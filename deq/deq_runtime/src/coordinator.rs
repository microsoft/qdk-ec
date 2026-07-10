use crate::decoder::BlackBoxDecoderClient;
#[cfg(feature = "cli")]
use crate::misc::util::help_message;
#[cfg(feature = "cli")]
use clap::ValueEnum;
use serde::Serialize;
use std::sync::Arc;
#[cfg(feature = "cli")]
use tonic::transport::Endpoint;
#[cfg(feature = "cli")]
use tonic::transport::server::Router;
use tonic::{Request, Status};

// Re-export so that generated proto code in window_coordinator::trace can
// reference `super::super::bin::*` (i.e. coordinator::bin).
pub(crate) use crate::bin;

include!("proto/deq.coordinator.rs");
#[cfg(feature = "cli")]
use coordinator_server::CoordinatorServer;

/// Replace each bit of `outcomes` whose position is set in `loss_mask`
/// with a uniformly random bit drawn from `rng`.  This is the default
/// **loss-random-imputation** strategy: lost measurements (`loss_mask`
/// bits set to 1) are filled with random bits before the coordinator
/// computes the parity-check syndrome.
///
/// Panics if `outcomes.size != loss_mask.size`, since a length mismatch
/// indicates a wire-format bug at the controller boundary.
pub fn apply_loss_random_imputation<R: rand::Rng>(
    outcomes: &mut crate::util::BitVector,
    loss_mask: &crate::util::BitVector,
    rng: &mut R,
) {
    use crate::misc::bit_vector;
    use rand::RngExt;
    assert_eq!(
        outcomes.size, loss_mask.size,
        "loss_mask size {} does not match outcomes size {}",
        loss_mask.size, outcomes.size,
    );
    for i in 0..outcomes.size {
        if bit_vector::get_bit(loss_mask, i) {
            bit_vector::set_bit(outcomes, i, rng.random::<bool>());
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Debug)]
#[cfg_attr(feature = "cli", derive(ValueEnum))]
pub enum CoordinatorType {
    /// a coordinator that does nothing but returning all-0 or random readouts
    Naive,
    /// a monolithic coordinator that only decode when all the output ports are
    /// connected and the measurements are loaded.
    Monolithic,
    /// window decoding
    Window,
}

impl crate::controller::ParseByName for CoordinatorType {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "naive" => Some(Self::Naive),
            "monolithic" => Some(Self::Monolithic),
            "window" => Some(Self::Window),
            _ => None,
        }
    }

    fn variant_names() -> Vec<&'static str> {
        vec!["naive", "monolithic", "window"]
    }
}

pub mod naive_coordinator;
pub use naive_coordinator::NaiveCoordinator;

pub mod monolithic_coordinator;
pub use monolithic_coordinator::MonolithicCoordinator;

pub mod window_coordinator;
pub use window_coordinator::WindowCoordinator;

pub mod mock_coordinator;
pub use mock_coordinator::MockCoordinator;

pub mod decoder_cache_key;
pub use decoder_cache_key::{
    DecoderCacheKey, ErrorModelFingerprint, FingerprintSource, ProbabilityModifierBits, build_modifier_fingerprints,
};

impl CoordinatorType {
    pub fn create(&self, config: serde_json::Value, black_box_decoder: Option<BlackBoxDecoderClient>) -> DynCoordinator {
        match self {
            Self::Naive => DynCoordinator::Naive(Arc::new(NaiveCoordinator::new(config))),
            Self::Monolithic | Self::Window => {
                let black_box_decoder =
                    black_box_decoder.expect("the provided decoder type does not support black box decoder interface");
                match self {
                    Self::Monolithic => {
                        DynCoordinator::Monolithic(Arc::new(MonolithicCoordinator::new(config, black_box_decoder)))
                    }
                    Self::Window => DynCoordinator::Window(Arc::new(WindowCoordinator::new(config, black_box_decoder))),
                    _ => unreachable!(),
                }
            }
        }
    }

    #[cfg(feature = "cli")]
    pub fn config_help() -> String {
        help_message::<naive_coordinator::NaiveCoordinatorConfig>("NaiveCoordinatorConfig:")
            + &*help_message::<monolithic_coordinator::MonolithicCoordinatorConfig>("MonolithicCoordinatorConfig:")
    }

    #[cfg(not(feature = "cli"))]
    pub fn config_help() -> String {
        String::new()
    }
}

#[derive(Clone)]
pub enum DynCoordinator {
    None,
    Naive(Arc<NaiveCoordinator>),
    Monolithic(Arc<MonolithicCoordinator>),
    Window(Arc<WindowCoordinator>),
    Mock(Arc<MockCoordinator>),
}

impl DynCoordinator {
    pub fn inner(&self) -> Arc<dyn coordinator_server::Coordinator> {
        match self {
            DynCoordinator::None => panic!("DynCoordinator::None has no inner coordinator"),
            DynCoordinator::Naive(v) => v.clone(),
            DynCoordinator::Monolithic(v) => v.clone(),
            DynCoordinator::Window(v) => v.clone(),
            DynCoordinator::Mock(v) => v.clone(),
        }
    }

    #[cfg(feature = "cli")]
    fn add_service_by(router: Router, service: &Arc<impl coordinator_server::Coordinator>) -> Router {
        let service = CoordinatorServer::from_arc(service.clone()).max_decoding_message_size(usize::MAX);
        router.add_service(service)
    }

    #[cfg(feature = "cli")]
    pub fn add_service(&self, router: Router) -> Router {
        match self {
            DynCoordinator::None => router,
            DynCoordinator::Naive(c) => Self::add_service_by(router, c),
            DynCoordinator::Monolithic(c) => Self::add_service_by(router, c),
            DynCoordinator::Window(c) => Self::add_service_by(router, c),
            DynCoordinator::Mock(c) => Self::add_service_by(router, c),
        }
    }

    pub async fn start(&self) {}

    /// Fire each underlying coordinator's cancellation token to abort pending
    /// decode tasks. Coordinators without a cancellation surface (`Naive`,
    /// `Mock`) are no-ops. Used by [`crate::server::LocalServer::shutdown`].
    pub async fn cancel_pending(&self) {
        match self {
            DynCoordinator::None => {}
            DynCoordinator::Naive(_) => {}
            DynCoordinator::Monolithic(c) => c.cancel_pending().await,
            DynCoordinator::Window(c) => c.cancel_pending().await,
            DynCoordinator::Mock(_) => {}
        }
    }
}

/// a client wrapper that can either be a remote gRPC client or a local reference
#[derive(Clone)]
pub enum CoordinatorClient {
    #[cfg(feature = "cli")]
    Remote(coordinator_client::CoordinatorClient<tonic::transport::Channel>),
    Local(DynCoordinator),
}

impl CoordinatorClient {
    #[cfg(feature = "cli")]
    pub async fn from_endpoint(endpoint: Endpoint) -> Self {
        CoordinatorClient::Remote(
            crate::coordinator::coordinator_client::CoordinatorClient::connect(endpoint)
                .await
                .unwrap(),
        )
    }

    /// Create a CoordinatorClient from a MockCoordinator for testing.
    pub fn from_mock(mock: Arc<MockCoordinator>) -> Self {
        CoordinatorClient::Local(DynCoordinator::Mock(mock))
    }

    pub async fn reset(&self, flags: ResetRequest) -> std::result::Result<(), Status> {
        let request = Request::new(flags);
        (match self {
            #[cfg(feature = "cli")]
            CoordinatorClient::Remote(client) => client.clone().reset(request).await,
            CoordinatorClient::Local(local) => local.inner().reset(request).await,
        })
        .map(|v| v.into_inner())
    }

    pub async fn load_library(&self, library: crate::bin::Library) -> std::result::Result<(), Status> {
        let request = Request::new(library);
        (match self {
            #[cfg(feature = "cli")]
            CoordinatorClient::Remote(client) => client.clone().load_library(request).await,
            CoordinatorClient::Local(local) => local.inner().load_library(request).await,
        })
        .map(|v| v.into_inner())
    }

    pub async fn unload(&self, _unload: UnloadLibrary) -> std::result::Result<(), Status> {
        unimplemented!()
    }

    pub async fn execute(&self, instruction: crate::bin::Instruction) -> std::result::Result<ExecuteResponse, Status> {
        let request = Request::new(instruction);
        (match self {
            #[cfg(feature = "cli")]
            CoordinatorClient::Remote(client) => client.clone().execute(request).await,
            CoordinatorClient::Local(local) => local.inner().execute(request).await,
        })
        .map(|v| v.into_inner())
    }

    pub async fn decode(&self, outcomes: Outcomes) -> std::result::Result<Readouts, Status> {
        let request = Request::new(outcomes);
        (match self {
            #[cfg(feature = "cli")]
            CoordinatorClient::Remote(client) => client.clone().decode(request).await,
            CoordinatorClient::Local(local) => local.inner().decode(request).await,
        })
        .map(|v| v.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::bit_vector;
    use crate::simulator::DeterministicRng;
    use crate::util::BitVector;
    use rand::SeedableRng;

    #[test]
    fn apply_loss_random_imputation_leaves_non_loss_bits_untouched() {
        // outcomes = [1, 0, 1, 0], loss_mask = [0, 0, 0, 0]
        let mut outcomes = BitVector {
            size: 4,
            data: vec![0b1010_0000],
        };
        let loss_mask = BitVector {
            size: 4,
            data: vec![0b0000_0000],
        };
        let mut rng = DeterministicRng::seed_from_u64(42);

        let before = outcomes.clone();
        apply_loss_random_imputation(&mut outcomes, &loss_mask, &mut rng);
        assert_eq!(
            outcomes, before,
            "no loss_mask bits set → no imputation, outcomes must stay byte-identical",
        );
    }

    #[test]
    fn apply_loss_random_imputation_only_replaces_marked_bits() {
        // 1000 trials with loss_mask = [0, 1, 0, 1].  bits 0 and 2 must
        // stay at their input values (1 and 1); bits 1 and 3 must take
        // both 0 and 1 over the trial set (with overwhelming probability).
        let mut rng = DeterministicRng::seed_from_u64(1);
        let loss_mask = BitVector {
            size: 4,
            data: vec![0b0101_0000],
        };
        let mut bit1_zero_count = 0usize;
        let mut bit3_zero_count = 0usize;
        let trials = 1000usize;
        for _ in 0..trials {
            let mut outcomes = BitVector {
                size: 4,
                data: vec![0b1010_0000],
            };
            apply_loss_random_imputation(&mut outcomes, &loss_mask, &mut rng);
            assert!(bit_vector::get_bit(&outcomes, 0), "bit 0 not in loss_mask, must be preserved");
            assert!(bit_vector::get_bit(&outcomes, 2), "bit 2 not in loss_mask, must be preserved");
            if !bit_vector::get_bit(&outcomes, 1) {
                bit1_zero_count += 1;
            }
            if !bit_vector::get_bit(&outcomes, 3) {
                bit3_zero_count += 1;
            }
        }
        let lo = trials / 4;
        let hi = 3 * trials / 4;
        assert!(
            (lo..hi).contains(&bit1_zero_count),
            "bit 1 imputed zero {bit1_zero_count}/{trials} times; expected roughly balanced",
        );
        assert!(
            (lo..hi).contains(&bit3_zero_count),
            "bit 3 imputed zero {bit3_zero_count}/{trials} times; expected roughly balanced",
        );
    }

    #[test]
    fn apply_loss_random_imputation_is_deterministic_with_same_seed() {
        let loss_mask = BitVector {
            size: 8,
            data: vec![0b1111_1111],
        };
        let initial = BitVector {
            size: 8,
            data: vec![0b0000_0000],
        };

        let mut a = initial.clone();
        let mut rng_a = DeterministicRng::seed_from_u64(123);
        apply_loss_random_imputation(&mut a, &loss_mask, &mut rng_a);

        let mut b = initial.clone();
        let mut rng_b = DeterministicRng::seed_from_u64(123);
        apply_loss_random_imputation(&mut b, &loss_mask, &mut rng_b);

        assert_eq!(a, b, "same seed → identical imputation");
    }

    #[test]
    #[should_panic(expected = "does not match outcomes size")]
    fn apply_loss_random_imputation_panics_on_size_mismatch() {
        let mut outcomes = BitVector {
            size: 4,
            data: vec![0b0000_0000],
        };
        let loss_mask = BitVector {
            size: 5,
            data: vec![0b0000_1000],
        };
        let mut rng = DeterministicRng::seed_from_u64(0);
        apply_loss_random_imputation(&mut outcomes, &loss_mask, &mut rng);
    }
}
