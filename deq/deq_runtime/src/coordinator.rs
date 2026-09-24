use crate::decoder::DynDecoder;
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

pub mod reweight_handler;
pub use reweight_handler::{DecodeProjection, DecoderReweighting, LoadedDecoder};

pub mod loss_handler;
pub use loss_handler::{EnvelopeReweightPolicy, LossHandler, LossStrategy, ReweightScale};

mod forced_gap_handler;

/// Latches the first seed for a shot and rejects later differences.
/// `None` means no call yet; `Some(None)` latches an absent seed.
pub(crate) fn accept_decoder_seed(slot: &mut Option<Option<u64>>, decoder_seed: Option<u64>) -> Result<(), Status> {
    let expected = *slot.get_or_insert(decoder_seed);
    if expected == decoder_seed {
        Ok(())
    } else {
        Err(Status::invalid_argument(format!(
            "decoder_seed cannot change within a shot: expected {expected:?}, received {decoder_seed:?}"
        )))
    }
}

/// Returns the common seed, or `InvalidArgument` if the inputs differ.
/// Empty input returns `None`.
pub(crate) fn common_decoder_seed(seeds: impl IntoIterator<Item = Option<u64>>) -> Result<Option<u64>, Status> {
    let mut seeds = seeds.into_iter();
    let Some(first) = seeds.next() else {
        return Ok(None);
    };
    match seeds.find(|&seed| seed != first) {
        None => Ok(first),
        Some(other) => Err(Status::invalid_argument(format!(
            "decoder_seed must match across gadgets decoded together: found {first:?} and {other:?}"
        ))),
    }
}

#[cfg(test)]
mod decoder_seed_tests {
    use super::*;

    #[test]
    fn decoder_seed_is_fixed_until_reset() {
        let mut slot = None;
        accept_decoder_seed(&mut slot, Some(42)).unwrap();
        accept_decoder_seed(&mut slot, Some(42)).unwrap();

        let error = accept_decoder_seed(&mut slot, Some(23)).unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
        assert_eq!(
            error.message(),
            "decoder_seed cannot change within a shot: expected Some(42), received Some(23)"
        );

        slot = None;
        accept_decoder_seed(&mut slot, None).unwrap();
    }

    #[test]
    fn gadgets_decoded_together_share_one_seed() {
        assert_eq!(common_decoder_seed(std::iter::empty()).unwrap(), None);
        assert_eq!(common_decoder_seed([None, None]).unwrap(), None);
        assert_eq!(common_decoder_seed([Some(42), Some(42)]).unwrap(), Some(42));

        let error = common_decoder_seed([Some(42), Some(23)]).unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
        assert!(error.message().contains("found Some(42) and Some(23)"));
        assert!(
            common_decoder_seed([Some(0), None]).is_err(),
            "seed zero is not an absent seed"
        );
    }
}

impl CoordinatorType {
    pub fn create(&self, config: serde_json::Value, decoder: DynDecoder) -> DynCoordinator {
        self.create_with_gap_decoder(config, decoder, None)
    }

    /// Create a coordinator with an optional backend for forced-gap alternatives.
    /// `None` reuses the hard decoder; the naive coordinator does not score gaps.
    #[must_use]
    pub fn create_with_gap_decoder(
        &self,
        config: serde_json::Value,
        decoder: DynDecoder,
        gap_decoder: Option<DynDecoder>,
    ) -> DynCoordinator {
        match self {
            Self::Naive => DynCoordinator::Naive(Arc::new(NaiveCoordinator::new(config))),
            Self::Monolithic => DynCoordinator::Monolithic(Arc::new(MonolithicCoordinator::with_gap_decoder(
                config,
                decoder,
                gap_decoder,
            ))),
            Self::Window => {
                DynCoordinator::Window(Arc::new(WindowCoordinator::with_gap_decoder(config, decoder, gap_decoder)))
            }
        }
    }

    #[cfg(feature = "cli")]
    pub fn config_help() -> String {
        help_message::<naive_coordinator::NaiveCoordinatorConfig>("NaiveCoordinatorConfig:")
            + &*help_message::<monolithic_coordinator::MonolithicCoordinatorConfig>("MonolithicCoordinatorConfig:")
            + &*help_message::<window_coordinator::WindowCoordinatorConfig>("WindowCoordinatorConfig:")
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

#[cfg(any(feature = "cli", feature = "simulator"))]
impl Readouts {
    pub(crate) fn gather(gadget_readouts: &[Self]) -> Result<Self, Status> {
        let mut result = Self {
            readouts: Some(crate::util::BitVector::default()),
            ..Default::default()
        };
        for gadget in gadget_readouts {
            let readouts = gadget
                .readouts
                .as_ref()
                .ok_or_else(|| Status::internal("decoder returned no readouts"))?;
            crate::misc::bit_vector::append(result.readouts.as_mut().unwrap(), readouts);
            result.probabilities.extend_from_slice(&gadget.probabilities);
            result.syndrome_count += gadget.syndrome_count;
            result.correction_count += gadget.correction_count;
            result.correction_weight += gadget.correction_weight;
        }
        Ok(result)
    }
}
