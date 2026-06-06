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
