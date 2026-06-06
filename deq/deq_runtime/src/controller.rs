use crate::coordinator::CoordinatorClient;
#[cfg(feature = "cli")]
use crate::misc::util::help_message;
#[cfg(feature = "cli")]
use clap::ValueEnum;
use serde::Serialize;
use std::sync::Arc;
#[cfg(feature = "cli")]
use tonic::transport::server::Router;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Debug)]
#[cfg_attr(feature = "cli", derive(ValueEnum))]
pub enum ControllerType {
    /// do not provide controller
    None,
    /// a controller that runs the static reference program in the deq-bin and only takes
    /// measurement outcomes as input during runtime
    Static,
    /// a JIT controller that compiles gadgets, check models, and error models on-the-fly
    Jit,
}

pub mod static_controller;
pub use static_controller::StaticController;

pub mod jit_controller;
pub use jit_controller::JitController;

impl ControllerType {
    pub fn create(&self, config: serde_json::Value) -> DynController {
        match self {
            Self::None => DynController::None,
            Self::Static => DynController::Static(Arc::new(StaticController::new(config))),
            Self::Jit => DynController::Jit(JitController::new(config)),
        }
    }

    #[cfg(feature = "cli")]
    pub fn config_help() -> String {
        help_message::<static_controller::StaticControllerConfig>("StaticControllerConfig:")
            + &*help_message::<jit_controller::JitControllerConfig>("JitControllerConfig:")
    }

    #[cfg(not(feature = "cli"))]
    pub fn config_help() -> String {
        String::new()
    }
}

#[derive(Clone)]
pub enum DynController {
    None,
    Static(Arc<StaticController>),
    Jit(Arc<JitController>),
}

impl DynController {
    #[cfg(feature = "cli")]
    pub fn add_service(&self, router: Router) -> Router {
        match self {
            DynController::None => router,
            DynController::Static(controller) => StaticController::add_service(controller, router),
            DynController::Jit(controller) => JitController::add_service(controller, router),
        }
    }

    pub async fn start(&self, coordinator: CoordinatorClient) {
        match self {
            DynController::None => {}
            DynController::Static(controller) => {
                controller.start(coordinator).await;
            }
            DynController::Jit(controller) => {
                controller.start(coordinator).await;
            }
        }
    }
}

/// Trait for parsing enum variants from their kebab-case string names.
/// When the `cli` feature is enabled, this delegates to clap's `ValueEnum`.
/// Otherwise, each enum provides its own manual implementation.
pub trait ParseByName: Sized {
    fn from_name(name: &str) -> Option<Self>;
    fn variant_names() -> Vec<&'static str>;
}
