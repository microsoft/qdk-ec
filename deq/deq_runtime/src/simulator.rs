#[cfg(all(feature = "cli", feature = "simulator"))]
use crate::misc::util::help_message;
#[cfg(feature = "cli")]
use clap::ValueEnum;
use rand::Rng;
use serde::Serialize;
#[cfg(feature = "cli")]
use tokio::sync::oneshot::Sender;

include!("proto/deq.simulator.rs");

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Debug)]
#[cfg_attr(feature = "cli", derive(ValueEnum))]
pub enum SimulatorType {
    /// do not run simulator
    None,
    /// a monolithic simulator that samples from a discrete distribution and run
    /// the logical error rate and decoding time evaluation
    Static,
    /// a simulator that uses the JIT controller to execute programs before sampling
    JitStatic,
    /// a simulator that drives TableauSimulator with retry-from-BEGIN semantics
    Preselect,
    /// a simulator that draws each shot's measurements from a user-supplied
    /// Python sampler (e.g. wrapping ``qdk.stim.run``); loss outcomes
    /// (returned as ``-`` by the Python adapter) are replaced with
    /// uniformly random bits so the decoder protocol is unchanged.
    #[cfg(all(feature = "simulator", feature = "python"))]
    Python,
}

pub mod common;
#[cfg(feature = "simulator")]
pub mod jit_static_simulator;
#[cfg(feature = "simulator")]
pub mod preselect_directives;
#[cfg(feature = "simulator")]
pub mod preselect_simulator;
#[cfg(all(feature = "simulator", feature = "python"))]
pub mod python_sampler;
#[cfg(all(feature = "simulator", feature = "python"))]
pub mod python_simulator;
#[cfg(feature = "simulator")]
pub mod rhai_assert;
#[cfg(feature = "simulator")]
pub mod static_simulator;
#[cfg(feature = "simulator")]
pub mod stim_delays;
#[cfg(feature = "simulator")]
pub mod tableau_preselect_sampler;
#[cfg(feature = "simulator")]
pub use jit_static_simulator::JitStaticSimulator;
#[cfg(feature = "simulator")]
pub use preselect_simulator::PreselectSimulator;
#[cfg(all(feature = "simulator", feature = "python"))]
pub use python_simulator::PythonSimulator;
#[cfg(feature = "simulator")]
pub use static_simulator::StaticSimulator;

#[allow(dead_code)]
/// use Xoshiro256StarStar for deterministic random number generator
pub type DeterministicRng = rand_xoshiro::Xoshiro256StarStar;

pub trait F64Rng {
    fn next_f64(&mut self) -> f64;
}

impl F64Rng for DeterministicRng {
    fn next_f64(&mut self) -> f64 {
        // 0x3ff: 10 bits among 11 bits exponent
        // shift right 12 (11 bits exponent + 1 bit sign)
        f64::from_bits((0x3ff << 52) | (self.next_u64() >> 12)) - 1.0
    }
}

pub trait F32Rng {
    fn next_f32(&mut self) -> f32;
}

impl F32Rng for DeterministicRng {
    fn next_f32(&mut self) -> f32 {
        // 0x7f: 7 bits among 8 bits exponent
        // shift right 9 (8 bits exponent + 1 bit sign)
        f32::from_bits((0x7f << 23) | (self.next_u32() >> 9)) - 1.0
    }
}

impl SimulatorType {
    pub fn create(&self, config: serde_json::Value) -> DynSimulator {
        match self {
            Self::None => DynSimulator::None,
            #[cfg(feature = "simulator")]
            Self::Static => DynSimulator::Static(Box::new(StaticSimulator::new(config))),
            #[cfg(feature = "simulator")]
            Self::JitStatic => DynSimulator::JitStatic(Box::new(JitStaticSimulator::new(config))),
            #[cfg(feature = "simulator")]
            Self::Preselect => DynSimulator::Preselect(Box::new(PreselectSimulator::new(config))),
            #[cfg(all(feature = "simulator", feature = "python"))]
            Self::Python => DynSimulator::Python(Box::new(PythonSimulator::new(config))),
            #[cfg(not(feature = "simulator"))]
            Self::Static | Self::JitStatic | Self::Preselect => {
                let _ = config;
                panic!("simulator requires the `simulator` feature; rebuild with `--features simulator`")
            }
        }
    }

    #[cfg(feature = "cli")]
    pub fn config_help() -> String {
        #[cfg(feature = "simulator")]
        {
            #[cfg(feature = "python")]
            let python_help = help_message::<python_simulator::PythonSimulatorConfig>("PythonSimulatorConfig:");
            #[cfg(not(feature = "python"))]
            let python_help = String::new();
            help_message::<static_simulator::StaticSimulatorConfig>("StaticSimulatorConfig:")
                + &*help_message::<jit_static_simulator::JitStaticSimulatorConfig>("JitStaticSimulatorConfig:")
                + &*help_message::<preselect_simulator::PreselectSimulatorConfig>("PreselectSimulatorConfig:")
                + &*python_help
        }
        #[cfg(not(feature = "simulator"))]
        String::new()
    }

    #[cfg(not(feature = "cli"))]
    pub fn config_help() -> String {
        String::new()
    }
}

pub enum DynSimulator {
    None,
    #[cfg(feature = "simulator")]
    Static(Box<StaticSimulator>),
    #[cfg(feature = "simulator")]
    JitStatic(Box<JitStaticSimulator>),
    #[cfg(feature = "simulator")]
    Preselect(Box<PreselectSimulator>),
    #[cfg(all(feature = "simulator", feature = "python"))]
    Python(Box<PythonSimulator>),
}

impl DynSimulator {
    #[cfg(feature = "cli")]
    pub async fn start(self, endpoint: tonic::transport::Endpoint, shutdown_signal: Sender<()>) {
        match self {
            DynSimulator::None => {
                let _ = std::mem::ManuallyDrop::new(shutdown_signal);
                let _ = endpoint;
            }
            #[cfg(feature = "simulator")]
            DynSimulator::Static(simulator) => {
                simulator.start(endpoint, shutdown_signal).await;
            }
            #[cfg(feature = "simulator")]
            DynSimulator::JitStatic(simulator) => {
                simulator.start(endpoint, shutdown_signal).await;
            }
            #[cfg(feature = "simulator")]
            DynSimulator::Preselect(simulator) => {
                simulator.start(endpoint, shutdown_signal).await;
            }
            #[cfg(all(feature = "simulator", feature = "python"))]
            DynSimulator::Python(simulator) => {
                simulator.start(endpoint, shutdown_signal).await;
            }
        }
    }
}
