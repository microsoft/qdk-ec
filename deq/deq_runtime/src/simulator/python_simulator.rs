//! Python Simulator
//!
//! A simulator that draws each shot's measurements from a user-supplied
//! Python sampler (e.g. wrapping `qdk.stim.run`) and feeds the result to
//! the standard static decoder controller — identical wire protocol to
//! [`StaticSimulator`], identical decoder-side handling.
//!
//! Loss-as-flip is applied inside [`PythonSampler`]: any ``'-'`` returned
//! by the Python sampler is replaced with a uniformly random bit before
//! the measurement record is packed.  No protocol change reaches the
//! decoder.
//!
//! [`StaticSimulator`]: crate::simulator::static_simulator::StaticSimulator
//! [`PythonSampler`]: crate::simulator::python_sampler::PythonSampler

#[cfg(feature = "cli")]
use crate::controller::static_controller::static_controller_client::StaticControllerClient;
#[cfg(feature = "cli")]
use crate::coordinator;
#[cfg(feature = "cli")]
use crate::misc::bit_vector;
use crate::simulator::DeterministicRng;
use crate::simulator::common::{CommonSimulatorConfig, DelayBatch, Sampler};
#[cfg(feature = "cli")]
use crate::simulator::common::{DecoderClient, ErrorSet, run_simulation_loop};
use crate::simulator::python_sampler::{PythonSampler, PythonSamplerConfig};
#[cfg(feature = "cli")]
use crate::util::BitVector;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
#[cfg(feature = "cli")]
use structdoc::StructDoc;
#[cfg(feature = "cli")]
use tokio::sync::oneshot::Sender;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct PythonSimulatorConfig {
    /// the filepath to a Stim circuit file
    pub filepath: String,
    /// the Python sampler configuration (forwarded to the Python class constructor)
    #[serde(flatten)]
    pub sampler: PythonSamplerConfig,
    /// common simulation configuration
    #[serde(flatten)]
    pub common: CommonSimulatorConfig,
}

pub struct PythonSimulator {
    pub config: PythonSimulatorConfig,
    pub rng: DeterministicRng,
    pub sampler: Box<dyn Sampler>,
    #[cfg_attr(not(feature = "cli"), allow(dead_code))]
    delay_schedule: Vec<DelayBatch>,
    #[cfg_attr(not(feature = "cli"), allow(dead_code))]
    embedded_rhai_script: Option<String>,
}

impl PythonSimulator {
    pub fn new(config: serde_json::Value) -> Self {
        let config: PythonSimulatorConfig = serde_json::from_value(config).unwrap();
        let seed: u64 = config.common.seed.unwrap_or_else(|| rand::rng().next_u64());

        let circuit_text = std::fs::read_to_string(&config.filepath)
            .unwrap_or_else(|e| panic!("Failed to read Stim circuit file '{}': {e}", config.filepath));
        let embedded_rhai_script = crate::simulator::rhai_assert::extract_rhai_script(&circuit_text);

        // Count measurements by scanning the text line-by-line rather than via
        // the upstream `stim` crate.  The python sampler is the plug-point for
        // non-stim backends (e.g. QDK's stabilizer simulator with `LOSS_ERROR`),
        // so we must not require that the Stim file be parseable by the upstream
        // crate.  `count_measurements` only needs to recognize the standard
        // measurement-producing instruction names.
        let num_measurements = crate::simulator::stim_delays::count_measurements(&circuit_text);

        // Strip `#!rhai` blocks before handing the circuit to the Python
        // sampler: the sampler has no business with the logical-error
        // assertion script.
        let sampler_circuit_text = crate::simulator::rhai_assert::strip_rhai_scripts(&circuit_text);

        let sampler = PythonSampler::new(
            &sampler_circuit_text,
            &config.sampler,
            seed,
            config.common.skip_shots,
            num_measurements,
        );

        let delay_schedule = crate::simulator::stim_delays::extract_delay_schedule(&circuit_text, num_measurements);

        Self {
            config,
            rng: DeterministicRng::seed_from_u64(seed),
            sampler: Box::new(sampler),
            delay_schedule,
            embedded_rhai_script,
        }
    }

    #[cfg(feature = "cli")]
    pub async fn start(mut self, endpoint: tonic::transport::Endpoint, shutdown: Sender<()>) {
        let rhai_engine = crate::simulator::rhai_assert::RhaiAssertEngine::build(
            &self.config.filepath,
            self.embedded_rhai_script.as_deref(),
            self.config.common.logical_assert_filepath.as_deref(),
        );

        let delay_schedule = if self.config.common.strict_timing {
            self.delay_schedule.clone()
        } else {
            vec![]
        };
        let mut client = PythonSimDecoderClient {
            client: None,
            endpoint,
            delay_schedule,
            last_latency_secs: 0.0,
        };
        run_simulation_loop(
            &self.config.common,
            self.sampler.as_ref(),
            &mut self.rng,
            &mut client,
            shutdown,
            &rhai_engine,
        )
        .await;
    }
}

#[cfg(feature = "cli")]
struct PythonSimDecoderClient {
    client: Option<StaticControllerClient<tonic::transport::Channel>>,
    endpoint: tonic::transport::Endpoint,
    delay_schedule: Vec<DelayBatch>,
    last_latency_secs: f64,
}

#[cfg(feature = "cli")]
impl DecoderClient for PythonSimDecoderClient {
    async fn initialize(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.client = Some(StaticControllerClient::connect(self.endpoint.clone()).await?);
        Ok(())
    }

    async fn decode(&mut self, sample: &ErrorSet) -> Option<BitVector> {
        let client = self.client.as_mut().unwrap();

        if self.delay_schedule.is_empty() {
            let t0 = std::time::Instant::now();
            let response = client
                .decode(coordinator::Outcomes {
                    gid: 0,
                    outcomes: Some(sample.measurements.clone()),
                    modifiers: vec![],
                    loss_mask: sample.loss_mask.clone(),
                })
                .await
                .unwrap()
                .into_inner();
            self.last_latency_secs = t0.elapsed().as_secs_f64();
            return Some(response.readouts.unwrap());
        }

        let all_bits = bit_vector::unpack_bits(&sample.measurements.data, sample.measurements.size);
        let all_loss_bits: Option<Vec<bool>> =
            sample.loss_mask.as_ref().map(|bv| bit_vector::unpack_bits(&bv.data, bv.size));
        let mut accumulated_readouts: Vec<bool> = Vec::new();
        let mut prev_count = 0usize;
        let n_batches = self.delay_schedule.len();

        for (i, batch) in self.delay_schedule.iter().enumerate() {
            let sleep_duration = if i == 0 {
                batch.delay_seconds
            } else {
                batch.delay_seconds - self.delay_schedule[i - 1].delay_seconds
            };
            if sleep_duration > 0.0 {
                tokio::time::sleep(std::time::Duration::from_secs_f64(sleep_duration)).await;
            }

            let end = batch.cumulative_count.min(all_bits.len());
            let slice = &all_bits[prev_count..end];
            let partial = BitVector {
                size: slice.len() as u64,
                data: bit_vector::pack_bits(slice),
            };
            let partial_loss = all_loss_bits.as_ref().map(|lb| {
                let loss_slice = &lb[prev_count..end];
                BitVector {
                    size: loss_slice.len() as u64,
                    data: bit_vector::pack_bits(loss_slice),
                }
            });
            prev_count = end;

            let t0 = std::time::Instant::now();
            let response = client
                .decode(coordinator::Outcomes {
                    gid: 0,
                    outcomes: Some(partial),
                    modifiers: vec![],
                    loss_mask: partial_loss,
                })
                .await
                .unwrap()
                .into_inner();
            if i == n_batches - 1 {
                self.last_latency_secs = t0.elapsed().as_secs_f64();
            }
            let readouts = response.readouts.unwrap();
            let bits = bit_vector::unpack_bits(&readouts.data, readouts.size);
            accumulated_readouts.extend_from_slice(&bits);
        }

        Some(BitVector {
            size: accumulated_readouts.len() as u64,
            data: bit_vector::pack_bits(&accumulated_readouts),
        })
    }

    async fn reset(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let client = self.client.as_mut().unwrap();
        client.reset(()).await?;
        Ok(())
    }

    fn simulator_name(&self) -> &'static str {
        "python_simulator"
    }

    fn last_decode_latency_secs(&self) -> f64 {
        self.last_latency_secs
    }
}
