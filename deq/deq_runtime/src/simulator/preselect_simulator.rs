//! Preselect Simulator
//!
//! A simulator that uses `TableauPreselectSampler` to drive
//! `stim::TableauSimulator` directly with retry-from-BEGIN semantics.

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
use crate::simulator::tableau_preselect_sampler::TableauPreselectSampler;
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
pub struct PreselectSimulatorConfig {
    /// the filepath to a Stim circuit file
    pub filepath: String,
    /// common simulation configuration
    #[serde(flatten)]
    pub common: CommonSimulatorConfig,
}

pub struct PreselectSimulator {
    pub config: PreselectSimulatorConfig,
    pub rng: DeterministicRng,
    pub sampler: Box<dyn Sampler>,
    #[cfg_attr(not(feature = "cli"), allow(dead_code))]
    delay_schedule: Vec<DelayBatch>,
    #[cfg_attr(not(feature = "cli"), allow(dead_code))]
    embedded_rhai_script: Option<String>,
}

impl PreselectSimulator {
    pub fn new(config: serde_json::Value) -> Self {
        let config: PreselectSimulatorConfig = serde_json::from_value(config).unwrap();
        let seed: u64 = config.common.seed.unwrap_or_else(|| rand::rng().next_u64());

        let circuit_text = std::fs::read_to_string(&config.filepath)
            .unwrap_or_else(|e| panic!("Failed to read Stim circuit file '{}': {e}", config.filepath));
        let embedded_rhai_script = crate::simulator::rhai_assert::extract_rhai_script(&circuit_text);
        let sampler = TableauPreselectSampler::new(
            &circuit_text,
            seed,
            config.common.skip_shots,
            config.common.strict_timing,
            config.common.preselect_max_attempts,
        );
        let delay_schedule = {
            let circuit: stim::Circuit = circuit_text
                .parse()
                .expect("Failed to parse Stim circuit for measurement counting");
            let expected =
                usize::try_from(circuit.num_measurements()).expect("Stim circuit measurement count exceeds usize");
            crate::simulator::stim_delays::extract_delay_schedule(&circuit_text, expected)
        };

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
        let mut client = PreselectDecoderClient {
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
struct PreselectDecoderClient {
    client: Option<StaticControllerClient<tonic::transport::Channel>>,
    endpoint: tonic::transport::Endpoint,
    delay_schedule: Vec<DelayBatch>,
    last_latency_secs: f64,
}

#[cfg(feature = "cli")]
impl DecoderClient for PreselectDecoderClient {
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
        "preselect"
    }

    fn last_decode_latency_secs(&self) -> f64 {
        self.last_latency_secs
    }
}
