//! JIT Static Simulator
//!
//! A simulator that uses the JIT controller to execute programs before sampling.
//! This allows simulating circuits that are compiled at runtime using the JIT compiler.

#[cfg(feature = "cli")]
use crate::controller::jit_controller::jit_controller_client::JitControllerClient;
#[cfg(feature = "cli")]
use crate::coordinator;
use crate::jit;
#[cfg(feature = "cli")]
use crate::misc::bit_vector;
use crate::simulator::DeterministicRng;
use crate::simulator::common::{CommonSimulatorConfig, DelayBatch, Sampler, load_stim_circuit};
#[cfg(feature = "cli")]
use crate::simulator::common::{DecoderClient, ErrorSet, run_simulation_loop};
#[cfg(feature = "cli")]
use crate::util::BitVector;
#[cfg(feature = "cli")]
use hashbrown::HashMap;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
#[cfg(feature = "cli")]
use std::sync::Arc;
#[cfg(feature = "cli")]
use structdoc::StructDoc;
#[cfg(feature = "cli")]
use tokio::sync::oneshot::Sender;
#[cfg(feature = "cli")]
use tokio::sync::watch;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct JitStaticSimulatorConfig {
    /// the filepath to a Stim circuit file for sampling
    pub filepath: String,
    /// the filepath to the JIT library containing gadget types, port types, and program
    pub jit_library_filepath: String,
    /// common simulation configuration
    #[serde(flatten)]
    pub common: CommonSimulatorConfig,
}

pub struct JitStaticSimulator {
    pub config: JitStaticSimulatorConfig,
    pub rng: DeterministicRng,
    pub sampler: Box<dyn Sampler>,
    pub jit_library: jit::JitLibrary,
    #[cfg_attr(not(feature = "cli"), allow(dead_code))]
    delay_schedule: Vec<DelayBatch>,
    #[cfg_attr(not(feature = "cli"), allow(dead_code))]
    embedded_rhai_script: Option<String>,
}

impl JitStaticSimulator {
    pub fn new(config: serde_json::Value) -> Self {
        let config: JitStaticSimulatorConfig = serde_json::from_value(config).unwrap();
        let seed: u64 = config.common.seed.unwrap_or_else(|| rand::rng().next_u64());

        // Load JIT library (always needed for controller setup)
        let jit_data = std::fs::read(&config.jit_library_filepath).unwrap();
        let jit_library: jit::JitLibrary = prost::Message::decode(&mut jit_data.as_slice()).unwrap();

        let (sampler, delay_schedule, embedded_rhai_script) =
            load_stim_circuit(&config.filepath, seed, config.common.skip_shots, config.common.strict_timing);
        Self {
            config,
            rng: DeterministicRng::seed_from_u64(seed),
            sampler,
            jit_library,
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
        let mut client = JitDecoderClient {
            client: None,
            endpoint,
            jit_library: self.jit_library.clone(),
            measurement_ranges: Vec::new(),
            expected_gid_to_index: Arc::new(HashMap::new()),
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
struct JitDecoderClient {
    client: Option<JitControllerClient<tonic::transport::Channel>>,
    endpoint: tonic::transport::Endpoint,
    jit_library: jit::JitLibrary,
    /// Pre-computed (measurement_start, measurement_count) for each instruction.
    measurement_ranges: Vec<(usize, usize)>,
    /// Map from expected gid (in the JIT library) to instruction index.
    /// Used to determine which watch channel to wait on for dependencies.
    expected_gid_to_index: Arc<HashMap<u64, usize>>,
    /// Delay schedule: when measurement batches become available.
    delay_schedule: Vec<DelayBatch>,
    /// Latency of the last decode call (seconds).
    last_latency_secs: f64,
}

#[cfg(feature = "cli")]
impl DecoderClient for JitDecoderClient {
    async fn initialize(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.client = Some(JitControllerClient::connect(self.endpoint.clone()).await?);

        // Pre-compute measurement ranges for each instruction
        let mut gtype_to_measurement_count: HashMap<u64, usize> = HashMap::new();
        for gadget_type in &self.jit_library.gadget_types {
            if let Some(base) = &gadget_type.base {
                gtype_to_measurement_count.insert(base.gtype, base.measurements.len());
            }
        }

        self.measurement_ranges.clear();
        let mut expected_gid_to_index = HashMap::new();
        let mut measurement_offset = 0;
        for (index, instruction) in self.jit_library.program.iter().enumerate() {
            let gadget = instruction.gadget.as_ref().unwrap();
            let gtype = gadget.gtype;
            let measurement_count = gtype_to_measurement_count.get(&gtype).copied().unwrap();
            self.measurement_ranges.push((measurement_offset, measurement_count));
            measurement_offset += measurement_count;
            let gid = if gadget.gid > 0 { gadget.gid } else { (index + 1) as u64 };
            assert!(!expected_gid_to_index.contains_key(&gid), "Duplicate gid {}", gid);
            expected_gid_to_index.insert(gid, index);
        }
        self.expected_gid_to_index = Arc::new(expected_gid_to_index);

        Ok(())
    }

    async fn decode(&mut self, sample: &ErrorSet) -> Option<BitVector> {
        let t0 = std::time::Instant::now();
        let measurements = sample.measurements.clone();

        let instruction_delays: Vec<f64> = if self.delay_schedule.is_empty() {
            vec![0.0; self.jit_library.program.len()]
        } else {
            self.measurement_ranges
                .iter()
                .map(|&(start, count)| {
                    let meas_end = start + count;
                    self.delay_schedule
                        .iter()
                        .find(|b| b.cumulative_count >= meas_end)
                        .map_or_else(
                            || self.delay_schedule.last().map_or(0.0, |b| b.delay_seconds),
                            |b| b.delay_seconds,
                        )
                })
                .collect()
        };
        let max_delay = instruction_delays.iter().copied().fold(0.0_f64, f64::max);

        let gid_signals: Arc<Vec<watch::Sender<Option<u64>>>> =
            Arc::new((0..self.jit_library.program.len()).map(|_| watch::channel(None).0).collect());

        let decode_futures: Vec<_> = self
            .jit_library
            .program
            .clone()
            .into_iter()
            .zip(self.measurement_ranges.iter().copied())
            .zip(instruction_delays)
            .enumerate()
            .map(
                |(index, ((instruction, (measurement_start, measurement_count)), delay_secs))| {
                    let client = self.client.as_ref().unwrap().clone();
                    let gadget_measurements = bit_vector::slice(&measurements, measurement_start, measurement_count);
                    let gid_signals = Arc::clone(&gid_signals);
                    let expected_gid_to_index = Arc::clone(&self.expected_gid_to_index);

                    let dependency_indices: Vec<usize> = instruction
                        .gadget
                        .as_ref()
                        .unwrap()
                        .connectors
                        .iter()
                        .map(|connector| expected_gid_to_index[&connector.gid])
                        .collect();

                    tokio::spawn(async move {
                        for &dep_index in &dependency_indices {
                            let mut rx = gid_signals[dep_index].subscribe();
                            rx.wait_for(|v| v.is_some()).await.ok();
                        }

                        let mut client = client;

                        let response = client.execute(instruction).await.unwrap().into_inner();
                        let gid = response.id;

                        gid_signals[index].send_replace(Some(gid));

                        if delay_secs > 0.0 {
                            let elapsed = t0.elapsed().as_secs_f64();
                            let remaining = delay_secs - elapsed;
                            if remaining > 0.0 {
                                tokio::time::sleep(std::time::Duration::from_secs_f64(remaining)).await;
                            }
                        }

                        let outcomes = coordinator::Outcomes {
                            gid,
                            outcomes: Some(gadget_measurements),
                            modifiers: vec![],
                            loss_mask: None,
                        };
                        let response = client.decode(outcomes).await.unwrap().into_inner();
                        (index, response.readouts)
                    })
                },
            )
            .collect();

        let mut results: Vec<(usize, Option<BitVector>)> = Vec::new();
        for handle in decode_futures {
            results.push(handle.await.unwrap());
        }
        let total_elapsed = t0.elapsed().as_secs_f64();
        self.last_latency_secs = total_elapsed - max_delay;

        results.sort_by_key(|(index, _)| *index);
        let mut all_readouts: Option<BitVector> = None;
        for (_, readouts) in results {
            if let Some(r) = readouts {
                match &mut all_readouts {
                    None => all_readouts = Some(r),
                    Some(existing) => bit_vector::append(existing, &r),
                }
            }
        }

        all_readouts
    }

    async fn reset(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let client = self.client.as_mut().unwrap();
        client
            .reset(coordinator::ResetRequest {
                reset_library: false,
                reset_decoder_service: false,
                custom: String::new(),
            })
            .await?;
        Ok(())
    }

    fn simulator_name(&self) -> &'static str {
        "jit_static_simulator"
    }

    fn last_decode_latency_secs(&self) -> f64 {
        self.last_latency_secs
    }
}
