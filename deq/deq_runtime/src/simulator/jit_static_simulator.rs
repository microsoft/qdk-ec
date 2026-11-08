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
    /// In strict timing, register gadgets this many seconds before their
    /// measurements arrive instead of registering the entire program upfront.
    #[serde(default)]
    pub registration_lead_seconds: Option<f64>,
    /// common simulation configuration
    #[serde(flatten)]
    pub common: CommonSimulatorConfig,
}

impl JitStaticSimulatorConfig {
    fn validate_registration_timing(&self) -> Result<(), &'static str> {
        if let Some(lead) = self.registration_lead_seconds {
            if !self.common.strict_timing {
                return Err("registration_lead_seconds requires strict_timing");
            }
            if !lead.is_finite() || lead < 0.0 {
                return Err("registration_lead_seconds must be finite and nonnegative");
            }
        }
        Ok(())
    }
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
        config
            .validate_registration_timing()
            .expect("invalid JIT registration timing");
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
            registration_lead_seconds: self.config.registration_lead_seconds,
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
    registration_lead_seconds: Option<f64>,
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
        validate_submission_dependencies(&self.jit_library, &self.expected_gid_to_index)?;
        annotate_remote_references(&mut self.jit_library);

        Ok(())
    }

    async fn decode(
        &mut self,
        sample: &ErrorSet,
    ) -> Result<Vec<coordinator::Readouts>, Box<dyn std::error::Error + Send + Sync>> {
        self.last_latency_secs = 0.0;
        let t0 = std::time::Instant::now();
        // Unresolved readout RPCs must not occupy the connection needed to publish future inputs.
        let decoding = JitControllerClient::connect(self.endpoint.clone()).await?;
        let measurements = sample.measurements.clone();
        let timer = Arc::new(crate::misc::deadline::DeadlineTimer::new()?);

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
                    let registration_lead_seconds = self.registration_lead_seconds;
                    let timer = Arc::clone(&timer);
                    let client = self.client.as_ref().unwrap().clone();
                    let mut decoding = decoding.clone();
                    let gadget_measurements = bit_vector::slice(&measurements, measurement_start, measurement_count);
                    let loss_mask = sample
                        .loss_mask
                        .as_ref()
                        .map(|mask| bit_vector::slice(mask, measurement_start, measurement_count));
                    let gid_signals = Arc::clone(&gid_signals);
                    let expected_gid_to_index = Arc::clone(&self.expected_gid_to_index);

                    let dependency_indices: Vec<usize> = instruction
                        .gadget
                        .as_ref()
                        .unwrap()
                        .submission_dependencies()
                        .map(|gid| expected_gid_to_index[&gid])
                        .collect();

                    async move {
                        if let Some(lead) = registration_lead_seconds {
                            let deadline = t0 + std::time::Duration::from_secs_f64((delay_secs - lead).max(0.0));
                            timer.wait_until(deadline).await?;
                        }
                        for &dep_index in &dependency_indices {
                            let mut rx = gid_signals[dep_index].subscribe();
                            rx.wait_for(|v| v.is_some()).await?;
                        }

                        let mut client = client;

                        let response = client.execute(instruction).await?.into_inner();
                        let gid = response.id;

                        gid_signals[index].send_replace(Some(gid));

                        if delay_secs > 0.0 {
                            timer.wait_until(t0 + std::time::Duration::from_secs_f64(delay_secs)).await?;
                        }

                        let outcomes = coordinator::Outcomes {
                            gid,
                            outcomes: Some(gadget_measurements),
                            modifiers: vec![],
                            loss_mask,
                            decoder_seed: None,
                        };
                        let response = decoding.decode(outcomes).await?.into_inner();
                        Ok::<_, Box<dyn std::error::Error + Send + Sync>>((index, response))
                    }
                },
            )
            .collect();

        let results = futures_util::future::try_join_all(decode_futures).await;
        let total_elapsed = t0.elapsed().as_secs_f64();
        self.last_latency_secs = total_elapsed - max_delay;

        let mut results = results?;
        results.sort_by_key(|(index, _)| *index);
        Ok(results.into_iter().map(|(_, response)| response).collect())
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

#[cfg(feature = "cli")]
fn validate_submission_dependencies(library: &jit::JitLibrary, indices: &HashMap<u64, usize>) -> Result<(), tonic::Status> {
    let mut successors = vec![vec![]; library.program.len()];
    let mut remaining = vec![0; library.program.len()];
    for (index, instruction) in library.program.iter().enumerate() {
        for gid in instruction.gadget.as_ref().unwrap().submission_dependencies() {
            let predecessor = *indices
                .get(&gid)
                .ok_or_else(|| tonic::Status::invalid_argument(format!("unknown predecessor gid={gid}")))?;
            successors[predecessor].push(index);
            remaining[index] += 1;
        }
    }
    let mut ready: std::collections::BinaryHeap<_> = remaining
        .iter()
        .enumerate()
        .filter(|(_, count)| **count == 0)
        .map(|(index, _)| std::cmp::Reverse(index))
        .collect();
    let mut visited = 0;
    while let Some(std::cmp::Reverse(index)) = ready.pop() {
        visited += 1;
        for &successor in &successors[index] {
            remaining[successor] -= 1;
            if remaining[successor] == 0 {
                ready.push(std::cmp::Reverse(successor));
            }
        }
    }
    if visited != library.program.len() {
        return Err(tonic::Status::invalid_argument("cyclic JIT program dependencies"));
    }
    Ok(())
}

#[cfg(feature = "cli")]
fn annotate_remote_references(library: &mut jit::JitLibrary) {
    let mut references = HashMap::<u64, u64>::new();
    for instruction in &library.program {
        let gadget = instruction.gadget.as_ref().unwrap();
        if let Some(remote) = gadget
            .modifier
            .as_ref()
            .and_then(|modifier| modifier.remote_conditional_correction.as_ref())
        {
            for source in remote
                .remote_readouts
                .iter()
                .map(|readout| readout.gid)
                .collect::<hashbrown::HashSet<_>>()
            {
                *references.entry(source).or_default() += 1;
            }
        }
    }
    for (index, instruction) in library.program.iter_mut().enumerate() {
        let gadget = instruction.gadget.as_mut().unwrap();
        let gid = if gadget.gid == 0 {
            u64::try_from(index).unwrap() + 1
        } else {
            gadget.gid
        };
        gadget.expected_remote_references = gadget
            .expected_remote_references
            .max(references.get(&gid).copied().unwrap_or(0));
    }
}

#[cfg(all(test, feature = "cli"))]
mod tests {
    use super::*;

    #[test]
    fn static_programs_hint_future_conditional_readouts() {
        let mut library = future_context_library(5);
        library.program[4].gadget.as_mut().unwrap().modifier = Some(crate::bin::GadgetModifier {
            remote_conditional_correction: Some(crate::bin::RemoteConditionalCorrection {
                remote_readouts: vec![crate::bin::remote_conditional_correction::RemoteReadout {
                    gid: 1,
                    readout_index: 0,
                }],
                ..Default::default()
            }),
            ..Default::default()
        });
        annotate_remote_references(&mut library);
        assert_eq!(library.program[0].gadget.as_ref().unwrap().expected_remote_references, 1);
        for instruction in &library.program[1..] {
            assert_eq!(instruction.gadget.as_ref().unwrap().expected_remote_references, 0);
        }
        library.program[0].gadget.as_mut().unwrap().expected_remote_references = u64::MAX;
        annotate_remote_references(&mut library);
        assert_eq!(
            library.program[0].gadget.as_ref().unwrap().expected_remote_references,
            u64::MAX
        );
    }

    fn future_context_library(gadget_count: u64) -> jit::JitLibrary {
        use crate::bin;
        use crate::misc::bit_matrix::zeros;

        jit::JitLibrary {
            port_types: vec![jit::JitPortType {
                base: Some(bin::PortType {
                    ptype: 1,
                    ..Default::default()
                }),
                ..Default::default()
            }],
            gadget_types: (1..=3)
                .map(|gtype| jit::JitGadgetType {
                    base: Some(bin::GadgetType {
                        gtype,
                        inputs: if gtype == 1 {
                            vec![]
                        } else {
                            vec![bin::gadget_type::Port {
                                ptype: 1,
                                ..Default::default()
                            }]
                        },
                        outputs: if gtype == 3 {
                            vec![]
                        } else {
                            vec![bin::gadget_type::Port {
                                ptype: 1,
                                ..Default::default()
                            }]
                        },
                        measurements: vec![bin::gadget_type::Measurement::default()],
                        correction_propagation: Some(zeros(0, 1)),
                        readout_propagation: Some(zeros(0, 1)),
                        logical_correction: Some(zeros(0, 0)),
                        physical_correction: Some(zeros(0, 1)),
                        ..Default::default()
                    }),
                    finished_checks: vec![jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![jit::jit_gadget_type::PresentMeasurement::default()],
                    }],
                    errors: vec![jit::jit_gadget_type::Error {
                        base: Some(bin::error_model_type::Error {
                            probability: 0.1,
                            ..Default::default()
                        }),
                        finished_checks: vec![0],
                        ..Default::default()
                    }],
                    ..Default::default()
                })
                .collect(),
            program: (1..=gadget_count)
                .map(|gid| jit::JitInstruction {
                    gadget: Some(bin::Gadget {
                        gid,
                        gtype: if gid == 1 {
                            1
                        } else if gid == gadget_count {
                            3
                        } else {
                            2
                        },
                        connectors: if gid == 1 {
                            vec![]
                        } else {
                            vec![bin::gadget::Connector { gid: gid - 1, port: 0 }]
                        },
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn future_inputs_progress_while_graph_slots_are_full() {
        for coordinator in ["window", "monolithic"] {
            for workers in [1, 2] {
                for lead in [None, Some(0.002)] {
                    check_future_context_capacity(coordinator, lead, false, workers).await;
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reset_cancels_ready_work_waiting_for_graph_slots() {
        for coordinator in ["window", "monolithic"] {
            check_future_context_capacity(coordinator, None, true, 1).await;
        }
    }

    #[allow(
        clippy::too_many_lines,
        reason = "Keep the shared saturation, cancellation, and reset scenario together."
    )]
    async fn check_future_context_capacity(
        coordinator_name: &str,
        registration_lead_seconds: Option<f64>,
        cancel: bool,
        workers: u32,
    ) {
        use crate::coordinator::{CoordinatorClient, DynCoordinator};
        use crate::server::ServerConfigs;
        use clap::Parser;
        use std::time::Duration;

        const GADGETS: u64 = 65;
        let library = future_context_library(GADGETS);
        let decoder_config = serde_json::json!({"parallel": workers}).to_string();
        let config = ServerConfigs::try_parse_from([
            "server",
            "--decoder",
            "black-box-relay-bp",
            "--decoder-config",
            &decoder_config,
            "--coordinator",
            coordinator_name,
            "--coordinator-config",
            if coordinator_name == "window" {
                "{\"buffer_radius\":128,\"lookahead_radius\":0,\"persistent_decoder\":false}"
            } else {
                "{\"persistent_decoder\":false}"
            },
            "--controller",
            "jit",
        ])
        .unwrap();
        let server = config.build_local().await;
        server.jit_controller().unwrap().load_library(library.clone()).await.unwrap();
        let CoordinatorClient::Local(coordinator) = server.coordinator_client() else {
            panic!("expected a local coordinator");
        };
        let (slots, decoder) = match &coordinator {
            DynCoordinator::Window(window) => (&window.decoding_slots, &window.decoder),
            DynCoordinator::Monolithic(monolithic) => (&monolithic.decoding_slots, &monolithic.decoder),
            _ => panic!("expected a decoding coordinator"),
        };
        let capacity = 2 * workers;
        assert_eq!(slots.available_permits(), usize::try_from(capacity).unwrap());
        let graph_slot = slots.acquire_many(capacity).await.unwrap();
        let endpoint =
            tonic::transport::Endpoint::from_shared(server.bind_grpc(([127, 0, 0, 1], 0).into()).await.unwrap()).unwrap();
        let mut client = JitDecoderClient {
            client: None,
            endpoint,
            jit_library: library,
            measurement_ranges: vec![],
            expected_gid_to_index: Arc::new(HashMap::new()),
            delay_schedule: if registration_lead_seconds.is_some() {
                (1..=GADGETS)
                    .map(|gid| DelayBatch {
                        cumulative_count: usize::try_from(gid).unwrap(),
                        delay_seconds: f64::from(u32::try_from(gid).unwrap()) * 0.001,
                    })
                    .collect()
            } else {
                vec![]
            },
            last_latency_secs: 0.0,
            registration_lead_seconds,
        };
        client.initialize().await.unwrap();
        let sample = ErrorSet {
            measurements: bit_vector::from_sparse_indices(GADGETS, &[GADGETS - 1]),
            loss_mask: None,
            errors: vec![],
        };
        let decoding = tokio::spawn(async move {
            let result = client.decode(&sample).await;
            (client, sample, result)
        });
        let supplied = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let supplied = match &coordinator {
                    DynCoordinator::Window(window) => window
                        .gadgets
                        .read()
                        .await
                        .get(&GADGETS)
                        .is_some_and(|gadget| gadget.outcomes.borrow().is_some()),
                    DynCoordinator::Monolithic(monolithic) => {
                        server.jit_controller().unwrap().compiler.contains_gid(GADGETS).await
                            && monolithic.gadgets.read().await.is_empty()
                    }
                    _ => unreachable!(),
                };
                if supplied {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await;
        supplied.expect("occupied graph slots must not prevent publishing future inputs");
        let crate::decoder::DynDecoder::BlackBoxRelayBP(decoder) = decoder else {
            panic!("expected a pooled decoder");
        };
        assert_eq!(
            *decoder.pending_decodes().borrow(),
            0,
            "graph saturation must defer backend construction"
        );
        assert!(!decoding.is_finished());
        if cancel {
            tokio::time::timeout(
                Duration::from_secs(10),
                server.jit_controller().unwrap().reset(coordinator::ResetRequest::default()),
            )
            .await
            .unwrap()
            .unwrap();
        }
        drop(graph_slot);
        let (mut client, sample, result) = tokio::time::timeout(Duration::from_secs(10), decoding)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(slots.available_permits(), usize::try_from(capacity).unwrap());
        if cancel {
            assert!(result.is_err());
        } else {
            assert!(result.is_ok());
        }
        client.reset().await.unwrap();
        let repeated = tokio::time::timeout(Duration::from_secs(10), client.decode(&sample))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            repeated.iter().map(|readout| readout.gid).collect::<Vec<_>>(),
            (1..=GADGETS).collect::<Vec<_>>()
        );
        assert_eq!(repeated.iter().map(|readout| readout.correction_count).sum::<u64>(), 1);
        if let Ok(readouts) = result {
            assert_eq!(repeated, readouts);
        }
        server.shutdown().await.unwrap();
    }

    #[test]
    fn registration_timing_requires_strict_finite_nonnegative_lead() {
        for (strict_timing, lead, valid) in [
            (false, None, true),
            (true, None, true),
            (true, Some(0.0), true),
            (true, Some(2.0), true),
            (false, Some(1.0), false),
            (true, Some(-1.0), false),
            (true, Some(f64::INFINITY), false),
            (true, Some(f64::NAN), false),
        ] {
            let mut config: JitStaticSimulatorConfig = serde_json::from_value(serde_json::json!({
                "filepath": "unused",
                "jit_library_filepath": "unused",
                "strict_timing": strict_timing,
            }))
            .unwrap();
            config.registration_lead_seconds = lead;
            assert_eq!(config.validate_registration_timing().is_ok(), valid);
        }
    }

    #[test]
    fn submission_dependencies_accept_forward_references_but_reject_cycles_and_unknown_ids() {
        let mut library = jit::JitLibrary {
            program: [3, 1, 2]
                .into_iter()
                .map(|gid| jit::JitInstruction {
                    gadget: Some(crate::bin::Gadget {
                        gid,
                        connectors: if gid == 1 {
                            vec![]
                        } else {
                            vec![crate::bin::gadget::Connector { gid: gid - 1, port: 0 }]
                        },
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        };
        let indices = HashMap::from([(3, 0), (1, 1), (2, 2)]);
        validate_submission_dependencies(&library, &indices).unwrap();
        library.program[1]
            .gadget
            .as_mut()
            .unwrap()
            .connectors
            .push(crate::bin::gadget::Connector { gid: 3, port: 0 });
        assert_eq!(
            validate_submission_dependencies(&library, &indices).unwrap_err().code(),
            tonic::Code::InvalidArgument
        );
        library.program[1].gadget.as_mut().unwrap().connectors[0].gid = 99;
        assert_eq!(
            validate_submission_dependencies(&library, &indices).unwrap_err().code(),
            tonic::Code::InvalidArgument
        );
    }
}
