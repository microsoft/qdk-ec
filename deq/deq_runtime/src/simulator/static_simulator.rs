#[cfg(feature = "cli")]
use crate::controller::static_controller::{
    StaticController, static_controller_client::StaticControllerClient, static_controller_server::StaticController as _,
};
#[cfg(feature = "cli")]
use crate::coordinator;
use crate::simulator::DeterministicRng;
use crate::simulator::common::{CommonSimulatorConfig, DelayBatch, Sampler, load_stim_circuit};
#[cfg(feature = "cli")]
use crate::simulator::common::{DecoderClient, ErrorSet, run_simulation_loop};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
#[cfg(feature = "cli")]
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
#[cfg(feature = "cli")]
use structdoc::StructDoc;
#[cfg(feature = "cli")]
use tokio::sync::oneshot::Sender;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct StaticSimulatorConfig {
    /// the filepath to a Stim circuit file
    #[serde(default)]
    pub filepath: String,
    /// Python sampling from typed PROGRAM metadata instead of a Stim file.
    #[cfg(feature = "python")]
    #[serde(default)]
    pub python_sampler: Option<super::python_sampler::PythonSamplerConfig>,
    /// Static library supplying the Python sampler's measurement count and schedule.
    #[cfg(feature = "python")]
    #[serde(default)]
    pub program_filepath: Option<String>,
    /// common simulation configuration
    #[serde(flatten)]
    pub common: CommonSimulatorConfig,
}

pub struct StaticSimulator {
    pub config: StaticSimulatorConfig,
    pub rng: DeterministicRng,
    pub sampler: Box<dyn Sampler>,
    #[cfg_attr(not(feature = "cli"), allow(dead_code))]
    delay_schedule: Vec<DelayBatch>,
    #[cfg_attr(not(feature = "cli"), allow(dead_code))]
    embedded_rhai_script: Option<String>,
}

impl StaticSimulator {
    pub fn new(config: serde_json::Value) -> Self {
        let config: StaticSimulatorConfig = serde_json::from_value(config).unwrap();
        let seed: u64 = config.common.seed.unwrap_or_else(|| rand::rng().next_u64());
        #[cfg(feature = "python")]
        let python_sampling = config.python_sampler.as_ref().map(|sampler| {
            assert!(
                config.filepath.is_empty(),
                "Python PROGRAM sampling cannot also specify a Stim filepath"
            );
            assert!(
                config.common.logical_assert_filepath.is_some(),
                "Python PROGRAM sampling requires logical assertions"
            );
            let path = config
                .program_filepath
                .as_ref()
                .expect("Python PROGRAM sampling requires program_filepath");
            let data = std::fs::read(path).expect("failed to read the sampling program");
            let library: crate::bin::Library = prost::Message::decode(data.as_slice()).expect("invalid sampling program");
            let timing = super::program_timing::validate(
                library
                    .program
                    .iter()
                    .filter_map(|instruction| match instruction.create.as_ref() {
                        Some(crate::bin::instruction::Create::Gadget(gadget)) => Some(gadget),
                        _ => None,
                    }),
                &library.gadget_types,
            )
            .expect("invalid sampling program timing")
            .expect("Python PROGRAM sampling requires typed timing metadata");
            super::python_sampler::from_timed_program(sampler, seed, config.common.skip_shots, timing)
        });
        #[cfg(feature = "python")]
        assert!(
            config.python_sampler.is_some() || config.program_filepath.is_none(),
            "program_filepath requires python_sampler"
        );
        #[cfg(not(feature = "python"))]
        let python_sampling = None;
        let (sampler, delay_schedule, embedded_rhai_script) = python_sampling.unwrap_or_else(|| {
            load_stim_circuit(&config.filepath, seed, config.common.skip_shots, config.common.strict_timing)
        });
        Self {
            config,
            rng: DeterministicRng::seed_from_u64(seed),
            sampler,
            delay_schedule,
            embedded_rhai_script,
        }
    }

    #[cfg(feature = "cli")]
    pub async fn start(self, endpoint: tonic::transport::Endpoint, shutdown: Sender<()>) {
        self.run(endpoint, shutdown, None).await;
    }

    /// Use typed program timing and local delivery in strict timing mode.
    #[cfg(feature = "cli")]
    pub async fn start_with_controller(
        self,
        endpoint: tonic::transport::Endpoint,
        shutdown: Sender<()>,
        controller: Arc<StaticController>,
    ) {
        self.run(endpoint, shutdown, Some(controller)).await;
    }

    #[cfg(feature = "cli")]
    async fn run(
        mut self,
        endpoint: tonic::transport::Endpoint,
        shutdown: Sender<()>,
        controller: Option<Arc<StaticController>>,
    ) {
        #[cfg(feature = "python")]
        assert!(
            self.config.python_sampler.is_none() || controller.is_some(),
            "Python PROGRAM sampling requires a local static controller"
        );
        let rhai_engine = crate::simulator::rhai_assert::RhaiAssertEngine::build(
            &self.config.filepath,
            self.embedded_rhai_script.as_deref(),
            self.config.common.logical_assert_filepath.as_deref(),
        );
        let timing = controller.as_ref().and_then(|controller| {
            super::program_timing::validate(
                controller
                    .library
                    .program
                    .iter()
                    .filter_map(|instruction| match instruction.create.as_ref() {
                        Some(crate::bin::instruction::Create::Gadget(gadget)) => Some(gadget),
                        _ => None,
                    }),
                &controller.library.gadget_types,
            )
            .expect("invalid program timing")
        });
        if let Some(timing) = &timing {
            super::program_timing::validate_stim(timing, &self.delay_schedule).expect("invalid Stim/program timing");
        }
        let schedule = if !self.config.common.strict_timing {
            vec![]
        } else if let Some(timing) = &timing {
            DeliveryBatch::from_program(timing)
        } else {
            self.delay_schedule
                .iter()
                .map(|batch| DeliveryBatch {
                    cumulative_count: batch.cumulative_count,
                    completion: Duration::from_nanos(batch.delay_ns),
                    completed_gadgets: None,
                })
                .collect()
        };
        let mut client = StaticDecoderClient {
            client: None,
            endpoint,
            local: controller.filter(|_| self.config.common.strict_timing),
            schedule,
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
struct DeliveryBatch {
    cumulative_count: usize,
    completion: Duration,
    completed_gadgets: Option<usize>,
}

#[cfg(feature = "cli")]
impl DeliveryBatch {
    fn from_program(timing: &super::program_timing::ProgramTiming) -> Vec<Self> {
        let mut batches: Vec<Self> = Vec::new();
        for (index, (&cumulative_count, &completion)) in
            timing.cumulative_measurements.iter().zip(&timing.completions).enumerate()
        {
            if let Some(previous) = batches.last_mut()
                && previous.completion == completion
            {
                previous.cumulative_count = cumulative_count;
                previous.completed_gadgets = Some(index + 1);
            } else {
                batches.push(Self {
                    cumulative_count,
                    completion,
                    completed_gadgets: Some(index + 1),
                });
            }
        }
        batches
    }
}

#[cfg(feature = "cli")]
struct StaticDecoderClient {
    client: Option<StaticControllerClient<tonic::transport::Channel>>,
    endpoint: tonic::transport::Endpoint,
    local: Option<Arc<StaticController>>,
    schedule: Vec<DeliveryBatch>,
    last_latency_secs: f64,
}

#[cfg(feature = "cli")]
impl StaticDecoderClient {
    async fn deliver(
        &mut self,
        sample: &ErrorSet,
        start: usize,
        end: usize,
        boundary: Option<usize>,
    ) -> Result<Vec<coordinator::Readouts>, tonic::Status> {
        let request = coordinator::Outcomes {
            outcomes: Some(crate::misc::bit_vector::slice(&sample.measurements, start, end - start)),
            loss_mask: sample
                .loss_mask
                .as_ref()
                .map(|mask| crate::misc::bit_vector::slice(mask, start, end - start)),
            ..Default::default()
        };
        let response = if let Some(local) = &self.local {
            local.decode_gadgets_through(tonic::Request::new(request), boundary).await?
        } else {
            self.client.as_mut().unwrap().decode_gadgets(request).await?
        };
        Ok(response.into_inner().readouts)
    }
}

#[cfg(feature = "cli")]
impl DecoderClient for StaticDecoderClient {
    async fn initialize(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.local.is_none() {
            self.client = Some(StaticControllerClient::connect(self.endpoint.clone()).await?);
        }

        Ok(())
    }

    async fn decode(
        &mut self,
        sample: &ErrorSet,
    ) -> Result<Vec<coordinator::Readouts>, Box<dyn std::error::Error + Send + Sync>> {
        self.decode_at(sample, Instant::now()).await
    }

    async fn decode_at(
        &mut self,
        sample: &ErrorSet,
        origin: Instant,
    ) -> Result<Vec<coordinator::Readouts>, Box<dyn std::error::Error + Send + Sync>> {
        self.last_latency_secs = 0.0;
        let total = usize::try_from(sample.measurements.size)?;
        crate::misc::validation::validate_outcomes(
            &sample.measurements,
            sample.loss_mask.as_ref(),
            sample.measurements.size,
        )?;
        if self.schedule.is_empty() {
            let result = self.deliver(sample, 0, total, None).await;
            self.last_latency_secs = origin.elapsed().as_secs_f64();
            return Ok(result?);
        }
        let mut previous = 0;
        let mut previous_completion = Duration::ZERO;
        for batch in &self.schedule {
            if batch.cumulative_count < previous || batch.cumulative_count > total || batch.completion < previous_completion
            {
                return Err("invalid chronological measurement schedule".into());
            }
            origin.checked_add(batch.completion).ok_or("completion deadline overflow")?;
            previous = batch.cumulative_count;
            previous_completion = batch.completion;
        }
        if previous != total {
            return Err("schedule does not consume every sample measurement".into());
        }
        let timer = crate::misc::deadline::DeadlineTimer::new()?;
        let mut previous = 0;
        let mut readouts = vec![];
        for index in 0..self.schedule.len() {
            let batch = &self.schedule[index];
            let end = batch.cumulative_count;
            let boundary = batch.completed_gadgets;
            let mut delivery = crate::decoder::worker_trace::WorkerTrace::root(
                "input_delivery",
                crate::misc::fastrace::system_diagnostics_enabled(),
            );
            delivery.property("scheduled_offset_ns", || batch.completion.as_nanos().to_string());
            delivery.property("completed_gadgets", || format!("{boundary:?}"));
            delivery.property("measurement_count", || (end - previous).to_string());
            timer
                .wait_until(origin.checked_add(batch.completion).ok_or("completion deadline overflow")?)
                .await?;
            delivery.event("timer_ready");
            readouts = self.deliver(sample, previous, end, boundary).await?;
            delivery.event("submitted");
            previous = end;
        }
        self.last_latency_secs = origin.elapsed().saturating_sub(previous_completion).as_secs_f64();
        Ok(readouts)
    }

    async fn reset(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(local) = &self.local {
            local.reset(tonic::Request::new(())).await?;
        } else {
            self.client.as_mut().unwrap().reset(()).await?;
        }
        Ok(())
    }

    fn simulator_name(&self) -> &'static str {
        "static_simulator"
    }

    fn last_decode_latency_secs(&self) -> f64 {
        self.last_latency_secs
    }
}

#[cfg(all(test, feature = "cli"))]
#[path = "../../tests/unit/static_simulator_timing_test.rs"]
mod tests;
