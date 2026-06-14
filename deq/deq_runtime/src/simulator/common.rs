//! Common simulation functionality shared between static and JIT static simulators.

#[cfg(feature = "simulator")]
use crate::SIGNAL_CHECKER;
#[cfg(feature = "simulator")]
use crate::misc::bit_vector::{self, bit_vector_to_string};
#[cfg(feature = "simulator")]
use crate::misc::fastrace::{Event, Span, SpanContext};
use crate::simulator::DeterministicRng;
use crate::util::BitVector;
#[cfg(all(feature = "cli", feature = "simulator"))]
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use serde::{Deserialize, Serialize};
#[cfg(all(feature = "cli", feature = "simulator"))]
use std::io::IsTerminal;
#[cfg(feature = "simulator")]
use std::time::Instant;
#[cfg(feature = "cli")]
use structdoc::StructDoc;
#[cfg(feature = "simulator")]
use tokio::sync::oneshot::Sender;

/// Common configuration fields shared by all static-style simulators.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct CommonSimulatorConfig {
    /// the optional seed for the random number generator
    pub seed: Option<u64>,
    /// skipping the first N shots (without actually decoding them)
    #[serde(default)]
    pub skip_shots: usize,
    /// the number of shots to simulate, by default 4000; setting it to 0 means
    /// infinite shots until reaching the error limit
    #[serde(default = "default_shots")]
    pub shots: usize,
    /// the number of logical errors before terminating, by default 4000; setting it to 0 means
    /// infinite errors until reaching the shot limit
    #[serde(default = "default_shots")]
    pub errors: usize,
    /// iterate single errors, helpful for testing if a decoder works properly
    #[serde(default)]
    pub iterate_single_error: bool,
    #[serde(default)]
    pub print_all: bool,
    #[serde(default)]
    pub print_on_error: bool,
    /// when true, (1) sampling is synchronous with decoding so that decoder
    /// timing measurements are not distorted by concurrent sampling work,
    /// and (2) measurement delays specified via ``#!delay`` directives in
    /// the Stim file are respected (the simulator sleeps before delivering
    /// each batch); when false, all measurements are delivered at once and
    /// the sampler runs ahead in the background for throughput
    #[serde(default)]
    pub strict_timing: bool,
    /// optional path to a Rhai script that defines
    /// `fn is_logical_error(shot, readouts, expected_readouts, measurements)`;
    /// when set, the script is the sole arbiter of logical errors instead of
    /// the built-in readout comparison
    #[serde(default)]
    pub logical_assert_filepath: Option<String>,
    /// Maximum number of resample attempts when preselect checks fail.
    /// Only used when the Stim circuit contains `#!preselect_expect` directives.
    #[serde(default = "default_preselect_max_attempts")]
    pub preselect_max_attempts: u64,
}

pub fn default_preselect_max_attempts() -> u64 {
    1_000_000
}

pub fn default_shots() -> usize {
    4000
}

/// Trait for decoder clients that can be used with the common simulation loop.
#[allow(async_fn_in_trait)]
pub trait DecoderClient: Send {
    /// Initialize the client connection and any setup required before simulation.
    async fn initialize(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Decode a sample and return the readouts.
    async fn decode(&mut self, sample: &ErrorSet) -> Option<BitVector>;

    /// Reset the decoder state for the next shot.
    async fn reset(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Get the name of the simulator for logging purposes.
    fn simulator_name(&self) -> &'static str;

    /// Return the latency of the last decode batch (time from sending the final
    /// measurement batch to receiving readouts). For batch mode this equals the
    /// total decode time; for streaming mode it measures only the final blocking
    /// call. Returns 0.0 if not applicable.
    fn last_decode_latency_secs(&self) -> f64 {
        0.0
    }
}

/// A delay schedule entry: send measurements [0..count) at the given delay.
#[derive(Debug, Clone)]
pub struct DelayBatch {
    /// Cumulative measurement count up to and including this batch.
    pub cumulative_count: usize,
    /// Time in seconds at which this batch becomes available.
    pub delay_seconds: f64,
}

/// Run the common simulation loop with the given decoder client.
#[cfg(feature = "simulator")]
pub async fn run_simulation_loop<C: DecoderClient>(
    config: &CommonSimulatorConfig,
    sampler: &dyn Sampler,
    rng: &mut DeterministicRng,
    client: &mut C,
    shutdown: Sender<()>,
    rhai_engine: &crate::simulator::rhai_assert::RhaiAssertEngine,
) {
    // Initialize the client
    if let Err(e) = client.initialize().await {
        eprintln!("Failed to initialize client: {e}");
        shutdown.send(()).unwrap();
        return;
    }

    let mut max_shots = if config.shots == 0 { usize::MAX } else { config.shots };
    let mut max_errors = if config.errors == 0 { usize::MAX } else { config.errors };
    if config.iterate_single_error {
        max_shots = sampler.count_single_error();
        max_errors = max_shots;
    }

    // Progress bar setup (only with indicatif feature)
    #[cfg(feature = "cli")]
    let (multi, shot_bar, error_bar, stats_bar) = {
        let is_tty = std::io::stdout().is_terminal() && std::io::stderr().is_terminal();
        let multi = if is_tty {
            MultiProgress::new()
        } else {
            MultiProgress::with_draw_target(ProgressDrawTarget::hidden())
        };
        let shot_bar = ProgressBar::new(max_shots as u64);
        shot_bar.set_style(ProgressStyle::with_template(" shots: [{eta_precise}] {wide_bar} {pos}/{len}").unwrap());
        multi.add(shot_bar.clone());
        let error_bar = ProgressBar::new(max_errors as u64);
        error_bar.set_style(ProgressStyle::with_template("errors: [{eta_precise}] {wide_bar} {pos}/{len}").unwrap());
        multi.add(error_bar.clone());
        let stats_bar = ProgressBar::new(1_u64);
        stats_bar.set_style(ProgressStyle::with_template("elapsed: [{elapsed_precise}] {msg}").unwrap());
        multi.add(stats_bar.clone());
        (multi, shot_bar, error_bar, stats_bar)
    };

    for _ in 0..config.skip_shots {
        #[cfg(feature = "cli")]
        shot_bar.inc(1);
        rng.jump();
    }

    let mut decode_elapsed = 0.0;
    let mut latency_elapsed = 0.0;
    let mut actual_shots = 0;
    let mut logical_errors = 0;
    let mut interrupted = false;
    let simulator_name = client.simulator_name();

    for shot in config.skip_shots..max_shots {
        // Check for Ctrl+C signal
        if SIGNAL_CHECKER.check().is_err() {
            println!("\nInterrupted by user (Ctrl+C)");
            interrupted = true;
            break;
        }

        let span = Span::root(simulator_name, SpanContext::random());
        span.add_property(|| ("shot", format!("{shot}")));
        actual_shots += 1;
        #[cfg(feature = "cli")]
        shot_bar.inc(1);

        // Sample errors
        let sample = if config.iterate_single_error {
            sampler.sample_single_error(shot)
        } else {
            sampler.sample(rng)
        };
        span.add_property(|| ("sample", format!("{sample:?}")));
        rng.jump();

        // Decode
        span.add_event(Event::new("start_decoding"));
        let decode_start = Instant::now();
        let readouts = client.decode(&sample).await;
        decode_elapsed += decode_start.elapsed().as_secs_f64();
        let shot_latency = client.last_decode_latency_secs();
        latency_elapsed += shot_latency;
        span.add_property(|| ("last_decode_latency_secs", format!("{shot_latency}")));

        // Process results
        span.add_event(Event::new("process_result"));
        let is_logical_error = rhai_engine.is_logical_error(shot, readouts.as_ref(), &sample.measurements);
        if is_logical_error {
            logical_errors += 1;
            #[cfg(feature = "cli")]
            error_bar.inc(1);
        }
        span.add_property(|| ("readouts", format!("{readouts:?}")));
        span.add_property(|| ("error", (if is_logical_error { "1" } else { "0" }).to_string()));

        if config.print_all || (config.print_on_error && is_logical_error) {
            let logical_str = if is_logical_error { "(error)" } else { "" };
            let readouts_str = readouts
                .as_ref()
                .map(bit_vector_to_string)
                .unwrap_or_else(|| "None".to_string());
            let physical_str = sample
                .errors
                .iter()
                .map(|(mi, ei)| format!("({mi},{ei})'{}'", sampler.error_tag(*mi, *ei)))
                .collect::<Vec<_>>()
                .join(",");
            let message = format!(
                "\n[{}]{}: readouts:{}, physical_errors:{}",
                shot, logical_str, readouts_str, physical_str
            );
            #[cfg(feature = "cli")]
            let _ = multi.println(message);
            #[cfg(not(feature = "cli"))]
            println!("{}", message);
        }

        // Reset for next shot
        span.add_event(Event::new("reset"));
        if let Err(e) = client.reset().await {
            eprintln!("Failed to reset client: {e}");
            break;
        }

        // Update progress bar
        #[cfg(feature = "cli")]
        {
            error_bar.tick();
            let error_rate: f64 = (logical_errors as f64) / (actual_shots as f64);
            let confidence_interval_95_percent =
                1.96 * (error_rate * (1. - error_rate) / (actual_shots as f64)).sqrt() / error_rate;
            stats_bar.set_message(format!(
                "decoding time: {:.3e}s ({:.3e}s per shots), logical error rate: {:.3e} ± {:.1e}",
                decode_elapsed,
                decode_elapsed / (actual_shots as f64),
                error_rate,
                confidence_interval_95_percent
            ));
        }

        if logical_errors >= max_errors {
            #[cfg(feature = "cli")]
            {
                shot_bar.force_draw();
                error_bar.force_draw();
                stats_bar.force_draw();
            }
            break;
        }
    }

    // Print summary
    let status = if interrupted { "Interrupted" } else { "Complete" };
    let error_rate: f64 = if actual_shots > 0 {
        (logical_errors as f64) / (actual_shots as f64)
    } else {
        0.0
    };
    let confidence_interval = if actual_shots > 0 {
        1.96 * (error_rate * (1. - error_rate) / (actual_shots as f64)).sqrt()
    } else {
        0.0
    };
    println!("=== Simulation {status} ===");
    if max_shots == usize::MAX {
        println!("  Shots: {}", actual_shots);
    } else {
        println!("  Shots: {}/{}", actual_shots, max_shots);
    }
    if max_errors == usize::MAX {
        println!("  Logical errors: {}", logical_errors);
    } else {
        println!("  Logical errors: {}/{}", logical_errors, max_errors);
    }
    let retries = sampler.filtered_count();
    if retries > 0 {
        let total = retries + actual_shots as u64;
        let pct = 100.0 * retries as f64 / total.max(1) as f64;
        println!("  Retries: {retries} ({pct:.2}%)");
    }
    println!("  Error rate: {:.6e} ± {:.2e}", error_rate, confidence_interval);
    println!(
        "  Decoding time: {:.3}s ({:.3e}s per shot)",
        decode_elapsed,
        if actual_shots > 0 {
            decode_elapsed / (actual_shots as f64)
        } else {
            0.0
        }
    );
    if latency_elapsed > 0.0 {
        println!(
            "  Last-batch latency: {:.3}s ({:.3e}s per shot)",
            latency_elapsed,
            if actual_shots > 0 {
                latency_elapsed / (actual_shots as f64)
            } else {
                0.0
            }
        );
    }

    // Don't drop the progress bar so that the final printings are preserved
    #[cfg(feature = "cli")]
    {
        std::mem::forget(multi);
        std::mem::forget(shot_bar);
        std::mem::forget(error_bar);
        std::mem::forget(stats_bar);
    }

    let _ = std::io::Write::flush(&mut std::io::stdout());

    let _ = std::io::Write::flush(&mut std::io::stdout());

    shutdown.send(()).unwrap();
}

#[derive(Clone, Debug)]
pub struct ErrorSet {
    /// (index of the marginal, index of the error in the marginal)
    pub errors: Vec<(usize, usize)>,
    pub measurements: BitVector,
    pub expected_readouts: BitVector,
}

/// Trait for measurement samplers used by the simulation loop.
pub trait Sampler: Send + Sync {
    /// Sample a set of errors.
    fn sample(&self, rng: &mut DeterministicRng) -> ErrorSet;

    /// Return the error set for a specific single error by index.
    fn sample_single_error(&self, index: usize) -> ErrorSet;

    /// Total number of individual errors across all marginals.
    fn count_single_error(&self) -> usize;

    /// Check if actual readouts match expected, respecting the readout mask.
    fn readouts_match(&self, actual: &BitVector, expected: &BitVector) -> bool;

    /// Get the tag string for a specific error in a specific marginal.
    fn error_tag(&self, marginal_index: usize, error_index: usize) -> &str;

    /// Number of samples that were filtered out (e.g. due to preselect
    /// failures).  Returns 0 by default.
    fn filtered_count(&self) -> u64 {
        0
    }
}

/// Load a Stim circuit file and return the sampler, delay schedule, and
/// any embedded Rhai script.
#[cfg(feature = "simulator")]
pub fn load_stim_circuit(
    filepath: &str,
    seed: u64,
    skip_shots: usize,
    strict_timing: bool,
) -> (Box<dyn Sampler>, Vec<DelayBatch>, Option<String>) {
    let circuit_text =
        std::fs::read_to_string(filepath).unwrap_or_else(|e| panic!("Failed to read Stim circuit file '{filepath}': {e}"));
    let embedded_rhai_script = crate::simulator::rhai_assert::extract_rhai_script(&circuit_text);
    let sampler = StimSampler::new(&circuit_text, seed, skip_shots, strict_timing);
    let delay_schedule = {
        let circuit: stim::Circuit = circuit_text
            .parse()
            .expect("Failed to parse Stim circuit for measurement counting");
        let expected = usize::try_from(circuit.num_measurements()).expect("Stim circuit measurement count exceeds usize");
        crate::simulator::stim_delays::extract_delay_schedule(&circuit_text, expected)
    };
    let preselect_schedule = crate::simulator::preselect_directives::extract_preselect_schedule(&circuit_text);
    let sampler: Box<dyn Sampler> = if preselect_schedule.is_empty() {
        Box::new(sampler)
    } else {
        Box::new(ResamplePreselectSampler::new(
            Box::new(sampler),
            preselect_schedule,
            1_000_000,
        ))
    };
    (sampler, delay_schedule, embedded_rhai_script)
}

/// A wrapper sampler that resamples until all preselect checks pass.
///
/// This is used by the static and jit-static simulators: after drawing a
/// sample from the inner sampler, it checks the measurement record against
/// the preselect schedule and retries (up to `max_attempts`) if any check
/// fails.
#[cfg(feature = "simulator")]
pub struct ResamplePreselectSampler {
    inner: Box<dyn Sampler>,
    schedule: crate::simulator::preselect_directives::PreselectSchedule,
    max_attempts: u64,
    filtered: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "simulator")]
impl ResamplePreselectSampler {
    pub fn new(
        inner: Box<dyn Sampler>,
        schedule: crate::simulator::preselect_directives::PreselectSchedule,
        max_attempts: u64,
    ) -> Self {
        Self {
            inner,
            schedule,
            max_attempts,
            filtered: std::sync::atomic::AtomicU64::new(0),
        }
    }

    fn checks_pass(&self, measurements: &BitVector) -> bool {
        for check in &self.schedule.checks {
            let idx = check.abs_meas_idx;
            if idx as u64 >= measurements.size {
                return false;
            }
            let actual = bit_vector::get_bit(measurements, idx as u64);
            if actual != check.expected {
                return false;
            }
        }
        true
    }
}

#[cfg(feature = "simulator")]
impl Sampler for ResamplePreselectSampler {
    fn sample(&self, rng: &mut DeterministicRng) -> ErrorSet {
        for _ in 0..self.max_attempts {
            let sample = self.inner.sample(rng);
            if self.checks_pass(&sample.measurements) {
                return sample;
            }
            self.filtered.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        panic!(
            "Preselect checks failed after {} attempts; \
             consider increasing preselect_max_attempts",
            self.max_attempts
        );
    }

    fn sample_single_error(&self, index: usize) -> ErrorSet {
        self.inner.sample_single_error(index)
    }

    fn count_single_error(&self) -> usize {
        self.inner.count_single_error()
    }

    fn readouts_match(&self, actual: &BitVector, expected: &BitVector) -> bool {
        self.inner.readouts_match(actual, expected)
    }

    fn error_tag(&self, marginal_index: usize, error_index: usize) -> &str {
        self.inner.error_tag(marginal_index, error_index)
    }

    fn filtered_count(&self) -> u64 {
        self.filtered.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// A sampler that pulls measurements from a Stim circuit via a background
/// thread.  The thread parses the circuit text, builds a `stim::MeasurementSampler`,
/// and streams one shot at a time through a channel.  The thread stops
/// automatically when the `StimSampler` (and its receiver) is dropped.
#[cfg(feature = "simulator")]
pub struct StimSampler {
    receiver: std::sync::Mutex<std::sync::mpsc::Receiver<Vec<bool>>>,
    /// In strict-timing mode, a request is sent before each sample so the
    /// background thread only works on-demand.
    request: Option<std::sync::mpsc::SyncSender<()>>,
}

#[cfg(feature = "simulator")]
impl StimSampler {
    /// Create a new stim sampler.
    ///
    /// When `strict_timing` is true the background thread only samples
    /// on-demand (one request → one sample) so it does not consume CPU while
    /// the decoder is running, keeping timing measurements accurate.  When
    /// false the thread samples ahead into a bounded buffer for throughput.
    pub fn new(circuit_text: &str, seed: u64, skip_shots: usize, strict_timing: bool) -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<bool>>(if strict_timing { 0 } else { 16 });
        let circuit_text = circuit_text.to_owned();

        let (request, request_rx) = if strict_timing {
            let (req_tx, req_rx) = std::sync::mpsc::sync_channel::<()>(0);
            (Some(req_tx), Some(req_rx))
        } else {
            (None, None)
        };

        std::thread::Builder::new()
            .name("stim-sampler".into())
            .spawn(move || {
                let circuit: stim::Circuit = circuit_text.parse().expect("Failed to parse Stim circuit");
                let mut sampler = circuit.compile_sampler_with_seed(false, seed);

                for _ in 0..skip_shots {
                    let _ = sampler.sample(1);
                }

                loop {
                    if let Some(ref req_rx) = request_rx
                        && req_rx.recv().is_err()
                    {
                        break;
                    }

                    let batch = sampler.sample(1);
                    let shot: Vec<bool> = batch.row(0).iter().copied().collect();
                    if tx.send(shot).is_err() {
                        break;
                    }
                }
            })
            .expect("Failed to spawn stim sampler thread");

        Self {
            receiver: std::sync::Mutex::new(rx),
            request,
        }
    }

    fn recv_sample(&self) -> ErrorSet {
        if let Some(ref req) = self.request {
            req.send(()).expect("Stim sampling thread stopped unexpectedly");
        }
        let rx = self.receiver.lock().unwrap();
        let measurements_bool = rx.recv().expect("Stim sampling thread stopped unexpectedly");
        ErrorSet {
            errors: vec![],
            measurements: BitVector {
                size: measurements_bool.len() as u64,
                data: bit_vector::pack_bits(&measurements_bool),
            },
            expected_readouts: BitVector { size: 0, data: vec![] },
        }
    }
}

#[cfg(feature = "simulator")]
impl Sampler for StimSampler {
    fn sample(&self, _rng: &mut DeterministicRng) -> ErrorSet {
        self.recv_sample()
    }

    fn sample_single_error(&self, _index: usize) -> ErrorSet {
        panic!("sample_single_error is not supported for StimSampler")
    }

    fn count_single_error(&self) -> usize {
        panic!("count_single_error is not supported for StimSampler")
    }

    fn readouts_match(&self, _actual: &BitVector, _expected: &BitVector) -> bool {
        true // readout assertion not yet implemented for stim
    }

    fn error_tag(&self, _marginal_index: usize, _error_index: usize) -> &str {
        ""
    }
}

#[cfg(all(test, feature = "simulator"))]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn resample_preselect_filters_and_counts() {
        // Circuit: H 0 then M 0 — measurement is random 50/50.
        // Preselect expects measurement 0 == false (i.e. result 0).
        // About half the samples should be rejected.
        let circuit_text = "H 0\nM 0\n#!preselect_expect 0 0\n";

        let inner = StimSampler::new(circuit_text, 42, 0, false);
        let schedule = crate::simulator::preselect_directives::extract_preselect_schedule(circuit_text);
        assert!(!schedule.is_empty(), "schedule should have 1 check");

        let sampler = ResamplePreselectSampler::new(Box::new(inner), schedule, 1_000_000);

        let mut rng = DeterministicRng::seed_from_u64(99);
        for _ in 0..20 {
            let sample = sampler.sample(&mut rng);
            // Measurement 0 must always be false (preselected to 0)
            let bit = bit_vector::get_bit(&sample.measurements, 0);
            assert!(!bit, "preselect should enforce measurement 0 == false");
        }

        let filtered = sampler.filtered_count();
        assert!(
            filtered > 0,
            "with 50/50 outcomes over 20 shots, some should have been filtered; got {filtered}"
        );
        println!("Filtered {filtered} samples out of 20 successful shots");
    }
}
