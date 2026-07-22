//! A sampler that drives `stim::TableauSimulator` directly with
//! retry-from-block semantics for `PREPARE { ... REQUIRE ... }` blocks.
//!
//! Instead of sampling the full circuit and filtering, this sampler
//! splits the circuit into [`PreselectBlock`]s and, for each shot,
//! executes them in order — retrying blocks with `REQUIRE` checks
//! until every check passes, running plain (no-check) blocks exactly
//! once.

use crate::misc::bit_vector;
use crate::simulator::DeterministicRng;
use crate::simulator::common::{ErrorSet, Sampler, default_preselect_max_attempts};
use crate::simulator::preselect_directives::{PreselectBlock, PreselectCheck, RequireTarget, extract_preselect_blocks};
use crate::util::BitVector;
use serde::{Deserialize, Serialize};

/// JSON config for the [`crate::simulator::common::SamplerType::Preselect`]
/// backend.
///
/// Separate from [`crate::simulator::preselect_simulator::PreselectSimulatorConfig`],
/// which configures the full LER simulator harness; this one only carries the
/// knobs that apply to the bare sampler.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PreselectSamplerConfig {
    /// Maximum number of retry-from-checkpoint attempts before giving up
    /// on a shot.
    pub preselect_max_attempts: u64,
}

impl Default for PreselectSamplerConfig {
    fn default() -> Self {
        Self {
            preselect_max_attempts: default_preselect_max_attempts(),
        }
    }
}

/// A sampler that uses `stim::TableauSimulator` with retry-from-block.
///
/// A background thread parses the Stim text into
/// [`PreselectBlock`]s and, for each shot, drives the
/// `TableauSimulator` block by block.  Plain (check-less) blocks run
/// once; blocks with `REQUIRE` checks re-run from their opening `{`
/// until every check passes.
///
/// Since `stim::Circuit` and `stim::TableauSimulator` are `!Send`,
/// everything runs inside the background thread.
pub struct TableauPreselectSampler {
    receiver: std::sync::Mutex<std::sync::mpsc::Receiver<Vec<bool>>>,
    request: Option<std::sync::mpsc::SyncSender<()>>,
    total_retries: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl TableauPreselectSampler {
    pub fn new(circuit_text: &str, seed: u64, skip_shots: usize, strict_timing: bool, max_attempts: u64) -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<bool>>(if strict_timing { 0 } else { 16 });
        let circuit_text = circuit_text.to_owned();

        let total_retries = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let retries_ref = total_retries.clone();

        let (request, request_rx) = if strict_timing {
            let (req_tx, req_rx) = std::sync::mpsc::sync_channel::<()>(0);
            (Some(req_tx), Some(req_rx))
        } else {
            (None, None)
        };

        std::thread::Builder::new()
            .name("tableau-preselect-sampler".into())
            .spawn(move || {
                let blocks = extract_preselect_blocks(&circuit_text);
                let parsed: Vec<stim::Circuit> = blocks
                    .iter()
                    .map(|block| {
                        block
                            .stim_text
                            .parse::<stim::Circuit>()
                            .expect("Failed to parse simulation block")
                    })
                    .collect();

                let mut rng = <crate::simulator::DeterministicRng as rand::SeedableRng>::seed_from_u64(seed);

                for shot in 0u64.. {
                    let sim_seed = rand::Rng::next_u64(&mut rng);
                    if shot < skip_shots as u64 {
                        continue;
                    }

                    if let Some(ref req_rx) = request_rx
                        && req_rx.recv().is_err()
                    {
                        break;
                    }

                    let measurements = run_one_shot(&blocks, &parsed, sim_seed, max_attempts, &retries_ref);

                    if tx.send(measurements).is_err() {
                        break;
                    }
                }
            })
            .expect("Failed to spawn tableau preselect sampler thread");

        Self {
            receiver: std::sync::Mutex::new(rx),
            request,
            total_retries,
        }
    }
}

/// Execute one shot, block by block.
///
/// A single `TableauSimulator` is kept for the entire shot.  We
/// maintain `accepted_indices` — a mapping from nominal measurement
/// index (as the circuit expects) to actual index in the simulator's
/// growing record (which includes junk from failed retries).  Every
/// `rec[-k]` target is remapped through this table so that references
/// across blocks stay correct.
fn run_one_shot(
    blocks: &[PreselectBlock],
    parsed: &[stim::Circuit],
    initial_seed: u64,
    max_attempts: u64,
    total_retries: &std::sync::atomic::AtomicU64,
) -> Vec<bool> {
    let mut sim = stim::TableauSimulator::with_seed(initial_seed);
    let mut accepted_indices: Vec<usize> = Vec::new();

    for (block_idx, block) in blocks.iter().enumerate() {
        let circuit = &parsed[block_idx];
        let mut attempts = 0u64;
        loop {
            let new = execute_with_mapping(&mut sim, circuit, &accepted_indices);

            if block.checks.is_empty() {
                accepted_indices.extend(new);
                break;
            }

            let record = sim.current_measurement_record();
            let base_nominal = accepted_indices.len();
            let all_pass = block
                .checks
                .iter()
                .all(|check| resolve_check(check, base_nominal, &new).is_satisfied(|actual_idx| record[actual_idx]));

            if all_pass {
                accepted_indices.extend(new);
                break;
            }

            attempts += 1;
            total_retries.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            if attempts >= max_attempts {
                panic!(
                    "Preselect retry limit ({max_attempts}) exceeded at block {block_idx}. \
                     This usually means the retry region interacts with data qubits \
                     whose state is corrupted by failed attempts. Use --simulator static \
                     (resample mode) instead of --simulator preselect for this circuit."
                );
            }
        }
    }

    let full_record = sim.current_measurement_record();
    accepted_indices.iter().map(|&i| full_record[i]).collect()
}

/// Translate a [`PreselectCheck`] parsed from the block's REQUIRE line
/// into a check whose target indices refer to actual positions in the
/// simulator's growing measurement record.
///
/// The parser stores each target's `abs_meas_idx` in the *nominal*
/// numbering — the index the circuit would see if no retries had ever
/// happened.  During execution we track the actual record indices of
/// the block's fresh measurements in `block_new_actual`; the base
/// nominal index of the block is `base_nominal` (the total number of
/// measurements accepted before the block started).
fn resolve_check(check: &PreselectCheck, base_nominal: usize, block_new_actual: &[usize]) -> PreselectCheck {
    let targets: Vec<RequireTarget> = check
        .targets
        .iter()
        .map(|t| {
            let nominal_offset = t
                .abs_meas_idx
                .checked_sub(base_nominal)
                .expect("REQUIRE target predates the enclosing PREPARE block");
            let actual_idx = *block_new_actual
                .get(nominal_offset)
                .expect("REQUIRE target beyond the block's most recent measurement");
            RequireTarget {
                abs_meas_idx: actual_idx,
                negated: t.negated,
            }
        })
        .collect();
    PreselectCheck { targets }
}

/// Execute `circuit` on `sim`, remapping every `rec[-k]` target through
/// the nominal→actual index mapping in `accepted`.  Returns the actual
/// record indices of the measurements produced by this execution.
fn execute_with_mapping(sim: &mut stim::TableauSimulator, circuit: &stim::Circuit, accepted: &[usize]) -> Vec<usize> {
    let mut new_actual: Vec<usize> = Vec::new();
    let nominal_base = accepted.len();

    for item in circuit {
        match item {
            stim::CircuitItem::Instruction(inst) => {
                let has_rec = inst.targets().iter().any(|t| t.is_measurement_record_target());

                let before = sim.current_measurement_record().len();

                if has_rec {
                    let nominal_now = nominal_base + new_actual.len();

                    let new_targets: Vec<stim::GateTarget> = inst
                        .targets()
                        .iter()
                        .map(|&t| {
                            if t.is_measurement_record_target() {
                                let k = (-t.value()) as usize;
                                let nominal_idx = nominal_now.checked_sub(k).expect("rec target underflow");
                                let actual_idx = if nominal_idx >= nominal_base {
                                    new_actual[nominal_idx - nominal_base]
                                } else {
                                    accepted[nominal_idx]
                                };
                                let new_k = before - actual_idx;
                                stim::GateTarget::rec(-(new_k as i32)).unwrap()
                            } else {
                                t
                            }
                        })
                        .collect();
                    let remapped = stim::CircuitInstruction::new(
                        inst.gate(),
                        new_targets,
                        inst.gate_args().iter().copied(),
                        inst.tag(),
                    )
                    .unwrap();
                    sim.do_circuit(&remapped.to_string().parse::<stim::Circuit>().unwrap());
                } else {
                    sim.do_circuit(&inst.to_string().parse::<stim::Circuit>().unwrap());
                }

                let after = sim.current_measurement_record().len();
                new_actual.extend(before..after);
            }
            stim::CircuitItem::RepeatBlock(_) => {
                panic!(
                    "The preselect simulator does not support REPEAT blocks. \
                     Use --simulator static (resample mode) for circuits with REPEAT."
                );
            }
        }
    }

    new_actual
}

impl Sampler for TableauPreselectSampler {
    fn sample(&self, _rng: &mut DeterministicRng) -> ErrorSet {
        if let Some(ref req) = self.request {
            req.send(()).expect("Tableau preselect sampling thread stopped unexpectedly");
        }
        let rx = self.receiver.lock().unwrap();
        let measurements_bool = rx.recv().expect("Tableau preselect sampling thread stopped unexpectedly");
        ErrorSet {
            errors: vec![],
            measurements: BitVector {
                size: measurements_bool.len() as u64,
                data: bit_vector::pack_bits(&measurements_bool),
            },
            loss_mask: None,
        }
    }

    fn sample_single_error(&self, _index: usize) -> ErrorSet {
        unimplemented!("single error iteration not supported for preselect sampler")
    }

    fn count_single_error(&self) -> usize {
        0
    }

    fn readouts_match(&self, actual: &BitVector, expected: &BitVector) -> bool {
        actual == expected
    }

    fn error_tag(&self, _marginal_index: usize, _error_index: usize) -> &str {
        ""
    }

    fn filtered_count(&self) -> u64 {
        self.total_retries.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_blocks_no_directives() {
        let blocks = extract_preselect_blocks("H 0\nM 0\n");
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].checks.is_empty());
        assert!(blocks[0].stim_text.contains("H 0"));
    }

    #[test]
    fn extract_blocks_with_prepare_block() {
        let text = "\
R 0
H 0
M 0
PREPARE {
H 0
M 0
REQUIRE rec[-1]
}
CZ 0 1
";
        let blocks = extract_preselect_blocks(text);
        assert_eq!(blocks.len(), 3);
        assert!(blocks[0].checks.is_empty());
        assert!(blocks[0].stim_text.contains("M 0"));
        assert_eq!(blocks[1].checks.len(), 1);
        assert_eq!(blocks[1].checks[0].targets.len(), 1);
        assert_eq!(blocks[1].checks[0].targets[0].abs_meas_idx, 1);
        assert!(!blocks[1].checks[0].targets[0].negated);
        assert!(blocks[1].stim_text.contains("M 0"));
        assert!(blocks[2].checks.is_empty());
        assert!(blocks[2].stim_text.contains("CZ 0 1"));
    }

    #[test]
    fn basic_preselect_sampling() {
        let text = "\
R 0
M 0
PREPARE {
H 0
M 0
REQUIRE rec[-1]
}
";
        let sampler = TableauPreselectSampler::new(text, 42, 0, false, 1_000_000);
        let mut rng = <crate::simulator::DeterministicRng as rand::SeedableRng>::seed_from_u64(123);
        let sample = sampler.sample(&mut rng);
        let bits = bit_vector::unpack_bits(&sample.measurements.data, sample.measurements.size);
        assert!(!bits[0]);
        assert!(!bits[1]);
    }

    #[test]
    fn preselect_reports_retries() {
        let text = "\
PREPARE {
H 0
M 0
REQUIRE rec[-1]
}
";
        let sampler = TableauPreselectSampler::new(text, 77, 0, false, 1_000_000);
        let mut rng = <crate::simulator::DeterministicRng as rand::SeedableRng>::seed_from_u64(0);
        for _ in 0..10 {
            let sample = sampler.sample(&mut rng);
            let bits = bit_vector::unpack_bits(&sample.measurements.data, sample.measurements.size);
            assert!(!bits[0], "preselect should enforce measurement == 0");
        }
        let retries = sampler.filtered_count();
        assert!(retries > 0, "10 shots of 50/50 should produce some retries, got 0");
    }

    #[test]
    fn no_preselect_passthrough() {
        let text = "X 0\nM 0\n";
        let sampler = TableauPreselectSampler::new(text, 42, 0, false, 1_000_000);
        let mut rng = <crate::simulator::DeterministicRng as rand::SeedableRng>::seed_from_u64(0);
        let sample = sampler.sample(&mut rng);
        let bits = bit_vector::unpack_bits(&sample.measurements.data, sample.measurements.size);
        assert!(bits[0], "X|0> then M should give 1");
        assert_eq!(sampler.filtered_count(), 0);
    }

    #[test]
    fn deterministic_measurement_no_retry() {
        let text = "\
PREPARE {
R 0
M 0
REQUIRE rec[-1]
}
";
        let sampler = TableauPreselectSampler::new(text, 42, 0, false, 1_000_000);
        let mut rng = <crate::simulator::DeterministicRng as rand::SeedableRng>::seed_from_u64(0);
        for _ in 0..10 {
            sampler.sample(&mut rng);
        }
        assert_eq!(sampler.filtered_count(), 0, "deterministic circuit should never retry");
    }

    #[test]
    fn many_shots_consistent() {
        let text = "\
R 0
PREPARE {
H 0
M 0
REQUIRE rec[-1]
}
H 0
M 0
";
        let sampler = TableauPreselectSampler::new(text, 123, 0, false, 1_000_000);
        let mut rng = <crate::simulator::DeterministicRng as rand::SeedableRng>::seed_from_u64(0);
        for shot in 0..50 {
            let sample = sampler.sample(&mut rng);
            let bits = bit_vector::unpack_bits(&sample.measurements.data, sample.measurements.size);
            assert_eq!(bits.len(), 2, "should have 2 measurements");
            assert!(!bits[0], "shot {shot}: preselected meas 0 should be 0");
        }
    }

    #[test]
    fn multi_target_odd_parity() {
        // `REQUIRE !rec[-1] rec[-2]` succeeds when the XOR of the two
        // measurements is 1 (odd parity).  Measure both qubits in the
        // X basis: |0>, when measured in X, is `(|+> + |->)/sqrt(2)`,
        // so each MX outcome is a fair coin flip.  That makes preselect
        // retries the only mechanism enforcing the assertion — on
        // average half of attempts have even parity and must be
        // retried.
        let text = "\
PREPARE {
R 0 1
MX 0
MX 1
REQUIRE !rec[-1] rec[-2]
}
";
        let sampler = TableauPreselectSampler::new(text, 7, 0, false, 1_000_000);
        let mut rng = <crate::simulator::DeterministicRng as rand::SeedableRng>::seed_from_u64(0);
        for _ in 0..100 {
            let sample = sampler.sample(&mut rng);
            let bits = bit_vector::unpack_bits(&sample.measurements.data, sample.measurements.size);
            assert_eq!(bits.len(), 2);
            assert!(bits[0] ^ bits[1], "odd parity requirement should be satisfied");
        }
        assert!(
            sampler.filtered_count() > 0,
            "with random X-basis outcomes, some shots must have been retried"
        );
    }
}
