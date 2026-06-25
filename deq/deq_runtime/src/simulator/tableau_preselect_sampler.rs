//! A sampler that drives `stim::TableauSimulator` directly with
//! retry-from-BEGIN semantics for preselect checks.
//!
//! Instead of sampling the full circuit and filtering, this sampler
//! splits the circuit at `#!preselect_begin` boundaries and replays
//! only the failing segment on preselect failure.

use crate::misc::bit_vector;
use crate::simulator::DeterministicRng;
use crate::simulator::common::{ErrorSet, Sampler};
use crate::util::BitVector;

/// A sampler that uses `stim::TableauSimulator` with retry-from-BEGIN.
///
/// A background thread parses the Stim text, splits it into segments
/// at `#!preselect_begin` markers, and for each shot drives the
/// `TableauSimulator` segment by segment.  When a `#!preselect_expect`
/// check fails, the simulator replays from the most recent BEGIN.
///
/// Since `stim::Circuit` and `stim::TableauSimulator` are `!Send`,
/// everything runs inside the background thread.
pub struct TableauPreselectSampler {
    receiver: std::sync::Mutex<std::sync::mpsc::Receiver<Vec<bool>>>,
    request: Option<std::sync::mpsc::SyncSender<()>>,
    total_retries: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

/// A preselect check parsed from the Stim text.
struct SegmentCheck {
    /// Absolute measurement index in the full circuit.
    abs_meas_idx: usize,
    expected: bool,
}

/// A contiguous block between two `#!preselect_begin` markers.
///
/// Split into a *retry* portion (BEGIN → last EXPECT) and a *tail*
/// (last EXPECT → next BEGIN).  Only the retry portion is re-executed
/// on check failure; the tail runs once after all checks pass.
struct Segment {
    /// Circuit text for the retryable portion (up to the last check).
    /// Empty when the segment has no checks.
    retry_stim_text: String,
    /// Checks verified after executing the retry portion.
    checks: Vec<SegmentCheck>,
    /// Circuit text after the last check — entangles with data qubits
    /// and must execute exactly once.
    tail_stim_text: String,
}

/// Parse the full Stim circuit text into segments.
fn parse_segments(stim_text: &str) -> Vec<Segment> {
    let mut segments: Vec<Segment> = Vec::new();
    let mut current_lines: Vec<&str> = Vec::new();
    let mut current_checks: Vec<SegmentCheck> = Vec::new();
    let mut last_check_line_pos: Option<usize> = None;

    for line in stim_text.lines() {
        let trimmed = line.trim();

        if trimmed == "#!preselect_begin" {
            flush_segment(
                &mut segments,
                &mut current_lines,
                &mut current_checks,
                &mut last_check_line_pos,
            );
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix("#!preselect_expect") {
            let rest = rest.trim();
            let mut parts = rest.split_whitespace();
            let abs_idx: usize = parts
                .next()
                .and_then(|s| s.parse().ok())
                .expect("invalid #!preselect_expect abs_meas_idx");
            let expected_int: u8 = parts
                .next()
                .and_then(|s| s.parse().ok())
                .expect("invalid #!preselect_expect expected value");
            current_checks.push(SegmentCheck {
                abs_meas_idx: abs_idx,
                expected: expected_int != 0,
            });
            last_check_line_pos = Some(current_lines.len());
            continue;
        }

        if trimmed.starts_with("#!") {
            continue;
        }

        current_lines.push(line);
    }

    flush_segment(
        &mut segments,
        &mut current_lines,
        &mut current_checks,
        &mut last_check_line_pos,
    );
    segments
}

fn flush_segment(
    segments: &mut Vec<Segment>,
    lines: &mut Vec<&str>,
    checks: &mut Vec<SegmentCheck>,
    last_check_pos: &mut Option<usize>,
) {
    let (retry, tail) = match *last_check_pos {
        Some(pos) => (lines[..pos].join("\n"), lines[pos..].join("\n")),
        None => (String::new(), lines.join("\n")),
    };
    segments.push(Segment {
        retry_stim_text: retry,
        checks: std::mem::take(checks),
        tail_stim_text: tail,
    });
    lines.clear();
    *last_check_pos = None;
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
                let segments = parse_segments(&circuit_text);
                let parsed: Vec<(stim::Circuit, stim::Circuit)> = segments
                    .iter()
                    .map(|seg| {
                        let retry = seg
                            .retry_stim_text
                            .parse::<stim::Circuit>()
                            .expect("Failed to parse retry segment");
                        let tail = seg
                            .tail_stim_text
                            .parse::<stim::Circuit>()
                            .expect("Failed to parse tail segment");
                        (retry, tail)
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

                    let measurements = run_one_shot(&segments, &parsed, sim_seed, max_attempts, &retries_ref);

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

/// Execute one shot with retry-from-BEGIN semantics.
///
/// A single `TableauSimulator` is kept for the entire shot.  We
/// maintain `accepted_indices` — a mapping from nominal measurement
/// index (as the circuit expects) to actual index in the simulator's
/// growing record (which includes junk from failed retries).  Every
/// `rec[-k]` target is remapped through this table so that references
/// across segments stay correct.
fn run_one_shot(
    segments: &[Segment],
    parsed: &[(stim::Circuit, stim::Circuit)],
    initial_seed: u64,
    max_attempts: u64,
    total_retries: &std::sync::atomic::AtomicU64,
) -> Vec<bool> {
    let mut sim = stim::TableauSimulator::with_seed(initial_seed);
    let mut accepted_indices: Vec<usize> = Vec::new();

    for (seg_idx, segment) in segments.iter().enumerate() {
        let (retry_circuit, tail_circuit) = &parsed[seg_idx];

        if segment.checks.is_empty() {
            let new = execute_with_mapping(&mut sim, tail_circuit, &accepted_indices);
            accepted_indices.extend(new);
        } else {
            let mut attempts = 0u64;
            loop {
                let new = execute_with_mapping(&mut sim, retry_circuit, &accepted_indices);

                let record = sim.current_measurement_record();
                let all_pass = segment.checks.iter().all(|check| {
                    let idx_in_seg = check.abs_meas_idx - accepted_indices.len();
                    idx_in_seg < new.len() && record[new[idx_in_seg]] == check.expected
                });

                if all_pass {
                    accepted_indices.extend(new);
                    break;
                }

                attempts += 1;
                total_retries.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                if attempts >= max_attempts {
                    panic!(
                        "Preselect retry limit ({max_attempts}) exceeded at segment {seg_idx}. \
                         This usually means the retry region interacts with data qubits \
                         whose state is corrupted by failed attempts. Use --simulator static \
                         (resample mode) instead of --simulator preselect for this circuit."
                    );
                }
            }

            // Tail: execute once after checks pass.
            let new = execute_with_mapping(&mut sim, tail_circuit, &accepted_indices);
            accepted_indices.extend(new);
        }
    }

    let full_record = sim.current_measurement_record();
    accepted_indices.iter().map(|&i| full_record[i]).collect()
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
            expected_readouts: BitVector { size: 0, data: vec![] },
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
    fn parse_segments_no_directives() {
        let segments = parse_segments("H 0\nM 0\n");
        assert_eq!(segments.len(), 1);
        assert!(segments[0].checks.is_empty());
        assert!(segments[0].retry_stim_text.is_empty());
        assert!(segments[0].tail_stim_text.contains("H 0"));
    }

    #[test]
    fn parse_segments_with_begin_and_checks() {
        let text = "\
R 0
H 0
M 0
#!preselect_begin
H 0
M 0
#!preselect_expect 1 0
CZ 0 1
";
        let segments = parse_segments(text);
        assert_eq!(segments.len(), 2);
        assert!(segments[0].checks.is_empty());
        assert_eq!(segments[1].checks.len(), 1);
        assert_eq!(segments[1].checks[0].abs_meas_idx, 1);
        assert!(segments[1].retry_stim_text.contains("M 0"));
        assert!(segments[1].tail_stim_text.contains("CZ 0 1"));
    }

    #[test]
    fn basic_preselect_sampling() {
        let text = "\
R 0
M 0
#!preselect_begin
H 0
M 0
#!preselect_expect 1 0
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
#!preselect_begin
H 0
M 0
#!preselect_expect 0 0
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
#!preselect_begin
R 0
M 0
#!preselect_expect 0 0
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
#!preselect_begin
H 0
M 0
#!preselect_expect 0 0
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
}
