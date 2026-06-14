//! Parse `#!delay` directives embedded in Stim circuit files.
//!
//! A `#!delay <seconds>` line in a Stim file inserts a measurement delay at
//! the current measurement index.  The directive is a Stim comment (starts
//! with `#`) so the Stim parser ignores it, but we extract it here to build
//! a streaming delay schedule.
//!
//! ## Example
//!
//! ```text
//! R 0 1
//! X_ERROR(0.01) 0 1
//! M 0 1
//! #!delay 0.5
//! X_ERROR(0.01) 0 1
//! M 0 1
//! ```
//!
//! The `#!delay 0.5` appears after the first `M 0 1` (2 measurement bits)
//! and before the second.  This produces a schedule where the first 2
//! measurements are sent immediately and the remaining arrive 0.5 s later.
//!
//! ## Limitations
//!
//! `REPEAT` blocks are **not supported** because measurement counting
//! inside repeated blocks requires expansion.  The parser panics if it
//! encounters a `REPEAT` instruction.

use super::common::DelayBatch;

/// Stim instruction names that produce measurement bits.
///
/// Each target of these instructions contributes one measurement bit,
/// except for `MPP` where each complete Pauli product term contributes one.
const MEASUREMENT_INSTRUCTIONS: &[&str] = &[
    "M", "MZ", "MX", "MY", "MR", "MRX", "MRY", "MRZ", "MXX", "MYY", "MZZ", "MPP", "MPAD",
];

/// Parse `#!delay` directives and measurement instructions from a Stim
/// circuit's text to build a streaming delay schedule.
///
/// After building the schedule, asserts that the counted measurement total
/// equals `expected_measurements` (typically from
/// `stim::Circuit::num_measurements()`).
///
/// # Panics
///
/// - If a `REPEAT` instruction is encountered.
/// - If the counted measurement total does not match `expected_measurements`.
pub(crate) fn extract_delay_schedule(stim_text: &str, expected_measurements: usize) -> Vec<DelayBatch> {
    let mut delays: Vec<(usize, f64)> = Vec::new(); // (measurement_index, delay_seconds)
    let mut measurement_count: usize = 0;

    for line in stim_text.lines() {
        let trimmed = line.trim();

        // Skip empty lines
        if trimmed.is_empty() {
            continue;
        }

        // Handle #!delay directive
        if let Some(rest) = trimmed.strip_prefix("#!delay") {
            let rest = rest.trim();
            let delay_seconds: f64 = rest.parse().unwrap_or_else(|e| {
                panic!(
                    "#!delay directive has invalid delay value '{rest}': {e}\n\
                     Expected: #!delay <seconds>  (e.g. #!delay 0.5)"
                )
            });
            delays.push((measurement_count, delay_seconds));
            continue;
        }

        // Skip all other comments (including #!rhai blocks)
        if trimmed.starts_with('#') {
            continue;
        }

        // Parse instruction name (first whitespace-delimited token)
        let instr_name = trimmed.split(|c: char| c.is_whitespace() || c == '(').next().unwrap_or("");

        // Reject REPEAT blocks
        if instr_name == "REPEAT" {
            panic!(
                "REPEAT blocks are not supported in Stim files with #!delay directives.\n\
                 The delay schedule requires flat (non-repeated) circuits so that \
                 measurement indices can be counted unambiguously."
            );
        }

        // Count measurement bits
        let name_upper = instr_name.to_uppercase();
        if MEASUREMENT_INSTRUCTIONS.contains(&name_upper.as_str()) {
            measurement_count += count_measurement_targets(trimmed, &name_upper);
        }
    }

    // Assert measurement count matches what stim reports
    assert_eq!(
        measurement_count, expected_measurements,
        "Measurement count mismatch: counted {measurement_count} measurement \
         bits by parsing the Stim file line-by-line, but stim reports \
         {expected_measurements}. This may indicate an unsupported instruction \
         or a parsing bug."
    );

    // Build delay schedule (same logic as build_delay_schedule in static_simulator)
    build_schedule_from_delays(&delays, measurement_count)
}

/// Count the number of measurement bits produced by a single measurement
/// instruction line.
///
/// For most instructions, each qubit target produces one measurement bit.
/// For `MPP`, each `*`-separated Pauli product term produces one bit.
fn count_measurement_targets(line: &str, instr_name: &str) -> usize {
    // Strip instruction name and optional parenthesized arguments
    let after_name = line
        .trim()
        .strip_prefix(instr_name)
        .or_else(|| {
            // Case-insensitive: try the original line's instruction
            let first_token_len = line
                .trim()
                .find(|c: char| c.is_whitespace() || c == '(')
                .unwrap_or(line.trim().len());
            Some(&line.trim()[first_token_len..])
        })
        .unwrap_or("");

    // Skip past parenthesized arguments like (0.01)
    let targets_str = if let Some(paren_end) = after_name.find(')') {
        after_name[paren_end + 1..].trim()
    } else {
        after_name.trim()
    };

    if targets_str.is_empty() {
        return 0;
    }

    if instr_name == "MPP" {
        // MPP: each product term (separated by `*`) contributes one measurement.
        // Terms are separated by spaces, products within a term by `*`.
        // e.g. "MPP X0*X1 Z2*Z3" → 2 measurement bits
        // The terms are separated by spaces, but within each term targets are
        // joined by `*`.  Count the number of space-separated groups.
        //
        // However, stim allows combiner targets: "MPP X0*X1 Z2*Z3" means 2
        // products = 2 measurements.  Just count groups that don't start with
        // a combiner.
        targets_str.split_whitespace().filter(|t| !t.starts_with('*')).count()
    } else if instr_name == "MXX" || instr_name == "MYY" || instr_name == "MZZ" {
        // Two-qubit parity measurements: each pair of targets produces one bit
        let n_targets = targets_str.split_whitespace().count();
        n_targets / 2
    } else {
        // Standard measurement: one bit per target
        targets_str.split_whitespace().count()
    }
}

/// Build a `Vec<DelayBatch>` from raw `(measurement_index, delay_seconds)` pairs.
fn build_schedule_from_delays(delays: &[(usize, f64)], total_measurements: usize) -> Vec<DelayBatch> {
    if delays.is_empty() {
        return vec![];
    }

    let mut schedule = Vec::new();
    let mut cumulative_delay = 0.0;

    // If the first delay doesn't start at measurement 0, prepend an immediate batch
    if delays[0].0 > 0 {
        schedule.push(DelayBatch {
            cumulative_count: delays[0].0,
            delay_seconds: 0.0,
        });
    }

    for (i, &(start_idx, relative_delay)) in delays.iter().enumerate() {
        cumulative_delay += relative_delay;
        let end = if i + 1 < delays.len() {
            delays[i + 1].0
        } else {
            total_measurements
        };
        if end > start_idx {
            schedule.push(DelayBatch {
                cumulative_count: end,
                delay_seconds: cumulative_delay,
            });
        }
    }

    schedule
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_delays_empty_schedule() {
        let stim = "R 0 1\nX_ERROR(0.01) 0 1\nM 0 1\n";
        let schedule = extract_delay_schedule(stim, 2);
        assert!(schedule.is_empty());
    }

    #[test]
    fn single_delay_between_measurements() {
        let stim = "\
R 0 1
M 0 1
#!delay 0.5
M 0 1
";
        let schedule = extract_delay_schedule(stim, 4);
        assert_eq!(schedule.len(), 2);
        // First batch: measurements 0..2, no delay
        assert_eq!(schedule[0].cumulative_count, 2);
        assert!((schedule[0].delay_seconds - 0.0).abs() < 1e-9);
        // Second batch: measurements 2..4, 0.5s delay
        assert_eq!(schedule[1].cumulative_count, 4);
        assert!((schedule[1].delay_seconds - 0.5).abs() < 1e-9);
    }

    #[test]
    fn multiple_delays() {
        let stim = "\
M 0
#!delay 0.3
M 1
#!delay 0.7
M 0 1
";
        let schedule = extract_delay_schedule(stim, 4);
        assert_eq!(schedule.len(), 3);
        assert_eq!(schedule[0].cumulative_count, 1);
        assert!((schedule[0].delay_seconds - 0.0).abs() < 1e-9);
        assert_eq!(schedule[1].cumulative_count, 2);
        assert!((schedule[1].delay_seconds - 0.3).abs() < 1e-9);
        assert_eq!(schedule[2].cumulative_count, 4);
        assert!((schedule[2].delay_seconds - 1.0).abs() < 1e-9);
    }

    #[test]
    fn delay_at_start() {
        let stim = "\
#!delay 1.0
M 0 1
";
        let schedule = extract_delay_schedule(stim, 2);
        assert_eq!(schedule.len(), 1);
        assert_eq!(schedule[0].cumulative_count, 2);
        assert!((schedule[0].delay_seconds - 1.0).abs() < 1e-9);
    }

    #[test]
    fn noise_and_gate_instructions_ignored() {
        let stim = "\
R 0 1
H 0
CNOT 0 1
DEPOLARIZE1(0.01) 0 1
X_ERROR(0.001) 0
TICK
M 0 1
";
        let schedule = extract_delay_schedule(stim, 2);
        assert!(schedule.is_empty());
    }

    #[test]
    fn multi_target_measurement() {
        let stim = "M 0 1 2 3\n";
        let schedule = extract_delay_schedule(stim, 4);
        assert!(schedule.is_empty());
    }

    #[test]
    fn mr_instruction_counted() {
        let stim = "\
MR 0 1
#!delay 0.5
MR 2
";
        let schedule = extract_delay_schedule(stim, 3);
        assert_eq!(schedule.len(), 2);
        assert_eq!(schedule[0].cumulative_count, 2);
        assert_eq!(schedule[1].cumulative_count, 3);
    }

    #[test]
    fn comments_and_rhai_blocks_skipped() {
        let stim = "\
# This is a comment
#!rhai
# fn is_logical_error(shot, readouts, measurements) { false }
M 0
#!delay 0.5
# another comment
M 1
";
        let schedule = extract_delay_schedule(stim, 2);
        assert_eq!(schedule.len(), 2);
        assert_eq!(schedule[0].cumulative_count, 1);
        assert_eq!(schedule[1].cumulative_count, 2);
    }

    #[test]
    #[should_panic(expected = "REPEAT blocks are not supported")]
    fn repeat_block_panics() {
        let stim = "\
M 0
REPEAT 10 {
    M 0
}
";
        extract_delay_schedule(stim, 11);
    }

    #[test]
    #[should_panic(expected = "Measurement count mismatch")]
    fn measurement_count_mismatch_panics() {
        let stim = "M 0 1\n";
        extract_delay_schedule(stim, 5); // claim 5, but only 2
    }

    #[test]
    fn case_insensitive_instructions() {
        let stim = "m 0 1\n";
        let schedule = extract_delay_schedule(stim, 2);
        assert!(schedule.is_empty());
    }

    #[test]
    fn mxx_produces_one_bit_per_pair() {
        let stim = "MXX 0 1 2 3\n";
        // 4 targets → 2 pairs → 2 measurement bits
        let schedule = extract_delay_schedule(stim, 2);
        assert!(schedule.is_empty());
    }
}
