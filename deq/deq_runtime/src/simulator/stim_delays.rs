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
pub const MEASUREMENT_INSTRUCTIONS: &[&str] = &[
    "M", "MZ", "MX", "MY", "MR", "MRX", "MRY", "MRZ", "MXX", "MYY", "MZZ", "MPP", "MPAD",
];

/// Count measurement bits in a Stim circuit by scanning the text line by line.
///
/// Unlike `stim::Circuit::num_measurements()`, this only requires that the
/// measurement-producing instruction *names* be recognized (`M`, `MR`, `MZ`,
/// etc.); arbitrary unrecognized instructions on other lines are silently
/// ignored.  This makes the function safe to use on Stim files that contain
/// extensions not understood by the upstream `stim` crate (e.g. QDK's
/// `LOSS_ERROR`).
///
/// # Panics
///
/// - If a `REPEAT` instruction is encountered (we do not expand loops; the
///   measurement count would otherwise depend on dynamic state).
pub fn count_measurements(stim_text: &str) -> usize {
    let mut measurement_count: usize = 0;
    for line in stim_text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let instr_name = instruction_name(trimmed);
        if instr_name == "REPEAT" {
            panic!("REPEAT blocks are not supported.");
        }
        let name_upper = instr_name.to_uppercase();
        if MEASUREMENT_INSTRUCTIONS.contains(&name_upper.as_str()) {
            measurement_count += count_measurement_targets(trimmed, &name_upper);
        }
    }
    measurement_count
}

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
    let mut delays: Vec<(usize, u64)> = Vec::new();
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
            let delay_ns = parse_delay_ns(rest).unwrap_or_else(|error| {
                panic!(
                    "#!delay directive has invalid delay value '{rest}': {error}\n\
                     Expected: #!delay <seconds>  (e.g. #!delay 0.5)"
                )
            });
            delays.push((measurement_count, delay_ns));
            continue;
        }

        // Skip all other comments (including #!rhai blocks)
        if trimmed.starts_with('#') {
            continue;
        }

        // Parse instruction name (first whitespace-delimited token)
        let instr_name = instruction_name(trimmed);

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
/// Return the number of measurement bits contributed by a single Stim
/// instruction line.  `instr_name` must be the uppercased opcode from
/// [`MEASUREMENT_INSTRUCTIONS`]; the caller is responsible for that check.
pub fn count_measurement_targets(line: &str, instr_name: &str) -> usize {
    let line = line.trim();
    let mut targets_str = line[instruction_name(line).len()..].trim_start();
    while let Some(closing) = match targets_str.as_bytes().first() {
        Some(b'[') => Some(']'),
        Some(b'(') => Some(')'),
        _ => None,
    } {
        let end = targets_str.find(closing).expect("unterminated Stim tag or arguments");
        targets_str = targets_str[end + 1..].trim_start();
    }
    let targets_str = targets_str.split('#').next().unwrap_or("").trim();

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

fn instruction_name(line: &str) -> &str {
    line.split(|character: char| character.is_whitespace() || character == '(' || character == '[')
        .next()
        .unwrap_or("")
}

fn parse_delay_ns(seconds: &str) -> Result<u64, &'static str> {
    let seconds = seconds.strip_prefix('+').unwrap_or(seconds);
    let (mantissa, exponent) = seconds.split_once(['e', 'E']).unwrap_or((seconds, "0"));
    let exponent = exponent.parse::<i64>().map_err(|_| "invalid decimal exponent")?;
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    if whole.len() + fraction.len() == 0 || !whole.bytes().chain(fraction.bytes()).all(|digit| digit.is_ascii_digit()) {
        return Err("delay must be a nonnegative decimal number");
    }
    let digits = format!("{whole}{fraction}");
    let digits = digits.trim_start_matches('0');
    if digits.is_empty() {
        return Ok(0);
    }
    let significant = digits.trim_end_matches('0');
    let trailing_zeros = i64::try_from(digits.len() - significant.len()).map_err(|_| "decimal is too long")?;
    let fractional_digits = i64::try_from(fraction.len()).map_err(|_| "decimal is too long")?;
    let scale = exponent
        .checked_add(9)
        .and_then(|scale| scale.checked_sub(fractional_digits))
        .and_then(|scale| scale.checked_add(trailing_zeros))
        .ok_or("decimal scale overflow")?;
    let scale = u32::try_from(scale).map_err(|_| "delay must be a whole number of nanoseconds")?;
    let coefficient = significant.parse::<u64>().map_err(|_| "nanosecond delay overflow")?;
    let multiplier = 10u64.checked_pow(scale).ok_or("nanosecond delay overflow")?;
    coefficient
        .checked_mul(multiplier)
        .filter(|&value| value <= i64::MAX.cast_unsigned())
        .ok_or("delay exceeds signed 63-bit nanoseconds")
}

/// Build a `Vec<DelayBatch>` from exact `(measurement_index, delay_ns)` pairs.
fn build_schedule_from_delays(delays: &[(usize, u64)], total_measurements: usize) -> Vec<DelayBatch> {
    if delays.is_empty() {
        return vec![];
    }

    let mut schedule = Vec::new();
    let mut cumulative_delay = 0u64;

    // If the first delay doesn't start at measurement 0, prepend an immediate batch
    if delays[0].0 > 0 {
        schedule.push(DelayBatch {
            cumulative_count: delays[0].0,
            delay_ns: 0,
        });
    }

    for (i, &(start_idx, relative_delay)) in delays.iter().enumerate() {
        cumulative_delay = cumulative_delay
            .checked_add(relative_delay)
            .filter(|&value| value <= i64::MAX.cast_unsigned())
            .expect("#!delay cumulative duration overflows signed 63-bit nanoseconds");
        let end = if i + 1 < delays.len() {
            delays[i + 1].0
        } else {
            total_measurements
        };
        assert!(end >= start_idx, "invalid measurement schedule");
        schedule.push(DelayBatch {
            cumulative_count: end,
            delay_ns: cumulative_delay,
        });
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
        assert_eq!(schedule[0].delay_ns, 0);
        // Second batch: measurements 2..4, 0.5s delay
        assert_eq!(schedule[1].cumulative_count, 4);
        assert_eq!(schedule[1].delay_ns, 500_000_000);
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
        assert_eq!(schedule[0].delay_ns, 0);
        assert_eq!(schedule[1].cumulative_count, 2);
        assert_eq!(schedule[1].delay_ns, 300_000_000);
        assert_eq!(schedule[2].cumulative_count, 4);
        assert_eq!(schedule[2].delay_ns, 1_000_000_000);
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
        assert_eq!(schedule[0].delay_ns, 1_000_000_000);
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

    #[test]
    fn tagged_measurements_and_inline_comments_preserve_timing_counts() {
        let text = "#!delay 0.01\nM[syndrome](0.01) 0 1 # result\n#!delay 0.01\nMPP[readout] X0*X1 Z2\n";
        assert_eq!(count_measurements(text), 4);
        let schedule = extract_delay_schedule(text, 4);
        assert_eq!(schedule.len(), 2);
        assert_eq!(schedule[0].cumulative_count, 2);
        assert_eq!(schedule[1].cumulative_count, 4);
        assert_eq!(count_measurement_targets("m(0.01)[tag] 0", "M"), 1);
    }

    #[test]
    fn rejects_invalid_delay_numbers() {
        for value in [
            "NaN",
            "inf",
            "-0.1",
            "1e100",
            ".",
            "1e",
            "0.0000000001",
            "9223372036.854775808",
        ] {
            assert!(std::panic::catch_unwind(|| extract_delay_schedule(&format!("#!delay {value}\nM 0"), 1)).is_err());
        }
    }

    #[test]
    fn decimal_delays_are_parsed_as_exact_integer_nanoseconds() {
        for (text, expected) in [
            ("16777216.000000001", 16_777_216_000_000_001),
            ("9223372036.854775807", i64::MAX.cast_unsigned()),
            ("1e-9", 1),
            ("1.5E-8", 15),
            ("+000.0100000000000000000000", 10_000_000),
            (".1", 100_000_000),
            ("1.", 1_000_000_000),
            ("0e100", 0),
        ] {
            assert_eq!(parse_delay_ns(text), Ok(expected), "{text}");
        }
    }

    #[test]
    #[should_panic(expected = "cumulative duration overflows")]
    fn cumulative_nanosecond_overflow_is_rejected() {
        extract_delay_schedule("#!delay 9223372036.854775807\n#!delay 0.000000001\n", 0);
    }
}
