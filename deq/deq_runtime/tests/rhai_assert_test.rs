#![cfg(feature = "simulator")]

use deq_runtime::misc::bit_vector;
use deq_runtime::simulator::rhai_assert::{self, RhaiAssertEngine};
use deq_runtime::util::BitVector;

fn write_temp_script(content: &str) -> (tempfile::NamedTempFile, String) {
    use std::io::Write;
    let mut f = tempfile::Builder::new().suffix(".rhai").tempfile().unwrap();
    f.write_all(content.as_bytes()).unwrap();
    let path = f.path().to_str().unwrap().to_owned();
    (f, path)
}

fn make_bitvec(bits: &[bool]) -> BitVector {
    BitVector {
        size: bits.len() as u64,
        data: bit_vector::pack_bits(bits),
    }
}

#[test]
fn script_detects_no_error() {
    let script = r#"
fn is_logical_error(shot, readouts, measurements) {
    // No error if readout 0 is false
    readouts.len() > 0 && readouts[0]
}
"#;
    let (_f, path) = write_temp_script(script);
    let engine = RhaiAssertEngine::new(&path);

    let readouts = make_bitvec(&[false, true, false]);
    let measurements = make_bitvec(&[true, false]);

    assert!(!engine.is_logical_error(0, Some(&readouts), &measurements));
}

#[test]
fn script_detects_error() {
    let script = r#"
fn is_logical_error(shot, readouts, measurements) {
    readouts.len() > 0 && readouts[0]
}
"#;
    let (_f, path) = write_temp_script(script);
    let engine = RhaiAssertEngine::new(&path);

    let readouts = make_bitvec(&[true, false]);
    let measurements = make_bitvec(&[true]);

    assert!(engine.is_logical_error(5, Some(&readouts), &measurements));
}

#[test]
fn script_receives_correct_shot_index() {
    let script = r#"
fn is_logical_error(shot, readouts, measurements) {
    shot == 42
}
"#;
    let (_f, path) = write_temp_script(script);
    let engine = RhaiAssertEngine::new(&path);

    let empty = make_bitvec(&[]);
    assert!(engine.is_logical_error(42, Some(&empty), &empty));
    assert!(!engine.is_logical_error(0, Some(&empty), &empty));
}

#[test]
fn script_can_access_measurements() {
    let script = r#"
fn is_logical_error(shot, readouts, measurements) {
    measurements.len() > 0 && measurements[0]
}
"#;
    let (_f, path) = write_temp_script(script);
    let engine = RhaiAssertEngine::new(&path);

    let empty = make_bitvec(&[]);
    let meas_true = make_bitvec(&[true, false]);
    let meas_false = make_bitvec(&[false, false]);

    assert!(engine.is_logical_error(0, Some(&empty), &meas_true));
    assert!(!engine.is_logical_error(0, Some(&empty), &meas_false));
}

#[test]
fn script_handles_none_readouts() {
    let script = r#"
fn is_logical_error(shot, readouts, measurements) {
    readouts.len() == 0
}
"#;
    let (_f, path) = write_temp_script(script);
    let engine = RhaiAssertEngine::new(&path);

    let measurements = make_bitvec(&[]);

    // None readouts → empty array
    assert!(engine.is_logical_error(0, None, &measurements));
}

#[test]
fn script_with_print_does_not_crash() {
    let script = r#"
fn is_logical_error(shot, readouts, measurements) {
    print(`Shot ${shot}: readouts len = ${readouts.len()}`);
    false
}
"#;
    let (_f, path) = write_temp_script(script);
    let engine = RhaiAssertEngine::new(&path);

    let r = make_bitvec(&[true, false]);
    let m = make_bitvec(&[]);

    assert!(!engine.is_logical_error(7, Some(&r), &m));
}

#[test]
#[should_panic(expected = "Failed to read Rhai script")]
fn nonexistent_file_panics() {
    RhaiAssertEngine::new("/nonexistent/path/script.rhai");
}

#[test]
#[should_panic(expected = "Failed to compile Rhai script")]
fn syntax_error_panics() {
    let script = "fn bad_syntax(( { }";
    let (_f, path) = write_temp_script(script);
    RhaiAssertEngine::new(&path);
}

#[test]
#[should_panic(expected = "is_logical_error")]
fn missing_function_panics_on_call() {
    let script = r#"
fn some_other_function(x) { x }
"#;
    let (_f, path) = write_temp_script(script);
    let engine = RhaiAssertEngine::new(&path);

    let empty = make_bitvec(&[]);
    engine.is_logical_error(0, Some(&empty), &empty);
}

#[test]
fn engine_can_be_called_many_times() {
    let script = r#"
fn is_logical_error(shot, readouts, measurements) {
    shot % 2 == 0
}
"#;
    let (_f, path) = write_temp_script(script);
    let engine = RhaiAssertEngine::new(&path);

    let empty = make_bitvec(&[]);
    for i in 0..100 {
        let result = engine.is_logical_error(i, Some(&empty), &empty);
        assert_eq!(result, i % 2 == 0);
    }
}

#[test]
fn readout_parity_check_script() {
    let script = r#"
fn is_logical_error(shot, readouts, measurements) {
    let parity = false;
    for b in readouts {
        parity = parity ^ b;
    }
    parity
}
"#;
    let (_f, path) = write_temp_script(script);
    let engine = RhaiAssertEngine::new(&path);

    let m = make_bitvec(&[]);

    // Even parity → no error
    assert!(!engine.is_logical_error(0, Some(&make_bitvec(&[false, false])), &m));
    // Odd parity → error
    assert!(engine.is_logical_error(0, Some(&make_bitvec(&[true, false])), &m));
    // Even parity → no error
    assert!(!engine.is_logical_error(0, Some(&make_bitvec(&[true, true])), &m));
}

// ── extract_rhai_script tests ──────────────────────────────────────────────

#[test]
fn extract_no_rhai_block_returns_none() {
    let stim = "H 0\nCNOT 0 1\nM 0 1\n";
    assert!(rhai_assert::extract_rhai_script(stim).is_none());
}

#[test]
fn extract_empty_rhai_block_returns_none() {
    let stim = "#!rhai\nH 0\nM 0\n";
    assert!(rhai_assert::extract_rhai_script(stim).is_none());
}

#[test]
fn extract_single_block() {
    let stim = "\
#!rhai
# fn is_logical_error(shot, readouts, measurements) {
#     false
# }
H 0
M 0
";
    let script = rhai_assert::extract_rhai_script(stim).unwrap();
    assert!(script.contains("fn is_logical_error"));
    assert!(script.contains("false"));
    assert!(!script.contains("# fn"));
}

#[test]
fn extract_strips_hash_without_space() {
    let stim = "\
#!rhai
#fn foo() { 42 }
H 0
";
    let script = rhai_assert::extract_rhai_script(stim).unwrap();
    assert!(script.contains("fn foo() { 42 }"));
}

#[test]
fn extract_multiple_blocks_concatenated() {
    let stim = "\
#!rhai
# let x = 1;
H 0
#!rhai
# fn is_logical_error(shot, readouts, measurements) {
#     false
# }
M 0
";
    let script = rhai_assert::extract_rhai_script(stim).unwrap();
    assert!(script.contains("let x = 1;"));
    assert!(script.contains("fn is_logical_error"));
}

#[test]
fn extract_block_at_eof() {
    let stim = "\
H 0
M 0
#!rhai
# fn is_logical_error(shot, readouts, measurements) { false }";
    let script = rhai_assert::extract_rhai_script(stim).unwrap();
    assert!(script.contains("fn is_logical_error"));
}

#[test]
fn extract_preserves_blank_comment_lines() {
    let stim = "\
#!rhai
# fn is_logical_error(shot, readouts, measurements) {
#
#     false
# }
H 0
";
    let script = rhai_assert::extract_rhai_script(stim).unwrap();
    assert!(script.contains("\n\n"));
}

#[test]
fn extract_ignores_regular_comments() {
    let stim = "\
# This is a normal stim comment
H 0
#!rhai
# fn is_logical_error(shot, readouts, measurements) { false }
M 0
";
    let script = rhai_assert::extract_rhai_script(stim).unwrap();
    assert!(!script.contains("normal stim comment"));
    assert!(script.contains("fn is_logical_error"));
}

// ── from_source tests ──────────────────────────────────────────────────────

#[test]
fn from_source_compiles_and_runs() {
    let script = r#"
fn is_logical_error(shot, readouts, measurements) {
    readouts.len() > 1
}
"#;
    let engine = RhaiAssertEngine::from_source("test", script);
    let short = make_bitvec(&[true]);
    let long = make_bitvec(&[true, false]);
    assert!(!engine.is_logical_error(0, Some(&short), &make_bitvec(&[])));
    assert!(engine.is_logical_error(0, Some(&long), &make_bitvec(&[])));
}

#[test]
fn embedded_script_end_to_end() {
    let stim = "\
#!rhai
# fn is_logical_error(shot, readouts, measurements) {
#     // Error if first measurement is true
#     measurements.len() > 0 && measurements[0]
# }
H 0
M 0
";
    let script = rhai_assert::extract_rhai_script(stim).unwrap();
    let engine = RhaiAssertEngine::from_source("embedded", &script);

    assert!(!engine.is_logical_error(0, Some(&make_bitvec(&[])), &make_bitvec(&[false])));
    assert!(engine.is_logical_error(0, Some(&make_bitvec(&[])), &make_bitvec(&[true])));
}

// ── Example: realistic Stim circuit with embedded Rhai assertion ───────────

/// This test demonstrates the recommended way to embed a Rhai logical-error
/// assertion directly inside a Stim circuit file.  The circuit, the noise
/// model, and the assertion logic all live in one self-contained file.
///
/// ## How it works
///
/// Stim treats every line starting with `#` as a comment.  We piggy-back on
/// this: a line containing exactly `#!rhai` marks the beginning of a Rhai
/// script block.  All subsequent `#`-prefixed lines are collected as Rhai
/// source code (the leading `#` and one optional space are stripped).  The
/// block ends at the first line that does *not* start with `#`.
///
/// Multiple `#!rhai` blocks are concatenated, so you can place helper
/// functions at the top and the main assertion at the bottom.
///
/// The Rhai script must define:
///
/// ```rhai
/// fn is_logical_error(shot, readouts, measurements) -> bool
/// ```
///
/// Where:
///   - `shot`         : i64          — the current shot index (0-based)
///   - `readouts`     : Array of bool — the decoder's output bits
///   - `measurements` : Array of bool — all physical measurement outcomes
///
/// Return `true` to flag the shot as a logical error, `false` otherwise.
/// You may call `print(...)` from the script for debugging.
#[test]
fn example_stim_circuit_with_embedded_rhai() {
    // ── The Stim circuit file contents ─────────────────────────────────
    //
    // This example encodes a simple repetition code:
    //
    //   1. Prepare two data qubits (q0, q1) in |0⟩.
    //   2. Apply X_ERROR(0.1) to each — 10 % chance of a bit-flip.
    //   3. Measure both qubits.
    //
    // The logical readout is the parity of the two measurement results:
    //   - Even parity (both 0 or both 1) → no logical error.
    //   - Odd parity (one flipped)       → logical error.
    //
    // The Rhai script embedded in the comments implements exactly this
    // parity check.

    let stim_file = "\
# ================================================================
# Example: two-qubit repetition code with embedded Rhai assertion
# ================================================================
#
# This file is a valid Stim circuit.  The `#!rhai` block below
# defines the logical-error criterion that the simulator will use
# instead of the default readout comparison.
#
# ── Rhai assertion script ──────────────────────────────────────
#
# Usage:
#   The function `is_logical_error` is called once per shot.
#   It receives three arguments:
#
#     shot         - i64        : 0-based shot index
#     readouts     - [bool; N]  : decoded readout bits
#     measurements - [bool; M]  : raw physical measurements
#
#   Return `true` for a logical error, `false` for success.
#   `print(...)` can be used for debugging.
#
#!rhai
# // Helper: compute parity (XOR) of a boolean array
# fn parity(arr) {
#     let p = false;
#     for b in arr {
#         p = p ^ b;     // XOR each element
#     }
#     p
# }
#
# fn is_logical_error(shot, readouts, measurements) {
#     // The logical observable is the parity of all measurements.
#     // In the noiseless case the parity is even (false).
#     // Any single bit-flip makes the parity odd -> logical error.
#     let meas_parity = parity(measurements);
#
#     if meas_parity {
#         print(`Shot ${shot}: odd parity detected -> logical error`);
#     }
#
#     meas_parity
# }

# ── Circuit ────────────────────────────────────────────────────
# Prepare |00⟩, apply 10% X noise, and measure.
X_ERROR(0.1) 0 1
M 0 1
";

    // ── Step 1: Extract the Rhai script from the Stim comments ─────
    let script = rhai_assert::extract_rhai_script(stim_file).expect("The stim file should contain a #!rhai block");

    // Verify the extracted script contains our helper and main function
    assert!(script.contains("fn parity(arr)"), "helper function extracted");
    assert!(script.contains("fn is_logical_error("), "main function extracted");
    assert!(!script.contains("#!rhai"), "marker itself is not in the script");
    assert!(!script.contains("# fn"), "comment prefixes are stripped");

    // ── Step 2: Compile the script ─────────────────────────────────
    let engine = RhaiAssertEngine::from_source("example.stim", &script);

    // ── Step 3: Test the assertion logic ───────────────────────────
    let empty_readouts = make_bitvec(&[]);

    // No errors: both measurements are 0 → parity is even → no error
    let meas_00 = make_bitvec(&[false, false]);
    assert!(
        !engine.is_logical_error(0, Some(&empty_readouts), &meas_00),
        "parity(00) = even → not a logical error"
    );

    // No errors: both measurements are 1 → parity is still even
    let meas_11 = make_bitvec(&[true, true]);
    assert!(
        !engine.is_logical_error(1, Some(&empty_readouts), &meas_11),
        "parity(11) = even → not a logical error"
    );

    // Single bit flip: measurements are [1, 0] → parity is odd → error
    let meas_10 = make_bitvec(&[true, false]);
    assert!(
        engine.is_logical_error(2, Some(&empty_readouts), &meas_10),
        "parity(10) = odd → logical error"
    );

    // Single bit flip: measurements are [0, 1] → parity is odd → error
    let meas_01 = make_bitvec(&[false, true]);
    assert!(
        engine.is_logical_error(3, Some(&empty_readouts), &meas_01),
        "parity(01) = odd → logical error"
    );
}
