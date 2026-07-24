//! Parse `PREPARE { ... REQUIRE ... }` blocks from Stim text (QDK v1.30+).
//!
//! These blocks are emitted by the Python `export_program_stim` function
//! to encode preselection: a `PREPARE { ... }` region runs as a
//! repeat-until-success unit, restarting from the opening brace whenever
//! any embedded `REQUIRE` check fails.  Because the upstream `stim`
//! crate does not understand `PREPARE` / `REQUIRE`, callers that need to
//! feed the text to `stim::Circuit::from_str` must first call
//! [`strip_preselect_directives`] to drop the block markers and require
//! lines while preserving every real instruction.
//!
//! Two consumers exist:
//!
//! * **Resample mode** (static / jit-static simulators): uses
//!   [`extract_preselect_schedule`] to get a flat list of parity checks.
//!   After sampling a full shot the simulator verifies every check and
//!   resamples on failure.
//!
//! * **Retry mode** (preselect simulator): the
//!   `TableauPreselectSampler` uses [`extract_preselect_blocks`] to get
//!   a flat sequence of [`PreselectBlock`]s where blocks with checks
//!   are retried until every REQUIRE passes and blocks without checks
//!   run once.
//!
//! ## Syntax
//!
//! ```text
//! PREPARE {
//!     H 0
//!     M 0
//!     REQUIRE rec[-1]           // succeeds when the last measurement == 0
//! }
//! ```
//!
//! * `REQUIRE t1 t2 ...` succeeds when the XOR of the (possibly
//!   negated) targets equals 0.
//! * A target may be negated: `REQUIRE !rec[-1]` flips that record's
//!   contribution.  Equivalently, `REQUIRE !rec[-1]` succeeds when the
//!   record equals 1.
//! * `rec[-k]` inside a `REQUIRE` refers to the k-th most recent
//!   measurement produced **inside the enclosing PREPARE block**.
//!   Referencing a measurement produced outside the block is a parse
//!   error.
//! * `PREPARE` blocks do **not** nest: the deq JIT emitter always
//!   produces at most one top-level `PREPARE` per gadget, and opening
//!   a second block before closing the current one is a parse error.

use super::stim_delays::{MEASUREMENT_INSTRUCTIONS, count_measurement_targets};

/// A single target in a `REQUIRE` line: an absolute measurement index
/// plus an optional negation flag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequireTarget {
    pub abs_meas_idx: usize,
    pub negated: bool,
}

/// A `REQUIRE` clause: the XOR of the (possibly negated) target bits
/// must equal 0 for the check to pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreselectCheck {
    pub targets: Vec<RequireTarget>,
}

impl PreselectCheck {
    /// Evaluate this check against a bit lookup that returns each
    /// target's measured value.  Returns `true` when the check passes,
    /// i.e. the XOR of the (possibly negated) target bits equals 0.
    pub fn is_satisfied(&self, mut get_bit: impl FnMut(usize) -> bool) -> bool {
        let mut acc = false;
        for target in &self.targets {
            acc ^= get_bit(target.abs_meas_idx) ^ target.negated;
        }
        !acc
    }
}

/// The full preselect schedule extracted from a Stim file.
#[derive(Debug, Clone, Default)]
pub struct PreselectSchedule {
    /// Ordered list of REQUIRE checks in the order they appear.
    pub checks: Vec<PreselectCheck>,
}

impl PreselectSchedule {
    /// Whether any preselect directives were found.
    pub fn is_empty(&self) -> bool {
        self.checks.is_empty()
    }
}

/// A single simulation block extracted from the Stim text.
///
/// The extractor slices the input into a flat sequence of blocks in
/// source order.  Two flavours exist:
///
/// * **PREPARE bodies** (`checks` non-empty): the real instructions
///   inside a `PREPARE { ... }` region, together with the `REQUIRE`
///   clauses that gate acceptance.  The sampler re-executes the whole
///   `stim_text` until every check passes.
/// * **Plain segments** (`checks` empty): any contiguous run of real
///   instructions outside a `PREPARE` region — text before the first
///   `PREPARE {`, text between one block's closing `}` and the next
///   opening `{`, and text after the final `}`.  These execute exactly
///   once.
///
/// The concatenation of every block's `stim_text` (joined with a
/// newline) is exactly what upstream `stim` needs to parse the
/// circuit — see [`strip_preselect_directives`].
#[derive(Debug, Clone)]
pub struct PreselectBlock {
    pub stim_text: String,
    pub checks: Vec<PreselectCheck>,
}

/// Return the [`PreselectSchedule`] for a stim text, ignoring block
/// structure.  Convenience wrapper around [`extract_preselect_blocks`].
pub fn extract_preselect_schedule(stim_text: &str) -> PreselectSchedule {
    let mut checks = Vec::new();
    for block in extract_preselect_blocks(stim_text) {
        checks.extend(block.checks);
    }
    PreselectSchedule { checks }
}

/// Strip every `PREPARE {` / matching `}` / `REQUIRE ...` line from the
/// stim text so the residual is parseable by the upstream `stim` crate.
///
/// The rest of the text (including comments, `#!delay`, and real
/// instructions) is preserved verbatim, including original whitespace.
///
/// Prefer [`extract_preselect_blocks`] when you also need the schedule
/// or block structure — it computes the same slicing once.
pub fn strip_preselect_directives(stim_text: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    for block in extract_preselect_blocks(stim_text) {
        parts.push(block.stim_text);
    }
    parts.join("\n")
}

/// Parse `PREPARE` / `REQUIRE` directives out of the stim text and
/// return a flat sequence of [`PreselectBlock`]s.
///
/// The sequence alternates between plain segments (no `checks`) and
/// `PREPARE` bodies (non-empty `checks`), in source order.  Empty
/// plain segments are omitted, so between two consecutive `PREPARE`
/// bodies with nothing between them the returned vector jumps from one
/// check-carrying block to the next.
///
/// # Panics
///
/// Panics on malformed input: unclosed `PREPARE`, a nested `PREPARE {`
/// opened before the current block is closed, a `REQUIRE` outside any
/// `PREPARE` block, a `rec[-0]` target, or a `rec[-k]` that references
/// a measurement not produced inside the enclosing block.
pub fn extract_preselect_blocks(stim_text: &str) -> Vec<PreselectBlock> {
    let mut blocks: Vec<PreselectBlock> = Vec::new();

    // A single active `PREPARE { ... }` frame.  The deq JIT emitter
    // never nests these, and hand-authored circuits should follow the
    // same contract; opening a second `PREPARE {` while one is already
    // active is a parse error.
    struct Frame {
        block_start_global_meas: usize,
        lines: Vec<String>,
        checks: Vec<PreselectCheck>,
    }
    let mut current: Option<Frame> = None;

    // Global measurement count across the whole circuit so far.
    let mut global_meas: usize = 0;

    // Buffer for real instructions that live outside any PREPARE
    // block; flushed into a plain (no-check) `PreselectBlock` whenever
    // a `PREPARE {` opens or the input ends.
    let mut plain_lines: Vec<String> = Vec::new();

    let flush_plain = |plain_lines: &mut Vec<String>, blocks: &mut Vec<PreselectBlock>| {
        if !plain_lines.is_empty() {
            blocks.push(PreselectBlock {
                stim_text: plain_lines.join("\n"),
                checks: Vec::new(),
            });
            plain_lines.clear();
        }
    };

    for raw_line in stim_text.lines() {
        let trimmed = raw_line.trim();

        // PREPARE { — open a block.
        if let Some(rest) = trimmed.strip_prefix("PREPARE") {
            let rest = rest.trim_start();
            let open_brace_ok = rest.starts_with('{') && rest.trim_end_matches(|c: char| c.is_whitespace()) == "{";
            assert!(
                open_brace_ok,
                "PREPARE must be immediately followed by '{{' on the same line; got: {raw_line:?}"
            );
            assert!(
                current.is_none(),
                "nested PREPARE blocks are not supported; close the current \
                 block with `}}` before opening another one"
            );
            flush_plain(&mut plain_lines, &mut blocks);
            current = Some(Frame {
                block_start_global_meas: global_meas,
                lines: Vec::new(),
                checks: Vec::new(),
            });
            continue;
        }

        // } — close the active block, but only if there is one.  A
        // standalone `}` outside a PREPARE block is left in the plain
        // buffer (e.g. a REPEAT block's closing brace).
        if trimmed == "}" && current.is_some() {
            let frame = current.take().expect("current non-empty (just checked)");
            blocks.push(PreselectBlock {
                stim_text: frame.lines.join("\n"),
                checks: frame.checks,
            });
            continue;
        }

        // REQUIRE inside a block — parse and record.
        if let Some(rest) = trimmed.strip_prefix("REQUIRE") {
            let frame = current
                .as_mut()
                .expect("REQUIRE outside a PREPARE block; wrap the REQUIRE in `PREPARE { ... }`");
            let block_meas_count = global_meas - frame.block_start_global_meas;
            let mut targets: Vec<RequireTarget> = Vec::new();
            for token in rest.split_whitespace() {
                let (negated, body) = if let Some(stripped) = token.strip_prefix('!') {
                    (true, stripped)
                } else {
                    (false, token)
                };
                let k = parse_rec_offset(body)
                    .unwrap_or_else(|| panic!("REQUIRE target must be `rec[-N]` or `!rec[-N]` with N >= 1; got: {token:?}"));
                assert!(k >= 1, "REQUIRE target rec[-0] is not allowed; got: {token:?}");
                assert!(
                    k <= block_meas_count,
                    "REQUIRE target rec[-{k}] references a measurement outside the \
                     enclosing PREPARE block (block has produced only {block_meas_count} \
                     measurement(s) so far)"
                );
                let abs_meas_idx = global_meas - k;
                targets.push(RequireTarget { abs_meas_idx, negated });
            }
            assert!(!targets.is_empty(), "REQUIRE requires at least one measurement target");
            frame.checks.push(PreselectCheck { targets });
            continue;
        }

        // Anything else — a real instruction, comment, or blank line.
        // Update measurement counter if it produces measurements.
        if !trimmed.is_empty() && !trimmed.starts_with('#') {
            let instr_name = trimmed
                .split(|c: char| c.is_whitespace() || c == '(')
                .next()
                .unwrap_or("")
                .to_uppercase();
            assert!(
                instr_name != "REPEAT",
                "REPEAT blocks are not supported alongside PREPARE/REQUIRE"
            );
            if MEASUREMENT_INSTRUCTIONS.contains(&instr_name.as_str()) {
                global_meas += count_measurement_targets(trimmed, &instr_name);
            }
        }

        // Route the line to the active block if any; otherwise it is a
        // plain-segment line that will be flushed when the next
        // PREPARE opens or the input ends.
        if let Some(frame) = current.as_mut() {
            frame.lines.push(raw_line.to_owned());
        } else {
            plain_lines.push(raw_line.to_owned());
        }
    }

    assert!(
        current.is_none(),
        "unterminated PREPARE block (missing `}}` before end of circuit)"
    );

    flush_plain(&mut plain_lines, &mut blocks);
    blocks
}

/// Parse a `rec[-N]` token and return `N` (positive).  Returns `None`
/// if the token doesn't match the expected shape.
fn parse_rec_offset(token: &str) -> Option<usize> {
    let inner = token.strip_prefix("rec[")?.strip_suffix(']')?;
    let inner = inner.strip_prefix('-')?;
    inner.parse::<usize>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        let schedule = extract_preselect_schedule("");
        assert!(schedule.is_empty());
        assert!(extract_preselect_blocks("").is_empty());
    }

    #[test]
    fn no_directives() {
        let text = "H 0\nM 0\n#!delay 0.5\n";
        let schedule = extract_preselect_schedule(text);
        assert!(schedule.is_empty());

        let blocks = extract_preselect_blocks(text);
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].checks.is_empty());
        assert_eq!(blocks[0].stim_text, "H 0\nM 0\n#!delay 0.5");
    }

    #[test]
    fn strip_leaves_plain_text_alone() {
        let text = "R 0\nH 0\nM 0\n";
        assert_eq!(strip_preselect_directives(text), "R 0\nH 0\nM 0");
    }

    #[test]
    fn single_target_require() {
        let text = "PREPARE {\nH 0\nM 0\nREQUIRE rec[-1]\n}\n";
        let blocks = extract_preselect_blocks(text);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].stim_text, "H 0\nM 0");
        assert_eq!(blocks[0].checks.len(), 1);
        assert_eq!(
            blocks[0].checks[0].targets,
            vec![RequireTarget {
                abs_meas_idx: 0,
                negated: false,
            }]
        );
    }

    #[test]
    fn negated_target_require() {
        let text = "PREPARE {\nH 0\nM 0\nREQUIRE !rec[-1]\n}\n";
        let schedule = extract_preselect_schedule(text);
        assert_eq!(schedule.checks.len(), 1);
        assert!(schedule.checks[0].targets[0].negated);
    }

    #[test]
    fn multi_target_require() {
        let text = "\
PREPARE {
H 0
H 1
M 0
M 1
REQUIRE !rec[-1] rec[-2]
}
";
        let schedule = extract_preselect_schedule(text);
        assert_eq!(schedule.checks.len(), 1);
        let targets = &schedule.checks[0].targets;
        assert_eq!(targets.len(), 2);
        // rec[-1] refers to the most recent measurement (M 1 -> abs idx 1);
        // rec[-2] refers to the second most recent (M 0 -> abs idx 0).
        assert_eq!(targets[0].abs_meas_idx, 1);
        assert!(targets[0].negated);
        assert_eq!(targets[1].abs_meas_idx, 0);
        assert!(!targets[1].negated);
    }

    #[test]
    fn parity_encoded_via_single_negation_is_odd_parity() {
        // `REQUIRE !rec[-1] rec[-2]` succeeds when the XOR of the two
        // measurements is 1 (odd parity).
        let text = "PREPARE {\nH 0\nH 1\nM 0\nM 1\nREQUIRE !rec[-1] rec[-2]\n}\n";
        let schedule = extract_preselect_schedule(text);
        let check = &schedule.checks[0];

        // (0, 0): XOR == 0 → odd-parity requirement fails
        assert!(!check.is_satisfied(|_| false));
        // (1, 1): XOR == 0 → still fails
        assert!(!check.is_satisfied(|_| true));
        // (1, 0): m1=1, m0=0 → satisfied
        assert!(check.is_satisfied(|idx| idx == 1));
        // (0, 1): m1=0, m0=1 → satisfied
        assert!(check.is_satisfied(|idx| idx == 0));
    }

    #[test]
    fn prefix_body_and_tail_split_into_three_blocks() {
        let text = "\
R 0 1
PREPARE {
H 0
M 0
REQUIRE rec[-1]
}
CZ 0 1
M 0
";
        let blocks = extract_preselect_blocks(text);
        assert_eq!(blocks.len(), 3);
        // Plain prefix.
        assert_eq!(blocks[0].stim_text, "R 0 1");
        assert!(blocks[0].checks.is_empty());
        // PREPARE body.
        assert_eq!(blocks[1].stim_text, "H 0\nM 0");
        assert_eq!(blocks[1].checks.len(), 1);
        // Plain tail.
        assert_eq!(blocks[2].stim_text, "CZ 0 1\nM 0");
        assert!(blocks[2].checks.is_empty());

        // The stripped text is exactly the concatenation of every
        // block's stim_text.
        assert_eq!(strip_preselect_directives(text), "R 0 1\nH 0\nM 0\nCZ 0 1\nM 0");
    }

    #[test]
    fn multiple_prepare_blocks() {
        let text = "\
PREPARE {
H 0
M 0
REQUIRE rec[-1]
}
CZ 0 1
PREPARE {
H 1
M 1
REQUIRE !rec[-1]
}
";
        let blocks = extract_preselect_blocks(text);
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].checks[0].targets[0].abs_meas_idx, 0);
        assert!(!blocks[0].checks[0].targets[0].negated);
        assert_eq!(blocks[1].stim_text, "CZ 0 1");
        assert!(blocks[1].checks.is_empty());
        assert_eq!(blocks[2].checks[0].targets[0].abs_meas_idx, 1);
        assert!(blocks[2].checks[0].targets[0].negated);
    }

    #[test]
    fn adjacent_prepare_blocks_skip_empty_plain_segment() {
        let text = "\
PREPARE {
H 0
M 0
REQUIRE rec[-1]
}
PREPARE {
H 1
M 1
REQUIRE rec[-1]
}
";
        let blocks = extract_preselect_blocks(text);
        assert_eq!(blocks.len(), 2, "no empty plain block between two PREPAREs");
        assert!(!blocks[0].checks.is_empty());
        assert!(!blocks[1].checks.is_empty());
    }

    #[test]
    #[should_panic(expected = "REQUIRE outside a PREPARE block")]
    fn require_outside_prepare_panics() {
        let _ = extract_preselect_blocks("M 0\nREQUIRE rec[-1]\n");
    }

    #[test]
    #[should_panic(expected = "unterminated PREPARE block")]
    fn unterminated_prepare_panics() {
        let _ = extract_preselect_blocks("PREPARE {\nM 0\nREQUIRE rec[-1]\n");
    }

    #[test]
    #[should_panic(expected = "nested PREPARE blocks are not supported")]
    fn nested_prepare_panics() {
        let text = "\
PREPARE {
M 0
PREPARE {
M 1
REQUIRE rec[-1]
}
REQUIRE rec[-1]
}
";
        let _ = extract_preselect_blocks(text);
    }

    #[test]
    #[should_panic(expected = "rec[-0] is not allowed")]
    fn rec_zero_panics() {
        let _ = extract_preselect_blocks("PREPARE {\nM 0\nREQUIRE rec[-0]\n}\n");
    }

    #[test]
    #[should_panic(expected = "references a measurement outside the enclosing PREPARE block")]
    fn out_of_block_reference_panics() {
        // Only one measurement inside the block, but REQUIRE asks for
        // rec[-2], which would reference something before PREPARE.
        let text = "M 0\nPREPARE {\nM 1\nREQUIRE rec[-2]\n}\n";
        let _ = extract_preselect_blocks(text);
    }
}
