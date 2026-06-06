//! Parse `#!preselect_begin` and `#!preselect_expect` directives from Stim text.
//!
//! These directives are emitted by the Python `export_program_stim` function
//! and instruct the simulator to enforce measurement outcomes.  The Stim
//! parser ignores them because they are comments (`#`-prefixed).
//!
//! Two consumers exist:
//!
//! * **Resample mode** (static / jit-static simulators): uses
//!   [`extract_preselect_schedule`] to get a flat list of
//!   `(abs_meas_idx, expected)` checks.  After sampling a full shot the
//!   simulator verifies the checks and resamples on failure.
//!
//! * **Retry mode** (preselect simulator): the
//!   `TableauPreselectSampler` parses `#!preselect_begin` boundaries
//!   and replays only the failing segment on check failure.

/// A single preselect check: the measurement at absolute index
/// `abs_meas_idx` must equal `expected`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreselectCheck {
    pub abs_meas_idx: usize,
    pub expected: bool,
}

/// The full preselect schedule extracted from a Stim file.
#[derive(Debug, Clone)]
pub struct PreselectSchedule {
    /// Ordered list of measurement checks.
    pub checks: Vec<PreselectCheck>,
}

impl PreselectSchedule {
    /// Whether any preselect directives were found.
    pub fn is_empty(&self) -> bool {
        self.checks.is_empty()
    }
}

/// Scan `stim_text` for `#!preselect_expect` directives and return the
/// schedule.  `#!preselect_begin` markers are not stored here —
/// they are parsed separately by `TableauPreselectSampler`.
pub fn extract_preselect_schedule(stim_text: &str) -> PreselectSchedule {
    let mut checks = Vec::new();

    for line in stim_text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix("#!preselect_expect") {
            let rest = rest.trim();
            let mut parts = rest.split_whitespace();
            let abs_idx: usize = parts.next().and_then(|s| s.parse().ok()).unwrap_or_else(|| {
                panic!(
                    "#!preselect_expect directive has invalid abs_meas_idx: {rest}\n\
                         Expected: #!preselect_expect <abs_idx> <0|1>"
                )
            });
            let expected_int: u8 = parts.next().and_then(|s| s.parse().ok()).unwrap_or_else(|| {
                panic!(
                    "#!preselect_expect directive has invalid expected value: {rest}\n\
                         Expected: #!preselect_expect <abs_idx> <0|1>"
                )
            });
            checks.push(PreselectCheck {
                abs_meas_idx: abs_idx,
                expected: expected_int != 0,
            });
        }
        // #!preselect_begin is ignored in this schedule — only used by
        // the retry-mode TableauPreselectSampler.
    }

    PreselectSchedule { checks }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        let schedule = extract_preselect_schedule("");
        assert!(schedule.is_empty());
    }

    #[test]
    fn no_preselect_directives() {
        let schedule = extract_preselect_schedule("H 0\nM 0\n#!delay 0.5\n");
        assert!(schedule.is_empty());
    }

    #[test]
    fn single_check() {
        let schedule = extract_preselect_schedule("M 0\n#!preselect_expect 0 0\n");
        assert_eq!(schedule.checks.len(), 1);
        assert_eq!(
            schedule.checks[0],
            PreselectCheck {
                abs_meas_idx: 0,
                expected: false,
            }
        );
    }

    #[test]
    fn multiple_checks_with_begin() {
        let text = "\
            R 0 1\n\
            #!preselect_begin\n\
            H 0\n\
            M 0\n\
            #!preselect_expect 0 0\n\
            H 1\n\
            M 1\n\
            #!preselect_expect 1 1\n\
        ";
        let schedule = extract_preselect_schedule(text);
        assert_eq!(schedule.checks.len(), 2);
        assert_eq!(schedule.checks[0].abs_meas_idx, 0);
        assert!(!schedule.checks[0].expected);
        assert_eq!(schedule.checks[1].abs_meas_idx, 1);
        assert!(schedule.checks[1].expected);
    }
}
