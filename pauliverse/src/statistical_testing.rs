//! Test utilities for statistical assertions in sampling tests.
//!
//! This module provides shared helpers for verifying:
//! - Event rates match expected probabilities (e.g., fault injection rate tests)
//! - Distributions are uniform over discrete outcomes (chi-squared style tests)
//!
//! Standard tolerance thresholds:
//! - ±5% for high sample counts (≥100K samples)
//! - ±10% for lower sample counts or high-variance scenarios

use binar::Bitwise;
use paulimer::pauli::Pauli;

/// Tolerance for tests with ≥100K samples.
pub const TOLERANCE_HIGH_SAMPLES: f64 = 0.05;

/// Tolerance for tests with fewer samples or higher variance.
pub const TOLERANCE_LOW_SAMPLES: f64 = 0.10;

/// Assert that the observed event rate matches the expected probability within tolerance.
///
/// # Arguments
/// - `observed`: Number of events observed
/// - `total`: Total number of trials
/// - `expected_probability`: The expected probability of an event
/// - `relative_tolerance`: Acceptable relative deviation (e.g., 0.05 for ±5%)
/// - `context`: Description for error messages
///
/// # Panics
/// Panics if the observed rate deviates from expected by more than the tolerance.
#[allow(clippy::cast_precision_loss)]
pub fn assert_rate_within_tolerance(
    observed: usize,
    total: usize,
    expected_probability: f64,
    relative_tolerance: f64,
    context: &str,
) {
    let observed_rate = observed as f64 / total as f64;
    let lower = expected_probability * (1.0 - relative_tolerance);
    let upper = expected_probability * (1.0 + relative_tolerance);
    assert!(
        (lower..upper).contains(&observed_rate),
        "{context}: observed rate {observed_rate:.4} deviates from expected \
         {expected_probability:.4} by more than {:.0}%",
        relative_tolerance * 100.0
    );
}

/// Assert that counts are uniformly distributed within tolerance.
///
/// Uses a per-bucket check that each count is within `relative_tolerance` of the expected
/// count (total / `num_buckets`).
///
/// # Arguments
/// - `counts`: Observed counts for each outcome
/// - `total_trials`: Total number of samples taken
/// - `relative_tolerance`: Acceptable relative deviation per bucket (e.g., 0.05 for ±5%)
/// - `context`: Description for error messages
///
/// # Panics
/// Panics if any bucket's count deviates from expected by more than the tolerance.
#[allow(clippy::cast_precision_loss)]
pub fn assert_uniform_distribution(counts: &[u32], total_trials: usize, relative_tolerance: f64, context: &str) {
    let expected = total_trials as f64 / counts.len() as f64;
    for (index, &count) in counts.iter().enumerate() {
        let ratio = f64::from(count) / expected;
        assert!(
            ((1.0 - relative_tolerance)..(1.0 + relative_tolerance)).contains(&ratio),
            "{context}: outcome {index} count {count} deviates from expected {expected:.0} \
             by {:.1}% (tolerance: ±{:.0}%)",
            (ratio - 1.0).abs() * 100.0,
            relative_tolerance * 100.0
        );
    }
}

/// Encode a two-qubit Pauli operator as a 4-bit index (0-15).
///
/// The encoding is: `x0 | (z0 << 1) | (x1 << 2) | (z1 << 3)`
///
/// This maps the 16 two-qubit Paulis {I, X, Y, Z}⊗{I, X, Y, Z} to indices 0-15,
/// where 0 is II (identity) and 1-15 are the non-identity Paulis.
pub fn two_qubit_pauli_to_bit_index<P: Pauli>(pauli: &P) -> u32 {
    let x_q0 = u32::from(pauli.x_bits().index(0));
    let z_q0 = u32::from(pauli.z_bits().index(0));
    let x_q1 = u32::from(pauli.x_bits().index(1));
    let z_q1 = u32::from(pauli.z_bits().index(1));
    x_q0 | (z_q0 << 1) | (x_q1 << 2) | (z_q1 << 3)
}
