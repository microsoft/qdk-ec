//! Shared sampling utilities for noisy simulation.

use rand::Rng;

/// Batched geometric sampler for efficient sparse event generation.
///
/// Generates skip counts for geometric sampling with parameter p, allowing
/// efficient iteration over sparse events (gates that fail with probability p).
pub struct GeometricSampler {
    log_one_minus_p: f64,
    random_buffer: [f64; Self::BATCH_SIZE],
    skip_buffer: [usize; Self::BATCH_SIZE],
    buffer_index: usize,
}

impl GeometricSampler {
    const BATCH_SIZE: usize = 1024;

    /// Create a sampler for events occurring with probability `p_error`.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `p_error` is in `(0.0, 1.0]`.
    #[must_use]
    pub fn new(p_error: f64) -> Self {
        debug_assert!(
            p_error > 0.0 && p_error <= 1.0,
            "p_error must be in (0, 1], got {p_error}"
        );
        // For p=1.0, ln(0) = -inf, which makes floor(x / -inf) = 0 for all x > 0.
        // This correctly produces skip=0 (every shot faults).
        Self {
            log_one_minus_p: (1.0 - p_error).ln(),
            random_buffer: [0.0; Self::BATCH_SIZE],
            skip_buffer: [0; Self::BATCH_SIZE],
            buffer_index: Self::BATCH_SIZE,
        }
    }

    /// Returns the number of events to skip before the next fault.
    pub fn next_skip<R: Rng>(&mut self, rng: &mut R) -> usize {
        if self.buffer_index >= Self::BATCH_SIZE {
            self.refill_buffers(rng);
        }
        let skip = self.skip_buffer[self.buffer_index];
        self.buffer_index += 1;
        skip
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn refill_buffers<R: Rng>(&mut self, rng: &mut R) {
        for value in &mut self.random_buffer {
            *value = rng.gen();
        }
        for (skip, &uniform) in self.skip_buffer.iter_mut().zip(self.random_buffer.iter()) {
            *skip = (uniform.ln() / self.log_one_minus_p).floor() as usize;
        }
        self.buffer_index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistical_testing::{assert_rate_within_tolerance, TOLERANCE_HIGH_SAMPLES, TOLERANCE_LOW_SAMPLES};
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    fn count_geometric_events(sampler: &mut GeometricSampler, rng: &mut SmallRng, total_trials: usize) -> usize {
        let mut event_count = 0;
        let mut position = 0;
        while position < total_trials {
            let skip = sampler.next_skip(rng);
            position += skip + 1;
            if position <= total_trials {
                event_count += 1;
            }
        }
        event_count
    }

    #[test]
    fn geometric_sampler_produces_correct_rate() {
        let mut rng = SmallRng::seed_from_u64(42);
        let probability = 0.1;
        let mut sampler = GeometricSampler::new(probability);
        let total_trials = 100_000;

        let event_count = count_geometric_events(&mut sampler, &mut rng, total_trials);

        assert_rate_within_tolerance(
            event_count,
            total_trials,
            probability,
            TOLERANCE_HIGH_SAMPLES,
            "GeometricSampler p=0.1",
        );
    }

    #[test]
    fn geometric_sampler_probability_one_produces_no_skips() {
        let mut rng = SmallRng::seed_from_u64(42);
        let mut sampler = GeometricSampler::new(1.0);

        for _ in 0..1000 {
            assert_eq!(sampler.next_skip(&mut rng), 0, "p=1.0 should never skip");
        }
    }

    #[test]
    fn geometric_sampler_low_probability_produces_sparse_events() {
        let mut rng = SmallRng::seed_from_u64(42);
        let probability = 0.001;
        let mut sampler = GeometricSampler::new(probability);
        let total_trials = 1_000_000;

        let event_count = count_geometric_events(&mut sampler, &mut rng, total_trials);

        assert_rate_within_tolerance(
            event_count,
            total_trials,
            probability,
            TOLERANCE_LOW_SAMPLES,
            "GeometricSampler p=0.001",
        );
    }
}
