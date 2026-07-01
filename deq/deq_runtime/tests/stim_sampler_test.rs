#![cfg(feature = "simulator")]

use deq_runtime::misc::bit_vector;
use deq_runtime::simulator::DeterministicRng;
use deq_runtime::simulator::common::{Sampler, StimSampler};
use rand::SeedableRng;

/// A deterministic circuit: H on qubit 0, CNOT 0→1, measure both.
/// Always produces measurements [0, 0] or [1, 1] (no noise).
const BELL_CIRCUIT: &str = "\
H 0
CNOT 0 1
M 0 1
";

/// A circuit with depolarizing noise that will flip measurements.
const NOISY_CIRCUIT: &str = "\
X_ERROR(1.0) 0
M 0 1
";

#[test]
fn basic_sampling_returns_correct_measurement_count() {
    let sampler = StimSampler::new(BELL_CIRCUIT, 42, 0, false);
    let mut rng = DeterministicRng::seed_from_u64(0);
    let sample = sampler.sample(&mut rng);
    assert_eq!(sample.measurements.size, 2, "Bell circuit has 2 measurements");
}

#[test]
fn deterministic_seeding_produces_reproducible_results() {
    let sampler1 = StimSampler::new(BELL_CIRCUIT, 123, 0, false);
    let sampler2 = StimSampler::new(BELL_CIRCUIT, 123, 0, false);
    let mut rng = DeterministicRng::seed_from_u64(0);

    let mut results1 = Vec::new();
    let mut results2 = Vec::new();
    for _ in 0..20 {
        results1.push(sampler1.sample(&mut rng).measurements);
        results2.push(sampler2.sample(&mut rng).measurements);
    }
    assert_eq!(results1, results2, "Same seed must produce identical samples");
}

#[test]
fn different_seeds_produce_different_results() {
    let sampler1 = StimSampler::new(BELL_CIRCUIT, 1, 0, false);
    let sampler2 = StimSampler::new(BELL_CIRCUIT, 2, 0, false);
    let mut rng = DeterministicRng::seed_from_u64(0);

    let mut any_different = false;
    for _ in 0..50 {
        let s1 = sampler1.sample(&mut rng);
        let s2 = sampler2.sample(&mut rng);
        if s1.measurements != s2.measurements {
            any_different = true;
            break;
        }
    }
    assert!(any_different, "Different seeds should produce different samples");
}

#[test]
fn bell_state_measurements_are_correlated() {
    let sampler = StimSampler::new(BELL_CIRCUIT, 99, 0, false);
    let mut rng = DeterministicRng::seed_from_u64(0);

    for _ in 0..100 {
        let sample = sampler.sample(&mut rng);
        let bits = bit_vector::unpack_bits(&sample.measurements.data, sample.measurements.size);
        assert_eq!(bits[0], bits[1], "Bell pair measurements must be equal");
    }
}

#[test]
fn noisy_circuit_flips_qubit_zero() {
    let sampler = StimSampler::new(NOISY_CIRCUIT, 0, 0, false);
    let mut rng = DeterministicRng::seed_from_u64(0);

    for _ in 0..10 {
        let sample = sampler.sample(&mut rng);
        let bits = bit_vector::unpack_bits(&sample.measurements.data, sample.measurements.size);
        assert!(bits[0], "X_ERROR(1.0) must always flip qubit 0");
        assert!(!bits[1], "Qubit 1 has no noise, should remain 0");
    }
}

#[test]
fn skip_shots_advances_the_sampler() {
    let sampler_no_skip = StimSampler::new(BELL_CIRCUIT, 42, 0, false);
    let sampler_skip_5 = StimSampler::new(BELL_CIRCUIT, 42, 5, false);
    let mut rng = DeterministicRng::seed_from_u64(0);

    // Advance sampler_no_skip by 5 to match
    for _ in 0..5 {
        sampler_no_skip.sample(&mut rng);
    }

    for _ in 0..10 {
        let s1 = sampler_no_skip.sample(&mut rng);
        let s2 = sampler_skip_5.sample(&mut rng);
        assert_eq!(
            s1.measurements, s2.measurements,
            "Skipped sampler should match advanced sampler"
        );
    }
}

#[test]
fn strict_timing_produces_valid_samples() {
    let sampler = StimSampler::new(BELL_CIRCUIT, 42, 0, true);
    let mut rng = DeterministicRng::seed_from_u64(0);

    for _ in 0..20 {
        let sample = sampler.sample(&mut rng);
        assert_eq!(sample.measurements.size, 2);
        let bits = bit_vector::unpack_bits(&sample.measurements.data, sample.measurements.size);
        assert_eq!(bits[0], bits[1], "Bell pair measurements must be equal (strict timing)");
    }
}

#[test]
fn strict_timing_is_deterministic() {
    let sampler1 = StimSampler::new(BELL_CIRCUIT, 77, 0, true);
    let sampler2 = StimSampler::new(BELL_CIRCUIT, 77, 0, true);
    let mut rng = DeterministicRng::seed_from_u64(0);

    for _ in 0..20 {
        let s1 = sampler1.sample(&mut rng);
        let s2 = sampler2.sample(&mut rng);
        assert_eq!(
            s1.measurements, s2.measurements,
            "Strict timing with same seed must be deterministic"
        );
    }
}

#[test]
fn error_set_fields_are_empty_for_stim() {
    let sampler = StimSampler::new(BELL_CIRCUIT, 42, 0, false);
    let mut rng = DeterministicRng::seed_from_u64(0);
    let sample = sampler.sample(&mut rng);

    assert!(sample.errors.is_empty(), "StimSampler should not track errors");
}

#[test]
fn shot_sample_loss_mask_is_absent_for_stim() {
    use deq_runtime::simulator::common::error_set_to_shot_sample;

    let sampler = StimSampler::new(BELL_CIRCUIT, 42, 0, false);
    let mut rng = DeterministicRng::seed_from_u64(0);
    for _ in 0..5 {
        let sample = sampler.sample(&mut rng);
        assert!(
            sample.loss_mask.is_none(),
            "StimSampler ErrorSet.loss_mask must be None (no loss awareness)"
        );
        let shot = error_set_to_shot_sample(&sample);
        assert!(
            shot.loss_mask.is_none(),
            "ShotSample.loss_mask must be absent when the sampler does not track loss"
        );
        assert!(shot.outcomes.is_some(), "ShotSample.outcomes must still be populated");
    }
}

#[test]
fn readouts_match_always_returns_true() {
    use deq_runtime::util::BitVector;

    let sampler = StimSampler::new(BELL_CIRCUIT, 42, 0, false);
    let a = BitVector {
        size: 2,
        data: vec![0b11],
    };
    let b = BitVector {
        size: 2,
        data: vec![0b00],
    };
    assert!(
        sampler.readouts_match(&a, &b),
        "StimSampler readouts_match should always return true"
    );
}

#[test]
#[should_panic(expected = "sample_single_error is not supported")]
fn sample_single_error_panics() {
    let sampler = StimSampler::new(BELL_CIRCUIT, 42, 0, false);
    sampler.sample_single_error(0);
}

#[test]
#[should_panic(expected = "count_single_error is not supported")]
fn count_single_error_panics() {
    let sampler = StimSampler::new(BELL_CIRCUIT, 42, 0, false);
    sampler.count_single_error();
}

#[test]
fn error_tag_returns_empty() {
    let sampler = StimSampler::new(BELL_CIRCUIT, 42, 0, false);
    assert_eq!(sampler.error_tag(0, 0), "");
}

#[test]
fn sampler_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<StimSampler>();
}

#[test]
fn drop_stops_background_thread() {
    // Create and immediately drop — should not hang or leak
    let sampler = StimSampler::new(BELL_CIRCUIT, 42, 0, false);
    drop(sampler);
}

#[test]
fn many_samples_do_not_deadlock() {
    let sampler = StimSampler::new(BELL_CIRCUIT, 42, 0, false);
    let mut rng = DeterministicRng::seed_from_u64(0);
    for _ in 0..1000 {
        let sample = sampler.sample(&mut rng);
        assert_eq!(sample.measurements.size, 2);
    }
}
