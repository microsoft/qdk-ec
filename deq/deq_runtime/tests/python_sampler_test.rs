#![cfg(all(feature = "simulator", feature = "python"))]

//! Unit tests for `PythonSampler`.
//!
//! Uses a tiny in-test Python sampler that returns canned shots so the
//! tests do not require any third-party Python package (e.g. ``qdk``) to
//! be installed in the embedded interpreter.

use deq_runtime::misc::bit_vector;
use deq_runtime::simulator::DeterministicRng;
use deq_runtime::simulator::common::Sampler;
use deq_runtime::simulator::python_sampler::{PythonSampler, PythonSamplerConfig};
use rand::SeedableRng;
use std::io::Write;

const TEST_CIRCUIT: &str = "H 0\nCNOT 0 1\nM 0 1\n";

/// Write a Python sampler module to a tempfile and return both the
/// tempfile (kept alive for the duration of the test) and its path.
fn write_python_sampler(body: &str) -> (tempfile::NamedTempFile, String) {
    let mut f = tempfile::Builder::new().suffix(".py").tempfile().expect("create tempfile");
    f.write_all(body.as_bytes()).expect("write tempfile");
    let path = f.path().to_string_lossy().into_owned();
    (f, path)
}

fn make_sampler(py_body: &str, num_measurements: usize) -> (tempfile::NamedTempFile, PythonSampler) {
    let (file, path) = write_python_sampler(py_body);
    let config = PythonSamplerConfig {
        sampler: path,
        name: "Sampler".to_string(),
        py_config: None,
    };
    let sampler = PythonSampler::new(TEST_CIRCUIT, &config, 42, 0, num_measurements);
    (file, sampler)
}

const CANNED_NO_LOSS: &str = r#"
class Sampler:
    def __init__(self, circuit_text, config):
        self.shots = ["01", "10", "11", "00"]
        self.i = 0

    def sample(self):
        shot = self.shots[self.i % len(self.shots)]
        self.i += 1
        return shot
"#;

const CANNED_ALL_LOSS: &str = r#"
class Sampler:
    def __init__(self, circuit_text, config):
        pass

    def sample(self):
        return "--"
"#;

const CANNED_MIXED: &str = r#"
class Sampler:
    def __init__(self, circuit_text, config):
        pass

    def sample(self):
        return "1-"
"#;

#[test]
fn sample_returns_expected_measurement_count() {
    let (_file, sampler) = make_sampler(CANNED_NO_LOSS, 2);
    let mut rng = DeterministicRng::seed_from_u64(0);
    let s = sampler.sample(&mut rng);
    assert_eq!(s.measurements.size, 2);
}

#[test]
fn zero_and_one_chars_map_to_bits() {
    let (_file, sampler) = make_sampler(CANNED_NO_LOSS, 2);
    let mut rng = DeterministicRng::seed_from_u64(0);

    let expected = [[false, true], [true, false], [true, true], [false, false]];
    for row in expected {
        let s = sampler.sample(&mut rng);
        let bits = bit_vector::unpack_bits(&s.measurements.data, s.measurements.size);
        assert_eq!(bits, row.to_vec());
        // No '-' chars in this canned shot → loss_mask is all-zero.
        let loss_bv = s.loss_mask.as_ref().expect("PythonSampler always reports loss_mask");
        let loss_bits = bit_vector::unpack_bits(&loss_bv.data, loss_bv.size);
        assert_eq!(loss_bits, vec![false, false]);
        rng.jump();
    }
}

#[test]
fn loss_mask_marks_dash_positions() {
    // Mixed canned shot "1-" → measurements are 1 + random; loss_mask is [0, 1].
    let (_file, sampler) = make_sampler(CANNED_MIXED, 2);
    let mut rng = DeterministicRng::seed_from_u64(42);
    for _ in 0..10 {
        let s = sampler.sample(&mut rng);
        let loss_bv = s.loss_mask.as_ref().expect("PythonSampler always reports loss_mask");
        let loss_bits = bit_vector::unpack_bits(&loss_bv.data, loss_bv.size);
        assert_eq!(loss_bits, vec![false, true], "loss_mask should mark only '-' positions");
        rng.jump();
    }
}

#[test]
fn dash_maps_to_placeholder_false() {
    // The sampler no longer randomizes loss bits — the coordinator does
    // that via `apply_loss_random_imputation`.  Each `'-'` here becomes
    // a deterministic `false` placeholder in `measurements`, with the
    // corresponding `loss_mask` bit set.
    let (_file, sampler) = make_sampler(CANNED_ALL_LOSS, 2);
    let mut rng = DeterministicRng::seed_from_u64(1);
    for _ in 0..10 {
        let s = sampler.sample(&mut rng);
        let bits = bit_vector::unpack_bits(&s.measurements.data, s.measurements.size);
        assert_eq!(bits, vec![false, false], "'-' should produce placeholder false bits");
        let loss_bv = s.loss_mask.as_ref().expect("PythonSampler always reports loss_mask");
        let loss_bits = bit_vector::unpack_bits(&loss_bv.data, loss_bv.size);
        assert_eq!(loss_bits, vec![true, true]);
        rng.jump();
    }
}

#[test]
fn sample_is_deterministic_across_samplers() {
    // Two independent samplers built from the same canned shot produce
    // byte-identical outputs; the sampler no longer depends on the rng.
    let (_file, sampler_a) = make_sampler(CANNED_MIXED, 2);
    let (_file2, sampler_b) = make_sampler(CANNED_MIXED, 2);

    let mut rng_a = DeterministicRng::seed_from_u64(7);
    let mut rng_b = DeterministicRng::seed_from_u64(99);
    for _ in 0..20 {
        let a = sampler_a.sample(&mut rng_a);
        let b = sampler_b.sample(&mut rng_b);
        assert_eq!(a.measurements, b.measurements);
        assert_eq!(a.loss_mask, b.loss_mask);
        rng_a.jump();
        rng_b.jump();
    }
}

#[test]
#[should_panic(expected = "expected '0', '1', or '-'")]
fn invalid_char_panics() {
    let body = r#"
class Sampler:
    def __init__(self, circuit_text, config):
        pass

    def sample(self):
        return "0X"
"#;
    let (_file, sampler) = make_sampler(body, 2);
    let mut rng = DeterministicRng::seed_from_u64(0);
    let _ = sampler.sample(&mut rng);
}

#[test]
#[should_panic(expected = "expected 2 measurement chars")]
fn wrong_length_panics() {
    let body = r#"
class Sampler:
    def __init__(self, circuit_text, config):
        pass

    def sample(self):
        return "0"
"#;
    let (_file, sampler) = make_sampler(body, 2);
    let mut rng = DeterministicRng::seed_from_u64(0);
    let _ = sampler.sample(&mut rng);
}

#[test]
fn custom_class_name_is_honored() {
    let body = r#"
class MySampler:
    def __init__(self, circuit_text, config):
        self.call = 0

    def sample(self):
        self.call += 1
        return "01"
"#;
    let (file, path) = write_python_sampler(body);
    let config = PythonSamplerConfig {
        sampler: path,
        name: "MySampler".to_string(),
        py_config: None,
    };
    let sampler = PythonSampler::new(TEST_CIRCUIT, &config, 0, 0, 2);
    let mut rng = DeterministicRng::seed_from_u64(0);
    let s = sampler.sample(&mut rng);
    let bits = bit_vector::unpack_bits(&s.measurements.data, s.measurements.size);
    assert_eq!(bits, vec![false, true]);
    drop(file);
}

#[test]
fn py_config_is_forwarded_with_injected_keys() {
    // The sampler asserts that the auto-injected `seed`, `skip_shots`, and
    // `num_measurements` keys are present and equal to what the Rust side
    // provided.  We assert this from inside Python and let the constructor
    // raise on mismatch.
    let body = r#"
class Sampler:
    def __init__(self, circuit_text, config):
        assert config["seed"] == 99, f"seed={config['seed']!r}"
        assert config["skip_shots"] == 3, f"skip_shots={config['skip_shots']!r}"
        assert config["num_measurements"] == 2, f"num_measurements={config['num_measurements']!r}"
        assert config["custom"] == "hello", f"custom={config['custom']!r}"

    def sample(self):
        return "00"
"#;
    let (file, path) = write_python_sampler(body);
    let config = PythonSamplerConfig {
        sampler: path,
        name: "Sampler".to_string(),
        py_config: Some(serde_json::json!({"custom": "hello"})),
    };
    let _sampler = PythonSampler::new(TEST_CIRCUIT, &config, 99, 3, 2);
    drop(file);
}
