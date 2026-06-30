//! Python sampler
//!
//! Calls a sampler implemented in Python with the following protocol:
//!
//! ```python
//! class Sampler:
//!     def __init__(self, circuit_text: str, config: dict) -> None: ...
//!     def sample(self) -> str:
//!         """Return one shot as a length-N string of '0', '1', or '-' chars,
//!         where N == circuit.num_measurements().  '-' means the qubit was
//!         lost during the corresponding measurement; the Rust side packs a
//!         placeholder `false` bit at that position and sets the
//!         corresponding bit of `ErrorSet.loss_mask` to 1.  The coordinator
//!         then decides what to do with those flagged bits (random
//!         imputation by default; see `apply_loss_random_imputation`).
//!         ('-' is used rather than 'L' because 'L' and '1' look nearly
//!         identical in a fixed-width string.)
//!         """
//! ```
//!
//! The class name defaults to ``Sampler`` and can be overridden via the
//! ``name`` field in the sampler JSON config.  Constructor ``config`` is
//! the user-supplied ``py_config`` dictionary augmented with the simulator
//! ``seed``, ``skip_shots`` and ``num_measurements`` so that the adapter
//! can seed its own RNG and pre-advance shots if desired.
//!
//! ## Loss handling
//!
//! The Rust side maps each ``'-'`` to a placeholder `false` bit in
//! ``ErrorSet.measurements`` and sets the corresponding bit of
//! ``ErrorSet.loss_mask`` to 1.  The actual decision of what to do with
//! lost bits — random imputation, erasure handling, etc. — lives in the
//! coordinator (see ``coordinator::apply_loss_random_imputation``).  That
//! way any future loss-aware sampler "just works": it only needs to emit
//! correct ``loss_mask`` bits and a sensible placeholder; the coordinator
//! handles the rest.
use crate::misc::bit_vector;
use crate::misc::python::{get_or_load_module, json_value_to_py};
use crate::simulator::DeterministicRng;
use crate::simulator::common::{ErrorSet, Sampler};
use crate::util::BitVector;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
#[cfg(feature = "cli")]
use structdoc::StructDoc;

/// Configuration for a Python-backed sampler.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
pub struct PythonSamplerConfig {
    /// The entry file of the Python sampler (a ``*.py`` file).
    pub file: String,
    /// The name of the sampler class inside the Python file; defaults to ``Sampler``.
    #[serde(default = "default_class_name")]
    pub name: String,
    /// Arbitrary Python sampler parameters; forwarded as the second
    /// argument to the Python class constructor.  The simulator-level
    /// ``seed``, ``skip_shots`` and ``num_measurements`` fields are
    /// auto-merged in unless already present.
    #[cfg_attr(feature = "cli", structdoc(skip))]
    #[serde(default)]
    pub py_config: Option<serde_json::Value>,
}

fn default_class_name() -> String {
    "Sampler".to_string()
}

/// A sampler that forwards each shot request to a Python ``sample()`` method.
///
/// The Python object is held behind a [`Mutex`] so the sampler can implement
/// [`Sampler`] with `&self`.  Each call:
///
/// 1. Acquires the GIL via [`Python::attach`].
/// 2. Calls ``instance.sample()`` and extracts a ``str``.
/// 3. Maps each char to a bool: ``'0' -> false``, ``'1' -> true``,
///    ``'-' -> false`` (placeholder; the coordinator decides what to do
///    with the corresponding ``loss_mask`` bit).
/// 4. Packs the bools into a [`BitVector`] and returns an [`ErrorSet`].
pub struct PythonSampler {
    instance: Mutex<Py<PyAny>>,
    num_measurements: usize,
}

impl PythonSampler {
    /// Construct a new Python sampler.
    ///
    /// `circuit_text` is passed verbatim as the first constructor argument.
    /// `seed`, `skip_shots`, and `num_measurements` are merged into the
    /// `py_config` dictionary (without overwriting user-provided keys) so
    /// the Python adapter can seed its own RNG and validate the expected
    /// shot length.
    pub fn new(
        circuit_text: &str,
        config: &PythonSamplerConfig,
        seed: u64,
        skip_shots: usize,
        num_measurements: usize,
    ) -> Self {
        let mut py_cfg_json = config.py_config.clone().unwrap_or_else(|| serde_json::json!({}));
        if let serde_json::Value::Object(ref mut map) = py_cfg_json {
            map.entry("seed".to_string()).or_insert(serde_json::json!(seed));
            map.entry("skip_shots".to_string()).or_insert(serde_json::json!(skip_shots));
            map.entry("num_measurements".to_string())
                .or_insert(serde_json::json!(num_measurements));
        } else {
            panic!("py_config must be a JSON object, got: {py_cfg_json}");
        }

        let instance = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let module = get_or_load_module(py, &config.file)?;
            let sampler_class = module.getattr(config.name.as_str())?;
            let py_cfg = json_value_to_py(py, &py_cfg_json)?;
            let inst = sampler_class.call1((circuit_text, py_cfg))?;
            Ok(inst.unbind())
        })
        .expect("failed to construct Python sampler");

        Self {
            instance: Mutex::new(instance),
            num_measurements,
        }
    }

    fn next_shot_string(&self) -> String {
        let guard = self.instance.lock().expect("PythonSampler mutex poisoned");
        Python::attach(|py| -> PyResult<String> {
            let inst = guard.bind(py);
            let py_result = inst.call_method0("sample")?;
            py_result.extract::<String>()
        })
        .expect("Python sampler.sample() raised an exception")
    }
}

impl Sampler for PythonSampler {
    fn sample(&self, _rng: &mut DeterministicRng) -> ErrorSet {
        let shot = self.next_shot_string();
        assert_eq!(
            shot.chars().count(),
            self.num_measurements,
            "Python sampler returned {} chars, expected {} measurement chars",
            shot.chars().count(),
            self.num_measurements
        );

        // Build `measurements` and `loss_flags` side by side.  Every `'-'`
        // contributes a placeholder `false` to `measurements` and a `true`
        // to `loss_flags`; the coordinator is responsible for replacing
        // those placeholder bits with random bits via its
        // `loss_random_imputation` policy.  Doing the imputation at the
        // coordinator (rather than here at the sampler) means any future
        // loss-aware sampler just emits the `loss_mask` and "just works",
        // and loss-aware decoders can opt out of imputation entirely.
        let mut bits: Vec<bool> = Vec::with_capacity(self.num_measurements);
        let mut loss_flags: Vec<bool> = Vec::with_capacity(self.num_measurements);
        for c in shot.chars() {
            match c {
                '0' => {
                    bits.push(false);
                    loss_flags.push(false);
                }
                '1' => {
                    bits.push(true);
                    loss_flags.push(false);
                }
                '-' => {
                    bits.push(false);
                    loss_flags.push(true);
                }
                other => panic!("Python sampler returned invalid char {other:?} (expected '0', '1', or '-')"),
            }
        }

        ErrorSet {
            errors: vec![],
            measurements: BitVector {
                size: bits.len() as u64,
                data: bit_vector::pack_bits(&bits),
            },
            loss_mask: Some(BitVector {
                size: loss_flags.len() as u64,
                data: bit_vector::pack_bits(&loss_flags),
            }),
        }
    }

    fn sample_single_error(&self, _index: usize) -> ErrorSet {
        panic!("sample_single_error is not supported for PythonSampler")
    }

    fn count_single_error(&self) -> usize {
        panic!("count_single_error is not supported for PythonSampler")
    }

    fn readouts_match(&self, _actual: &BitVector, _expected: &BitVector) -> bool {
        // Loss-as-flip discards expected-readout information; treat every
        // shot as decode-only (no logical-error comparison) by default.
        // Users who want logical-error tracking should supply a Rhai
        // ``is_logical_error`` script via ``logical_assert_filepath``.
        true
    }

    fn error_tag(&self, _marginal_index: usize, _error_index: usize) -> &str {
        ""
    }
}
