//! Python decoder
//!
//! Calling another decoder written in Python language with the following APIs:
//!
//! class Decoder:
//!     def __init__(self, hypergraph: DecodingHypergraph, config: Dict): ...
//!     def decode(self, syndrome: list[int]) -> list[int]: ...
//!     def reset(self) -> None: ...
//!
//! The class name defaults to `Decoder` and can be overridden by setting the
//! top-level `name` field in the decoder JSON config.
//!

use crate::decoder::blackbox_decoder::{DecodingHypergraph, ParityFactor};
use crate::decoder::thread_pooling::{DecoderInstance, ThreadPoolingConfig, ThreadPoolingDecoder};
use crate::misc::bit_vector::to_sparse_indices;
use crate::misc::python::{get_or_load_module, get_or_load_module_from_source, json_value_to_py};
use crate::util::BitVector;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use serde::{Deserialize, Serialize};
#[cfg(feature = "cli")]
use structdoc::StructDoc;

/// Compile-time-embedded Python decoder adapters.
///
/// When a [`PythonDecoderConfig::file`] value starts with `@`, the
/// string after the `@` is looked up here instead of being treated as
/// a filesystem path.  Ships baked into the ``deq_runtime`` binary so
/// callers never need to know where the reference decoder adapters
/// live on disk.  The `@` prefix is reserved: no filesystem path
/// starting with `@` will be opened by the decoder.
mod builtin_decoders {
    /// Return `(virtual_filename, source_code)` for a named builtin, or
    /// `None` if the name is unknown.  ``virtual_filename`` is what
    /// Python tracebacks display (typically the `@name` sentinel).
    pub fn lookup(name: &str) -> Option<(&'static str, &'static str)> {
        match name {
            "naive_decoder" => Some(("@naive_decoder", include_str!("naive_decoder.py"))),
            "relay_bp_decoder" => Some(("@relay_bp_decoder", include_str!("relay_bp_decoder.py"))),
            "tesseract_decoder" => Some(("@tesseract_decoder", include_str!("tesseract_decoder.py"))),
            _ => None,
        }
    }

    /// All known builtin decoder names (without the leading `@`).
    pub fn names() -> &'static [&'static str] {
        &["naive_decoder", "relay_bp_decoder", "tesseract_decoder"]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct PythonDecoderConfig {
    /// we want to recognize all the thread pooling config fields
    #[serde(flatten)]
    pub thread_pooling_config: ThreadPoolingConfig,
    /// Where to find the Python decoder.
    ///
    /// * A filesystem path to a ``*.py`` file, or
    /// * a ``@name`` sentinel that resolves to a compile-time-embedded
    ///   adapter in the [`builtin_decoders`] registry above (currently
    ///   ``@naive_decoder``, ``@relay_bp_decoder``, ``@tesseract_decoder``).
    ///   The ``@`` prefix is reserved and never opens a real file.
    pub file: String,
    /// the name of the decoder class inside the Python file; defaults to "Decoder"
    #[serde(default = "default_decoder_class_name")]
    pub name: String,
    /// Python decoder parameters
    #[structdoc(skip)]
    pub py_config: Option<serde_json::Value>,
}

fn default_decoder_class_name() -> String {
    "Decoder".to_string()
}

#[pyclass(name = "DecodingHypergraph")]
pub struct PyDecodingHypergraph {
    #[pyo3(get, set)]
    pub vertex_num: u64,
    #[pyo3(get, set)]
    pub hyperedges: Py<PyList>, // PyHyperedge
}

#[pymethods]
impl PyDecodingHypergraph {
    fn __repr__(&self) -> PyResult<String> {
        Python::attach(|py| {
            let hyperedges = self.hyperedges.bind(py);
            Ok(format!(
                "DecodingHypergraph(vertex_num={}, hyperedges=[...{}...])",
                self.vertex_num,
                hyperedges.len()
            ))
        })
    }
}

impl PyDecodingHypergraph {
    pub fn new(py: Python, hypergraph: &DecodingHypergraph) -> PyResult<Self> {
        let py_hyperedges = PyList::empty(py);
        for e in &hypergraph.hyperedges {
            let py_e = PyHyperedge {
                vertices: e.vertices.clone(),
                probability: e.probability,
            };
            py_hyperedges.append(py_e)?;
        }
        Ok(Self {
            vertex_num: hypergraph.vertex_num,
            hyperedges: py_hyperedges.unbind(),
        })
    }
}

#[pyclass(name = "Hyperedge")]
#[derive(Debug)]
pub struct PyHyperedge {
    #[pyo3(get, set)]
    pub vertices: Vec<u64>,
    #[pyo3(get, set)]
    pub probability: f64,
}

#[pymethods]
impl PyHyperedge {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }
}

pub struct PythonDecoderInstance {
    decoder: Py<PyAny>,
}

impl DecoderInstance for PythonDecoderInstance {
    fn new(hypergraph: &DecodingHypergraph, config: &serde_json::Value) -> Self {
        let config: PythonDecoderConfig = serde_json::from_value(config.clone()).unwrap();
        let decoder = Python::attach(|py| {
            let module = if let Some(builtin_name) = config.file.strip_prefix('@') {
                let (fname, source) = builtin_decoders::lookup(builtin_name).ok_or_else(|| {
                    let known = builtin_decoders::names()
                        .iter()
                        .map(|n| format!("@{n}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    PyValueError::new_err(format!("unknown builtin decoder '@{builtin_name}'.  Known builtins: {known}"))
                })?;
                get_or_load_module_from_source(py, fname, source)?
            } else {
                get_or_load_module(py, &config.file)?
            };
            let py_hypergraph = PyDecodingHypergraph::new(py, hypergraph)?;
            let py_config = json_value_to_py(py, &config.py_config.unwrap_or_else(|| serde_json::json!({})))?;
            let decoder_class = module.getattr(config.name.as_str())?;
            let decoder = decoder_class.call1((py_hypergraph, py_config))?;
            Ok::<Py<PyAny>, PyErr>(decoder.unbind())
        })
        .unwrap();
        Self { decoder }
    }

    fn decode(&mut self, syndrome: &BitVector) -> ParityFactor {
        let subgraph = Python::attach(|py| {
            let decoder = self.decoder.bind(py);
            let py_syndrome = PyList::empty(py);
            for index in to_sparse_indices(syndrome) {
                py_syndrome.append(index)?;
            }
            let py_result = decoder.call_method1("decode", (py_syndrome,))?;
            py_result.extract::<Vec<u64>>()
        })
        .unwrap();
        ParityFactor { subgraph }
    }

    fn reset(&mut self) {
        Python::attach(|py| {
            let decoder = self.decoder.bind(py);
            decoder.call_method0("reset")?;
            Ok::<(), PyErr>(())
        })
        .unwrap();
    }
}

pub type PythonDecoder = ThreadPoolingDecoder<PythonDecoderInstance>;
