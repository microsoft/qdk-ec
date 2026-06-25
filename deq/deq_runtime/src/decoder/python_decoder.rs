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
use crate::util::BitVector;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::{Deserialize, Serialize};
#[cfg(feature = "cli")]
use structdoc::StructDoc;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct PythonDecoderConfig {
    /// we want to recognize all the thread pooling config fields
    #[serde(flatten)]
    pub thread_pooling_config: ThreadPoolingConfig,
    /// the entry file of the Python decoder (should be a file *.py)
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
            let module = get_or_load_module(py, &config.file)?;
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

fn get_or_load_module<'py>(py: Python<'py>, file: &str) -> PyResult<Bound<'py, PyAny>> {
    // In principle, you should not need to modify sys.path; if you encounter some problem
    // with module loading, try `export LD_LIBRARY_PATH="$HOME/miniforge3/lib/:$LD_LIBRARY_PATH"`
    //
    // Use the file path as part of the module name so different Python decoder files
    // loaded by the same process get distinct modules (otherwise the first-loaded file
    // would be returned for every subsequent file).
    let module_name = format!(
        "deq_python_decoder_{}",
        file.replace(|c: char| !c.is_ascii_alphanumeric(), "_")
    );
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    if let Ok(existing) = modules.get_item(&module_name) {
        return Ok(existing);
    }
    let util = py.import("importlib.util")?;
    let spec = util.call_method1("spec_from_file_location", (&module_name, file))?;
    let module = util.call_method1("module_from_spec", (spec.clone(),))?;
    // Note: register in sys.modules AFTER exec_module finishes. Heavy imports
    // (e.g. numpy / scipy) inside the loaded module release the GIL, which
    // would otherwise let another rayon worker thread observe a half-loaded
    // module and call `getattr` for the decoder class before it is defined.
    spec.getattr("loader")?.call_method1("exec_module", (module.clone(),))?;
    modules.set_item(&module_name, module.clone())?;
    Ok(module)
}

/// Convert a [`serde_json::Value`] directly into a Python object.
///
/// We don't use the `pythonize` crate here because this crate enables
/// `serde_json/arbitrary_precision`, under which `serde_json::Number` is
/// serialized as the sentinel map `{"$serde_json::private::Number": "10"}`
/// instead of a plain integer. That sentinel breaks any downstream Python
/// callee that expects a real `int` or `float` (e.g. pybind11-bound C++
/// constructors with strict type checks). Walking the `Value` tree manually
/// lets us emit native Python scalars.
fn json_value_to_py<'py>(py: Python<'py>, value: &serde_json::Value) -> PyResult<Bound<'py, PyAny>> {
    match value {
        serde_json::Value::Null => Ok(py.None().into_bound(py)),
        serde_json::Value::Bool(b) => Ok(pyo3::types::PyBool::new(py, *b).to_owned().into_any()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any())
            } else if let Some(u) = n.as_u64() {
                Ok(u.into_pyobject(py)?.into_any())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any())
            } else {
                // Arbitrary-precision number that doesn't fit any of the above:
                // fall back to a Python string so the callee can convert.
                Ok(n.to_string().into_pyobject(py)?.into_any())
            }
        }
        serde_json::Value::String(s) => Ok(s.into_pyobject(py)?.into_any()),
        serde_json::Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_value_to_py(py, item)?)?;
            }
            Ok(list.into_any())
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            for (key, val) in map {
                dict.set_item(key, json_value_to_py(py, val)?)?;
            }
            Ok(dict.into_any())
        }
    }
}

pub type PythonDecoder = ThreadPoolingDecoder<PythonDecoderInstance>;
