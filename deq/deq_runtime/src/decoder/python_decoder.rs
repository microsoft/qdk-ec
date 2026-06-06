//! Python decoder
//!
//! Calling another decoder written in Python language with the following APIs:
//!
//! fn new(hypergraph: &DecodingHypergraph, config: Dict | None) -> Decoder
//! fn decode(self: Decoder, syndrome: list[int]) -> list[int]
//! fn reset(self: Decoder) -> None
//!

use crate::decoder::blackbox_decoder::{DecodingHypergraph, ParityFactor};
use crate::decoder::thread_pooling::{DecoderInstance, ThreadPoolingConfig, ThreadPoolingDecoder};
use crate::misc::bit_vector::to_sparse_indices;
use crate::util::BitVector;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pythonize::pythonize;
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
    /// Python decoder parameters
    #[structdoc(skip)]
    pub py_config: Option<serde_json::Value>,
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
            let py_config = pythonize(py, &config.py_config.unwrap_or_else(|| serde_json::json!({}))).unwrap();
            let decoder = module.getattr("new")?.call1((py_hypergraph, py_config))?;
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
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    if modules.get_item("deq_python_decoder").is_ok() {
        return modules.get_item("deq_python_decoder");
    }
    let util = py.import("importlib.util")?;
    let spec = util.call_method1("spec_from_file_location", ("deq_python_decoder", file))?;
    let module = util.call_method1("module_from_spec", (spec.clone(),))?;
    sys.getattr("modules")?.set_item("deq_python_decoder", module.clone())?;
    spec.getattr("loader")?.call_method1("exec_module", (module.clone(),))?;
    Ok(module)
}

pub type PythonDecoder = ThreadPoolingDecoder<PythonDecoderInstance>;
