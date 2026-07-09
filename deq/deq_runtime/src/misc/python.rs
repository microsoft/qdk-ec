//! Shared Python interop helpers used by Rust-side plugins that load
//! user-supplied Python files (e.g. the Python decoder, the Python
//! sampler).
//!
//! Both helpers were originally private to `decoder::python_decoder`;
//! they moved here when a second consumer (the Python sampler) needed
//! them.  Behaviour is unchanged.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Load (or re-fetch) a Python module from a filesystem path.
///
/// The module is keyed in ``sys.modules`` by a name derived from the
/// file path so different files loaded by the same process get distinct
/// modules (otherwise the first-loaded file would be returned for every
/// subsequent file).
///
/// In principle, you should not need to modify ``sys.path``; if you
/// encounter some problem with module loading, try
/// ``export LD_LIBRARY_PATH="$HOME/miniforge3/lib/:$LD_LIBRARY_PATH"``
/// (only needed for standalone Rust binaries — the Python extension
/// module path does not need it).
pub(crate) fn get_or_load_module<'py>(py: Python<'py>, file: &str) -> PyResult<Bound<'py, PyAny>> {
    let module_name = format!(
        "deq_python_module_{}",
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
    // module and call `getattr` for the user-defined class before it is defined.
    spec.getattr("loader")?.call_method1("exec_module", (module.clone(),))?;
    modules.set_item(&module_name, module.clone())?;
    Ok(module)
}

/// Load (or re-fetch) a Python module from an in-memory source string.
///
/// Companion to [`get_or_load_module`] for callers that carry their own
/// source (e.g. the compile-time-embedded builtin samplers registry).
/// ``virtual_filename`` is used as ``__file__`` on the module and as the
/// filename shown in Python tracebacks, so pick something descriptive
/// like ``"@qdk_sampler"`` rather than ``"<string>"``.
///
/// The module is registered in ``sys.modules`` under a deterministic
/// key derived from ``virtual_filename`` so subsequent lookups reuse
/// the already-loaded module.
pub(crate) fn get_or_load_module_from_source<'py>(
    py: Python<'py>,
    virtual_filename: &str,
    source: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let module_name = format!(
        "deq_python_module_{}",
        virtual_filename.replace(|c: char| !c.is_ascii_alphanumeric(), "_")
    );
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    if let Ok(existing) = modules.get_item(&module_name) {
        return Ok(existing);
    }
    let types = py.import("types")?;
    let module = types.getattr("ModuleType")?.call1((&module_name,))?;
    module.setattr("__file__", virtual_filename)?;
    let builtins = py.import("builtins")?;
    let code = builtins.getattr("compile")?.call1((source, virtual_filename, "exec"))?;
    let dict = module.getattr("__dict__")?;
    // Same ordering as `get_or_load_module`: exec the module body
    // BEFORE registering in sys.modules so half-loaded modules can't
    // leak to other worker threads mid-import.
    builtins.getattr("exec")?.call1((code, dict))?;
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
pub(crate) fn json_value_to_py<'py>(py: Python<'py>, value: &serde_json::Value) -> PyResult<Bound<'py, PyAny>> {
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
