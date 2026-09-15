// PyO3 hands extracted arguments over as owned values; taking them by reference
// would mean cloning at every call site instead.
#![allow(clippy::needless_pass_by_value)]

use pyo3::prelude::*;

mod codes;
mod container;
mod display;
mod gadgets;
mod nodes;
mod parsers;
mod types;

use codes::PyCode;
use container::{PyLayer, PyQodec};
use gadgets::{PyCircuit, PyEncoding, PyFlag, PyGadget, PyOutcome, PyReadout, PyReference};
use types::{
    PyBlock, PyBlockOperand, PyClifford, PyCondition, PyInstruction, PyInstructionCall, PyInstructionSet, PyObserve,
    PyParameter, PyPauli, PyRotate, PyStabilize,
};

/// Convert a core [`qodec::Metadata`] mapping into a Python `dict`.
pub(crate) fn metadata_to_py<'py>(py: Python<'py>, metadata: &qodec::Metadata) -> PyResult<Bound<'py, PyAny>> {
    pythonize::pythonize(py, metadata).map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

/// A Pauli operator accepted from Python: a `str`, or a
/// `qodec.codes.PauliExpression`.
///
/// `PauliExpression` is deliberately not a `str` subclass — it is a validated
/// value with a small surface, the same shape as `Reference` — so it needs an
/// explicit conversion here rather than riding on `String` extraction. Nothing
/// else is accepted: `str(obj)` on an arbitrary object would turn `42` into a
/// Pauli named `"42"` and defer the failure to load time.
pub(crate) fn pauli_text(object: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(text) = object.extract::<String>() {
        return Ok(text);
    }
    let class = object.py().import("qodec.codes")?.getattr("PauliExpression")?;
    if object.is_instance(&class)? {
        return object.str()?.extract();
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "expected a Pauli string or PauliExpression, got {}",
        object.get_type().name()?
    )))
}

/// [`pauli_text`] over a list.
pub(crate) fn pauli_strings(objects: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<qodec::PauliString>> {
    objects
        .iter()
        .map(|object| pauli_text(object).map(qodec::PauliString))
        .collect()
}

/// Convert an optional Python value into a core [`qodec::Metadata`] mapping.
///
/// `None` (the unset constructor default) yields an empty mapping. A value
/// that is not a mapping is rejected, which enforces the object-only rule at
/// the boundary with no separate type check.
pub(crate) fn metadata_from_py(value: Option<&Bound<'_, PyAny>>) -> PyResult<qodec::Metadata> {
    match value {
        None => Ok(qodec::Metadata::default()),
        Some(value) => {
            pythonize::depythonize(value).map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
        }
    }
}

pyo3::create_exception!(
    _native,
    QodecError,
    pyo3::exceptions::PyException,
    "Base class for every qodec domain error."
);
pyo3::create_exception!(
    _native,
    QodecLoadError,
    QodecError,
    "Raised when reading, parsing, or validating a qodec or artifact fails."
);
pyo3::create_exception!(
    _native,
    QodecSaveError,
    QodecError,
    "Raised when preparing, serializing, or writing a qodec or artifact fails."
);

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("QodecError", py.get_type::<QodecError>())?;
    module.add("QodecLoadError", py.get_type::<QodecLoadError>())?;
    module.add("QodecSaveError", py.get_type::<QodecSaveError>())?;
    module.add_class::<PyQodec>()?;
    module.add_class::<PyLayer>()?;
    module.add_class::<PyInstructionSet>()?;
    module.add_class::<PyBlock>()?;
    module.add_class::<PyInstruction>()?;
    module.add_class::<PyBlockOperand>()?;
    module.add_class::<PyParameter>()?;
    module.add_class::<PyCondition>()?;
    module.add_class::<PyStabilize>()?;
    module.add_class::<PyClifford>()?;
    module.add_class::<PyPauli>()?;
    module.add_class::<PyObserve>()?;
    module.add_class::<PyRotate>()?;
    module.add_class::<PyCode>()?;
    module.add_class::<PyGadget>()?;
    module.add_class::<PyCircuit>()?;
    module.add_class::<PyOutcome>()?;
    module.add_class::<PyFlag>()?;
    module.add_class::<PyReadout>()?;
    module.add_class::<PyReference>()?;
    module.add_class::<PyEncoding>()?;
    module.add_class::<PyInstructionCall>()?;
    module.add_class::<nodes::PyModelPath>()?;
    module.add_function(wrap_pyfunction!(nodes::_node_query, module)?)?;
    module.add_function(wrap_pyfunction!(parsers::register, module)?)?;
    #[cfg(feature = "test-support")]
    {
        module.add_function(wrap_pyfunction!(container::_test_node_snapshots, module)?)?;
        module.add_function(wrap_pyfunction!(parsers::test_support::_test_native_calls, module)?)?;
        module.add_function(wrap_pyfunction!(
            parsers::test_support::_test_register_native_empty,
            module
        )?)?;
    }
    Ok(())
}
