//! Python bindings for `deqagram`, the `.deq` parser.
//!
//! `parse` returns deqagram's pure-Python dataclasses, built directly from its
//! Rust AST by the `convert` module.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod convert;

/// Parses a `.deq` source into a `deqagram.DeqFile`.
///
/// Raises `ValueError` on a parse error.
#[pyfunction]
fn parse<'py>(py: Python<'py>, source: &str) -> PyResult<Bound<'py, PyAny>> {
    let file: ::deqagram::ast::DeqFile = source.parse().map_err(|e| PyValueError::new_err(format!("{e}")))?;
    convert::deq_file(py, file, source)
}

/// The `deqagram` extension module (imported as `deqagram.deqagram`).
#[pymodule]
fn deqagram(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse, m)?)?;
    m.add("__all__", ["parse"])?;
    Ok(())
}
