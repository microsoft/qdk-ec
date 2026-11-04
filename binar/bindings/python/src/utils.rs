use pyo3::{
    exceptions::{PyIndexError, PyTypeError, PyValueError},
    prelude::*,
};

pub fn py_value_err(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

pub fn py_type_err(msg: impl Into<String>) -> PyErr {
    PyTypeError::new_err(msg.into())
}

pub fn py_index_err(msg: impl Into<String>) -> PyErr {
    PyIndexError::new_err(msg.into())
}

/// Convert a Python-style index (negative values count from the end) into an
/// in-bounds `usize` index.
///
/// # Errors
///
/// Returns an `IndexError` if the index is out of range for `len`.
pub fn try_py_index(index: isize, len: usize) -> PyResult<usize> {
    let resolved = if index < 0 {
        len.checked_sub(index.unsigned_abs())
    } else {
        Some(index.unsigned_abs())
    };
    match resolved {
        Some(resolved) if resolved < len => Ok(resolved),
        _ => Err(py_index_err(format!(
            "Index {index} is out-of-bounds for extent {len}."
        ))),
    }
}

/// Apply [`try_py_index`] to each index.
///
/// # Errors
///
/// Returns an `IndexError` if any index is out of range for `len`.
pub fn try_py_indices(indices: Vec<isize>, len: usize) -> PyResult<Vec<usize>> {
    indices.into_iter().map(|i| try_py_index(i, len)).collect()
}
