use pyo3::{exceptions::PyIndexError, prelude::*};

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
        _ => Err(PyIndexError::new_err(format!(
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
