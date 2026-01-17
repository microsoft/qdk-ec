use pyo3::prelude::*;

mod linalg;
mod py_bitmatrix;
mod py_bitvec;

pub use py_bitmatrix::PyBitMatrix;
pub use py_bitvec::PyBitVec;

/// # Errors
///
/// Returns an error if any class or function registration fails.
#[pymodule]
pub fn binar(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBitMatrix>()?;
    m.add_class::<PyBitVec>()?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::py_row_stacked, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::py_rank, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::py_null_space, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::py_inverse, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::py_determinant, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(linalg::py_solve, m)?)?;
    Ok(())
}
