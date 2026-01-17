use crate::{PyBitMatrix, PyBitVec};
use binar::{BitMatrix, matrix::bitmatrix_row_stacked};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyIterator};

#[pyfunction]
#[pyo3(name = "vstack", signature=(matrices))]
/// # Errors
/// Will return an error if the extraction of `BitMatrix` objects fails.
pub fn py_row_stacked(matrices: &Bound<'_, PyAny>) -> PyResult<PyBitMatrix> {
    let iter = PyIterator::from_object(matrices)?;
    let mut pymatrices = Vec::new();
    for item in iter {
        let item = item?;
        let bitmatrix: PyBitMatrix = item.extract()?;
        pymatrices.push(bitmatrix);
    }
    let bitmatrices = pymatrices.iter().map(std::convert::Into::into);
    let stack = bitmatrix_row_stacked(bitmatrices);
    Ok(stack.into())
}

#[pyfunction]
#[pyo3(name = "rank")]
pub fn py_rank(matrix: &PyBitMatrix) -> usize {
    BitMatrix::rank(matrix)
}

#[pyfunction]
#[pyo3(name = "null_space")]
pub fn py_null_space(matrix: &PyBitMatrix) -> PyBitMatrix {
    BitMatrix::kernel(matrix).into()
}

#[pyfunction]
#[pyo3(name = "inv")]
pub fn py_inverse(matrix: &PyBitMatrix) -> PyResult<PyBitMatrix> {
    if matrix.rowcount() != matrix.columncount() {
        return Err(PyValueError::new_err("Matrix must be square to compute the inverse."));
    }
    Ok(BitMatrix::inverted(matrix).into())
}

#[pyfunction]
#[pyo3(name = "det")]
pub fn py_determinant(matrix: &PyBitMatrix) -> PyResult<bool> {
    if matrix.rowcount() != matrix.columncount() {
        return Err(PyValueError::new_err(
            "Matrix must be square to compute the determinant.",
        ));
    }
    Ok(matrix.rank() == matrix.rowcount())
}

#[pyfunction]
#[pyo3(name = "solve")]
pub fn py_solve(matrix: &PyBitMatrix, b: &PyBitVec) -> Option<PyBitVec> {
    use binar::matrix::EchelonForm;
    let form = EchelonForm::new(matrix.clone().into());
    form.solve(&b.as_view()).map(std::convert::Into::into)
}
