use crate::{PyBitMatrix, PyBitVec};
use binar::matrix::EchelonForm;
use pyo3::prelude::*;

#[pyclass(name = "EchelonForm", module = "binar")]
#[derive(Clone)]
pub struct PyEchelonForm {
    inner: EchelonForm,
}

#[pymethods]
impl PyEchelonForm {
    #[new]
    fn new(matrix: PyBitMatrix) -> Self {
        Self {
            inner: EchelonForm::new(matrix.into()),
        }
    }

    /// The reduced row echelon form matrix.
    #[getter]
    fn matrix(&self) -> PyBitMatrix {
        self.inner.matrix().into()
    }

    /// The transformation matrix T such that T * original = RREF.
    #[getter]
    fn transform(&self) -> PyBitMatrix {
        self.inner.transform().into()
    }

    /// The inverse transpose of the transformation matrix.
    #[getter]
    fn transform_inv_t(&self) -> PyBitMatrix {
        self.inner.transform_inv_t().into()
    }

    /// The pivot column indices (rank profile).
    #[getter]
    fn pivots(&self) -> Vec<usize> {
        self.inner.pivots().to_vec()
    }

    /// Solve the linear system Ax = b.
    fn solve(&self, b: &PyBitVec) -> Option<PyBitVec> {
        self.inner.solve(&b.as_view()).map(Into::into)
    }

    /// Solve the linear system A^T x = b.
    fn transpose_solve(&self, b: &PyBitVec) -> Option<PyBitVec> {
        self.inner.transpose_solve(&b.as_view()).map(Into::into)
    }
}
