use binar::vec::AlignedBitVec;
use binar::BitwisePairMut;
use binar::{Bitwise, BitwisePair};
use derive_more::{Deref, DerefMut, From, Into};
use paulimer::pauli::generic::pauli_string;
use paulimer::pauli::{commutes_with, dense_from, DensePauli, Pauli};
use paulimer::traits::NeutralElement;
use pyo3::exceptions::PyValueError;
use pyo3::{
    exceptions::PyNotImplementedError,
    prelude::*,
    pyclass::CompareOp,
    types::{PyComplex, PyIterator},
    PyResult,
};

use crate::py_sparse_pauli::PySparsePauli;

#[derive(Clone, Deref, DerefMut, From, Into)]
#[must_use]
#[pyclass(name = "DensePauli", module = "paulimer")]
pub struct PyDensePauli {
    #[deref]
    #[deref_mut]
    pub(crate) inner: DensePauli,
    pub(crate) size: usize,
}

impl<'life> From<&'life PyDensePauli> for &'life DensePauli {
    fn from(py_dense_pauli: &'life PyDensePauli) -> Self {
        &py_dense_pauli.inner
    }
}

#[pymethods]
impl PyDensePauli {
    #[new]
    #[pyo3(signature = (characters=""))]
    /// # Errors
    /// Will return an error if the string is not a valid Pauli representation
    fn new(characters: &str) -> PyResult<Self> {
        match characters.parse() {
            Ok(pauli) => Ok(Self {
                inner: pauli,
                size: characters.len(),
            }),
            Err(_) => Err(pyo3::exceptions::PyValueError::new_err("Invalid character string.")),
        }
    }

    #[staticmethod]
    fn identity(size: usize) -> Self {
        PyDensePauli {
            inner: DensePauli::neutral_element_of_size(size),
            size,
        }
    }

    #[staticmethod]
    fn from_sparse(pauli: &PySparsePauli, qubit_count: usize) -> Self {
        PyDensePauli {
            inner: dense_from(&pauli.inner, qubit_count),
            size: qubit_count,
        }
    }

    #[staticmethod]
    fn x(qubit_id: usize, qubit_count: usize) -> Self {
        PyDensePauli {
            inner: DensePauli::x(qubit_id, qubit_count),
            size: qubit_count,
        }
    }

    #[staticmethod]
    fn y(qubit_id: usize, qubit_count: usize) -> Self {
        PyDensePauli {
            inner: DensePauli::y(qubit_id, qubit_count),
            size: qubit_count,
        }
    }

    #[staticmethod]
    fn z(qubit_id: usize, qubit_count: usize) -> Self {
        PyDensePauli {
            inner: DensePauli::z(qubit_id, qubit_count),
            size: qubit_count,
        }
    }

    #[getter]
    #[must_use]
    pub fn exponent(&self) -> u8 {
        self.inner.xz_phase_exponent()
    }

    #[getter]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn phase(&self, py: Python<'_>) -> Py<PyComplex> {
        let y_count = self.inner.x_bits().and_weight(self.inner.z_bits());
        // Safety: y_count % 4 is always 0-3, which fits in u8
        let exponent = (self.inner.xz_phase_exponent() + 4u8 - (y_count % 4) as u8) % 4;
        match exponent {
            0 => PyComplex::from_doubles(py, 1.0, 0.0).into(),
            1 => PyComplex::from_doubles(py, 0.0, 1.0).into(),
            2 => PyComplex::from_doubles(py, -1.0, 0.0).into(),
            3 => PyComplex::from_doubles(py, 0.0, -1.0).into(),
            _ => unreachable!(),
        }
    }

    #[getter]
    #[must_use]
    pub fn characters(&self) -> String {
        pauli_string(&self.inner, 0, false, false, true, Some(self.size))
            .clone()
            .trim_start_matches(['+', '-', 'i', ' ', '𝑖'])
            .to_string()
    }

    #[getter]
    #[must_use]
    pub fn support(&self) -> Vec<usize> {
        self.inner.support().collect()
    }

    #[getter]
    #[must_use]
    pub fn weight(&self) -> usize {
        self.inner.weight()
    }

    #[getter]
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// # Errors
    /// Will return an error if the extraction of DensePaulis(s) fails.
    pub fn commutes_with(&self, others: &Bound<'_, PyAny>) -> PyResult<bool> {
        // Try to extract as a single DensePauli first
        if let Ok(other) = others.extract::<PyDensePauli>() {
            return Ok(commutes_with(&self.inner, &other.inner));
        }

        let iter = PyIterator::from_object(others)?;
        for item in iter {
            let item = item?;
            let pauli: PyDensePauli = item.extract()?;
            if !commutes_with(&self.inner, &pauli.inner) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// # Errors
    ///
    /// Will return error for >,>=,<,<=
    pub fn __richcmp__(&self, other: &PyDensePauli, comparison: CompareOp) -> PyResult<bool> {
        match comparison {
            CompareOp::Eq => Ok(self.inner == other.inner),
            CompareOp::Ne => Ok(self.inner != other.inner),
            _ => Err(PyNotImplementedError::new_err("")),
        }
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn __mul__(&self, other: &PyDensePauli) -> PyResult<PyDensePauli> {
        if self.size != other.size {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot multiply DensePaulis of different sizes.",
            ));
        }
        Ok(PyDensePauli {
            inner: self.inner.clone() * &other.inner,
            size: self.size,
        })
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn __imul__(&mut self, other: &PyDensePauli) -> PyResult<()> {
        if self.size != other.size {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot multiply DensePaulis of different sizes.",
            ));
        }
        self.inner *= &other.inner;
        Ok(())
    }

    pub fn __add__(&self, other: &PyDensePauli) -> PyDensePauli {
        let mut x_bits = AlignedBitVec::zeros(self.size + other.size);
        let mut z_bits = AlignedBitVec::zeros(self.size + other.size);
        x_bits.assign(self.inner.x_bits());
        z_bits.assign(self.inner.z_bits());
        x_bits.assign_with_offset(other.inner.x_bits(), self.size, other.size);
        z_bits.assign_with_offset(other.inner.z_bits(), self.size, other.size);
        PyDensePauli {
            inner: DensePauli::from_bits(x_bits, z_bits, 0),
            size: self.size + other.size,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn __abs__(&self) -> PyDensePauli {
        let y_count = self.inner.x_bits().and_weight(self.inner.z_bits());
        // Safety: y_count % 4 is always 0-3, which fits in u8
        let exponent = (y_count % 4) as u8;
        PyDensePauli {
            inner: DensePauli::from_bits(self.inner.x_bits().clone(), self.inner.z_bits().clone(), exponent),
            size: self.size,
        }
    }

    pub fn __neg__(&mut self) -> PyDensePauli {
        PyDensePauli {
            inner: -self.inner.clone(),
            size: self.size,
        }
    }

    /// # Errors
    /// Will return an error if the index is out of bounds.
    pub fn __getitem__(&self, index: usize) -> PyResult<&str> {
        if index >= self.size {
            return Err(PyValueError::new_err("Index out of bounds."));
        }
        let is_x = self.inner.x_bits().index(index);
        let is_z = self.inner.z_bits().index(index);
        match (is_x, is_z) {
            (false, false) => Ok("I"),
            (true, false) => Ok("X"),
            (true, true) => Ok("Y"),
            (false, true) => Ok("Z"),
        }
    }

    #[must_use]
    pub fn __str__(&self) -> String {
        self.inner.to_string()
    }

    #[must_use]
    pub fn __repr__(&self) -> String {
        self.inner.to_string()
    }

    /// Returns state for pickle serialization.
    #[must_use]
    pub fn __getstate__(&self) -> (Vec<u64>, Vec<u64>, u8, usize) {
        let x_words = self.inner.x_bits().as_words().to_vec();
        let z_words = self.inner.z_bits().as_words().to_vec();
        (x_words, z_words, self.inner.xz_phase_exponent(), self.size)
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn __setstate__(&mut self, state: (Vec<u64>, Vec<u64>, u8, usize)) {
        let x_bits = AlignedBitVec::from_words(&state.0);
        let z_bits = AlignedBitVec::from_words(&state.1);
        self.inner = DensePauli::from_bits(x_bits, z_bits, state.2);
        self.size = state.3;
    }
}
