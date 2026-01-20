use binar::{Bitwise, BitwisePair, IndexSet};
use derive_more::{Deref, DerefMut, From};
use paulimer::pauli::{as_sparse, commutes_with, Pauli, PauliMutable, SparsePauli};
use paulimer::traits::NeutralElement;
use pyo3::{
    exceptions::{PyNotImplementedError, PyValueError},
    prelude::*,
    pyclass::CompareOp,
    types::{PyComplex, PyIterator},
    PyResult,
};
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use crate::py_dense_pauli::PyDensePauli;

#[must_use]
#[pyclass(name = "SparsePauli", module = "paulimer")]
#[repr(transparent)]
#[derive(Clone, Debug, Deref, DerefMut, From)]
pub struct PySparsePauli {
    #[deref]
    #[deref_mut]
    pub(crate) inner: SparsePauli,
}

impl<'life> From<&'life PySparsePauli> for &'life SparsePauli {
    fn from(py_sparse_pauli: &'life PySparsePauli) -> Self {
        &py_sparse_pauli.inner
    }
}

#[pymethods]
impl PySparsePauli {
    #[new]
    #[pyo3(signature = (characters=None, exponent=0))]
    /// # Errors
    /// Will return an error if the character map does not represent a valid Pauli operator
    pub fn new(characters: Option<&Bound<'_, PyAny>>, exponent: u8) -> PyResult<Self> {
        // If no characters provided, return identity
        let Some(chars) = characters else {
            return Ok(Self {
                inner: SparsePauli::neutral_element_of_size(0),
            });
        };

        // Try to extract as a string first
        if let Ok(s) = chars.extract::<String>() {
            return match s.parse::<SparsePauli>() {
                Ok(mut pauli) => {
                    pauli.add_assign_phase_exp(exponent);
                    Ok(Self { inner: pauli })
                }
                Err(_) => Err(PyValueError::new_err("Invalid character string.")),
            };
        }

        // Try to extract as a dict
        if let Ok(dict) = chars.extract::<HashMap<usize, char>>() {
            return match SparsePauli::try_from(dict) {
                Ok(mut pauli) => {
                    pauli.add_assign_phase_exp(exponent);
                    Ok(Self { inner: pauli })
                }
                Err(_) => Err(PyValueError::new_err("Invalid character map.")),
            };
        }

        Err(PyValueError::new_err("Expected string or dict for characters."))
    }

    #[staticmethod]
    pub fn identity() -> Self {
        SparsePauli::neutral_element_of_size(0).into()
    }

    #[staticmethod]
    /// # Errors
    /// Will return an error if the string is not a valid Pauli representation
    pub fn from_string(characters: &str) -> PyResult<Self> {
        match characters.parse() {
            Ok(pauli) => Ok(Self { inner: pauli }),
            Err(_) => Err(PyValueError::new_err("Invalid character string.")),
        }
    }

    #[staticmethod]
    fn from_dense(dense_pauli: &PyDensePauli) -> Self {
        Self {
            inner: as_sparse(&dense_pauli.inner),
        }
    }

    #[staticmethod]
    fn x(qubit_id: usize) -> Self {
        SparsePauli::x(qubit_id, 0).into()
    }

    #[staticmethod]
    fn y(qubit_id: usize) -> Self {
        SparsePauli::y(qubit_id, 0).into()
    }

    #[staticmethod]
    fn z(qubit_id: usize) -> Self {
        SparsePauli::z(qubit_id, 0).into()
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
        let support: Vec<usize> = self.inner.support().collect();
        support
            .iter()
            .map(|&index| {
                let is_x = self.inner.x_bits().index(index);
                let is_z = self.inner.z_bits().index(index);
                match (is_x, is_z) {
                    (true, false) => 'X',
                    (true, true) => 'Y',
                    (false, true) => 'Z',
                    (false, false) => unreachable!(),
                }
            })
            .collect()
    }

    #[getter]
    #[must_use]
    pub fn support(&self) -> Vec<usize> {
        self.inner.support().collect()
    }

    #[getter]
    #[must_use]
    pub fn weight(&self) -> usize {
        self.inner.support().count()
    }

    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// # Errors
    /// Will return an error if the extraction of SparsePauli(s) fails.
    pub fn commutes_with(&self, others: &Bound<'_, PyAny>) -> PyResult<bool> {
        // Try to extract as a single SparsePauli first
        if let Ok(other) = others.extract::<PySparsePauli>() {
            return Ok(commutes_with(&self.inner, &other.inner));
        }

        // Try to extract as an iterable of SparsePauli
        let iter = PyIterator::from_object(others)?;
        for item in iter {
            let item = item?;
            let pauli: PySparsePauli = item.extract()?;
            if !commutes_with(&self.inner, &pauli.inner) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// # Errors
    ///
    /// Will return for >, <, >=, <= comparisons
    pub fn __richcmp__(&self, other: &Self, comparison: CompareOp) -> PyResult<bool> {
        match comparison {
            CompareOp::Eq => Ok(self.inner == other.inner),
            CompareOp::Ne => Ok(self.inner != other.inner),
            _ => Err(PyNotImplementedError::new_err("")),
        }
    }

    pub fn __mul__(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.clone() * &other.inner,
        }
    }

    pub fn __imul__(&mut self, other: &Self) {
        self.inner *= &other.inner;
    }

    pub fn __neg__(&mut self) -> Self {
        Self {
            inner: -self.inner.clone(),
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn __abs__(&self) -> Self {
        let y_count = self.inner.x_bits().and_weight(self.inner.z_bits());
        // Safety: y_count % 4 is always 0-3, which fits in u8
        let exponent = (y_count % 4) as u8;
        Self {
            inner: SparsePauli::from_bits(self.inner.x_bits().clone(), self.inner.z_bits().clone(), exponent),
        }
    }

    #[must_use]
    pub fn __getitem__(&self, index: usize) -> &str {
        let is_x = self.inner.x_bits().index(index);
        let is_z = self.inner.z_bits().index(index);
        match (is_x, is_z) {
            (false, false) => "I",
            (true, false) => "X",
            (true, true) => "Y",
            (false, true) => "Z",
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

    #[must_use]
    pub fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.to_string().hash(&mut hasher);
        hasher.finish()
    }

    /// # Errors
    ///
    /// Will not return
    pub fn __getstate__(&self) -> PyResult<(Vec<usize>, Vec<usize>, u8)> {
        let x_bits: Vec<usize> = self.inner.x_bits().clone().into_iter().collect();
        let z_bits: Vec<usize> = self.inner.z_bits().clone().into_iter().collect();
        Ok((x_bits, z_bits, self.inner.xz_phase_exponent()))
    }

    /// # Errors
    ///
    /// Will not return
    pub fn __setstate__(&mut self, state: (Vec<usize>, Vec<usize>, u8)) -> PyResult<()> {
        let (x, z, exponent) = state;
        let x_bits = IndexSet::from_iter(x);
        let z_bits = IndexSet::from_iter(z);
        self.inner = SparsePauli::from_bits(x_bits, z_bits, exponent);
        Ok(())
    }
}
