use derive_more::{Deref, DerefMut, From};
use paulimer::pauli::SparsePauli;
use paulimer::pauli_group::{centralizer_of, centralizer_within, symplectic_form_of, PauliGroup};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator};

use crate::py_sparse_pauli::PySparsePauli;

#[pyclass]
#[pyo3(name = "PauliGroupIterator", module = "paulimer")]
pub struct PyPauliGroupIterator {
    // Store a concrete iterator that owns its data
    iter: Box<dyn Iterator<Item = SparsePauli> + Send + Sync>,
}

#[pymethods]
impl PyPauliGroupIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PySparsePauli> {
        slf.iter.next().map(|pauli| PySparsePauli { inner: pauli })
    }
}

#[pyclass]
#[pyo3(name = "PauliGroup", module = "paulimer")]
#[derive(Clone, Deref, DerefMut, From)]
pub struct PyPauliGroup {
    #[deref]
    #[deref_mut]
    inner: PauliGroup,
    is_abelian_promise: Option<bool>,
}

fn extract_sparse_pauli_from_iterable(iterable: &Bound<'_, PyAny>) -> PyResult<Vec<SparsePauli>> {
    let mut result = Vec::new();
    let iter = PyIterator::from_object(iterable)?;
    for item in iter {
        let item = item?;
        let pauli: PySparsePauli = item.extract()?;
        result.push(pauli.inner);
    }
    Ok(result)
}

#[pymethods]
impl PyPauliGroup {
    #[new]
    #[pyo3(signature = (generators, all_commute = None))]
    fn new(generators: &Bound<'_, PyAny>, all_commute: Option<bool>) -> PyResult<Self> {
        let pauli_vec = extract_sparse_pauli_from_iterable(generators)?;
        if all_commute.is_none() {
            return Ok(Self {
                inner: PauliGroup::new(&pauli_vec),
                is_abelian_promise: None,
            });
        }
        Ok(Self {
            inner: PauliGroup::with_promise(&pauli_vec, all_commute.unwrap()),
            is_abelian_promise: all_commute,
        })
    }

    #[getter]
    fn generators(&self) -> Vec<PySparsePauli> {
        self.inner.generators().iter().cloned().map(Into::into).collect()
    }

    #[getter]
    fn standard_generators(&self) -> Vec<PySparsePauli> {
        self.inner
            .standard_generators()
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    #[getter]
    fn support(&self) -> Vec<usize> {
        self.inner.support().clone()
    }

    #[getter]
    fn log2_size(&self) -> usize {
        self.inner.log2_size()
    }

    #[getter]
    fn elements(&self) -> PyPauliGroupIterator {
        PyPauliGroupIterator {
            iter: Box::new(self.inner.elements()),
        }
    }

    #[getter]
    fn binary_rank(&self) -> usize {
        self.inner.binary_rank()
    }

    #[getter]
    fn phases(&self) -> Vec<usize> {
        self.inner.phases().iter().map(|exponent| *exponent as usize).collect()
    }

    fn factorization_of(&self, element: &PySparsePauli) -> Option<Vec<PySparsePauli>> {
        self.inner.factorization_of(&element.inner).map(|factorization| {
            factorization
                .into_iter()
                .map(|pauli| PySparsePauli { inner: pauli })
                .collect()
        })
    }

    fn factorizations_of(&self, elements: &Bound<'_, PyAny>) -> Vec<Option<Vec<PySparsePauli>>> {
        let sparse_elements = to_sparse_pauli_vec(elements);
        let inner_elements = sparse_elements.clone();
        let factorizations = self.inner.factorizations_of(inner_elements.as_slice());
        factorizations
            .into_iter()
            .map(|factorization| {
                factorization.map(|factors| {
                    factors
                        .into_iter()
                        .map(|pauli| PySparsePauli { inner: pauli })
                        .collect()
                })
            })
            .collect()
    }

    #[getter]
    fn is_abelian(&self) -> bool {
        self.inner.is_abelian()
    }

    #[getter]
    fn is_stabilizer_group(&self) -> bool {
        self.inner.is_stabilizer_group()
    }

    fn __contains__(&self, element: &PySparsePauli) -> bool {
        self.inner.contains(&element.inner)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __lt__(&self, other: &Self) -> bool {
        self.inner < other.inner
    }

    fn __le__(&self, other: &Self) -> bool {
        self.inner <= other.inner
    }

    fn __truediv__(&self, other: &Self) -> PyResult<Self> {
        Python::attach(|py| {
            let warnings = py.import("warnings")?;
            warnings.call_method1(
                "warn",
                (
                    "PauliGroup division (/) is deprecated. Use modulo (%) for coset representatives instead.",
                    py.get_type::<pyo3::exceptions::PyDeprecationWarning>(),
                ),
            )?;
            Ok::<_, PyErr>(())
        })?;

        #[allow(deprecated)]
        self.inner
            .try_quotient(&other.inner)
            .map(|quotient| Self {
                inner: quotient,
                is_abelian_promise: None,
            })
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "Cannot compute quotient: the divisor is not a subgroup of the dividend",
                )
            })
    }

    fn __mod__(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.modulo(&other.inner),
            is_abelian_promise: None,
        }
    }

    fn __or__(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.clone() | &other.inner,
            is_abelian_promise: None, // Result of union doesn't preserve original promise
        }
    }

    fn __and__(&self, other: &Self) -> Self {
        Self {
            inner: &self.inner & &other.inner,
            is_abelian_promise: None, // Result of intersection doesn't preserve original promise
        }
    }

    fn __str__(&self) -> String {
        let generator_strs: Vec<String> = self.inner.generators().iter().map(ToString::to_string).collect();
        format!("⟨{}⟩", generator_strs.join(", "))
    }

    fn __repr__(&self) -> String {
        let generator_reprs: Vec<String> = self
            .inner
            .generators()
            .iter()
            .map(|g| format!("SparsePauli(\"{g}\")"))
            .collect();
        format!("PauliGroup([{}])", generator_reprs.join(", "))
    }

    fn __getstate__(&self) -> (Vec<PySparsePauli>, Option<bool>) {
        let generators = self.generators();
        let is_abelian_promise = self.is_abelian_promise;
        (generators, is_abelian_promise)
    }

    fn __setstate__(&mut self, state: (Vec<PySparsePauli>, Option<bool>)) {
        let (state_generators, promise_option) = state;
        let generators: Vec<SparsePauli> = state_generators.into_iter().map(|py_pauli| py_pauli.inner).collect();

        if let Some(promise) = promise_option {
            self.inner = PauliGroup::with_promise(&generators, promise);
        } else {
            self.inner = PauliGroup::new(&generators);
        }
        self.is_abelian_promise = promise_option;
    }

    fn __reduce__(&self) -> (Py<PyAny>, (Vec<PySparsePauli>, Option<bool>)) {
        let generators = self.generators();
        let is_abelian_promise = self.is_abelian_promise;

        Python::attach(|py| {
            let cls = py.get_type::<Self>();
            (cls.into(), (generators, is_abelian_promise))
        })
    }
}

#[pyfunction]
#[pyo3(name = "centralizer_of", signature=(group, supported_by=None))]
/// # Errors
/// Will return an error if the extraction of indices from `supported_by` fails.
pub fn py_centralizer_of(group: &PyPauliGroup, supported_by: Option<&Bound<'_, PyAny>>) -> PyResult<PyPauliGroup> {
    if let Some(supported_by_iterable) = supported_by {
        let mut support_vec = Vec::new();
        let iter = PyIterator::from_object(supported_by_iterable)?;
        for item in iter {
            let item = item?;
            let index: usize = item.extract()?;
            support_vec.push(index);
        }
        return Ok(PyPauliGroup {
            inner: centralizer_within(&support_vec, &group.inner),
            is_abelian_promise: None,
        });
    }
    Ok(PyPauliGroup {
        inner: centralizer_of(&group.inner),
        is_abelian_promise: None,
    })
}

#[pyfunction]
#[pyo3(name = "symplectic_form_of")]
/// # Errors
/// Will return an error if the extraction of SparsePauli(s) fails.
pub fn py_symplectic_form_of(generators: &Bound<'_, PyAny>) -> PyResult<Vec<PySparsePauli>> {
    let sparse_generators = to_sparse_pauli_vec(generators);

    Ok(symplectic_form_of(&sparse_generators)
        .iter()
        .map(|pauli| PySparsePauli { inner: pauli.clone() })
        .collect())
}

fn to_sparse_pauli_vec(elements: &Bound<'_, PyAny>) -> Vec<SparsePauli> {
    let mut pauli_vec = Vec::new();
    let iter = PyIterator::from_object(elements).unwrap();
    for item in iter {
        let item = item.unwrap();
        let pauli: PySparsePauli = item.extract().unwrap();
        pauli_vec.push(pauli.inner);
    }
    pauli_vec
}
