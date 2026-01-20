use derive_more::{Deref, DerefMut, From, Into};
use paulimer::clifford::{
    group_encoding_clifford_of, split_phased_css, split_qubit_cliffords_and_css, Clifford, CliffordMutable,
    CliffordUnitary, XOrZ,
};
use paulimer::pauli::{as_sparse, DensePauli, SparsePauli};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PySliceIndices;

use crate::enums::PyUnitaryOp;
use crate::py_dense_pauli::PyDensePauli;
use crate::py_sparse_pauli::PySparsePauli;

pub enum PyPauliInput {
    Dense(DensePauli),
    Sparse(SparsePauli),
}

impl PyPauliInput {
    pub fn to_sparse(&self) -> SparsePauli {
        match self {
            PyPauliInput::Dense(dense) => as_sparse(dense),
            PyPauliInput::Sparse(sparse) => sparse.clone(),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyPauliInput {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(dense) = ob.extract::<PyRef<PyDensePauli>>() {
            Ok(PyPauliInput::Dense(dense.inner.clone()))
        } else if let Ok(sparse) = ob.extract::<PyRef<PySparsePauli>>() {
            Ok(PyPauliInput::Sparse(sparse.inner.clone()))
        } else {
            Err(PyValueError::new_err("Expected DensePauli or SparsePauli"))
        }
    }
}

#[derive(Clone, Deref, DerefMut, From, Into)]
#[must_use]
#[pyclass(name = "CliffordUnitary", module = "paulimer")]
pub struct PyCliffordUnitary {
    #[deref]
    #[deref_mut]
    pub(crate) inner: CliffordUnitary,
}

impl<'life> From<&'life PyCliffordUnitary> for &'life CliffordUnitary {
    fn from(py_clifford: &'life PyCliffordUnitary) -> Self {
        &py_clifford.inner
    }
}

#[pymethods]
impl PyCliffordUnitary {
    /// Creates an empty 0-qubit Clifford (used internally for pickle deserialization)
    #[new]
    fn new() -> Self {
        Self {
            inner: CliffordUnitary::identity(0),
        }
    }

    #[getter]
    fn qubit_count(&self) -> usize {
        self.inner.num_qubits()
    }

    /// # Panic
    ///
    /// Will panic when number of qubits does not fit into isize
    fn qubits(&self) -> PySliceIndices {
        let range = self.inner.qubits();
        PySliceIndices::new(range.start.try_into().unwrap(), range.end.try_into().unwrap(), 1)
    }

    #[getter]
    fn is_valid(&self) -> bool {
        self.inner.is_valid()
    }

    #[getter]
    fn is_identity(&self) -> bool {
        self.inner.is_identity()
    }

    #[staticmethod]
    fn identity(num_qubits: usize) -> Self {
        Self {
            inner: CliffordUnitary::identity(num_qubits),
        }
    }

    #[staticmethod]
    fn from_string(characters: &str) -> PyResult<Self> {
        match characters.parse() {
            Ok(clifford) => Ok(Self { inner: clifford }),
            Err(_) => Err(PyValueError::new_err("Invalid clifford string.")),
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[staticmethod]
    fn from_name(unitary_op: &str, qubits: Vec<usize>, qubit_count: usize) -> PyResult<Self> {
        let mut res = CliffordUnitary::identity(qubit_count);
        let Ok(unitary_op) = unitary_op.parse::<PyUnitaryOp>() else {
            return Err(PyValueError::new_err("Invalid unitary operation name."));
        };
        res.left_mul(unitary_op.into(), &qubits);
        Ok(res.into())
    }

    #[staticmethod]
    fn zero(num_qubits: usize) -> Self {
        CliffordUnitary::zero(num_qubits).into()
    }

    #[allow(clippy::needless_pass_by_value)]
    #[staticmethod]
    fn group_encoding_clifford_of(generators: Vec<PySparsePauli>, qubit_count: usize) -> Self {
        group_encoding_clifford_of(
            &generators.iter().map(|x| x.inner.clone()).collect::<Vec<_>>(),
            qubit_count,
        )
        .into()
    }

    fn split_qubit_cliffords_and_css(&self) -> Option<(Self, Self)> {
        let res = split_qubit_cliffords_and_css(&self.inner.clone().into());
        let (left_mod_pauli, _) = res?;
        let left = CliffordUnitary::from(left_mod_pauli);
        let right = left.inverse().multiply_with(&self.inner);
        Some((left.into(), right.into()))
    }

    fn split_phased_css(&self) -> Option<(Self, Self)> {
        //Optional[Tuple[CliffordUnitary, CliffordUnitary]]
        let res = split_phased_css(&self.inner.clone().into());
        let (left_mod_pauli, _) = res?;
        let left = CliffordUnitary::from(left_mod_pauli);
        let right = left.inverse().multiply_with(&self.inner);
        Some((left.into(), right.into()))
    }

    fn preimage_x(&self, qubit_index: usize) -> PyDensePauli {
        PyDensePauli {
            inner: self.inner.preimage_x(qubit_index),
            size: self.qubit_count(),
        }
    }

    fn preimage_z(&self, qubit_index: usize) -> PyDensePauli {
        PyDensePauli {
            inner: self.inner.preimage_z(qubit_index),
            size: self.qubit_count(),
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    fn preimage_of(&self, pauli: PyPauliInput) -> PyDensePauli {
        let inner = match pauli {
            PyPauliInput::Dense(ref dense) => self.inner.preimage(dense),
            PyPauliInput::Sparse(ref sparse) => self.inner.preimage(sparse),
        };
        PyDensePauli {
            inner,
            size: self.qubit_count(),
        }
    }

    fn image_x(&self, qubit_index: usize) -> PyDensePauli {
        PyDensePauli {
            inner: self.inner.image_x(qubit_index),
            size: self.qubit_count(),
        }
    }

    fn image_z(&self, qubit_index: usize) -> PyDensePauli {
        PyDensePauli {
            inner: self.inner.image_z(qubit_index),
            size: self.qubit_count(),
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    fn image_of(&self, pauli: PyPauliInput) -> PyDensePauli {
        let inner = match pauli {
            PyPauliInput::Dense(ref dense) => self.inner.image(dense),
            PyPauliInput::Sparse(ref sparse) => self.inner.image(sparse),
        };
        PyDensePauli {
            inner,
            size: self.qubit_count(),
        }
    }

    fn tensor(&self, rhs: &Self) -> Self {
        Self {
            inner: self.inner.tensor(&rhs.inner),
        }
    }

    fn inverse(&self) -> Self {
        Self {
            inner: self.inner.inverse(),
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[staticmethod]
    fn from_preimages(preimages: Vec<PyDensePauli>) -> Self {
        let inner_preimages: Vec<DensePauli> = preimages.iter().map(|x| x.inner.clone()).collect();

        Self {
            inner: CliffordUnitary::from_preimages(&inner_preimages[..]),
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[staticmethod]
    fn from_images(images: Vec<PyDensePauli>) -> Self {
        let inner_images: Vec<DensePauli> = images.iter().map(|x| x.inner.clone()).collect();
        Self {
            inner: CliffordUnitary::from_images(&inner_images[..]),
        }
    }

    fn __mul__(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.multiply_with(&other.inner),
        }
    }

    #[getter]
    fn is_css(&self) -> bool {
        self.inner.is_css()
    }

    fn is_diagonal(&self, axis: &str) -> bool {
        let axis_enum: XOrZ = match axis {
            "X" => XOrZ::X,
            "Z" => XOrZ::Z,
            _ => return false,
        };
        self.inner.is_diagonal(axis_enum)
    }

    fn is_diagonal_resource_encoder(&self, axis: &str) -> bool {
        let axis_enum: XOrZ = match axis {
            "X" => XOrZ::X,
            "Z" => XOrZ::Z,
            _ => return false,
        };
        self.inner.is_diagonal_resource_encoder(axis_enum)
    }

    fn unitary_from_diagonal_resource_state(&self, axis: &str) -> Option<Self> {
        let axis_enum: XOrZ = match axis {
            "X" => XOrZ::X,
            "Z" => XOrZ::Z,
            _ => return None,
        };
        self.inner
            .unitary_from_diagonal_resource_state(axis_enum)
            .map(Into::into)
    }

    #[getter]
    fn symplectic_matrix(&self) -> binar::BitMatrix {
        self.inner.symplectic_matrix().into()
    }

    #[allow(clippy::needless_pass_by_value)]
    fn left_mul(&mut self, unitary_op: PyUnitaryOp, support: Vec<usize>) {
        self.inner.left_mul(unitary_op.into(), &support);
    }

    #[allow(clippy::needless_pass_by_value)]
    fn left_mul_clifford(&mut self, clifford: &PyCliffordUnitary, support: Vec<usize>) {
        self.inner.left_mul_clifford(&clifford.inner, &support);
    }

    #[allow(clippy::needless_pass_by_value)]
    fn left_mul_permutation(&mut self, permutation: Vec<usize>, support: Vec<usize>) {
        self.inner.left_mul_permutation(&permutation, &support);
    }

    #[allow(clippy::needless_pass_by_value)]
    fn left_mul_pauli(&mut self, pauli: PyPauliInput) {
        match pauli {
            PyPauliInput::Dense(ref dense) => self.inner.left_mul_pauli(dense),
            PyPauliInput::Sparse(ref sparse) => self.inner.left_mul_pauli(sparse),
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    fn left_mul_pauli_exp(&mut self, pauli: PyPauliInput) {
        match pauli {
            PyPauliInput::Dense(ref dense) => self.inner.left_mul_pauli_exp(dense),
            PyPauliInput::Sparse(ref sparse) => self.inner.left_mul_pauli_exp(sparse),
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    fn left_mul_controlled_pauli(&mut self, control: PyPauliInput, target: PyPauliInput) {
        match (&control, &target) {
            (PyPauliInput::Dense(c), PyPauliInput::Dense(t)) => {
                self.inner.left_mul_controlled_pauli(c, t);
            }
            (PyPauliInput::Sparse(c), PyPauliInput::Sparse(t)) => {
                self.inner.left_mul_controlled_pauli(c, t);
            }
            _ => {
                let c = control.to_sparse();
                let t = target.to_sparse();
                self.inner.left_mul_controlled_pauli(&c, &t);
            }
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

    fn __pow__(&self, exponent: isize, _modulo: Option<Py<PyAny>>) -> Self {
        if exponent < 0 {
            self.inner.inverse().power(exponent.unsigned_abs()).into()
        } else {
            self.inner.power(exponent.unsigned_abs()).into()
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __ne__(&self, other: &Self) -> bool {
        self.inner != other.inner
    }

    /// Returns (words, phases) for efficient portable serialization
    #[must_use]
    pub fn __getstate__(&self) -> (Vec<u64>, Vec<u8>) {
        let (words, phases) = self.inner.as_words();
        (words.to_vec(), phases.to_vec())
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn __setstate__(&mut self, state: (Vec<u64>, Vec<u8>)) {
        let (words, phases) = state;
        self.inner = CliffordUnitary::from_words(&words, phases);
    }
}

// Standalone functions

#[pyfunction]
#[pyo3(name = "is_diagonal_resource_encoder")]
#[allow(clippy::must_use_candidate)]
pub fn py_is_diagonal_resource_encoder(clifford: &PyCliffordUnitary, axis: &str) -> bool {
    let axis_enum: XOrZ = match axis {
        "X" => XOrZ::X,
        "Z" => XOrZ::Z,
        _ => return false,
    };
    clifford.inner.is_diagonal_resource_encoder(axis_enum)
}

#[pyfunction]
#[pyo3(name = "unitary_from_diagonal_resource_state")]
pub fn py_unitary_from_diagonal_resource_state(clifford: &PyCliffordUnitary, axis: &str) -> Option<PyCliffordUnitary> {
    let axis_enum: XOrZ = match axis {
        "X" => XOrZ::X,
        "Z" => XOrZ::Z,
        _ => return None,
    };
    clifford
        .inner
        .unitary_from_diagonal_resource_state(axis_enum)
        .map(Into::into)
}

#[pyfunction]
#[pyo3(name = "split_qubit_cliffords_and_css")]
#[allow(clippy::must_use_candidate)]
pub fn py_split_qubit_cliffords_and_css(
    clifford: &PyCliffordUnitary,
) -> Option<(PyCliffordUnitary, PyCliffordUnitary)> {
    let res = split_qubit_cliffords_and_css(&clifford.inner.clone().into());
    let (left_mod_pauli, _) = res?;
    let left = CliffordUnitary::from(left_mod_pauli);
    let right = left.inverse().multiply_with(&clifford.inner);
    Some((left.into(), right.into()))
}

#[pyfunction]
#[pyo3(name = "split_phased_css")]
#[allow(clippy::must_use_candidate)]
pub fn py_split_phased_css(clifford: &PyCliffordUnitary) -> Option<(PyCliffordUnitary, PyCliffordUnitary)> {
    let res = split_phased_css(&clifford.inner.clone().into());
    let (left_mod_pauli, _) = res?;
    let left = CliffordUnitary::from(left_mod_pauli);
    let right = left.inverse().multiply_with(&clifford.inner);
    Some((left.into(), right.into()))
}

#[pyfunction]
#[pyo3(name = "encoding_clifford_of")]
#[allow(clippy::needless_pass_by_value)]
pub fn py_encoding_clifford_of(generators: Vec<PyPauliInput>, qubit_count: usize) -> PyCliffordUnitary {
    let pauli_vec: Vec<SparsePauli> = generators.iter().map(PyPauliInput::to_sparse).collect();
    group_encoding_clifford_of(&pauli_vec, qubit_count).into()
}
