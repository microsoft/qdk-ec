use ::paulimer::Pauli;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod enums;
mod format_spec;
mod py_clifford;
mod py_dense_pauli;
mod py_faulty_simulation;
mod py_frame_propagator;
mod py_noise;
mod py_pauli_group;
mod py_sparse_pauli;
mod simulation;

pub use enums::PyUnitaryOp;
pub use py_clifford::{
    py_encoding_clifford_of, py_is_diagonal_resource_encoder, py_split_phased_css, py_split_qubit_cliffords_and_css,
    py_unitary_from_diagonal_resource_state, PyCliffordUnitary,
};
pub use py_dense_pauli::PyDensePauli;
pub use py_faulty_simulation::PyFaultySimulation;
pub use py_frame_propagator::PyFramePropagator;
pub use py_noise::{PyFault, PyOutcomeCondition, PyPauliDistribution};
pub use py_pauli_group::{py_centralizer_of, py_symplectic_form_of, PyPauliGroup};
pub use py_sparse_pauli::PySparsePauli;
pub use simulation::{PyOutcomeCompleteSimulation, PyOutcomeFreeSimulation, PyOutcomeSpecificSimulation};

pub(crate) fn validate_qubit_index(index: usize, qubit_count: usize) -> PyResult<()> {
    if index >= qubit_count {
        return Err(PyValueError::new_err(format!(
            "Qubit index {index} out of bounds for {qubit_count} qubits."
        )));
    }
    Ok(())
}

pub(crate) fn validate_pauli_support(pauli: &impl Pauli, qubit_count: usize) -> PyResult<()> {
    if let Some(index) = pauli.max_support() {
        validate_qubit_index(index, qubit_count)?;
    }
    Ok(())
}

pub(crate) fn validate_qubit_indices(indices: &[usize], qubit_count: usize) -> PyResult<()> {
    for &index in indices {
        validate_qubit_index(index, qubit_count)?;
    }
    Ok(())
}

/// # Errors
///
/// Will return Err
#[pymodule]
pub fn paulimer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDensePauli>()?;
    m.add_class::<PySparsePauli>()?;
    m.add_class::<PyPauliGroup>()?;
    m.add_class::<PyCliffordUnitary>()?;
    m.add_class::<PyUnitaryOp>()?;
    m.add_class::<PyOutcomeCompleteSimulation>()?;
    m.add_class::<PyOutcomeSpecificSimulation>()?;
    m.add_class::<PyOutcomeFreeSimulation>()?;
    m.add_class::<PyFaultySimulation>()?;
    m.add_class::<PyFramePropagator>()?;
    m.add_class::<PyFault>()?;
    m.add_class::<PyPauliDistribution>()?;
    m.add_class::<PyOutcomeCondition>()?;
    m.add_function(wrap_pyfunction!(py_centralizer_of, m)?)?;
    m.add_function(wrap_pyfunction!(py_symplectic_form_of, m)?)?;
    m.add_function(wrap_pyfunction!(py_is_diagonal_resource_encoder, m)?)?;
    m.add_function(wrap_pyfunction!(py_unitary_from_diagonal_resource_state, m)?)?;
    m.add_function(wrap_pyfunction!(py_split_qubit_cliffords_and_css, m)?)?;
    m.add_function(wrap_pyfunction!(py_split_phased_css, m)?)?;
    m.add_function(wrap_pyfunction!(py_encoding_clifford_of, m)?)?;
    Ok(())
}
