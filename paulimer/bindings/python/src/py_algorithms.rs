use paulimer::pauli::{indexed_anticommutators_of, SparsePauli};
use pyo3::prelude::*;
use pyo3::types::PyIterator;

use crate::py_clifford::PyPauliInput;

#[pyfunction]
#[pyo3(name = "indexed_anticommutators_of")]
#[allow(clippy::needless_pass_by_value)]
/// Returns the indices of Pauli operators in `paulis` that anticommute with `observable`.
///
/// # Errors
/// Will return an error if `paulis` is not iterable over `SparsePauli` or `DensePauli`.
pub fn py_indexed_anticommutators_of(observable: PyPauliInput, paulis: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let obs = observable.to_sparse();
    let sparse: Vec<SparsePauli> = PyIterator::from_object(paulis)?
        .map(|item| Ok(item?.extract::<PyPauliInput>()?.to_sparse()))
        .collect::<PyResult<_>>()?;
    Ok(indexed_anticommutators_of(&obs, &sparse).into_iter().collect())
}
