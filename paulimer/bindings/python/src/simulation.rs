use std::ops::{Deref, DerefMut};

use binar::{BitMatrix, BitVec, IntoBitIterator};
use paulimer::clifford::CliffordUnitary;
use pauliverse::outcome_complete_simulation::OutcomeCompleteSimulation;
use pauliverse::outcome_free_simulation::OutcomeFreeSimulation;
use pauliverse::outcome_specific_simulation::OutcomeSpecificSimulation;
use pauliverse::Simulation;
use pyo3::prelude::*;

use crate::enums::PyUnitaryOp;
use crate::py_clifford::PyCliffordUnitary;
use crate::py_sparse_pauli::PySparsePauli;

#[derive(derive_more::Deref, derive_more::DerefMut, derive_more::From)]
#[must_use]
#[pyclass(name = "OutcomeCompleteSimulation", module = "paulimer")]
pub struct PyOutcomeCompleteSimulation {
    inner: OutcomeCompleteSimulation,
}

#[derive(derive_more::Deref, derive_more::DerefMut, derive_more::From)]
#[must_use]
#[pyclass(name = "OutcomeSpecificSimulation", module = "paulimer")]
pub struct PyOutcomeSpecificSimulation {
    inner: OutcomeSpecificSimulation,
}

#[derive(derive_more::Deref, derive_more::DerefMut, derive_more::From)]
#[must_use]
#[pyclass(name = "OutcomeFreeSimulation", module = "paulimer")]
pub struct PyOutcomeFreeSimulation {
    inner: OutcomeFreeSimulation,
}

macro_rules! impl_simulation {
    ($struct_name:ty, $wrapper_struct:ty { $($inside:tt)* }) => {
        #[pymethods]
        impl $wrapper_struct {
            #[new]
            #[pyo3(signature=(qubit_count=0))]
            pub fn new(qubit_count: usize) -> Self {
                <$struct_name>::new(qubit_count).into()
            }

            #[getter]
            #[must_use]
            pub fn qubit_count(&self) -> usize {
                self.deref().qubit_count()
            }

            #[getter]
            #[must_use]
            pub fn qubit_capacity(&self) -> usize {
                self.deref().qubit_capacity()
            }

            #[getter]
            #[must_use]
            pub fn outcome_count(&self) -> usize {
                self.deref().outcome_count()
            }

            #[getter]
            #[must_use]
            pub fn outcome_capacity(&self) -> usize {
                self.deref().outcome_capacity()
            }

            #[getter]
            #[must_use]
            pub fn random_outcome_count(&self) -> usize {
                self.deref().random_outcome_count()
            }

            #[getter]
            #[must_use]
            pub fn random_outcome_capacity(&self) -> usize {
                self.deref().random_outcome_capacity()
            }

            #[getter]
            #[must_use]
            pub fn random_bit_count(&self) -> usize {
                self.deref().random_outcome_count()
            }

            #[allow(clippy::needless_pass_by_value)]
            pub fn apply_unitary(&mut self, unitary_op: PyUnitaryOp, support: Vec<usize>) {
                Simulation::unitary_op(self.deref_mut(), unitary_op.into(), &support);
            }

            pub fn apply_pauli_exp(&mut self, observable: &PySparsePauli) {
                Simulation::pauli_exp(self.deref_mut(), &observable.inner);
            }

            #[pyo3(signature=(observable, controlled_by=None))]
            pub fn apply_pauli(&mut self, observable: &PySparsePauli, controlled_by: Option<&PySparsePauli>) {
                if let Some(control) = controlled_by {
                    Simulation::controlled_pauli(self.deref_mut(), &observable.inner, &control.inner);
                } else {
                    Simulation::pauli(self.deref_mut(), &observable.inner);
                }
            }

            #[allow(clippy::needless_pass_by_value)]
            #[pyo3(signature=(observable, outcomes, parity=true))]
            pub fn apply_conditional_pauli(&mut self, observable: &PySparsePauli, outcomes: Vec<usize>, parity: bool) {
                Simulation::conditional_pauli(self.deref_mut(), &observable.inner, &outcomes, parity);
            }

            #[allow(clippy::needless_pass_by_value)]
            #[pyo3(signature=(permutation, supported_by=None))]
            pub fn apply_permutation(&mut self, permutation: Vec<usize>, supported_by: Option<Vec<usize>>) {
                let support = supported_by.unwrap_or_else(|| (0..self.deref().qubit_count()).collect());
                Simulation::permute(self.deref_mut(), &permutation, &support);
            }

            #[allow(clippy::needless_pass_by_value)]
            #[pyo3(signature=(clifford, supported_by=None))]
            pub fn apply_clifford(&mut self, clifford: &PyCliffordUnitary, supported_by: Option<Vec<usize>>) {
                let support = supported_by.unwrap_or_else(|| (0..self.deref().qubit_count()).collect());
                Simulation::clifford(self.deref_mut(), &clifford.inner, &support);
            }

            #[pyo3(signature=(observable, hint=None))]
            pub fn measure(&mut self, observable: &PySparsePauli, hint: Option<&PySparsePauli>) -> usize {
                if let Some(h) = hint {
                    Simulation::measure_with_hint(self.deref_mut(), &observable.inner, &h.inner)
                } else {
                    Simulation::measure(self.deref_mut(), &observable.inner)
                }
            }

            pub fn allocate_random_bit(&mut self) -> usize {
                self.inner.allocate_random_bit()
            }

            pub fn reserve_qubits(&mut self, new_qubit_capacity: usize) {
                Simulation::reserve_qubits(self.deref_mut(), new_qubit_capacity);
            }

            pub fn reserve_outcomes(&mut self, new_outcome_capacity: usize, new_random_outcome_capacity: usize) {
                Simulation::reserve_outcomes(self.deref_mut(), new_outcome_capacity, new_random_outcome_capacity);
            }

            #[pyo3(signature=(observable, ignore_sign=false, sign_parity=None))]
            #[must_use]
            pub fn is_stabilizer(&self, observable: &PySparsePauli, ignore_sign: bool, sign_parity: Option<Vec<usize>>) -> bool {
                if ignore_sign {
                    Simulation::is_stabilizer_up_to_sign(self.deref(), &observable.inner)
                } else if let Some(parity_outcomes) = sign_parity {
                    Simulation::is_stabilizer_with_conditional_sign(self.deref(), &observable.inner, &parity_outcomes)
                } else {
                    Simulation::is_stabilizer(self.deref(), &observable.inner)
                }
            }

            #[staticmethod]
            #[pyo3(signature=(num_qubits, num_outcomes, num_random_outcomes))]
            pub fn with_capacity(num_qubits: usize, num_outcomes: usize, num_random_outcomes: usize) -> Self {
                <$struct_name>::with_capacity(num_qubits, num_outcomes, num_random_outcomes).into()
            }

            #[getter]
            pub fn random_outcome_indicator(&self) -> BitVec {
                self.deref().random_outcome_indicator().iter().copied().collect::<BitVec>()
            }

            $($inside)*
        }
    };
}

impl_simulation!(
    OutcomeCompleteSimulation,
    PyOutcomeCompleteSimulation {
        #[getter]
        pub fn clifford(&self) -> PyCliffordUnitary {
            PyCliffordUnitary {
                inner: self.deref().state_encoder().clone(),
            }
        }

        #[getter]
        pub fn sign_matrix(&self) -> BitMatrix {
            BitMatrix::from(self.deref().aligned_sign_matrix().clone())
        }

        #[getter]
        pub fn outcome_matrix(&self) -> BitMatrix {
            BitMatrix::from(self.deref().aligned_outcome_matrix().clone())
        }

        #[getter]
        pub fn outcome_shift(&self) -> BitVec {
            self.deref().outcome_shift().iter_bits().collect::<BitVec>()
        }
});

impl_simulation!(
    OutcomeFreeSimulation,
    PyOutcomeFreeSimulation {
        #[getter]
        pub fn clifford(&self) -> PyCliffordUnitary {
            let c: CliffordUnitary = self.deref().state_encoder().clone().into();
            PyCliffordUnitary { inner: c }
        }
});

impl_simulation!(
    OutcomeSpecificSimulation,
    PyOutcomeSpecificSimulation {
        #[getter]
        pub fn clifford(&self) -> PyCliffordUnitary {
            PyCliffordUnitary {
                inner: self.deref().state_encoder().clone(),
            }
        }

        #[getter]
        pub fn outcome_vector(&self) -> BitVec {
            self.deref().outcome_vector().iter().copied().collect::<BitVec>()
        }

        #[staticmethod]
        #[pyo3(signature=(num_qubits))]
        pub fn with_zero_outcomes(num_qubits: usize) -> Self {
            OutcomeSpecificSimulation::new_with_zero_outcomes(num_qubits).into()
        }

        #[staticmethod]
        #[pyo3(signature=(num_qubits, seed = 0))]
        pub fn new_with_seeded_random_outcomes(num_qubits: usize, seed: u64) -> Self {
            OutcomeSpecificSimulation::new_with_seeded_random_outcomes(num_qubits, seed).into()
        }
});
