#![allow(clippy::must_use_candidate)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::needless_pass_by_value)]

use crate::{PyCliffordUnitary, PyFault, PySparsePauli, PyUnitaryOp};
use binar::BitMatrix;
use paulimer::clifford::Clifford;
use pauliverse::{FaultySimulation, Simulation};
use pyo3::prelude::*;
use rand::{rngs::SmallRng, SeedableRng};

/// Frame-based noisy simulation with circuit-builder interface.
///
/// Implements the StabilizerSimulation protocol: call gate methods to build
/// a circuit, then call `sample()` to get noisy outcomes.
///
/// Example:
///     sim = FaultySimulation()
///     sim.apply_unitary(UnitaryOpcode.Hadamard, [0])
///     sim.apply_unitary(UnitaryOpcode.ControlledX, [0, 1])
///     sim.measure(SparsePauli("ZI"))
///     sim.measure(SparsePauli("IZ"))
///     sim.apply_fault(PauliFault.depolarizing(0.01, [0, 1]))
///     outcomes = sim.sample(1000)
#[pyclass(name = "FaultySimulation", unsendable)]
pub struct PyFaultySimulation {
    inner: FaultySimulation,
}

#[pymethods]
impl PyFaultySimulation {
    /// Create a new simulation.
    ///
    /// Args:
    ///     qubit_count: Expected number of qubits (optional, for pre-allocation).
    ///     outcome_count: Expected number of measurement outcomes (optional).
    ///     instruction_count: Expected number of instructions (optional).
    ///
    /// Pre-allocating capacity can improve performance for large circuits.
    #[new]
    #[pyo3(signature = (qubit_count=None, outcome_count=None, instruction_count=None))]
    pub fn new(qubit_count: Option<usize>, outcome_count: Option<usize>, instruction_count: Option<usize>) -> Self {
        match (qubit_count, outcome_count, instruction_count) {
            (Some(q), Some(o), Some(i)) => PyFaultySimulation {
                inner: FaultySimulation::with_capacity(q, o, i),
            },
            (Some(q), Some(o), None) => PyFaultySimulation {
                inner: FaultySimulation::with_capacity(q, o, 0),
            },
            (Some(q), None, Some(i)) => PyFaultySimulation {
                inner: FaultySimulation::with_capacity(q, 0, i),
            },
            (Some(q), None, None) => PyFaultySimulation {
                inner: FaultySimulation::with_capacity(q, 0, 0),
            },
            _ => PyFaultySimulation {
                inner: FaultySimulation::new(),
            },
        }
    }

    // ========== StabilizerSimulation protocol methods ==========

    #[getter]
    pub fn qubit_count(&self) -> usize {
        Simulation::qubit_count(&self.inner)
    }

    #[getter]
    pub fn outcome_count(&self) -> usize {
        Simulation::outcome_count(&self.inner)
    }

    #[getter]
    pub fn fault_count(&self) -> usize {
        self.inner.fault_count()
    }

    /// Apply a unitary gate.
    pub fn apply_unitary(&mut self, opcode: &PyUnitaryOp, qubits: Vec<usize>) {
        self.inner.unitary_op(opcode.clone().into(), &qubits);
    }

    /// Apply a Clifford unitary.
    #[pyo3(signature = (clifford, qubits=None))]
    pub fn apply_clifford(&mut self, clifford: &PyCliffordUnitary, qubits: Option<Vec<usize>>) {
        let qubits = qubits.unwrap_or_else(|| (0..clifford.inner.num_qubits()).collect());
        self.inner.clifford(&clifford.inner, &qubits);
    }

    /// Apply a Pauli gate.
    #[pyo3(signature = (pauli, controlled_by=None))]
    pub fn apply_pauli(&mut self, pauli: &PySparsePauli, controlled_by: Option<&PySparsePauli>) {
        if let Some(control) = controlled_by {
            self.inner.controlled_pauli(&control.inner, &pauli.inner);
        } else {
            self.inner.pauli(&pauli.inner);
        }
    }

    /// Apply a Pauli exponential (e^{-iπ/4 P}).
    pub fn apply_pauli_exp(&mut self, pauli: &PySparsePauli) {
        self.inner.pauli_exp(&pauli.inner);
    }

    /// Apply a permutation.
    #[pyo3(signature = (permutation, qubits=None))]
    pub fn apply_permutation(&mut self, permutation: Vec<usize>, qubits: Option<Vec<usize>>) {
        let qubits = qubits.unwrap_or_else(|| (0..permutation.len()).collect());
        self.inner.permute(&permutation, &qubits);
    }

    /// Apply a conditional Pauli based on measurement outcomes.
    #[pyo3(signature = (pauli, outcomes, parity=true))]
    pub fn apply_conditional_pauli(&mut self, pauli: &PySparsePauli, outcomes: Vec<usize>, parity: bool) {
        self.inner.conditional_pauli(&pauli.inner, &outcomes, parity);
    }

    /// Measure an observable, returning the outcome index.
    ///
    /// Args:
    ///     observable: The Pauli observable to measure.
    ///     hint: Optional anti-commuting stabilizer hint. Accepted for API compatibility
    ///           with other simulation backends but ignored by FaultySimulation.
    #[pyo3(signature = (observable, hint=None))]
    pub fn measure(&mut self, observable: &PySparsePauli, hint: Option<&PySparsePauli>) -> usize {
        let _ = hint;
        Simulation::measure(&mut self.inner, &observable.inner)
    }

    /// Allocate a random bit, returning the outcome index.
    pub fn allocate_random_bit(&mut self) -> usize {
        Simulation::allocate_random_bit(&mut self.inner)
    }

    /// Add a fault (noise) instruction.
    pub fn apply_fault(&mut self, fault: &PyFault) {
        self.inner.apply_fault(fault.inner.clone());
    }

    /// Sample measurement outcomes.
    ///
    /// Args:
    ///     shots: Number of shots to sample.
    ///     seed: Optional random seed for reproducibility.
    ///
    /// Returns:
    ///     A BitMatrix of shape (shots, outcome_count).
    #[pyo3(signature = (shots, seed=None))]
    pub fn sample(&self, shots: usize, seed: Option<u64>) -> BitMatrix {
        match seed {
            Some(s) => {
                let mut rng = SmallRng::seed_from_u64(s);
                self.inner.sample_with_rng(shots, &mut rng)
            }
            None => self.inner.sample(shots),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "FaultySimulation(qubit_count={}, outcome_count={}, fault_count={})",
            self.inner.qubit_count(),
            self.inner.outcome_count(),
            self.inner.fault_count()
        )
    }
}
