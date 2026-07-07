#![allow(clippy::must_use_candidate)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::needless_pass_by_value)]

use crate::{PyCliffordUnitary, PySparsePauli, PyUnitaryOp};
use binar::BitMatrix;
use paulimer::clifford::Clifford;
use paulimer::pauli::Pauli;
use pauliverse::{FramePropagator, Simulation};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

/// Heisenberg-picture batched Pauli error frame propagator.
///
/// Tracks Pauli errors injected at arbitrary points in a circuit across many
/// shots in parallel using `(qubit_count × shot_count)` X/Z bit matrices and
/// an `(outcome_count × shot_count)` matrix of outcome deltas (per-shot XOR
/// against the noiseless trajectory).
///
/// Typical workflow:
///
/// 1. Construct with the qubit/outcome/shot capacities of your circuit.
/// 2. Walk the circuit; at the desired locations call `inject_pauli` to inject
///    the fault for one shot.
/// 3. Apply gates via `apply_unitary` / `apply_clifford` / etc. and record
///    measurements via `measure`.
/// 4. Read `outcome_deltas` to get the per-shot outcome flips relative to the
///    noiseless trajectory.
///
/// Pauli gates are no-ops in frame propagation (they commute through Pauli
/// errors up to a phase). The `parity` argument of `apply_conditional_pauli`
/// is accepted for API compatibility but ignored: only the delta of the
/// condition matters when propagating error frames.
#[pyclass(name = "FramePropagator", unsendable)]
pub struct PyFramePropagator {
    inner: FramePropagator,
}

#[pymethods]
impl PyFramePropagator {
    /// Create a new propagator.
    ///
    /// Args:
    ///     qubit_count: Number of qubits to track.
    ///     outcome_count: Number of measurement outcomes the circuit will produce.
    ///     shot_count: Number of independent shots to run in parallel.
    #[new]
    #[pyo3(signature = (qubit_count, outcome_count, shot_count))]
    pub fn new(qubit_count: usize, outcome_count: usize, shot_count: usize) -> Self {
        PyFramePropagator {
            inner: FramePropagator::new(qubit_count, outcome_count, shot_count),
        }
    }

    #[getter]
    pub fn qubit_count(&self) -> usize {
        Simulation::qubit_count(&self.inner)
    }

    #[getter]
    pub fn outcome_count(&self) -> usize {
        Simulation::outcome_count(&self.inner)
    }

    #[getter]
    pub fn shot_count(&self) -> usize {
        self.inner.shot_count()
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

    /// Apply a Pauli gate (no-op for frame propagation).
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

    /// Apply a conditional Pauli based on previous outcome ids.
    ///
    /// `parity` is accepted for API compatibility but ignored: in frame
    /// propagation, only the delta of the condition matters.
    #[pyo3(signature = (pauli, outcomes, parity=true))]
    pub fn apply_conditional_pauli(&mut self, pauli: &PySparsePauli, outcomes: Vec<usize>, parity: bool) {
        self.inner.conditional_pauli(&pauli.inner, &outcomes, parity);
    }

    /// Record a measurement of `observable` and return the assigned outcome id.
    ///
    /// `hint` is accepted for API compatibility with other simulation backends
    /// and ignored.
    #[pyo3(signature = (observable, hint=None))]
    pub fn measure(&mut self, observable: &PySparsePauli, hint: Option<&PySparsePauli>) -> usize {
        let _ = hint;
        Simulation::measure(&mut self.inner, &observable.inner)
    }

    /// Allocate a random measurement outcome (no anti-commutation update).
    pub fn allocate_random_bit(&mut self) -> usize {
        Simulation::allocate_random_bit(&mut self.inner)
    }

    /// Inject a Pauli error into a specific shot at the current circuit position.
    ///
    /// Raises:
    ///     IndexError: if `shot` is out of range, or `pauli` acts on a qubit
    ///         beyond `qubit_count`.
    ///
    /// # Errors
    ///
    /// Returns a Python `IndexError` if `shot >= shot_count` or `pauli` acts on
    /// a qubit index `>= qubit_count`.
    pub fn inject_pauli(&mut self, shot: usize, pauli: &PySparsePauli) -> PyResult<()> {
        let shot_count = self.inner.shot_count();
        if shot >= shot_count {
            return Err(PyIndexError::new_err(format!(
                "shot {shot} out of range (shot_count = {shot_count})"
            )));
        }
        let qubit_count = Simulation::qubit_count(&self.inner);
        if let Some(max_qubit) = pauli.inner.max_support() {
            if max_qubit >= qubit_count {
                return Err(PyIndexError::new_err(format!(
                    "pauli acts on qubit {max_qubit} out of range (qubit_count = {qubit_count})"
                )));
            }
        }
        self.inner.inject_pauli(shot, &pauli.inner);
        Ok(())
    }

    /// Reset a qubit, clearing its accumulated error frame across all shots.
    ///
    /// Raises:
    ///     IndexError: if `qubit` is out of range.
    ///
    /// # Errors
    ///
    /// Returns a Python `IndexError` if `qubit >= qubit_count`.
    pub fn reset_qubit(&mut self, qubit: usize) -> PyResult<()> {
        let qubit_count = Simulation::qubit_count(&self.inner);
        if qubit >= qubit_count {
            return Err(PyIndexError::new_err(format!(
                "qubit {qubit} out of range (qubit_count = {qubit_count})"
            )));
        }
        self.inner.reset_qubit(qubit);
        Ok(())
    }

    /// Outcome delta matrix: `(outcome_count × shot_count)` row-major bits.
    ///
    /// Bit `(o, s)` is true iff outcome `o` in shot `s` differs from the
    /// noiseless trajectory.
    #[getter]
    pub fn outcome_deltas(&self) -> BitMatrix {
        self.inner.outcome_deltas().clone().into()
    }

    fn __repr__(&self) -> String {
        format!(
            "FramePropagator(qubit_count={}, outcome_count={}, shot_count={})",
            Simulation::qubit_count(&self.inner),
            Simulation::outcome_count(&self.inner),
            self.inner.shot_count(),
        )
    }
}
