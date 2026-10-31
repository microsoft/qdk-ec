use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use binar::{BitMatrix, BitVec};
use paulimer::clifford::CliffordUnitary;
use pauliverse::action::{phased_action_from_simulation, PhasedCircuitAction};
use pauliverse::outcome_complete_simulation::OutcomeCompleteSimulation;
use pauliverse::outcome_free_simulation::OutcomeFreeSimulation;
use pauliverse::outcome_specific_simulation::OutcomeSpecificSimulation;
use pauliverse::phased_outcome_complete_simulation::PhasedOutcomeCompleteSimulation;
use pauliverse::Simulation;
use pyo3::exceptions::{PyDeprecationWarning, PyNotImplementedError, PyValueError};
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

#[derive(derive_more::Deref, derive_more::DerefMut)]
#[must_use]
#[pyclass(name = "PhasedOutcomeCompleteSimulation", module = "paulimer")]
pub struct PyPhasedOutcomeCompleteSimulation {
    #[deref]
    #[deref_mut]
    inner: PhasedOutcomeCompleteSimulation,
    angle_owner: Arc<()>,
}

impl From<PhasedOutcomeCompleteSimulation> for PyPhasedOutcomeCompleteSimulation {
    fn from(inner: PhasedOutcomeCompleteSimulation) -> Self {
        Self {
            inner,
            angle_owner: Arc::new(()),
        }
    }
}

/// An opaque handle to a symbolic angle `α` of a parameterised circuit.
///
/// A symbolic angle is the free parameter of a Pauli exponent `e^{iα P}`. Obtain one from
/// `PhasedOutcomeCompleteSimulation.allocate_symbolic_angle` (or a batch from
/// `allocate_symbolic_angles`) and pass it to `apply_symbolic_pauli_exp`. The handle is opaque:
/// its only observable feature is its `index`, the angle's subscript `k` in `α_k`, fixed by the
/// order in which angles are allocated. When two circuits are compared with `phased_action`, angles
/// with the same `index` are required to correspond, so describing both circuits in terms of the
/// `k`-th angle is what makes the comparison meaningful -- regardless of how the rest of each
/// circuit is written.
///
/// Handles compare equal only when they identify the same angle in the same simulation.
#[derive(Clone)]
#[pyclass(name = "SymbolicAngle", module = "paulimer", frozen)]
pub struct PySymbolicAngle {
    outcome: usize,
    index: usize,
    owner: Arc<()>,
}

#[pymethods]
impl PySymbolicAngle {
    /// The subscript `k` identifying this angle as `α_k`, set by allocation order.
    #[getter]
    #[must_use]
    pub fn index(&self) -> usize {
        self.index
    }

    #[must_use]
    pub fn __repr__(&self) -> String {
        format!("SymbolicAngle(index={})", self.index)
    }

    #[must_use]
    pub fn __eq__(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner) && self.outcome == other.outcome
    }

    #[must_use]
    pub fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        Arc::as_ptr(&self.owner).hash(&mut hasher);
        self.outcome.hash(&mut hasher);
        hasher.finish()
    }
}

fn warn_renamed(py: Python<'_>, old_name: &str, new_name: &str) -> PyResult<()> {
    py.import("warnings")?.call_method1(
        "warn",
        (
            format!("{old_name} is deprecated. Use {new_name} instead."),
            py.get_type::<PyDeprecationWarning>(),
            1,
        ),
    )?;
    Ok(())
}

macro_rules! impl_simulation {
    ($struct_name:ty, $wrapper_struct:ty, clifford_supported = $clifford_supported:literal { $($inside:tt)* }) => {
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
            pub fn branch_bit_count(&self) -> usize {
                self.deref().branch_bit_count()
            }

            #[getter]
            #[must_use]
            pub fn branch_bit_capacity(&self) -> usize {
                self.deref().branch_bit_capacity()
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

            #[allow(clippy::needless_pass_by_value, clippy::missing_errors_doc)]
            #[pyo3(signature=(clifford, supported_by=None))]
            pub fn apply_clifford(&mut self, clifford: &PyCliffordUnitary, supported_by: Option<Vec<usize>>) -> PyResult<()> {
                if !$clifford_supported {
                    return Err(PyNotImplementedError::new_err(
                        "this simulator tracks the exact global phase, which a phaseless CliffordUnitary \
                         does not determine; apply Cliffords through apply_unitary, apply_pauli, or \
                         apply_pauli_exp instead",
                    ));
                }
                let support = supported_by.unwrap_or_else(|| (0..self.deref().qubit_count()).collect());
                Simulation::clifford(self.deref_mut(), &clifford.inner, &support);
                Ok(())
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

            /// Reserve capacity. The keyword name remains compatible with earlier releases.
            pub fn reserve_outcomes(&mut self, new_outcome_capacity: usize, new_random_outcome_capacity: usize) {
                Simulation::reserve_outcomes(self.deref_mut(), new_outcome_capacity, new_random_outcome_capacity);
            }

            #[allow(clippy::needless_pass_by_value)]
            #[pyo3(signature=(observable, ignore_sign=false, sign_parity=Vec::new()))]
            #[must_use]
            pub fn is_stabilizer(&self, observable: &PySparsePauli, ignore_sign: bool, sign_parity: Vec<usize>) -> bool {
                if ignore_sign {
                    Simulation::is_stabilizer_up_to_sign(self.deref(), &observable.inner)
                } else {
                    if !sign_parity.is_empty() {
                        Simulation::is_stabilizer_with_conditional_sign(self.deref(), &observable.inner, &sign_parity)
                    } else {
                        Simulation::is_stabilizer(self.deref(), &observable.inner)
                    }
                }
            }

            /// Construct a simulation. The keyword names remain compatible with earlier releases.
            #[staticmethod]
            #[pyo3(signature=(num_qubits, num_outcomes, num_random_outcomes))]
            pub fn with_capacity(num_qubits: usize, num_outcomes: usize, num_random_outcomes: usize) -> Self {
                <$struct_name>::with_capacity(num_qubits, num_outcomes, num_random_outcomes).into()
            }

            #[getter]
            pub fn outcome_introduces_branch_bit(&self) -> BitVec {
                self.deref().outcome_introduces_branch_bit().iter().copied().collect::<BitVec>()
            }


            /// Deprecated alias for `branch_bit_count`.
            ///
            /// # Errors
            ///
            /// Returns an error if Python rejects the deprecation warning.
            #[getter]
            pub fn random_outcome_count(&self, py: Python<'_>) -> PyResult<usize> {
                warn_renamed(py, "random_outcome_count", "branch_bit_count")?;
                Ok(self.deref().branch_bit_count())
            }

            /// Deprecated alias for `branch_bit_count`.
            ///
            /// # Errors
            ///
            /// Returns an error if Python rejects the deprecation warning.
            #[getter]
            pub fn random_bit_count(&self, py: Python<'_>) -> PyResult<usize> {
                warn_renamed(py, "random_bit_count", "branch_bit_count")?;
                Ok(self.deref().branch_bit_count())
            }

            /// Deprecated alias for `branch_bit_capacity`.
            ///
            /// # Errors
            ///
            /// Returns an error if Python rejects the deprecation warning.
            #[getter]
            pub fn random_outcome_capacity(&self, py: Python<'_>) -> PyResult<usize> {
                warn_renamed(py, "random_outcome_capacity", "branch_bit_capacity")?;
                Ok(self.deref().branch_bit_capacity())
            }

            /// Deprecated alias for `outcome_introduces_branch_bit`.
            ///
            /// # Errors
            ///
            /// Returns an error if Python rejects the deprecation warning.
            #[getter]
            pub fn random_outcome_indicator(&self, py: Python<'_>) -> PyResult<BitVec> {
                warn_renamed(py, "random_outcome_indicator", "outcome_introduces_branch_bit")?;
                Ok(self.outcome_introduces_branch_bit())
            }

            $($inside)*
        }
    };
}

impl_simulation!(
    OutcomeCompleteSimulation,
    PyOutcomeCompleteSimulation,
    clifford_supported = true {
        #[getter]
        pub fn clifford(&self) -> PyCliffordUnitary {
            PyCliffordUnitary {
                inner: self.deref().state_encoder().clone(),
            }
        }

        #[getter]
        pub fn sign_matrix(&self) -> BitMatrix {
            self.inner.sign_matrix()
        }

        #[getter]
        pub fn outcome_matrix(&self) -> BitMatrix {
            self.inner.outcome_matrix()
        }

        #[getter]
        pub fn outcome_shift(&self) -> BitVec {
            self.inner.outcome_shift()
        }
});

impl_simulation!(
    OutcomeFreeSimulation,
    PyOutcomeFreeSimulation,
    clifford_supported = true {
        #[getter]
        pub fn clifford(&self) -> PyCliffordUnitary {
            let c: CliffordUnitary = self.deref().state_encoder().clone().into();
            PyCliffordUnitary { inner: c }
        }
});

impl_simulation!(
    OutcomeSpecificSimulation,
    PyOutcomeSpecificSimulation,
    clifford_supported = true {
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
        pub fn with_zero_samples(num_qubits: usize) -> Self {
            OutcomeSpecificSimulation::new_with_zero_samples(num_qubits).into()
        }

        #[staticmethod]
        #[pyo3(signature=(num_qubits, seed = 0))]
        pub fn with_seeded_random_samples(num_qubits: usize, seed: u64) -> Self {
            OutcomeSpecificSimulation::new_with_seeded_random_samples(num_qubits, seed).into()
        }

        #[staticmethod]
        #[pyo3(signature=(num_qubits))]
        pub fn with_random_samples(num_qubits: usize) -> Self {
            OutcomeSpecificSimulation::new_with_random_samples(num_qubits).into()
        }

        /// Deprecated alias for `with_zero_samples`.
        ///
        /// # Errors
        ///
        /// Returns an error if Python rejects the deprecation warning.
        #[staticmethod]
        pub fn with_zero_outcomes(py: Python<'_>, num_qubits: usize) -> PyResult<Self> {
            warn_renamed(py, "with_zero_outcomes", "with_zero_samples")?;
            Ok(Self::with_zero_samples(num_qubits))
        }

        /// Deprecated alias for `with_seeded_random_samples`.
        ///
        /// # Errors
        ///
        /// Returns an error if Python rejects the deprecation warning.
        #[staticmethod]
        #[pyo3(signature=(num_qubits, seed=0))]
        pub fn new_with_seeded_random_outcomes(py: Python<'_>, num_qubits: usize, seed: u64) -> PyResult<Self> {
            warn_renamed(py, "new_with_seeded_random_outcomes", "with_seeded_random_samples")?;
            Ok(Self::with_seeded_random_samples(num_qubits, seed))
        }
});

impl_simulation!(
    PhasedOutcomeCompleteSimulation,
    PyPhasedOutcomeCompleteSimulation,
    clifford_supported = false {
        #[getter]
        pub fn clifford(&self) -> PyCliffordUnitary {
            PyCliffordUnitary {
                inner: self.deref().state_encoder(),
            }
        }

        #[getter]
        pub fn sign_matrix(&self) -> BitMatrix {
            self.inner.sign_matrix()
        }

        #[getter]
        pub fn quadratic_phase_matrix(&self) -> BitMatrix {
            self.inner.quadratic_phase_matrix()
        }

        #[getter]
        pub fn outcome_matrix(&self) -> BitMatrix {
            self.inner.outcome_matrix()
        }

        #[getter]
        pub fn outcome_shift(&self) -> BitVec {
            self.inner.outcome_shift()
        }

        #[getter]
        pub fn linear_i_phase(&self) -> BitVec {
            self.inner.linear_i_phase()
        }

        #[getter]
        pub fn linear_sign_phase(&self) -> BitVec {
            self.inner.linear_sign_phase()
        }

        /// Return the scalar phase exponent for an independent branch assignment.
        ///
        /// # Errors
        ///
        /// Returns a `ValueError` if `random_bits` has fewer entries than `branch_bit_count`.
        /// Entries past `branch_bit_count` are ignored.
        #[allow(clippy::needless_pass_by_value)]
        pub fn output_phase_exponent(&self, random_bits: Vec<bool>) -> PyResult<u8> {
            let branch_bit_count = self.inner.branch_bit_count();
            if random_bits.len() < branch_bit_count {
                return Err(PyValueError::new_err(format!(
                    "random_bits has length {}, but the simulation has {branch_bit_count} branch bits",
                    random_bits.len()
                )));
            }
            Ok(self.inner.output_phase_exponent(&random_bits))
        }

        /// Allocate a fresh symbolic angle `α`.
        ///
        /// Returns an opaque `SymbolicAngle` handle; pass it to `apply_symbolic_pauli_exp` to
        /// apply `e^{iα P}`. Angles are numbered by allocation order (the returned handle's
        /// `index`), and when two circuits are compared with `phased_action` the angle with a given
        /// `index` in one must correspond to the same `index` in the other. To allocate several at
        /// once, use `allocate_symbolic_angles`.
        pub fn allocate_symbolic_angle(&mut self) -> PySymbolicAngle {
            let index = self.inner.symbolic_angle_indicator().iter().filter(|&&is_angle| is_angle).count();
            let outcome = self.inner.allocate_symbolic_angle();
            PySymbolicAngle {
                outcome,
                index,
                owner: Arc::clone(&self.angle_owner),
            }
        }

        /// Allocate `count` fresh symbolic angles `α_0, ..., α_{count-1}` at once.
        ///
        /// Returns the `SymbolicAngle` handles in order, so `angles[k]` is `α_k`. Allocating all of
        /// a circuit's angles up front and then referring to them by index keeps the correspondence
        /// between two circuits explicit and independent of how either circuit is otherwise written.
        pub fn allocate_symbolic_angles(&mut self, count: usize) -> Vec<PySymbolicAngle> {
            (0..count).map(|_| self.allocate_symbolic_angle()).collect()
        }

        /// All symbolic angles allocated on this simulation so far, in order (`angles[k]` is `α_k`).
        #[getter]
        #[must_use]
        pub fn symbolic_angles(&self) -> Vec<PySymbolicAngle> {
            let symbolic_angles = self.inner.symbolic_angle_indicator();
            self.inner.outcome_introduces_branch_bit()
                .iter()
                .enumerate()
                .filter(|(_, introduces_branch)| **introduces_branch)
                .zip(symbolic_angles)
                .filter(|(_, is_angle)| **is_angle)
                .enumerate()
                .map(|(index, ((outcome, _), _))| PySymbolicAngle {
                    outcome,
                    index,
                    owner: Arc::clone(&self.angle_owner),
                })
                .collect()
        }

        /// Apply a symbolic Pauli exponent `e^{iα P}` parameterised by `angle`.
        ///
        /// `angle` must be a `SymbolicAngle` obtained from `allocate_symbolic_angle` or
        /// `allocate_symbolic_angles`. This is the high-level way to add a free-angle exponent
        /// `e^{iα P}` for an arbitrary Pauli `P`. Each angle must parameterise exactly one exponent.
        /// This initial simulation layer does not enforce that restriction when building an action.
        /// Angles with matching `index` in two circuits correspond when their phased actions are compared.
        ///
        /// # Errors
        ///
        /// Returns a `ValueError` if `angle` comes from another simulation.
        #[allow(clippy::needless_pass_by_value)]
        pub fn apply_symbolic_pauli_exp(&mut self, observable: &PySparsePauli, angle: &PySymbolicAngle) -> PyResult<()> {
            if !Arc::ptr_eq(&self.angle_owner, &angle.owner) {
                return Err(PyValueError::new_err(format!(
                    "symbolic angle {} belongs to another simulation",
                    angle.index
                )));
            }
            self.inner.symbolic_pauli_exp(&observable.inner, angle.outcome);
            Ok(())
        }

        #[allow(clippy::needless_pass_by_value)]
        /// # Errors
        ///
        /// Returns a `ValueError` if the non-output system qubits remain entangled.
        pub fn phased_action(
            &self,
            input_qubits: Vec<usize>,
            output_qubits: Vec<usize>,
        ) -> PyResult<PyPhasedCircuitAction> {
            phased_action_from_simulation(&self.inner, &input_qubits, &output_qubits)
                .map(|action| PyPhasedCircuitAction { inner: action })
                .map_err(|error| PyValueError::new_err(format!("{error:?}")))
        }
});

#[derive(derive_more::From)]
#[must_use]
#[pyclass(name = "PhasedCircuitAction", module = "paulimer")]
pub struct PyPhasedCircuitAction {
    inner: PhasedCircuitAction,
}

#[pymethods]
impl PyPhasedCircuitAction {
    #[getter]
    #[must_use]
    pub fn choi_state_stabilizers(&self) -> Vec<PySparsePauli> {
        self.inner
            .choi_state_stabilizers()
            .iter()
            .map(|pauli| PySparsePauli { inner: pauli.clone() })
            .collect()
    }

    #[must_use]
    pub fn is_equivalent(&self, other: &PyPhasedCircuitAction) -> bool {
        self.inner.is_equivalent(&other.inner).is_ok()
    }

    #[must_use]
    pub fn is_equivalent_up_to_signs(&self, other: &PyPhasedCircuitAction) -> bool {
        self.inner.is_equivalent_up_to_signs(&other.inner).is_ok()
    }
}
