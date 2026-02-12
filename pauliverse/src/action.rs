use std::fmt::Debug;

use crate::{
    circuit::{Circuit, SimulationError},
    OutcomeCompleteSimulation, Simulation,
};
use binar::{AffineMap, BitMatrix, BitVec, Bitwise, BitwiseMut, IndexSet};
use paulimer::{clifford::standard_restriction_with_sign_matrix, CliffordUnitary, Pauli, PauliMutable, SparsePauli};

type QubitId = usize;

// ================================================================================================
// Public Types
// ================================================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct CircuitAction {
    /// The observables measured by the circuit, that is Paulis whose measurement outcomes are part of circuit outcomes
    observables: GeneratorsWithSigns,
    /// The stabilizers of the output state of the circuit for all inputs
    stabilizers: GeneratorsWithSigns,
    /// The stabilizers of the choi state of the circuit
    choi_state_stabilizers: GeneratorsWithSigns,
    /// The stabilizers of auxiliary qubits used by the circuit
    auxiliary_stabilizers: GeneratorsWithSigns,
    /// The map from circuit outcomes to inner random bits
    random_from_outcomes: AffineMap,
    /// The map from inner random bits to circuit outcomes
    outcomes_from_random: AffineMap,
}

#[derive(Clone, PartialEq)]
#[must_use]
pub struct SignedPauli {
    pub pauli: SparsePauli,
    /// The sign of pauli is determined by the inner product of `outcomes_sign_mask` and outcome.
    pub outcomes_sign_mask: BitVec,
}

impl Debug for SignedPauli {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(-1)^<{:?},outcome> {}", self.outcomes_sign_mask, self.pauli)
    }
}

#[derive(Debug, derive_more::From)]
pub enum ActionError {
    AuxiliaryQubitsEntangled {
        state_encoder: CliffordUnitary,
        auxiliary_qubits: Vec<QubitId>,
    },
    #[from]
    SimulationFailed(SimulationError),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ActionsInequivalenceReason {
    /// See [`CircuitAction::input_qubits`] for details.
    InputQubitCount,
    /// See [`CircuitAction::output_qubits`] for details.
    OutputQubitCount,
    /// See [`CircuitAction::observables`] for details.
    Observables,
    /// See [`CircuitAction::observables`] for details.
    ObservablesCount,
    /// See [`CircuitAction::signed_observables`] for details.
    ObservablesSigns,
    /// See [`CircuitAction::stabilizers`] for details.
    Stabilizers,
    /// See [`CircuitAction::stabilizers`] for details.
    StabilizersCount,
    /// See [`CircuitAction::signed_stabilizers`] for details.
    StabilizersSigns,
    /// See [`CircuitAction::choi_state_stabilizers`] for details.
    ChoiState,
    /// See [`CircuitAction::signed_choi_state_stabilizers`] for details.
    ChoiStateSigns,
}

/// [`Circuit`]s in pauliverse include fixed number of qubits and do not have prepare and destroy instructions.
/// For this reason, we provide indexes of input and output qubits via `input_qubits` and `output_qubits`.
/// The qubits that are not `output_qubits` at the end of circuit execution are considered auxiliary qubits (see [`CircuitAction::auxiliary_qubits`]).
/// If they are entangled with qubits in the choi state, then action is undefined.
///
/// # Errors
///
/// Returns [`ActionError`] if action calculation fails.
pub fn action_of(
    circuit: &Circuit,
    input_qubits: &[QubitId],
    output_qubits: &[QubitId],
) -> Result<CircuitAction, ActionError> {
    let qubit_count = circuit.qubit_count();
    let reference_qubits: Vec<QubitId> = (qubit_count..qubit_count + input_qubits.len()).collect();
    let outcome_count = circuit.outcome_count();
    let mut simulation =
        OutcomeCompleteSimulation::with_capacity(qubit_count + input_qubits.len(), outcome_count, outcome_count);

    for (input_qubit, reference_qubit) in input_qubits.iter().zip(reference_qubits.iter()) {
        simulation.unitary_op(paulimer::UnitaryOp::PrepareBell, &[*input_qubit, *reference_qubit]);
    }

    circuit.simulate(&mut simulation)?;
    let sign_matrix = simulation.sign_matrix();
    let state_encoder = simulation.state_encoder();

    let auxiliary_qubits: Vec<QubitId> = output_qubits
        .iter()
        .copied()
        .collect::<IndexSet>()
        .complement(qubit_count)
        .into_iter()
        .collect();
    let auxiliary_stabilizers =
        GeneratorsWithSigns::from_restriction(&state_encoder, &sign_matrix, &auxiliary_qubits, false);
    if auxiliary_stabilizers.canonical_generators.len() < auxiliary_qubits.len() {
        return Err(ActionError::AuxiliaryQubitsEntangled {
            state_encoder,
            auxiliary_qubits,
        });
    }

    let observables = GeneratorsWithSigns::from_restriction(&state_encoder, &sign_matrix, &reference_qubits, true);
    let stabilizers = GeneratorsWithSigns::from_restriction(&state_encoder, &sign_matrix, output_qubits, false);
    let choi_state_stabilizers = GeneratorsWithSigns::from_restriction(
        &state_encoder,
        &sign_matrix,
        &(reference_qubits
            .iter()
            .chain(output_qubits.iter())
            .copied()
            .collect::<Vec<_>>()),
        false,
    );

    let indicators = simulation.random_outcome_indicator();
    let random_bit_map_matrix = random_bit_map_matrix(indicators);
    let random_bit_map_shift = &random_bit_map_matrix * &simulation.outcome_shift().as_view();
    let outcome_to_random_bit_map = AffineMap::affine(random_bit_map_matrix.clone(), random_bit_map_shift.clone());
    let outcomes_from_random = AffineMap::affine(simulation.outcome_matrix(), simulation.outcome_shift().clone());

    let action = CircuitAction {
        observables,
        stabilizers,
        choi_state_stabilizers,
        auxiliary_stabilizers,
        random_from_outcomes: outcome_to_random_bit_map,
        outcomes_from_random,
    };
    Ok(action)
}

impl CircuitAction {
    /// Canonical choice of circuit observables, that is Paulis measured by the circuit
    /// Qubits are reindexed to the range `[0, input_qubits.len())` and ordered according to [`CircuitAction::input_qubits`].
    pub fn observables(&self) -> &[SparsePauli] {
        self.observables.abs()
    }

    /// Canonical choice of circuit stabilizers, that is Paulis that stabilize output state of the circuit
    /// for all circuit inputs
    /// Qubits are reindexed to the range `[0, output_qubits.len())` and ordered according to [`CircuitAction::output_qubits`].
    pub fn stabilizers(&self) -> &[SparsePauli] {
        self.stabilizers.abs()
    }

    /// Canonical choice of circuit choi state stabilizers
    /// Qubits are reindexed to the range `[0, input_qubits.len() + output_qubits.len())` and ordered according to [`CircuitAction::input_qubits`]
    /// concatenated with [`CircuitAction::output_qubits`].
    pub fn choi_state_stabilizers(&self) -> &[SparsePauli] {
        self.choi_state_stabilizers.abs()
    }

    /// Returns `Ok(())` if actions are equivalent up to signs, otherwise returns reasons for inequivalence.
    ///
    /// # Errors
    ///
    /// Returns a list of [`ActionsInequivalenceReason`] if the actions differ.
    pub fn is_equivalent_up_to_signs(&self, other: &CircuitAction) -> Result<(), Vec<ActionsInequivalenceReason>> {
        let mut reasons = Vec::new();
        if self.input_qubits().len() != other.input_qubits().len() {
            reasons.push(ActionsInequivalenceReason::InputQubitCount);
        }
        if self.output_qubits().len() != other.output_qubits().len() {
            reasons.push(ActionsInequivalenceReason::OutputQubitCount);
        }
        if !reasons.is_empty() {
            return Err(reasons);
        }

        if self.observables.abs().len() != other.observables.abs().len() {
            reasons.push(ActionsInequivalenceReason::ObservablesCount);
        }
        if self.stabilizers.abs().len() != other.stabilizers.abs().len() {
            reasons.push(ActionsInequivalenceReason::StabilizersCount);
        }
        if !reasons.is_empty() {
            return Err(reasons);
        }

        if self.observables.abs() != other.observables.abs() {
            reasons.push(ActionsInequivalenceReason::Observables);
        }
        if self.stabilizers.abs() != other.stabilizers.abs() {
            reasons.push(ActionsInequivalenceReason::Stabilizers);
        }
        if !reasons.is_empty() {
            return Err(reasons);
        }

        if self.choi_state_stabilizers.abs() != other.choi_state_stabilizers.abs() {
            reasons.push(ActionsInequivalenceReason::ChoiState);
        }
        if reasons.is_empty() {
            Ok(())
        } else {
            Err(reasons)
        }
    }

    /// Check if two actions are equivalent when outcomes are remapped.
    /// Outcomes of self `o_self` = `A(o_other)` where A is `self_outcomes_from_other_outcomes` and `o_other` are outcomes of other.
    /// When map is none, it is assumed that zero map is used, as common for circuits with unitary action.
    ///
    /// # Errors
    ///
    /// Returns a list of [`ActionsInequivalenceReason`] if the actions differ.
    pub fn is_equivalent_with_map(
        &self,
        other: &CircuitAction,
        self_outcomes_from_other_outcomes: Option<&AffineMap>,
    ) -> Result<(), Vec<ActionsInequivalenceReason>> {
        self.is_equivalent_up_to_signs(other)?;

        let zero_map = zero_map(self, other);
        let self_outcomes_from_other_outcomes = self_outcomes_from_other_outcomes.unwrap_or(&zero_map);

        let self_outcomes_from_other_random = self_outcomes_from_other_outcomes.dot(&other.outcomes_from_random);
        let self_random_from_other_random = self.random_from_outcomes.dot(&self_outcomes_from_other_random);

        let mut reasons = Vec::new();

        if self
            .observables
            .is_equivalent_with_map(&other.observables, &self_random_from_other_random)
        {
            reasons.push(ActionsInequivalenceReason::ObservablesSigns);
        }
        if self
            .stabilizers
            .is_equivalent_with_map(&other.stabilizers, &self_random_from_other_random)
        {
            reasons.push(ActionsInequivalenceReason::StabilizersSigns);
        }
        if self
            .choi_state_stabilizers
            .is_equivalent_with_map(&other.choi_state_stabilizers, &self_random_from_other_random)
        {
            reasons.push(ActionsInequivalenceReason::ChoiStateSigns);
        }
        if reasons.is_empty() {
            Ok(())
        } else {
            Err(reasons)
        }
    }

    /// Canonical stabilizers of auxiliary qubits used by the circuit
    pub fn auxiliary_stabilizers(&self) -> &[SparsePauli] {
        self.auxiliary_stabilizers.abs()
    }

    /// Same as [`CircuitAction::observables`] but with signs as a function of circuit outcomes.
    #[must_use]
    pub fn signed_observables(&self) -> Vec<SignedPauli> {
        self.observables.with_transformed_signs(&self.random_from_outcomes)
    }

    /// Same as [`CircuitAction::stabilizers`] but with signs as a function of circuit outcomes.
    #[must_use]
    pub fn signed_stabilizers(&self) -> Vec<SignedPauli> {
        self.stabilizers.with_transformed_signs(&self.random_from_outcomes)
    }

    /// Same as [`CircuitAction::choi_state_stabilizers`] but with signs as a function of circuit outcomes.
    #[must_use]
    pub fn signed_choi_state_stabilizers(&self) -> Vec<SignedPauli> {
        self.choi_state_stabilizers
            .with_transformed_signs(&self.random_from_outcomes)
    }

    /// Same as [`CircuitAction::auxiliary_stabilizers`] but with signs as a function of circuit outcomes.
    #[must_use]
    pub fn signed_auxiliary_stabilizers(&self) -> Vec<SignedPauli> {
        self.auxiliary_stabilizers
            .with_transformed_signs(&self.random_from_outcomes)
    }

    #[must_use]
    pub fn input_qubits(&self) -> &[QubitId] {
        &self.observables.canonical_to_original
    }

    #[must_use]
    pub fn output_qubits(&self) -> &[QubitId] {
        &self.stabilizers.canonical_to_original
    }

    #[must_use]
    pub fn auxiliary_qubits(&self) -> &[QubitId] {
        &self.auxiliary_stabilizers.canonical_to_original
    }

    #[must_use]
    pub fn outcome_count(&self) -> usize {
        self.random_from_outcomes.input_dimension()
    }
}

// ================================================================================================
// Private Types
// ================================================================================================

#[derive(Debug, Clone, PartialEq)]
struct GeneratorsWithSigns {
    /// Canonical choice of generators, with canonical signs
    canonical_generators: Vec<SparsePauli>,
    /// The sign of generator j is `<e_j, A(r)>` where A is `sign_from_random` and r is the vector of inner random bits.
    sign_from_random: AffineMap,
    /// support ids to original circuit qubit ids
    canonical_to_original: Vec<QubitId>,
}

impl GeneratorsWithSigns {
    fn new(canonical_generators: Vec<SparsePauli>, sign_from_random: AffineMap, qubits: &[QubitId]) -> Self {
        assert_eq!(canonical_generators.len(), sign_from_random.output_dimension());

        Self {
            canonical_generators,
            sign_from_random,
            canonical_to_original: qubits.to_vec(),
        }
    }

    fn from_restriction(
        clifford: &CliffordUnitary,
        sign_matrix: &BitMatrix,
        support: &[QubitId],
        conjugate: bool,
    ) -> Self {
        let (mut paulis, random_to_sign_linear) = standard_restriction_with_sign_matrix(clifford, sign_matrix, support);
        let mut random_to_sign_translation = BitVec::zeros(random_to_sign_linear.row_count());
        for (index, pauli) in paulis.iter_mut().enumerate() {
            if conjugate {
                pauli.complex_conjugate();
            }
            let adjusted = adjust_phase_to_canonical(pauli);
            random_to_sign_translation.assign_index(index, adjusted);
        }
        let random_to_sign_bit_map = AffineMap::affine(random_to_sign_linear, random_to_sign_translation);
        Self::new(paulis, random_to_sign_bit_map, support)
    }

    fn abs(&self) -> &[SparsePauli] {
        &self.canonical_generators
    }

    fn with_transformed_signs(&self, random_from_outcomes: &AffineMap) -> Vec<SignedPauli> {
        let sign_from_outcome = self.sign_from_random.dot(random_from_outcomes);
        let mut result = Vec::new();
        for (index, generator) in self.canonical_generators.iter().enumerate() {
            let mut observable = generator.clone();
            if sign_from_outcome.shift().index(index) {
                observable.add_assign_phase_exp(2);
            }
            let outcomes_sign_mask = (&(sign_from_outcome.matrix().row(index))).into();
            result.push(SignedPauli {
                pauli: observable,
                outcomes_sign_mask,
            });
        }
        result
    }

    fn is_equivalent_with_map(&self, other: &GeneratorsWithSigns, self_random_from_other_random: &AffineMap) -> bool {
        self.sign_from_random.dot(self_random_from_other_random) != other.sign_from_random
    }
}

// ================================================================================================
// Helper Functions
// ================================================================================================

fn random_bit_map_matrix(indicators: &[bool]) -> BitMatrix {
    let pivots = indicators.support().collect::<Vec<_>>();
    let mut random_bit_map_matrix = BitMatrix::zeros(pivots.len(), indicators.len());
    for (random_bit_index, pivot) in pivots.iter().enumerate() {
        random_bit_map_matrix.set((random_bit_index, *pivot), true);
    }
    random_bit_map_matrix
}

fn adjust_phase_to_canonical(pauli: &mut SparsePauli) -> bool {
    debug_assert!(pauli.is_order_two());
    if pauli.xyz_phase_exponent() == 0 {
        false
    } else {
        pauli.add_assign_phase_exp(2);
        true
    }
}

fn zero_map(to: &CircuitAction, from: &CircuitAction) -> AffineMap {
    AffineMap::zero(from.outcome_count(), to.outcome_count())
}
