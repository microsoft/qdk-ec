mod phased;
pub use phased::{PhasedCircuitAction, phased_action_from_simulation, phased_action_of};

use std::fmt::Debug;

use crate::{
    OutcomeCompleteSimulation, PhasedOutcomeCompleteSimulation, Simulation,
    circuit::{Circuit, SimulationError},
};
use binar::{AffineMap, BitMatrix, BitVec, Bitwise, BitwiseMut, IndexSet};
use paulimer::{CliffordUnitary, Pauli, PauliMutable, SparsePauli, clifford::standard_restriction_with_sign_matrix};

type QubitId = crate::circuit::QubitId;

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
    /// The map from circuit outcomes to branch bits
    branch_bits_from_outcomes: AffineMap,
    /// The map from branch bits to circuit outcomes
    outcomes_from_branch_bits: AffineMap,
    /// The caller-supplied input qubit IDs
    input_qubit_ids: Vec<QubitId>,
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

impl SignedPauli {
    #[must_use]
    pub fn sign_support(&self) -> Vec<usize> {
        self.outcomes_sign_mask.support().collect()
    }
}

#[derive(Debug, derive_more::From)]
pub enum ActionError {
    AuxiliaryQubitsEntangled {
        state_encoder: CliffordUnitary,
        auxiliary_qubits: Vec<QubitId>,
    },
    /// A symbolic angle parameterises more than one rotation, so the recorded action does not
    /// determine the operator. `angle` is the outcome id returned by
    /// [`Simulation::allocate_symbolic_angle`].
    /// See [`PhasedOutcomeCompleteSimulation::reused_symbolic_angle`].
    SymbolicAngleReused { angle: usize },
    /// A symbolic angle parameterises no rotation. Its branch bit would encode the global phase
    /// `exp(i alpha)` rather than the identity. `angle` is the outcome id returned by
    /// [`Simulation::allocate_symbolic_angle`].
    /// See [`PhasedOutcomeCompleteSimulation::unused_symbolic_angle`].
    SymbolicAngleUnused { angle: usize },
    /// A discarded auxiliary qubit carries a stabilizer sign that depends on a symbolic angle.
    /// Discarding it would decohere that angle, so the circuit has no phased action. `angle` is
    /// the outcome id returned by [`Simulation::allocate_symbolic_angle`].
    AuxiliaryQubitsCarrySymbolicAngle { angle: usize },
    /// The qubits do not describe a Choi-state layout. Either `input_qubits` or `output_qubits`
    /// names a qubit twice, or the simulation does not hold every system and reference qubit.
    InvalidQubits {
        input_qubits: Vec<QubitId>,
        output_qubits: Vec<QubitId>,
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
    /// The relative `ζ₈` phases between branches of the Choi state differ.
    /// Only produced by [`PhasedCircuitAction`]; see its documentation for details.
    RelativePhase,
    /// The two phased actions have different numbers of symbolic-angle (virtual) random bits, so no
    /// one-to-one correspondence between their symbolic rotations exists.
    /// Only produced by [`PhasedCircuitAction`]; see its documentation for details.
    SymbolicAngleCount,
    /// A supplied outcome remapping would affinely mix symbolic-angle (virtual) random bits, either
    /// with one another or with true (measurement) random bits, which does not correspond to any
    /// operator equality. Only produced by [`PhasedCircuitAction::is_equivalent_with_map`].
    SymbolicAngleMixed,
    /// The supplied correspondence omits a nonredundant outcome or selects an impossible record of `self`.
    RandomOutcomeCoverage,
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
    build_action::<OutcomeCompleteSimulation>(circuit, input_qubits, output_qubits).map(|(action, _)| action)
}

/// Stabilizer simulators that expose the encoder data required to compute a [`CircuitAction`].
trait ActionSimulation: Simulation {
    fn encoder(&self) -> CliffordUnitary;
    fn signs(&self) -> BitMatrix;
    fn branch_bit_count(&self) -> usize;
    fn branch_bit_locations(&self) -> impl Iterator<Item = crate::outcome_complete_simulation::BranchBitLocation> + '_;
    fn outcomes(&self) -> BitMatrix;
    fn outcome_offset(&self) -> BitVec;
}

impl ActionSimulation for OutcomeCompleteSimulation {
    fn encoder(&self) -> CliffordUnitary {
        self.state_encoder()
    }
    fn signs(&self) -> BitMatrix {
        self.sign_matrix()
    }
    fn branch_bit_count(&self) -> usize {
        self.branch_bit_count()
    }
    fn branch_bit_locations(&self) -> impl Iterator<Item = crate::outcome_complete_simulation::BranchBitLocation> + '_ {
        self.branch_bit_locations()
    }
    fn outcomes(&self) -> BitMatrix {
        self.outcome_matrix()
    }
    fn outcome_offset(&self) -> BitVec {
        self.outcome_shift()
    }
}

impl ActionSimulation for PhasedOutcomeCompleteSimulation {
    fn encoder(&self) -> CliffordUnitary {
        self.state_encoder()
    }
    fn signs(&self) -> BitMatrix {
        self.sign_matrix()
    }
    fn branch_bit_count(&self) -> usize {
        self.branch_bit_count()
    }
    fn branch_bit_locations(&self) -> impl Iterator<Item = crate::outcome_complete_simulation::BranchBitLocation> + '_ {
        self.branch_bit_locations()
    }
    fn outcomes(&self) -> BitMatrix {
        self.outcome_matrix()
    }
    fn outcome_offset(&self) -> BitVec {
        self.outcome_shift()
    }
}

/// Computes a [`CircuitAction`] using simulator `S`, returning both the action and the consumed
/// simulator so that phase-aware callers can additionally read out its phase data.
fn build_action<S: ActionSimulation>(
    circuit: &Circuit,
    input_qubits: &[QubitId],
    output_qubits: &[QubitId],
) -> Result<(CircuitAction, S), ActionError> {
    let qubit_count = circuit
        .qubit_count()
        .max(input_qubits.iter().max().map_or(0, |&q| q + 1))
        .max(output_qubits.iter().max().map_or(0, |&q| q + 1));
    let reference_qubits: Vec<QubitId> = (qubit_count..qubit_count + input_qubits.len()).collect();
    let outcome_count = circuit.outcome_count();
    let mut simulation = S::with_capacity(qubit_count + input_qubits.len(), outcome_count, outcome_count);

    for (input_qubit, reference_qubit) in input_qubits.iter().zip(reference_qubits.iter()) {
        simulation.unitary_op(paulimer::UnitaryOp::PrepareBell, &[*input_qubit, *reference_qubit]);
    }

    circuit.simulate(&mut simulation)?;
    let action = action_from_simulation(&simulation, input_qubits, output_qubits, &reference_qubits, qubit_count)?;
    Ok((action, simulation))
}

/// Canonicalizes the Choi state recorded in `simulation` into a [`CircuitAction`].
///
/// This is the post-simulation core shared by [`build_action`] (which prepares the Bell pairs and
/// replays a [`Circuit`]) and [`phased_action_from_simulation`] (which canonicalizes a Choi state the
/// caller has already prepared). The caller is responsible for having entangled `input_qubits[k]`
/// with `reference_qubits[k]` via a Bell pair before applying the circuit.
fn action_from_simulation<S: ActionSimulation>(
    simulation: &S,
    input_qubits: &[QubitId],
    output_qubits: &[QubitId],
    reference_qubits: &[QubitId],
    qubit_count: usize,
) -> Result<CircuitAction, ActionError> {
    let sign_matrix = simulation.signs();
    let state_encoder = simulation.encoder();

    // Auxiliary qubits are every qubit the simulation holds that is neither an output nor a
    // reference. Taking the total from the simulation, rather than from `qubit_count`, covers
    // qubits the caller allocated beyond the named system block.
    let auxiliary_qubits: Vec<QubitId> = output_qubits
        .iter()
        .chain(reference_qubits.iter())
        .copied()
        .collect::<IndexSet>()
        .complement(qubit_count.max(simulation.qubit_count()))
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

    let observables = GeneratorsWithSigns::from_restriction(&state_encoder, &sign_matrix, reference_qubits, true);
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

    let branch_bit_map_matrix = branch_bit_map_matrix(simulation);
    let branch_bit_map_shift = &branch_bit_map_matrix * &simulation.outcome_offset().as_view();
    let outcome_to_branch_bit_map = AffineMap::affine(branch_bit_map_matrix.clone(), branch_bit_map_shift.clone());
    let outcomes_from_branch_bits = AffineMap::affine(simulation.outcomes(), simulation.outcome_offset().clone());

    Ok(CircuitAction {
        observables,
        stabilizers,
        choi_state_stabilizers,
        auxiliary_stabilizers,
        branch_bits_from_outcomes: outcome_to_branch_bit_map,
        outcomes_from_branch_bits,
        input_qubit_ids: input_qubits.to_vec(),
    })
}

impl CircuitAction {
    /// Canonical choice of circuit observables, that is Paulis measured by the circuit.
    /// Qubits are reindexed to the range `[0, input_qubits.len())` where the k-th qubit corresponds
    /// to the k-th entry of [`CircuitAction::input_qubits`].
    #[must_use]
    pub fn observables(&self) -> &[SparsePauli] {
        self.observables.abs()
    }

    /// Canonical choice of circuit stabilizers, that is Paulis that stabilize output state of the circuit
    /// for all circuit inputs.
    /// Qubits are reindexed to the range `[0, output_qubits.len())` where the k-th qubit corresponds
    /// to the k-th entry of [`CircuitAction::output_qubits`].
    #[must_use]
    pub fn stabilizers(&self) -> &[SparsePauli] {
        self.stabilizers.abs()
    }

    /// Canonical choice of circuit choi state stabilizers.
    /// Qubits are reindexed to the range `[0, input_qubits.len() + output_qubits.len())` where the first
    /// `input_qubits.len()` qubits correspond positionally to [`CircuitAction::input_qubits`]
    /// and the remaining qubits correspond positionally to [`CircuitAction::output_qubits`].
    #[must_use]
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
        if reasons.is_empty() { Ok(()) } else { Err(reasons) }
    }

    /// Check if two actions are equivalent when outcomes are remapped.
    /// Outcomes of self `o_self` = `A(o_other)` where A is `self_outcomes_from_other_outcomes` and `o_other` are outcomes of other.
    /// When map is none, it is assumed that zero map is used, as common for circuits with unitary action.
    /// The map must send possible records to possible records and cover every distinct operation of
    /// `self`. Proportional outcome operations may be compressed, preserving their summed probabilities.
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

        let self_branch_bits_from_other_branch_bits =
            self.branch_map_with_outcome_map(other, self_outcomes_from_other_outcomes)?;

        let mut reasons = self.sign_reasons_with_branch_map(other, &self_branch_bits_from_other_branch_bits);
        let signs = self.choi_state_stabilizers.sign_from_branch_bits.matrix();
        if (signs * self_branch_bits_from_other_branch_bits.matrix()).rank() != signs.rank() {
            reasons.push(ActionsInequivalenceReason::RandomOutcomeCoverage);
        }
        if reasons.is_empty() { Ok(()) } else { Err(reasons) }
    }

    /// Canonical stabilizers of auxiliary qubits used by the circuit
    #[must_use]
    pub fn auxiliary_stabilizers(&self) -> &[SparsePauli] {
        self.auxiliary_stabilizers.abs()
    }

    /// Same as [`CircuitAction::observables`] but with signs as a function of circuit outcomes.
    #[must_use]
    pub fn signed_observables(&self) -> Vec<SignedPauli> {
        self.observables.with_transformed_signs(&self.branch_bits_from_outcomes)
    }

    /// Same as [`CircuitAction::stabilizers`] but with signs as a function of circuit outcomes.
    #[must_use]
    pub fn signed_stabilizers(&self) -> Vec<SignedPauli> {
        self.stabilizers.with_transformed_signs(&self.branch_bits_from_outcomes)
    }

    /// Same as [`CircuitAction::choi_state_stabilizers`] but with signs as a function of circuit outcomes.
    #[must_use]
    pub fn signed_choi_state_stabilizers(&self) -> Vec<SignedPauli> {
        self.choi_state_stabilizers
            .with_transformed_signs(&self.branch_bits_from_outcomes)
    }

    /// Same as [`CircuitAction::auxiliary_stabilizers`] but with signs as a function of circuit outcomes.
    #[must_use]
    pub fn signed_auxiliary_stabilizers(&self) -> Vec<SignedPauli> {
        self.auxiliary_stabilizers
            .with_transformed_signs(&self.branch_bits_from_outcomes)
    }

    #[must_use]
    pub fn input_qubits(&self) -> &[QubitId] {
        &self.input_qubit_ids
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
        self.branch_bits_from_outcomes.input_dimension()
    }

    fn branch_map_with_outcome_map(
        &self,
        other: &CircuitAction,
        outcome_map: Option<&AffineMap>,
    ) -> Result<AffineMap, Vec<ActionsInequivalenceReason>> {
        let zero = zero_map(self, other);
        let outcome_map = outcome_map.unwrap_or(&zero);
        if outcome_map.input_dimension() != other.outcome_count()
            || outcome_map.output_dimension() != self.outcome_count()
        {
            return Err(vec![ActionsInequivalenceReason::RandomOutcomeCoverage]);
        }
        let outcomes_from_other_branch_bits = outcome_map.dot(&other.outcomes_from_branch_bits);
        let branch_map = self.branch_bits_from_outcomes.dot(&outcomes_from_other_branch_bits);
        if self.outcomes_from_branch_bits.dot(&branch_map) != outcomes_from_other_branch_bits {
            return Err(vec![ActionsInequivalenceReason::RandomOutcomeCoverage]);
        }
        Ok(branch_map)
    }

    fn sign_reasons_with_branch_map(
        &self,
        other: &CircuitAction,
        self_branch_bits_from_other_branch_bits: &AffineMap,
    ) -> Vec<ActionsInequivalenceReason> {
        let mut reasons = Vec::new();
        if self
            .observables
            .signs_differ_with_map(&other.observables, self_branch_bits_from_other_branch_bits)
        {
            reasons.push(ActionsInequivalenceReason::ObservablesSigns);
        }
        if self
            .stabilizers
            .signs_differ_with_map(&other.stabilizers, self_branch_bits_from_other_branch_bits)
        {
            reasons.push(ActionsInequivalenceReason::StabilizersSigns);
        }
        if self
            .choi_state_stabilizers
            .signs_differ_with_map(&other.choi_state_stabilizers, self_branch_bits_from_other_branch_bits)
        {
            reasons.push(ActionsInequivalenceReason::ChoiStateSigns);
        }
        reasons
    }
}

#[derive(Debug, Clone, PartialEq)]
struct GeneratorsWithSigns {
    /// Canonical choice of generators, with canonical signs
    canonical_generators: Vec<SparsePauli>,
    /// The sign of generator j is `<e_j, A(r)>` where A is `sign_from_branch_bits` and r is the vector of branch bits.
    sign_from_branch_bits: AffineMap,
    /// support ids to original circuit qubit ids
    canonical_to_original: Vec<QubitId>,
}

impl GeneratorsWithSigns {
    fn new(canonical_generators: Vec<SparsePauli>, sign_from_branch_bits: AffineMap, qubits: &[QubitId]) -> Self {
        assert_eq!(canonical_generators.len(), sign_from_branch_bits.output_dimension());

        Self {
            canonical_generators,
            sign_from_branch_bits,
            canonical_to_original: qubits.to_vec(),
        }
    }

    fn from_restriction(
        clifford: &CliffordUnitary,
        sign_matrix: &BitMatrix,
        support: &[QubitId],
        conjugate: bool,
    ) -> Self {
        let (mut paulis, branch_bits_to_sign_linear) =
            standard_restriction_with_sign_matrix(clifford, sign_matrix, support);
        let mut branch_bits_to_sign_translation = BitVec::zeros(branch_bits_to_sign_linear.row_count());
        for (index, pauli) in paulis.iter_mut().enumerate() {
            if conjugate {
                pauli.complex_conjugate();
            }
            let adjusted = adjust_phase_to_canonical(pauli);
            branch_bits_to_sign_translation.assign_index(index, adjusted);
        }
        let branch_bits_to_sign_bit_map =
            AffineMap::affine(branch_bits_to_sign_linear, branch_bits_to_sign_translation);
        Self::new(paulis, branch_bits_to_sign_bit_map, support)
    }

    fn abs(&self) -> &[SparsePauli] {
        &self.canonical_generators
    }

    fn with_transformed_signs(&self, branch_bits_from_outcomes: &AffineMap) -> Vec<SignedPauli> {
        let sign_from_outcome = self.sign_from_branch_bits.dot(branch_bits_from_outcomes);
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

    fn signs_differ_with_map(
        &self,
        other: &GeneratorsWithSigns,
        self_branch_bits_from_other_branch_bits: &AffineMap,
    ) -> bool {
        self.sign_from_branch_bits.dot(self_branch_bits_from_other_branch_bits) != other.sign_from_branch_bits
    }
}

// ================================================================================================
// Helper Functions
// ================================================================================================

fn branch_bit_map_matrix(simulation: &impl ActionSimulation) -> BitMatrix {
    let mut matrix = BitMatrix::zeros(simulation.branch_bit_count(), simulation.outcome_count());
    for location in simulation.branch_bit_locations() {
        matrix.set((location.branch_bit_index, location.outcome_id), true);
    }
    matrix
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

/// Returns the length-`dimension` bit vector with the bits in `set_indices` set to one.
fn unit_vector(dimension: usize, set_indices: &[usize]) -> BitVec {
    let mut vector = BitVec::zeros(dimension);
    for &index in set_indices {
        vector.assign_index(index, true);
    }
    vector
}
