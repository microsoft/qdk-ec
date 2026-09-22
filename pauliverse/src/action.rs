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
    /// The map from circuit outcomes to inner random bits
    random_from_outcomes: AffineMap,
    /// The map from inner random bits to circuit outcomes
    outcomes_from_random: AffineMap,
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
///
/// The method names differ from the inherent accessors of the same purpose to avoid shadowing them
/// inside the forwarding implementations.
trait ActionSimulation: Simulation {
    fn encoder(&self) -> CliffordUnitary;
    fn signs(&self) -> BitMatrix;
    fn random_indicator(&self) -> &[bool];
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
    fn random_indicator(&self) -> &[bool] {
        self.random_outcome_indicator()
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
    fn random_indicator(&self) -> &[bool] {
        self.random_outcome_indicator()
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

    let indicators = simulation.random_indicator();
    let random_bit_map_matrix = random_bit_map_matrix(indicators);
    let random_bit_map_shift = &random_bit_map_matrix * &simulation.outcome_offset().as_view();
    let outcome_to_random_bit_map = AffineMap::affine(random_bit_map_matrix.clone(), random_bit_map_shift.clone());
    let outcomes_from_random = AffineMap::affine(simulation.outcomes(), simulation.outcome_offset().clone());

    Ok(CircuitAction {
        observables,
        stabilizers,
        choi_state_stabilizers,
        auxiliary_stabilizers,
        random_from_outcomes: outcome_to_random_bit_map,
        outcomes_from_random,
        input_qubit_ids: input_qubits.to_vec(),
    })
}

impl CircuitAction {
    /// Canonical choice of circuit observables, that is Paulis measured by the circuit.
    /// Qubits are reindexed to the range `[0, input_qubits.len())` where the k-th qubit corresponds
    /// to the k-th entry of [`CircuitAction::input_qubits`].
    pub fn observables(&self) -> &[SparsePauli] {
        self.observables.abs()
    }

    /// Canonical choice of circuit stabilizers, that is Paulis that stabilize output state of the circuit
    /// for all circuit inputs.
    /// Qubits are reindexed to the range `[0, output_qubits.len())` where the k-th qubit corresponds
    /// to the k-th entry of [`CircuitAction::output_qubits`].
    pub fn stabilizers(&self) -> &[SparsePauli] {
        self.stabilizers.abs()
    }

    /// Canonical choice of circuit choi state stabilizers.
    /// Qubits are reindexed to the range `[0, input_qubits.len() + output_qubits.len())` where the first
    /// `input_qubits.len()` qubits correspond positionally to [`CircuitAction::input_qubits`]
    /// and the remaining qubits correspond positionally to [`CircuitAction::output_qubits`].
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
        if reasons.is_empty() { Ok(()) } else { Err(reasons) }
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
        self.random_from_outcomes.input_dimension()
    }
}

/// The exact-global-phase analog of [`CircuitAction`], computed with a
/// [`PhasedOutcomeCompleteSimulation`] so that the **relative `ζ₈` phases between branches** of the
/// circuit's Choi state are retained in addition to the phaseless stabilizer data.
///
/// A [`CircuitAction`] determines the Choi state only up to phase, so it cannot distinguish circuits
/// that act identically on the Pauli group but differ by branch-dependent phases — for example
/// `e^{iα Z}` and `e^{-iα Z}`, whose conditioned Paulis `+Z` and `-Z` share a symplectic action.
/// [`PhasedCircuitAction`] additionally compares the per-branch phase function
/// `φ(r) = i^⟨p, r⟩ (-1)^⟨B r + s, r⟩`, capturing exactly that information.
///
/// The comparison is *up to a single global phase* common to all branches: the encoder's absolute
/// phase is not exposed, so two Choi states that differ only by an overall scalar are reported as
/// equivalent. Pinning down that absolute phase as well requires the auxiliary-qubit separation of
/// §4.5 of [arXiv:2603.24717](https://arxiv.org/abs/2603.24717), a planned follow-up.
#[derive(Debug, Clone, PartialEq)]
pub struct PhasedCircuitAction {
    action: CircuitAction,
    phase: PhaseData,
    /// Indicator over the inner random bits: `true` where the bit is a symbolic rotation angle (a
    /// "virtual" random bit allocated via [`Simulation::allocate_symbolic_angle`]) rather than a
    /// genuine measurement-derived random bit.
    symbolic_angles: BitVec,
}

/// Computes a [`PhasedCircuitAction`] for `circuit` with the given input and output qubits.
///
/// Behaves exactly like [`action_of`] but uses a [`PhasedOutcomeCompleteSimulation`], additionally
/// recording the branch phase function of the circuit's Choi state.
///
/// # Errors
///
/// Returns [`ActionError`] if action calculation fails.
pub fn phased_action_of(
    circuit: &Circuit,
    input_qubits: &[QubitId],
    output_qubits: &[QubitId],
) -> Result<PhasedCircuitAction, ActionError> {
    let (action, simulation) = build_action::<PhasedOutcomeCompleteSimulation>(circuit, input_qubits, output_qubits)?;
    Ok(phased_action(action, &simulation))
}

/// Computes a [`PhasedCircuitAction`] directly from a [`PhasedOutcomeCompleteSimulation`] whose Choi
/// state the caller has already prepared.
///
/// This is the simulator-native counterpart of [`phased_action_of`], matching the convention used by
/// the Python bindings where the simulator itself records the circuit. The caller must, before
/// applying the circuit, have entangled each `input_qubits[k]` with a reference qubit via
/// `UnitaryOp::PrepareBell`, following the same layout as [`phased_action_of`]: the reference qubit
/// for `input_qubits[k]` is `system_qubit_count + k`, where `system_qubit_count` is one past the
/// largest index appearing in `input_qubits` or `output_qubits`.
///
/// # Errors
///
/// Returns [`ActionError::AuxiliaryQubitsEntangled`] if the non-output system qubits remain
/// entangled with the rest of the state.
pub fn phased_action_from_simulation(
    simulation: &PhasedOutcomeCompleteSimulation,
    input_qubits: &[QubitId],
    output_qubits: &[QubitId],
) -> Result<PhasedCircuitAction, ActionError> {
    let system_qubit_count = input_qubits
        .iter()
        .chain(output_qubits.iter())
        .copied()
        .max()
        .map_or(0, |qubit| qubit + 1);
    let reference_qubits: Vec<QubitId> = (system_qubit_count..system_qubit_count + input_qubits.len()).collect();
    let action = action_from_simulation(
        simulation,
        input_qubits,
        output_qubits,
        &reference_qubits,
        system_qubit_count,
    )?;
    Ok(phased_action(action, simulation))
}

/// Assembles a [`PhasedCircuitAction`] from a computed `action` and the `simulation` that recorded
/// the branch phase function.
fn phased_action(action: CircuitAction, simulation: &PhasedOutcomeCompleteSimulation) -> PhasedCircuitAction {
    let symbolic_angles: BitVec = simulation.symbolic_angle_indicator().iter().copied().collect();
    PhasedCircuitAction {
        action,
        phase: PhaseData::from_simulation(simulation),
        symbolic_angles,
    }
}

impl PhasedCircuitAction {
    /// The underlying phaseless [`CircuitAction`].
    #[must_use]
    pub fn action(&self) -> &CircuitAction {
        &self.action
    }

    /// Canonical choi state stabilizers; see [`CircuitAction::choi_state_stabilizers`].
    pub fn choi_state_stabilizers(&self) -> &[SparsePauli] {
        self.action.choi_state_stabilizers()
    }

    /// Returns `Ok(())` if the phaseless actions are equivalent up to signs, otherwise the reasons.
    ///
    /// This ignores phase entirely; use [`Self::is_equivalent_with_map`] to additionally compare the
    /// relative branch phases.
    ///
    /// # Errors
    ///
    /// Returns a list of [`ActionsInequivalenceReason`] if the phaseless actions differ.
    pub fn is_equivalent_up_to_signs(
        &self,
        other: &PhasedCircuitAction,
    ) -> Result<(), Vec<ActionsInequivalenceReason>> {
        self.action.is_equivalent_up_to_signs(&other.action)
    }

    /// Verifies that two phased actions implement the same operator on every input, enforcing the
    /// **virtual/true random-bit distinction**: symbolic-angle (virtual) random bits must correspond
    /// *one to one* between the two actions, while true (measurement-derived) random bits may be
    /// marginalized.
    ///
    /// A symbolic rotation `e^{iα P}` is modelled by conditioning `P` on a bit allocated via
    /// [`Simulation::allocate_symbolic_angle`]. Two encodings of the same parameterised circuit are
    /// equivalent only when their angle bits match up identically — angle `α_k` of one maps to angle
    /// `α_k` of the other, in allocation order, with no affine mixing. True random bits (allocated
    /// via [`Simulation::allocate_random_bit`] or produced by a genuine measurement) carry no such
    /// constraint: surplus true bits present in only one action are projected out, matching the way
    /// the phaseless [`CircuitAction::is_equivalent_with_map`] marginalizes measurement randomness.
    /// This is exactly what makes measurement-based "ejection" gadgets compare equal to the operation
    /// they implement directly.
    ///
    /// The two actions must have the same number of symbolic angles (otherwise
    /// [`ActionsInequivalenceReason::SymbolicAngleCount`] is returned). When both actions also share
    /// the same true random bits and those bits must be related non-trivially, use
    /// [`Self::is_equivalent_with_map`] with an explicit correspondence.
    ///
    /// # Errors
    ///
    /// Returns a list of [`ActionsInequivalenceReason`] if the actions differ.
    pub fn is_equivalent(&self, other: &PhasedCircuitAction) -> Result<(), Vec<ActionsInequivalenceReason>> {
        let map = self.provenance_random_map(other).map_err(|reason| vec![reason])?;
        self.check_with_random_map(other, &map)
    }

    /// Check if two phased actions are equivalent (up to a single global phase) when outcomes are
    /// remapped, comparing both the [`CircuitAction`] data and the relative branch phases.
    ///
    /// The outcome remapping `self_outcomes_from_other_outcomes` follows the same convention as
    /// [`CircuitAction::is_equivalent_with_map`]: outcomes of `self` equal `A(o_other)`. When the map
    /// is `None`, the zero map is used, as is common for circuits with unitary action.
    ///
    /// This is the lower-level escape hatch behind [`Self::is_equivalent`]. The supplied map may
    /// affinely remap *true* random bits, but **symbolic-angle (virtual) random bits must be mapped
    /// one to one** in allocation order. Any map whose induced random-bit correspondence affinely
    /// combines angle bits, or mixes them with true random bits, is rejected with
    /// [`ActionsInequivalenceReason::SymbolicAngleMixed`]; prefer [`Self::is_equivalent`] unless you
    /// specifically need to relabel true random bits.
    ///
    /// # Errors
    ///
    /// Returns a list of [`ActionsInequivalenceReason`] if the actions differ; the additional
    /// [`ActionsInequivalenceReason::RelativePhase`] is returned when only the branch phases differ.
    pub fn is_equivalent_with_map(
        &self,
        other: &PhasedCircuitAction,
        self_outcomes_from_other_outcomes: Option<&AffineMap>,
    ) -> Result<(), Vec<ActionsInequivalenceReason>> {
        let zero = zero_map(&self.action, &other.action);
        let outcome_map = self_outcomes_from_other_outcomes.unwrap_or(&zero);
        let self_outcomes_from_other_random = outcome_map.dot(&other.action.outcomes_from_random);
        let self_random_from_other_random = self.action.random_from_outcomes.dot(&self_outcomes_from_other_random);

        if !self.angle_correspondence_is_clean(other, &self_random_from_other_random) {
            return Err(vec![ActionsInequivalenceReason::SymbolicAngleMixed]);
        }
        self.check_with_random_map(other, &self_random_from_other_random)
    }

    /// Builds the random-bit correspondence used by [`Self::is_equivalent`]: identity (in allocation
    /// order) on the symbolic-angle bits, identity on the true bits shared by both actions, and a
    /// projection to zero of any surplus true bits present only in `other`.
    fn provenance_random_map(&self, other: &PhasedCircuitAction) -> Result<AffineMap, ActionsInequivalenceReason> {
        let self_angles: Vec<usize> = self.symbolic_angles.support().collect();
        let other_angles: Vec<usize> = other.symbolic_angles.support().collect();
        if self_angles.len() != other_angles.len() {
            return Err(ActionsInequivalenceReason::SymbolicAngleCount);
        }
        let self_random = self.symbolic_angles.len();
        let other_random = other.symbolic_angles.len();
        let self_trues = (0..self_random).filter(|&index| !self.symbolic_angles.index(index));
        let other_trues: Vec<usize> = (0..other_random)
            .filter(|&index| !other.symbolic_angles.index(index))
            .collect();

        let mut matrix = BitMatrix::zeros(self_random, other_random);
        for (&self_bit, &other_bit) in self_angles.iter().zip(other_angles.iter()) {
            matrix.set((self_bit, other_bit), true);
        }
        for (self_bit, &other_bit) in self_trues.zip(other_trues.iter()) {
            matrix.set((self_bit, other_bit), true);
        }
        Ok(AffineMap::linear(matrix))
    }

    /// Runs the count, sign and relative-phase comparisons under a given random-bit correspondence
    /// `self_random_from_other_random` (branch `r` of `other` corresponds to branch
    /// `self_random_from_other_random(r)` of `self`).
    fn check_with_random_map(
        &self,
        other: &PhasedCircuitAction,
        self_random_from_other_random: &AffineMap,
    ) -> Result<(), Vec<ActionsInequivalenceReason>> {
        self.action.is_equivalent_up_to_signs(&other.action)?;

        let mut reasons = Vec::new();
        if self
            .action
            .observables
            .is_equivalent_with_map(&other.action.observables, self_random_from_other_random)
        {
            reasons.push(ActionsInequivalenceReason::ObservablesSigns);
        }
        if self
            .action
            .stabilizers
            .is_equivalent_with_map(&other.action.stabilizers, self_random_from_other_random)
        {
            reasons.push(ActionsInequivalenceReason::StabilizersSigns);
        }
        if self
            .action
            .choi_state_stabilizers
            .is_equivalent_with_map(&other.action.choi_state_stabilizers, self_random_from_other_random)
        {
            reasons.push(ActionsInequivalenceReason::ChoiStateSigns);
        }
        if !self.relative_phase_matches(other) {
            reasons.push(ActionsInequivalenceReason::RelativePhase);
        }
        if reasons.is_empty() { Ok(()) } else { Err(reasons) }
    }

    /// Guards against an outcome remapping that does not respect the virtual/true distinction.
    ///
    /// Returns `true` iff the induced random-bit correspondence maps the symbolic-angle bits of
    /// `other` one to one onto those of `self` (in allocation order) with no leakage: each angle bit
    /// of `other` maps exactly to the matching angle bit of `self`, and no angle bit of `self` is
    /// driven by a true (non-angle) bit of `other`.
    fn angle_correspondence_is_clean(
        &self,
        other: &PhasedCircuitAction,
        self_random_from_other_random: &AffineMap,
    ) -> bool {
        let self_angles: Vec<usize> = self.symbolic_angles.support().collect();
        let other_angles: Vec<usize> = other.symbolic_angles.support().collect();
        if self_angles.len() != other_angles.len() {
            return false;
        }
        let matrix = self_random_from_other_random.matrix();
        if self_angles
            .iter()
            .any(|&self_bit| self_random_from_other_random.shift().index(self_bit))
        {
            return false;
        }
        for (&self_angle, &other_angle) in self_angles.iter().zip(other_angles.iter()) {
            for self_bit in 0..matrix.row_count() {
                let expected = self_bit == self_angle;
                if matrix.get((self_bit, other_angle)) != expected {
                    return false;
                }
            }
        }
        for &self_angle in &self_angles {
            for other_bit in 0..matrix.column_count() {
                if !other_angles.contains(&other_bit) && matrix.get((self_angle, other_bit)) {
                    return false;
                }
            }
        }
        true
    }

    /// Checks that the branch phase functions of `self` and `other` agree up to a global phase
    /// **on the symbolic-angle (virtual) random bits only**.
    ///
    /// Symbolic-angle bits carry the coherent, observable relative phases of the modelled rotations
    /// `e^{iα P}`, so they must match. True (measurement-derived) random bits label incoherent,
    /// traced-out measurement branches whose per-branch global phase is physically unobservable, so
    /// the comparison ignores them entirely (it sets every true bit to zero). This is what makes the
    /// phased equivalence reduce *exactly* to the phaseless [`CircuitAction`] equivalence when no
    /// symbolic angles are present, and what lets measurement-based "ejection" gadgets — whose
    /// corrected ancilla branches differ only by an unobservable per-branch phase — compare equal to
    /// the operation they implement directly.
    ///
    /// The angle bits of `self` and `other` correspond one to one in allocation order. The phase
    /// function `φ(r) = 2⟨p, r⟩ + 4⟨B r + s, r⟩ (mod 8)` restricted to the angle subspace is a
    /// degree-≤2 polynomial, so it is fully determined by its values on the zero vector, the angle
    /// unit vectors, and their pairwise sums. Equality up to a global phase is therefore equivalent
    /// to equality of the linear coefficients `φ(e_i) − φ(0)` and the quadratic coefficients
    /// `φ(e_i + e_j) − φ(e_i) − φ(e_j) + φ(0)` (ignoring the constant `φ(0)`, i.e. the global phase).
    fn relative_phase_matches(&self, other: &PhasedCircuitAction) -> bool {
        let self_angles: Vec<usize> = self.symbolic_angles.support().collect();
        let other_angles: Vec<usize> = other.symbolic_angles.support().collect();
        if self_angles.len() != other_angles.len() {
            return false;
        }
        let angle_count = self_angles.len();
        let self_dimension = self.phase.random_count();
        let other_dimension = other.phase.random_count();

        let phase_self = |angles: &[usize]| {
            let indices: Vec<usize> = angles.iter().map(|&order| self_angles[order]).collect();
            self.phase.phase_exponent(&unit_vector(self_dimension, &indices))
        };
        let phase_other = |angles: &[usize]| {
            let indices: Vec<usize> = angles.iter().map(|&order| other_angles[order]).collect();
            other.phase.phase_exponent(&unit_vector(other_dimension, &indices))
        };

        let constant_self = i32::from(phase_self(&[]));
        let constant_other = i32::from(phase_other(&[]));

        let mut linear_self = vec![0i32; angle_count];
        let mut linear_other = vec![0i32; angle_count];
        for order in 0..angle_count {
            linear_self[order] = (i32::from(phase_self(&[order])) - constant_self).rem_euclid(8);
            linear_other[order] = (i32::from(phase_other(&[order])) - constant_other).rem_euclid(8);
        }
        if linear_self != linear_other {
            return false;
        }

        for first in 0..angle_count {
            for second in (first + 1)..angle_count {
                let quadratic_self = (i32::from(phase_self(&[first, second]))
                    - constant_self
                    - linear_self[first]
                    - linear_self[second])
                    .rem_euclid(8);
                let quadratic_other = (i32::from(phase_other(&[first, second]))
                    - constant_other
                    - linear_other[first]
                    - linear_other[second])
                    .rem_euclid(8);
                if quadratic_self != quadratic_other {
                    return false;
                }
            }
        }
        true
    }
}

// ================================================================================================
// Private Types
// ================================================================================================

/// Branch phase function of a Choi state, indexed by the inner random bits.
///
/// The `ζ₈` phase of branch `r` is `ζ₈^φ(r)` with `φ(r) = 2⟨p, r⟩ + 4⟨B r + s, r⟩ (mod 8)`, matching
/// [`PhasedOutcomeCompleteSimulation::output_phase_exponent`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PhaseData {
    /// `p`: linear `i` phase.
    linear_i: BitVec,
    /// `s`: linear `-1` phase.
    linear_sign: BitVec,
    /// `B`: quadratic `-1` phase.
    quadratic: BitMatrix,
}

impl PhaseData {
    /// Extracts the branch phase function recorded by `simulation`.
    pub(crate) fn from_simulation(simulation: &PhasedOutcomeCompleteSimulation) -> Self {
        PhaseData {
            linear_i: simulation.linear_i_phase(),
            linear_sign: simulation.linear_sign_phase(),
            quadratic: simulation.quadratic_phase_matrix(),
        }
    }

    fn random_count(&self) -> usize {
        self.linear_i.len()
    }

    /// The `ζ₈` exponent `φ(r) = 2⟨p, r⟩ + 4⟨B r + s, r⟩ (mod 8)` for the branch `random_bits`.
    pub(crate) fn phase_exponent(&self, random_bits: &BitVec) -> u8 {
        phase_form_exponent(
            self.random_count(),
            |index| random_bits.index(index),
            |index| self.linear_i.index(index),
            |index| self.linear_sign.index(index),
            |row, column| self.quadratic.get((row, column)),
        )
    }
}

/// Evaluates the `ζ₈ = e^{iπ/4}` exponent of the F₂ phase form `i^⟨p, r⟩ (-1)^⟨B r + s, r⟩` for a
/// random-bit assignment `r`.
///
/// The coefficients are read through accessor closures so the phased simulator and its lowered
/// [`crate::action`] `PhaseData` — which store `p`, `s`, `B` and `r` in different (aligned vs.
/// unaligned) representations — share a single implementation. `random_bit`, `linear_i` (`p`) and
/// `linear_sign` (`s`) are indexed by column and `quadratic` reads `B[(row, column)]`, all over
/// `0..random_count`.
pub(crate) fn phase_form_exponent(
    random_count: usize,
    random_bit: impl Fn(usize) -> bool,
    linear_i: impl Fn(usize) -> bool,
    linear_sign: impl Fn(usize) -> bool,
    quadratic: impl Fn(usize, usize) -> bool,
) -> u8 {
    let mut linear_i_parity = false;
    let mut sign = false;
    for column in 0..random_count {
        if !random_bit(column) {
            continue;
        }
        linear_i_parity ^= linear_i(column);
        sign ^= linear_sign(column);
        for row in 0..random_count {
            if random_bit(row) && quadratic(row, column) {
                sign = !sign;
            }
        }
    }
    (2 * u8::from(linear_i_parity) + 4 * u8::from(sign)) % 8
}

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

/// Returns the length-`dimension` bit vector with the bits in `set_indices` set to one.
fn unit_vector(dimension: usize, set_indices: &[usize]) -> BitVec {
    let mut vector = BitVec::zeros(dimension);
    for &index in set_indices {
        vector.assign_index(index, true);
    }
    vector
}
