use std::fmt::Debug;

use crate::{
    OutcomeCompleteSimulation, PhasedOutcomeCompleteSimulation, Simulation,
    circuit::{Circuit, SimulationError},
};
use binar::{AffineMap, BitMatrix, BitVec, Bitwise, BitwiseMut, IndexSet};
use paulimer::{
    CliffordMutable, CliffordUnitary, Pauli, PauliMutable, SparsePauli,
    clifford::{
        AuxiliarySeparationError, Clifford, group_encoding_clifford_of, separate_auxiliary_qubits,
        standard_restriction_with_sign_matrix,
    },
};

type QubitId = crate::circuit::QubitId;

#[cfg(test)]
mod encoder_phase_tests {
    use super::*;
    use dense_oracle::{C, Dense, gate_matrix, zeta8};
    use paulimer::{UnitaryOp, clifford::clifford_to_pauli_exponents, core::x};
    use rand::{RngExt, SeedableRng};

    fn apply_gate(state: &mut Dense, operation: UnitaryOp, support: &[usize]) {
        match operation {
            UnitaryOp::ControlledX => state.apply_cx(support[0], support[1]),
            UnitaryOp::ControlledZ => state.apply_cz(support[0], support[1]),
            _ => state.apply1(support[0], gate_matrix(operation)),
        }
    }

    fn basis_state(qubit_count: usize, label: &BitVec) -> Dense {
        let mut state = Dense::zero(qubit_count);
        for bit in label.support() {
            state.apply1(bit, gate_matrix(UnitaryOp::X));
        }
        state
    }

    fn reordered(amplitudes: &[C], order: &[usize]) -> Vec<C> {
        (0..amplitudes.len())
            .map(|index| {
                let original = order.iter().enumerate().fold(0, |bits, (position, &qubit)| {
                    bits | (((index >> (order.len() - 1 - position)) & 1) << (order.len() - 1 - qubit))
                });
                amplitudes[original]
            })
            .collect()
    }

    fn check_family(qubit_count: usize, output: &[usize], gates: &[(UnitaryOp, Vec<usize>)]) {
        let mut simulation = PhasedOutcomeCompleteSimulation::with_capacity(qubit_count, qubit_count, qubit_count);
        for bit in 0..qubit_count {
            let coin = simulation.allocate_random_bit();
            let observable: SparsePauli = [x(bit)].as_slice().into();
            simulation.conditional_pauli(&observable, &[coin], true);
        }
        for (operation, support) in gates {
            simulation.unitary_op(*operation, support);
        }
        let action = phased_action_from_simulation(&simulation, &[], output).unwrap();
        let choi = &action.action.choi_state_stabilizers;
        let auxiliary = &action.action.auxiliary_stabilizers;
        let reference = group_encoding_clifford_of(&choi.canonical_generators, output.len()).tensor(
            &group_encoding_clifford_of(&auxiliary.canonical_generators, qubit_count - output.len()),
        );
        let exponents = clifford_to_pauli_exponents(&reference);
        let order: Vec<_> = output.iter().chain(&auxiliary.canonical_to_original).copied().collect();
        let mut constant_phase = None;
        for assignment in 0..1 << qubit_count {
            let random: BitVec = (0..qubit_count).map(|bit| assignment & (1 << bit) != 0).collect();
            let mut actual = basis_state(qubit_count, &random);
            for (operation, support) in gates {
                apply_gate(&mut actual, *operation, support);
            }
            let actual = Dense {
                qubit_count,
                amp: reordered(&actual.amp, &order),
            };
            let mut label = BitVec::zeros(qubit_count);
            for bit in 0..qubit_count {
                let observable = reference.image_z(bit);
                let x_bits: Vec<_> = (0..qubit_count).map(|bit| observable.x_bits().index(bit)).collect();
                let z_bits: Vec<_> = (0..qubit_count).map(|bit| observable.z_bits().index(bit)).collect();
                let transformed = actual.pauli_applied(&x_bits, &z_bits, i64::from(observable.xz_phase_exponent() % 4));
                let expectation: C = actual
                    .amp
                    .iter()
                    .zip(transformed)
                    .map(|(first, second)| first.conj() * second)
                    .sum();
                assert!((expectation.re.abs() - 1.0).abs() < 1e-9);
                label.assign_index(bit, expectation.re < 0.0);
            }
            let mut expected = basis_state(qubit_count, &label);
            for exponent in &exponents {
                let x_bits: Vec<_> = (0..qubit_count).map(|bit| exponent.x_bits().index(bit)).collect();
                let z_bits: Vec<_> = (0..qubit_count).map(|bit| exponent.z_bits().index(bit)).collect();
                expected.apply_pauli_exp(&x_bits, &z_bits, i64::from(exponent.xz_phase_exponent() % 4));
            }
            let phase = zeta8(i64::from(action.phase.phase_exponent(&random)));
            for amplitude in &mut expected.amp {
                *amplitude *= phase;
            }
            let constant = *constant_phase.get_or_insert_with(|| {
                actual
                    .amp
                    .iter()
                    .zip(&expected.amp)
                    .map(|(first, second)| first * second.conj())
                    .sum::<C>()
            });
            assert!((constant.norm() - 1.0).abs() < 1e-9);
            for (first, second) in actual.amp.iter().zip(&expected.amp) {
                assert!(
                    (*first - constant * second).norm() < 1e-9,
                    "{qubit_count} qubits, output={output:?}, assignment={assignment}, gates={gates:?}",
                );
            }
        }
    }

    #[test]
    fn encoder_phase_matches_dense_basis_families() {
        check_family(2, &[0, 1], &[(UnitaryOp::ControlledZ, vec![0, 1])]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5a1f_2026);
        let single = [
            UnitaryOp::Hadamard,
            UnitaryOp::SqrtZ,
            UnitaryOp::SqrtZInv,
            UnitaryOp::X,
            UnitaryOp::Y,
            UnitaryOp::Z,
            UnitaryOp::SqrtX,
        ];
        for qubit_count in 1..=4 {
            for trial in 0..32 {
                let output_count = if trial % 2 == 0 { qubit_count } else { qubit_count - 1 };
                let output: Vec<_> = (0..output_count).rev().collect();
                let mut gates = Vec::new();
                for _ in 0..24 {
                    let first = rng.random_range(0..qubit_count);
                    let kind = rng.random_range(0..9);
                    if kind >= single.len() && first < output_count && output_count >= 2 {
                        let second = (first + rng.random_range(1..output_count)) % output_count;
                        let operation = if kind == 7 {
                            UnitaryOp::ControlledX
                        } else {
                            UnitaryOp::ControlledZ
                        };
                        gates.push((operation, vec![first, second]));
                    } else {
                        gates.push((single[kind % single.len()], vec![first]));
                    }
                }
                check_family(qubit_count, &output, &gates);
            }
        }
    }
}

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
    /// A discarded auxiliary qubit carries a stabilizer sign that changes with a symbolic angle
    /// that no physical outcome reveals. Discarding it would decohere that angle, so this
    /// representation cannot record the circuit. `angle` is the outcome id returned by
    /// [`Simulation::allocate_symbolic_angle`].
    ///
    /// The check is conservative. A rotation confined to discarded qubits is harmless, yet it
    /// carries the same recorded data as a rotation that spans a retained qubit, so both are
    /// refused.
    AuxiliaryQubitsCarrySymbolicAngle { angle: usize },
    /// The qubits do not describe a Choi-state layout. Either `input_qubits` or `output_qubits`
    /// names a qubit twice, or the simulation does not hold every system and reference qubit.
    InvalidQubits {
        input_qubits: Vec<QubitId>,
        output_qubits: Vec<QubitId>,
    },
    #[from]
    AuxiliarySeparationFailed(AuxiliarySeparationError),
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
    /// The two phased actions agree up to a global phase but their *absolute* global `ζ₈` phases
    /// differ. Only produced by [`PhasedCircuitAction::is_equivalent_with_global_phase`]; see its
    /// documentation for details.
    GlobalPhase,
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

        let reasons = self.sign_reasons_with_random_map(other, &self_random_from_other_random);
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

    fn sign_reasons_with_random_map(
        &self,
        other: &CircuitAction,
        self_random_from_other_random: &AffineMap,
    ) -> Vec<ActionsInequivalenceReason> {
        let mut reasons = Vec::new();
        if self
            .observables
            .is_equivalent_with_map(&other.observables, self_random_from_other_random)
        {
            reasons.push(ActionsInequivalenceReason::ObservablesSigns);
        }
        if self
            .stabilizers
            .is_equivalent_with_map(&other.stabilizers, self_random_from_other_random)
        {
            reasons.push(ActionsInequivalenceReason::StabilizersSigns);
        }
        if self
            .choi_state_stabilizers
            .is_equivalent_with_map(&other.choi_state_stabilizers, self_random_from_other_random)
        {
            reasons.push(ActionsInequivalenceReason::ChoiStateSigns);
        }
        reasons
    }
}

/// The phase-aware analog of [`CircuitAction`], computed with a
/// [`PhasedOutcomeCompleteSimulation`] so that the **relative `ζ₈` phases between branches** of the
/// circuit's Choi state are retained in addition to the phaseless stabilizer data.
///
/// A [`CircuitAction`] determines the Choi state only up to phase, so it cannot distinguish circuits
/// that act identically on the Pauli group but differ by branch-dependent phases — for example
/// `e^{iα Z}` and `e^{-iα Z}`, whose conditioned Paulis `+Z` and `-Z` share a symplectic action.
/// [`PhasedCircuitAction`] compares the simulator's explicit branch phase and the encoder's
/// label-dependent phase in a common unsigned stabilizer frame.
///
/// The comparison ignores an angle-independent phase within each physical-outcome sector.
/// It retains relative phases between virtual assignments in that sector.
/// Use [`Self::is_equivalent_with_global_phase`] to include the
/// absolute phase recovered by the §4.3 auxiliary-qubit separation of
/// [arXiv:2603.24717](https://arxiv.org/abs/2603.24717), or inspect it with [`Self::global_phase`].
///
/// Dense branch-vector tests cover this comparison. They are not a proof of soundness or
/// completeness. The default correspondence does not search all possible outcome relabelings.
#[derive(Debug, Clone, PartialEq)]
pub struct PhasedCircuitAction {
    action: CircuitAction,
    phase: PhaseData,
    /// Outcome map with symbolic-angle allocation rows omitted.
    physical_outcomes_from_random: BitMatrix,
    /// Indicator over the inner random bits: `true` where the bit is a symbolic rotation angle (a
    /// "virtual" random bit allocated via [`Simulation::allocate_symbolic_angle`]) rather than a
    /// genuine measurement-derived random bit.
    symbolic_angles: BitVec,
    /// The absolute global `ζ₈` phase of the Choi-state encoder, recovered via the §4.3 auxiliary
    /// separation (the constant term `l` of the separation phase polynomial). Used only by
    /// [`PhasedCircuitAction::is_equivalent_with_global_phase`]; the up-to-global-phase checks ignore
    /// it.
    global_phase: u8,
}

/// Computes a [`PhasedCircuitAction`] for `circuit` with the given input and output qubits.
///
/// Behaves exactly like [`action_of`] but uses a [`PhasedOutcomeCompleteSimulation`], additionally
/// recording the branch phase function of the circuit's Choi state.
///
/// # Errors
///
/// Returns [`ActionError::InvalidQubits`] if `input_qubits` or `output_qubits` names a qubit twice,
/// [`ActionError::AuxiliaryQubitsEntangled`] if a non-output system qubit remains entangled,
/// [`ActionError::AuxiliarySeparationFailed`] if the transient auxiliary qubits cannot be separated,
/// or [`ActionError::SimulationFailed`] if circuit simulation fails.
pub fn phased_action_of(
    circuit: &Circuit,
    input_qubits: &[QubitId],
    output_qubits: &[QubitId],
) -> Result<PhasedCircuitAction, ActionError> {
    if has_repeated_qubit(input_qubits) || has_repeated_qubit(output_qubits) {
        return Err(invalid_qubits(input_qubits, output_qubits));
    }
    let (action, simulation) = build_action::<PhasedOutcomeCompleteSimulation>(circuit, input_qubits, output_qubits)?;
    let qubit_count = circuit
        .qubit_count()
        .max(input_qubits.iter().max().map_or(0, |&q| q + 1))
        .max(output_qubits.iter().max().map_or(0, |&q| q + 1));
    let reference_qubits: Vec<QubitId> = (qubit_count..qubit_count + input_qubits.len()).collect();
    phased_action(action, &simulation, &reference_qubits, output_qubits)
}

/// Computes a [`PhasedCircuitAction`] directly from a [`PhasedOutcomeCompleteSimulation`] whose Choi
/// state the caller has already prepared.
///
/// This is the simulator-native counterpart of [`phased_action_of`], matching the convention used by
/// the Python bindings where the simulator itself records the circuit. The caller must, before
/// applying the circuit, have entangled each `input_qubits[k]` with a reference qubit via
/// `UnitaryOp::PrepareBell`, following the same layout as [`phased_action_of`]: the reference qubit
/// for `input_qubits[k]` is `system_qubit_count + k`, where `system_qubit_count` is one past the
/// largest index appearing in `input_qubits` or `output_qubits`. Any further qubit of the
/// simulation is an auxiliary qubit and must be disentangled from the rest of the state.
///
/// # Errors
///
/// Returns [`ActionError::InvalidQubits`] if `input_qubits` or `output_qubits` names a qubit twice,
/// or if the simulation has fewer than `system_qubit_count + input_qubits.len()` qubits. Returns
/// [`ActionError::AuxiliaryQubitsEntangled`] if the non-output system qubits remain entangled with
/// the rest of the state, or [`ActionError::AuxiliarySeparationFailed`] if the transient auxiliary
/// qubits cannot be separated.
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
    if system_qubit_count + input_qubits.len() > simulation.qubit_count()
        || has_repeated_qubit(input_qubits)
        || has_repeated_qubit(output_qubits)
    {
        return Err(invalid_qubits(input_qubits, output_qubits));
    }
    let action = action_from_simulation(
        simulation,
        input_qubits,
        output_qubits,
        &reference_qubits,
        system_qubit_count,
    )?;
    phased_action(action, simulation, &reference_qubits, output_qubits)
}

fn has_repeated_qubit(qubits: &[QubitId]) -> bool {
    let mut seen = std::collections::HashSet::with_capacity(qubits.len());
    qubits.iter().any(|qubit| !seen.insert(*qubit))
}

fn invalid_qubits(input_qubits: &[QubitId], output_qubits: &[QubitId]) -> ActionError {
    ActionError::InvalidQubits {
        input_qubits: input_qubits.to_vec(),
        output_qubits: output_qubits.to_vec(),
    }
}

/// The outcome id of a symbolic angle whose branch a discarded auxiliary qubit records but no
/// physical outcome reveals.
///
/// Two branches that share a physical-outcome record differ by a vector in the kernel of
/// `physical_outcomes_from_random`. Discarding the auxiliary qubits keeps their coherence exactly
/// when every auxiliary stabiliser sign is constant along that kernel. Every ordinary random bit is
/// itself a physical outcome, so such a direction is supported on symbolic-angle bits alone.
///
/// Revealing an angle indirectly is enough: two rotations measured together expose only their
/// parity, and that parity fixes the auxiliary state, so the pair is accepted.
fn symbolic_angle_on_auxiliary_signs(
    action: &CircuitAction,
    simulation: &PhasedOutcomeCompleteSimulation,
    physical_outcomes_from_random: &BitMatrix,
) -> Option<usize> {
    let signs = action.auxiliary_stabilizers.sign_from_random.matrix();
    debug_assert!(
        signs.column_count() <= physical_outcomes_from_random.column_count(),
        "both matrices are indexed by inner random bit, so the signs cannot name more of them"
    );
    let kernel = physical_outcomes_from_random.kernel();
    let unrevealed = (0..kernel.row_count())
        .map(|row| BitVec::from(&kernel.row(row)))
        .find(|direction| (0..signs.row_count()).any(|generator| sign_changes_along(signs, generator, direction)))?;
    let angles = simulation.symbolic_angle_indicator();
    let random_bit = unrevealed
        .support()
        .find(|bit| angles.get(*bit).copied().unwrap_or(false));
    debug_assert!(
        random_bit.is_some(),
        "every ordinary random bit holds its own physical outcome row, so a direction the outcomes \
         leave free can only run along symbolic angles"
    );
    let outcome = random_bit_outcome_id(simulation, random_bit?);
    debug_assert!(
        outcome.is_some(),
        "every inner random bit is reported by one public outcome"
    );
    outcome
}

/// Whether the sign of auxiliary generator `generator` differs between two branches separated by
/// `direction`.
fn sign_changes_along(signs: &BitMatrix, generator: usize, direction: &BitVec) -> bool {
    direction
        .support()
        .filter(|bit| *bit < signs.column_count() && signs[(generator, *bit)])
        .count()
        % 2
        == 1
}

/// The public outcome id that reports inner random bit `random_bit`.
fn random_bit_outcome_id(simulation: &PhasedOutcomeCompleteSimulation, random_bit: usize) -> Option<usize> {
    simulation
        .random_outcome_indicator()
        .iter()
        .enumerate()
        .filter(|(_, is_random)| **is_random)
        .nth(random_bit)
        .map(|(outcome, _)| outcome)
}

/// Assembles a [`PhasedCircuitAction`] from a computed `action` and the `simulation` that recorded
/// the branch phase function, recovering the absolute global phase from the reference and output
/// qubits.
fn phased_action(
    action: CircuitAction,
    simulation: &PhasedOutcomeCompleteSimulation,
    reference_qubits: &[QubitId],
    output_qubits: &[QubitId],
) -> Result<PhasedCircuitAction, ActionError> {
    if let Some(angle) = simulation.reused_symbolic_angle() {
        return Err(ActionError::SymbolicAngleReused { angle });
    }
    if let Some(angle) = simulation.unused_symbolic_angle() {
        return Err(ActionError::SymbolicAngleUnused { angle });
    }
    // arXiv:2309.08676 deallocates a qubit only when its state is the same on every branch. An
    // auxiliary state that the physical outcomes fix meets this; one that also follows an
    // unrevealed symbolic angle would decohere that angle when discarded.
    let physical_outcomes_from_random = physical_outcome_matrix(simulation);
    if let Some(angle) = symbolic_angle_on_auxiliary_signs(&action, simulation, &physical_outcomes_from_random) {
        return Err(ActionError::AuxiliaryQubitsCarrySymbolicAngle { angle });
    }
    let symbolic_angles: BitVec = simulation.symbolic_angle_indicator().iter().copied().collect();
    let global_phase = separation_global_phase(simulation, reference_qubits, output_qubits)?;
    let phase = PhaseData::from_simulation(simulation, &action);
    Ok(PhasedCircuitAction {
        action,
        phase,
        physical_outcomes_from_random,
        symbolic_angles,
        global_phase,
    })
}

fn physical_outcome_matrix(simulation: &PhasedOutcomeCompleteSimulation) -> BitMatrix {
    let outcomes = simulation.outcome_matrix();
    let mut physical_rows = Vec::new();
    let mut random_bit = 0;
    for (outcome, &is_random) in simulation.random_outcome_indicator().iter().enumerate() {
        let is_symbolic_angle = if is_random {
            let is_symbolic_angle = simulation.symbolic_angle_indicator()[random_bit];
            random_bit += 1;
            is_symbolic_angle
        } else {
            false
        };
        if !is_symbolic_angle {
            physical_rows.push(outcome);
        }
    }
    debug_assert_eq!(random_bit, simulation.symbolic_angle_indicator().len());

    let mut physical = BitMatrix::zeros(physical_rows.len(), outcomes.column_count());
    for (row, &source_row) in physical_rows.iter().enumerate() {
        for column in outcomes.row(source_row).support() {
            physical.set((row, column), true);
        }
    }
    physical
}

fn separation_global_phase(
    simulation: &PhasedOutcomeCompleteSimulation,
    reference_qubits: &[QubitId],
    output_qubits: &[QubitId],
) -> Result<u8, AuxiliarySeparationError> {
    let support: Vec<usize> = reference_qubits.iter().chain(output_qubits.iter()).copied().collect();
    let separation = separate_auxiliary_qubits(&simulation.phased_state_encoder(), &support)?;
    Ok(separation.phase().constant())
}

impl PhasedCircuitAction {
    /// The underlying phaseless [`CircuitAction`].
    #[must_use]
    pub fn action(&self) -> &CircuitAction {
        &self.action
    }

    /// The global `ζ₈` phase of the Choi-state encoder recovered by the §4.3 auxiliary separation,
    /// as an exponent in `0..8`. Compared only by [`Self::is_equivalent_with_global_phase`].
    ///
    /// The value is fixed by the canonical marginal-encoder convention, not by the operator alone,
    /// so it is a reference point rather than an absolute quantity. The identity circuit gives `2`,
    /// not `0`. Two circuits that apply the same operator to the outputs can still differ here when
    /// they leave a discarded auxiliary qubit in different states: `S` on qubit `0` gives `0` with
    /// the auxiliary qubit untouched, `2` after an `X` on it, and `4` after an `H` or a `Y`.
    ///
    /// Differences are meaningful when both actions are built the same way. `XZ` and `Y`, for
    /// instance, differ by `6`, which is the factor `-i` that relates them.
    #[must_use]
    pub fn global_phase(&self) -> u8 {
        self.global_phase
    }

    /// Canonical choi state stabilizers; see [`CircuitAction::choi_state_stabilizers`].
    #[must_use]
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

    /// Compares actions under the default angle and outcome correspondence.
    ///
    /// At each physical outcome, Kraus operators must agree for every angle value up to an
    /// angle-independent phase. Extra true-random records are ignored only when they uniformly
    /// refine the same outcome operation, with angle-independent weights. Outcomes with
    /// angle-dependent proportionality factors are not merged.
    ///
    /// Dense branch-vector tests cover this comparison. They are not a proof of soundness or
    /// completeness. The default correspondence does not search all possible outcome relabelings.
    ///
    /// A symbolic rotation `e^{iα P}` is modelled by conditioning `P` on a bit allocated via
    /// [`Simulation::allocate_symbolic_angle`]. Two encodings of the same parameterised circuit are
    /// equivalent only when their angle bits match up identically — angle `α_k` of one maps to angle
    /// `α_k` of the other, in allocation order, with no affine mixing. True random bits (allocated
    /// via [`Simulation::allocate_random_bit`] or produced by a genuine measurement) are checked in
    /// both comparison directions.
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
        let self_random_from_other_random = self.provenance_random_map(other).map_err(|reason| vec![reason])?;
        let other_random_from_self_random = other.provenance_random_map(self).map_err(|reason| vec![reason])?;
        self.check_with_random_map(
            other,
            &self_random_from_other_random,
            Some(&other_random_from_self_random),
        )
    }

    /// Like [`Self::is_equivalent`], but additionally requires the two actions' global `ζ₈` phases
    /// to agree.
    ///
    /// [`Self::is_equivalent`] ignores angle-independent phases within physical-outcome sectors.
    /// This stronger check also pins [`Self::global_phase`], which distinguishes `Co` from `-Co`.
    /// When the operators
    /// agree only up to a non-trivial global phase,
    /// [`ActionsInequivalenceReason::GlobalPhase`] is added to the reasons.
    ///
    /// Read the check against the convention described on [`Self::global_phase`]. Two circuits that
    /// apply the same operator to the outputs are reported as different here when they leave a
    /// discarded auxiliary qubit in different states. Use this only when both actions are built the
    /// same way and exact equality of that recorded phase is what you want.
    ///
    /// The comparison quotients by the physical outcome record first, so a phase difference that
    /// every outcome sector reveals is treated as global within its sector.
    ///
    /// # Errors
    ///
    /// Returns a list of [`ActionsInequivalenceReason`] if the actions differ.
    pub fn is_equivalent_with_global_phase(
        &self,
        other: &PhasedCircuitAction,
    ) -> Result<(), Vec<ActionsInequivalenceReason>> {
        self.is_equivalent(other)?;
        if self.global_phase != other.global_phase {
            return Err(vec![ActionsInequivalenceReason::GlobalPhase]);
        }
        Ok(())
    }

    /// Compares actions under a supplied outcome correspondence, using the per-outcome phase
    /// convention of [`Self::is_equivalent`].
    ///
    /// The outcome remapping `self_outcomes_from_other_outcomes` follows the same convention as
    /// [`CircuitAction::is_equivalent_with_map`]: outcomes of `self` equal `A(o_other)`. When the map
    /// is `None`, the zero map is used, as is common for circuits with unitary action.
    ///
    /// This is the lower-level escape hatch behind [`Self::is_equivalent`]. The supplied map may
    /// affinely remap *true* random bits, but every symbolic-angle row must equal the corresponding
    /// angle coordinate in allocation order. A true-random row can depend on an angle coordinate
    /// only when the physical outcome record determines that dependence.
    ///
    /// The map must preserve physical-outcome fibres in both directions and cover every
    /// nonredundant outcome of `self`. Selecting only some active outcomes is not equivalence.
    /// It must also map possible records to possible records, including deterministic outcomes.
    ///
    /// # Errors
    ///
    /// Returns a list of [`ActionsInequivalenceReason`] if the actions differ; the additional
    /// [`ActionsInequivalenceReason::RelativePhase`] also reports incompatible physical-outcome
    /// fibres. [`ActionsInequivalenceReason::RandomOutcomeCoverage`] reports omitted outcome classes
    /// or a map that selects impossible records.
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
        if self.action.outcomes_from_random.dot(&self_random_from_other_random) != self_outcomes_from_other_random {
            return Err(vec![ActionsInequivalenceReason::RandomOutcomeCoverage]);
        }
        self.check_with_random_map(other, &self_random_from_other_random, None)?;
        if !self.random_map_covers_self(&self_random_from_other_random) {
            return Err(vec![ActionsInequivalenceReason::RandomOutcomeCoverage]);
        }
        Ok(())
    }

    /// Covers every self assignment modulo true-random translations that preserve Choi signs
    /// and change phase only by a constant on each physical-outcome fibre.
    ///
    /// A debug assertion checks that physical-kernel directions contain only angle bits. The polar
    /// form's diagonal then adds only constraints already imposed by the angle rows, so it is omitted.
    fn random_map_covers_self(&self, random_map: &AffineMap) -> bool {
        let dimension = self.phase.random_count();
        if random_map.matrix().rank() == dimension {
            return true;
        }
        let (_, quadratic) =
            quadratic_phase_coefficients(&|random| i32::from(self.phase.phase_exponent(random)), dimension);
        let mut phase_polar_form = BitMatrix::zeros(dimension, dimension);
        for first in 0..dimension {
            for second in first + 1..dimension {
                let coupled = quadratic[first * dimension + second] & 4 != 0;
                phase_polar_form.set((first, second), coupled);
                phase_polar_form.set((second, first), coupled);
            }
        }
        let physical_kernel = self.physical_outcomes_from_random.kernel();
        debug_assert!(
            (0..physical_kernel.row_count()).all(|row| physical_kernel
                .row(row)
                .support()
                .all(|bit| self.symbolic_angles.index(bit))),
            "a physical-kernel direction contains a true-random bit"
        );
        let phase_constraints = &physical_kernel * &phase_polar_form;
        let signs = self.action.choi_state_stabilizers.sign_from_random.matrix();
        let angle_count = self.symbolic_angles.support().count();
        let mut constraints = BitMatrix::zeros(
            angle_count + signs.row_count() + phase_constraints.row_count(),
            dimension,
        );
        for (row, angle) in self.symbolic_angles.support().enumerate() {
            constraints.set((row, angle), true);
        }
        for row in 0..signs.row_count() {
            for bit in signs.row(row).support() {
                constraints.set((angle_count + row, bit), true);
            }
        }
        for row in 0..phase_constraints.row_count() {
            for bit in phase_constraints.row(row).support() {
                constraints.set((angle_count + signs.row_count() + row, bit), true);
            }
        }
        (&constraints * random_map.matrix()).rank() == constraints.rank()
    }

    /// Builds the random-bit correspondence used by [`Self::is_equivalent`]. Symbolic-angle bits and
    /// shared true-random bits map by allocation order; surplus true-random bits map to zero.
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
        reverse_random_map: Option<&AffineMap>,
    ) -> Result<(), Vec<ActionsInequivalenceReason>> {
        self.action.is_equivalent_up_to_signs(&other.action)?;

        let mut reasons = self
            .action
            .sign_reasons_with_random_map(&other.action, self_random_from_other_random);
        if let Some(other_random_from_self_random) = reverse_random_map {
            for reason in other
                .action
                .sign_reasons_with_random_map(&self.action, other_random_from_self_random)
            {
                if !reasons.contains(&reason) {
                    reasons.push(reason);
                }
            }
        }
        let relative_phase_matches = self.relative_phase_matches(other, self_random_from_other_random)
            && reverse_random_map.is_none_or(|other_random_from_self_random| {
                other.relative_phase_matches(self, other_random_from_self_random)
            });
        if !relative_phase_matches {
            reasons.push(ActionsInequivalenceReason::RelativePhase);
        }
        if reasons.is_empty() { Ok(()) } else { Err(reasons) }
    }

    /// Guards against an outcome remapping that does not respect the virtual/true distinction.
    ///
    /// Returns `true` iff each symbolic-angle row is the corresponding angle coordinate of `other`.
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
            for other_bit in 0..matrix.column_count() {
                let expected = other_bit == other_angle;
                if matrix.get((self_angle, other_bit)) != expected {
                    return false;
                }
            }
        }
        true
    }

    /// Checks that the phase difference is constant on every physical-outcome fibre.
    fn relative_phase_matches(&self, other: &PhasedCircuitAction, self_random_from_other_random: &AffineMap) -> bool {
        let other_dimension = other.phase.random_count();
        let phase_difference = |other_random: &BitVec| {
            let self_random = self_random_from_other_random.apply(other_random);
            (i32::from(self.phase.phase_exponent(&self_random)) - i32::from(other.phase.phase_exponent(other_random)))
                .rem_euclid(8)
        };
        let (linear, quadratic) = quadratic_phase_coefficients(&phase_difference, other_dimension);
        let kernel = other.physical_outcomes_from_random.kernel();
        if kernel.row_count() != self.physical_outcomes_from_random.kernel().row_count() {
            return false;
        }
        for row in 0..kernel.row_count() {
            let direction: BitVec = (&kernel.row(row)).into();
            let mapped = self_random_from_other_random.matrix() * &direction.as_view();
            let observed = &self.physical_outcomes_from_random * &mapped.as_view();
            if observed.support().next().is_some() {
                return false;
            }
            if !phase_is_invariant_along(&linear, &quadratic, &direction) {
                return false;
            }
        }
        true
    }
}

fn quadratic_phase_coefficients(phase: &impl Fn(&BitVec) -> i32, dimension: usize) -> (Vec<i32>, Vec<i32>) {
    let constant = phase(&BitVec::zeros(dimension));
    let mut linear = vec![0; dimension];
    for (first, coefficient) in linear.iter_mut().enumerate() {
        *coefficient = (phase(&unit_vector(dimension, &[first])) - constant).rem_euclid(8);
    }

    let mut quadratic = vec![0; dimension * dimension];
    for first in 0..dimension {
        for second in (first + 1)..dimension {
            quadratic[first * dimension + second] =
                (phase(&unit_vector(dimension, &[first, second])) - constant - linear[first] - linear[second])
                    .rem_euclid(8);
        }
    }
    (linear, quadratic)
}

fn phase_is_invariant_along(linear: &[i32], quadratic: &[i32], direction: &BitVec) -> bool {
    let dimension = linear.len();
    let mut derivative_constant = 0;
    let mut derivative_linear = vec![0; dimension];
    for bit in direction.support() {
        derivative_constant += linear[bit];
        derivative_linear[bit] -= 2 * linear[bit];
    }
    for first in 0..dimension {
        for second in (first + 1)..dimension {
            let coefficient = quadratic[first * dimension + second];
            match (direction.index(first), direction.index(second)) {
                (true, true) => {
                    derivative_constant += coefficient;
                    derivative_linear[first] -= coefficient;
                    derivative_linear[second] -= coefficient;
                }
                (true, false) => {
                    if (2 * coefficient).rem_euclid(8) != 0 {
                        return false;
                    }
                    derivative_linear[second] += coefficient;
                }
                (false, true) => {
                    if (2 * coefficient).rem_euclid(8) != 0 {
                        return false;
                    }
                    derivative_linear[first] += coefficient;
                }
                (false, false) => {}
            }
        }
    }
    if derivative_constant.rem_euclid(8) != 0 {
        return false;
    }
    if derivative_linear
        .iter()
        .any(|coefficient| coefficient.rem_euclid(8) != 0)
    {
        return false;
    }
    true
}

// ================================================================================================
// Private Types
// ================================================================================================

/// Branch phase function of a Choi state, indexed by the inner random bits.
///
/// Adds the simulator's explicit phase to the encoder phase in an unsigned marginal frame.
/// A branch-independent constant is omitted.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PhaseData {
    /// `p`: linear `i` phase.
    linear_i: BitVec,
    /// `s`: linear `-1` phase.
    linear_sign: BitVec,
    /// `B`: quadratic `-1` phase.
    quadratic: BitMatrix,
    /// `A`: branch labels as a function of the inner random bits.
    branch_labels: BitMatrix,
    encoder_linear: Vec<u8>,
    encoder_quadratic: BitMatrix,
}

impl PhaseData {
    /// Uses the tensor of unsigned marginal encoders as the reference.
    /// For `D|0> = scalar |d>` and `D X_i D† = i^p_i X^x_i Z^z_i`, the linear coefficient is
    /// `2 p_i + 4 <z_i, d>` and the quadratic coefficient is `4 <z_i, x_j>`.
    fn from_simulation(simulation: &PhasedOutcomeCompleteSimulation, action: &CircuitAction) -> Self {
        let choi = &action.choi_state_stabilizers;
        let auxiliary = &action.auxiliary_stabilizers;
        let choi_count = choi.canonical_generators.len();
        let qubit_count = simulation.qubit_count();
        let reference = group_encoding_clifford_of(&choi.canonical_generators, choi_count).tensor(
            &group_encoding_clifford_of(&auxiliary.canonical_generators, qubit_count - choi_count),
        );
        let order: Vec<_> = choi
            .canonical_to_original
            .iter()
            .chain(&auxiliary.canonical_to_original)
            .copied()
            .collect();
        let mut encoder = simulation.state_encoder();
        encoder.left_mul_permutation(&order, &(0..qubit_count).collect::<Vec<_>>());
        let images: Vec<_> = (0..qubit_count)
            .map(|bit| reference.preimage(&encoder.image_x(bit)))
            .collect();
        let offset: BitVec = (0..qubit_count)
            .map(|bit| {
                let preimage = encoder.preimage(&reference.image_z(bit));
                debug_assert!(preimage.x_bits().is_zero());
                preimage.xz_phase_exponent() % 4 == 2
            })
            .collect();
        let encoder_linear = images
            .iter()
            .map(|image| {
                let shifted_sign = image.z_bits().support().filter(|&bit| offset.index(bit)).count() % 2 == 1;
                (2 * (image.xz_phase_exponent() % 4) + 4 * u8::from(shifted_sign)) % 8
            })
            .collect();
        let mut encoder_quadratic = BitMatrix::zeros(qubit_count, qubit_count);
        for first in 0..qubit_count {
            for second in first + 1..qubit_count {
                let coupled = images[first]
                    .z_bits()
                    .support()
                    .filter(|&bit| images[second].x_bits().index(bit))
                    .count()
                    % 2
                    == 1;
                encoder_quadratic.set((first, second), coupled);
            }
        }
        PhaseData {
            linear_i: simulation.linear_i_phase(),
            linear_sign: simulation.linear_sign_phase(),
            quadratic: simulation.quadratic_phase_matrix(),
            branch_labels: simulation.sign_matrix(),
            encoder_linear,
            encoder_quadratic,
        }
    }

    fn random_count(&self) -> usize {
        self.linear_i.len()
    }

    /// The relative phase exponent in the unsigned marginal frame.
    pub(crate) fn phase_exponent(&self, random_bits: &BitVec) -> u8 {
        let mut phase = phase_form_exponent(
            self.random_count(),
            |index| random_bits.index(index),
            |index| self.linear_i.index(index),
            |index| self.linear_sign.index(index),
            |row, column| self.quadratic.get((row, column)),
        );
        let label = &self.branch_labels * &random_bits.as_view();
        for first in label.support() {
            phase = (phase + self.encoder_linear[first]) % 8;
            for second in label.support().filter(|&bit| bit > first) {
                if self.encoder_quadratic.get((first, second)) {
                    phase = (phase + 4) % 8;
                }
            }
        }
        phase
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
