use super::{
    ActionError, ActionsInequivalenceReason, CircuitAction, QubitId, action_from_simulation, build_action, unit_vector,
    zero_map,
};
use crate::phased_outcome_complete_simulation::phase_form_exponent;
use crate::{PhasedOutcomeCompleteSimulation, circuit::Circuit};
use binar::{AffineMap, BitMatrix, BitVec, Bitwise};
use paulimer::SparsePauli;

/// The initial phase-aware analog of [`CircuitAction`], computed with a
/// [`PhasedOutcomeCompleteSimulation`] so that the **relative `ζ₈` phases between branches** of the
/// circuit's Choi state are retained in addition to the phaseless stabilizer data.
///
/// A [`CircuitAction`] determines the Choi state only up to phase, so it cannot distinguish circuits
/// that act identically on the Pauli group but differ by branch-dependent phases — for example
/// `e^{iα Z}` and `e^{-iα Z}`, whose conditioned Paulis `+Z` and `-Z` share a symplectic action.
/// [`PhasedCircuitAction`] additionally compares the explicit per-branch phase function
/// `φ(r) = i^⟨p, r⟩ (-1)^⟨B r + s, r⟩`.
///
/// This initial comparison has known false matches. It does not include the later common-frame
/// comparison of encoder phases, and the encoder's global phase is not exposed.
/// Auxiliary-qubit separation is described in §4.3 of
/// [arXiv:2603.24717](https://arxiv.org/abs/2603.24717).
#[derive(Debug, Clone, PartialEq)]
pub struct PhasedCircuitAction {
    action: CircuitAction,
    phase: PhaseData,
    /// Indicator over the branch bits: `true` where the bit is a symbolic rotation angle (a
    /// "virtual" branch bit allocated via [`crate::Simulation::allocate_symbolic_angle`]) rather than a
    /// genuine measurement-derived random bit.
    symbolic_angles: BitVec,
}

/// Computes a [`PhasedCircuitAction`] for `circuit` with the given input and output qubits.
///
/// Behaves exactly like [`action_of`](super::action_of) but uses a [`PhasedOutcomeCompleteSimulation`], additionally
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
/// Every other simulation qubit is auxiliary unless it is an output.
///
/// # Errors
///
/// Returns [`ActionError::AuxiliaryQubitsEntangled`] if the auxiliary qubits remain
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

    /// Applies the initial phased-action comparison, enforcing the symbolic/measurement branch-bit
    /// distinction. Symbolic-angle bits correspond one to one between the two actions.
    /// This comparison has known false matches; see [`PhasedCircuitAction`].
    ///
    /// A symbolic rotation `e^{iα P}` is modelled by conditioning `P` on a bit allocated via
    /// [`crate::Simulation::allocate_symbolic_angle`]. Two encodings of the same parameterised circuit are
    /// equivalent only when their angle bits match up identically — angle `α_k` of one maps to angle
    /// `α_k` of the other, in allocation order, with no affine mixing. True random bits (allocated
    /// via [`crate::Simulation::allocate_random_bit`] or produced by a genuine measurement) carry no such
    /// constraint: surplus true bits present in only one action are projected out, matching the way
    /// the phaseless [`CircuitAction::is_equivalent_with_map`] marginalizes measurement randomness.
    /// This correspondence supports the tested measurement-based "ejection" gadgets.
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
        let map = self.provenance_branch_map(other).map_err(|reason| vec![reason])?;
        self.check_with_branch_map(other, &map)
    }

    /// Applies the initial comparison with remapped outcomes, using [`CircuitAction`] data and
    /// the explicit branch phase form. This comparison has known false matches; see [`PhasedCircuitAction`].
    ///
    /// The outcome remapping `self_outcomes_from_other_outcomes` follows the same convention as
    /// [`CircuitAction::is_equivalent_with_map`]: outcomes of `self` equal `A(o_other)`. When the map
    /// is `None`, the zero map is used, as is common for circuits with unitary action.
    ///
    /// This is the lower-level escape hatch behind [`Self::is_equivalent`]. The supplied map may
    /// affinely remap *true* random bits, but **symbolic-angle (virtual) random bits must be mapped
    /// one to one** in allocation order. Any map whose induced branch-bit correspondence affinely
    /// combines angle bits, or mixes them with true random bits, is rejected with
    /// [`ActionsInequivalenceReason::SymbolicAngleMixed`]; prefer [`Self::is_equivalent`] unless you
    /// specifically need to relabel true branch bits.
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
        let self_outcomes_from_other_branch_bits = outcome_map.dot(&other.action.outcomes_from_branch_bits);
        let self_branch_bits_from_other_branch_bits = self
            .action
            .branch_bits_from_outcomes
            .dot(&self_outcomes_from_other_branch_bits);

        if !self.angle_correspondence_is_clean(other, &self_branch_bits_from_other_branch_bits) {
            return Err(vec![ActionsInequivalenceReason::SymbolicAngleMixed]);
        }
        self.check_with_branch_map(other, &self_branch_bits_from_other_branch_bits)
    }

    /// Builds the branch-bit correspondence used by [`Self::is_equivalent`]: identity (in allocation
    /// order) on the symbolic-angle bits, identity on the true bits shared by both actions, and a
    /// projection to zero of any surplus true bits present only in `other`.
    fn provenance_branch_map(&self, other: &PhasedCircuitAction) -> Result<AffineMap, ActionsInequivalenceReason> {
        let self_angles: Vec<usize> = self.symbolic_angles.support().collect();
        let other_angles: Vec<usize> = other.symbolic_angles.support().collect();
        if self_angles.len() != other_angles.len() {
            return Err(ActionsInequivalenceReason::SymbolicAngleCount);
        }
        let self_branch_bits = self.symbolic_angles.len();
        let other_branch_bits = other.symbolic_angles.len();
        let self_trues = (0..self_branch_bits).filter(|&index| !self.symbolic_angles.index(index));
        let other_trues: Vec<usize> = (0..other_branch_bits)
            .filter(|&index| !other.symbolic_angles.index(index))
            .collect();

        let mut matrix = BitMatrix::zeros(self_branch_bits, other_branch_bits);
        for (&self_bit, &other_bit) in self_angles.iter().zip(other_angles.iter()) {
            matrix.set((self_bit, other_bit), true);
        }
        for (self_bit, &other_bit) in self_trues.zip(other_trues.iter()) {
            matrix.set((self_bit, other_bit), true);
        }
        Ok(AffineMap::linear(matrix))
    }

    /// Runs the count, sign and relative-phase comparisons under a given branch-bit correspondence
    /// `self_branch_bits_from_other_branch_bits` (branch `r` of `other` corresponds to branch
    /// `self_branch_bits_from_other_branch_bits(r)` of `self`).
    fn check_with_branch_map(
        &self,
        other: &PhasedCircuitAction,
        self_branch_bits_from_other_branch_bits: &AffineMap,
    ) -> Result<(), Vec<ActionsInequivalenceReason>> {
        self.action.is_equivalent_up_to_signs(&other.action)?;

        let mut reasons = Vec::new();
        if self
            .action
            .observables
            .signs_differ_with_map(&other.action.observables, self_branch_bits_from_other_branch_bits)
        {
            reasons.push(ActionsInequivalenceReason::ObservablesSigns);
        }
        if self
            .action
            .stabilizers
            .signs_differ_with_map(&other.action.stabilizers, self_branch_bits_from_other_branch_bits)
        {
            reasons.push(ActionsInequivalenceReason::StabilizersSigns);
        }
        if self.action.choi_state_stabilizers.signs_differ_with_map(
            &other.action.choi_state_stabilizers,
            self_branch_bits_from_other_branch_bits,
        ) {
            reasons.push(ActionsInequivalenceReason::ChoiStateSigns);
        }
        if !self.relative_phase_matches(other) {
            reasons.push(ActionsInequivalenceReason::RelativePhase);
        }
        if reasons.is_empty() { Ok(()) } else { Err(reasons) }
    }

    /// Guards against an outcome remapping that does not respect the virtual/true distinction.
    ///
    /// Returns `true` iff the induced branch-bit correspondence maps the symbolic-angle bits of
    /// `other` one to one onto those of `self` (in allocation order) with no leakage: each angle bit
    /// of `other` maps exactly to the matching angle bit of `self`, and no angle bit of `self` is
    /// driven by a true (non-angle) bit of `other`.
    fn angle_correspondence_is_clean(
        &self,
        other: &PhasedCircuitAction,
        self_branch_bits_from_other_branch_bits: &AffineMap,
    ) -> bool {
        let self_angles: Vec<usize> = self.symbolic_angles.support().collect();
        let other_angles: Vec<usize> = other.symbolic_angles.support().collect();
        if self_angles.len() != other_angles.len() {
            return false;
        }
        let matrix = self_branch_bits_from_other_branch_bits.matrix();
        if self_angles
            .iter()
            .any(|&self_bit| self_branch_bits_from_other_branch_bits.shift().index(self_bit))
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
        let self_dimension = self.phase.branch_count();
        let other_dimension = other.phase.branch_count();

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

/// Branch phase function of a Choi state, indexed by the branch bits.
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

    fn branch_count(&self) -> usize {
        self.linear_i.len()
    }

    /// The `ζ₈` exponent `φ(r) = 2⟨p, r⟩ + 4⟨B r + s, r⟩ (mod 8)` for the branch `branch_assignment`.
    pub(crate) fn phase_exponent(&self, branch_assignment: &BitVec) -> u8 {
        phase_form_exponent(
            self.branch_count(),
            |index| branch_assignment.index(index),
            |index| self.linear_i.index(index),
            |index| self.linear_sign.index(index),
            |row, column| self.quadratic.get((row, column)),
        )
    }
}

#[cfg(test)]
mod tests;
