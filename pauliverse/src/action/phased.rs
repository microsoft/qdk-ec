use super::{
    ActionError, ActionsInequivalenceReason, CircuitAction, QubitId, action_from_simulation, build_action, unit_vector,
    zero_map,
};
use crate::phased_outcome_complete_simulation::phase_form_exponent;
use crate::{PhasedOutcomeCompleteSimulation, Simulation, circuit::Circuit};
use binar::{AffineMap, BitMatrix, BitVec, Bitwise};
use paulimer::clifford::separate_auxiliary_qubits;
use paulimer::{
    CliffordMutable,
    clifford::{Clifford, group_encoding_clifford_of},
};
use paulimer::{Pauli, SparsePauli};

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
/// convention-defined encoder phase recovered by the §4.3 auxiliary-qubit separation of
/// [arXiv:2603.24717](https://arxiv.org/abs/2603.24717), or inspect it with [`Self::global_phase`].
///
/// Dense branch-vector tests cover this comparison. They are not a proof of soundness or
/// completeness. The default correspondence does not search all possible outcome relabelings.
#[derive(Debug, Clone, PartialEq)]
pub struct PhasedCircuitAction {
    action: CircuitAction,
    phase: PhaseData,
    /// Outcome map with symbolic-angle allocation rows omitted.
    physical_outcomes_from_branch_bits: BitMatrix,
    /// Indicator over the branch bits: `true` where the bit is a symbolic rotation angle (a
    /// "virtual" branch bit allocated via [`Simulation::allocate_symbolic_angle`]) rather than a
    /// genuine measurement-derived random bit.
    symbolic_angles: BitVec,
    /// The convention-defined `ζ₈` phase of the Choi-state encoder, recovered via the §4.3 auxiliary
    /// separation (the constant term `l` of the separation phase polynomial). Used only by
    /// [`PhasedCircuitAction::is_equivalent_with_global_phase`]; the up-to-global-phase checks ignore
    /// it.
    global_phase: u8,
}

/// Computes a [`PhasedCircuitAction`] for `circuit` with the given input and output qubits.
///
/// Behaves exactly like [`action_of`](super::action_of) but uses a [`PhasedOutcomeCompleteSimulation`], additionally
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
    phased_action(action, &simulation)
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
    phased_action(action, simulation)
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
/// `physical_outcomes_from_branch_bits`. Discarding the auxiliary qubits keeps their coherence exactly
/// when every auxiliary stabiliser sign is constant along that kernel. Every ordinary random bit is
/// itself a physical outcome, so such a direction is supported on symbolic-angle bits alone.
///
/// Revealing an angle indirectly is enough: two rotations measured together expose only their
/// parity, and that parity fixes the auxiliary state, so the pair is accepted.
fn symbolic_angle_on_auxiliary_signs(
    action: &CircuitAction,
    simulation: &PhasedOutcomeCompleteSimulation,
    physical_outcomes_from_branch_bits: &BitMatrix,
) -> Option<usize> {
    let signs = action.auxiliary_stabilizers.sign_from_branch_bits.matrix();
    debug_assert!(
        signs.column_count() <= physical_outcomes_from_branch_bits.column_count(),
        "both matrices are indexed by branch bit, so the signs cannot name more of them"
    );
    let kernel = physical_outcomes_from_branch_bits.kernel();
    let unrevealed = (0..kernel.row_count())
        .map(|row| BitVec::from(&kernel.row(row)))
        .find(|direction| (0..signs.row_count()).any(|generator| sign_changes_along(signs, generator, direction)))?;
    let angles = simulation.symbolic_angle_indicator();
    let branch_bit = unrevealed
        .support()
        .find(|bit| angles.get(*bit).copied().unwrap_or(false));
    debug_assert!(
        branch_bit.is_some(),
        "every ordinary random bit holds its own physical outcome row, so a direction the outcomes \
         leave free can only run along symbolic angles"
    );
    let outcome = branch_bit_outcome_id(simulation, branch_bit?);
    debug_assert!(outcome.is_some(), "every branch bit is reported by one public outcome");
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

/// The public outcome id that reports branch bit `branch_bit`.
fn branch_bit_outcome_id(simulation: &PhasedOutcomeCompleteSimulation, branch_bit_index: usize) -> Option<usize> {
    simulation
        .branch_bit_locations()
        .find(|location| location.branch_bit_index == branch_bit_index)
        .map(|location| location.outcome_id)
}

/// Assembles a [`PhasedCircuitAction`] from a computed `action` and the `simulation` that recorded
/// the branch phase function, recovering the convention-defined encoder phase from the reference and output
/// qubits.
fn phased_action(
    action: CircuitAction,
    simulation: &PhasedOutcomeCompleteSimulation,
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
    let physical_outcomes_from_branch_bits = physical_outcome_matrix(simulation);
    if let Some(angle) = symbolic_angle_on_auxiliary_signs(&action, simulation, &physical_outcomes_from_branch_bits) {
        return Err(ActionError::AuxiliaryQubitsCarrySymbolicAngle { angle });
    }
    let symbolic_angles: BitVec = simulation.symbolic_angle_indicator().iter().copied().collect();
    let global_phase = separate_auxiliary_qubits(
        &simulation.phased_state_encoder(),
        &action.choi_state_stabilizers.canonical_to_original,
    )?
    .phase()
    .constant();
    let phase = PhaseData::from_simulation(simulation, &action);
    Ok(PhasedCircuitAction {
        action,
        phase,
        physical_outcomes_from_branch_bits,
        symbolic_angles,
        global_phase,
    })
}

fn physical_outcome_matrix(simulation: &PhasedOutcomeCompleteSimulation) -> BitMatrix {
    let outcomes = simulation.outcome_matrix();
    debug_assert_eq!(
        simulation.branch_bit_locations().count(),
        simulation.symbolic_angle_indicator().len()
    );
    let mut is_symbolic_outcome = vec![false; simulation.outcome_count()];
    for location in simulation.branch_bit_locations() {
        is_symbolic_outcome[location.outcome_id] = simulation.symbolic_angle_indicator()[location.branch_bit_index];
    }
    let physical_rows: Vec<_> = (0..simulation.outcome_count())
        .filter(|&outcome| !is_symbolic_outcome[outcome])
        .collect();

    let mut physical = BitMatrix::zeros(physical_rows.len(), outcomes.column_count());
    for (row, &source_row) in physical_rows.iter().enumerate() {
        for column in outcomes.row(source_row).support() {
            physical.set((row, column), true);
        }
    }
    physical
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
        let self_branch_bits_from_other_branch_bits =
            self.provenance_branch_map(other).map_err(|reason| vec![reason])?;
        let other_branch_bits_from_self_branch_bits =
            other.provenance_branch_map(self).map_err(|reason| vec![reason])?;
        self.check_with_branch_map(
            other,
            &self_branch_bits_from_other_branch_bits,
            Some(&other_branch_bits_from_self_branch_bits),
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
        let self_outcomes_from_other_branch_bits = outcome_map.dot(&other.action.outcomes_from_branch_bits);
        let self_branch_bits_from_other_branch_bits = self
            .action
            .branch_bits_from_outcomes
            .dot(&self_outcomes_from_other_branch_bits);

        if !self.angle_correspondence_is_clean(other, &self_branch_bits_from_other_branch_bits) {
            return Err(vec![ActionsInequivalenceReason::SymbolicAngleMixed]);
        }
        if self
            .action
            .outcomes_from_branch_bits
            .dot(&self_branch_bits_from_other_branch_bits)
            != self_outcomes_from_other_branch_bits
        {
            return Err(vec![ActionsInequivalenceReason::RandomOutcomeCoverage]);
        }
        self.check_with_branch_map(other, &self_branch_bits_from_other_branch_bits, None)?;
        if !self.branch_map_covers_self(&self_branch_bits_from_other_branch_bits) {
            return Err(vec![ActionsInequivalenceReason::RandomOutcomeCoverage]);
        }
        Ok(())
    }

    /// Covers every self assignment modulo true-random translations that preserve Choi signs
    /// and change phase only by a constant on each physical-outcome fibre.
    ///
    /// A debug assertion checks that physical-kernel directions contain only angle bits. The polar
    /// form's diagonal then adds only constraints already imposed by the angle rows, so it is omitted.
    fn branch_map_covers_self(&self, branch_map: &AffineMap) -> bool {
        let dimension = self.phase.branch_count();
        if branch_map.matrix().rank() == dimension {
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
        let physical_kernel = self.physical_outcomes_from_branch_bits.kernel();
        debug_assert!(
            (0..physical_kernel.row_count()).all(|row| physical_kernel
                .row(row)
                .support()
                .all(|bit| self.symbolic_angles.index(bit))),
            "a physical-kernel direction contains a true-random bit"
        );
        let phase_constraints = &physical_kernel * &phase_polar_form;
        let signs = self.action.choi_state_stabilizers.sign_from_branch_bits.matrix();
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
        (&constraints * branch_map.matrix()).rank() == constraints.rank()
    }

    /// Builds the branch-bit correspondence used by [`Self::is_equivalent`]. Symbolic-angle bits and
    /// shared true-random bits map by allocation order; surplus true-random bits map to zero.
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
        reverse_branch_map: Option<&AffineMap>,
    ) -> Result<(), Vec<ActionsInequivalenceReason>> {
        self.action.is_equivalent_up_to_signs(&other.action)?;

        let mut reasons = self
            .action
            .sign_reasons_with_branch_map(&other.action, self_branch_bits_from_other_branch_bits);
        if let Some(other_branch_bits_from_self_branch_bits) = reverse_branch_map {
            for reason in other
                .action
                .sign_reasons_with_branch_map(&self.action, other_branch_bits_from_self_branch_bits)
            {
                if !reasons.contains(&reason) {
                    reasons.push(reason);
                }
            }
        }
        let relative_phase_matches = self.relative_phase_matches(other, self_branch_bits_from_other_branch_bits)
            && reverse_branch_map.is_none_or(|other_branch_bits_from_self_branch_bits| {
                other.relative_phase_matches(self, other_branch_bits_from_self_branch_bits)
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
    fn relative_phase_matches(
        &self,
        other: &PhasedCircuitAction,
        self_branch_bits_from_other_branch_bits: &AffineMap,
    ) -> bool {
        let other_dimension = other.phase.branch_count();
        let phase_difference = |other_branch_bits: &BitVec| {
            let self_branch_bits = self_branch_bits_from_other_branch_bits.apply(other_branch_bits);
            (i32::from(self.phase.phase_exponent(&self_branch_bits))
                - i32::from(other.phase.phase_exponent(other_branch_bits)))
            .rem_euclid(8)
        };
        let (linear, quadratic) = quadratic_phase_coefficients(&phase_difference, other_dimension);
        let kernel = other.physical_outcomes_from_branch_bits.kernel();
        if kernel.row_count() != self.physical_outcomes_from_branch_bits.kernel().row_count() {
            return false;
        }
        for row in 0..kernel.row_count() {
            let direction: BitVec = (&kernel.row(row)).into();
            let mapped = self_branch_bits_from_other_branch_bits.matrix() * &direction.as_view();
            let observed = &self.physical_outcomes_from_branch_bits * &mapped.as_view();
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

/// Branch phase function of a Choi state, indexed by the branch bits.
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
    /// `A`: branch labels as a function of the branch bits.
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

    fn branch_count(&self) -> usize {
        self.linear_i.len()
    }

    /// The relative phase exponent in the unsigned marginal frame.
    pub(crate) fn phase_exponent(&self, branch_assignment: &BitVec) -> u8 {
        let mut phase = phase_form_exponent(
            self.branch_count(),
            |index| branch_assignment.index(index),
            |index| self.linear_i.index(index),
            |index| self.linear_sign.index(index),
            |row, column| self.quadratic.get((row, column)),
        );
        let label = &self.branch_labels * &branch_assignment.as_view();
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

#[cfg(test)]
mod tests;
