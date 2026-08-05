use crate::Simulation;
use crate::action::phase_form_exponent;
use crate::outcome_complete_simulation::row_sum;
use crate::outcome_free_simulation::{max_pair_support, max_support};
use binar::{BitMatrix, BitVec};
use binar::{Bitwise, BitwiseMut, BitwisePair, BitwisePairMut, IndexSet, matrix::AlignedBitMatrix, vec::AlignedBitVec};
use paulimer::clifford::{Clifford, CliffordUnitary, PhasedCliffordUnitary};
use paulimer::pauli::{Pauli, PauliBits, PauliUnitary, anti_commutes_with, generic::PhaseExponent};
use paulimer::pauli::{PauliBinaryOps, PauliMutable};
use paulimer::{CLIFFORD_BIT_ALIGNMENT, UnitaryOp};
use rand::RngExt;

type SparsePauli = paulimer::pauli::SparsePauli;

/// Asymptotically efficient stabilizer simulation tracking all outcomes *and* the exact global phase.
///
/// This is the global-phase-tracking generalization of [`crate::OutcomeCompleteSimulation`]. Where
/// the latter implements Algorithm 5.3 of [KBP](https://arxiv.org/abs/2309.08676) and represents the
/// simulated state only *up to a global phase*, this simulator implements Algorithm 4.2 of the
/// [phased simulation paper](https://arxiv.org/abs/2603.24717) and tracks the global phase exactly.
///
/// Exact phase tracking is what enables verification of circuits that contain *non-stabilizer*
/// resources such as symbolic single-qubit rotations: an equality `C₁ e^{iαZ} C₂|0⟩ = D₁ e^{iαZ} D₂|0⟩`
/// for all `α` reduces to a pair of exact stabilizer-state equalities, and exactness — not equality
/// up to a global phase — is precisely what makes the reduction valid.
///
/// # Representation
///
/// For a circuit with `n` output qubits and `n_M` outcomes the simulator maintains a vector
/// `q ∈ {1, 1/2}^{n_M}` of conditional outcome probabilities together with a *phased* Clifford
/// encoder `R` and `𝔽₂` data `A`, `B`, `M`, `v₀`, `p`, `s`. For a random-bit vector
/// `r ∈ {0,1}^{n_r}` (where `n_r` is the number of random outcomes) the outcome vector is
/// `v = v₀ + M r` and the exact output state is
///
/// ```text
///     i^⟨p, r⟩ · (-1)^⟨B r + s, r⟩ · R |A r⟩.
/// ```
///
/// Here `⟨p, r⟩` is linear in `r` (an `i`-phase), while `⟨B r + s, r⟩` is a quadratic form in `r`
/// (a `±1`-phase). The encoder `R` additionally fixes the overall `ζ₈ = e^{iπ/4}` phase of `R|Ar⟩`.
///
/// | code field             | paper symbol | meaning                                       |
/// |------------------------|--------------|-----------------------------------------------|
/// | `phased_clifford`      | `R`          | phased state encoder (exact global phase)     |
/// | `sign_matrix`          | `A`          | random bits → computational-basis register    |
/// | `quadratic_phase_matrix` | `B`        | quadratic `-1` phase                          |
/// | `outcome_matrix`       | `M`          | random bits → outcome vector                  |
/// | `outcome_shift`        | `v₀`         | deterministic outcome shift                   |
/// | `linear_i_phase`       | `p`          | linear `i` phase                              |
/// | `linear_sign_phase`    | `s`          | linear `-1` phase                             |
/// | `random_outcome_indicator` | `q`      | which outcomes are random (probability 1/2)   |
///
/// # Phase-resolved Clifford application
///
/// A bare [`CliffordUnitary`] is *phaseless*: its symplectic tableau fixes the operator on the Pauli
/// group but not the overall `ζ₈` factor. Exact phase tracking therefore requires Cliffords to be
/// applied through phase-resolved entry points — the named gates of [`Simulation::unitary_op`],
/// [`Simulation::pauli`] and [`Simulation::pauli_exp`], plus [`Simulation::permute`],
/// [`Simulation::conditional_pauli`] and the measurement methods. Applying an opaque phaseless
/// [`CliffordUnitary`] via [`Simulation::clifford`] would leave the global phase undetermined, so
/// that method is unsupported here (it panics); see its documentation for the rationale.
///
/// # Examples
///
/// ```
/// use pauliverse::{PhasedOutcomeCompleteSimulation, Simulation};
/// use paulimer::UnitaryOp;
///
/// let mut sim = PhasedOutcomeCompleteSimulation::new(2);
/// sim.unitary_op(UnitaryOp::Hadamard, &[0]);
/// sim.unitary_op(UnitaryOp::ControlledX, &[0, 1]);
///
/// // The encoder now prepares a Bell pair with an exactly-known global phase.
/// assert_eq!(sim.random_outcome_count(), 0);
/// ```
#[must_use]
pub struct PhasedOutcomeCompleteSimulation {
    phased_clifford: PhasedCliffordUnitary,   // R (phased encoder)
    sign_matrix: AlignedBitMatrix,            // A
    quadratic_phase_matrix: AlignedBitMatrix, // B
    outcome_matrix: AlignedBitMatrix,         // M
    outcome_shift: AlignedBitVec,             // v_0
    linear_i_phase: AlignedBitVec,            // p
    linear_sign_phase: AlignedBitVec,         // s
    random_outcome_indicator: Vec<bool>,      // vec(q), [j] is true iff vec(q)_j = 1/2
    symbolic_angle_indicator: Vec<bool>,      // [k] is true iff random bit k is a symbolic rotation angle
    random_bit_count: usize,
    qubit_count: usize,
}

impl std::fmt::Debug for PhasedOutcomeCompleteSimulation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhasedOutcomeCompleteSimulation")
            .field("state_encoder", &self.phased_clifford.clone().into_clifford())
            .field("sign_matrix", &self.sign_matrix())
            .field("quadratic_phase_matrix", &self.quadratic_phase_matrix())
            .field("outcome_matrix", &self.outcome_matrix())
            .field("outcome_shift", &self.outcome_shift().iter().collect::<Vec<bool>>())
            .field("linear_i_phase", &self.linear_i_phase().iter().collect::<Vec<bool>>())
            .field(
                "linear_sign_phase",
                &self.linear_sign_phase().iter().collect::<Vec<bool>>(),
            )
            .field("random_outcome_indicator", &self.random_outcome_indicator)
            .field("symbolic_angle_indicator", &self.symbolic_angle_indicator)
            .field("random_bit_count", &self.random_bit_count)
            .field("qubit_count", &self.qubit_count)
            .finish()
    }
}

impl Default for PhasedOutcomeCompleteSimulation {
    fn default() -> Self {
        PhasedOutcomeCompleteSimulation::with_capacity(0, 0, 0)
    }
}

impl PhasedOutcomeCompleteSimulation {
    /// Get the phaseless Clifford unitary encoding the current stabilizer state.
    ///
    /// This is the unitary `R` such that `R|0⟩` represents the stabilizer state, with the global
    /// phase discarded. Use [`Self::phased_state_encoder`] to retain the exact global phase.
    pub fn state_encoder(&self) -> CliffordUnitary {
        self.phased_state_encoder().into_clifford()
    }

    /// Get the phased Clifford unitary encoding the current stabilizer state with exact global phase.
    pub fn phased_state_encoder(&self) -> PhasedCliffordUnitary {
        let mut res = self.phased_clifford.clone();
        res.resize(self.qubit_count);
        res
    }

    /// Get the sign matrix `A` tracking how computational-basis registers depend on random bits.
    ///
    /// Returns a cache-aligned reference for efficiency.
    pub fn aligned_sign_matrix(&self) -> &AlignedBitMatrix {
        &self.sign_matrix
    }

    /// Get a copy of the sign matrix `A` without alignment constraints.
    pub fn sign_matrix(&self) -> BitMatrix {
        BitMatrix::from_aligned(AlignedBitMatrix::from_row_iter(
            self.sign_matrix.row_iterator(0..self.qubit_count()),
            self.random_outcome_count(),
        ))
    }

    /// Get the quadratic phase matrix `B` (cache-aligned).
    ///
    /// The `±1` phase contributed by `B` is `(-1)^⟨B r, r⟩`.
    pub fn aligned_quadratic_phase_matrix(&self) -> &AlignedBitMatrix {
        &self.quadratic_phase_matrix
    }

    /// Get a copy of the quadratic phase matrix `B` without alignment constraints.
    pub fn quadratic_phase_matrix(&self) -> BitMatrix {
        BitMatrix::from_aligned(AlignedBitMatrix::from_row_iter(
            self.quadratic_phase_matrix.row_iterator(0..self.random_outcome_count()),
            self.random_outcome_count(),
        ))
    }

    /// Get the outcome matrix `M` encoding all `2^{n_r}` measurement branches (cache-aligned).
    pub fn aligned_outcome_matrix(&self) -> &AlignedBitMatrix {
        &self.outcome_matrix
    }

    /// Get a copy of the outcome matrix `M` without alignment constraints.
    pub fn outcome_matrix(&self) -> BitMatrix {
        BitMatrix::from_aligned(AlignedBitMatrix::from_row_iter(
            self.outcome_matrix.row_iterator(0..self.outcome_count()),
            self.random_outcome_count(),
        ))
    }

    /// Get the outcome shift vector `v₀` (cache-aligned).
    pub fn aligned_outcome_shift(&self) -> &AlignedBitVec {
        &self.outcome_shift
    }

    /// Get a copy of the outcome shift vector `v₀` without alignment constraints.
    pub fn outcome_shift(&self) -> BitVec {
        BitVec::from_aligned(self.outcome_count(), self.outcome_shift.clone())
    }

    /// Get the linear `i`-phase vector `p` (cache-aligned).
    ///
    /// The `i` phase contributed by `p` is `i^⟨p, r⟩`.
    pub fn aligned_linear_i_phase(&self) -> &AlignedBitVec {
        &self.linear_i_phase
    }

    /// Get a copy of the linear `i`-phase vector `p` without alignment constraints.
    pub fn linear_i_phase(&self) -> BitVec {
        BitVec::from_aligned(self.random_outcome_count(), self.linear_i_phase.clone())
    }

    /// Get the linear `-1`-phase vector `s` (cache-aligned).
    ///
    /// The `±1` phase contributed by `s` is `(-1)^⟨s, r⟩`.
    pub fn aligned_linear_sign_phase(&self) -> &AlignedBitVec {
        &self.linear_sign_phase
    }

    /// Get a copy of the linear `-1`-phase vector `s` without alignment constraints.
    pub fn linear_sign_phase(&self) -> BitVec {
        BitVec::from_aligned(self.random_outcome_count(), self.linear_sign_phase.clone())
    }

    /// Returns the `ζ₈ = e^{iπ/4}` exponent of the scalar `i^⟨p, r⟩ (-1)^⟨B r + s, r⟩` for the given
    /// random-bit assignment `r`.
    ///
    /// This is the `r`-dependent prefactor multiplying `R|A r⟩` in the output state. The remaining
    /// phase of `R|A r⟩` itself is carried by [`Self::phased_state_encoder`].
    ///
    /// # Panics
    ///
    /// Panics if `random_bits` has fewer than [`Self::random_outcome_count`] entries.
    #[must_use]
    pub fn output_phase_exponent(&self, random_bits: &[bool]) -> u8 {
        let n_random = self.random_outcome_count();
        assert!(
            random_bits.len() >= n_random,
            "random_bits is shorter than the number of random outcomes"
        );
        phase_form_exponent(
            n_random,
            |index| random_bits[index],
            |index| self.linear_i_phase.index(index),
            |index| self.linear_sign_phase.index(index),
            |row, column| self.quadratic_phase_matrix.get((row, column)),
        )
    }

    /// Sample measurement outcomes from all `2^{n_r}` branches.
    pub fn sample(&self, shots: usize) -> BitMatrix {
        let mut rng = rand::rng();
        self.sample_with_rng(shots, &mut rng)
    }

    /// Sample measurement outcomes using a provided random number generator.
    pub fn sample_with_rng<R: RngExt>(&self, num_shots: usize, rng: &mut R) -> BitMatrix {
        let num_outcomes = self.outcome_count();
        let num_random_bits = self.random_outcome_count();

        if num_outcomes == 0 {
            return BitMatrix::from_aligned(AlignedBitMatrix::zeros(num_shots, 0));
        }

        let random_matrix = AlignedBitMatrix::random_with_rng(num_shots, num_random_bits, rng);
        let outcome_matrix =
            AlignedBitMatrix::from_row_iter(self.outcome_matrix.row_iterator(0..num_outcomes), num_random_bits);
        let mut result = random_matrix.mul_transpose(&outcome_matrix);
        for shot in 0..num_shots {
            result.row_mut(shot).bitxor_assign(&self.outcome_shift.as_view());
        }
        BitMatrix::from_aligned(result)
    }

    fn ensure_qubit_capacity(&mut self, max_qubit_id: Option<usize>) {
        if let Some(max_qubit_id) = max_qubit_id {
            self.qubit_count = std::cmp::max(self.qubit_count, max_qubit_id + 1);
            if max_qubit_id >= self.qubit_capacity() {
                let new_capacity = (max_qubit_id + 1).next_power_of_two();
                self.reserve_qubits(new_capacity);
            }
        }
    }

    #[inline]
    fn ensure_outcome_capacity(&mut self, random_outcome: bool) {
        let mut new_outcome_capacity = self.outcome_capacity();
        let next_outcome_pos = self.outcome_count();
        if next_outcome_pos >= self.outcome_capacity() {
            new_outcome_capacity = (next_outcome_pos + 1).next_power_of_two();
        }

        let mut new_random_outcome_capacity = self.random_outcome_capacity();
        if random_outcome {
            let next_random_bit = self.random_outcome_count();
            if next_random_bit >= self.random_outcome_capacity() {
                new_random_outcome_capacity = (next_random_bit + 1).next_power_of_two();
            }
        }

        self.reserve_outcomes(new_outcome_capacity, new_random_outcome_capacity);
    }

    /// Applies a Pauli `P` conditioned on the random-bit parity `⟨indicator, r⟩`, updating the
    /// state matrix `A` and the phase data `B`, `p`, `s` accordingly.
    ///
    /// This realises step 4b of Algorithm 4.2: with preimage `i^l X^x Z^z = R† P R`,
    /// `A ← A + x indicator^T` and the conditional factor `(i^l (-1)^{⟨z, A r⟩})^⟨indicator, r⟩` is
    /// absorbed into the tracked phase:
    ///
    /// * the `(-1)^{⟨z, A r⟩ ⟨indicator, r⟩}` factor adds `indicator (z^T A)^T` to `B`;
    /// * for odd `l`, the extra `i^⟨indicator, r⟩` flips `p` by `indicator`, and its carry against the
    ///   existing i-phase, `(-1)^{⟨p, r⟩⟨indicator, r⟩}`, adds the outer product `p ⊗ indicator` to `B`;
    /// * for `l` with the two-bit set, the `(-1)^⟨indicator, r⟩` factor flips `s` by `indicator`.
    fn apply_pauli_conditioned_on_inner_random_bits<Bits: PauliBits, Phase: PhaseExponent>(
        &mut self,
        pauli: &PauliUnitary<Bits, Phase>,
        inner_bits_indicator: &AlignedBitVec,
    ) {
        let preimage = self.phased_clifford.clifford().preimage(pauli);
        let z_times_sign_matrix = row_sum(&self.sign_matrix, preimage.z_bits().support());
        for row in inner_bits_indicator.support() {
            self.quadratic_phase_matrix
                .row_mut(row)
                .bitxor_assign(&z_times_sign_matrix);
        }
        for x_bit_pos in preimage.x_bits().support() {
            self.sign_matrix.row_mut(x_bit_pos).bitxor_assign(inner_bits_indicator);
        }
        let l = preimage.xz_phase_exponent().value();
        if l & 1 == 1 {
            // Adding i^⟨indicator, r⟩ to the i-phase, which is tracked mod 2 in `p`. The carry
            // i·i = -1 from overlap with the existing i-phase is the quadratic cross term
            // (-1)^{⟨p, r⟩⟨indicator, r⟩}, absorbed into `B` as the outer product `p ⊗ indicator`.
            for row in self.linear_i_phase.support() {
                self.quadratic_phase_matrix
                    .row_mut(row)
                    .bitxor_assign(inner_bits_indicator);
            }
            self.linear_i_phase.bitxor_assign(inner_bits_indicator);
        }
        if l & 2 == 2 {
            self.linear_sign_phase.bitxor_assign(inner_bits_indicator);
        }
    }

    pub fn with_capacity(qubit_count: usize, outcome_count: usize, random_outcome_count: usize) -> Self {
        const MIN_CAPACITY: usize = CLIFFORD_BIT_ALIGNMENT;
        let outcome_capacity = outcome_count.max(MIN_CAPACITY);
        let random_capacity = random_outcome_count.max(MIN_CAPACITY);

        PhasedOutcomeCompleteSimulation {
            phased_clifford: PhasedCliffordUnitary::identity(qubit_count),
            sign_matrix: AlignedBitMatrix::zeros(qubit_count, random_capacity),
            quadratic_phase_matrix: AlignedBitMatrix::zeros(random_capacity, random_capacity),
            outcome_matrix: AlignedBitMatrix::zeros(outcome_capacity, random_capacity),
            outcome_shift: AlignedBitVec::zeros(outcome_capacity),
            linear_i_phase: AlignedBitVec::zeros(random_capacity),
            linear_sign_phase: AlignedBitVec::zeros(random_capacity),
            random_outcome_indicator: Vec::with_capacity(outcome_count),
            symbolic_angle_indicator: Vec::with_capacity(random_outcome_count),
            random_bit_count: 0,
            qubit_count,
        }
    }

    /// Measures a Pauli observable using an anti-commuting hint operator, tracking the exact phase.
    ///
    /// Implements case 5 of Algorithm 4.2. Given an anti-commuting hint `P'` with preimage
    /// `R† P' R = (-1)^α Z^{b'}`, the encoder is updated by `R ← e^{iπ/4 (i P' P)} R`, the quadratic
    /// and linear `-1` phases absorb the outcome-dependent stabiliser sign, and the `(-1)^α` sign
    /// relabels the reported outcome (`m = r ⊕ α`) via `outcome_shift` rather than a global phase.
    ///
    /// # Panics
    ///
    /// Panics if `hint` does not anti-commute with `observable`.
    pub fn measure_pauli_with_hint_generic<HintBits: PauliBits, HintPhase: PhaseExponent>(
        &mut self,
        observable: &SparsePauli,
        hint: &PauliUnitary<HintBits, HintPhase>,
    ) {
        assert!(
            anti_commutes_with(observable, hint),
            "observable={observable}, hint={hint}"
        );
        let preimage = self.phased_clifford.clifford().preimage(hint);
        if preimage.x_bits().support().next().is_some() {
            // hint is not a stabilizer of the encoded state family
            self.measure(observable);
        } else {
            // Ensure capacity for the new random bit before sizing the indicator vectors.
            self.ensure_outcome_capacity(true);

            // R <- (-1)^alpha e^{i pi/4 (i P' P)} R.
            // i P' P = -i P P', so the rotation Pauli is (observable * hint) with an i^3 = -i phase.
            let alpha = preimage.xz_phase_exponent().value() / 2;
            let mut rotation = observable.clone();
            rotation.mul_assign_right(hint);
            rotation.add_assign_phase_exp(3);
            self.phased_clifford.left_mul_pauli_exp(&rotation);

            // a = A^T b', with the new random bit appended: a_with_zero and a_with_one = a ⊕ {0,1}.
            let a_with_zero = row_sum(&self.sign_matrix, preimage.z_bits().support());
            let mut a_with_one = a_with_zero.clone();
            a_with_one.assign_index(self.random_bit_count, true);
            let new_random_bit = self.random_bit_count;
            self.allocate_random_bit();

            // B <- B + (a ⊕ 0)(a ⊕ 1)^T, s_{n(s)} <- s_{n(s)} + alpha.
            for row in a_with_zero.support() {
                self.quadratic_phase_matrix.row_mut(row).bitxor_assign(&a_with_one);
            }
            if alpha == 1 {
                self.linear_sign_phase
                    .assign_index(new_random_bit, !self.linear_sign_phase.index(new_random_bit));
                // (-1)^alpha relabels the reported outcome (m = r ⊕ alpha), not the global phase.
                let outcome_position = self.outcome_count() - 1;
                self.outcome_shift
                    .assign_index(outcome_position, !self.outcome_shift.index(outcome_position));
            }

            // Apply P' conditioned on the random bits indicated by (a ⊕ 1).
            self.apply_pauli_conditioned_on_inner_random_bits(hint, &a_with_one);
        }
    }

    fn measure_deterministic<Bits: PauliBits, Phase: PhaseExponent>(&mut self, preimage: &PauliUnitary<Bits, Phase>) {
        self.ensure_outcome_capacity(false);
        let outcome_matrix_row = row_sum(&self.sign_matrix, preimage.z_bits().support());
        let outcome_position = self.random_outcome_indicator.len();
        self.outcome_matrix
            .row_mut(outcome_position)
            .assign(&outcome_matrix_row);
        debug_assert!(preimage.xz_phase_exponent().is_even());
        if preimage.xz_phase_exponent().value() == 2 {
            self.outcome_shift.assign_index(outcome_position, true);
        }
        self.random_outcome_indicator.push(false);
    }

    /// Get the number of random (non-deterministic) measurement outcomes.
    #[must_use]
    pub fn random_outcome_count(&self) -> usize {
        self.random_bit_count
    }

    /// Get indicators for which outcomes are random.
    #[must_use]
    pub fn random_outcome_indicator(&self) -> &[bool] {
        &self.random_outcome_indicator
    }

    /// Get indicators for which random bits are symbolic rotation angles.
    ///
    /// The returned slice is indexed by random-bit index (`0..random_outcome_count()`). Entry `k` is
    /// `true` iff random bit `k` was allocated via [`Simulation::allocate_symbolic_angle`] (a virtual
    /// rotation parameter) rather than [`Simulation::allocate_random_bit`] or a genuine measurement.
    #[must_use]
    pub fn symbolic_angle_indicator(&self) -> &[bool] {
        &self.symbolic_angle_indicator
    }

    fn allocate_random_bit_with_provenance(&mut self, is_symbolic_angle: bool) -> usize {
        self.ensure_outcome_capacity(true);
        let outcome_pos = self.random_outcome_indicator.len();
        self.outcome_matrix
            .row_mut(outcome_pos)
            .assign_index(self.random_bit_count, true);
        self.random_outcome_indicator.push(true);
        self.symbolic_angle_indicator.push(is_symbolic_angle);
        self.random_bit_count += 1;
        self.random_bit_count - 1
    }
}

impl Simulation for PhasedOutcomeCompleteSimulation {
    fn allocate_random_bit(&mut self) -> usize {
        self.allocate_random_bit_with_provenance(false)
    }

    fn allocate_symbolic_angle(&mut self) -> usize {
        self.allocate_random_bit_with_provenance(true)
    }

    fn clifford(&mut self, _clifford: &crate::Unitary, _support: &[crate::QubitId]) {
        unimplemented!(
            "PhasedOutcomeCompleteSimulation tracks the exact global phase, which a phaseless \
             CliffordUnitary does not determine; apply Cliffords through unitary_op, pauli or \
             pauli_exp instead"
        );
    }

    fn unitary_op(&mut self, unitary_op: UnitaryOp, support: &[crate::QubitId]) {
        self.ensure_qubit_capacity(max_support(support));
        self.phased_clifford.left_mul(unitary_op, support);
    }

    fn permute(&mut self, permutation: &[usize], support: &[crate::QubitId]) {
        self.ensure_qubit_capacity(max_support(support));
        self.phased_clifford.left_mul_permutation(permutation, support);
    }

    fn controlled_pauli(&mut self, observable1: &SparsePauli, observable2: &SparsePauli) {
        self.ensure_qubit_capacity(max_pair_support(observable1, observable2));
        self.controlled_pauli_phase_resolved(observable1, observable2);
    }

    fn pauli(&mut self, observable: &SparsePauli) {
        self.ensure_qubit_capacity(observable.max_support());
        self.phased_clifford.left_mul_pauli(observable);
    }

    fn pauli_exp(&mut self, observable: &SparsePauli) {
        self.ensure_qubit_capacity(observable.max_support());
        self.phased_clifford.left_mul_pauli_exp(observable);
    }

    fn is_stabilizer_up_to_sign(&self, observable: &SparsePauli) -> bool {
        self.phased_clifford.clifford().preimage(observable).x_bits().is_zero()
    }

    fn qubit_count(&self) -> usize {
        self.qubit_count
    }

    fn conditional_pauli(&mut self, observable: &SparsePauli, outcomes: &[usize], parity: bool) {
        self.ensure_qubit_capacity(observable.max_support());
        let bit_indicator = outcomes.iter().copied().collect::<IndexSet>();
        let is_p_applied: bool = !parity ^ bit_indicator.dot(&self.outcome_shift);
        if is_p_applied {
            self.pauli(observable);
        }
        let inner_bits_indicator = row_sum(&self.outcome_matrix, outcomes);
        self.apply_pauli_conditioned_on_inner_random_bits(observable, &inner_bits_indicator);
    }

    fn is_stabilizer(&self, observable: &SparsePauli) -> bool {
        let preimage = self.phased_clifford.clifford().preimage(observable);
        if preimage.x_bits().is_zero() {
            let sign_parity_indicator = row_sum(&self.sign_matrix, preimage.z_bits().support());
            sign_parity_indicator.is_zero()
        } else {
            false
        }
    }

    fn is_stabilizer_with_conditional_sign(&self, observable: &SparsePauli, outcomes: &[crate::OutcomeId]) -> bool {
        let preimage = self.phased_clifford.clifford().preimage(observable);
        if preimage.x_bits().is_zero() {
            let sign_parity_indicator = row_sum(&self.sign_matrix, preimage.z_bits().support());
            debug_assert!(preimage.xz_phase_exponent().is_even());
            let shift = preimage.xz_phase_exponent().value() / 2 == 1;
            let expected_parity_indicator = row_sum(&self.outcome_matrix, outcomes.iter().copied());
            let expected_shift = outcomes
                .iter()
                .copied()
                .map(|o| self.outcome_shift.index(o))
                .fold(false, |acc, v| acc ^ v);
            (sign_parity_indicator == expected_parity_indicator) && (shift == expected_shift)
        } else {
            false
        }
    }

    fn measure(&mut self, observable: &SparsePauli) -> usize {
        self.ensure_qubit_capacity(observable.max_support());
        let preimage = self.phased_clifford.clifford().preimage(observable);
        let non_zero_pos = preimage.x_bits().support().next();
        match non_zero_pos {
            Some(pos) => {
                let hint = self.phased_clifford.clifford().image_z(pos);
                self.measure_pauli_with_hint_generic(observable, &hint);
            }
            None => {
                self.measure_deterministic(&preimage);
            }
        }
        self.outcome_count() - 1
    }

    fn measure_with_hint(&mut self, observable: &SparsePauli, hint: &SparsePauli) -> usize {
        self.ensure_qubit_capacity(max_pair_support(observable, hint));
        self.measure_pauli_with_hint_generic(observable, hint);
        self.outcome_count() - 1
    }

    fn outcome_count(&self) -> usize {
        self.random_outcome_indicator.len()
    }

    fn with_capacity(qubit_count: usize, outcome_count: usize, random_outcome_count: usize) -> Self
    where
        Self: Sized,
    {
        PhasedOutcomeCompleteSimulation::with_capacity(qubit_count, outcome_count, random_outcome_count)
    }

    fn qubit_capacity(&self) -> usize {
        debug_assert_eq!(self.phased_clifford.num_qubits(), self.sign_matrix.row_count());
        self.phased_clifford.num_qubits()
    }

    fn outcome_capacity(&self) -> usize {
        self.outcome_matrix.row_count()
    }

    fn random_outcome_capacity(&self) -> usize {
        debug_assert_eq!(self.outcome_matrix.column_count(), self.sign_matrix.column_count());
        self.outcome_matrix.column_count()
    }

    fn reserve_qubits(&mut self, new_capacity: usize) {
        if new_capacity > self.qubit_capacity() {
            self.sign_matrix.resize(new_capacity, self.sign_matrix.column_count());
            self.phased_clifford.resize(new_capacity);
        }
    }

    fn reserve_outcomes(&mut self, new_outcome_capacity: usize, new_random_outcome_capacity: usize) {
        assert!(
            new_outcome_capacity >= new_random_outcome_capacity,
            "outcome capacity must be at least random outcome capacity"
        );
        let new_outcome_capacity = new_outcome_capacity.max(self.outcome_capacity());
        let new_random_outcome_capacity = new_random_outcome_capacity.max(self.random_outcome_capacity());

        self.outcome_matrix
            .resize(new_outcome_capacity, new_random_outcome_capacity);
        self.sign_matrix
            .resize(self.sign_matrix.row_count(), new_random_outcome_capacity);
        self.quadratic_phase_matrix
            .resize(new_random_outcome_capacity, new_random_outcome_capacity);
        if self.outcome_shift.len() < new_outcome_capacity {
            self.outcome_shift.resize(new_outcome_capacity);
        }
        if self.linear_i_phase.len() < new_random_outcome_capacity {
            self.linear_i_phase.resize(new_random_outcome_capacity);
        }
        if self.linear_sign_phase.len() < new_random_outcome_capacity {
            self.linear_sign_phase.resize(new_random_outcome_capacity);
        }
    }
}

impl PhasedOutcomeCompleteSimulation {
    /// Applies the controlled-Pauli `Λ(observable1, observable2)` to the encoder, phase-resolved.
    ///
    /// For commuting Hermitian involutions `P₁`, `P₂` the controlled-Pauli is
    /// `Λ(P₁, P₂) = (I + P₁)/2 + (I − P₁)/2 · P₂ = exp(iπ/4 · (I − P₁)(I − P₂))`. Since the factors
    /// commute this is
    ///
    /// ```text
    ///     Λ(P₁, P₂) = e^{iπ/4} · e^{-iπ/4 P₁} · e^{-iπ/4 P₂} · e^{iπ/4 P₁P₂},
    /// ```
    ///
    /// each factor of which is applied through the phase-exact primitive, so the global phase of the
    /// encoder is tracked exactly while the symplectic action matches
    /// [`paulimer::clifford::CliffordMutable::left_mul_controlled_pauli`].
    fn controlled_pauli_phase_resolved(&mut self, observable1: &SparsePauli, observable2: &SparsePauli) {
        debug_assert!(
            paulimer::pauli::commutes_with(observable1, observable2),
            "controlled_pauli requires commuting observables"
        );
        let mut negated1 = observable1.clone();
        negated1.add_assign_phase_exp(2);
        let mut negated2 = observable2.clone();
        negated2.add_assign_phase_exp(2);
        let mut product = observable1.clone();
        product.mul_assign_right(observable2);

        self.phased_clifford.left_mul_pauli_exp(&negated1);
        self.phased_clifford.left_mul_pauli_exp(&negated2);
        self.phased_clifford.left_mul_pauli_exp(&product);
        self.phased_clifford.left_mul_global_phase(1);
    }
}
