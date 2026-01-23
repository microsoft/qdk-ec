use crate::outcome_free_simulation::{max_pair_support, max_support};
use crate::Simulation;
use binar::{matrix::AlignedBitMatrix, vec::AlignedBitVec, Bitwise, BitwiseMut, BitwisePair, BitwisePairMut, IndexSet};
use binar::{BitMatrix, BitVec};
use paulimer::clifford::{Clifford, CliffordMutable, CliffordUnitary};
use paulimer::pauli::{anti_commutes_with, generic::PhaseExponent, Pauli, PauliBits, PauliUnitary};
use paulimer::pauli::{PauliBinaryOps, PauliMutable};
use paulimer::{UnitaryOp, CLIFFORD_BIT_ALIGNMENT};
use rand::{thread_rng, Rng};
use std::borrow::Borrow;

type SparsePauli = paulimer::pauli::SparsePauli;

/// Asymptotically efficient stabilizer simulation tracking all measurement outcomes.
///
/// Instead of running separate simulations for each possible measurement outcome,
/// this simulator tracks all `2^n_random` outcome branches simultaneously where `n_random` is the
/// number of random measurements. This admits an asymptotic improvement over outcome-specific
/// simulation for many use cases.
///
/// # Use Cases
///
/// - **Exhaustive enumeration**: Compute quantities over all possible outcomes
/// - **Exact probability distributions**: Calculate measurement statistics without sampling
/// - **Circuit verification**: Analyze complete behavior across all measurement branches
/// - **Outcome codes**: Study encoding/decoding that depends on measurement outcomes
///
/// # Performance
///
/// - **Complexity**: `O(n_gates × n_qubits²)` worst-case, like other simulators
/// - **Key advantage**: Simulate once, then sample any number of shots efficiently
/// - **Compared to `OutcomeSpecific`**: Saves a factor of `n_random` (number of random measurements)
///   when collecting many samples, since the circuit isn't re-executed per shot
/// - **Sampling cost**: `O(shots × n_random)` to generate outcome samples after simulation
/// - **Space**: `O(n_qubits² + n_random²)` for sign and outcome matrices
///
/// # Theory
///
/// A circuit with `n_random` with random outcomes has a (worst-case) `2^n_random` possible execution paths.
/// This simulator represents all branches implicitly using:
/// - A Clifford unitary encoding the stabilizer state
/// - A sign matrix tracking measurement outcome correlations  
/// - An outcome matrix encoding how outcomes depend on `n_random` random bits
/// - A deterministic outcome shift vector
///
/// The simulation cost is linear in `n_random`, not exponential. The `2^n_random` outcomes are
/// represented compactly and can be sampled efficiently.
///
/// # Examples
///
/// ```
/// use pauliverse::{OutcomeCompleteSimulation, Simulation};
/// use paulimer::{UnitaryOp, SparsePauli};
///
/// let mut sim = OutcomeCompleteSimulation::new(3);
/// sim.unitary_op(UnitaryOp::Hadamard, &[0]);
/// sim.unitary_op(UnitaryOp::ControlledX, &[0, 1]);
///
/// let observable: SparsePauli = "ZII".parse().unwrap();
/// sim.measure(&observable);
///
/// // All 2^n_random branches tracked without separate simulation runs
/// let num_branches = 1 << sim.random_outcome_count();
/// println!("Tracking {} outcome branches", num_branches);
/// ```
///
/// # Alternatives
///
/// - Use [`crate::OutcomeSpecificSimulation`] for Monte Carlo sampling
/// - Use [`crate::OutcomeFreeSimulation`] when outcomes don't matter
/// - Use [`crate::FaultySimulation`] for noisy simulations
#[must_use]
///
/// Outcome-complete stabilizer simulation implementation.
/// See <https://arxiv.org/pdf/2309.08676#page=27> for details.
pub struct OutcomeCompleteSimulation {
    clifford: CliffordUnitary,           // R
    sign_matrix: AlignedBitMatrix,       // A
    outcome_matrix: AlignedBitMatrix,    // M
    outcome_shift: AlignedBitVec,        // v_0
    random_outcome_indicator: Vec<bool>, // vec(p), [j] is true iff vec(p)_j = 1/2
    random_bit_count: usize,
    qubit_count: usize,
}

impl std::fmt::Debug for OutcomeCompleteSimulation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutcomeCompleteSimulation")
            .field("state_encoder", &self.clifford.clone().resize(self.qubit_count))
            .field("sign_matrix", &self.sign_matrix())
            .field("outcome_matrix", &self.outcome_matrix())
            .field("outcome_shift", &self.outcome_shift().iter().collect::<Vec<bool>>())
            .field("random_outcome_indicator", &self.random_outcome_indicator)
            .field("random_bit_count", &self.random_bit_count)
            .field("qubit_count", &self.qubit_count)
            .finish()
    }
}

impl Default for OutcomeCompleteSimulation {
    fn default() -> Self {
        OutcomeCompleteSimulation::with_capacity(0, 0, 0)
    }
}

impl OutcomeCompleteSimulation {
    /// Get the Clifford unitary encoding the current stabilizer state.
    ///
    /// This is the unitary R such that R|0⟩ represents the stabilizer state.
    pub fn state_encoder(&self) -> CliffordUnitary {
        let mut res = self.clifford.clone();
        res.resize(self.qubit_count);
        res
    }

    /// Get the sign matrix tracking measurement outcome correlations.
    ///
    /// The sign matrix A encodes how Pauli signs depend on random outcomes.
    /// Returns a cache-aligned reference for efficiency.
    pub fn aligned_sign_matrix(&self) -> &AlignedBitMatrix {
        &self.sign_matrix
    }

    /// Get a copy of the sign matrix without alignment constraints.
    pub fn sign_matrix(&self) -> BitMatrix {
        BitMatrix::from_aligned(AlignedBitMatrix::from_row_iter(
            self.sign_matrix.row_iterator(0..self.qubit_count()),
            self.random_outcome_count(),
        ))
    }

    /// Get the outcome matrix encoding all 2^k measurement branches.
    ///
    /// Each row corresponds to a measurement outcome, each column to a random bit.
    /// Returns a cache-aligned reference for efficiency.
    pub fn aligned_outcome_matrix(&self) -> &AlignedBitMatrix {
        &self.outcome_matrix
    }

    /// Get a copy of the outcome matrix without alignment constraints.
    pub fn outcome_matrix(&self) -> BitMatrix {
        BitMatrix::from_aligned(AlignedBitMatrix::from_row_iter(
            self.outcome_matrix.row_iterator(0..self.outcome_count()),
            self.random_outcome_count(),
        ))
    }

    /// Get the outcome shift vector (deterministic outcome values).
    ///
    /// Returns a cache-aligned reference for efficiency.
    pub fn aligned_outcome_shift(&self) -> &AlignedBitVec {
        &self.outcome_shift
    }

    /// Get a copy of the outcome shift vector without alignment constraints.
    pub fn outcome_shift(&self) -> BitVec {
        BitVec::from_aligned(self.outcome_count(), self.outcome_shift.clone())
    }

    /// Sample measurement outcomes from all 2^k branches.
    ///
    /// Each shot corresponds to one random selection of the `n_random` random bits.
    /// Returns a matrix where each row is one shot's outcome vector.
    pub fn sample(&self, shots: usize) -> BitMatrix {
        let mut rng = thread_rng();
        self.sample_with_rng(shots, &mut rng)
    }

    /// Sample measurement outcomes using a provided random number generator.
    ///
    /// Useful for reproducible sampling with a seeded RNG.
    pub fn sample_with_rng<R: Rng>(&self, num_shots: usize, rng: &mut R) -> BitMatrix {
        let num_outcomes = self.outcome_count();
        let num_random_bits = self.random_outcome_count();

        if num_outcomes == 0 {
            return BitMatrix::from_aligned(AlignedBitMatrix::zeros(num_shots, 0));
        }

        // Generate all random bits at once as a matrix (num_shots × num_random_bits)
        let random_matrix = AlignedBitMatrix::random_with_rng(num_shots, num_random_bits, rng);

        // Extract the active portion of outcome_matrix (it may have extra capacity)
        let outcome_matrix =
            AlignedBitMatrix::from_row_iter(self.outcome_matrix.row_iterator(0..num_outcomes), num_random_bits);

        // Compute result = random_matrix * outcome_matrix^T using M4RI
        // This precomputes lookup tables for efficient parallel XOR
        let mut result = random_matrix.mul_transpose(&outcome_matrix);

        // XOR each row with outcome_shift
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

    /// Ensure matrices have enough capacity for the next measurement.
    /// Automatically resizes if needed, doubling capacity each time.
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

    fn apply_pauli_conditioned_on_inner_random_bits<Bits: PauliBits, Phase: PhaseExponent>(
        &mut self,
        pauli: &PauliUnitary<Bits, Phase>,
        inner_bits_indicator: &AlignedBitVec,
    ) {
        let preimage = self.clifford.preimage(pauli);
        for x_bit_pos in preimage.x_bits().support() {
            self.sign_matrix.row_mut(x_bit_pos).bitxor_assign(inner_bits_indicator);
        }
    }

    pub fn with_capacity(qubit_count: usize, outcome_count: usize, random_outcome_count: usize) -> Self {
        const MIN_CAPACITY: usize = CLIFFORD_BIT_ALIGNMENT;
        let outcome_capacity = outcome_count.max(MIN_CAPACITY);
        let random_capacity = random_outcome_count.max(MIN_CAPACITY);

        OutcomeCompleteSimulation {
            clifford: CliffordUnitary::identity(qubit_count),
            outcome_matrix: AlignedBitMatrix::zeros(outcome_capacity, random_capacity),
            sign_matrix: AlignedBitMatrix::zeros(qubit_count, random_capacity),
            outcome_shift: AlignedBitVec::zeros(outcome_capacity),
            random_outcome_indicator: Vec::with_capacity(outcome_count),
            random_bit_count: 0,
            qubit_count,
        }
    }

    /// Measures a Pauli observable using an anti-commuting hint operator.
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
        let preimage = self.clifford.preimage(hint);
        if preimage.x_bits().support().next().is_some() {
            // hint is not true
            self.measure(observable);
        } else {
            // Ensure we have capacity for the next random bit before creating
            // random_bits_indicator, since it's sized to sign_matrix.column_count()
            self.ensure_outcome_capacity(true);
            let mut pauli = observable.clone();
            pauli.mul_assign_right(hint);
            pauli.add_assign_phase_exp(3u8.wrapping_sub(preimage.xz_phase_exponent().raw_value()));
            self.pauli_exp(&pauli);
            let mut random_bits_indicator = row_sum(&self.sign_matrix, preimage.z_bits().support());
            random_bits_indicator.assign_index(self.random_bit_count, true);
            self.allocate_random_bit();
            self.apply_pauli_conditioned_on_inner_random_bits(hint, &random_bits_indicator);
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
}

#[macro_export]
macro_rules! implement_common_simulation_methods {
    () => {
        fn clifford(&mut self, clifford: &$crate::Unitary, support: &[$crate::QubitId]) {
            self.ensure_qubit_capacity(max_support(support));
            self.clifford.left_mul_clifford(clifford, support);
        }

        fn unitary_op(&mut self, unitary_op: UnitaryOp, support: &[$crate::QubitId]) {
            self.ensure_qubit_capacity(max_support(support));
            let clifford = &mut self.clifford;
            clifford.left_mul(unitary_op, support);
        }

        fn permute(&mut self, permutation: &[usize], support: &[$crate::QubitId]) {
            self.ensure_qubit_capacity(max_support(support));
            self.clifford.left_mul_permutation(permutation, support);
        }

        fn controlled_pauli(&mut self, observable1: &$crate::Pauli, observable2: &SparsePauli) {
            self.ensure_qubit_capacity(max_pair_support(observable1, observable2));
            self.clifford.left_mul_controlled_pauli(observable1, observable2);
        }

        fn pauli(&mut self, observable: &$crate::Pauli) {
            self.ensure_qubit_capacity(observable.max_support());
            self.clifford.left_mul_pauli(observable);
        }

        fn pauli_exp(&mut self, observable: &$crate::Pauli) {
            self.ensure_qubit_capacity(observable.max_support());
            self.clifford.left_mul_pauli_exp(observable);
        }

        fn is_stabilizer_up_to_sign(&self, observable: &$crate::Pauli) -> bool {
            self.clifford.preimage(observable).x_bits().is_zero()
        }

        fn qubit_count(&self) -> usize {
            self.qubit_count
        }
    };
}

impl Simulation for OutcomeCompleteSimulation {
    fn allocate_random_bit(&mut self) -> usize {
        self.ensure_outcome_capacity(true);
        let outcome_pos = self.random_outcome_indicator.len();
        self.outcome_matrix
            .row_mut(outcome_pos)
            .assign_index(self.random_bit_count, true);
        self.random_outcome_indicator.push(true);
        self.random_bit_count += 1;
        self.random_bit_count - 1
    }

    implement_common_simulation_methods!();

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
        let preimage = self.clifford.preimage(observable);
        if preimage.x_bits().is_zero() {
            let sign_parity_indicator = row_sum(&self.sign_matrix, preimage.z_bits().support());
            sign_parity_indicator.is_zero()
        } else {
            false
        }
    }

    fn is_stabilizer_with_conditional_sign(&self, observable: &SparsePauli, outcomes: &[crate::OutcomeId]) -> bool {
        let preimage = self.clifford.preimage(observable);
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
        let preimage = self.clifford.preimage(observable);
        let non_zero_pos = preimage.x_bits().support().next();
        match non_zero_pos {
            Some(pos) => {
                let hint = self.clifford.image_z(pos);
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

    fn random_outcome_count(&self) -> usize {
        self.random_bit_count
    }

    fn random_outcome_indicator(&self) -> &[bool] {
        &self.random_outcome_indicator
    }

    fn outcome_count(&self) -> usize {
        self.random_outcome_indicator.len()
    }

    fn with_capacity(qubit_count: usize, outcome_count: usize, random_outcome_count: usize) -> Self
    where
        Self: Sized,
    {
        OutcomeCompleteSimulation::with_capacity(qubit_count, outcome_count, random_outcome_count)
    }

    fn qubit_capacity(&self) -> usize {
        debug_assert_eq!(self.clifford.num_qubits(), self.sign_matrix.row_count());
        self.clifford.num_qubits()
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
            self.clifford.resize(new_capacity);
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
        if self.outcome_shift.len() < new_outcome_capacity {
            self.outcome_shift.resize(new_outcome_capacity);
        }
    }
}

pub fn row_sum<Index>(matrix: &AlignedBitMatrix, rows_to_sum: impl IntoIterator<Item = Index>) -> AlignedBitVec
where
    Index: Borrow<usize>,
{
    let mut res = AlignedBitVec::zeros(matrix.column_count());
    for row_id in rows_to_sum {
        res.bitxor_assign(&matrix.row(*row_id.borrow()));
    }
    res
}
