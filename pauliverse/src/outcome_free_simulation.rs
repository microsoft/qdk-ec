use binar::Bitwise;
use paulimer::{
    clifford::{Clifford, CliffordMutable, CliffordUnitaryModPauli, MutablePreImages, PreimageViews},
    pauli::{
        Pauli, PauliBinaryOps, PauliBits, PauliMutable, PauliUnitaryProjective, SparsePauliProjective,
        anti_commutes_with,
    },
};

use crate::Simulation;

type SparsePauli = paulimer::pauli::SparsePauli;
type OutcomeId = crate::circuit::OutcomeId;
type QubitId = crate::circuit::QubitId;
type Operation = paulimer::UnitaryOp;
type Unitary = paulimer::clifford::CliffordUnitary;

/// Stabilizer simulation without tracking specific measurement outcomes.
///
/// This simulator tracks the quantum state evolution through measurements without
/// committing to specific outcome values. Measurements update the stabilizer state, modulo
/// Paulis, but outcomes remain unspecified, which permits faster simulation.
///
/// The state is represented purely by a Clifford unitary (modulo Paulis) that gets
/// updated after each measurement without branching or recording outcome values.
///
/// # Use Cases
///
/// - **Circuit validation**: Verify stabilizer evolution, up to signs.
///
/// # Performance
///
/// - **Time**: `O(n_gates × n_qubits²)`
/// - **Space**: `O(n_qubits²)`
/// - **Most lightweight**: No sign tracking overhead
///
/// # Examples
///
/// ```
/// use pauliverse::{OutcomeFreeSimulation, Simulation};
/// use paulimer::{UnitaryOp, SparsePauli};
///
/// let mut sim = OutcomeFreeSimulation::new(3);
/// sim.unitary_op(UnitaryOp::Hadamard, &[0]);
/// sim.unitary_op(UnitaryOp::ControlledX, &[0, 1]);
///
/// let observable: SparsePauli = "ZII".parse().unwrap();
/// sim.measure(&observable);
///
/// // Query stabilizers without caring about the outcome value
/// let test_pauli: SparsePauli = "ZII".parse().unwrap();
/// assert!(sim.is_stabilizer(&test_pauli));
///
/// let test_pauli2: SparsePauli = "XII".parse().unwrap();
/// assert!(!sim.is_stabilizer(&test_pauli2));
/// ```
///
/// # Alternatives
///
/// - Use [`crate::OutcomeSpecificSimulation`] when you need concrete outcome values
/// - Use [`crate::OutcomeCompleteSimulation`] to track all possible outcomes
/// - Use [`crate::FaultySimulation`] for noisy simulations
#[must_use]
#[derive(Debug)]
pub struct OutcomeFreeSimulation {
    clifford: CliffordUnitaryModPauli,
    random_outcome_indicator: Vec<bool>, // vec(p), [j] is true iff vec(p)_j = 1/2
    random_bit_count: usize,
    qubit_count: usize,
}

impl Default for OutcomeFreeSimulation {
    fn default() -> Self {
        OutcomeFreeSimulation::with_capacity(0, 0, 0)
    }
}

impl OutcomeFreeSimulation {
    /// Measure a Pauli observable with an anticommuting hint for optimization.
    ///
    /// # Panics
    /// Panics if `hint` commutes with `observable`
    pub fn measure_with_hint_generic<HintBits: PauliBits>(
        &mut self,
        observable: &SparsePauliProjective,
        hint: &PauliUnitaryProjective<HintBits>,
    ) {
        assert!(
            anti_commutes_with(observable, hint),
            "observable={observable}, hint={hint}"
        );
        let preimage = self.clifford.preimage(hint);
        if preimage.x_bits().support().next().is_some() {
            // hint is not true
            self.measure_projective(observable);
        } else {
            let mut pauli: SparsePauliProjective = observable.clone();
            pauli.mul_assign_right(hint);
            self.clifford.left_mul_pauli_exp(&pauli);
            self.allocate_random_bit();
        }
    }

    /// Measure a projective Pauli observable without tracking the outcome value.
    ///
    /// Updates the stabilizer state but doesn't record the specific outcome.
    /// Returns the outcome ID (which indicates a measurement occurred).
    pub fn measure_projective(&mut self, observable: &SparsePauliProjective) -> OutcomeId {
        let preimage = self.clifford.preimage(observable);
        let non_zero_pos = preimage.x_bits().support().next();
        match non_zero_pos {
            Some(pos) => {
                self.allocate_random_bit();
                update_encoder_for_random_outcome(&mut self.clifford, observable, preimage, pos, false);
            }
            None => {
                self.random_outcome_indicator.push(false);
            }
        }
        self.outcome_count() - 1
    }

    /// Get the Clifford unitary encoding the current stabilizer state.
    ///
    /// Returns the unitary modulo Pauli operators (since outcome values aren't tracked).
    pub fn state_encoder(&self) -> CliffordUnitaryModPauli {
        let mut res = self.clifford.clone();
        res.resize(self.qubit_count);
        res
    }

    /// Get the number of random (non-deterministic) measurement outcomes.
    #[must_use]
    pub fn random_outcome_count(&self) -> usize {
        self.random_bit_count
    }

    /// Get indicators for which outcomes are random.
    ///
    /// Returns a slice where `[i]` is true if outcome `i` was random.
    #[must_use]
    pub fn random_outcome_indicator(&self) -> &[bool] {
        &self.random_outcome_indicator
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
}

impl Simulation for OutcomeFreeSimulation {
    fn allocate_random_bit(&mut self) -> OutcomeId {
        self.random_bit_count += 1;
        self.random_outcome_indicator.push(true);
        self.random_outcome_indicator.len() - 1
    }

    fn reserve_qubits(&mut self, new_capacity: usize) {
        if new_capacity > self.qubit_capacity() {
            self.clifford.resize(new_capacity);
        }
    }

    fn clifford(&mut self, clifford: &Unitary, support: &[QubitId]) {
        self.clifford.left_mul_clifford(clifford.as_ref(), support);
    }

    fn conditional_pauli(&mut self, _observable: &SparsePauli, _outcomes: &[OutcomeId], _parity: bool) {}

    fn controlled_pauli(&mut self, observable1: &SparsePauli, observable2: &SparsePauli) {
        self.ensure_qubit_capacity(max_pair_support(observable1, observable2));
        self.clifford
            .left_mul_controlled_pauli(observable1.as_ref(), observable2.as_ref());
    }

    fn pauli(&mut self, _observable: &SparsePauli) {}

    fn pauli_exp(&mut self, sparse_pauli: &SparsePauli) {
        self.ensure_qubit_capacity(sparse_pauli.max_support());
        self.clifford.left_mul_pauli_exp(sparse_pauli.as_ref());
    }

    fn permute(&mut self, permutation: &[usize], support: &[QubitId]) {
        self.ensure_qubit_capacity(max_support(support));
        self.clifford.left_mul_permutation(permutation, support);
    }

    fn unitary_op(&mut self, operation: Operation, support: &[QubitId]) {
        self.ensure_qubit_capacity(max_support(support));
        self.clifford.left_mul(operation, support);
    }

    /// Sign-blind: this simulator tracks the state only modulo Pauli operators, so
    /// `P` and `-P` are indistinguishable. Equivalent to
    /// [`Simulation::is_stabilizer_up_to_sign`].
    fn is_stabilizer(&self, observable: &SparsePauli) -> bool {
        self.is_stabilizer_up_to_sign(observable)
    }

    fn is_stabilizer_up_to_sign(&self, observable: &SparsePauli) -> bool {
        self.clifford.preimage(observable.as_ref()).x_bits().is_zero()
    }

    fn is_stabilizer_with_conditional_sign(&self, observable: &SparsePauli, _outcomes: &[OutcomeId]) -> bool {
        self.is_stabilizer_up_to_sign(observable)
    }

    fn measure(&mut self, observable: &SparsePauli) -> OutcomeId {
        self.ensure_qubit_capacity(observable.max_support());
        self.measure_projective(observable.as_ref());
        self.outcome_count() - 1
    }

    fn measure_with_hint(&mut self, observable: &SparsePauli, anti_commuting_stabilizer: &SparsePauli) -> OutcomeId {
        self.ensure_qubit_capacity(observable.max_support());
        self.measure_with_hint_generic(observable.as_ref(), anti_commuting_stabilizer.as_ref());
        self.outcome_count() - 1
    }

    fn outcome_count(&self) -> usize {
        self.random_outcome_indicator.len()
    }

    fn with_capacity(qubit_count: usize, outcome_count: usize, _random_outcome_count: usize) -> Self
    where
        Self: Sized,
    {
        OutcomeFreeSimulation {
            clifford: CliffordUnitaryModPauli::identity(qubit_count),
            random_outcome_indicator: Vec::with_capacity(outcome_count),
            random_bit_count: 0,
            qubit_count,
        }
    }

    fn qubit_count(&self) -> usize {
        self.qubit_count
    }

    fn qubit_capacity(&self) -> usize {
        self.clifford.num_qubits()
    }

    fn outcome_capacity(&self) -> usize {
        self.random_outcome_indicator.capacity()
    }

    fn random_outcome_capacity(&self) -> usize {
        self.outcome_capacity()
    }

    fn reserve_outcomes(&mut self, new_outcome_capacity: usize, new_random_outcome_capacity: usize) {
        if new_outcome_capacity > self.outcome_capacity()
            || new_random_outcome_capacity > self.random_outcome_capacity()
        {
            let new_capacity = new_outcome_capacity.max(new_random_outcome_capacity);
            self.random_outcome_indicator
                .reserve(new_capacity - self.random_outcome_indicator.capacity());
        }
    }
}

pub(crate) fn max_support(support: &[usize]) -> Option<usize> {
    support.iter().copied().max()
}

pub(crate) fn max_pair_support<PauliLike1: Pauli, PauliLike2: Pauli>(a: &PauliLike1, b: &PauliLike2) -> Option<usize> {
    match (a.max_support(), b.max_support()) {
        (None, None) => None,
        (Some(id), None) | (None, Some(id)) => Some(id),
        (Some(id1), Some(id2)) => Some(std::cmp::max(id1, id2)),
    }
}

/// Updates `encoder` for a measurement of `observable` with random outcome `outcome`, given the `preimage` of
/// `observable` under `encoder` and a qubit `pivot` on which `preimage` has an X or Y component.
///
/// The stabilizer `image_z(pivot)` anticommutes with `observable`. The preimage of each generator that anticommutes
/// with the product of `observable` and `image_z(pivot)` is multiplied on the right by `preimage · Z_pivot`. When
/// `outcome` is true, the preimage of each generator that anticommutes with `image_z(pivot)` is negated. A generator
/// anticommutes with `image_z(pivot)` exactly when its preimage has an X or Y component on `pivot`. Only generators
/// that anticommute with `image_z(pivot)` or with `observable` are visited, because the others are unchanged.
pub(crate) fn update_encoder_for_random_outcome<Encoder>(
    encoder: &mut Encoder,
    observable: &impl Pauli,
    preimage: Encoder::DensePauli,
    pivot: usize,
    outcome: bool,
) where
    Encoder: Clifford + MutablePreImages + PreimageViews,
    for<'life> Encoder::PreImageViewMut<'life>: PauliBinaryOps<Encoder::DensePauli>,
{
    let mut factor = preimage;
    factor.mul_assign_right_z(pivot);
    let (x_generators, z_generators) = {
        let z_image_support = encoder.z_image_view_up_to_phase(pivot);
        (
            support_union(z_image_support.z_bits(), observable.z_bits(), encoder.num_qubits()),
            support_union(z_image_support.x_bits(), observable.x_bits(), encoder.num_qubits()),
        )
    };
    for qubit in x_generators {
        let anticommutes_with_observable = observable.z_bits().index(qubit);
        let mut x_preimage = encoder.preimage_x_view_mut(qubit);
        update_generator_preimage(&mut x_preimage, &factor, pivot, anticommutes_with_observable, outcome);
    }
    for qubit in z_generators {
        let anticommutes_with_observable = observable.x_bits().index(qubit);
        let mut z_preimage = encoder.preimage_z_view_mut(qubit);
        update_generator_preimage(&mut z_preimage, &factor, pivot, anticommutes_with_observable, outcome);
    }
}

fn support_union(first: &impl Bitwise, second: &impl Bitwise, capacity: usize) -> Vec<usize> {
    let mut indices = Vec::with_capacity(capacity);
    indices.extend(first.support());
    indices.extend(second.support().filter(|&index| !first.index(index)));
    indices
}

fn update_generator_preimage<Factor: Pauli>(
    generator_preimage: &mut impl PauliBinaryOps<Factor>,
    factor: &Factor,
    pivot: usize,
    anticommutes_with_observable: bool,
    outcome: bool,
) {
    let anticommutes_with_z_image = generator_preimage.x_bits().index(pivot);
    if anticommutes_with_z_image != anticommutes_with_observable {
        generator_preimage.mul_assign_right(factor);
    }
    if anticommutes_with_z_image && outcome {
        generator_preimage.negate();
    }
}
