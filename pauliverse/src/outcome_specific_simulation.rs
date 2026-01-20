use crate::outcome_free_simulation::{max_pair_support, max_support};
use crate::{OutcomeId, Simulation};
use binar::Bitwise;
use paulimer::clifford::{Clifford, CliffordMutable, CliffordUnitary};
use paulimer::pauli::{anti_commutes_with, generic::PhaseExponent, Pauli, PauliBits, PauliUnitary};
use paulimer::pauli::{PauliBinaryOps, PauliMutable};
use paulimer::UnitaryOp;
use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};

type SparsePauli = paulimer::pauli::SparsePauli;

#[must_use]
pub struct OutcomeSpecificSimulation {
    clifford: CliffordUnitary, // R
    outcome_vector: Vec<bool>,
    bit_source: Box<dyn Iterator<Item = bool> + Send + Sync>,
    random_outcome_indicator: Vec<bool>, // vec(p), [j] is true iff vec(p)_j = 1/2
    num_random_bits: usize,
    qubit_count: usize,
}

impl std::fmt::Debug for OutcomeSpecificSimulation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutcomeSpecificSimulation")
            .field("state_encoder", &self.clifford.clone().resize(self.qubit_count))
            .field("outcome_vector", &self.outcome_vector)
            .field("bit_source", &"<bit source iterator>")
            .field("random_outcome_indicator", &self.random_outcome_indicator)
            .field("num_random_bits", &self.num_random_bits)
            .field("qubit_count", &self.qubit_count)
            .finish()
    }
}

impl Default for OutcomeSpecificSimulation {
    fn default() -> Self {
        OutcomeSpecificSimulation::with_capacity(0, 0, 0)
    }
}

impl OutcomeSpecificSimulation {
    fn ensure_qubit_capacity(&mut self, max_qubit_id: Option<usize>) {
        if let Some(max_qubit_id) = max_qubit_id {
            self.qubit_count = std::cmp::max(self.qubit_count, max_qubit_id + 1);
            if max_qubit_id >= self.qubit_capacity() {
                let new_capacity = (max_qubit_id + 1).next_power_of_two();
                self.reserve_qubits(new_capacity);
            }
        }
    }

    pub fn new_with_bit_source(
        num_qubits: usize,
        bit_source: impl Iterator<Item = bool> + 'static + Send + Sync,
    ) -> Self {
        OutcomeSpecificSimulation {
            clifford: CliffordUnitary::identity(num_qubits),
            outcome_vector: Vec::new(),
            bit_source: Box::new(bit_source),
            random_outcome_indicator: Vec::new(),
            num_random_bits: 0,
            qubit_count: num_qubits,
        }
    }

    pub fn with_bit_source_and_capacity(
        num_qubits: usize,
        bit_source: impl Iterator<Item = bool> + 'static + Send + Sync,
        num_outcomes: usize,
    ) -> Self {
        OutcomeSpecificSimulation {
            clifford: CliffordUnitary::identity(num_qubits),
            outcome_vector: Vec::with_capacity(num_outcomes),
            bit_source: Box::new(bit_source),
            random_outcome_indicator: Vec::with_capacity(num_outcomes),
            num_random_bits: 0,
            qubit_count: num_qubits,
        }
    }

    pub fn new_with_random_outcomes(num_qubits: usize) -> Self {
        Self::new_with_bit_source(num_qubits, SeededRandomBitIterator::new(rand::thread_rng().gen()))
    }

    pub fn new_with_seeded_random_outcomes(num_qubits: usize, seed: u64) -> Self {
        Self::new_with_bit_source(num_qubits, SeededRandomBitIterator::new(seed))
    }

    pub fn new_with_zero_outcomes(num_qubits: usize) -> Self {
        Self::new_with_bit_source(num_qubits, ZeroBitIterator)
    }

    pub fn with_zero_outcomes_and_capacity(num_qubits: usize, num_outcomes: usize) -> Self {
        Self::with_bit_source_and_capacity(num_qubits, ZeroBitIterator, num_outcomes)
    }

    pub fn state_encoder(&self) -> CliffordUnitary {
        let mut res = self.clifford.clone();
        res.resize(self.qubit_count);
        res
    }

    #[must_use]
    pub fn outcome_vector(&self) -> &Vec<bool> {
        &self.outcome_vector
    }

    pub fn with_capacity(num_qubits: usize, num_outcomes: usize, _num_random_outcomes: usize) -> Self {
        Self::with_bit_source_and_capacity(
            num_qubits,
            SeededRandomBitIterator::new(rand::thread_rng().gen()),
            num_outcomes,
        )
    }

    pub fn new(num_qubits: usize) -> Self {
        Self::new_with_random_outcomes(num_qubits)
    }

    /// # Panics
    /// Panics if `hint` commutes with `observable`
    fn measure_with_hint_generic<HintBits: PauliBits, HintPhase: PhaseExponent>(
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
            let mut pauli = observable.clone();
            pauli.mul_assign_right(hint);
            pauli.add_assign_phase_exp(3u8.wrapping_sub(preimage.xz_phase_exponent().raw_value()));
            self.clifford.left_mul_pauli_exp(&pauli);
            self.allocate_random_bit();
            self.apply_conditional_pauli_generic(hint, &[self.outcome_count() - 1], true);
        }
    }

    fn measure_deterministic<Bits: PauliBits, Phase: PhaseExponent>(&mut self, preimage: &PauliUnitary<Bits, Phase>) {
        debug_assert!(preimage.xz_phase_exponent().is_even());
        self.outcome_vector.push(preimage.xz_phase_exponent().value() == 2);
        self.random_outcome_indicator.push(false);
    }

    fn apply_conditional_pauli_generic<Bits: PauliBits, Phase: PhaseExponent>(
        &mut self,
        pauli: &PauliUnitary<Bits, Phase>,
        outcomes_indicator: &[usize],
        parity: bool,
    ) {
        if total_parity(self.outcome_vector(), outcomes_indicator) == parity {
            self.clifford.left_mul_pauli(pauli);
        }
    }
}

impl Simulation for OutcomeSpecificSimulation {
    fn allocate_random_bit(&mut self) -> usize {
        let random_bit = self.bit_source.next().expect("Bit source iterator should be infinite");

        self.outcome_vector.push(random_bit);
        self.random_outcome_indicator.push(true);
        self.num_random_bits += 1;
        self.num_random_bits - 1
    }

    fn conditional_pauli(&mut self, observable: &crate::Pauli, outcomes: &[OutcomeId], parity: bool) {
        self.apply_conditional_pauli_generic(observable, outcomes, parity);
    }

    implement_common_simulation_methods!();

    fn is_stabilizer(&self, observable: &SparsePauli) -> bool {
        let preimage = self.clifford.preimage(observable);
        preimage.x_bits().weight() == 0 && preimage.xz_phase_exponent().value() == 0
    }

    fn is_stabilizer_with_conditional_sign(&self, observable: &crate::Pauli, outcomes: &[OutcomeId]) -> bool {
        let parity = total_parity(self.outcome_vector(), outcomes);
        let preimage = self.clifford.preimage(observable);
        preimage.x_bits().weight() == 0 && (preimage.xz_phase_exponent().value() == 0) != parity
    }

    fn measure(&mut self, observable: &crate::Pauli) -> OutcomeId {
        self.ensure_qubit_capacity(observable.max_support());
        let preimage = self.clifford.preimage(observable);
        let non_zero_pos = preimage.x_bits().support().next();
        match non_zero_pos {
            Some(pos) => {
                let hint = self.clifford.image_z(pos);
                self.measure_with_hint_generic(observable, &hint);
            }
            None => {
                self.measure_deterministic(&preimage);
            }
        }
        self.outcome_count() - 1
    }

    fn measure_with_hint(&mut self, observable: &SparsePauli, hint: &SparsePauli) -> OutcomeId {
        self.ensure_qubit_capacity(max_pair_support(observable, hint));
        self.measure_with_hint_generic(observable, hint);
        self.outcome_vector().len() - 1
    }

    fn outcome_count(&self) -> usize {
        self.random_outcome_indicator().len()
    }

    fn random_outcome_count(&self) -> usize {
        self.num_random_bits
    }

    fn random_outcome_indicator(&self) -> &[bool] {
        &self.random_outcome_indicator
    }

    fn with_capacity(qubit_count: usize, outcome_count: usize, random_outcome_count: usize) -> Self
    where
        Self: Sized,
    {
        OutcomeSpecificSimulation::with_capacity(qubit_count, outcome_count, random_outcome_count)
    }

    fn qubit_capacity(&self) -> usize {
        self.clifford.num_qubits()
    }

    fn reserve_qubits(&mut self, new_capacity: usize) {
        if new_capacity > self.qubit_capacity() {
            self.clifford.resize(new_capacity);
        }
    }

    fn outcome_capacity(&self) -> usize {
        self.random_outcome_indicator.capacity()
    }

    fn random_outcome_capacity(&self) -> usize {
        self.outcome_capacity()
    }

    fn reserve_outcomes(&mut self, new_outcome_capacity: usize, new_random_outcome_capacity: usize) {
        let new_capacity = new_outcome_capacity.max(new_random_outcome_capacity);
        if new_capacity > self.outcome_capacity() {
            self.outcome_vector
                .reserve(new_outcome_capacity - self.outcome_capacity());
            self.random_outcome_indicator
                .reserve(new_outcome_capacity - self.outcome_capacity());
        }
    }
}

fn total_parity(outcome_vector: &[bool], outcomes_indicator: &[usize]) -> bool {
    let mut res = false;
    for j in outcomes_indicator {
        res ^= outcome_vector[*j];
    }
    res
}

//TODO: move to tests module
#[test]
fn init_test() {
    let outcome_specific_simulation = OutcomeSpecificSimulation::new(2);
    // Verify it initializes correctly with just qubit count
    assert_eq!(outcome_specific_simulation.outcome_vector().len(), 0);
}

pub struct RandomBitIterator {
    rng: rand::rngs::ThreadRng,
}

impl RandomBitIterator {
    #[must_use]
    pub fn new() -> Self {
        Self { rng: thread_rng() }
    }
}

impl Default for RandomBitIterator {
    fn default() -> Self {
        Self::new()
    }
}

impl Iterator for RandomBitIterator {
    type Item = bool;

    fn next(&mut self) -> Option<bool> {
        Some(self.rng.gen::<bool>())
    }
}

pub struct SeededRandomBitIterator {
    rng: StdRng,
}

impl SeededRandomBitIterator {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Iterator for SeededRandomBitIterator {
    type Item = bool;

    fn next(&mut self) -> Option<bool> {
        Some(self.rng.gen::<bool>())
    }
}

/// Iterator that always returns false
pub struct ZeroBitIterator;

impl Iterator for ZeroBitIterator {
    type Item = bool;

    fn next(&mut self) -> Option<bool> {
        Some(false)
    }
}
