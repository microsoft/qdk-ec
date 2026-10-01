//! Checks that `measure` leaves each simulation in the same state as `measure_with_hint(observable, image_z(pivot))`.

use binar::Bitwise;
use paulimer::{
    clifford::{Clifford, CliffordUnitary},
    pauli::{DensePauli, Pauli, PauliMutable, SparsePauli, SparsePauliProjective},
    traits::NeutralElement,
};
use pauliverse::{OutcomeCompleteSimulation, OutcomeFreeSimulation, OutcomeSpecificSimulation, Simulation};
use proptest::prelude::*;
use rand::{RngExt, SeedableRng, rngs::StdRng};

trait EquivalenceCheck: Simulation {
    fn new_for_test(qubit_count: usize, seed: u64) -> Self;
    fn stabilizer_hint(&self, observable: &SparsePauli) -> Option<SparsePauli>;
    fn assert_same_state(&self, other: &Self);
    fn assert_measured(&self, observable: &SparsePauli, outcome_id: usize);
}

impl EquivalenceCheck for OutcomeSpecificSimulation {
    fn new_for_test(_qubit_count: usize, seed: u64) -> Self {
        OutcomeSpecificSimulation::new_with_seeded_random_outcomes(0, seed)
    }
    fn stabilizer_hint(&self, observable: &SparsePauli) -> Option<SparsePauli> {
        let encoder = self.state_encoder();
        let pivot = encoder.preimage(observable).x_bits().support().next()?;
        Some(encoder.image_z(pivot).into())
    }
    fn assert_same_state(&self, other: &Self) {
        assert_eq!(self.outcome_vector(), other.outcome_vector());
        assert_eq!(self.random_outcome_indicator(), other.random_outcome_indicator());
        assert_eq!(self.random_outcome_count(), other.random_outcome_count());
        assert_eq!(self.qubit_count(), other.qubit_count());
        assert_eq!(self.state_encoder(), other.state_encoder());
    }
    fn assert_measured(&self, observable: &SparsePauli, outcome_id: usize) {
        assert!(self.is_stabilizer_with_conditional_sign(observable, &[outcome_id]));
    }
}

impl EquivalenceCheck for OutcomeCompleteSimulation {
    fn new_for_test(_qubit_count: usize, _seed: u64) -> Self {
        OutcomeCompleteSimulation::default()
    }
    fn stabilizer_hint(&self, observable: &SparsePauli) -> Option<SparsePauli> {
        let encoder = self.state_encoder();
        let pivot = encoder.preimage(observable).x_bits().support().next()?;
        Some(encoder.image_z(pivot).into())
    }
    fn assert_same_state(&self, other: &Self) {
        assert_eq!(self.random_outcome_indicator(), other.random_outcome_indicator());
        assert_eq!(self.random_outcome_count(), other.random_outcome_count());
        assert_eq!(self.qubit_count(), other.qubit_count());
        assert_eq!(self.state_encoder(), other.state_encoder());
        assert_eq!(self.aligned_sign_matrix(), other.aligned_sign_matrix());
        assert_eq!(self.aligned_outcome_matrix(), other.aligned_outcome_matrix());
        assert_eq!(
            self.outcome_shift().iter().collect::<Vec<bool>>(),
            other.outcome_shift().iter().collect::<Vec<bool>>()
        );
    }
    fn assert_measured(&self, observable: &SparsePauli, outcome_id: usize) {
        assert!(self.is_stabilizer_with_conditional_sign(observable, &[outcome_id]));
    }
}

impl EquivalenceCheck for OutcomeFreeSimulation {
    fn new_for_test(qubit_count: usize, _seed: u64) -> Self {
        OutcomeFreeSimulation::with_capacity(qubit_count, 0, 0)
    }
    fn stabilizer_hint(&self, observable: &SparsePauli) -> Option<SparsePauli> {
        let encoder = self.state_encoder();
        let projective: &SparsePauliProjective = observable.as_ref();
        let pivot = encoder.preimage(projective).x_bits().support().next()?;
        Some(SparsePauli::from(DensePauli::from(encoder.image_z(pivot))))
    }
    fn assert_same_state(&self, other: &Self) {
        assert_eq!(self.random_outcome_indicator(), other.random_outcome_indicator());
        assert_eq!(self.random_outcome_count(), other.random_outcome_count());
        assert_eq!(self.qubit_count(), other.qubit_count());
        assert_eq!(self.state_encoder(), other.state_encoder());
    }
    fn assert_measured(&self, observable: &SparsePauli, _outcome_id: usize) {
        assert!(self.is_stabilizer_up_to_sign(observable));
    }
}

fn random_pauli(qubit_count: usize, rng: &mut impl RngExt) -> SparsePauli {
    loop {
        let mut dense = DensePauli::neutral_element_of_size(qubit_count);
        dense.set_random_order_two(qubit_count, rng);
        if dense.weight() > 0 {
            return dense.into();
        }
    }
}

fn single_qubit_z(qubit: usize, qubit_count: usize) -> SparsePauli {
    let mut dense = DensePauli::neutral_element_of_size(qubit_count);
    dense.mul_assign_right_z(qubit);
    dense.into()
}

fn random_support(qubit_count: usize, size: usize, rng: &mut impl RngExt) -> Vec<usize> {
    let mut support: Vec<usize> = Vec::new();
    while support.len() < size {
        let qubit = rng.random_range(0..qubit_count);
        if !support.contains(&qubit) {
            support.push(qubit);
        }
    }
    support
}

fn check_equivalence<S: EquivalenceCheck>(qubit_count: usize, step_count: usize, seed: u64) {
    let rng = &mut StdRng::seed_from_u64(seed);
    let mut simulation = S::new_for_test(qubit_count, seed);
    let mut reference = S::new_for_test(qubit_count, seed);
    let qubits: Vec<usize> = (0..qubit_count).collect();
    let unitary = CliffordUnitary::random(qubit_count, rng);
    simulation.clifford(&unitary, &qubits);
    reference.clifford(&unitary, &qubits);
    let mut observables: Vec<SparsePauli> = Vec::new();
    for _ in 0..step_count {
        match rng.random_range(0..10) {
            0 => {
                let size = rng.random_range(1..=qubit_count.min(3));
                let support = random_support(qubit_count, size, rng);
                let unitary = CliffordUnitary::random(size, rng);
                simulation.clifford(&unitary, &support);
                reference.clifford(&unitary, &support);
            }
            1 => {
                let pauli = random_pauli(qubit_count, rng);
                simulation.pauli(&pauli);
                reference.pauli(&pauli);
            }
            2 if simulation.outcome_count() > 0 => {
                let pauli = random_pauli(qubit_count, rng);
                let outcome_count = simulation.outcome_count();
                let condition_size = rng.random_range(1..=outcome_count.min(3));
                let outcomes = random_support(outcome_count, condition_size, rng);
                let parity = rng.random_bool(0.5);
                simulation.conditional_pauli(&pauli, &outcomes, parity);
                reference.conditional_pauli(&pauli, &outcomes, parity);
            }
            3 => {
                let pauli = random_pauli(qubit_count, rng);
                simulation.pauli_exp(&pauli);
                reference.pauli_exp(&pauli);
            }
            choice => {
                let observable = if choice < 6 {
                    single_qubit_z(rng.random_range(0..qubit_count), qubit_count)
                } else if choice == 6 && !observables.is_empty() {
                    observables[rng.random_range(0..observables.len())].clone()
                } else {
                    random_pauli(qubit_count, rng)
                };
                let outcome_id = simulation.measure(&observable);
                let reference_outcome_id = match reference.stabilizer_hint(&observable) {
                    Some(hint) => reference.measure_with_hint(&observable, &hint),
                    None => reference.measure(&observable),
                };
                assert_eq!(outcome_id, reference_outcome_id);
                simulation.assert_measured(&observable, outcome_id);
                observables.push(observable);
            }
        }
        simulation.assert_same_state(&reference);
    }
}

proptest! {
    #[test]
    fn outcome_specific_measure_proptest(qubit_count in 1..12usize, seed in any::<u64>()) {
        check_equivalence::<OutcomeSpecificSimulation>(qubit_count, 6 * qubit_count, seed);
    }

    #[test]
    fn outcome_complete_measure_proptest(qubit_count in 1..12usize, seed in any::<u64>()) {
        check_equivalence::<OutcomeCompleteSimulation>(qubit_count, 6 * qubit_count, seed);
    }

    #[test]
    fn outcome_free_measure_proptest(qubit_count in 1..12usize, seed in any::<u64>()) {
        check_equivalence::<OutcomeFreeSimulation>(qubit_count, 6 * qubit_count, seed);
    }
}

// More than 64 qubits, so that the columns of the state encoder span several 64-row chunks.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]
    #[test]
    fn outcome_specific_measure_many_qubits_proptest(qubit_count in 65..80usize, seed in any::<u64>()) {
        check_equivalence::<OutcomeSpecificSimulation>(qubit_count, 5 * qubit_count, seed);
    }

    #[test]
    fn outcome_complete_measure_many_qubits_proptest(qubit_count in 65..80usize, seed in any::<u64>()) {
        check_equivalence::<OutcomeCompleteSimulation>(qubit_count, 5 * qubit_count, seed);
    }

    #[test]
    fn outcome_free_measure_many_qubits_proptest(qubit_count in 65..80usize, seed in any::<u64>()) {
        check_equivalence::<OutcomeFreeSimulation>(qubit_count, 5 * qubit_count, seed);
    }
}
