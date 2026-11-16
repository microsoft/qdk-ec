//! Checks that the pivot `measure` chooses for a random measurement changes neither the outcomes nor the state.
//!
//! The simulation measures with `measure`. The reference measures with `measure_with_hint` and passes `image_z` of the
//! lowest candidate pivot, so it always uses the lowest candidate. Both run the same random circuit of named gates,
//! Pauli operators, resets and measurements. After every step they must report the same outcomes and have the same
//! stabilizers.

use binar::Bitwise;
use paulimer::{
    UnitaryOp,
    clifford::{Clifford, CliffordUnitary},
    pauli::{DensePauli, Pauli, PauliMutable, SparsePauli, SparsePauliProjective},
    traits::NeutralElement,
};
use pauliverse::{OutcomeCompleteSimulation, OutcomeFreeSimulation, OutcomeSpecificSimulation, Simulation};
use proptest::prelude::*;
use rand::{RngExt, SeedableRng, rngs::StdRng};

const SINGLE_QUBIT_GATES: [UnitaryOp; 10] = [
    UnitaryOp::Hadamard,
    UnitaryOp::SqrtZ,
    UnitaryOp::SqrtZInv,
    UnitaryOp::SqrtX,
    UnitaryOp::SqrtXInv,
    UnitaryOp::SqrtY,
    UnitaryOp::SqrtYInv,
    UnitaryOp::X,
    UnitaryOp::Y,
    UnitaryOp::Z,
];

const TWO_QUBIT_GATES: [UnitaryOp; 3] = [UnitaryOp::ControlledX, UnitaryOp::ControlledZ, UnitaryOp::Swap];

trait PivotChoiceCheck: Simulation {
    fn new_for_test(qubit_count: usize, seed: u64) -> Self;
    /// Returns `image_z` of the lowest candidate pivot for measuring `observable`, or `None` if the outcome is
    /// deterministic.
    fn lowest_candidate_hint(&self, observable: &SparsePauli) -> Option<SparsePauli>;
    /// Asserts that `other` reports the same outcomes and has the same stabilizers, with the same signs where the
    /// simulation fixes them.
    fn assert_same_state(&self, other: &Self);
}

impl PivotChoiceCheck for OutcomeSpecificSimulation {
    fn new_for_test(qubit_count: usize, seed: u64) -> Self {
        OutcomeSpecificSimulation::new_with_seeded_random_samples(qubit_count, seed)
    }
    fn lowest_candidate_hint(&self, observable: &SparsePauli) -> Option<SparsePauli> {
        hint_from_lowest_candidate(&self.state_encoder(), observable)
    }
    fn assert_same_state(&self, other: &Self) {
        assert_eq!(self.outcome_vector(), other.outcome_vector());
        assert_eq!(
            self.outcome_introduces_branch_bit(),
            other.outcome_introduces_branch_bit()
        );
        assert_eq!(self.branch_bit_count(), other.branch_bit_count());
        let encoder = self.state_encoder();
        for qubit in 0..self.qubit_count() {
            assert!(other.is_stabilizer(&encoder.image_z(qubit).into()));
        }
    }
}

impl PivotChoiceCheck for OutcomeCompleteSimulation {
    fn new_for_test(qubit_count: usize, _seed: u64) -> Self {
        OutcomeCompleteSimulation::new(qubit_count)
    }
    fn lowest_candidate_hint(&self, observable: &SparsePauli) -> Option<SparsePauli> {
        hint_from_lowest_candidate(&self.state_encoder(), observable)
    }
    fn assert_same_state(&self, other: &Self) {
        assert_eq!(
            self.outcome_introduces_branch_bit(),
            other.outcome_introduces_branch_bit()
        );
        assert_eq!(self.branch_bit_count(), other.branch_bit_count());
        assert_eq!(self.outcome_matrix(), other.outcome_matrix());
        assert_eq!(
            self.outcome_shift().iter().collect::<Vec<bool>>(),
            other.outcome_shift().iter().collect::<Vec<bool>>()
        );
        let encoder = self.state_encoder();
        for qubit in 0..self.qubit_count() {
            assert!(other.is_stabilizer_up_to_sign(&encoder.image_z(qubit).into()));
        }
    }
}

impl PivotChoiceCheck for OutcomeFreeSimulation {
    fn new_for_test(qubit_count: usize, _seed: u64) -> Self {
        OutcomeFreeSimulation::with_capacity(qubit_count, 0, 0)
    }
    fn lowest_candidate_hint(&self, observable: &SparsePauli) -> Option<SparsePauli> {
        let encoder = self.state_encoder();
        let projective: &SparsePauliProjective = observable.as_ref();
        let pivot = encoder.preimage(projective).x_bits().support().next()?;
        Some(SparsePauli::from(DensePauli::from(encoder.image_z(pivot))))
    }
    fn assert_same_state(&self, other: &Self) {
        assert_eq!(
            self.outcome_introduces_branch_bit(),
            other.outcome_introduces_branch_bit()
        );
        assert_eq!(self.branch_bit_count(), other.branch_bit_count());
        let encoder = self.state_encoder();
        for qubit in 0..self.qubit_count() {
            let generator = SparsePauli::from(DensePauli::from(encoder.image_z(qubit)));
            assert!(other.is_stabilizer_up_to_sign(&generator));
        }
    }
}

fn hint_from_lowest_candidate(encoder: &CliffordUnitary, observable: &SparsePauli) -> Option<SparsePauli> {
    let pivot = encoder.preimage(observable).x_bits().support().next()?;
    Some(encoder.image_z(pivot).into())
}

fn measure_both<S: PivotChoiceCheck>(simulation: &mut S, reference: &mut S, observable: &SparsePauli) -> usize {
    let outcome_id = simulation.measure(observable);
    let reference_outcome_id = match reference.lowest_candidate_hint(observable) {
        Some(hint) => reference.measure_with_hint(observable, &hint),
        None => reference.measure(observable),
    };
    assert_eq!(outcome_id, reference_outcome_id);
    outcome_id
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

fn single_qubit_x(qubit: usize, qubit_count: usize) -> SparsePauli {
    let mut dense = DensePauli::neutral_element_of_size(qubit_count);
    dense.mul_assign_right_x(qubit);
    dense.into()
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

fn check_pivot_choice<S: PivotChoiceCheck>(qubit_count: usize, step_count: usize, seed: u64) {
    let rng = &mut StdRng::seed_from_u64(seed);
    let mut simulation = S::new_for_test(qubit_count, seed);
    let mut reference = S::new_for_test(qubit_count, seed);
    let mut observables: Vec<SparsePauli> = Vec::new();
    for _ in 0..step_count {
        match rng.random_range(0..12) {
            0 | 1 => {
                let gate = SINGLE_QUBIT_GATES[rng.random_range(0..SINGLE_QUBIT_GATES.len())];
                let support = [rng.random_range(0..qubit_count)];
                simulation.unitary_op(gate, &support);
                reference.unitary_op(gate, &support);
            }
            2 | 3 if qubit_count > 1 => {
                let gate = TWO_QUBIT_GATES[rng.random_range(0..TWO_QUBIT_GATES.len())];
                let support = random_support(qubit_count, 2, rng);
                simulation.unitary_op(gate, &support);
                reference.unitary_op(gate, &support);
            }
            4 => {
                let pauli = random_pauli(qubit_count, rng);
                simulation.pauli(&pauli);
                reference.pauli(&pauli);
            }
            5 => {
                let pauli = random_pauli(qubit_count, rng);
                simulation.pauli_exp(&pauli);
                reference.pauli_exp(&pauli);
            }
            6 if simulation.outcome_count() > 0 => {
                let pauli = random_pauli(qubit_count, rng);
                let outcome_count = simulation.outcome_count();
                let condition_size = rng.random_range(1..=outcome_count.min(3));
                let outcomes = random_support(outcome_count, condition_size, rng);
                let parity = rng.random_bool(0.5);
                simulation.conditional_pauli(&pauli, &outcomes, parity);
                reference.conditional_pauli(&pauli, &outcomes, parity);
            }
            7 => {
                let qubit = rng.random_range(0..qubit_count);
                let outcome_id = measure_both(&mut simulation, &mut reference, &single_qubit_z(qubit, qubit_count));
                let flip = single_qubit_x(qubit, qubit_count);
                simulation.conditional_pauli(&flip, &[outcome_id], true);
                reference.conditional_pauli(&flip, &[outcome_id], true);
            }
            choice => {
                let observable = if choice < 10 {
                    single_qubit_z(rng.random_range(0..qubit_count), qubit_count)
                } else if choice == 10 && !observables.is_empty() {
                    observables[rng.random_range(0..observables.len())].clone()
                } else {
                    random_pauli(qubit_count, rng)
                };
                measure_both(&mut simulation, &mut reference, &observable);
                observables.push(observable);
            }
        }
        simulation.assert_same_state(&reference);
    }
}

proptest! {
    #[test]
    fn outcome_specific_pivot_choice_proptest(qubit_count in 1..12usize, seed in any::<u64>()) {
        check_pivot_choice::<OutcomeSpecificSimulation>(qubit_count, 8 * qubit_count, seed);
    }

    #[test]
    fn outcome_complete_pivot_choice_proptest(qubit_count in 1..12usize, seed in any::<u64>()) {
        check_pivot_choice::<OutcomeCompleteSimulation>(qubit_count, 8 * qubit_count, seed);
    }

    #[test]
    fn outcome_free_pivot_choice_proptest(qubit_count in 1..12usize, seed in any::<u64>()) {
        check_pivot_choice::<OutcomeFreeSimulation>(qubit_count, 8 * qubit_count, seed);
    }
}

// More than 64 qubits, so that the candidate pivots span several 64-bit words.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]
    #[test]
    fn outcome_specific_pivot_choice_many_qubits_proptest(qubit_count in 65..80usize, seed in any::<u64>()) {
        check_pivot_choice::<OutcomeSpecificSimulation>(qubit_count, 5 * qubit_count, seed);
    }

    #[test]
    fn outcome_complete_pivot_choice_many_qubits_proptest(qubit_count in 65..80usize, seed in any::<u64>()) {
        check_pivot_choice::<OutcomeCompleteSimulation>(qubit_count, 5 * qubit_count, seed);
    }

    #[test]
    fn outcome_free_pivot_choice_many_qubits_proptest(qubit_count in 65..80usize, seed in any::<u64>()) {
        check_pivot_choice::<OutcomeFreeSimulation>(qubit_count, 5 * qubit_count, seed);
    }
}
