//! Checks that `measure` leaves each simulation in the same state as `measure_with_hint(observable, image_z(pivot))`.
//!
//! `check_pivot_choice` checks that the choice of pivot changes neither the outcomes nor the stabilizers. Its
//! reference always passes `image_z` of the lowest candidate pivot to `measure_with_hint`, so the state encoders can
//! differ.

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

trait EquivalenceCheck: Simulation {
    fn new_for_test(qubit_count: usize, seed: u64) -> Self;
    fn stabilizer_hint(&self, observable: &SparsePauli, pivots: &mut PivotReplica) -> Option<SparsePauli>;
    /// Returns `image_z` of the lowest candidate pivot for measuring `observable`, or `None` if the outcome is
    /// deterministic.
    fn lowest_candidate_hint(&self, observable: &SparsePauli) -> Option<SparsePauli>;
    fn assert_same_state(&self, other: &Self);
    /// Asserts that `other` reports the same outcomes and has the same stabilizers, with the same signs where the
    /// simulation fixes them.
    fn assert_same_stabilizers(&self, other: &Self);
    fn assert_measured(&self, observable: &SparsePauli, outcome_id: usize);
}

/// Replicates how `measure` chooses the pivot of a random measurement: the candidate that served as a pivot most
/// recently, or the lowest candidate among ties.
#[derive(Default)]
struct PivotReplica {
    refreshed_at: Vec<u64>,
    clock: u64,
}

impl PivotReplica {
    fn choose(&mut self, candidates: impl Iterator<Item = usize>) -> Option<usize> {
        let mut best: Option<(usize, u64)> = None;
        for candidate in candidates {
            let time = self.refreshed_at.get(candidate).copied().unwrap_or(0);
            if best.is_none_or(|(_, best_time)| time > best_time) {
                best = Some((candidate, time));
            }
        }
        let (pivot, _) = best?;
        if pivot >= self.refreshed_at.len() {
            self.refreshed_at.resize(pivot + 1, 0);
        }
        self.clock += 1;
        self.refreshed_at[pivot] = self.clock;
        Some(pivot)
    }
}

impl EquivalenceCheck for OutcomeSpecificSimulation {
    fn new_for_test(qubit_count: usize, seed: u64) -> Self {
        OutcomeSpecificSimulation::new_with_seeded_random_samples(qubit_count, seed)
    }
    fn stabilizer_hint(&self, observable: &SparsePauli, pivots: &mut PivotReplica) -> Option<SparsePauli> {
        let encoder = self.state_encoder();
        let pivot = pivots.choose(encoder.preimage(observable).x_bits().support())?;
        Some(encoder.image_z(pivot).into())
    }
    fn lowest_candidate_hint(&self, observable: &SparsePauli) -> Option<SparsePauli> {
        let encoder = self.state_encoder();
        let pivot = encoder.preimage(observable).x_bits().support().next()?;
        Some(encoder.image_z(pivot).into())
    }
    fn assert_same_state(&self, other: &Self) {
        assert_eq!(self.outcome_vector(), other.outcome_vector());
        assert_eq!(
            self.outcome_introduces_branch_bit(),
            other.outcome_introduces_branch_bit()
        );
        assert_eq!(self.branch_bit_count(), other.branch_bit_count());
        assert_eq!(self.qubit_count(), other.qubit_count());
        assert_eq!(self.state_encoder(), other.state_encoder());
    }
    fn assert_same_stabilizers(&self, other: &Self) {
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
    fn assert_measured(&self, observable: &SparsePauli, outcome_id: usize) {
        assert!(self.is_stabilizer_with_conditional_sign(observable, &[outcome_id]));
    }
}

impl EquivalenceCheck for OutcomeCompleteSimulation {
    fn new_for_test(qubit_count: usize, _seed: u64) -> Self {
        OutcomeCompleteSimulation::with_capacity(qubit_count, 0, 0)
    }
    fn stabilizer_hint(&self, observable: &SparsePauli, pivots: &mut PivotReplica) -> Option<SparsePauli> {
        let encoder = self.state_encoder();
        let pivot = pivots.choose(encoder.preimage(observable).x_bits().support())?;
        Some(encoder.image_z(pivot).into())
    }
    fn lowest_candidate_hint(&self, observable: &SparsePauli) -> Option<SparsePauli> {
        let encoder = self.state_encoder();
        let pivot = encoder.preimage(observable).x_bits().support().next()?;
        Some(encoder.image_z(pivot).into())
    }
    fn assert_same_state(&self, other: &Self) {
        assert_eq!(
            self.outcome_introduces_branch_bit(),
            other.outcome_introduces_branch_bit()
        );
        assert_eq!(self.branch_bit_count(), other.branch_bit_count());
        assert_eq!(self.qubit_count(), other.qubit_count());
        assert_eq!(self.state_encoder(), other.state_encoder());
        assert_eq!(self.aligned_sign_matrix(), other.aligned_sign_matrix());
        assert_eq!(self.aligned_outcome_matrix(), other.aligned_outcome_matrix());
        assert_eq!(
            self.outcome_shift().iter().collect::<Vec<bool>>(),
            other.outcome_shift().iter().collect::<Vec<bool>>()
        );
    }
    fn assert_same_stabilizers(&self, other: &Self) {
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
    fn assert_measured(&self, observable: &SparsePauli, outcome_id: usize) {
        assert!(self.is_stabilizer_with_conditional_sign(observable, &[outcome_id]));
    }
}

impl EquivalenceCheck for OutcomeFreeSimulation {
    fn new_for_test(qubit_count: usize, _seed: u64) -> Self {
        OutcomeFreeSimulation::with_capacity(qubit_count, 0, 0)
    }
    fn stabilizer_hint(&self, observable: &SparsePauli, pivots: &mut PivotReplica) -> Option<SparsePauli> {
        let encoder = self.state_encoder();
        let projective: &SparsePauliProjective = observable.as_ref();
        let pivot = pivots.choose(encoder.preimage(projective).x_bits().support())?;
        Some(SparsePauli::from(DensePauli::from(encoder.image_z(pivot))))
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
        assert_eq!(self.qubit_count(), other.qubit_count());
        assert_eq!(self.state_encoder(), other.state_encoder());
    }
    fn assert_same_stabilizers(&self, other: &Self) {
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

fn single_qubit_x(qubit: usize, qubit_count: usize) -> SparsePauli {
    let mut dense = DensePauli::neutral_element_of_size(qubit_count);
    dense.mul_assign_right_x(qubit);
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
    let mut pivots = PivotReplica::default();
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
                let reference_outcome_id = match reference.stabilizer_hint(&observable, &mut pivots) {
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

fn measure_both<S: EquivalenceCheck>(simulation: &mut S, reference: &mut S, observable: &SparsePauli) -> usize {
    let outcome_id = simulation.measure(observable);
    let reference_outcome_id = match reference.lowest_candidate_hint(observable) {
        Some(hint) => reference.measure_with_hint(observable, &hint),
        None => reference.measure(observable),
    };
    assert_eq!(outcome_id, reference_outcome_id);
    outcome_id
}

fn check_pivot_choice<S: EquivalenceCheck>(qubit_count: usize, step_count: usize, seed: u64) {
    let single_qubit_gates = [
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
    let two_qubit_gates = [UnitaryOp::ControlledX, UnitaryOp::ControlledZ, UnitaryOp::Swap];
    let rng = &mut StdRng::seed_from_u64(seed);
    let mut simulation = S::new_for_test(qubit_count, seed);
    let mut reference = S::new_for_test(qubit_count, seed);
    let mut observables: Vec<SparsePauli> = Vec::new();
    for _ in 0..step_count {
        match rng.random_range(0..12) {
            0 | 1 => {
                let gate = single_qubit_gates[rng.random_range(0..single_qubit_gates.len())];
                let support = [rng.random_range(0..qubit_count)];
                simulation.unitary_op(gate, &support);
                reference.unitary_op(gate, &support);
            }
            2 | 3 if qubit_count > 1 => {
                let gate = two_qubit_gates[rng.random_range(0..two_qubit_gates.len())];
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
        simulation.assert_same_stabilizers(&reference);
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
