use std::borrow::Borrow;

use binar::{BitMatrix, BitView, Bitwise, BitwisePairMut, IndexSet};
use paulimer::{
    clifford::{Clifford, CliffordMutable, CliffordUnitary},
    operations::UnitaryOp,
    pauli::{Pauli, SparsePauli},
};
use pauliverse::{
    outcome_complete_simulation::OutcomeCompleteSimulation, outcome_free_simulation::OutcomeFreeSimulation,
    outcome_specific_simulation::OutcomeSpecificSimulation, Simulation,
};
use quantum_core::{x, z, PositionedPauliObservable};

trait SimulationForTest: Simulation + Default {
    fn measure_o(&mut self, observable: &[PositionedPauliObservable]) -> usize {
        pauliverse::Simulation::measure(self, &observable.into())
    }

    fn assert_stabilizer_o(&self, observable: &[PositionedPauliObservable]) {
        assert!(self.is_stabilizer(&observable.into()));
    }

    fn apply_conditional_pauli_o(
        &mut self,
        observable: &[PositionedPauliObservable],
        outcomes: &[usize],
        parity: bool,
    ) {
        self.conditional_pauli(&observable.into(), outcomes, parity);
    }

    fn assert_stabilizer_up_to_sign_o(&self, observable: &[PositionedPauliObservable]) {
        assert!(self.is_stabilizer_up_to_sign(&observable.into()));
    }

    fn assert_stabilizer_with_conditional_sign(&self, observable: &[PositionedPauliObservable], outcomes: &[usize]) {
        assert!(self.is_stabilizer_with_conditional_sign(&observable.into(), outcomes));
    }

    fn apply_pauli_o(&mut self, observable: &[PositionedPauliObservable]) {
        self.pauli(&observable.into());
    }
}

impl SimulationForTest for OutcomeCompleteSimulation {}
impl SimulationForTest for OutcomeSpecificSimulation {}
impl SimulationForTest for OutcomeFreeSimulation {}

fn measure_and_fix(
    sim: &mut impl SimulationForTest,
    observable: &[PositionedPauliObservable],
    fix: &[PositionedPauliObservable],
) {
    let max_qubit_id = observable
        .iter()
        .chain(fix.iter())
        .map(|p| p.qubit_id)
        .max()
        .unwrap_or(0);

    if max_qubit_id < sim.qubit_count() {
        sim.assert_stabilizer_o(fix);
    }

    let r = sim.measure_o(observable);
    sim.assert_stabilizer_up_to_sign_o(observable);
    sim.assert_stabilizer_with_conditional_sign(observable, &[r]);
    sim.apply_conditional_pauli_o(fix, &[r], true);
    sim.assert_stabilizer_o(observable);
}

fn cx_via_measure(sim: &mut impl SimulationForTest, control: usize, target: usize, helper: usize) {
    let (q0, q1, q2) = (control, helper, target);
    measure_and_fix(sim, &[x(q1)], &[z(q1)]);
    measure_and_fix(sim, &[z(q0), z(q1)], &[x(q1)]);
    measure_and_fix(sim, &[x(q1), x(q2)], &[z(q0), z(q1)]);
    measure_and_fix(sim, &[z(q1)], &[x(q1), x(q2)]);
}

fn cz_via_measure(sim: &mut impl SimulationForTest, control: usize, target: usize, helper: usize) {
    let (q0, q1, q2) = (control, helper, target);
    measure_and_fix(sim, &[x(q1)], &[z(q1)]);
    measure_and_fix(sim, &[z(q0), z(q1)], &[x(q1)]);
    measure_and_fix(sim, &[x(q1), z(q2)], &[z(q0), z(q1)]);
    measure_and_fix(sim, &[z(q1)], &[x(q1), z(q2)]);
}

fn prep_bell_state(sim: &mut impl SimulationForTest, target: (usize, usize)) {
    let (q0, q1) = target;
    measure_and_fix(sim, &[x(q0)], &[z(q0)]);
    measure_and_fix(sim, &[x(q1)], &[z(q1)]);
    measure_and_fix(sim, &[z(q0), z(q1)], &[x(q0)]);
}

fn assert_bell(sim: &impl SimulationForTest, target: (usize, usize)) {
    let (q0, q1) = target;
    sim.assert_stabilizer_o(&[x(q0), x(q1)]);
    sim.assert_stabilizer_o(&[z(q0), z(q1)]);
}

// two random outcomes
fn random_and_deterministic_outcome_sequence<SimulationKind: SimulationForTest>(sim: &mut SimulationKind) {
    sim.measure_o(&[x(0)]); // 0: random o1            | 0b1
    sim.measure_o(&[z(1)]); // 1: deterministic 0      | 0b0
    sim.measure_o(&[x(0)]); // 2: deterministic = o1   | 0b1
    sim.apply_pauli_o(&[z(0)]);
    sim.measure_o(&[x(0)]); // 3: deterministic = o1+1 |0b1
    sim.apply_pauli_o(&[z(0)]);
    sim.measure_o(&[x(0)]); // 4: deterministic = o1   |0b1
    sim.measure_o(&[x(0)]); // 5: deterministic = o1   |0b1
    sim.measure_o(&[z(0)]); // 6: random o2            |0b10
    sim.apply_pauli_o(&[x(1)]);
    sim.measure_o(&[z(1)]); // 7: deterministic 1      |0b0
                            // outcome shift : 0b_1000_1000
}

fn cx_cz_test<SimulationKind: SimulationForTest>() {
    // just run cnot via measure circuit
    {
        let mut sim = SimulationKind::with_capacity(3, 4, 4);
        cx_via_measure(&mut sim, 0, 1, 2);
        assert_eq!(sim.random_outcome_count(), 4);
        assert_eq!(sim.random_outcome_indicator().len(), 4);
    }
    // test that cnot via measurement followed by a builtin cnot is identity
    {
        let mut sim = SimulationKind::with_capacity(5, 10, 10);
        choi_state_of_cx_via_measure(&mut sim);
    }
    // test that cz via measurement followed by a builtin cz is identity
    {
        let mut sim = SimulationKind::with_capacity(5, 10, 10);
        choi_state_of_cz_via_measure(&mut sim);
    }
}

fn choi_state_of_cz_via_measure<SimulationKind: SimulationForTest>(sim: &mut SimulationKind) {
    let (control, helper, target, target_ref, control_ref) = (0, 1, 2, 3, 4usize);
    prep_bell_state(sim, (control, control_ref));
    prep_bell_state(sim, (target, target_ref));
    cz_via_measure(sim, control, target, helper);
    sim.unitary_op(UnitaryOp::ControlledZ, &[control, target]);
    assert_bell(sim, (control, control_ref));
    assert_bell(sim, (target, target_ref));
}

fn choi_state_of_cx_via_measure<SimulationKind: SimulationForTest>(sim: &mut SimulationKind) {
    let (control, helper, target, target_ref, control_ref) = (0, 1, 2, 3, 4usize);
    prep_bell_state(sim, (control, control_ref));
    prep_bell_state(sim, (target, target_ref));
    assert_bell(sim, (control, control_ref));
    assert_bell(sim, (target, target_ref));
    cx_via_measure(sim, control, target, helper);

    sim.assert_stabilizer_o(&[z(helper)]);
    // check that we get a Choi state of a cnot by listing its stabilizers
    sim.assert_stabilizer_o(&[z(control_ref), z(control)]);
    sim.assert_stabilizer_o(&[x(control_ref), x(control), x(target)]);
    sim.assert_stabilizer_o(&[z(target_ref), z(control), z(target)]);
    sim.assert_stabilizer_o(&[x(target_ref), x(target)]);

    sim.unitary_op(UnitaryOp::ControlledX, &[control, target]);
    // println!("{}",clifford_images_as_sparse_string(sim.clifford()));
    assert_bell(sim, (control, control_ref));
    assert_bell(sim, (target, target_ref));
}

#[test]
fn cx_cz_outcome_complete_test() {
    cx_cz_test::<OutcomeCompleteSimulation>();
}

#[test]
fn cx_cz_outcome_specific_test() {
    cx_cz_test::<OutcomeSpecificSimulation>();
}

#[test]
fn cx_cz_outcome_free_test() {
    cx_cz_test::<OutcomeFreeSimulation>();
}

#[test]
fn measure_and_fix_outcome_complete_test() {
    let mut sim = OutcomeCompleteSimulation::with_capacity(3, 4, 4);
    for j in 0..3 {
        sim.assert_stabilizer_o(&[z(j)]);
    }
    measure_and_fix(&mut sim, &[x(1)], &[z(1)]);
}

fn check_random_outcomes_bit_shift(sim: &OutcomeCompleteSimulation) {
    for k in 0..sim.random_outcome_indicator().len() {
        if sim.random_outcome_indicator()[k] {
            assert!(!sim.outcome_shift().index(k));
        }
    }
}

fn check_outcome_matrix_and_shift_properties(sim: &OutcomeCompleteSimulation) {
    check_random_outcomes_bit_shift(sim);
    let rank_profile: Vec<usize> = sim
        .random_outcome_indicator()
        .iter()
        .enumerate()
        .filter_map(|(k, is_random)| if *is_random { Some(k) } else { None })
        .collect();
    assert!(is_column_reduced_with_profile(&sim.outcome_matrix(), &rank_profile));
}

#[must_use]
pub fn is_column_reduced_with_profile(matrix: &BitMatrix, rank_profile: &[usize]) -> bool {
    for (col, row) in rank_profile.iter().enumerate() {
        if !is_standard_basis_element(&matrix.row(*row), col) {
            return false;
        }
    }
    let mut current_pivot_pos = 0;
    for row in 0..matrix.rowcount() {
        if current_pivot_pos < rank_profile.len() - 1 && row >= rank_profile[current_pivot_pos + 1] {
            current_pivot_pos += 1;
        }
        if !is_supported_on_first_k_bits(&matrix.row(row), current_pivot_pos + 1) {
            return false;
        }
    }
    true
}

#[must_use]
pub fn is_standard_basis_element(bitstring: &BitView, pos: usize) -> bool {
    bitstring.index(pos) && bitstring.weight() == 1
}

#[must_use]
pub fn is_supported_on_first_k_bits(bitstring: &BitView, k: usize) -> bool {
    (k..bitstring.len()).all(|index| !bitstring.index(index))
}

#[test]
fn outcome_sequence_test() {
    let mut sim = OutcomeCompleteSimulation::with_capacity(2, 8, 2);
    random_and_deterministic_outcome_sequence(&mut sim);
    assert_eq!(sim.random_outcome_count(), 2);
    assert_eq!(sim.random_outcome_indicator().len(), 8);
    // println!("{}", bitmatrix_to_string(sim.outcome_matrix()));
    // println!("s{}", to_string(&sim.outcome_shift().view()));
    // println!("{}", bitmatrix_to_string(sim.sign_matrix()));
    assert_eq!(
        &vec![true, false, false, false, false, false, true, false],
        sim.random_outcome_indicator()
    );
    check_outcome_matrix_and_shift_properties(&sim);

    // Check the relevant part of the outcome matrix (first 8 rows, 2 columns)
    let expected_values = [
        [true, false],
        [false, false],
        [true, false],
        [true, false],
        [true, false],
        [true, false],
        [false, true],
        [false, false],
    ];
    for (r, row) in expected_values.iter().enumerate() {
        for (c, &expected) in row.iter().enumerate() {
            assert_eq!(
                sim.outcome_matrix().row(r).index(c),
                expected,
                "Mismatch at position ({r}, {c})"
            );
        }
    }

    // Check the relevant part of outcome_shift (first 8 bits)
    let expected_shift = [false, false, false, true, false, false, false, true];
    for (i, &expected) in expected_shift.iter().enumerate() {
        assert_eq!(
            sim.outcome_shift().index(i),
            expected,
            "Mismatch at outcome_shift position {i}"
        );
    }
}

#[test]
fn large_capacity_test() {
    let mut sim = OutcomeCompleteSimulation::with_capacity(3, 10000, 10000);
    let m_id = sim.measure_o(&[x(0)]);
    sim.apply_conditional_pauli_o(&[z(0)], &[m_id], true);
}

#[test]
fn test_outcome_specific_grows_automatically() {
    let mut sim = OutcomeSpecificSimulation::new(5);

    // Should start with no measurements
    assert_eq!(sim.outcome_vector().len(), 0);

    // Apply some gates
    sim.unitary_op(UnitaryOp::Hadamard, &[0]);
    sim.unitary_op(UnitaryOp::ControlledX, &[0, 1]);

    // Measure - should grow automatically
    let _outcome = sim.measure_o(&[z(0)]);

    // Should have grown
    assert_eq!(sim.outcome_vector().len(), 1);
}

#[test]
fn test_outcome_specific_many_measurements() {
    let mut sim = OutcomeSpecificSimulation::new(10);

    // Perform many measurements beyond initial capacity
    for i in 0..50 {
        sim.unitary_op(UnitaryOp::Hadamard, &[i % 10]);
        let _outcome = sim.measure_o(&[z(i % 10)]);
    }

    // Should have grown to accommodate all measurements
    assert_eq!(sim.outcome_vector().len(), 50);
}

#[test]
fn test_outcome_complete_grows_automatically() {
    let mut sim = OutcomeCompleteSimulation::new(5);

    // Should start with no measurements
    assert_eq!(sim.random_outcome_indicator().len(), 0);

    // Apply some gates
    sim.unitary_op(UnitaryOp::Hadamard, &[0]);
    sim.unitary_op(UnitaryOp::ControlledX, &[0, 1]);

    // Measure - should grow automatically
    let _outcome = sim.measure_o(&[z(0)]);

    // Should have grown
    assert_eq!(sim.random_outcome_indicator().len(), 1);
}

#[test]
fn test_outcome_complete_many_measurements() {
    let mut sim = OutcomeCompleteSimulation::new(10);

    // Perform many measurements beyond initial capacity
    for i in 0..50 {
        sim.unitary_op(UnitaryOp::Hadamard, &[i % 10]);
        let _outcome = sim.measure_o(&[z(i % 10)]);
    }

    // Should have grown to accommodate all measurements
    assert_eq!(sim.random_outcome_indicator().len(), 50);
}

#[test]
fn test_simulation_grows_beyond_initial_capacity() {
    // Verify automatic growth works correctly
    let num_qubits = 5;

    let mut sim = OutcomeCompleteSimulation::new(num_qubits);

    // Perform many measurements to trigger multiple resizes
    for i in 0..100 {
        sim.unitary_op(UnitaryOp::Hadamard, &[i % num_qubits]);
        let _outcome = sim.measure_o(&[z(i % num_qubits)]);
    }

    // Should have grown to accommodate all measurements
    assert_eq!(sim.random_outcome_indicator().len(), 100);
}

#[test]
fn test_custom_bit_source() {
    struct AllTrueIterator;

    impl Iterator for AllTrueIterator {
        type Item = bool;
        fn next(&mut self) -> Option<bool> {
            Some(true)
        }
    }

    let mut sim = OutcomeSpecificSimulation::new_with_bit_source(2, AllTrueIterator);

    // Measuring Z on |+⟩ is random and uses the bit source
    sim.unitary_op(UnitaryOp::Hadamard, &[0]);
    sim.measure_o(&[z(0)]);

    sim.unitary_op(UnitaryOp::Hadamard, &[1]);
    sim.measure_o(&[z(1)]);

    // Verify bit source values are used directly as outcomes
    assert!(sim.outcome_vector()[0]);
    assert!(sim.outcome_vector()[1]);
}

#[test]
fn test_bit_source_with_capacity() {
    struct CountingBitIterator {
        count: usize,
    }

    impl Iterator for CountingBitIterator {
        type Item = bool;
        fn next(&mut self) -> Option<bool> {
            let result = self.count % 2 == 1;
            self.count += 1;
            Some(result)
        }
    }

    let mut sim = OutcomeSpecificSimulation::with_bit_source_and_capacity(3, CountingBitIterator { count: 0 }, 10);

    for i in 0..5 {
        sim.unitary_op(UnitaryOp::Hadamard, &[i % 3]);
        sim.measure_o(&[z(i % 3)]);
    }

    assert_eq!(sim.outcome_vector().len(), 5);
}

#[test]
fn test_zero_outcomes_with_capacity() {
    let mut sim = OutcomeSpecificSimulation::with_zero_outcomes_and_capacity(4, 20);

    for i in 0..10 {
        sim.unitary_op(UnitaryOp::Hadamard, &[i % 4]);
        sim.measure_o(&[z(i % 4)]);
        assert!(!sim.outcome_vector()[i]);
    }

    assert_eq!(sim.outcome_vector().len(), 10);
}

#[test]
fn test_seeded_random_outcomes() {
    let mut sim1 = OutcomeSpecificSimulation::new_with_seeded_random_outcomes(2, 42);
    let mut sim2 = OutcomeSpecificSimulation::new_with_seeded_random_outcomes(2, 42);

    // Both simulations with same seed should produce identical outcomes
    for _ in 0..5 {
        sim1.unitary_op(UnitaryOp::Hadamard, &[0]);
        sim1.measure_o(&[z(0)]);

        sim2.unitary_op(UnitaryOp::Hadamard, &[0]);
        sim2.measure_o(&[z(0)]);
    }

    assert_eq!(sim1.outcome_vector(), sim2.outcome_vector());
}

fn assert_outcome_complete_simulation_properties_consistent(sim: &OutcomeCompleteSimulation) {
    let random_outcome_count = sim.random_outcome_count();
    let outcome_count = sim.outcome_count();
    // let qubit_count = sim.state_encoder().num_qubits();
    // assert_eq!(qubit_count, sim.state_encoder().num_qubits());
    // assert_eq!(qubit_count, sim.sign_matrix().rowcount());
    assert_eq!(random_outcome_count, sim.sign_matrix().columncount());

    assert_eq!(outcome_count, sim.outcome_matrix().rowcount());
    assert_eq!(random_outcome_count, sim.outcome_matrix().columncount());

    assert_eq!(outcome_count, sim.outcome_shift().len());
    assert_eq!(outcome_count, sim.random_outcome_indicator().len());
}

fn assert_outcome_free_simulation_properties_consistent(sim: &OutcomeFreeSimulation) {
    let random_outcome_count = sim.random_outcome_count();
    let outcome_count = sim.outcome_count();
    let qubit_count = sim.qubit_count();
    assert_eq!(qubit_count, sim.state_encoder().num_qubits());
    assert_eq!(outcome_count, sim.random_outcome_indicator().len());
    assert!(random_outcome_count <= outcome_count);
}

fn assert_outcome_specific_simulation_properties_consistent(sim: &OutcomeSpecificSimulation) {
    let random_outcome_count = sim.random_outcome_count();
    let outcome_count = sim.outcome_vector().len();
    let qubit_count = sim.qubit_count();
    assert_eq!(qubit_count, sim.state_encoder().num_qubits());
    assert_eq!(outcome_count, sim.outcome_vector().len());
    assert_eq!(outcome_count, sim.random_outcome_indicator().len());
    assert!(random_outcome_count <= outcome_count);
}

fn assert_simulations_properties_equal(sim1: &impl Simulation, sim2: &impl Simulation) {
    assert_eq!(sim1.qubit_count(), sim2.qubit_count());
    assert_eq!(sim1.random_outcome_count(), sim2.random_outcome_count());
    assert_eq!(sim1.random_outcome_indicator(), sim2.random_outcome_indicator());
    assert_eq!(sim1.outcome_count(), sim2.outcome_count());
}

fn does_encode_all_zero(clifford: &CliffordUnitary) -> bool {
    for qubit_index in 0..clifford.num_qubits() {
        if !clifford.preimage_z(qubit_index).is_pauli_z(qubit_index) {
            return false;
        }
    }
    true
}

fn compare_simulations(
    outcome_free_circuit: impl Fn(&mut OutcomeFreeSimulation),
    outcome_complete_circuit: impl Fn(&mut OutcomeCompleteSimulation),
    outcome_specific_circuit: impl Fn(&mut OutcomeSpecificSimulation),
) {
    let mut outcome_free = OutcomeFreeSimulation::default();
    let mut outcome_complete = OutcomeCompleteSimulation::default();
    let mut outcome_specific_zero = OutcomeSpecificSimulation::new_with_zero_outcomes(0);

    outcome_free_circuit(&mut outcome_free);
    outcome_complete_circuit(&mut outcome_complete);
    outcome_specific_circuit(&mut outcome_specific_zero);

    assert_outcome_complete_simulation_properties_consistent(&outcome_complete);
    assert_outcome_free_simulation_properties_consistent(&outcome_free);
    assert_outcome_specific_simulation_properties_consistent(&outcome_specific_zero);

    assert_simulations_properties_equal(&outcome_free, &outcome_complete);
    assert_simulations_properties_equal(&outcome_complete, &outcome_specific_zero);
    assert_eq!(&outcome_free.state_encoder(), outcome_complete.state_encoder().as_ref());
    assert_eq!(outcome_complete.state_encoder(), outcome_specific_zero.state_encoder());
    assert_eq!(
        outcome_complete.outcome_shift().iter().collect::<Vec<_>>(),
        *outcome_specific_zero.outcome_vector()
    );
    let outcome_matrix = outcome_complete.outcome_matrix();
    let outcome_shift = outcome_complete.outcome_shift().iter().collect::<Vec<_>>();
    let sign_matrix = outcome_complete.sign_matrix();
    let oc_encoder_inverse = outcome_complete.state_encoder().inverse();

    for non_zero_random_bit in 0..outcome_complete.random_outcome_count() {
        let bit = non_zero_random_bit;
        let bit_source = (0..).map(move |i| i == bit);
        let mut outcome_specific = OutcomeSpecificSimulation::new_with_bit_source(0, bit_source);
        outcome_specific_circuit(&mut outcome_specific);

        assert_simulations_properties_equal(&outcome_specific_zero, &outcome_specific);
        assert_outcome_specific_simulation_properties_consistent(&outcome_specific_zero);

        let mut expected_outcome_vector = outcome_matrix.column(non_zero_random_bit).iter().collect::<Vec<_>>();
        expected_outcome_vector.bitxor_assign(&outcome_shift);
        assert_eq!(expected_outcome_vector, *outcome_specific.outcome_vector());

        // State encoders match up to phases
        assert_eq!(
            outcome_specific.state_encoder().as_ref(),
            outcome_specific_zero.state_encoder().as_ref()
        );
        let mut encoder_diff = oc_encoder_inverse.multiply_with(&outcome_specific.state_encoder());
        let signs_support: IndexSet = sign_matrix.column(non_zero_random_bit).borrow().into();
        let zero_bits = IndexSet::default();
        encoder_diff.left_mul_pauli(&SparsePauli::from_bits(signs_support, zero_bits, 0));
        assert!(does_encode_all_zero(&encoder_diff));
    }
}

macro_rules! test_sims {
    ($circuit_function:ident) => {
        let (of, oc, os) = (
            $circuit_function::<OutcomeFreeSimulation>,
            $circuit_function::<OutcomeCompleteSimulation>,
            $circuit_function::<OutcomeSpecificSimulation>,
        );
        compare_simulations(of, oc, os);
    };
}

#[test]
fn test_compare_simulations() {
    test_sims!(choi_state_of_cx_via_measure);
    test_sims!(choi_state_of_cz_via_measure);
    test_sims!(random_and_deterministic_outcome_sequence);
}
