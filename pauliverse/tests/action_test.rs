use std::borrow::Borrow;
use std::str::FromStr;

use binar::matrix::AlignedBitMatrix;
use binar::{AffineMap, Bitwise, BitwiseMut, IndexSet};
use paulimer::clifford::{XOrZ, group_encoding_clifford_of, random_clifford_via_operations_sampling};
use paulimer::core::{x, y, z};
use paulimer::operations::diagonal_operations;
use paulimer::pauli::remapped_sparse;
use paulimer::{Clifford, CliffordMutable, CliffordUnitary, Pauli, PauliGroup, PauliMutable, SparsePauli};
use paulimer::{PositionedPauliObservable, UnitaryOp};
use pauliverse::action::action_of;
use pauliverse::{Circuit, CircuitBuilder, OutcomeId, QubitId, Simulation};
use rand::{Rng, SeedableRng};

#[test]
fn clifford_unitary_action_tests() {
    let seed = 12345;
    let qubit_count = 3;

    let random_number_generator = &mut rand::rngs::StdRng::seed_from_u64(seed);
    let unitary = CliffordUnitary::random(qubit_count, random_number_generator);
    clifford_unitary_action_test(&unitary);
}

fn clifford_unitary_action_test(unitary: &CliffordUnitary) {
    let (circuit, input, output) = one_unitary_circuit_with_io(unitary);
    let action = action_of(&circuit, &input, &output).expect("unitary action");
    check_unitary_action(unitary, &input, &output, &action);
}

#[test]
fn measurement_action_test() {
    let pauli_strings = ["X", "Y", "Z", "XX", "XY", "XZ", "YY", "YZ", "-ZZ", "-XYZ"];
    for pauli_string in pauli_strings {
        let pauli = SparsePauli::from_str(pauli_string).unwrap();
        pauli_measurement_action_test(&pauli);
    }

    let pauli = &[z(0), z(1)].into();
    let (circuit0, input0, output0, sign_support0) = zz_via_plus_with_io();
    let (circuit1, input1, output1, sign_support1) = measure_circuit_with_io(pauli);
    let action0 = action_of(&circuit0, &input0, &output0).expect("measurement action");
    let action1 = action_of(&circuit1, &input1, &output1).expect("measurement action");
    check_pauli_measurement_action(pauli, &input0, &output0, &action0, &sign_support0);
    check_pauli_measurement_action(pauli, &input1, &output1, &action1, &sign_support1);

    assert_eq!(sign_support1.len(), 1);
    let affine_map = affine_map_from_sparse(
        action0.outcome_count(),
        action1.outcome_count(),
        vec![(sign_support1[0], false, sign_support0.as_slice())],
    );
    action1
        .is_equivalent_with_map(&action0, Some(&affine_map))
        .expect("actions must be equivalent with outcome mapping");
}

fn pauli_measurement_action_test(pauli: &SparsePauli) {
    let (circuit, input, output, sign_support) = measure_circuit_with_io(pauli);
    let action = action_of(&circuit, &input, &output).expect("measurement action");
    check_pauli_measurement_action(pauli, &input, &output, &action, &sign_support);
}

#[test]
fn prepare_bell_action_test() {
    let (circuit0, input0, output0) = bell_pair_with_io();
    let (circuit1, input1, output1) = long_range_bell_pair_with_io();
    let action0 = action_of(&circuit0, &input0, &output0).expect("Bell pair preparation action");
    let action1 = action_of(&circuit1, &input1, &output1).expect("Long range Bell pair preparation action");
    action0
        .is_equivalent_up_to_signs(&action1)
        .expect("actions must be equivalent up to signs");
    action0
        .is_equivalent_with_map(&action1, None)
        .expect("actions must be equivalent");
    action1
        .is_equivalent_with_map(&action0, None)
        .expect("actions must be equivalent");

    check_bell_pair(&circuit0, &input0, &output0);
    check_bell_pair(&circuit1, &input1, &output1);
}

#[test]
fn long_range_cnot_test() {
    let (circuit, input, output) = cnot_via_bell_with_io();
    let mut cnot_01 = CliffordUnitary::identity(2);
    cnot_01.left_mul_cx(0, 1);
    let action = action_of(&circuit, &input, &output).expect("CNOT via Bell pair action");
    check_unitary_action(&cnot_01, &input, &output, &action);
}

#[test]
fn diagonal_unitary_ejection_test() {
    let seed = 54654;
    let qubit_count = 3;

    let random_number_generator = &mut rand::rngs::StdRng::seed_from_u64(seed);

    let z_diagonal_unitary = random_diagonal_clifford_unitary(qubit_count, random_number_generator);
    let (circuit, input) = diagonal_unitary_ejection_circuit_with_io(&z_diagonal_unitary);
    check_and_compare_unitary(&z_diagonal_unitary, &circuit, &input);
}

#[test]
fn diagonal_unitary_injection_test() {
    let seed = 54654;
    let qubit_count = 3;

    let random_number_generator = &mut rand::rngs::StdRng::seed_from_u64(seed);

    let z_diagonal_unitary = random_diagonal_clifford_unitary(qubit_count, random_number_generator);
    let (circuit, input) = diagonal_unitary_injection_circuit_with_io(&z_diagonal_unitary);
    check_and_compare_unitary(&z_diagonal_unitary, &circuit, &input);
}

#[test]
fn diagonal_measure_ejection_test() {
    let seed = 54654;
    let qubit_count = 3;
    let random_number_generator = &mut rand::rngs::StdRng::seed_from_u64(seed);

    let z_diagonal_paulis = random_independent_z_paulis(qubit_count, qubit_count, random_number_generator);
    let pauli_count = z_diagonal_paulis.len();
    let (ejection_circuit, input_output, outcome_map) = diagonal_measure_ejection_circuit_with_io(&z_diagonal_paulis);
    let ejection_action =
        action_of(&ejection_circuit, &input_output, &input_output).expect("diagonal measure ejection action");
    let (measure_circuit, measure_input_output) = multi_measure_circuit_with_io(&z_diagonal_paulis);
    let measure_action =
        action_of(&measure_circuit, &measure_input_output, &measure_input_output).expect("diagonal measure action");
    let map = affine_map_from_sparse(2 * pauli_count, pauli_count, outcome_map);
    measure_action
        .is_equivalent_with_map(&ejection_action, Some(&map))
        .expect(
            "diagonal measure ejection action should be equivalent to diagonal measure action with outcome mapping",
        );
}

fn check_and_compare_unitary(
    z_diagonal_unitary: &CliffordUnitary,
    circuit: &Circuit,
    input_and_output_qubits: &[usize],
) {
    assert!(z_diagonal_unitary.is_diagonal(XOrZ::Z));
    let action =
        action_of(circuit, input_and_output_qubits, input_and_output_qubits).expect("diagonal ejection action");
    check_unitary_action(
        z_diagonal_unitary,
        input_and_output_qubits,
        input_and_output_qubits,
        &action,
    );

    let (unitary_circuit, unitary_input, unitary_output) = one_unitary_circuit_with_io(z_diagonal_unitary);
    let unitary_action = action_of(&unitary_circuit, &unitary_input, &unitary_output).expect("diagonal unitary action");
    unitary_action
        .is_equivalent_with_map(&action, None)
        .expect("diagonal ejection action should be equivalent to unitary action");
}

/// Validation of some common kinds of action
fn check_bell_pair(circuit: &Circuit, input: &[usize], output: &[usize]) {
    let action = action_of(circuit, input, output).expect("Bell pair preparation action");
    assert!(
        action.observables().is_empty(),
        "Bell pair preparation should have no observables"
    );
    assert_eq!(
        action.stabilizers().len(),
        2,
        "Bell pair preparation should have two stabilizers"
    );

    let bell_stabilizers = PauliGroup::from_strings(&["XX", "ZZ"]);
    let actual_stabilizers = PauliGroup::new(
        &action
            .signed_stabilizers()
            .iter()
            .map(|s| s.pauli.clone())
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        bell_stabilizers.standard_generators(),
        actual_stabilizers.standard_generators(),
        "Bell pair stabilizers should be XX and ZZ"
    );

    for signed in action.signed_stabilizers() {
        assert!(
            signed.outcomes_sign_mask.is_zero(),
            "Bell pair stabilizer signs must be fixed, got {}",
            signed.outcomes_sign_mask
        );
    }
}

fn check_unitary_action(
    unitary: &CliffordUnitary,
    input: &[usize],
    output: &[usize],
    action: &pauliverse::action::CircuitAction,
) {
    assert!(
        action.observables().is_empty(),
        "unitary action should have no observables, got {:?}",
        action.observables()
    );
    assert!(
        action.stabilizers().is_empty(),
        "unitary action should have no stabilizers, got {:?}",
        action.stabilizers()
    );

    let choi_group = choi_group(action);
    let output_support = output_support(input, output);
    for qubit_id in unitary.qubits() {
        let image_x = remapped_sparse(&unitary.image_x(qubit_id), &output_support);
        let image_z = remapped_sparse(&unitary.image_z(qubit_id), &output_support);
        let choi_stabilizer_x = SparsePauli::x(qubit_id, 0) * &image_x;
        let choi_stabilizer_z = SparsePauli::z(qubit_id, 0) * &image_z;
        assert!(choi_group.contains(&choi_stabilizer_x));
        assert!(choi_group.contains(&choi_stabilizer_z));
    }

    for sign_pauli in action.signed_choi_state_stabilizers() {
        assert!(
            sign_pauli.outcomes_sign_mask.is_zero(),
            "unitary action should have no sign dependence, got {}",
            sign_pauli.outcomes_sign_mask
        );
    }
}

fn check_pauli_measurement_action(
    pauli: &paulimer::pauli::PauliUnitary<binar::IndexSet, u8>,
    input: &[usize],
    output: &[usize],
    action: &pauliverse::action::CircuitAction,
    expected_sign_support: &[usize],
) {
    assert_eq!(
        action.observables().len(),
        1,
        "measurement action should have exactly one observable"
    );
    assert_eq!(
        action.stabilizers().len(),
        1,
        "measurement action should have exactly one stabilizer"
    );
    assert_eq!(
        action.signed_observables()[0].pauli,
        pauli,
        "observable should match the measured Pauli"
    );
    assert_eq!(
        action.signed_stabilizers()[0].pauli,
        pauli,
        "stabilizer should match the measured Pauli"
    );
    assert_eq!(
        action.signed_observables()[0].sign_support(),
        expected_sign_support,
        "observable sign is determined by the measurement outcome"
    );
    assert_eq!(
        action.signed_stabilizers()[0].sign_support(),
        expected_sign_support,
        "stabilizer sign is determined by the measurement outcome"
    );

    let encoder = group_encoding_clifford_of(std::slice::from_ref(pauli), input.len());
    assert_eq!(encoder.image_z(0), pauli);
    let choi_group = choi_group(action);
    let output_support = output_support(input, output);
    for qubit_id in 1..input.len() {
        let mut image_x: SparsePauli = encoder.image_x(qubit_id).into();
        let mut image_z: SparsePauli = encoder.image_z(qubit_id).into();
        let remapped_image_x = remapped_sparse(&image_x, &output_support);
        let remapped_image_z = remapped_sparse(&image_z, &output_support);
        image_x.complex_conjugate();
        image_z.complex_conjugate();
        assert!(choi_group.contains(&(&image_x * &remapped_image_x)));
        assert!(choi_group.contains(&(&image_z * &remapped_image_z)));
    }
}

///// Circuit generation methods

fn one_unitary_circuit_with_io(unitary: &CliffordUnitary) -> (Circuit, Vec<QubitId>, Vec<QubitId>) {
    let qubits = unitary.qubits().collect::<Vec<QubitId>>();

    let circuit = empty_builder().clifford(unitary, &qubits).into_circuit();

    (circuit, qubits.clone(), qubits)
}

fn measure_circuit_with_io(pauli: &SparsePauli) -> (Circuit, Vec<QubitId>, Vec<QubitId>, Vec<OutcomeId>) {
    let max_qubit_id = pauli.max_support().expect("Non trivial support required");
    let qubits = (0..=max_qubit_id).collect::<Vec<QubitId>>();

    let circuit = empty_builder().measure_sparse(pauli, 0).into_circuit();

    (circuit, qubits.clone(), qubits, vec![0])
}

fn long_range_bell_pair_with_io() -> (Circuit, Vec<QubitId>, Vec<QubitId>) {
    let (q0, q1, q2, q3) = (0, 1, 2, 3);
    let input_qubits = vec![];
    let output_qubits = vec![q0, q3];

    let circuit = empty_builder()
        .prepare_bell(q0, q1)
        .prepare_bell(q2, q3)
        .measure_xx(q1, q2, 0)
        .measure_zz(q1, q2, 1)
        .conditional_zz(q2, q3, &[0], true)
        .conditional_xx(q2, q3, &[1], true)
        .into_circuit();

    (circuit, input_qubits, output_qubits)
}

fn bell_pair_with_io() -> (Circuit, Vec<QubitId>, Vec<QubitId>) {
    let (q0, q1) = (0, 1);
    let input_qubits = vec![];
    let output_qubits = vec![q0, q1];

    let circuit = empty_builder().prepare_bell(q0, q1).into_simulator().into_circuit();

    (circuit, input_qubits, output_qubits)
}

fn cnot_via_bell_with_io() -> (Circuit, Vec<QubitId>, Vec<QubitId>) {
    let (control, b1, b2, target) = (0, 3, 1, 2);
    let (o_z, o_x) = (0, 1);
    let input_qubits = vec![control, target];
    let output_qubits = vec![control, target];

    let circuit = empty_builder()
        .prepare_bell(b1, b2)
        .cx(control, b1)
        .measure_z(b1, o_z)
        .cx(b2, target)
        .measure_x(b2, o_x)
        .conditional_z(control, &[o_x], true)
        .conditional_x(target, &[o_z], true)
        .into_circuit();

    (circuit, input_qubits, output_qubits)
}

fn zz_via_plus_with_io() -> (Circuit, Vec<QubitId>, Vec<QubitId>, Vec<OutcomeId>) {
    let (q0, aux0, q1) = (0, 1, 2);
    let input_qubits = vec![q0, q1];
    let output_qubits = vec![q0, q1];
    let (zz0, zz1, x_aux) = (0, 1, 2);

    let circuit = empty_builder()
        .h(aux0)
        .measure_zz(aux0, q0, zz0)
        .measure_zz(aux0, q1, zz1)
        .measure_x(aux0, x_aux)
        .conditional_z(q1, &[x_aux], true)
        .into_circuit();

    (circuit, input_qubits, output_qubits, vec![zz0, zz1])
}

/// Implements `z_diagonal_unitary` via diagonal ejection
fn diagonal_unitary_ejection_circuit_with_io(z_diagonal_unitary: &CliffordUnitary) -> (Circuit, Vec<QubitId>) {
    assert!(z_diagonal_unitary.is_diagonal(XOrZ::Z));
    let qubit_count = z_diagonal_unitary.num_qubits();
    let targets = (0..qubit_count).collect::<Vec<QubitId>>();
    let references = (qubit_count..2 * qubit_count).collect::<Vec<QubitId>>();

    let mut b = empty_builder();
    for (&target, &reference) in targets.iter().zip(references.iter()) {
        b = b.cnot(target, reference);
    }
    b = b.clifford(z_diagonal_unitary, &references);
    for (id, (&target, &reference)) in targets.iter().zip(references.iter()).enumerate() {
        b = b.measure_x(reference, id).conditional_z(target, &[id], true);
    }

    (b.into_circuit(), targets)
}

type OutcomeMapping = Vec<(OutcomeId, bool, Vec<OutcomeId>)>;

fn diagonal_measure_ejection_circuit_with_io(
    z_diagonal_paulis: &[SparsePauli],
) -> (Circuit, Vec<QubitId>, OutcomeMapping) {
    let qubit_count = z_diagonal_paulis
        .iter()
        .map(|p| p.max_support().expect("non trivial support"))
        .max()
        .map(|max_support| max_support + 1)
        .expect("at least one pauli should be provided");

    let targets = (0..qubit_count).collect::<Vec<QubitId>>();
    let references = (qubit_count..2 * qubit_count).collect::<Vec<QubitId>>();

    let mut b = empty_builder();
    for (&target, &reference) in targets.iter().zip(references.iter()) {
        b = b.cnot(target, reference);
    }

    let pauli_outcome_ids = 0..z_diagonal_paulis.len();
    for (pauli, outcome_id) in z_diagonal_paulis.iter().zip(pauli_outcome_ids.clone()) {
        let reference_pauli = remapped_sparse(pauli, &references);
        b = b.measure_sparse(&reference_pauli, outcome_id);
    }

    let x_outcome_ids = z_diagonal_paulis.len()..2 * z_diagonal_paulis.len();
    for (id, (&target, &reference)) in x_outcome_ids.zip(targets.iter().zip(references.iter())) {
        b = b.measure_x(reference, id).conditional_z(target, &[id], true);
    }

    let outcome_map = pauli_outcome_ids.map(|id| (id, false, vec![id])).collect::<Vec<_>>();
    (b.into_circuit(), targets, outcome_map)
}

fn multi_measure_circuit_with_io(z_diagonal_paulis: &[SparsePauli]) -> (Circuit, Vec<QubitId>) {
    let max_qubit_id = z_diagonal_paulis
        .iter()
        .map(|pauli| pauli.max_support().expect("Non trivial support required"))
        .max()
        .expect("At least one pauli should be provided");
    let qubits = (0..=max_qubit_id).collect::<Vec<QubitId>>();

    let mut b = empty_builder();
    for (id, pauli) in z_diagonal_paulis.iter().enumerate() {
        b = b.measure_sparse(pauli, id);
    }

    (b.into_circuit(), qubits)
}

fn diagonal_unitary_injection_circuit_with_io(z_diagonal_unitary: &CliffordUnitary) -> (Circuit, Vec<QubitId>) {
    assert!(z_diagonal_unitary.is_diagonal(XOrZ::Z));
    let qubit_count = z_diagonal_unitary.num_qubits();
    let targets = (0..qubit_count).collect::<Vec<QubitId>>();
    let references = (qubit_count..2 * qubit_count).collect::<Vec<QubitId>>();

    let mut b = empty_builder();

    // prepare magic state
    for &reference in &references {
        b = b.h(reference);
    }
    b = b.clifford(z_diagonal_unitary, &references);

    // transversal cnot from targets to references
    for (&target, &reference) in targets.iter().zip(references.iter()) {
        b = b.cnot(target, reference);
    }

    // measure references and apply corrections
    for (id, &reference) in references.iter().enumerate() {
        let pauli = remapped_sparse(&z_diagonal_unitary.image_x(id), &targets);
        b = b
            .measure_z(reference, id)
            .conditional_x(id, &[id], true)
            .conditional_sparse_pauli(&pauli, &[id], true);
    }

    (b.into_circuit(), targets)
}

/// A helper for building circuits & running simulations
#[must_use]
pub struct SimulationBuilder<Simulator: Simulation> {
    simulator: Simulator,
}

impl<Simulator: Simulation> SimulationBuilder<Simulator> {
    pub fn new(simulator: Simulator) -> Self {
        Self { simulator }
    }

    pub fn into_simulator(self) -> Simulator {
        self.simulator
    }

    pub fn into_circuit(self) -> Circuit
    where
        Simulator: Into<Circuit>,
    {
        self.simulator.into()
    }

    pub fn h(mut self, qubit: QubitId) -> Self {
        self.simulator.unitary_op(UnitaryOp::Hadamard, &[qubit]);
        self
    }

    pub fn hadamard(self, qubit: QubitId) -> Self {
        self.h(qubit)
    }

    pub fn swap(mut self, q0: QubitId, q1: QubitId) -> Self {
        self.simulator.unitary_op(UnitaryOp::Swap, &[q0, q1]);
        self
    }

    pub fn cnot(mut self, control: QubitId, target: QubitId) -> Self {
        self.simulator.unitary_op(UnitaryOp::ControlledX, &[control, target]);
        self
    }

    pub fn cx(self, control: QubitId, target: QubitId) -> Self {
        self.cnot(control, target)
    }

    pub fn cz(mut self, control: QubitId, target: QubitId) -> Self {
        self.simulator.unitary_op(UnitaryOp::ControlledZ, &[control, target]);
        self
    }

    pub fn prepare_bell(mut self, q0: QubitId, q1: QubitId) -> Self {
        self.simulator.unitary_op(UnitaryOp::PrepareBell, &[q0, q1]);
        self
    }

    pub fn conditional_pauli(
        mut self,
        pauli: &[PositionedPauliObservable],
        outcome_ids: &[usize],
        parity: bool,
    ) -> Self {
        self.simulator.conditional_pauli(&pauli.into(), outcome_ids, parity);
        self
    }

    pub fn conditional_sparse_pauli(mut self, pauli: &SparsePauli, outcome_ids: &[usize], parity: bool) -> Self {
        self.simulator.conditional_pauli(pauli, outcome_ids, parity);
        self
    }

    pub fn conditional_x(self, qubit: QubitId, outcome_ids: &[usize], parity: bool) -> Self {
        self.conditional_pauli(&[x(qubit)], outcome_ids, parity)
    }

    pub fn conditional_y(self, qubit: QubitId, outcome_ids: &[usize], parity: bool) -> Self {
        self.conditional_pauli(&[y(qubit)], outcome_ids, parity)
    }

    pub fn conditional_z(self, qubit: QubitId, outcome_ids: &[usize], parity: bool) -> Self {
        self.conditional_pauli(&[z(qubit)], outcome_ids, parity)
    }

    pub fn conditional_xx(self, q0: QubitId, q1: QubitId, outcome_ids: &[usize], parity: bool) -> Self {
        self.conditional_pauli(&[x(q0), x(q1)], outcome_ids, parity)
    }

    pub fn conditional_yy(self, q0: QubitId, q1: QubitId, outcome_ids: &[usize], parity: bool) -> Self {
        self.conditional_pauli(&[y(q0), y(q1)], outcome_ids, parity)
    }

    pub fn conditional_zz(self, q0: QubitId, q1: QubitId, outcome_ids: &[usize], parity: bool) -> Self {
        self.conditional_pauli(&[z(q0), z(q1)], outcome_ids, parity)
    }

    pub fn pauli(mut self, pauli: &[PositionedPauliObservable]) -> Self {
        self.simulator.pauli(&pauli.into());
        self
    }

    pub fn pauli_x(self, qubit: QubitId) -> Self {
        self.pauli(&[x(qubit)])
    }

    pub fn pauli_y(self, qubit: QubitId) -> Self {
        self.pauli(&[y(qubit)])
    }

    pub fn pauli_z(self, qubit: QubitId) -> Self {
        self.pauli(&[z(qubit)])
    }

    /// Measures the given observable and asserts the outcome id matches.
    ///
    /// # Panics
    ///
    /// Panics if the returned outcome id doesn't match the expected `outcome`.
    pub fn measure(mut self, observable: &[PositionedPauliObservable], outcome: OutcomeId) -> Self {
        let outcome_id = self.simulator.measure(&observable.into());
        assert_eq!(
            outcome_id, outcome,
            "Expected measurement outcome id {outcome} but got {outcome_id}"
        );
        self
    }

    pub fn measure_x(self, qubit: QubitId, outcome: OutcomeId) -> Self {
        self.measure(&[x(qubit)], outcome)
    }

    pub fn measure_y(self, qubit: QubitId, outcome: OutcomeId) -> Self {
        self.measure(&[y(qubit)], outcome)
    }

    pub fn measure_z(self, qubit: QubitId, outcome: OutcomeId) -> Self {
        self.measure(&[z(qubit)], outcome)
    }

    pub fn measure_xx(self, q0: QubitId, q1: QubitId, outcome: OutcomeId) -> Self {
        self.measure(&[x(q0), x(q1)], outcome)
    }

    pub fn measure_yy(self, q0: QubitId, q1: QubitId, outcome: OutcomeId) -> Self {
        self.measure(&[y(q0), y(q1)], outcome)
    }

    pub fn measure_zz(self, q0: QubitId, q1: QubitId, outcome: OutcomeId) -> Self {
        self.measure(&[z(q0), z(q1)], outcome)
    }

    pub fn sqrt_x(mut self, qubit: QubitId) -> Self {
        self.simulator.unitary_op(UnitaryOp::SqrtX, &[qubit]);
        self
    }

    pub fn sqrt_y(mut self, qubit: QubitId) -> Self {
        self.simulator.unitary_op(UnitaryOp::SqrtY, &[qubit]);
        self
    }

    pub fn sqrt_z(mut self, qubit: QubitId) -> Self {
        self.simulator.unitary_op(UnitaryOp::SqrtZ, &[qubit]);
        self
    }

    pub fn controlled_pauli(
        mut self,
        control: &[PositionedPauliObservable],
        target: &[PositionedPauliObservable],
    ) -> Self {
        self.simulator.controlled_pauli(&control.into(), &target.into());
        self
    }

    pub fn clifford(mut self, clifford: &CliffordUnitary, qubits: &[QubitId]) -> Self {
        self.simulator.clifford(clifford, qubits);
        self
    }

    /// Measures the given sparse Pauli observable and asserts the outcome id matches.
    ///
    /// # Panics
    ///
    /// Panics if the returned outcome id doesn't match the expected `outcome`.
    pub fn measure_sparse(mut self, observable: &SparsePauli, outcome: OutcomeId) -> Self {
        let outcome_id = self.simulator.measure(observable);
        assert_eq!(
            outcome_id, outcome,
            "Expected measurement outcome id {outcome} but got {outcome_id}"
        );
        self
    }
}

fn empty_builder() -> SimulationBuilder<CircuitBuilder> {
    SimulationBuilder::new(CircuitBuilder::new())
}

fn output_support(input: &[usize], output: &[usize]) -> Vec<usize> {
    (input.len()..output.len() + input.len()).collect::<Vec<usize>>()
}

fn choi_group(action: &pauliverse::action::CircuitAction) -> PauliGroup {
    PauliGroup::new(
        action
            .signed_choi_state_stabilizers()
            .iter()
            .map(|s| s.pauli.clone())
            .collect::<Vec<_>>()
            .as_slice(),
    )
}

fn affine_map_from_sparse<T, U>(
    input_dimension: usize,
    output_dimension: usize,
    sparse_map: impl IntoIterator<Item = (OutcomeId, bool, T)>,
) -> AffineMap
where
    T: IntoIterator<Item = U>,
    U: Borrow<usize>,
{
    let mut res = AffineMap::zero(input_dimension, output_dimension);
    for (outcome_id, parity, dependencies) in sparse_map {
        res.shift_mut().assign_index(outcome_id, parity);
        for dep in dependencies {
            res.matrix_mut().set((outcome_id, *dep.borrow()), true);
        }
    }
    res
}

fn random_diagonal_clifford_unitary(qubit_count: usize, random_number_generator: &mut impl Rng) -> CliffordUnitary {
    let operations = diagonal_operations(qubit_count);
    random_clifford_via_operations_sampling(
        qubit_count,
        qubit_count * qubit_count,
        &operations,
        random_number_generator,
    )
}

fn random_independent_z_paulis(
    qubit_count: usize,
    count: usize,
    random_number_generator: &mut impl Rng,
) -> Vec<SparsePauli> {
    let matrix = AlignedBitMatrix::random_invertible(qubit_count, random_number_generator);
    (0..count)
        .map(|i| {
            let x_bits = IndexSet::new();
            let z_bits = matrix.row(i).support().collect();
            SparsePauli::from_bits(x_bits, z_bits, 0)
        })
        .collect()
}
