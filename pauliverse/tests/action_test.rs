use std::str::FromStr;

use binar::Bitwise;
use paulimer::clifford::group_encoding_clifford_of;
use paulimer::core::{x, z};
use paulimer::pauli::remapped_sparse;
use paulimer::UnitaryOp;
use paulimer::{Clifford, CliffordMutable, CliffordUnitary, Pauli, PauliGroup, PauliMutable, SparsePauli};
use pauliverse::action::action_of;
use pauliverse::{Circuit, CircuitBuilder, QubitId, Simulation};
use rand::SeedableRng;

#[test]
fn clifford_unitary_action_tests() {
    let seed = 12345;
    let qubit_count = 3;

    let random_number_generator = &mut rand::rngs::StdRng::seed_from_u64(seed);
    let unitary = CliffordUnitary::random(qubit_count, random_number_generator);
    clifford_unitary_action_test(&unitary);
}

#[test]
fn measurement_action_test() {
    let pauli_strings = ["X", "Y", "Z", "XX", "XY", "XZ", "YY", "YZ", "-ZZ", "-XYZ"];
    for pauli_string in pauli_strings {
        let pauli = SparsePauli::from_str(pauli_string).unwrap();
        pauli_measurement_action_test(&pauli);
    }
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

fn clifford_unitary_action_test(unitary: &CliffordUnitary) {
    let (circuit, input, output) = one_unitary_circuit_with_io(unitary);
    let action = action_of(&circuit, &input, &output).expect("unitary action");
    check_unitary_action(unitary, &input, &output, &action);
}

fn check_unitary_action(
    unitary: &CliffordUnitary,
    input: &[usize],
    output: &[usize],
    action: &pauliverse::action::CircuitAction,
) {
    assert!(
        action.observables().is_empty(),
        "unitary action should have no observables"
    );
    assert!(
        action.stabilizers().is_empty(),
        "unitary action should have no stabilizers"
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

fn pauli_measurement_action_test(pauli: &SparsePauli) {
    let (circuit, input, output) = measure_circuit_with_io(pauli);
    let action = action_of(&circuit, &input, &output).expect("measurement action");
    check_pauli_measurement_action(pauli, &input, &output, &action);
}

fn check_pauli_measurement_action(
    pauli: &paulimer::pauli::PauliUnitary<binar::IndexSet, u8>,
    input: &[usize],
    output: &[usize],
    action: &pauliverse::action::CircuitAction,
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
        action.signed_observables()[0]
            .outcomes_sign_mask
            .support()
            .collect::<Vec<usize>>(),
        &[0],
        "observable sign is determined by the measurement outcome"
    );
    assert_eq!(
        action.signed_stabilizers()[0]
            .outcomes_sign_mask
            .support()
            .collect::<Vec<usize>>(),
        &[0],
        "observable sign is determined by the measurement outcome"
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

fn one_unitary_circuit_with_io(unitary: &CliffordUnitary) -> (Circuit, Vec<QubitId>, Vec<QubitId>) {
    let mut builder = CircuitBuilder::new();
    let qubits = unitary.qubits().collect::<Vec<QubitId>>();
    builder.clifford(unitary, &qubits);
    (builder.into_circuit(), qubits.clone(), qubits)
}

fn measure_circuit_with_io(pauli: &SparsePauli) -> (Circuit, Vec<QubitId>, Vec<QubitId>) {
    let mut builder = CircuitBuilder::new();
    let max_qubit_id = pauli.max_support().expect("Non trivial support required");
    let qubits = (0..=max_qubit_id).collect::<Vec<QubitId>>();
    builder.measure(pauli);
    (builder.into_circuit(), qubits.clone(), qubits)
}

fn long_range_bell_pair_with_io() -> (Circuit, Vec<QubitId>, Vec<QubitId>) {
    let mut builder = CircuitBuilder::new();
    let (q0, q1, q2, q3) = (0, 1, 2, 3);
    builder.unitary_op(UnitaryOp::PrepareBell, &[q0, q1]);
    builder.unitary_op(UnitaryOp::PrepareBell, &[q2, q3]);
    let x_outcome = builder.measure(&[x(q1), x(q2)].into());
    let z_outcome = builder.measure(&[z(q1), z(q2)].into());
    builder.conditional_pauli(&[z(q2), z(q3)].into(), &[x_outcome], true);
    builder.conditional_pauli(&[x(q2), x(q3)].into(), &[z_outcome], true);
    (builder.into_circuit(), vec![], vec![q0, q3])
}

fn bell_pair_with_io() -> (Circuit, Vec<QubitId>, Vec<QubitId>) {
    let mut builder = CircuitBuilder::new();
    let q0 = 0;
    let q1 = 1;
    builder.unitary_op(UnitaryOp::PrepareBell, &[q0, q1]);
    let input_qubits = vec![];
    let output_qubits = vec![q0, q1];
    (builder.into_circuit(), input_qubits, output_qubits)
}

fn cnot_via_bell_with_io() -> (Circuit, Vec<QubitId>, Vec<QubitId>) {
    let mut builder = CircuitBuilder::new();
    let (control, b1, b2, target) = (0, 3, 1, 2);
    builder.unitary_op(UnitaryOp::PrepareBell, &[b1, b2]);
    builder.unitary_op(UnitaryOp::ControlledX, &[control, b1]);
    let z_outcome = builder.measure(&[z(b1)].into());
    builder.unitary_op(UnitaryOp::ControlledX, &[b2, target]);
    let x_outcome = builder.measure(&[x(b2)].into());
    builder.conditional_pauli(&[z(control)].into(), &[x_outcome], true);
    builder.conditional_pauli(&[x(target)].into(), &[z_outcome], true);
    (builder.into_circuit(), vec![control, target], vec![control, target])
}
