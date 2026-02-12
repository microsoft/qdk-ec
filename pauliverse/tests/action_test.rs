use std::str::FromStr;

use binar::Bitwise;
use paulimer::{Clifford, CliffordUnitary, Pauli, PauliGroup, SparsePauli};
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
    let (circuit, input, output) = bell_pair_with_io();
    let action = action_of(&circuit, &input, &output).expect("Bell pair preparation action");
    assert!(
        action.observables().is_empty(),
        "Bell pair preparation should have no observables"
    );
    assert_eq!(
        action.stabilizers().len(),
        2,
        "Bell pair preparation should have two stabilizers"
    );
    for signed in action.signed_stabilizers() {
        assert!(
            signed.outcomes_sign_mask.is_empty(),
            "Bell pair stabilizer signs are fixed"
        );
    }
    let bell_stabilizers = PauliGroup::from_strings(&["XX","ZZ"]);
    let actual_stabilizers =  PauliGroup::new(&action.signed_stabilizers().iter().map(|s| s.pauli.clone()).collect::<Vec<_>>());
    
    assert_eq!(bell_stabilizers.standard_generators(), actual_stabilizers.standard_generators(), "Bell pair stabilizers should be XX and ZZ");
}

fn clifford_unitary_action_test(unitary: &CliffordUnitary) {
    let (circuit, input, output) = one_unitary_circuit_with_io(unitary);
    let action = action_of(&circuit, &input, &output).expect("unitary action");
    assert!(
        action.observables().is_empty(),
        "unitary action should have no observables"
    );
    assert!(
        action.stabilizers().is_empty(),
        "unitary action should have no stabilizers"
    );
}

fn pauli_measurement_action_test(pauli: &SparsePauli) {
    let (circuit, input, output) = measure_circuit_with_io(pauli);
    let action = action_of(&circuit, &input, &output).expect("measurement action");
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

/// Build a long-range Bell pair circuit using entanglement swapping.
///
/// This creates a Bell pair between q0 and q3 by:
/// 1. Preparing local Bell pairs on (q0, q1) and (q2, q3)
/// 2. Performing a Bell measurement on the middle qubits (q1, q2)
/// 3. Applying Pauli corrections to q3 based on measurement outcomes
///
/// Returns the circuit and the input/output qubit lists (q0 and q3).
#[allow(dead_code)]
fn long_range_bell_pair_with_io() -> (Circuit, Vec<QubitId>, Vec<QubitId>) {
    use paulimer::UnitaryOp;

    let mut builder = CircuitBuilder::new();

    let q0 = 0;
    let q1 = 1;
    let q2 = 2;
    let q3 = 3;

    builder.unitary_op(UnitaryOp::PrepareBell, &[q0, q1]);
    builder.unitary_op(UnitaryOp::PrepareBell, &[q2, q3]);

    let xx_observable = SparsePauli::from_str("IXXI").unwrap();
    let zz_observable = SparsePauli::from_str("IZZI").unwrap();

    let x_outcome = builder.measure(&xx_observable);
    let z_outcome = builder.measure(&zz_observable);

    let z_on_q3 = SparsePauli::from_str("IIIZ").unwrap();
    let x_on_q3 = SparsePauli::from_str("IIIX").unwrap();

    builder.conditional_pauli(&z_on_q3, &[z_outcome], true);
    builder.conditional_pauli(&x_on_q3, &[x_outcome], true);

    let input_qubits = vec![];
    let output_qubits = vec![q0, q3];

    (builder.into_circuit(), input_qubits, output_qubits)
}

fn bell_pair_with_io() -> (Circuit, Vec<QubitId>, Vec<QubitId>) {
    use paulimer::UnitaryOp;
    let mut builder = CircuitBuilder::new();
    let q0 = 0;
    let q1 = 1;
    builder.unitary_op(UnitaryOp::PrepareBell, &[q0, q1]);
    let input_qubits = vec![];
    let output_qubits = vec![q0, q1];
    (builder.into_circuit(), input_qubits, output_qubits)
}

/// Build a CNOT gate implemented via Bell pair consumption.
///
/// This implements CNOT(control -> target) by:
/// 1. Preparing a Bell pair on auxiliary qubits (b1, b2)
/// 2. Applying CNOT(control, b1) and measuring b1 in Z basis
/// 3. Applying CNOT(b2, target) and measuring b2 in X basis
/// 4. Applying conditional corrections based on measurement outcomes
///
/// The Bell pair is consumed in the process.
/// Returns the circuit and the input/output qubit lists (control, target).
#[allow(dead_code)]
fn cnot_via_bell_with_io() -> (Circuit, Vec<QubitId>, Vec<QubitId>) {
    use paulimer::UnitaryOp;

    let mut builder = CircuitBuilder::new();

    let control = 0;
    let b1 = 1;
    let b2 = 2;
    let target = 3;

    builder.unitary_op(UnitaryOp::PrepareBell, &[b1, b2]);

    builder.unitary_op(UnitaryOp::ControlledX, &[control, b1]);
    let z_on_b1 = SparsePauli::from_str("IZII").unwrap();
    let b_outcome = builder.measure(&z_on_b1);

    builder.unitary_op(UnitaryOp::ControlledX, &[b2, target]);
    let x_on_b2 = SparsePauli::from_str("IIXI").unwrap();
    let a_outcome = builder.measure(&x_on_b2);

    let x_on_target = SparsePauli::from_str("IIIX").unwrap();
    let z_on_control = SparsePauli::from_str("ZIII").unwrap();

    builder.conditional_pauli(&x_on_target, &[a_outcome], true);
    builder.conditional_pauli(&z_on_control, &[b_outcome], true);

    let input_qubits = vec![control, target];
    let output_qubits = vec![control, target];

    (builder.into_circuit(), input_qubits, output_qubits)
}
