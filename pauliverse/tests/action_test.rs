use std::str::FromStr;

use binar::Bitwise;
use paulimer::{Clifford, CliffordUnitary, Pauli, SparsePauli};
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

// MORE TESTS TO BE ADDED
