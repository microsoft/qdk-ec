use super::*;

// ================================================================================================
// Section 4.1 of arXiv:2603.24717: verifying parameterized state-preparation circuits.
//
// To decide whether two parameterized circuits prepare the same state for *every* rotation angle,
//   C₁ exp(iα Z) C₂|0…0>  ==  D₁ exp(iα Z) D₂|0…0>   (for all α),
// it suffices to check a single EXACT stabilizer-state equality with the angle replaced by a binary
// symbolic exponent,
//   C₁ Z^a C₂|0…0>  ==  D₁ Z^a D₂|0…0>,
// because exactness — equality including the relative phase between the a = 0 and a = 1 branches —
// pins down the rotation phase for every α. This needs no dedicated verification entry point: the
// check is exactly `phased_action_of` + `PhasedCircuitAction::is_equivalent`, the phased analog of
// how `OutcomeCompleteSimulation` performs phaseless equality checking. `Z^a` is realized by an
// `allocate_symbolic_angle` angle applied with `symbolic_pauli_exp`.
// ================================================================================================

/// Records a state-preparation gadget `C₁ (∏ₖ Z_k^{a_k}) C₂ |0…0>` as a phased action with no input
/// qubits — the `inputs = []` (state-preparation) case of `phased_action_of`.
pub(super) fn prepared_state_action(
    qubit_count: usize,
    build: impl FnOnce(&mut CircuitBuilder),
) -> PhasedCircuitAction {
    let outputs: Vec<QubitId> = (0..qubit_count).collect();
    let circuit = build_circuit(build);
    phased_action_of(&circuit, &[], &outputs).expect("state preparation action")
}

/// Prepares `|+…+>` by Hadamarding every qubit in `0..qubit_count`.
pub(super) fn prepare_plus(builder: &mut CircuitBuilder, qubit_count: usize) {
    for qubit in 0..qubit_count {
        builder.unitary_op(UnitaryOp::Hadamard, &[qubit]);
    }
}

/// Two different Clifford factorizations `C₁ Z^a C₂` of the same parameterized state must verify as
/// equivalent: `exp(iα Z₀Z₁)|++>` realized directly, versus the CNOT-conjugated single-qubit rotation
/// `CNOT₀₁ exp(iα Z₁) CNOT₀₁ |++>` (using `CNOT₀₁ Z₁ CNOT₀₁ = Z₀Z₁` and `CNOT₀₁|++> = |++>`).
#[test]
pub(super) fn verifies_equal_state_preparation_factorizations() {
    let direct = prepared_state_action(2, |builder| {
        prepare_plus(builder, 2);
        let angle = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[z(0), z(1)]), angle);
    });
    let conjugated = prepared_state_action(2, |builder| {
        prepare_plus(builder, 2);
        builder.unitary_op(UnitaryOp::ControlledX, &[0, 1]);
        let angle = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[z(1)]), angle);
        builder.unitary_op(UnitaryOp::ControlledX, &[0, 1]);
    });

    direct
        .is_equivalent(&conjugated)
        .expect("§4.1: the two factorizations prepare the same parameterized state");
    conjugated
        .is_equivalent(&direct)
        .expect("verification must be symmetric");
}

/// The check is phase-sensitive: `exp(+iα Z₀)|+>` and `exp(-iα Z₀)|+>` have identical stabilizer data
/// but opposite branch phase, so they must be distinguished — by exactly one `RelativePhase` reason.
#[test]
pub(super) fn detects_phase_only_state_preparation_difference() {
    let positive = prepared_state_action(1, |builder| {
        prepare_plus(builder, 1);
        let angle = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[z(0)]), angle);
    });
    let negative = prepared_state_action(1, |builder| {
        prepare_plus(builder, 1);
        let angle = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&-sparse(&[z(0)]), angle);
    });

    positive
        .is_equivalent_up_to_signs(&negative)
        .expect("the phaseless data is identical");
    let reasons = positive
        .is_equivalent(&negative)
        .expect_err("exp(+iαZ) and exp(-iαZ) prepare states differing only in branch phase");
    assert_eq!(reasons, vec![ActionsInequivalenceReason::RelativePhase]);
}

/// The §4.1 reduction generalizes to several independent symbolic angles. `exp(iα Z₀Z₁) exp(iβ Z₀)|++>`
/// verifies equal to its CNOT-conjugated factorization (angles allocated in the same order, so the
/// virtual-angle bits correspond one to one), while negating the second rotation's Pauli yields a
/// pure branch-phase difference that is detected.
#[test]
pub(super) fn verifies_multi_angle_state_preparation() {
    let direct = |negate_second: bool| {
        prepared_state_action(2, move |builder| {
            prepare_plus(builder, 2);
            let first = builder.allocate_symbolic_angle();
            builder.symbolic_pauli_exp(&sparse(&[z(0), z(1)]), first);
            let second = builder.allocate_symbolic_angle();
            let pauli = if negate_second {
                -sparse(&[z(0)])
            } else {
                sparse(&[z(0)])
            };
            builder.symbolic_pauli_exp(&pauli, second);
        })
    };
    let conjugated = prepared_state_action(2, |builder| {
        prepare_plus(builder, 2);
        builder.unitary_op(UnitaryOp::ControlledX, &[0, 1]);
        let first = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[z(1)]), first);
        builder.unitary_op(UnitaryOp::ControlledX, &[0, 1]);
        let second = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[z(0)]), second);
    });

    direct(false)
        .is_equivalent(&conjugated)
        .expect("§4.1: the multi-angle factorizations prepare the same parameterized state");
    let reasons = direct(true)
        .is_equivalent(&conjugated)
        .expect_err("negating one rotation must produce a detectable branch-phase difference");
    assert_eq!(reasons, vec![ActionsInequivalenceReason::RelativePhase]);
}
