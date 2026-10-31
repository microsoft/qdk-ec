use super::*;

#[test]
fn native_action_checks_qubits_beyond_named_outputs() {
    let circuit = build_circuit(|builder| builder.unitary_op(UnitaryOp::PrepareBell, &[0, 1]));
    let mut simulation = PhasedOutcomeCompleteSimulation::new(2);
    circuit.simulate(&mut simulation).unwrap();
    for action in [
        phased_action_of(&circuit, &[], &[0]),
        phased_action_from_simulation(&simulation, &[], &[0]),
    ] {
        assert!(matches!(
            action,
            Err(ActionError::AuxiliaryQubitsEntangled { auxiliary_qubits, .. }) if auxiliary_qubits == [1]
        ));
    }

    simulation.unitary_op(UnitaryOp::ControlledX, &[0, 1]);
    let action = phased_action_from_simulation(&simulation, &[], &[0]).unwrap();
    assert_eq!(action.action().auxiliary_qubits(), [1]);
}

#[test]
fn native_action_excludes_reference_qubits_from_auxiliary_support() {
    let mut simulation = PhasedOutcomeCompleteSimulation::new(3);
    simulation.unitary_op(UnitaryOp::PrepareBell, &[0, 1]);
    let action = phased_action_from_simulation(&simulation, &[0], &[0]).unwrap();
    assert_eq!(action.action().auxiliary_qubits(), [2]);

    simulation.unitary_op(UnitaryOp::ControlledX, &[0, 2]);
    assert!(matches!(
        phased_action_from_simulation(&simulation, &[0], &[0]),
        Err(ActionError::AuxiliaryQubitsEntangled { auxiliary_qubits, .. }) if auxiliary_qubits == [2]
    ));
}

/// Conditional `±Z₀` gadget: the two signs share the same symplectic action but differ only in the
/// branch phase of the odd branch.
pub(super) fn signed_z_rotation(negate: bool) -> (Circuit, Vec<QubitId>, Vec<QubitId>) {
    let circuit = build_circuit(|builder| {
        let branch = builder.allocate_symbolic_angle();
        let observable = if negate { -sparse(&[z(0)]) } else { sparse(&[z(0)]) };
        builder.symbolic_pauli_exp(&observable, branch);
    });
    (circuit, vec![0], vec![0])
}

// ================================================================================================
// Negative and randomized tests: a phased equivalence is exact, so any sign flip on a symbolic Pauli
// exponent or any permutation of the symbolic angles on the right-hand side must be detected. The
// phaseless action is unchanged (the symplectic content is identical), so the failure is a pure
// `RelativePhase` (sign flips) or a support/phase mismatch (permutations). Random instances exercise
// the same invariants over many angle supports, qubit counts, and qubit assignments.
// ================================================================================================

/// Direct Z-channel, but the `k`-th symbolic Z-rotation is `exp(±iα Z…)` according to `signs[k]`.
pub(super) fn signed_z_channel(n: usize, angle_supports: &[Vec<usize>], signs: &[bool]) -> Circuit {
    let system: Vec<QubitId> = (0..n).collect();
    build_circuit(|builder| {
        for (qubits, &negate) in angle_supports.iter().zip(signs.iter()) {
            let angle = builder.allocate_symbolic_angle();
            let pauli = if negate {
                -z_product(qubits, &system)
            } else {
                z_product(qubits, &system)
            };
            builder.symbolic_pauli_exp(&pauli, angle);
        }
    })
}

/// Direct Z-channel whose `k`-th allocated angle drives the rotation on `angle_supports[perm[k]]`.
pub(super) fn permuted_z_channel(n: usize, angle_supports: &[Vec<usize>], perm: &[usize]) -> Circuit {
    let system: Vec<QubitId> = (0..n).collect();
    build_circuit(|builder| {
        for &target in perm {
            let angle = builder.allocate_symbolic_angle();
            builder.symbolic_pauli_exp(&z_product(&angle_supports[target], &system), angle);
        }
    })
}

/// All non-trivial Z products on `n` qubits, in ascending-mask order: distinct, independent supports.
pub(super) fn distinct_z_supports(n: usize) -> Vec<Vec<usize>> {
    (1u32..(1 << n))
        .map(|mask| (0..n).filter(|bit| mask & (1 << bit) != 0).collect())
        .collect()
}

#[test]
pub(super) fn flipping_any_sign_yields_relative_phase() {
    let n = 2;
    let supports = distinct_z_supports(n);
    let system: Vec<QubitId> = (0..n).collect();
    let baseline = signed_z_channel(n, &supports, &vec![false; supports.len()]);
    let baseline_action = phased_action_of(&baseline, &system, &system).expect("baseline action");
    for mask in 1u32..(1 << supports.len()) {
        let signs: Vec<bool> = (0..supports.len()).map(|k| mask & (1 << k) != 0).collect();
        let flipped = signed_z_channel(n, &supports, &signs);
        let flipped_action = phased_action_of(&flipped, &system, &system).expect("flipped action");
        baseline_action
            .is_equivalent_up_to_signs(&flipped_action)
            .expect("sign flips leave the phaseless action unchanged");
        let reasons = baseline_action
            .is_equivalent(&flipped_action)
            .expect_err("sign mask must be detected");
        assert_eq!(
            reasons,
            vec![ActionsInequivalenceReason::RelativePhase],
            "mask {mask:b}"
        );
    }
}

#[test]
pub(super) fn permuting_distinct_angles_is_detected() {
    let n = 2;
    let supports = distinct_z_supports(n);
    let system: Vec<QubitId> = (0..n).collect();
    let identity: Vec<usize> = (0..supports.len()).collect();
    let base = permuted_z_channel(n, &supports, &identity);
    let base_action = phased_action_of(&base, &system, &system).expect("identity action");
    base_action
        .is_equivalent(&base_action)
        .expect("identity permutation is self-equivalent");
    let swapped = [1usize, 0, 2];
    let swapped_action = phased_action_of(&permuted_z_channel(n, &supports, &swapped), &system, &system).expect("swap");
    base_action
        .is_equivalent(&swapped_action)
        .expect_err("permuting distinct angles must be inequivalent");
}
