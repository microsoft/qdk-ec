use super::*;

// ================================================================================================
// "Ejection": remote/measurement-based execution of a Z-diagonal channel.
//
// A Z-diagonal channel on `n` system qubits is applied indirectly: `n` ancillas (prepared in |0⟩)
// receive a transversal CNOT from the system qubits, the Z-diagonal channel acts on the ancillas,
// each ancilla is destructively measured in the X basis, and a "−" outcome triggers a conditional Z
// correction on the corresponding system qubit. The action must equal applying the same Z-diagonal
// channel directly to the system qubits. The X-basis measurements introduce *true* random bits that
// must be marginalized, while the channel's symbolic rotation angles are *virtual* bits that must
// correspond one-to-one — exactly the mixed case the virtual/true distinction is built for.
// ================================================================================================

/// A tensor product of `Z` operators, one on each qubit `support[i]` selected by `local_indices`.
/// `local_indices` name positions *within* `support`, letting the same channel description
/// (`angle_supports`) be applied either to the system qubits or to the ancillas.
pub(super) fn z_product(local_indices: &[usize], support: &[QubitId]) -> SparsePauli {
    let positioned: Vec<PositionedPauliObservable> = local_indices.iter().map(|&index| z(support[index])).collect();
    (&positioned[..]).into()
}

/// Applies the symbolic Z-rotations indexed by `angle_supports` (each a tensor product of `Z`s, with
/// its own symbolic angle) to the qubits named by `support`, in allocation order.
pub(super) fn apply_symbolic_z_rotations(
    builder: &mut CircuitBuilder,
    angle_supports: &[Vec<usize>],
    support: &[QubitId],
) {
    for local_indices in angle_supports {
        let angle = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&z_product(local_indices, support), angle);
    }
}

/// The direct circuit: the symbolic Z-rotations applied straight to the `n` system qubits.
pub(super) fn direct_z_channel(n: usize, angle_supports: &[Vec<usize>]) -> Circuit {
    let system: Vec<QubitId> = (0..n).collect();
    build_circuit(|builder| {
        apply_symbolic_z_rotations(builder, angle_supports, &system);
    })
}

/// The ejection circuit: the same Z-rotations are executed *remotely* on `n` fresh ancillas and then
/// teleported back onto the system. Qubits `0..n` are the system, `n..2n` the ancillas. Three phases:
///   1. entangle — a transversal CNOT copies each system qubit onto its ancilla,
///   2. rotate remotely — the symbolic Z-rotations act on the ancillas instead of the system,
///   3. measure and correct — each ancilla is measured in the X basis (a *true* random bit), and a
///      `−` outcome triggers a conditional `Z` correction on the matching system qubit.
///
/// The net channel on the system must equal [`direct_z_channel`].
pub(super) fn z_ejection_channel(n: usize, angle_supports: &[Vec<usize>]) -> Circuit {
    let system: Vec<QubitId> = (0..n).collect();
    let ancillas: Vec<QubitId> = (n..2 * n).collect();
    build_circuit(|builder| {
        for (&system_qubit, &ancilla) in system.iter().zip(ancillas.iter()) {
            builder.unitary_op(UnitaryOp::ControlledX, &[system_qubit, ancilla]);
        }
        apply_symbolic_z_rotations(builder, angle_supports, &ancillas);
        for (&system_qubit, &ancilla) in system.iter().zip(ancillas.iter()) {
            let minus_outcome = builder.measure(&sparse(&[x(ancilla)]));
            builder.conditional_pauli(&sparse(&[z(system_qubit)]), &[minus_outcome], true);
        }
    })
}

/// Asserts the ejection invariant: executing the Z-rotations remotely and teleporting them back is
/// the same channel — including every branch phase — as applying them directly. Checked both ways so
/// the equivalence is symmetric.
pub(super) fn check_z_ejection(n: usize, angle_supports: &[Vec<usize>]) {
    let system: Vec<QubitId> = (0..n).collect();
    let direct = direct_z_channel(n, angle_supports);
    let ejection = z_ejection_channel(n, angle_supports);

    let direct_action = phased_action_of(&direct, &system, &system).expect("direct channel action");
    let ejection_action = phased_action_of(&ejection, &system, &system).expect("ejection channel action");

    direct_action.is_equivalent(&ejection_action).unwrap_or_else(|reasons| {
        panic!("ejection of {angle_supports:?} on {n} qubits must equal the direct channel: {reasons:?}")
    });
    ejection_action
        .is_equivalent(&direct_action)
        .expect("ejection equivalence must be symmetric");
}

#[test]
pub(super) fn single_qubit_z_rotation_ejection() {
    check_z_ejection(1, &[vec![0]]);
}

#[test]
pub(super) fn two_qubit_z_rotation_ejections() {
    check_z_ejection(2, &[vec![0]]);
    check_z_ejection(2, &[vec![1]]);
    check_z_ejection(2, &[vec![0, 1]]);
    check_z_ejection(2, &[vec![0], vec![1], vec![0, 1]]);
}

#[test]
pub(super) fn three_qubit_all_z_products_ejection() {
    let all_nontrivial: Vec<Vec<usize>> = (1u32..8)
        .map(|mask| (0..3).filter(|bit| mask & (1 << bit) != 0).collect())
        .collect();
    check_z_ejection(3, &all_nontrivial);
}

#[test]
pub(super) fn repeated_angles_ejection() {
    check_z_ejection(2, &[vec![0], vec![0], vec![0, 1], vec![0, 1]]);
}

/// A wrong correction (conditioning on the wrong measurement outcome) must be detected: the branch
/// phase then depends on a true measurement bit that the direct channel cannot reproduce.
#[test]
pub(super) fn miscorrected_ejection_is_detected() {
    let n = 2;
    let angle_supports = [vec![0, 1]];
    let system: Vec<QubitId> = (0..n).collect();
    let ancillas: Vec<QubitId> = (n..2 * n).collect();
    let broken = build_circuit(|builder| {
        for (&system_qubit, &ancilla) in system.iter().zip(ancillas.iter()) {
            builder.unitary_op(UnitaryOp::ControlledX, &[system_qubit, ancilla]);
        }
        apply_symbolic_z_rotations(builder, &angle_supports, &ancillas);
        let mut outcomes = Vec::new();
        for &ancilla in &ancillas {
            outcomes.push(builder.measure(&sparse(&[x(ancilla)])));
        }
        // Apply only the first correction, dropping the second: leaves a residual outcome dependence.
        builder.conditional_pauli(&sparse(&[z(system[0])]), &[outcomes[0]], true);
    });

    let direct = direct_z_channel(n, &angle_supports);
    let direct_action = phased_action_of(&direct, &system, &system).expect("direct action");
    let broken_action = phased_action_of(&broken, &system, &system).expect("broken action");

    direct_action
        .is_equivalent(&broken_action)
        .expect_err("a missing correction must make the ejection inequivalent");
}

// A Z-diagonal *Clifford* (here `S` on each ancilla plus a `CZ`) carries a non-trivial phase but no
// symbolic angles. Ejecting it must equal applying it directly — the **no-angle** case, where the
// phased equivalence reduces exactly to the phaseless `OutcomeCompleteSimulation` behaviour: the only
// random bits are the corrected true X-measurement outcomes, so the relative-phase check is vacuous
// and the residual `|±⟩` ancillas (left uncleaned, exactly as in the phaseless ejection precedent)
// do not affect the comparison.

pub(super) fn apply_z_diagonal_clifford(builder: &mut CircuitBuilder, support: &[QubitId]) {
    for &qubit in support {
        builder.unitary_op(UnitaryOp::SqrtZ, &[qubit]);
    }
    for window in support.windows(2) {
        builder.unitary_op(UnitaryOp::ControlledZ, &[window[0], window[1]]);
    }
}

pub(super) fn direct_z_clifford_channel(n: usize) -> Circuit {
    let system: Vec<QubitId> = (0..n).collect();
    build_circuit(|builder| apply_z_diagonal_clifford(builder, &system))
}

pub(super) fn z_clifford_ejection_channel(n: usize) -> Circuit {
    let system: Vec<QubitId> = (0..n).collect();
    let ancillas: Vec<QubitId> = (n..2 * n).collect();
    build_circuit(|builder| {
        for (&system_qubit, &ancilla) in system.iter().zip(ancillas.iter()) {
            builder.unitary_op(UnitaryOp::ControlledX, &[system_qubit, ancilla]);
        }
        apply_z_diagonal_clifford(builder, &ancillas);
        for (&system_qubit, &ancilla) in system.iter().zip(ancillas.iter()) {
            let outcome = builder.measure(&sparse(&[x(ancilla)]));
            builder.conditional_pauli(&sparse(&[z(system_qubit)]), &[outcome], true);
        }
    })
}

#[test]
pub(super) fn z_diagonal_clifford_ejection_without_angles() {
    for n in 1..=3 {
        let system: Vec<QubitId> = (0..n).collect();
        let direct = direct_z_clifford_channel(n);
        let ejection = z_clifford_ejection_channel(n);
        let direct_action = phased_action_of(&direct, &system, &system).expect("direct clifford action");
        let ejection_action = phased_action_of(&ejection, &system, &system).expect("ejection clifford action");

        direct_action.is_equivalent(&ejection_action).unwrap_or_else(|reasons| {
            panic!("no-angle Z-diagonal Clifford ejection on {n} qubits must equal direct: {reasons:?}")
        });
        ejection_action
            .is_equivalent(&direct_action)
            .expect("no-angle ejection equivalence must be symmetric");
    }
}

// ================================================================================================
// X-basis "ejection": the Hadamard dual of the Z-basis gadget above.
//
// An X-diagonal channel on `n` system qubits is applied indirectly: `n` ancillas (prepared in |+⟩)
// drive a transversal CNOT *into* the system qubits (control = ancilla, target = system), the
// X-diagonal channel acts on the ancillas, each ancilla is destructively measured in the Z basis,
// and a `1` outcome triggers a conditional X correction on the corresponding system qubit. Because
// this whole gadget is the conjugation of the (already verified) Z-basis gadget by a Hadamard on
// every system and ancilla qubit, it must equal applying the same X-diagonal channel directly. As in
// the Z case, the Z-basis measurements introduce *true* random bits that must be marginalized, while
// the symbolic rotation angles are *virtual* bits that correspond one-to-one.
// ================================================================================================

/// `X` on each `qubits[i]`-th entry of `support` (a tensor product of `X` operators).
pub(super) fn x_product(qubits: &[usize], support: &[QubitId]) -> SparsePauli {
    let positioned: Vec<PositionedPauliObservable> = qubits.iter().map(|&qubit| x(support[qubit])).collect();
    (&positioned[..]).into()
}

/// Applies the symbolic X-rotations indexed by `angle_supports` (each a tensor product of `X`s, with
/// its own symbolic angle) to the qubits named by `support`, in allocation order.
pub(super) fn apply_symbolic_x_rotations(
    builder: &mut CircuitBuilder,
    angle_supports: &[Vec<usize>],
    support: &[QubitId],
) {
    for qubits in angle_supports {
        let angle = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&x_product(qubits, support), angle);
    }
}

/// The direct circuit: the symbolic X-rotations applied straight to the `n` system qubits.
pub(super) fn direct_x_channel(n: usize, angle_supports: &[Vec<usize>]) -> Circuit {
    let system: Vec<QubitId> = (0..n).collect();
    build_circuit(|builder| {
        apply_symbolic_x_rotations(builder, angle_supports, &system);
    })
}

/// The ejection circuit: the same symbolic X-rotations executed remotely on `n` ancillas.
pub(super) fn x_ejection_channel(n: usize, angle_supports: &[Vec<usize>]) -> Circuit {
    let system: Vec<QubitId> = (0..n).collect();
    let ancillas: Vec<QubitId> = (n..2 * n).collect();
    build_circuit(|builder| {
        for &ancilla in &ancillas {
            builder.unitary_op(UnitaryOp::Hadamard, &[ancilla]);
        }
        for (&system_qubit, &ancilla) in system.iter().zip(ancillas.iter()) {
            builder.unitary_op(UnitaryOp::ControlledX, &[ancilla, system_qubit]);
        }
        apply_symbolic_x_rotations(builder, angle_supports, &ancillas);
        for (&system_qubit, &ancilla) in system.iter().zip(ancillas.iter()) {
            let outcome = builder.measure(&sparse(&[z(ancilla)]));
            builder.conditional_pauli(&sparse(&[x(system_qubit)]), &[outcome], true);
        }
    })
}

pub(super) fn check_x_ejection(n: usize, angle_supports: &[Vec<usize>]) {
    let system: Vec<QubitId> = (0..n).collect();
    let direct = direct_x_channel(n, angle_supports);
    let ejection = x_ejection_channel(n, angle_supports);

    let direct_action = phased_action_of(&direct, &system, &system).expect("direct channel action");
    let ejection_action = phased_action_of(&ejection, &system, &system).expect("ejection channel action");

    direct_action.is_equivalent(&ejection_action).unwrap_or_else(|reasons| {
        panic!("X ejection of {angle_supports:?} on {n} qubits must equal the direct channel: {reasons:?}")
    });
    ejection_action
        .is_equivalent(&direct_action)
        .expect("X ejection equivalence must be symmetric");
}

#[test]
pub(super) fn single_qubit_x_rotation_ejection() {
    check_x_ejection(1, &[vec![0]]);
}

#[test]
pub(super) fn two_qubit_x_rotation_ejections() {
    check_x_ejection(2, &[vec![0]]);
    check_x_ejection(2, &[vec![1]]);
    check_x_ejection(2, &[vec![0, 1]]);
    check_x_ejection(2, &[vec![0], vec![1], vec![0, 1]]);
}

#[test]
pub(super) fn three_qubit_all_x_products_ejection() {
    let all_nontrivial: Vec<Vec<usize>> = (1u32..8)
        .map(|mask| (0..3).filter(|bit| mask & (1 << bit) != 0).collect())
        .collect();
    check_x_ejection(3, &all_nontrivial);
}

#[test]
pub(super) fn repeated_x_angles_ejection() {
    check_x_ejection(2, &[vec![0], vec![0], vec![0, 1], vec![0, 1]]);
}

// ================================================================================================
// Ejection of a *general* Z-diagonal channel: symbolic Z-rotations (virtual bits) mixed with
// non-destructive Z-basis measurements (true observed bits). Ejecting the channel onto ancillas adds
// a third kind of bit, the destructive X-readout outcomes (true auxiliary bits, marginalized). The
// default `is_equivalent` resolves all three automatically: the virtual angle bits correspond one to
// one, the observed measurement bits map identity-by-allocation-order, and the readout bits are
// projected out. This is the first gadget that exercises all three provenance classes at once.
// ================================================================================================

/// Applies a Z-diagonal *channel* to `support`: first the symbolic Z-rotations indexed by
/// `angle_supports`, then non-destructive Z-basis measurements of the tensor products indexed by
/// `measure_supports`, all in allocation order.
pub(super) fn apply_z_diagonal_channel(
    builder: &mut CircuitBuilder,
    angle_supports: &[Vec<usize>],
    measure_supports: &[Vec<usize>],
    support: &[QubitId],
) {
    apply_symbolic_z_rotations(builder, angle_supports, support);
    for qubits in measure_supports {
        let _ = builder.measure(&z_product(qubits, support));
    }
}

pub(super) fn direct_z_channel_with_measurements(
    n: usize,
    angle_supports: &[Vec<usize>],
    measure_supports: &[Vec<usize>],
) -> Circuit {
    let system: Vec<QubitId> = (0..n).collect();
    build_circuit(|builder| {
        apply_z_diagonal_channel(builder, angle_supports, measure_supports, &system);
    })
}

pub(super) fn z_ejection_channel_with_measurements(
    n: usize,
    angle_supports: &[Vec<usize>],
    measure_supports: &[Vec<usize>],
) -> Circuit {
    let system: Vec<QubitId> = (0..n).collect();
    let ancillas: Vec<QubitId> = (n..2 * n).collect();
    build_circuit(|builder| {
        for (&system_qubit, &ancilla) in system.iter().zip(ancillas.iter()) {
            builder.unitary_op(UnitaryOp::ControlledX, &[system_qubit, ancilla]);
        }
        apply_z_diagonal_channel(builder, angle_supports, measure_supports, &ancillas);
        for (&system_qubit, &ancilla) in system.iter().zip(ancillas.iter()) {
            let outcome = builder.measure(&sparse(&[x(ancilla)]));
            builder.conditional_pauli(&sparse(&[z(system_qubit)]), &[outcome], true);
        }
    })
}

pub(super) fn check_z_ejection_with_measurements(
    n: usize,
    angle_supports: &[Vec<usize>],
    measure_supports: &[Vec<usize>],
) {
    let system: Vec<QubitId> = (0..n).collect();
    let direct = direct_z_channel_with_measurements(n, angle_supports, measure_supports);
    let ejection = z_ejection_channel_with_measurements(n, angle_supports, measure_supports);

    let direct_action = phased_action_of(&direct, &system, &system).expect("direct channel action");
    let ejection_action = phased_action_of(&ejection, &system, &system).expect("ejection channel action");

    direct_action.is_equivalent(&ejection_action).unwrap_or_else(|reasons| {
        panic!("Z ejection of angles {angle_supports:?} and measurements {measure_supports:?} on {n} qubits must equal the direct channel: {reasons:?}")
    });
    ejection_action
        .is_equivalent(&direct_action)
        .expect("Z channel ejection equivalence must be symmetric");
}

#[test]
pub(super) fn single_qubit_z_channel_ejection() {
    check_z_ejection_with_measurements(1, &[vec![0]], &[vec![0]]);
    check_z_ejection_with_measurements(1, &[], &[vec![0]]);
}

#[test]
pub(super) fn two_qubit_z_channel_ejections() {
    check_z_ejection_with_measurements(2, &[vec![0]], &[vec![1]]);
    check_z_ejection_with_measurements(2, &[vec![0], vec![1]], &[vec![0, 1]]);
    check_z_ejection_with_measurements(2, &[vec![0, 1]], &[vec![0], vec![1]]);
    check_z_ejection_with_measurements(2, &[], &[vec![0], vec![1], vec![0, 1]]);
}

#[test]
pub(super) fn three_qubit_z_channel_ejection() {
    check_z_ejection_with_measurements(3, &[vec![0, 1, 2]], &[vec![0], vec![1, 2]]);
}

// ================================================================================================
// X-basis dual of the general-channel ejection above: symbolic X-rotations mixed with
// non-destructive X-basis measurements, ejected through `|+⟩` ancillas with reversed CNOTs,
// destructive Z-readout, and conditional X corrections.
// ================================================================================================

/// Applies an X-diagonal *channel* to `support`: the symbolic X-rotations indexed by
/// `angle_supports`, then non-destructive X-basis measurements indexed by `measure_supports`.
pub(super) fn apply_x_diagonal_channel(
    builder: &mut CircuitBuilder,
    angle_supports: &[Vec<usize>],
    measure_supports: &[Vec<usize>],
    support: &[QubitId],
) {
    apply_symbolic_x_rotations(builder, angle_supports, support);
    for qubits in measure_supports {
        let _ = builder.measure(&x_product(qubits, support));
    }
}

pub(super) fn direct_x_channel_with_measurements(
    n: usize,
    angle_supports: &[Vec<usize>],
    measure_supports: &[Vec<usize>],
) -> Circuit {
    let system: Vec<QubitId> = (0..n).collect();
    build_circuit(|builder| {
        apply_x_diagonal_channel(builder, angle_supports, measure_supports, &system);
    })
}

pub(super) fn x_ejection_channel_with_measurements(
    n: usize,
    angle_supports: &[Vec<usize>],
    measure_supports: &[Vec<usize>],
) -> Circuit {
    let system: Vec<QubitId> = (0..n).collect();
    let ancillas: Vec<QubitId> = (n..2 * n).collect();
    build_circuit(|builder| {
        for &ancilla in &ancillas {
            builder.unitary_op(UnitaryOp::Hadamard, &[ancilla]);
        }
        for (&system_qubit, &ancilla) in system.iter().zip(ancillas.iter()) {
            builder.unitary_op(UnitaryOp::ControlledX, &[ancilla, system_qubit]);
        }
        apply_x_diagonal_channel(builder, angle_supports, measure_supports, &ancillas);
        for (&system_qubit, &ancilla) in system.iter().zip(ancillas.iter()) {
            let outcome = builder.measure(&sparse(&[z(ancilla)]));
            builder.conditional_pauli(&sparse(&[x(system_qubit)]), &[outcome], true);
        }
    })
}

pub(super) fn check_x_ejection_with_measurements(
    n: usize,
    angle_supports: &[Vec<usize>],
    measure_supports: &[Vec<usize>],
) {
    let system: Vec<QubitId> = (0..n).collect();
    let direct = direct_x_channel_with_measurements(n, angle_supports, measure_supports);
    let ejection = x_ejection_channel_with_measurements(n, angle_supports, measure_supports);

    let direct_action = phased_action_of(&direct, &system, &system).expect("direct channel action");
    let ejection_action = phased_action_of(&ejection, &system, &system).expect("ejection channel action");

    direct_action.is_equivalent(&ejection_action).unwrap_or_else(|reasons| {
        panic!("X ejection of angles {angle_supports:?} and measurements {measure_supports:?} on {n} qubits must equal the direct channel: {reasons:?}")
    });
    ejection_action
        .is_equivalent(&direct_action)
        .expect("X channel ejection equivalence must be symmetric");
}

#[test]
pub(super) fn single_qubit_x_channel_ejection() {
    check_x_ejection_with_measurements(1, &[vec![0]], &[vec![0]]);
    check_x_ejection_with_measurements(1, &[], &[vec![0]]);
}

#[test]
pub(super) fn two_qubit_x_channel_ejections() {
    check_x_ejection_with_measurements(2, &[vec![0]], &[vec![1]]);
    check_x_ejection_with_measurements(2, &[vec![0], vec![1]], &[vec![0, 1]]);
    check_x_ejection_with_measurements(2, &[vec![0, 1]], &[vec![0], vec![1]]);
    check_x_ejection_with_measurements(2, &[], &[vec![0], vec![1], vec![0, 1]]);
}

#[test]
pub(super) fn three_qubit_x_channel_ejection() {
    check_x_ejection_with_measurements(3, &[vec![0, 1, 2]], &[vec![0], vec![1, 2]]);
}
