use paulimer::core::{x, y, z};
use paulimer::pauli::SparsePauli;
use paulimer::{PositionedPauliObservable, UnitaryOp};
use pauliverse::action::{
    ActionsInequivalenceReason, PhasedCircuitAction, phased_action_from_simulation, phased_action_of,
};
use pauliverse::phased_outcome_complete_simulation::PhasedOutcomeCompleteSimulation;
use pauliverse::{Circuit, CircuitBuilder, QubitId, Simulation};
use proptest::prelude::*;
use rand::SeedableRng;
use std::ops::Range;

fn build_circuit(build: impl FnOnce(&mut CircuitBuilder)) -> Circuit {
    let mut builder = CircuitBuilder::new();
    build(&mut builder);
    builder.into()
}

fn sparse(observable: &[PositionedPauliObservable]) -> SparsePauli {
    observable.into()
}

/// `exp(iα Z₀Z₁)` represented as a symbolic rotation gadget: allocate a random branch bit, then
/// conditionally apply `Z₀Z₁` on the odd branch.
fn zz_rotation() -> (Circuit, Vec<QubitId>, Vec<QubitId>) {
    let circuit = build_circuit(|builder| {
        let branch = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[z(0), z(1)]), branch);
    });
    (circuit, vec![0, 1], vec![0, 1])
}

/// `CNOT₀₁ · exp(iα Z₁) · CNOT₀₁`, which should equal `exp(iα Z₀Z₁)` as a channel.
fn cnot_conjugated_z_rotation() -> (Circuit, Vec<QubitId>, Vec<QubitId>) {
    let circuit = build_circuit(|builder| {
        builder.unitary_op(UnitaryOp::ControlledX, &[0, 1]);
        let branch = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[z(1)]), branch);
        builder.unitary_op(UnitaryOp::ControlledX, &[0, 1]);
    });
    (circuit, vec![0, 1], vec![0, 1])
}

/// `exp(iα Z₁)` on its own, which differs from `exp(iα Z₀Z₁)` in symplectic action.
fn z_rotation() -> (Circuit, Vec<QubitId>, Vec<QubitId>) {
    let circuit = build_circuit(|builder| {
        let branch = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[z(1)]), branch);
    });
    (circuit, vec![0, 1], vec![0, 1])
}

/// Conditional `±Z₀` gadget: the two signs share the same symplectic action but differ only in the
/// branch phase of the odd branch.
fn signed_z_rotation(negate: bool) -> (Circuit, Vec<QubitId>, Vec<QubitId>) {
    let circuit = build_circuit(|builder| {
        let branch = builder.allocate_symbolic_angle();
        let observable = if negate { -sparse(&[z(0)]) } else { sparse(&[z(0)]) };
        builder.symbolic_pauli_exp(&observable, branch);
    });
    (circuit, vec![0], vec![0])
}

#[test]
fn zz_rotation_equals_cnot_conjugated_z_rotation() {
    let (direct, direct_input, direct_output) = zz_rotation();
    let (conjugated, conjugated_input, conjugated_output) = cnot_conjugated_z_rotation();

    let direct_action = phased_action_of(&direct, &direct_input, &direct_output).expect("direct action");
    let conjugated_action =
        phased_action_of(&conjugated, &conjugated_input, &conjugated_output).expect("conjugated action");

    direct_action
        .is_equivalent(&conjugated_action)
        .expect("symbolic rotations must agree including branch phase");
    conjugated_action
        .is_equivalent(&direct_action)
        .expect("equivalence must be symmetric");
}

#[test]
fn zz_rotation_differs_from_z_rotation() {
    let (zz, zz_input, zz_output) = zz_rotation();
    let (single, single_input, single_output) = z_rotation();

    let zz_action = phased_action_of(&zz, &zz_input, &zz_output).expect("zz action");
    let single_action = phased_action_of(&single, &single_input, &single_output).expect("z action");

    let reasons = zz_action
        .is_equivalent(&single_action)
        .expect_err("rotations with different supports must differ");
    assert!(!reasons.is_empty());
}

#[test]
fn opposite_sign_rotations_differ_only_in_relative_phase() {
    let (positive, positive_input, positive_output) = signed_z_rotation(false);
    let (negative, negative_input, negative_output) = signed_z_rotation(true);

    let positive_action = phased_action_of(&positive, &positive_input, &positive_output).expect("positive action");
    let negative_action = phased_action_of(&negative, &negative_input, &negative_output).expect("negative action");

    positive_action
        .is_equivalent_up_to_signs(&negative_action)
        .expect("phaseless actions must be identical");

    let reasons = positive_action
        .is_equivalent(&negative_action)
        .expect_err("opposite signs must be distinguished by the phased action");
    assert_eq!(reasons, vec![ActionsInequivalenceReason::RelativePhase]);
}

#[test]
fn rotation_equals_itself() {
    let (zz, zz_input, zz_output) = zz_rotation();
    let action = phased_action_of(&zz, &zz_input, &zz_output).expect("action");
    action
        .is_equivalent(&action)
        .expect("a rotation must be equivalent to itself");
}

/// Conjugating a `Z` rotation by a Hadamard produces the corresponding `X` rotation:
/// `H₀ · exp(iα Z₀) · H₀ == exp(iα X₀)`, including branch phase.
#[test]
fn hadamard_conjugated_z_rotation_equals_x_rotation() {
    let conjugated = build_circuit(|builder| {
        builder.unitary_op(UnitaryOp::Hadamard, &[0]);
        let branch = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[z(0)]), branch);
        builder.unitary_op(UnitaryOp::Hadamard, &[0]);
    });
    let x_rotation = build_circuit(|builder| {
        let branch = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[x(0)]), branch);
    });

    let conjugated_action = phased_action_of(&conjugated, &[0], &[0]).expect("conjugated action");
    let x_action = phased_action_of(&x_rotation, &[0], &[0]).expect("x action");

    conjugated_action
        .is_equivalent(&x_action)
        .expect("HZH must equal an X rotation, including branch phase");
    x_action
        .is_equivalent(&conjugated_action)
        .expect("equivalence must be symmetric");
}

/// A mixed-basis exponent equals its single-qubit Hadamard conjugate:
/// `exp(iα X₀Z₁) == H₀ · exp(iα Z₀Z₁) · H₀`, while the un-conjugated `exp(iα Z₀Z₁)` differs.
#[test]
fn mixed_basis_exponent_equals_hadamard_conjugated_zz() {
    let mixed = build_circuit(|builder| {
        let branch = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[x(0), z(1)]), branch);
    });
    let conjugated_zz = build_circuit(|builder| {
        builder.unitary_op(UnitaryOp::Hadamard, &[0]);
        let branch = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[z(0), z(1)]), branch);
        builder.unitary_op(UnitaryOp::Hadamard, &[0]);
    });

    let mixed_action = phased_action_of(&mixed, &[0, 1], &[0, 1]).expect("mixed action");
    let conjugated_action = phased_action_of(&conjugated_zz, &[0, 1], &[0, 1]).expect("conjugated action");

    mixed_action
        .is_equivalent(&conjugated_action)
        .expect("mixed-basis exponent must equal the Hadamard-conjugated ZZ exponent");

    let (bare_zz, bare_input, bare_output) = zz_rotation();
    let bare_action = phased_action_of(&bare_zz, &bare_input, &bare_output).expect("bare zz action");
    mixed_action
        .is_equivalent(&bare_action)
        .expect_err("the un-conjugated ZZ exponent is a different channel");
}

/// Builds the Choi state of a single-system-qubit gadget directly in a phased simulation, mirroring
/// the simulator-native idiom used by the Python bindings: Bell-pair every system qubit `q` in
/// `0..n` with its reference `q + n`, allocate one random branch bit, then apply `build_gadget`.
fn choi_simulation(
    system_qubit_count: usize,
    build_gadget: impl FnOnce(&mut PhasedOutcomeCompleteSimulation, usize),
) -> PhasedOutcomeCompleteSimulation {
    let mut simulation = PhasedOutcomeCompleteSimulation::new(2 * system_qubit_count);
    for system_qubit in 0..system_qubit_count {
        simulation.unitary_op(
            UnitaryOp::PrepareBell,
            &[system_qubit, system_qubit + system_qubit_count],
        );
    }
    let branch = simulation.allocate_symbolic_angle();
    build_gadget(&mut simulation, branch);
    simulation
}

#[test]
fn simulator_native_action_matches_circuit_action() {
    let (zz, zz_input, zz_output) = zz_rotation();
    let circuit_action = phased_action_of(&zz, &zz_input, &zz_output).expect("circuit action");

    let simulation = choi_simulation(2, |simulation, branch| {
        simulation.symbolic_pauli_exp(&sparse(&[z(0), z(1)]), branch);
    });
    let simulation_action = phased_action_from_simulation(&simulation, &[0, 1], &[0, 1]).expect("simulation action");

    simulation_action
        .is_equivalent(&circuit_action)
        .expect("simulator-native action must match the circuit action");
}

#[test]
fn simulator_native_distinguishes_opposite_signs() {
    let positive = choi_simulation(1, |simulation, branch| {
        simulation.symbolic_pauli_exp(&sparse(&[z(0)]), branch);
    });
    let negative = choi_simulation(1, |simulation, branch| {
        simulation.symbolic_pauli_exp(&-sparse(&[z(0)]), branch);
    });

    let positive_action = phased_action_from_simulation(&positive, &[0], &[0]).expect("positive action");
    let negative_action = phased_action_from_simulation(&negative, &[0], &[0]).expect("negative action");

    positive_action
        .is_equivalent_up_to_signs(&negative_action)
        .expect("phaseless actions must be identical");
    let reasons = positive_action
        .is_equivalent(&negative_action)
        .expect_err("opposite signs differ only in relative phase");
    assert_eq!(reasons, vec![ActionsInequivalenceReason::RelativePhase]);
}

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
fn z_product(local_indices: &[usize], support: &[QubitId]) -> SparsePauli {
    let positioned: Vec<PositionedPauliObservable> = local_indices.iter().map(|&index| z(support[index])).collect();
    (&positioned[..]).into()
}

/// Applies the symbolic Z-rotations indexed by `angle_supports` (each a tensor product of `Z`s, with
/// its own symbolic angle) to the qubits named by `support`, in allocation order.
fn apply_symbolic_z_rotations(builder: &mut CircuitBuilder, angle_supports: &[Vec<usize>], support: &[QubitId]) {
    for local_indices in angle_supports {
        let angle = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&z_product(local_indices, support), angle);
    }
}

/// The direct circuit: the symbolic Z-rotations applied straight to the `n` system qubits.
fn direct_z_channel(n: usize, angle_supports: &[Vec<usize>]) -> Circuit {
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
fn z_ejection_channel(n: usize, angle_supports: &[Vec<usize>]) -> Circuit {
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
fn check_z_ejection(n: usize, angle_supports: &[Vec<usize>]) {
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
fn single_qubit_z_rotation_ejection() {
    check_z_ejection(1, &[vec![0]]);
}

#[test]
fn two_qubit_z_rotation_ejections() {
    check_z_ejection(2, &[vec![0]]);
    check_z_ejection(2, &[vec![1]]);
    check_z_ejection(2, &[vec![0, 1]]);
    check_z_ejection(2, &[vec![0], vec![1], vec![0, 1]]);
}

#[test]
fn three_qubit_all_z_products_ejection() {
    let all_nontrivial: Vec<Vec<usize>> = (1u32..8)
        .map(|mask| (0..3).filter(|bit| mask & (1 << bit) != 0).collect())
        .collect();
    check_z_ejection(3, &all_nontrivial);
}

#[test]
fn repeated_angles_ejection() {
    check_z_ejection(2, &[vec![0], vec![0], vec![0, 1], vec![0, 1]]);
}

/// A wrong correction (conditioning on the wrong measurement outcome) must be detected: the branch
/// phase then depends on a true measurement bit that the direct channel cannot reproduce.
#[test]
fn miscorrected_ejection_is_detected() {
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

fn apply_z_diagonal_clifford(builder: &mut CircuitBuilder, support: &[QubitId]) {
    for &qubit in support {
        builder.unitary_op(UnitaryOp::SqrtZ, &[qubit]);
    }
    for window in support.windows(2) {
        builder.unitary_op(UnitaryOp::ControlledZ, &[window[0], window[1]]);
    }
}

fn direct_z_clifford_channel(n: usize) -> Circuit {
    let system: Vec<QubitId> = (0..n).collect();
    build_circuit(|builder| apply_z_diagonal_clifford(builder, &system))
}

fn z_clifford_ejection_channel(n: usize) -> Circuit {
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
fn z_diagonal_clifford_ejection_without_angles() {
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
fn x_product(qubits: &[usize], support: &[QubitId]) -> SparsePauli {
    let positioned: Vec<PositionedPauliObservable> = qubits.iter().map(|&qubit| x(support[qubit])).collect();
    (&positioned[..]).into()
}

/// Applies the symbolic X-rotations indexed by `angle_supports` (each a tensor product of `X`s, with
/// its own symbolic angle) to the qubits named by `support`, in allocation order.
fn apply_symbolic_x_rotations(builder: &mut CircuitBuilder, angle_supports: &[Vec<usize>], support: &[QubitId]) {
    for qubits in angle_supports {
        let angle = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&x_product(qubits, support), angle);
    }
}

/// The direct circuit: the symbolic X-rotations applied straight to the `n` system qubits.
fn direct_x_channel(n: usize, angle_supports: &[Vec<usize>]) -> Circuit {
    let system: Vec<QubitId> = (0..n).collect();
    build_circuit(|builder| {
        apply_symbolic_x_rotations(builder, angle_supports, &system);
    })
}

/// The ejection circuit: the same symbolic X-rotations executed remotely on `n` ancillas.
fn x_ejection_channel(n: usize, angle_supports: &[Vec<usize>]) -> Circuit {
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

fn check_x_ejection(n: usize, angle_supports: &[Vec<usize>]) {
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
fn single_qubit_x_rotation_ejection() {
    check_x_ejection(1, &[vec![0]]);
}

#[test]
fn two_qubit_x_rotation_ejections() {
    check_x_ejection(2, &[vec![0]]);
    check_x_ejection(2, &[vec![1]]);
    check_x_ejection(2, &[vec![0, 1]]);
    check_x_ejection(2, &[vec![0], vec![1], vec![0, 1]]);
}

#[test]
fn three_qubit_all_x_products_ejection() {
    let all_nontrivial: Vec<Vec<usize>> = (1u32..8)
        .map(|mask| (0..3).filter(|bit| mask & (1 << bit) != 0).collect())
        .collect();
    check_x_ejection(3, &all_nontrivial);
}

#[test]
fn repeated_x_angles_ejection() {
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
fn apply_z_diagonal_channel(
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

fn direct_z_channel_with_measurements(
    n: usize,
    angle_supports: &[Vec<usize>],
    measure_supports: &[Vec<usize>],
) -> Circuit {
    let system: Vec<QubitId> = (0..n).collect();
    build_circuit(|builder| {
        apply_z_diagonal_channel(builder, angle_supports, measure_supports, &system);
    })
}

fn z_ejection_channel_with_measurements(
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

fn check_z_ejection_with_measurements(n: usize, angle_supports: &[Vec<usize>], measure_supports: &[Vec<usize>]) {
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
fn single_qubit_z_channel_ejection() {
    check_z_ejection_with_measurements(1, &[vec![0]], &[vec![0]]);
    check_z_ejection_with_measurements(1, &[], &[vec![0]]);
}

#[test]
fn two_qubit_z_channel_ejections() {
    check_z_ejection_with_measurements(2, &[vec![0]], &[vec![1]]);
    check_z_ejection_with_measurements(2, &[vec![0], vec![1]], &[vec![0, 1]]);
    check_z_ejection_with_measurements(2, &[vec![0, 1]], &[vec![0], vec![1]]);
    check_z_ejection_with_measurements(2, &[], &[vec![0], vec![1], vec![0, 1]]);
}

#[test]
fn three_qubit_z_channel_ejection() {
    check_z_ejection_with_measurements(3, &[vec![0, 1, 2]], &[vec![0], vec![1, 2]]);
}

// ================================================================================================
// X-basis dual of the general-channel ejection above: symbolic X-rotations mixed with
// non-destructive X-basis measurements, ejected through `|+⟩` ancillas with reversed CNOTs,
// destructive Z-readout, and conditional X corrections.
// ================================================================================================

/// Applies an X-diagonal *channel* to `support`: the symbolic X-rotations indexed by
/// `angle_supports`, then non-destructive X-basis measurements indexed by `measure_supports`.
fn apply_x_diagonal_channel(
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

fn direct_x_channel_with_measurements(
    n: usize,
    angle_supports: &[Vec<usize>],
    measure_supports: &[Vec<usize>],
) -> Circuit {
    let system: Vec<QubitId> = (0..n).collect();
    build_circuit(|builder| {
        apply_x_diagonal_channel(builder, angle_supports, measure_supports, &system);
    })
}

fn x_ejection_channel_with_measurements(
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

fn check_x_ejection_with_measurements(n: usize, angle_supports: &[Vec<usize>], measure_supports: &[Vec<usize>]) {
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
fn single_qubit_x_channel_ejection() {
    check_x_ejection_with_measurements(1, &[vec![0]], &[vec![0]]);
    check_x_ejection_with_measurements(1, &[], &[vec![0]]);
}

#[test]
fn two_qubit_x_channel_ejections() {
    check_x_ejection_with_measurements(2, &[vec![0]], &[vec![1]]);
    check_x_ejection_with_measurements(2, &[vec![0], vec![1]], &[vec![0, 1]]);
    check_x_ejection_with_measurements(2, &[vec![0, 1]], &[vec![0], vec![1]]);
    check_x_ejection_with_measurements(2, &[], &[vec![0], vec![1], vec![0, 1]]);
}

#[test]
fn three_qubit_x_channel_ejection() {
    check_x_ejection_with_measurements(3, &[vec![0, 1, 2]], &[vec![0], vec![1, 2]]);
}

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
fn prepared_state_action(qubit_count: usize, build: impl FnOnce(&mut CircuitBuilder)) -> PhasedCircuitAction {
    let outputs: Vec<QubitId> = (0..qubit_count).collect();
    let circuit = build_circuit(build);
    phased_action_of(&circuit, &[], &outputs).expect("state preparation action")
}

/// Prepares `|+…+>` by Hadamarding every qubit in `0..qubit_count`.
fn prepare_plus(builder: &mut CircuitBuilder, qubit_count: usize) {
    for qubit in 0..qubit_count {
        builder.unitary_op(UnitaryOp::Hadamard, &[qubit]);
    }
}

/// Two different Clifford factorizations `C₁ Z^a C₂` of the same parameterized state must verify as
/// equivalent: `exp(iα Z₀Z₁)|++>` realized directly, versus the CNOT-conjugated single-qubit rotation
/// `CNOT₀₁ exp(iα Z₁) CNOT₀₁ |++>` (using `CNOT₀₁ Z₁ CNOT₀₁ = Z₀Z₁` and `CNOT₀₁|++> = |++>`).
#[test]
fn verifies_equal_state_preparation_factorizations() {
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
fn detects_phase_only_state_preparation_difference() {
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
fn verifies_multi_angle_state_preparation() {
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

// ================================================================================================
// Negative and randomized tests: a phased equivalence is exact, so any sign flip on a symbolic Pauli
// exponent or any permutation of the symbolic angles on the right-hand side must be detected. The
// phaseless action is unchanged (the symplectic content is identical), so the failure is a pure
// `RelativePhase` (sign flips) or a support/phase mismatch (permutations). Random instances exercise
// the same invariants over many angle supports, qubit counts, and qubit assignments.
// ================================================================================================

/// Direct Z-channel, but the `k`-th symbolic Z-rotation is `exp(±iα Z…)` according to `signs[k]`.
fn signed_z_channel(n: usize, angle_supports: &[Vec<usize>], signs: &[bool]) -> Circuit {
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
fn permuted_z_channel(n: usize, angle_supports: &[Vec<usize>], perm: &[usize]) -> Circuit {
    let system: Vec<QubitId> = (0..n).collect();
    build_circuit(|builder| {
        for &target in perm {
            let angle = builder.allocate_symbolic_angle();
            builder.symbolic_pauli_exp(&z_product(&angle_supports[target], &system), angle);
        }
    })
}

/// All non-trivial Z products on `n` qubits, in ascending-mask order: distinct, independent supports.
fn distinct_z_supports(n: usize) -> Vec<Vec<usize>> {
    (1u32..(1 << n))
        .map(|mask| (0..n).filter(|bit| mask & (1 << bit) != 0).collect())
        .collect()
}

#[test]
fn flipping_any_sign_yields_relative_phase() {
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
fn permuting_distinct_angles_is_detected() {
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

prop_compose! {
    fn arbitrary_signed_z_channel(qubit_range: Range<usize>)
        (n in qubit_range)(signs in proptest::collection::vec(any::<bool>(), 1..(1usize << n)), n in Just(n))
        -> (usize, Vec<bool>)
    { (n, signs) }
}

proptest! {
    #[test]
    fn random_sign_flip_is_pure_relative_phase((n, signs) in arbitrary_signed_z_channel(2..4usize)) {
        let supports = distinct_z_supports(n);
        let mut signs: Vec<bool> = signs.into_iter().take(supports.len()).collect();
        signs.resize(supports.len(), false);
        let system: Vec<QubitId> = (0..n).collect();
        let baseline = signed_z_channel(n, &supports, &vec![false; supports.len()]);
        let flipped = signed_z_channel(n, &supports, &signs);
        let a = phased_action_of(&baseline, &system, &system).expect("baseline");
        let b = phased_action_of(&flipped, &system, &system).expect("flipped");
        a.is_equivalent_up_to_signs(&b).expect("phaseless equal");
        match a.is_equivalent(&b) {
            Ok(()) => prop_assert!(signs.iter().all(|s| !s), "only the all-positive mask is equivalent"),
            Err(reasons) => prop_assert_eq!(reasons, vec![ActionsInequivalenceReason::RelativePhase]),
        }
    }
}

prop_compose! {
    fn arbitrary_permutation(qubit_range: Range<usize>)(n in qubit_range, seed in any::<u64>()) -> (usize, Vec<usize>) {
        use rand::seq::SliceRandom;
        let mut perm: Vec<usize> = (0..((1usize << n) - 1)).collect();
        perm.shuffle(&mut rand::rngs::StdRng::seed_from_u64(seed));
        (n, perm)
    }
}

proptest! {
    #[test]
    fn random_angle_permutation_matches_iff_identity((n, perm) in arbitrary_permutation(2..4usize)) {
        let supports = distinct_z_supports(n);
        let system: Vec<QubitId> = (0..n).collect();
        let identity: Vec<usize> = (0..supports.len()).collect();
        let base = phased_action_of(&permuted_z_channel(n, &supports, &identity), &system, &system).expect("base");
        let permuted = phased_action_of(&permuted_z_channel(n, &supports, &perm), &system, &system).expect("perm");
        if perm == identity {
            prop_assert!(base.is_equivalent(&permuted).is_ok());
        } else {
            prop_assert!(base.is_equivalent(&permuted).is_err(), "permutation {perm:?} maps distinct angles to wrong Paulis");
        }
    }

    #[test]
    fn random_multi_angle_channel_self_equivalent((n, signs) in arbitrary_signed_z_channel(2..4usize)) {
        let supports = distinct_z_supports(n);
        let mut signs: Vec<bool> = signs.into_iter().take(supports.len()).collect();
        signs.resize(supports.len(), false);
        let system: Vec<QubitId> = (0..n).collect();
        let a = phased_action_of(&signed_z_channel(n, &supports, &signs), &system, &system).expect("a");
        let b = phased_action_of(&signed_z_channel(n, &supports, &signs), &system, &system).expect("b");
        prop_assert!(a.is_equivalent(&b).is_ok(), "a channel must be exactly equivalent to itself");
    }
}

/// `Y₀ = i·X₀Z₀`, so applying `Y` differs from applying `X` then `Z` only by the global phase `i`.
/// Both are equivalent up to a global phase, but their absolute encoder phases differ.
#[test]
fn pauli_y_equals_x_then_z_up_to_global_phase() {
    let direct = build_circuit(|builder| builder.pauli(&sparse(&[y(0)])));
    let factored = build_circuit(|builder| {
        builder.pauli(&sparse(&[x(0)]));
        builder.pauli(&sparse(&[z(0)]));
    });

    let direct_action = phased_action_of(&direct, &[0], &[0]).expect("direct action");
    let factored_action = phased_action_of(&factored, &[0], &[0]).expect("factored action");

    direct_action
        .is_equivalent(&factored_action)
        .expect("Y and XZ agree up to a global phase");

    let reasons = direct_action
        .is_equivalent_with_global_phase(&factored_action)
        .expect_err("Y and XZ differ by the global phase i");
    assert_eq!(reasons, vec![ActionsInequivalenceReason::GlobalPhase]);
}

#[test]
fn identical_circuit_matches_global_phase() {
    let circuit = build_circuit(|builder| builder.pauli(&sparse(&[y(0)])));
    let lhs = phased_action_of(&circuit, &[0], &[0]).expect("lhs action");
    let rhs = phased_action_of(&circuit, &[0], &[0]).expect("rhs action");

    lhs.is_equivalent_with_global_phase(&rhs)
        .expect("a circuit must equal itself including the global phase");
}
