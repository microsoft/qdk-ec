use super::*;
pub(super) fn build_circuit(build: impl FnOnce(&mut CircuitBuilder)) -> Circuit {
    let mut builder = CircuitBuilder::new();
    build(&mut builder);
    builder.into()
}

pub(super) fn sparse(observable: &[PositionedPauliObservable]) -> SparsePauli {
    observable.into()
}

/// `exp(iα Z₀Z₁)` represented as a symbolic rotation gadget: allocate a random branch bit, then
/// conditionally apply `Z₀Z₁` on the odd branch.
pub(super) fn zz_rotation() -> (Circuit, Vec<QubitId>, Vec<QubitId>) {
    let circuit = build_circuit(|builder| {
        let branch = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[z(0), z(1)]), branch);
    });
    (circuit, vec![0, 1], vec![0, 1])
}

/// `CNOT₀₁ · exp(iα Z₁) · CNOT₀₁`, which should equal `exp(iα Z₀Z₁)` as a channel.
pub(super) fn cnot_conjugated_z_rotation() -> (Circuit, Vec<QubitId>, Vec<QubitId>) {
    let circuit = build_circuit(|builder| {
        builder.unitary_op(UnitaryOp::ControlledX, &[0, 1]);
        let branch = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[z(1)]), branch);
        builder.unitary_op(UnitaryOp::ControlledX, &[0, 1]);
    });
    (circuit, vec![0, 1], vec![0, 1])
}

/// `exp(iα Z₁)` on its own, which differs from `exp(iα Z₀Z₁)` in symplectic action.
pub(super) fn z_rotation() -> (Circuit, Vec<QubitId>, Vec<QubitId>) {
    let circuit = build_circuit(|builder| {
        let branch = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[z(1)]), branch);
    });
    (circuit, vec![0, 1], vec![0, 1])
}

#[test]
pub(super) fn zz_rotation_equals_cnot_conjugated_z_rotation() {
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
pub(super) fn zz_rotation_differs_from_z_rotation() {
    let (zz, zz_input, zz_output) = zz_rotation();
    let (single, single_input, single_output) = z_rotation();

    let zz_action = phased_action_of(&zz, &zz_input, &zz_output).expect("zz action");
    let single_action = phased_action_of(&single, &single_input, &single_output).expect("z action");

    let reasons = zz_action
        .is_equivalent(&single_action)
        .expect_err("rotations with different supports must differ");
    assert_ne!(reasons, [], "the failure must name a reason");
}

#[test]
pub(super) fn opposite_sign_rotations_differ_only_in_relative_phase() {
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
pub(super) fn rotation_equals_itself() {
    let (zz, zz_input, zz_output) = zz_rotation();
    let action = phased_action_of(&zz, &zz_input, &zz_output).expect("action");
    action
        .is_equivalent(&action)
        .expect("a rotation must be equivalent to itself");
}

/// Conjugating a `Z` rotation by a Hadamard produces the corresponding `X` rotation:
/// `H₀ · exp(iα Z₀) · H₀ == exp(iα X₀)`, including branch phase.
#[test]
pub(super) fn hadamard_conjugated_z_rotation_equals_x_rotation() {
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
pub(super) fn mixed_basis_exponent_equals_hadamard_conjugated_zz() {
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
pub(super) fn choi_simulation(
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
pub(super) fn simulator_native_action_matches_circuit_action() {
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
pub(super) fn simulator_native_distinguishes_opposite_signs() {
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
