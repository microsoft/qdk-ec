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
pub(super) fn surplus_measurement_conditioned_rotation_sign_is_phase_relevant() {
    let unconditional = build_circuit(|builder| {
        let angle = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[z(0)]), angle);
    });
    let conditional = build_circuit(|builder| {
        let measurement = builder.allocate_random_bit();
        let angle = builder.allocate_symbolic_angle();
        builder.conditional_pauli(&sparse(&[x(0)]), &[measurement], true);
        builder.symbolic_pauli_exp(&sparse(&[z(0)]), angle);
        builder.conditional_pauli(&sparse(&[x(0)]), &[measurement], true);
    });

    let unconditional_action = phased_action_of(&unconditional, &[0], &[0]).expect("unconditional action");
    let conditional_action = phased_action_of(&conditional, &[0], &[0]).expect("conditional action");

    let reasons = unconditional_action
        .is_equivalent(&conditional_action)
        .expect_err("a surplus measurement-conditioned rotation sign must be phase relevant");
    assert_eq!(reasons, vec![ActionsInequivalenceReason::RelativePhase]);
    let reverse_reasons = conditional_action
        .is_equivalent(&unconditional_action)
        .expect_err("surplus mixed measurement-angle phase detection must be symmetric");
    assert_eq!(reverse_reasons, vec![ActionsInequivalenceReason::RelativePhase]);
}

#[test]
pub(super) fn surplus_randomness_is_checked_symmetrically() {
    let pure = randomizes_prepared_output(false);
    let randomized = randomizes_prepared_output(true);
    assert!(pure.is_equivalent(&randomized).is_err());
    assert!(randomized.is_equivalent(&pure).is_err());
    let harmless = phased_action_of(
        &build_circuit(|builder| {
            builder.allocate_random_bit();
        }),
        &[],
        &[0],
    )
    .expect("harmless random action");

    pure.is_equivalent(&harmless)
        .expect("unused surplus randomness must not change the action");
    harmless
        .is_equivalent(&pure)
        .expect("harmless surplus randomness equivalence must be symmetric");
}

#[test]
pub(super) fn true_random_coordinates_must_follow_physical_records() {
    let source = build_circuit(|builder| {
        let first = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[z(0)]), first);
        let second = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[z(0)]), second);
        builder.allocate_random_bit();
    });
    let target = build_circuit(|builder| {
        let first = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[z(0)]), first);
        builder.allocate_random_bit();
        let second = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[z(0)]), second);
    });
    let source_action = phased_action_of(&source, &[0], &[0]).expect("source action");
    let target_action = phased_action_of(&target, &[0], &[0]).expect("target action");

    let mut matrix = BitMatrix::zeros(3, 3);
    matrix.set((0, 0), true);
    matrix.set((1, 2), true);
    matrix.set((2, 1), true);
    let outcome_map = AffineMap::linear(matrix.clone());

    source_action
        .is_equivalent_with_map(&target_action, Some(&outcome_map))
        .expect("the map must preserve the physical coin record");

    matrix.set((2, 0), true);
    let invalid_map = AffineMap::linear(matrix);
    assert_eq!(
        source_action.is_equivalent_with_map(&target_action, Some(&invalid_map)),
        Err(vec![ActionsInequivalenceReason::RelativePhase])
    );

    let measured = build_circuit(|builder| {
        let angle = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[x(0)]), angle);
        let _ = builder.measure(&sparse(&[z(0)]));
        builder.allocate_random_bit();
    });
    let measured_action = phased_action_of(&measured, &[], &[0]).expect("measured action");
    let mut physical_map = BitMatrix::identity(3);
    physical_map.set((2, 1), true);
    measured_action
        .is_equivalent_with_map(&measured_action, Some(&AffineMap::linear(physical_map)))
        .expect("a recorded measurement can relabel an unused physical coin");
}

#[test]
fn repeated_measurement_sign_is_an_outcome_relabeling() {
    let action = |negative| {
        let circuit = build_circuit(|builder| {
            builder.measure(&sparse(&[z(0)]));
            let observable = if negative { -sparse(&[z(0)]) } else { sparse(&[z(0)]) };
            builder.measure(&observable);
        });
        phased_action_of(&circuit, &[0], &[0]).expect("repeated measurement")
    };
    let positive = action(false);
    let negative = action(true);
    positive.is_equivalent(&negative).expect("outcome labels may differ");
    negative.is_equivalent(&positive).expect("relabeling is reversible");
}

#[test]
fn corrected_measurements_are_a_uniform_outcome_refinement() {
    let action = |first| {
        let circuit = build_circuit(|builder| {
            builder.measure(&sparse(&[z(0)]));
            let angle = builder.allocate_symbolic_angle();
            builder.symbolic_pauli_exp(&sparse(&[z(0)]), angle);
            builder.measure(&sparse(&[first]));
            let last = builder.measure(&sparse(&[x(0)]));
            builder.conditional_pauli(&sparse(&[z(0)]), &[last], false);
        });
        phased_action_of(&circuit, &[0], &[0]).expect("corrected measurements")
    };
    let independent = action(paulimer::core::y(0));
    let repeated = action(x(0));
    independent
        .is_equivalent(&repeated)
        .expect("the extra random record only refines the same operation");
    repeated
        .is_equivalent(&independent)
        .expect("compression preserves the summed outcome probabilities");
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

/// `Y₀ = i·X₀Z₀`, so applying `Y` differs from applying `X` then `Z` only by the global phase `i`.
/// Both are equivalent up to a global phase, but their absolute encoder phases differ.
#[test]
pub(super) fn pauli_y_equals_x_then_z_up_to_global_phase() {
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
pub(super) fn identical_circuit_matches_global_phase() {
    let circuit = build_circuit(|builder| builder.pauli(&sparse(&[y(0)])));
    let lhs = phased_action_of(&circuit, &[0], &[0]).expect("lhs action");
    let rhs = phased_action_of(&circuit, &[0], &[0]).expect("rhs action");

    lhs.is_equivalent_with_global_phase(&rhs)
        .expect("a circuit must equal itself including the global phase");
}

#[test]
pub(super) fn global_phase_check_preserves_unrelated_inequivalence_reasons() {
    let direct = build_circuit(|builder| builder.pauli(&sparse(&[y(0)])));
    let factored = build_circuit(|builder| {
        builder.pauli(&sparse(&[x(0)]));
        builder.pauli(&sparse(&[z(0)]));
    });

    let direct_action = phased_action_of(&direct, &[0], &[0]).expect("direct action");
    let unrelated_action = phased_action_of(&factored, &[], &[0]).expect("unrelated action");
    assert_ne!(direct_action.global_phase(), unrelated_action.global_phase());

    let base_reasons = direct_action
        .is_equivalent(&unrelated_action)
        .expect_err("actions with different input counts are unrelated");
    let exact_reasons = direct_action
        .is_equivalent_with_global_phase(&unrelated_action)
        .expect_err("unrelated actions remain inequivalent");

    assert_eq!(exact_reasons, base_reasons);
    assert!(!exact_reasons.contains(&ActionsInequivalenceReason::GlobalPhase));
}

#[test]
pub(super) fn simulator_native_rejects_entangled_auxiliary_qubits() {
    // Qubit 2 is a system qubit that no argument names. Leaving it entangled with an output must
    // be an error, not a silently truncated action.
    let mut simulation = PhasedOutcomeCompleteSimulation::new(4);
    simulation.unitary_op(UnitaryOp::PrepareBell, &[0, 3]);
    simulation.unitary_op(UnitaryOp::ControlledX, &[0, 2]);

    let result = phased_action_from_simulation(&simulation, &[0], &[0]);

    assert!(
        result.is_err(),
        "an entangled auxiliary qubit must be reported, got {result:?}"
    );
}

#[test]
pub(super) fn simulator_native_rejects_a_reused_symbolic_angle() {
    // One angle drives two rotations, so both collapse onto the same branch bit. The recorded
    // action then cannot tell exp(2 i alpha X) from exp(2 i alpha Z), and must be rejected.
    let mut simulation = PhasedOutcomeCompleteSimulation::new(2);
    simulation.unitary_op(UnitaryOp::PrepareBell, &[0, 1]);
    let angle = simulation.allocate_symbolic_angle();
    simulation.symbolic_pauli_exp(&sparse(&[x(0)]), angle);
    simulation.symbolic_pauli_exp(&sparse(&[x(0)]), angle);

    let result = phased_action_from_simulation(&simulation, &[0], &[0]);

    assert!(
        matches!(result, Err(ActionError::SymbolicAngleReused { angle: reused }) if reused == angle),
        "a reused symbolic angle must be reported, got {result:?}"
    );
}

#[test]
pub(super) fn symbolic_rotation_may_not_reach_a_discarded_qubit() {
    // Qubit 2 is discarded. A rotation that spans qubits 0 and 2 correlates the angle with the
    // discarded qubit, so the retained state loses coherence and the action is not well defined.
    fn action(observable: &[PositionedPauliObservable]) -> Result<PhasedCircuitAction, ActionError> {
        let mut simulation = PhasedOutcomeCompleteSimulation::new(3);
        simulation.unitary_op(UnitaryOp::PrepareBell, &[0, 1]);
        let angle = simulation.allocate_symbolic_angle();
        simulation.symbolic_pauli_exp(&sparse(observable), angle);
        phased_action_from_simulation(&simulation, &[0], &[0])
    }

    action(&[x(0)]).expect("a rotation on the retained qubit alone has an action");

    let spanning = action(&[x(0), x(2)]);
    assert!(
        matches!(spanning, Err(ActionError::AuxiliaryQubitsCarrySymbolicAngle { angle }) if angle == 0),
        "a rotation reaching a discarded qubit must be reported, got {spanning:?}"
    );
}

/// Two rotations on one discarded qubit, measured together, reveal only their combined parity.
/// That parity still fixes the auxiliary state, so the action exists. A rule demanding that each
/// angle be revealed on its own would refuse this.
#[test]
pub(super) fn a_revealed_parity_of_two_angles_keeps_the_action() {
    let circuit = build_circuit(|builder| {
        let first = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[x(0)]), first);
        let second = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[x(0)]), second);
        let _ = builder.measure(&sparse(&[z(0)]));
    });

    phased_action_of(&circuit, &[], &[]).expect("the measured parity fixes the discarded qubit");
}

/// The angle counter is indexed by outcome id. A deterministic measurement takes an outcome id
/// without taking a random bit, so a counter indexed by random bit would miss the reuse.
#[test]
pub(super) fn angle_reuse_is_caught_after_a_deterministic_measurement() {
    let mut simulation = PhasedOutcomeCompleteSimulation::new(2);
    simulation.unitary_op(UnitaryOp::PrepareBell, &[0, 1]);
    let deterministic = simulation.measure(&sparse(&[z(0), z(1)]));
    assert_eq!(deterministic, 0, "the Bell parity is deterministic");

    let angle = simulation.allocate_symbolic_angle();
    simulation.symbolic_pauli_exp(&sparse(&[x(0)]), angle);
    simulation.symbolic_pauli_exp(&sparse(&[x(0)]), angle);

    let result = phased_action_from_simulation(&simulation, &[0], &[0]);

    assert!(
        matches!(result, Err(ActionError::SymbolicAngleReused { angle: reused }) if reused == angle),
        "a reused angle must be reported whatever its outcome id, got {result:?}"
    );
}

/// Conditioning a Pauli on an angle directly is another way to drive a rotation, so it counts
/// against the same angle.
#[test]
pub(super) fn angle_reuse_through_a_conditional_pauli_is_caught() {
    let mut simulation = PhasedOutcomeCompleteSimulation::new(2);
    simulation.unitary_op(UnitaryOp::PrepareBell, &[0, 1]);
    let angle = simulation.allocate_symbolic_angle();
    simulation.symbolic_pauli_exp(&sparse(&[x(0)]), angle);
    simulation.conditional_pauli(&sparse(&[z(0)]), &[angle], true);

    let result = phased_action_from_simulation(&simulation, &[0], &[0]);

    assert!(
        matches!(result, Err(ActionError::SymbolicAngleReused { angle: reused }) if reused == angle),
        "conditioning on an angle counts as a use, got {result:?}"
    );
}

/// The control of a conditional Pauli is the parity of the outcomes it names, so an angle named
/// twice drives nothing. Counting each mention would refuse a valid circuit.
#[test]
pub(super) fn an_angle_named_twice_in_one_control_is_not_a_reuse() {
    let mut simulation = PhasedOutcomeCompleteSimulation::new(2);
    simulation.unitary_op(UnitaryOp::PrepareBell, &[0, 1]);
    let angle = simulation.allocate_symbolic_angle();
    simulation.symbolic_pauli_exp(&sparse(&[x(0)]), angle);
    simulation.conditional_pauli(&sparse(&[z(0)]), &[angle, angle], true);

    phased_action_from_simulation(&simulation, &[0], &[0])
        .expect("a control that names an angle twice applies nothing");
}

/// An angle named three times leaves the same control as naming it once, so it is one use and the
/// rotation before it makes two.
#[test]
pub(super) fn an_angle_named_three_times_in_one_control_is_one_use() {
    let mut simulation = PhasedOutcomeCompleteSimulation::new(2);
    simulation.unitary_op(UnitaryOp::PrepareBell, &[0, 1]);
    let angle = simulation.allocate_symbolic_angle();
    simulation.conditional_pauli(&sparse(&[z(0)]), &[angle, angle, angle], true);

    phased_action_from_simulation(&simulation, &[0], &[0]).expect("one control is one use");

    simulation.symbolic_pauli_exp(&sparse(&[x(0)]), angle);
    let result = phased_action_from_simulation(&simulation, &[0], &[0]);

    assert!(
        matches!(result, Err(ActionError::SymbolicAngleReused { angle: reused }) if reused == angle),
        "a second use of the angle must be reported, got {result:?}"
    );
}

/// Supports that do not describe a Choi-state layout must be reported as errors. Each case once
/// panicked inside canonicalization or produced an action for a layout that does not exist.
#[test]
pub(super) fn phased_action_from_simulation_rejects_invalid_qubits() {
    let mut one_qubit = PhasedOutcomeCompleteSimulation::new(1);
    one_qubit.unitary_op(UnitaryOp::Hadamard, &[0]);
    let mut three_qubits = PhasedOutcomeCompleteSimulation::new(3);
    three_qubits.unitary_op(UnitaryOp::PrepareBell, &[0, 1]);
    three_qubits.unitary_op(UnitaryOp::PrepareBell, &[0, 2]);

    let cases: [(&str, &PhasedOutcomeCompleteSimulation, &[QubitId], &[QubitId]); 4] = [
        ("output outside the simulation", &one_qubit, &[], &[5]),
        ("reference qubit outside the simulation", &one_qubit, &[0], &[0]),
        ("repeated input", &three_qubits, &[0, 0], &[0]),
        ("repeated output", &one_qubit, &[], &[0, 0]),
    ];
    for (name, simulation, inputs, outputs) in cases {
        let result = phased_action_from_simulation(simulation, inputs, outputs);
        assert!(
            matches!(
                &result,
                Err(ActionError::InvalidQubits { input_qubits, output_qubits })
                    if input_qubits == inputs && output_qubits == outputs
            ),
            "{name}: expected InvalidQubits, got {result:?}"
        );
    }

    phased_action_from_simulation(&one_qubit, &[], &[0]).expect("a valid layout has an action");

    let circuit = build_circuit(|builder| builder.unitary_op(UnitaryOp::Hadamard, &[0]));
    let repeated_output = phased_action_of(&circuit, &[0], &[0, 0]);
    assert!(
        matches!(repeated_output, Err(ActionError::InvalidQubits { .. })),
        "a circuit with a repeated output must be reported, got {repeated_output:?}"
    );
    let repeated_input = phased_action_of(&circuit, &[0, 0], &[0]);
    assert!(
        matches!(repeated_input, Err(ActionError::InvalidQubits { .. })),
        "a circuit with a repeated input must be reported, got {repeated_input:?}"
    );
}

/// An angle that drives no rotation would encode `exp(i alpha I)`. On `|0>` that matches the
/// rotation `exp(i alpha Z)`, so an unused angle once compared equal to a real rotation.
#[test]
pub(super) fn phased_action_rejects_an_unused_symbolic_angle() {
    let rotated = {
        let mut simulation = PhasedOutcomeCompleteSimulation::new(1);
        let angle = simulation.allocate_symbolic_angle();
        simulation.symbolic_pauli_exp(&sparse(&[z(0)]), angle);
        simulation
    };
    phased_action_from_simulation(&rotated, &[], &[0]).expect("a used angle has an action");

    let mut unused = PhasedOutcomeCompleteSimulation::new(1);
    let angle = unused.allocate_symbolic_angle();
    let result = phased_action_from_simulation(&unused, &[], &[0]);
    assert!(
        matches!(result, Err(ActionError::SymbolicAngleUnused { angle: reported }) if reported == angle),
        "an unused angle must be reported, got {result:?}"
    );

    let mut after_measurement = PhasedOutcomeCompleteSimulation::new(2);
    after_measurement.unitary_op(UnitaryOp::PrepareBell, &[0, 1]);
    after_measurement.measure(&sparse(&[z(0), z(1)]));
    let used = after_measurement.allocate_symbolic_angle();
    let idle = after_measurement.allocate_symbolic_angle();
    after_measurement.symbolic_pauli_exp(&sparse(&[x(0)]), used);
    after_measurement.conditional_pauli(&sparse(&[z(0)]), &[idle, idle], true);
    let result = phased_action_from_simulation(&after_measurement, &[0], &[0]);
    assert!(
        matches!(result, Err(ActionError::SymbolicAngleUnused { angle: reported }) if reported == idle),
        "an angle named only in pairs is unused and must be reported by outcome id, got {result:?}"
    );

    let circuit = build_circuit(|builder| {
        builder.allocate_symbolic_angle();
    });
    let result = phased_action_of(&circuit, &[0], &[0]);
    assert!(
        matches!(result, Err(ActionError::SymbolicAngleUnused { .. })),
        "a circuit with an unused angle must be reported, got {result:?}"
    );
}

/// A deterministic measurement takes an outcome id without a random bit. The auxiliary error must
/// still name the angle by the outcome id that the caller received, not by its random bit.
#[test]
pub(super) fn auxiliary_angle_error_names_the_outcome_id() {
    let mut simulation = PhasedOutcomeCompleteSimulation::new(3);
    simulation.unitary_op(UnitaryOp::PrepareBell, &[0, 1]);
    let deterministic = simulation.measure(&sparse(&[z(0), z(1)]));
    assert_eq!(deterministic, 0, "the Bell parity is deterministic");
    let angle = simulation.allocate_symbolic_angle();
    assert_eq!(angle, 1, "the angle takes the next outcome id");
    simulation.symbolic_pauli_exp(&sparse(&[x(0), x(2)]), angle);

    let result = phased_action_from_simulation(&simulation, &[0], &[0]);

    assert!(
        matches!(result, Err(ActionError::AuxiliaryQubitsCarrySymbolicAngle { angle: reported }) if reported == angle),
        "the auxiliary error must name outcome id {angle}, got {result:?}"
    );
}
