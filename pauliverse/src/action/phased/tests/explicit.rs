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

mod shifted_phase_regressions {
    use super::super::*;
    use binar::{AffineMap, BitMatrix, BitVec};
    use dense_oracle::{C, Dense, gate_matrix};
    use paulimer::core::y;

    const ANGLES: [f64; 5] = [0.0, 0.3, std::f64::consts::FRAC_PI_4, std::f64::consts::FRAC_PI_2, -0.6];

    fn measured_rotation(final_pauli: Option<UnitaryOp>) -> PhasedCircuitAction {
        prepared_measurement(&[], UnitaryOp::X, UnitaryOp::Z, final_pauli)
    }

    fn pauli(operation: UnitaryOp) -> SparsePauli {
        match operation {
            UnitaryOp::X => sparse(&[x(0)]),
            UnitaryOp::Y => sparse(&[y(0)]),
            UnitaryOp::Z => sparse(&[z(0)]),
            _ => panic!("expected a Pauli"),
        }
    }

    fn prepared_measurement(
        preparation: &[UnitaryOp],
        measurement: UnitaryOp,
        rotation: UnitaryOp,
        final_pauli: Option<UnitaryOp>,
    ) -> PhasedCircuitAction {
        let circuit = build_circuit(|builder| {
            for &operation in preparation {
                builder.unitary_op(operation, &[0]);
            }
            builder.measure(&pauli(measurement));
            let angle = builder.allocate_symbolic_angle();
            builder.symbolic_pauli_exp(&pauli(rotation), angle);
            if let Some(pauli) = final_pauli {
                builder.unitary_op(pauli, &[0]);
            }
        });
        phased_action_of(&circuit, &[], &[0]).unwrap()
    }

    fn flipped_record() -> AffineMap {
        let shift: BitVec = [true, false].into_iter().collect();
        AffineMap::affine(BitMatrix::identity(2), shift)
    }

    #[test]
    fn measured_rotation_distinguishes_final_y_and_z() {
        let first = measured_rotation(Some(UnitaryOp::Y));
        let second = measured_rotation(Some(UnitaryOp::Z));
        assert!(first.is_equivalent(&second).is_err());
        assert!(second.is_equivalent(&first).is_err());
    }

    #[test]
    fn measured_rotation_distinguishes_identity_and_final_x() {
        let first = measured_rotation(None);
        let second = measured_rotation(Some(UnitaryOp::X));
        assert!(first.is_equivalent(&second).is_err());
        assert!(second.is_equivalent(&first).is_err());
    }

    #[test]
    fn shifted_record_rejects_final_y() {
        let first = measured_rotation(None);
        let second = measured_rotation(Some(UnitaryOp::Y));
        let map = flipped_record();
        assert!(first.is_equivalent_with_map(&second, Some(&map)).is_err());
        assert!(second.is_equivalent_with_map(&first, Some(&map)).is_err());
    }

    #[test]
    fn shifted_record_accepts_final_z() {
        let first = measured_rotation(None);
        let second = measured_rotation(Some(UnitaryOp::Z));
        let map = flipped_record();
        first.is_equivalent_with_map(&second, Some(&map)).unwrap();
        second.is_equivalent_with_map(&first, Some(&map)).unwrap();
    }

    #[test]
    fn encoder_quadratic_phase_preserves_conjugated_program() {
        let build = |conjugated| {
            let circuit = build_circuit(|builder| {
                let angle = builder.allocate_symbolic_angle();
                builder.unitary_op(UnitaryOp::SqrtZ, &[0]);
                builder.unitary_op(UnitaryOp::ControlledX, &[0, 1]);
                builder.unitary_op(UnitaryOp::Hadamard, &[0]);
                builder.unitary_op(UnitaryOp::ControlledZ, &[0, 1]);
                if conjugated {
                    builder.unitary_op(UnitaryOp::Hadamard, &[1]);
                    builder.unitary_op(UnitaryOp::Hadamard, &[1]);
                    builder.unitary_op(UnitaryOp::SqrtXInv, &[0]);
                    builder.unitary_op(UnitaryOp::SqrtXInv, &[1]);
                    builder.unitary_op(UnitaryOp::SqrtZ, &[0]);
                    builder.unitary_op(UnitaryOp::ControlledX, &[0, 1]);
                    builder.unitary_op(UnitaryOp::Hadamard, &[1]);
                    builder.unitary_op(UnitaryOp::ControlledZ, &[0, 1]);
                }
                let first_measurement = if conjugated {
                    -sparse(&[z(0), y(1)])
                } else {
                    -sparse(&[y(0), z(1)])
                };
                builder.measure(&first_measurement);
                let rotation = if conjugated {
                    -sparse(&[x(0), x(1)])
                } else {
                    sparse(&[z(0), y(1)])
                };
                builder.symbolic_pauli_exp(&rotation, angle);
                let second_measurement = if conjugated { -sparse(&[x(0)]) } else { sparse(&[z(0)]) };
                builder.measure(&second_measurement);
                if conjugated {
                    builder.unitary_op(UnitaryOp::ControlledZ, &[0, 1]);
                    builder.unitary_op(UnitaryOp::Hadamard, &[1]);
                    builder.unitary_op(UnitaryOp::ControlledX, &[0, 1]);
                    builder.unitary_op(UnitaryOp::SqrtZInv, &[0]);
                    builder.unitary_op(UnitaryOp::SqrtX, &[1]);
                    builder.unitary_op(UnitaryOp::SqrtX, &[0]);
                    builder.unitary_op(UnitaryOp::Hadamard, &[1]);
                    builder.unitary_op(UnitaryOp::Hadamard, &[1]);
                }
            });
            phased_action_of(&circuit, &[], &[1]).unwrap()
        };
        let direct = build(false);
        let conjugated = build(true);
        let shift: BitVec = [false, false, true].into_iter().collect();
        let map = AffineMap::affine(BitMatrix::identity(3), shift);
        direct
            .is_equivalent_with_map(&conjugated, Some(&map))
            .expect("flipping the final record must preserve the encoder quadratic phase");
        conjugated
            .is_equivalent_with_map(&direct, Some(&map))
            .expect("the reverse comparison must preserve the encoder quadratic phase");
    }

    #[test]
    fn custom_map_covers_angle_and_random_signs_separately() {
        let build = |randomize| {
            let circuit = build_circuit(|builder| {
                let angle = builder.allocate_symbolic_angle();
                builder.symbolic_pauli_exp(&sparse(&[x(0)]), angle);
                if randomize {
                    let coin = builder.allocate_random_bit();
                    builder.conditional_pauli(&sparse(&[x(0)]), &[coin], true);
                }
            });
            phased_action_of(&circuit, &[], &[0]).unwrap()
        };
        let source = build(true);
        let target = build(false);
        let mut matrix = BitMatrix::zeros(2, 1);
        matrix.set((0, 0), true);
        assert_eq!(
            source.is_equivalent_with_map(&target, Some(&AffineMap::linear(matrix))),
            Err(vec![ActionsInequivalenceReason::RandomOutcomeCoverage]),
        );
    }

    #[test]
    fn shifted_map_respects_deterministic_records() {
        let build = |negative| {
            let circuit = build_circuit(|builder| {
                let observable = if negative { -sparse(&[z(0)]) } else { sparse(&[z(0)]) };
                builder.measure(&observable);
                let angle = builder.allocate_symbolic_angle();
                builder.symbolic_pauli_exp(&sparse(&[x(0)]), angle);
            });
            phased_action_of(&circuit, &[], &[0]).unwrap()
        };
        let positive = build(false);
        let negative = build(true);
        let map = flipped_record();
        assert_eq!(
            positive.is_equivalent_with_map(&positive, Some(&map)),
            Err(vec![ActionsInequivalenceReason::RandomOutcomeCoverage]),
        );
        positive.is_equivalent_with_map(&negative, Some(&map)).unwrap();
        negative.is_equivalent_with_map(&positive, Some(&map)).unwrap();
    }

    fn pauli_applied(state: &Dense, operation: UnitaryOp) -> Vec<C> {
        state.pauli_applied(
            &[matches!(operation, UnitaryOp::X | UnitaryOp::Y)],
            &[matches!(operation, UnitaryOp::Y | UnitaryOp::Z)],
            i64::from(operation == UnitaryOp::Y),
        )
    }

    fn dense_branch(
        preparation: &[UnitaryOp],
        measurement: UnitaryOp,
        rotation: UnitaryOp,
        final_pauli: Option<UnitaryOp>,
        outcome: bool,
        angle: f64,
    ) -> Vec<C> {
        let mut state = Dense::zero(1);
        for &operation in preparation {
            state.apply1(0, gate_matrix(operation));
        }
        let measured = pauli_applied(&state, measurement);
        for (amplitude, transformed) in state.amp.iter_mut().zip(measured) {
            *amplitude = (*amplitude + if outcome { -transformed } else { transformed }) * 0.5;
        }
        let rotated = pauli_applied(&state, rotation);
        for (amplitude, transformed) in state.amp.iter_mut().zip(rotated) {
            *amplitude = *amplitude * angle.cos() + C::I * transformed * angle.sin();
        }
        if let Some(operation) = final_pauli {
            state.apply1(0, gate_matrix(operation));
        }
        state.amp
    }

    fn dense_records_match(
        preparation: &[UnitaryOp],
        measurement: UnitaryOp,
        rotation: UnitaryOp,
        source_final: Option<UnitaryOp>,
        target_final: Option<UnitaryOp>,
        flipped: bool,
    ) -> bool {
        for outcome in [false, true] {
            let mut constant_phase = None;
            for angle in ANGLES {
                let source = dense_branch(
                    preparation,
                    measurement,
                    rotation,
                    source_final,
                    outcome ^ flipped,
                    angle,
                );
                let target = dense_branch(preparation, measurement, rotation, target_final, outcome, angle);
                let source_norm: f64 = source.iter().map(C::norm_sqr).sum();
                let target_norm: f64 = target.iter().map(C::norm_sqr).sum();
                if (source_norm - target_norm).abs() > 1e-10 {
                    return false;
                }
                if target_norm < 1e-10 {
                    continue;
                }
                let phase = *constant_phase.get_or_insert_with(|| {
                    source
                        .iter()
                        .zip(&target)
                        .map(|(first, second)| first * second.conj())
                        .sum::<C>()
                        / target_norm
                });
                if (phase.norm() - 1.0).abs() > 1e-10
                    || source
                        .iter()
                        .zip(&target)
                        .any(|(first, second)| (*first - phase * second).norm() > 1e-10)
                {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn affine_measurement_maps_match_dense_branch_vectors() {
        let preparations: [&[UnitaryOp]; 6] = [
            &[],
            &[UnitaryOp::X],
            &[UnitaryOp::Hadamard],
            &[UnitaryOp::Hadamard, UnitaryOp::Z],
            &[UnitaryOp::Hadamard, UnitaryOp::SqrtZ],
            &[UnitaryOp::Hadamard, UnitaryOp::SqrtZInv],
        ];
        let axes = [UnitaryOp::X, UnitaryOp::Y, UnitaryOp::Z];
        let finals = [None, Some(UnitaryOp::X), Some(UnitaryOp::Y), Some(UnitaryOp::Z)];
        let identity = AffineMap::linear(BitMatrix::identity(2));
        let shifted = flipped_record();
        let mut mismatches = Vec::new();
        let mut count = 0;
        for preparation in preparations {
            for measurement in axes {
                for rotation in axes {
                    let actions: Vec<_> = finals
                        .iter()
                        .map(|&final_pauli| prepared_measurement(preparation, measurement, rotation, final_pauli))
                        .collect();
                    for (first, source) in actions.iter().enumerate() {
                        for (second, target) in actions.iter().enumerate() {
                            for (kind, map) in [None, Some(&identity), Some(&shifted)].iter().enumerate() {
                                let expected = dense_records_match(
                                    preparation,
                                    measurement,
                                    rotation,
                                    finals[first],
                                    finals[second],
                                    kind == 2,
                                );
                                let actual = map
                                    .map_or_else(
                                        || source.is_equivalent(target),
                                        |map| source.is_equivalent_with_map(target, Some(map)),
                                    )
                                    .is_ok();
                                count += 1;
                                if actual != expected {
                                    mismatches.push(format!(
                                        "{preparation:?}, {measurement:?}, {rotation:?}, {:?}/{:?}, map={kind}, actual={actual}, expected={expected}",
                                        finals[first], finals[second],
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(count, 2592);
        assert!(
            mismatches.is_empty(),
            "{} mismatches:\n{}",
            mismatches.len(),
            mismatches.iter().take(16).cloned().collect::<Vec<_>>().join("\n"),
        );
    }
}

mod per_outcome_regressions {
    use super::super::*;
    use binar::{AffineMap, BitMatrix};

    fn prepared_angle(coin_count: usize, active: bool) -> PhasedCircuitAction {
        let circuit = build_circuit(|builder| {
            let angle = builder.allocate_symbolic_angle();
            builder.symbolic_pauli_exp(&sparse(&[x(0)]), angle);
            let coins: Vec<_> = (0..coin_count).map(|_| builder.allocate_random_bit()).collect();
            if active {
                builder.conditional_pauli(&sparse(&[x(1)]), &coins, true);
            }
        });
        phased_action_of(&circuit, &[], &[0, 1]).unwrap()
    }

    fn angle_map(rows: usize, columns: usize) -> BitMatrix {
        let mut matrix = BitMatrix::zeros(rows, columns);
        matrix.set((0, 0), true);
        matrix
    }

    #[test]
    fn custom_map_requires_active_random_coverage() {
        let source = prepared_angle(1, true);
        let target = prepared_angle(0, false);
        let map = AffineMap::linear(angle_map(2, 1));
        assert_eq!(
            source.is_equivalent_with_map(&target, Some(&map)),
            Err(vec![ActionsInequivalenceReason::RandomOutcomeCoverage])
        );
    }

    #[test]
    fn custom_map_allows_unused_random_coverage() {
        let source = prepared_angle(1, false);
        let target = prepared_angle(0, false);
        let map = AffineMap::linear(angle_map(2, 1));
        source.is_equivalent_with_map(&target, Some(&map)).unwrap();
    }

    #[test]
    fn custom_map_allows_redundant_random_parity() {
        let source = prepared_angle(2, true);
        let target = prepared_angle(1, true);
        let mut matrix = angle_map(3, 2);
        matrix.set((1, 1), true);
        source
            .is_equivalent_with_map(&target, Some(&AffineMap::linear(matrix)))
            .unwrap();
    }

    #[test]
    fn custom_map_rejects_uncovered_mixed_phase() {
        let source = build_circuit(|builder| {
            let coin = builder.allocate_random_bit();
            builder.conditional_pauli(&sparse(&[x(0)]), &[coin], true);
            let angle = builder.allocate_symbolic_angle();
            builder.symbolic_pauli_exp(&sparse(&[z(0)]), angle);
            builder.conditional_pauli(&sparse(&[x(0)]), &[coin], true);
        });
        let target = build_circuit(|builder| {
            let angle = builder.allocate_symbolic_angle();
            builder.symbolic_pauli_exp(&sparse(&[z(0)]), angle);
        });
        let source = phased_action_of(&source, &[0], &[0]).unwrap();
        let target = phased_action_of(&target, &[0], &[0]).unwrap();
        let mut matrix = BitMatrix::zeros(2, 1);
        matrix.set((1, 0), true);
        assert_eq!(
            source.is_equivalent_with_map(&target, Some(&AffineMap::linear(matrix))),
            Err(vec![ActionsInequivalenceReason::RandomOutcomeCoverage])
        );
    }

    #[test]
    fn custom_map_allows_record_constant_phase() {
        let source = build_circuit(|builder| {
            let angle = builder.allocate_symbolic_angle();
            builder.symbolic_pauli_exp(&sparse(&[x(0)]), angle);
            let coin = builder.allocate_random_bit();
            builder.conditional_pauli(&-sparse(&[]), &[coin], true);
        });
        let source = phased_action_of(&source, &[], &[0, 1]).unwrap();
        let target = prepared_angle(0, false);
        source
            .is_equivalent_with_map(&target, Some(&AffineMap::linear(angle_map(2, 1))))
            .unwrap();
    }

    #[test]
    fn custom_map_covers_both_coherent_fibres() {
        let build = |measure| {
            let circuit = build_circuit(|builder| {
                let angle = builder.allocate_symbolic_angle();
                builder.symbolic_pauli_exp(&sparse(&[x(0)]), angle);
                if measure {
                    builder.measure(&sparse(&[z(0)]));
                }
            });
            phased_action_of(&circuit, &[], &[0]).unwrap()
        };
        let unmeasured = build(false);
        let measured = build(true);
        let forward = AffineMap::linear(angle_map(1, 2));
        let mut reverse = angle_map(2, 1);
        reverse.set((1, 0), true);
        assert!(unmeasured.is_equivalent_with_map(&measured, Some(&forward)).is_err());
        assert!(
            measured
                .is_equivalent_with_map(&unmeasured, Some(&AffineMap::linear(reverse)))
                .is_err()
        );
        assert!(unmeasured.is_equivalent(&measured).is_err());
        assert!(measured.is_equivalent(&unmeasured).is_err());
    }

    #[test]
    fn reset_is_not_per_outcome_equivalence() {
        let build = |reset| {
            let circuit = build_circuit(|builder| {
                let angle = builder.allocate_symbolic_angle();
                let observable = if reset { sparse(&[x(1)]) } else { sparse(&[z(1)]) };
                builder.symbolic_pauli_exp(&observable, angle);
                if reset {
                    let outcome = builder.measure(&sparse(&[z(1)]));
                    builder.conditional_pauli(&sparse(&[x(1)]), &[outcome], true);
                }
            });
            phased_action_of(&circuit, &[0], &[0, 1]).unwrap()
        };
        let direct = build(false);
        let reset = build(true);
        assert!(direct.is_equivalent(&reset).is_err());
        assert!(reset.is_equivalent(&direct).is_err());
    }

    #[test]
    fn measured_angle_supports_must_match() {
        let build = |measured_qubit| {
            let circuit = build_circuit(|builder| {
                for qubit in 0..2 {
                    let angle = builder.allocate_symbolic_angle();
                    builder.symbolic_pauli_exp(&sparse(&[x(qubit)]), angle);
                }
                builder.measure(&sparse(&[z(measured_qubit)]));
            });
            phased_action_of(&circuit, &[], &[0, 1]).unwrap()
        };
        let first = build(0);
        let second = build(1);
        assert!(first.is_equivalent(&second).is_err());
        assert!(second.is_equivalent(&first).is_err());
    }

    #[test]
    fn channel_measurement_changes_per_outcome_action() {
        let build = |measure| {
            let circuit = build_circuit(|builder| {
                let angle = builder.allocate_symbolic_angle();
                builder.symbolic_pauli_exp(&sparse(&[x(1)]), angle);
                if measure {
                    builder.measure(&sparse(&[z(1)]));
                }
            });
            phased_action_of(&circuit, &[0], &[0, 1]).unwrap()
        };
        let direct = build(false);
        let measured = build(true);
        assert!(direct.is_equivalent(&measured).is_err());
        assert!(measured.is_equivalent(&direct).is_err());
    }
}
