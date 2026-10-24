use super::*;

pub(super) fn randomizes_prepared_output(randomize: bool) -> PhasedCircuitAction {
    let circuit = build_circuit(|builder| {
        if randomize {
            let bit = builder.allocate_random_bit();
            builder.conditional_pauli(&sparse(&[x(0)]), &[bit], true);
        }
    });
    phased_action_of(&circuit, &[], &[0]).expect("prepared output action")
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

pub(super) fn x_rotation_with_z(pre: bool, post: bool) -> PhasedCircuitAction {
    prepared_state_action(1, |builder| {
        if pre {
            builder.unitary_op(UnitaryOp::Z, &[0]);
        }
        let angle = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[x(0)]), angle);
        if post {
            builder.unitary_op(UnitaryOp::Z, &[0]);
        }
    })
}

#[test]
pub(super) fn canonical_phase_includes_encoder_carried_branch_phase() {
    let baseline = x_rotation_with_z(false, false);
    let pre_z = x_rotation_with_z(true, false);
    let post_z = x_rotation_with_z(false, true);

    baseline
        .is_equivalent(&pre_z)
        .expect("a Z before the rotation fixes the prepared input state");
    pre_z
        .is_equivalent(&baseline)
        .expect("the equivalent state-preparation comparison must be symmetric");

    let reasons = baseline
        .is_equivalent(&post_z)
        .expect_err("a Z after the rotation changes its sign");
    assert_eq!(reasons, vec![ActionsInequivalenceReason::RelativePhase]);
    let reasons = post_z
        .is_equivalent(&baseline)
        .expect_err("the inequivalent state-preparation comparison must be symmetric");
    assert_eq!(reasons, vec![ActionsInequivalenceReason::RelativePhase]);
}

pub(super) fn measured_x_rotation(post_z: bool) -> PhasedCircuitAction {
    let circuit = build_circuit(|builder| {
        let angle = builder.allocate_symbolic_angle();
        builder.symbolic_pauli_exp(&sparse(&[x(0)]), angle);
        if post_z {
            builder.unitary_op(UnitaryOp::Z, &[0]);
        }
        let _ = builder.measure(&sparse(&[z(0)]));
    });
    phased_action_of(&circuit, &[], &[]).expect("measured state-preparation action")
}

#[test]
pub(super) fn physical_outcome_sectors_quotient_branch_global_phase() {
    let baseline = measured_x_rotation(false);
    let post_z = measured_x_rotation(true);

    baseline
        .is_equivalent(&post_z)
        .expect("the Z phase is global after the physical measurement reveals the angle branch");
    post_z
        .is_equivalent(&baseline)
        .expect("outcome-sector phase equivalence must be symmetric");
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

mod encoder_phase_tests {
    use super::super::dense_reference::{C, Dense, gate_matrix, zeta8};
    use super::*;
    use paulimer::{UnitaryOp, clifford::clifford_to_pauli_exponents, core::x};
    use rand::{RngExt, SeedableRng};

    fn apply_gate(state: &mut Dense, operation: UnitaryOp, support: &[usize]) {
        match operation {
            UnitaryOp::ControlledX => state.apply_cx(support[0], support[1]),
            UnitaryOp::ControlledZ => state.apply_cz(support[0], support[1]),
            _ => state.apply1(support[0], gate_matrix(operation)),
        }
    }

    fn basis_state(qubit_count: usize, label: &BitVec) -> Dense {
        let mut state = Dense::zero(qubit_count);
        for bit in label.support() {
            state.apply1(bit, gate_matrix(UnitaryOp::X));
        }
        state
    }

    fn reordered(amplitudes: &[C], order: &[usize]) -> Vec<C> {
        (0..amplitudes.len())
            .map(|index| {
                let original = order.iter().enumerate().fold(0, |bits, (position, &qubit)| {
                    bits | (((index >> (order.len() - 1 - position)) & 1) << (order.len() - 1 - qubit))
                });
                amplitudes[original]
            })
            .collect()
    }

    fn check_family(qubit_count: usize, output: &[usize], gates: &[(UnitaryOp, Vec<usize>)]) {
        let mut simulation = PhasedOutcomeCompleteSimulation::with_capacity(qubit_count, qubit_count, qubit_count);
        for bit in 0..qubit_count {
            let coin = simulation.allocate_random_bit();
            let observable: SparsePauli = [x(bit)].as_slice().into();
            simulation.conditional_pauli(&observable, &[coin], true);
        }
        for (operation, support) in gates {
            simulation.unitary_op(*operation, support);
        }
        let action = phased_action_from_simulation(&simulation, &[], output).unwrap();
        let choi = &action.action.choi_state_stabilizers;
        let auxiliary = &action.action.auxiliary_stabilizers;
        let reference = group_encoding_clifford_of(&choi.canonical_generators, output.len()).tensor(
            &group_encoding_clifford_of(&auxiliary.canonical_generators, qubit_count - output.len()),
        );
        let exponents = clifford_to_pauli_exponents(&reference);
        let order: Vec<_> = output.iter().chain(&auxiliary.canonical_to_original).copied().collect();
        let mut constant_phase = None;
        for assignment in 0..1 << qubit_count {
            let random: BitVec = (0..qubit_count).map(|bit| assignment & (1 << bit) != 0).collect();
            let mut actual = basis_state(qubit_count, &random);
            for (operation, support) in gates {
                apply_gate(&mut actual, *operation, support);
            }
            let actual = Dense {
                qubit_count,
                amp: reordered(&actual.amp, &order),
            };
            let mut label = BitVec::zeros(qubit_count);
            for bit in 0..qubit_count {
                let observable = reference.image_z(bit);
                let x_bits: Vec<_> = (0..qubit_count).map(|bit| observable.x_bits().index(bit)).collect();
                let z_bits: Vec<_> = (0..qubit_count).map(|bit| observable.z_bits().index(bit)).collect();
                let transformed = actual.pauli_applied(&x_bits, &z_bits, i64::from(observable.xz_phase_exponent() % 4));
                let expectation: C = actual
                    .amp
                    .iter()
                    .zip(transformed)
                    .map(|(first, second)| first.conj() * second)
                    .sum();
                assert!((expectation.re.abs() - 1.0).abs() < 1e-9);
                label.assign_index(bit, expectation.re < 0.0);
            }
            let mut expected = basis_state(qubit_count, &label);
            for exponent in &exponents {
                let x_bits: Vec<_> = (0..qubit_count).map(|bit| exponent.x_bits().index(bit)).collect();
                let z_bits: Vec<_> = (0..qubit_count).map(|bit| exponent.z_bits().index(bit)).collect();
                expected.apply_pauli_exp(&x_bits, &z_bits, i64::from(exponent.xz_phase_exponent() % 4));
            }
            let phase = zeta8(i64::from(action.phase.phase_exponent(&random)));
            for amplitude in &mut expected.amp {
                *amplitude *= phase;
            }
            let constant = *constant_phase.get_or_insert_with(|| {
                actual
                    .amp
                    .iter()
                    .zip(&expected.amp)
                    .map(|(first, second)| first * second.conj())
                    .sum::<C>()
            });
            assert!((constant.norm() - 1.0).abs() < 1e-9);
            for (first, second) in actual.amp.iter().zip(&expected.amp) {
                assert!(
                    (*first - constant * second).norm() < 1e-9,
                    "{qubit_count} qubits, output={output:?}, assignment={assignment}, gates={gates:?}",
                );
            }
        }
    }

    #[test]
    fn encoder_phase_matches_dense_basis_families() {
        check_family(2, &[0, 1], &[(UnitaryOp::ControlledZ, vec![0, 1])]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5a1f_2026);
        let single = [
            UnitaryOp::Hadamard,
            UnitaryOp::SqrtZ,
            UnitaryOp::SqrtZInv,
            UnitaryOp::X,
            UnitaryOp::Y,
            UnitaryOp::Z,
            UnitaryOp::SqrtX,
        ];
        for qubit_count in 1..=4 {
            for trial in 0..32 {
                let output_count = if trial % 2 == 0 { qubit_count } else { qubit_count - 1 };
                let output: Vec<_> = (0..output_count).rev().collect();
                let mut gates = Vec::new();
                for _ in 0..24 {
                    let first = rng.random_range(0..qubit_count);
                    let kind = rng.random_range(0..9);
                    if kind >= single.len() && first < output_count && output_count >= 2 {
                        let second = (first + rng.random_range(1..output_count)) % output_count;
                        let operation = if kind == 7 {
                            UnitaryOp::ControlledX
                        } else {
                            UnitaryOp::ControlledZ
                        };
                        gates.push((operation, vec![first, second]));
                    } else {
                        gates.push((single[kind % single.len()], vec![first]));
                    }
                }
                check_family(qubit_count, &output, &gates);
            }
        }
    }
}
