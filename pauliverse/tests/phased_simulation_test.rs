mod dense_reference;

#[test]
fn specific_signed_queries_reject_imaginary_paulis_on_negative_outcomes() {
    use pauliverse::{OutcomeSpecificSimulation, Simulation};
    let mut simulation = OutcomeSpecificSimulation::new_with_zero_samples(1);
    let outcome = simulation.measure(&"-Z".parse().unwrap());
    for text in ["iZ", "-iZ"] {
        let pauli = text.parse().unwrap();
        assert!(
            !simulation.is_stabilizer_with_conditional_sign(&pauli, &[outcome]),
            "{text}"
        );
    }
}

#[test]
fn signed_stabilizer_queries_reject_negative_and_imaginary_paulis() {
    use pauliverse::{PhasedOutcomeCompleteSimulation, Simulation};
    let mut simulation = PhasedOutcomeCompleteSimulation::new(1);
    let negative = simulation.measure(&"-Z".parse().unwrap());
    for (text, positive, negative) in [
        ("Z", true, false),
        ("-Z", false, true),
        ("iZ", false, false),
        ("-iZ", false, false),
    ] {
        let pauli = text.parse().unwrap();
        assert_eq!(simulation.is_stabilizer(&pauli), positive, "{text}");
        assert_eq!(
            simulation.is_stabilizer_with_conditional_sign(&pauli, &[]),
            positive,
            "{text}"
        );
        assert_eq!(
            simulation.is_stabilizer_with_conditional_sign(&pauli, &[0]),
            negative,
            "{text}"
        );
    }
    assert_eq!(negative, 0);
}

mod dense {
    //! Dense statevector validation of `PhasedOutcomeCompleteSimulation`.
    //!
    //! For random circuits (Clifford gates, Pauli exponentials, Paulis, controlled-Paulis, conditional
    //! Paulis and Pauli measurements) this test enumerates every random-bit assignment `r`, materialises
    //! the simulator's claimed output state `i^⟨p,r⟩ (-1)^⟨Br+s,r⟩ R|Ar⟩`, and compares it — *exactly,
    //! including the global phase* — against a brute-force dense statevector simulation whose
    //! measurement outcomes are forced to the branch selected by `r`.

    #![allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::too_many_lines
    )]

    use binar::vec::AlignedBitVec;
    use binar::{Bitwise, BitwiseMut};
    use paulimer::clifford::Clifford;
    use paulimer::pauli::commutes_with;
    use paulimer::{DensePauli, SparsePauli, UnitaryOp};
    use pauliverse::{PhasedOutcomeCompleteSimulation, Simulation};
    use rand::RngExt;

    use super::dense_reference::{C, Dense, close, gate_matrix, normalize, pauli_arrays, statevector, zeta8};

    #[test]
    fn conditional_pauli_matches_exact_dense_branches_for_every_phase_and_parity() {
        for prefix in ["", "-", "i", "-i"] {
            for letters in ["I", "X", "Y", "Z"] {
                let text = format!("{prefix}{letters}");
                let pauli: SparsePauli = text.parse().unwrap();
                let dense_pauli: DensePauli = text.parse().unwrap();
                let (x_bits, z_bits, phase) = pauli_arrays(&dense_pauli, 1);
                for parity in [false, true] {
                    for shift in [false, true] {
                        for duplicated in [false, true] {
                            let mut simulation = PhasedOutcomeCompleteSimulation::new(1);
                            let fixed = simulation.measure(&if shift { "-Z" } else { "Z" }.parse().unwrap());
                            let first = simulation.allocate_random_bit();
                            let second = simulation.allocate_random_bit();
                            simulation.unitary_op(UnitaryOp::Hadamard, &[0]);
                            simulation.unitary_op(UnitaryOp::SqrtZ, &[0]);
                            let mut outcomes = vec![fixed, first, second];
                            if duplicated {
                                outcomes.extend([fixed, fixed, first, first]);
                            }
                            simulation.conditional_pauli(&pauli, &outcomes, parity);
                            for first_bit in [false, true] {
                                for second_bit in [false, true] {
                                    let mut expected = Dense::zero(1);
                                    expected.apply1(0, gate_matrix(UnitaryOp::Hadamard));
                                    expected.apply1(0, gate_matrix(UnitaryOp::SqrtZ));
                                    if shift ^ first_bit ^ second_bit == parity {
                                        expected.apply_pauli(&x_bits, &z_bits, phase);
                                    }
                                    let actual = claimed_state(&simulation, &[first_bit, second_bit], 1);
                                    for (actual, expected) in actual.iter().zip(&expected.amp) {
                                        assert!(
                                            (*actual - *expected).norm() < 1e-9,
                                            "{text}, parity={parity}, shift={shift}, duplicated={duplicated}, branch={first_bit},{second_bit}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[derive(Clone)]
    enum Op {
        Gate(UnitaryOp, Vec<usize>),
        Pauli(String),
        PauliExp(String),
        ControlledPauli(String, String),
        ConditionalPauli(String, Vec<usize>, bool),
        Measure(String),
    }

    fn random_hermitian_pauli(rng: &mut impl RngExt, qubit_count: usize) -> String {
        loop {
            let mut letters = String::new();
            let mut any = false;
            for _ in 0..qubit_count {
                match rng.random_range(0..4) {
                    0 => letters.push('I'),
                    1 => {
                        letters.push('X');
                        any = true;
                    }
                    2 => {
                        letters.push('Z');
                        any = true;
                    }
                    _ => {
                        letters.push('Y');
                        any = true;
                    }
                }
            }
            if !any {
                continue;
            }
            let sign = if rng.random_range(0..2) == 0 { "-" } else { "" };
            return format!("{sign}{letters}");
        }
    }

    fn two_distinct(rng: &mut impl RngExt, qubit_count: usize) -> (usize, usize) {
        let first = rng.random_range(0..qubit_count);
        let mut second = rng.random_range(0..qubit_count);
        while second == first {
            second = rng.random_range(0..qubit_count);
        }
        (first, second)
    }

    fn random_circuit(rng: &mut impl RngExt, qubit_count: usize) -> Vec<Op> {
        let single_qubit_gates = [
            UnitaryOp::Hadamard,
            UnitaryOp::X,
            UnitaryOp::Y,
            UnitaryOp::Z,
            UnitaryOp::SqrtZ,
            UnitaryOp::SqrtZInv,
            UnitaryOp::SqrtX,
            UnitaryOp::SqrtXInv,
            UnitaryOp::SqrtY,
            UnitaryOp::SqrtYInv,
        ];
        let two_qubit_gates = [UnitaryOp::ControlledX, UnitaryOp::ControlledZ, UnitaryOp::Swap];
        let mut ops = Vec::new();
        let mut measurement_count = 0usize;
        let op_count = rng.random_range(6..14);
        for _ in 0..op_count {
            match rng.random_range(0..7) {
                0 => {
                    let qubit = rng.random_range(0..qubit_count);
                    ops.push(Op::Gate(
                        single_qubit_gates[rng.random_range(0..single_qubit_gates.len())],
                        vec![qubit],
                    ));
                }
                1 => {
                    let (first_qubit, second_qubit) = two_distinct(rng, qubit_count);
                    ops.push(Op::Gate(
                        two_qubit_gates[rng.random_range(0..two_qubit_gates.len())],
                        vec![first_qubit, second_qubit],
                    ));
                }
                2 => ops.push(Op::Pauli(random_hermitian_pauli(rng, qubit_count))),
                3 => ops.push(Op::PauliExp(random_hermitian_pauli(rng, qubit_count))),
                4 => {
                    let first_pauli_string = random_hermitian_pauli(rng, qubit_count);
                    let mut second_pauli_string = random_hermitian_pauli(rng, qubit_count);
                    let mut guard = 0;
                    loop {
                        let first_sparse_pauli: SparsePauli = first_pauli_string.parse().unwrap();
                        let second_sparse_pauli: SparsePauli = second_pauli_string.parse().unwrap();
                        if commutes_with(&first_sparse_pauli, &second_sparse_pauli) {
                            break;
                        }
                        second_pauli_string = random_hermitian_pauli(rng, qubit_count);
                        guard += 1;
                        if guard > 32 {
                            break;
                        }
                    }
                    let first_sparse_pauli: SparsePauli = first_pauli_string.parse().unwrap();
                    let second_sparse_pauli: SparsePauli = second_pauli_string.parse().unwrap();
                    if commutes_with(&first_sparse_pauli, &second_sparse_pauli) {
                        ops.push(Op::ControlledPauli(first_pauli_string, second_pauli_string));
                    }
                }
                5 => {
                    if measurement_count > 0 && rng.random_range(0..2) == 0 {
                        let mut outcomes = Vec::new();
                        for outcome_index in 0..measurement_count {
                            if rng.random_range(0..2) == 0 {
                                outcomes.push(outcome_index);
                            }
                        }
                        if !outcomes.is_empty() {
                            ops.push(Op::ConditionalPauli(
                                random_hermitian_pauli(rng, qubit_count),
                                outcomes,
                                rng.random_range(0..2) == 1,
                            ));
                        }
                    }
                }
                _ => {
                    if measurement_count < 5 {
                        ops.push(Op::Measure(random_hermitian_pauli(rng, qubit_count)));
                        measurement_count += 1;
                    }
                }
            }
        }
        ops
    }

    fn run_simulation(ops: &[Op], qubit_count: usize) -> PhasedOutcomeCompleteSimulation {
        let mut sim = PhasedOutcomeCompleteSimulation::new(qubit_count);
        for op in ops {
            match op {
                Op::Gate(gate, support) => sim.unitary_op(*gate, support),
                Op::Pauli(pauli) => sim.pauli(&pauli.parse().unwrap()),
                Op::PauliExp(pauli) => sim.pauli_exp(&pauli.parse().unwrap()),
                Op::ControlledPauli(first_pauli, second_pauli) => {
                    sim.controlled_pauli(&first_pauli.parse().unwrap(), &second_pauli.parse().unwrap());
                }
                Op::ConditionalPauli(pauli, outcomes, parity) => {
                    sim.conditional_pauli(&pauli.parse().unwrap(), outcomes, *parity);
                }
                Op::Measure(pauli) => {
                    sim.measure(&pauli.parse().unwrap());
                }
            }
        }
        sim
    }

    fn dense_reference(ops: &[Op], outcome_bits: &[bool], qubit_count: usize) -> Vec<C> {
        let mut dense = Dense::zero(qubit_count);
        let mut measurement_index = 0usize;
        for op in ops {
            match op {
                Op::Gate(gate, support) => match gate {
                    UnitaryOp::ControlledX => dense.apply_cx(support[0], support[1]),
                    UnitaryOp::ControlledZ => dense.apply_cz(support[0], support[1]),
                    UnitaryOp::Swap => dense.apply_swap(support[0], support[1]),
                    other => dense.apply1(support[0], gate_matrix(*other)),
                },
                Op::Pauli(pauli) => {
                    let (x_bits, z_bits, phase) = pauli_arrays(&pauli.parse::<DensePauli>().unwrap(), qubit_count);
                    dense.apply_pauli(&x_bits, &z_bits, phase);
                }
                Op::PauliExp(pauli) => {
                    let (x_bits, z_bits, phase) = pauli_arrays(&pauli.parse::<DensePauli>().unwrap(), qubit_count);
                    dense.apply_pauli_exp(&x_bits, &z_bits, phase);
                }
                Op::ControlledPauli(first_pauli, second_pauli) => {
                    let first_arrays = pauli_arrays(&first_pauli.parse::<DensePauli>().unwrap(), qubit_count);
                    let second_arrays = pauli_arrays(&second_pauli.parse::<DensePauli>().unwrap(), qubit_count);
                    dense.apply_controlled_pauli(&first_arrays, &second_arrays);
                }
                Op::ConditionalPauli(pauli, outcomes, parity) => {
                    let condition = outcomes
                        .iter()
                        .fold(false, |acc, &outcome_index| acc ^ outcome_bits[outcome_index]);
                    if condition == *parity {
                        let (x_bits, z_bits, phase) = pauli_arrays(&pauli.parse::<DensePauli>().unwrap(), qubit_count);
                        dense.apply_pauli(&x_bits, &z_bits, phase);
                    }
                }
                Op::Measure(pauli) => {
                    let (x_bits, z_bits, phase) = pauli_arrays(&pauli.parse::<DensePauli>().unwrap(), qubit_count);
                    dense.project(&x_bits, &z_bits, phase, outcome_bits[measurement_index]);
                    measurement_index += 1;
                }
            }
        }
        dense.amp
    }

    fn claimed_state(sim: &PhasedOutcomeCompleteSimulation, branch_assignment: &[bool], qubit_count: usize) -> Vec<C> {
        let encoder = sim.phased_state_encoder();
        let base = statevector(&encoder);

        let sign_matrix = sim.aligned_sign_matrix();
        let branch_bit_count = sim.branch_bit_count();
        let mut register = AlignedBitVec::zeros(qubit_count);
        for qubit in 0..qubit_count {
            let mut bit = false;
            for (column, branch_bit) in branch_assignment.iter().enumerate().take(branch_bit_count) {
                if *branch_bit && sign_matrix[(qubit, column)] {
                    bit = !bit;
                }
            }
            register.assign_index(qubit, bit);
        }

        let image = encoder.clifford().image_x_bits(&register);
        let (x_bits, z_bits, phase) = pauli_arrays(&image, qubit_count);
        let mut dense = Dense { qubit_count, amp: base };
        dense.apply_pauli(&x_bits, &z_bits, phase);

        let exponent = i64::from(sim.output_phase_exponent(branch_assignment));
        for amplitude in &mut dense.amp {
            *amplitude *= zeta8(exponent);
        }
        let mut amp = dense.amp;
        normalize(&mut amp);
        amp
    }

    fn outcome_vector(sim: &PhasedOutcomeCompleteSimulation, branch_assignment: &[bool]) -> Vec<bool> {
        let outcome_matrix = sim.aligned_outcome_matrix();
        let shift = sim.aligned_outcome_shift();
        let branch_bit_count = sim.branch_bit_count();
        (0..sim.outcome_count())
            .map(|row| {
                let mut bit = shift.index(row);
                for (column, branch_bit) in branch_assignment.iter().enumerate().take(branch_bit_count) {
                    if *branch_bit && outcome_matrix.row(row).index(column) {
                        bit = !bit;
                    }
                }
                bit
            })
            .collect()
    }

    fn describe(ops: &[Op]) -> String {
        ops.iter()
            .map(|op| match op {
                Op::Gate(gate, support) => format!("Gate({gate:?},{support:?})"),
                Op::Pauli(pauli) => format!("Pauli({pauli})"),
                Op::PauliExp(pauli) => format!("PauliExp({pauli})"),
                Op::ControlledPauli(first_pauli, second_pauli) => format!("CPauli({first_pauli},{second_pauli})"),
                Op::ConditionalPauli(pauli, outcomes, parity) => format!("CondPauli({pauli},{outcomes:?},{parity})"),
                Op::Measure(pauli) => format!("Measure({pauli})"),
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }

    fn verify(ops: &[Op], qubit_count: usize) {
        let sim = run_simulation(ops, qubit_count);
        let branch_bit_count = sim.branch_bit_count();
        assert!(branch_bit_count <= 12, "too many random bits to enumerate");
        for assignment in 0..(1usize << branch_bit_count) {
            let branch_assignment: Vec<bool> = (0..branch_bit_count).map(|bit| (assignment >> bit) & 1 == 1).collect();
            let outcomes = outcome_vector(&sim, &branch_assignment);
            let reference = dense_reference(ops, &outcomes, qubit_count);
            let claimed = claimed_state(&sim, &branch_assignment, qubit_count);
            assert!(
                close(&claimed, &reference),
                "mismatch: ops=[{}] branch_assignment={branch_assignment:?}",
                describe(ops)
            );
        }
    }

    #[test]
    fn single_pauli_measurement() {
        verify(&[Op::Measure("X".into())], 1);
        verify(&[Op::Measure("Y".into())], 1);
        verify(&[Op::Measure("-X".into())], 1);
        verify(&[Op::Measure("-Y".into())], 1);
    }

    #[test]
    fn two_pauli_measurements() {
        verify(&[Op::Measure("X".into()), Op::Measure("Y".into())], 1);
        verify(&[Op::Measure("Y".into()), Op::Measure("X".into())], 1);
        verify(&[Op::Measure("X".into()), Op::Measure("Z".into())], 1);
        verify(&[Op::Measure("-X".into()), Op::Measure("-Y".into())], 1);
    }

    #[test]
    fn measurement_then_conditional() {
        verify(
            &[
                Op::Measure("X".into()),
                Op::ConditionalPauli("Z".into(), vec![0], false),
            ],
            1,
        );
        verify(
            &[Op::Measure("Y".into()), Op::ConditionalPauli("X".into(), vec![0], true)],
            1,
        );
    }

    #[test]
    fn controlled_pauli_no_randomness() {
        verify(
            &[
                Op::Gate(UnitaryOp::Hadamard, vec![0]),
                Op::ControlledPauli("ZI".into(), "IX".into()),
            ],
            2,
        );
        verify(
            &[
                Op::Gate(UnitaryOp::Hadamard, vec![0]),
                Op::ControlledPauli("ZZ".into(), "XX".into()),
            ],
            2,
        );
        verify(
            &[
                Op::Gate(UnitaryOp::SqrtX, vec![0]),
                Op::ControlledPauli("YI".into(), "IY".into()),
            ],
            2,
        );
    }

    #[test]
    fn entangling_then_two_measurements() {
        verify(
            &[
                Op::Gate(UnitaryOp::Hadamard, vec![0]),
                Op::Gate(UnitaryOp::ControlledX, vec![0, 1]),
                Op::Measure("XX".into()),
                Op::Measure("ZI".into()),
            ],
            2,
        );
        verify(
            &[
                Op::Measure("X".into()),
                Op::Gate(UnitaryOp::SqrtX, vec![0]),
                Op::Measure("X".into()),
            ],
            1,
        );
    }

    #[test]
    fn shrink_a() {
        verify(
            &[
                Op::Gate(UnitaryOp::Swap, vec![2, 0]),
                Op::Measure("-ZIX".into()),
                Op::Gate(UnitaryOp::ControlledX, vec![2, 1]),
                Op::Gate(UnitaryOp::SqrtX, vec![0]),
                Op::ConditionalPauli("-ZXI".into(), vec![0], false),
                Op::Gate(UnitaryOp::ControlledZ, vec![2, 1]),
                Op::Measure("XIY".into()),
            ],
            3,
        );
    }

    #[test]
    fn shrink_b_no_conditional() {
        verify(
            &[
                Op::Gate(UnitaryOp::Swap, vec![2, 0]),
                Op::Measure("-ZIX".into()),
                Op::Gate(UnitaryOp::ControlledX, vec![2, 1]),
                Op::Gate(UnitaryOp::SqrtX, vec![0]),
                Op::Gate(UnitaryOp::ControlledZ, vec![2, 1]),
                Op::Measure("XIY".into()),
            ],
            3,
        );
    }

    #[test]
    fn shrink_c_conditional_between() {
        verify(
            &[
                Op::Measure("-ZIX".into()),
                Op::ConditionalPauli("-ZXI".into(), vec![0], false),
                Op::Measure("XIY".into()),
            ],
            3,
        );
    }

    #[test]
    fn shrink_d_two_meas_gate() {
        verify(
            &[
                Op::Measure("ZIX".into()),
                Op::Gate(UnitaryOp::SqrtX, vec![0]),
                Op::Measure("XIY".into()),
            ],
            3,
        );
    }

    #[test]
    fn captured_regression_one() {
        verify(
            &[
                Op::Gate(UnitaryOp::Swap, vec![2, 0]),
                Op::Measure("-ZIX".into()),
                Op::Gate(UnitaryOp::ControlledX, vec![2, 1]),
                Op::Gate(UnitaryOp::SqrtX, vec![0]),
                Op::ConditionalPauli("-ZXI".into(), vec![0], false),
                Op::Gate(UnitaryOp::ControlledZ, vec![2, 1]),
                Op::Measure("XIY".into()),
                Op::Gate(UnitaryOp::SqrtY, vec![1]),
                Op::Gate(UnitaryOp::ControlledX, vec![1, 2]),
                Op::Gate(UnitaryOp::Y, vec![2]),
            ],
            3,
        );
    }

    #[test]
    fn phased_outcome_complete_tracks_dense_statevector() {
        let mut rng = rand::rng();
        for _trial in 0..600 {
            let qubit_count = 3usize;
            let ops = random_circuit(&mut rng, qubit_count);
            let sim = run_simulation(&ops, qubit_count);
            let branch_bit_count = sim.branch_bit_count();
            if branch_bit_count > 8 {
                continue;
            }
            for assignment in 0..(1usize << branch_bit_count) {
                let branch_assignment: Vec<bool> =
                    (0..branch_bit_count).map(|bit| (assignment >> bit) & 1 == 1).collect();
                let outcomes = outcome_vector(&sim, &branch_assignment);
                let reference = dense_reference(&ops, &outcomes, qubit_count);
                let claimed = claimed_state(&sim, &branch_assignment, qubit_count);
                assert!(
                    close(&claimed, &reference),
                    "mismatch: ops=[{}] branch_assignment={branch_assignment:?}",
                    describe(&ops)
                );
            }
        }
    }
}

mod hinted {
    //! Sign-correctness of hinted Pauli measurements (`measure_with_hint`).
    //!
    //! Defining property of a projective measurement: once `P` has been measured, `P` is a stabilizer of
    //! the post-measurement state whose sign equals the reported outcome. In particular, this must hold
    //! no matter which anti-commuting `hint` is supplied — including hints carrying a negative sign,
    //! which drive the `(-1)^alpha` branch of case 5 of Algorithm 4.2.

    use paulimer::UnitaryOp;
    use paulimer::clifford::{Clifford, PhasedCliffordUnitary};
    use paulimer::pauli::{SparsePauli, as_sparse};
    use pauliverse::{PhasedOutcomeCompleteSimulation, Simulation};
    use proptest::collection::vec;
    use proptest::prelude::*;

    #[derive(Clone, Debug)]
    enum Gate {
        Single { op: UnitaryOp, qubit: usize },
        Two { op: UnitaryOp, first: usize, second: usize },
    }

    fn distinct_pair(qubit_count: usize) -> impl Strategy<Value = (usize, usize)> {
        (0..qubit_count, 0..qubit_count - 1)
            .prop_map(|(first, second)| (first, if second < first { second } else { second + 1 }))
    }

    fn gate_strategy(qubit_count: usize) -> BoxedStrategy<Gate> {
        use UnitaryOp::{ControlledX, Hadamard, SqrtX, SqrtZ};
        let single = (prop::sample::select(vec![Hadamard, SqrtZ, SqrtX]), 0..qubit_count)
            .prop_map(|(op, qubit)| Gate::Single { op, qubit });
        if qubit_count >= 2 {
            let two = distinct_pair(qubit_count).prop_map(|(first, second)| Gate::Two {
                op: ControlledX,
                first,
                second,
            });
            prop_oneof![3 => single, 1 => two].boxed()
        } else {
            single.boxed()
        }
    }

    /// Generates a qubit count, a random Clifford preparation circuit, whether to negate the hint, and
    /// the qubit whose stabilizer/destabilizer images drive the measurement.
    fn scenario() -> impl Strategy<Value = (usize, Vec<Gate>, bool, usize)> {
        (1usize..4).prop_flat_map(|qubit_count| {
            (
                Just(qubit_count),
                vec(gate_strategy(qubit_count), 0..12),
                any::<bool>(),
                0..qubit_count,
            )
        })
    }

    fn apply(gate: &Gate, sim: &mut PhasedOutcomeCompleteSimulation, mirror: &mut PhasedCliffordUnitary) {
        match *gate {
            Gate::Single { op, qubit } => {
                sim.unitary_op(op, &[qubit]);
                mirror.left_mul(op, &[qubit]);
            }
            Gate::Two { op, first, second } => {
                sim.unitary_op(op, &[first, second]);
                mirror.left_mul(op, &[first, second]);
            }
        }
    }

    proptest! {
        /// For a random stabilizer state, measuring the destabilizer `X`-image of qubit `target` while
        /// hinting with the (optionally negated) stabilizer `Z`-image of `target` must leave the
        /// observable a stabilizer whose conditional sign matches the reported outcome.
        #[test]
        fn measure_with_hint_outcome_sign_is_correct((qubit_count, gates, negate_hint, target) in scenario()) {
            let mut sim = PhasedOutcomeCompleteSimulation::new(qubit_count);
            let mut mirror = PhasedCliffordUnitary::identity(qubit_count);
            for gate in &gates {
                apply(gate, &mut sim, &mut mirror);
            }

            // `image_z(target)` is a stabilizer of the prepared state; `image_x(target)` anti-commutes
            // with it, so measuring the latter is a genuine (random) case-5 measurement.
            let stabilizer: SparsePauli = as_sparse(&mirror.clifford().image_z(target));
            let observable: SparsePauli = as_sparse(&mirror.clifford().image_x(target));
            let hint = if negate_hint { -stabilizer } else { stabilizer };

            let outcome = sim.measure_with_hint(&observable, &hint);

            prop_assert!(
                sim.is_stabilizer_with_conditional_sign(&observable, &[outcome]),
                "measured observable {observable} is not a stabilizer with the reported outcome sign \
                 (qubit_count={qubit_count}, negate_hint={negate_hint}, target={target})"
            );
        }
    }
}
