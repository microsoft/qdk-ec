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

use dense_oracle::{C, Dense, close, gate_matrix, normalize, pauli_arrays, statevector, zeta8};

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

fn claimed_state(sim: &PhasedOutcomeCompleteSimulation, random_bits: &[bool], qubit_count: usize) -> Vec<C> {
    let encoder = sim.phased_state_encoder();
    let base = statevector(&encoder);

    let sign_matrix = sim.aligned_sign_matrix();
    let random_outcome_count = sim.random_outcome_count();
    let mut register = AlignedBitVec::zeros(qubit_count);
    for qubit in 0..qubit_count {
        let mut bit = false;
        for (column, random_bit) in random_bits.iter().enumerate().take(random_outcome_count) {
            if *random_bit && sign_matrix[(qubit, column)] {
                bit = !bit;
            }
        }
        register.assign_index(qubit, bit);
    }

    let image = encoder.clifford().image_x_bits(&register);
    let (x_bits, z_bits, phase) = pauli_arrays(&image, qubit_count);
    let mut dense = Dense { qubit_count, amp: base };
    dense.apply_pauli(&x_bits, &z_bits, phase);

    let exponent = i64::from(sim.output_phase_exponent(random_bits));
    for amplitude in &mut dense.amp {
        *amplitude *= zeta8(exponent);
    }
    let mut amp = dense.amp;
    normalize(&mut amp);
    amp
}

fn outcome_vector(sim: &PhasedOutcomeCompleteSimulation, random_bits: &[bool]) -> Vec<bool> {
    let outcome_matrix = sim.aligned_outcome_matrix();
    let shift = sim.aligned_outcome_shift();
    let random_outcome_count = sim.random_outcome_count();
    (0..sim.outcome_count())
        .map(|row| {
            let mut bit = shift.index(row);
            for (column, random_bit) in random_bits.iter().enumerate().take(random_outcome_count) {
                if *random_bit && outcome_matrix.row(row).index(column) {
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
    let random_outcome_count = sim.random_outcome_count();
    assert!(random_outcome_count <= 12, "too many random bits to enumerate");
    for assignment in 0..(1usize << random_outcome_count) {
        let random_bits: Vec<bool> = (0..random_outcome_count)
            .map(|bit| (assignment >> bit) & 1 == 1)
            .collect();
        let outcomes = outcome_vector(&sim, &random_bits);
        let reference = dense_reference(ops, &outcomes, qubit_count);
        let claimed = claimed_state(&sim, &random_bits, qubit_count);
        assert!(
            close(&claimed, &reference),
            "mismatch: ops=[{}] random_bits={random_bits:?}",
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
        let random_outcome_count = sim.random_outcome_count();
        if random_outcome_count > 8 {
            continue;
        }
        for assignment in 0..(1usize << random_outcome_count) {
            let random_bits: Vec<bool> = (0..random_outcome_count)
                .map(|bit| (assignment >> bit) & 1 == 1)
                .collect();
            let outcomes = outcome_vector(&sim, &random_bits);
            let reference = dense_reference(&ops, &outcomes, qubit_count);
            let claimed = claimed_state(&sim, &random_bits, qubit_count);
            assert!(
                close(&claimed, &reference),
                "mismatch: ops=[{}] random_bits={random_bits:?}",
                describe(&ops)
            );
        }
    }
}
