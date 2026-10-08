use dense_oracle::{Dense, close, gate_matrix, pauli_arrays, statevector};
use paulimer::clifford::PhasedCliffordUnitary;
use paulimer::pauli::Pauli;
use paulimer::{DensePauli, UnitaryOp};
use proptest::collection::vec;
use proptest::prelude::*;

#[derive(Clone, Debug)]
enum Gate {
    Single { op: UnitaryOp, qubit: usize },
    Two { op: UnitaryOp, first: usize, second: usize },
    Pauli(DensePauli),
    PauliExp(DensePauli),
}

fn distinct_pair(qubit_count: usize) -> impl Strategy<Value = (usize, usize)> {
    (0..qubit_count, 0..qubit_count - 1)
        .prop_map(|(first, second)| (first, if second < first { second } else { second + 1 }))
}

fn pauli_strategy(qubit_count: usize, hermitian: bool) -> impl Strategy<Value = DensePauli> {
    (vec(any::<bool>(), qubit_count), vec(any::<bool>(), qubit_count), 0u8..4)
        .prop_map(|(x_bits, z_bits, phase)| {
            DensePauli::from_bits(x_bits.into_iter().collect(), z_bits.into_iter().collect(), phase)
        })
        .prop_filter("non-identity, Hermitian when required", move |pauli| {
            !pauli.is_identity() && (!hermitian || pauli.is_order_two())
        })
}

fn gate_strategy(qubit_count: usize) -> impl Strategy<Value = Gate> {
    use UnitaryOp::{
        ControlledX, ControlledZ, Hadamard, PrepareBell, SqrtX, SqrtXInv, SqrtY, SqrtYInv, SqrtZ, SqrtZInv, Swap, X, Y,
        Z,
    };
    let single = (
        prop::sample::select(vec![
            Hadamard, X, Y, Z, SqrtZ, SqrtZInv, SqrtX, SqrtXInv, SqrtY, SqrtYInv,
        ]),
        0..qubit_count,
    )
        .prop_map(|(op, qubit)| Gate::Single { op, qubit });
    let two = (
        prop::sample::select(vec![ControlledX, ControlledZ, Swap, PrepareBell]),
        distinct_pair(qubit_count),
    )
        .prop_map(|(op, (first, second))| Gate::Two { op, first, second });
    prop_oneof![
        10 => single,
        4 => two,
        1 => pauli_strategy(qubit_count, false).prop_map(Gate::Pauli),
        1 => pauli_strategy(qubit_count, true).prop_map(Gate::PauliExp),
    ]
}

fn apply(gate: &Gate, dense: &mut Dense, phased: &mut PhasedCliffordUnitary) {
    use UnitaryOp::{ControlledX, ControlledZ, Hadamard, PrepareBell, Swap};
    let qubit_count = dense.qubit_count;
    match gate {
        &Gate::Single { op, qubit } => {
            dense.apply1(qubit, gate_matrix(op));
            phased.left_mul(op, &[qubit]);
        }
        &Gate::Two { op, first, second } => {
            match op {
                ControlledX => dense.apply_cx(first, second),
                ControlledZ => dense.apply_cz(first, second),
                Swap => dense.apply_swap(first, second),
                PrepareBell => {
                    dense.apply1(first, gate_matrix(Hadamard));
                    dense.apply_cx(first, second);
                }
                _ => unreachable!("Gate::Two only carries two-qubit ops"),
            }
            phased.left_mul(op, &[first, second]);
        }
        Gate::Pauli(pauli) => {
            let (x_bits, z_bits, phase) = pauli_arrays(pauli, qubit_count);
            dense.apply_pauli(&x_bits, &z_bits, phase);
            phased.left_mul_pauli(pauli);
        }
        Gate::PauliExp(pauli) => {
            let (x_bits, z_bits, phase) = pauli_arrays(pauli, qubit_count);
            dense.apply_pauli_exp(&x_bits, &z_bits, phase);
            phased.left_mul_pauli_exp(pauli);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]
    #[test]
    fn phased_clifford_tracks_dense_statevector(gates in vec(gate_strategy(4), 0..40)) {
        let qubit_count = 4;
        let mut dense = Dense::zero(qubit_count);
        let mut phased = PhasedCliffordUnitary::identity(qubit_count);
        for gate in &gates {
            apply(gate, &mut dense, &mut phased);
        }
        prop_assert!(close(&statevector(&phased), &dense.amp), "diverged on {gates:?}");
    }
}

#[test]
fn zero_qubit_amplitude_phase_is_zero() {
    // The empty system has the single amplitude 1, so its zeta_8 exponent is 0. Reading it must
    // not index a generator that does not exist.
    let unitary = PhasedCliffordUnitary::identity(0);
    let basis = binar::vec::AlignedBitVec::zeros(0);

    assert_eq!(unitary.state_amplitude_phase_exponent(&basis), Some(0));
    assert_eq!(unitary.state_amplitude_phase_exponent_usize(0), Some(0));
}

/// A Clifford with no qubits has only the scalar Paulis, so `exp(iπ/4 · P)` is a number: `exp(iπ/4)`
/// for `I` and `exp(-iπ/4)` for `-I`. The zero qubit path used to return without recording it.
#[test]
fn zero_qubit_pauli_exponential_keeps_its_phase() {
    use binar::vec::AlignedBitVec;
    let scalar = |phase: u8| DensePauli::from_bits(AlignedBitVec::zeros(0), AlignedBitVec::zeros(0), phase);

    for (phase, expected) in [(0_u8, 1_u8), (2, 7)] {
        let mut unitary = PhasedCliffordUnitary::identity(0);
        unitary.left_mul_pauli_exp(&scalar(phase));
        assert_eq!(
            unitary.state_amplitude_phase_exponent(&AlignedBitVec::zeros(0)),
            Some(expected),
            "a scalar Pauli with phase {phase} contributes the eighth root {expected}"
        );
    }

    let mut unitary = PhasedCliffordUnitary::identity(0);
    unitary.left_mul_pauli_exp(&scalar(0));
    unitary.left_mul_pauli_exp(&scalar(2));
    assert_eq!(
        unitary.state_amplitude_phase_exponent(&AlignedBitVec::zeros(0)),
        Some(0),
        "opposite rotations cancel"
    );
}
