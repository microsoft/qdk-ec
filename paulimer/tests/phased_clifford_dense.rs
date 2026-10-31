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

#[test]
fn usize_amplitude_zero_extends_at_platform_width() {
    let width = usize::BITS as usize;
    for qubit_count in [width - 1, width, width + 1, width * 2 + 1] {
        let mut clifford = PhasedCliffordUnitary::identity(qubit_count);
        clifford.left_mul(UnitaryOp::X, &[0]);
        assert_eq!(clifford.state_amplitude_phase_exponent_usize(1), Some(0));
        assert_eq!(clifford.state_amplitude_phase_exponent_usize(0), None);
        if qubit_count >= width {
            clifford.left_mul(UnitaryOp::X, &[width - 1]);
            assert_eq!(
                clifford.state_amplitude_phase_exponent_usize(1 | (1usize << (width - 1))),
                Some(0)
            );
        }
        if qubit_count > width {
            clifford.left_mul(UnitaryOp::X, &[width]);
            assert_eq!(
                clifford.state_amplitude_phase_exponent_usize(1 | (1usize << (width - 1))),
                None
            );
        }
    }
}

mod auxiliary_separation {
    //! Validation of the §4.3 auxiliary-qubit separation primitive (arXiv:2603.24717).
    //!
    //! For random encoders whose auxiliary qubits are disentangled from the output qubits, the recovered
    //! `(Co₁, Co₂, A₁, A₂, φ)` are checked against the encoder state for *every* label `r`:
    //! `encoder|r⟩ == ζ₈^{φ(r)} (Co₁|A₁r⟩ ⊗ Co₂|A₂r⟩)`, comparing exact `ζ₈` phases.

    use std::collections::BTreeSet;

    use binar::matrix::AlignedBitMatrix;
    use binar::vec::AlignedBitVec;
    use binar::{Bitwise, BitwiseMut};
    use paulimer::clifford::{
        AuxiliarySeparationError, Clifford, PhasedCliffordUnitary, clifford_to_pauli_exponents,
        random_clifford_via_operations_sampling, separate_auxiliary_qubits,
    };
    use paulimer::operations::{css_operations, diagonal_operations};
    use paulimer::{CliffordMutable, CliffordUnitary, DensePauli};
    use rand::SeedableRng;

    fn rand_below(rng: &mut impl rand::RngExt, bound: usize) -> usize {
        usize::try_from(rng.next_u64() % bound as u64).expect("a value below bound fits in usize")
    }

    fn phased_from_clifford(clifford: &CliffordUnitary) -> PhasedCliffordUnitary {
        let mut phased = PhasedCliffordUnitary::identity(clifford.num_qubits());
        for exponent in clifford_to_pauli_exponents(clifford) {
            phased.left_mul_pauli_exp(&exponent);
        }
        phased
    }

    fn encoded_basis_state(clifford: &PhasedCliffordUnitary, bits: &AlignedBitVec) -> PhasedCliffordUnitary {
        let num_qubits = clifford.num_qubits();
        let pauli = DensePauli::from_bits(bits.clone(), AlignedBitVec::zeros(num_qubits), 0);
        let image = clifford.clifford().image(&pauli);
        let mut state = clifford.clone();
        state.left_mul_pauli(&image);
        state
    }

    /// The set of `ζ₈` exponent differences between two phased states over their common-support basis
    /// states. A single element means the states agree up to one global phase.
    fn phase_diffs(reference: &PhasedCliffordUnitary, candidate: &PhasedCliffordUnitary) -> BTreeSet<i64> {
        let num_qubits = reference.num_qubits();
        let mut diffs = BTreeSet::new();
        for value in 0..(1usize << num_qubits) {
            let mut basis = AlignedBitVec::zeros(num_qubits);
            for qubit in 0..num_qubits {
                if (value >> qubit) & 1 == 1 {
                    basis.assign_index(qubit, true);
                }
            }
            if let (Some(a), Some(b)) = (
                reference.state_amplitude_phase_exponent(&basis),
                candidate.state_amplitude_phase_exponent(&basis),
            ) {
                diffs.insert((i64::from(a) - i64::from(b)).rem_euclid(8));
            }
        }
        diffs
    }

    fn label_bits(value: usize, num_qubits: usize) -> AlignedBitVec {
        let mut bits = AlignedBitVec::zeros(num_qubits);
        for qubit in 0..num_qubits {
            if (value >> qubit) & 1 == 1 {
                bits.assign_index(qubit, true);
            }
        }
        bits
    }

    /// Concatenate `A₁ r` (on the leading qubits) and `A₂ r` (on the trailing qubits) into one label.
    fn block_labels(
        output_map: &AlignedBitMatrix,
        auxiliary_map: &AlignedBitMatrix,
        label: &AlignedBitVec,
    ) -> AlignedBitVec {
        let output = output_map * &label.as_view();
        let auxiliary = auxiliary_map * &label.as_view();
        let output_count = output_map.row_count();
        let mut combined = AlignedBitVec::zeros(output_count + auxiliary_map.row_count());
        for bit in output.support() {
            combined.assign_index(bit, true);
        }
        for bit in auxiliary.support() {
            combined.assign_index(output_count + bit, true);
        }
        combined
    }

    fn check_separation_with_outputs(co_clifford: &CliffordUnitary, output: &[usize]) {
        let num_qubits = co_clifford.num_qubits();
        let encoder = phased_from_clifford(co_clifford);
        let output_set: BTreeSet<usize> = output.iter().copied().collect();
        let auxiliary: Vec<usize> = (0..num_qubits).filter(|qubit| !output_set.contains(qubit)).collect();
        let qubit_order: Vec<usize> = output.iter().chain(&auxiliary).copied().collect();
        let all_qubits: Vec<usize> = (0..num_qubits).collect();
        let mut ordered_encoder = encoder.clone();
        ordered_encoder.left_mul_permutation(&qubit_order, &all_qubits);

        let separation = separate_auxiliary_qubits(&encoder, output).expect("disentangled auxiliary qubits separate");
        let blocks = phased_from_clifford(&separation.output_encoder().tensor(separation.auxiliary_encoder()));

        for value in 0..(1usize << num_qubits) {
            let label = label_bits(value, num_qubits);
            let encoder_state = encoded_basis_state(&ordered_encoder, &label);
            let block_label = block_labels(separation.output_basis_map(), separation.auxiliary_basis_map(), &label);
            let block_state = encoded_basis_state(&blocks, &block_label);

            let diffs = phase_diffs(&encoder_state, &block_state);
            assert_eq!(
                diffs.len(),
                1,
                "n={num_qubits} output={output:?} r={value}: not a single global phase: {diffs:?}"
            );
            let expected = i64::from(separation.phase().evaluate(&label)).rem_euclid(8);
            assert_eq!(
                *diffs.iter().next().unwrap(),
                expected,
                "n={num_qubits} output={output:?} r={value}: phase polynomial mismatch",
            );
        }
    }

    fn check_separation(co_clifford: &CliffordUnitary, k1: usize) {
        check_separation_with_outputs(co_clifford, &(0..k1).collect::<Vec<_>>());
    }

    #[test]
    fn block_diagonal_separation() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for _ in 0..120 {
            let k1 = 1 + rand_below(&mut rng, 2);
            let k2 = 1 + rand_below(&mut rng, 2);
            let c1 = CliffordUnitary::random(k1, &mut rng);
            let c2 = CliffordUnitary::random(k2, &mut rng);
            check_separation(&c1.tensor(&c2), k1);
        }
    }

    #[test]
    fn general_separable_separation() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(23);
        for _ in 0..80 {
            let k1 = 1 + rand_below(&mut rng, 2);
            let k2 = 1 + rand_below(&mut rng, 2);
            let n = k1 + k2;
            let c1 = CliffordUnitary::random(k1, &mut rng);
            let c2 = CliffordUnitary::random(k2, &mut rng);
            let diagonal: CliffordUnitary =
                random_clifford_via_operations_sampling(n, n * n, &diagonal_operations(n), &mut rng);
            let css: CliffordUnitary = random_clifford_via_operations_sampling(n, n * n, &css_operations(n), &mut rng);
            let co_clifford = c1.tensor(&c2).multiply_with(&diagonal).multiply_with(&css);
            check_separation(&co_clifford, k1);
        }
    }

    #[test]
    fn entangled_auxiliary_qubits_are_rejected() {
        let mut clifford = CliffordUnitary::identity(2);
        clifford.left_mul_hadamard(0);
        clifford.left_mul_cx(0, 1);
        let encoder = phased_from_clifford(&clifford);
        assert_eq!(
            separate_auxiliary_qubits(&encoder, &[0]).unwrap_err(),
            AuxiliarySeparationError::AuxiliaryQubitsEntangled,
        );
    }

    #[test]
    fn invalid_output_qubits_are_rejected() {
        let encoder = PhasedCliffordUnitary::identity(2);
        assert_eq!(
            separate_auxiliary_qubits(&encoder, &[2]).unwrap_err(),
            AuxiliarySeparationError::InvalidOutputQubits,
        );
        assert_eq!(
            separate_auxiliary_qubits(&encoder, &[0, 0]).unwrap_err(),
            AuxiliarySeparationError::InvalidOutputQubits,
        );
    }

    #[test]
    fn non_leading_output_subset_preserves_exact_phase_and_basis_map() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(47);
        let output = [2, 0];
        let inverse_order = [1, 2, 0];
        let all_qubits = [0, 1, 2];
        for _ in 0..40 {
            let ordered_product = CliffordUnitary::random(2, &mut rng).tensor(&CliffordUnitary::random(1, &mut rng));
            let mut physical_encoder = ordered_product;
            physical_encoder.left_mul_permutation(&inverse_order, &all_qubits);
            check_separation_with_outputs(&physical_encoder, &output);
        }
    }

    #[test]
    fn zero_qubit_separation_is_trivial() {
        let encoder = PhasedCliffordUnitary::identity(0);
        let separation = separate_auxiliary_qubits(&encoder, &[]).expect("the empty encoder separates");
        assert_eq!(separation.output_encoder().num_qubits(), 0);
        assert_eq!(separation.auxiliary_encoder().num_qubits(), 0);
        assert_eq!(separation.phase().num_bits(), 0);
        assert_eq!(separation.phase().constant(), 0, "the identity carries no phase");
    }
}
