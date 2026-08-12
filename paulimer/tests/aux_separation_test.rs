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

/// Run the separation on `co_clifford` with output qubits `[0, k1)` and check eq. aux-out-separation
/// for every label `r`.
fn check_separation(co_clifford: &CliffordUnitary, k1: usize) {
    let num_qubits = co_clifford.num_qubits();
    let output: Vec<usize> = (0..k1).collect();
    let encoder = phased_from_clifford(co_clifford);

    let separation = separate_auxiliary_qubits(&encoder, &output).expect("disentangled auxiliary qubits separate");
    let blocks = phased_from_clifford(&separation.output_encoder().tensor(separation.auxiliary_encoder()));

    for value in 0..(1usize << num_qubits) {
        let label = label_bits(value, num_qubits);
        let encoder_state = encoded_basis_state(&encoder, &label);
        let block_label = block_labels(separation.output_basis_map(), separation.auxiliary_basis_map(), &label);
        let block_state = encoded_basis_state(&blocks, &block_label);

        let diffs = phase_diffs(&encoder_state, &block_state);
        assert_eq!(
            diffs.len(),
            1,
            "n={num_qubits} k1={k1} r={value}: not a single global phase: {diffs:?}"
        );
        let expected = i64::from(separation.phase().evaluate(&label)).rem_euclid(8);
        assert_eq!(
            *diffs.iter().next().unwrap(),
            expected,
            "n={num_qubits} k1={k1} r={value}: phase polynomial mismatch",
        );
    }
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
