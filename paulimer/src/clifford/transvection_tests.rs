use super::{assign_positive_hermitian_phase, clifford_to_transvections, symplectic_basis, symplectic_residue_matrix};
use crate::clifford::{Clifford, CliffordMutable, CliffordUnitary};
use crate::{DensePauli, Pauli, PauliBinaryOps, SparsePauli, anti_commutes_with};
use binar::Bitwise;
use binar::matrix::AlignedBitMatrix;
use rand::{SeedableRng, rngs::StdRng};

fn reference_residue(pauli: &SparsePauli, image: &DensePauli) -> SparsePauli {
    let mut vector: SparsePauli = image.clone().into();
    vector.mul_assign_left(pauli);
    assign_positive_hermitian_phase(&mut vector);
    vector
}

fn reference_next_transvection(clifford: &CliffordUnitary) -> Option<SparsePauli> {
    let basis: Vec<_> = symplectic_basis(clifford.num_qubits()).collect();
    let images: Vec<_> = basis.iter().map(|pauli| clifford.image(pauli)).collect();
    for (pauli, image) in basis.iter().zip(&images) {
        if anti_commutes_with(pauli, image) {
            return Some(reference_residue(pauli, image));
        }
    }
    for first in 0..basis.len() {
        for second in first + 1..basis.len() {
            if anti_commutes_with(&basis[first], &images[second]) ^ anti_commutes_with(&basis[second], &images[first]) {
                let mut sum = basis[first].clone();
                sum.mul_assign_left(&basis[second]);
                let mut image = images[first].clone();
                image.mul_assign_left(&images[second]);
                return Some(reference_residue(&sum, &image));
            }
        }
    }
    basis
        .iter()
        .zip(&images)
        .map(|(pauli, image)| reference_residue(pauli, image))
        .find(|vector| !vector.x_bits().is_zero() || !vector.z_bits().is_zero())
}

fn assert_matches_reference(clifford: &CliffordUnitary) {
    let qubit_count = clifford.num_qubits();
    let mut expected_residue = AlignedBitMatrix::zeros(2 * qubit_count, 2 * qubit_count);
    for (row, basis) in symplectic_basis(qubit_count).enumerate() {
        let vector = reference_residue(&basis, &clifford.image(&basis));
        for qubit in 0..qubit_count {
            expected_residue.set((row, qubit), vector.x_bits().index(qubit));
            expected_residue.set((row, qubit_count + qubit), vector.z_bits().index(qubit));
        }
    }
    assert_eq!(symplectic_residue_matrix(clifford), expected_residue);

    let mut working = clifford.clone();
    let mut expected = Vec::new();
    while let Some(vector) = reference_next_transvection(&working) {
        working.left_mul_pauli_exp(&vector);
        expected.push(vector);
        assert!(expected.len() <= 4 * qubit_count + 2);
    }
    expected.reverse();
    assert_eq!(clifford_to_transvections(clifford), expected);
}

#[test]
fn packed_reduction_preserves_image_convention_and_factor_order() {
    assert_matches_reference(&CliffordUnitary::identity(0));
    for qubit_count in [1, 2, 3, 7, 31, 32, 33, 63, 64, 65] {
        for seed in 0..3 {
            let clifford = CliffordUnitary::random(qubit_count, &mut StdRng::seed_from_u64(seed));
            assert_matches_reference(&clifford);
        }
    }
}

#[test]
fn packed_reduction_crosses_aligned_block_boundaries() {
    for qubit_count in [255, 256, 257] {
        let mut clifford = CliffordUnitary::identity(qubit_count);
        clifford.left_mul_hadamard(qubit_count - 1);
        clifford.left_mul_cx(0, qubit_count - 1);
        clifford.left_mul_root_z(0);
        assert_matches_reference(&clifford);
    }
}
