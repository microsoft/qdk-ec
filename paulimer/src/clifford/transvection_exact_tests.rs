use super::{
    right_multiply_transvection, span_vectors, standard_basis, subspace_key, transvection_matrix,
    triangularize_subspace,
};
use binar::Bitwise;
use binar::matrix::AlignedBitMatrix;
use binar::vec::AlignedBitVec;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::cell::Cell;
use std::collections::HashSet;

thread_local! {
    pub(super) static SPAN_VECTOR_VISITS: Cell<usize> = const { Cell::new(0) };
}

#[test]
fn rank_one_replay_matches_every_one_qubit_matrix_and_vector() {
    for matrix_bits in 0..16 {
        let mut matrix = AlignedBitMatrix::zeros(2, 2);
        for row in 0..2 {
            for column in 0..2 {
                matrix.set((row, column), matrix_bits & (1 << (2 * row + column)) != 0);
            }
        }
        for vector_bits in 0..4 {
            let vector: AlignedBitVec = (0..2).map(|bit| vector_bits & (1 << bit) != 0).collect();
            let expected = matrix.dot(&transvection_matrix(&vector, 1));
            let mut actual = matrix.clone();
            right_multiply_transvection(&mut actual, &vector, 1);
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn rank_one_replay_matches_matrix_products_across_packed_boundaries() {
    let mut generator = StdRng::seed_from_u64(217);
    for qubit_count in [0, 1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257] {
        let dimension = 2 * qubit_count;
        let mut actual = AlignedBitMatrix::random_with_rng(dimension, dimension, &mut generator);
        let mut expected = actual.clone();
        let mut vectors = vec![AlignedBitVec::zeros(dimension), (0..dimension).map(|_| true).collect()];
        for _ in 0..8 {
            vectors.push((0..dimension).map(|_| generator.random::<bool>()).collect());
        }
        for vector in vectors {
            expected = expected.dot(&transvection_matrix(&vector, qubit_count));
            right_multiply_transvection(&mut actual, &vector, qubit_count);
            assert_eq!(actual, expected, "rank-one replay differs at {qubit_count} qubits");
        }
    }
}

#[test]
fn alternating_restriction_does_not_enumerate_span() {
    let dimension = 14;
    let mut core = AlignedBitMatrix::zeros(dimension, dimension);
    for bit in 0..12 {
        core.set((bit, bit ^ 1), true);
    }
    core.set((12, 12), true);
    core.set((13, 13), true);
    let basis: Vec<AlignedBitVec> = standard_basis(dimension).into_iter().take(12).collect();
    let mut unsolvable = HashSet::new();
    SPAN_VECTOR_VISITS.set(0);

    let key = subspace_key(&basis, dimension);
    assert!(triangularize_subspace(&core, &basis, &key, dimension, &mut unsolvable).is_none());
    assert!(unsolvable.contains(&key));
    assert_eq!(SPAN_VECTOR_VISITS.get(), 0);
}

#[test]
fn zero_diagonal_does_not_imply_alternating() {
    let dimension = 3;
    let mut core = AlignedBitMatrix::zeros(dimension, dimension);
    for bit in 0..dimension {
        core.set((bit, (bit + 1) % dimension), true);
    }
    let basis = standard_basis(dimension);
    SPAN_VECTOR_VISITS.set(0);

    let key = subspace_key(&basis, dimension);
    triangularize_subspace(&core, &basis, &key, dimension, &mut HashSet::new());
    assert!(SPAN_VECTOR_VISITS.get() > 0);
}

#[test]
fn span_vectors_supports_more_than_u64_bits_lazily() {
    let dimension = 65;
    let basis = standard_basis(dimension);
    let vectors: Vec<AlignedBitVec> = span_vectors(&basis).take(4).collect();

    assert_eq!(vectors.len(), 4);
    assert_eq!(vectors[0].weight(), 1);
    assert_eq!(vectors[1].weight(), 1);
    assert_eq!(vectors[2].weight(), 2);
    assert_eq!(vectors[3].weight(), 1);
}
