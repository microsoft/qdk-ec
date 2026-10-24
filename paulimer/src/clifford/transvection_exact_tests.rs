use super::{span_vectors, subspace_key, triangularize_subspace};
use binar::matrix::AlignedBitMatrix;
use std::cell::Cell;
use std::collections::HashSet;

thread_local! {
    pub(super) static SPAN_VECTOR_VISITS: Cell<usize> = const { Cell::new(0) };
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
    let basis: Vec<Vec<bool>> = (0..12)
        .map(|row| (0..dimension).map(|column| row == column).collect())
        .collect();
    let mut unsolvable = HashSet::new();
    SPAN_VECTOR_VISITS.set(0);

    assert!(triangularize_subspace(&core, &basis, dimension, &mut unsolvable).is_none());
    assert!(unsolvable.contains(&subspace_key(&basis, dimension)));
    assert_eq!(SPAN_VECTOR_VISITS.get(), 0);
}

#[test]
fn zero_diagonal_does_not_imply_alternating() {
    let dimension = 3;
    let mut core = AlignedBitMatrix::zeros(dimension, dimension);
    for bit in 0..dimension {
        core.set((bit, (bit + 1) % dimension), true);
    }
    let basis: Vec<Vec<bool>> = (0..dimension)
        .map(|row| (0..dimension).map(|column| row == column).collect())
        .collect();
    SPAN_VECTOR_VISITS.set(0);

    triangularize_subspace(&core, &basis, dimension, &mut HashSet::new());
    assert!(SPAN_VECTOR_VISITS.get() > 0);
}

#[test]
fn span_vectors_supports_more_than_u64_bits_lazily() {
    let dimension = 65;
    let basis: Vec<Vec<bool>> = (0..dimension)
        .map(|row| (0..dimension).map(|column| row == column).collect())
        .collect();
    let vectors: Vec<Vec<bool>> = span_vectors(&basis).take(4).collect();

    assert_eq!(vectors.len(), 4);
    assert_eq!(vectors[0].iter().filter(|&&bit| bit).count(), 1);
    assert_eq!(vectors[1].iter().filter(|&&bit| bit).count(), 1);
    assert_eq!(vectors[2].iter().filter(|&&bit| bit).count(), 2);
    assert_eq!(vectors[3].iter().filter(|&&bit| bit).count(), 1);
}
