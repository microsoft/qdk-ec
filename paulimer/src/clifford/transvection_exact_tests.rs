use super::{
    SearchBudget, TransvectionSearchError, action_matrix, congruence_triangularize_with_budget,
    find_fix_vector_with_budget, residue_core, right_multiply_transvection, span_vectors, subspace_key,
    transvection_matrix, triangularize_subspace,
};
use crate::clifford::{Clifford, CliffordMutable, CliffordUnitary};
use binar::matrix::AlignedBitMatrix;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::cell::Cell;
use std::collections::HashSet;

thread_local! {
    pub(super) static SPAN_VECTOR_VISITS: Cell<usize> = const { Cell::new(0) };
}

#[test]
fn search_budget_is_shared_by_recursive_subspaces() {
    let core = AlignedBitMatrix::identity(4);
    for limit in 0..12 {
        let mut budget = SearchBudget { remaining: Some(limit) };
        SPAN_VECTOR_VISITS.set(0);
        let result = congruence_triangularize_with_budget(&core, &mut budget);
        assert!(SPAN_VECTOR_VISITS.get() <= limit);
        if result == Err(TransvectionSearchError::BudgetExhausted) {
            assert_eq!(budget.remaining, Some(0));
        } else {
            assert!(result.unwrap().is_some());
        }
    }
}

#[test]
fn retained_fix_search_honors_budget_and_reports_exhaustion() {
    let mut clifford = CliffordUnitary::identity(2);
    clifford.left_mul_swap(0, 1);
    let action = action_matrix(&clifford);
    let (basis, rank, _) = residue_core(&action, 2);
    let mut budget = SearchBudget { remaining: Some(0) };
    assert_eq!(
        find_fix_vector_with_budget(&action, 2, &basis, rank, &mut budget),
        Err(TransvectionSearchError::BudgetExhausted)
    );
    assert_eq!(
        find_fix_vector_with_budget(
            &action,
            2,
            &AlignedBitMatrix::zeros(0, 4),
            0,
            &mut SearchBudget::default()
        ),
        Err(TransvectionSearchError::NoFixVector)
    );
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
            let vector: Vec<bool> = (0..2).map(|bit| vector_bits & (1 << bit) != 0).collect();
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
        let mut vectors = vec![vec![false; dimension], vec![true; dimension]];
        for _ in 0..8 {
            vectors.push((0..dimension).map(|_| generator.random()).collect());
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
