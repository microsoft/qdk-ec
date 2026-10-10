use super::{
    action_matrix, congruence_triangularize, find_fix_vector, matrix_row, residue_core, row_reduce_with_transform,
    transvection_matrix, vector_to_pauli, vectors_to_matrix,
};
use crate::clifford::{Clifford, CliffordMutable, CliffordUnitary};
use binar::matrix::AlignedBitMatrix;

mod oracle {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/symplectic_oracle.rs"
    ));
}

fn check_retained_search(qubit_count: usize, group_order: usize, expected_fixes: usize) {
    let dimension = 2 * qubit_count;
    let reached = oracle::enumerate_actions(qubit_count);
    assert_eq!(
        reached.len(),
        group_order,
        "the retained-search oracle must cover the full group"
    );
    let mut fixes = 0;
    for (packed, (minimum, _, _)) in reached {
        let mut action = AlignedBitMatrix::zeros(dimension, dimension);
        for row in 0..dimension {
            for column in 0..dimension {
                action.set((row, column), packed >> (row * dimension + column) & 1 == 1);
            }
        }
        let (basis, rank, core) = residue_core(&action, qubit_count);
        if minimum == rank {
            continue;
        }
        assert_eq!(minimum, rank + 1);
        assert!(congruence_triangularize(&core).is_err());
        let fix = find_fix_vector(&action, qubit_count, &basis, rank);
        assert!(
            fix.iter().any(|&bit| bit),
            "the retained search must return a nonzero fix"
        );
        let mut spanning: Vec<Vec<bool>> = (0..rank).map(|row| matrix_row(&basis, row, dimension)).collect();
        spanning.push(fix.clone());
        assert_eq!(
            row_reduce_with_transform(&vectors_to_matrix(&spanning, dimension))
                .0
                .row_count(),
            rank,
            "the retained fix must lie in the residue space"
        );
        let updated = action.dot(&transvection_matrix(&fix, qubit_count));
        let (updated_basis, updated_rank, updated_core) = residue_core(&updated, qubit_count);
        assert_eq!(updated_rank, rank, "the retained fix must preserve residue rank");
        let transform =
            congruence_triangularize(&updated_core).expect("the retained fix must produce a triangularizable core");
        let defining = transform.dot(&updated_basis);
        let mut rebuilt = CliffordUnitary::identity(qubit_count);
        for row in 0..rank {
            rebuilt.left_mul_pauli_exp(&vector_to_pauli(&matrix_row(&defining, row, dimension), qubit_count));
        }
        rebuilt.left_mul_pauli_exp(&vector_to_pauli(&fix, qubit_count));
        assert!(rebuilt.is_valid(), "the retained search must rebuild a valid tableau");
        assert_eq!(
            action_matrix(&rebuilt),
            action,
            "the retained search must reproduce the input action"
        );
        fixes += 1;
    }
    assert_eq!(fixes, expected_fixes, "the retained-search case count must match");
}

#[test]
fn retained_search_covers_one_and_two_qubit_actions() {
    check_retained_search(1, 6, 0);
    check_retained_search(2, 720, 225);
}

#[test]
#[ignore = "enumerates all three-qubit actions and directly checks every rank-plus-one case"]
fn retained_search_covers_three_qubit_actions() {
    check_retained_search(3, 1_451_520, 150_255);
}
