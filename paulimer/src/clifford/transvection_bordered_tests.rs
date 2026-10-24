use super::tests::SPAN_VECTOR_VISITS;
use super::{
    BorderedFamily, BorderedFamilyIssue, action_matrix, bordered_decomposition, bordered_family,
    clifford_to_transvections_minimal, final_pauli_correction, minimal_decomposition, residue_core,
    transvection_matrix, vector_to_pauli, vectors_to_matrix, verified_decomposition_or_search,
    verifies_bordered_decomposition,
};
use crate::clifford::{Clifford, CliffordMutable, CliffordUnitary};
use crate::{Pauli, SparsePauli};
use binar::IndexSet;
use binar::matrix::AlignedBitMatrix;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::cell::Cell;

thread_local! {
    pub(super) static FALLBACK_CALLS: Cell<usize> = const { Cell::new(0) };
}

fn class_a_clifford() -> CliffordUnitary {
    let mut clifford = CliffordUnitary::identity(2);
    for pauli in [
        SparsePauli::x(0, 2),
        SparsePauli::x(1, 2),
        SparsePauli::from_bits([0, 1].into_iter().collect(), IndexSet::new(), 0),
        SparsePauli::z(0, 2),
    ] {
        clifford.left_mul_pauli_exp(&pauli);
    }
    clifford
}

fn class_a_action() -> AlignedBitMatrix {
    action_matrix(&class_a_clifford())
}

fn class_a_with_signs(mask: usize) -> CliffordUnitary {
    let vector: Vec<bool> = (0..4).map(|bit| mask & (1 << bit) != 0).collect();
    let mut clifford = class_a_clifford();
    clifford.left_mul_pauli(&vector_to_pauli(&vector, 2));
    clifford
}

fn small_matrix(rows: &[u8], columns: usize) -> AlignedBitMatrix {
    let vectors: Vec<Vec<bool>> = rows
        .iter()
        .map(|&row| (0..columns).map(|column| row >> column & 1 == 1).collect())
        .collect();
    vectors_to_matrix(&vectors, columns)
}

fn small_family(rows: &[u8], columns: usize, relation: u8) -> BorderedFamily {
    BorderedFamily {
        vectors: rows
            .iter()
            .map(|&row| (0..columns).map(|column| row >> column & 1 == 1).collect())
            .collect(),
        relation: (0..rows.len()).map(|index| relation >> index & 1 == 1).collect(),
    }
}

fn assert_bordered_replay(action: &AlignedBitMatrix, qubit_count: usize, vectors: &[Vec<bool>]) {
    let mut rebuilt = CliffordUnitary::identity(qubit_count);
    for vector in vectors {
        let pauli = vector_to_pauli(vector, qubit_count);
        assert!(pauli.is_order_two(), "bordered factors must be Hermitian");
        assert_eq!(
            pauli.xyz_phase_exponent(),
            0,
            "bordered factors must have positive phase"
        );
        rebuilt.left_mul_pauli_exp(&pauli);
    }
    assert!(rebuilt.is_valid(), "bordered replay must form a valid tableau");
    assert_eq!(
        action_matrix(&rebuilt),
        *action,
        "bordered factors must reproduce the input action"
    );
}

#[test]
fn bordered_family_handles_nonsymmetric_zero_diagonal() {
    let mut core = AlignedBitMatrix::zeros(3, 3);
    for index in 0..3 {
        core.set((index, (index + 1) % 3), true);
    }
    let standard = vec![
        vec![true, false, false],
        vec![false, true, false],
        vec![false, false, true],
    ];
    let family = bordered_family(&core, &standard).expect("a nonsymmetric zero diagonal requires a pair pivot");
    assert!(
        family.is_valid(&core),
        "the pair pivot must produce a valid bordered family"
    );
    assert!(
        !family.relation[0],
        "a non-alternating form must peel a vector before its simplex"
    );

    let empty = AlignedBitMatrix::zeros(0, 0);
    let family = bordered_family(&empty, &[]).expect("the zero-dimensional simplex must exist");
    assert!(family.is_valid(&empty));
    assert_eq!(family.vectors, vec![Vec::<bool>::new()]);
    assert_eq!(family.relation, vec![true]);
}

#[test]
fn bordered_core_requires_square_shape() {
    let family = small_family(&[1, 2, 0], 2, 4);
    let core = small_matrix(&[1, 2], 3);
    let result =
        std::panic::catch_unwind(|| family.validate(&core)).expect("a nonsquare core must be rejected without a panic");
    assert_eq!(
        result,
        Err(vec![BorderedFamilyIssue::CoreShape]),
        "the core shape check must run"
    );
}

#[test]
fn bordered_family_requires_vector_count() {
    let family = small_family(&[1], 1, 0);
    assert_eq!(
        family.validate(&small_matrix(&[1], 1)),
        Err(vec![
            BorderedFamilyIssue::VectorCount,
            BorderedFamilyIssue::EvenRelation
        ]),
        "validation must report the wrong vector count"
    );
}

#[test]
fn bordered_family_requires_relation_length() {
    let mut family = small_family(&[1, 2, 0], 2, 4);
    family.relation.push(false);
    assert_eq!(
        family.validate(&AlignedBitMatrix::identity(2)),
        Err(vec![BorderedFamilyIssue::RelationLength]),
        "validation must report the wrong relation length"
    );
}

#[test]
fn bordered_family_requires_vector_length() {
    let mut family = small_family(&[1, 2, 0], 2, 4);
    family.vectors[0].push(false);
    assert_eq!(
        family.validate(&AlignedBitMatrix::identity(2)),
        Err(vec![BorderedFamilyIssue::VectorLength]),
        "validation must report the wrong vector length"
    );
}

#[test]
fn bordered_family_requires_odd_relation() {
    let family = small_family(&[1, 0], 1, 0);
    assert_eq!(
        family.validate(&small_matrix(&[1], 1)),
        Err(vec![
            BorderedFamilyIssue::EvenRelation,
            BorderedFamilyIssue::NonUnitLower
        ]),
        "validation must report an even relation"
    );
}

#[test]
fn bordered_family_requires_unit_lower_form() {
    let family = small_family(&[1, 2, 0], 2, 4);
    assert_eq!(
        family.validate(&small_matrix(&[2, 3], 2)),
        Err(vec![BorderedFamilyIssue::NonUnitLower]),
        "validation must report a non-unit-lower form"
    );
}

#[test]
fn bordered_family_requires_zero_relation_sum() {
    let family = small_family(&[1, 2, 4, 7], 3, 13);
    assert_eq!(
        family.validate(&small_matrix(&[4, 3, 1], 3)),
        Err(vec![BorderedFamilyIssue::NonzeroRelation]),
        "validation must report a nonzero relation sum"
    );
}

#[test]
fn bordered_family_requires_full_rank() {
    let family = small_family(&[1, 0, 0], 2, 4);
    assert_eq!(
        family.validate(&AlignedBitMatrix::identity(2)),
        Err(vec![
            BorderedFamilyIssue::NonUnitLower,
            BorderedFamilyIssue::MissingRank
        ]),
        "validation must report a missing rank"
    );
}

#[test]
fn bordered_public_nonsymmetric_path_does_not_fall_back() {
    let clifford = class_a_clifford();
    let action = action_matrix(&clifford);
    let (_, rank, core) = residue_core(&action, 2);
    assert_ne!(core, core.transposed());
    assert_ne!(action.dot(&action), AlignedBitMatrix::identity(4));
    for mask in 0..16 {
        let clifford = class_a_with_signs(mask);
        FALLBACK_CALLS.set(0);
        let decomposition = clifford_to_transvections_minimal(&clifford);
        assert_eq!(
            FALLBACK_CALLS.get(),
            0,
            "the public nonsymmetric path must use the construction"
        );
        assert_eq!(decomposition.transvections.len(), rank + 1);
        let mut rebuilt = CliffordUnitary::identity(2);
        for factor in &decomposition.transvections {
            rebuilt.left_mul_pauli_exp(factor);
        }
        assert!(rebuilt.is_valid());
        assert_eq!(action_matrix(&rebuilt), action);
        rebuilt.left_mul_pauli(&decomposition.pauli_correction);
        assert_eq!(rebuilt, clifford);
    }
}

#[test]
fn bordered_verification_rejects_identity_padding() {
    let vector = vec![true, false];
    let action = transvection_matrix(&vector, 1);
    let vectors = vec![vector, vec![false; 2]];
    assert_bordered_replay(&action, 1, &vectors);
    assert!(
        !verifies_bordered_decomposition(&action, 1, 1, &vectors),
        "identity padding must not pass verification"
    );
}

#[test]
fn bordered_verification_requires_rank_plus_one_factors() {
    let vector = vec![true, false];
    let action = transvection_matrix(&vector, 1);
    for vectors in [vec![vector.clone()], vec![vector.clone(); 3]] {
        assert_bordered_replay(&action, 1, &vectors);
        assert!(
            !verifies_bordered_decomposition(&action, 1, 1, &vectors),
            "verification must reject the wrong factor count"
        );
    }
}

#[test]
fn bordered_swap_layers_do_not_search() {
    for qubit_count in [2, 4, 8, 16, 32, 40] {
        let mut clifford = CliffordUnitary::identity(qubit_count);
        for qubit in (0..qubit_count).step_by(2) {
            clifford.left_mul_swap(qubit, qubit + 1);
        }
        let action = action_matrix(&clifford);
        SPAN_VECTOR_VISITS.set(0);
        let vectors = minimal_decomposition(&action, qubit_count);
        assert_eq!(
            SPAN_VECTOR_VISITS.get(),
            0,
            "the bordered SWAP path must not enumerate a span"
        );
        assert!(
            verifies_bordered_decomposition(&action, qubit_count, qubit_count, &vectors),
            "the simplex must supply a valid rank plus one decomposition"
        );
        assert_bordered_replay(&action, qubit_count, &vectors);
    }
}

#[test]
fn bordered_verification_rejects_invalid_candidates() {
    let action = class_a_action();
    let (basis, rank, core) = residue_core(&action, 2);
    let candidate = bordered_decomposition(&core, &basis).expect("class A must have a constructed candidate");
    assert!(verifies_bordered_decomposition(&action, 2, rank, &candidate));
    let mut reversed = candidate.clone();
    reversed.reverse();
    let mut zero_factor = candidate.clone();
    zero_factor[0].fill(false);
    let mut short_vector = candidate.clone();
    short_vector[0].pop();
    let mut long_vector = candidate.clone();
    long_vector[0].push(true);
    for (vectors, label) in [
        (candidate[..rank].to_vec(), "wrong factor count"),
        (zero_factor, "zero factor"),
        (short_vector, "short vector"),
        (long_vector, "long vector"),
        (reversed, "reversed factors"),
    ] {
        assert!(
            !verifies_bordered_decomposition(&action, 2, rank, &vectors),
            "invalid candidate passed verification: {label}"
        );
    }
}

#[test]
fn bordered_fallback_recovers_from_candidate_failure() {
    let action = class_a_action();
    let (basis, rank, core) = residue_core(&action, 2);
    let mut reversed = bordered_decomposition(&core, &basis).expect("class A must have a constructed candidate");
    reversed.reverse();
    for candidate in [None, Some(Vec::new()), Some(reversed)] {
        SPAN_VECTOR_VISITS.set(0);
        let vectors = verified_decomposition_or_search(&action, 2, &basis, candidate);
        assert!(
            SPAN_VECTOR_VISITS.get() > 0,
            "a failed candidate must use the retained search"
        );
        assert!(
            verifies_bordered_decomposition(&action, 2, rank, &vectors),
            "the retained search must recover a valid decomposition"
        );
        assert_bordered_replay(&action, 2, &vectors);
        let factors: Vec<_> = vectors.iter().map(|vector| vector_to_pauli(vector, 2)).collect();
        for mask in 0..16 {
            let clifford = class_a_with_signs(mask);
            let correction = final_pauli_correction(&clifford, &factors);
            assert_eq!(correction.xyz_phase_exponent(), 0);
            let mut rebuilt = CliffordUnitary::identity(2);
            for factor in &factors {
                rebuilt.left_mul_pauli_exp(factor);
            }
            rebuilt.left_mul_pauli(&correction);
            assert_eq!(rebuilt, clifford, "the retained list needs its own final Pauli");
        }
    }
}

#[test]
fn bordered_seeded_larger_actions_replay() {
    for qubit_count in [4, 6, 8, 12, 16, 32] {
        for seed in 0..32 {
            let mut generator = StdRng::seed_from_u64(seed);
            let conjugator = CliffordUnitary::random(qubit_count, &mut generator);
            let action = action_matrix(&conjugator);
            let (basis, rank, core) = residue_core(&action, qubit_count);
            let vectors = bordered_decomposition(&core, &basis).expect("a sampled core must produce a bordered family");
            assert_eq!(
                vectors.len(),
                rank + 1,
                "a bordered family must have rank plus one vectors"
            );
            assert_bordered_replay(&action, qubit_count, &vectors);

            let mut swap = CliffordUnitary::identity(qubit_count);
            for first in (0..qubit_count).step_by(2) {
                let support: IndexSet = [first, first + 1].into_iter().collect();
                for pauli in [
                    SparsePauli::from_bits(support.clone(), IndexSet::new(), 0),
                    SparsePauli::from_bits(IndexSet::new(), support.clone(), 0),
                    SparsePauli::from_bits(support.clone(), support, 2),
                ] {
                    swap.left_mul_pauli_exp(&conjugator.image(&pauli));
                }
            }
            let action = action_matrix(&swap);
            SPAN_VECTOR_VISITS.set(0);
            let vectors = minimal_decomposition(&action, qubit_count);
            assert_eq!(
                SPAN_VECTOR_VISITS.get(),
                0,
                "a conjugated SWAP layer must use the construction"
            );
            assert!(verifies_bordered_decomposition(
                &action,
                qubit_count,
                qubit_count,
                &vectors
            ));
            assert_bordered_replay(&action, qubit_count, &vectors);
        }
    }
}
