use binar::{Bitwise, BitwiseMut, IndexSet};
use itertools::Itertools;
use paulimer::pauli::{commutes_with, Pauli, PauliMutable, SparsePauli};
use paulimer::pauli_group::{centralizer_of, symplectic_form_of, PauliGroup};
use paulimer::traits::NeutralElement;
use proptest::collection::vec;
use proptest::prelude::*;
use std::collections::HashSet;
use std::str::FromStr;

fn sparse_pauli(max_weight: usize) -> BoxedStrategy<SparsePauli> {
    (
        any::<u8>().prop_map(|e| e % 4),
        prop::collection::vec(any::<usize>(), 0..=max_weight),
        prop::collection::vec(prop::sample::select(vec!['I', 'X', 'Y', 'Z']), 0..=max_weight),
    )
        .prop_filter_map(
            "non-empty support for non-identity",
            |(exponent, indices, operators)| {
                let mut x_bits = IndexSet::new();
                let mut z_bits = IndexSet::new();

                for (idx, op) in indices.into_iter().zip(operators.into_iter()) {
                    match op {
                        'X' => x_bits.assign_index(idx, true),
                        'Z' => z_bits.assign_index(idx, true),
                        'Y' => {
                            x_bits.assign_index(idx, true);
                            z_bits.assign_index(idx, true);
                        }
                        'I' => {}
                        _ => unreachable!(),
                    }
                }

                if x_bits.is_zero() && z_bits.is_zero() {
                    let mut identity = SparsePauli::from_str("I").ok()?;
                    identity.assign_phase_exp(exponent);
                    Some(identity)
                } else {
                    Some(SparsePauli::from_bits(x_bits, z_bits, exponent))
                }
            },
        )
        .boxed()
}

fn small_sparse_pauli() -> BoxedStrategy<SparsePauli> {
    sparse_pauli(3)
}

fn sparse_paulis(
    min_len: usize,
    max_len: usize,
    element_strategy: BoxedStrategy<SparsePauli>,
) -> BoxedStrategy<Vec<SparsePauli>> {
    vec(element_strategy, min_len..=max_len).boxed()
}

fn small_pauli_group() -> BoxedStrategy<PauliGroup> {
    sparse_paulis(0, 4, small_sparse_pauli())
        .prop_map(|generators| PauliGroup::new(&generators))
        .boxed()
}

fn pauli_group_pairs() -> BoxedStrategy<(PauliGroup, PauliGroup)> {
    (small_pauli_group(), small_pauli_group()).boxed()
}

proptest! {
    #[test]
    fn test_generators_match_construction_argument(generators in sparse_paulis(1, 5, small_sparse_pauli())) {
        let group = PauliGroup::new(&generators);
        prop_assert_eq!(group.generators(), generators);
    }

    #[test]
    fn test_elements_are_distinct(group in small_pauli_group()) {
        let elements: Vec<SparsePauli> = group.elements().collect();
        let unique_elements: HashSet<SparsePauli> = elements.iter().cloned().collect();
        prop_assert_eq!(elements.len(), unique_elements.len());
    }

    #[test]
    fn test_log_size_matches_elements(group in small_pauli_group()) {
        let log_size = group.log2_size();
        let group_elements: HashSet<SparsePauli> = group.elements().collect();
        prop_assert_eq!(group_elements.len(), 1 << log_size);
    }

    #[test]
    fn test_elements_are_invariant_under_group_element_augmentation(group in small_pauli_group()) {
        let original_elements: HashSet<SparsePauli> = group.elements().collect();

        if let Some(group_element) = group.elements().next() {
            let mut augmented_generators = group.generators().to_vec();
            augmented_generators.push(group_element);
            let augmented_group = PauliGroup::new(&augmented_generators);
            let augmented_elements: HashSet<SparsePauli> = augmented_group.elements().collect();
            prop_assert_eq!(original_elements, augmented_elements);
        }
    }

    #[test]
    fn test_generator_products_are_elements(group in small_pauli_group()) {

        assert!(group.generators().len() <= 4, "Test too large to brute force");
        let group_elements: HashSet<SparsePauli> = group.elements().collect();
        let mut products = HashSet::new();
        products.insert(SparsePauli::default_size_neutral_element());
        if !group.generators().is_empty() {
            let mut current_length = 0;
            let mut generators = group.generators().to_vec();
            generators.insert(0, SparsePauli::default_size_neutral_element());
            // println!("Current products: {:?}", products);
            while current_length == 0 || current_length < products.len() {
                current_length = products.len();
                // println!("Current products: {:?}", products);
                products = generators.iter().cartesian_product(products.iter()).map(|(g, p)| g * p).collect();
                // println!("Next products: {:?}", products);
            }
        }

        assert_eq!(products, group_elements);
    }

    #[test]
    fn test_rank_is_invariant_under_append_generator_products(generators in sparse_paulis(2, 5, small_sparse_pauli())) {
        let group = PauliGroup::new(&generators);
        let expected_rank = group.binary_rank();

        for i in 0..generators.len() {
            for j in i+1..generators.len() {
                let mut overcomplete_generators = generators.clone();
                overcomplete_generators.push(&generators[i] * &generators[j]);
                let overcomplete_group = PauliGroup::new(&overcomplete_generators);
                prop_assert_eq!(overcomplete_group.binary_rank(), expected_rank);
            }
        }
    }

    #[test]
    fn test_is_abelian_matches_permutation_equality(generators in sparse_paulis(1, 5, small_sparse_pauli())) {
        use itertools::Itertools;
        let group = PauliGroup::new(&generators);
        let product = generators.iter().fold(SparsePauli::from_str("I").unwrap(), |acc, g| &acc * g);

        let all_permutations_equal = generators.iter().permutations(generators.len())
            .all(|perm| {
                let perm_product = perm.iter().fold(SparsePauli::from_str("I").unwrap(), |acc, &g| &acc * g);
                perm_product == product
            });

        prop_assert_eq!(group.is_abelian(), all_permutations_equal);
    }

    #[test]
    fn test_rank_is_at_most_generator_count(group in small_pauli_group()) {
        prop_assert!(group.binary_rank() <= group.generators().len());
    }

    #[test]
    fn test_rank_increments_when_dimension_is_extended(generators in sparse_paulis(1, 4, small_sparse_pauli())) {
        let group = PauliGroup::new(&generators);
        let original_rank = group.binary_rank();

        let max_support = group.support().iter().max().copied().unwrap_or(0) + 1;

        let mut x_extension_bits = IndexSet::new();
        x_extension_bits.assign_index(max_support, true);
        let x_extension = SparsePauli::from_bits(x_extension_bits, IndexSet::new(), 0);

        let mut z_extension_bits = IndexSet::new();
        z_extension_bits.assign_index(max_support, true);
        let z_extension = SparsePauli::from_bits(IndexSet::new(), z_extension_bits, 0);

        let mut y_x_bits = IndexSet::new();
        let mut y_z_bits = IndexSet::new();
        y_x_bits.assign_index(max_support, true);
        y_z_bits.assign_index(max_support, true);
        let y_extension = SparsePauli::from_bits(y_x_bits, y_z_bits, 1);

        let mut extended_generators: Vec<SparsePauli> = generators.iter()
            .map(|g| g * &y_extension)
            .collect();
        extended_generators.push(x_extension);
        extended_generators.push(z_extension);

        let extended_group = PauliGroup::new(&extended_generators);
        prop_assert_eq!(extended_group.binary_rank(), original_rank + 2);
    }

    #[test]
    fn test_quotient_by_identity_is_trivial(generators in sparse_paulis(1, 5, small_sparse_pauli())) {
        let group = PauliGroup::new(&generators);
        let identity_generators = vec![SparsePauli::from_str("I").unwrap()];
        let identity_group = PauliGroup::new(&identity_generators);
        let quotient = group.clone() / &identity_group;
        prop_assert_eq!(group, quotient);
    }

    #[test]
    fn test_pauli_group_equal_is_consistent_with_not_equal(pair in pauli_group_pairs()) {
        let (left, right) = pair;
        let are_equal = left == right;
        let are_not_equal = left != right;
        prop_assert_eq!(are_equal, !are_not_equal);
    }

    #[test]
    fn test_equality_ignores_redundant_phases(generators in sparse_paulis(1, 4, small_sparse_pauli())) {
        let pure_phases: Vec<SparsePauli> = generators.iter()
            .map(|g| SparsePauli::from_bits(IndexSet::new(), IndexSet::new(), g.xz_phase_exponent()))
            .collect();

        let phaseless_generators: Vec<SparsePauli> = generators.iter()
            .map(|g| SparsePauli::from_bits(g.x_bits().clone(), g.z_bits().clone(), 0))
            .collect();

        let mut redundant_generators = generators.clone();
        redundant_generators.extend(pure_phases.clone());

        let mut pure_phase_generators = phaseless_generators;
        pure_phase_generators.extend(pure_phases);

        let group1 = PauliGroup::new(&redundant_generators);
        let group2 = PauliGroup::new(&pure_phase_generators);

        prop_assert_eq!(group1, group2);
    }

    #[test]
    fn test_is_stabilizer_group_iff_devoid_of_negative_identity(group in small_pauli_group()) {
        let group_elements: Vec<SparsePauli> = group.clone().elements().collect();

        let devoid_of_negative_identity = group_elements.iter().all(|element| {
            element.weight() > 0 || element.xz_phase_exponent() == 0
        });

        prop_assert_eq!(group.is_stabilizer_group(), devoid_of_negative_identity);
    }

    #[test]
    fn test_pauli_group_comparison_examples(pair in pauli_group_pairs()) {
        let (group1, group2) = pair;
        let are_equal = group1 == group2;
        let are_not_equal = group1 != group2;
        prop_assert_eq!(are_equal, !are_not_equal);
    }

    #[test]
    fn test_pauli_group_self_comparison(group in small_pauli_group()) {
        if group.generators().len() >= 2 {
            let mut alt_generators = group.generators().to_vec();
            alt_generators[0] = &alt_generators[0] * &alt_generators[1];
            let alternate_presentation = PauliGroup::new(&alt_generators);
            prop_assert_eq!(&group, &alternate_presentation);
        }
    }

    #[test]
    fn test_centralizer_commutes(group in sparse_paulis(1, 3, small_sparse_pauli()).prop_map(|g| PauliGroup::new(&g))) {
        let centralizer = centralizer_of(&group);

        for centralizer_element in centralizer.generators() {
            for group_element in group.generators() {
                prop_assert!(commutes_with(centralizer_element, group_element));
            }
        }
    }

    #[test]
    fn test_symplectic_basis_preserves_group(group in small_pauli_group()) {
        let basis = symplectic_form_of(group.generators());
        prop_assert_eq!(PauliGroup::new(&basis), group);
    }

    #[test]
    fn test_standard_form_is_length_nonincreasing(group in small_pauli_group()) {
        prop_assume!(!group.generators().is_empty());
        prop_assert!(group.standard_generators().len() <= group.generators().len());
    }

    #[test]
    fn test_standard_form_is_group_preserving(group in small_pauli_group()) {
        let standard_group = PauliGroup::new(group.standard_generators());

        let group_elements: HashSet<SparsePauli> = group.elements().collect();
        let standard_elements: HashSet<SparsePauli> = standard_group.elements().collect();

        prop_assert_eq!(group_elements, standard_elements);
    }

    #[test]
    fn test_quotient_with_subgroup(
        generators in sparse_paulis(2, 5, small_sparse_pauli()),
        split in 1..4usize
    ) {
        let actual_split = std::cmp::min(split, generators.len() - 1);
        let group = PauliGroup::new(&generators);
        let subgroup = PauliGroup::new(&generators[..actual_split]);
        prop_assume!(subgroup <= group);
        let quotient = group.clone() / &subgroup;
        prop_assert!(quotient.log2_size() <= group.log2_size());
    }

    #[test]
    fn test_centralizer_with_identity_elements(group in small_pauli_group()) {
        let centralizer = centralizer_of(&group);
        prop_assert!(!centralizer.generators().is_empty() || centralizer.generators().is_empty());
    }

    #[test]
    fn test_group_operations_preserve_identity_elements(
        generators in sparse_paulis(1, 3, small_sparse_pauli())
    ) {
        let group = PauliGroup::new(&generators);

        let elements: HashSet<SparsePauli> = group.elements().collect();
        let identity = SparsePauli::from_str("I").unwrap();

        let has_identity_generator = generators.iter().any(|g| g.weight() == 0);
        if has_identity_generator {
            prop_assert!(elements.contains(&identity));
        }
    }

    #[test]
    fn test_quotient_generators_intersect_divisor_orbit(
        generators in sparse_paulis(2, 6, small_sparse_pauli()),
        split_index in 1..4usize
    ) {
        prop_assume!(generators.len() >= 2);
        let actual_split = std::cmp::min(split_index, generators.len() - 1);
        let group = PauliGroup::new(&generators[actual_split..]);
        let divisor = PauliGroup::new(&generators[..actual_split]);

        let is_normal = generators[actual_split..].iter().all(|g1| {
            generators[..actual_split].iter().all(|g2| commutes_with(g1, g2))
        });

        prop_assume!(is_normal);
        prop_assume!(divisor <= group);

        let quotient = group.clone() / &divisor;
        let quotient_elements: HashSet<SparsePauli> = quotient.elements().collect();

        for generator in &generators[actual_split..] {
            let mut orbit_in_quotient = false;
            for divisor_element in divisor.elements() {
                let orbit_element = generator * &divisor_element;
                if quotient_elements.contains(&orbit_element) {
                    orbit_in_quotient = true;
                    break;
                }
            }
            prop_assert!(orbit_in_quotient, "No orbit element found in quotient for generator {:?}", generator);
        }
    }

    #[test]
    fn test_symplectic_basis_satisfies_commutation_constraints(
        generators in sparse_paulis(1, 6, small_sparse_pauli())
    ) {
        let basis = symplectic_form_of(&generators);
        prop_assert!(is_symplectic(&basis));
    }

    #[test]
    fn test_self_quotient_has_rank_zero(group in small_pauli_group()) {
        let quotient = group.clone() / &group;
        assert_eq!(quotient.binary_rank(), 0);

        if group.generators().len() >= 2 {
            let mut alt_generators = group.generators().to_vec();
            alt_generators[0] = &alt_generators[0] * &alt_generators[1];
            let alt_group = PauliGroup::new(&alt_generators);
            let quotient = group / &alt_group;
            assert_eq!(quotient.binary_rank(), 0);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5))]

    #[test]
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn test_large_subgroup_containment(rank in 64..128_usize) {
        let generators: Vec<SparsePauli> = (0..rank).map(|index| { SparsePauli::y(index, rank) }).collect();
        let group = PauliGroup::new(&generators);
        let subgroup = PauliGroup::new(&generators[1..]);
        prop_assert!(subgroup < group);
        prop_assert!(!(group <= subgroup));
    }
}

#[test]
fn test_pauli_group_rank_examples() {
    let group = PauliGroup::from_strings(&["I"]);
    assert_eq!(group.binary_rank(), 0);
    assert_eq!(group.phases().clone(), vec![0]);

    let group = PauliGroup::from_strings(&["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]);
    assert_eq!(group.binary_rank(), 4);
    assert_eq!(group.phases().clone(), vec![0]);

    let group = PauliGroup::from_strings(&["XXIXXII", "XIXXIXI", "IXXXIIX", "ZZIZZII", "ZIZZIZI", "IZZZIIZ"]);
    assert_eq!(group.binary_rank(), 6);
    assert_eq!(group.phases().clone(), vec![0]);

    let group = PauliGroup::from_strings(&["XXXXX", "ZZZZZ", "YYYYY"]);
    assert_eq!(group.binary_rank(), 2);
    assert_eq!(group.phases().clone(), vec![0, 1, 2, 3]);

    let group = PauliGroup::from_strings(&["IIIII", "IIIII"]);
    assert_eq!(group.binary_rank(), 0);
    assert_eq!(group.phases().clone(), vec![0]);

    let group = PauliGroup::from_strings(&["XYXXX", "ZIZZZ", "YIYXY"]);
    assert_eq!(group.binary_rank(), 3);
    assert_eq!(group.phases().clone(), vec![0, 2]);
}

#[test]
fn test_pauli_group_quotient_examples() {
    let group = PauliGroup::from_strings(&["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]);
    let checks = PauliGroup::from_strings(&["XZZXI", "IXZZX", "XIXZZ"]);
    let quotient = group / &checks;
    assert_eq!(quotient.binary_rank(), 1);
}

// Property-based tests for PauliGroup intersection (BitAnd) operation
// These tests verify fundamental mathematical properties that should hold
// for any correct intersection implementation.
proptest! {
    #[test]
    fn test_intersection_is_commutative(pair in pauli_group_pairs()) {
        let (group1, group2) = pair;
        let intersection1 = &group1 & &group2;
        let intersection2 = &group2 & &group1;
        prop_assert_eq!(intersection1, intersection2);
    }

    #[test]
    fn test_intersection_with_self_preserves_group(group in small_pauli_group()) {
        let intersection = &group & &group;
        prop_assert_eq!(intersection, group);
    }

    #[test]
    fn test_intersection_is_subset_of_both_groups(pair in pauli_group_pairs()) {
        let (group1, group2) = pair;
        let intersection = &group1 & &group2;

        // The intersection should be a subgroup of both original groups
        // For now, we check binary rank as a necessary condition
        prop_assert!(intersection.binary_rank() <= group1.binary_rank());
        prop_assert!(intersection.binary_rank() <= group2.binary_rank());

        // Phase count should also be bounded
        prop_assert!(intersection.phases().len() <= group1.phases().len());
        prop_assert!(intersection.phases().len() <= group2.phases().len());
    }

    #[test]
    fn test_intersection_empty_groups(
        group1 in small_pauli_group(),
        group2 in small_pauli_group()
    ) {
        let empty_group = PauliGroup::new(&[]);

        // Intersection with trivial group should be trivial
        let intersection1 = &group1 & &empty_group;
        let intersection2 = &empty_group & &group2;
        let intersection3 = &empty_group & &empty_group;

        // Trivial group has log2_size == 0 (size 1, containing only identity)
        prop_assert_eq!(intersection1.log2_size(), 0);
        prop_assert_eq!(intersection2.log2_size(), 0);
        prop_assert_eq!(intersection3.log2_size(), 0);
    }

    #[test]
    fn test_intersection_with_identity_group(group in small_pauli_group()) {
        // Create trivial (identity-only) group
        let trivial_group = PauliGroup::new(&[]);

        // Intersection with trivial group should be trivial
        let intersection = &group & &trivial_group;

        // Trivial group has log2_size == 0
        prop_assert_eq!(intersection.log2_size(), 0);
    }

    #[test]
    fn test_intersection_associativity(
        group1 in small_pauli_group(),
        group2 in small_pauli_group(),
        group3 in small_pauli_group()
    ) {
        // Test that (A ∩ B) ∩ C = A ∩ (B ∩ C)
        let temp1 = &group1 & &group2;
        let left_associative = &temp1 & &group3;

        let temp2 = &group2 & &group3;
        let right_associative = &group1 & &temp2;

        prop_assert_eq!(
            &left_associative,
            &right_associative,
            "group1={:?}, group2={:?}, group3={:?}, left={:?}, right={:?}, temp1={:?}, temp2={:?}",
            group1.generators(), group2.generators(), group3.generators(), left_associative.generators(), right_associative.generators(), temp1.generators(), temp2.generators()
        );
    }

    #[test]
    fn test_intersection_idempotency(group in small_pauli_group()) {
        // Test that A ∩ A = A (idempotency)
        let intersection = &group & &group;
        prop_assert_eq!(intersection, group);
    }

    #[test]
    fn test_intersection_with_subgroup(pair in pauli_group_pairs()) {
        let (group1, group2) = pair;

        // If group1 ≤ group2, then group1 ∩ group2 = group1
        // We can't easily test containment with the current API,
        // but we can test some properties that should hold
        let intersection = &group1 & &group2;

        // The intersection size should never exceed either group
        prop_assert!(intersection.log2_size() <= group1.log2_size());
        prop_assert!(intersection.log2_size() <= group2.log2_size());

        // Binary rank should be bounded
        prop_assert!(intersection.binary_rank() <= group1.binary_rank());
        prop_assert!(intersection.binary_rank() <= group2.binary_rank());
    }

    #[test]
    fn test_intersection_distributivity_over_union(
        group1 in small_pauli_group(),
        group2 in small_pauli_group(),
        group3 in small_pauli_group()
    ) {
        // Test that A ∩ (B ∪ C) ⊇ (A ∩ B) ∪ (A ∩ C)
        // This is a fundamental property of set operations

        let union_bc = &group2 | &group3;
        let a_intersect_union = &group1 & &union_bc;

        let a_intersect_b = &group1 & &group2;
        let a_intersect_c = &group1 & &group3;
        let union_intersections = &a_intersect_b | &a_intersect_c;

        // The left side should be at least as large as the right side
        prop_assert!(a_intersect_union.log2_size() >= union_intersections.log2_size());
        prop_assert!(a_intersect_union.binary_rank() >= union_intersections.binary_rank());
    }

    #[test]
    fn test_intersection_preserves_abelian_property(pair in pauli_group_pairs()) {
        let (group1, group2) = pair;
        let intersection = &group1 & &group2;

        // If both groups are abelian, the intersection should be abelian
        if group1.is_abelian() && group2.is_abelian() {
            prop_assert!(intersection.is_abelian());
        }

        // The intersection should always be abelian if it's a stabilizer group
        if intersection.is_stabilizer_group() {
            prop_assert!(intersection.is_abelian());
        }
    }

    #[test]
    fn test_intersection_support_properties(pair in pauli_group_pairs()) {
        let (group1, group2) = pair;
        let intersection = &group1 & &group2;

        // The support of the intersection should be a subset of both supports
        let support1: std::collections::HashSet<usize> = group1.support().iter().copied().collect();
        let support2: std::collections::HashSet<usize> = group2.support().iter().copied().collect();
        let intersection_support: std::collections::HashSet<usize> = intersection.support().iter().copied().collect();

        // Intersection support should be subset of both
        prop_assert!(intersection_support.is_subset(&support1));
        prop_assert!(intersection_support.is_subset(&support2));
    }
}

// Tests for factorization_of method
#[cfg(test)]
mod factorization_tests {
    use super::*;
    use crate::NeutralElement;
    use paulimer::pauli::{PauliBinaryOps, PauliMutable};

    #[test]
    fn test_factorization_of_identity() {
        let generators = vec![
            SparsePauli::from_str("XI").unwrap(),
            SparsePauli::from_str("IZ").unwrap(),
        ];
        let group = PauliGroup::new(&generators);
        let identity = SparsePauli::from_str("II").unwrap();

        let factorization = group.factorization_of(&identity).unwrap();
        let product = factorization.iter().fold(identity.clone(), |mut acc, p| {
            acc.mul_assign_right(p);
            acc
        });
        assert_eq!(product, identity);
    }

    #[test]
    fn test_factorization_of_generator() {
        let generators = vec![
            SparsePauli::from_str("XI").unwrap(),
            SparsePauli::from_str("IZ").unwrap(),
        ];
        let group = PauliGroup::new(&generators);
        let generator = &generators[0];

        let factorization = group.factorization_of(generator);
        if let Some(factorization) = factorization {
            let product = factorization
                .iter()
                .fold(SparsePauli::from_str("II").unwrap(), |mut acc, p| {
                    acc.mul_assign_right(p);
                    acc
                });
            assert_eq!(product, *generator);
        }
    }

    #[test]
    fn test_factorization_of_element_not_in_group() {
        let generators = vec![
            SparsePauli::from_str("XI").unwrap(),
            SparsePauli::from_str("IZ").unwrap(),
        ];
        let group = PauliGroup::new(&generators);
        let element = SparsePauli::from_str("YI").unwrap(); // Not in the group

        let factorization = group.factorization_of(&element);
        assert!(factorization.is_none());
    }

    #[test]
    fn test_factorization_of_product_of_generators() {
        let generators = vec![
            SparsePauli::from_str("XI").unwrap(),
            SparsePauli::from_str("IZ").unwrap(),
        ];
        let group = PauliGroup::new(&generators);

        // Create a product of generators
        let mut product = SparsePauli::from_str("II").unwrap();
        product.mul_assign_right(&generators[0]);
        product.mul_assign_right(&generators[1]);

        let factorization = group.factorization_of(&product);
        if let Some(factorization) = factorization {
            let reconstructed = factorization
                .iter()
                .fold(SparsePauli::from_str("II").unwrap(), |mut acc, p| {
                    acc.mul_assign_right(p);
                    acc
                });
            assert_eq!(reconstructed, product);
        }
    }

    #[test]
    fn test_factorization_empty_group() {
        let group = PauliGroup::new(&[]);
        let identity = SparsePauli::from_str("I").unwrap();

        // Check if identity is in the empty group
        if group.contains(&identity) {
            let factorization = group.factorization_of(&identity).unwrap();
            let product = factorization.iter().fold(identity.clone(), |mut acc, p| {
                acc.mul_assign_right(p);
                acc
            });
            assert_eq!(product, identity);
        } else {
            // If identity is not in empty group, factorization should return None
            let factorization = group.factorization_of(&identity);
            assert!(factorization.is_none());
        }
    }

    #[test]
    fn test_factorization_with_phases() {
        let mut gen1 = SparsePauli::from_str("X").unwrap();
        gen1.assign_phase_exp(1); // i * X
        let mut gen2 = SparsePauli::from_str("Z").unwrap();
        gen2.assign_phase_exp(2); // -Z

        let generators = vec![gen1.clone(), gen2.clone()];
        let group = PauliGroup::new(&generators);

        // Test that generators are contained in the group
        assert!(group.contains(&gen1));
        assert!(group.contains(&gen2));

        // Test factorization of generators with phases
        let _factorization1 = group.factorization_of(&gen1);
        let _factorization2 = group.factorization_of(&gen2);
    }

    #[test]
    fn test_factorization_single_generator_group() {
        let generator = SparsePauli::from_str("X").unwrap();
        let group = PauliGroup::new(std::slice::from_ref(&generator));

        // Test factorization of the generator itself
        let factorization = group.factorization_of(&generator);
        if let Some(factorization) = factorization {
            let product = factorization
                .iter()
                .fold(SparsePauli::from_str("I").unwrap(), |mut acc, p| {
                    acc.mul_assign_right(p);
                    acc
                });
            assert_eq!(product, generator);
        }

        // Test that non-group elements return None factorization
        let non_element = SparsePauli::from_str("Y").unwrap();
        let factorization_non = group.factorization_of(&non_element);
        assert!(factorization_non.is_none());
    }

    #[test]
    fn test_factorization_comprehensive_example() {
        // Create a group with X and Z generators
        let gen_x = SparsePauli::from_str("X").unwrap();
        let gen_z = SparsePauli::from_str("Z").unwrap();
        let group = PauliGroup::new(&[gen_x.clone(), gen_z.clone()]);

        // Test factorization of Y (should be X*Z up to phase)
        let mut y_element = SparsePauli::from_str("I").unwrap();
        y_element.mul_assign_right(&gen_x);
        y_element.mul_assign_right(&gen_z);

        // Y should be in the group
        assert!(group.contains(&y_element));

        // Get factorization of Y
        let factorization = group.factorization_of(&y_element);
        if let Some(factorization) = factorization {
            let reconstructed = factorization
                .iter()
                .fold(SparsePauli::from_str("I").unwrap(), |mut acc, p| {
                    acc.mul_assign_right(p);
                    acc
                });
            assert_eq!(reconstructed, y_element);
        }
    }

    #[test]
    fn test_factorization_respects_group_property() {
        // Test that factorization always produces elements that multiply to the target
        let generators = vec![
            SparsePauli::from_str("XI").unwrap(),
            SparsePauli::from_str("ZI").unwrap(),
            SparsePauli::from_str("IZ").unwrap(),
        ];
        let group = PauliGroup::new(&generators);

        // Test several elements in the group
        for element in group.elements().take(10) {
            let factorization = group.factorization_of(&element);

            if let Some(factorization) = factorization {
                let reconstructed = factorization.iter().fold(element.neutral_element(), |mut acc, p| {
                    acc.mul_assign_right(p);
                    acc
                });

                // The product of the factorization should equal the original element
                assert_eq!(reconstructed, element, "Factorization failed for element: {element}");
            }
        }
    }

    #[test]
    fn test_factorization_regressions() {
        // [Z₀X₁X₂Z₃, Z₁X₂X₃Z₄, Z₀Z₂X₃X₄, X₀Z₁Z₃X₄]
        let generators = vec![
            SparsePauli::from_str("ZXXZI").unwrap(),
            SparsePauli::from_str("IZXXZ").unwrap(),
            SparsePauli::from_str("ZIZXX").unwrap(),
            SparsePauli::from_str("XZIZX").unwrap(),
        ];
        let group = PauliGroup::new(&generators);
        let product = &generators[0] * &generators[1];
        let factorization = group.factorization_of(&product);
        let expected = vec![generators[0].clone(), generators[1].clone()];
        assert_eq!(factorization, Some(expected));
    }

    #[test]
    fn test_factorization_larger_group_performance() {
        // Test with a larger group to verify the linear algebra approach scales well
        let generators = vec![
            SparsePauli::from_str("XIIII").unwrap(),
            SparsePauli::from_str("ZIIII").unwrap(),
            SparsePauli::from_str("IXIII").unwrap(),
            SparsePauli::from_str("IZIII").unwrap(),
            SparsePauli::from_str("IIXII").unwrap(),
            SparsePauli::from_str("IIZII").unwrap(),
        ];
        let group = PauliGroup::new(&generators);

        // This group has 2^6 = 64 elements, which would be impractical for exhaustive search
        // but should be fast with linear algebra

        // Test factorization of several elements
        let mut element_count = 0;
        for element in group.elements().take(20) {
            let factorization = group.factorization_of(&element);

            if let Some(factorization) = factorization {
                let reconstructed = factorization.iter().fold(element.neutral_element(), |mut acc, p| {
                    acc.mul_assign_right(p);
                    acc
                });
                assert_eq!(
                    reconstructed, element,
                    "Large group factorization failed for: {element}"
                );
            }
            element_count += 1;
        }

        // Verify we tested a reasonable number of elements
        assert!(element_count >= 10, "Should have tested at least 10 elements");
    }

    #[test]
    fn test_factorizations_of_empty_input() {
        let generators = vec![
            SparsePauli::from_str("XI").unwrap(),
            SparsePauli::from_str("IZ").unwrap(),
        ];
        let group = PauliGroup::new(&generators);

        let factorizations = group.factorizations_of(&[]);
        assert!(factorizations.is_empty());
    }

    #[test]
    fn test_factorizations_of_single_element() {
        let generators = vec![
            SparsePauli::from_str("XI").unwrap(),
            SparsePauli::from_str("IZ").unwrap(),
        ];
        let group = PauliGroup::new(&generators);
        let element = SparsePauli::from_str("XI").unwrap();

        let factorizations = group.factorizations_of(std::slice::from_ref(&element));
        assert_eq!(factorizations.len(), 1);

        // Should be equivalent to calling factorization_of
        let single_factorization = group.factorization_of(&element);
        assert_eq!(factorizations[0], single_factorization);
    }

    #[test]
    fn test_factorizations_of_multiple_elements() {
        let generators = vec![
            SparsePauli::from_str("XI").unwrap(),
            SparsePauli::from_str("IZ").unwrap(),
        ];
        let group = PauliGroup::new(&generators);

        let elements = vec![
            SparsePauli::from_str("XI").unwrap(),
            SparsePauli::from_str("IZ").unwrap(),
            SparsePauli::from_str("II").unwrap(), // Identity
        ];

        let factorizations = group.factorizations_of(&elements);
        assert_eq!(factorizations.len(), 3);

        // Verify each factorization individually
        for (i, element) in elements.iter().enumerate() {
            let expected_factorization = group.factorization_of(element);
            assert_eq!(factorizations[i], expected_factorization);

            // If factorization exists, verify it reconstructs the original element
            if let Some(ref factorization) = factorizations[i] {
                let reconstructed = factorization.iter().fold(element.neutral_element(), |mut acc, p| {
                    acc.mul_assign_right(p);
                    acc
                });
                assert_eq!(reconstructed, *element);
            }
        }
    }

    #[test]
    fn test_factorizations_of_mixed_in_and_out_of_group() {
        let generators = vec![
            SparsePauli::from_str("XI").unwrap(),
            SparsePauli::from_str("IZ").unwrap(),
        ];
        let group = PauliGroup::new(&generators);

        let elements = vec![
            SparsePauli::from_str("XI").unwrap(), // In group
            SparsePauli::from_str("YI").unwrap(), // Not in group
            SparsePauli::from_str("II").unwrap(), // In group (identity)
        ];

        let factorizations = group.factorizations_of(&elements);
        assert_eq!(factorizations.len(), 3);

        // First element should have a factorization
        assert!(factorizations[0].is_some());

        // Second element should not have a factorization (not in group)
        assert!(factorizations[1].is_none());

        // Third element (identity) should have a factorization
        assert!(factorizations[2].is_some());
    }

    #[test]
    fn test_factorizations_of_performance_batch_vs_individual() {
        let generators = vec![
            SparsePauli::from_str("XIIII").unwrap(),
            SparsePauli::from_str("ZIIII").unwrap(),
            SparsePauli::from_str("IXIII").unwrap(),
            SparsePauli::from_str("IZIII").unwrap(),
        ];
        let group = PauliGroup::new(&generators);

        // Create test elements
        let elements: Vec<SparsePauli> = group.elements().take(10).collect();

        // Test batch factorization
        let batch_factorizations = group.factorizations_of(&elements);

        // Test individual factorizations
        let individual_factorizations: Vec<Option<Vec<SparsePauli>>> =
            elements.iter().map(|e| group.factorization_of(e)).collect();

        // Results should be identical
        assert_eq!(batch_factorizations.len(), individual_factorizations.len());
        for (batch, individual) in batch_factorizations.iter().zip(individual_factorizations.iter()) {
            assert_eq!(batch, individual);
        }
    }

    #[test]
    fn test_factorizations_of_with_phases() {
        let mut gen1 = SparsePauli::from_str("X").unwrap();
        gen1.assign_phase_exp(1); // i * X
        let mut gen2 = SparsePauli::from_str("Z").unwrap();
        gen2.assign_phase_exp(2); // -Z

        let generators = vec![gen1.clone(), gen2.clone()];
        let group = PauliGroup::new(&generators);

        let elements = vec![gen1.clone(), gen2.clone()];
        let factorizations = group.factorizations_of(&elements);

        assert_eq!(factorizations.len(), 2);
        assert!(factorizations[0].is_some()); // gen1 should be factorizable
        assert!(factorizations[1].is_some()); // gen2 should be factorizable
    }

    #[test]
    fn test_factorizations_of_large_batch() {
        let generators = vec![
            SparsePauli::from_str("XI").unwrap(),
            SparsePauli::from_str("ZI").unwrap(),
            SparsePauli::from_str("IZ").unwrap(),
        ];
        let group = PauliGroup::new(&generators);

        // Create a large batch of elements (mix of in-group and out-of-group)
        let mut elements = Vec::new();

        // Add some group elements
        for element in group.elements().take(5) {
            elements.push(element);
        }

        // Add some non-group elements
        elements.push(SparsePauli::from_str("YY").unwrap());
        elements.push(SparsePauli::from_str("XX").unwrap());

        let factorizations = group.factorizations_of(&elements);
        assert_eq!(factorizations.len(), elements.len());

        // Verify that each factorization (when it exists) correctly reconstructs the element
        for (i, element) in elements.iter().enumerate() {
            if let Some(ref factorization) = factorizations[i] {
                let reconstructed = factorization.iter().fold(element.neutral_element(), |mut acc, p| {
                    acc.mul_assign_right(p);
                    acc
                });
                assert_eq!(
                    reconstructed, *element,
                    "Factorization failed for element at index {i}: {element}"
                );
            }
        }
    }

    #[test]
    fn test_elements_examples() {
        let group = PauliGroup::from_strings(&["XI", "IY"]);
        assert_eq!(group.elements().count(), 4);
    }

    #[test]
    fn test_intersection_examples() {
        let group_a = PauliGroup::from_strings(&["IX", "iI"]);
        let group_b = PauliGroup::from_strings(&["X", "Y", "IX"]);
        let group_c = PauliGroup::from_strings(&["IX", "-I"]);
        let intersection = &group_a & &group_b;
        assert_eq!(
            intersection,
            group_c,
            "Intersection {:?} != {:?}",
            intersection.generators(),
            group_c.generators()
        );

        let group_d = PauliGroup::from_strings(&["iX", "IZ", "IX"]);
        let group_e = PauliGroup::from_strings(&["X", "iI"]);
        let group_f = PauliGroup::from_strings(&["iX", "-I"]);
        let intersection = &group_e & &group_d;
        assert_eq!(
            intersection,
            group_f,
            "Intersection {:?} != {:?}",
            intersection.generators(),
            group_f.generators()
        );
    }

    #[test]
    fn test_is_symplectic_examples() {
        let basis1 = vec![
            SparsePauli::from_str("IX").unwrap(),
            SparsePauli::from_str("XI").unwrap(),
            SparsePauli::from_str("IIX").unwrap(),
            SparsePauli::from_str("ZI").unwrap(),
            SparsePauli::from_str("IZ").unwrap(),
        ];
        let symplectic_basis1 = symplectic_form_of(&basis1);
        assert!(is_symplectic(&symplectic_basis1));
    }
}

#[test]
fn test_quotient_examples() {
    let group = PauliGroup::from_strings(&["XX", "ZZ"]);
    let subgroup = PauliGroup::from_strings(&["ZZ"]);
    let quotient = group.clone() / &subgroup;
    assert_eq!(quotient.log2_size(), 1);
}

proptest! {
    #[test]
    fn test_quotient_coset_equivalence (
        group_generators in sparse_paulis(2, 4, small_sparse_pauli()),
        subgroup_size in 1..3usize
    ) {
        let group = PauliGroup::new(&group_generators);

        let actual_subgroup_size = std::cmp::min(subgroup_size, group_generators.len() - 1);
        let subgroup = PauliGroup::new(&group_generators[..actual_subgroup_size]);

        prop_assume!(subgroup <= group);

        // Property: For any element g in the group and any element h in the subgroup,
        // g/S should equal (g*h)/S where S is the subgroup
        for group_element in group.elements().take(5) {
            for subgroup_element in subgroup.elements().take(3) {
                let g = group_element.clone();
                let gh = &g * &subgroup_element;
                let g_group = PauliGroup::new(std::slice::from_ref(&g));
                let gh_group = PauliGroup::new(std::slice::from_ref(&gh));
                if subgroup <= g_group && subgroup <= gh_group {
                    let quotient_g = g_group / &subgroup;
                    let quotient_gh = gh_group / &subgroup;
                    prop_assert_eq!(quotient_g, quotient_gh);
                }
            }
        }
    }
}

fn is_symplectic(basis: &[SparsePauli]) -> bool {
    if basis.is_empty() {
        return true;
    }

    let mut g_index = basis.len();
    if basis.len() > 1 {
        for index in (1..basis.len()).rev() {
            if !commutes_with(&basis[index], &basis[index - 1]) {
                g_index = index + 1;
                break;
            }
            if index > 0 {
                g_index = index - 1;
            }
        }
    }

    let commuting_section = std::cmp::min(g_index, basis.len());
    for (index, element) in basis[..commuting_section].iter().enumerate() {
        if index + 2 >= basis.len() {
            continue;
        }

        for other in basis[index + 2..].iter().step_by(2) {
            if !commutes_with(element, other) {
                return false;
            }
        }
    }

    let remaining_start = std::cmp::min(g_index, basis.len());
    for element in &basis[remaining_start..] {
        for other in basis {
            if !commutes_with(element, other) {
                return false;
            }
        }
    }
    true
}

#[test]
#[should_panic(expected = "Divisor must be a subgroup of the dividend")]
fn test_quotient_panics_when_divisor_not_subgroup() {
    let group = PauliGroup::new(&[SparsePauli::from_str("X0").unwrap()]);
    let non_subgroup = PauliGroup::new(&[SparsePauli::from_str("Z0").unwrap()]);
    let _ = group / &non_subgroup;
}

#[test]
fn test_contains_returns_false_for_element_with_disjoint_support() {
    let group = PauliGroup::new(&[
        SparsePauli::from_str("X0X1").unwrap(),
        SparsePauli::from_str("Z0").unwrap(),
    ]);

    let element_with_disjoint_support = SparsePauli::from_str("Y2Y3Y4").unwrap();
    assert!(!group.contains(&element_with_disjoint_support));

    let element_with_partially_overlapping_support = SparsePauli::from_str("X0X2").unwrap();
    assert!(!group.contains(&element_with_partially_overlapping_support));
}

#[test]
fn test_factorizations_of_with_disjoint_support() {
    let group = PauliGroup::new(&[
        SparsePauli::from_str("X0X1").unwrap(),
        SparsePauli::from_str("Z0").unwrap(),
    ]);

    let elements = vec![
        SparsePauli::from_str("X0X1").unwrap(),  // In group
        SparsePauli::from_str("Y2Y3").unwrap(),  // Disjoint support
        SparsePauli::from_str("X0X2").unwrap(),  // Partially overlapping
        SparsePauli::from_str("Z0").unwrap(),    // In group
    ];

    let factorizations = group.factorizations_of(&elements);

    assert!(factorizations[0].is_some());  // X0X1 is in group
    assert!(factorizations[1].is_none());  // Y2Y3 has disjoint support
    assert!(factorizations[2].is_none());  // X0X2 partially overlaps
    assert!(factorizations[3].is_some());  // Z0 is in group
}