use super::*;

fn assert_signed_replay(clifford: &CliffordUnitary) -> usize {
    let decomposition = clifford_to_transvections_minimal(clifford);
    for pauli in decomposition
        .transvections
        .iter()
        .chain(std::iter::once(&decomposition.pauli_correction))
    {
        assert_eq!(pauli.xyz_phase_exponent(), 0);
        assert_eq!(usize::from(pauli.xz_phase_exponent()), pauli.y_weight() % 4);
    }
    let mut rebuilt = symplectic_action_from_transvections(&decomposition.transvections, clifford.num_qubits());
    rebuilt.left_mul_pauli(&decomposition.pauli_correction);
    assert_eq!(
        rebuilt, *clifford,
        "rotations followed by the Pauli must reproduce every sign"
    );
    decomposition.transvections.len()
}

#[test]
fn all_one_and_two_qubit_signed_tableaux_replay() {
    for (qubit_count, signed_count) in [(1, 24), (2, 11_520)] {
        let group = enumerate_symplectic_group(qubit_count);
        let mut visited = 0;
        for (minimum, representative) in group.values() {
            for mask in 0..(1usize << (2 * qubit_count)) {
                let vector: Vec<bool> = (0..2 * qubit_count).map(|bit| mask & (1 << bit) != 0).collect();
                let mut clifford = representative.clone();
                clifford.left_mul_pauli(&pauli_of_vector(&vector, qubit_count));
                assert_eq!(assert_signed_replay(&clifford), *minimum);
                visited += 1;
            }
        }
        assert_eq!(visited, signed_count);
    }
}

#[test]
fn zero_qubits_and_pauli_corrections() {
    assert_eq!(assert_signed_replay(&CliffordUnitary::identity(0)), 0);
    let empty = clifford_to_transvections_minimal(&CliffordUnitary::identity(0));
    assert_eq!(empty.transvections, Vec::<SparsePauli>::new());
    assert!(!is_non_identity(&empty.pauli_correction));

    for (gate, text) in [(UnitaryOp::X, "X"), (UnitaryOp::Y, "Y"), (UnitaryOp::Z, "Z")] {
        let mut clifford = CliffordUnitary::identity(1);
        clifford.left_mul(gate, &[0]);
        let decomposition = clifford_to_transvections_minimal(&clifford);
        let expected: SparsePauli = text.parse().unwrap();
        assert_eq!(decomposition.transvections, Vec::<SparsePauli>::new());
        assert_eq!(decomposition.pauli_correction, expected);
        assert_eq!(assert_signed_replay(&clifford), 0);
        assert_eq!(clifford_to_transvections(&clifford), Vec::<SparsePauli>::new());
    }

    let mut hadamard = CliffordUnitary::identity(1);
    hadamard.left_mul_hadamard(0);
    let decomposition = clifford_to_transvections_minimal(&hadamard);
    assert_eq!(decomposition.transvections, vec!["Y".parse::<SparsePauli>().unwrap()]);
    assert_eq!(decomposition.pauli_correction, "Z".parse::<SparsePauli>().unwrap());
    assert_eq!(assert_signed_replay(&hadamard), 1);

    let mut wrong_order = CliffordUnitary::identity(1);
    wrong_order.left_mul_pauli(&decomposition.pauli_correction);
    for transvection in &decomposition.transvections {
        wrong_order.left_mul_pauli_exp(transvection);
    }
    assert_ne!(wrong_order, hadamard, "the correction must follow the rotations");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn random_signed_quarter_turns_replay(
        qubit_count in 1usize..=4,
        rotations in vec((any::<u8>(), any::<bool>()), 0..16),
        final_mask in any::<u8>(),
    ) {
        let mut clifford = CliffordUnitary::identity(qubit_count);
        for (mask, negative) in rotations {
            let vector: Vec<bool> = (0..2 * qubit_count).map(|bit| mask & (1 << bit) != 0).collect();
            let mut pauli = pauli_of_vector(&vector, qubit_count);
            if negative {
                pauli.add_assign_phase_exp(2);
            }
            clifford.left_mul_pauli_exp(&pauli);
        }
        let vector: Vec<bool> = (0..2 * qubit_count).map(|bit| final_mask & (1 << bit) != 0).collect();
        clifford.left_mul_pauli(&pauli_of_vector(&vector, qubit_count));
        assert_signed_replay(&clifford);
    }
}
