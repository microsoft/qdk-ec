#[cfg(feature = "serde")]
mod serde_tests {

    use binar::{BitwiseMut, IndexSet};
    use paulimer::clifford::{Clifford, CliffordMutable, CliffordUnitary};
    use paulimer::pauli::{Pauli, SparsePauli};
    use proptest::prelude::*;
    use rand::prelude::*;
    use std::str::FromStr;

    pub fn arbitrary_clifford(qubit_count_range: std::ops::Range<usize>) -> impl Strategy<Value = CliffordUnitary> {
        qubit_count_range.prop_flat_map(|qubit_count| {
            any::<u64>().prop_map(move |seed| {
                let mut random_number_generator = StdRng::seed_from_u64(seed);
                CliffordUnitary::random(qubit_count.max(1), &mut random_number_generator)
            })
        })
    }

    pub fn arbitrary_sparse_pauli(max_weight: usize) -> BoxedStrategy<SparsePauli> {
        (
            any::<u8>().prop_map(|exponent| exponent % 4),
            prop::collection::vec(any::<usize>().prop_map(|index| index % 100), 1..=max_weight),
            prop::collection::vec(prop::sample::select(vec!['X', 'Y', 'Z']), 1..=max_weight),
        )
            .prop_map(|(exponent, indices, operators)| {
                let mut x_bits = IndexSet::new();
                let mut z_bits = IndexSet::new();

                for (index, operator) in indices.into_iter().zip(operators.into_iter()) {
                    match operator {
                        'X' => x_bits.assign_index(index, true),
                        'Z' => z_bits.assign_index(index, true),
                        'Y' => {
                            x_bits.assign_index(index, true);
                            z_bits.assign_index(index, true);
                        }
                        _ => unreachable!(),
                    }
                }

                SparsePauli::from_bits(x_bits, z_bits, exponent)
            })
            .boxed()
    }

    proptest! {
        #[test]
        fn clifford_serde_roundtrip(clifford in arbitrary_clifford(0..5)) {
            let serialized = serde_json::to_string(&clifford).expect("Failed to serialize CliffordUnitary");
            let deserialized: CliffordUnitary = serde_json::from_str(&serialized).expect("Failed to deserialize CliffordUnitary");
            assert_eq!(clifford, deserialized);
        }

        #[test]
        fn sparse_pauli_serde_roundtrip(pauli in arbitrary_sparse_pauli(10)) {
            let serialized = serde_json::to_string(&pauli).expect("Failed to serialize SparsePauli");
            let deserialized: SparsePauli = serde_json::from_str(&serialized).expect("Failed to deserialize SparsePauli");
            assert_eq!(pauli, deserialized);
        }
    }

    #[test]
    fn clifford_identity_serde() {
        let identity = CliffordUnitary::identity(3);
        let serialized = serde_json::to_string(&identity).expect("Failed to serialize identity");
        let deserialized: CliffordUnitary = serde_json::from_str(&serialized).expect("Failed to deserialize identity");
        assert!(deserialized.is_identity());
        assert_eq!(identity, deserialized);
    }

    #[test]
    fn clifford_hadamard_serde() {
        let mut hadamard = CliffordUnitary::identity(1);
        hadamard.left_mul_hadamard(0);
        let serialized = serde_json::to_string(&hadamard).expect("Failed to serialize hadamard");
        let deserialized: CliffordUnitary = serde_json::from_str(&serialized).expect("Failed to deserialize hadamard");
        assert_eq!(hadamard, deserialized);
    }

    #[test]
    fn sparse_pauli_identity_serde() {
        let identity = SparsePauli::from_str("I").expect("Failed to parse identity");
        let serialized = serde_json::to_string(&identity).expect("Failed to serialize identity");
        let deserialized: SparsePauli = serde_json::from_str(&serialized).expect("Failed to deserialize identity");
        assert_eq!(identity, deserialized);
    }

    #[test]
    fn sparse_pauli_xyz_serde() {
        let x_bits: IndexSet = [0, 5].into_iter().collect();
        let z_bits: IndexSet = [5, 10].into_iter().collect();
        let pauli = SparsePauli::from_bits(x_bits, z_bits, 0);

        let serialized = serde_json::to_string(&pauli).expect("Failed to serialize pauli");
        let deserialized: SparsePauli = serde_json::from_str(&serialized).expect("Failed to deserialize pauli");
        assert_eq!(pauli, deserialized);
    }

    #[test]
    fn sparse_pauli_with_phase_serde() {
        let pauli = SparsePauli::from_bits(IndexSet::singleton(0), IndexSet::new(), 1);

        let serialized = serde_json::to_string(&pauli).expect("Failed to serialize pauli");
        let deserialized: SparsePauli = serde_json::from_str(&serialized).expect("Failed to deserialize pauli");
        assert_eq!(pauli.xz_phase_exponent(), deserialized.xz_phase_exponent());
        assert_eq!(pauli, deserialized);
    }

    #[test]
    fn sparse_pauli_negative_phase_serde() {
        let pauli = SparsePauli::from_bits(IndexSet::new(), IndexSet::singleton(3), 2);

        let serialized = serde_json::to_string(&pauli).expect("Failed to serialize pauli");
        let deserialized: SparsePauli = serde_json::from_str(&serialized).expect("Failed to deserialize pauli");
        assert_eq!(pauli, deserialized);
    }

    #[test]
    fn unitary_op_serde() {
        use paulimer::UnitaryOp;

        let ops = unitary_op_examples();

        for op in ops {
            let serialized = serde_json::to_string(&op).expect("Failed to serialize UnitaryOp");
            let deserialized: UnitaryOp = serde_json::from_str(&serialized).expect("Failed to deserialize UnitaryOp");
            assert_eq!(op as u8, deserialized as u8);
        }
    }

    pub fn unitary_op_examples() -> [paulimer::UnitaryOp; 15] {
        use paulimer::UnitaryOp;
        [
            UnitaryOp::I,
            UnitaryOp::X,
            UnitaryOp::Y,
            UnitaryOp::Z,
            UnitaryOp::SqrtX,
            UnitaryOp::SqrtXInv,
            UnitaryOp::SqrtY,
            UnitaryOp::SqrtYInv,
            UnitaryOp::SqrtZ,
            UnitaryOp::SqrtZInv,
            UnitaryOp::Hadamard,
            UnitaryOp::Swap,
            UnitaryOp::ControlledX,
            UnitaryOp::ControlledZ,
            UnitaryOp::PrepareBell,
        ]
    }
}

#[cfg(feature = "schemars")]
mod schemars_tests {
    use super::serde_tests::{arbitrary_clifford, arbitrary_sparse_pauli};
    use jsonschema::Validator;
    use paulimer::{
        clifford::{Clifford, CliffordMutable, CliffordUnitary},
        pauli::SparsePauli,
    };
    use proptest::prelude::*;
    use std::str::FromStr;

    fn validate_against_schema<T: schemars::JsonSchema>(value: &serde_json::Value) {
        let schema = schemars::schema_for!(T);
        let schema_json = serde_json::to_value(&schema).expect("Failed to convert schema to JSON");
        let validator = Validator::new(&schema_json).expect("Failed to compile schema");
        assert!(
            validator.is_valid(value),
            "Value does not match schema: {:?}",
            validator.iter_errors(value).collect::<Vec<_>>()
        );
    }

    #[test]
    fn clifford_unitary_schema_validates_serde_output() {
        let identity = CliffordUnitary::identity(3);
        let serialized = serde_json::to_value(&identity).expect("Failed to serialize");
        validate_against_schema::<CliffordUnitary>(&serialized);

        let mut hadamard = CliffordUnitary::identity(1);
        hadamard.left_mul_hadamard(0);
        let serialized = serde_json::to_value(&hadamard).expect("Failed to serialize");
        validate_against_schema::<CliffordUnitary>(&serialized);
    }

    #[test]
    fn sparse_pauli_schema_validates_serde_output() {
        let identity = SparsePauli::from_str("I").expect("Failed to parse");
        let serialized = serde_json::to_value(&identity).expect("Failed to serialize");
        validate_against_schema::<SparsePauli>(&serialized);

        let pauli = SparsePauli::from_str("-X₀Y₂Z₃").expect("Failed to parse");
        let serialized = serde_json::to_value(&pauli).expect("Failed to serialize");
        validate_against_schema::<SparsePauli>(&serialized);
    }

    #[test]
    fn unitary_op_schema_validates_serde_output() {
        use paulimer::UnitaryOp;

        let ops = super::serde_tests::unitary_op_examples();

        for op in ops {
            let serialized = serde_json::to_value(op).expect("Failed to serialize");
            validate_against_schema::<UnitaryOp>(&serialized);
        }
    }

    proptest! {
        #[test]
        fn clifford_schema_validates_random(clifford in arbitrary_clifford(0..5)) {
            let serialized = serde_json::to_value(&clifford).expect("Failed to serialize");
            validate_against_schema::<CliffordUnitary>(&serialized);
        }

        #[test]
        fn sparse_pauli_schema_validates_random(pauli in arbitrary_sparse_pauli(10)) {
            let serialized = serde_json::to_value(&pauli).expect("Failed to serialize");
            validate_against_schema::<SparsePauli>(&serialized);
        }
    }
}
