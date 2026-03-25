use core::fmt;
use std::collections::HashSet;
use std::str::FromStr;

use binar::vec::AlignedBitViewMut as MutableBitView;
use paulimer::StringLayout::{Dense, Sparse};
use paulimer::StringNotation::{Ascii, Tex, Unicode};
use paulimer::core::{x, y, z};
use paulimer::pauli::{
    DensePauli, DensePauliProjective, Pauli, PauliBinaryOps, PauliMutable, PauliUnitary, Phase, SparsePauli,
    SparsePauliProjective, commutes_with, generic::PhaseExponent,
};
use proptest::prelude::*;

proptest! {
    #[test]
    fn from_bits(pauli in arbitrary_pauli(1000)) {
        let from_bits = PauliUnitary::from_bits(pauli.x_bits().clone(), pauli.z_bits().clone(), pauli.xz_phase_exponent());
        assert_eq!(pauli.x_bits(), from_bits.x_bits());
        assert_eq!(pauli.z_bits(), from_bits.z_bits());
        assert_eq!(pauli.xz_phase_exponent(), from_bits.xz_phase_exponent());
        assert_eq!(pauli, from_bits);
    }

    #[test]
    fn from_references(pauli in arbitrary_pauli(1000)) {
        let from_reference = PauliUnitary::from_bits(pauli.x_bits().as_slice(), pauli.z_bits().as_slice(), pauli.xz_phase_exponent());
        assert_eq!(pauli.x_bits(), from_reference.x_bits());
        assert_eq!(pauli.z_bits(), from_reference.z_bits());
        assert_eq!(pauli.xz_phase_exponent(), from_reference.xz_phase_exponent());
        assert_eq!(pauli, from_reference);
    }

    #[test]
    fn is_hermitian(pauli in arbitrary_pauli(1000)) {
        let square = pauli.clone() * &pauli;
        let expect_hermitian = square.xz_phase_exponent().value() == 0;
        assert_eq!(pauli.is_order_two(), expect_hermitian);
    }

    #[test]
    fn commutes_with_matches_explicit_commutator((left, right) in equal_length_paulis(1)) {
        let leftright = left.clone() * &right;
        let rightleft = right.clone() * &left;
        let expect_commutes_with = leftright == rightleft;
        assert_eq!(commutes_with(&left,&right), expect_commutes_with);
    }

    #[test]
    fn cayley_table(length in 1..1000usize, mut index in 0..999usize) {
        index %= length;
        let zeros = vec![false; length];
        let mut bits = zeros.clone();
        bits[index] = true;
        let identity = PauliUnitary::from_bits(zeros.clone(), zeros.clone(), 0u8);
        let i = &PauliUnitary::from_bits(zeros.clone(), zeros.clone(), 1u8);
        let neg = &PauliUnitary::from_bits(zeros.clone(), zeros.clone(), 2u8);
        let x = &PauliUnitary::from_bits(bits.clone(), zeros.clone(), 0u8);
        let z = &PauliUnitary::from_bits(zeros.clone(), bits.clone(), 0u8);
        let y = &PauliUnitary::from_bits(bits.clone(), bits.clone(), 1u8);

        assert_eq!(x.clone() * x, identity);
        assert_eq!(y.clone() * y, identity);
        assert_eq!(z.clone() * z, identity);

        assert_eq!(x * x, identity);
        assert_eq!(y * y, identity);
        assert_eq!(z * z, identity);

        assert_eq!(z.clone() * x, y.clone() * i);
        assert_eq!(x.clone() * z, y.clone() * i * neg);
        assert_eq!(y.clone() * z, x.clone() * i);
        assert_eq!(z.clone() * y, x.clone() * i * neg);
        assert_eq!(x.clone() * y, z.clone() * i);
        assert_eq!(y.clone() * x, z.clone() * i * neg);

        assert_eq!(z * x, i * y );
        assert_eq!(x * z, -i * y );
        assert_eq!(y * z, i * x);
        assert_eq!(z * y, -i * x);
        assert_eq!(x * y, i * z );
        assert_eq!(y * x, -i * z);

        assert_eq!(i * x * z, y);
    }

    #[test]
    fn unsigned_int_paulis(index in 0..32usize) {
        let one_bit = 1u32 << index;
        let zero_bit = 0u32;
        let x = &PauliUnitary::from_bits(one_bit, zero_bit, 0u8);
        let z = &PauliUnitary::from_bits(zero_bit, one_bit, 0u8);
        let y = &PauliUnitary::from_bits(one_bit, one_bit, 1u8);
        let i = &PauliUnitary::from_bits(zero_bit, zero_bit, 1u8);
        assert_eq!(i * x * z, y);
    }

    #[test]
    fn mul_assign((mut left, right) in equal_length_paulis(1000)) {
        let product = left.clone() * &right;
        left *= &right;
        assert_eq!(left, product);
    }

    #[test]
    fn mul_assign_phase(mut pauli in arbitrary_pauli(1000), raw_exponent in 0..4u8) {
        let original = pauli.clone();
        let exponent = raw_exponent;
        pauli *= Phase::from_exponent(exponent);
        assert_eq!(original.x_bits(), pauli.x_bits());
        assert_eq!(original.z_bits(), pauli.z_bits());
        assert_eq!(
            original.xz_phase_exponent().raw_value().wrapping_add(exponent) % 4u8,
            pauli.xz_phase_exponent().value()
        );
    }

    #[test]
    fn format_round_trip(pauli in arbitrary_pauli(50)) {
        let str_dense = format!("{pauli}");
        let str_sparse = format!("{pauli:#}");
        test_round_trip::<SparsePauli>(&str_dense, &str_sparse, true);
        test_round_trip::<DensePauli>(&str_dense, &str_sparse, true);
        test_round_trip::<SparsePauliProjective>(&str_dense, &str_sparse, false);
        test_round_trip::<DensePauliProjective>(&str_dense, &str_sparse, false);
    }

    #[test]
    fn left_multiply((left, right) in equal_length_paulis(1000)) {
        let right_product = left.clone() * &right;
        let left_product = &left * right;
        assert_eq!(right_product, left_product);
    }

    #[test]
    fn test_pauli_unitary_hash_consistency(mut pauli in arbitrary_pauli(50), phase_offset in 1..20u8) {

        let original_pauli = pauli.clone();

        let original_phase = pauli.xz_phase_exponent().value();
        let equivalent_phase = original_phase.wrapping_add(phase_offset * 4);
        pauli.assign_phase_exp(equivalent_phase);

        prop_assert_eq!(&original_pauli, &pauli);

        let mut set = HashSet::new();
        set.insert(original_pauli.clone());
        set.insert(pauli.clone());
        prop_assert_eq!(set.len(), 1);
    }

}

prop_compose! {
   fn arbitrary_pauli(max_dimension: usize)(dimension in 0..max_dimension) -> PauliUnitary<Vec<bool>, u8>{
        arbitrary_pauli_of_length(dimension)
   }
}
prop_compose! {
   fn equal_length_paulis(max_dimension: usize)(dimension in 0..max_dimension) -> (PauliUnitary<Vec<bool>, u8>, PauliUnitary<Vec<bool>, u8>){
        (arbitrary_pauli_of_length(dimension), arbitrary_pauli_of_length(dimension))
   }
}

fn arbitrary_pauli_of_length(length: usize) -> PauliUnitary<Vec<bool>, u8> {
    paulimer::pauli::pauli_random(length, &mut rand::rng())
}

#[test]
fn pauli_product_test() {
    let target: DensePauli = [z(0), z(1)].into();
    let preimage: DensePauli = [x(0)].into();
    let mut phase = preimage.xz_phase_exponent().raw_value();

    let (mut x_bits, mut z_bits) = preimage.to_xz_bits();
    let mut preimage_view =
        PauliUnitary::<MutableBitView, &mut u8>::from_bits(x_bits.as_view_mut(), z_bits.as_view_mut(), &mut phase);

    let control: DensePauli = [y(0), x(1)].into();
    let preimage_r: DensePauli = [-y(1)].into();
    preimage_view.mul_assign_right(&target);
    println!("{preimage_view}");
    preimage_view.mul_assign_left(&control);
    assert!(preimage_view == preimage_r);
}

fn test_round_trip<PauliLike: Pauli + FromStr<Err: fmt::Debug> + Eq + fmt::Debug + fmt::Display>(
    str_dense: &str,
    str_sparse: &str,
    with_phases: bool,
) {
    let pauli1 = str_dense.parse::<PauliLike>().expect(str_dense);
    let pauli2 = str_sparse.parse::<PauliLike>().expect(str_sparse);
    assert_eq!(pauli1, pauli2);

    let str_dense_1 = format!("{pauli1}");
    let str_sparse_1 = format!("{pauli1:#}");

    let str_dense_2 = format!("{pauli2}");
    let str_sparse_2 = format!("{pauli2:#}");

    assert_eq!(str_dense_2, str_dense_1);
    assert_eq!(str_sparse_2, str_sparse_1);

    if with_phases {
        assert_eq!(str_dense, str_dense_2);
        assert_eq!(str_sparse, str_sparse_2);
    }

    let pauli3 = str_dense_2.parse::<PauliLike>().expect(str_dense);
    let pauli4 = str_sparse_2.parse::<PauliLike>().expect(str_sparse);
    assert_eq!(pauli1, pauli3);
    assert_eq!(pauli1, pauli4);
}

#[test]
#[allow(clippy::missing_panics_doc)]
fn xyz_phase_test() {
    let positive_xyz_exp_examples = ["I", "Y", "XY", "YZ", "XZ", "XYZ", "YY", "YYY"];
    let prefix_phase_exp_pair = [("", 0u8), ("i", 1u8), ("-", 2u8), ("-i", 3u8)];
    for s in positive_xyz_exp_examples {
        for (prefix, expected_exp) in prefix_phase_exp_pair {
            let pauli = SparsePauli::from_str(&format!("{prefix}{s}")).unwrap();
            assert_eq!(
                pauli.xyz_phase_exponent(),
                expected_exp,
                "Pauli {prefix}{s} should have xyz phase exponent {expected_exp}"
            );
        }
    }
}

proptest! {
    #[test]
    fn to_string_with_round_trip(pauli in arbitrary_pauli(50)) {
        let combos: Vec<(&str, String)> = vec![
            ("dense+unicode", pauli.to_string()),
            ("dense+ascii", pauli.to_string_with(Dense, Ascii)),
            ("sparse+unicode", pauli.to_string_with(Sparse, Unicode)),
            ("sparse+ascii", pauli.to_string_with(Sparse, Ascii)),
        ];
        for (label, string) in &combos {
            let parsed_dense: DensePauli = string.parse().unwrap();
            let parsed_sparse: SparsePauli = string.parse().unwrap();

            let (rt_dense, rt_sparse) = match *label {
                "dense+unicode" => (parsed_dense.to_string(), parsed_sparse.to_string()),
                "dense+ascii" => (parsed_dense.to_string_with(Dense, Ascii), parsed_sparse.to_string_with(Dense, Ascii)),
                "sparse+unicode" => (parsed_dense.to_string_with(Sparse, Unicode), parsed_sparse.to_string_with(Sparse, Unicode)),
                "sparse+ascii" => (parsed_dense.to_string_with(Sparse, Ascii), parsed_sparse.to_string_with(Sparse, Ascii)),
                _ => unreachable!(),
            };
            prop_assert_eq!(string, &rt_dense, "DensePauli {}", label);
            prop_assert_eq!(string, &rt_sparse, "SparsePauli {}", label);
        }
    }

    #[test]
    fn ascii_and_unicode_parse_to_same_pauli(pauli in arbitrary_pauli(50)) {
        let dense_ascii = pauli.to_string_with(Dense, Ascii);
        let dense_unicode = pauli.to_string();
        let from_dense_ascii: DensePauli = dense_ascii.parse().unwrap();
        let from_dense_unicode: DensePauli = dense_unicode.parse().unwrap();
        prop_assert_eq!(&from_dense_ascii, &from_dense_unicode, "dense layout");

        let sparse_ascii = pauli.to_string_with(Sparse, Ascii);
        let sparse_unicode = pauli.to_string_with(Sparse, Unicode);
        let from_sparse_ascii: DensePauli = sparse_ascii.parse().unwrap();
        let from_sparse_unicode: DensePauli = sparse_unicode.parse().unwrap();
        prop_assert_eq!(&from_sparse_ascii, &from_sparse_unicode, "sparse layout");
    }

    #[test]
    fn sparse_and_dense_parse_to_same_pauli(pauli in arbitrary_pauli(50)) {
        let sparse = pauli.to_string_with(Sparse, Ascii);
        let dense = pauli.to_string_with(Dense, Ascii);
        let from_sparse: DensePauli = sparse.parse().unwrap();
        let from_dense: DensePauli = dense.parse().unwrap();
        prop_assert_eq!(&from_sparse, &from_dense);
    }

    #[test]
    fn projective_string_methods_omit_phase(pauli in arbitrary_projective_pauli(50)) {
        let combos = [
            pauli.to_string(),
            pauli.to_string_with(Dense, Ascii),
            pauli.to_string_with(Sparse, Unicode),
            pauli.to_string_with(Sparse, Ascii),
        ];
        let phase_prefixes = ["+", "-", "i", "-i", "𝑖", "-𝑖"];
        for string in &combos {
            for prefix in phase_prefixes {
                prop_assert!(
                    !string.starts_with(prefix),
                    "unexpected prefix {:?} in {:?}",
                    prefix, string
                );
            }
        }
    }
}

#[test]
#[allow(clippy::similar_names)]
fn pauli_weight_test() {
    let pauli_strings = ["I", "X", "Y", "Z", "XX", "XY", "XZ", "YY", "YZ", "ZZ", "XYZ"];
    for pauli_string in pauli_strings {
        let pauli = SparsePauli::from_str(pauli_string).unwrap();
        let x_weight = pauli.x_weight();
        let y_weight = pauli.y_weight();
        let z_weight = pauli.z_weight();
        let formatted = format!("{pauli}");
        let count_x = formatted.matches('X').count();
        let count_y = formatted.matches('Y').count();
        let count_z = formatted.matches('Z').count();
        assert_eq!(x_weight, count_x, "Pauli {pauli_string} should have x_weight {count_x}");
        assert_eq!(y_weight, count_y, "Pauli {pauli_string} should have y_weight {count_y}");
        assert_eq!(z_weight, count_z, "Pauli {pauli_string} should have z_weight {count_z}");
    }
}

#[test]
fn sparse_parsing_with_large_qubit_index() {
    let pauli: DensePauli = "X_512".parse().unwrap();
    assert!(pauli.size() >= 513);

    let pauli: DensePauli = "X_0Z_1023".parse().unwrap();
    assert!(pauli.size() >= 1024);
}

prop_compose! {
    fn arbitrary_projective_pauli(max_dimension: usize)(dimension in 0..max_dimension) -> DensePauliProjective {
        let pauli = arbitrary_pauli_of_length(dimension);
        let (x_bits, z_bits) = pauli.to_xz_bits();
        let zero_phase = PauliUnitary::from_bits(x_bits, z_bits, 0u8);
        DensePauliProjective::from_str(
            &zero_phase.to_string_with(Sparse, Ascii),
        ).unwrap()
    }
}

#[test]
fn dense_pauli_to_string_identity() {
    let pauli: DensePauli = "I".parse().unwrap();
    assert_eq!(pauli.to_string_with(Sparse, Ascii), "I");
    assert_eq!(pauli.to_string_with(Dense, Ascii), "I");
    assert_eq!(pauli.to_string_with(Sparse, Unicode), "I");
    assert_eq!(pauli.to_string(), "I");
}

#[test]
fn dense_pauli_to_string_single_qubit() {
    let x: DensePauli = "X".parse().unwrap();
    assert_eq!(x.to_string_with(Sparse, Ascii), "X_0");
    assert_eq!(x.to_string_with(Sparse, Unicode), "X₀");
    assert_eq!(x.to_string_with(Dense, Ascii), "X");
    assert_eq!(x.to_string(), "X");

    let y: DensePauli = "Y".parse().unwrap();
    assert_eq!(y.to_string_with(Sparse, Ascii), "Y_0");
    assert_eq!(y.to_string_with(Sparse, Unicode), "Y₀");
    assert_eq!(y.to_string_with(Dense, Ascii), "Y");
    assert_eq!(y.to_string(), "Y");

    let z: DensePauli = "Z".parse().unwrap();
    assert_eq!(z.to_string_with(Sparse, Ascii), "Z_0");
    assert_eq!(z.to_string_with(Sparse, Unicode), "Z₀");
    assert_eq!(z.to_string_with(Dense, Ascii), "Z");
    assert_eq!(z.to_string(), "Z");
}

#[test]
fn dense_pauli_to_string_with_phase() {
    let neg_x: DensePauli = "-X".parse().unwrap();
    assert_eq!(neg_x.to_string_with(Sparse, Ascii), "-X_0");
    assert_eq!(neg_x.to_string_with(Sparse, Unicode), "-X₀");
    assert_eq!(neg_x.to_string_with(Dense, Ascii), "-X");
    assert_eq!(neg_x.to_string(), "-X");

    let iy: DensePauli = "iY".parse().unwrap();
    assert_eq!(iy.to_string_with(Sparse, Ascii), "iY_0");
    assert_eq!(iy.to_string_with(Sparse, Unicode), "𝑖Y₀");
    assert_eq!(iy.to_string_with(Dense, Ascii), "iY");
    assert_eq!(iy.to_string(), "𝑖Y");

    let neg_iz: DensePauli = "-iZ".parse().unwrap();
    assert_eq!(neg_iz.to_string_with(Sparse, Ascii), "-iZ_0");
    assert_eq!(neg_iz.to_string_with(Sparse, Unicode), "-𝑖Z₀");
    assert_eq!(neg_iz.to_string_with(Dense, Ascii), "-iZ");
    assert_eq!(neg_iz.to_string(), "-𝑖Z");
}

#[test]
fn dense_pauli_to_string_multi_qubit() {
    let xyz: DensePauli = "XYZ".parse().unwrap();
    assert_eq!(xyz.to_string_with(Sparse, Ascii), "X_0 Y_1 Z_2");
    assert_eq!(xyz.to_string_with(Sparse, Unicode), "X₀Y₁Z₂");
    assert_eq!(xyz.to_string_with(Dense, Ascii), "XYZ");
    assert_eq!(xyz.to_string(), "XYZ");
}

#[test]
fn dense_pauli_to_string_sparse_skips_identities() {
    let xiz: DensePauli = "XIZ".parse().unwrap();
    assert_eq!(xiz.to_string_with(Sparse, Ascii), "X_0 Z_2");
    assert_eq!(xiz.to_string_with(Sparse, Unicode), "X₀Z₂");
    assert_eq!(xiz.to_string_with(Dense, Ascii), "XIZ");
    assert_eq!(xiz.to_string(), "XIZ");
}

#[test]
fn projective_pauli_to_string_no_phase() {
    let pauli: DensePauliProjective = "XYZ".parse().unwrap();
    assert_eq!(pauli.to_string_with(Sparse, Ascii), "X_0 Y_1 Z_2");
    assert_eq!(pauli.to_string_with(Sparse, Unicode), "X₀Y₁Z₂");
    assert_eq!(pauli.to_string_with(Dense, Ascii), "XYZ");
    assert_eq!(pauli.to_string(), "XYZ");

    let identity: DensePauliProjective = "I".parse().unwrap();
    assert_eq!(identity.to_string_with(Sparse, Ascii), "I");
    assert_eq!(identity.to_string_with(Dense, Ascii), "I");
}

#[test]
fn pauli_tex_notation() {
    let pauli: DensePauli = "XYZ".parse().unwrap();
    assert_eq!(pauli.to_string_with(Dense, Tex), "XYZ");
    assert_eq!(pauli.to_string_with(Sparse, Tex), "X_{0} Y_{1} Z_{2}");

    let xiz: DensePauli = "XIZ".parse().unwrap();
    assert_eq!(xiz.to_string_with(Sparse, Tex), "X_{0} Z_{2}");

    let neg_ix: DensePauli = "-iX".parse().unwrap();
    assert_eq!(neg_ix.to_string_with(Dense, Tex), "-\\mathrm{i}X");
    assert_eq!(neg_ix.to_string_with(Sparse, Tex), "-\\mathrm{i}X_{0}");

    let identity: DensePauli = "I".parse().unwrap();
    assert_eq!(identity.to_string_with(Dense, Tex), "I");
}
