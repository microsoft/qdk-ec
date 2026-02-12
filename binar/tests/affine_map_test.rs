use binar::{AffineMap, BitMatrix, BitVec, BitwiseMut};
use proptest::prelude::*;

fn affine_map_strategy(output_dim: usize, input_dim: usize) -> impl Strategy<Value = AffineMap> {
    (
        prop::collection::vec(any::<bool>(), output_dim * input_dim),
        prop::collection::vec(any::<bool>(), output_dim),
    )
        .prop_map(move |(matrix_bits, shift_bits)| {
            affine_map_from_bits(output_dim, input_dim, &matrix_bits, &shift_bits)
        })
}

fn affine_map_from_bits(
    output_dim: usize,
    input_dim: usize,
    matrix_bits: &[bool],
    shift_bits: &[bool],
) -> AffineMap {
    AffineMap::affine(
        bitmatrix_from_bits(output_dim, input_dim, matrix_bits),
        bitvec_from_bits(output_dim, shift_bits),
    )
}

fn composable_maps_with_input(max_dimension: usize) -> impl Strategy<Value = (AffineMap, AffineMap, BitVec)> {
    (1..=max_dimension, 1..=max_dimension, 1..=max_dimension).prop_flat_map(|(dim_a, dim_b, dim_c)| {
        (
            affine_map_strategy(dim_c, dim_b),
            affine_map_strategy(dim_b, dim_a),
            prop::collection::vec(any::<bool>(), dim_a),
            Just(dim_a),
        )
    }).prop_map(|(first, second, input_bits, dim_a)| {
        (first, second, bitvec_from_bits(dim_a, &input_bits))
    })
}

fn three_composable_maps(max_dimension: usize) -> impl Strategy<Value = (AffineMap, AffineMap, AffineMap)> {
    (1..=max_dimension, 1..=max_dimension, 1..=max_dimension, 1..=max_dimension)
        .prop_flat_map(|(dim_a, dim_b, dim_c, dim_d)| {
            (
                affine_map_strategy(dim_d, dim_c),
                affine_map_strategy(dim_c, dim_b),
                affine_map_strategy(dim_b, dim_a),
            )
        })
}

fn affine_map_with_input(max_dimension: usize) -> impl Strategy<Value = (AffineMap, BitVec)> {
    (1..=max_dimension, 1..=max_dimension).prop_flat_map(|(input_dim, output_dim)| {
        (
            affine_map_strategy(output_dim, input_dim),
            prop::collection::vec(any::<bool>(), input_dim),
            Just(input_dim),
        )
    }).prop_map(|(map, input_bits, input_dim)| {
        (map, bitvec_from_bits(input_dim, &input_bits))
    })
}

fn bitmatrix_from_bits(rows: usize, cols: usize, bits: &[bool]) -> BitMatrix {
    let mut matrix = BitMatrix::zeros(rows, cols);
    for row in 0..rows {
        for col in 0..cols {
            let index = row * cols + col;
            if index < bits.len() {
                matrix.set((row, col), bits[index]);
            }
        }
    }
    matrix
}

fn bitvec_from_bits(length: usize, bits: &[bool]) -> BitVec {
    let mut vec = BitVec::zeros(length);
    for (i, &bit) in bits.iter().take(length).enumerate() {
        vec.assign_index(i, bit);
    }
    vec
}

proptest! {
    #[test]
    fn dot_apply_consistency((first, second, input) in composable_maps_with_input(20)) {
        let composed = first.dot(&second);
        let via_composition = composed.apply(&input);
        let via_sequential = first.apply(&second.apply(&input));
        prop_assert_eq!(via_composition, via_sequential);
    }

    #[test]
    fn dot_associativity((f, g, h) in three_composable_maps(10)) {
        let left_associated = f.dot(&g).dot(&h);
        let right_associated = f.dot(&g.dot(&h));
        prop_assert_eq!(left_associated, right_associated);
    }

    #[test]
    fn identity_composition((map, _) in affine_map_with_input(20)) {
        let identity = AffineMap::linear(BitMatrix::identity(map.input_dimension()));
        let composed = map.dot(&identity);
        prop_assert_eq!(map.matrix(), composed.matrix());
        prop_assert_eq!(map.shift(), composed.shift());
    }

    #[test]
    fn translation_apply(shift_bits in prop::collection::vec(any::<bool>(), 1..50)) {
        let shift = bitvec_from_bits(shift_bits.len(), &shift_bits);
        let translation = AffineMap::translation(shift.clone());
        let input = BitVec::zeros(shift.len());
        let result = translation.apply(&input);
        prop_assert_eq!(result, shift);
    }

    #[test]
    fn linear_map_zero_at_zero((map, _) in affine_map_with_input(20)) {
        let linear = AffineMap::linear(map.matrix().clone());
        let zero_input = BitVec::zeros(linear.input_dimension());
        let result = linear.apply(&zero_input);
        let expected = BitVec::zeros(linear.output_dimension());
        prop_assert_eq!(result, expected);
    }
}
