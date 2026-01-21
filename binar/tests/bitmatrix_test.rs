use binar::matrix::tiny_matrix::{tiny_matrix_from_bitmatrix, tiny_matrix_rref};
use binar::matrix::{AlignedBitMatrix, AlignedEchelonForm as EchelonForm, directly_summed, kernel_basis_matrix};
use binar::vec::AlignedBitVec;
use binar::{Bitwise, BitwiseMut, BitwisePairMut};
use proptest::prelude::*;
use rand::Rng;
use rand::prelude::*;
use sorted_iter::SortedIterator;
use sorted_iter::assume::AssumeSortedByItemExt;
use std::collections::{BTreeMap, HashSet};
use std::str::FromStr;

proptest! {
    #[test]
    fn shape(row_count in 0..100usize, column_count in 0..100usize) {
        let matrix = AlignedBitMatrix::with_shape(row_count, column_count);
        assert_eq!(matrix.row_count(), row_count);
        assert_eq!(matrix.column_count(), column_count);
        assert_eq!(matrix.shape(), (row_count, column_count));
    }

    #[test]
    fn zeros(row_count in 0..100usize, column_count in 0..100usize) {
        let matrix = AlignedBitMatrix::zeros(row_count, column_count);
        for irow in 0..matrix.row_count() {
            for icol in 0..matrix.column_count() {
                assert!(!matrix[(irow, icol)]);
            }
        }
    }

    // #[test]
    // fn ones(row_count in 0..100usize, column_count in 0..100usize) {
    //     let matrix = AlignedBitMatrix::ones(row_count, column_count);
    //     for index in iproduct!(0..matrix.row_count(), 0..matrix.column_count()) {
    //         assert_eq!(matrix[index], true);
    //     }
    // }

    #[test]
    fn indexing(matrix in arbitrary_bitmatrix(100)) {
        for irow in 0..matrix.row_count() {
            for icol in 0..matrix.column_count() {
                assert_eq!(matrix[(irow, icol)], matrix[[irow, icol]]);
            }
        }
    }

    #[test]
    fn clone(matrix in arbitrary_bitmatrix(100)) {
        assert_eq!(matrix, matrix.clone());
    }

    #[test]
    fn swap_rows(matrix in nonempty_bitmatrix(100), raw_row_indexes in (0..100usize, 0..100usize)) {
        let row_indexes = [raw_row_indexes.0 % matrix.row_count(), raw_row_indexes.1 % matrix.row_count()];
        let mut swapped = matrix.clone();
        swapped.swap_rows(row_indexes[0], row_indexes[1]);
        for column_index in 0..matrix.column_count() {
            assert_eq!(matrix[[row_indexes[0], column_index]], swapped[[row_indexes[1], column_index]]);
        }
        for row_index in (0..matrix.row_count()).collect::<HashSet<usize>>().difference(&HashSet::from(row_indexes)) {
            for column_index in 0..matrix.column_count() {
                assert_eq!(matrix[[*row_index, column_index]], swapped[[*row_index, column_index]]);
            }
        }
    }

    #[test]
    fn swap_columns(matrix in nonempty_bitmatrix(100), raw_column_indexes in (0..100usize, 0..100usize)) {
        let column_indexes = [raw_column_indexes.0 % matrix.column_count(), raw_column_indexes.1 % matrix.column_count()];
        let mut swapped = matrix.clone();
        swapped.swap_columns(column_indexes[0], column_indexes[1]);
        for row_index in 0..matrix.row_count() {
            assert_eq!(matrix[[row_index, column_indexes[0]]], swapped[[row_index, column_indexes[1]]]);
        }
        for column_index in (0..matrix.column_count()).collect::<HashSet<usize>>().difference(&HashSet::from(column_indexes)) {
            for row_index in 0..matrix.row_count() {
                assert_eq!(matrix[[row_index, *column_index]], swapped[[row_index, *column_index]]);
            }
        }
    }

    #[test]
    fn addition((left, right) in equal_shape_bitmatrices(100)) {
        let sum = &left + &right;
        for irow in 0..left.row_count() {
            for icol in 0..right.column_count() {
                let index = (irow, icol);
                assert_eq!(sum[index], left[index] ^ right[index]);
            }
        }
        assert_eq!(sum, &right + &left);
    }

    #[test]
    fn addition_inplace((mut left, right) in equal_shape_bitmatrices(100)) {
        let sum = &left + &right;
        left += &right;
        assert_eq!(sum, left);
    }

    #[test]
    fn xor((left, right) in equal_shape_bitmatrices(100)) {
        assert_eq!(&left ^ &right, &left + &right);
    }

    #[test]
    fn xor_inplace((mut left, right) in equal_shape_bitmatrices(100)) {
        let xor = &left ^ &right;
        left ^= &right;
        assert_eq!(xor, left);
    }

    #[test]
    fn and((left, right) in equal_shape_bitmatrices(100)) {
        let and = &left & &right;
        for irow in 0..left.row_count() {
            for icol in 0..left.column_count() {
                let index = (irow, icol);
                assert_eq!(and[index], left[index] & right[index]);
            }
        }
        assert_eq!(and, &right & &left);
    }


    #[test]
    fn and_inplace((mut left, right) in equal_shape_bitmatrices(100)) {
        let and = &left & &right;
        left &= &right;
        assert_eq!(and, left);
    }

    #[test]
    fn equality(left in arbitrary_bitmatrix(100), right in arbitrary_bitmatrix(100)) {
        let mut are_equal = left.shape() == right.shape();
        if are_equal {
            for irow in 0..left.row_count() {
                for icol in 0..right.column_count() {
                    let index = (irow, icol);
                    are_equal &= left[index] == right[index];
                }
            }
        }
        assert_eq!(left == right, are_equal);
    }

    #[test]
    fn transpose(matrix in arbitrary_bitmatrix(100)) {
        let transposed = matrix.transposed();
        for row in 0..matrix.row_count() {
            for column in 0..matrix.column_count() {
                assert_eq!(matrix[(row, column)], transposed[(column, row)]);
            }
        }
    }

    #[test]
    fn inverse(matrix in invertible_bitmatrix(100)) {
        let inverted = matrix.inverted();
        let identity = AlignedBitMatrix::identity(matrix.row_count());
        assert_eq!(&matrix * &inverted, identity);
    }

    #[test]
    fn echelon_form(matrix in arbitrary_bitmatrix(100)) {
        let mut echeloned = matrix.clone();
        let profile = echeloned.echelonize();
        assert!(is_rref(&echeloned, &profile));
        assert!(preserves_rowspan_of(&matrix, &echeloned));
    }

    #[test]
    fn tiny_matrix_echelon_form(aligned_matrix in fixed_size_bitmatrix(32,60)) {
        use binar::BitMatrix;
        let matrix = BitMatrix::from_aligned(aligned_matrix);
        let mut echeloned = matrix.clone();
        let _ = echeloned.echelonize();
        let mut tiny1 = tiny_matrix_from_bitmatrix::<32>(&matrix);
        tiny_matrix_rref::<32,60>(&mut tiny1);
        let tiny2 = tiny_matrix_from_bitmatrix::<32>(&echeloned);
        assert_eq!(tiny1,tiny2);
    }

    #[test]
    fn direct_sum(left in arbitrary_bitmatrix(100), right in arbitrary_bitmatrix(100)) {
        let summed = directly_summed([&left, &right]);
        let expected_shape = (left.row_count() + right.row_count(), left.column_count() + right.column_count());
        assert_eq!(expected_shape, summed.shape());
        for row_index in 0..left.row_count() {
            for column_index in 0..left.column_count() {
                assert_eq!(left[(row_index, column_index)], summed[(row_index, column_index)]);
            }
            for column_index in left.column_count()..summed.column_count() {
                assert!(!summed[(row_index, column_index)]);
            }
        }
        for row_index in 0..right.row_count() {
            for column_index in 0..right.column_count() {
                assert_eq!(right[(row_index, column_index)], summed[(left.row_count() + row_index, left.column_count() + column_index)]);
            }
            for column_index in 0..left.column_count() {
                assert!(!summed[(left.row_count() + row_index, column_index)]);
            }
        }
    }

}

macro_rules! bitmatrix{
    ($($t:tt)+) => {
        AlignedBitMatrix::from_str(stringify!($($t)+)).unwrap()
    };
}

prop_compose! {
   fn arbitrary_bitmatrix(max_dimension: usize)(shape in (0..=max_dimension, 0..=max_dimension)) -> AlignedBitMatrix {
       random_bitmatrix(shape.0, shape.1)
   }
}

prop_compose! {
   fn fixed_size_bitmatrix(row_count: usize, column_count: usize)(_ in 0..column_count) -> AlignedBitMatrix {
       random_bitmatrix(row_count, column_count)
   }
}

prop_compose! {
   fn invertible_bitmatrix(max_dimension: usize)(dimension in 1..=max_dimension) -> AlignedBitMatrix {
       let mut matrix = AlignedBitMatrix::identity(dimension);
       for _ in 0..dimension^2 {
            let from_index = thread_rng().gen_range(0..dimension);
            let to_index = thread_rng().gen_range(0..dimension);
            if from_index != to_index {
                matrix.add_into_row(to_index, from_index);
            }
       }
       for _ in 0..dimension.pow(2) {
            let from_index = thread_rng().gen_range(0..dimension);
            let to_index = thread_rng().gen_range(0..dimension);
            matrix.swap_rows(from_index, to_index);
       }
       matrix
   }
}

prop_compose! {
   fn nonempty_bitmatrix(max_dimension: usize)(shape in (1..=max_dimension, 1..=max_dimension)) -> AlignedBitMatrix {
       random_bitmatrix(shape.0, shape.1)
   }
}

prop_compose! {
   fn equal_shape_bitmatrices(max_dimension: usize)(shape in (1..=max_dimension, 1..=max_dimension)) -> (AlignedBitMatrix, AlignedBitMatrix) {
       (random_bitmatrix(shape.0, shape.1), random_bitmatrix(shape.0, shape.1))
   }
}

// #[test]
// fn reduce() {
//     for _ in 0..100 {
//         let array = random_bitmatrix(100, 100);
//         let reduced = rref(array);
//         assert!(is_rref(&reduced));
//     }

//     for _ in 0..100 {
//         let array = random_bitmatrix(50, 100);
//         let (reduced, profile) = rref_with_rank_profile(array);
//         assert_eq!(profile.len(), reduced.row_count());
//         assert!(is_rref(&reduced));
//     }

//     {
//         let matrix = bitmatrix!(
//             |10 011 01|
//             |.. 111 01|
//             |.. ... 10|);
//         assert!(is_rref(&matrix));
//         let (reduced, profile) = rref_with_rank_profile(matrix);
//         assert!(is_rref(&reduced));
//         assert_eq!(profile, vec![0, 2, 5]);
//     }
// }

#[test]
fn test_echelon_form() {
    for _ in 0..100 {
        check_echelon_form_on_random_matrix(100, 100);
    }
    for _ in 0..100 {
        check_echelon_form_on_random_matrix(50, 100);
    }
}

fn check_echelon_form_on_random_matrix(nrows: usize, ncols: usize) {
    let array = random_bitmatrix(nrows, ncols);
    let echelon_form = EchelonForm::new(array.clone());
    assert!(is_rref(&echelon_form.matrix, &echelon_form.pivots));
    assert_eq!(&echelon_form.transform * &array, echelon_form.matrix);
    assert_eq!(
        &echelon_form.transform * &echelon_form.transform_inv_t.transposed(),
        AlignedBitMatrix::identity(array.row_count())
    );
}

#[test]
fn test_mul() {
    println!("0");
    let x = bitmatrix!(
        |01|
        |10|);
    let id = bitmatrix!(
        |10|
        |01|);
    println!("1");
    assert_eq!(&x * &x, id);
    assert_eq!(&x * &id, x);
    assert_eq!(&id * &x, x);

    // multiplication is associative
    println!("2");
    for _ in 0..100 {
        let a = random_bitmatrix(10, 10);
        let b = random_bitmatrix(10, 10);
        let c = random_bitmatrix(10, 10);
        assert_eq!(&(&a * &b) * &c, &a * &(&b * &c));
    }

    println!("3");
    // multiplication by zero is zero
    for _ in 0..100 {
        let a = random_bitmatrix(10, 10);
        let z = AlignedBitMatrix::zeros(10, 10);
        assert_eq!(&a * &z, z);
    }

    // multiplication by id
    for _ in 0..100 {
        let a = random_bitmatrix(3, 3);
        let id = AlignedBitMatrix::identity(3);
        assert_eq!(&a * &id, a);
    }
}

#[test]
fn test_mul_large_matrices() {
    // Test multiplication at sizes where M4RI optimizations matter
    for size in [100, 500, 1000] {
        let a = random_bitmatrix(size, size);
        let id = AlignedBitMatrix::identity(size);

        // A * I = A
        assert_eq!(&a * &id, a, "A * I != A for size {size}");

        // I * A = A
        assert_eq!(&id * &a, a, "I * A != A for size {size}");

        // A * 0 = 0
        let zero = AlignedBitMatrix::zeros(size, size);
        assert_eq!(&a * &zero, zero, "A * 0 != 0 for size {size}");
    }
}

#[test]
fn test_mul_transpose_large_matrices() {
    // Test mul_transpose at various sizes
    for size in [100, 500, 1000] {
        let a = random_bitmatrix(size, size);
        let b = random_bitmatrix(size, size);

        // A * B^T via mul_transpose should equal A * B.transposed()
        let result_transpose = a.mul_transpose(&b);
        let result_manual = &a * &b.transposed();
        assert_eq!(
            result_transpose, result_manual,
            "mul_transpose_m4ri != mul(transposed) for size {size}"
        );
    }
}

#[test]
fn test_mul_non_square_matrices() {
    // Test non-square matrix multiplication
    let sizes = [(100, 200, 150), (200, 100, 300), (500, 1000, 500)];

    for (rows_a, cols_a_rows_b, cols_b) in sizes {
        let a = random_bitmatrix(rows_a, cols_a_rows_b);
        let b = random_bitmatrix(cols_a_rows_b, cols_b);

        let result = &a * &b;
        assert_eq!(
            result.shape(),
            (rows_a, cols_b),
            "Wrong result shape for ({rows_a}x{cols_a_rows_b}) * ({cols_a_rows_b}x{cols_b})"
        );

        // Verify with identity: A * I = A
        let id = AlignedBitMatrix::identity(cols_a_rows_b);
        assert_eq!(&a * &id, a);
    }
}

#[test]
fn test_mul_associativity_large() {
    // Test associativity at larger sizes
    for size in [100, 300] {
        let a = random_bitmatrix(size, size);
        let b = random_bitmatrix(size, size);
        let c = random_bitmatrix(size, size);

        let left = &(&a * &b) * &c;
        let right = &a * &(&b * &c);
        assert_eq!(left, right, "(A*B)*C != A*(B*C) for size {size}");
    }
}

proptest::proptest! {
    #[test]
    fn test_echelon_form_solve(matrix in nonempty_bitmatrix(5)) {
        test_solve_of(&matrix);
    }

    #[test]
    fn test_echelon_form_transpose_solve(matrix in nonempty_bitmatrix(5)) {
        test_transpose_solve_of(&matrix);
    }
}

#[test]
fn test_kernel_basis() {
    let num_cols = 100;
    for _ in 0..100 {
        let mut matrix = random_bitmatrix(50, 100);
        let rrp = matrix.echelonize();
        let mut kernel_basis_matrix = kernel_basis_matrix(&matrix);
        let prod = &matrix * &kernel_basis_matrix.transposed();
        assert!(prod.is_zero());
        let rrpc = kernel_basis_matrix.echelonize();
        assert_eq!(rrp.len() + rrpc.len(), num_cols);
    }
}

fn preserves_rowspan_of(matrix: &AlignedBitMatrix, rref_matrix: &AlignedBitMatrix) -> bool {
    let profile = fast_profile_of(rref_matrix);
    let mut profile_rows = BTreeMap::new();
    for (row_index, column_index) in profile.iter().enumerate() {
        profile_rows.insert(column_index, row_index);
    }
    for row in matrix.rows() {
        let mut reduced = AlignedBitVec::from_view(&row);
        let support = row
            .support()
            .assume_sorted_by_item()
            .intersection(profile.iter().copied().assume_sorted_by_item());

        for column_index in support {
            let row_index = profile_rows[&column_index];
            let rref_row = AlignedBitVec::from_view(&rref_matrix.row(row_index));
            reduced.bitxor_assign(&rref_row);
        }
        if reduced.weight() > 0 {
            return false;
        }
    }
    true
}

fn is_rref(matrix: &AlignedBitMatrix, with_profile: &[usize]) -> bool {
    let expected_profile = fast_profile_of(matrix);
    (expected_profile == with_profile) && columns_are_pivots_of(matrix, with_profile)
}

fn columns_are_pivots_of(matrix: &AlignedBitMatrix, column_indexes: &[usize]) -> bool {
    for &column_index in column_indexes {
        let column = matrix.column(column_index);
        if column.weight() != 1 {
            return false;
        }
    }
    true
}

fn fast_profile_of(matrix: &AlignedBitMatrix) -> Vec<usize> {
    let mut profile = vec![];
    for row_index in 0..matrix.row_count() {
        let row = matrix.row(row_index);
        let pivot = row.into_iter().position(|bit| bit);
        if pivot.is_none() {
            break;
        }
        profile.push(pivot.unwrap());
    }
    profile
}

fn random_bitmatrix(row_count: usize, column_count: usize) -> AlignedBitMatrix {
    let mut matrix = AlignedBitMatrix::with_shape(row_count, column_count);
    let mut bits = std::iter::from_fn(move || Some(thread_rng().r#gen::<bool>()));
    for row_index in 0..row_count {
        for column_index in 0..column_count {
            matrix.set((row_index, column_index), bits.next().expect("boom"));
        }
    }
    for _ in 0..row_count {
        let from_index = thread_rng().gen_range(0..row_count);
        let to_index = thread_rng().gen_range(0..row_count);
        matrix.swap_rows(from_index, to_index);
    }
    matrix
}

#[test]
fn test_echelon_form_solve_examples() {
    let mut matrix = AlignedBitMatrix::zeros(2, 4);
    matrix.set((0, 2), true);
    matrix.set((0, 3), true);
    matrix.set((1, 0), true);
    matrix.set((1, 1), true);
    matrix.set((1, 2), true);
    matrix.set((1, 3), true);
    test_solve_of(&matrix);
}

#[test]
fn test_echelon_form_examples() {
    let mut matrix = AlignedBitMatrix::zeros(2, 4);
    matrix.set((0, 2), true);
    matrix.set((0, 3), true);
    matrix.set((1, 0), true);
    matrix.set((1, 1), true);
    matrix.set((1, 2), true);
    matrix.set((1, 3), true);
    let echelon_form = EchelonForm::new(matrix.clone());
    assert_eq!(matrix, &echelon_form.transform_inv_t * &echelon_form.matrix);
    assert_eq!(&echelon_form.transform * &matrix, echelon_form.matrix);
}

fn test_solve_of(matrix: &AlignedBitMatrix) {
    let echelon_form = EchelonForm::new(matrix.clone());
    let all_combinations = column_combinations_of(matrix);
    let target_count = 1 << matrix.row_count();

    for target_index in 0..target_count {
        let target = bitvec_from_usize(target_index, matrix.row_count());
        let solution = echelon_form.solve(&target.as_view());
        if all_combinations.contains(&target) {
            let solution_bitvec = solution.expect("Expected a solution but got None");
            let result = matrix * &solution_bitvec.as_view();
            assert_eq!(result, target);
        } else {
            assert!(solution.is_none());
        }
    }
}

fn test_transpose_solve_of(matrix: &AlignedBitMatrix) {
    let echelon_form = EchelonForm::new(matrix.clone());
    let transpose = matrix.transposed();
    let all_combinations = column_combinations_of(&transpose);
    let target_count = 1 << transpose.row_count();

    for target_index in 0..target_count {
        let target = bitvec_from_usize(target_index, transpose.row_count());
        let solution = echelon_form.transpose_solve(&target.as_view());
        if all_combinations.contains(&target) {
            let solution_bitvec = solution.expect("Expected a solution but got None");
            let result = &transpose * &solution_bitvec.as_view();
            assert_eq!(
                result, target,
                "Failed to verify solution for target {:?}, result: {:?}, solution: {:?}, matrix: \n{} \n rref: \n{}",
                target, result, solution_bitvec, matrix, echelon_form.matrix
            );
        } else {
            assert!(solution.is_none());
        }
    }
}

fn bitvec_from_usize(value: usize, size: usize) -> AlignedBitVec {
    let mut bitvec = AlignedBitVec::zeros(size);
    for bit in 0..size {
        if (value >> bit) & 1 == 1 {
            bitvec.assign_index(bit, true);
        }
    }
    bitvec
}

fn column_combinations_of(matrix: &AlignedBitMatrix) -> std::collections::HashSet<AlignedBitVec> {
    let mut all_combinations = std::collections::HashSet::new();
    let num_combinations = 1 << matrix.column_count(); // 2^column_count

    for combination in 0..num_combinations {
        let mut result = AlignedBitVec::zeros(matrix.row_count());

        // For each bit in the combination, add the corresponding column to the result
        for col in 0..matrix.column_count() {
            if (combination >> col) & 1 == 1 {
                // Add column `col` to the result using XOR (since we're in GF(2))
                for row in 0..matrix.row_count() {
                    if matrix.get((row, col)) {
                        result.assign_index(row, !result.index(row));
                    }
                }
            }
        }

        all_combinations.insert(result);
    }

    all_combinations
}

#[test]
fn transpose_kernel_test() {
    let mut random_data: [u64; 64] = [0; 64];
    let mut rng = rand::thread_rng();
    for value in &mut random_data {
        *value = rng.r#gen();
    }

    let mut transpose_data = random_data;
    binar::matrix::transpose_kernel::transpose_64x64_inplace(&mut transpose_data);
    for (i, &row) in random_data.iter().enumerate() {
        for (j, &column) in transpose_data.iter().enumerate() {
            let original_bit = (row >> j) & 1;
            let transposed_bit = (column >> i) & 1;
            assert_eq!(original_bit, transposed_bit, "Mismatch at ({i}, {j})");
        }
    }
}
