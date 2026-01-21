//! M4RI (Method of Four Russians) algorithm for GF(2) matrix multiplication.
//!
//! Binary matrix multiplication `C = A × B` is computed as a sum of stripe products:
//!
//! ```text
//!   C = Σᵢ A[:,iK:(i+1)K] × B[iK:(i+1)K,:]
//! ```
//!
//! For each stripe, we precompute all 2^K linear combinations of K rows of B into a
//! lookup table (using Gray code for efficiency). Each row of A contributes a K-bit
//! pattern that indexes into this table.
//!
//! A is transposed upfront so that K-bit patterns can be extracted efficiently using
//! an 8×64 bit transpose (yielding 64 patterns at a time).
//!
//! With K=8, this is faster than naive multiplication when matrices are large enough
//! for the table construction overhead to be amortized.
//!
//! # References
//!
//! - Arlazarov, Dinic, Kronrod, Faradzev (1970): Original "Method of Four Russians"
//! - M4RI library: <https://github.com/malb/m4ri>
//! - Albrecht, Bard, Hart (2008): <https://arxiv.org/abs/0811.1714>

use super::aligned_bitmatrix::{AlignedBitMatrix, Row};
use super::transpose_kernel::transpose_8x64;
use crate::{BitBlock, BitwisePairMut};

/// Inner stripe width: 2^8 = 256 table entries per sub-table.
const INNER_STRIPE_BITS: usize = 8;

/// Super-stripe width: 4 inner stripes processed together for cache efficiency.
const SUPER_STRIPE_COUNT: usize = 4;

/// Total bits per super-stripe.
const SUPER_STRIPE_BITS: usize = INNER_STRIPE_BITS * SUPER_STRIPE_COUNT;

/// Compute `left * right` using M4RI (Method of Four Russians).
///
/// # Panics
///
/// Panics if `left.column_count() != right.row_count()`.
#[allow(clippy::similar_names)]
pub fn mul(left: &AlignedBitMatrix, right: &AlignedBitMatrix) -> AlignedBitMatrix {
    assert_eq!(left.column_count(), right.row_count());

    let num_result_rows = left.row_count();
    let inner_dimension = left.column_count();
    let num_result_cols = right.column_count();

    if inner_dimension == 0 || num_result_cols == 0 {
        return AlignedBitMatrix::zeros(num_result_rows, num_result_cols);
    }

    let left_transposed = left.transposed();
    let mut result = AlignedBitMatrix::zeros(num_result_rows, num_result_cols);
    let mut row_table = RowStripeTable::new(num_result_cols);

    let num_stripes = inner_dimension.div_ceil(SUPER_STRIPE_BITS);
    for stripe_index in 0..num_stripes {
        let bit_start = stripe_index * SUPER_STRIPE_BITS;
        let stripe_bits = SUPER_STRIPE_BITS.min(inner_dimension - bit_start);

        row_table.populate(right, bit_start, stripe_bits);

        let column_patterns = StripeColumnExtractor::new(&left_transposed, bit_start, stripe_bits);
        multiply_stripes(column_patterns, &row_table, &mut result);
    }

    result
}

/// Lookup tables for a super-stripe.
///
/// Each sub-table maps 8-bit patterns to precomputed row XOR combinations.
struct RowStripeTable {
    sub_tables: [AlignedBitMatrix; SUPER_STRIPE_COUNT],
}

impl RowStripeTable {
    fn new(num_cols: usize) -> Self {
        let num_patterns = 1 << INNER_STRIPE_BITS;
        Self {
            sub_tables: std::array::from_fn(|_| AlignedBitMatrix::zeros(num_patterns, num_cols)),
        }
    }

    /// Populate all sub-tables using Gray code.
    fn populate(&mut self, matrix: &AlignedBitMatrix, bit_start: usize, stripe_bits: usize) {
        let num_inner_stripes = stripe_bits.div_ceil(INNER_STRIPE_BITS);

        for inner_idx in 0..num_inner_stripes {
            let inner_bit_start = bit_start + inner_idx * INNER_STRIPE_BITS;
            let inner_bits = INNER_STRIPE_BITS.min(stripe_bits - inner_idx * INNER_STRIPE_BITS);

            let num_patterns = 1usize << inner_bits;
            let table = &mut self.sub_tables[inner_idx];

            table.row_mut(0).blocks.fill(BitBlock::default());

            let mut previous_gray = 0usize;

            for pattern in 1..num_patterns {
                let current_gray = pattern ^ (pattern >> 1);
                let changed_bit = (previous_gray ^ current_gray).trailing_zeros() as usize;
                let src_row = matrix.row(inner_bit_start + changed_bit);

                let (prev_row, mut curr_row) = table.rows_mut(previous_gray, current_gray);
                curr_row.blocks.clone_from_slice(prev_row.blocks);
                curr_row.bitxor_assign(&src_row);

                previous_gray = current_gray;
            }
        }
    }

    #[inline]
    fn get(&self, table_idx: usize, pattern: usize) -> Row<'_> {
        self.sub_tables[table_idx].row(pattern)
    }
}

/// Extracts column patterns from a transposed matrix.
///
/// Yields arrays of 8-bit patterns, one per inner stripe.
struct StripeColumnExtractor<'a> {
    stripe_rows: [&'a [u64]; SUPER_STRIPE_BITS],
    num_columns: usize,
    num_inner_stripes: usize,
    stripe_bits: usize,
    current_column: usize,
    cached_patterns: [[u64; 8]; SUPER_STRIPE_COUNT],
}

impl<'a> StripeColumnExtractor<'a> {
    fn new(transposed_matrix: &'a AlignedBitMatrix, bit_start: usize, stripe_bits: usize) -> Self {
        let num_columns = transposed_matrix.column_count();
        let num_inner_stripes = stripe_bits.div_ceil(INNER_STRIPE_BITS);

        let mut stripe_rows = [transposed_matrix.row_words(bit_start); SUPER_STRIPE_BITS];
        for (row_ref, offset) in stripe_rows.iter_mut().zip(0..stripe_bits).skip(1) {
            *row_ref = transposed_matrix.row_words(bit_start + offset);
        }

        let mut extractor = Self {
            stripe_rows,
            num_columns,
            num_inner_stripes,
            stripe_bits,
            current_column: 0,
            cached_patterns: [[0; 8]; SUPER_STRIPE_COUNT],
        };

        if num_columns > 0 {
            extractor.load_block(0);
        }

        extractor
    }

    fn load_block(&mut self, word_index: usize) {
        for inner_idx in 0..self.num_inner_stripes {
            let row_offset = inner_idx * INNER_STRIPE_BITS;
            let inner_bits = INNER_STRIPE_BITS.min(self.stripe_bits.saturating_sub(row_offset));

            let mut input = [0u64; 8];
            for (slot, bit) in input.iter_mut().zip(0..inner_bits) {
                *slot = self.stripe_rows[row_offset + bit][word_index];
            }
            self.cached_patterns[inner_idx] = transpose_8x64(input);
        }
    }

    #[inline]
    fn current_patterns(&self) -> [usize; SUPER_STRIPE_COUNT] {
        let col_in_block = self.current_column % 64;
        let word_index = col_in_block / 8;
        let shift = (col_in_block % 8) * 8;

        std::array::from_fn(|inner_idx| ((self.cached_patterns[inner_idx][word_index] >> shift) & 0xFF) as usize)
    }

    #[inline]
    fn increment_column(&mut self) {
        self.current_column += 1;
        if self.current_column.is_multiple_of(64) && self.current_column < self.num_columns {
            self.load_block(self.current_column / 64);
        }
    }
}

impl Iterator for StripeColumnExtractor<'_> {
    type Item = [usize; SUPER_STRIPE_COUNT];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_column >= self.num_columns {
            return None;
        }
        let patterns = self.current_patterns();
        self.increment_column();
        Some(patterns)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.num_columns - self.current_column;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for StripeColumnExtractor<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.num_columns - self.current_column
    }
}

fn multiply_stripes(
    column_patterns: StripeColumnExtractor<'_>,
    row_table: &RowStripeTable,
    result: &mut AlignedBitMatrix,
) {
    let num_inner_stripes = column_patterns.num_inner_stripes;

    for (row_index, patterns) in column_patterns.enumerate() {
        let mut result_row = result.row_mut(row_index);
        for (inner_idx, &pattern) in patterns.iter().take(num_inner_stripes).enumerate() {
            let contribution = row_table.get(inner_idx, pattern);
            result_row.bitxor_assign(&contribution);
        }
    }
}
