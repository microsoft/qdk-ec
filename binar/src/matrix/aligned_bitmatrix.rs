use crate::BitLength;
use crate::matrix::Column;
use crate::vec::{AlignedBitVec, AlignedBitView, AlignedBitViewMut};
use crate::vec::{BIT_BLOCK_WORD_COUNT, BitAccessor, BitBlock, Word};
use crate::{Bitwise, BitwiseMut, BitwisePair, BitwisePairMut};
use rand::Rng;
use sorted_iter::SortedIterator;
use sorted_iter::assume::AssumeSortedByItemExt;
use std::cmp::PartialEq;
use std::hash::Hash;
use std::iter::FromIterator;
use std::mem::size_of;
use std::ops::Index;
use std::ops::{Add, AddAssign, BitAnd, BitAndAssign, BitXor, BitXorAssign, Mul};

use std::str::FromStr;

/// Result of reduced row echelon form computation with transforms
#[derive(Debug, Clone)]
pub struct EchelonForm {
    /// The matrix in reduced row echelon form
    pub matrix: AlignedBitMatrix,
    /// Transform matrix T such that T * original = matrix
    pub transform: AlignedBitMatrix,
    /// Inverse transpose of the transform matrix
    pub transform_inv_t: AlignedBitMatrix,
    /// Column indices of the pivot positions (rank profile)
    pub pivots: Vec<usize>,
}

impl EchelonForm {
    #[must_use]
    pub fn new(mut matrix: AlignedBitMatrix) -> Self {
        let num_rows = matrix.row_count();
        let mut transform = AlignedBitMatrix::identity(num_rows);
        let mut transform_inv_t = AlignedBitMatrix::identity(num_rows);
        let mut pivot = pivot_of(&matrix, (0, 0));
        let mut rank_profile = Vec::<usize>::with_capacity(matrix.column_count());

        for row_index in 0..matrix.row_count() {
            if pivot.1 >= matrix.column_count() {
                break;
            }

            matrix.swap_rows(pivot.0, row_index);
            transform_inv_t.swap_rows(pivot.0, row_index);
            transform.swap_rows(pivot.0, row_index);

            pivot.0 = row_index;
            rank_profile.push(pivot.1);
            reduce_with_transforms(&mut matrix, &mut transform, &mut transform_inv_t, pivot);
            pivot = pivot_of(&matrix, (pivot.0 + 1, pivot.1 + 1));
        }

        Self {
            matrix,
            transform,
            transform_inv_t,
            pivots: rank_profile,
        }
    }

    /// Solve the linear system represented by this echelon form for a given right-hand side
    /// target.
    ///
    /// Given the original matrix A and right-hand side b, this solves Ax = b by finding
    /// the coefficients x.
    /// If the system has no solution (b is not in the column space of A), returns None.
    /// If the system has a solution, returns Some(AlignedBitVec).
    ///
    /// # Panics
    ///
    /// Panics if the target length exceeds the matrix column capacity.
    #[must_use]
    pub fn solve(&self, target: &AlignedBitView) -> Option<AlignedBitVec> {
        // Solve: A * x = target, or equivalently rref_matrix * x = transform * target.

        let (_, column_capacity) = self.matrix.capacity();
        assert!(
            target.len() <= column_capacity,
            "Target length {} exceeds matrix column capacity {}",
            target.len(),
            column_capacity
        );
        let transformed_target = &self.transform * target;
        solve_rref_system(&self.matrix, &self.pivots, &transformed_target.as_view())
    }

    /// Solve the linear system represented by the transpose of this echelon form for
    /// a given right-hand side target.
    ///
    /// Given the original matrix A and right-hand side b, this solves Aᵀx = b by finding
    /// the coefficients x.
    /// If the system has no solution (b is not in the row space of A), returns None.
    /// If the system has a solution, returns Some(AlignedBitVec).
    #[must_use]
    pub fn transpose_solve(&self, target: &AlignedBitView) -> Option<AlignedBitVec> {
        // Use A^T * x = b ⟺ x^T * A = b^T
        // and A = transform_inv * rref_matrix to get
        // (x^T * transform_inv) * rref_matrix = rhs^T.
        // First, solve y^T * rref_matrix = rhs^T for y^T.
        // Then use y^T =x^T transform_inv_t, or equivalently
        // y^T * transform = x^T.

        let rref_solution = transpose_solve_rref_system(&self.matrix, &self.pivots, target)?;
        Some(self.transform.right_multiply(&rref_solution.as_view()))
    }
}

fn solve_rref_system(matrix: &AlignedBitMatrix, pivots: &[usize], target: &AlignedBitView) -> Option<AlignedBitVec> {
    let mut residual_target = AlignedBitVec::from_view(target);
    let mut solution = AlignedBitVec::zeros(matrix.column_count());

    // Back-substitution: work backwards from the last pivot row to the first
    for row_index in (0..pivots.len()).rev() {
        if residual_target.index(row_index) {
            let column_index = pivots[row_index];
            let matrix_column = matrix.column(column_index);
            residual_target.bitxor_assign(&matrix_column);
            solution.assign_index(column_index, true);
        }
    }
    residual_target.is_zero().then_some(solution)
}

fn transpose_solve_rref_system(
    matrix: &AlignedBitMatrix,
    pivots: &[usize],
    target: &AlignedBitView,
) -> Option<AlignedBitVec> {
    let mut residual_target = AlignedBitVec::from_view(target);
    let mut solution = AlignedBitVec::zeros(matrix.row_count());

    for (row_index, column_index) in pivots.iter().enumerate() {
        if residual_target.index(*column_index) {
            let matrix_row = matrix.row(row_index);
            residual_target.bitxor_assign(&matrix_row);
            solution.assign_index(row_index, true);
        }
    }
    residual_target.is_zero().then_some(solution)
}

#[must_use]
#[derive(Eq)]
pub struct AlignedBitMatrix {
    blocks: Vec<BitBlock>,
    rows: Vec<*mut BitBlock>,
    column_count: usize,
}

impl Hash for AlignedBitMatrix {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.blocks.hash(state);
    }
}

unsafe impl Sync for AlignedBitMatrix {}

pub type Row<'life> = AlignedBitView<'life>; // should we use View in the name to indicate that it is a view and not a copy of a row ?
pub type MutableRow<'life> = AlignedBitViewMut<'life>; // should we use View in the name to indicate that it is a view and not a copy of a row ?

impl AlignedBitMatrix {
    pub fn with_shape(rows: usize, columns: usize) -> Self {
        Self::zeros(rows, columns)
    }

    pub fn zeros(row_count: usize, column_count: usize) -> Self {
        let rowstride = Self::rowstride_of(column_count);
        let buffer = vec![BitBlock::default(); row_count * rowstride];
        Self::from_blocks(buffer, (row_count, column_count))
    }

    pub fn ones(row_count: usize, column_count: usize) -> Self {
        let rowstride = Self::rowstride_of(column_count);
        let buffer = vec![BitBlock::ones(); row_count * rowstride];
        Self::from_blocks(buffer, (row_count, column_count))
    }

    pub fn identity(dimension: usize) -> Self {
        let mut res = Self::zeros(dimension, dimension);
        for index in 0..dimension {
            res.set((index, index), true);
        }
        res
    }

    pub fn from_row_iter<'life>(iter: impl ExactSizeIterator<Item = AlignedBitView<'life>>, columns: usize) -> Self {
        let rows = iter.len();
        let mut matrix = Self::zeros(rows, columns);
        for (row_from, mut row_to) in std::iter::zip(iter, matrix.row_iterator_mut(0..rows)) {
            row_to.assign(&row_from);
        }
        matrix
    }

    pub fn from_iter<Row, Rows>(iter: Rows, column_count: usize) -> Self
    where
        Row: IntoIterator<Item = bool>,
        Rows: IntoIterator<Item = Row>,
    {
        let mut rows = Vec::<Vec<bool>>::new();
        let mut row_count = 0;
        for row in iter {
            rows.push(row.into_iter().collect());
            row_count += 1;
        }
        let mut matrix = AlignedBitMatrix::with_shape(row_count, column_count);
        for (row_index, row) in rows.iter().enumerate() {
            for (column_index, value) in row.iter().take(column_count).enumerate() {
                matrix.set((row_index, column_index), *value);
            }
        }
        matrix
    }

    pub fn with_value(value: bool, shape: (usize, usize)) -> Self {
        if value {
            Self::ones(shape.0, shape.1)
        } else {
            Self::zeros(shape.0, shape.1)
        }
    }

    /// Create a random bit matrix with the given dimensions.
    ///
    /// Uses `rand::thread_rng()` for random number generation.
    pub fn random(rows: usize, columns: usize) -> Self {
        Self::random_with_rng(rows, columns, &mut rand::thread_rng())
    }

    /// Create a random bit matrix with the given dimensions using a provided RNG.
    ///
    /// Efficiently generates random bits by filling u64 words directly.
    /// Bits beyond `columns` in each row are guaranteed to be zero.
    pub fn random_with_rng<R: Rng>(rows: usize, columns: usize, rng: &mut R) -> Self {
        let mut result = Self::zeros(rows, columns);

        if result.blocks.is_empty() {
            return result;
        }

        let total_words = result.blocks.len() * BIT_BLOCK_WORD_COUNT;
        let words: &mut [Word] =
            unsafe { std::slice::from_raw_parts_mut(result.blocks.as_mut_ptr().cast::<Word>(), total_words) };
        rng.fill(words);

        let excess_bits = columns % BitBlock::BLOCK_BIT_LEN;
        if excess_bits > 0 {
            let mask = (1u64 << (excess_bits % Word::BITS as usize)) - 1;
            let words_to_keep = excess_bits / Word::BITS as usize;
            let rowstride_words = Self::rowstride_of(columns) * BIT_BLOCK_WORD_COUNT;
            for row in 0..rows {
                let row_start = row * rowstride_words;
                if mask != 0 {
                    words[row_start + words_to_keep] &= mask;
                }
                for word in &mut words[row_start + words_to_keep + 1..row_start + rowstride_words] {
                    *word = 0;
                }
            }
        }

        result
    }

    /// View the matrix data as a flat slice of words (u64).
    #[must_use]
    pub fn as_words(&self) -> &[Word] {
        unsafe {
            std::slice::from_raw_parts(
                self.blocks.as_ptr().cast::<Word>(),
                self.blocks.len() * BIT_BLOCK_WORD_COUNT,
            )
        }
    }

    /// View the matrix data as a byte slice (native endianness).
    /// Use for fast serialization when endianness is known to match.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.blocks.as_ptr().cast::<u8>(),
                self.blocks.len() * std::mem::size_of::<BitBlock>(),
            )
        }
    }

    /// Deserialize a matrix from a flat vector of words.
    /// The words should have been produced by `as_words`.
    ///
    /// # Panics
    ///
    /// Panics if `words.len()` is not a multiple of `BIT_BLOCK_WORD_COUNT`.
    pub fn from_words(words: &[Word], column_count: usize) -> Self {
        assert!(
            words.len().is_multiple_of(BIT_BLOCK_WORD_COUNT),
            "words length {} must be a multiple of BIT_BLOCK_WORD_COUNT ({})",
            words.len(),
            BIT_BLOCK_WORD_COUNT
        );

        let rowstride = Self::rowstride_of(column_count);
        let block_count = words.len() / BIT_BLOCK_WORD_COUNT;
        let row_count = block_count / rowstride;

        let blocks: Vec<BitBlock> = words
            .chunks_exact(BIT_BLOCK_WORD_COUNT)
            .map(|chunk| {
                let arr: [u64; BIT_BLOCK_WORD_COUNT] = chunk.try_into().unwrap();
                BitBlock::from(arr)
            })
            .collect();

        Self::from_blocks(blocks, (row_count, column_count))
    }

    /// Deserialize a matrix from bytes (native endianness).
    /// Use for fast deserialization when endianness is known to match.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` is not a multiple of `size_of::<BitBlock>()`.
    pub fn from_bytes(data: &[u8], column_count: usize) -> Self {
        let block_size = std::mem::size_of::<BitBlock>();
        assert!(
            data.len().is_multiple_of(block_size),
            "bytes length {} must be a multiple of BitBlock size ({})",
            data.len(),
            block_size
        );

        let block_count = data.len() / block_size;
        let rowstride = Self::rowstride_of(column_count);
        let row_count = block_count / rowstride;

        let blocks: Vec<BitBlock> = data
            .chunks_exact(block_size)
            .map(|chunk| {
                let mut block = BitBlock::default();
                for (i, word_bytes) in chunk.chunks_exact(8).enumerate() {
                    block.blocks[i] = u64::from_ne_bytes(word_bytes.try_into().unwrap());
                }
                block
            })
            .collect();

        Self::from_blocks(blocks, (row_count, column_count))
    }

    fn from_blocks(mut buffer: Vec<BitBlock>, shape: (usize, usize)) -> Self {
        let rows = Self::rows_of(buffer.as_mut_slice(), shape.0);
        Self::from_blocks_and_rows(buffer, shape, rows)
    }

    fn from_blocks_and_rows(buffer: Vec<BitBlock>, shape: (usize, usize), rows: Vec<*mut BitBlock>) -> Self {
        let matrix = Self {
            blocks: buffer,
            rows,
            column_count: shape.1,
        };
        debug_assert!(matrix.is_aligned());
        matrix
    }

    #[must_use]
    pub fn is_zero(&self) -> bool {
        let zero = BitBlock::default();
        for block in &self.blocks {
            if *block != zero {
                return false;
            }
        }
        true
    }

    fn is_aligned(&self) -> bool {
        let alignment = (self.blocks.as_ptr() as usize) % size_of::<BitBlock>();
        if alignment != 0 {
            return false;
        }
        for row in &self.rows {
            let alignment = (*row as usize) % size_of::<BitBlock>();
            if alignment != 0 {
                return false;
            }
        }
        true
    }

    fn rowstride(&self) -> usize {
        Self::rowstride_of(self.column_count)
    }

    pub(super) fn rowstride_of(column_count: usize) -> usize {
        let rowstride = column_count / BitBlock::BLOCK_BIT_LEN;
        let adjustment = !column_count.is_multiple_of(BitBlock::BLOCK_BIT_LEN);
        rowstride + usize::from(adjustment)
    }

    fn rows_of(blocks: &mut [BitBlock], row_count: usize) -> Vec<*mut BitBlock> {
        let mut rows = Vec::<*mut BitBlock>::new();
        let rowstride = if row_count == 0 { 0 } else { blocks.len() / row_count };
        if rowstride == 0 {
            rows = vec![blocks.as_mut_ptr(); row_count];
        } else {
            for row in blocks.chunks_exact_mut(rowstride) {
                rows.push(row.as_mut_ptr());
            }
        }
        rows
    }

    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    #[must_use]
    pub fn column_count(&self) -> usize {
        self.column_count
    }

    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.row_count(), self.column_count())
    }

    #[must_use]
    pub fn capacity(&self) -> (usize, usize) {
        (self.row_count(), self.rowstride() * BitBlock::BLOCK_BIT_LEN)
    }

    /// Resize the matrix to new dimensions, preserving existing data.
    /// New rows/columns are filled with zeros.
    pub fn resize(&mut self, new_rows: usize, new_cols: usize) {
        let old_rows = self.row_count();
        let old_cols = self.column_count();

        if new_rows == old_rows && new_cols == old_cols {
            return; // No-op
        }

        // Create new matrix with new dimensions
        let mut new_matrix = Self::zeros(new_rows, new_cols);

        // Copy old data
        for r in 0..old_rows {
            for c in 0..old_cols {
                let val = self.row(r).index(c);
                new_matrix.row_mut(r).assign_index(c, val);
            }
        }

        *self = new_matrix;
    }

    pub fn row(&self, index: usize) -> Row<'_> {
        Row {
            blocks: unsafe { std::slice::from_raw_parts(&raw const (*self.rows[index]), self.block_count()) },
        }
    }

    #[must_use]
    pub fn rows(&self) -> impl ExactSizeIterator<Item = Row<'_>> {
        self.row_iterator(0..self.row_count())
    }

    pub fn row_iterator(
        &self,
        index_iterator: impl ExactSizeIterator<Item = usize>,
    ) -> impl ExactSizeIterator<Item = Row<'_>> {
        index_iterator.map(|index| self.row(index))
    }

    pub fn row_iterator_mut(
        &mut self,
        index_iterator: impl ExactSizeIterator<Item = usize>,
    ) -> impl ExactSizeIterator<Item = MutableRow<'_>> {
        index_iterator.map(|index| self.build_mutable_row(index))
    }

    pub fn row_mut(&mut self, index: usize) -> MutableRow<'_> {
        self.build_mutable_row(index)
    }

    #[inline]
    fn block_count(&self) -> usize {
        let mut block_count = self.column_count() / BitBlock::BLOCK_BIT_LEN;
        if !self.column_count().is_multiple_of(BitBlock::BLOCK_BIT_LEN) {
            block_count += 1;
        }
        block_count
    }

    fn build_mutable_row(&self, index: usize) -> MutableRow<'_> {
        let ptr = self.rows[index];
        MutableRow {
            blocks: unsafe { std::slice::from_raw_parts_mut(&raw mut (*ptr), self.block_count()) },
        }
    }

    pub fn rows_mut(&mut self, index: usize, index2: usize) -> (MutableRow<'_>, MutableRow<'_>) {
        (self.build_mutable_row(index), self.build_mutable_row(index2))
    }

    pub fn rows2_mut(&mut self, index: (usize, usize)) -> (MutableRow<'_>, MutableRow<'_>) {
        (self.build_mutable_row(index.0), self.build_mutable_row(index.1))
    }

    pub fn rows2(&self, index: (usize, usize)) -> (Row<'_>, Row<'_>) {
        (self.row(index.0), self.row(index.1))
    }

    /// # Safety
    /// Does not check if all indexes are distinct
    pub unsafe fn rows4_mut(
        &mut self,
        index: (usize, usize, usize, usize),
    ) -> (MutableRow<'_>, MutableRow<'_>, MutableRow<'_>, MutableRow<'_>) {
        (
            self.build_mutable_row(index.0),
            self.build_mutable_row(index.1),
            self.build_mutable_row(index.2),
            self.build_mutable_row(index.3),
        )
    }

    /// # Safety
    /// Does not check if all indexes are distinct
    pub unsafe fn rows8_mut(&mut self, index: crate::Tuple8<usize>) -> crate::Tuple8<MutableRow<'_>> {
        (
            self.build_mutable_row(index.0),
            self.build_mutable_row(index.1),
            self.build_mutable_row(index.2),
            self.build_mutable_row(index.3),
            self.build_mutable_row(index.4),
            self.build_mutable_row(index.5),
            self.build_mutable_row(index.6),
            self.build_mutable_row(index.7),
        )
    }

    #[must_use]
    pub fn column(&self, index: usize) -> Column<'_> {
        let block_index = index / BitBlock::BLOCK_BIT_LEN;
        let bit_index = index % BitBlock::BLOCK_BIT_LEN;
        Column {
            rows: &self.rows,
            accessor: BitAccessor::for_index::<BitBlock>(bit_index),
            block_index,
        }
    }

    #[must_use]
    pub fn columns(&self) -> impl ExactSizeIterator<Item = Column<'_>> {
        let indexes = 0..self.column_count();
        indexes.map(|index| self.column(index))
    }

    /// # Panics
    ///
    /// Will panic if index out of range
    pub fn set(&mut self, index: (usize, usize), to: bool) {
        assert!(index.0 < self.row_count() && index.1 < self.column_count());
        unsafe { self.set_unchecked(index, to) };
    }

    /// # Safety
    /// Dose not check if index is out of bounds
    pub unsafe fn set_unchecked(&mut self, index: (usize, usize), to: bool) {
        let (block, bit_index) = self.block_index_of_mut(index);
        block.assign_index(bit_index, to);
    }

    /// # Panics
    ///
    /// Will panic if index out of range
    #[must_use]
    pub fn get(&self, index: (usize, usize)) -> bool {
        assert!(index.0 < self.row_count() && index.1 < self.column_count());
        unsafe { self.get_unchecked(index) }
    }

    /// # Safety
    /// Does not check if index is out of bounds
    #[must_use]
    pub unsafe fn get_unchecked(&self, index: (usize, usize)) -> bool {
        let (block, bit_index) = self.block_index_of(index);
        block.index(bit_index)
    }

    /// Toggle the bit at the given (row, column) index.
    ///
    /// This is more efficient than `row_mut(row).negate_index(col)` as it
    /// avoids creating a row view.
    ///
    /// # Panics
    ///
    /// Will panic if index is out of range.
    pub fn negate(&mut self, index: (usize, usize)) {
        assert!(index.0 < self.row_count() && index.1 < self.column_count());
        unsafe { self.negate_unchecked(index) };
    }

    /// Toggle the bit at the given (row, column) index without bounds checking.
    ///
    /// # Safety
    ///
    /// Does not check if index is out of bounds.
    pub unsafe fn negate_unchecked(&mut self, index: (usize, usize)) {
        let (block, bit_index) = self.block_index_of_mut(index);
        block.negate_index(bit_index);
    }

    pub fn echelonize(&mut self) -> Vec<usize> {
        let mut pivot = pivot_of(self, (0, 0));
        let mut rank_profile = Vec::<usize>::with_capacity(self.column_count());

        for row_index in 0..self.row_count() {
            if pivot.1 >= self.column_count() {
                break;
            }
            self.swap_rows(pivot.0, row_index);
            pivot.0 = row_index;
            rank_profile.push(pivot.1);
            reduce(self, pivot);
            pivot = pivot_of(self, (pivot.0 + 1, pivot.1 + 1));
        }
        rank_profile
    }

    #[must_use]
    pub fn rank(&self) -> usize {
        self.clone().echelonize().len()
    }
    pub fn transposed(&self) -> Self {
        const TILE_SIZE: usize = 64;
        use crate::matrix::transpose_kernel::transpose_64x64_inplace;

        let mut res = Self::with_shape(self.column_count(), self.row_count());
        let (full_row_blocks, remainder_rows) = (self.row_count() / TILE_SIZE, self.row_count() % TILE_SIZE);
        let (full_col_blocks, remainder_cols) = (self.column_count() / TILE_SIZE, self.column_count() % TILE_SIZE);

        // Fast path: transpose every 64x64 tile using the specialized kernel.
        let mut tile = [0u64; 64];
        for block_row in 0..full_row_blocks {
            for block_col in 0..full_col_blocks {
                read_64x64_tile(self, block_row, block_col, &mut tile);
                transpose_64x64_inplace(&mut tile);
                write_64x64_tile(&mut res, block_row, block_col, &tile);
            }
        }

        // Columns beyond the last full 64-wide tile: process them a block of 64 rows at a time.
        if remainder_cols > 0 {
            let mut tile = [0u64; 64];
            for block_row in 0..full_row_blocks {
                read_64x64_tile(self, block_row, full_col_blocks, &mut tile);
                transpose_64x64_inplace(&mut tile);
                write_64x64_tile_partial(&mut res, block_row, full_col_blocks, &tile, remainder_cols);
            }
        }

        if remainder_rows > 0 {
            // Rows beyond the last full 64-high tile for every fully processed column block.
            for block_col in 0..full_col_blocks {
                let mut tile = [0u64; 64];
                read_64x64_tile_partial(self, full_row_blocks, block_col, &mut tile, remainder_rows);
                transpose_64x64_inplace(&mut tile);
                write_64x64_tile(&mut res, full_row_blocks, block_col, &tile);
            }

            // Bottom-right corner where both row and column remainders are present.
            if remainder_cols > 0 {
                let mut tile = [0u64; 64];
                read_64x64_tile_partial(self, full_row_blocks, full_col_blocks, &mut tile, remainder_rows);
                transpose_64x64_inplace(&mut tile);
                write_64x64_tile_partial(&mut res, full_row_blocks, full_col_blocks, &tile, remainder_cols);
            }
        }

        res
    }

    #[must_use]
    pub(super) fn row_words(&self, index: usize) -> &[u64] {
        use crate::bit::bitblock::BIT_BLOCK_WORD_COUNT;
        let block_ptr = self.rows[index];
        let block_count = self.block_count();
        let word_ptr = block_ptr.cast_const().cast::<u64>();
        let word_len = block_count * BIT_BLOCK_WORD_COUNT;
        unsafe { std::slice::from_raw_parts(word_ptr, word_len) }
    }

    #[must_use]
    pub(super) fn row_words_mut(&mut self, index: usize) -> &mut [u64] {
        use crate::bit::bitblock::BIT_BLOCK_WORD_COUNT;
        let block_ptr = self.rows[index];
        let block_count = self.block_count();
        let word_ptr = block_ptr.cast::<u64>();
        let word_len = block_count * BIT_BLOCK_WORD_COUNT;
        unsafe { std::slice::from_raw_parts_mut(word_ptr, word_len) }
    }

    pub fn submatrix(&self, rows: &[usize], columns: &[usize]) -> Self {
        let mut res = Self::with_shape(rows.len(), columns.len());
        for (row_index, &row) in rows.iter().enumerate() {
            for (column_index, &column) in columns.iter().enumerate() {
                res.set((row_index, column_index), self[(row, column)]);
            }
        }
        res
    }

    /// # Panics
    ///
    /// Will panic if matrix is not invertible
    pub fn inverted(&self) -> AlignedBitMatrix {
        assert!(self.column_count() == self.row_count());
        let echelon_form = EchelonForm::new(self.clone());
        assert!(echelon_form.pivots.len() == self.row_count());
        debug_assert_eq!(
            self * &echelon_form.transform,
            AlignedBitMatrix::identity(self.row_count())
        );
        echelon_form.transform
    }

    pub fn swap_rows(&mut self, left_row_index: usize, right_row_index: usize) {
        self.rows.swap(left_row_index, right_row_index);
    }

    pub fn swap_columns(&mut self, left_column_index: usize, right_column_index: usize) {
        for row_index in 0..self.row_count() {
            let left_bit = self.get((row_index, left_column_index));
            let right_bit = self.get((row_index, right_column_index));
            self.set((row_index, left_column_index), right_bit);
            self.set((row_index, right_column_index), left_bit);
        }
    }

    pub fn permute_rows(&mut self, permutation: &[usize]) {
        let old_rows = self.rows.clone();
        for index in 0..permutation.len() {
            self.rows[index] = old_rows[permutation[index]];
        }
    }

    pub fn add_into_row(&mut self, to_index: usize, from_index: usize) {
        let mut to_block = self.rows[to_index];
        let mut from_block = self.rows[from_index];
        for _ in 0..self.rowstride() {
            unsafe {
                BitwisePairMut::bitxor_assign(&mut *to_block, &*from_block);
                to_block = to_block.add(1);
                from_block = from_block.add(1);
            }
        }
    }

    /// # Panics
    ///
    /// Will panic if matrix dimensions are incompatible
    pub fn dot(&self, rhs: &AlignedBitMatrix) -> AlignedBitMatrix {
        super::m4ri::mul(self, rhs)
    }

    /// # Panics
    /// Will panic if the bitview dimensions are incompatible.
    pub fn right_multiply(&self, left: &AlignedBitView) -> AlignedBitVec {
        assert!(left.len() >= self.row_count());
        let mut result = AlignedBitVec::zeros(self.column_count());
        for row_index in left.support() {
            let row = self.row(row_index);
            result.bitxor_assign(&row);
        }
        result
    }

    fn block_index_of_mut(&mut self, index: (usize, usize)) -> (&mut BitBlock, usize) {
        let column_block = index.1 / BitBlock::BLOCK_BIT_LEN;
        let column_remainder = index.1 % BitBlock::BLOCK_BIT_LEN;
        let bit_index = column_remainder % BitBlock::BLOCK_BIT_LEN;
        unsafe {
            let block = self.rows[index.0].add(column_block);
            (&mut *block, bit_index)
        }
    }

    fn block_index_of(&self, index: (usize, usize)) -> (&BitBlock, usize) {
        let column_block = index.1 / BitBlock::BLOCK_BIT_LEN;
        let column_remainder = index.1 % BitBlock::BLOCK_BIT_LEN;
        let bit_index = column_remainder % BitBlock::BLOCK_BIT_LEN;
        unsafe {
            let block = self.rows[index.0].add(column_block);
            (&*block, bit_index)
        }
    }
}

#[inline]
fn write_64x64_tile(matrix: &mut AlignedBitMatrix, block_row: usize, block_col: usize, tile: &[u64; 64]) {
    for (offset, &value) in tile.iter().enumerate() {
        let dest_words = matrix.row_words_mut(block_col * 64 + offset);
        dest_words[block_row] = value;
    }
}

#[inline]
fn write_64x64_tile_partial(
    matrix: &mut AlignedBitMatrix,
    block_row: usize,
    block_col: usize,
    tile: &[u64; 64],
    column_count: usize,
) {
    for (offset, &value) in tile.iter().enumerate().take(column_count) {
        let dest_words = matrix.row_words_mut(block_col * 64 + offset);
        dest_words[block_row] = value;
    }
}

#[inline]
fn read_64x64_tile(matrix: &AlignedBitMatrix, block_row: usize, block_col: usize, tile: &mut [u64; 64]) {
    for (offset, word) in tile.iter_mut().enumerate() {
        let src_words = matrix.row_words(block_row * 64 + offset);
        *word = src_words[block_col];
    }
}

#[inline]
fn read_64x64_tile_partial(
    matrix: &AlignedBitMatrix,
    block_row: usize,
    block_col: usize,
    tile: &mut [u64; 64],
    row_count: usize,
) {
    for (offset, word) in tile.iter_mut().enumerate().take(row_count) {
        let src_words = matrix.row_words(block_row * 64 + offset);
        *word = src_words[block_col];
    }
}

unsafe impl Send for AlignedBitMatrix {}

impl Clone for AlignedBitMatrix {
    fn clone(&self) -> Self {
        let mut blocks = self.blocks.clone();
        let mut rows = Vec::<*mut BitBlock>::new();
        let offset = unsafe { blocks.as_mut_ptr().offset_from(self.blocks.as_ptr()) };
        for row in &self.rows {
            rows.push(unsafe { row.offset(offset) });
        }
        AlignedBitMatrix::from_blocks_and_rows(blocks, self.shape(), rows)
    }
}

impl Index<[usize; 2]> for AlignedBitMatrix {
    type Output = bool;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self[(index[0], index[1])]
    }
}

impl Index<(usize, usize)> for AlignedBitMatrix {
    type Output = bool;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if self.get(index) {
            return &true;
        }
        &false
    }
}

impl PartialEq for AlignedBitMatrix {
    fn eq(&self, other: &Self) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        for irow in 0..self.row_count() {
            for icol in 0..self.column_count() {
                if self[(irow, icol)] != other[(irow, icol)] {
                    return false;
                }
            }
        }
        true
    }
}

impl AddAssign<&AlignedBitMatrix> for AlignedBitMatrix {
    fn add_assign(&mut self, other: &AlignedBitMatrix) {
        assert_eq!(self.shape(), other.shape());
        for index in 0..self.row_count() {
            self.row_mut(index).bitxor_assign(&other.row(index));
        }
    }
}

impl Add for &AlignedBitMatrix {
    type Output = AlignedBitMatrix;

    fn add(self, other: Self) -> Self::Output {
        let mut clone = (*self).clone();
        clone += other;
        clone
    }
}

impl BitXor for &AlignedBitMatrix {
    type Output = AlignedBitMatrix;

    fn bitxor(self, other: Self) -> Self::Output {
        self.add(other)
    }
}

impl BitXorAssign<&AlignedBitMatrix> for AlignedBitMatrix {
    fn bitxor_assign(&mut self, other: &AlignedBitMatrix) {
        self.add_assign(other);
    }
}

impl BitAndAssign<&AlignedBitMatrix> for AlignedBitMatrix {
    fn bitand_assign(&mut self, other: &Self) {
        assert_eq!(self.shape(), other.shape());
        for index in 0..self.row_count() {
            self.row_mut(index).bitand_assign(&other.row(index));
        }
    }
}

impl BitAnd for &AlignedBitMatrix {
    type Output = AlignedBitMatrix;

    fn bitand(self, other: Self) -> Self::Output {
        let mut clone = (*self).clone();
        clone &= other;
        clone
    }
}

impl Mul for &AlignedBitMatrix {
    type Output = AlignedBitMatrix;

    fn mul(self, other: Self) -> Self::Output {
        self.dot(other)
    }
}

impl AlignedBitMatrix {
    /// Compute `self * other^T`.
    ///
    /// # Panics
    ///
    /// Panics if `self.column_count() != other.column_count()`.
    pub fn mul_transpose(&self, other: &Self) -> Self {
        self.dot(&other.transposed())
    }
}

impl Mul<&AlignedBitView<'_>> for &AlignedBitMatrix {
    type Output = AlignedBitVec;

    fn mul(self, right: &AlignedBitView) -> Self::Output {
        assert!(right.len() >= self.column_count());
        let dots = self.rows().map(|row| row.dot(right));
        dots.collect()
    }
}

impl std::fmt::Display for AlignedBitMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            write!(f, "[")?;
        }
        for row_index in 0..self.row_count() {
            for column_index in 0..self.column_count() {
                let value = i32::from(self.get((row_index, column_index)));
                write!(f, "{value:?}")?;
            }
            if f.alternate() {
                write!(f, "|")?;
            } else {
                writeln!(f)?;
            }
        }
        if f.alternate() {
            write!(f, "]")?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for AlignedBitMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AlignedBitMatrix(shape={:?},value={:#})", self.shape(), self)
    }
}

impl FromStr for AlignedBitMatrix {
    type Err = usize;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut rows = Vec::<AlignedBitVec>::new();
        let mut column_count = 0;
        for row_string in s.split(&['|', '[', ']', '(', ')', ';', '\n']) {
            if !row_string.is_empty() {
                let mut res = Vec::<bool>::new();
                for char in row_string.chars() {
                    match char {
                        '0' | '.' | '▫' | '□' => res.push(false),
                        '1' | '▪' | '■' => res.push(true),
                        ' ' | '-' | ',' => {}
                        _ => return Err(0),
                    }
                }
                if !res.is_empty() {
                    column_count = column_count.max(res.len());
                    rows.push(res.into_iter().collect());
                }
            }
        }
        Ok(AlignedBitMatrix::from_iter(rows.iter(), column_count))
    }
}

/// # Panics
///
/// Should not panic.
pub fn row_stacked<'t, Matrices>(matrices: Matrices) -> AlignedBitMatrix
where
    Matrices: IntoIterator<Item = &'t AlignedBitMatrix>,
{
    let mut buffer = Vec::<BitBlock>::new();
    let mut column_count: Option<usize> = None;
    let mut row_count = 0;
    for matrix in matrices {
        debug_assert!(column_count.is_none() || column_count.unwrap() == matrix.column_count());
        buffer.append(&mut matrix.blocks.clone());
        column_count = Some(matrix.column_count());
        row_count += matrix.row_count();
    }
    AlignedBitMatrix::from_blocks(buffer, (row_count, *column_count.get_or_insert(0)))
}

pub fn directly_summed<'t, Matrices>(matrices: Matrices) -> AlignedBitMatrix
where
    Matrices: IntoIterator<Item = &'t AlignedBitMatrix>,
{
    let mut row_count = 0;
    let mut column_count = 0;
    let vec_matrices = Vec::from_iter(matrices);
    for matrix in &vec_matrices {
        row_count += matrix.row_count();
        column_count += matrix.column_count();
    }
    let mut sum = AlignedBitMatrix::zeros(row_count, column_count);
    let mut sum_row_offset = 0;
    let mut sum_column_offset = 0;
    for matrix in &vec_matrices {
        for row_index in 0..matrix.row_count() {
            for column_index in 0..matrix.column_count() {
                sum.set(
                    (row_index + sum_row_offset, column_index + sum_column_offset),
                    matrix[(row_index, column_index)],
                );
            }
        }
        sum_row_offset += matrix.row_count();
        sum_column_offset += matrix.column_count();
    }
    sum
}

fn pivot_of(matrix: &AlignedBitMatrix, starting_at: (usize, usize)) -> (usize, usize) {
    let (mut row_index, mut column_index) = starting_at;
    if row_index >= matrix.row_count() || column_index >= matrix.column_count() {
        return (row_index, column_index);
    }
    while !matrix.get((row_index, column_index)) {
        row_index += 1;
        if row_index == matrix.row_count() {
            column_index += 1;
            row_index = starting_at.0;
            if column_index == matrix.column_count() {
                break;
            }
        }
    }
    (row_index, column_index)
}

struct ReductionData {
    column_accessor: BitAccessor,
    blocks_per_row: usize,
    rowstride: usize,
    base_block: *const BitBlock,
    from_block: *mut BitBlock,
}

impl ReductionData {
    pub fn for_pivot(pivot: (usize, usize), within: &AlignedBitMatrix) -> Self {
        let start_block_offset = pivot.1 / BitBlock::BLOCK_BIT_LEN;
        let bit_index = pivot.1 % BitBlock::BLOCK_BIT_LEN;
        let from_block = unsafe { within.rows.get_unchecked(pivot.0).add(start_block_offset) };
        let base_block = unsafe { within.blocks.as_ptr().add(start_block_offset) };
        let rowstride = within.rowstride();

        ReductionData {
            column_accessor: BitAccessor::for_index::<BitBlock>(bit_index),
            blocks_per_row: rowstride - start_block_offset,
            rowstride,
            from_block,
            base_block,
        }
    }
}

fn reduce(matrix: &mut AlignedBitMatrix, from: (usize, usize)) {
    let data = ReductionData::for_pivot(from, matrix);
    let mut to_block = data.from_block;
    to_block = reduce_backward_until(data.base_block, to_block, &data);
    to_block = unsafe { to_block.add(data.rowstride * matrix.row_count()) };
    let until_block = unsafe { data.from_block.add(data.rowstride) };
    reduce_backward_until(until_block, to_block, &data);
}

fn reduce_backward_until(
    until_block: *const BitBlock,
    mut to_block: *mut BitBlock,
    data: &ReductionData,
) -> *mut BitBlock {
    while until_block != to_block {
        to_block = unsafe { to_block.sub(data.rowstride) };
        let column_value = unsafe { data.column_accessor.array_value_of(&(*to_block)) };
        if column_value {
            add_into_block(to_block, data.from_block, data.blocks_per_row);
        }
    }
    to_block
}

fn add_into_block(mut to_block: *mut BitBlock, mut from_block: *const BitBlock, block_count: usize) {
    for _ in 0..block_count {
        unsafe {
            <BitBlock as BitwisePairMut>::bitxor_assign(&mut *to_block, &*from_block);
            to_block = to_block.add(1);
            from_block = from_block.add(1);
        }
    }
}

fn reduce_with_transforms(
    matrix: &mut AlignedBitMatrix,
    transform: &mut AlignedBitMatrix,
    transform_inv_t: &mut AlignedBitMatrix,
    from: (usize, usize),
) {
    let row_count = matrix.row_count();
    for row_index in 0..from.0 {
        xor_if_column_with_transforms(from.1, matrix, transform, transform_inv_t, row_index, from.0);
    }
    for row_index in from.0 + 1..row_count {
        xor_if_column_with_transforms(from.1, matrix, transform, transform_inv_t, row_index, from.0);
    }
}

fn xor_if_column_with_transforms(
    column_index: usize,
    matrix: &mut AlignedBitMatrix,
    transform: &mut AlignedBitMatrix,
    transform_inv_t: &mut AlignedBitMatrix,
    row_index: usize,
    from_row_index: usize,
) {
    if matrix[(row_index, column_index)] {
        matrix.add_into_row(row_index, from_row_index);
        transform.add_into_row(row_index, from_row_index);
        transform_inv_t.add_into_row(from_row_index, row_index);
    }
}

pub fn kernel_basis_matrix(matrix: &AlignedBitMatrix) -> AlignedBitMatrix {
    let num_cols = matrix.column_count();
    let mut rr = matrix.clone();
    let rank_profile = rr.echelonize();
    let rank_profile_complement = complement(&rank_profile, num_cols);
    let res_row_count = num_cols - rank_profile.len();
    let mut res = AlignedBitMatrix::zeros(res_row_count, num_cols);
    for (index, elt) in rank_profile.into_iter().enumerate() {
        for (row_pos, col_src) in rank_profile_complement.iter().enumerate().take(res_row_count) {
            res.set((row_pos, elt), rr[(index, *col_src)]);
        }
    }
    for (index, position) in rank_profile_complement.into_iter().enumerate() {
        res.set((index, position), true);
    }
    res
}

#[must_use]
pub fn complement(v: &[usize], index_bound: usize) -> Vec<usize> {
    let values = v.iter().copied().assume_sorted_by_item();
    (0..index_bound).difference(values).collect()
}
