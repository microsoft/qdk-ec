use crate::matrix::column::Column;
use crate::matrix::{
    AlignedBitMatrix, AlignedEchelonForm, kernel_basis_matrix as aligned_kernel, row_stacked as aligned_row_stacked,
};
use crate::vec::Word;
use crate::{BitVec, BitView, BitViewMut};
use derive_more::{From, Into};
use std::cmp::PartialEq;
use std::hash::Hash;
use std::ops::Index;
use std::ops::{Add, AddAssign, BitAnd, BitAndAssign, BitXor, BitXorAssign, Mul};
use std::str::FromStr;

/// Result of reduced row echelon form computation with transforms
#[derive(Debug, Clone)]
pub struct EchelonForm {
    aligned: AlignedEchelonForm,
}

impl EchelonForm {
    #[must_use]
    pub fn new(matrix: BitMatrix) -> Self {
        Self {
            aligned: AlignedEchelonForm::new(matrix.aligned),
        }
    }

    /// Solve the linear system represented by this echelon form for a given right-hand side
    /// target.
    ///
    /// Given the original matrix A and right-hand side b, this solves Ax = b by finding
    /// the coefficients x.
    /// If the system has no solution (b is not in the column space of A), returns None.
    /// If the system has a solution, returns Some(BitVec).
    ///
    /// # Panics
    ///
    /// Panics if the target length does not equal the matrix column count.
    #[must_use]
    pub fn solve(&self, target: &BitView) -> Option<BitVec> {
        assert_eq!(target.len(), self.aligned.matrix.columncount());
        let solution = self.aligned.solve(&target.bits)?;
        Some(BitVec::from_aligned(self.aligned.matrix.rowcount(), solution))
    }

    /// Solve the linear system represented by the transpose of this echelon form for
    /// a given right-hand side target.
    ///
    /// Given the original matrix A and right-hand side b, this solves Aᵀx = b by finding
    /// the coefficients x.
    /// If the system has no solution (b is not in the row space of A), returns None.
    /// If the system has a solution, returns Some(BitVec).
    ///
    /// # Panics
    ///
    /// Panics if the target length does not equal the matrix row count.
    #[must_use]
    pub fn transpose_solve(&self, target: &BitView) -> Option<BitVec> {
        assert_eq!(target.len(), self.aligned.matrix.rowcount());
        let solution = self.aligned.transpose_solve(&target.bits)?;
        Some(BitVec::from_aligned(self.aligned.matrix.columncount(), solution))
    }
}

#[must_use]
#[derive(Eq, From, Into)]
pub struct BitMatrix {
    #[from]
    #[into]
    aligned: AlignedBitMatrix,
}

impl AsRef<AlignedBitMatrix> for BitMatrix {
    fn as_ref(&self) -> &AlignedBitMatrix {
        &self.aligned
    }
}

impl Hash for BitMatrix {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.aligned.hash(state);
    }
}

unsafe impl Sync for BitMatrix {}

pub type Row<'life> = BitView<'life>; // should we use View in the name to indicate that it is a view and not a copy of a row ?
pub type RowMut<'life> = BitViewMut<'life>; // should we use View in the name to indicate that it is a view and not a copy of a row ?

impl BitMatrix {
    pub fn with_shape(rows: usize, columns: usize) -> Self {
        Self::zeros(rows, columns)
    }

    pub fn zeros(rows: usize, columns: usize) -> Self {
        Self {
            aligned: AlignedBitMatrix::zeros(rows, columns),
        }
    }

    pub fn ones(rows: usize, columns: usize) -> Self {
        Self {
            aligned: AlignedBitMatrix::ones(rows, columns),
        }
    }

    pub fn identity(dimension: usize) -> Self {
        Self {
            aligned: AlignedBitMatrix::identity(dimension),
        }
    }

    pub fn from_row_iter<'life>(iter: impl ExactSizeIterator<Item = BitView<'life>>, columns: usize) -> Self {
        Self {
            aligned: AlignedBitMatrix::from_row_iter(iter.map(|view| view.bits.clone()), columns),
        }
    }

    pub fn from_iter<Row, Rows>(iter: Rows, columncount: usize) -> Self
    where
        Row: IntoIterator<Item = bool>,
        Rows: IntoIterator<Item = Row>,
    {
        Self {
            aligned: AlignedBitMatrix::from_iter(iter, columncount),
        }
    }

    pub fn from_aligned(aligned: AlignedBitMatrix) -> Self {
        Self { aligned }
    }

    /// View the matrix data as a flat slice of words (u64s).
    #[must_use]
    pub fn as_words(&self) -> &[Word] {
        self.aligned.as_words()
    }

    /// View the matrix data as a byte slice (native endianness).
    /// Use for fast serialization when endianness is known to match.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.aligned.as_bytes()
    }

    /// Deserialize a matrix from words (u64s).
    ///
    /// # Panics
    ///
    /// Panics if `words.len()` is not a multiple of `BIT_BLOCK_WORD_COUNT`.
    pub fn from_words(words: &[Word], columncount: usize) -> Self {
        Self {
            aligned: AlignedBitMatrix::from_words(words, columncount),
        }
    }

    /// Deserialize a matrix from bytes (native endianness).
    /// Use for fast deserialization when endianness is known to match.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` is not a multiple of `size_of::<BitBlock>()`.
    pub fn from_bytes(data: &[u8], columncount: usize) -> Self {
        Self {
            aligned: AlignedBitMatrix::from_bytes(data, columncount),
        }
    }

    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.aligned.is_zero()
    }

    #[must_use]
    pub fn rowcount(&self) -> usize {
        self.aligned.rowcount()
    }

    #[must_use]
    pub fn columncount(&self) -> usize {
        self.aligned.columncount()
    }

    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.aligned.shape()
    }

    /// Resize the matrix to new dimensions, preserving existing data.
    /// New rows/columns are filled with zeros.
    pub fn resize(&mut self, new_rows: usize, new_cols: usize) {
        self.aligned.resize(new_rows, new_cols);
    }

    pub fn row(&self, index: usize) -> Row<'_> {
        Row::from_aligned(self.columncount(), self.aligned.row(index))
    }

    #[must_use]
    pub fn rows(&self) -> impl ExactSizeIterator<Item = Row<'_>> {
        self.aligned
            .rows()
            .map(|aligned_row| Row::from_aligned(self.columncount(), aligned_row))
    }

    pub fn row_mut(&mut self, index: usize) -> RowMut<'_> {
        RowMut::from_aligned(self.columncount(), self.aligned.row_mut(index))
    }

    #[must_use]
    pub fn column(&self, index: usize) -> Column<'_> {
        self.aligned.column(index)
    }

    #[must_use]
    pub fn columns(&self) -> impl ExactSizeIterator<Item = Column<'_>> {
        self.aligned.columns()
    }

    /// # Panics
    ///
    /// Will panic if index out of range
    pub fn set(&mut self, index: (usize, usize), to: bool) {
        self.aligned.set(index, to);
    }

    /// # Safety
    /// Does not check if index is out of bounds
    pub unsafe fn set_unchecked(&mut self, index: (usize, usize), to: bool) {
        unsafe { self.aligned.set_unchecked(index, to) };
    }

    /// # Panics
    ///
    /// Will panic if index out of range
    #[must_use]
    pub fn get(&self, index: (usize, usize)) -> bool {
        self.aligned.get(index)
    }

    /// # Safety
    /// Does not check if index is out of bounds
    #[must_use]
    pub unsafe fn get_unchecked(&self, index: (usize, usize)) -> bool {
        unsafe { self.aligned.get_unchecked(index) }
    }

    pub fn echelonize(&mut self) -> Vec<usize> {
        self.aligned.echelonize()
    }

    #[must_use]
    pub fn rank(&self) -> usize {
        self.aligned.rank()
    }

    pub fn transposed(&self) -> Self {
        Self {
            aligned: self.aligned.transposed(),
        }
    }

    pub fn submatrix(&self, rows: &[usize], columns: &[usize]) -> Self {
        Self {
            aligned: self.aligned.submatrix(rows, columns),
        }
    }

    /// # Panics
    ///
    /// Will panic if matrix is not invertible
    pub fn inverted(&self) -> Self {
        Self {
            aligned: self.aligned.inverted(),
        }
    }

    pub fn swap_rows(&mut self, left_row_index: usize, right_row_index: usize) {
        self.aligned.swap_rows(left_row_index, right_row_index);
    }

    pub fn swap_columns(&mut self, left_column_index: usize, right_column_index: usize) {
        self.aligned.swap_columns(left_column_index, right_column_index);
    }

    pub fn permute_rows(&mut self, permutation: &[usize]) {
        self.aligned.permute_rows(permutation);
    }

    pub fn add_into_row(&mut self, to_index: usize, from_index: usize) {
        self.aligned.add_into_row(to_index, from_index);
    }

    /// Compute `self * other^T` without allocating a transposed matrix.
    ///
    /// This is equivalent to `self * other.transposed()` but avoids the
    /// transpose allocation by computing `row_i(self) · row_j(other)` directly.
    ///
    /// # Panics
    ///
    /// Panics if `self.columncount() != other.columncount()`.
    pub fn mul_transpose(&self, other: &Self) -> Self {
        Self {
            aligned: self.aligned.mul_transpose(&other.aligned),
        }
    }

    /// # Panics
    /// Will panic if the bitview dimensions are incompatible.
    pub fn right_multiply(&self, left: &BitView) -> BitVec {
        assert!(left.len() == self.rowcount());
        BitVec::from_aligned(self.columncount(), self.aligned.right_multiply(&left.bits))
    }

    pub fn kernel(&self) -> BitMatrix {
        let aligned = aligned_kernel(&self.aligned);
        BitMatrix::from_aligned(aligned)
    }
}

unsafe impl Send for BitMatrix {}

impl Clone for BitMatrix {
    fn clone(&self) -> Self {
        Self {
            aligned: self.aligned.clone(),
        }
    }
}

impl Index<[usize; 2]> for BitMatrix {
    type Output = bool;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        self.aligned.index(index)
    }
}

impl Index<(usize, usize)> for BitMatrix {
    type Output = bool;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.aligned.index(index)
    }
}

impl PartialEq for BitMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.aligned.eq(&other.aligned)
    }
}

impl AddAssign<&BitMatrix> for BitMatrix {
    fn add_assign(&mut self, other: &BitMatrix) {
        self.aligned.add_assign(&other.aligned);
    }
}

impl Add for &BitMatrix {
    type Output = BitMatrix;

    fn add(self, other: Self) -> Self::Output {
        let mut clone = (*self).clone();
        clone += other;
        clone
    }
}

impl BitXor for &BitMatrix {
    type Output = BitMatrix;

    fn bitxor(self, other: Self) -> Self::Output {
        self.add(other)
    }
}

impl BitXorAssign<&BitMatrix> for BitMatrix {
    fn bitxor_assign(&mut self, other: &BitMatrix) {
        self.add_assign(other);
    }
}

impl BitAndAssign<&BitMatrix> for BitMatrix {
    fn bitand_assign(&mut self, other: &Self) {
        self.aligned.bitand_assign(&other.aligned);
    }
}

impl BitAnd for &BitMatrix {
    type Output = BitMatrix;

    fn bitand(self, other: Self) -> Self::Output {
        let mut clone = (*self).clone();
        clone &= other;
        clone
    }
}

impl Mul for &BitMatrix {
    type Output = BitMatrix;

    fn mul(self, other: Self) -> Self::Output {
        Self::Output {
            aligned: &self.aligned * &other.aligned,
        }
    }
}

impl Mul<&BitView<'_>> for &BitMatrix {
    type Output = BitVec;

    fn mul(self, right: &BitView) -> Self::Output {
        assert!(right.len() == self.columncount());
        BitVec::from_aligned(self.rowcount(), &self.aligned * &right.bits)
    }
}

impl std::fmt::Display for BitMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.aligned.fmt(f)
    }
}

impl std::fmt::Debug for BitMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BitMatrix(shape={:?},value={:#})", self.shape(), self)
    }
}

impl FromStr for BitMatrix {
    type Err = usize;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self {
            aligned: AlignedBitMatrix::from_str(s)?,
        })
    }
}

pub fn row_stacked<'t, Matrices>(matrices: Matrices) -> BitMatrix
where
    Matrices: IntoIterator<Item = &'t BitMatrix>,
{
    let aligned = aligned_row_stacked(matrices.into_iter().map(|m| &m.aligned));
    BitMatrix::from_aligned(aligned)
}
