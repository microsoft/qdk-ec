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

/// Result of reduced row echelon form computation.
///
/// `EchelonForm` represents a matrix in reduced row echelon form (RREF) and provides
/// methods for solving linear systems over GF(2). This form is useful for:
///
/// - Solving systems of linear equations (Ax = b)
/// - Computing matrix rank
/// - Finding null space (kernel)
/// - Testing linear independence
///
/// # Example
///
/// ```
/// use binar::{BitMatrix, BitVec, Bitwise, EchelonForm};
///
/// // Create a 3x3 linear system
/// let rows = vec![
///     BitVec::from_iter([true, true, false]),
///     BitVec::from_iter([false, true, true]),
///     BitVec::from_iter([true, false, true]),
/// ];
/// let m = BitMatrix::from_row_iter(rows.iter().map(|r| r.as_view()), 3);
/// let echelon = EchelonForm::new(m);
///
/// // Solve Ax = b over GF(2)
/// let b = BitVec::from_iter([true, true, false]);
/// let solution = echelon.solve(&b.as_view());
/// assert!(solution.is_some());
/// ```
///
/// See also [`EchelonForm::new`] for creating an echelon form.
#[derive(Debug, Clone)]
pub struct EchelonForm {
    aligned: AlignedEchelonForm,
}

impl EchelonForm {
    /// Creates a new `EchelonForm` by reducing the given matrix to row echelon form.
    ///
    /// This constructor computes the reduced row echelon form (RREF) of the matrix,
    /// which can then be used to solve linear systems efficiently.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::{BitMatrix, BitVec, EchelonForm};
    ///
    /// let m = BitMatrix::identity(3);
    /// let echelon = EchelonForm::new(m);
    ///
    /// // Solve a linear system
    /// let b = BitVec::ones(3);
    /// let solution = echelon.solve(&b.as_view());
    /// ```
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
        assert_eq!(target.len(), self.aligned.matrix.column_count());
        let solution = self.aligned.solve(&target.bits)?;
        Some(BitVec::from_aligned(self.aligned.matrix.row_count(), solution))
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
        assert_eq!(target.len(), self.aligned.matrix.row_count());
        let solution = self.aligned.transpose_solve(&target.bits)?;
        Some(BitVec::from_aligned(self.aligned.matrix.column_count(), solution))
    }
}

/// A 2D matrix of bits with a convenient, user-friendly API.
///
/// `BitMatrix` is the primary matrix type in this crate, wrapping [`AlignedBitMatrix`]
/// to provide an easier-to-use interface while maintaining the same performance characteristics.
/// It stores a matrix of bits (0 or 1) with cache-aligned memory for efficient operations,
/// particularly optimized for linear algebra over GF(2).
///
/// # When to Use
///
/// Use `BitMatrix` for:
/// - Linear algebra over GF(2) (the binary field)
/// - Representing linear transformations in discrete systems
/// - Solving systems of linear equations mod 2
/// - Computing matrix rank, kernel, and inverses
///
/// Consider [`matrix::AlignedBitMatrix`](crate::matrix::AlignedBitMatrix) directly when you
/// need more control over memory layout or specific performance tuning.
///
/// # Construction
///
/// ```
/// use binar::BitMatrix;
///
/// // Create from dimensions
/// let zeros = BitMatrix::zeros(10, 20);
/// let ones = BitMatrix::ones(10, 20);
/// let identity = BitMatrix::identity(10);
///
/// // Create from row iterators
/// let rows = vec![
///     vec![true, false, true],
///     vec![false, true, false],
/// ];
/// let matrix = BitMatrix::from_iter(rows, 3);
/// ```
///
/// # Accessing Elements
///
/// ```
/// use binar::{BitMatrix, Bitwise, BitwiseMut};
///
/// let mut m = BitMatrix::zeros(5, 5);
///
/// // Get and set individual elements
/// m.set((2, 3), true);
/// assert_eq!(m.get((2, 3)), true);
/// assert_eq!(m[(2, 3)], true);  // Index notation also works
///
/// // Access rows (as views)
/// let row = m.row(2);
/// assert_eq!(row.index(3), true);
///
/// // Mutable row access
/// let mut row = m.row_mut(2);
/// row.assign_index(4, true);
/// ```
///
/// # Linear Algebra Operations
///
/// ```
/// use binar::BitMatrix;
///
/// let mut m = BitMatrix::identity(3);
/// m.set((0, 1), true);
/// m.set((1, 2), true);
///
/// // Matrix multiplication (over GF(2))
/// let product = &m * &m;
///
/// // Addition (XOR, since we're in GF(2))
/// let sum = &m ^ &m;  // XOR all elements
///
/// // Compute rank
/// let rank = m.rank();
///
/// // Compute kernel (null space)
/// let kernel = m.kernel();
///
/// // Transpose
/// let transposed = m.transposed();
///
/// // Invert (if square and full rank)
/// let inverse = m.inverted();
/// assert_eq!(&m * &inverse, BitMatrix::identity(3));
/// ```
///
/// # Row Echelon Form
///
/// ```
/// use binar::{BitMatrix, BitVec, EchelonForm};
///
/// let mut m = BitMatrix::from_iter(
///     vec![
///         vec![true, false, true],
///         vec![false, true, false],
///         vec![true, true, false],
///     ],
///     3,
/// );
///
/// // Reduce to echelon form
/// let echelon = EchelonForm::new(m.clone());
///
/// // Solve linear system Ax = b
/// let b = BitVec::from_iter([true, false, true]);
/// if let Some(solution) = echelon.solve(&b.as_view()) {
///     println!("Found solution!");
/// }
/// ```
///
/// # Serialization
///
/// For efficient serialization, use [`as_words`](BitMatrix::as_words) or
/// [`as_bytes`](BitMatrix::as_bytes):
///
/// ```
/// use binar::BitMatrix;
///
/// let m = BitMatrix::identity(64);
/// let words = m.as_words();
/// let restored = BitMatrix::from_words(words, 64);
/// assert_eq!(m, restored);
/// ```
///
/// # See Also
///
/// - [`matrix::AlignedBitMatrix`](crate::matrix::AlignedBitMatrix) - The underlying aligned type
/// - [`BitVec`](crate::BitVec) - 1D bit vector, used for rows
/// - [`EchelonForm`] - Reduced row echelon form with solving capabilities
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
    /// Creates a matrix with the given shape (alias for [`zeros`](BitMatrix::zeros)).
    pub fn with_shape(rows: usize, columns: usize) -> Self {
        Self::zeros(rows, columns)
    }

    /// Creates a new matrix with all bits set to zero.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::BitMatrix;
    ///
    /// let m = BitMatrix::zeros(10, 20);
    /// assert_eq!(m.shape(), (10, 20));
    /// assert!(m.is_zero());
    /// ```
    pub fn zeros(rows: usize, columns: usize) -> Self {
        Self {
            aligned: AlignedBitMatrix::zeros(rows, columns),
        }
    }

    /// Creates a new matrix with all bits set to one.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::BitMatrix;
    ///
    /// let m = BitMatrix::ones(3, 4);
    /// assert_eq!(m.get((0, 0)), true);
    /// assert_eq!(m.get((2, 3)), true);
    /// ```
    pub fn ones(rows: usize, columns: usize) -> Self {
        Self {
            aligned: AlignedBitMatrix::ones(rows, columns),
        }
    }

    /// Creates an identity matrix of the given dimension.
    ///
    /// An identity matrix has ones on the main diagonal and zeros elsewhere.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::BitMatrix;
    ///
    /// let id = BitMatrix::identity(3);
    /// assert_eq!(id.get((0, 0)), true);
    /// assert_eq!(id.get((1, 1)), true);
    /// assert_eq!(id.get((0, 1)), false);
    /// ```
    pub fn identity(dimension: usize) -> Self {
        Self {
            aligned: AlignedBitMatrix::identity(dimension),
        }
    }

    /// Creates a matrix from an iterator of row views.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::{BitMatrix, BitVec};
    ///
    /// let rows = vec![BitVec::zeros(5), BitVec::ones(5)];
    /// let views: Vec<_> = rows.iter().map(|r| r.as_view()).collect();
    /// let m = BitMatrix::from_row_iter(views.into_iter(), 5);
    /// assert_eq!(m.shape(), (2, 5));
    /// ```
    pub fn from_row_iter<'life>(iter: impl ExactSizeIterator<Item = BitView<'life>>, columns: usize) -> Self {
        Self {
            aligned: AlignedBitMatrix::from_row_iter(iter.map(|view| view.bits.clone()), columns),
        }
    }

    /// Creates a matrix from nested iterators of boolean values.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::BitMatrix;
    ///
    /// let rows = vec![
    ///     vec![true, false, true],
    ///     vec![false, true, false],
    /// ];
    /// let m = BitMatrix::from_iter(rows, 3);
    /// assert_eq!(m.shape(), (2, 3));
    /// assert_eq!(m.get((0, 0)), true);
    /// ```
    pub fn from_iter<Row, Rows>(iter: Rows, column_count: usize) -> Self
    where
        Row: IntoIterator<Item = bool>,
        Rows: IntoIterator<Item = Row>,
    {
        Self {
            aligned: AlignedBitMatrix::from_iter(iter, column_count),
        }
    }

    /// Creates a `BitMatrix` from an `AlignedBitMatrix`.
    ///
    /// This is useful when working directly with [`AlignedBitMatrix`] and needing
    /// to convert to the more convenient `BitMatrix` API.
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
    pub fn from_words(words: &[Word], column_count: usize) -> Self {
        Self {
            aligned: AlignedBitMatrix::from_words(words, column_count),
        }
    }

    /// Deserialize a matrix from bytes (native endianness).
    /// Use for fast deserialization when endianness is known to match.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` is not a multiple of `size_of::<BitBlock>()`.
    pub fn from_bytes(data: &[u8], column_count: usize) -> Self {
        Self {
            aligned: AlignedBitMatrix::from_bytes(data, column_count),
        }
    }

    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.aligned.is_zero()
    }

    /// Returns the number of rows in the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::BitMatrix;
    ///
    /// let m = BitMatrix::zeros(10, 20);
    /// assert_eq!(m.row_count(), 10);
    /// ```
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.aligned.row_count()
    }

    /// Returns the number of columns in the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::BitMatrix;
    ///
    /// let m = BitMatrix::zeros(10, 20);
    /// assert_eq!(m.column_count(), 20);
    /// ```
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.aligned.column_count()
    }

    /// Returns the matrix dimensions as `(rows, columns)`.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::BitMatrix;
    ///
    /// let m = BitMatrix::zeros(10, 20);
    /// assert_eq!(m.shape(), (10, 20));
    /// ```
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.aligned.shape()
    }

    /// Resizes the matrix to new dimensions, preserving existing data.
    ///
    /// New rows and columns are filled with zeros. If the matrix is shrunk,
    /// data is truncated.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::BitMatrix;
    ///
    /// let mut m = BitMatrix::identity(3);
    /// m.resize(5, 5);
    /// assert_eq!(m.shape(), (5, 5));
    /// assert_eq!(m.get((0, 0)), true);  // Preserved
    /// assert_eq!(m.get((4, 4)), false); // New element
    /// ```
    pub fn resize(&mut self, new_rows: usize, new_cols: usize) {
        self.aligned.resize(new_rows, new_cols);
    }

    /// Returns a view of the specified row.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::{BitMatrix, Bitwise};
    ///
    /// let m = BitMatrix::identity(5);
    /// let row0 = m.row(0);
    /// assert_eq!(row0.index(0), true);
    /// assert_eq!(row0.weight(), 1);
    /// ```
    pub fn row(&self, index: usize) -> Row<'_> {
        Row::from_aligned(self.column_count(), self.aligned.row(index))
    }

    /// Returns an iterator over all rows.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::{BitMatrix, Bitwise};
    ///
    /// let m = BitMatrix::identity(3);
    /// let weights: Vec<_> = m.rows().map(|row| row.weight()).collect();
    /// assert_eq!(weights, vec![1, 1, 1]);
    /// ```
    #[must_use]
    pub fn rows(&self) -> impl ExactSizeIterator<Item = Row<'_>> {
        self.aligned
            .rows()
            .map(|aligned_row| Row::from_aligned(self.column_count(), aligned_row))
    }

    /// Returns a mutable view of the specified row.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::{BitMatrix, BitwiseMut};
    ///
    /// let mut m = BitMatrix::zeros(3, 5);
    /// let mut row = m.row_mut(1);
    /// row.assign_index(2, true);
    /// drop(row);
    /// assert_eq!(m.get((1, 2)), true);
    /// ```
    pub fn row_mut(&mut self, index: usize) -> RowMut<'_> {
        RowMut::from_aligned(self.column_count(), self.aligned.row_mut(index))
    }

    /// Returns a view of the specified column.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::{BitMatrix, Bitwise};
    ///
    /// let m = BitMatrix::identity(5);
    /// let col2 = m.column(2);
    /// assert_eq!(col2.index(2), true);
    /// assert_eq!(col2.weight(), 1);
    /// ```
    #[must_use]
    pub fn column(&self, index: usize) -> Column<'_> {
        self.aligned.column(index)
    }

    /// Returns an iterator over all columns.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::{BitMatrix, Bitwise};
    ///
    /// let m = BitMatrix::identity(3);
    /// let weights: Vec<_> = m.columns().map(|col| col.weight()).collect();
    /// assert_eq!(weights, vec![1, 1, 1]);
    /// ```
    #[must_use]
    pub fn columns(&self) -> impl ExactSizeIterator<Item = Column<'_>> {
        self.aligned.columns()
    }

    /// Sets the bit at the given position.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::BitMatrix;
    ///
    /// let mut m = BitMatrix::zeros(3, 3);
    /// m.set((1, 2), true);
    /// assert_eq!(m.get((1, 2)), true);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn set(&mut self, index: (usize, usize), to: bool) {
        self.aligned.set(index, to);
    }

    /// Sets the bit at the given position without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is within bounds.
    pub unsafe fn set_unchecked(&mut self, index: (usize, usize), to: bool) {
        unsafe { self.aligned.set_unchecked(index, to) };
    }

    /// Gets the bit at the given position.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::BitMatrix;
    ///
    /// let m = BitMatrix::identity(3);
    /// assert_eq!(m.get((0, 0)), true);
    /// assert_eq!(m.get((0, 1)), false);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    #[must_use]
    pub fn get(&self, index: (usize, usize)) -> bool {
        self.aligned.get(index)
    }

    /// Gets the bit at the given position without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is within bounds.
    #[must_use]
    pub unsafe fn get_unchecked(&self, index: (usize, usize)) -> bool {
        unsafe { self.aligned.get_unchecked(index) }
    }

    /// Reduces the matrix to row echelon form in place.
    ///
    /// Returns the pivot column indices.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::BitMatrix;
    ///
    /// let mut m = BitMatrix::from_iter(
    ///     vec![
    ///         vec![true, false, true],
    ///         vec![false, true, false],
    ///     ],
    ///     3,
    /// );
    /// let pivots = m.echelonize();
    /// assert_eq!(pivots.len(), 2);
    /// ```
    pub fn echelonize(&mut self) -> Vec<usize> {
        self.aligned.echelonize()
    }

    /// Computes the rank of the matrix (dimension of the row/column space).
    ///
    /// # Example
    ///
    /// ```
    /// use binar::BitMatrix;
    ///
    /// let id = BitMatrix::identity(5);
    /// assert_eq!(id.rank(), 5);
    ///
    /// let singular = BitMatrix::zeros(5, 5);
    /// assert_eq!(singular.rank(), 0);
    /// ```
    #[must_use]
    pub fn rank(&self) -> usize {
        self.aligned.rank()
    }

    /// Returns the transpose of the matrix (rows and columns swapped).
    ///
    /// # Example
    ///
    /// ```
    /// use binar::BitMatrix;
    ///
    /// let m = BitMatrix::zeros(3, 5);
    /// let mt = m.transposed();
    /// assert_eq!(mt.shape(), (5, 3));
    /// ```
    pub fn transposed(&self) -> Self {
        Self {
            aligned: self.aligned.transposed(),
        }
    }

    /// Extracts a submatrix by selecting specific rows and columns.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::BitMatrix;
    ///
    /// let m = BitMatrix::identity(5);
    /// let sub = m.submatrix(&[0, 2, 4], &[1, 3]);
    /// assert_eq!(sub.shape(), (3, 2));
    /// ```
    pub fn submatrix(&self, rows: &[usize], columns: &[usize]) -> Self {
        Self {
            aligned: self.aligned.submatrix(rows, columns),
        }
    }

    /// Computes the inverse of this matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::BitMatrix;
    ///
    /// let m = BitMatrix::identity(3);
    /// let inv = m.inverted();
    /// assert_eq!(&m * &inv, BitMatrix::identity(3));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square or not invertible.
    pub fn inverted(&self) -> Self {
        Self {
            aligned: self.aligned.inverted(),
        }
    }

    /// Swaps two rows in place.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::BitMatrix;
    ///
    /// let mut m = BitMatrix::identity(3);
    /// m.swap_rows(0, 2);
    /// assert_eq!(m.get((0, 0)), false);
    /// assert_eq!(m.get((0, 2)), true);
    /// ```
    pub fn swap_rows(&mut self, left_row_index: usize, right_row_index: usize) {
        self.aligned.swap_rows(left_row_index, right_row_index);
    }

    /// Swaps two columns in place.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::BitMatrix;
    ///
    /// let mut m = BitMatrix::identity(3);
    /// m.swap_columns(0, 2);
    /// assert_eq!(m.get((0, 0)), false);
    /// assert_eq!(m.get((2, 0)), true);
    /// ```
    pub fn swap_columns(&mut self, left_column_index: usize, right_column_index: usize) {
        self.aligned.swap_columns(left_column_index, right_column_index);
    }

    /// Permutes the rows according to the given permutation.
    ///
    /// After calling this method, row `i` will contain what was previously in row `permutation[i]`.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::BitMatrix;
    ///
    /// let mut m = BitMatrix::identity(3);
    /// m.permute_rows(&[2, 0, 1]);  // Rotate rows
    /// assert_eq!(m.get((0, 2)), true);  // Row 0 now has old row 2
    /// ```
    pub fn permute_rows(&mut self, permutation: &[usize]) {
        self.aligned.permute_rows(permutation);
    }

    /// Adds (XORs) one row into another.
    ///
    /// This performs `row[to_index] ^= row[from_index]` (addition in GF(2)).
    ///
    /// # Example
    ///
    /// ```
    /// use binar::{BitMatrix, Bitwise};
    ///
    /// let mut m = BitMatrix::identity(3);
    /// m.add_into_row(0, 1);  // Add row 1 to row 0
    /// assert_eq!(m.row(0).weight(), 2);  // Now has bits from both rows
    /// ```
    pub fn add_into_row(&mut self, to_index: usize, from_index: usize) {
        self.aligned.add_into_row(to_index, from_index);
    }

    /// Compute `self * other^T`.
    ///
    /// This is equivalent to `self * other.transposed()` but may avoid the
    /// explicit transpose operation.
    ///
    /// # Panics
    ///
    /// Panics if `self.column_count() != other.column_count()`.
    pub fn mul_transpose(&self, other: &Self) -> Self {
        Self {
            aligned: self.aligned.mul_transpose(&other.aligned),
        }
    }

    /// Multiplies a row vector by this matrix from the left.
    ///
    /// Computes `left * self`, where `left` is treated as a row vector.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::{BitMatrix, BitVec, Bitwise};
    ///
    /// let m = BitMatrix::identity(3);
    /// let v = BitVec::ones(3);
    /// let result = m.right_multiply(&v.as_view());
    /// assert_eq!(result.weight(), 3);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `left.len() != self.row_count()`.
    pub fn right_multiply(&self, left: &BitView) -> BitVec {
        assert!(left.len() == self.row_count());
        BitVec::from_aligned(self.column_count(), self.aligned.right_multiply(&left.bits))
    }

    /// Computes the kernel (null space) of this matrix.
    ///
    /// Returns a matrix whose rows form a basis for the kernel.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::BitMatrix;
    ///
    /// let m = BitMatrix::zeros(2, 3);
    /// let ker = m.kernel();
    /// assert_eq!(ker.row_count(), 3);  // Full kernel for zero matrix
    /// ```
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
        assert!(right.len() == self.column_count());
        BitVec::from_aligned(self.row_count(), &self.aligned * &right.bits)
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
