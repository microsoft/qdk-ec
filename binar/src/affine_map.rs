//! Affine maps over GF(2).
//!
//! An affine map is a linear map plus a translation: `f(x) = Ax + b`.

use crate::{BitMatrix, BitVec};

/// An affine map over GF(2), represented as `f(x) = Ax + b`.
#[derive(Debug, Clone, PartialEq)]
pub struct AffineMap {
    matrix: BitMatrix,
    shift: BitVec,
}

impl AffineMap {
    /// Creates an affine map from a matrix and a shift vector.
    ///
    /// # Panics
    ///
    /// Panics if the matrix row count does not match the shift length.
    #[must_use]
    pub fn affine(matrix: BitMatrix, shift: BitVec) -> Self {
        assert_eq!(matrix.row_count(), shift.len());
        Self { matrix, shift }
    }

    /// Creates a linear map (affine map with zero shift).
    #[must_use]
    pub fn linear(matrix: BitMatrix) -> Self {
        let shift = BitVec::zeros(matrix.row_count());
        Self { matrix, shift }
    }

    /// Creates a translation map (affine map with identity matrix).
    #[must_use]
    pub fn translation(shift: BitVec) -> Self {
        let matrix = BitMatrix::identity(shift.len());
        Self { matrix, shift }
    }

    #[must_use]
    pub fn zero(input_dimension: usize, output_dimension: usize) -> Self {
        let matrix = BitMatrix::zeros(output_dimension, input_dimension);
        let shift = BitVec::zeros(output_dimension);
        Self { matrix, shift }
    }

    /// Affine map evaluated at `input` : `self(input)`
    pub fn apply(&self, input: &BitVec) -> BitVec {
        &self.matrix * &input.as_view() + &self.shift
    }

    /// Computes the composition of two affine maps equal to `self(other(x))`.
    ///
    /// # Panics
    ///
    /// Panics if input dimension of self does not match output dimension of other.
    #[must_use]
    pub fn dot(&self, other: &AffineMap) -> AffineMap {
        assert_eq!(self.input_dimension(), other.output_dimension());
        let matrix = &self.matrix * &other.matrix;
        let shift = &self.matrix * &other.shift.as_view() + &self.shift;
        AffineMap::affine(matrix, shift)
    }

    /// Returns the input dimension of this map (number of columns in the matrix).
    #[must_use]
    pub fn input_dimension(&self) -> usize {
        self.matrix.column_count()
    }

    /// Returns the output dimension of this map (number of rows in the matrix).
    #[must_use]
    pub fn output_dimension(&self) -> usize {
        self.matrix.row_count()
    }

    /// Returns a reference to the linear part (matrix) of this affine map.
    pub fn matrix(&self) -> &BitMatrix {
        &self.matrix
    }

    /// Returns a reference to the translation part (shift) of this affine map.
    pub fn shift(&self) -> &BitVec {
        &self.shift
    }
}
