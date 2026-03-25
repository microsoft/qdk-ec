mod aligned_bitmatrix;
mod bitmatrix;
mod column;
mod m4ri;
pub mod tiny_matrix;
pub mod transpose_kernel;

use derive_more::{Display, Error};

/// Error returned when constructing a matrix from sparse data with
/// incompatible dimensions.
#[derive(Debug, Display, Error)]
pub enum SparseConversionError {
    #[display("more {kind}s provided ({provided}) than {kind}_count ({declared})")]
    TooManyEntries {
        kind: &'static str,
        provided: usize,
        declared: usize,
    },
    #[display("{kind} index {index} out of bounds for {kind}_count {bound}")]
    IndexOutOfBounds {
        kind: &'static str,
        index: usize,
        bound: usize,
    },
}

pub use aligned_bitmatrix::{
    AlignedBitMatrix, EchelonForm as AlignedEchelonForm, MutableRow, Row, complete_to_full_rank_row_basis,
    directly_summed, kernel_basis_matrix, row_stacked,
};
pub use bitmatrix::{
    BitMatrix, EchelonForm, Row as MatrixRow,
    complete_to_full_rank_row_basis as bitmatrix_complete_to_full_rank_row_basis, row_stacked as bitmatrix_row_stacked,
};
pub use column::Column;
