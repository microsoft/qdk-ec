mod aligned_bitmatrix;
mod bitmatrix;
mod column;
mod m4ri;
pub mod tiny_matrix;
pub mod transpose_kernel;

pub use aligned_bitmatrix::{
    AlignedBitMatrix, EchelonForm as AlignedEchelonForm, MutableRow, Row, directly_summed, kernel_basis_matrix,
    row_stacked, complete_to_full_rank_row_basis
};
pub use bitmatrix::{BitMatrix, EchelonForm, Row as MatrixRow, row_stacked as bitmatrix_row_stacked, complete_to_full_rank_row_basis as bitmatrix_complete_to_full_rank_row_basis};
pub use column::Column;
