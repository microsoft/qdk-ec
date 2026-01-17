pub mod bit;
pub use bit::{BitBlock, BitLength, Bitwise, BitwiseMut, BitwisePair, BitwisePairMut, FromBits, IntoBitIterator};

type Tuple8<T> = (T, T, T, T, T, T, T, T);

pub mod vec;
pub use vec::{BitVec, BitView, BitViewMut, IndexSet, remapped};

pub mod matrix;
pub use matrix::{BitMatrix, EchelonForm};

#[cfg(feature = "python")]
pub mod python;

pub const BIT_MATRIX_ALIGNMENT: usize = crate::bit::BitBlock::BLOCK_BIT_LEN;
