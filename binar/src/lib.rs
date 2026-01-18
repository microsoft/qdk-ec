//! High-performance binary arithmetic: bit vectors, bit matrices, and linear algebra over GF(2).
//!
//! # Overview
//!
//! `binar` provides efficient data structures and operations for working with bits, bit vectors, and bit matrices.
//! It's designed for applications requiring fast linear algebra over GF(2) (the field with two elements: 0 and 1),
//! where addition is XOR and multiplication is AND.
//!
//! # Getting Started
//!
//! The primary types you'll work with are:
//! - [`BitVec`] - A variable-length bit vector
//! - [`BitMatrix`] - A matrix of bits supporting linear algebra operations
//! - [`IndexSet`] - A sparse representation of a bit vector using sorted indices
//!
//! ## Example: Basic Bit Vector Operations
//!
//! ```
//! use binar::{BitVec, Bitwise, BitwiseMut};
//!
//! // Create a bit vector with 100 bits, all initialized to false
//! let mut v = BitVec::zeros(100);
//!
//! // Set some bits
//! v.assign_index(5, true);
//! v.assign_index(42, true);
//!
//! // Count the number of set bits
//! assert_eq!(v.weight(), 2);
//!
//! // XOR with another vector
//! let mut v2 = BitVec::zeros(100);
//! v2.assign_index(42, true);
//! v.bitxor_assign(&v2);
//!
//! // Bit 42 is now unset (true XOR true = false)
//! assert_eq!(v.weight(), 1);
//! ```
//!
//! ## Example: Linear Algebra over GF(2)
//!
//! ```
//! use binar::{BitMatrix, Bitwise};
//!
//! // Create a 5x5 identity matrix
//! let mut matrix = BitMatrix::identity(5);
//!
//! // Modify the matrix
//! matrix.set((0, 4), true);
//! matrix.set((2, 3), true);
//!
//! // Compute row echelon form
//! let pivots = matrix.echelonize();
//!
//! // Matrix-vector multiplication
//! let x: BitVec = vec![true, false, true, false, true].into_iter().collect();
//! let y = &matrix * &x;
//! ```
//!
//! # Traits
//!
//! The library provides a trait system for generic programming with bit-like types:
//! - [`Bitwise`] - Read-only operations: indexing, counting, iteration
//! - [`BitwiseMut`] - Mutable operations: setting and clearing bits
//! - [`BitwisePair`] - Binary operations between two bit structures (AND, OR, XOR, dot product)
//! - [`BitwisePairMut`] - Mutable binary operations (in-place XOR, AND, OR)
//!
//! These traits allow you to write generic code that works with [`BitVec`], [`BitMatrix`] rows,
//! [`IndexSet`], and other bit-like types. The traits are also implemented for standard types
//! like `u64`, `u32`, `[u64; N]`, and other unsigned integer types, allowing seamless interoperability
//! with raw bit patterns.
//!
//! # Why Choose binar?
//!
//! While many Rust crates offer bit manipulation, `binar` is built around a different philosophy:
//! performance should be portable, and correctness should be easy.
//!
//! - **Portable performance** - Achieves competitive speed with state-of-the-art implementations
//!   without relying on platform-specific intrinsics or SIMD. Your code runs consistently fast
//!   everywhere, from ARM microcontrollers to x86 servers.
//!
//! - **Pure Rust** - No C dependencies means simpler builds and easier cross-compilation.
//!
//! - **Flexible licensing** - MIT license removes barriers for both open source and commercial projects.
//!
//! - **Purpose-built for GF(2)** - The API is designed specifically for linear algebra workflows
//!   over GF(2), not retrofitted from general bit manipulation. You get the operations you actually
//!   need when working with binary matrices and vectors.
//!
//! # Performance-Oriented Types
//!
//! For performance-critical applications, aligned variants are available:
//! - [`vec::AlignedBitVec`] - Cache-aligned bit vector
//! - [`matrix::AlignedBitMatrix`] - Cache-aligned bit matrix
//!
//! Note that [`BitVec`] and [`BitMatrix`] are thin wrappers around these aligned types,
//! providing a more convenient API while maintaining the same performance characteristics.
//! The aligned types use aligned memory allocation and may offer better performance in tight loops
//! or with large datasets. Start with [`BitVec`] and [`BitMatrix`], and consider using the aligned
//! variants directly when you need more control over memory layout or when profiling indicates it would help.

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
