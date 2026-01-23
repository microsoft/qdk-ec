//! Pauli and Clifford algebra for quantum computing.
//!
//! `paulimer` provides efficient implementations of Pauli operators and Clifford unitaries,
//! the building blocks for stabilizer quantum mechanics and quantum error correction.
//!
//! Implementations are based on data structures and algorithms described in [arXiv:2309.08676](https://arxiv.org/abs/2309.08676).
//!
//! # Overview
//!
//! This crate offers:
//! - **Pauli operators**: Dense ([`DensePauli`]) and sparse ([`SparsePauli`]) representations with phase tracking
//! - **Pauli groups**: Subgroup representation ([`PauliGroup`]) for stabilizer groups and code analysis
//! - **Clifford unitaries**: Efficient representation ([`CliffordUnitary`]) enabling fast conjugation
//!
//! All types support efficient operations on quantum error correction codes, stabilizer
//! circuit simulation, and Clifford circuit analysis.
//!
//! # Quick Start
//!
//! ```
//! use paulimer::{DensePauli, SparsePauli, commutes_with};
//! use paulimer::{CliffordUnitary, Clifford};
//!
//! // Create Pauli operators from strings
//! let pauli1: DensePauli = "XII".parse().unwrap();  // X ⊗ I ⊗ I
//! let pauli2: SparsePauli = "Z0".parse().unwrap(); // Sparse: Z₀ ⊗ I ⊗ I
//!
//! // Check commutation (X and Z anticommute)
//! assert!(!commutes_with(&pauli1, &pauli2));
//!
//! // Create and apply Clifford gates
//! let clifford = CliffordUnitary::identity(3);
//! let image = clifford.image(&pauli1);
//! ```
//!
//! # Pauli Operators
//!
//! Pauli operators are tensor products of single-qubit Paulis {I, X, Y, Z} with phases {±1, ±i}.
//! Two complementary representations optimize for different use cases:
//!
//! - **[`DensePauli`]**: Bit vectors storing X and Z components for all qubits.
//!   - Best for: operators on most qubits, small to moderate system sizes
//!   - Memory: O(n) for n qubits
//!   - String format: `"XYZI"` (one character per qubit)
//!
//! - **[`SparsePauli`]**: Index sets storing only non-identity positions.
//!   - Best for: operators on few qubits in large systems, weight << n
//!   - Memory: O(k) for weight k
//!   - String format: `"X0 Z5 Y100"` (positioned Paulis)
//!
//! Both representations share a common trait-based interface ([`Pauli`], [`PauliBinaryOps`])
//! enabling generic code that works with either type.
//!
//! # Pauli Groups
//!
//! The [`PauliGroup`] struct represents a subgroup of the Pauli group generated
//! by a set of Pauli operators:
//!
//! - **Membership testing**: Check if an operator is in the group
//! - **Factorization**: Decompose elements into generators
//! - **Enumeration**: Iterate over all group elements
//! - **Structure queries**: Abelian, stabilizer, and rank properties
//!
//! `PauliGroup` is essential for:
//! - Representing stabilizer groups for quantum error correcting codes
//! - Checking code properties (distance, logical operators, normalizers)
//! - Analyzing symmetries and logical equivalence of quantum circuits
//! - Computing with discrete Pauli subgroups efficiently
//!
//! # Clifford Unitaries
//!
//! [`CliffordUnitary`] represents Clifford gates as the images of Pauli basis elements.
//! A Clifford on n qubits is stored as a 2n×2n binary matrix plus phase information,
//! enabling O(n²) Pauli conjugation instead of naive O(2ⁿ).
//!
//! Key capabilities:
//! - **Conjugation**: Propagate Paulis forward and backward through gates
//! - **Composition**: Build gate sequences (O(n³))
//! - **Standard gates**: Apply via [`CliffordMutable`] methods or [`UnitaryOp`] enum
//! - **Construction**: Build from Pauli basis images or gate sequences
//!
//! This representation is the foundation for efficient stabilizer simulation
//! and Clifford circuit verification.
//!
//! # Performance
//!
//! Built on [`binar`] for efficient bit matrix operations:
//! - SIMD acceleration where available
//! - Cache-aligned data structures
//! - Optimized Gaussian elimination
//! - Binary symplectic representation enabling O(n²) Clifford conjugation
//!
//! The algorithms in this crate support efficient verification and characterization
//! of stabilizer circuits, including equivalence checking and logical action analysis.
//!
//! # Features
//!
//! - `python`: Python bindings via `PyO3`
//! - `serde`: Serialization support
//! - `schemars`: JSON schema generation
//!
//! # Examples
//!
//! ## Pauli Group Operations
//!
//! ```
//! use paulimer::{DensePauli, commutes_with, Pauli, PauliBinaryOps};
//!
//! let x: DensePauli = "XII".parse().unwrap();
//! let z: DensePauli = "ZII".parse().unwrap();
//!
//! // Anticommutation
//! assert!(!commutes_with(&x, &z));
//!
//! // Multiplication: XZ = iY
//! let mut y = x.clone();
//! y.mul_assign_right(&z);
//! assert_eq!(y.weight(), 1);
//! ```
//!
//! ## Clifford Propagation
//!
//! ```
//! use paulimer::{CliffordUnitary, CliffordMutable, Clifford};
//! use paulimer::{DensePauli, Pauli, UnitaryOp};
//!
//! let mut clifford = CliffordUnitary::identity(2);
//!
//! // Apply CNOT: maps X₀ → X₀ ⊗ X₁
//! clifford.left_mul_cx(0, 1);
//!
//! let x0: DensePauli = "XI".parse().unwrap();
//! let image = clifford.image(&x0);
//! let expected: DensePauli = "XX".parse().unwrap();
//! assert_eq!(image, expected);
//! ```

pub mod binar_impls;
pub mod clifford;
pub mod operations;
pub mod pauli;
pub mod pauli_group;
pub mod setwise;
pub mod traits;

pub use clifford::{Clifford, CliffordMutable, CliffordUnitary};
pub use operations::UnitaryOp;
pub use pauli::{
    commutes_with, anti_commutes_with, DensePauli, SparsePauli, Pauli, PauliBinaryOps,
    PauliMutable, Phase,
};
pub use pauli_group::PauliGroup;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "serde")]
mod serde;

#[cfg(feature = "schemars")]
mod schemars;

// Type aliases from original lib.rs
type Tuple2<T> = (T, T);
type Tuple4<T> = (T, T, T, T);
type Tuple8<T> = (T, T, T, T, T, T, T, T);
type Tuple2x2<T> = Tuple2<Tuple2<T>>;
type Tuple4x2<T> = Tuple4<Tuple2<T>>;

// Utility functions
#[must_use]
pub fn subscript_digits(number: usize) -> String {
    let mut res = String::new();
    for char in number.to_string().chars() {
        let digit = char.to_digit(10).unwrap_or_default() as usize;
        res.push(SUB_CHARS[digit]);
    }
    res
}

pub const SUB_CHARS: [char; 10] = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
pub const CLIFFORD_BIT_ALIGNMENT: usize = binar::BIT_MATRIX_ALIGNMENT;
