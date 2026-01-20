pub mod binar_impls;
pub mod clifford;
pub mod operations;
pub mod pauli;
pub mod pauli_group;
pub mod setwise;
pub mod traits;
pub use operations::UnitaryOp;

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
