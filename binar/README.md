# binar

High-performance bit manipulation library providing optimized bit vectors, bit matrices, and linear algebra over GF(2).

[![Crates.io](https://img.shields.io/crates/v/binar.svg)](https://crates.io/crates/binar)
[![Documentation](https://docs.rs/binar/badge.svg)](https://docs.rs/binar)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)

## Features

- **Bit Vectors**: Efficient bit vector implementations with `BitVec` and `AlignedBitVec`
- **Bit Matrices**: High-performance bit matrix operations with `BitMatrix` and `AlignedBitMatrix`
- **Linear Algebra over GF(2)**: Gaussian elimination, matrix multiplication, and rank computation
- **Bitwise Traits**: Generic trait system for bitwise operations across different types
- **Portable Performance**: Designed to leverage SIMD without explicit intrinsics, ensuring portability

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
binar = "0.1"
```

## Usage

```rust
use binar::{AlignedBitMatrix, AlignedBitVec, Bitwise, BitwiseMut};

// Create and manipulate bit vectors
let mut v1 = AlignedBitVec::zeros(100);
v1.assign_index(42, true);
v1.assign_index(17, true);

let mut v2 = AlignedBitVec::zeros(100);
v2.assign_index(42, true);
v2.assign_index(99, true);

// Bitwise operations
v1.bitxor_assign(&v2);
assert_eq!(v1.weight(), 3);  // Bits at positions 17, 42 XOR 42, 99 = 17, 99

// Linear algebra over GF(2)
let mut matrix = AlignedBitMatrix::identity(5);
matrix.set((0, 4), true);
matrix.set((2, 3), true);

// Gaussian elimination
let pivots = matrix.echelonize();
println!("Pivot columns: {:?}", pivots);

// Matrix-vector multiplication
let x = AlignedBitVec::from_fn(5, |i| i % 2 == 0);
let y = &matrix * &x;
```

## Python Bindings

Python bindings are available separately. See [bindings/python](bindings/python) for details.

## Performance

This library prioritizes performance while maintaining portability. Operations are designed to benefit from SIMD optimizations without using explicit intrinsics, allowing the compiler to generate efficient code across different platforms.

Key use cases include:
- Sparse matrix operations
- Linear equation solving over GF(2)
- Rank computation
- Kernel and image computation

Benchmarks can be run with:

```bash
cargo bench
```

## Documentation

Full API documentation is available on [docs.rs](https://docs.rs/binar).

## License

Licensed under the [MIT License](../LICENSE).
