# binar - High-performance binary arithmetic

Fast bit vectors and bit matrices with linear algebra over GF(2).

## Overview

`binar` provides efficient Python bindings to high-performance Rust implementations of:
- **Bit vectors** (`BitVector`) - Variable-length sequences of bits
- **Bit matrices** (`BitMatrix`) - 2D arrays of bits with linear algebra operations
- Operations optimized for quantum computing and error correction

The library is designed for applications requiring fast linear algebra over GF(2) (the binary field with elements {0, 1}), where addition is XOR and multiplication is AND.

## Installation

```bash
pip install binar
```

For development:
```bash
cd binar/bindings/python
maturin develop --release
```

## Quick Start

```python
import binar

# Bit vectors: create, manipulate, and compute
v1 = binar.BitVector("10110")
v2 = binar.BitVector([True, False, True, False, False])
print(v1.weight)  # 3 (number of 1s)
print(v1.support)  # [0, 2, 3] (indices of 1s)

# Boolean operations
v3 = v1 ^ v2  # XOR
print(v1.dot(v2))  # Inner product over GF(2): True

# Bit matrices: linear algebra over GF(2)
m = binar.BitMatrix([
    "1010",
    "0110",
    "1100",
    "0011"
])
print(m.shape)  # (4, 4)

# Matrix operations
identity = binar.BitMatrix.identity(4)
product = m @ identity  # Matrix multiplication
m_rref = m.echelonized()  # Row echelon form
kernel = m.kernel()  # Null space basis
```


## Key Features

### BitVector

- Create from strings, lists, or factory methods
- Boolean operations: XOR, AND, OR
- Hamming weight and parity computation
- Inner product over GF(2)
- Support (indices of set bits)

### BitMatrix

- Create from rows or factory methods
- Matrix multiplication over GF(2)
- Element-wise boolean operations
- Row echelon form and reduced row echelon form
- Null space (kernel) computation
- Transpose and submatrix extraction

## Use Cases

`binar` is particularly useful for:

- **Quantum error correction**: Parity check matrices, stabilizer codes
- **Linear codes**: Generator and check matrices over GF(2)
- **Graph theory**: Adjacency matrices, graph algorithms
- **Cryptography**: Linear feedback shift registers, boolean functions
- **Computational algebra**: Gaussian elimination, system solving over GF(2)

## Performance

Built on optimized Rust code with:
- SIMD acceleration for bit operations
- Cache-friendly memory layout
- Efficient Gaussian elimination algorithms
- Zero-copy integration between Python and Rust

## Examples

### Solving Linear Systems over GF(2)

```python
import binar

# Coefficient matrix
A = binar.BitMatrix([
    "110",
    "101",
    "011"
])

# Find kernel (solutions to Ax = 0)
kernel = A.kernel()
print(f"Null space dimension: {kernel.row_count}")

# Verify solution
for row in kernel.rows:
    result = A @ row
    assert result.is_zero  # Ax = 0
```

### Parity Check Matrix for [7,4,3] Hamming Code

```python
import binar

# Parity check matrix for [7,4,3] Hamming code
H = binar.BitMatrix([
    "1010101",
    "0110011",
    "0001111"
])

# Check syndrome for error vector
error = binar.BitVector("0001000")  # Error on bit 3
syndrome = H @ error
print(f"Syndrome: {syndrome}")  # Points to error location

# Generate all codewords by finding kernel
codewords = H.kernel()
print(f"Code dimension: {codewords.row_count}")  # 4
```

## API Reference

See the [type stubs file](binar.pyi) for complete API documentation with type hints.

## Related Packages

- **paulimer**: Pauli and Clifford algebra built on binar

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! See [github.com/microsoft/qdk-ec](https://github.com/microsoft/qdk-ec) for guidelines.
