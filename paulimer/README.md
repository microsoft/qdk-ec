# paulimer

Pauli and Clifford algebra for quantum computing.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)

## Overview

paulimer provides efficient implementations of Pauli operators and Clifford unitaries,
the building blocks for stabilizer quantum mechanics and quantum error correction.

**Key Features:**
- **Pauli Operators**: Dense and sparse representations with phase tracking (±1, ±i)
  - [`DensePauli`]: Bit vectors optimized for operators on most qubits (O(n) memory)
  - [`SparsePauli`]: Index sets for operators on few qubits in large systems (O(k) memory)

- **Pauli Groups**: Subgroup representation for stabilizer codes
  - [`PauliGroup`]: Membership testing, factorization, and structure queries
  - Essential for code verification and logical operator analysis

- **Clifford Unitaries**: Efficient representation enabling fast operations
  - [`CliffordUnitary`]: O(n²) Pauli conjugation via binary symplectic matrix
  - Supports all standard Clifford gates (H, S, CNOT, etc.)

Based on algorithms from [arXiv:2309.08676](https://arxiv.org/abs/2309.08676).

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
paulimer = "0.1.0"
```

## Quick Start

### Pauli Operators

```rust
use paulimer::{DensePauli, SparsePauli, Pauli, commutes_with};

// Dense notation: one character per qubit
let dense: DensePauli = "YIZI".parse().unwrap();  // Y₀ ⊗ I₁ ⊗ Z₂ ⊗ I₃
assert_eq!(dense.weight(), 2);  // Acts on 2 qubits

// Sparse notation: only specify non-identity positions  
let sparse: SparsePauli = "Y0 Z2".parse().unwrap();  // Same operator, compact
assert_eq!(sparse.weight(), 2);

// Check commutation
let x: DensePauli = "XII".parse().unwrap();
let z: DensePauli = "ZII".parse().unwrap();
assert!(!commutes_with(&x, &z));  // X and Z anticommute
```

### Pauli Groups

```rust
use paulimer::{PauliGroup, SparsePauli};

// Define stabilizer group for 3-qubit repetition code
let generators = vec![
    "ZZI".parse::<SparsePauli>().unwrap(),
    "IZZ".parse::<SparsePauli>().unwrap(),
];
let group = PauliGroup::new(&generators);

// Check properties
assert!(group.is_stabilizer_group());
assert_eq!(group.log2_size(), 2);  // 4 elements

// Test membership
let zzi = "ZZI".parse::<SparsePauli>().unwrap();
assert!(group.contains(&zzi));
```

### Clifford Unitaries

```rust
use paulimer::{CliffordUnitary, Clifford, CliffordMutable};
use paulimer::{DensePauli, UnitaryOp};

// Build CNOT gate
let mut cnot = CliffordUnitary::identity(2);
cnot.left_mul_cx(0, 1);

// Propagate Pauli through gate: X₀ → X₀ ⊗ X₁
let x0: DensePauli = "XI".parse().unwrap();
let image = cnot.image(&x0);
assert_eq!(image, "XX".parse::<DensePauli>().unwrap());

// Build circuits with UnitaryOp
let mut circuit = CliffordUnitary::identity(2);
circuit.left_mul(UnitaryOp::Hadamard, &[0]);
circuit.left_mul(UnitaryOp::ControlledX, &[0, 1]);
```

## When to Use Each Type

| Type | Best For | Memory | Example |
|------|----------|--------|---------|
| `DensePauli` | Operators on most qubits | O(n) | Error correction on small codes |
| `SparsePauli` | Few qubits in large systems | O(k) | Syndrome extraction, weight-k errors |
| `PauliGroup` | Stabilizer groups, code analysis | O(k·n) | Checking stabilizer properties |
| `CliffordUnitary` | Gate sequences, circuit analysis | O(n²) | Clifford circuit simulation |

## Features

The crate supports optional features:

- `python`: Enables Python bindings via PyO3
- `serde`: Enables serialization/deserialization support
- `schemars`: Enables JSON schema generation

Enable features in your `Cargo.toml`:

```toml
[dependencies]
paulimer = { version = "0.0.1", features = ["serde"] }
```

## Python Bindings

Python bindings are available in `bindings/python/`:

```bash
cd paulimer/bindings/python
maturin develop --release
```

Then in Python:

```python
from paulimer import DensePauli, Phase

# Create Pauli operators
pauli = DensePauli.x(0, 4)
print(pauli)  # X₀

# Multiply operators
result = pauli * DensePauli.z(0, 4)
print(result)  # Should be Y₀ (X*Z = iY)
```

## Performance

paulimer is designed for high performance:

- Built on top of [binar](../binar) for optimized bit operations
- Cache-aligned data structures for efficient memory access
- Sparse representations for large but sparse operators
- Benchmarks available in `benches/`

Run benchmarks:

```bash
cargo bench
```

## Documentation

Build and view comprehensive API documentation:

```bash
cargo doc --open --package paulimer
```

Key documentation:
- [`DensePauli`](src/pauli/dense.rs) - Dense Pauli representation with examples
- [`SparsePauli`](src/pauli/sparse.rs) - Sparse Pauli representation for large systems
- [`PauliGroup`](src/pauli_group.rs) - Subgroup operations and stabilizer groups
- [`CliffordUnitary`](src/clifford.rs) - Clifford gates and Pauli conjugation
- [Trait documentation](src/lib.rs) - `Pauli`, `Clifford`, and other core traits

## Contributing

This project welcomes contributions. See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
