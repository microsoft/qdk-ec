# paulimer

A library for Pauli and Clifford algebra.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)

## Features

- **Pauli Operators**:
  - `DensePauli`: Dense representation using bit vectors for X and Z components
  - `SparsePauli`: Sparse representation using index sets for memory efficiency
  
- **Pauli Groups**: 
  - Commutation checking between Pauli operators
  - Basis completion algorithms
  - Group structure utilities
  
- **Clifford Unitaries**:
  - Clifford tableau representation using stabilizer formalism
  - Pauli conjugation (image & preimage)
  - Multiplication and composition

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
paulimer = "0.0.1"
```

## Usage

### Pauli Operators

```rust
use paulimer::pauli::{DensePauli, Phase};

// Create a Pauli operator: Y₀ Z₂
let mut pauli = DensePauli::identity(4);
pauli.assign_x(0, true);  // Y = XZ
pauli.assign_z(0, true);
pauli.assign_z(2, true);

// Check properties
assert_eq!(pauli.weight(), 2);  // Acts on 2 qubits
println!("Support: {:?}", pauli.support().collect::<Vec<_>>());  // [0, 2]

// Multiply Pauli operators
let x1 = DensePauli::x(1, 4);
let result = pauli.multiply_with(&x1);
```

### Sparse Pauli for Large Systems

```rust
use paulimer::pauli::SparsePauli;

// Efficient for operators acting on few qubits in large systems
let sparse = SparsePauli::from_xz(
    vec![0, 5, 100].into_iter(),  // X positions
    vec![2, 100, 1000].into_iter(), // Z positions
    0  // phase
);

// Memory usage: O(k) where k is the number of non-identity Paulis
assert_eq!(sparse.weight(), 4);  // 4 non-identity positions
```

### Clifford Gates

```rust
use paulimer::clifford::tableau::Tableau;
use paulimer::pauli::DensePauli;

// Create a Clifford gate (e.g., CNOT)
let mut clifford = Tableau::identity(2);
// CNOT: X₀ → X₀X₁, Z₀ → Z₀, X₁ → X₁, Z₁ → Z₀Z₁

// Propagate Pauli through Clifford
let pauli_in = DensePauli::x(0, 2);
let pauli_out = clifford.image(&pauli_in);

// Random Clifford for testing
let mut rng = rand::thread_rng();
let random_clifford = Tableau::random(5, &mut rng);
```

### Commutation and Group Operations

```rust
use paulimer::pauli::{DensePauli, are_mutually_commuting};

let pauli1 = DensePauli::x(0, 3);
let pauli2 = DensePauli::x(1, 3);
let pauli3 = DensePauli::z(0, 3);

// Check if operators commute
assert!(pauli1.commutes_with(&pauli2));  // X₀ and X₁ commute
assert!(!pauli1.commutes_with(&pauli3)); // X₀ and Z₀ anticommute

// Check mutual commutation for stabilizer groups
let stabilizers = vec![pauli1, pauli2];
assert!(are_mutually_commuting(&stabilizers));
```

## Features

The crate supports optional features:

- `python`: Enables Python bindings via PyO3
- `serde`: Enables serialization/deserialization support
- `schemars`: Enables JSON schema generation

Enable features in your `Cargo.toml`:

```toml
[dependencies]
paulimer = { version = "0.1.0", features = ["serde"] }
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

For detailed API documentation, see:
- [Pauli module documentation](src/pauli/)
- [Clifford module documentation](src/clifford/)

## Contributing

This project welcomes contributions. See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
