# Quantum Development Kit for Error Correction (QDK-EC)

Welcome to the Quantum Development Kit for Error Correction!

This repository is part of the [Azure Quantum Development Kit](https://github.com/microsoft/qdk) and provides high-performance tooling for quantum error correction research and development. It includes Rust crates and Python packages for Pauli algebra, Clifford gates, stabilizer simulation, and circuit synthesis.

## Components

This repository contains several interconnected crates:

- [binar](binar): A high-performance bit manipulation library providing bit vectors, bit matrices, and bitwise operations over GF(2).
- [paulimer](paulimer): A library for Pauli operators and Clifford gates, built on binar.
- [pauliverse](pauliverse): Fast stabilizer simulators. 

### Python Bindings

Python bindings are available for several crates:

- [binar](binar/bindings/python): Python bindings for the binar crate.
- [paulimer](paulimer/bindings/python): Python bindings for the paulimer and pauliverse crates.

## Building

### Prerequisites

To build this repository, you need:

- [Rust](https://www.rust-lang.org/tools/install) (stable toolchain)
- [Python](https://python.org/) (3.9 or later)
- [maturin](https://github.com/PyO3/maturin) (for building Python bindings)

### Building the Rust Crates

To build all crates:

```bash
cargo build --release
```

To run tests:

```bash
cargo test
```

### Building Python Bindings

To build and install the Python bindings for development:

For binar:
```bash
cd binar/bindings/python
maturin develop --release
```

For paulimer:
```bash
cd paulimer/bindings/python
maturin develop --release
```

## Installation

### Rust (from crates.io - available soon)

Add crates to your `Cargo.toml`:

```toml
[dependencies]
binar = "0.1"
paulimer = "0.0.1"
pauliverse = "0.0.1"
```

Or use `cargo add`:

```bash
cargo add binar
cargo add paulimer
cargo add pauliverse
```

### Python (from PyPI - available soon)

```bash
pip install binar
pip install paulimer  # Includes pauliverse bindings
```

### From Source

Follow the building instructions above to install the crates and Python bindings locally.

## Quick Start

### Working with Bit Vectors and Matrices (binar)

```rust
use binar::{BitVec, BitMatrix, Bitwise, BitwiseMut};

// Create and manipulate bit vectors
let mut v = BitVec::zeros(100);
v.assign_index(42, true);
v.assign_index(17, true);
assert_eq!(v.weight(), 2);

// Linear algebra over GF(2)
let mut matrix = BitMatrix::identity(5);
matrix.set((0, 4), true);
let pivots = matrix.echelonize();
```

### Working with Pauli Operators (paulimer)

```rust
use paulimer::pauli::{DensePauli, SparsePauli};

// Create Pauli operators
let pauli_x = DensePauli::x(0, 4);  // X₀ on 4 qubits
let pauli_z = DensePauli::z(1, 4);  // Z₁ on 4 qubits

// Check commutation
assert!(pauli_x.commutes_with(&pauli_z));  // X₀ and Z₁ commute

// Sparse representation for large systems
let sparse = SparsePauli::from_xz(
    vec![0, 100].into_iter(),  // X on qubits 0, 100
    vec![50].into_iter(),       // Z on qubit 50
    0
);
```

### Working with Clifford Gates (paulimer)

```rust
use paulimer::clifford::tableau::Tableau;
use paulimer::pauli::DensePauli;

// Create identity Clifford on 2 qubits
let clifford = Tableau::identity(2);

// Propagate Pauli through Clifford
let pauli_in = DensePauli::x(0, 2);
let pauli_out = clifford.image(&pauli_in);
```

### Stabilizer Simulation (pauliverse)

```rust
use pauliverse::outcome_free_simulation::OutcomeFreeSimulation;
use pauliverse::Simulation;
use paulimer::pauli::DensePauli;
use paulimer::UnitaryOp;

// Create simulation with 3 qubits
let mut sim = OutcomeFreeSimulation::new(3);

// Apply gates
sim.unitary_op(UnitaryOp::H, &[0]);
sim.unitary_op(UnitaryOp::CNOT, &[0, 1]);

// Measure
let z_obs = DensePauli::z(0, 3);
let outcome = sim.measure(&z_obs);
```

### Multi-Shot Simulation with Noise (pauliverse)

```rust
use pauliverse::frame_propagator::FramePropagator;

// Simulate 1000 shots with error propagation
let mut propagator = FramePropagator::new(
    qubit_count,
    outcome_count,
    1000  // shots
);

// Apply gates (errors propagate through all shots)
propagator.apply_h(0);
propagator.apply_cnot(0, 1);
propagator.measure(&z_observable);

// Get outcome deltas (error syndromes)
let deltas = propagator.into_outcome_deltas();
```

## Benchmarks

This repository uses [Criterion](https://github.com/bheisler/criterion.rs) for Rust benchmarks and [ASV](https://asv.readthedocs.io/) for Python benchmarks.

To run Rust benchmarks:

```bash
cargo bench
```

## Citation

If you use QDK-EC in your research, please cite as follows:

```bibtex
@software{Microsoft_QDK_EC,
  author = {{Microsoft}},
  license = {MIT},
  title = {{Quantum Development Kit for Error Correction}},
  url = {https://github.com/microsoft/qdk-ec}
}
```

## Feedback

If you have feedback about the content in this repository, please let us know by filing a [new issue](https://github.com/microsoft/qdk-ec/issues/new/choose)!

## Reporting Security Issues

Security issues and bugs should be reported privately following our [security issue documentation](SECURITY.md).

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit <https://cla.opensource.microsoft.com/>.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Legal and Licensing

### Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.