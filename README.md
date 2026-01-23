# Quantum Development Kit for Error Correction (QDK-EC)

Welcome to the Quantum Development Kit for Error Correction!

This repository is part of the [Azure Quantum Development Kit](https://github.com/microsoft/qdk) and provides high-performance tooling for quantum error correction research and development. It includes Rust crates and Python packages for Pauli algebra, Clifford gates, stabilizer simulation, and circuit synthesis.

Implementations are based on data structures and algorithms described in [arXiv:2309.08676](https://arxiv.org/abs/2309.08676).

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
paulimer = "0.1"
pauliverse = "0.1"
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

### Bit Vectors and Matrices (binar)

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

### Pauli Operators and Clifford Gates (paulimer)

```rust
use paulimer::{DensePauli, SparsePauli, commutes_with};
use paulimer::{CliffordUnitary, CliffordMutable, Clifford};

// Create Pauli operators from strings
let x: DensePauli = "XII".parse().unwrap();  // X ⊗ I ⊗ I
let z: SparsePauli = "Z0".parse().unwrap();  // Sparse: Z₀

// Check commutation (X and Z anticommute)
assert!(!commutes_with(&x, &z));

// Build Clifford gates and propagate Paulis
let mut clifford = CliffordUnitary::identity(2);
clifford.left_mul_cx(0, 1);  // Apply CNOT: X₀ → X₀ ⊗ X₁

let x0: DensePauli = "XI".parse().unwrap();
let image = clifford.image(&x0);
assert_eq!(image, "XX".parse::<DensePauli>().unwrap());
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