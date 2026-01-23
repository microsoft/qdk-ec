# pauliverse

Fast stabilizer simulators.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)

## Overview

pauliverse provides multiple stabilizer simulation implementations optimized for different use cases:

- **`OutcomeSpecificSimulation`**: Traditional stabilizer simulation that draws random measurement outcomes as needed. Best for Monte Carlo sampling when the number of shots is much smaller than the number of random measurements.
- **`OutcomeCompleteSimulation`**: Tracks all 2^n_random outcome branches simultaneously. Best for analyzing entire circuits, or when shots >> 2^n_random.
- **`OutcomeFreeSimulation`**: Tracks stabilizer modulo measurement outcomes. Best for circuit verification and logical operator analysis.
- **`FaultySimulation`**: Extends OutcomeCompleteSimulation with frame-based noise propagation. Best for estimating logical error rates under Pauli noise models.

All simulators support the full Clifford group and Pauli measurements. Based on algorithms from [arXiv:2309.08676](https://arxiv.org/abs/2309.08676).

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
pauliverse = "0.1.0"
```

## Quick Start

Here's a basic example using `OutcomeSpecificSimulation` to create a Bell pair and measure it:

```rust
use pauliverse::{OutcomeSpecificSimulation, Simulation};
use paulimer::{UnitaryOp, SparsePauli};

// Create Bell pair and collect measurement statistics
let mut outcome_counts = [0, 0];

for _ in 0..1000 {
    // Initialize fresh simulation for each shot
    let mut sim = OutcomeSpecificSimulation::new_with_random_outcomes(2);
    
    // Create Bell pair: |00⟩ + |11⟩
    sim.unitary_op(UnitaryOp::Hadamard, &[0]);
    sim.unitary_op(UnitaryOp::ControlledX, &[0, 1]);
    
    // Measure ZZ (both qubits have same parity)
    let zz: SparsePauli = "ZZ".parse().unwrap();
    let outcome_id = sim.measure(&zz);
    
    // Record outcome
    let outcome = sim.outcome_vector()[outcome_id] as usize;
    outcome_counts[outcome] += 1;
}

println!("ZZ outcomes: +1: {}, -1: {}", outcome_counts[0], outcome_counts[1]);
// Output: ZZ outcomes: +1: 1000, -1: 0
// (Bell pair always has even parity)
```

For more examples and choosing the right simulator, see the [API documentation](https://docs.rs/pauliverse).

## Python Bindings

Python bindings for pauliverse are packaged with paulimer bindings:

```bash
cd paulimer/bindings/python
maturin develop --release
```

Then in Python:

```python
from paulimer import OutcomeFreeSimulation, DensePauli

# Create simulation
sim = OutcomeFreeSimulation(qubit_count=3)

# Apply gates
sim.h(0)
sim.cnot(0, 1)

# Measure
pauli_z = DensePauli.z(0, 3)
outcome = sim.measure(pauli_z)
```

## Choosing a Simulator

| Simulator | Best For | Key Advantage |
|-----------|----------|---------------|
| `OutcomeSpecificSimulation` | Monte Carlo sampling with many shots | Minimal overhead per shot, simple API |
| `OutcomeCompleteSimulation` | Exact distributions, circuit verification | Simulates once, sample many times efficiently |
| `OutcomeFreeSimulation` | State verification, logical operators | Tracks stabilizers without measurement records |
| `FaultySimulation` | Logical error rates, decoder testing | Frame-based noise propagation |

All simulators have O(n_gates × n_qubits²) worst-case complexity per simulation, though actual performance depends on circuit structure.

## Performance

pauliverse is optimized for quantum error correction research:

- **Efficient Clifford operations**: Leverages bit matrix operations via [binar](../binar)
- **SIMD acceleration**: Parallel bitwise operations for large stabilizer tableaux
- **Smart memory management**: Preallocate capacity to avoid reallocations in hot loops
- **Flexible measurement tracking**: Choose the simulator that matches your use case

Run benchmarks:

```bash
cargo bench --package pauliverse
```

## Architecture

- Built on [paulimer](../paulimer) for Pauli and Clifford operations
- Uses [binar](../binar) for efficient bit matrix operations

## Use Cases

- **Logical error rate estimation**: Monte Carlo sampling for error correction performance
- **Decoder validation**: Generate test data with controlled error patterns
- **Circuit verification**: Validate that circuits implement intended logical operations

## Documentation

Build and view the full API documentation:

```bash
cargo doc --open --package pauliverse
```

Key resources:
- [Simulation trait](src/lib.rs) - 40+ methods for gates, measurements, and state queries
- [OutcomeSpecificSimulation](src/outcome_specific_simulation.rs) - Traditional simulation with random outcomes
- [OutcomeCompleteSimulation](src/outcome_complete_simulation.rs) - All-branches simulation for exact analysis
- [FaultySimulation](src/faulty_simulation.rs) - Noisy simulation with frame propagation

## Contributing

This project welcomes contributions. See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
