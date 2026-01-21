# pauliverse

Fast stabilizer simulators.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)

## Features

pauliverse provides multiple stabilizer simulation implementations optimized for different use cases:

- **`OutcomeFreeSimulation`**: Standard stabilizer tableau simulation
- **`OutcomeSpecificSimulation`**: Simulation conditioned on specific measurement outcomes
- **`OutcomeCompleteSimulation`**: Full tracking with complete outcome history
- **`FaultySimulation`**: Simulation with noise and error tracking
- **`FramePropagator`**: Batch multi-shot simulation with *O(n_gates × n_qubits)* complexity

All simulators support the full Clifford group and Pauli measurements.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
pauliverse = "0.0.1"
```

## Usage

### Basic Stabilizer Simulation

```rust
use pauliverse::outcome_free_simulation::OutcomeFreeSimulation;
use pauliverse::Simulation;
use paulimer::pauli::DensePauli;

// Create a stabilizer simulation with 3 qubits
let mut sim = OutcomeFreeSimulation::new(3);

// Apply Clifford gates
sim.unitary_op(UnitaryOp::H, &[0]);
sim.unitary_op(UnitaryOp::CNOT, &[0, 1]);

// Measure in Z basis
let z_obs = DensePauli::z(0, 3);
let outcome = sim.measure(&z_obs);

// Check if operator is a stabilizer
assert!(sim.is_stabilizer(&z_obs));
```

### Frame-Based Multi-Shot Simulation

```rust
use pauliverse::frame_propagator::FramePropagator;

// Simulate 1000 shots of a circuit with errors
let mut propagator = FramePropagator::new(
    qubit_count,    // Number of qubits
    outcome_count,  // Number of measurements
    1000,           // Number of shots
);

// Apply gates (propagates errors through all shots)
propagator.apply_h(0);
propagator.apply_cnot(0, 1);

// Measure observable (computes outcome for each shot)
propagator.measure(&z_observable);

// Get outcome deltas (error syndromes)
let deltas = propagator.into_outcome_deltas();
```

### Noisy Circuit Simulation

```rust
use pauliverse::faulty_simulation::FaultySimulation;
use pauliverse::noise::{PauliFault, PauliDistribution};

// Create simulation with capacity for noise
let mut sim = FaultySimulation::with_capacity(
    qubit_count,
    outcome_count,
    random_outcome_count
);

// Define depolarizing noise (uniform over X, Y, Z)
let noise = PauliFault {
    probability: 0.01,  // 1% error rate
    distribution: PauliDistribution::DepolarizingOnQubits(vec![0]),
    fault_set: None,
    condition: None,
};

// Apply noisy gate
sim.unitary_op(UnitaryOp::H, &[0]);
sim.apply_fault(&noise);

// Continue with more gates and measurements...
```

### Custom Noise Distributions

```rust
use pauliverse::noise::{PauliDistribution, PauliFault};
use paulimer::pauli::SparsePauli;

// Single deterministic error
let single_fault = PauliFault {
    probability: 0.05,
    distribution: PauliDistribution::Single(SparsePauli::x(0, 3)),
    fault_set: None,
    condition: None,
};

// Weighted distribution over multiple Paulis
let weighted_fault = PauliFault {
    probability: 0.1,
    distribution: PauliDistribution::Weighted(vec![
        (SparsePauli::x(0, 3), 0.7),  // 70% X error
        (SparsePauli::z(0, 3), 0.3),  // 30% Z error
    ]),
    fault_set: None,
    condition: None,
};

// Depolarizing noise on multiple qubits
let depol_fault = PauliFault {
    probability: 0.01,
    distribution: PauliDistribution::DepolarizingOnQubits(vec![0, 1]),
    fault_set: None,
    condition: None,
};
```

### Conditional Noise

```rust
use pauliverse::noise::OutcomeCondition;

// Apply noise only when measurement outcome is 1
let conditional_fault = PauliFault {
    probability: 0.05,
    distribution: PauliDistribution::Single(SparsePauli::x(0, 3)),
    fault_set: None,
    condition: Some(OutcomeCondition {
        outcomes: vec![outcome_id],
        parity: true,  // Apply when outcome is 1
    }),
};
```

## Simulation Modes

pauliverse provides three simulation modes optimized for different use cases:

1. **`OutcomeFreeSimulation`**: Standard stabilizer simulation
   - Tracks stabilizer tableau and measurement outcomes
   - Use when you need the quantum state evolution

2. **`OutcomeSpecificSimulation`**: Simulation conditioned on specific outcomes
   - Allows conditioning evolution on predetermined measurement results
   - Useful for post-selection and conditional logic

3. **`OutcomeCompleteSimulation`**: Full tracking with outcome history
   - Maintains complete history of all outcomes and random bits
   - Use when you need detailed tracking for analysis

All modes implement the `Simulation` trait, allowing generic code.

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

## Performance

pauliverse is designed for high-performance quantum error correction research:

- **Frame propagator**: *O(n_gates × n_qubits)* complexity for multi-shot simulation
- **Bit matrix operations**: Leverages SIMD via binar for parallel operations
- **Batch processing**: Simulates thousands of shots simultaneously with minimal overhead
- **Memory efficiency**: Sparse representations for large systems with localized errors

Run benchmarks:

```bash
cargo bench --package pauliverse
```

## Architecture

- Built on [paulimer](../paulimer) for Pauli and Clifford operations
- Uses [binar](../binar) for efficient bit matrix operations
- `SmallVec` for avoiding heap allocations in hot paths (1-4 element collections)

## Use Cases

- **Surface code simulation**: Efficient syndrome extraction with frame propagation
- **Logical error rate estimation**: Monte Carlo sampling with batch simulation
- **Decoder testing**: Generate error syndromes for decoder validation
- **Noise characterization**: Study error propagation in different noise models

## Documentation

For detailed API documentation, see:
- [Simulation trait](src/lib.rs) - Core simulation interface
- [Frame propagator](src/frame_propagator.rs) - Batch multi-shot simulation
- [Noise module](src/noise.rs) - Noise modeling and distributions
- [Faulty simulation](src/faulty_simulation.rs) - Combined simulation with noise

## Contributing

This project welcomes contributions. See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
