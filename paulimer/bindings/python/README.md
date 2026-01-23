# paulimer - Paulis, Cliffords and stabilizer simulation

High-performance Pauli operators, Clifford unitaries, and stabilizer simulation for quantum computing.

## Installation

```bash
pip install paulimer
```

## Quick Start

```python
import paulimer

# Pauli operators
p = paulimer.DensePauli("XYZ")
q = paulimer.SparsePauli("X0 Z100")
print(q * q)  # Identity

# Clifford gates
h = paulimer.CliffordUnitary.from_name("Hadamard", [0], qubit_count=1)
print(h.image_of(paulimer.DensePauli("X")))  # Z

# Stabilizer simulation
sim = paulimer.OutcomeCompleteSimulation(2)
sim.apply_unitary(paulimer.UnitaryOpcode.Hadamard, [0])
sim.apply_unitary(paulimer.UnitaryOpcode.ControlledX, [0, 1])
sim.measure(paulimer.SparsePauli("Z0"))
```

## Features

- **DensePauli / SparsePauli** - Pauli operators with phase tracking and multiplication
- **CliffordUnitary** - Clifford gates with conjugation and composition
- **PauliGroup** - Group operations including membership testing and factorization
- **Stabilizer Simulation** - Noiseless (OutcomeComplete, OutcomeFree, OutcomeSpecific) and noisy (Faulty) modes

## Use Cases

Designed for quantum error correction research, including stabilizer circuit analysis and Clifford circuit verification.

## Performance

Built on `binar` for SIMD-accelerated binary linear algebra. 

## API Reference

See [paulimer.pyi](paulimer.pyi) for complete type hints and documentation.

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! See [github.com/microsoft/qdk-ec](https://github.com/microsoft/qdk-ec) for guidelines.
