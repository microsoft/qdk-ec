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

# Decompose a Clifford into pi/4 Pauli exponents (Clifford transvections)
cnot = paulimer.CliffordUnitary.from_name("ControlledX", [0, 1], qubit_count=2)
factors, correction = cnot.to_transvections_minimal()
rebuilt = paulimer.CliffordUnitary.identity(2)
for pauli in factors:
    rebuilt.left_mul_pauli_exp(pauli)
rebuilt.left_mul_pauli(correction)
assert rebuilt == cnot

# Stabilizer simulation
sim = paulimer.OutcomeCompleteSimulation(2)
sim.apply_unitary(paulimer.UnitaryOpcode.Hadamard, [0])
sim.apply_unitary(paulimer.UnitaryOpcode.ControlledX, [0, 1])
sim.measure(paulimer.SparsePauli("Z0"))
```

The minimal API returns the fewest rotations for the unsigned action and a final Pauli for all tableau signs.
The minimum excludes that Pauli. The greedy `to_transvections()` API still returns only an unsigned list.
Neither API reconstructs global phase because the input tableau does not record it.

## Features

- **DensePauli / SparsePauli** - Pauli operators with phase tracking and multiplication
- **CliffordUnitary** - Clifford gates with conjugation, composition, and decomposition into `π/4`
  Pauli exponents (`to_transvections`, `to_transvections_minimal`)
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
