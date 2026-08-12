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

### Verifying parameterised circuits with symbolic angles

`PhasedOutcomeCompleteSimulation` tracks the exact global phase, so two circuits that share the same
free rotation angles can be checked for exact equality — even for exponents of higher-weight Paulis:

```python
from paulimer import PhasedOutcomeCompleteSimulation, SparsePauli, UnitaryOpcode


def prepared_action(build):
    sim = PhasedOutcomeCompleteSimulation(2)
    for qubit in range(2):
        sim.apply_unitary(UnitaryOpcode.Hadamard, [qubit])  # |++>
    build(sim)
    return sim.phased_action([], [0, 1])  # state-preparation action


def direct(sim):  # e^{i alpha Z0 Z1} |++>
    alpha = sim.allocate_symbolic_angle()
    sim.apply_symbolic_pauli_exp(SparsePauli("Z_0 Z_1"), alpha)


def conjugated(sim):  # CNOT . e^{i alpha Z1} . CNOT |++>
    sim.apply_unitary(UnitaryOpcode.ControlledX, [0, 1])
    alpha = sim.allocate_symbolic_angle()
    sim.apply_symbolic_pauli_exp(SparsePauli("Z_1"), alpha)
    sim.apply_unitary(UnitaryOpcode.ControlledX, [0, 1])


assert prepared_action(direct).is_equivalent(prepared_action(conjugated))
```

### Decomposing a Clifford into pi/4 Pauli exponents

`CliffordUnitary.to_pauli_exponents()` returns an ordered product of `pi/4` Pauli exponents that
reproduces the Clifford exactly, including the Pauli-image signs — so replaying it with exact phase
tracking yields a well-defined global phase:

```python
from paulimer import CliffordUnitary, UnitaryOpcode

clifford = CliffordUnitary.identity(2)
clifford.left_mul(UnitaryOpcode.Hadamard, [0])
clifford.left_mul(UnitaryOpcode.ControlledX, [0, 1])

exponents = clifford.to_pauli_exponents()  # list[SparsePauli], each factor exp(+-i pi/4 P)

rebuilt = CliffordUnitary.identity(2)
for pauli in exponents:
    rebuilt.left_mul_pauli_exp(pauli)
assert rebuilt == clifford
```

## Features

- **DensePauli / SparsePauli** - Pauli operators with phase tracking and multiplication
- **CliffordUnitary** - Clifford gates with conjugation and composition
  (including `to_pauli_exponents()`, an exact decomposition into `pi/4` Pauli exponents)
- **PauliGroup** - Group operations including membership testing and factorization
- **Stabilizer Simulation** - Noiseless (OutcomeComplete, OutcomeFree, OutcomeSpecific) and noisy (Faulty) modes
- **PhasedOutcomeCompleteSimulation** - Outcome-complete simulation that additionally tracks the exact global phase, enabling exact equality checking of parameterised (symbolic-angle) circuits

## Use Cases

Designed for quantum error correction research, including stabilizer circuit analysis and Clifford circuit verification. `PhasedOutcomeCompleteSimulation` extends this to exact verification of non-stabilizer circuits built from symbolic Pauli rotations `e^{i alpha P}`.

## Performance

Built on `binar` for SIMD-accelerated binary linear algebra. 

## API Reference

See [paulimer.pyi](paulimer.pyi) for complete type hints and documentation.

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! See [github.com/microsoft/qdk-ec](https://github.com/microsoft/qdk-ec) for guidelines.
