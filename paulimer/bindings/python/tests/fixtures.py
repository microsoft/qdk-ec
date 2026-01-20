"""Simulation factory functions for paulimer tests.

These functions create common circuit patterns using FaultySimulation's
circuit-builder API.
"""

from paulimer import (
    FaultySimulation,
    SparsePauli,
    UnitaryOpcode,
)


def make_bell_circuit() -> FaultySimulation:
    """Create a Bell state preparation with ZZ and XX measurements."""
    sim = FaultySimulation()
    sim.apply_unitary(UnitaryOpcode.Hadamard, [0])
    sim.apply_unitary(UnitaryOpcode.ControlledX, [0, 1])
    sim.measure(SparsePauli("ZZ"))
    sim.measure(SparsePauli("XX"))
    return sim


def make_ghz_circuit(num_qubits: int) -> FaultySimulation:
    """Create a GHZ state preparation with Z...Z measurement."""
    sim = FaultySimulation()
    sim.apply_unitary(UnitaryOpcode.Hadamard, [0])
    for i in range(num_qubits - 1):
        sim.apply_unitary(UnitaryOpcode.ControlledX, [i, i + 1])
    all_z = "Z" * num_qubits
    sim.measure(SparsePauli(all_z))
    return sim


def make_repetition_code_circuit(distance: int, rounds: int) -> FaultySimulation:
    """Create a repetition code with ZZ stabilizer measurements."""
    sim = FaultySimulation()
    for _ in range(rounds):
        for i in range(distance - 1):
            pauli_str = "I" * i + "ZZ" + "I" * (distance - i - 2)
            sim.measure(SparsePauli(pauli_str))
    return sim
