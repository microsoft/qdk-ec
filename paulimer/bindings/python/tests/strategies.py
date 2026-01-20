"""Hypothesis strategies for property-based testing of paulimer.

These strategies generate random instances of paulimer types for use
with hypothesis-based property tests.
"""

from hypothesis import strategies as st

from paulimer import (
    FaultySimulation,
    SparsePauli,
    UnitaryOpcode,
)


@st.composite
def sparse_pauli_strategy(draw, max_qubits: int = 10):
    """Generate a random SparsePauli string."""
    num_qubits = draw(st.integers(min_value=1, max_value=max_qubits))
    chars = draw(st.lists(
        st.sampled_from(["I", "X", "Y", "Z"]),
        min_size=num_qubits,
        max_size=num_qubits,
    ))
    return SparsePauli("".join(chars))


@st.composite
def unitary_instruction_strategy(draw, sim: FaultySimulation, num_qubits: int):
    """Apply a random unitary instruction to the simulation."""
    single_qubit_ops = [
        UnitaryOpcode.Hadamard,
        UnitaryOpcode.SqrtZ,
        UnitaryOpcode.SqrtZInv,
        UnitaryOpcode.SqrtX,
        UnitaryOpcode.SqrtXInv,
        UnitaryOpcode.SqrtY,
        UnitaryOpcode.SqrtYInv,
    ]
    two_qubit_ops = [
        UnitaryOpcode.ControlledX,
        UnitaryOpcode.ControlledZ,
        UnitaryOpcode.Swap,
    ]

    if num_qubits < 2:
        opcode = draw(st.sampled_from(single_qubit_ops))
        qubit = draw(st.integers(min_value=0, max_value=num_qubits - 1))
        sim.apply_unitary(opcode, [qubit])
        return

    is_two_qubit = draw(st.booleans())
    if is_two_qubit:
        opcode = draw(st.sampled_from(two_qubit_ops))
        qubit_a = draw(st.integers(min_value=0, max_value=num_qubits - 1))
        qubit_b = draw(st.integers(min_value=0, max_value=num_qubits - 1).filter(lambda x: x != qubit_a))
        sim.apply_unitary(opcode, [qubit_a, qubit_b])
    else:
        opcode = draw(st.sampled_from(single_qubit_ops))
        qubit = draw(st.integers(min_value=0, max_value=num_qubits - 1))
        sim.apply_unitary(opcode, [qubit])


@st.composite
def simulation_strategy(
    draw,
    min_qubits: int = 1,
    max_qubits: int = 10,
    min_instructions: int = 1,
    max_instructions: int = 20,
    allow_measurements: bool = True,
):
    """Generate a random FaultySimulation for property-based testing."""
    num_qubits = draw(st.integers(min_value=min_qubits, max_value=max_qubits))
    num_instructions = draw(st.integers(min_value=min_instructions, max_value=max_instructions))

    sim = FaultySimulation()

    for _ in range(num_instructions):
        instruction_type = draw(st.integers(min_value=0, max_value=9 if allow_measurements else 6))

        if instruction_type <= 6:
            draw(unitary_instruction_strategy(sim, num_qubits))
        elif instruction_type <= 8:
            weight = draw(st.integers(min_value=1, max_value=min(3, num_qubits)))
            chars = ["I"] * num_qubits
            positions = draw(st.lists(
                st.integers(min_value=0, max_value=num_qubits - 1),
                min_size=weight,
                max_size=weight,
                unique=True,
            ))
            for pos in positions:
                chars[pos] = draw(st.sampled_from(["X", "Y", "Z"]))
            observable = SparsePauli("".join(chars))
            sim.measure(observable)
        else:
            sim.allocate_random_bit()

    return sim
