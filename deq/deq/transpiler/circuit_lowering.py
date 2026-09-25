"""Shared circuit preparation for decoder analyses.

This module owns body views, Clifford lowering, and source-boundary mappings.
It does not depend on check discovery, logical-flow analysis, or fault
propagation. Physical simulation uses the unlowered simulation view.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import remainder

import stim

from deq.circuit.model import (
    CheckStatement,
    ConditionalStatement,
    ErrorStatement,
    GadgetStatement,
    InputPort,
    Instruction,
    LossStatement,
    LossTarget,
    OutputPort,
    PauliTarget,
    PreselectStatement,
    PropagateStatement,
    QubitTarget,
    ReadoutStatement,
    RepeatBlock,
    VirtualLogicalStatement,
)
from deq.defaults import DEFAULT_U3_AXIS_TOLERANCE
from deq.transpiler.stim_constants import (
    ANNOTATION_INSTRUCTIONS,
    NON_CLIFFORD_AXES,
    NON_CLIFFORD_INSTRUCTIONS,
    NON_CLIFFORD_PAIR_AXES,
    NON_CLIFFORD_PRODUCT_GATES,
    NOISE_INSTRUCTIONS_ALL,
    non_clifford_pauli_products,
    validate_non_clifford_instruction,
)

_KNOWN_INSTRUCTION_DECORATORS = frozenset({"SIMULATE_ONLY", "DECODE_ONLY"})

_FLAT_METADATA_TYPES = (
    InputPort,
    OutputPort,
    ReadoutStatement,
    CheckStatement,
    ErrorStatement,
    LossStatement,
    ConditionalStatement,
    VirtualLogicalStatement,
    PropagateStatement,
    PreselectStatement,
)


def is_simulation_only(stmt: GadgetStatement) -> bool:
    """True if the statement carries an ``@SIMULATE_ONLY`` decorator."""
    return isinstance(stmt, Instruction) and any(
        decorator.name == "SIMULATE_ONLY" for decorator in stmt.decorators
    )


def is_decode_only(stmt: GadgetStatement) -> bool:
    """True if the statement carries an ``@DECODE_ONLY`` decorator."""
    return isinstance(stmt, Instruction) and any(
        decorator.name == "DECODE_ONLY" for decorator in stmt.decorators
    )


def _validate_instruction_decorators(stmt: GadgetStatement) -> None:
    """Raise on unrecognized or conflicting instruction-level decorators."""
    if not isinstance(stmt, Instruction) or not stmt.decorators:
        return
    names = set()
    for deco in stmt.decorators:
        if deco.name not in _KNOWN_INSTRUCTION_DECORATORS:
            raise ValueError(
                f"unrecognized instruction decorator @{deco.name} on "
                f"'{stmt.name}'; known instruction decorators are: "
                f"{', '.join(sorted(_KNOWN_INSTRUCTION_DECORATORS))}"
            )
        names.add(deco.name)
    if "SIMULATE_ONLY" in names and "DECODE_ONLY" in names:
        raise ValueError(
            f"instruction '{stmt.name}' has both @SIMULATE_ONLY and "
            f"@DECODE_ONLY; these are mutually exclusive"
        )


def flatten_body(
    statements: Sequence[GadgetStatement],
    *,
    for_simulate: bool = False,
) -> list[GadgetStatement]:
    """Expand ``REPEAT`` blocks inline; filter by decode/simulate view.

    Parameters
    ----------
    for_simulate : bool
        ``False`` (default): decode view, excluding ``@SIMULATE_ONLY``.
        ``True``: simulate view, excluding ``@DECODE_ONLY``.
    """
    flat: list[GadgetStatement] = []
    for stmt in statements:
        if isinstance(stmt, RepeatBlock):
            body = list(stmt.body)
            for _ in range(stmt.count):
                flat.extend(flatten_body(body, for_simulate=for_simulate))
        else:
            _validate_instruction_decorators(stmt)
            if not for_simulate and is_simulation_only(stmt):
                continue
            if for_simulate and is_decode_only(stmt):
                continue
            flat.append(stmt)
    return flat


def max_qubit_index(statements: Sequence[GadgetStatement]) -> int:
    """Return the largest physical qubit index referenced anywhere in the body."""
    max_idx = -1
    for stmt in statements:
        if isinstance(stmt, Instruction):
            for target in stmt.targets:
                if isinstance(target, QubitTarget):
                    max_idx = max(max_idx, target.index)
                elif isinstance(target, (PauliTarget, LossTarget)):
                    max_idx = max(max_idx, target.index)
        elif isinstance(stmt, RepeatBlock):
            max_idx = max(max_idx, max_qubit_index(list(stmt.body)))
        elif isinstance(stmt, (InputPort, OutputPort)):
            for qubit in stmt.qubit_indices:
                max_idx = max(max_idx, qubit)
    return max_idx


def _u3_dephasing_axes(arguments: Sequence[float]) -> tuple[str, ...]:
    """Recognize Pauli axes up to global phase within an angular tolerance.

    For U = Rz(phi) Ry(theta) Rz(lam), its X, Y, Z coefficients are
    proportional to -sin(theta/2) sin((phi-lam)/2),
    sin(theta/2) cos((phi-lam)/2), and cos(theta/2) sin((phi+lam)/2).
    Angles in these formulas are radians. Modular relations on the stored
    half-turn arguments use an absolute tolerance of 1e-12, with no relative
    tolerance. This decoder-only approximation leaves physical angles intact.
    """
    theta, phi, lam = (remainder(value, 2) for value in arguments)
    if abs(theta) <= DEFAULT_U3_AXIS_TOLERANCE:
        return ("Z",)
    if (
        abs(abs(theta) - 1) <= DEFAULT_U3_AXIS_TOLERANCE
        or abs(remainder(phi + lam, 2)) <= DEFAULT_U3_AXIS_TOLERANCE
    ):
        difference = abs(remainder(phi - lam, 2))
        if abs(difference - 1) <= DEFAULT_U3_AXIS_TOLERANCE:
            return ("X",)
        if difference <= DEFAULT_U3_AXIS_TOLERANCE:
            return ("Y",)
    return ("Z", "Y", "Z")


def non_clifford_dephasing_circuit(
    instruction: Instruction, auxiliary_qubit: int
) -> stim.Circuit:
    """Lower a non-Clifford gate to a conservative channel without record bits.

    Each Pauli rotation uses one fresh |+> auxiliary controlling the entire
    Pauli product. Resetting it makes successive rotations independent.
    U3 first selects a Pauli axis within the angular tolerance, otherwise
    dephases its Z-Y-Z expansion. CH uses Ry(pi/4), CX, Ry(-pi/4). CCZ uses
    the seven nonconstant Z products in its phase polynomial, with CCX
    obtained by conjugating the target by H. Every constituent rotation is
    dephased, without simplifying its angle or cancelling adjacent rotations.

    The auxiliary must be outside the physical-qubit range. This lowering is
    shared by check, flow, and fault analysis, never physical simulation output.
    """
    validate_non_clifford_instruction(instruction)
    circuit = stim.Circuit()

    def dephase(terms: Sequence[tuple[int, str]]) -> None:
        circuit.append("RX", [auxiliary_qubit])
        for qubit, axis in terms:
            circuit.append("C" + axis, [auxiliary_qubit, qubit])
        circuit.append("R", [auxiliary_qubit])

    name = instruction.name.upper()
    if name in NON_CLIFFORD_PRODUCT_GATES:
        for product in non_clifford_pauli_products(instruction):
            dephase([
                (qubit, "IXYZ"[product[qubit]])
                for qubit in range(len(product)) if product[qubit]
            ])
        return circuit
    qubits = [target.index for target in instruction.targets if isinstance(target, QubitTarget)]
    if name in NON_CLIFFORD_AXES:
        for qubit in qubits:
            dephase([(qubit, NON_CLIFFORD_AXES[name])])
    elif name in NON_CLIFFORD_PAIR_AXES:
        for offset in range(0, len(qubits), 2):
            dephase([
                (qubit, NON_CLIFFORD_PAIR_AXES[name])
                for qubit in qubits[offset:offset + 2]
            ])
    elif name in {"U", "U3"}:
        axes = _u3_dephasing_axes(instruction.arguments)
        for qubit in qubits:
            for axis in axes:
                dephase([(qubit, axis)])
    elif name == "CH":
        for offset in range(0, len(qubits), 2):
            control, target = qubits[offset:offset + 2]
            dephase([(target, "Y")])
            circuit.append("CX", [control, target])
            dephase([(target, "Y")])
    elif name in {"CCX", "CCZ"}:
        for offset in range(0, len(qubits), 3):
            first, second, target = qubits[offset:offset + 3]
            if name == "CCX":
                circuit.append("H", [target])
            for support in (
                [first], [second], [target], [first, second],
                [first, target], [second, target], [first, second, target],
            ):
                dephase([(qubit, "Z") for qubit in support])
            if name == "CCX":
                circuit.append("H", [target])
    else:
        raise ValueError(f"No conservative lowering defined for {name}")
    return circuit


@dataclass(frozen=True)
class DecomposedBody:
    """Shared Clifford analysis circuit with source and measurement boundaries.

    ``qubit_count`` includes decoder-visible ports, noise targets, and any
    analysis-only auxiliary qubit, even when no primitive touches a port qubit.
    """

    instructions: tuple[stim.CircuitInstruction, ...]
    measurement_start_at: tuple[int, ...]
    total_measurements: int
    body_start_at: tuple[int, ...]
    qubit_count: int


def build_decomposed_body(
    flat_body: Sequence[GadgetStatement],
) -> DecomposedBody:
    """Lower a flattened decode body before check, flow, and fault analysis.

    Non-Clifford rotations become Clifford dephasing circuits on a private
    auxiliary qubit. Resets keep successive uses independent without adding
    physical measurement records. Source statements and simulation output
    remain unchanged; their boundaries map into the lowered instructions.

    Each source gate is decomposed independently so adjacent statements do
    not merge. Metadata and noise map to the next gate boundary. Call
    :func:`flatten_body` first to expand repeats and select the decode view.
    """
    instructions: list[stim.CircuitInstruction] = []
    gate_body_indices: list[int] = []
    gate_starts: list[int] = []
    auxiliary_qubit = max_qubit_index(list(flat_body)) + 1
    qubit_count = auxiliary_qubit
    for body_index, statement in enumerate(flat_body):
        if isinstance(statement, RepeatBlock):
            raise ValueError(
                "build_decomposed_body requires a flattened gadget body; "
                "call flatten_body before decomposing REPEAT blocks"
            )
        if not isinstance(statement, Instruction):
            if not isinstance(statement, _FLAT_METADATA_TYPES):
                raise TypeError(
                    "unsupported gadget body statement in analysis: "
                    f"{type(statement).__name__}"
                )
            continue
        name = statement.name.upper()
        if name in NOISE_INSTRUCTIONS_ALL or name in ANNOTATION_INSTRUCTIONS:
            continue
        gate_body_indices.append(body_index)
        gate_starts.append(len(instructions))
        if name in NON_CLIFFORD_INSTRUCTIONS:
            qubit_count = auxiliary_qubit + 1
            circuit = non_clifford_dephasing_circuit(statement, auxiliary_qubit)
        else:
            circuit = stim.Circuit(
                str(Instruction(
                    name=statement.name,
                    arguments=statement.arguments,
                    targets=statement.targets,
                ))
            )
        instructions.extend(circuit.decomposed())

    measurement_starts: list[int] = []
    measurement_count = 0
    for instruction in instructions:
        measurement_starts.append(measurement_count)
        measurement_count += instruction.num_measurements

    body_starts: list[int] = []
    gate_cursor = 0
    for body_index in range(len(flat_body)):
        if (
            gate_cursor < len(gate_body_indices)
            and body_index == gate_body_indices[gate_cursor]
        ):
            body_starts.append(gate_starts[gate_cursor])
            gate_cursor += 1
        elif gate_cursor < len(gate_body_indices):
            body_starts.append(gate_starts[gate_cursor])
        else:
            body_starts.append(len(instructions))

    return DecomposedBody(
        instructions=tuple(instructions),
        measurement_start_at=tuple(measurement_starts),
        total_measurements=measurement_count,
        body_start_at=tuple(body_starts),
        qubit_count=qubit_count,
    )
