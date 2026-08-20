"""Shared propagation of circuit-level Pauli faults into decoder error rows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import stim
from paulimer import FramePropagator, SparsePauli, UnitaryOpcode

import deq.proto.deq_bin_pb2 as bin_pb
import deq.proto.deq_jit_pb2 as jit_pb
import deq.proto.util_pb2 as util_pb
from deq.circuit.model import (
    CheckStatement,
    CodeDefinition,
    ConditionalStatement,
    ErrorStatement,
    GadgetStatement,
    InputPort,
    Instruction,
    LossStatement,
    OutputPort,
    PreselectStatement,
    PropagateStatement,
    ReadoutStatement,
    RepeatBlock,
    VirtualLogicalStatement,
)
from deq.transpiler.jit_transpiler import (
    PortColumnLayout,
    select_stabilizer_generators,
)
from deq.transpiler.stim_constants import (
    ANNOTATION_INSTRUCTIONS,
    NOISE_INSTRUCTIONS_ALL,
    format_pauli_string,
    instruction_num_measurements,
    pauli_product_to_stim,
    pauli_string_to_sparse,
)


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


@dataclass(frozen=True)
class DecomposedBody:
    """Primitive instructions and body-index mapping for one gadget body."""

    instructions: tuple[stim.CircuitInstruction, ...]
    measurement_start_at: tuple[int, ...]
    total_measurements: int
    body_start_at: tuple[int, ...]


def build_decomposed_body(
    flat_body: Sequence[GadgetStatement],
) -> DecomposedBody:
    """Decompose a flattened gadget body without merging source statements.

    Each source instruction is decomposed independently so its start boundary
    remains explicit without inserting a circuit instruction as a separator.
    Non-gate body entries map to the next gate boundary; entries after the last
    gate map to the terminal boundary.

    .. warning::
        ``flat_body`` must already have been processed by :func:`flatten_body`.
        Unflattened repeat blocks and unknown statement types are rejected.
    """
    instructions: list[stim.CircuitInstruction] = []
    gate_body_indices: list[int] = []
    gate_starts: list[int] = []
    for body_index, statement in enumerate(flat_body):
        if isinstance(statement, RepeatBlock):
            raise ValueError(
                "build_decomposed_body requires a flattened gadget body; "
                "call flatten_body before decomposing REPEAT blocks"
            )
        if not isinstance(statement, Instruction):
            if not isinstance(statement, _FLAT_METADATA_TYPES):
                raise TypeError(
                    "unsupported gadget body statement in fault propagation: "
                    f"{type(statement).__name__}"
                )
            continue
        name = statement.name.upper()
        if name in NOISE_INSTRUCTIONS_ALL or name in ANNOTATION_INSTRUCTIONS:
            continue
        gate_body_indices.append(body_index)
        gate_starts.append(len(instructions))
        instructions.extend(
            stim.Circuit(
                str(
                    Instruction(
                        name=statement.name,
                        arguments=statement.arguments,
                        targets=statement.targets,
                    )
                )
            )
            .decomposed()
        )

    measurement_starts: list[int] = []
    measurement_count = 0
    for instruction in instructions:
        measurement_starts.append(measurement_count)
        measurement_count += instruction_num_measurements(str(instruction))

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
    )


def build_port_paulis(
    ports: Sequence[InputPort | OutputPort],
    codes: dict[str, CodeDefinition],
    num_qubits: int,
) -> tuple[list[stim.PauliString], list[stim.PauliString]]:
    """Build output-stabilizer and frame-column Paulis for concatenated ports."""
    output_stabilizer_paulis: list[stim.PauliString] = []
    frame_column_paulis: list[stim.PauliString] = []
    for port in ports:
        code = codes[port.code_name]
        local_to_global = {
            local_qubit: global_qubit
            for local_qubit, global_qubit in enumerate(port.qubit_indices)
        }
        for stabilizer in code.stabilizers:
            output_stabilizer_paulis.append(
                pauli_product_to_stim(
                    stabilizer, num_qubits, local_to_global
                )
            )
        for logical in code.logicals:
            frame_column_paulis.append(
                pauli_product_to_stim(
                    logical.x_operator, num_qubits, local_to_global
                )
            )
            frame_column_paulis.append(
                pauli_product_to_stim(
                    logical.z_operator, num_qubits, local_to_global
                )
            )
        selected = select_stabilizer_generators(code)
        for generator_index in selected.generator_indices:
            frame_column_paulis.append(
                pauli_product_to_stim(
                    code.stabilizers[generator_index],
                    num_qubits,
                    local_to_global,
                )
            )
    return output_stabilizer_paulis, frame_column_paulis


@dataclass(frozen=True)
class MechanismFlips:
    """Measurement, output-stabilizer, and frame-column propagation result."""

    flipped_real: set[int]
    output_stabilizer_flips: Sequence[bool]
    frame_column_flips: Sequence[bool]


@dataclass(frozen=True)
class ErrorProjectionContext:
    """Per-gadget invariants used to lower propagated faults to error rows."""

    input_virtual_count: int
    finished_member_lists: Sequence[frozenset[int]]
    unfinished_member_lists: Sequence[frozenset[int]]
    output_stabilizer_measurement_offset: int
    readout_measurement_sets: Sequence[set[int]]
    logical_columns: set[int]
    unfinished_to_column: Sequence[int | None]
    physical_correction_by_logical: dict[int, set[int]]

    def output_stabilizer_measurement_index(
        self, stabilizer_index: int
    ) -> int:
        """Map an output-stabilizer position to the global measurement index."""
        return self.output_stabilizer_measurement_offset + stabilizer_index


def build_error_projection_context(
    *,
    output_ports: Sequence[OutputPort],
    codes: dict[str, CodeDefinition],
    input_virtual_count: int,
    finished_checks: Sequence[tuple[frozenset[int], bool]],
    unfinished_checks: Sequence[tuple[frozenset[int], bool]],
    output_virtual_start: int,
    readouts: Sequence[bin_pb.GadgetType.Readout],
    physical_correction: util_pb.BitMatrix,
) -> ErrorProjectionContext:
    """Build the shared lowering context for one gadget."""
    output_layout = PortColumnLayout(output_ports, codes)
    logical_columns = output_layout.logical_columns
    physical_correction_by_logical = {
        row: set() for row in logical_columns
    }
    for row, column in zip(physical_correction.i, physical_correction.j):
        if row in physical_correction_by_logical:
            physical_correction_by_logical[row].add(column)
    return ErrorProjectionContext(
        input_virtual_count=input_virtual_count,
        finished_member_lists=[members for members, _ in finished_checks],
        unfinished_member_lists=[members for members, _ in unfinished_checks],
        output_stabilizer_measurement_offset=output_virtual_start,
        readout_measurement_sets=[
            set(readout.measurement_indices) for readout in readouts
        ],
        logical_columns=logical_columns,
        unfinished_to_column=output_layout.stab_to_column,
        physical_correction_by_logical=physical_correction_by_logical,
    )


_FRAME_H = UnitaryOpcode.Hadamard
_FRAME_S = UnitaryOpcode.SqrtZ
_FRAME_CX = UnitaryOpcode.ControlledX


def _apply_instruction(
    propagator: FramePropagator,
    instruction: stim.CircuitInstruction,
    real_measurement_outcomes: list[int],
) -> None:
    targets = instruction.targets_copy()
    match instruction.name:
        case "H":
            for target in targets:
                propagator.apply_unitary(_FRAME_H, [target.value])
        case "S":
            for target in targets:
                propagator.apply_unitary(_FRAME_S, [target.value])
        case "CX":
            for index in range(0, len(targets), 2):
                control, target = targets[index], targets[index + 1]
                if control.is_measurement_record_target:
                    record_index = len(real_measurement_outcomes) + control.value
                    assert 0 <= record_index < len(real_measurement_outcomes)
                    propagator.apply_conditional_pauli(
                        SparsePauli.x(target.value),
                        [real_measurement_outcomes[record_index]],
                    )
                else:
                    propagator.apply_unitary(
                        _FRAME_CX, [control.value, target.value]
                    )
        case "M":
            for target in targets:
                real_measurement_outcomes.append(
                    propagator.measure(SparsePauli.z(target.value))
                )
        case "R":
            for target in targets:
                propagator.reset_qubit(target.value)
        case "MPAD":
            for _target in targets:
                real_measurement_outcomes.append(
                    propagator.measure(SparsePauli.identity())
                )
        case other:
            raise ValueError(
                f"fault propagation encountered unexpected primitive {other}"
            )


def propagate_pauli_mechanisms(
    mechanisms: Sequence[tuple[int, stim.PauliString]],
    body: DecomposedBody,
    num_qubits: int,
    output_stabilizer_paulis: Sequence[stim.PauliString],
    frame_column_paulis: Sequence[stim.PauliString],
) -> list[MechanismFlips]:
    """Propagate all injected Pauli mechanisms in one batched frame walk."""
    shot_count = len(mechanisms)
    propagator = FramePropagator(
        num_qubits,
        body.total_measurements
        + len(output_stabilizer_paulis)
        + len(frame_column_paulis),
        shot_count,
    )
    shots_by_start: dict[int, list[int]] = {}
    for shot, (start, _pauli) in enumerate(mechanisms):
        shots_by_start.setdefault(start, []).append(shot)

    injected = 0

    def inject_at(boundary: int) -> None:
        nonlocal injected
        for shot in shots_by_start.get(boundary, ()):
            propagator.inject_pauli(
                shot, pauli_string_to_sparse(mechanisms[shot][1])
            )
            injected += 1

    real_measurement_outcomes: list[int] = []
    for boundary, instruction in enumerate(body.instructions):
        inject_at(boundary)
        _apply_instruction(propagator, instruction, real_measurement_outcomes)
    inject_at(len(body.instructions))
    assert injected == shot_count, "each mechanism must be injected exactly once"

    output_stabilizer_outcomes = [
        propagator.measure(pauli_string_to_sparse(pauli))
        for pauli in output_stabilizer_paulis
    ]
    frame_column_outcomes = [
        propagator.measure(pauli_string_to_sparse(pauli))
        for pauli in frame_column_paulis
    ]
    shots_by_outcome = [row.support for row in propagator.outcome_deltas.rows]

    flipped_real = [set() for _ in range(shot_count)]
    for real_index, outcome in enumerate(real_measurement_outcomes):
        for shot in shots_by_outcome[outcome]:
            flipped_real[shot].add(real_index)
    output_stabilizer_flips = [
        [False] * len(output_stabilizer_paulis) for _ in range(shot_count)
    ]
    for stabilizer_index, outcome in enumerate(output_stabilizer_outcomes):
        for shot in shots_by_outcome[outcome]:
            output_stabilizer_flips[shot][stabilizer_index] = True
    frame_column_flips = [
        [False] * len(frame_column_paulis) for _ in range(shot_count)
    ]
    for frame_column, outcome in enumerate(frame_column_outcomes):
        for shot in shots_by_outcome[outcome]:
            frame_column_flips[shot][frame_column] = True
    return [
        MechanismFlips(
            flipped_real=flipped_real[shot],
            output_stabilizer_flips=output_stabilizer_flips[shot],
            frame_column_flips=frame_column_flips[shot],
        )
        for shot in range(shot_count)
    ]


def build_error_row_from_flips(
    *,
    site_name: str,
    site_pauli: stim.PauliString,
    probability: float,
    flips: MechanismFlips,
    context: ErrorProjectionContext,
) -> jit_pb.JitGadgetType.Error | None:
    """Lower one propagated mechanism footprint to a decoder error row."""
    flipped_measurements = {
        real + context.input_virtual_count for real in flips.flipped_real
    }
    for stabilizer_index, flipped in enumerate(
        flips.output_stabilizer_flips
    ):
        if flipped:
            flipped_measurements.add(
                context.output_stabilizer_measurement_index(stabilizer_index)
            )

    finished_flipped = [
        check_index
        for check_index, members in enumerate(context.finished_member_lists)
        if len(members & flipped_measurements) % 2 == 1
    ]
    unfinished_flipped = [
        check_index
        for check_index, members in enumerate(context.unfinished_member_lists)
        if len(members & flipped_measurements) % 2 == 1
    ]

    residual: set[int] = {
        frame_column
        for frame_column, flipped in enumerate(flips.frame_column_flips)
        if frame_column in context.logical_columns and flipped
    }
    for logical_row, columns in context.physical_correction_by_logical.items():
        if len(columns & flips.flipped_real) % 2 == 1:
            residual ^= {logical_row}
    for check_index in unfinished_flipped:
        column = context.unfinished_to_column[check_index]
        if column is not None:
            residual ^= {column}

    readout_flips = [
        readout_index
        for readout_index, measurements in enumerate(
            context.readout_measurement_sets
        )
        if len(measurements & flips.flipped_real) % 2 == 1
    ]
    if not (
        finished_flipped or unfinished_flipped or residual or readout_flips
    ):
        return None

    return jit_pb.JitGadgetType.Error(
        base=bin_pb.ErrorModelType.Error(
            tag=f"{site_name} {format_pauli_string(site_pauli)}",
            residual=sorted(residual),
            readout_flips=readout_flips,
            probability=probability,
        ),
        finished_checks=finished_flipped,
        unfinished_checks=unfinished_flipped,
    )