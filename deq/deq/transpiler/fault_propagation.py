"""Shared propagation of circuit-level Pauli faults into decoder error rows."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field

import stim
from paulimer import FramePropagator, SparsePauli, UnitaryOpcode

import deq.proto.deq_bin_pb2 as bin_pb
import deq.proto.deq_jit_pb2 as jit_pb
import deq.proto.util_pb2 as util_pb
from deqagram import (
    CodeDefinition,
    InputPort,
    OutputPort,
)
from deq.transpiler.circuit_lowering import DecomposedBody
from deq.transpiler.jit_transpiler import (
    PortColumnLayout,
    select_stabilizer_generators,
)
from deq.transpiler.stim_constants import (
    format_pauli_string,
    pauli_product_to_stim,
    pauli_string_to_sparse,
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
    """Measurement, output-stabilizer, and frame-column propagation result.

    Each field lists only the flipped positions.
    """

    flipped_real: set[int]
    flipped_output_stabilizers: Sequence[int]
    flipped_frame_columns: Sequence[int]


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
    finished_checks_by_measurement: dict[int, list[int]] = field(init=False)
    unfinished_checks_by_measurement: dict[int, list[int]] = field(init=False)
    readouts_by_real_measurement: dict[int, list[int]] = field(init=False)
    logical_rows_by_real_measurement: dict[int, list[int]] = field(init=False)

    def __post_init__(self) -> None:
        for name, groups in (
            ("finished_checks_by_measurement", enumerate(self.finished_member_lists)),
            ("unfinished_checks_by_measurement", enumerate(self.unfinished_member_lists)),
            ("readouts_by_real_measurement", enumerate(self.readout_measurement_sets)),
            ("logical_rows_by_real_measurement", self.physical_correction_by_logical.items()),
        ):
            object.__setattr__(self, name, _invert_membership(groups))

    def output_stabilizer_measurement_index(
        self, stabilizer_index: int
    ) -> int:
        """Map an output-stabilizer position to the global measurement index."""
        return self.output_stabilizer_measurement_offset + stabilizer_index


def _invert_membership(
    groups: Iterable[tuple[int, Collection[int]]],
) -> dict[int, list[int]]:
    """Map each member to the keys of the groups containing it."""
    keys_by_member: dict[int, list[int]] = {}
    for key, members in groups:
        for member in members:
            keys_by_member.setdefault(member, []).append(key)
    return keys_by_member


def _odd_parity_keys(
    keys_by_member: dict[int, list[int]], members: Iterable[int]
) -> set[int]:
    """Return the keys of the groups that contain an odd number of distinct *members*."""
    odd: set[int] = set()
    for member in members:
        odd.symmetric_difference_update(keys_by_member.get(member, ()))
    return odd


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


def _measure_with_record_flips(
    propagator: FramePropagator,
    observable: SparsePauli,
    real_measurement_outcomes: list[int],
    measurement_fault_columns: dict[int, int],
) -> None:
    fault_column = measurement_fault_columns.get(len(real_measurement_outcomes))
    outcome = propagator.measure(observable)
    real_measurement_outcomes.append(outcome)
    if fault_column is not None:
        propagator.inject_outcome_flip(fault_column, outcome)


def _apply_instruction(
    propagator: FramePropagator,
    instruction: stim.CircuitInstruction,
    real_measurement_outcomes: list[int],
    measurement_fault_columns: dict[int, int],
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
                _measure_with_record_flips(
                    propagator, SparsePauli.z(target.value), real_measurement_outcomes, measurement_fault_columns,
                )
        case "R":
            for target in targets:
                propagator.reset_qubit(target.value)
        case "MPAD":
            for _target in targets:
                _measure_with_record_flips(
                    propagator, SparsePauli.identity(), real_measurement_outcomes, measurement_fault_columns,
                )
        case other:
            raise ValueError(
                f"fault propagation encountered unexpected primitive {other}"
            )


def propagate_pauli_mechanisms(
    mechanisms: Sequence[tuple[int, stim.PauliString]],
    body: DecomposedBody,
    output_stabilizer_paulis: Sequence[stim.PauliString],
    frame_column_paulis: Sequence[stim.PauliString],
    *,
    measurement_flips: Sequence[int] = (),
) -> list[MechanismFlips]:
    """Propagate Pauli faults, then record-only faults, through the lowered body.

    Each batched shot is an independent fault column, not a sampled execution.
    ``measurement_flips`` contains distinct real measurement indices.
    Record-only faults toggle outcome deltas without changing qubit frames.
    Both real and padded outcomes propagate through subsequent feedback normally.
    """
    shot_count = len(mechanisms) + len(measurement_flips)
    propagator = FramePropagator(
        body.qubit_count,
        body.total_measurements
        + len(output_stabilizer_paulis)
        + len(frame_column_paulis),
        shot_count,
    )
    shots_by_start: dict[int, list[int]] = {}
    for shot, (start, _pauli) in enumerate(mechanisms):
        shots_by_start.setdefault(start, []).append(shot)
    measurement_fault_columns: dict[int, int] = {}
    for fault_column, real_index in enumerate(measurement_flips, start=len(mechanisms)):
        if not 0 <= real_index < body.total_measurements:
            raise ValueError("measurement fault index is outside the lowered body")
        if real_index in measurement_fault_columns:
            raise ValueError("measurement fault indices must be distinct")
        measurement_fault_columns[real_index] = fault_column

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
        _apply_instruction(
            propagator, instruction, real_measurement_outcomes, measurement_fault_columns,
        )
    inject_at(len(body.instructions))
    assert injected == len(mechanisms), "each Pauli mechanism must be injected exactly once"

    output_stabilizer_outcomes = [
        propagator.measure(pauli_string_to_sparse(pauli))
        for pauli in output_stabilizer_paulis
    ]
    frame_column_outcomes = [
        propagator.measure(pauli_string_to_sparse(pauli))
        for pauli in frame_column_paulis
    ]
    roles_by_outcome: dict[int, list[tuple[int, int]]] = {}
    for role, outcomes in enumerate(
        (real_measurement_outcomes, output_stabilizer_outcomes, frame_column_outcomes)
    ):
        for index, outcome in enumerate(outcomes):
            roles_by_outcome.setdefault(outcome, []).append((role, index))

    mechanism_flips: list[MechanismFlips] = []
    for flipped_outcomes in propagator.outcome_deltas.sparse_columns():
        flipped_by_role: tuple[list[int], list[int], list[int]] = ([], [], [])
        for outcome in flipped_outcomes:
            for role, index in roles_by_outcome.get(outcome, ()):
                flipped_by_role[role].append(index)
        flipped_real, flipped_output_stabilizers, flipped_frame_columns = flipped_by_role
        mechanism_flips.append(
            MechanismFlips(
                flipped_real=set(flipped_real),
                flipped_output_stabilizers=flipped_output_stabilizers,
                flipped_frame_columns=flipped_frame_columns,
            )
        )
    return mechanism_flips


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
    flipped_measurements.update(
        context.output_stabilizer_measurement_index(stabilizer_index)
        for stabilizer_index in flips.flipped_output_stabilizers
    )

    finished_flipped = sorted(
        _odd_parity_keys(
            context.finished_checks_by_measurement, flipped_measurements
        )
    )
    unfinished_flipped = sorted(
        _odd_parity_keys(
            context.unfinished_checks_by_measurement, flipped_measurements
        )
    )

    residual: set[int] = {
        frame_column
        for frame_column in flips.flipped_frame_columns
        if frame_column in context.logical_columns
    }
    residual ^= _odd_parity_keys(
        context.logical_rows_by_real_measurement, flips.flipped_real
    )
    for check_index in unfinished_flipped:
        column = context.unfinished_to_column[check_index]
        if column is not None:
            residual ^= {column}

    readout_flips = sorted(
        _odd_parity_keys(
            context.readouts_by_real_measurement, flips.flipped_real
        )
    )
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