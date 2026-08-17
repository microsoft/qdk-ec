"""Whole-gadget traversal and individual-gate dispatch for loss models."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import stim

from deq.circuit.model import (
    CombinerTarget,
    GadgetDefinition,
    Instruction,
    MeasurementRecordTarget,
    PauliTarget,
    QubitTarget,
)
from deq.transpiler.fault_propagation import build_decomposed_body
from deq.transpiler.jit_transpiler import flatten_body
from deq.transpiler.loss.api import (
    LossAnalysisState,
    LossGate,
    LossModel,
    UnsupportedLossModelError,
)
from deq.transpiler.loss.loss_graph import (
    LossBranch,
    LossEvent,
    LossEventGraph,
    PauliInsertion,
    build_loss_event_graph,
)
from deq.transpiler.stim_constants import (
    ANNOTATION_INSTRUCTIONS,
    NOISE_INSTRUCTIONS_ALL,
    instruction_num_measurements,
    split_mpp_targets,
)


@dataclass(frozen=True)
class LossAnalysisResult:
    """Loss graph plus physical-qubit mappings at the gadget boundary."""

    graph: LossEventGraph
    exit_qubits_by_event: dict[int, tuple[int, ...]]
    input_event_id_by_qubit: dict[int, int]


@dataclass
class _PendingLossBranch:
    qubit: int
    loss_boundary: int
    loss_measurements: set[int] = field(default_factory=set)
    continuation_pauli_insertions: set[PauliInsertion] = field(default_factory=set)
    active: bool = True
    successor_event_id: int | None = None

    def finish(self) -> LossBranch:
        return LossBranch(
            qubit=self.qubit,
            loss_boundary=self.loss_boundary,
            loss_measurements=tuple(self.loss_measurements),
            continuation_pauli_insertions=tuple(self.continuation_pauli_insertions),
            successor_event_id=self.successor_event_id,
        )


@dataclass
class _PendingLossEvent:
    event_id: int
    body_index: int
    target_index: int
    source_qubit: int
    loss_probability: float
    source_boundary: int
    branches: list[_PendingLossBranch]
    source_pauli_insertions: set[PauliInsertion] = field(default_factory=set)

    def finish(self) -> LossEvent:
        return LossEvent(
            event_id=self.event_id,
            body_index=self.body_index,
            target_index=self.target_index,
            source_qubit=self.source_qubit,
            loss_probability=self.loss_probability,
            source_boundary=self.source_boundary,
            branches=tuple(branch.finish() for branch in self.branches),
            source_pauli_insertions=tuple(self.source_pauli_insertions),
        )


class _MutableLossAnalysisState(LossAnalysisState):
    def __init__(self) -> None:
        self.events: dict[int, _PendingLossEvent] = {}
        self.pending: dict[int, list[tuple[_PendingLossEvent, _PendingLossBranch]]] = (
            defaultdict(list)
        )

    def add_source_event(
        self,
        *,
        event_id: int,
        body_index: int,
        target_index: int,
        qubit: int,
        probability: float,
        boundary: int,
    ) -> None:
        self._link_prior_losses_to_new_source(qubit, event_id)
        branch = _PendingLossBranch(
            qubit=qubit,
            loss_boundary=boundary,
        )
        event = _PendingLossEvent(
            event_id=event_id,
            body_index=body_index,
            target_index=target_index,
            source_qubit=qubit,
            loss_probability=probability,
            source_boundary=boundary,
            branches=[branch],
        )
        self.events[event_id] = event
        self.pending[qubit].append((event, branch))

    def _link_prior_losses_to_new_source(self, qubit: int, successor_event_id: int) -> None:
        """Share a new source's suffix with prior single-branch losses."""

        retained: list[tuple[_PendingLossEvent, _PendingLossBranch]] = []
        for event, branch in self.pending.get(qubit, ()):
            active_branch_count = sum(
                candidate_branch.active for candidate_branch in event.branches
            )
            # Share the later suffix only when this is the event's sole lifetime.
            if branch.active and active_branch_count == 1:
                branch.active = False
                branch.successor_event_id = successor_event_id
            else:
                retained.append((event, branch))
        # Linked branches continue through the successor instead of pending here.
        if retained:
            self.pending[qubit] = retained
        else:
            self.pending.pop(qubit, None)

    def active_event_ids(self, qubit: int) -> tuple[int, ...]:
        return tuple(
            sorted({event.event_id for event, _ in self.pending.get(qubit, ())})
        )

    def event_has_active_loss(self, event_id: int, qubit: int) -> bool:
        return any(
            event.event_id == event_id for event, _ in self.pending.get(qubit, ())
        )

    def add_continuation_pauli_insertion(
        self,
        qubit: int,
        boundary: int,
        generators: tuple[str, ...] = ("X", "Z"),
    ) -> None:
        for _, branch in self.pending.get(qubit, ()):
            branch.continuation_pauli_insertions.add(
                PauliInsertion(
                    boundary=boundary,
                    qubit=qubit,
                    generators=generators,
                )
            )

    # This is the event-specific form of the broadcast operation above.
    def add_event_continuation_pauli_insertion(
        self,
        event_id: int,
        *,
        branch_qubit: int,
        qubit: int,
        boundary: int,
        generators: tuple[str, ...] = ("X", "Z"),
    ) -> None:
        for event, branch in self.pending.get(branch_qubit, ()):
            if event.event_id == event_id:
                branch.continuation_pauli_insertions.add(
                    PauliInsertion(
                        boundary=boundary,
                        qubit=qubit,
                        generators=generators,
                    )
                )
                return
        raise ValueError(
            f"event {event_id} has no active loss branch on qubit {branch_qubit}"
        )

    def add_source_pauli_insertion(
        self,
        event_id: int,
        generators: tuple[str, ...] = ("X", "Z"),
    ) -> None:
        event = self.events[event_id]
        event.source_pauli_insertions.add(
            PauliInsertion(
                boundary=event.source_boundary,
                qubit=event.source_qubit,
                generators=generators,
            )
        )

    def add_loss_controlled_pauli_insertion(
        self,
        measurement_index: int,
        qubit: int,
        boundary: int,
        generators: tuple[str, ...],
    ) -> None:
        for event in self.events.values():
            if self.event_has_active_loss(event.event_id, qubit):
                continue
            for branch in event.branches:
                if measurement_index in branch.loss_measurements:
                    branch.continuation_pauli_insertions.add(
                        PauliInsertion(boundary, qubit, generators)
                    )

    def record_loss_measurement(self, qubit: int, measurement_index: int) -> None:
        for _, branch in self.pending.get(qubit, ()):
            branch.loss_measurements.add(measurement_index)

    def clear_loss(self, qubit: int) -> None:
        for _, branch in self.pending.pop(qubit, ()):
            branch.active = False

    def _add_active_branch(
        self,
        event: _PendingLossEvent,
        qubit: int,
        boundary: int,
        *,
        add_continuation_insertion: bool,
    ) -> None:
        """Add one active physical branch unless the event already occupies it."""

        if self.event_has_active_loss(event.event_id, qubit):
            return
        insertions = (
            {PauliInsertion(boundary=boundary, qubit=qubit)}
            if add_continuation_insertion
            else set()
        )
        branch = _PendingLossBranch(
            qubit=qubit,
            loss_boundary=boundary,
            continuation_pauli_insertions=insertions,
        )
        event.branches.append(branch)
        self.pending[qubit].append((event, branch))

    def propagate_loss(
        self,
        event_id: int,
        *,
        lost_qubit: int,
        new_qubit: int,
        boundary: int,
    ) -> None:
        if not self.event_has_active_loss(event_id, lost_qubit):
            raise ValueError(
                f"event {event_id} has no active loss branch on qubit {lost_qubit}"
            )
        self._add_active_branch(
            self.events[event_id],
            new_qubit,
            boundary,
            add_continuation_insertion=True,
        )

    def swap_losses(
        self, event_id: int, first_qubit: int, second_qubit: int, boundary: int
    ) -> None:
        event = self.events[event_id]
        first_lost = self.event_has_active_loss(event_id, first_qubit)
        second_lost = self.event_has_active_loss(event_id, second_qubit)
        if first_lost == second_lost:
            return
        source = first_qubit if first_lost else second_qubit
        destination = second_qubit if first_lost else first_qubit
        retained = []
        for candidate_event, branch in self.pending.get(source, ()):
            if candidate_event.event_id == event_id:
                branch.active = False
            else:
                retained.append((candidate_event, branch))
        if retained:
            self.pending[source] = retained
        else:
            self.pending.pop(source, None)
        self._add_active_branch(
            event, destination, boundary, add_continuation_insertion=False
        )

    def finish(self, measurement_count: int) -> LossEventGraph:
        return build_loss_event_graph(
            (event.finish() for event in self.events.values()),
            measurement_count=measurement_count,
        )


def _collect_loss_sources(
    body: Sequence[object],
) -> list[tuple[int, int, int, float, int]]:
    sources: list[tuple[int, int, int, float, int]] = []
    next_event_id = 0
    for body_index, statement in enumerate(body):
        if not isinstance(statement, Instruction):
            continue
        if statement.name.upper() != "LOSS_ERROR":
            continue
        if len(statement.arguments) != 1:
            raise ValueError(
                f"LOSS_ERROR at flattened body index {body_index} requires "
                "exactly one probability"
            )
        probability = float(statement.arguments[0])
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"LOSS_ERROR probability at flattened body index {body_index} "
                f"must be in [0, 1], got {probability}"
            )
        targets = [
            (target_index, target)
            for target_index, target in enumerate(statement.targets)
            if isinstance(target, QubitTarget)
        ]
        if len(targets) != len(statement.targets):
            raise ValueError(
                f"LOSS_ERROR at flattened body index {body_index} accepts "
                "only qubit targets"
            )
        if not targets:
            raise ValueError(
                f"LOSS_ERROR at flattened body index {body_index} requires at "
                "least one qubit target"
            )
        qubits = [target.index for _, target in targets]
        if len(set(qubits)) != len(qubits):
            raise ValueError(
                f"LOSS_ERROR at flattened body index {body_index} contains a "
                "duplicate qubit target"
            )
        for target_index, target in targets:
            if target.inverted:
                raise ValueError("LOSS_ERROR qubit targets cannot be inverted")
            if probability == 0.0:
                continue
            sources.append(
                (
                    next_event_id,
                    body_index,
                    target_index,
                    probability,
                    target.index,
                )
            )
            next_event_id += 1
    return sources


def _split_source_occurrences(statement: Instruction) -> list[Instruction]:
    gate = stim.gate_data(statement.name.upper())
    if gate.name == "MPAD":
        return [
            Instruction(
                name=statement.name,
                arguments=statement.arguments,
                targets=[target],
            )
            for target in statement.targets
        ]
    if gate.takes_pauli_targets:
        occurrences = []
        for group in split_mpp_targets(list(statement.targets)):
            targets = []
            for index, target in enumerate(group):
                if index:
                    targets.append(CombinerTarget())
                targets.append(target)
            occurrences.append(
                Instruction(
                    name=statement.name,
                    arguments=statement.arguments,
                    targets=targets,
                )
            )
        return occurrences

    targets = [
        target
        for target in statement.targets
        if isinstance(target, (QubitTarget, PauliTarget, MeasurementRecordTarget))
    ]
    if len(targets) != len(statement.targets):
        raise UnsupportedLossModelError(
            f"gate {statement.name} with non-qubit targets is not supported by "
            "loss-model analysis"
        )
    group_size = 2 if gate.is_two_qubit_gate else 1
    if len(targets) % group_size != 0:
        raise ValueError(
            f"{statement.name} requires target groups of size {group_size}"
        )
    occurrences = []
    for index in range(0, len(targets), group_size):
        group = targets[index : index + group_size]
        record_targets = [
            target for target in group if isinstance(target, MeasurementRecordTarget)
        ]
        if record_targets and not (
            group_size == 2
            and isinstance(group[0], MeasurementRecordTarget)
            and isinstance(group[1], QubitTarget)
        ):
            raise UnsupportedLossModelError(
                f"gate {statement.name} has an unsupported classical target group"
            )
        occurrences.append(
            Instruction(
            name=statement.name,
            arguments=statement.arguments,
                targets=group,
            )
        )
    return occurrences


def _loss_gate_from_source_occurrence(
    occurrence: Instruction,
    *,
    body_index: int,
    measurement_index: int | None,
    prior_measurement_count: int,
    boundary: int,
    boundary_after: int,
) -> LossGate:
    try:
        gate = stim.gate_data(occurrence.name.upper())
    except IndexError:
        raise UnsupportedLossModelError(
            f"unknown gate {occurrence.name} is not supported by Stim"
        ) from None
    control_measurement_indices = [
        prior_measurement_count - target.offset
        for target in occurrence.targets
        if isinstance(target, MeasurementRecordTarget)
    ]
    if len(control_measurement_indices) > 1:
        raise UnsupportedLossModelError(
            f"gate {occurrence.name} has multiple measurement-record controls"
        )
    if any(index < 0 for index in control_measurement_indices):
        raise ValueError(
            f"gate {occurrence.name} references a measurement before the gadget"
        )
    return LossGate(
        name=gate.name,
        source_name=occurrence.name.upper(),
        arguments=tuple(float(argument) for argument in occurrence.arguments),
        qubits=tuple(
            target.index
            for target in occurrence.targets
            if isinstance(target, (QubitTarget, PauliTarget))
        ),
        measurement_index=measurement_index,
        control_measurement_index=(
            control_measurement_indices[0] if control_measurement_indices else None
        ),
        body_index=body_index,
        boundary_before=boundary,
        boundary_after=boundary_after,
        produces_measurement=gate.produces_measurements,
        resets_qubits=gate.is_reset,
        is_source_gate=True,
    )


def _decompose_instruction_for_loss(
    statement: Instruction,
    *,
    body_index: int,
    measurement_indices: tuple[int, ...],
    measurement_index: int,
    boundary: int,
    span: int,
) -> list[LossGate]:
    decomposed = stim.Circuit(str(statement)).decomposed()
    gates: list[LossGate] = []
    measurement_cursor = 0
    instruction_offset = 0
    for instruction in decomposed:
        if not isinstance(instruction, stim.CircuitInstruction):
            raise UnsupportedLossModelError(
                f"decomposition of {statement.name} contains a repeat block"
            )
        name = instruction.name
        if name == "TICK":
            continue
        if name == "I":
            continue
        boundary_before = boundary + instruction_offset
        instruction_offset += 1
        if name == "MPAD":
            measurement_cursor += instruction_num_measurements(str(instruction))
            continue
        if name not in {"H", "S", "CX", "M", "R"}:
            raise UnsupportedLossModelError(
                f"Stim decomposition of {statement.name} produced unsupported "
                f"primitive {name}"
            )
        targets = instruction.targets_copy()
        group_size = 2 if name == "CX" else 1
        if len(targets) % group_size != 0:
            raise ValueError(
                f"decomposed {name} requires target groups of size {group_size}"
            )
        for index in range(0, len(targets), group_size):
            group = targets[index : index + group_size]
            control_measurement_index = None
            if group[0].is_measurement_record_target:
                if len(group) != 2 or not group[1].is_qubit_target:
                    raise UnsupportedLossModelError(
                        f"decomposition of {statement.name} contains an "
                        "unsupported classical target group"
                    )
                control_index = measurement_index + measurement_cursor + group[0].value
                if control_index < 0:
                    raise ValueError(
                        f"gate {statement.name} references a measurement before "
                        "the gadget"
                    )
                control_measurement_index = control_index
                qubits = (group[1].value,)
            else:
                if not all(target.is_qubit_target for target in group):
                    raise UnsupportedLossModelError(
                        f"decomposition of {statement.name} contains a non-qubit target"
                    )
                qubits = tuple(target.value for target in group)
            primitive_measurement_index = None
            if name == "M":
                if measurement_cursor >= len(measurement_indices):
                    raise ValueError(
                        f"decomposition of {statement.name} produced too many "
                        "measurements"
                    )
                primitive_measurement_index = measurement_indices[measurement_cursor]
                measurement_cursor += 1
            gates.append(
                LossGate(
                    name=name,
                    source_name=statement.name.upper(),
                    arguments=tuple(instruction.gate_args_copy()),
                    qubits=qubits,
                    measurement_index=primitive_measurement_index,
                    control_measurement_index=control_measurement_index,
                    body_index=body_index,
                    boundary_before=boundary_before,
                    boundary_after=boundary_before + 1,
                    produces_measurement=name == "M",
                    resets_qubits=name == "R",
                    is_source_gate=False,
                )
            )
    if instruction_offset != span:
        raise ValueError(
            f"decomposition of {statement.name} occupied {instruction_offset} "
            f"instructions, expected timeline span {span}"
        )
    if measurement_cursor != len(measurement_indices):
        raise ValueError(
            f"decomposition of {statement.name} produced {measurement_cursor} "
            f"measurements, expected {len(measurement_indices)}"
        )
    return gates


def _loss_gates_for_instruction(
    statement: Instruction,
    *,
    body_index: int,
    measurement_index: int,
    boundary: int,
    span: int,
    native_gates: frozenset[str],
) -> tuple[list[LossGate], int]:
    source_name = statement.name.upper()
    try:
        source_gate = stim.gate_data(source_name)
    except IndexError:
        raise UnsupportedLossModelError(
            f"unknown gate {statement.name} is not supported by Stim"
        ) from None
    statement_measurement_count = instruction_num_measurements(str(statement))
    if source_gate.name == "MPAD":
        return [], measurement_index + statement_measurement_count

    if source_gate.name not in native_gates:
        measurement_indices = tuple(
            range(
                measurement_index,
                measurement_index + statement_measurement_count,
            )
        )
        return (
            _decompose_instruction_for_loss(
                statement,
                body_index=body_index,
                measurement_indices=measurement_indices,
                measurement_index=measurement_index,
                boundary=boundary,
                span=span,
            ),
            measurement_index + statement_measurement_count,
        )

    gates: list[LossGate] = []
    for occurrence in _split_source_occurrences(statement):
        occurrence_measurement_count = instruction_num_measurements(str(occurrence))
        if occurrence_measurement_count > 1:
            raise ValueError(
                f"atomized {occurrence.name} produced "
                f"{occurrence_measurement_count} measurements"
            )
        prior_measurement_count = measurement_index
        occurrence_measurement_index = (
            measurement_index if occurrence_measurement_count else None
        )
        measurement_index += occurrence_measurement_count
        gates.append(
            _loss_gate_from_source_occurrence(
                occurrence,
                body_index=body_index,
                measurement_index=occurrence_measurement_index,
                prior_measurement_count=prior_measurement_count,
                boundary=boundary,
                boundary_after=boundary + span,
            )
        )
    return gates, measurement_index


def analyze_loss_events(
    gadget: GadgetDefinition,
    model: LossModel,
) -> LossAnalysisResult:
    """Traverse one gadget, returning its graph and physical boundary mappings.

    Each declared input qubit receives a synthetic source at boundary ``0``,
    modelling a loss that entered from an upstream gadget. Synthetic event IDs
    follow the gadget's own ``LOSS_ERROR`` event IDs.

    ``exit_qubits_by_event`` maps each event ID to the physical qubits that still
    carry an active loss branch once the body finishes -- the qubits on which the
    loss leaves the gadget. ``input_event_id_by_qubit`` maps each seeded input
    physical qubit to its event ID.
    """

    body = flatten_body(list(gadget.body))
    decomposed_body = build_decomposed_body(body)
    loss_sources = _collect_loss_sources(body)
    if not isinstance(model, LossModel):
        raise TypeError(
            f"{type(model).__name__} does not implement the LossModel protocol"
        )
    native_gates = frozenset(name.upper() for name in model.native_gates)
    total_measurements = sum(
        instruction_num_measurements(str(statement))
        for statement in body
        if isinstance(statement, Instruction)
    )

    input_qubits = [
        qubit for port in gadget.input_ports for qubit in port.qubit_indices
    ]
    input_event_id_by_qubit = {
        qubit: len(loss_sources) + offset
        for offset, qubit in enumerate(input_qubits)
    }

    if not loss_sources and not input_event_id_by_qubit:
        return LossAnalysisResult(
            graph=build_loss_event_graph((), measurement_count=total_measurements),
            exit_qubits_by_event={},
            input_event_id_by_qubit={},
        )

    sources_by_body_index: dict[int, list[tuple[int, int, int, float, int]]] = (
        defaultdict(list)
    )
    for source in loss_sources:
        sources_by_body_index[source[1]].append(source)

    state = _MutableLossAnalysisState()
    for qubit, event_id in input_event_id_by_qubit.items():
        state.add_source_event(
            event_id=event_id,
            body_index=0,
            target_index=0,
            qubit=qubit,
            probability=1.0,
            boundary=0,
        )
        model.handle_loss_source(event_id, state)
    measurement_index = 0
    for body_index, statement in enumerate(body):
        boundary = decomposed_body.body_start_at[body_index]
        for event_id, _, target_index, probability, qubit in sources_by_body_index.get(
            body_index, ()
        ):
            state.add_source_event(
                event_id=event_id,
                body_index=body_index,
                target_index=target_index,
                qubit=qubit,
                probability=probability,
                boundary=boundary,
            )
            model.handle_loss_source(event_id, state)

        if not isinstance(statement, Instruction):
            continue
        source_name = statement.name.upper()
        if (
            source_name in NOISE_INSTRUCTIONS_ALL
            or source_name in ANNOTATION_INSTRUCTIONS
        ):
            continue
        boundary_after = (
            decomposed_body.body_start_at[body_index + 1]
            if body_index + 1 < len(decomposed_body.body_start_at)
            else len(decomposed_body.instructions)
        )
        gates, measurement_index = _loss_gates_for_instruction(
            statement,
            body_index=body_index,
            measurement_index=measurement_index,
            boundary=boundary,
            span=boundary_after - boundary,
            native_gates=native_gates,
        )
        for gate in gates:
            model.handle_gate(gate, state)

    assert measurement_index == total_measurements
    # Qubits still carrying an active, *unheralded* loss branch when the body
    # ends are the qubits on which the loss leaves the gadget. A branch that was
    # measured (has loss_measurements) is resolved in-gadget and does not exit.
    exit_qubits_by_event: dict[int, set[int]] = defaultdict(set)
    for qubit, entries in state.pending.items():
        for event, branch in entries:
            if branch.active and not branch.loss_measurements:
                exit_qubits_by_event[event.event_id].add(qubit)
    exits = {
        event_id: tuple(sorted(qubits))
        for event_id, qubits in exit_qubits_by_event.items()
    }
    return LossAnalysisResult(
        graph=state.finish(measurement_count=measurement_index),
        exit_qubits_by_event=exits,
        input_event_id_by_qubit=input_event_id_by_qubit,
    )
