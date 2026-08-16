"""Loss event graph types and validation.

The immutable result of loss analysis has this ownership hierarchy::

    LossEventGraph
     └─ LossEvent                 one possible source loss
         ├─ PauliInsertion       generators at the source boundary
         └─ LossBranch           one physical qubit carrying that loss
             ├─ PauliInsertion  generators accumulated along the branch
             └─ successor_event_id

A loss propagation creates another ``LossBranch`` within the same event.
A successor link instead connects alternative source events on one continuing
loss lifetime, allowing them to share their common suffix.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class PauliInsertion:
    """One single-qubit Pauli generator basis at a circuit boundary.

    The default ``("X", "Z")`` spans the full single-qubit Pauli group. Any
    two distinct generators span that same group and canonicalize to this basis.
    """

    boundary: int
    qubit: int
    generators: tuple[str, ...] = ("X", "Z")

    def __post_init__(self) -> None:
        if self.boundary < 0:
            raise ValueError("Pauli insertion boundary must be non-negative")
        if self.qubit < 0:
            raise ValueError("Pauli insertion qubit must be non-negative")
        if any(generator not in {"X", "Y", "Z"} for generator in self.generators):
            raise ValueError("Pauli insertion generators support only X, Y, and Z")
        generators = tuple(sorted(set(self.generators), key="XYZ".index))
        if not generators:
            raise ValueError("Pauli insertion requires at least one generator")
        if len(generators) > 1:
            generators = ("X", "Z")
        object.__setattr__(self, "generators", generators)


@dataclass(frozen=True)
class LossBranch:
    """One physical loss branch caused directly or through propagation.

    Boundaries are positions in the loss model's ideal-operation stream. A
    branch may produce zero or multiple loss measurements.
    """

    qubit: int
    loss_boundary: int
    loss_measurements: tuple[int, ...]
    continuation_pauli_insertions: tuple[PauliInsertion, ...]
    successor_event_id: int | None = None

    def __post_init__(self) -> None:
        if self.qubit < 0 or self.loss_boundary < 0:
            raise ValueError("loss branch source must be non-negative")
        measurements = tuple(sorted(set(self.loss_measurements)))
        insertions = tuple(sorted(set(self.continuation_pauli_insertions)))
        if any(index < 0 for index in measurements):
            raise ValueError("loss measurement indices must be non-negative")
        if self.successor_event_id is not None and self.successor_event_id < 0:
            raise ValueError("successor event ID must be non-negative")
        object.__setattr__(self, "loss_measurements", measurements)
        object.__setattr__(self, "continuation_pauli_insertions", insertions)


@dataclass(frozen=True)
class LossEvent:
    """One source loss event with one or more propagated loss branches."""

    event_id: int
    body_index: int
    target_index: int
    source_qubit: int
    loss_probability: float
    source_boundary: int
    branches: tuple[LossBranch, ...]
    source_pauli_insertions: tuple[PauliInsertion, ...] = ()

    def __post_init__(self) -> None:
        if self.event_id < 0:
            raise ValueError("loss event ID must be non-negative")
        if self.body_index < 0 or self.target_index < 0:
            raise ValueError("loss event source indices must be non-negative")
        if self.source_qubit < 0 or self.source_boundary < 0:
            raise ValueError("loss event source must be non-negative")
        if not 0.0 < self.loss_probability <= 1.0:
            raise ValueError("loss event probability must be in (0, 1]")
        branches = tuple(
            sorted(
                set(self.branches),
                key=lambda branch: (
                    branch.loss_boundary,
                    branch.qubit,
                    branch.loss_measurements,
                    branch.continuation_pauli_insertions,
                    branch.successor_event_id,
                ),
            )
        )
        if not branches:
            raise ValueError("loss event must contain at least one branch")
        if not any(
            branch.qubit == self.source_qubit
            and branch.loss_boundary == self.source_boundary
            for branch in branches
        ):
            raise ValueError("loss event must contain its source branch")
        object.__setattr__(self, "branches", branches)
        object.__setattr__(
            self,
            "source_pauli_insertions",
            tuple(sorted(set(self.source_pauli_insertions))),
        )

    @property
    def affected_qubits(self) -> tuple[int, ...]:
        """Return all qubits lost directly or through propagation."""

        return tuple(sorted({branch.qubit for branch in self.branches}))

    @property
    def loss_measurements(self) -> tuple[int, ...]:
        """Return local loss measurements before following successor links."""

        return tuple(
            sorted(
                {
                    measurement_index
                    for branch in self.branches
                    for measurement_index in branch.loss_measurements
                }
            )
        )

    @property
    def continuation_pauli_insertions(self) -> tuple[PauliInsertion, ...]:
        """Return inheritable effects after this event's source location."""

        return tuple(
            sorted(
                {
                    insertion
                    for branch in self.branches
                    for insertion in branch.continuation_pauli_insertions
                }
            )
        )

    @property
    def local_pauli_insertions(self) -> tuple[PauliInsertion, ...]:
        """Return this event's source-only and local continuation effects."""

        insertions = set(self.source_pauli_insertions)
        insertions.update(self.continuation_pauli_insertions)
        return tuple(sorted(insertions))


@dataclass(frozen=True)
class LossEventGraph:
    """Physical loss events and their definite forward-suffix links."""

    measurement_count: int
    events: tuple[LossEvent, ...]
    successor_event_ids: tuple[tuple[int, ...], ...]


def build_loss_event_graph(
    events: Iterable[LossEvent], measurement_count: int
) -> LossEventGraph:
    """Validate and canonicalize the physical loss-event DAG."""

    ordered_events = tuple(sorted(events, key=lambda event: event.event_id))
    event_ids = [event.event_id for event in ordered_events]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("loss event IDs must be unique")
    event_index = {event_id: index for index, event_id in enumerate(event_ids)}
    successor_event_ids: list[set[int]] = [set() for _ in ordered_events]
    predecessor_event_ids: list[set[int]] = [set() for _ in ordered_events]
    for event_index_value, event in enumerate(ordered_events):
        for measurement_index in event.loss_measurements:
            if measurement_index >= measurement_count:
                raise ValueError(
                    f"loss event {event.event_id} references measurement "
                    f"{measurement_index}, outside measurement_count={measurement_count}"
                )
        for branch in event.branches:
            successor_id = branch.successor_event_id
            if successor_id is None:
                continue
            if successor_id not in event_index:
                raise ValueError(
                    f"loss event {event.event_id} references unknown successor "
                    f"{successor_id}"
                )
            if successor_id == event.event_id:
                raise ValueError("loss event cannot imply itself")
            successor_event_ids[event_index_value].add(successor_id)
            predecessor_event_ids[event_index[successor_id]].add(event.event_id)

    remaining_predecessors = [
        len(predecessors) for predecessors in predecessor_event_ids
    ]
    pending = [
        event_id
        for event_id in event_ids
        if remaining_predecessors[event_index[event_id]] == 0
    ]
    topological_order: list[int] = []
    while pending:
        event_id = pending.pop()
        topological_order.append(event_id)
        for successor_id in successor_event_ids[event_index[event_id]]:
            successor_index = event_index[successor_id]
            remaining_predecessors[successor_index] -= 1
            if remaining_predecessors[successor_index] == 0:
                pending.append(successor_id)
    if len(topological_order) != len(event_ids):
        raise ValueError("loss-event implication graph contains a cycle")

    return LossEventGraph(
        measurement_count=measurement_count,
        events=ordered_events,
        successor_event_ids=tuple(
            tuple(sorted(successors)) for successors in successor_event_ids
        ),
    )
