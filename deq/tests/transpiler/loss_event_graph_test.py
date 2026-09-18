"""Validation tests for the physical loss-event compiler IR."""

import pytest

from deq.transpiler.loss.loss_graph import (
    LossBranch,
    LossEvent,
    PauliInsertion,
    build_loss_event_graph,
)


def _event(
    event_id: int,
    *,
    successor: int | None = None,
    measurements: tuple[int, ...] = (),
) -> LossEvent:
    return LossEvent(
        event_id=event_id,
        body_index=event_id,
        target_index=0,
        source_qubit=0,
        loss_probability=0.1,
        source_boundary=event_id,
        branches=(
            LossBranch(
                qubit=0,
                loss_boundary=event_id,
                loss_measurements=measurements,
                continuation_pauli_insertions=(),
                successor_event_id=successor,
            ),
        ),
    )


def test_graph_canonicalizes_event_and_successor_order() -> None:
    graph = build_loss_event_graph(
        (_event(1), _event(0, successor=1)), measurement_count=0
    )

    assert [event.event_id for event in graph.events] == [0, 1]
    assert graph.successor_event_ids == ((1,), ())


def test_graph_rejects_cycles() -> None:
    with pytest.raises(ValueError, match="contains a cycle"):
        build_loss_event_graph(
            (_event(0, successor=1), _event(1, successor=0)),
            measurement_count=0,
        )


def test_graph_rejects_unknown_successor() -> None:
    with pytest.raises(ValueError, match="unknown successor"):
        build_loss_event_graph((_event(0, successor=1),), measurement_count=0)


def test_graph_rejects_out_of_range_measurement() -> None:
    with pytest.raises(ValueError, match="outside measurement_count"):
        build_loss_event_graph((_event(0, measurements=(1,)),), measurement_count=1)


def test_graph_rejects_duplicate_event_ids() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        build_loss_event_graph((_event(0), _event(0)), measurement_count=0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"boundary": -1, "qubit": 0}, "boundary must be non-negative"),
        ({"boundary": 0, "qubit": -1}, "qubit must be non-negative"),
        (
            {"boundary": 0, "qubit": 0, "generators": ("I",)},
            "generators support only X, Y, and Z",
        ),
        (
            {"boundary": 0, "qubit": 0, "generators": ()},
            "requires at least one generator",
        ),
    ],
)
def test_pauli_insertion_rejects_invalid_values(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PauliInsertion(**kwargs)


def test_pauli_insertion_canonicalizes_generators() -> None:
    insertion = PauliInsertion(0, 0, ("Y", "X", "Y"))

    assert insertion.generators == ("X", "Z")


@pytest.mark.parametrize(
    "qubit, boundary",
    [
        (-1, 0),
        (0, -1),
    ],
)
def test_loss_branch_rejects_negative_source(qubit: int, boundary: int) -> None:
    with pytest.raises(ValueError, match="source must be non-negative"):
        LossBranch(qubit, boundary, (), ())


def test_loss_branch_rejects_negative_measurement() -> None:
    with pytest.raises(ValueError, match="measurement indices must be non-negative"):
        LossBranch(0, 0, (-1,), ())


def test_loss_branch_rejects_negative_successor() -> None:
    with pytest.raises(ValueError, match="successor event ID must be non-negative"):
        LossBranch(0, 0, (), (), successor_event_id=-1)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"event_id": -1}, "event ID must be non-negative"),
        ({"body_index": -1}, "source indices must be non-negative"),
        ({"target_index": -1}, "source indices must be non-negative"),
        ({"source_qubit": -1}, "source must be non-negative"),
        ({"source_boundary": -1}, "source must be non-negative"),
        ({"loss_probability": 0.0}, r"probability must be in \(0, 1\]"),
        ({"loss_probability": 1.1}, r"probability must be in \(0, 1\]"),
        ({"branches": ()}, "must contain at least one branch"),
        (
            {"branches": (LossBranch(1, 0, (), ()),)},
            "must contain its source branch",
        ),
    ],
)
def test_loss_event_rejects_invalid_values(
    overrides: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "event_id": 0,
        "body_index": 0,
        "target_index": 0,
        "source_qubit": 0,
        "loss_probability": 0.1,
        "source_boundary": 0,
        "branches": (LossBranch(0, 0, (), ()),),
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        LossEvent(**values)


def test_loss_event_canonicalizes_and_aggregates_branch_data() -> None:
    first_insertion = PauliInsertion(2, 0, ("X",))
    second_insertion = PauliInsertion(3, 1)
    source_branch = LossBranch(
        qubit=0,
        loss_boundary=1,
        loss_measurements=(0, 2, 2),
        continuation_pauli_insertions=(first_insertion, first_insertion),
    )
    child_branch = LossBranch(
        qubit=1,
        loss_boundary=2,
        loss_measurements=(1,),
        continuation_pauli_insertions=(second_insertion,),
    )
    source_insertion = PauliInsertion(1, 0)

    event = LossEvent(
        event_id=0,
        body_index=0,
        target_index=0,
        source_qubit=0,
        loss_probability=0.1,
        source_boundary=1,
        branches=(child_branch, source_branch, source_branch),
        source_pauli_insertions=(source_insertion, source_insertion),
    )

    assert event.branches == (source_branch, child_branch)
    assert event.affected_qubits == (0, 1)
    assert event.loss_measurements == (0, 1, 2)
    assert event.continuation_pauli_insertions == (
        first_insertion,
        second_insertion,
    )
    assert event.local_pauli_insertions == (
        source_insertion,
        first_insertion,
        second_insertion,
    )


def test_graph_rejects_self_successor() -> None:
    with pytest.raises(ValueError, match="cannot imply itself"):
        build_loss_event_graph((_event(0, successor=0),), measurement_count=0)
