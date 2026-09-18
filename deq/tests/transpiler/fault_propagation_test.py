"""Tests for the shared circuit-fault propagation timeline."""

from typing import cast

import pytest

from deq.circuit.model import GadgetDefinition, GadgetStatement
from deq.circuit.parser import parse
from deq.transpiler.fault_propagation import (
    ErrorProjectionContext,
    build_decomposed_body,
)
from deq.transpiler.jit_transpiler import flatten_body


def _raw_body(source: str):
    qfile = parse(source)
    gadget = next(
        definition
        for definition in qfile.definitions
        if isinstance(definition, GadgetDefinition)
    )
    return list(gadget.body)


def _body(source: str):
    return flatten_body(_raw_body(source))


def test_loss_error_uses_the_following_gate_boundary() -> None:
    timeline = build_decomposed_body(
        _body("GADGET G { H 0 LOSS_ERROR(0.1) 0 MX 0 }")
    )

    assert timeline.body_start_at == (0, 1, 1)
    assert [instruction.name for instruction in timeline.instructions] == [
        "H",
        "H",
        "M",
        "H",
    ]
    assert timeline.measurement_start_at == (0, 0, 0, 1)
    assert timeline.total_measurements == 1


def test_adjacent_source_gates_keep_distinct_boundaries() -> None:
    timeline = build_decomposed_body(_body("GADGET G { H 0 H 1 }"))

    assert timeline.body_start_at == (0, 1)
    assert [str(instruction) for instruction in timeline.instructions] == [
        "H 0",
        "H 1",
    ]


def test_user_tick_does_not_participate_in_boundary_mapping() -> None:
    timeline = build_decomposed_body(_body("GADGET G { H 0 TICK MX 1 }"))

    assert timeline.body_start_at == (0, 1, 1)
    assert [str(instruction) for instruction in timeline.instructions] == [
        "H 0",
        "H 1",
        "M 1",
        "H 1",
    ]


def test_unflattened_repeat_block_is_rejected() -> None:
    with pytest.raises(ValueError, match="call flatten_body"):
        build_decomposed_body(
            _raw_body("GADGET G { REPEAT 2 { M 0 } }")
        )


def test_unknown_body_block_is_rejected() -> None:
    class UnknownBlock:
        body: list[object] = []

    with pytest.raises(TypeError, match="UnknownBlock"):
        build_decomposed_body(
            [cast(GadgetStatement, UnknownBlock())]
        )


def test_measurement_count_uses_stim_instruction_metadata() -> None:
    timeline = build_decomposed_body(
        _body("GADGET G { MPAD 0 H 0 }")
    )

    assert timeline.measurement_start_at == (0, 1)
    assert timeline.total_measurements == 1


def test_output_stabilizer_measurement_index_uses_offset() -> None:
    context = ErrorProjectionContext(
        input_virtual_count=0,
        finished_member_lists=(),
        unfinished_member_lists=(),
        output_stabilizer_measurement_offset=7,
        readout_measurement_sets=(),
        logical_columns=set(),
        unfinished_to_column=(),
        physical_correction_by_logical={},
    )

    assert context.output_stabilizer_measurement_index(3) == 10