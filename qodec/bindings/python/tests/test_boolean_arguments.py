from __future__ import annotations

import json
from pathlib import Path

import pytest

import qodec
from qodec.gadgets import Circuit
from qodec.instructions import InstructionCall


@pytest.mark.parametrize("value", [True, False, 1, 0])
def test_instruction_call_preserves_argument_type(value: bool | int) -> None:
    call = InstructionCall("probe", operands=[0], arguments={"select": value})
    assert call.arguments["select"] == value
    assert type(call.arguments["select"]) is type(value)
    assert type(call.operands[0]) is int


def test_instruction_call_does_not_coerce_stored_boolean_lists() -> None:
    values: list[int] = [True, False]
    call = InstructionCall("probe", arguments={"values": values})
    stored = call.arguments["values"]
    assert stored == values
    from collections.abc import MutableSequence

    assert isinstance(stored, MutableSequence)
    assert [type(value) for value in stored] == [bool, bool]


@pytest.fixture
def boolean_isa() -> qodec.InstructionSet:
    fixture = Path(__file__).resolve().parents[2] / "c/tests/fixtures/argument-shapes/argument-shapes.qodec.yaml"
    return qodec.Qodec.load(fixture).layers[1].instruction_set


@pytest.mark.parametrize("value", [True, False])
def test_instruction_call_rejects_boolean_selection_bits(value: bool) -> None:
    with pytest.raises(TypeError, match="select.*integer 0 or 1, not bool"):
        InstructionCall("select", select=[{"select": value}])


@pytest.mark.parametrize("value", [True, False])
def test_callback_rejects_boolean_selection_bits(boolean_isa: qodec.InstructionSet, value: bool) -> None:
    circuit = Circuit(boolean_isa, "source", format="selection-probe")
    with pytest.raises(TypeError, match="select.*integer 0 or 1, not bool"):
        circuit.calls(parser=lambda source, target: [InstructionCall("select", select=[{"select": value}])])


@pytest.mark.parametrize("value", [0, 1])
def test_integer_selection_bits_agree_across_parsers(boolean_isa: qodec.InstructionSet, value: int) -> None:
    expected = [{"select": value}]
    call = InstructionCall("select", select=expected)
    source = json.dumps([{"select": {"select": expected}}])
    circuit = Circuit(boolean_isa, source, format="yaml")
    assert call.select == circuit.calls()[0].select == circuit.calls(parser=lambda source, target: [call])[0].select
    assert type(call.select[0]["select"]) is int


@pytest.mark.parametrize(("value", "error"), [(2, ValueError), (-1, OverflowError), (256, OverflowError)])
def test_out_of_range_selection_bits_are_rejected(value: int, error: type[Exception]) -> None:
    with pytest.raises(error):
        InstructionCall("select", select=[{"select": value}])


@pytest.mark.parametrize("value", [True, False, 1, 0, "true", "false"])
@pytest.mark.parametrize("shorthand", [False, True])
def test_circuit_preserves_literal_type(
    boolean_isa: qodec.InstructionSet, value: bool | int | str, shorthand: bool
) -> None:
    call: object = [0, {"select": value}] if shorthand else {"operands": [0], "arguments": {"select": value}}
    circuit = Circuit(boolean_isa, json.dumps([{"boolean": call}]), format="yaml")
    parsed = circuit.calls()[0]
    assert parsed.arguments == {"select": value}
    assert type(parsed.arguments["select"]) is type(value)
    assert parsed.operands == [0]
    assert type(parsed.operands[0]) is int
    assert parsed.select == []


@pytest.mark.parametrize("literal", ["true", "false"])
@pytest.mark.parametrize(
    "source, error",
    [
        ("- boolean: {{operands: [{literal}]}}", "an operand must be an integer index or a block name"),
        ("- boolean: [{literal}]", "an operand must be an integer index or a block name"),
        ("- boolean: {{arguments: {{select: [{literal}]}}}}", "unsupported value shape"),
        ("- boolean: [0, select: [{literal}]]", "unsupported value shape"),
        ("- boolean: {{select: [{{select: {literal}}}]}}", "bit must be a non-negative integer"),
    ],
)
def test_circuit_rejects_booleans_outside_scalar_arguments(
    boolean_isa: qodec.InstructionSet, literal: str, source: str, error: str
) -> None:
    circuit = Circuit(boolean_isa, source.format(literal=literal), format="yaml")
    with pytest.raises(ValueError, match=error):
        _ = circuit.calls()