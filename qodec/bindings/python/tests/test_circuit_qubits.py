"""``Circuit.blocks``: the block-instance labels a circuit's calls name.

This is the label space an encoding's ``support`` is written in, not a
physical layout — qodec assigns block instances no addresses.
"""

from __future__ import annotations

import pathlib

import pytest

import qodec

EXAMPLES = pathlib.Path(__file__).resolve().parents[3] / "examples"


def _gadgets(protocol: qodec.Qodec) -> list[qodec.Gadget]:
    return [gadget for layer in protocol.layers for gadget in layer.gadgets.values()]


def _labels_from_operands(circuit: qodec.gadgets.Circuit) -> list[str]:
    """First-appearance order over every block operand, done the long way."""
    seen: list[str] = []
    for call in circuit.calls():
        for operand in call.operands:
            label = str(operand)
            if label not in seen:
                seen.append(label)
    return seen


@pytest.fixture(scope="module")
def c4c6() -> qodec.Qodec:
    return qodec.Qodec.load(EXAMPLES / "c4c6" / "qodec.yaml")


@pytest.mark.parametrize(
    "example", ["repetition3/repetition3.qodec.yaml", "c4c6/qodec.yaml", "c422-c832-arch/qodec.yaml"]
)
def test_blocks_match_first_seen_operand_labels(example: str) -> None:
    protocol = qodec.Qodec.load(EXAMPLES / example)
    for gadget in _gadgets(protocol):
        assert gadget.circuit.blocks == _labels_from_operands(gadget.circuit)


def test_order_is_first_appearance_not_sorted(c4c6: qodec.Qodec) -> None:
    """``idle`` touches its ancilla block before its data block."""
    blocks = c4c6.layers[0].gadgets["idle"].circuit.blocks
    assert blocks == ["3", "4", "5", "6", "7", "8", "0", "1", "2"]


def test_an_empty_circuit_names_no_blocks(c4c6: qodec.Qodec) -> None:
    circuit = qodec.gadgets.Circuit(c4c6.layers[1].instruction_set, "[]", format="yaml")
    assert circuit.blocks == []


def test_a_circuit_need_not_touch_every_encoded_qubit() -> None:
    """Support is not a subset of circuit blocks: a circuit may leave qubits idle."""
    protocol = qodec.Qodec.load(EXAMPLES / "repetition3" / "repetition3.qodec.yaml")
    gadget = protocol.layers[0].gadgets["rotate_z"]
    support = {label for encoding in gadget.inputs for label in encoding.support}
    assert gadget.circuit.blocks == ["0"]
    assert not support <= set(gadget.circuit.blocks)


def test_multiqubit_blocks_are_not_expanded() -> None:
    instruction_set = qodec.InstructionSet(
        "test",
        blocks=[qodec.instructions.Block("pair", encodes=2)],
        instructions=[qodec.Instruction("idle", inputs=[qodec.instructions.BlockOperand("pair")])],
    )
    circuit = qodec.gadgets.Circuit(
        instruction_set,
        "- idle: [named]\n- idle: [7]\n- idle: [named]\n- idle: [3]",
        format="yaml",
    )
    assert circuit.blocks == ["named", "7", "3"]
