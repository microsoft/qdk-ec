"""Unit tests for the qodec Python bindings' authoring surface.

These exercise the small, self-contained pieces of the binding API —
actions and the flat Pauli builders — without needing an on-disk
codec. Codec load/save behavior is
covered separately by the Rust integration tests and the example tests.

Run with::

    pytest bindings/python/tests
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import qodec
from qodec.actions import (
    Clifford,
    Condition,
    Observe,
    Pauli,
    Rotate,
    Stabilize,
)
from qodec.codes import pauli
from qodec.instructions import Block, BlockOperand, Instruction, InstructionSet


def test_stabilize_roundtrips_operators_and_condition() -> None:
    condition = Condition(["reject"], invert=True)
    stabilize = Stabilize(["Z_0", "Z_1"], condition=condition)
    assert stabilize.operators == ("Z_0", "Z_1")
    assert stabilize.condition is not None
    assert stabilize.condition.predicates == ("reject",)
    assert stabilize.condition.invert is True


def test_stabilize_condition_defaults_to_none() -> None:
    assert Stabilize(["X_0"]).condition is None


def test_stabilize_introduces_temporary_qubit_in_instruction_set() -> None:
    instruction = Instruction(
        "channel",
        inputs=[BlockOperand("qubit")],
        outputs=[BlockOperand("qubit")],
        action=[
            Stabilize(["Z_1"]),
            Rotate("X_0 Y_1", 0.3),
            Clifford({"X_0": "X_0 X_1", "Z_1": "Z_0 Z_1"}),
            Observe(["Z_1"]),
            Pauli("X_1", condition=Condition(["outcomes[0]"])),
        ],
    )
    instruction_set = InstructionSet(
        "Channels",
        blocks=[Block("qubit", encodes=1)],
        instructions=[instruction],
    )
    assert instruction_set.instructions["channel"] == instruction


@pytest.mark.parametrize(
    "action",
    [Pauli("X_1"), Rotate("Y_1", 0.3), Observe(["Z_1"]), Clifford({"X_1": "Z_1", "Z_1": "X_1"})],
)
def test_unprepared_temporary_is_preserved_for_audit(action: qodec.Action) -> None:
    instruction = Instruction(
        "channel",
        inputs=[BlockOperand("qubit")],
        outputs=[BlockOperand("qubit")],
        action=[action, Stabilize(["Z_1"])],
    )
    instruction_set = InstructionSet("Channels", blocks=[Block("qubit", encodes=1)], instructions=[instruction])
    assert instruction_set.instructions["channel"].action == instruction.action


def _write_joint_measurement_instruction(tmp_path: Path) -> Path:
    source = tmp_path / "joint-measurements.isa.yaml"
    source.write_text(
        "name: JointMeasurements\n"
        "blocks: {qubit: 1}\n"
        "instructions:\n"
        "  - mnemonic: measure_both_one\n"
        "    description: Measure whether both qubits are one, without measuring each separately.\n"
        "    in: [qubit, qubit]\n"
        "    out: [qubit, qubit]\n"
        "    action:\n"
        "      - stabilize: Z_2\n"
        "      - rotate: {pauli: Y_2, angle: 0.7853981633974483}\n"
        "      - rotate: {pauli: Z_0 Y_2, angle: -0.7853981633974483}\n"
        "      - rotate: {pauli: Z_1 Y_2, angle: -0.7853981633974483}\n"
        "      - rotate: {pauli: Z_0 Z_1 Y_2, angle: 0.7853981633974483}\n"
        "      - observe: Z_2\n"
    )
    return source


def test_temporary_qubits_load_from_yaml(tmp_path: Path) -> None:
    source = _write_joint_measurement_instruction(tmp_path)
    instruction = InstructionSet.load(source).instructions["measure_both_one"]
    assert instruction.inputs == instruction.outputs == [BlockOperand("qubit"), BlockOperand("qubit")]
    assert instruction.action == [
        Stabilize(["Z_2"]),
        Rotate("Y_2", 0.7853981633974483),
        Rotate("Z_0 Y_2", -0.7853981633974483),
        Rotate("Z_1 Y_2", -0.7853981633974483),
        Rotate("Z_0 Z_1 Y_2", 0.7853981633974483),
        Observe(["Z_2"]),
    ]
    assert sum(len(action.observables) for action in instruction.action if isinstance(action, Observe)) == 1


def test_clifford_accepts_dict_and_preserves_map() -> None:
    clifford = Clifford({"X": "Z", "Z": "X"})
    assert clifford.generators == {"X": "Z", "Z": "X"}


def test_clifford_rejects_non_dict() -> None:
    not_a_dict: Any = ["X", "Z"]
    with pytest.raises(ValueError, match=r"generators must be a dict"):
        Clifford(not_a_dict)


def test_pauli_atom() -> None:
    pauli_atom = Pauli("Y_0")
    assert pauli_atom.operator == "Y_0"
    assert pauli_atom.condition is None


def test_observe_accepts_pauli_strings_and_expressions() -> None:
    observe = Observe(["Z_0", "X_0", pauli("Y_0")])
    assert observe.observables == ("Z_0", "X_0", "Y_0")


def test_observe_rejects_bad_item() -> None:
    bad_item: Any = 123
    with pytest.raises(TypeError):
        Observe([bad_item])


def test_observe_takes_no_condition() -> None:
    """A gated observe would make the measurement record runtime-sized."""
    assert not hasattr(Observe(["Z_0"]), "condition")
    with pytest.raises(TypeError):
        Observe(["Z_0"], condition=Condition(["reject"]))  # type: ignore[call-arg]


def test_a_conditional_observe_is_preserved_but_cannot_be_projected(tmp_path: Path) -> None:
    source = tmp_path / "t.isa.yaml"
    source.write_text(
        "name: T\n"
        "blocks: {q: 1}\n"
        "instructions:\n"
        "  - mnemonic: mz\n"
        "    description: a\n"
        "    in: [q]\n"
        "    out: [q]\n"
        "    flags: [reject]\n"
        '    action: [{observe: "Z_0", unless: [reject]}]\n'
    )
    instruction_set = InstructionSet.load(source)
    destination = tmp_path / "saved.isa.yaml"
    instruction_set.save(destination)
    assert "unless:" in destination.read_text()
    with pytest.raises(ValueError, match="conditional observe"):
        list(instruction_set.instructions["mz"].action)


def test_rotate_keeps_angle() -> None:
    rotate = Rotate("Z_0", 1.5)
    assert rotate.pauli == "Z_0"
    assert rotate.angle == pytest.approx(1.5)


@pytest.mark.parametrize("angle", ["theta", "1.5"])
def test_rotate_accepts_symbolic_angle(angle: str) -> None:
    # A str angle is a symbolic parameter name (Scalar::Parameter), returned
    # verbatim; a numeric angle stays a literal float (Scalar::Literal).
    symbolic = Rotate("Z_0", angle)
    assert symbolic.pauli == "Z_0"
    assert symbolic.angle == angle
    assert isinstance(symbolic.angle, str)

    literal = Rotate("Z_0", 1.5)
    assert isinstance(literal.angle, float)

    # Literal and operand forms are distinct even when string-equal.
    assert symbolic == Rotate("Z_0", angle)
    assert literal != symbolic


def test_condition_accepts_str_predicates() -> None:
    condition = Condition(["reject", "accept"])
    assert condition.predicates == ("reject", "accept")
    assert condition.invert is False


def test_flat_pauli_factory() -> None:
    assert pauli("X_0") == "X_0"
    assert pauli("Z_0", "Z_1") == "Z_0 Z_1"
    # PauliExpression supports tensor-product composition via ``*``.
    assert pauli("X_0") * pauli("Z_1") == "X_0 Z_1"
    assert pauli("X_0") * "Z_1" == "X_0 Z_1"


def test_flat_pauli_requires_a_token() -> None:
    with pytest.raises(ValueError, match=r"requires at least one token"):
        pauli()
