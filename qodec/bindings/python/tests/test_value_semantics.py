"""Structural equality, repr, and sharing across every value type.

An ``__eq__`` comparing identity instead of contents silently breaks ``==``
and ``in``; a ``__repr__`` that panics or renders ``<object at 0x...>`` only
shows up in a debugger. Sharing decides whether assigning through a getter
takes effect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

import qodec
from qodec.actions import Clifford, Condition, Observe, Pauli, Rotate, Stabilize
from qodec.codes import Code
from qodec.gadgets import Circuit, Encoding
from qodec.instructions import Block, BlockOperand, Instruction, InstructionCall, InstructionSet, Parameter

# (label, build an instance, build a structurally different instance)
CASES: list[tuple[str, Callable[[], Any], Callable[[], Any]]] = [
    ("Block", lambda: Block("q", encodes=1), lambda: Block("q", encodes=2)),
    ("BlockOperand", lambda: BlockOperand("q"), lambda: BlockOperand("r")),
    ("Parameter", lambda: Parameter("theta", Parameter.Kind.NUMBER), lambda: Parameter("phi", Parameter.Kind.NUMBER)),
    ("Stabilize", lambda: Stabilize(["Z_0"]), lambda: Stabilize(["X_0"])),
    ("Pauli", lambda: Pauli("X_0"), lambda: Pauli("Y_0")),
    ("Observe", lambda: Observe(["Z_0"]), lambda: Observe(["Z_1"])),
    ("Clifford", lambda: Clifford({"X_0": "X_0"}), lambda: Clifford({"X_0": "Z_0"})),
    ("Rotate", lambda: Rotate("Z_0", "theta"), lambda: Rotate("X_0", "theta")),
    ("Condition", lambda: Condition(["a"]), lambda: Condition(["a"], invert=True)),
    (
        "Instruction",
        lambda: Instruction(mnemonic="i", inputs=[BlockOperand("q")]),
        lambda: Instruction(mnemonic="j", inputs=[BlockOperand("q")]),
    ),
    (
        "Code",
        lambda: Code(name="c", stabilizers=["Z_0 Z_1"], x=["X_0 X_1"], z=["Z_0"]),
        lambda: Code(name="d", stabilizers=["Z_0 Z_1"], x=["X_0 X_1"], z=["Z_0"]),
    ),
]

IDS = [label for label, _, _ in CASES]


def _isa() -> InstructionSet:
    return InstructionSet(
        name="phys",
        description="d",
        blocks=[Block("q", encodes=1)],
        instructions=[Instruction(mnemonic="R", outputs=[BlockOperand("q")], action=[Stabilize(["Z_0"])])],
    )


@pytest.mark.parametrize(("label", "build", "build_other"), CASES, ids=IDS)
def test_equality_is_structural(label: str, build: Callable[[], Any], build_other: Callable[[], Any]) -> None:
    first, second, other = build(), build(), build_other()
    assert first is not second, f"{label}: the two builds must be distinct objects for this to mean anything"
    assert first == second, f"{label}: structurally equal values must compare equal"
    assert not (first != second), f"{label}: __ne__ must agree with __eq__"
    assert first != other, f"{label}: structurally different values must not compare equal"


@pytest.mark.parametrize(("label", "build", "build_other"), CASES, ids=IDS)
def test_equality_against_a_foreign_type_is_false_not_an_error(
    label: str, build: Callable[[], Any], build_other: Callable[[], Any]
) -> None:
    value = build()
    for foreign in (None, 42, "text", object()):
        assert value != foreign, f"{label}: must not equal {foreign!r}"


@pytest.mark.parametrize(("label", "build", "build_other"), CASES, ids=IDS)
def test_repr_names_the_type_and_is_not_the_default(
    label: str, build: Callable[[], Any], build_other: Callable[[], Any]
) -> None:
    rendered = repr(build())
    assert rendered, f"{label}: repr must not be empty"
    assert "object at 0x" not in rendered, f"{label}: repr is the default, not a real one: {rendered}"
    assert label in rendered, f"{label}: repr should name its type, got {rendered}"


def test_pyo3_value_types_are_unhashable() -> None:
    # Structural __eq__ without __hash__ makes these unhashable by design;
    # they are used as dict values, not keys.
    with pytest.raises(TypeError):
        hash(Block("q", encodes=1))


@pytest.mark.parametrize(("attribute", "value"), [("name", "r"), ("encodes", 2)])
def test_block_properties_are_read_only(attribute: str, value: str | int) -> None:
    block = Block("q", encodes=1)
    with pytest.raises(AttributeError, match=attribute):
        setattr(block, attribute, value)
    assert block.name == "q"
    assert block.encodes == 1
    assert block == Block("q", encodes=1)
    assert not (block != Block("q", encodes=1))


def test_instruction_call_exposes_operands_arguments_and_repr() -> None:
    circuit = Circuit(instruction_set=_isa(), source="R 0 1")
    calls = circuit.calls()
    assert len(calls) == 2
    call = calls[0]

    assert call.mnemonic == "R"
    assert list(call.operands) == [0]
    assert dict(call.arguments) == {}
    rendered = repr(call)
    assert "InstructionCall" in rendered and "R" in rendered, rendered

    assert call == circuit.calls()[0]
    assert call != calls[1]


def test_instruction_exposes_action_and_parameters() -> None:
    instruction = Instruction(
        mnemonic="rotate",
        inputs=[BlockOperand("q")],
        parameters=[Parameter("theta", Parameter.Kind.NUMBER)],
        action=[Rotate("Z_0", "theta")],
    )
    assert [p.name for p in instruction.parameters] == ["theta"]
    assert len(instruction.action) == 1
    assert "rotate" in repr(instruction)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda condition: Stabilize(["Z_0"], condition=condition), id="Stabilize"),
        pytest.param(lambda condition: Clifford({"X_0": "Z_0"}, condition=condition), id="Clifford"),
        pytest.param(lambda condition: Pauli("X_0", condition=condition), id="Pauli"),
        pytest.param(lambda condition: Rotate("Z_0", "theta", condition=condition), id="Rotate"),
    ],
)
def test_action_conditions_remain_values_through_instruction_conversion(build: Callable[[Condition], Any]) -> None:
    predicates = ["enabled"]
    condition = Condition(predicates, invert=True)
    action = build(condition)
    predicates.append("changed")
    instruction = Instruction("guarded", action=[action])

    assert action.condition == Condition(["enabled"], invert=True)
    assert action.condition is not condition
    assert instruction.action == [action]
    restored = instruction.action[0]
    assert isinstance(restored, (Stabilize, Clifford, Pauli, Rotate))
    assert restored.condition == condition


@pytest.mark.parametrize("use_mapping", [False, True])
def test_instruction_set_accepts_reused_value_inputs(use_mapping: bool) -> None:
    block = Block("q", encodes=1)
    operand = BlockOperand("q")
    parameter = Parameter("theta", Parameter.Kind.NUMBER)
    instruction = Instruction(
        "rotate", inputs=[operand], outputs=[operand], parameters=[parameter], action=[Rotate("Z_0", "theta")]
    )
    instructions = {"ignored": instruction} if use_mapping else [instruction]
    instruction_set = InstructionSet("logical", blocks=[block], instructions=instructions)
    instruction_set.blocks = [block]
    instruction_set.instructions = instructions

    assert instruction_set.blocks == [block]
    assert instruction_set.instructions == {"rotate": instruction}
    assert instruction_set.instructions["rotate"] is not instruction
    assert instruction.inputs == instruction.outputs == [operand]
    assert instruction.parameters == [parameter]


@pytest.mark.parametrize("block", ["q", "q..."])
@pytest.mark.parametrize("is_variadic", [None, False, True])
def test_variadic_suffix_enables_block_operand(block: str, is_variadic: bool | None) -> None:
    operand = BlockOperand(block) if is_variadic is None else BlockOperand(block, is_variadic=is_variadic)
    assert operand.block == "q"
    assert operand.is_variadic == (block.endswith("...") or is_variadic is True)


@pytest.mark.parametrize("left, right", [(True, 1), (False, 0), (True, 1.0), (1, 1.0), ([True], [1]), ([[1]], [[1.0]])])
def test_call_equality_distinguishes_literal_kinds(left: Any, right: Any) -> None:
    first = InstructionCall("draft", arguments={"value": left})
    second = InstructionCall("draft", arguments={"value": right})
    assert first != second
    assert second != first
    assert first == InstructionCall("draft", arguments={"value": left})


def test_call_equality_handles_cyclic_draft_lists() -> None:
    first_values: list[Any] = [True]
    second_values: list[Any] = [True]
    first_values.append(first_values)
    second_values.append(second_values)
    first = InstructionCall("draft", arguments={"value": first_values})
    second = InstructionCall("draft", arguments={"value": second_values})
    assert first == second
    second_values[0] = 1
    assert first != second


def test_call_equality_does_not_coerce_draft_integers() -> None:
    class Count(int):
        pass

    assert InstructionCall("draft", arguments={"value": Count(1)}) == InstructionCall("draft", arguments={"value": 1})
    assert InstructionCall("draft", arguments={"value": 1 << 100}) == InstructionCall("draft", arguments={"value": 1 << 100})


def test_call_equality_propagates_comparison_errors() -> None:
    class InvalidComparison:
        def __eq__(self, other: object) -> bool:
            raise RuntimeError("comparison failed")

    value: Any = InvalidComparison()
    with pytest.raises(RuntimeError, match="comparison failed"):
        _ = InstructionCall("draft", arguments={"value": value}) == InstructionCall("draft", arguments={"value": value})


def test_container_types_compare_structurally() -> None:
    code = Code(name="c", stabilizers=["Z_0 Z_1"], x=["X_0 X_1"], z=["Z_0"])
    assert Encoding(code=code, support=["0", "1"]) == Encoding(code=code, support=["0", "1"])
    assert Encoding(code=code, support=["0", "1"]) != Encoding(code=code, support=["0", "2"])
    assert _isa() == _isa()
    assert Circuit(instruction_set=_isa(), source="R 0") == Circuit(instruction_set=_isa(), source="R 0")
    assert Circuit(instruction_set=_isa(), source="R 0") != Circuit(instruction_set=_isa(), source="R 1")


@pytest.mark.parametrize(
    ("block_types", "different_block_types"),
    [([], ["q", "q"]), (["q", "q"], ["q", "r"]), (["q", "r"], ["r", "q"])],
)
def test_encoding_block_types_affect_equality(
    block_types: list[str], different_block_types: list[str]
) -> None:
    code = Code(name="c", stabilizers=["Z_0 Z_1"], x=["X_0 X_1"], z=["Z_0"])
    first = Encoding(code=code, support=["0", "1"], block_types=block_types)
    second = Encoding(code=code, support=["0", "1"], block_types=block_types)
    assert first is not second
    assert first == second
    assert not (first != second)

    second.block_types = different_block_types
    assert not (first == second)
    assert first != second

    first.block_types = different_block_types
    assert first == second
    assert not (first != second)


def _container_equality_case(container: str) -> tuple[Encoding, qodec.Gadget, qodec.Gadget, object, object]:
    code = Code(name="c", stabilizers=["Z_0 Z_1"], x=["X_0 X_1"], z=["Z_0"])
    instruction = Instruction("idle", inputs=[BlockOperand("q")], outputs=[BlockOperand("q")])
    instruction_set = InstructionSet(name="logical", blocks=[Block("q", encodes=1)], instructions=[instruction])
    encoding = Encoding(code=code, support=["0", "1"], block_types=["q", "q"])
    first_gadget = qodec.Gadget(
        instruction, Circuit(instruction_set=_isa(), source=""), inputs=[encoding], outputs=[encoding]
    )
    second_gadget = qodec.Gadget(
        instruction, Circuit(instruction_set=_isa(), source=""), inputs=[encoding], outputs=[encoding]
    )
    first_layer = qodec.Layer(instruction_set, gadgets=[first_gadget])
    second_layer = qodec.Layer(instruction_set, gadgets=[second_gadget])
    first, second = {
        "Gadget": (first_gadget, second_gadget),
        "Layer": (first_layer, second_layer),
        "Qodec": (
            qodec.Qodec([first_layer, qodec.Layer(_isa())]),
            qodec.Qodec([second_layer, qodec.Layer(_isa())]),
        ),
    }[container]
    return encoding, first_gadget, second_gadget, first, second


@pytest.mark.parametrize("boundary", ["inputs", "outputs"])
@pytest.mark.parametrize("container", ["Gadget", "Layer", "Qodec"])
def test_encoding_block_types_affect_container_equality(boundary: str, container: str) -> None:
    encoding, first_gadget, second_gadget, first, second = _container_equality_case(container)
    assert first is not second
    assert first == second
    assert not (first != second)

    differing = Encoding(code=encoding.code, support=encoding.support, block_types=["q", "r"])
    setattr(second_gadget, boundary, [differing])
    assert not (first == second)
    assert first != second

    setattr(first_gadget, boundary, [differing])
    assert first == second
    assert not (first != second)


def test_encodings_are_shared_like_every_other_gadget_component() -> None:
    encoding, first_gadget, _, _, _ = _container_equality_case("Gadget")
    assert first_gadget.inputs[0] is first_gadget.inputs[0]
    assert first_gadget.inputs[0] is encoding

    first_gadget.inputs[0].support = ["2", "3"]
    assert first_gadget.inputs[0].support == ["2", "3"]
    assert encoding.support == ["2", "3"]


def test_error_types_are_exposed_and_ordered(tmp_path: Path) -> None:
    # The bindings never exercised error formatting, so a regression in a
    # message the Python user actually sees would not have been caught.
    assert issubclass(qodec.QodecLoadError, qodec.QodecError)
    missing_file = tmp_path / "does_not_exist.qodec.yaml"
    with pytest.raises(qodec.QodecLoadError) as caught:
        qodec.Qodec.load(missing_file)
    assert str(caught.value), "a load failure must carry a message"
    assert str(missing_file) in str(caught.value)
