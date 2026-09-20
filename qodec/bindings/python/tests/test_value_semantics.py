"""Structural equality, repr, and sharing across every value type.

An ``__eq__`` comparing identity instead of contents silently breaks ``==``
and ``in``; a ``__repr__`` that panics or renders ``<object at 0x...>`` only
shows up in a debugger. Sharing decides whether assigning through a getter
takes effect.
"""

from __future__ import annotations

from collections.abc import MutableSequence
from pathlib import Path
from copy import copy, deepcopy
from typing import Any, Callable

import pytest

import qodec
from qodec.actions import Clifford, Condition, Observe, Pauli, Rotate, Stabilize
from qodec.codes import Code
from qodec import Reference
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
def test_instruction_set_preserves_shared_instruction_inputs(use_mapping: bool) -> None:
    block = Block("q", encodes=1)
    operand = BlockOperand("q")
    parameter = Parameter("theta", Parameter.Kind.NUMBER)
    instruction = Instruction(
        "rotate", inputs=[operand], outputs=[operand], parameters=[parameter], action=[Rotate("Z_0", "theta")]
    )
    instructions = {"rotate": instruction} if use_mapping else [instruction]
    instruction_set = InstructionSet("logical", blocks=[block], instructions=instructions)
    instruction_set.blocks = [block]
    instruction_set.instructions = instructions

    assert instruction_set.blocks == [block]
    assert instruction_set.instructions == {"rotate": instruction}
    assert instruction_set.instructions["rotate"] is instruction
    instruction_set.instructions["rotate"].flags.append("reject")
    assert instruction.flags == ["reject"]
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


def test_gadget_mapping_writes_through_and_survives_bulk_replacement() -> None:
    instruction = Instruction("R", outputs=[BlockOperand("q")])
    gadget = qodec.Gadget(
        instruction, Circuit(_isa(), "R 0"),
        outputs=[Encoding(Code("q", [], ["X_0"], ["Z_0"]), support=["0"])],
    )
    layer = qodec.Layer(_isa())
    view = layer.gadgets
    view["R"] = gadget
    assert layer.gadgets["R"] is gadget
    layer.gadgets = view
    layer.instruction_set.instructions = layer.instruction_set.instructions
    layer.instruction_set.metadata = layer.instruction_set.metadata
    layer.gadgets = []
    assert not view


@pytest.mark.parametrize("field", ["stabilizers", "x", "z"])
def test_operator_sequences_write_through_and_survive_bulk_replacement(field: str) -> None:
    code = Code("q", [], [], [])
    operators = getattr(code, field)
    assert isinstance(operators, MutableSequence)
    operators.extend(["X_0", "Z_1"])
    operators.insert(1, "Y_2")
    assert list(getattr(code, field)) == ["X_0", "Y_2", "Z_1"]
    operators[1:] = ["Z_3", "X_4"]
    del operators[::2]
    operators.reverse()
    assert operators.pop() == "Z_3"
    assert not getattr(code, field)
    code_values = ["Y_0"]
    setattr(code, field, code_values)
    code_values.clear()
    assert list(operators) == ["Y_0"]
    operators.clear()
    assert not getattr(code, field)


def test_sequence_snapshots_do_not_write_through() -> None:
    encoding = Encoding(Code("q", [], ["X_0"], ["Z_0"]), support=["0", "1"])
    support = encoding.support
    for snapshot in (support[:], copy(support)):
        snapshot[0] = "changed"
    assert list(encoding.support) == ["0", "1"]
    with pytest.raises(IndexError):
        support[10] = "3"
    with pytest.raises(ValueError):
        support[::2] = []
    assert list(encoding.support) == ["0", "1"]


def test_nested_metadata_views_follow_the_owner() -> None:
    code = Code("q", [], ["X_0"], ["Z_0"], metadata={"nested": {"values": [1]}})
    nested = code.metadata["nested"]
    code.metadata["nested"]["values"].append(2)
    assert code.metadata == {"nested": {"values": [1, 2]}}
    code.metadata = {"nested": {"values": [3]}}
    nested["values"].append(4)
    assert code.metadata == {"nested": {"values": [3, 4]}}
    before = deepcopy(dict(code.metadata))
    with pytest.raises((TypeError, ValueError)):
        nested["invalid"] = object()
    assert code.metadata == before


def test_instruction_mapping_rejects_mismatched_keys_atomically() -> None:
    instruction_set = _isa()
    before = dict(instruction_set.instructions)
    with pytest.raises(ValueError, match="key.*mnemonic"):
        instruction_set.instructions.update({"new": Instruction("new"), "wrong": Instruction("other")})
    assert instruction_set.instructions == before


def test_instruction_replacement_changes_only_the_mapping_slot() -> None:
    instruction = Instruction("R", outputs=[BlockOperand("q")])
    instruction_set = InstructionSet("physical", blocks=[Block("q", 1)], instructions=[instruction])
    gadget = qodec.Gadget(
        instruction, Circuit(_isa(), "R 0"),
        outputs=[Encoding(Code("q", [], ["X_0"], ["Z_0"]), support=["0"])],
    )
    replacement = Instruction("R", outputs=instruction.outputs, flags=["reject"])
    instruction_set.instructions["R"] = replacement
    assert instruction_set.instructions["R"] is replacement
    assert gadget.implements is instruction
    assert not gadget.implements.flags
    with pytest.raises(AttributeError):
        instruction.mnemonic = "renamed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="mnemonic"):
        gadget.implements = Instruction("renamed")
    assert gadget.implements is instruction


def test_live_instruction_edits_reach_parsing_and_serialization() -> None:
    instruction_set = _isa()
    protocol = qodec.Qodec([qodec.Layer(instruction_set)])
    circuit = Circuit(instruction_set, "- R: [0]", format="yaml")
    assert not circuit.readouts
    instruction_set.instructions["R"].action.append(Observe(["Z_0"]))
    assert len(circuit.readouts) == 1
    restored = qodec.Qodec.loads(protocol.dumps())
    assert restored == protocol
    assert restored.layers[0].instruction_set.instructions["R"].action[-1] == Observe(["Z_0"])


@pytest.mark.parametrize("field, item", [
    ("flags", "reject"),
    ("parameters", Parameter("enabled", Parameter.Kind.BIT)),
])
def test_instruction_sequence_updates_reject_duplicates_atomically(field: str, item: Any) -> None:
    instruction = Instruction("draft")
    values = getattr(instruction, field)
    values.append(item)
    with pytest.raises(ValueError, match="duplicate"):
        values.extend([item])
    assert list(getattr(instruction, field)) == [item]


def test_nested_collection_defaults_and_removal() -> None:
    code = Code("q", [], ["X_0"], ["Z_0"])
    code.metadata.setdefault("nested", {})["values"] = [1, 2]
    assert code.metadata == {"nested": {"values": [1, 2]}}
    removed = code.metadata.pop("nested")
    assert removed == {"values": [1, 2]} and not code.metadata
    code.metadata["rows"] = [[1], [2]]
    row = code.metadata["rows"].pop(0)
    assert row == [1]
    row.append(3)
    assert code.metadata == {"rows": [[2]]}
    assert code.metadata.setdefault("rows", []) == [[2]]
    assert code.metadata.pop("missing", "default") == "default"
    with pytest.raises(KeyError):
        code.metadata.pop("missing")
    key, value = code.metadata.popitem()
    assert key == "rows" and value == [[2]] and not code.metadata
    code.metadata.update({"keep": 1, "drop": 2})
    del code.metadata["drop"]
    snapshot = dict(code.metadata)
    snapshot.clear()
    assert code.metadata == {"keep": 1}
    code.metadata.clear()
    assert not code.metadata


@pytest.mark.parametrize("field", ["instructions", "blocks"])
def test_copying_collection_views_makes_explicit_snapshots(field: str) -> None:
    instruction_set = _isa()
    view = getattr(instruction_set, field)
    shallow, detached = copy(view), deepcopy(view)
    if field == "instructions":
        assert isinstance(shallow, dict) and isinstance(detached, dict)
        assert shallow["R"] is instruction_set.instructions["R"]
        assert detached["R"] is not instruction_set.instructions["R"]
    else:
        assert isinstance(shallow, list) and isinstance(detached, list)
    view.clear()
    assert shallow and detached


def test_model_copy_protocols_preserve_ownership() -> None:
    encoding, gadget, _, _, _ = _container_equality_case("Gadget")
    shallow = copy(gadget)
    assert shallow is not gadget and shallow == gadget
    assert shallow.circuit is gadget.circuit
    assert shallow.inputs[0] is encoding
    assert shallow.implements is gadget.implements
    detached, same_encoding = deepcopy((gadget, encoding))
    assert detached == gadget
    assert detached.circuit is not gadget.circuit
    assert detached.circuit.instruction_set is not gadget.circuit.instruction_set
    assert detached.inputs[0] is detached.outputs[0] is same_encoding
    assert same_encoding.code is not encoding.code
    detached.implements.flags.append("reject")
    detached.inputs[0].support[0] = "7"
    assert not gadget.implements.flags
    assert encoding.support == ["0", "1"]


def test_replace_preserves_unspecified_fields_and_shares_children() -> None:
    instruction = Instruction("prepare", description="retained", metadata={"version": [1]})
    changed = instruction.__replace__(flags=["reject"])
    assert changed is not instruction
    assert changed.flags == ["reject"] and not instruction.flags
    assert changed.description == instruction.description
    changed.metadata["version"].append(2)
    assert instruction.metadata == {"version": [1]}
    circuit = Circuit(_isa(), "R 0", format="stim")
    changed_circuit = circuit.__replace__(source="R 1", format=None)
    assert changed_circuit.source == "R 1" and changed_circuit.format is None
    assert changed_circuit.instruction_set is circuit.instruction_set
    assert circuit.source == "R 0" and circuit.format == "stim"
    assert instruction.__replace__() is not instruction
    with pytest.raises(TypeError, match="no replaceable field"):
        instruction.__replace__(unknown=1)


def test_loaded_instructions_and_deep_copies_share_within_their_protocols() -> None:
    protocol = qodec.Qodec.load(Path(__file__).parents[3] / "examples/repetition3/repetition3.qodec.yaml")
    copied, copied_layer = deepcopy((protocol, protocol.layers[0]))
    assert copied == protocol and copied.layers[0] is copied_layer
    assert copied.manifest_filename == protocol.manifest_filename
    for mnemonic, gadget in copied_layer.gadgets.items():
        assert gadget.implements is copied_layer.instruction_set.instructions[mnemonic]
        assert gadget.circuit.instruction_set is copied.layers[1].instruction_set
        assert gadget.implements is not protocol.layers[0].gadgets[mnemonic].implements
    instruction = copied_layer.instruction_set.instructions["prepare_z"]
    instruction.flags.append("reject")
    assert copied_layer.gadgets["prepare_z"].implements.flags == ["reject"]
    assert not protocol.layers[0].gadgets["prepare_z"].implements.flags
    node_instruction = copied.resolve('layers[0].instruction_set.instructions["prepare_z"]').value(Instruction)
    assert node_instruction is instruction


def test_derived_instruction_index_is_live_and_read_only() -> None:
    protocol = qodec.Qodec([qodec.Layer(_isa())])
    index = protocol.instruction_sets
    protocol.layers.append(qodec.Layer(InstructionSet("other")))
    assert set(index) == {"phys", "other"}
    with pytest.raises(TypeError):
        index["new"] = _isa()  # type: ignore[index]
    assert copy(index)["phys"] is protocol.layers[0].instruction_set
    assert deepcopy(index)["phys"] is not protocol.layers[0].instruction_set


@pytest.mark.parametrize("build, field", [
    (lambda: Condition(["enabled"]), "predicates"),
    (lambda: Stabilize(["Z_0"]), "operators"),
    (lambda: Observe(["Z_0"]), "observables"),
    (lambda: Clifford({"X_0": "Z_0"}), "generators"),
])
def test_action_value_containers_reject_mutation(build: Callable[[], Any], field: str) -> None:
    value = getattr(build(), field)
    with pytest.raises(TypeError):
        value[0 if isinstance(value, tuple) else "X_0"] = "changed"


@pytest.mark.parametrize("label, build, build_other", CASES, ids=IDS)
def test_value_copy_protocols_preserve_all_fields(
    label: str, build: Callable[[], Any], build_other: Callable[[], Any]
) -> None:
    original = build()
    for clone in (copy(original), deepcopy(original), original.__replace__()):
        assert type(clone) is type(original), label
        assert clone == original, label
    assert original.__replace__() is not original


def test_reference_replacement_preserves_authored_spelling() -> None:
    reference = Reference("circuit.readouts[00]")
    unchanged = reference.__replace__()
    changed = reference.__replace__(value="circuit.readouts[1:3]")
    assert unchanged is not reference and unchanged.path == reference.path
    assert changed.path == "circuit.readouts[1:3]"
    assert reference.path == "circuit.readouts[00]"
    with pytest.raises(ValueError):
        reference.__replace__(value="not a reference")


def test_gadget_replacement_shares_children_but_copies_equations() -> None:
    encoding, gadget, _, _, _ = _container_equality_case("Gadget")
    gadget.checks = [["circuit.readouts[0]"]]
    replacement = gadget.__replace__(checks=[])
    assert replacement is not gadget
    assert replacement.implements is gadget.implements
    assert replacement.circuit is gadget.circuit
    assert replacement.inputs[0] is replacement.outputs[0] is encoding
    assert not replacement.checks and len(gadget.checks) == 1
    replacement.checks.append((1,))
    assert gadget.checks[0] == ("circuit.readouts[0]",)
    with pytest.raises(ValueError):
        gadget.__replace__(outputs=[])
    assert gadget.outputs[0] is encoding


def test_model_replacement_does_not_install_or_rebind_children() -> None:
    instruction_set = _isa()
    layer = qodec.Layer(instruction_set)
    replacement = layer.__replace__(codes={"q": Code("q", [], ["X_0"], ["Z_0"])})
    assert replacement.instruction_set is instruction_set
    assert not layer.codes
    changed_set = instruction_set.__replace__(description="edited")
    assert changed_set.instructions["R"] is instruction_set.instructions["R"]
    assert layer.instruction_set is instruction_set
    assert instruction_set.description == "d"
    encoding = Encoding(replacement.codes["q"], support=["0"])
    changed_encoding = encoding.__replace__(support=["7"])
    assert changed_encoding.code is encoding.code
    assert list(encoding.support) == ["0"]


def test_qodec_replacement_preserves_loaded_history(tmp_path: Path) -> None:
    original = qodec.Qodec.load(Path(__file__).parents[3] / "examples/repetition3/repetition3.qodec.yaml")
    unchanged = original.__replace__()
    assert unchanged is not original and unchanged == original
    assert unchanged.layers[0] is original.layers[0]
    original_location = original.resolve("").source_location
    copied_location = unchanged.resolve("").source_location
    assert original_location is not None and copied_location is not None
    assert copied_location.path == original_location.path
    edited = original.__replace__(name=None, description=None, metadata={})
    assert edited.name == edited.description == ""
    assert original.name == "repetition3"
    assert edited.manifest_filename == original.manifest_filename
    restored = qodec.Qodec.load(edited.save(tmp_path, single_file=True))
    assert restored == edited


def test_copying_drafts_neither_validates_nor_parses() -> None:
    code = Code("draft", [], ["X_0"], [])
    instruction = Instruction("draft")
    gadget = qodec.Gadget(instruction, Circuit(_isa(), "not valid source", format="unknown"))
    gadget.outputs.append(Encoding(code, support=["0"]))
    for cloned in (copy(gadget), deepcopy(gadget)):
        assert cloned == gadget
        assert cloned.circuit.source == "not valid source"
        assert cloned.outputs[0].code.x == ["X_0"]
        assert not cloned.outputs[0].code.z


def test_replacement_does_not_inspect_the_replaced_action(tmp_path: Path) -> None:
    source = tmp_path / "conditional.isa.yaml"
    source.write_text(
        "name: T\nblocks: {q: 1}\ninstructions:\n"
        "- mnemonic: M\n  description: draft\n  in: [q]\n"
        "  flags: [reject]\n  action: [{observe: Z_0, unless: [reject]}]\n",
        encoding="utf-8",
    )
    original = InstructionSet.load(source).instructions["M"]
    replacement = original.__replace__(action=[Observe(["Z_0"])])
    assert replacement.action == [Observe(["Z_0"])]
    assert replacement.description == "draft"
    with pytest.raises(ValueError, match="conditional observe"):
        list(original.action)


def test_iterated_metadata_containers_remain_live() -> None:
    code = Code("q", [], [], [], metadata={"rows": [[1], [2]]})
    for row in code.metadata["rows"]:
        row.append(3)
    assert code.metadata == {"rows": [[1, 3], [2, 3]]}


def test_call_deepcopy_preserves_cycles_and_shared_arguments() -> None:
    values: list[Any] = [1]
    values.append(values)
    call = InstructionCall("draft", arguments={"first": values, "second": values})
    memo: dict[int, Any] = {}
    cloned = deepcopy(call, memo)
    assert cloned is not call
    assert call.__deepcopy__(memo) is cloned
    assert cloned == call
    cloned_values: list[Any] = memo[id(values)]
    assert cloned.arguments["first"] is cloned_values
    assert cloned.arguments["second"] is cloned_values
    assert cloned_values[1] is cloned_values
    cloned_values[0] = 2
    assert cloned.arguments["first"] == cloned_values
    assert values[0] == 1


def test_call_argument_assignment_preserves_recursive_values() -> None:
    values: list[Any] = []
    values.append(values)
    call = InstructionCall("draft")
    call.arguments.update({"first": values, "second": values})
    stored: Any = call.arguments["first"]
    assert id(stored) == id(call.arguments["second"])
    assert stored[0] is stored
    stored.append(2)
    current: Any = call.arguments["first"]
    assert len(current) == 2
    assert len(values) == 1


def test_error_types_are_exposed_and_ordered(tmp_path: Path) -> None:
    # The bindings never exercised error formatting, so a regression in a
    # message the Python user actually sees would not have been caught.
    assert issubclass(qodec.QodecLoadError, qodec.QodecError)
    missing_file = tmp_path / "does_not_exist.qodec.yaml"
    with pytest.raises(qodec.QodecLoadError) as caught:
        qodec.Qodec.load(missing_file)
    assert str(caught.value), "a load failure must carry a message"
    assert str(missing_file) in str(caught.value)
