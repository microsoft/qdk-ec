"""What each module exports, pinned.

Asserted as **set equality**, not membership: a subset check would let a new
export slip in unnoticed. Every addition or removal below should be a
deliberate line in a diff.
"""

from __future__ import annotations

import importlib
import inspect
import pickle
from typing import Any

import pytest

import qodec

SURFACE: dict[str, set[str]] = {
    "qodec": {
        "Code",
        "Gadget",
        "Instruction",
        "InstructionSet",
        "Layer",
        "Qodec",
        "Node",
        "SourceLocation",
        "Reference",
        "ReferenceLike",
        "QodecError",
        "QodecLoadError",
        "QodecSaveError",
        "__version__",
        "register",
        "actions",
        "codes",
        "gadgets",
        "instructions",
    },
    "qodec.codes": {"Code", "PauliExpression", "pauli"},
    "qodec.gadgets": {
        "Check",
        "Circuit",
        "Encoding",
        "Flag",
        "Gadget",
        "Outcome",
        "Readout",
        "ReadoutLike",
    },
    "qodec.actions": {
        "Clifford",
        "Condition",
        "Observe",
        "Pauli",
        "Rotate",
        "Stabilize",
    },
    "qodec.instructions": {
        "Block",
        "BlockOperand",
        "Instruction",
        "InstructionCall",
        "InstructionSet",
        "Parameter",
    },
}


@pytest.mark.parametrize("module_name", sorted(SURFACE))
def test_module_exports_exactly(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert set(module.__all__) == SURFACE[module_name]


@pytest.mark.parametrize("module_name", sorted(SURFACE))
def test_every_exported_name_resolves(module_name: str) -> None:
    """``__all__`` is a promise about attributes, not just a list of strings."""
    module = importlib.import_module(module_name)
    for name in module.__all__:
        assert hasattr(module, name), f"{module_name}.{name} does not resolve"


@pytest.mark.parametrize("module_name", sorted(SURFACE))
def test_exported_classes_have_canonical_pickle_paths(module_name: str) -> None:
    module = importlib.import_module(module_name)
    for name in module.__all__:
        value = getattr(module, name)
        if inspect.isclass(value):
            expected = "qodec" if getattr(qodec, name, None) is value else module_name
            assert value.__module__ == expected
            assert pickle.loads(pickle.dumps(value)) is value


def test_actions_have_one_home() -> None:
    """The action classes are reachable by one path only."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("qodec.instructions.actions")
    for name in SURFACE["qodec.actions"]:
        assert getattr(qodec.actions, name).__module__ == "qodec.actions"


def test_top_level_re_exports_are_the_curated_set() -> None:
    """Dual-exported names are deliberate, and are the same objects."""
    dual = {
        "Code": qodec.codes,
        "Gadget": qodec.gadgets,
        "Instruction": qodec.instructions,
        "InstructionSet": qodec.instructions,
    }
    for name, submodule in dual.items():
        assert getattr(qodec, name) is getattr(submodule, name)


def test_validation_error_is_gone() -> None:
    """Argument checks raise ``ValueError``, not a bespoke type."""
    assert not hasattr(qodec, "ValidationError")
    protocol = qodec.Qodec([], name="empty")
    with pytest.raises(ValueError, match=r"start index 2 must be <= stop index 1"):
        protocol.slice(2, 1)

def test_references_have_one_public_home() -> None:
    assert qodec.Reference.__module__ == "qodec"
    assert not hasattr(qodec.gadgets, "Reference")
    assert not hasattr(qodec.gadgets, "ReferenceLike")


def test_parameter_kind_is_nested_and_picklable() -> None:
    """The kind enum lives on the type it describes, and only there."""
    assert not hasattr(qodec.instructions, "ParameterKind")
    assert not hasattr(qodec.instructions, "Kind")
    kind = qodec.instructions.Parameter.Kind
    assert qodec.instructions.Parameter("theta", "number").kind is kind.NUMBER
    assert pickle.loads(pickle.dumps(kind.BIT)) is kind.BIT


def test_pauli_letter_builders_are_gone() -> None:
    """``pauli()`` is the one builder; the single letters shadowed too much."""
    for name in ("I", "X", "Y", "Z"):
        assert not hasattr(qodec.codes, name)
    assert qodec.codes.pauli("X_0", "X_1") * "Z_2" == "X_0 X_1 Z_2"


PINNED_MEMBERS: list[tuple[type, set[str]]] = [
        (qodec.Qodec, {"name", "description", "schema_version", "manifest_filename", "metadata", "layers", "instruction_sets", "codes", "load", "loads", "save", "dumps", "slice", "validate", "resolve"}),
        (qodec.Layer, {"instruction_set", "codes", "gadgets"}),
        (qodec.instructions.InstructionCall, {"mnemonic", "operands", "arguments", "select"}),
        (qodec.gadgets.Circuit, {"instruction_set", "source", "format", "effective_format", "calls", "blocks", "readouts"}),
        (qodec.gadgets.Readout, {"position", "name", "is_flag", "equation"}),
        (qodec.Reference, {"path", "segments", "expand", "Field", "Key", "Index", "Slice", "Union"}),
        (qodec.Reference.Field, {"name"}),
        (qodec.Reference.Key, {"value"}),
        (qodec.Reference.Index, {"value"}),
        (qodec.Reference.Slice, {"start", "stop", "step"}),
        (qodec.Reference.Union, {"indices"}),
        (qodec.InstructionSet, {"name", "description", "blocks", "instructions", "metadata", "load", "save"}),
        (qodec.Instruction, {"mnemonic", "description", "inputs", "outputs", "flags", "observe_count", "parameters", "action", "metadata"}),
        (qodec.Code, {"name", "description", "stabilizers", "x", "z", "logical_count", "physical_qubit_count", "metadata", "load", "save"}),
        (qodec.Gadget, {"implements", "circuit", "inputs", "outputs", "checks", "readouts", "frames", "parameter_bindings", "metadata", "resolve"}),
        (qodec.Node, {"path", "source_location", "resolve", "value", "sequence_nodes", "mapping_nodes"}),
        (qodec.SourceLocation, {"path", "line"}),
        (qodec.gadgets.Encoding, {"code", "support", "block_types"}),
        (qodec.gadgets.Outcome, {"observable", "instruction"}),
        (qodec.gadgets.Flag, {"name", "instruction"}),
        (qodec.instructions.Block, {"name", "encodes"}),
        (qodec.instructions.BlockOperand, {"block", "is_variadic"}),
        (qodec.instructions.Parameter, {"name", "kind", "Kind"}),
        (qodec.codes.PauliExpression, {"text"}),
        (qodec.actions.Condition, {"predicates", "invert"}),
        (qodec.actions.Stabilize, {"operators", "condition"}),
        (qodec.actions.Clifford, {"generators", "condition"}),
        (qodec.actions.Pauli, {"operator", "condition"}),
        (qodec.actions.Observe, {"observables"}),
        (qodec.actions.Rotate, {"pauli", "angle", "condition"}),
]


@pytest.mark.parametrize(("owner", "expected"), PINNED_MEMBERS)
def test_exported_class_members_are_exact(owner: type, expected: set[str]) -> None:
    assert {name for name in dir(owner) if not name.startswith("_")} == expected


def test_circuit_blocks_are_a_read_only_property_without_qubits_alias() -> None:
    circuit = qodec.gadgets.Circuit(qodec.InstructionSet("test"), "[]", format="yaml")
    assert circuit.blocks == []
    assert not hasattr(circuit, "qubits")
    with pytest.raises(AttributeError):
        setattr(circuit, "blocks", [])


def test_every_exported_class_is_pinned() -> None:
    """A newly exported class must join the parametrization, or nothing pins it."""
    pinned = {owner for owner, _ in PINNED_MEMBERS}
    exported = {
        getattr(module, name)
        for module in (qodec, qodec.actions, qodec.codes, qodec.gadgets, qodec.instructions)
        for name in module.__all__
        if isinstance(getattr(module, name), type) and not issubclass(getattr(module, name), Exception)
    }
    assert exported - pinned == set()


def test_instruction_observe_count_excludes_flags() -> None:
    instruction = qodec.Instruction("measure", flags=["reject"], action=[
        qodec.actions.Observe(["Z_0", "Z_1"]),
        qodec.actions.Pauli("X_0"),
        qodec.actions.Observe(["Z_0"]),
    ])
    assert instruction.observe_count == 3


def test_instruction_calls_reject_predicates_without_affecting_action_guards() -> None:
    call_type: Any = qodec.instructions.InstructionCall
    assert "predicates" not in inspect.signature(call_type).parameters
    with pytest.raises(TypeError, match="predicates"):
        call_type("M", **{"predicates": ["enabled"]})
    assert qodec.actions.Condition(["enabled"]).predicates == ("enabled",)


def test_instruction_set_keywords_and_call_lookup() -> None:
    from qodec.gadgets import Circuit
    from qodec.instructions import Block

    instruction = qodec.Instruction("idle")
    instruction_set = qodec.InstructionSet("test", blocks=[Block("qubit", encodes=1)], instructions=[instruction])
    layer = qodec.Layer(instruction_set=instruction_set)
    circuit = Circuit(instruction_set=layer.instruction_set, source="- idle: []", format="yaml")
    call = circuit.calls()[0]
    assert circuit.instruction_set.instructions[call.mnemonic] == instruction
    assert circuit.instruction_set.blocks == instruction_set.blocks
    for owner in (qodec.Layer, Circuit):
        parameters = inspect.signature(owner).parameters
        assert "instruction_set" in parameters
        assert "isa" not in parameters
    replacement = qodec.InstructionSet("replacement", instructions=[instruction])
    layer.instruction_set = replacement
    circuit.instruction_set = replacement
    assert layer.instruction_set is replacement
    assert circuit.instruction_set is replacement
    assert [call.mnemonic for call in circuit.calls()] == ["idle"]
