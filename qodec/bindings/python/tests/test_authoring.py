"""End-to-end authoring test for the qodec Python bindings.

Builds a minimal but complete qodec entirely from Python objects — the
distance-3 bit-flip repetition code, mirroring
``examples/repetition3`` — then exercises the build -> save -> load
round-trip. This pins the public authoring surface (constructor
signatures, parameter names, module paths) so the documented example
cannot silently drift from the API again.

Run with::

    pytest bindings/python/tests/test_authoring.py
"""

from __future__ import annotations

import ast
from collections import UserDict
import json
from pathlib import Path

import pytest

import qodec
from qodec.actions import Observe, Stabilize
from qodec.codes import Code
from qodec.instructions import BlockOperand, Instruction, InstructionSet, Parameter


def _build_repetition3() -> qodec.Qodec:
    """Assemble the distance-3 repetition-code qodec from scratch."""
    logical_isa = InstructionSet(
        name="repetition3",
        description="Logical ISA for the 3-qubit bit-flip repetition code.",
        blocks=[qodec.instructions.Block("repetition3", encodes=1)],
        instructions=[
            Instruction(
                mnemonic="prepare_z",
                outputs=[BlockOperand("repetition3")],
                action=[Stabilize(["Z_0"])],
            ),
            Instruction(
                mnemonic="measure_z",
                inputs=[BlockOperand("repetition3")],
                action=[Observe(["Z_0"])],
            ),
        ],
    )

    physical_isa = InstructionSet(
        name="stim",
        description="Minimal stim-like physical ISA.",
        blocks=[qodec.instructions.Block("qubit", encodes=1)],
        instructions=[
            Instruction(
                mnemonic="R",
                outputs=[BlockOperand("qubit")],
                action=[Stabilize(["Z_0"])],
            ),
            Instruction(
                mnemonic="M",
                inputs=[BlockOperand("qubit")],
                action=[Observe(["Z_0"])],
            ),
        ],
    )

    code = Code(
        name="repetition3",
        stabilizers=["Z_0 Z_1", "Z_1 Z_2"],
        x=["X_0 X_1 X_2"],
        z=["Z_0"],
    )
    encoding = qodec.gadgets.Encoding(code=code, support=["0", "1", "2"])

    prepare = qodec.Gadget(
        implements=logical_isa.instructions["prepare_z"],
        circuit=qodec.gadgets.Circuit(instruction_set=physical_isa, source="R 0 1 2"),
        outputs=[encoding],
    )
    measure = qodec.Gadget(
        implements=logical_isa.instructions["measure_z"],
        circuit=qodec.gadgets.Circuit(instruction_set=physical_isa, source="M 0 1 2"),
        inputs=[encoding],
        checks=[
            ["circuit.readouts[0:2]", "in[0].stabilizers[0]"],
            ["circuit.readouts[1:3]", "in[0].stabilizers[1]"],
        ],
        readouts=[["circuit.readouts[0]", "in[0].z[0]"]],
    )

    return qodec.Qodec(
        layers=[
            qodec.Layer(logical_isa, gadgets=[prepare, measure]),
            qodec.Layer(physical_isa),
        ],
        name="repetition3",
    )


def test_build_qodec_from_scratch() -> None:
    codec = _build_repetition3()
    assert codec.name == "repetition3"
    assert [layer.instruction_set.name for layer in codec.layers] == ["repetition3", "stim"]

    layer = codec.layers[0]
    assert set(layer.gadgets) == {"prepare_z", "measure_z"}
    (readout,) = layer.gadgets["measure_z"].readouts
    assert readout.equation == ("circuit.readouts[0]", "in[0].z[0]")
    assert codec.layers[1].gadgets == {}

def test_frames_are_sparse_copied_and_preserved() -> None:
    protocol = _build_repetition3()
    gadget = protocol.layers[0].gadgets["prepare_z"]
    assert gadget.frames == {}
    gadget.frames = {"out[0].z[0]": ["circuit.readouts[0:2]"], "out[0].x[0]": []}
    snapshot = gadget.frames
    snapshot.clear()
    assert gadget.frames == {"out[0].z[0]": ("circuit.readouts[0:2]",), "out[0].x[0]": ()}
    restored = qodec.Qodec.loads(protocol.dumps())
    assert restored.layers[0].gadgets["prepare_z"].frames == gadget.frames
    assert "frames:" in str(gadget)
    restored.layers[0].gadgets["prepare_z"].frames = {}
    assert restored != qodec.Qodec.loads(protocol.dumps())


def test_bad_frame_term_does_not_replace_existing_frames() -> None:
    gadget = _build_repetition3().layers[0].gadgets["prepare_z"]
    gadget.frames = {"out[0].z[0]": []}
    with pytest.raises(ValueError, match="reference"):
        gadget.frames = {"out[0].z[0]": ["not-a-reference"]}
    assert gadget.frames == {"out[0].z[0]": ()}

def test_frame_constructor_and_setter_accept_mappings() -> None:
    original = _build_repetition3().layers[0].gadgets["prepare_z"]
    frames: UserDict[str, qodec.gadgets.Check] = UserDict({"out[0].z[0]": (1,)})
    gadget = qodec.Gadget(original.implements, original.circuit, outputs=original.outputs, frames=frames)
    assert gadget.frames == {"out[0].z[0]": (1,)}
    gadget.frames = UserDict({"out[0].z[0]": []})
    assert gadget.frames == {"out[0].z[0]": ()}

def test_literal_bits_survive_all_equation_roles() -> None:
    protocol = _build_repetition3()
    gadget = protocol.layers[0].gadgets["measure_z"]
    gadget.checks = [[0, 1, "circuit.readouts[0]"]]
    gadget.readouts = [[1, "circuit.readouts[0]"]]
    gadget.frames = {"out[0].z[0]": [1]}
    restored = qodec.Qodec.loads(protocol.dumps()).layers[0].gadgets["measure_z"]
    assert restored.checks == gadget.checks
    assert restored.readouts == gadget.readouts
    assert restored.frames == gadget.frames
    assert type(restored.checks[0][0]) is int
    assert ast.literal_eval(str(restored.readouts[0])) == [1, "circuit.readouts[0]"]


@pytest.mark.parametrize("invalid", [True, False, 2, -1, 1.0, "1", None])
def test_equation_constants_require_integer_bits(invalid: object) -> None:
    gadget = _build_repetition3().layers[0].gadgets["prepare_z"]
    gadget.frames = {"out[0].z[0]": [1]}
    with pytest.raises((TypeError, ValueError)):
        gadget.frames = {"out[0].z[0]": [invalid]}  # type: ignore[list-item]
    assert gadget.frames == {"out[0].z[0]": (1,)}


@pytest.mark.parametrize("round_trip", [False, True])
def test_invalid_algebra_round_trips(round_trip: bool, tmp_path: Path) -> None:
    codec = _build_repetition3()
    if round_trip:
        codec = qodec.Qodec.loads(codec.dumps())
    codec.validate()

    code = codec.codes["repetition3"]
    code.z = [pauli for pauli in code.x]
    codec.validate()
    assert qodec.Qodec.loads(codec.dumps()).codes[code.name] == code
    for single_file in [False, True]:
        destination = tmp_path / str(single_file)
        codec.save(destination, single_file=single_file)
        assert qodec.Qodec.load(destination / codec.manifest_filename).codes[code.name] == code
    standalone = Code("invalid", stabilizers=["X_0", "Z_0"], x=["X_1"], z=["X_1"])
    path = tmp_path / "invalid.code.yaml"
    standalone.save(path)
    assert Code.load(path) == standalone


def test_unresolved_parity_references_round_trip() -> None:
    codec = _build_repetition3()
    measure = codec.layers[0].gadgets["measure_z"]
    measure.checks = [["circuit.readouts[99]"]]
    codec.validate()
    assert qodec.Qodec.loads(codec.dumps()).layers[0].gadgets["measure_z"].checks == measure.checks
    assert not hasattr(codec, "validation_issues")
    assert not hasattr(qodec, "ValidationIssue")


def test_openqasm_source_roundtrips_without_parsing(tmp_path: Path) -> None:
    codec = _build_repetition3()
    circuit = codec.layers[0].gadgets["measure_z"].circuit
    circuit.source = "OPENQASM 3.0;\nqubit[3] data;\n"
    circuit.format = "openqasm"
    codec.validate()
    reloaded = qodec.Qodec.loads(codec.dumps())
    assert reloaded.layers[0].gadgets["measure_z"].circuit == circuit
    for single_file in [False, True]:
        destination = tmp_path / str(single_file)
        codec.save(destination, single_file=single_file)
        reloaded = qodec.Qodec.load(destination / codec.manifest_filename)
        saved_circuit = reloaded.layers[0].gadgets["measure_z"].circuit
        assert saved_circuit == circuit
        with pytest.raises(ValueError, match="openqasm"):
            _ = saved_circuit.calls()


def _build_two_block_protocol() -> tuple[qodec.Qodec, qodec.Gadget]:
    from qodec.gadgets import Circuit, Encoding
    from qodec.instructions import Block

    operation = Instruction(
        "idle",
        inputs=[BlockOperand("q"), BlockOperand("q")],
        outputs=[BlockOperand("q"), BlockOperand("q")],
    )
    source = InstructionSet("logical", blocks=[Block("q", encodes=1)], instructions=[operation])
    target = InstructionSet("physical", blocks=[Block("a", encodes=1), Block("b", encodes=1)])
    code = Code("qubit", stabilizers=[], x=["X_0"], z=["Z_0"])
    encodings = [
        Encoding(code=code, support=["0"], block_types=["a"]),
        Encoding(code=code, support=["1"], block_types=["b"]),
    ]
    gadget = qodec.Gadget(operation, Circuit(target, "[]", format="yaml"), inputs=encodings, outputs=encodings)
    codec = qodec.Qodec([qodec.Layer(source, gadgets=[gadget]), qodec.Layer(target)])
    codec.validate()
    qodec.Qodec.loads(codec.dumps()).validate()
    return codec, gadget


def _overlap_boundary_blocks(gadget: qodec.Gadget, side: str) -> None:
    boundary = gadget.inputs if side == "input" else gadget.outputs
    boundary[1].support = ["0"]
    if side == "input":
        gadget.inputs = boundary
    else:
        gadget.outputs = boundary


@pytest.mark.parametrize("side", ["input", "output"])
def test_encoding_type_conflicts_fail_before_saving(side: str, tmp_path: Path) -> None:
    codec, gadget = _build_two_block_protocol()
    _overlap_boundary_blocks(gadget, side)
    with pytest.raises(ValueError, match="conflicting block types 'a' and 'b'"):
        codec.validate()
    with pytest.raises(qodec.QodecSaveError, match="conflicting block types 'a' and 'b'"):
        codec.dumps()
    for single_file in [False, True]:
        destination = tmp_path / str(single_file)
        with pytest.raises(qodec.QodecSaveError, match="conflicting block types 'a' and 'b'"):
            codec.save(destination, single_file=single_file)
        assert not destination.exists()


def test_a_gadget_key_must_match_the_instruction_it_implements() -> None:
    codec = _build_repetition3()
    layer = codec.layers[0]
    with pytest.raises(ValueError, match="does not match the implemented instruction"):
        layer.gadgets = {"missing": layer.gadgets["measure_z"]}


def test_validate_checks_layer_relationships() -> None:
    codec = _build_repetition3()
    layer = codec.layers[0]
    instruction_set = layer.instruction_set
    instruction_set.instructions = [
        instruction for mnemonic, instruction in instruction_set.instructions.items() if mnemonic != "measure_z"
    ]
    with pytest.raises(ValueError, match="not declared by the layer"):
        codec.validate()
    for stop in [0, 1]:
        partial = qodec.Qodec([qodec.Layer(item.instruction_set) for item in codec.layers[:stop]])
        partial.validate()
        assert len(qodec.Qodec.loads(partial.dumps()).layers) == stop


def test_gadget_analytical_surface_returns_references() -> None:
    from qodec.gadgets import Reference

    codec = _build_repetition3()
    measure = codec.layers[0].gadgets["measure_z"]

    checks = measure.checks
    assert all(isinstance(reference, Reference) for equation in checks for reference in equation)
    # Compact slice selectors round-trip verbatim.
    assert checks[0][0] == "circuit.readouts[0:2]"
    assert checks[0][1] == "in[0].stabilizers[0]"

    readout_entry = measure.readouts[0]
    readout_reference = readout_entry.equation[1]
    assert isinstance(readout_reference, Reference)
    assert readout_reference == "in[0].z[0]"


def test_gadget_readouts_roundtrip_anonymous_and_named() -> None:
    from qodec.gadgets import Reference

    gadget = _build_repetition3().layers[0].gadgets["measure_z"]
    gadget.readouts = [
        ["circuit.readouts[0]", "in[0].z[0]"],
        {"reject": [Reference("circuit.readouts[1]"), "circuit.readouts[2]"]},
    ]

    readouts = gadget.readouts
    anonymous, named = readouts
    assert (anonymous.position, anonymous.name) == (0, None)
    assert (named.position, named.name) == (1, "reject")
    assert anonymous.equation == ("circuit.readouts[0]", "in[0].z[0]")
    assert named.equation == ("circuit.readouts[1]", "circuit.readouts[2]")
    assert isinstance(anonymous.equation[0], Reference)
    assert isinstance(named.equation[0], Reference)
    assert (anonymous.is_flag, named.is_flag) == (False, True)


def test_readouts_accept_returned_values_and_rebind_positions() -> None:
    original = _build_repetition3().layers[0].gadgets["measure_z"]
    original.readouts = (
        ("circuit.readouts[0]", "in[0].z[0]"),
        {"reject": ("circuit.readouts[1:3]",)},
    )
    returned = original.readouts
    clone = qodec.Gadget(
        original.implements, original.circuit, inputs=original.inputs,
        checks=original.checks, readouts=returned,
    )
    assert clone.readouts == returned
    clone.readouts = returned
    assert clone.readouts == returned
    clone.readouts = returned[::-1]
    first, second = clone.readouts
    assert (first.position, first.is_flag, first.name) == (0, False, "reject")
    assert (second.position, second.is_flag, second.name) == (1, True, None)
    assert first.equation == returned[1].equation
    assert returned[1].position == 1
    assert original.readouts == returned


def test_readout_roles_follow_the_current_instruction() -> None:
    gadget = _build_repetition3().layers[0].gadgets["measure_z"]
    original = gadget.readouts
    instruction = gadget.implements
    gadget.implements = qodec.Instruction(
        instruction.mnemonic, inputs=instruction.inputs, outputs=instruction.outputs,
    )
    assert original[0].is_flag is False
    assert gadget.readouts[0].is_flag is True
    assert gadget.readouts[0].equation == original[0].equation


def test_parity_getters_are_immutable_snapshots() -> None:
    gadget = _build_repetition3().layers[0].gadgets["measure_z"]
    checks, readouts = gadget.checks, gadget.readouts
    assert isinstance(checks, tuple)
    assert isinstance(checks[0], tuple)
    assert isinstance(readouts, tuple)
    assert isinstance(readouts[0].equation, tuple)
    for value in (checks, checks[0], readouts, readouts[0].equation):
        with pytest.raises(AttributeError):
            getattr(value, "append")("circuit.readouts[0]")
    gadget.checks = (*checks, ("circuit.readouts[2]",))
    gadget.readouts = (("circuit.readouts[2]",),)
    assert len(gadget.checks) == len(checks) + 1
    assert readouts[0].equation != gadget.readouts[0].equation
    assert checks[0][0] == "circuit.readouts[0:2]"


def test_mixed_readout_inputs_are_atomic() -> None:
    gadget = _build_repetition3().layers[0].gadgets["measure_z"]
    original = gadget.readouts[0]
    named: UserDict[str, tuple[str, ...]] = UserDict({"reject": ("circuit.readouts[1:3]",)})
    gadget.readouts = (original, named, ())
    assert gadget.readouts[0] == original
    assert gadget.readouts[1].name == "reject"
    assert gadget.readouts[1].equation == ("circuit.readouts[1:3]",)
    assert str(gadget.readouts[2]) == "[]"
    _assert_invalid_readout_update_is_atomic(gadget, original, named)
    _assert_invalid_check_update_is_atomic(gadget)
    gadget.readouts = ()
    gadget.checks = ()
    assert gadget.readouts == gadget.checks == ()


def _assert_invalid_readout_update_is_atomic(
    gadget: qodec.Gadget, original: qodec.gadgets.Readout, named: UserDict[str, tuple[str, ...]]
) -> None:
    before = gadget.readouts
    with pytest.raises(ValueError, match="invalid"):
        gadget.readouts = (original, named, ("invalid",))
    assert gadget.readouts == before
    with pytest.raises(ValueError, match="invalid"):
        qodec.Gadget(
            gadget.implements, gadget.circuit, inputs=gadget.inputs,
            readouts=(original, named, ("invalid",)),
        )


def _assert_invalid_check_update_is_atomic(gadget: qodec.Gadget) -> None:
    checks = gadget.checks
    with pytest.raises(ValueError, match="invalid"):
        gadget.checks = (*checks, ("invalid",))
    assert gadget.checks == checks


@pytest.mark.parametrize("name", [None, "reject", "quote'\"\\\n", "\U0001f680"])
def test_readout_display_preserves_authored_data(name: str | None) -> None:
    gadget = _build_repetition3().layers[0].gadgets["measure_z"]
    equation = ["circuit.readouts[0,2]", "in[0].z[0]"]
    authored = equation if name is None else {name: equation}
    gadget.readouts = [authored]
    readout = gadget.readouts[0]
    assert ast.literal_eval(str(readout)) == authored
    assert json.loads(str(readout)) == authored
    assert repr(readout) == (
        f"Readout(position=0, name={name!r}, is_flag=False, equation={equation!r})"
    )
    assert repr(readout) in repr(gadget.readouts)
    gadget.readouts = [ast.literal_eval(str(readout))]
    assert gadget.readouts == (readout,)


def test_gadget_readouts_reject_multi_key_named_entry() -> None:
    from qodec.gadgets import Reference

    gadget = _build_repetition3().layers[0].gadgets["measure_z"]
    # A named readout must be a single-key {name: equation} mapping; a
    # multi-key dict is rejected.
    with pytest.raises((TypeError, ValueError)):
        gadget.readouts = [
            {
                "a": [Reference("circuit.readouts[0]")],
                "b": [Reference("circuit.readouts[1]")],
            }
        ]



def _assert_bad_equation_setters_preserve_values(gadget: qodec.Gadget, path: str) -> None:
    checks = gadget.checks
    readouts = gadget.readouts
    with pytest.raises(ValueError) as error:
        gadget.checks = [["circuit.readouts[0]"], [path]]
    assert path in str(error.value)
    assert gadget.checks == checks
    for named in [False, True]:
        with pytest.raises(ValueError) as error:
            if named:
                gadget.readouts = [["circuit.readouts[0]"], {"reject": [path]}]
            else:
                gadget.readouts = [["circuit.readouts[0]"], [path]]
        assert path in str(error.value)
        assert gadget.readouts == readouts


@pytest.mark.parametrize("path", ["checks[0]", "circuit.readouts[3:1]", "readouts[0:0]", "in[0].z[0:2:0]"])
def test_invalid_equations_are_rejected_before_mutation(path: str) -> None:
    gadget = _build_repetition3().layers[0].gadgets["measure_z"]
    _assert_bad_equation_setters_preserve_values(gadget, path)
    with pytest.raises(ValueError) as error:
        qodec.Gadget(implements=gadget.implements, circuit=gadget.circuit, inputs=gadget.inputs, checks=[[path]])
    assert path in str(error.value)
    with pytest.raises(ValueError) as error:
        qodec.Gadget(implements=gadget.implements, circuit=gadget.circuit, inputs=gadget.inputs, readouts=[[path]])
    assert path in str(error.value)


def test_loaded_reference_getters_do_not_reparse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from qodec.gadgets import Reference

    codec = _build_repetition3()
    codec.save(str(tmp_path), single_file=True)
    loaded = qodec.Qodec.load(str(tmp_path / codec.manifest_filename))

    def reject_reference(value: object) -> Reference:
        raise AssertionError(f"getter reparsed {value}")

    monkeypatch.setattr(qodec.gadgets, "Reference", reject_reference)
    gadget = loaded.layers[0].gadgets["measure_z"]
    for _ in range(2):
        reference = gadget.checks[0][0]
        assert isinstance(reference, Reference)
        assert reference.path == "circuit.readouts[0:2]"
        assert [term.index for term in reference.expand()] == [0, 1]
        logical = gadget.readouts[0].equation[1]
        assert isinstance(logical, Reference)
        assert logical.encoding_property == "z"
    gadget.checks = [[term for term in equation] for equation in gadget.checks]
    assert gadget.checks[0][0] == reference


def test_save_load_roundtrip(tmp_path: Path) -> None:
    codec = _build_repetition3()

    directory = tmp_path / "repetition3"
    codec.save(str(directory))
    reloaded = qodec.Qodec.load(str(directory / codec.manifest_filename))
    assert reloaded.name == "repetition3"
    assert [layer.instruction_set.name for layer in reloaded.layers] == ["repetition3", "stim"]
    assert set(reloaded.layers[0].gadgets) == {"prepare_z", "measure_z"}


def test_save_load_single_file_roundtrip(tmp_path: Path) -> None:
    codec = _build_repetition3()

    bundle = tmp_path / "repetition3"
    codec.save(str(bundle), single_file=True)
    reloaded = qodec.Qodec.load(str(bundle / codec.manifest_filename))
    assert reloaded.name == "repetition3"
    assert set(reloaded.layers[0].gadgets) == {"prepare_z", "measure_z"}


def _preparation_instruction_set() -> InstructionSet:
    return InstructionSet(
        name="repetition3",
        description="Logical ISA for the 3-qubit bit-flip repetition code.",
        blocks=[qodec.instructions.Block("repetition3", encodes=1)],
        instructions=[
            Instruction(
                mnemonic="prepare_z",
                outputs=[BlockOperand("repetition3")],
                action=[Stabilize(["Z_0"])],
            ),
        ],
    )


def test_instruction_set_save_load_roundtrip(tmp_path: Path) -> None:
    instruction_set = _preparation_instruction_set()
    path = tmp_path / "repetition3.isa.yaml"
    instruction_set.save(str(path))
    reloaded = InstructionSet.load(str(path))

    assert reloaded.name == instruction_set.name
    assert reloaded.description == instruction_set.description
    assert [block.name for block in reloaded.blocks] == ["repetition3"]
    assert list(reloaded.instructions) == ["prepare_z"]
    assert reloaded.instructions["prepare_z"].mnemonic == "prepare_z"


def test_instruction_set_instructions_accepts_dict_and_rejects_duplicates() -> None:
    prepare = Instruction(
        mnemonic="prepare_z",
        outputs=[BlockOperand("repetition3")],
        action=[Stabilize(["Z_0"])],
    )

    blocks = [qodec.instructions.Block("repetition3", encodes=1)]
    instruction_set = InstructionSet(name="repetition3", blocks=blocks, instructions={"prepare_z": prepare})
    assert set(instruction_set.instructions) == {"prepare_z"}
    assert instruction_set.instructions["prepare_z"].mnemonic == "prepare_z"

    instruction_set.instructions = [prepare]
    assert instruction_set.instructions["prepare_z"] == prepare

    with pytest.raises(ValueError, match=r'duplicate instruction "prepare_z"'):
        InstructionSet(name="repetition3", blocks=blocks, instructions=[prepare, prepare])



@pytest.mark.parametrize("collection", ["instructions", "gadgets"])
@pytest.mark.parametrize("invalid_shape", [False, True])
def test_rejected_collection_updates_preserve_the_protocol(collection: str, invalid_shape: bool) -> None:
    protocol = qodec.Qodec.loads(_build_repetition3().dumps())
    before = qodec.Qodec.loads(protocol.dumps())
    assert protocol == before
    layer = protocol.layers[0]
    owner = layer.instruction_set if collection == "instructions" else layer
    first = next(iter(getattr(owner, collection).values()))
    replacement = 42 if invalid_shape else [first, first]
    expected = "must be a list" if invalid_shape else "duplicate"
    with pytest.raises(ValueError, match=expected):
        setattr(owner, collection, replacement)
    assert protocol == before
    assert qodec.Qodec.loads(protocol.dumps()) == before


@pytest.mark.parametrize("version", [0, 2])
def test_invalid_schema_version_does_not_replace_the_current_version(version: int) -> None:
    protocol = qodec.Qodec.loads(_build_repetition3().dumps())
    protocol.schema_version = 1
    before = qodec.Qodec.loads(protocol.dumps())
    assert protocol == before
    with pytest.raises(ValueError, match="schema_version must be 1 or None"):
        protocol.schema_version = version
    assert protocol == before
    with pytest.raises(ValueError, match="schema_version must be 1 or None"):
        qodec.Qodec(protocol.layers, schema_version=version)
    protocol.schema_version = None
    assert qodec.Qodec.loads(protocol.dumps()).schema_version is None


@pytest.mark.parametrize(("mnemonic", "boundary"), [("prepare_z", "output"), ("measure_z", "input")])
def test_gadget_constructor_requires_matching_encoding_counts(mnemonic: str, boundary: str) -> None:
    gadget = _build_repetition3().layers[0].gadgets[mnemonic]
    with pytest.raises(ValueError, match=f"declares 1 {boundary} operand.*provides 0 {boundary} encoding"):
        qodec.Gadget(gadget.implements, gadget.circuit)


def test_code_save_load_roundtrip(tmp_path: Path) -> None:
    code = Code(
        name="repetition3",
        stabilizers=["Z_0 Z_1", "Z_1 Z_2"],
        x=["X_0 X_1 X_2"],
        z=["Z_0"],
    )

    path = tmp_path / "repetition3.code.yaml"
    code.save(str(path))
    reloaded = Code.load(str(path))

    assert reloaded.name == code.name
    assert reloaded.stabilizers == code.stabilizers
    assert reloaded.x == code.x
    assert reloaded.z == code.z


def _replace_protocol_components(protocol: qodec.Qodec) -> None:
    protocol.description = "Edited protocol"
    protocol.manifest_filename = "nested/protocol.yaml"
    protocol.layers[0].instruction_set.description = "Edited logical instructions"
    code = _repetition3_code()
    code.description = "Replacement code"
    protocol.layers[0].codes = {block: code for block in protocol.layers[0].codes}
    for gadget in protocol.layers[0].gadgets.values():
        for boundary in ("inputs", "outputs"):
            encodings = getattr(gadget, boundary)
            for encoding in encodings:
                encoding.code = code
            setattr(gadget, boundary, encodings)
    measure = protocol.layers[0].gadgets["measure_z"]
    measure.circuit = qodec.gadgets.Circuit(
        protocol.layers[1].instruction_set, "# replacement circuit\nM 0 1 2\n", format="stim"
    )


@pytest.mark.parametrize("single_file", [False, True])
def test_replaced_components_survive_each_save_layout(single_file: bool, tmp_path: Path) -> None:
    protocol = qodec.Qodec.loads(_build_repetition3().dumps())
    _replace_protocol_components(protocol)
    protocol.save(tmp_path, single_file=single_file)
    restored = qodec.Qodec.load(tmp_path / "nested/protocol.yaml")
    assert restored == protocol
    assert restored.description == "Edited protocol"
    assert restored.codes["repetition3"].description == "Replacement code"
    assert restored.layers[0].instruction_set.description == "Edited logical instructions"
    assert restored.layers[0].gadgets["measure_z"].circuit.format == "stim"
    assert restored.layers[0].gadgets["measure_z"].circuit.source == "# replacement circuit\nM 0 1 2\n"


@pytest.mark.parametrize("single_file", [False, True])
def test_failed_save_keeps_memory_and_can_be_retried(single_file: bool, tmp_path: Path) -> None:
    protocol = qodec.Qodec.loads(_build_repetition3().dumps())
    before = qodec.Qodec.loads(protocol.dumps())
    blocked = tmp_path / protocol.manifest_filename
    blocked.mkdir()
    sentinel = tmp_path / "unrelated.txt"
    sentinel.write_text("keep this file")
    with pytest.raises(qodec.QodecSaveError):
        protocol.save(tmp_path, single_file=single_file)
    assert protocol == before
    assert sentinel.read_text() == "keep this file"
    blocked.rmdir()
    protocol.save(tmp_path, single_file=single_file)
    assert qodec.Qodec.load(blocked) == before
    assert sentinel.read_text() == "keep this file"


@pytest.mark.parametrize("single_file", [False, True])
def test_invalid_model_does_not_overwrite_existing_output(single_file: bool, tmp_path: Path) -> None:
    protocol = _build_repetition3()
    manifest = tmp_path / protocol.manifest_filename
    manifest.write_text("prior content")
    protocol.codes["repetition3"].stabilizers = ["Z_bad"]
    with pytest.raises(qodec.QodecSaveError, match="invalid qubit index"):
        protocol.save(tmp_path, single_file=single_file)
    assert manifest.read_text() == "prior content"
    assert sorted(path.name for path in tmp_path.iterdir()) == [protocol.manifest_filename]
    assert protocol.codes["repetition3"].stabilizers == ["Z_bad"]


@pytest.mark.parametrize("kind", ["code", "instruction_set"])
def test_standalone_save_reports_io_failure_and_preserves_values(kind: str, tmp_path: Path) -> None:
    artifact = _repetition3_code() if kind == "code" else _preparation_instruction_set()
    original = str(artifact)
    blocker = tmp_path / "file"
    blocker.write_text("unchanged")
    with pytest.raises(qodec.QodecSaveError, match="file"):
        artifact.save(blocker / "artifact.yaml")
    with pytest.raises(qodec.QodecSaveError):
        artifact.save(tmp_path)
    assert str(artifact) == original
    assert blocker.read_text() == "unchanged"
    artifact.save(tmp_path / "artifact.yaml")
    assert type(artifact).load(tmp_path / "artifact.yaml") == artifact


def test_instruction_set_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(qodec.QodecLoadError):
        InstructionSet.load(str(tmp_path / "does_not_exist.isa.yaml"))


def test_instruction_value_equality() -> None:
    def make() -> Instruction:
        return Instruction(
            mnemonic="prepare_z",
            outputs=[BlockOperand("repetition3")],
            action=[Stabilize(["Z_0"])],
        )

    assert make() == make()
    assert make() != Instruction(
        mnemonic="measure_z",
        inputs=[BlockOperand("repetition3")],
        action=[Observe(["Z_0"])],
    )
    assert make() != object()


def _assert_simple_value_equality() -> None:
    assert Code(name="c", stabilizers=["Z_0 Z_1"], x=["X_0 X_1"], z=["Z_0"]) == Code(
        name="c", stabilizers=["Z_0 Z_1"], x=["X_0 X_1"], z=["Z_0"]
    )
    assert Code(name="c", stabilizers=["Z_0 Z_1"], x=["X_0 X_1"], z=["Z_0"]) != Code(
        name="d", stabilizers=["Z_0 Z_1"], x=["X_0 X_1"], z=["Z_0"]
    )
    assert BlockOperand("q") == BlockOperand("q")
    assert BlockOperand("q") != BlockOperand("r")
    assert Parameter("theta", "number") == Parameter("theta", "number")
    assert Parameter("theta", "number") != Parameter("theta", Parameter.Kind.PAULI)
    assert qodec.instructions.Block("b", encodes=2) == qodec.instructions.Block("b", encodes=2)
    assert qodec.instructions.Block("b", encodes=2) != qodec.instructions.Block("b", encodes=3)

    # Actions compare by content too.
    assert Stabilize(["Z_0"]) == Stabilize(["Z_0"])
    assert Stabilize(["Z_0"]) != Stabilize(["X_0"])
    assert Observe(["Z_0"]) == Observe(["Z_0"])
    assert Observe(["Z_0"]) != Observe(["Z_1"])


def _assert_equal_protocol_and_components(left: qodec.Qodec, right: qodec.Qodec) -> None:
    assert left == right
    assert left != object()

    # The same holds for the composite sub-objects.
    left_measure = left.layers[0].gadgets["measure_z"]
    right_measure = right.layers[0].gadgets["measure_z"]
    assert left.layers[0] == right.layers[0]
    assert left.layers[0].instruction_set == right.layers[0].instruction_set
    assert left_measure == right_measure
    assert left_measure.circuit == right_measure.circuit
    assert left_measure.inputs[0] == right_measure.inputs[0]


def test_structural_equality() -> None:
    _assert_simple_value_equality()
    left, right = _build_repetition3(), _build_repetition3()
    _assert_equal_protocol_and_components(left, right)

    mutated = _build_repetition3()
    mutated.layers[0].gadgets["measure_z"].readouts = [["circuit.readouts[1]", "in[0].z[0]"]]
    assert mutated != left
    assert mutated.layers[0] != left.layers[0]
    assert mutated.layers[0].gadgets["measure_z"] != left.layers[0].gadgets["measure_z"]

    renamed = _build_repetition3()
    renamed.name = "different"
    assert renamed != left


def test_parameter_kind_enum() -> None:
    # The getter returns a Parameter.Kind enum member.
    parameter = Parameter("theta", "number")
    assert parameter.kind is Parameter.Kind.NUMBER
    assert parameter.kind.value == "number"

    # The constructor accepts both the string token and the enum member,
    # and they produce equal parameters.
    from_string = Parameter("theta", "pauli")
    from_enum = Parameter("theta", Parameter.Kind.PAULI)
    assert from_string.kind is Parameter.Kind.PAULI
    assert from_enum.kind is Parameter.Kind.PAULI

    # An unknown token is rejected.
    with pytest.raises(ValueError, match=r'unknown kind "complex"'):
        Parameter("theta", "complex")


def test_metadata_defaults_to_empty_dict() -> None:
    # Every definition exposes metadata; it defaults to an empty dict.
    assert Code(name="c", stabilizers=["Z_0 Z_1"], x=["X_0 X_1"], z=["Z_0"]).metadata == {}
    assert InstructionSet(name="isa").metadata == {}
    assert Instruction(mnemonic="noop").metadata == {}


def test_metadata_constructor_and_getter_roundtrip() -> None:
    # An arbitrary nested mapping passed to the constructor reads back verbatim.
    payload = {"duration_ns": 120, "tags": ["calibrated"], "vendor": {"ibm": {"native": True}}}
    instruction = Instruction(mnemonic="rotate_z", metadata=payload)
    assert instruction.metadata == payload

    code = Code(name="c", stabilizers=["Z_0 Z_1"], x=["X_0 X_1"], z=["Z_0"], metadata={"provenance": "arXiv:x"})
    assert code.metadata == {"provenance": "arXiv:x"}


def test_metadata_setter_on_mutable_types() -> None:
    code = Code(name="c", stabilizers=["Z_0 Z_1"], x=["X_0 X_1"], z=["Z_0"])
    code.metadata = {"a": 1}
    assert code.metadata == {"a": 1}

    instruction_set = InstructionSet(name="isa")
    instruction_set.metadata = {"backend": "ion-trap"}
    assert instruction_set.metadata == {"backend": "ion-trap"}

    codec = _build_repetition3()
    codec.metadata = {"authors": ["A. Researcher"]}
    assert codec.metadata == {"authors": ["A. Researcher"]}

    gadget = codec.layers[0].gadgets["prepare_z"]
    gadget.metadata = {"duration_ns": 800}
    assert gadget.metadata == {"duration_ns": 800}


def test_instruction_metadata_is_construct_only() -> None:
    # Instruction is an immutable value object: metadata has no setter.
    instruction = Instruction(mnemonic="noop", metadata={"a": 1})
    assert instruction.metadata == {"a": 1}
    with pytest.raises(AttributeError):
        instruction.metadata = {"b": 2}  # type: ignore[misc]


def test_metadata_rejects_non_mapping() -> None:
    code = Code(name="c", stabilizers=["Z_0 Z_1"], x=["X_0 X_1"], z=["Z_0"])
    # The object-only rule is enforced at the boundary: a scalar is rejected.
    with pytest.raises((TypeError, ValueError)):
        code.metadata = 5  # type: ignore[assignment]
    with pytest.raises((TypeError, ValueError)):
        Instruction(mnemonic="noop", metadata=[1, 2, 3])  # type: ignore[arg-type]


def test_metadata_participates_in_equality() -> None:
    def make(metadata: dict[str, object] | None = None) -> Code:
        return Code(name="c", stabilizers=["Z_0 Z_1"], x=["X_0 X_1"], z=["Z_0"], metadata=metadata)

    assert make({"a": 1}) == make({"a": 1})
    assert make({"a": 1}) != make({"a": 2})
    assert make({"a": 1}) != make()


def _code_with_metadata() -> Code:
    return Code(
        name="rep",
        stabilizers=["Z_0 Z_1"],
        x=["X_0 X_1"],
        z=["Z_0"],
        metadata={"provenance": "arXiv:x", "tags": ["css"]},
    )


def test_metadata_roundtrips_through_standalone_save_load(tmp_path: Path) -> None:
    code = _code_with_metadata()
    code_path = tmp_path / "rep.code.yaml"
    code.save(str(code_path))
    assert Code.load(str(code_path)).metadata == {"provenance": "arXiv:x", "tags": ["css"]}

    instruction_set = InstructionSet(
        name="instruction_set",
        metadata={"backend": "ion-trap"},
        instructions=[Instruction(mnemonic="noop", metadata={"duration_ns": 7})],
    )
    isa_path = tmp_path / "rep.isa.yaml"
    instruction_set.save(str(isa_path))
    reloaded = InstructionSet.load(str(isa_path))
    assert reloaded.metadata == {"backend": "ion-trap"}
    assert reloaded.instructions["noop"].metadata == {"duration_ns": 7}


def test_metadata_roundtrips_through_qodec_save_load(tmp_path: Path) -> None:
    codec = _build_repetition3()
    codec.metadata = {"authors": ["A. Researcher"]}
    codec.layers[0].instruction_set.metadata = {"backend": "ion-trap"}
    codec.layers[0].gadgets["measure_z"].inputs[0].code.metadata = {"provenance": "arXiv:x"}
    codec.layers[0].gadgets["prepare_z"].metadata = {"duration_ns": 800}

    directory = tmp_path / "repetition3"
    codec.save(str(directory))
    reloaded = qodec.Qodec.load(str(directory / codec.manifest_filename))

    assert reloaded.metadata == {"authors": ["A. Researcher"]}
    assert reloaded.layers[0].instruction_set.metadata == {"backend": "ion-trap"}
    assert reloaded.layers[0].gadgets["measure_z"].inputs[0].code.metadata == {"provenance": "arXiv:x"}
    assert reloaded.layers[0].gadgets["prepare_z"].metadata == {"duration_ns": 800}



def _repetition3_code() -> Code:
    return Code(name="repetition3", stabilizers=["Z_0 Z_1", "Z_1 Z_2"], x=["X_0 X_1 X_2"], z=["Z_0"])


def test_code_rejects_an_illegal_pauli_at_construction() -> None:
    # `Code`'s checks are self-contained, so the earliest point is the constructor.
    with pytest.raises(ValueError, match=r"invalid qubit index"):
        Code(name="bad", stabilizers=["Z_1 Z_O"], x=["X_0"], z=["Z_0"])


def test_save_rejects_what_load_would_reject(tmp_path: Path) -> None:
    # The setters accept anything, so `save` re-checks: writing an artifact that
    # cannot be read back loses the work where it is least recoverable.
    code = _repetition3_code()
    code.stabilizers = ["Z_1 Z_O"]
    with pytest.raises(qodec.QodecSaveError, match=r"invalid qubit index"):
        code.save(tmp_path / "bad.code.yaml")

    instruction_set = InstructionSet(
        name="i",
        blocks=[qodec.instructions.Block("b", encodes=1)],
        instructions=[Instruction(mnemonic="p", outputs=[BlockOperand("b")], action=[Stabilize(["Z_0"])])],
    )
    instruction_set.blocks = []
    instruction_set.save(tmp_path / "draft.isa.yaml")
    assert InstructionSet.load(tmp_path / "draft.isa.yaml").instructions == instruction_set.instructions


def test_qodec_save_rejects_a_mutated_illegal_code(tmp_path: Path) -> None:
    codec = qodec.Qodec.load(Path(__file__).parents[3] / "examples" / "repetition3" / "repetition3.qodec.yaml")
    codec.layers[0].gadgets["idle"].inputs[0].code.stabilizers = ["Z_1 Z_O"]
    with pytest.raises(qodec.QodecSaveError, match=r"code 'repetition3'"):
        codec.save(tmp_path / "out")


def test_instruction_set_allows_incomplete_declarations() -> None:
    empty = InstructionSet(name="i")
    assert empty.blocks == [] and dict(empty.instructions) == {}
    empty.blocks = [qodec.instructions.Block("b", encodes=1)]
    assert [block.name for block in empty.blocks] == ["b"]

    draft = InstructionSet(
        name="i",
        instructions=[Instruction(mnemonic="p", outputs=[BlockOperand("b")], action=[Stabilize(["Z_0"])])],
    )
    assert draft.instructions["p"].outputs == [BlockOperand("b")]
