from __future__ import annotations

import json
from pathlib import Path

import pytest

import qodec as qc
from qodec.actions import Condition, Pauli, Rotate
from qodec import Reference
from qodec.gadgets import Circuit, Encoding, Readout
from qodec.instructions import Block, BlockOperand, Parameter


def model() -> qc.Qodec:
    code = qc.Code("qubit", [], ["X_0"], ["Z_0"])
    instruction = qc.Instruction("idle", inputs=[BlockOperand("qubit")], outputs=[BlockOperand("qubit")],
        parameters=[Parameter("enabled", "bit")], action=[Pauli("X_0", condition=Condition(["enabled"])), Rotate("Z_0", 0.5)])
    instruction_set = qc.InstructionSet("test", blocks=[Block("qubit", encodes=1)], instructions=[instruction])
    gadget = qc.Gadget(instruction, Circuit(instruction_set, "unparseable source"), inputs=[Encoding(code)], outputs=[Encoding(code)],
        checks=[["in[0].z[0]"]], metadata={"values": [True, 1, 1.5, None, "text"]})
    return qc.Qodec([qc.Layer(instruction_set, gadgets=[gadget]), qc.Layer(instruction_set)])


def test_reference_lookup_uses_the_same_addresses_at_every_root(monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = model()
    node = protocol.resolve('layers[0].gadgets["idle"]')
    gadget = node.value(qc.Gadget)
    reference = Reference("in[00].z[0]")

    def forbidden_string_conversion(self: Reference) -> str:
        raise AssertionError("lookup converted a parsed Reference back to text")

    monkeypatch.setattr(Reference, "__str__", forbidden_string_conversion)
    selected = node.resolve(reference)
    assert selected.value(str) == "Z_0"
    assert selected == node.resolve("in[0].z[0]")
    assert selected == protocol.resolve(Reference('layers[0].gadgets["idle"].in[0].z[0]'))
    standalone = gadget.resolve(reference)
    assert standalone.value(str) == "Z_0"
    assert standalone.path == "in[0].z[0]"
    assert standalone != selected
    assert standalone.source_location is None
    assert gadget.resolve("").value(qc.Gadget) is gadget
    for path in ["inputs", "outputs", "implements.inputs", "implements.outputs", "circuit.readouts[0]"]:
        with pytest.raises(LookupError):
            gadget.resolve(path)


def test_direct_encoding_operators_retain_loaded_source_locations() -> None:
    protocol = qc.Qodec.load(Path(__file__).parents[3] / "examples/repetition3/repetition3.qodec.yaml")
    gadget = protocol.resolve('layers[0].gadgets["measure_z"]')
    direct = gadget.resolve(Reference("in[0].stabilizers[0]")).source_location
    through_code = gadget.resolve("in[0].code.stabilizers[0]").source_location
    assert direct is not None and through_code is not None
    assert (direct.path, direct.line) == (through_code.path, through_code.line)


def test_standalone_nodes_follow_mutation_and_distinguish_owners() -> None:
    from copy import copy

    gadget = model().layers[0].gadgets["idle"]
    node = gadget.resolve("in[0].z[0]")
    original_hash = hash(node)
    assert node == gadget.resolve(Reference("in[0].z[00]"))
    assert node != copy(gadget).resolve("in[0].z[0]")
    gadget.inputs[0].code.z = ["Z_1"]
    assert node.value(str) == "Z_1" and hash(node) == original_hash
    gadget.inputs.clear()
    with pytest.raises(LookupError):
        node.value(str)


def test_selection_nodes_preserve_order_duplicates_and_live_member_paths() -> None:
    protocol = qc.Qodec([], metadata={"values": [10, 20, 30]})
    values = protocol.resolve('metadata["values"]')
    selected = values.resolve(Reference("[2, 0,2]"))
    members = selected.as_sequence()
    assert tuple(member.value(int) for member in selected.as_sequence()) == (30, 10, 30)
    assert members == (values.resolve("[2]"), values.resolve("[0]"), values.resolve("[2]"))
    assert selected.resolve("[1]") == members[1]
    assert tuple(member.value(int) for member in values.resolve("[0:3:2]").as_sequence()) == (10, 30)
    assert values.resolve("[0:1]").as_sequence() == (values.resolve("[0]"),)
    assert tuple(member.value(int) for member in selected.resolve("[1,0]").as_sequence()) == (10, 30)
    assert selected.resolve("[1,0]").as_sequence() == (members[1], members[0])
    assert not selected.is_none
    with pytest.raises(TypeError):
        selected.as_mapping()
    protocol.metadata["values"][2] = 40
    assert tuple(member.value(int) for member in selected.as_sequence()) == (40, 10, 40)
    del protocol.metadata["values"][2]
    with pytest.raises(LookupError):
        selected.as_sequence()
    assert members[1].value(int) == 10

def test_large_union_children_have_actual_paths() -> None:
    protocol = qc.Qodec([], metadata={"values": [10]})
    selection = protocol.resolve('metadata["values"][' + ",".join(["0"] * 16384) + "]")
    members = selection.as_sequence()
    assert len(members) == 16384
    assert all(member == protocol.resolve('metadata["values"][0]') for member in members)


@pytest.mark.parametrize("suffix", ["[0,9]", "[0:4]", "[0,9][0]", "[0:2][9]", "[0:2].name"])
def test_selection_lookup_never_returns_partial_results(suffix: str) -> None:
    protocol = qc.Qodec([], metadata={"values": [10, 20, 30]})
    with pytest.raises(LookupError):
        protocol.resolve('metadata["values"]' + suffix)


def test_general_references_are_addresses_not_parity_terms() -> None:
    gadget = model().layers[0].gadgets["idle"]
    reference = Reference('metadata["values"][1]')
    assert gadget.resolve(reference).value(int) == 1
    field = reference.segments[0]
    assert isinstance(field, Reference.Field)
    assert gadget.resolve(field.name).as_mapping()
    assert Reference("name").expand() == [Reference("name")]
    values: list[Reference | str] = [reference, reference.path]
    for value in values:
        with pytest.raises(ValueError, match="not a parity reference"):
            gadget.checks = [[value]]
        with pytest.raises(ValueError, match="not a parity reference"):
            gadget.readouts = [[value]]
        with pytest.raises(ValueError, match="not a parity reference"):
            gadget.frames = {"out[0].z[0]": [value]}
    with pytest.raises(ValueError, match="not a parity reference"):
        gadget.frames = {reference.path: []}
    assert gadget.checks == [(Reference("in[0].z[0]"),)]


def test_node_types_and_relative_navigation() -> None:
    protocol = model()
    root = protocol.resolve("")
    assert root.value(qc.Qodec) is protocol
    layer = root.resolve("layers[0]")
    assert layer.value(qc.Layer) is protocol.layers[0]
    gadget = layer.resolve('gadgets["idle"]')
    assert gadget.value(qc.Gadget) is protocol.layers[0].gadgets["idle"]
    assert gadget.resolve("circuit").value(Circuit) is gadget.value(qc.Gadget).circuit
    assert gadget.resolve("in[0]").value(Encoding).code is gadget.resolve("in[0].code").value(qc.Code)
    assert gadget.resolve("implements").value(qc.Instruction) == protocol.layers[0].instruction_set.instructions["idle"]
    assert layer.resolve("instruction_set").value(qc.InstructionSet) is protocol.layers[0].instruction_set
    assert layer.resolve("instruction_set.blocks[0]").value(Block).name == "qubit"
    _assert_gadget_component_views(gadget)
    with pytest.raises(LookupError):
        gadget.resolve("circuit.calls")


def _assert_gadget_component_views(gadget: qc.Node) -> None:
    assert gadget.resolve("implements.in[0]").value(BlockOperand).block == "qubit"
    assert gadget.resolve("implements.parameters[0]").value(Parameter).name == "enabled"
    assert gadget.resolve("implements.parameters[0].kind").value(Parameter.Kind) is Parameter.Kind.BIT
    assert isinstance(gadget.resolve("implements.action[0]").as_action(), Pauli)
    assert gadget.resolve("implements.action[0].condition").value(Condition).predicates == ("enabled",)
    assert gadget.resolve("checks[0][0]").value(Reference) == Reference("in[0].z[0]")
    assert gadget.resolve("checks[0][0].path").value(str) == "in[0].z[0]"


def test_exact_scalar_types_and_null() -> None:
    values = model().resolve('layers[0].gadgets["idle"].metadata["values"]').as_sequence()
    assert values[0].value(bool) is True
    assert values[1].value(int) == 1
    assert values[2].value(float) == 1.5
    assert values[3].is_none
    assert values[4].value(str) == "text"
    for index, expected in [(0, int), (1, bool), (1, float), (4, qc.Gadget)]:
        with pytest.raises(TypeError):
            values[index].value(expected)
    with pytest.raises(TypeError, match=r"value\(bool\)"):
        bool(values[0])

def test_frame_navigation_distinguishes_constants_from_references() -> None:
    protocol = model()
    protocol.layers[0].gadgets["idle"].frames = {"out[0].z[0]": ["in[0].z[0]", 1]}
    frame = protocol.resolve('layers[0].gadgets["idle"].frames["out[0].z[0]"]')
    assert frame.resolve("[0]").value(Reference) == Reference("in[0].z[0]")
    assert frame.resolve("[1]").value(int) == 1
    with pytest.raises(TypeError):
        frame.resolve("[1]").value(Reference)


def test_live_handles_equality_and_missing_repr() -> None:
    protocol = model()
    first = protocol.resolve("layers[00]")
    name = first.resolve("instruction_set.name")
    second = protocol.resolve("layers[0]")
    assert first == second and hash(first) == hash(second)
    assert first != model().resolve("layers[0]")
    assert str(first) == "layers[0]"
    assert "Layer" in repr(first)
    original_hash = hash(first)
    replacement = qc.Layer(qc.InstructionSet("replacement"))
    protocol.layers = [replacement]
    assert first.value(qc.Layer) is replacement
    assert name.value(str) == "replacement"
    assert hash(first) == original_hash
    protocol.layers = []
    assert first == second
    assert "missing" in repr(first)
    with pytest.raises(LookupError):
        first.value(qc.Layer)
    assert first.__eq__(None) is NotImplemented
    assert first.__ne__(None) is NotImplemented


def test_collection_children_remain_live_after_replacement() -> None:
    protocol = qc.Qodec([], metadata={"items": [1, 2], 'a.b["c"]': "original"})
    entries = protocol.resolve("metadata").as_mapping()
    items = entries["items"].as_sequence()
    assert entries['a.b["c"]'] == protocol.resolve('metadata["a.b[\\"c\\"]"]')
    protocol.metadata = {"items": [3], 'a.b["c"]': "replacement"}
    assert items[0].value(int) == 3
    assert entries['a.b["c"]'].value(str) == "replacement"
    with pytest.raises(LookupError, match="model path does not exist"):
        items[1].value(int)
    protocol.metadata = {"items": None}
    assert entries["items"].is_none
    with pytest.raises(LookupError):
        items[0].value(int)
    with pytest.raises(LookupError):
        entries['a.b["c"]'].value(str)


def test_collections_are_explicit_and_surface_is_pinned() -> None:
    node = model().resolve("layers")
    for name in ("__getitem__", "__len__", "__iter__"):
        assert not hasattr(qc.Node, name)
    names = {"path", "source_location", "is_none", "resolve", "value", "as_action", "as_sequence", "as_mapping"}
    assert len(names) == 8
    assert {name for name in dir(qc.Node) if not name.startswith("_")} == names
    assert {name for name in dir(qc.SourceLocation) if not name.startswith("_")} == {"path", "line"}
    with pytest.raises(TypeError):
        node.as_mapping()
    mapping = model().resolve("layers[0].gadgets").as_mapping()
    assert set(mapping) == {"idle"}
    assert mapping["idle"].path == 'layers[0].gadgets["idle"]'
    for constructor in (qc.Node, qc.SourceLocation):
        with pytest.raises(TypeError):
            constructor()


@pytest.mark.parametrize("path", ["layers[-1]", "layers[*]", "layers[0:0]", ".layers", "layers[", "layers.__class__()"])
def test_invalid_path_syntax(path: str) -> None:
    with pytest.raises(ValueError):
        model().resolve(path)


def test_quoted_keys_are_literal() -> None:
    protocol = qc.Qodec([], metadata={'a.b["c"]': "value"})
    assert protocol.resolve('metadata["a.b[\\"c\\"]"]').value(str) == "value"
    with pytest.raises(LookupError):
        protocol.resolve("metadata.missing")


def test_observe_has_an_absent_model_condition() -> None:
    instruction = qc.Instruction("measure", action=[qc.actions.Observe(["Z_0"])])
    protocol = qc.Qodec([qc.Layer(qc.InstructionSet("logical", instructions=[instruction]))])
    node = protocol.resolve('layers[0].instruction_set.instructions["measure"].action[0]')
    assert node.resolve("condition").is_none
    assert node.resolve("observables[0]").value(str) == "Z_0"


@pytest.mark.parametrize("guard, invert", [("if", False), ("unless", True)])
def test_conditional_observe_scalar_navigation(guard: str, invert: bool) -> None:
    target = {"name": "logical", "blocks": {}, "instructions": [{
        "mnemonic": "M", "description": "", "flags": ["reject"],
        "action": [{"observe": "Z_0", guard: ["reject"]}],
    }]}
    text = 'qodec.yaml: {layers: [{instruction_set: target}]}\n---\n' + json.dumps({"target": target})
    protocol = qc.Qodec.loads(text)
    path = 'layers[0].instruction_set.instructions["M"].action[0]'
    action = protocol.resolve(path)
    actions = protocol.resolve(path.rsplit("[", 1)[0]).as_sequence()
    assert actions == (action,)
    assert action.resolve("condition").value(Condition).invert is invert
    assert action.resolve("condition.predicates").as_sequence()[0].value(str) == "reject"
    assert action.resolve("observables").as_sequence()[0].value(str) == "Z_0"
    assert protocol.resolve(path + ".condition.invert").value(bool) is invert
    assert action.resolve("condition.invert").value(bool) is invert
    assert action.resolve("condition.predicates[0]").value(str) == "reject"
    assert action.resolve("observables[0]").value(str) == "Z_0"
    assert not action.resolve("condition").is_none
    with pytest.raises(TypeError):
        action.resolve("condition.invert").value(int)
    with pytest.raises(ValueError, match="conditional observe"):
        action.as_action()
    with pytest.raises(ValueError, match="conditional observe"):
        action.value(qc.actions.Observe)


def test_lookup_does_not_project_unselected_actions() -> None:
    target = {"name": "logical", "blocks": {}, "instructions": [{
        "mnemonic": "M", "description": "", "flags": ["reject"],
        "action": [{"observe": "Z_0", "if": ["reject"]}, {"pauli": "X_0"}],
    }]}
    protocol = qc.Qodec.loads('entry: {layers: [{instruction_set: target}]}\n---\n' + json.dumps({"target": target}))
    action = protocol.resolve('layers[0].instruction_set.instructions["M"].action[1]')
    assert action.as_action() == Pauli("X_0")
    assert action.resolve("operator").value(str) == "X_0"


@pytest.mark.parametrize("path", ["layers[0].missing.name", 'metadata["missing"].name'])
def test_missing_paths_report_the_first_failing_prefix(path: str) -> None:
    with pytest.raises(LookupError) as error:
        model().resolve(path)
    assert str(error.value) == f'model path does not exist: {json.dumps(path.rsplit(".", 1)[0])}'


def test_readout_lookup_does_not_call_the_whole_list_getter(monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = model()
    gadget = protocol.layers[0].gadgets["idle"]
    gadget.readouts = [{"first": []}, {"large": [0] * 100_000}]

    def forbidden_getter(self: qc.Gadget) -> None:
        raise AssertionError("projected the whole readout list")

    monkeypatch.setattr(qc.Gadget, "readouts", property(forbidden_getter))
    readouts = protocol.resolve('layers[0].gadgets["idle"].readouts')
    assert len(readouts.as_sequence()) == 2
    assert readouts.resolve("[0].position").value(int) == 0
    assert readouts.resolve("[0].equation").as_sequence() == ()
    assert readouts.resolve("[0]").value(Readout).name == "first"


def test_collections_use_explicit_node_accessors() -> None:
    protocol = model()
    gadget = protocol.resolve('layers[0].gadgets["idle"]')
    sequences = [protocol.resolve("layers"), protocol.resolve("layers[0:1]"),
        gadget.resolve("checks"), gadget.resolve("checks[0]"), gadget.resolve("in"),
        gadget.resolve("metadata[\"values\"]"), gadget.resolve("implements.action")]
    mappings = [protocol.resolve("instruction_sets"), protocol.resolve("layers[0].gadgets"),
        gadget.resolve("metadata"), gadget.resolve("frames"), gadget.resolve("parameter_bindings")]
    for node in sequences + mappings:
        accessor = "as_sequence" if node in sequences else "as_mapping"
        for expected in [list, tuple, dict, object]:
            with pytest.raises(TypeError, match=accessor):
                node.value(expected)
    assert gadget.resolve("in[0]").value(Encoding) is protocol.layers[0].gadgets["idle"].inputs[0]
    predicates = gadget.resolve("implements.action[0].condition.predicates").as_sequence()
    assert tuple(node.value(str) for node in predicates) == ("enabled",)


def test_parameter_binding_nodes_support_mapping_lookup_and_live_updates() -> None:
    protocol = model()
    gadget = protocol.layers[0].gadgets["idle"]
    gadget.parameter_bindings = {"enabled": "bit"}
    for root in (gadget.resolve(""), protocol.resolve('layers[0].gadgets["idle"]')):
        mapping = root.resolve("parameter_bindings")
        entries = mapping.as_mapping()
        assert set(entries) == {"enabled"}
        assert entries["enabled"] == mapping.resolve('["enabled"]')
        assert entries["enabled"].value(str) == "bit"
        gadget.parameter_bindings["enabled"] = "next"
        assert entries["enabled"].value(str) == "next"
        with pytest.raises(TypeError, match="as_mapping"):
            mapping.value(object)
        gadget.parameter_bindings = {"enabled": "bit"}


def _readout_nodes_with_gadget_locations(protocol: qc.Qodec) -> list[qc.Node]:
    gadgets = protocol.resolve("layers[0].gadgets").as_mapping()
    for node in gadgets.values():
        location = node.source_location
        assert location is not None and location.line >= 1
        assert location.path.is_file()
    readouts = [readout for node in gadgets.values() for readout in node.resolve("readouts").as_sequence()]
    assert readouts
    return readouts


def _assert_equation_locations(readouts: list[qc.Node], source_file: Path | None = None) -> None:
    for readout in readouts:
        assert readout.value(Readout).equation
        location = readout.resolve("equation").source_location
        assert location is not None
        if source_file is not None:
            assert location.path == source_file
        assert "circuit.readouts" in location.path.read_text().splitlines()[location.line - 1]


def test_loaded_locations_and_bundle_lines(tmp_path: Path) -> None:
    manifest = Path(__file__).resolve().parents[3] / "examples/repetition3/repetition3.qodec.yaml"
    protocol = qc.Qodec.load(manifest)
    root_location = protocol.resolve("").source_location
    assert root_location is not None and root_location.path == manifest
    readouts = _readout_nodes_with_gadget_locations(protocol)
    _assert_equation_locations(readouts)
    bundle = tmp_path / "actual.bundle"
    bundle.write_text(protocol.dumps())
    bundled = qc.Qodec.load(bundle)
    _assert_equation_locations([bundled.resolve(readout.path) for readout in readouts], bundle)
    protocol.layers[0].instruction_set.name = "changed"
    assert protocol.resolve("").source_location is None
    assert protocol.slice(0, 1).resolve("").source_location is None
    assert qc.Qodec.loads(bundled.dumps()).resolve("").source_location is None


def _write_external_equation_fixture(tmp_path: Path) -> tuple[Path, Path]:
    documents = {
        "qodec.yaml": {"layers": [
            {"instruction_set": "logical.data", "codes": {"qubit": "code.data"}, "gadgets": {"measure_z": "gadget.data"}},
            {"instruction_set": "physical.data"},
        ]},
        "logical.data": {"name": "logical", "blocks": {"qubit": 1}, "instructions": [
            {"mnemonic": "measure_z", "description": "measure", "in": ["qubit"], "action": [{"observe": "Z_0"}]}
        ]},
        "physical.data": {"name": "physical", "blocks": {"qubit": 1}, "instructions": []},
        "code.data": {"name": "qubit", "stabilizers": [], "x": ["X_0"], "z": ["Z_0"]},
        "gadget.data": {"circuit": {"format": "stim", "source": "M 0"}, "readouts": "equations.data"},
    }
    for filename, document in documents.items():
        (tmp_path / filename).write_text(json.dumps(document))
    manifest = tmp_path / "qodec.yaml"
    readouts_file = tmp_path / "equations.data"
    readouts_file.write_text('# equation file\n\n[["circuit.readouts[0]"]]\n')
    return manifest, readouts_file


def test_external_equations_retain_their_own_file(tmp_path: Path) -> None:
    manifest, readouts_file = _write_external_equation_fixture(tmp_path)
    protocol = qc.Qodec.load(manifest)
    node = protocol.resolve('layers[0].gadgets["measure_z"].readouts[0].equation[0]')
    location = node.source_location
    assert location is not None and location.path == readouts_file
    assert location.line == 3
    assert node.value(Reference).path in readouts_file.read_text().splitlines()[location.line - 1]
    encoding = protocol.resolve('layers[0].gadgets["measure_z"].in[0].code')
    assert encoding.source_location is not None