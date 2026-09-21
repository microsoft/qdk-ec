"""Load -> save -> reload round-trip tests over the bundled examples.

Loads each example qodec from ``examples/`` and checks that saving it
back out (both as a directory bundle and as a single-file bundle) and
reloading reproduces the same structural surface. This exercises the
on-disk format end to end and pins the gadget
``inputs``/``outputs`` accessors against real artifacts.
"""

from __future__ import annotations

import json
from os.path import relpath
from pathlib import Path

import pytest
import yaml

import qodec
from qodec.instructions import InstructionCall

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"

EXAMPLE_PATHS = [
    EXAMPLES / "repetition3" / "repetition3.qodec.yaml",
    EXAMPLES / "c422-c832-arch" / "qodec.yaml",
    EXAMPLES / "c4c6" / "qodec.yaml",
]

@pytest.fixture
def external_protocol(tmp_path: Path) -> tuple[qodec.Qodec, Path]:
    source = tmp_path / "shared.yaml"
    qodec.InstructionSet("physical").save(source)
    project = tmp_path / "project"
    project.mkdir()
    manifest = project / "entry"
    manifest.write_text(json.dumps({"layers": [{"instruction_set": "../shared.yaml"}]}), encoding="utf-8")
    return qodec.Qodec.load(manifest), source


@pytest.mark.parametrize("edited", [False, True])
def test_external_files_are_linked_until_edited(
    external_protocol: tuple[qodec.Qodec, Path], tmp_path: Path, edited: bool
) -> None:
    protocol, source = external_protocol
    original = source.read_bytes()
    if edited:
        protocol.layers[0].instruction_set.description = "edited"
    protocol.manifest_filename = "nested/entry"
    manifest = protocol.save(tmp_path / "output")
    saved = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    target = (manifest.parent / saved["layers"][0]["instruction_set"]).resolve()
    assert (target == source) is not edited
    assert source.read_bytes() == original
    assert qodec.Qodec.load(manifest) == protocol


@pytest.mark.parametrize("missing", [False, True])
def test_external_file_changes_fail_before_writing(
    external_protocol: tuple[qodec.Qodec, Path], tmp_path: Path, missing: bool
) -> None:
    protocol, source = external_protocol
    if missing:
        source.unlink()
    else:
        source.write_text("name: changed\n", encoding="utf-8")
    destination = tmp_path / "output"
    with pytest.raises(qodec.QodecSaveError, match="cannot reuse external artifact|changed since loading"):
        protocol.save(destination)
    assert not destination.exists()
    assert qodec.Qodec.loads(protocol.dumps()) == protocol
    assert qodec.Qodec.load(protocol.save(destination, single_file=True)) == protocol


def _assert_gadget_preserved(expected: qodec.Gadget, actual: qodec.Gadget) -> None:
    assert actual.implements == expected.implements
    assert actual.inputs == expected.inputs
    assert actual.outputs == expected.outputs
    assert actual.circuit.instruction_set == expected.circuit.instruction_set
    assert actual.circuit.source == expected.circuit.source
    assert actual.circuit.effective_format == expected.circuit.effective_format
    assert actual.parameter_bindings == expected.parameter_bindings
    assert actual.checks == expected.checks
    assert actual.readouts == expected.readouts
    assert actual.frames == expected.frames
    assert actual.metadata == expected.metadata


def _assert_protocol_preserved(expected: qodec.Qodec, actual: qodec.Qodec) -> None:
    assert actual.name == expected.name
    assert actual.description == expected.description
    assert actual.schema_version == expected.schema_version
    assert Path(actual.manifest_filename) == Path(expected.manifest_filename)
    assert actual.metadata == expected.metadata
    assert actual.instruction_sets == expected.instruction_sets
    assert actual.codes == expected.codes
    assert len(actual.layers) == len(expected.layers)
    for expected_layer, actual_layer in zip(expected.layers, actual.layers):
        assert actual_layer.instruction_set == expected_layer.instruction_set
        assert actual_layer.gadgets.keys() == expected_layer.gadgets.keys()
        for mnemonic, gadget in expected_layer.gadgets.items():
            _assert_gadget_preserved(gadget, actual_layer.gadgets[mnemonic])


@pytest.fixture(params=EXAMPLE_PATHS, ids=lambda path: path.parent.name)
def example_path(request: pytest.FixtureRequest) -> Path:
    path = request.param
    assert path.is_file(), f"missing example fixture: {path}"
    return path


def _assert_example_has_content(protocol: qodec.Qodec) -> None:
    assert protocol.name
    assert protocol.layers
    assert any(layer.gadgets for layer in protocol.layers)
    assert any(
        encoding.block_types
        for layer in protocol.layers
        for gadget in layer.gadgets.values()
        for encoding in gadget.inputs
    ), "round-trip fixtures must exercise nonempty encoding block types"


def _save_and_reload(protocol: qodec.Qodec, destination: Path, *, single_file: bool = False) -> qodec.Qodec:
    manifest = protocol.save(str(destination), single_file=single_file)
    assert isinstance(manifest, Path)
    assert manifest == destination / protocol.manifest_filename
    if single_file:
        assert list(destination.iterdir()) == [manifest], "bundle must be the only saved file"
    return qodec.Qodec.load(manifest)


@pytest.mark.parametrize("single_file", [False, True], ids=["directory", "bundle"])
def test_example_roundtrip(example_path: Path, tmp_path: Path, single_file: bool) -> None:
    original = qodec.Qodec.load(str(example_path))
    _assert_example_has_content(original)
    reloaded = _save_and_reload(original, tmp_path / "saved", single_file=single_file)
    _assert_protocol_preserved(original, reloaded)


@pytest.mark.parametrize("single_file", [False, True], ids=["directory", "bundle"])
@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
@pytest.mark.parametrize("manifest_filename", ["qodec.yaml", "nested/protocol.yaml", "../entry"])
def test_save_returns_manifest_path(
    tmp_path: Path, single_file: bool, relative: bool, manifest_filename: str
) -> None:
    protocol = qodec.Qodec.load(EXAMPLE_PATHS[0])
    protocol.manifest_filename = manifest_filename
    destination = tmp_path / "saved"
    if relative:
        destination = Path(relpath(destination))
    manifest = protocol.save(destination, single_file=single_file)
    assert isinstance(manifest, Path)
    assert manifest == destination / manifest_filename
    assert manifest.is_absolute() is not relative
    assert qodec.Qodec.load(manifest) == protocol


@pytest.mark.parametrize("single_file", [False, True], ids=["directory", "bundle"])
def test_save_propagates_write_errors(tmp_path: Path, single_file: bool) -> None:
    protocol = qodec.Qodec([])
    (tmp_path / protocol.manifest_filename).mkdir()
    with pytest.raises(qodec.QodecSaveError):
        protocol.save(tmp_path, single_file=single_file)


def test_gadget_inputs_outputs_accessors(example_path: Path) -> None:
    """Every loaded gadget exposes the ``inputs``/``outputs`` accessors."""
    codec = qodec.Qodec.load(str(example_path))
    saw_encoding = False
    for layer in codec.layers:
        for name in layer.gadgets:
            gadget = layer.gadgets[name]
            for encoding in (*gadget.inputs, *gadget.outputs):
                saw_encoding = True
                assert encoding.code.name
                assert all(isinstance(block, str) for block in encoding.support)
    assert saw_encoding


def test_load_and_save_accept_path_like(example_path: Path, tmp_path: Path) -> None:
    """Every ``load``/``save`` takes ``os.PathLike``, not only ``str``."""
    codec = qodec.Qodec.load(example_path)

    bundle = tmp_path / "bundle"
    _assert_protocol_preserved(codec, qodec.Qodec.load(codec.save(bundle)))

    instruction_set = codec.layers[0].instruction_set
    isa_path = tmp_path / "standalone" / "layer.data"
    instruction_set.save(isa_path)
    assert qodec.InstructionSet.load(isa_path) == instruction_set

    code = next(iter(codec.codes.values()))
    code_path = tmp_path / "standalone" / "code"
    code.save(code_path)
    assert qodec.Code.load(code_path) == code


def test_bundle_string_roundtrip(example_path: Path) -> None:
    """``dumps`` and ``loads`` round-trip without touching the filesystem."""
    original = qodec.Qodec.load(example_path)
    reloaded = qodec.Qodec.loads(original.dumps())
    _assert_protocol_preserved(original, reloaded)


def test_loads_rejects_a_plain_manifest() -> None:
    """A single-document manifest is a directory qodec, not a bundle."""
    with pytest.raises(qodec.QodecLoadError):
        qodec.Qodec.loads("name: lonely\nlayers: []\n")


def _assert_selection_is_separate_from_arguments(full: InstructionCall, shorthand: InstructionCall) -> None:
    assert full.operands == shorthand.operands == [0]
    assert full.arguments == {"select": -2}
    assert full.select == [{"select": 0}]
    assert shorthand.arguments == {"select": -3}
    assert shorthand.select == []


def _assert_boolean_argument_types(calls: list[InstructionCall]) -> None:
    for call, expected in zip(calls, (True, False)):
        assert call.arguments == {"select": expected, "disabled": not expected, "one": 1, "zero": 0}
        assert type(call.arguments["select"]) is bool
        assert type(call.arguments["disabled"]) is bool
        assert type(call.arguments["one"]) is int
        assert type(call.arguments["zero"]) is int
        assert call.operands == [0]
        assert type(call.operands[0]) is int
    assert calls[0].select == [{"select": 1}]
    assert type(calls[0].select[0]["select"]) is int
    assert calls[1].select == []


def _assert_argument_shapes(protocol: qodec.Qodec, expected_source: str) -> None:
    assert protocol.schema_version == 1
    declared = protocol.layers[1].instruction_set.instructions["select"]
    assert declared.flags == ["select"]
    assert [parameter.name for parameter in declared.parameters] == ["select"]
    circuit = protocol.layers[0].gadgets["noop"].circuit
    assert circuit.effective_format == "yaml"
    assert circuit.source == expected_source
    calls = circuit.calls()
    assert [call.mnemonic for call in calls] == ["M", "probe", "select", "select", "boolean", "boolean"]
    _assert_selection_is_separate_from_arguments(calls[2], calls[3])
    _assert_boolean_argument_types(calls[4:])


@pytest.mark.parametrize("single_file", [False, True])
def test_inline_yaml_select_names_survive_loading_and_roundtrip(tmp_path: Path, single_file: bool) -> None:
    fixture = Path(__file__).resolve().parents[2] / "c/tests/fixtures/argument-shapes/argument-shapes.qodec.yaml"
    original = qodec.Qodec.load(fixture)
    saved = _save_and_reload(original, tmp_path, single_file=single_file)
    from_text = qodec.Qodec.loads(original.dumps())
    for protocol in (original, from_text, saved):
        _assert_argument_shapes(protocol, original.layers[0].gadgets["noop"].circuit.source)


@pytest.fixture
def reference_documents() -> dict[str, object]:
    return {
        "entry": {
            "schema_version": 1,
            "name": "reference-loading",
            "layers": [
                {
                    "instruction_set": "./unused/../logical.data",
                    "codes": {"trivial": "./trivial"},
                    "gadgets": {"noop": "./gadgets/noop.payload"},
                },
                {"instruction_set": "./physical"},
            ],
        },
        "./logical.data": {
            "name": "top",
            "description": "A single logical instruction.",
            "blocks": {"trivial": 1},
            "instructions": [{"mnemonic": "noop", "description": "Identity.", "in": ["trivial"], "out": ["trivial"]}],
        },
        "physical": {
            "name": "bottom",
            "description": "A single physical instruction.",
            "blocks": {"qubit": 1},
            "instructions": [{"mnemonic": "I", "description": "Identity.", "in": ["qubit"], "out": ["qubit"]}],
        },
        "trivial": {"name": "trivial", "stabilizers": [], "x": ["X_0"], "z": ["Z_0"]},
        "gadgets/noop.payload": {"circuit": "../../sources/./idle.stim"},
        "../sources/./idle.stim": "I 0\n",
        "qodec.yaml": {"not": "a manifest"},
        "unrelated/bad.isa.yaml": "[invalid YAML",
        "unrelated/bad.code.yaml": "[invalid YAML",
        "unrelated/bad.gadget.yaml": "[invalid YAML",
        "unrelated/orphan.gadget.yaml": {"circuit": "./missing.stim"},
    }


def _write_reference_documents(directory: Path, documents: dict[str, object], manifest_filename: str) -> Path:
    for filename, content in documents.items():
        path = directory / (manifest_filename if filename == "entry" else filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8", newline="\n")
    return directory / manifest_filename


@pytest.mark.parametrize("manifest_filename", ["myproto.qodec.yaml", "protocol.data", "protocol"])
def test_load_follows_references_and_preserves_manifest_filename(
    tmp_path: Path, reference_documents: dict[str, object], manifest_filename: str
) -> None:
    manifest = _write_reference_documents(tmp_path / "input", reference_documents, manifest_filename)
    codec = qodec.Qodec.load(manifest)
    assert codec.manifest_filename == manifest_filename
    _assert_referenced_model(codec)

    output = tmp_path / "output"
    saved = _save_and_reload(codec, output)
    assert saved.manifest_filename == manifest_filename
    assert (output / manifest_filename).is_file()
    assert not (output / "qodec.yaml").exists()
    _assert_protocol_preserved(codec, saved)

    (manifest.parent / "logical.data").write_text("[invalid YAML", encoding="utf-8")
    with pytest.raises(qodec.QodecLoadError, match=r"logical\.data"):
        qodec.Qodec.load(manifest)


def _assert_referenced_model(codec: qodec.Qodec) -> None:
    assert codec.schema_version == 1
    assert set(codec.instruction_sets) == {"top", "bottom"}
    assert set(codec.codes) == {"trivial"}
    assert set(codec.layers[0].gadgets) == {"noop"}
    assert codec.layers[0].gadgets["noop"].circuit.source == "I 0\n"


@pytest.mark.parametrize("single_file", [False, True])
def test_moving_loaded_manifest_to_parent_preserves_artifacts(
    tmp_path: Path, reference_documents: dict[str, object], single_file: bool
) -> None:
    bundle = "\n---\n".join(json.dumps({path: content}) for path, content in reference_documents.items())
    protocol = qodec.Qodec.loads(bundle)
    protocol.manifest_filename = "../entry"
    protocol.description = "edited"
    text = protocol.dumps()
    _assert_protocol_preserved(protocol, qodec.Qodec.loads(text))
    output = tmp_path / "root" / "output"
    manifest = protocol.save(output, single_file=single_file)
    assert manifest == output / protocol.manifest_filename
    restored = qodec.Qodec.load(manifest)
    assert restored == protocol
    assert Path(restored.manifest_filename) == Path("../entry" if single_file else "entry")
    _assert_referenced_model(restored)


def _write_reference_bundle(
    tmp_path: Path, reference_documents: dict[str, object], source_on_disk: bool
) -> tuple[Path, str]:
    directory = tmp_path / "input"
    directory.mkdir()
    if source_on_disk:
        source = reference_documents.pop("../sources/./idle.stim")
        assert isinstance(source, str)
        source_path = tmp_path / "sources" / "idle.stim"
        source_path.parent.mkdir()
        source_path.write_text(source, encoding="utf-8", newline="\n")
    bundle = "\n---\n".join(json.dumps({path: content}) for path, content in reference_documents.items())
    path = directory / "archive.data"
    path.write_text(bundle, encoding="utf-8")
    return path, bundle


@pytest.mark.parametrize("source_on_disk", [False, True])
@pytest.mark.parametrize("single_file", [False, True])
def test_bundle_uses_first_key_and_referenced_source(
    tmp_path: Path, reference_documents: dict[str, object], source_on_disk: bool, single_file: bool
) -> None:
    path, bundle = _write_reference_bundle(tmp_path, reference_documents, source_on_disk)
    codec = qodec.Qodec.load(path)
    assert codec.manifest_filename == "entry"
    assert codec.name == "reference-loading"
    assert codec.schema_version == 1
    circuit = codec.layers[0].gadgets["noop"].circuit
    assert circuit.source == "I 0\n"
    assert [call.mnemonic for call in circuit.calls()] == ["I"]
    if not source_on_disk:
        _assert_protocol_preserved(codec, qodec.Qodec.loads(bundle))

    output = tmp_path / "output"
    codec.save(output, single_file=single_file)
    _assert_protocol_preserved(codec, qodec.Qodec.load(output / codec.manifest_filename))


@pytest.mark.parametrize("schema_version", [0, 1, 2])
def test_bundle_schema_version_must_be_current(reference_documents: dict[str, object], schema_version: int) -> None:
    manifest = reference_documents["entry"]
    assert isinstance(manifest, dict)
    manifest["schema_version"] = schema_version
    bundle = "\n---\n".join(json.dumps({path: content}) for path, content in reference_documents.items())
    if schema_version == 1:
        assert qodec.Qodec.loads(bundle).schema_version == schema_version
    else:
        with pytest.raises(qodec.QodecLoadError, match=f"declares schema_version {schema_version}"):
            qodec.Qodec.loads(bundle)


@pytest.mark.parametrize("single_file", [False, True])
@pytest.mark.parametrize("remove_block", [False, True])
def test_unused_layer_codes_survive_python_saves(tmp_path: Path, single_file: bool, remove_block: bool) -> None:
    unused_code = {"name": "unused", "stabilizers": [], "x": ["X_0"], "z": ["Z_0"]}
    documents = [
        {"qodec.yaml": {"schema_version": 1, "layers": [
            {"instruction_set": "logical.yaml", "codes": {"spare": "unused.yaml"}},
            {"instruction_set": "physical.yaml"},
        ]}},
        {"logical.yaml": {"name": "logical", "blocks": {"spare": 1}, "instructions": []}},
        {"physical.yaml": {"name": "physical", "blocks": {"qubit": 1}, "instructions": []}},
        {"unused.yaml": unused_code},
    ]
    protocol = qodec.Qodec.loads("\n---\n".join(map(json.dumps, documents)))
    protocol.description = "edited"
    if remove_block:
        protocol.layers[0].instruction_set.blocks = []
    protocol.save(tmp_path, single_file=single_file)
    restored = qodec.Qodec.load(tmp_path / protocol.manifest_filename)
    saved = {key: value for document in yaml.safe_load_all(restored.dumps()) for key, value in document.items()}
    assert saved.get("unused.yaml") == (None if remove_block else unused_code)
    assert saved["qodec.yaml"]["layers"][0].get("codes", {}) == ({} if remove_block else {"spare": "unused.yaml"})
    assert saved["qodec.yaml"]["description"] == "edited"


@pytest.mark.parametrize("single_file", [False, True])
def test_code_edits_survive_last_gadget_removal(tmp_path: Path, single_file: bool) -> None:
    example = Path(__file__).resolve().parents[3] / "examples/repetition3/repetition3.qodec.yaml"
    protocol = qodec.Qodec.load(example)
    code = protocol.codes["repetition3"]
    code.z = ["Z_1"]
    manifest = protocol.save(tmp_path, single_file=single_file)
    assert qodec.Qodec.load(manifest).codes["repetition3"].z == ["Z_1"]
    protocol.layers[0].gadgets = {}
    protocol.validate()
    assert protocol.codes["repetition3"] is code
    assert protocol.resolve('codes["repetition3"]').value(qodec.Code) is code
    restored = qodec.Qodec.load(protocol.save(tmp_path, single_file=single_file))
    saved = {key: value for document in yaml.safe_load_all(restored.dumps()) for key, value in document.items()}
    assert saved["repetition3.code.yaml"]["z"] == code.z


def test_unused_layer_code_bindings_are_live_and_editable(tmp_path: Path) -> None:
    instruction_set = qodec.InstructionSet("logical", blocks=[qodec.instructions.Block("spare", encodes=1)])
    code = qodec.Code("unused", stabilizers=[], x=["X_0"], z=["Z_0"])
    layer = qodec.Layer(instruction_set, codes={"spare": code})
    protocol = qodec.Qodec([layer])
    node = protocol.resolve('layers[0].codes["spare"]')
    assert node.value(qodec.Code) is code
    assert set(protocol.resolve("codes").mapping_nodes()) == {"unused"}
    code.z = ["Z_1"]
    restored = qodec.Qodec.load(protocol.save(tmp_path))
    assert restored.layers[0].codes["spare"].z == ["Z_1"]
    layer.codes = {}
    with pytest.raises(LookupError):
        node.value(qodec.Code)
    assert protocol.codes == {}


@pytest.mark.parametrize("start, stop", [(0, 2), (0, 1), (1, 2)])
def test_slice_preserves_retained_layer_code_declarations(start: int, stop: int) -> None:
    example = Path(__file__).resolve().parents[3] / "examples/distillation-15/distillation-15.qodec.yaml"
    protocol = qodec.Qodec.load(example)
    original = {key: value for document in yaml.safe_load_all(protocol.dumps()) for key, value in document.items()}
    sliced = protocol.slice(start, stop)
    saved = {key: value for document in yaml.safe_load_all(sliced.dumps()) for key, value in document.items()}
    original_layers = original[protocol.manifest_filename]["layers"][start:stop]
    saved_layers = saved[sliced.manifest_filename]["layers"]
    assert [layer.get("codes", {}) for layer in saved_layers] == [layer.get("codes", {}) for layer in original_layers]
    for layer in original_layers:
        for reference in layer.get("codes", {}).values():
            assert saved[reference] == original[reference]


@pytest.mark.parametrize(
    "circuit",
    [
        {"source": [{"missing": [0]}]},
        {"source": [{"missing": {"operands": [0]}}]},
        {"source": [{"missing": {}}]},
        {"format": "stim", "source": "X 0"},
    ],
)
def test_unknown_circuit_calls_are_preserved_until_inspected(
    tmp_path: Path, reference_documents: dict[str, object], circuit: dict[str, object]
) -> None:
    reference_documents["gadgets/noop.payload"] = {"circuit": circuit}
    bundle = "\n---\n".join(json.dumps({path: content}) for path, content in reference_documents.items())
    bundle_path = tmp_path / "protocol"
    bundle_path.write_text(bundle, encoding="utf-8")
    for protocol in [qodec.Qodec.loads(bundle), qodec.Qodec.load(bundle_path)]:
        source = protocol.layers[0].gadgets["noop"].circuit.source
        reloaded = qodec.Qodec.loads(protocol.dumps())
        saved = reloaded.layers[0].gadgets["noop"].circuit
        assert saved.source == source
        with pytest.raises(ValueError, match="unknown instruction"):
            _ = saved.calls()
