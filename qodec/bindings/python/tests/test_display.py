from __future__ import annotations

import importlib
from pathlib import Path

import qodec as qc
import pytest
import yaml
from qodec.actions import Clifford, Condition, Observe, Pauli, Rotate, Stabilize
from qodec.gadgets import Circuit, Encoding, Reference
from qodec.instructions import Block, BlockOperand, Parameter


class Printer:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def text(self, text: str) -> None:
        self.parts.append(text)

    def break_(self) -> None:
        self.parts.append("\n")


def test_code_yaml_display() -> None:
    code = qc.Code("one", stabilizers=[], x=["X_0"], z=["Z_0"])
    assert yaml.safe_load(str(code)) == {"name": "one", "stabilizers": [], "x": ["X_0"], "z": ["Z_0"]}
    assert repr(code) == 'Code("one")'
    printer = Printer()
    code._repr_pretty_(printer, False)
    assert "".join(printer.parts) == str(code)
    cycle_printer = Printer()
    code._repr_pretty_(cycle_printer, True)
    assert "".join(cycle_printer.parts) == repr(code)


@pytest.fixture
def gadget() -> qc.Gadget:
    code = qc.Code("one", stabilizers=[], x=["X_0"], z=["Z_0"])
    instruction = qc.Instruction("measure", inputs=[BlockOperand("logical")], action=[Observe(["Z_0"])])
    physical = qc.InstructionSet("physical", blocks=[Block("qubit", 1)])
    return qc.Gadget(
        instruction,
        Circuit(physical, "M 0\n", format="stim"),
        inputs=[Encoding(code, support=["0"], block_types=["qubit"])],
        checks=[["in[0].stabilizers[0,1]", "circuit.readouts[9]"]],
        readouts=[{"result": ["circuit.readouts[0]", "in[0].z[0]"]}],
        parameter_bindings={"theta": "angle"},
        metadata={"note": "colon: # quote \" and newline\n", "enabled": True},
    )


def test_gadget_snippet_preserves_declarations(gadget: qc.Gadget) -> None:
    rendered = yaml.safe_load(str(gadget))
    assert set(rendered) == {"circuit", "in", "checks", "readouts", "parameter_bindings", "metadata"}
    assert rendered["circuit"] == {"source": "M 0\n", "format": "stim", "in": {"0": "qubit"}}
    assert rendered["in"] == [{"logical": [0]}]
    assert rendered["checks"] == [["in[0].stabilizers[0,1]", "circuit.readouts[9]"]]
    assert rendered["readouts"] == [{"result": ["circuit.readouts[0]", "in[0].z[0]"]}]
    assert rendered["parameter_bindings"] == {"theta": "circuit.source.angle"}
    assert rendered["metadata"] == gadget.metadata
    assert "source: |" in str(gadget)
    assert repr(gadget) == 'Gadget("measure")'


def test_flat_sequences_use_flow_style(gadget: qc.Gadget) -> None:
    rendered = str(gadget)
    assert "- logical: [0]" in rendered
    assert "\nchecks:\n- [" in rendered
    assert "\nreadouts:\n- result: [" in rendered
    assert yaml.safe_load(rendered)["circuit"]["source"] == gadget.circuit.source


def test_saved_artifacts_use_the_same_compact_output(tmp_path: Path) -> None:
    code = qc.Code("pair", stabilizers=["Z_0 Z_1"], x=["X_0 X_1"], z=["Z_0"])
    code_path = tmp_path / "code.yaml"
    code.save(code_path)
    assert code_path.read_text() == str(code)
    assert "stabilizers: [Z_0 Z_1]" in str(code)
    assert qc.Code.load(code_path) == code

    instruction = qc.Instruction("idle", inputs=[BlockOperand("pair")], outputs=[BlockOperand("pair")])
    instruction_set = qc.InstructionSet("logical", blocks=[Block("pair", 1)], instructions=[instruction])
    instruction_path = tmp_path / "instructions.yaml"
    instruction_set.save(instruction_path)
    assert instruction_path.read_text() == str(instruction_set)
    assert "in: [pair]" in str(instruction_set)
    assert qc.InstructionSet.load(instruction_path) == instruction_set


def _protocol_with_uninterpretable_source(source: str) -> qc.Qodec:
    code = qc.Code("pair", stabilizers=["Z_0 Z_1"], x=["X_0 X_1"], z=["Z_0"])
    instruction = qc.Instruction("idle", inputs=[BlockOperand("pair")], outputs=[BlockOperand("pair")])
    logical = qc.InstructionSet("logical", blocks=[Block("pair", 1)], instructions=[instruction])
    physical = qc.InstructionSet("physical", blocks=[Block("qubit", 1)])
    gadget = qc.Gadget(
        instruction,
        Circuit(physical, source, format="stim"),
        inputs=[Encoding(code, support=["0", "1"], block_types=["qubit", "qubit"])],
        outputs=[Encoding(code, support=["0", "1"], block_types=["qubit", "qubit"])],
        checks=[["in[0].stabilizers[0]", "out[0].stabilizers[0]"]],
    )
    return qc.Qodec([qc.Layer(logical, gadgets=[gadget]), qc.Layer(physical)])


def _saved_gadget_text(protocol: qc.Qodec, root: Path, single_file: bool) -> str:
    manifest_text = (root / protocol.manifest_filename).read_text()
    if single_file:
        return manifest_text
    manifest = yaml.safe_load(manifest_text)
    return (root / manifest["layers"][0]["gadgets"]["idle"]).read_text()


@pytest.mark.parametrize("single_file", [False, True])
def test_saved_qodec_compacts_lists_without_changing_source(tmp_path: Path, single_file: bool) -> None:
    source = "H 0\nnot even valid Stim: [\n\n"
    protocol = _protocol_with_uninterpretable_source(source)
    protocol.save(tmp_path, single_file=single_file)
    restored = qc.Qodec.load(tmp_path / protocol.manifest_filename)
    assert restored == protocol
    assert restored.layers[0].gadgets["idle"].circuit.source == source
    saved = _saved_gadget_text(protocol, tmp_path, single_file)
    assert "pair: [0, 1]" in saved
    assert qc.Qodec.loads(protocol.dumps()) == protocol
    assert "pair: [0, 1]" in protocol.dumps()


@pytest.mark.parametrize("source,format", [("not a circuit\n", "stim"), ("[broken", "yaml"), ("arbitrary\n", "unknown"), ("", None)])
def test_circuit_display_does_not_parse(source: str, format: str | None) -> None:
    circuit = Circuit(qc.InstructionSet("empty"), source, format=format)
    assert yaml.safe_load(str(circuit)) == {"source": source, "format": circuit.effective_format}


def test_display_uses_current_values_without_validation(gadget: qc.Gadget) -> None:
    gadget.circuit.source = "invalid stim\n"
    gadget.implements = qc.Instruction(gadget.implements.mnemonic)
    assert yaml.safe_load(str(gadget))["circuit"]["source"] == "invalid stim\n"
    # The draft instruction declares no operand for this encoding, so the block
    # type is unknown rather than borrowed from the code's name.
    assert yaml.safe_load(str(gadget))["in"] == [{"?": [0]}]
    code = gadget.inputs[0].code
    code.z = []
    assert yaml.safe_load(str(code))["z"] == []


@pytest.mark.parametrize(
    "value,expected",
    [
        (Block("data", 2), {"data": 2}),
        (BlockOperand("data"), "data"),
        (BlockOperand("data", is_variadic=True), ["data"]),
        (Parameter("theta", "number"), {"theta": "number"}),
        (Condition(["outcomes[0]"], invert=True), {"predicates": ["outcomes[0]"], "invert": True}),
        (Stabilize(["Z_0"]), {"stabilize": ["Z_0"]}),
        (Clifford({"X_0": "Z_0", "Z_0": "X_0"}), {"clifford": {"X_0": "Z_0", "Z_0": "X_0"}}),
        (Pauli("X_0", condition=Condition(["bit"], invert=True)), {"pauli": "X_0", "unless": ["bit"]}),
        (Observe(["Z_0"]), {"observe": "Z_0"}),
        (Rotate("Z_0", "theta", condition=Condition(["bit"])), {"rotate": {"pauli": "Z_0", "angle": "theta"}, "if": ["bit"]}),
    ],
)
def test_component_yaml(value: object, expected: object) -> None:
    assert yaml.safe_load(str(value)) == expected


def test_instruction_set_reuses_instruction_yaml(gadget: qc.Gadget) -> None:
    instruction = gadget.implements
    instruction_set = qc.InstructionSet("logical", blocks=[Block("logical", 1)], instructions=[instruction])
    rendered = yaml.safe_load(str(instruction_set))
    assert rendered["instructions"] == [yaml.safe_load(str(instruction))]
    instruction_set.instructions = []
    assert not yaml.safe_load(str(instruction_set)).get("instructions")


def test_pretty_matches_str_for_display_types(gadget: qc.Gadget) -> None:
    layer = qc.Layer(qc.InstructionSet("logical"), gadgets=[gadget])
    values = [
        gadget, gadget.circuit, gadget.implements, gadget.inputs[0].code,
        layer.instruction_set, layer, qc.Qodec([layer]),
        Block("data", 1), BlockOperand("data"), Parameter("angle", "number"),
        Condition(["bit"]), Stabilize(["Z_0"]), Clifford({}), Pauli("X_0"),
        Observe(["Z_0"]), Rotate("Z_0", 0.5), Reference("readouts[0]"), gadget.readouts[0],
        qc.codes.PauliExpression("X_0\nZ_1"),
    ]
    for value in values:
        printer = Printer()
        getattr(value, "_repr_pretty_")(printer, False)
        assert "".join(printer.parts) == str(value)
        cycle_printer = Printer()
        getattr(value, "_repr_pretty_")(cycle_printer, True)
        assert "".join(cycle_printer.parts) == repr(value)


def test_ipython_nested_display(gadget: qc.Gadget) -> None:
    pretty = pytest.importorskip("IPython.lib.pretty").pretty
    assert pretty(gadget) == str(gadget)
    nested = pretty([gadget], max_width=30)
    assert "  source:" in nested
    assert 'Gadget("measure")' not in nested
    recursive: list[object] = []
    recursive.append(recursive)
    assert pretty(recursive) == "[[...]]"


def test_display_hooks_are_the_exact_set() -> None:
    modules = [importlib.import_module(name) for name in (
        "qodec", "qodec.codes", "qodec.gadgets", "qodec.actions", "qodec.instructions"
    )]
    displayed = {
        value
        for module in modules
        for name in module.__all__
        if isinstance(value := getattr(module, name), type) and "_repr_pretty_" in value.__dict__
    }
    assert displayed == {
        qc.Qodec, qc.Layer, qc.Code, qc.Gadget, qc.InstructionSet, qc.Instruction,
        Circuit, Reference, qc.gadgets.Readout, qc.codes.PauliExpression,
        Block, BlockOperand, Parameter, Condition, Stabilize, Clifford, Pauli, Observe, Rotate,
    }


def test_mixed_physical_types_are_retained(gadget: qc.Gadget) -> None:
    gadget.inputs = [Encoding(gadget.inputs[0].code, support=["data", "aux"], block_types=["pair", "qubit"])]
    rendered = yaml.safe_load(str(gadget))
    assert rendered["circuit"]["in"] == {"data": "pair", "aux": "qubit"}
    assert rendered["in"] == [{"logical": ["data", "aux"]}]