"""Source interpretation is configurable without changing stored declarations."""

import json
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import pytest
import qodec as qc
from qodec.actions import Observe, Stabilize
from qodec.gadgets import Circuit
from qodec.instructions import Block, BlockOperand, InstructionCall


def circuit(format: str = "parser-test") -> Circuit:
    instruction_set = qc.InstructionSet("test", instructions=[qc.Instruction("idle")])
    return Circuit(instruction_set, "authored source", format=format)


def idle_parser(
    source: str, instruction_set: qc.InstructionSet
) -> tuple[InstructionCall, ...]:
    assert source == "authored source"
    assert instruction_set.name == "test"
    return (InstructionCall("idle", operands=[7]),)


def test_registered_and_explicit_parsers() -> None:
    source = circuit()
    qc.register(idle_parser, format="parser-test")
    assert source.calls()[0].operands == [7]
    assert source.blocks == ["7"]
    assert source.readouts == []
    assert source.calls(parser=idle_parser) == source.calls()
    qc.register(lambda text, target: [], format="parser-test")
    assert source.calls() == []
    assert source.blocks == []
    assert source.calls(parser=idle_parser)[0].operands == [7]
    assert source.source == "authored source"


def test_override_does_not_register() -> None:
    source = circuit("unregistered-test")
    assert source.calls(parser=idle_parser)[0].mnemonic == "idle"
    with pytest.raises(ValueError, match="No source parser"):
        source.calls()


@pytest.mark.parametrize("invalid", [17, "not callable"])
def test_invalid_registration_preserves_the_working_parser(invalid: Any) -> None:
    source = circuit("registration-failure")
    qc.register(idle_parser, format="registration-failure")
    with pytest.raises(TypeError, match="parser must be callable"):
        qc.register(invalid, format="registration-failure")
    with pytest.raises(TypeError, match="parser must be callable"):
        source.calls(parser=invalid)
    assert source.calls()[0].operands == [7]


@pytest.mark.parametrize("value", [True, False])
def test_callback_boolean_operands_are_not_integer_labels(value: bool) -> None:
    with pytest.raises(ValueError, match="operands must be non-negative integers or strings"):
        circuit().calls(parser=lambda source, target: [InstructionCall("idle", operands=[value])])


def test_unknown_instruction_from_callback_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown instruction ['\"]missing['\"]"):
        circuit().calls(
            parser=lambda source, instruction_set: [InstructionCall("missing")]
        )


def stim_circuit(source: str) -> Circuit:
    instruction_set = qc.InstructionSet(
        "physical",
        blocks=[Block("qubit", 1)],
        instructions=[
            qc.Instruction(
                "M", inputs=[BlockOperand("qubit")], action=[Observe(["Z_0"])]
            ),
            qc.Instruction(
                "R", outputs=[BlockOperand("qubit")], action=[Stabilize(["Z_0"])]
            ),
            qc.Instruction("MPAD0", action=[Observe(["I"])]),
            qc.Instruction("MPAD1", action=[Observe(["-I"])]),
        ],
    )
    return Circuit(instruction_set, source, format="stim")


def test_stim_repeats_preserve_calls_and_record_positions() -> None:
    source = stim_circuit("R 0\nREPEAT 2 {\n M 0 1\n}\nM 2\n")
    assert [call.operands for call in source.calls()] == [[0], [0], [1], [0], [1], [2]]
    assert [bit.instruction for bit in source.readouts] == [1, 2, 3, 4, 5]
    assert source.blocks == ["0", "1", "2"]


def test_deep_repeat_nesting_reports_a_parser_error() -> None:
    stim_circuit("M 0").calls()
    source = stim_circuit("REPEAT 1 {\n" * 200 + "M 0\n" + "}\n" * 200)
    previous = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(150)
        with pytest.raises(ValueError, match="repeat nesting exceeds"):
            source.calls()
    finally:
        sys.setrecursionlimit(previous)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("M 0\nMPAD(0.1) 0 1\nM 1", "MPAD noise"),
        ("M(0.1) 0", "gate arguments"),
        ("M !0", "inverted targets"),
        ("REPEAT 1000001 {\nM 0\n}", "expansion exceeds"),
    ],
)
def test_stim_rejects_lossy_translation(source: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        stim_circuit(source).calls()


def test_stim_official_parser_accepts_comments_and_aliases() -> None:
    source = stim_circuit("RZ 0 # prepare\nMZ 0\nDETECTOR rec[-1]\n")
    assert [call.mnemonic for call in source.calls()] == ["R", "M"]


def test_mpad_preserves_bits_and_later_measurement_positions() -> None:
    source = stim_circuit("M 3\nMPAD 0 1\nREPEAT 2 {\nMPAD 1\nM 5\n}\nMPAD(0) 0")
    calls = source.calls()
    assert [call.mnemonic for call in calls] == [
        "M",
        "MPAD0",
        "MPAD1",
        "MPAD1",
        "M",
        "MPAD1",
        "M",
        "MPAD0",
    ]
    assert [call.operands for call in calls] == [[3], [], [], [], [5], [], [5], []]
    assert [bit.instruction for bit in source.readouts] == list(range(8))
    assert [
        bit.observable for bit in source.readouts if isinstance(bit, qc.gadgets.Outcome)
    ] == ["Z_0", "I", "-I", "-I", "Z_0", "-I", "Z_0", "I"]
    assert source.blocks == ["3", "5"]


def test_mpad_requires_only_the_constant_instructions_it_uses() -> None:
    source = stim_circuit("MPAD 0")
    source.instruction_set.instructions = [
        qc.Instruction("MPAD0", action=[Observe(["I"])])
    ]
    assert source.calls()[0].mnemonic == "MPAD0"
    source.source = "MPAD 1"
    with pytest.raises(ValueError, match="requires instruction 'MPAD1'"):
        source.calls()


@pytest.mark.parametrize(
    "declaration",
    [
        qc.Instruction("MPAD1", action=[Observe(["I"])]),
        qc.Instruction("MPAD1", action=[Observe(["-I"])], flags=["reject"]),
        qc.Instruction(
            "MPAD1", inputs=[BlockOperand("qubit")], action=[Observe(["-I"])]
        ),
        qc.Instruction("MPAD1", action=[Observe(["-I", "-I"])]),
    ],
)
def test_mpad_rejects_incompatible_declarations(declaration: qc.Instruction) -> None:
    source = stim_circuit("MPAD 1")
    source.instruction_set.instructions = [declaration]
    with pytest.raises(ValueError, match="single observe of -I"):
        source.calls()


@pytest.mark.parametrize(
    "text", ["MPAD 0 1 0 1", "M 0\nMPAD 0 1 0", "REPEAT 2 {\nMPAD 0 1\n}"]
)
def test_padding_obeys_the_call_expansion_limit(
    text: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qodec import _stim

    monkeypatch.setattr(_stim, "_MAX_CALLS", 3)
    with pytest.raises(ValueError, match="expansion exceeds 3"):
        stim_circuit(text).calls()


def test_empty_padding_and_padding_only_circuits_have_no_blocks() -> None:
    assert stim_circuit("MPAD").calls() == []
    source = stim_circuit("MPAD 0 1")
    assert source.blocks == []
    assert len(source.readouts) == 2


def test_missing_stim_does_not_prevent_source_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "stim", None)
    source = stim_circuit("M 0")
    assert source.source == "M 0"
    assert "M 0" in str(source)
    with pytest.raises(ValueError, match=r"qodec\[parsers\]"):
        source.calls()


@pytest.mark.parametrize("format", ["yaml", "stim"])
def test_registration_replaces_native_and_supplied_parsers(format: str) -> None:
    source = """
import sys
import qodec as qc
from qodec.gadgets import Circuit
from qodec.instructions import Block, BlockOperand, InstructionCall
format = sys.argv[1]
target = qc.InstructionSet('physical', blocks=[Block('qubit', 1)], instructions=[
    qc.Instruction('M', inputs=[BlockOperand('qubit')], action=[qc.actions.Observe(['Z_0'])])])
circuit = Circuit(target, 'M 0' if format == 'stim' else '- M: [0]', format=format)
assert circuit.calls()[0].operands == [0]
qc.register(lambda source, target: [InstructionCall('M', operands=[7])], format=format)
assert circuit.calls()[0].operands == [7]
assert circuit.blocks == ['7'] and len(circuit.readouts) == 1
qc.register(lambda source, target: [], format=format)
assert circuit.calls() == circuit.blocks == circuit.readouts == []
"""
    subprocess.run([sys.executable, "-I", "-c", source, format], check=True)


def test_native_yaml_needs_no_python_registration() -> None:
    from qodec import _parsers

    assert not hasattr(_parsers, "_registered")
    assert not hasattr(_parsers, "resolve")
    source = stim_circuit("- M: [4]")
    source.format = "yaml"
    assert source.calls()[0].operands == [4]
    assert source.blocks == ["4"]
    assert len(source.readouts) == 1
    source.format = "unregistered-native-test"
    inspections: list[Callable[[], object]] = [
        source.calls,
        lambda: source.blocks,
        lambda: source.readouts,
    ]
    for inspect in inspections:
        with pytest.raises(ValueError, match="No source parser"):
            inspect()


def test_module_load_without_stim_uses_native_yaml() -> None:
    source = """
import sys
sys.modules['stim'] = None
import qodec as qc
from qodec import _parsers
from qodec.gadgets import Circuit
from qodec.instructions import InstructionCall
assert not hasattr(_parsers, '_registered')
target = qc.InstructionSet('test', instructions=[qc.Instruction('idle')])
assert Circuit(target, '- idle: []', format='yaml').calls()[0].mnemonic == 'idle'
circuit = Circuit(target, 'I 0', format='stim')
assert circuit.source == 'I 0'
try:
    circuit.calls()
except ValueError as error:
    assert str(error) == "No source parser registered for '.stim'"
else:
    raise AssertionError('Stim must not have a fallback parser')
qc.register(lambda source, target: [InstructionCall('idle')], format='stim')
assert circuit.calls()[0].mnemonic == 'idle'
"""
    subprocess.run([sys.executable, "-I", "-c", source], check=True)


def test_replacement_does_not_change_a_running_callback() -> None:
    def replace(source: str, target: qc.InstructionSet) -> tuple[InstructionCall, ...]:
        qc.register(lambda text, target: [], format="replace-in-callback")
        return idle_parser(source, target)

    qc.register(replace, format="replace-in-callback")
    source = circuit("replace-in-callback")
    assert source.calls()[0].operands == [7]
    assert source.calls() == []


def test_replaced_parser_is_destroyed_after_unlocking() -> None:
    observed: list[str] = []

    class Parser:
        def __call__(
            self, source: str, target: qc.InstructionSet
        ) -> tuple[InstructionCall, ...]:
            return ()

        def __del__(self) -> None:
            qc.register(idle_parser, format="registered-by-destructor")
            observed.append("destroyed")

    qc.register(Parser(), format="destruction-test")
    qc.register(idle_parser, format="destruction-test")
    assert observed == ["destroyed"]
    assert circuit("registered-by-destructor").calls()[0].operands == [7]


@pytest.mark.parametrize("result", [None, ["not a call"], iter(())])
def test_invalid_parser_results_are_rejected(result: Any) -> None:
    with pytest.raises(TypeError, match="sequence|InstructionCall"):
        circuit().calls(parser=lambda source, target: result)


def test_callback_errors_are_not_hidden() -> None:
    def fail(source: str, target: qc.InstructionSet) -> list[InstructionCall]:
        raise RuntimeError("adapter failed")

    with pytest.raises(RuntimeError, match="adapter failed"):
        circuit().calls(parser=fail)


@pytest.mark.parametrize("value", [-(1 << 63), -1, 0, 1, (1 << 63) - 1])
def test_callback_integer_arguments_preserve_signed_range(value: int) -> None:
    actual = circuit().calls(parser=lambda source, target: [InstructionCall("idle", arguments={"value": value})])
    assert actual[0].arguments["value"] == value
    assert type(actual[0].arguments["value"]) is int


@pytest.mark.parametrize("value", [-(1 << 63) - 1, 1 << 63])
def test_callback_integer_arguments_reject_overflow(value: int) -> None:
    with pytest.raises(OverflowError):
        circuit().calls(parser=lambda source, target: [InstructionCall("idle", arguments={"value": value})])


def test_registered_callback_round_trips_complete_call_values() -> None:
    indices = [0, 3]
    arguments: dict[str, InstructionCall.Argument] = {
        "enabled": True,
        "disabled": False,
        "zero": 0,
        "negative": -2,
        "angle": 0.5,
        "text": "forwarded",
        "indices": indices,
        "names": ["a", "b"],
        "empty": [],
        "readout": "circuit.readouts[0]",
    }
    expected = InstructionCall(
        "idle",
        operands=[5, "named"],
        arguments=arguments,
        select=[{"reject": 0}, {"reject": 1}],
    )
    qc.register(lambda source, target: [expected], format="complete-call-values")
    actual = circuit("complete-call-values").calls()[0]
    assert actual == expected
    assert type(actual.arguments["enabled"]) is bool
    assert type(actual.arguments["zero"]) is int
    assert circuit("complete-call-values").blocks == ["5", "named"]
    indices.append(9)
    assert actual.arguments["indices"] == [0, 3]


@pytest.mark.parametrize("value", [None, {"nested": 1}, [True], [1, "mixed"], [1.5]])
def test_registered_callback_rejects_invalid_argument_shapes(value: Any) -> None:
    qc.register(
        lambda source, target: [InstructionCall("idle", arguments={"value": value})],
        format="invalid-callback-argument",
    )
    with pytest.raises(TypeError, match="argument value shape"):
        circuit("invalid-callback-argument").calls()


@pytest.mark.parametrize(
    ("value", "expected"), [
        ("", ""), ("label", "label"), ("007", "007"), ("true", "true"),
        ("readouts_label[0]", "readouts_label[0]"), ("in[0].z[0]", "in[0].z[0]"),
        ("circuit.readouts[003]", "circuit.readouts[3]"),
        ("circuit.readouts[+3]", "circuit.readouts[3]"),
        ("circuit.readouts[0:1]", "circuit.readouts[0]"),
        ("circuit.readouts[03:04]", "circuit.readouts[3]"),
        ("circuit.readouts[3:5:2]", "circuit.readouts[3]"),
    ],
)
def test_yaml_and_callback_text_arguments_agree(value: str, expected: str) -> None:
    source = circuit()
    callback_call = source.calls(parser=lambda text, target: [InstructionCall("idle", arguments={"value": value})])[0]
    yaml_source = json.dumps([{"idle": {"arguments": {"value": value}}}])
    yaml_call = Circuit(source.instruction_set, yaml_source, format="yaml").calls()[0]
    assert callback_call.arguments == yaml_call.arguments == {"value": expected}


@pytest.mark.parametrize("value", [
    "readouts[0]", "circuit.readouts[0:0]", "circuit.readouts[0:2]", "circuit.readouts[0,1]",
    "circuit.readouts[0,0]", "circuit.readouts[0:1048576]", "circuit.readouts[0:1][0]",
    "circuit.readouts[-1]", "circuit.readouts[]", "circuit.readouts[0", "circuit.readouts[0]suffix",
])
def test_yaml_and_callbacks_reject_invalid_readout_arguments(value: str) -> None:
    source = circuit()
    with pytest.raises(ValueError, match="readout") as callback_error:
        source.calls(
            parser=lambda source, target: [
                InstructionCall("idle", arguments={"bit": value})
            ]
        )
    yaml_source = json.dumps([{"idle": {"arguments": {"bit": value}}}])
    with pytest.raises(ValueError, match="readout") as yaml_error:
        Circuit(source.instruction_set, yaml_source, format="yaml").calls()
    assert str(callback_error.value) == str(yaml_error.value)


def test_registered_callback_preserves_original_exception_and_clears_state() -> None:
    original = RuntimeError("registered callback failed")

    def fail(source: str, target: qc.InstructionSet) -> list[InstructionCall]:
        raise original

    qc.register(fail, format="registered-error-state")
    source = circuit("registered-error-state")
    for inspect in (
        lambda: source.calls(),
        lambda: source.blocks,
        lambda: source.readouts,
    ):
        with pytest.raises(RuntimeError) as caught:
            inspect()
        assert caught.value is original
    qc.register(idle_parser, format="registered-error-state")
    assert source.calls()[0].operands == [7]
    with pytest.raises(ValueError, match="No source parser"):
        circuit("missing-after-error").calls()


def test_nested_callback_errors_do_not_leak_into_the_outer_call() -> None:
    def outer(source: str, target: qc.InstructionSet) -> tuple[InstructionCall, ...]:
        with pytest.raises(ValueError, match="No source parser"):
            circuit("missing-nested").calls()
        with pytest.raises(RuntimeError, match="inner"):
            circuit("nested-failure").calls()
        return idle_parser(source, target)

    def inner(source: str, target: qc.InstructionSet) -> list[InstructionCall]:
        raise RuntimeError("inner")

    qc.register(inner, format="nested-failure")
    qc.register(outer, format="nested-success")
    assert circuit("nested-success").calls()[0].operands == [7]


def test_callback_receives_an_isolated_instruction_set_snapshot() -> None:
    source = circuit("snapshot-test")

    def parser(text: str, target: qc.InstructionSet) -> list[InstructionCall]:
        assert target == source.instruction_set and target is not source.instruction_set
        target.name = "changed snapshot"
        return [InstructionCall("idle")]

    qc.register(parser, format="snapshot-test")
    assert source.calls()[0].mnemonic == "idle"
    assert source.instruction_set.name == "test"


def test_shutdown_releases_registered_python_callables() -> None:
    source = """
import atexit
import weakref
import qodec as qc
class Parser:
    def __call__(self, source, target):
        return []
parser = Parser()
reference = weakref.ref(parser)
atexit.register(lambda: print('released' if reference() is None else 'retained'))
qc.register(parser, format='shutdown-probe')
del parser
assert reference() is not None
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", source], check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == "released", result.stderr


def test_registered_parser_preserves_flag_record_positions() -> None:
    declaration = qc.Instruction("probe", action=[Observe(["Z_0"])], flags=["reject"])
    source = Circuit(
        qc.InstructionSet("flagged", instructions=[declaration]),
        "text",
        format="flag-record-test",
    )
    qc.register(
        lambda text, target: [InstructionCall("probe"), InstructionCall("probe")],
        format="flag-record-test",
    )
    assert [bit.instruction for bit in source.readouts] == [0, 0, 1, 1]
    assert [type(bit).__name__ for bit in source.readouts] == [
        "Outcome",
        "Flag",
        "Outcome",
        "Flag",
    ]


def test_stim_rejects_an_isa_that_changes_the_record() -> None:
    source = stim_circuit("M 0")
    original = source.instruction_set.instructions["M"]
    source.instruction_set.instructions = [
        qc.Instruction(
            "M", inputs=original.inputs, action=original.action, flags=["extra"]
        )
    ]
    with pytest.raises(ValueError, match="record shape"):
        source.calls()


def test_empty_huge_repeat_does_not_expand_annotations() -> None:
    assert stim_circuit("REPEAT 1000000000000 {\nTICK\n}").calls() == []
