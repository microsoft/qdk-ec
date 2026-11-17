"""Source annotations reach both native output formats without changing decoding."""

from pathlib import Path

import pytest
from google.protobuf.json_format import MessageToDict

from deq.circuit.parser import parse
from deq.circuit.body_validation import validate_metadata
from deq.cli.simulate import simulate__ler
from deq.compiler.jit_compiler import static_jit_compiler
from deq.proto import deq_bin_pb2 as pb, deq_jit_pb2 as jit_pb
from deq.transpiler.jit_library_builder import build_jit_library
from deq.transpiler.loss.model_none import NoLossModel
from deq.transpiler.program_artifacts import transpile_program


SOURCE = '''
@METADATA("tools.code", {"tags": ["data", null], "enabled": true})
CODE C [[1,1]] {
    @METADATA("tools.logical", ["x", "z"])
    LOGICAL X0 Z0
}
@METADATA("tools.type", "prepare")
GADGET Prepare {
    R 0
    @METADATA("tools.port", {"direction": "out"})
    OUTPUT C 0
}
@CHECKS("manual", verify=0)
GADGET Read {
    @METADATA("tools.port", {"direction": "in"})
    INPUT C 0
    @METADATA("tools.measurement", {"basis": "Z"})
    M 0
    @METADATA("tools.readout", [1, false])
    READOUT M0
    @METADATA("tools.check", {"name": "check"})
    CHECK M0
    @METADATA("tools.error", {"name": "error"})
    ERROR(0.001) C0 R0
}
@METADATA("tools.type", {"composition": true})
COMPOSE Pair { Prepare 0 Read 0 }
@METADATA("tools.scope", "definition")
PROGRAM Sub {
    @METADATA("tools.scope", "inner")
    Prepare 0
    Read 0
}
@METADATA("tools.run", {"description": "metadata test", "options": {"seed": 17}})
PROGRAM Main {
    @METADATA("tools.scope", "loop")
    @METADATA("tools.loop", ["repeated"])
    @START(at=0.1)
    REPEAT 2 {
        @METADATA("tools.scope", "call")
        @DELAY(0.1)
        Sub()
    }
    @METADATA("tools.instance", null)
    Pair()
    ASSERT_EQ rec[-1] 0
}
'''


def annotations(message):
    return MessageToDict(message.metadata)


def check_outputs(library, compiled):
    assert annotations(library) == annotations(compiled)
    assert annotations(compiled)["tools.run"] == {"description": "metadata test", "options": {"seed": 17}}
    assert compiled.port_types[0] == library.port_types[0].base
    assert annotations(compiled.port_types[0])["tools.code"]["tags"] == ["data", None]
    assert all(annotations(observable)["tools.logical"] == ["x", "z"] for observable in compiled.port_types[0].observables)
    types = {item.base.name: item for item in library.gadget_types}
    assert annotations(types["Prepare"].base) == {"tools.type": "prepare"}
    assert annotations(types["Prepare"].base.outputs[0])["tools.port"]["direction"] == "out"
    assert annotations(types["Read"].base.inputs[0])["tools.port"]["direction"] == "in"
    for name in ("Read", "Pair"):
        gadget = types[name]
        assert annotations(gadget.base.measurements[0])["tools.measurement"] == {"basis": "Z"}
        assert annotations(gadget.base.readouts[0])["tools.readout"] == [1, False]
        assert annotations(gadget.finished_checks[0].base)["tools.check"] == {"name": "check"}
        assert annotations(gadget.errors[0].base)["tools.error"] == {"name": "error"}
    assert annotations(types["Pair"].base) == {"tools.type": {"composition": True}}
    gadgets = [item.gadget for item in compiled.program if item.WhichOneof("create") == "gadget"]
    assert gadgets == [item.gadget for item in library.program]
    assert [annotations(gadget).get("tools.scope") for gadget in gadgets] == [
        "inner", "call", "inner", "call", None,
    ]
    assert all(annotations(gadget)["tools.loop"] == ["repeated"] for gadget in gadgets[:4])
    assert annotations(gadgets[-1])["tools.instance"] is None


@pytest.mark.parametrize("compact", [False, True])
def test_metadata_survives_nested_calls_repeats_and_composition(compact):
    artifacts = transpile_program(parse(SOURCE), "Main")
    check_outputs(artifacts.library, static_jit_compiler(artifacts.library, compact=compact))


def test_annotation_roundtrip_retains_custom_metadata():
    from deq.transpiler.jit_annotate import annotate

    source = parse(SOURCE)
    annotated = annotate(source, loss_model=NoLossModel())
    artifacts = transpile_program(parse(annotated), "Main")
    check_outputs(artifacts.library, static_jit_compiler(artifacts.library))


def test_metadata_reaches_cli_output_files(tmp_path: Path):
    source = tmp_path / "source.deq"
    source.write_text(SOURCE)
    simulate__ler(
        str(source), program="Main", save=str(tmp_path / "output"),
        shots=1, errors=2, jobs=1, batch_size=1, simulator="static",
        strict_timing=True, loss_model="none", seed=17,
    )
    library = jit_pb.JitLibrary.FromString((tmp_path / "output/Main.deq.jit").read_bytes())
    compiled = pb.Library.FromString((tmp_path / "output/Main.deq.bin").read_bytes())
    check_outputs(library, compiled)


def test_stabilizer_annotations_survive_in_jit_and_static_observables():
    source = parse('''
CODE C [[3,1]] {
    LOGICAL X0*X1*X2 Z0
    @METADATA("tools.stabilizer", {"group": "Z"})
    STABILIZER Z0*Z1 Z1*Z2
}
''')
    library = build_jit_library(source, jobs=1, loss_model=NoLossModel())
    compiled = static_jit_compiler(library)
    assert all(annotations(item) == {"tools.stabilizer": {"group": "Z"}} for item in library.port_types[0].stabilizers)
    assert all(annotations(item) == {"tools.stabilizer": {"group": "Z"}} for item in compiled.port_types[0].observables[2:])

def test_loss_annotations_reach_both_native_formats():
    source = parse('''
@CHECKS("manual", verify=0)
GADGET G {
    R 0
    M 0
    CHECK M0
    ERROR(0.0) C0
    @METADATA("tools.loss", {"site": "source"})
    LOSS(0.1) SE0 CE0 M0
}
PROGRAM Main { G() }
''')
    artifacts = transpile_program(source, "Main")
    compiled = static_jit_compiler(artifacts.library)
    for gadget in (artifacts.library.gadget_types[0].base, compiled.gadget_types[0]):
        assert annotations(gadget.loss_model.losses[0]) == {"tools.loss": {"site": "source"}}

def test_composed_boundary_ports_and_input_losses_preserve_annotations():
    source = parse('''
CODE C [[1,1]] { LOGICAL X0 Z0 }
GADGET Pass {
    INPUT C 0
    OUTPUT C 0
    @METADATA("tools.loss", {"site": "input"})
    LOSS(IN0.L0) OUT0.L0
}
@METADATA("tools.type", "composed")
COMPOSE Wrapped {
    @METADATA("tools.port", "input")
    INPUT C 0
    Pass 0
    @METADATA("tools.port", "output")
    OUTPUT C 0
}
''')
    library = build_jit_library(source, jobs=1)
    wrapped = next(item.base for item in library.gadget_types if item.base.name == "Wrapped")
    assert annotations(wrapped.inputs[0]) == {"tools.port": "input"}
    assert annotations(wrapped.outputs[0]) == {"tools.port": "output"}
    assert annotations(wrapped.loss_model.input_losses[0]) == {"tools.loss": {"site": "input"}}


def test_inner_program_defaults_override_outer_loop_metadata():
    source = SOURCE.replace('@METADATA("tools.scope", "call")', "")
    artifacts = transpile_program(parse(source), "Main")
    assert [
        annotations(item.gadget).get("tools.scope") for item in artifacts.library.program
    ] == ["inner", "definition", "inner", "definition", None]


def test_transformed_annotated_checks_are_rejected():
    source = parse('''
GADGET G {
    R 0
    M 0 0
    @METADATA("tools.check", "custom parity")
    CHECK M0 M1
}
''')
    with pytest.raises(ValueError, match="annotated CHECK was transformed or removed"):
        build_jit_library(source, jobs=1, loss_model=NoLossModel())

def test_compact_export_does_not_merge_distinct_error_annotations():
    artifacts = transpile_program(parse('''
@CHECKS("manual", verify=0)
GADGET A { R 0 M 0 CHECK M0 @METADATA("label", "a") ERROR(0.1) C0 }
@CHECKS("manual", verify=0)
GADGET B { R 0 M 0 CHECK M0 @METADATA("label", "b") ERROR(0.1) C0 }
PROGRAM Main { REPEAT 2 { A() B() } }
'''), "Main")
    compiled = static_jit_compiler(artifacts.library, compact=True)
    assert len(compiled.error_model_types) == 2
    labels = {model.etype: annotations(model.errors[0])["label"] for model in compiled.error_model_types}
    assert [
        labels[item.error_model.etype] for item in compiled.program
        if item.WhichOneof("create") == "error_model"
    ] == ["a", "b", "a", "b"]


@pytest.mark.parametrize("annotation", [
    '@METADATA("key")', '@METADATA(1, {})', '@METADATA("", {})',
    '@METADATA("key", value=1)', '@METADATA("key", 9007199254740993)',
    '@METADATA("key", {"nested": 9007199254740993})',
    '@METADATA("key", 9007199254740993.0)',
    '@METADATA("key", {"nested": 1e999})',
    '@METADATA("key", {"nested": 1e-999})',
    '@METADATA("key", 1) @METADATA("key", 2)',
    '@METADATA("deq.simulation.timing", {})',
    '@METADATA("private_gadgets", [])',
])
def test_invalid_metadata_is_not_silently_discarded(annotation):
    with pytest.raises((ValueError, SyntaxError), match="METADATA"):
        parse(f"{annotation} GADGET G {{ R 0 M 0 }}")


@pytest.mark.parametrize("source", [
    'GADGET G { @METADATA("key", 1) H 0 }',
    'PROGRAM P { @METADATA("key", 1) ASSERT_EQ 0 }',
    'PROGRAM P { @METADATA("key", 1) INPUT C 0 }',
    'GADGET G { @METADATA("key", 1) }',
    'PROGRAM P { @METADATA("key", 1) VIRTUAL X0 0 }',
])
def test_metadata_requires_an_output_record(source):
    with pytest.raises((ValueError, SyntaxError), match="METADATA"):
        parse(source)


def test_public_metadata_validator_uses_explicit_context_recursively():
    from deqagram import Decorator, Instruction, QubitTarget, RepeatBlock

    call = Instruction(
        name="M", targets=[QubitTarget(0)],
        decorators=[Decorator("METADATA", ("tools.record", {"label": "measurement"}))],
        source_line=7,
    )
    body = (RepeatBlock(count=2, body=[call]),)
    validate_metadata(body, context="PROGRAM")
    validate_metadata(body, context="GADGET")
    with pytest.raises(SyntaxError, match=r"no output record on Instruction \(line 7\)"):
        validate_metadata(body, context="COMPOSE")
    call.name = "H"
    validate_metadata(body, context="PROGRAM")
    with pytest.raises(SyntaxError, match="no output record"):
        validate_metadata(body, context="GADGET")


def test_public_metadata_validator_checks_values_on_constructed_nodes():
    from deqagram import Decorator, ReadoutStatement

    readout = ReadoutStatement(
        decorators=[Decorator("METADATA", ("tools.record", float("inf")))]
    )
    with pytest.raises(ValueError, match="finite JSON-compatible"):
        validate_metadata([readout], context="GADGET")
