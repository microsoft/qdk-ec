# pylint: disable=no-member
"""Tests for the strip-tags CLI tool."""

import tempfile
from pathlib import Path

import deq.proto.deq_bin_pb2 as pb
import deq.proto.deq_jit_pb2 as jit_pb
from deq.cli.strip_tags import (
    strip_bin_library,
    strip_jit_library,
    strip_tags,
)

# ── BIN tests ────────────────────────────────────────────────────────────


def _sample_bin_library() -> pb.Library:
    return pb.Library(
        description="test library",
        gadget_types=[
            pb.GadgetType(
                gtype=1,
                name="MyGadget",
                description="gadget desc",
                measurements=[
                    pb.GadgetType.Measurement(tag="m0"),
                    pb.GadgetType.Measurement(tag="m1"),
                ],
                inputs=[pb.GadgetType.Port(ptype=1, tag="in0")],
                outputs=[pb.GadgetType.Port(ptype=1, tag="out0")],
                readouts=[pb.GadgetType.Readout(tag="r0")],
            ),
        ],
        port_types=[
            pb.PortType(
                ptype=1,
                name="MyPort",
                description="port desc",
                observables=[
                    pb.PortType.Observable(tag="LX0"),
                    pb.PortType.Observable(tag="LZ0"),
                ],
            ),
        ],
        check_model_types=[
            pb.CheckModelType(
                ctype=1,
                name="MyCheck",
                description="check desc",
                remote_gadgets=[
                    pb.CheckModelType.RemoteGadget(tag="prev"),
                ],
                checks=[
                    pb.CheckModelType.Check(tag="c0"),
                    pb.CheckModelType.Check(tag="c1"),
                ],
            ),
        ],
        error_model_types=[
            pb.ErrorModelType(
                etype=1,
                name="MyError",
                description="error desc",
                remote_check_models=[
                    pb.ErrorModelType.RemoteCheckModel(tag="rcm0"),
                ],
                errors=[
                    pb.ErrorModelType.Error(tag="E0", probability=0.01),
                    pb.ErrorModelType.Error(tag="E1", probability=0.02),
                ],
            ),
        ],
        program=[
            pb.Instruction(gadget=pb.Gadget(gid=1, gtype=1, tag="g-inst")),
            pb.Instruction(
                check_model=pb.CheckModel(cid=1, ctype=1, gid=1, tag="cm-inst")
            ),
            pb.Instruction(
                error_model=pb.ErrorModel(eid=1, etype=1, cid=1, tag="em-inst")
            ),
        ],
    )


def test_bin_strips_all_tags() -> None:
    lib = _sample_bin_library()
    stripped, count = strip_bin_library(lib)

    assert count == 16

    gt = stripped.gadget_types[0]
    assert gt.name == "MyGadget"
    assert gt.description == "gadget desc"
    assert all(m.tag == "" for m in gt.measurements)
    assert all(p.tag == "" for p in gt.inputs)
    assert all(p.tag == "" for p in gt.outputs)
    assert all(r.tag == "" for r in gt.readouts)

    pt = stripped.port_types[0]
    assert pt.name == "MyPort"
    assert pt.description == "port desc"
    assert all(o.tag == "" for o in pt.observables)

    cmt = stripped.check_model_types[0]
    assert cmt.name == "MyCheck"
    assert cmt.description == "check desc"
    assert all(rg.tag == "" for rg in cmt.remote_gadgets)
    assert all(c.tag == "" for c in cmt.checks)

    emt = stripped.error_model_types[0]
    assert emt.name == "MyError"
    assert emt.description == "error desc"
    assert all(rcm.tag == "" for rcm in emt.remote_check_models)
    assert all(e.tag == "" for e in emt.errors)

    for instr in stripped.program:
        if instr.HasField("gadget"):
            assert instr.gadget.tag == ""
        if instr.HasField("check_model"):
            assert instr.check_model.tag == ""
        if instr.HasField("error_model"):
            assert instr.error_model.tag == ""


def test_bin_preserves_description() -> None:
    lib = _sample_bin_library()
    stripped, _ = strip_bin_library(lib)
    assert stripped.description == "test library"


def test_bin_already_empty_tags() -> None:
    lib = pb.Library(
        gadget_types=[
            pb.GadgetType(gtype=1, measurements=[pb.GadgetType.Measurement()])
        ],
    )
    stripped, count = strip_bin_library(lib)
    assert count == 0
    assert len(stripped.gadget_types) == 1


# ── JIT tests ────────────────────────────────────────────────────────────


def _sample_jit_library() -> jit_pb.JitLibrary:
    return jit_pb.JitLibrary(
        description="jit lib",
        gadget_types=[
            jit_pb.JitGadgetType(
                base=pb.GadgetType(
                    gtype=1,
                    name="JitGadget",
                    description="jit gadget desc",
                    measurements=[pb.GadgetType.Measurement(tag="m0")],
                    inputs=[pb.GadgetType.Port(ptype=1, tag="in0")],
                    outputs=[pb.GadgetType.Port(ptype=1, tag="out0")],
                    readouts=[pb.GadgetType.Readout(tag="r0")],
                ),
                finished_checks=[
                    jit_pb.JitGadgetType.Check(
                        base=pb.CheckModelType.Check(tag="fc0"),
                    ),
                ],
                unfinished_checks=[
                    jit_pb.JitGadgetType.Check(
                        base=pb.CheckModelType.Check(tag="uc0"),
                    ),
                ],
                errors=[
                    jit_pb.JitGadgetType.Error(
                        base=pb.ErrorModelType.Error(tag="E0", probability=0.01),
                    ),
                ],
            ),
        ],
        port_types=[
            jit_pb.JitPortType(
                base=pb.PortType(
                    ptype=1,
                    name="JitPort",
                    description="jit port desc",
                    observables=[pb.PortType.Observable(tag="LX0")],
                ),
                k=0,
                stabilizers=[jit_pb.JitPortType.Stabilizer(tag="S0")],
            ),
        ],
    )


def test_jit_strips_all_tags() -> None:
    lib = _sample_jit_library()
    stripped, count = strip_jit_library(lib)

    assert count == 9

    gt = stripped.gadget_types[0]
    assert gt.base.name == "JitGadget"
    assert gt.base.description == "jit gadget desc"
    assert all(m.tag == "" for m in gt.base.measurements)
    assert all(p.tag == "" for p in gt.base.inputs)
    assert all(p.tag == "" for p in gt.base.outputs)
    assert all(r.tag == "" for r in gt.base.readouts)
    assert all(c.base.tag == "" for c in gt.finished_checks)
    assert all(c.base.tag == "" for c in gt.unfinished_checks)
    assert all(e.base.tag == "" for e in gt.errors)

    pt = stripped.port_types[0]
    assert pt.base.name == "JitPort"
    assert pt.base.description == "jit port desc"
    assert all(o.tag == "" for o in pt.base.observables)
    assert all(s.tag == "" for s in pt.stabilizers)


def test_jit_preserves_description() -> None:
    lib = _sample_jit_library()
    stripped, _ = strip_jit_library(lib)
    assert stripped.description == "jit lib"


# ── CLI integration ──────────────────────────────────────────────────────


def test_cli_jit_roundtrip() -> None:
    lib = _sample_jit_library()
    with tempfile.TemporaryDirectory() as tmp:
        infile = str(Path(tmp) / "test.deq.jit")
        outfile = str(Path(tmp) / "test.min.deq.jit")

        with open(infile, "wb") as f:
            f.write(lib.SerializeToString())

        strip_tags(infile, out=outfile)

        with open(outfile, "rb") as f:
            result = jit_pb.JitLibrary.FromString(f.read())

        assert all(e.base.tag == "" for e in result.gadget_types[0].errors)
        assert result.gadget_types[0].base.name == "JitGadget"


def test_cli_bin_roundtrip() -> None:
    lib = _sample_bin_library()
    with tempfile.TemporaryDirectory() as tmp:
        infile = str(Path(tmp) / "test.deq.bin")
        outfile = str(Path(tmp) / "test.min.deq.bin")

        with open(infile, "wb") as f:
            f.write(lib.SerializeToString())

        strip_tags(infile, out=outfile)

        with open(outfile, "rb") as f:
            result = pb.Library.FromString(f.read())

        assert all(e.tag == "" for e in result.error_model_types[0].errors)
        assert result.gadget_types[0].name == "MyGadget"
        assert result.description == "test library"
