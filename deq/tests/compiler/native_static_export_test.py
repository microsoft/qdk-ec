"""Regression coverage for the native materialized and compact export APIs."""

import deq_runtime
import pytest

from deq.proto import deq_bin_pb2 as pb
from deq.proto import deq_jit_pb2 as jit_pb
from deq.transpiler.program_timing import TIMING_METADATA_KEY, Timing, set_gadget_timing


@pytest.mark.parametrize("count", [4, 70])
@pytest.mark.parametrize("timed", [False, True])
def test_native_static_export_layout_is_opt_in(count: int, timed: bool) -> None:
    source = jit_pb.JitLibrary(
        port_types=[
            jit_pb.JitPortType(
                base=pb.PortType(
                    ptype=1, metadata={"custom.port_type": ["a", None]},
                    observables=[pb.PortType.Observable(metadata={"custom.observable": True})],
                ),
                stabilizers=[jit_pb.JitPortType.Stabilizer(metadata={"custom.stabilizer": "kept in JIT"})],
            )
        ],
        gadget_types=[jit_pb.JitGadgetType(
            base=pb.GadgetType(
                gtype=1, metadata={"custom.gadget_type": {"value": 7}},
                measurements=[pb.GadgetType.Measurement(metadata={"custom.measurement": "m"})],
                readouts=[pb.GadgetType.Readout(
                    measurement_indices=[0], metadata={"custom.readout": "r"}
                )],
            ),
            finished_checks=[jit_pb.JitGadgetType.Check(
                base=pb.CheckModelType.Check(metadata={"custom.check": "c"}),
                measurements=[jit_pb.JitGadgetType.PresentMeasurement(measurement_index=0)],
            )],
            errors=[jit_pb.JitGadgetType.Error(
                base=pb.ErrorModelType.Error(probability=0.1, metadata={"custom.error": "e"}),
                finished_checks=[0],
            )],
        )],
    )
    source.metadata.update({"source": "native-export-regression"})
    for gid in range(1, count + 1):
        gadget = source.program.add().gadget
        gadget.gtype = 1
        gadget.gid = gid
        gadget.metadata.update({"custom.instance": {"id": str(gid), "notes": [None, True]}})
        if timed:
            set_gadget_timing(gadget, Timing(gid * 10, 5))

    serialized = source.SerializeToString()
    for compile_library, expected_type_count in [
        (deq_runtime.static_jit_compile, count),
        (deq_runtime.static_jit_compile_compact, 1),
    ]:
        encoded = compile_library(jit_library=serialized)
        assert isinstance(encoded, bytes)
        compiled = pb.Library.FromString(encoded)
        assert len(compiled.check_model_types) == expected_type_count
        assert len(compiled.error_model_types) == expected_type_count
        assert compiled.metadata == source.metadata
        assert compiled.port_types[0] == source.port_types[0].base
        assert compiled.gadget_types[0] == source.gadget_types[0].base
        assert all(
            model.checks[0].metadata == source.gadget_types[0].finished_checks[0].base.metadata
            for model in compiled.check_model_types
        )
        assert all(
            model.errors[0].metadata == source.gadget_types[0].errors[0].base.metadata
            for model in compiled.error_model_types
        )
        gadgets = [
            instruction.gadget
            for instruction in compiled.program
            if instruction.WhichOneof("create") == "gadget"
        ]
        assert gadgets == [instruction.gadget for instruction in source.program]
        if expected_type_count == count:
            assert [kind.ctype for kind in compiled.check_model_types] == list(
                range(1, count + 1)
            )
            assert [kind.etype for kind in compiled.error_model_types] == list(
                range(1, count + 1)
            )


def test_native_compilation_does_not_interpret_simulator_metadata():
    source = jit_pb.JitLibrary(
        gadget_types=[jit_pb.JitGadgetType(base=pb.GadgetType(gtype=1))],
        program=[jit_pb.JitInstruction(gadget=pb.Gadget(
            gtype=1, metadata={TIMING_METADATA_KEY: "not a simulator schedule"},
        ))],
    )
    compiled = pb.Library.FromString(deq_runtime.static_jit_compile(source.SerializeToString()))
    assert compiled.program[0].gadget.metadata == source.program[0].gadget.metadata
