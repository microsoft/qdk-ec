"""03_jit_dynamic_circuit.py — dynamic logical circuit via the JIT controller.

This is the workflow most users actually want: a `.deq` source file describing
gadget types, transpiled into a `JitLibrary` in memory, loaded into the
runtime, and then driven instruction-by-instruction from Python.

The library reused here is the same `small_example.deq` from the intro
chapter: a distance-3 repetition code with `PrepareZ`, `Idle`, `MeasureZ`.

Run with: ``python 03_jit_dynamic_circuit.py``
"""
import asyncio
from pathlib import Path

from deq.circuit.parser import parse_file
from deq.proto import coordinator_pb2 as coord_pb
from deq.proto import deq_bin_pb2 as bin_pb
from deq.proto import deq_jit_pb2 as jit_pb
from deq.proto import util_pb2 as util_pb
from deq.runtime import Runtime
from deq.transpiler.jit_library_builder import build_jit_library

# Reuse the repetition-code example from the intro chapter.
THIS_DIR = Path(__file__).resolve().parent
REP_DEQ = THIS_DIR.parent / "intro" / "small_example.deq"


def _connector(gid: int) -> bin_pb.Gadget.Connector:
    """A connector that wires an input port to the output of `gid`'s port 0."""
    return bin_pb.Gadget.Connector(gid=gid, port=0)


def _make_instruction(*, gtype: int, gid: int, connectors=()) -> jit_pb.JitInstruction:
    """Build a JitInstruction that instantiates one gadget."""
    return jit_pb.JitInstruction(
        gadget=bin_pb.Gadget(gtype=gtype, gid=gid, connectors=list(connectors))
    )


async def main() -> None:
    # Stage 1: build a JitLibrary from the .deq source. This is the same
    # work `deq transpile` would do, just done in process so we can keep
    # the result as a proto message rather than a file.
    print(f"Transpiling {REP_DEQ.name}...")
    jit_library = build_jit_library(parse_file(REP_DEQ))
    gadget_names = {gt.base.gtype: gt.base.name for gt in jit_library.gadget_types}
    print(f"  {len(jit_library.gadget_types)} gadget types: {sorted(gadget_names.values())}")

    # We need the gtypes for the three gadgets we are going to stream.
    gtype_for = {name: gt for gt, name in gadget_names.items()}
    g_prep = gtype_for["PrepareZ"]
    g_idle = gtype_for["Idle"]
    g_meas = gtype_for["MeasureZ"]

    # Stage 2: bring up an in-process runtime with the JIT controller and a
    # *real* coordinator. The monolithic coordinator decodes a connected
    # subgraph once every gadget in it has received outcomes; the naive
    # decoder is fine here because we're only exercising the runtime
    # mechanics, not measuring logical error rates.
    async with Runtime(
        decoder="black-box-relay-bp",
        coordinator="monolithic",
        controller="jit",
    ) as runtime:
        jit = runtime.jit_controller

        # Stage 3: register the library. The JIT compiler stores the
        # `JitGadgetType`s, and the underlying `bin` port/gadget types are
        # forwarded to the coordinator so subsequent `execute` calls can
        # resolve them.
        await jit.load_library(jit_library)
        print(f"Loaded JitLibrary into {runtime!r}")

        # Stage 4: stream the dynamic program one instruction at a time.
        # In a real online setting these instructions would be produced by
        # whatever software is choreographing the logical circuit; here we
        # just write them out explicitly.
        prep_gid = await jit.execute(_make_instruction(gtype=g_prep, gid=1))
        print(f"  PrepareZ -> gid={prep_gid}")

        idle_gid = await jit.execute(
            _make_instruction(gtype=g_idle, gid=2, connectors=[_connector(prep_gid)])
        )
        print(f"  Idle     -> gid={idle_gid}")

        meas_gid = await jit.execute(
            _make_instruction(gtype=g_meas, gid=3, connectors=[_connector(idle_gid)])
        )
        print(f"  MeasureZ -> gid={meas_gid}")

        # Stage 5: feed in measurement outcomes and read back logical
        # readouts. A real coordinator commits the entire connected
        # subgraph at once, so we must call `decode` on *every* gadget in
        # the program — including PrepareZ, which has zero physical
        # measurements. We submit them concurrently via `asyncio.gather`
        # so they all sit pending on the runtime's worker pool together;
        # the coordinator commits once the last one arrives.
        prep_outcomes = coord_pb.Outcomes(
            gid=prep_gid, outcomes=util_pb.BitVector(size=0, data=b"")
        )
        idle_outcomes = coord_pb.Outcomes(
            gid=idle_gid, outcomes=util_pb.BitVector(size=2, data=b"\x00")
        )
        meas_outcomes = coord_pb.Outcomes(
            gid=meas_gid, outcomes=util_pb.BitVector(size=3, data=b"\x00")
        )
        prep_ro, idle_ro, meas_ro = await asyncio.gather(
            jit.decode(prep_outcomes),
            jit.decode(idle_outcomes),
            jit.decode(meas_outcomes),
        )
        print(f"PrepareZ readouts: size={prep_ro.readouts.size}")
        print(f"Idle     readouts: size={idle_ro.readouts.size}")
        print(f"MeasureZ readouts: size={meas_ro.readouts.size}  (the logical Z bit)")


if __name__ == "__main__":
    asyncio.run(main())
