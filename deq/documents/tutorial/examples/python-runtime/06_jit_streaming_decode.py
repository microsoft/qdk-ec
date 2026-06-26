"""06_jit_streaming_decode.py — decode is streaming, not call-and-return.

Real QEC decoders are not request-response: feeding in one gadget's
measurement outcomes does **not** immediately give you a readout. The
decoder needs a *window* of context — the syndromes of nearby gadgets —
before it can commit a result. This is fundamental to streaming/windowed
decoding, not an artefact of our wrapper.

This example switches to the **window coordinator** (the same coordinator a
real LER simulation would use) and walks through the consequence:

1. Execute a chain `PrepareZ → Idle → Idle → MeasureZ`.
2. Submit outcomes for the *first* Idle only and watch its `decode` sit
   pending — the window decoder cannot commit without seeing the
   neighbours' syndromes too.
3. Submit outcomes for the remaining gadgets. The window now has enough
   context; every pending decode completes and we recover the logical
   readout from the final `MeasureZ`.

A subtle but important detail: **every** gadget that the controller has
executed must have `decode()` called on it before the subgraph can commit
— including gadgets with zero measurements such as `PrepareZ`. The
coordinator tracks per-gadget completion, not measurement count.

Run with: ``python 06_jit_streaming_decode.py``
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

REP_DEQ = Path(__file__).resolve().parent.parent / "intro" / "small_example.deq"


def _outcomes(gid: int, num_bits: int) -> coord_pb.Outcomes:
    """Zero-bit measurement outcomes of length `num_bits` for `gid`."""
    n_bytes = max(1, (num_bits + 7) // 8) if num_bits > 0 else 0
    return coord_pb.Outcomes(
        gid=gid,
        outcomes=util_pb.BitVector(size=num_bits, data=bytes(n_bytes)),
    )


def _connector(gid: int) -> bin_pb.Gadget.Connector:
    return bin_pb.Gadget.Connector(gid=gid, port=0)


async def main() -> None:
    jit_library = build_jit_library(parse_file(REP_DEQ))
    gtype_for = {gt.base.name: gt.base.gtype for gt in jit_library.gadget_types}

    # Use the window coordinator — the same one a real LER simulation would
    # configure via ``--coordinator window``. `buffer_radius=1` means each
    # gadget needs 1 hop of context before it can be committed.
    async with Runtime(
        decoder="black-box-relay-bp",
        coordinator="window",
        coordinator_config={"buffer_radius": 1, "lookahead_radius": 0},
        controller="jit",
    ) as runtime:
        jit = runtime.jit_controller
        await jit.load_library(jit_library)

        # Chain: PrepareZ → Idle → Idle → MeasureZ.
        await jit.execute(jit_pb.JitInstruction(gadget=bin_pb.Gadget(gtype=gtype_for["PrepareZ"], gid=1)))
        await jit.execute(
            jit_pb.JitInstruction(
                gadget=bin_pb.Gadget(gtype=gtype_for["Idle"], gid=2, connectors=[_connector(1)])
            )
        )
        await jit.execute(
            jit_pb.JitInstruction(
                gadget=bin_pb.Gadget(gtype=gtype_for["Idle"], gid=3, connectors=[_connector(2)])
            )
        )
        await jit.execute(
            jit_pb.JitInstruction(
                gadget=bin_pb.Gadget(gtype=gtype_for["MeasureZ"], gid=4, connectors=[_connector(3)])
            )
        )
        print("Executed: prep(gid=1) → idle1(gid=2) → idle2(gid=3) → measure(gid=4)")

        # Step 1: submit ONLY idle1's outcomes. Spawn the decode as a task so
        # we can observe that it does not complete on its own.
        print("\nSubmitting outcomes for idle1 only (gid=2)...")
        idle1_decode = asyncio.create_task(jit.decode(_outcomes(gid=2, num_bits=2)))

        # Wait a full second. If a real decoder behaved like a request-response
        # API, idle1_decode would be done by now. It is not.
        await asyncio.sleep(1.0)
        assert not idle1_decode.done(), (
            "idle1's decode should still be pending — the window decoder "
            "needs to see neighbouring syndromes before it can commit."
        )
        print("  After 1s: idle1's decode is still pending. The window")
        print("  decoder cannot commit a gadget without seeing the syndromes")
        print("  of its neighbours.")

        # Step 2: submit outcomes for the rest. Even PrepareZ — which has
        # zero physical measurements — needs `decode()` called so the
        # coordinator knows the subgraph is fully loaded.
        print("\nSubmitting outcomes for prep, idle2, and measure...")
        prep_decode = asyncio.create_task(jit.decode(_outcomes(gid=1, num_bits=0)))
        idle2_decode = asyncio.create_task(jit.decode(_outcomes(gid=3, num_bits=2)))
        meas_decode = asyncio.create_task(jit.decode(_outcomes(gid=4, num_bits=3)))

        # With every gadget in the subgraph having submitted outcomes, the
        # window can commit. All four pending decodes complete.
        prep_ro, idle1_ro, idle2_ro, meas_ro = await asyncio.gather(
            prep_decode, idle1_decode, idle2_decode, meas_decode
        )
        print("  prep   readouts.size =", prep_ro.readouts.size)
        print("  idle1  readouts.size =", idle1_ro.readouts.size)
        print("  idle2  readouts.size =", idle2_ro.readouts.size)
        print("  meas   readouts.size =", meas_ro.readouts.size, "  (the logical Z bit)")


if __name__ == "__main__":
    asyncio.run(main())
