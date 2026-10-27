"""07_window_partial_streaming.py — window decoding commits as the circuit grows.

The previous example ([06](06_jit_streaming_decode.py)) submitted
outcomes for an incomplete subgraph and showed the pending decode
resolved only after the rest of the circuit's outcomes arrived. That
could leave the impression that decoding cannot start until the full
circuit is loaded — true for the ``monolithic`` coordinator, but **not**
for the ``window`` coordinator.

With ``buffer_radius=1, lookahead_radius=0`` the window decoder commits a
gadget as soon as its 1-hop neighbourhood is fully analyzed. A gadget far
enough from the open frontier can commit even while the circuit is still
being extended downstream.

This script executes a partial chain ``PrepareZ → Idle → Idle → Idle``
(no ``MeasureZ``) and shows:

- ``prep``, ``idle1``, and ``idle2`` commit. Their 1-hop neighbours have
    outcomes, and the boundary buffer can use its terminal error model to
    project unfinished checks without waiting for future gadgets.
- ``idle3`` stays pending because radius one still requires a successor.

When the ``async with`` block exits, the runtime's shutdown propagates
cancellation into every pending decode — the leftover frontier tasks
resolve cleanly with a ``RuntimeError`` instead of leaking past the
event loop into interpreter shutdown.

Run with: ``python 07_window_partial_streaming.py``
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
    n_bytes = (num_bits + 7) // 8
    return coord_pb.Outcomes(
        gid=gid,
        outcomes=util_pb.BitVector(size=num_bits, data=bytes(n_bytes)),
    )


def _connector(gid: int) -> bin_pb.Gadget.Connector:
    return bin_pb.Gadget.Connector(gid=gid, port=0)


async def main() -> None:
    jit_library = build_jit_library(parse_file(REP_DEQ))
    gtype_for = {gt.base.name: gt.base.gtype for gt in jit_library.gadget_types}

    async with Runtime(
        decoder="black-box-relay-bp",
        coordinator="window",
        coordinator_config={"buffer_radius": 1, "lookahead_radius": 0},
        controller="jit",
    ) as runtime:
        jit = runtime.jit_controller
        await jit.load_library(jit_library)

        # Partial chain: PrepareZ -> Idle -> Idle -> Idle (NO MeasureZ).
        # idle3 is the open frontier — its output port has no downstream.
        await jit.execute(
            jit_pb.JitInstruction(gadget=bin_pb.Gadget(gtype=gtype_for["PrepareZ"], gid=1))
        )
        for gid in (2, 3, 4):
            await jit.execute(
                jit_pb.JitInstruction(
                    gadget=bin_pb.Gadget(
                        gtype=gtype_for["Idle"], gid=gid, connectors=[_connector(gid - 1)]
                    )
                )
            )
        print("Executed: prep(gid=1) → idle1(gid=2) → idle2(gid=3) → idle3(gid=4)")
        print("           (no MeasureZ — idle3 is the open frontier)")

        # Submit outcomes for every gadget.
        prep_decode = asyncio.create_task(jit.decode(_outcomes(gid=1, num_bits=0)))
        idle1_decode = asyncio.create_task(jit.decode(_outcomes(gid=2, num_bits=2)))
        idle2_decode = asyncio.create_task(jit.decode(_outcomes(gid=3, num_bits=2)))
        idle3_decode = asyncio.create_task(jit.decode(_outcomes(gid=4, num_bits=2)))
        print("Submitted decodes for prep, idle1, idle2, idle3.")

        prep_ro, idle1_ro, idle2_ro = await asyncio.wait_for(
            asyncio.gather(prep_decode, idle1_decode, idle2_decode), timeout=30
        )
        print(f"  prep   readouts.size = {prep_ro.readouts.size}  (committed)")
        print(f"  idle1  readouts.size = {idle1_ro.readouts.size}  (committed)")
        print(f"  idle2  readouts.size = {idle2_ro.readouts.size}  (committed)")
        assert not idle3_decode.done(), "idle3 should still be pending (frontier with dangling output)"
        print("idle3 remains pending: buffer_radius=1 requires a future neighbour.")

    # The `async with` exit fires the runtime's cancellation tokens, which
    # unblocks every pending decode with a Cancelled error. No explicit
    # cancel needed — the runtime handles partial-circuit shutdown for us.
    # Await the leftover tasks so we surface (and clear) the exceptions.
    try:
        await idle3_decode
        raise AssertionError("idle3 should have raised, not returned")
    except RuntimeError as error:
        print(f"After shutdown: idle3.decode raised: {type(error).__name__}")


if __name__ == "__main__":
    asyncio.run(main())
