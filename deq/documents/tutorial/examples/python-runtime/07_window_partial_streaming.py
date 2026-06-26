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

- ``prep`` and ``idle1`` commit. Their 1-hop neighbours all have
  connected output ports, so every error model in their windows loads
  (the JIT compiler holds a gadget's error model open until its outputs
  are connected) and the window decoder gets the syndromes it needs.
- ``idle2`` and ``idle3`` stay pending until shutdown. ``idle3`` is
  the frontier — its output is dangling, so its error model never loads.
  ``idle2`` has ``idle3`` in its window, so it inherits the wait.

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

        # prep and idle1 both have a 1-hop neighbourhood where every gadget
        # has its outputs connected (prep→idle1; idle1→idle2). The JIT
        # compiler's error-model futures all resolve, decode_single passes
        # them to the coordinator, and the window committer fires.
        prep_ro, idle1_ro = await asyncio.gather(prep_decode, idle1_decode)
        print(f"  prep   readouts.size = {prep_ro.readouts.size}  (committed)")
        print(f"  idle1  readouts.size = {idle1_ro.readouts.size}  (committed)")

        # idle3 is the frontier: its output is dangling, so the JIT compiler
        # never resolves its error-model future, and decode_single blocks
        # inside the JIT controller. idle2's window {idle1, idle2, idle3}
        # needs idle3's syndrome too, so it inherits the wait.
        await asyncio.sleep(1.0)
        assert not idle2_decode.done(), "idle2 should still be pending (window reaches the frontier)"
        assert not idle3_decode.done(), "idle3 should still be pending (frontier with dangling output)"
        print("After 1s: idle2 and idle3 are still pending — both wait on the open frontier.")

    # The `async with` exit fires the runtime's cancellation tokens, which
    # unblocks every pending decode with a Cancelled error. No explicit
    # cancel needed — the runtime handles partial-circuit shutdown for us.
    # Await the leftover tasks so we surface (and clear) the exceptions.
    for name, task in [("idle2", idle2_decode), ("idle3", idle3_decode)]:
        try:
            await task
            raise AssertionError(f"{name} should have raised, not returned")
        except RuntimeError as e:
            print(f"After shutdown: {name}.decode raised: {type(e).__name__}")


if __name__ == "__main__":
    asyncio.run(main())
