"""02_concurrent_decode.py — concurrent decode via asyncio.gather.

Demonstrates that multiple in-flight `decode` calls actually overlap on
the runtime's worker pool — they don't get serialized by Python's GIL.

We set up several independent PrepareZ → Idle → MeasureZ chains and
``asyncio.gather`` over all of their decodes. Each chain is a separate
connected subgraph from the monolithic coordinator's point of view, so it
commits each chain independently as the corresponding outcomes are
submitted.

Run with: ``python 02_concurrent_decode.py``
"""
import asyncio
import time
from pathlib import Path

from deq.circuit.parser import parse_file
from deq.proto import coordinator_pb2 as coord_pb
from deq.proto import deq_bin_pb2 as bin_pb
from deq.proto import deq_jit_pb2 as jit_pb
from deq.proto import util_pb2 as util_pb
from deq.runtime import Runtime
from deq.transpiler.jit_library_builder import build_jit_library

REP_DEQ = Path(__file__).resolve().parent.parent / "intro" / "small_example.deq"
NUM_CHAINS = 8


def _connector(gid: int) -> bin_pb.Gadget.Connector:
    return bin_pb.Gadget.Connector(gid=gid, port=0)


async def main() -> None:
    jit_library = build_jit_library(parse_file(REP_DEQ))
    gtype_for = {gt.base.name: gt.base.gtype for gt in jit_library.gadget_types}

    async with Runtime(
        decoder="black-box-relay-bp",
        coordinator="monolithic",
        controller="jit",
    ) as runtime:
        jit = runtime.jit_controller
        await jit.load_library(jit_library)

        # Build NUM_CHAINS independent prep → idle → measure chains. Each
        # chain is its own connected subgraph; the monolithic coordinator
        # commits each one as soon as its three decodes arrive.
        instructions: list[jit_pb.JitInstruction] = []
        prep_gids, idle_gids, meas_gids = [], [], []
        next_gid = 1
        for _ in range(NUM_CHAINS):
            prep, idle, meas = next_gid, next_gid + 1, next_gid + 2
            next_gid += 3
            instructions += [
                jit_pb.JitInstruction(
                    gadget=bin_pb.Gadget(gtype=gtype_for["PrepareZ"], gid=prep)
                ),
                jit_pb.JitInstruction(
                    gadget=bin_pb.Gadget(
                        gtype=gtype_for["Idle"],
                        gid=idle,
                        connectors=[_connector(prep)],
                    )
                ),
                jit_pb.JitInstruction(
                    gadget=bin_pb.Gadget(
                        gtype=gtype_for["MeasureZ"],
                        gid=meas,
                        connectors=[_connector(idle)],
                    )
                ),
            ]
            prep_gids.append(prep)
            idle_gids.append(idle)
            meas_gids.append(meas)

        await jit.batch_execute(instructions)
        print(f"executed {NUM_CHAINS} chains ({3 * NUM_CHAINS} gadgets)")

        # Dispatch every gadget's decode in one gather. The runtime
        # schedules them all on its tokio worker pool simultaneously; each
        # subgraph commits as soon as its three decodes arrive.
        all_outcomes = (
            [coord_pb.Outcomes(gid=g, outcomes=util_pb.BitVector(size=0, data=b""))
             for g in prep_gids]
            + [coord_pb.Outcomes(gid=g, outcomes=util_pb.BitVector(size=2, data=b"\x00"))
               for g in idle_gids]
            + [coord_pb.Outcomes(gid=g, outcomes=util_pb.BitVector(size=3, data=b"\x00"))
               for g in meas_gids]
        )
        start = time.perf_counter()
        results = await asyncio.gather(
            *[jit.decode(o) for o in all_outcomes]
        )
        elapsed = time.perf_counter() - start
        print(f"decoded {len(results)} gadgets concurrently in {elapsed * 1000:.1f} ms")

        # gather() preserves input order, so the last NUM_CHAINS results
        # are the MeasureZ readouts (the logical bits).
        measure_readouts = results[-NUM_CHAINS:]
        assert all(r.readouts.size == 1 for r in measure_readouts)
        print(f"  all {NUM_CHAINS} MeasureZ gadgets returned 1-bit readouts")


if __name__ == "__main__":
    asyncio.run(main())
