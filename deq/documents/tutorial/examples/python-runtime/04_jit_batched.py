"""04_jit_batched.py — batched JIT execute and decode.

For programs where many gadget instances are known up front (e.g. a planned
syndrome-extraction round across many logical qubits), `batch_execute` and
`batch_decode` are more efficient than looping over single-shot calls: the
runtime schedules all of them on its tokio worker pool in one go, respecting
connector dependencies for compilation.

This script runs 3 parallel logical qubits, each going through one
PrepareZ → Idle → MeasureZ chain, then decodes the MeasureZ gadgets
concurrently.

Run with: ``python 04_jit_batched.py``
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

        # Three parallel chains: PrepareZ → Idle → MeasureZ for each. We
        # need MeasureZ to close each chain so the Idle gadget's unfinished
        # checks have something downstream to consume them (otherwise
        # `decode` on the Idle would block forever waiting for an error
        # model that can never resolve).
        #
        # batch_execute respects connector dependencies, so the order in
        # which we list the instructions doesn't matter as long as every
        # connector eventually points to an instruction in the same batch
        # (or to a previously-executed gadget).
        instructions: list[jit_pb.JitInstruction] = []
        prep_gids: list[int] = []
        idle_gids: list[int] = []
        meas_gids: list[int] = []
        next_gid = 1
        for _ in range(3):
            prep = next_gid
            idle = next_gid + 1
            meas = next_gid + 2
            next_gid += 3
            instructions.extend(
                [
                    jit_pb.JitInstruction(
                        gadget=bin_pb.Gadget(gtype=gtype_for["PrepareZ"], gid=prep)
                    ),
                    jit_pb.JitInstruction(
                        gadget=bin_pb.Gadget(
                            gtype=gtype_for["Idle"],
                            gid=idle,
                            connectors=[bin_pb.Gadget.Connector(gid=prep, port=0)],
                        )
                    ),
                    jit_pb.JitInstruction(
                        gadget=bin_pb.Gadget(
                            gtype=gtype_for["MeasureZ"],
                            gid=meas,
                            connectors=[bin_pb.Gadget.Connector(gid=idle, port=0)],
                        )
                    ),
                ]
            )
            prep_gids.append(prep)
            idle_gids.append(idle)
            meas_gids.append(meas)

        # One call, everything compiled and submitted at once.
        assigned = await jit.batch_execute(instructions)
        print(f"batch_execute returned {len(assigned)} gids in input order")

        # The monolithic coordinator only commits a connected subgraph once
        # every gadget in it has had `decode()` called. So we feed in
        # outcomes for *all* nine gadgets — preps with zero bits, idles
        # with 2 bits, measures with 3 bits — in a single batch.
        # batch_decode preserves input order in its return value.
        all_outcomes = (
            [
                coord_pb.Outcomes(gid=g, outcomes=util_pb.BitVector(size=0, data=b""))
                for g in prep_gids
            ]
            + [
                coord_pb.Outcomes(gid=g, outcomes=util_pb.BitVector(size=2, data=b"\x00"))
                for g in idle_gids
            ]
            + [
                coord_pb.Outcomes(gid=g, outcomes=util_pb.BitVector(size=3, data=b"\x00"))
                for g in meas_gids
            ]
        )
        readouts_list = await jit.batch_decode(all_outcomes)
        prep_readouts = readouts_list[: len(prep_gids)]
        idle_readouts = readouts_list[len(prep_gids) : len(prep_gids) + len(idle_gids)]
        meas_readouts = readouts_list[len(prep_gids) + len(idle_gids) :]
        print(f"batch_decode returned {len(readouts_list)} readouts ({len(prep_readouts)} prep, "
              f"{len(idle_readouts)} idle, {len(meas_readouts)} measure)")
        for gid, ro in zip(meas_gids, meas_readouts):
            print(f"  MeasureZ gid={gid} -> readout_size={ro.readouts.size}")


if __name__ == "__main__":
    asyncio.run(main())
