"""08_sampler.py — deq-native measurement sampling, end-to-end.

Some users want to drive a decoder with their own measurement outcomes
rather than have the runtime sample, decode, and check in one go. The
:class:`deq.runtime.Sampler` takes a ``.deq`` file plus the name of one of
its ``PROGRAM`` blocks, transpiles the program to a Stim circuit
internally, and produces per-gadget-partitioned shots ready to feed
straight into ``runtime.coordinator.decode`` /
``runtime.jit_controller.decode``.

The noise model is whatever is baked into the gadget bodies (``X_ERROR``,
``DEPOLARIZE1``, ``M(p)``, …) — no separate configuration needed; the
``.deq`` file is the source of truth.

This example samples the ``Simulation`` program from
``language/03_with_idle.deq`` (a distance-3 repetition-code memory
experiment: PrepareZ → Idle → MeasureZ), prints the per-gadget outcomes
for a few shots, then drives the in-process runtime to decode each
shot's outcomes through the JIT controller and prints the resulting
logical readouts.

Run with: ``python 08_sampler.py``
"""
import asyncio
from pathlib import Path

from deq.proto import coordinator_pb2 as coord_pb
from deq.proto import util_pb2 as util_pb
from deq.runtime import Runtime, Sampler

DEQ = Path(__file__).resolve().parent.parent / "language" / "03_with_idle.deq"


def _bits(bv: util_pb.BitVector) -> str:
    """Render a BitVector as an MSB-first binary string of length ``bv.size``.

    Matches the runtime's packing: bit ``i`` lives in ``data[i // 8]`` at
    position ``7 - (i % 8)``. Showing exactly ``size`` characters makes
    the per-gadget bit-count visible from the printed value (a 3-bit
    chunk reads ``'000'``, a 2-bit chunk reads ``'00'``), instead of
    hex which always rounds up to a byte.
    """
    return "".join(
        str((bv.data[i // 8] >> (7 - (i % 8))) & 1) for i in range(bv.size)
    )


async def main() -> None:
    sampler = Sampler(DEQ, program="Simulation", seed=42)
    print(f"sampler = {sampler!r}")
    print(f"  program       = {sampler.program}")
    print(f"  instructions  = {len(sampler.instructions)} gadgets")
    for i, instr in enumerate(sampler.instructions):
        gid = instr.gadget.gid
        # Look up the gadget type's name + measurement count for a readable line.
        gtype = next(
            gt for gt in sampler.library.gadget_types
            if gt.base.gtype == instr.gadget.gtype
        )
        print(f"    [{i}] gid={gid} {gtype.base.name:<8} ({len(gtype.base.measurements)} measurements)")

    print()
    shots = sampler.sample(num_shots=5)
    print(f"sampled {len(shots)} shots:")
    for i, shot in enumerate(shots):
        flat = _bits(shot.outcomes)
        per_gadget = [_bits(c) for c in sampler.split_outcomes(shot.outcomes)]
        print(f"  shot {i}: flat={flat!r}  per-gadget={per_gadget}")

    print()
    instructions = list(sampler.instructions)
    async with Runtime(
        decoder="black-box-relay-bp",
        coordinator="monolithic",
        controller="jit",
    ) as runtime:
        jit = runtime.jit_controller
        await jit.load_library(sampler.library)

        print(f"decoded {len(shots)} shots through runtime.jit_controller:")
        for i, shot in enumerate(shots):
            await jit.reset()
            await jit.batch_execute(instructions)

            chunks = sampler.split_outcomes(shot.outcomes)
            readouts = await jit.batch_decode([
                coord_pb.Outcomes(gid=instr.gadget.gid, outcomes=chunk)
                for instr, chunk in zip(instructions, chunks)
            ])
            print(f"  shot {i}: readouts={[_bits(r.readouts) for r in readouts]}")


if __name__ == "__main__":
    asyncio.run(main())
