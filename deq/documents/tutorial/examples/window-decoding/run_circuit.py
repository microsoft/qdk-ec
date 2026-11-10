"""Drive a JIT program with zero outcomes to illustrate window scheduling."""

# pylint: disable=no-member

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc

import deq.proto.deq_jit_pb2 as jit_pb
from deq.proto import coordinator_pb2, jit_controller_pb2_grpc, util_pb2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("--addr", default="localhost:50051")
    parser.add_argument("--stream-delay", type=float, default=0.0)
    args = parser.parse_args()
    if args.stream_delay < 0:
        parser.error("--stream-delay must be nonnegative")

    library = jit_pb.JitLibrary.FromString(args.library.read_bytes())
    instructions = list(library.program)
    if not instructions:
        parser.error("library has no program; transpile with --program NAME")
    measurement_counts = {
        gadget.base.gtype: len(gadget.base.measurements)
        for gadget in library.gadget_types
    }

    with grpc.insecure_channel(args.addr) as channel:
        grpc.channel_ready_future(channel).result(timeout=30)
        client = jit_controller_pb2_grpc.JitControllerStub(channel)

        def decode(instruction: jit_pb.JitInstruction) -> None:
            count = measurement_counts[instruction.gadget.gtype]
            outcomes = coordinator_pb2.Outcomes(
                gid=instruction.gadget.gid,
                outcomes=util_pb2.BitVector(
                    size=count,
                    data=bytes((count + 7) // 8),
                ),
            )
            client.Decode(outcomes, timeout=60)

        with ThreadPoolExecutor(max_workers=len(instructions)) as executor:
            pending = []
            if args.stream_delay == 0:
                for instruction in instructions:
                    client.Execute(instruction, timeout=30)
                pending = [
                    executor.submit(decode, instruction) for instruction in instructions
                ]
            else:
                for index, instruction in enumerate(instructions):
                    if index:
                        time.sleep(args.stream_delay)
                    client.Execute(instruction, timeout=30)
                    pending.append(executor.submit(decode, instruction))
            for future in pending:
                future.result()

        client.Reset(coordinator_pb2.ResetRequest(), timeout=30)

    print(f"Decoded {len(instructions)} gadgets; reset flushed the trace.")


if __name__ == "__main__":
    main()
