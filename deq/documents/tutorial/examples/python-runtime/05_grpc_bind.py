"""05_grpc_bind.py — also expose the in-process runtime as a gRPC server.

When you want Python to drive the runtime AND external clients (e.g. another
process, a separate machine, the existing `deq` CLI) to talk to the same
services, call `await runtime.bind(addr)`. The same `Runtime` instance now
serves both surfaces — in-process Python calls go through the in-process
`Local` clients (no tonic, no TCP), while gRPC clients hit the tonic router
on the bound socket.

Run with: ``python 05_grpc_bind.py``
"""
import asyncio
from pathlib import Path

from deq.circuit.parser import parse_file
from deq.runtime import Runtime
from deq.transpiler.jit_library_builder import build_jit_library

REP_DEQ = Path(__file__).resolve().parent.parent / "intro" / "small_example.deq"


async def main() -> None:
    jit_library = build_jit_library(parse_file(REP_DEQ))

    async with Runtime(
        decoder="black-box-relay-bp",
        coordinator="monolithic",
        controller="jit",
    ) as runtime:
        # Bind to an OS-chosen port. Pass an explicit address like
        # "127.0.0.1:50051" or "[::]:50051" for a fixed port.
        url = await runtime.bind("[::]:0")
        print(f"gRPC server bound at {url}")
        print(f"  bound_port = {runtime.bound_port()}")

        # In-process calls still go through the Local clients (no tonic,
        # no TCP); gRPC clients connect via the URL above.
        await runtime.jit_controller.load_library(jit_library)
        print("loaded library via the in-process JIT controller")

        # An external client (any language with gRPC + the deq protos) could
        # connect to `url` now and issue the same RPCs.

    # The async context manager triggers shutdown(); the gRPC serve loop
    # observes the shutdown signal and the runtime exits cleanly.
    print("runtime shut down, gRPC port released")


if __name__ == "__main__":
    asyncio.run(main())
