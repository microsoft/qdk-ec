"""01_hello_runtime.py — minimal Runtime lifecycle, coordinator interface.

The point of this script is to see the basic lifecycle of an in-process
Runtime: construct it, load a library, inspect it, shut it down. We
deliberately stop short of ``execute`` / ``decode`` here — those come in
``03_jit_dynamic_circuit.py``, where the JIT controller makes building
instructions easy.

We use the **coordinator** alone — the always-present, lower-level
``deq.bin`` interface. To feed it a library, we run the same JIT
compilation the controller would do internally, but eagerly via
``static_jit_compiler``: parse the source ``.deq`` file, build a
``JitLibrary``, and compile that down to a ``deq.bin.Library``. The
runtime then accepts the compiled library directly.

Run with: ``python 01_hello_runtime.py``
"""
import asyncio
from pathlib import Path

from deq.circuit.parser import parse_file
from deq.compiler.jit_compiler import static_jit_compiler
from deq.runtime import Runtime
from deq.transpiler.jit_library_builder import build_jit_library

# Reuse the repetition-code example from the intro chapter.
REP_DEQ = Path(__file__).resolve().parent.parent / "intro" / "small_example.deq"


async def main() -> None:
    # Build a JitLibrary in memory, then compile it down to a deq.bin.Library
    # via the static JIT compiler. The runtime's coordinator speaks deq.bin.
    jit_library = build_jit_library(parse_file(REP_DEQ))
    bin_library = static_jit_compiler(jit_library)
    print(
        f"Compiled bin.Library: {len(bin_library.gadget_types)} gadget types, "
        f"{len(bin_library.port_types)} port types"
    )

    # `async with` guarantees `shutdown()` runs on exit — no leaked
    # background tasks or gRPC ports.
    async with Runtime(
        decoder="black-box-relay-bp",
        coordinator="monolithic",
    ) as runtime:
        print(repr(runtime))
        print(f"  has_jit_controller = {runtime.has_jit_controller()}")
        print(f"  bound_port         = {runtime.bound_port()}")

        await runtime.coordinator.load_library(bin_library)
        print("loaded library via runtime.coordinator")


if __name__ == "__main__":
    asyncio.run(main())
