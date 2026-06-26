"""Integration tests for the in-process `deq.runtime.Runtime` wrapper.

Exercises the PyO3 bindings end-to-end. Covers:

- runtime lifecycle (bind / shutdown / context-manager),
- the namespaced ``runtime.coordinator`` (deq.bin) interface,
- the namespaced ``runtime.jit_controller`` (deq.jit) interface,
- raw-bytes pass-through and parsed-proto round-trip,
- concurrency.
"""

from __future__ import annotations

import asyncio
import pytest

from deq.proto import coordinator_pb2 as coord_pb
from deq.proto import deq_bin_pb2 as bin_pb
from deq.proto import deq_jit_pb2 as jit_pb
from deq.proto import util_pb2 as util_pb
from deq.runtime import (
    Coordinator,
    JitController,
    RawCoordinator,
    RawJitController,
    RawRuntime,
    Runtime,
)


# ── helpers ─────────────────────────────────────────────────────────────────


def _library_with_one_gadget_type(gtype: int = 1, readouts: int = 4) -> bin_pb.Library:
    """Build a minimal `deq.bin.Library` the naive coordinator will accept."""
    gadget_type = bin_pb.GadgetType(gtype=gtype, name="probe")
    for index in range(readouts):
        gadget_type.readouts.add(tag=f"r{index}")
    return bin_pb.Library(description="rt-test", gadget_types=[gadget_type])


# ── Runtime lifecycle ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_runtime_lifecycle_and_bind():
    runtime = Runtime(decoder="black-box-naive", coordinator="naive")
    assert runtime.bound_port() is None
    assert "unbound" in repr(runtime)

    url = await runtime.bind("[::]:0")
    assert url.startswith("http://")
    port = runtime.bound_port()
    assert port is not None and port > 0
    assert str(port) in url

    await runtime.shutdown()
    # shutdown is idempotent
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_runtime_async_context_manager_shuts_down():
    async with Runtime(decoder="black-box-naive", coordinator="naive") as runtime:
        await runtime.bind("[::]:0")
        assert runtime.bound_port() is not None
    # After context exit, shutdown ran; bound_port should still be None or
    # represent the finished state — what matters is that the next shutdown
    # is a no-op.
    await runtime.shutdown()


# ── Coordinator interface ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_coordinator_load_execute_decode_typed():
    async with Runtime(decoder="black-box-naive", coordinator="naive") as runtime:
        coord = runtime.coordinator
        assert isinstance(coord, Coordinator)

        await coord.load_library(_library_with_one_gadget_type(gtype=1, readouts=4))

        gid = await coord.execute(bin_pb.Instruction(gadget=bin_pb.Gadget(gtype=1, gid=100)))
        assert gid == 100

        readouts = await coord.decode(
            coord_pb.Outcomes(gid=gid, outcomes=util_pb.BitVector(size=0))
        )
        assert isinstance(readouts, coord_pb.Readouts)
        assert readouts.gid == gid
        assert readouts.readouts.size == 4


@pytest.mark.asyncio
async def test_coordinator_decode_raw_bytes_passthrough():
    async with Runtime(decoder="black-box-naive", coordinator="naive") as runtime:
        coord = runtime.coordinator
        await coord.load_library(
            _library_with_one_gadget_type(gtype=2, readouts=2).SerializeToString()
        )
        gid = await coord.execute(
            bin_pb.Instruction(gadget=bin_pb.Gadget(gtype=2, gid=7)).SerializeToString()
        )
        raw = await coord.decode(
            coord_pb.Outcomes(gid=gid, outcomes=util_pb.BitVector(size=0)),
            raw=True,
        )
        assert isinstance(raw, (bytes, bytearray))
        parsed = coord_pb.Readouts()
        parsed.ParseFromString(raw)
        assert parsed.gid == gid


@pytest.mark.asyncio
async def test_coordinator_reset_clears_state():
    runtime = Runtime(decoder="black-box-naive", coordinator="naive")
    try:
        coord = runtime.coordinator
        await coord.load_library(_library_with_one_gadget_type(gtype=3, readouts=1))
        await coord.execute(bin_pb.Instruction(gadget=bin_pb.Gadget(gtype=3, gid=1)))
        await coord.reset(reset_library=True)
        with pytest.raises(RuntimeError, match="gtype=3"):
            await coord.execute(bin_pb.Instruction(gadget=bin_pb.Gadget(gtype=3, gid=2)))
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_coordinator_concurrent_decodes():
    """Multiple decode requests should overlap on the tokio runtime."""
    async with Runtime(decoder="black-box-naive", coordinator="naive") as runtime:
        coord = runtime.coordinator
        await coord.load_library(_library_with_one_gadget_type(gtype=4, readouts=8))
        gids = await asyncio.gather(
            *[
                coord.execute(bin_pb.Instruction(gadget=bin_pb.Gadget(gtype=4, gid=i)))
                for i in range(10, 15)
            ]
        )
        assert sorted(gids) == [10, 11, 12, 13, 14]

        readouts_list = await asyncio.gather(
            *[
                coord.decode(coord_pb.Outcomes(gid=g, outcomes=util_pb.BitVector(size=0)))
                for g in gids
            ]
        )
        assert [r.gid for r in readouts_list] == list(gids)
        assert all(r.readouts.size == 8 for r in readouts_list)


# ── JIT controller interface ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_jit_controller_unavailable_when_not_configured():
    async with Runtime(decoder="black-box-naive", coordinator="naive") as runtime:
        assert runtime.has_jit_controller() is False
        with pytest.raises(AttributeError, match="jit_controller"):
            _ = runtime.jit_controller


@pytest.mark.asyncio
async def test_jit_controller_available_when_configured():
    async with Runtime(
        decoder="black-box-naive",
        coordinator="monolithic",
        controller="jit",
    ) as runtime:
        assert runtime.has_jit_controller() is True
        jit = runtime.jit_controller
        assert isinstance(jit, JitController)
        # The accessor returns the same wrapper each time.
        assert runtime.jit_controller is jit


@pytest.mark.asyncio
async def test_jit_controller_load_empty_library_and_reset():
    """Empty JitLibrary load and reset round-trip across the FFI boundary."""
    async with Runtime(
        decoder="black-box-naive",
        coordinator="monolithic",
        controller="jit",
    ) as runtime:
        jit = runtime.jit_controller
        await jit.load_library(jit_pb.JitLibrary(description="empty"))
        # batch_execute with no instructions is a valid no-op.
        gids = await jit.batch_execute([])
        assert gids == []
        gids_raw = await jit.batch_execute([])
        assert gids_raw == []
        # batch_decode with no outcomes returns an empty list.
        readouts = await jit.batch_decode([])
        assert readouts == []
        await jit.reset()


@pytest.mark.asyncio
async def test_jit_controller_repr_and_raw_access():
    async with Runtime(
        decoder="black-box-naive",
        coordinator="monolithic",
        controller="jit",
    ) as runtime:
        jit = runtime.jit_controller
        assert repr(jit) == "JitController()"
        assert isinstance(jit.raw, RawJitController)


# ── Raw re-exports & misc ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_raw_runtime_re_exports():
    """The raw Rust pyclasses remain accessible for bypass scenarios."""
    import deq_runtime as _drt

    assert RawRuntime is _drt.Runtime
    assert RawCoordinator is _drt.Coordinator
    assert RawJitController is _drt.JitController

    raw = RawRuntime(decoder="black-box-naive", coordinator="naive")
    # Exercise the raw coordinator API directly.
    coord = raw.coordinator
    assert isinstance(coord, RawCoordinator)
    await coord.load_library(_library_with_one_gadget_type(gtype=5).SerializeToString())
    await raw.shutdown()


@pytest.mark.asyncio
async def test_decoder_config_accepts_dict():
    """Configuration mappings are JSON-serialized transparently."""
    runtime = Runtime(
        decoder="mock",
        decoder_config={"decode_delay_ms": 0},
        coordinator="naive",
    )
    try:
        await runtime.coordinator.load_library(_library_with_one_gadget_type(gtype=6))
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_repr_includes_jit_controller_marker():
    async with Runtime(
        decoder="black-box-naive",
        coordinator="monolithic",
        controller="jit",
    ) as runtime:
        assert "controller=jit" in repr(runtime)
    async with Runtime(decoder="black-box-naive", coordinator="naive") as runtime:
        assert "controller=jit" not in repr(runtime)


@pytest.mark.asyncio
async def test_jit_controller_end_to_end_with_real_library():
    """Build a real JitLibrary from a .deq file and drive a gadget through it.

    Exercises the full JIT pipeline (load → execute) via Python: confirms
    that dynamically-loaded libraries propagate to the coordinator so that
    `execute` can resolve the gadget type. This catches the regression where
    `JitController::load_library` only updated the JIT compiler.
    """
    from pathlib import Path

    from deq.circuit.parser import parse_file
    from deq.transpiler.jit_library_builder import build_jit_library

    repo_root = Path(__file__).resolve().parents[1]
    deq_path = repo_root / "circuit" / "repetition_code" / "repetition_code_d3.deq"
    jit_library = build_jit_library(parse_file(deq_path))
    assert jit_library.gadget_types, "library should have at least one gadget type"

    async with Runtime(
        decoder="black-box-naive",
        coordinator="monolithic",
        controller="jit",
    ) as runtime:
        jit = runtime.jit_controller
        await jit.load_library(jit_library)

        # Pick the first input-free (preparation) gadget type and instantiate it.
        prep = next(
            (gt for gt in jit_library.gadget_types if not gt.base.inputs),
            None,
        )
        assert prep is not None, "expected at least one input-free gadget"
        instr = jit_pb.JitInstruction(
            gadget=bin_pb.Gadget(gtype=prep.base.gtype, gid=1)
        )
        gid = await jit.execute(instr)
        assert gid == 1

        await jit.reset(reset_library=True)
