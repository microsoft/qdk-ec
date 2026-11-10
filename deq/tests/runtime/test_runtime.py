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
import math
from pathlib import Path

import grpc
import pytest

from deq.circuit.parser import parse_file
from deq.proto import coordinator_pb2 as coord_pb
from deq.proto import deq_bin_pb2 as bin_pb
from deq.proto import deq_jit_pb2 as jit_pb
from deq.proto import jit_controller_pb2 as jit_controller_pb
from deq.proto.coordinator_pb2_grpc import CoordinatorStub
from deq.proto.jit_controller_pb2_grpc import JitControllerStub
from deq.proto import util_pb2 as util_pb
from google.protobuf.empty_pb2 import Empty
from deq.runtime import (
    Coordinator,
    JitController,
    RawCoordinator,
    RawJitController,
    RawRuntime,
    Runtime,
)
from deq.transpiler.jit_library_builder import build_jit_library


# ── helpers ─────────────────────────────────────────────────────────────────


def _library_with_one_gadget_type(gtype: int = 1, readouts: int = 4) -> bin_pb.Library:
    """Build a minimal `deq.bin.Library` the naive coordinator will accept."""
    gadget_type = bin_pb.GadgetType(gtype=gtype, name="probe")
    for index in range(readouts):
        gadget_type.readouts.add(tag=f"r{index}")
    return bin_pb.Library(description="rt-test", gadget_types=[gadget_type])


def _repetition_code_jit_library() -> jit_pb.JitLibrary:
    tests_root = Path(__file__).resolve().parents[1]
    return build_jit_library(
        parse_file(
            tests_root / "circuit" / "repetition_code" / "repetition_code_d3.deq"
        )
    )

@pytest.mark.asyncio
@pytest.mark.parametrize("use_grpc", [False, True])
async def test_jit_execution_failure_preserves_status_until_reset(use_grpc):
    async with Runtime(
        decoder="naive", coordinator="window", controller="jit"
    ) as runtime:
        await runtime.bind("127.0.0.1:0")
        controller = runtime.jit_controller
        instruction = jit_pb.JitInstruction(
            gadget=bin_pb.Gadget(gid=1, gtype=999)
        )
        async with grpc.aio.insecure_channel(
            f"127.0.0.1:{runtime.bound_port()}"
        ) as channel:
            client = JitControllerStub(channel)
            if use_grpc:
                with pytest.raises(grpc.aio.AioRpcError) as failure:
                    await client.BatchExecute(
                        jit_controller_pb.BatchExecuteRequest(instructions=[instruction]),
                        timeout=5,
                    )
                assert failure.value.code() == grpc.StatusCode.NOT_FOUND
            else:
                with pytest.raises(RuntimeError, match="gtype=999"):
                    await asyncio.wait_for(
                        controller.batch_execute([instruction]), timeout=5
                    )
            with pytest.raises(RuntimeError, match="gtype=999"):
                await controller.decode(coord_pb.Outcomes(gid=1))
            with pytest.raises(RuntimeError, match="gtype=999"):
                await controller.batch_execute([])
            await controller.memory_stats()
            await controller.reset()
            assert await controller.batch_execute([]) == []


# ── Runtime lifecycle ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_automatic_collection_matches_traced_results_and_rejects_expired_ids(tmp_path):
    library = _repetition_code_jit_library()
    preparation = next(gadget for gadget in library.gadget_types if not gadget.base.inputs and gadget.base.outputs)
    idle = next(gadget for gadget in library.gadget_types if gadget.base.name == "Syndrome")
    measurement = next(gadget for gadget in library.gadget_types if gadget.base.inputs and not gadget.base.outputs)
    future_context = 3 + int(not preparation.base.measurements)
    comparisons = []
    for collect in [False, True]:
        async with Runtime(
            decoder="black-box-relay-bp", decoder_config={"parallel": 1},
            coordinator="window",
            coordinator_config={
                "buffer_radius": 3, "lookahead_radius": 0, "persistent_decoder": False,
                **({} if collect else {"trace_filepath": str(tmp_path / "reference.trace")}),
            },
            controller="jit",
        ) as runtime:
            await runtime.bind("127.0.0.1:0")
            controller = runtime.jit_controller
            await controller.load_library(library)
            pending = {}
            results = []
            for gid in range(1, 101):
                gadget = preparation if gid == 1 else measurement if gid == 100 else idle
                await controller.execute(jit_pb.JitInstruction(gadget=bin_pb.Gadget(
                    gid=gid, gtype=gadget.base.gtype,
                    connectors=[] if gid == 1 else [bin_pb.Gadget.Connector(gid=gid - 1, port=0)],
                )))
                size = len(gadget.base.measurements)
                data = bytearray((size + 7) // 8)
                if gid in (20, 21):
                    data[0] = 0x80
                pending[gid] = asyncio.create_task(controller.decode(coord_pb.Outcomes(
                    gid=gid, outcomes=util_pb.BitVector(size=size, data=bytes(data)),
                )))
                for finished in [index for index in pending if index <= gid - future_context]:
                    result = await asyncio.wait_for(pending.pop(finished), timeout=10)
                    results.append((result.gid, result.readouts.SerializeToString()))
                if collect:
                    state = await controller.memory_stats()
                    assert state.gadgets <= 16 + 31  # Completed gadgets between 32-request collection passes.
                    assert state.jit_gadgets <= 16
            for task in pending.values():
                result = await asyncio.wait_for(task, timeout=10)
                results.append((result.gid, result.readouts.SerializeToString()))
            comparisons.append(results)
            if collect:
                async with grpc.aio.insecure_channel(f"127.0.0.1:{runtime.bound_port()}") as channel:
                    client = JitControllerStub(channel)
                    state = await client.GetMemoryStats(Empty(), timeout=10)
                    assert state.gadgets == state.jit_gadgets == state.error_models == 0
                    expired = jit_pb.JitInstruction(gadget=bin_pb.Gadget(gid=1, gtype=preparation.base.gtype))
                    with pytest.raises(grpc.aio.AioRpcError) as error:
                        await client.Execute(expired, timeout=5)
                    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
                    with pytest.raises(RuntimeError, match="collected"):
                        await controller.execute(expired)
                    assert not hasattr(controller, "retire_history")
    assert comparisons[0] == comparisons[1]


@pytest.mark.asyncio
async def test_memory_statistics_report_automatic_collection_policy():
    for config in [{}, {"persistent_decoder": False, "forced_gap": True}]:
        async with Runtime(coordinator="window", coordinator_config=config, controller="jit") as runtime:
            stats = await runtime.jit_controller.memory_stats()
            assert stats.automatic_collection is not config.get("forced_gap", False)
            assert stats.persistent_decoder is config.get("persistent_decoder", True)
            assert stats.gadgets == stats.retained_readouts == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("coordinator", ["window", "monolithic", "naive"])
async def test_memory_statistics_report_library_types(coordinator):
    async with Runtime(decoder="naive", coordinator=coordinator, controller="jit") as runtime:
        await runtime.bind("127.0.0.1:0")
        async with grpc.aio.insecure_channel(
            f"127.0.0.1:{runtime.bound_port()}"
        ) as channel:
            coordinator_client = CoordinatorStub(channel)
            jit_client = JitControllerStub(channel)

            async def assert_counts(port_types: int, gadget_types: int) -> None:
                snapshots = [
                    await runtime.coordinator.memory_stats(),
                    await runtime.jit_controller.memory_stats(),
                    await coordinator_client.GetMemoryStats(Empty(), timeout=5),
                    await jit_client.GetMemoryStats(Empty(), timeout=5),
                ]
                for snapshot in snapshots:
                    assert snapshot.port_types == (0 if coordinator == "naive" else port_types)
                    assert snapshot.gadget_types == gadget_types
                    assert snapshot.gadgets == 0

            await assert_counts(0, 0)
            await runtime.coordinator.load_library(bin_pb.Library(
                port_types=[bin_pb.PortType(ptype=1)],
                gadget_types=[bin_pb.GadgetType(gtype=1), bin_pb.GadgetType(gtype=2)],
            ))
            await assert_counts(1, 2)
            await runtime.coordinator.load_library(bin_pb.Library(
                port_types=[bin_pb.PortType(ptype=2), bin_pb.PortType(ptype=3)],
                gadget_types=[bin_pb.GadgetType(gtype=3)],
            ))
            await assert_counts(3, 3)
            await runtime.coordinator.reset()
            await assert_counts(3, 3)
            await runtime.coordinator.reset(reset_library=True)
            await assert_counts(0, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_grpc", [False, True])
@pytest.mark.parametrize("expected_references", [1, 1 << 63, (1 << 64) - 1])
async def test_hinted_readouts_survive_jit_collection_and_transfer_to_late_consumer(use_grpc, expected_references):
    from deq.circuit.parser import parse

    library = build_jit_library(parse("""
CODE C [[1,1,1]] { LOGICAL X0 Z0 }
GADGET Prepare { R 0 OUTPUT C 0 }
GADGET Read { INPUT C 0 M 0 READOUT M0 }
GADGET Idle { INPUT C 0 OUTPUT C 0 }
"""), jobs=1)
    kinds = {gadget.base.name: gadget.base.gtype for gadget in library.gadget_types}
    async with Runtime(
        decoder="naive", coordinator="window",
        coordinator_config={"buffer_radius": 0, "persistent_decoder": False},
        controller="jit",
    ) as runtime:
        await runtime.bind("127.0.0.1:0")
        controller = runtime.jit_controller
        await controller.load_library(library)
        async with grpc.aio.insecure_channel(f"127.0.0.1:{runtime.bound_port()}") as channel:
            client = JitControllerStub(channel)

            async def execute(gid, name, predecessor=None, expected=0, modifier=None):
                instruction = jit_pb.JitInstruction(gadget=bin_pb.Gadget(
                    gid=gid, gtype=kinds[name], expected_remote_references=expected,
                    connectors=[] if predecessor is None else [bin_pb.Gadget.Connector(gid=predecessor, port=0)],
                    modifier=modifier,
                ))
                if use_grpc:
                    return (await client.Execute(instruction, timeout=5)).id
                return await controller.execute(instruction)

            async def decode(gid, size=0, data=b""):
                outcome = coord_pb.Outcomes(gid=gid, outcomes=util_pb.BitVector(size=size, data=data))
                return await asyncio.wait_for(controller.decode(outcome), timeout=5)

            await execute(1, "Prepare")
            await decode(1)
            await execute(2, "Read", 1, expected=expected_references)
            await decode(2, 1, b"\x80")
            stats = await controller.memory_stats()
            assert stats.gadgets == stats.frame_gadgets == 0
            assert stats.retained_readouts == stats.retained_readout_bytes == 1
            await execute(3, "Prepare")
            await decode(3)
            modifier = bin_pb.GadgetModifier(remote_conditional_correction=bin_pb.RemoteConditionalCorrection(
                remote_readouts=[bin_pb.RemoteConditionalCorrection.RemoteReadout(gid=2, readout_index=99)],
                correction=util_pb.BitMatrix(rows=2, cols=1, i=[1], j=[0]),
            ))
            with pytest.raises((RuntimeError, grpc.aio.AioRpcError), match="out of range"):
                await execute(4, "Idle", 3, modifier=modifier)
            assert (await controller.memory_stats()).retained_readouts == 1
            modifier.remote_conditional_correction.remote_readouts[0].readout_index = 0
            await execute(4, "Idle", 3, modifier=modifier)
            assert (await controller.memory_stats()).retained_readouts == int(expected_references > 1)
            await decode(4)
            await execute(5, "Read", 4)
            result = await decode(5, 1, b"\0")
            assert result.readouts.data == b"\x80"
            await controller.reset()
            assert (await controller.memory_stats()).retained_readouts == 0


@pytest.mark.parametrize("expected_references", [-1, 1 << 64])
def test_remote_readout_hints_reject_out_of_range_values(expected_references):
    with pytest.raises(ValueError, match="out of range"):
        bin_pb.Gadget(expected_remote_references=expected_references)

@pytest.mark.asyncio
async def test_batch_registers_remote_sources_before_consumers_and_rejects_cycles():
    from deq.circuit.parser import parse

    library = build_jit_library(parse("""
CODE C [[1,1,1]] { LOGICAL X0 Z0 }
GADGET Prepare { R 0 OUTPUT C 0 }
GADGET Read { INPUT C 0 M 0 READOUT M0 }
GADGET Idle { INPUT C 0 OUTPUT C 0 }
"""), jobs=1)
    kinds = {gadget.base.name: gadget.base.gtype for gadget in library.gadget_types}

    def instruction(gid, name, parent=None, remote=None):
        gadget = bin_pb.Gadget(
            gid=gid, gtype=kinds[name],
            connectors=[] if parent is None else [bin_pb.Gadget.Connector(gid=parent, port=0)],
        )
        if remote is not None:
            gadget.modifier.remote_conditional_correction.CopyFrom(bin_pb.RemoteConditionalCorrection(
                remote_readouts=[bin_pb.RemoteConditionalCorrection.RemoteReadout(gid=remote, readout_index=0)],
                correction=util_pb.BitMatrix(rows=2, cols=1, i=[1], j=[0]),
            ))
        return jit_pb.JitInstruction(gadget=gadget)

    async with Runtime(
        decoder="naive", coordinator="window",
        coordinator_config={"buffer_radius": 0, "persistent_decoder": False},
        controller="jit",
    ) as runtime:
        controller = runtime.jit_controller
        await controller.load_library(library)
        program = [instruction(4, "Idle", 3, 2), instruction(2, "Read", 1),
                   instruction(3, "Prepare"), instruction(1, "Prepare")]
        assert await asyncio.wait_for(controller.batch_execute(program), timeout=5) == [4, 2, 3, 1]
        await asyncio.wait_for(controller.batch_decode([
            coord_pb.Outcomes(gid=gid, outcomes=util_pb.BitVector(
                size=1 if gid == 2 else 0, data=b"\x80" if gid == 2 else b"",
            )) for gid in (1, 2, 3, 4)
        ]), timeout=5)
        assert (await controller.memory_stats()).retained_readouts == 0
        await controller.execute(instruction(5, "Read", 4))
        result = await controller.decode(coord_pb.Outcomes(
            gid=5, outcomes=util_pb.BitVector(size=1, data=b"\0"),
        ))
        assert result.readouts.data == b"\x80"
        with pytest.raises(ValueError, match="cyclic"):
            await asyncio.wait_for(controller.batch_execute([
                instruction(6, "Prepare"), instruction(7, "Idle", 6, 7),
            ]), timeout=5)


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

        gid = await coord.execute(
            bin_pb.Instruction(gadget=bin_pb.Gadget(gtype=1, gid=100))
        )
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
            await coord.execute(
                bin_pb.Instruction(gadget=bin_pb.Gadget(gtype=3, gid=2))
            )
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
                coord.decode(
                    coord_pb.Outcomes(gid=g, outcomes=util_pb.BitVector(size=0))
                )
                for g in gids
            ]
        )
        assert [r.gid for r in readouts_list] == list(gids)
        assert all(r.readouts.size == 8 for r in readouts_list)


# ── JIT controller interface ───────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("coordinator", ["monolithic", "window"])
@pytest.mark.parametrize(
    "gap_decoder,gap_config",
    [
        (None, None),
        ("black-box-relay-bp", {"seed": 17}),
        ("black-box-tesseract", {"det_penalty": 30, "beam_climbing": True}),
        (None, {"det_penalty": 30}),
    ],
)
async def test_forced_gap_with_competing_readout_errors(
    coordinator: str, gap_decoder, gap_config
):
    fixture = Path(__file__).resolve().parents[1] / "circuit/fixtures/forced_gap.deq"
    library = build_jit_library(parse_file(fixture))
    assert len(library.gadget_types) == 1
    gadget_type = library.gadget_types[0]
    assert not gadget_type.base.inputs
    assert not gadget_type.base.outputs
    assert len(gadget_type.base.readouts) == 1
    assert len(gadget_type.base.measurements) == 2
    assert len(gadget_type.finished_checks) == 1
    assert not gadget_type.unfinished_checks
    assert [error.base.probability for error in gadget_type.errors] == pytest.approx(
        [0.1, 0.01]
    )
    assert [list(error.finished_checks) for error in gadget_type.errors] == [[0], [0]]
    assert [list(error.base.readout_flips) for error in gadget_type.errors] == [[0], []]

    async with Runtime(
        decoder="black-box-tesseract",
        decoder_config={"parallel": 1},
        gap_decoder=gap_decoder,
        gap_decoder_config=gap_config,
        coordinator=coordinator,
        coordinator_config={"forced_gap": True},
        controller="jit",
    ) as runtime:
        controller = runtime.jit_controller
        await controller.load_library(library)
        gid = await controller.execute(
            jit_pb.JitInstruction(gadget=bin_pb.Gadget(gtype=gadget_type.base.gtype))
        )
        measurements = util_pb.BitVector(size=2, data=b"\x80")
        readouts = await controller.decode(
            coord_pb.Outcomes(gid=gid, outcomes=measurements)
        )

    flipping_likelihood = 0.1 * (1.0 - 0.01)
    check_only_likelihood = 0.01 * (1.0 - 0.1)
    expected_probability = check_only_likelihood / (
        flipping_likelihood + check_only_likelihood
    )
    assert readouts.readouts == util_pb.BitVector(size=1, data=b"\x80")
    assert readouts.probabilities == pytest.approx([expected_probability])
    assert readouts.syndrome_count == 1
    assert readouts.correction_count == 1
    assert readouts.correction_weight == pytest.approx(math.log(9))


@pytest.mark.parametrize("runtime_class", [Runtime, RawRuntime])
@pytest.mark.parametrize("gap_decoder", [None, "black-box-tesseract"])
@pytest.mark.parametrize("parallel", [0, 1, 2, None, "auto"])
def test_runtime_rejects_gap_pool_size(runtime_class, gap_decoder, parallel):
    import json

    gap_config = {"parallel": parallel}
    with pytest.raises(ValueError, match="parallel.*--decoder-config"):
        runtime_class(
            decoder="black-box-tesseract",
            decoder_config='{"parallel":1}',
            gap_decoder=gap_decoder,
            gap_decoder_config=(
                gap_config if runtime_class is Runtime else json.dumps(gap_config)
            ),
        )


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
@pytest.mark.parametrize("coordinator", ["monolithic", "window"])
@pytest.mark.parametrize("reset_decoder_service", [True, False])
async def test_jit_controller_full_reset_allows_library_replacement(
    coordinator: str, reset_decoder_service: bool
):
    library = _repetition_code_jit_library()
    assert library.port_types and library.gadget_types
    async with Runtime(
        decoder="black-box-naive", coordinator=coordinator, controller="jit"
    ) as runtime:
        jit = runtime.jit_controller
        await jit.load_library(library)
        for replacement_name in (library.port_types[0].base.name, "replacement"):
            await jit.reset(
                reset_library=True, reset_decoder_service=reset_decoder_service
            )
            library.port_types[0].base.name = replacement_name
            await jit.load_library(library)


@pytest.mark.asyncio
async def test_jit_open_frontier_and_duplicate_decodes_across_native_and_grpc():
    library = _repetition_code_jit_library()
    preparation = next(
        gadget for gadget in library.gadget_types
        if not gadget.base.inputs and gadget.base.outputs
    )
    measurement_count = len(preparation.base.measurements)
    async with Runtime(
        decoder="black-box-naive",
        coordinator="window",
        coordinator_config={"buffer_radius": 0, "lookahead_radius": 0},
        controller="jit",
    ) as runtime:
        await runtime.bind("127.0.0.1:0")
        async with grpc.aio.insecure_channel(
            f"127.0.0.1:{runtime.bound_port()}"
        ) as channel:
            client = JitControllerStub(channel)
            await client.LoadLibrary(library, timeout=5)
            for native_first in (False, True):
                response = await client.Execute(
                    jit_pb.JitInstruction(
                        gadget=bin_pb.Gadget(gtype=preparation.base.gtype, gid=1)
                    ),
                    timeout=5,
                )
                outcomes = coord_pb.Outcomes(
                    gid=response.id,
                    outcomes=util_pb.BitVector(
                        size=measurement_count,
                        data=bytes((measurement_count + 7) // 8),
                    ),
                )
                if native_first:
                    readouts = await asyncio.wait_for(
                        runtime.jit_controller.decode(outcomes), timeout=5
                    )
                else:
                    readouts = await client.Decode(outcomes, timeout=5)
                assert readouts.gid == 1
                with pytest.raises(grpc.aio.AioRpcError) as duplicate:
                    await client.Decode(outcomes, timeout=5)
                assert duplicate.value.code() == grpc.StatusCode.INVALID_ARGUMENT
                with pytest.raises(RuntimeError, match="unknown or already-decoded"):
                    await runtime.jit_controller.decode(outcomes)
                await client.Reset(coord_pb.ResetRequest(), timeout=5)


@pytest.mark.asyncio
async def test_jit_controller_end_to_end_with_real_library():
    """Build a real JitLibrary from a .deq file and drive a gadget through it.

    Exercises the full JIT pipeline (load → execute) via Python: confirms
    that dynamically-loaded libraries propagate to the coordinator so that
    `execute` can resolve the gadget type. This catches the regression where
    `JitController::load_library` only updated the JIT compiler.
    """
    jit_library = _repetition_code_jit_library()
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
        await jit.load_library(jit_library)
        assert await jit.execute(instr) == 1
        await jit.reset()
        assert await jit.execute(instr) == 1
