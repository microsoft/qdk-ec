"""Verify the plotted quantity and reject incomplete or inconsistent timing data."""

from pathlib import Path

import pytest

from deq.proto.deq_jit_pb2 import JitLibrary
from deq.proto.fastrace_pb2 import TraceFile
from deq.proto.simulator_pb2 import SimulatorTrace
from deq.transpiler.program_timing import TIMING_METADATA_KEY, Timing, set_gadget_timing
from readout_latency import readout_samples, solver_call_durations


@pytest.fixture
def recorded_run(tmp_path: Path) -> dict:
    run = tmp_path / "run"
    run.mkdir()
    library = JitLibrary()
    definition = library.gadget_types.add().base
    definition.gtype = 1
    definition.readouts.add()
    for gid in (1, 2):
        gadget = library.program.add().gadget
        gadget.gid, gadget.gtype = gid, 1
        set_gadget_timing(gadget, Timing((gid - 1) * 10_000_000, 10_000_000))
    (run / "Brickwall.deq.jit").write_bytes(library.SerializeToString())
    shots = SimulatorTrace()
    trace = TraceFile()
    for shot in (0, 1):
        start = (shot + 1) * 1_000_000_000
        shots.shots.add(shot=shot).decode_result.SetInParent()
        span = trace.spans.add(name="jit_static_simulator", begin_time_unix_ns=start)
        span.properties.add(key="shot", value=str(shot))
        span.events.add(name="start_decoding", timestamp_unix_ns=start)
        span.events.add(name="process_result", timestamp_unix_ns=start + 100_000_000)
        for gid, elapsed in ((1, 25), (2, 27)):
            window = trace.spans.add(name="decode_window", begin_time_unix_ns=start + 1)
            window.properties.add(key="window", value="[1, 2]")
            window.properties.add(key="commit_region", value=f"[{gid}]")
            for name in (
                "outcomes_ready",
                "inputs_ready",
                "graph_slot_acquired",
                "decoding",
                "decoded",
                "pauli_frame_updated",
            ):
                window.properties.add(
                    key=f"{name}_offset_ns", value=str(elapsed * 1_000_000 - 1)
                )
    (run / "shots.pb").write_bytes(shots.SerializeToString())
    (run / "timing.pb").write_bytes(trace.SerializeToString())
    return {
        "controller": "jit",
        "run_directory": str(run),
        "shots": 2,
    }


def test_latency_uses_last_required_input_not_center_measurement(recorded_run: dict):
    samples, coverage = readout_samples(recorded_run)
    assert [(sample.region, sample.latency_ms) for sample in samples] == [
        ("intermediate", 5),
        ("final", 7),
        ("intermediate", 5),
        ("final", 7),
    ]
    assert len(coverage) == 4
    assert all(row["expected"] == row["observed"] == 1 for row in coverage)


def test_compiled_readiness_reports_required_and_own_input_latency(recorded_run: dict):
    path = Path(recorded_run["run_directory"]) / "timing.pb"
    trace = TraceFile.FromString(path.read_bytes())
    for span in trace.spans:
        if span.name != "decode_window":
            continue
        properties = {item.key: item.value for item in span.properties}
        gid = int(properties["commit_region"].strip("[]"))
        offset = properties["pauli_frame_updated_offset_ns"]
        span.name = "compiled_readout"
        del span.properties[:]
        span.events.add(
            name="ready", timestamp_unix_ns=span.begin_time_unix_ns + int(offset)
        )
        for key, value in (
            ("gid", str(gid)),
            ("input_gid", "2"),
            ("readout_count", "1"),
        ):
            span.properties.add(key=key, value=value)
    path.write_bytes(trace.SerializeToString())
    samples, coverage = readout_samples(recorded_run)
    assert [sample.latency_ms for sample in samples] == [5, 7, 5, 7]
    assert [sample.own_gadget_latency_ms for sample in samples] == [15, 7, 15, 7]
    assert all(row["expected"] == row["observed"] for row in coverage)
    trace.spans[1].properties[2].value = "2"
    path.write_bytes(trace.SerializeToString())
    with pytest.raises(ValueError, match="readout count"):
        readout_samples(recorded_run)


def test_warmups_are_validated_but_excluded_from_samples(recorded_run: dict):
    recorded_run["warmup_shots"] = 1
    samples, coverage = readout_samples(recorded_run)
    assert [sample.shot for sample in samples] == [1, 1]
    assert len(coverage) == 4
    assert all(row["expected"] == row["observed"] == 1 for row in coverage)
    path = Path(recorded_run["run_directory"]) / "shots.pb"
    shots = SimulatorTrace.FromString(path.read_bytes())
    shots.shots[0].ClearField("decode_result")
    path.write_bytes(shots.SerializeToString())
    with pytest.raises(ValueError, match="failed shots"):
        readout_samples(recorded_run)


def test_replay_rejects_best_effort_traces(recorded_run: dict):
    recorded_run["require_lossless_trace"] = True
    with pytest.raises(ValueError, match="lossless latency trace"):
        readout_samples(recorded_run)
    path = Path(recorded_run["run_directory"]) / "timing.pb"
    trace = TraceFile.FromString(path.read_bytes())
    for span in trace.spans:
        span.properties.add(key="timing_transport", value="lossless")
    path.write_bytes(trace.SerializeToString())
    assert len(readout_samples(recorded_run)[0]) == 4


def test_missing_warmup_commit_is_not_hidden(recorded_run: dict):
    recorded_run["warmup_shots"] = 1
    path = Path(recorded_run["run_directory"]) / "timing.pb"
    trace = TraceFile.FromString(path.read_bytes())
    del trace.spans[1]
    path.write_bytes(trace.SerializeToString())
    with pytest.raises(ValueError, match="missing 1 readout commits"):
        readout_samples(recorded_run)


@pytest.mark.parametrize("data", [b"\x80", b"\x0a\x05\x00", b"\x00"])
def test_streaming_reader_rejects_truncated_or_invalid_records(
    tmp_path: Path, data: bytes
):
    from readout_latency import protobuf_records

    path = tmp_path / "trace.pb"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        list(protobuf_records(path))


@pytest.mark.parametrize("elapsed_ns", [19_000_000, 20_000_000])
def test_non_readout_windows_are_not_latency_observations(
    recorded_run: dict, elapsed_ns: int
):
    expected = readout_samples(recorded_run)
    run = Path(recorded_run["run_directory"])
    library = JitLibrary.FromString((run / "Brickwall.deq.jit").read_bytes())
    library.gadget_types.add().base.gtype = 2
    gadget = library.program.add().gadget
    gadget.gid, gadget.gtype = 3, 2
    set_gadget_timing(gadget, Timing(0, 20_000_000))
    (run / "Brickwall.deq.jit").write_bytes(library.SerializeToString())
    trace = TraceFile.FromString((run / "timing.pb").read_bytes())
    window = trace.spans.add()
    window.CopyFrom(trace.spans[1])
    for property in window.properties:
        if property.key in ("window", "commit_region"):
            property.value = "[3]"
        elif property.key.endswith("_offset_ns"):
            property.value = str(elapsed_ns - 1)
    (run / "timing.pb").write_bytes(trace.SerializeToString())
    assert readout_samples(recorded_run) == expected


def test_missing_native_timing_is_rejected(recorded_run: dict):
    path = Path(recorded_run["run_directory"]) / "Brickwall.deq.jit"
    library = JitLibrary.FromString(path.read_bytes())
    del library.program[0].gadget.metadata[TIMING_METADATA_KEY]
    path.write_bytes(library.SerializeToString())
    with pytest.raises(ValueError, match="simulation timing metadata"):
        readout_samples(recorded_run)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "failed"])
def test_bad_shot_results_are_not_latency_samples(recorded_run: dict, mutation: str):
    path = Path(recorded_run["run_directory"]) / "shots.pb"
    shots = SimulatorTrace.FromString(path.read_bytes())
    if mutation == "duplicate":
        shots.shots[1].shot = 0
    elif mutation == "missing":
        del shots.shots[-1]
    else:
        shots.shots[0].ClearField("decode_result")
    path.write_bytes(shots.SerializeToString())
    with pytest.raises(ValueError):
        readout_samples(recorded_run)


@pytest.mark.parametrize(
    "mutation",
    ["duplicate-shot", "duplicate-commit", "missing-commit", "early", "late"],
)
def test_bad_timing_records_are_rejected(recorded_run: dict, mutation: str):
    path = Path(recorded_run["run_directory"]) / "timing.pb"
    trace = TraceFile.FromString(path.read_bytes())
    if mutation == "duplicate-shot":
        trace.spans.add().CopyFrom(trace.spans[0])
    elif mutation == "duplicate-commit":
        trace.spans.add().CopyFrom(trace.spans[1])
    elif mutation == "missing-commit":
        del trace.spans[1]
    elif mutation == "early":
        for property in trace.spans[1].properties:
            if property.key.endswith("_offset_ns"):
                property.value = str(19_000_000 - 1)
    else:
        trace.spans[1].properties[-1].value = str(101_000_000 - 1)
    path.write_bytes(trace.SerializeToString())
    with pytest.raises(ValueError):
        readout_samples(recorded_run)


def solver_trace() -> TraceFile:
    trace = TraceFile()
    for start, end in ((100, 500), (4000, 9000)):
        span = trace.spans.add(name="decoder_job")
        span.properties.add(key="initialized_offset_ns", value=str(start))
        span.properties.add(key="solve_finished_offset_ns", value=str(end))
        span.properties.add(key="solver_success", value="true")
    trace.spans.add(name="decoder_zero_syndrome")
    trace.spans.add(name="decode_window")
    return trace


def test_solver_distribution_excludes_non_calls():
    assert solver_call_durations(solver_trace()) == [0.0004, 0.005]
    assert solver_call_durations(TraceFile()) == []


@pytest.mark.parametrize("mutation", ["missing", "failed", "backwards"])
def test_invalid_solver_timing_is_rejected(mutation):
    trace = solver_trace()
    properties = trace.spans[0].properties
    if mutation == "missing":
        del properties[0]
    elif mutation == "failed":
        properties[2].value = "false"
    else:
        properties[1].value = "0"
    with pytest.raises(ValueError):
        solver_call_durations(trace)


def test_solver_distribution_command_reports_microseconds(tmp_path):
    import json
    import subprocess
    import sys
    import readout_latency

    trace = tmp_path / "timing.pb"
    trace.write_bytes(solver_trace().SerializeToString())
    output = tmp_path / "solver.json"
    result = subprocess.run(
        [sys.executable, readout_latency.__file__, str(trace), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "mean us | median us" in result.stdout
    statistics = json.loads(output.read_text())[str(trace)]
    assert statistics["samples"] == 2
    assert statistics["median_ms"] == pytest.approx(0.0027)
    assert statistics["quantiles_ms"] == [0.0004, 0.005]
    assert statistics["zero_syndrome_bypasses"] == 1
