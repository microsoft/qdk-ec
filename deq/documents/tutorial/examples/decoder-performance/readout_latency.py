"""Extract readout latencies and report individual solver-call distributions."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from pathlib import Path
import re

from deq.proto.deq_jit_pb2 import JitLibrary
from deq.proto.deq_bin_pb2 import Library
from deq.proto.fastrace_pb2 import SpanRecord, TraceFile
from deq.proto.simulator_pb2 import SimulatorTrace
from deq.transpiler.program_timing import gadget_timing

SIMULATOR_SPAN_NAMES = {"jit_static_simulator", "static_simulator"}


def window_event_times(window: SpanRecord) -> dict[str, int]:
    names = (
        "outcomes_ready",
        "inputs_ready",
        "graph_slot_acquired",
        "decoding",
        "decoded",
        "pauli_frame_updated",
    )
    properties = {item.key: item.value for item in window.properties}
    keys = [f"{name}_offset_ns" for name in names]
    if not all(key in properties for key in keys):
        raise ValueError("incomplete monotonic window timing")
    offsets = [int(properties[key]) for key in keys]
    if offsets != sorted(offsets) or min(offsets) < 0:
        raise ValueError("invalid monotonic window timing")
    return {
        name: window.begin_time_unix_ns + offset for name, offset in zip(names, offsets)
    }


@dataclass(frozen=True)
class ReadoutSample:
    shot: int
    gid: int
    region: str
    latency_ms: float


def solver_call_durations(trace: TraceFile) -> list[float]:
    """Return solver-stage milliseconds, excluding dispatch, construction and reset.

    Loaded-decoder traces include parity-result validation in this stage.
    Zero-syndrome shortcuts make no solver call and are not inserted as zeros.
    """
    durations = []
    for span in trace.spans:
        if span.name != "decoder_job":
            continue
        properties = {item.key: item.value for item in span.properties}
        required = ("initialized_offset_ns", "solve_finished_offset_ns", "solver_success")
        if not all(key in properties for key in required):
            raise ValueError("incomplete solver-call timing")
        if properties["solver_success"] != "true":
            raise ValueError("failed solver calls cannot form a successful-run distribution")
        start = int(properties["initialized_offset_ns"])
        end = int(properties["solve_finished_offset_ns"])
        if not 0 <= start <= end:
            raise ValueError("invalid monotonic solver-call timing")
        durations.append((end - start) / 1e6)
    return durations


def readout_samples(result: dict) -> tuple[list[ReadoutSample], list[dict]]:
    run = Path(result["run_directory"])
    shot_results = SimulatorTrace.FromString((run / "shots.pb").read_bytes())
    trace = TraceFile.FromString((run / "timing.pb").read_bytes())
    if result["controller"] == "static":
        static_library = Library.FromString(
            (run / "Brickwall.deq.bin").read_bytes()
        )
        counts = {
            item.gtype: len(item.readouts) for item in static_library.gadget_types
        }
        gadgets = [
            item.gadget for item in static_library.program
            if item.WhichOneof("create") == "gadget"
        ]
    else:
        library = JitLibrary.FromString((run / "Brickwall.deq.jit").read_bytes())
        counts = {
            item.base.gtype: len(item.base.readouts) for item in library.gadget_types
        }
        gadgets = [item.gadget for item in library.program]
    completion_ns = {}
    for gadget in gadgets:
        timing = gadget_timing(gadget)
        if timing is None:
            raise ValueError("native program is missing simulation timing metadata")
        completion_ns[gadget.gid] = timing.end
    if not completion_ns:
        raise ValueError("native program is missing simulation timing metadata")
    readouts = {
        gadget.gid: counts[gadget.gtype] for gadget in gadgets if counts[gadget.gtype]
    }
    final_completion = max(completion_ns.values())
    completed = {
        shot.shot: shot.HasField("decode_result") for shot in shot_results.shots
    }
    if len(completed) != len(shot_results.shots) or sorted(completed) != list(
        range(result["shots"])
    ):
        raise ValueError("missing or duplicate shot results")
    if not all(completed.values()):
        raise ValueError("failed shots cannot become latency observations")
    shots = []
    for span in trace.spans:
        if span.name in SIMULATOR_SPAN_NAMES:
            properties = {item.key: item.value for item in span.properties}
            events = {event.name: event.timestamp_unix_ns for event in span.events}
            shots.append(
                (
                    events["start_decoding"],
                    events["process_result"],
                    int(properties["shot"]),
                )
            )
    shots.sort()
    if len(shots) != len(completed) or {shot for _, _, shot in shots} != set(completed):
        raise ValueError("shot results and timing records disagree")
    if any(start >= end for start, end, _ in shots) or any(
        previous[1] > current[0] for previous, current in zip(shots, shots[1:])
    ):
        raise ValueError("shot timing records overlap or run backwards")
    starts = [start for start, _, _ in shots]
    samples = []
    observed: set[tuple[int, int]] = set()
    for span in trace.spans:
        if span.name != "decode_window":
            continue
        properties = {item.key: item.value for item in span.properties}
        if "pauli_frame_updated_offset_ns" not in properties:
            continue
        events = window_event_times(span)
        position = bisect_right(starts, span.begin_time_unix_ns) - 1
        if position < 0:
            raise ValueError("window precedes its simulator shot")
        start, end, shot = shots[position]
        committed = events["pauli_frame_updated"]
        if not start <= span.begin_time_unix_ns <= committed <= end:
            raise ValueError("commit is outside the successful shot")
        window = [int(gid) for gid in re.findall(r"\d+", properties["window"])]
        committed_gids = [
            int(gid) for gid in re.findall(r"\d+", properties["commit_region"])
        ]
        planned = start + max(completion_ns[gid] for gid in window)
        latency_ms = (committed - planned) / 1e6
        if not math.isfinite(latency_ms) or latency_ms <= 0:
            raise ValueError("readout latency must be finite and positive")
        for gid in committed_gids:
            if gid not in readouts:
                continue
            if (shot, gid) in observed:
                raise ValueError(f"readout gadget {gid} was committed more than once")
            observed.add((shot, gid))
            for _ in range(readouts[gid]):
                samples.append(
                    ReadoutSample(
                        shot=shot,
                        gid=gid,
                        region=(
                            "final" if completion_ns[gid] == final_completion else "intermediate"
                        ),
                        latency_ms=latency_ms,
                    )
                )
    coverage = []
    for shot in completed:
        missing = set(readouts) - {
            gid for observed_shot, gid in observed if observed_shot == shot
        }
        if missing:
            raise ValueError(
                f"successful shot {shot} is missing {len(missing)} readout commits"
            )
        for region in ("intermediate", "final"):
            expected = sum(
                count
                for gid, count in readouts.items()
                if ("final" if completion_ns[gid] == final_completion else "intermediate") == region
            )
            coverage.append(
                {
                    "shot": shot,
                    "region": region,
                    "expected": expected,
                    "observed": sum(
                        sample.shot == shot and sample.region == region
                        for sample in samples
                    ),
                }
            )
    return samples, coverage


def main() -> None:
    import argparse
    import json
    from summarize import quantile, summarize_population

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+", help="Native timing.pb files")
    parser.add_argument("--output", type=Path, help="Save statistics and distribution quantiles as JSON")
    arguments = parser.parse_args()
    report = {}
    for path in arguments.traces:
        trace = TraceFile.FromString(path.read_bytes())
        durations = sorted(solver_call_durations(trace))
        statistics = summarize_population(durations, 256)
        statistics["p90_ms"] = quantile(durations, 0.9) if durations else None
        statistics["maximum_ms"] = durations[-1] if durations else None
        statistics["zero_syndrome_bypasses"] = sum(
            span.name == "decoder_zero_syndrome" for span in trace.spans
        )
        report[str(path.resolve())] = statistics
        columns = [
            statistics.get(key)
            for key in ("mean_ms", "median_ms", "p90_ms", "p99_ms", "maximum_ms")
        ]
        print(path)
        print("calls | mean us | median us | p90 us | p99 us | max us")
        print(
            f"{len(durations)} | "
            + " | ".join("-" if value is None else f"{value * 1000:.3f}" for value in columns)
        )
        print(f"Zero-syndrome bypasses (not solver calls): {statistics['zero_syndrome_bypasses']}")
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
