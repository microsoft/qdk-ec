"""Extract readout latencies and report individual solver-call distributions."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
import math
from pathlib import Path
import re

from deq.proto.deq_jit_pb2 import JitLibrary
from deq.proto.deq_bin_pb2 import Library
from deq.proto.fastrace_pb2 import SpanRecord, TraceFile
from deq.proto.simulator_pb2 import SimulatorShot
from deq.transpiler.program_timing import gadget_timing

SIMULATOR_SPAN_NAMES = {"jit_static_simulator", "static_simulator"}


def _varint(stream, *, allow_eof: bool = False) -> int | None:
    value = 0
    for shift in range(0, 70, 7):
        byte = stream.read(1)
        if not byte:
            if shift == 0 and allow_eof:
                return None
            raise ValueError("truncated protobuf integer")
        value |= (byte[0] & 127) << shift
        if not byte[0] & 128:
            if value >= 1 << 64:
                raise ValueError("protobuf integer overflow")
            return value
    raise ValueError("protobuf integer overflow")


def protobuf_records(path: Path) -> Iterator[bytes]:
    """Read repeated field-one messages without retaining the whole trace."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        while (tag := _varint(stream, allow_eof=True)) is not None:
            field, wire = tag >> 3, tag & 7
            if field == 0:
                raise ValueError("invalid protobuf field")
            if wire == 0:
                _varint(stream)
                continue
            if wire == 2:
                length = _varint(stream)
                assert length is not None
            elif wire in (1, 5):
                length = 8 if wire == 1 else 4
            else:
                raise ValueError("unsupported protobuf wire type")
            if length > size - stream.tell():
                raise ValueError("truncated protobuf record")
            if field == 1 and wire == 2:
                yield stream.read(length)
            else:
                stream.seek(length, 1)


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
    own_gadget_latency_ms: float | None = None


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
        required = (
            "initialized_offset_ns",
            "solve_finished_offset_ns",
            "solver_success",
        )
        if not all(key in properties for key in required):
            raise ValueError("incomplete solver-call timing")
        if properties["solver_success"] != "true":
            raise ValueError(
                "failed solver calls cannot form a successful-run distribution"
            )
        start = int(properties["initialized_offset_ns"])
        end = int(properties["solve_finished_offset_ns"])
        if not 0 <= start <= end:
            raise ValueError("invalid monotonic solver-call timing")
        durations.append((end - start) / 1e6)
    return durations


def readout_samples(result: dict) -> tuple[list[ReadoutSample], list[dict]]:
    run = Path(result["run_directory"])
    warmups = result.get("warmup_shots", 0)
    if (
        isinstance(warmups, bool)
        or not isinstance(warmups, int)
        or not 0 <= warmups < result["shots"]
    ):
        raise ValueError("warmup_shots must leave at least one measured shot")
    if result["controller"] == "static":
        static_library = Library.FromString((run / "Brickwall.deq.bin").read_bytes())
        counts = {
            item.gtype: len(item.readouts) for item in static_library.gadget_types
        }
        gadgets = [
            item.gadget
            for item in static_library.program
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
    completed = {}
    for data in protobuf_records(run / "shots.pb"):
        shot = SimulatorShot.FromString(data)
        if shot.shot in completed:
            raise ValueError("missing or duplicate shot results")
        completed[shot.shot] = shot.HasField("decode_result")
    if sorted(completed) != list(range(result["shots"])):
        raise ValueError("missing or duplicate shot results")
    if not all(completed.values()):
        raise ValueError("failed shots cannot become latency observations")
    shots = []
    for data in protobuf_records(run / "timing.pb"):
        span = SpanRecord.FromString(data)
        if span.name in SIMULATOR_SPAN_NAMES:
            properties = {item.key: item.value for item in span.properties}
            if (
                result.get("require_lossless_trace")
                and properties.get("timing_transport") != "lossless"
            ):
                raise ValueError("replay requires the lossless latency trace transport")
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
    observed = {shot: set() for shot in completed}
    observed_counts = Counter()
    for data in protobuf_records(run / "timing.pb"):
        span = SpanRecord.FromString(data)
        if span.name not in ("decode_window", "compiled_readout"):
            continue
        properties = {item.key: item.value for item in span.properties}
        if (
            result.get("require_lossless_trace")
            and properties.get("timing_transport") != "lossless"
        ):
            raise ValueError("replay requires the lossless latency trace transport")
        if span.name == "compiled_readout":
            gid = int(properties["gid"])
            source_gid = int(properties["input_gid"])
            if gid not in readouts or source_gid not in completion_ns:
                raise ValueError("compiled readout references an unknown gadget")
            if int(properties["readout_count"]) != readouts[gid]:
                raise ValueError("compiled readout count does not match the library")
            readiness = [
                event.timestamp_unix_ns
                for event in span.events
                if event.name == "ready"
            ]
            if len(readiness) != 1:
                raise ValueError("compiled readout needs exactly one readiness event")
            committed = readiness[0]
            window = [gid, source_gid]
            committed_gids = [gid]
        else:
            if "pauli_frame_updated_offset_ns" not in properties:
                continue
            events = window_event_times(span)
            committed = events["pauli_frame_updated"]
            window = [int(gid) for gid in re.findall(r"\d+", properties["window"])]
            committed_gids = [
                int(gid) for gid in re.findall(r"\d+", properties["commit_region"])
            ]
        position = bisect_right(starts, span.begin_time_unix_ns) - 1
        if position < 0:
            raise ValueError("window precedes its simulator shot")
        start, end, shot = shots[position]
        if not start <= span.begin_time_unix_ns <= committed <= end:
            raise ValueError("commit is outside the successful shot")
        if not any(gid in readouts for gid in committed_gids):
            continue
        planned = start + max(completion_ns[gid] for gid in window)
        latency_ms = (committed - planned) / 1e6
        if not math.isfinite(latency_ms) or latency_ms <= 0:
            raise ValueError("readout latency must be finite and positive")
        for gid in committed_gids:
            if gid not in readouts:
                continue
            if gid in observed[shot]:
                raise ValueError(f"readout gadget {gid} was committed more than once")
            observed[shot].add(gid)
            region = (
                "final" if completion_ns[gid] == final_completion else "intermediate"
            )
            observed_counts[shot, region] += readouts[gid]
            for _ in range(readouts[gid] if shot >= warmups else 0):
                samples.append(
                    ReadoutSample(
                        shot=shot,
                        gid=gid,
                        region=region,
                        latency_ms=latency_ms,
                        own_gadget_latency_ms=(committed - start - completion_ns[gid])
                        / 1e6,
                    )
                )
    coverage = []
    for shot in completed:
        missing = set(readouts) - observed[shot]
        if missing:
            raise ValueError(
                f"successful shot {shot} is missing {len(missing)} readout commits"
            )
        for region in ("intermediate", "final"):
            expected = sum(
                count
                for gid, count in readouts.items()
                if (
                    "final"
                    if completion_ns[gid] == final_completion
                    else "intermediate"
                )
                == region
            )
            coverage.append(
                {
                    "shot": shot,
                    "region": region,
                    "expected": expected,
                    "observed": observed_counts[shot, region],
                }
            )
    return samples, coverage


def main() -> None:
    import argparse
    import json
    from summarize import quantile, summarize_population

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+", help="Native timing.pb files")
    parser.add_argument(
        "--output", type=Path, help="Save statistics and distribution quantiles as JSON"
    )
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
            + " | ".join(
                "-" if value is None else f"{value * 1000:.3f}" for value in columns
            )
        )
        print(
            f"Zero-syndrome bypasses (not solver calls): {statistics['zero_syndrome_bypasses']}"
        )
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
