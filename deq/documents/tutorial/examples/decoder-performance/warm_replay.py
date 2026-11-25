"""Evaluate all four decoder curves using validated, precomputed physical shots."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
import sys

import psutil

from deq.cli.jit import _render_rhai_assertion_block
from deq.compiler.jit_compiler import static_jit_compiler
from deq.defaults import DEFAULT_RAYON_NUM_THREADS, DEFAULT_TOKIO_WORKER_THREADS
from deq.proto.deq_jit_pb2 import JitLibrary
from deq.proto.simulator_pb2 import SimulatorShot
from plot import plot
from precompute_replay import MAX_FILE_BYTES, checkpoint, file_hash, verify_samples
from readout_latency import protobuf_records, readout_samples
from run import SERIES, Series, overloaded, provenance
from study_utils import (
    exclusive_output,
    interrupt_on_termination,
    population,
    run_process,
)
from summarize import write_summary

HERE = Path(__file__).resolve().parent
POOLS = [
    {"name": f"numa-{index}", "cpus": cpus, "max_initializations": 2}
    for index, cpus in enumerate(("30-53", "60-83", "90-113"), 1)
]


def _write_artifact(path: Path, data: bytes) -> None:
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"{path} exceeds the 10 GB file guard")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def compile_static_file(source: str, output: str) -> None:
    os.environ["TOKIO_WORKER_THREADS"] = str(DEFAULT_TOKIO_WORKER_THREADS)
    os.environ["RAYON_NUM_THREADS"] = str(DEFAULT_RAYON_NUM_THREADS)
    destination = Path(output)
    with exclusive_output(destination.with_suffix(".lock")):
        encoded = Path(source).read_bytes()
        source_hash = hashlib.sha256(encoded).hexdigest()
        runtime_hash = file_hash(_native_path())
        library = JitLibrary.FromString(encoded)
        _write_artifact(
            destination, static_jit_compiler(library, compact=True).SerializeToString()
        )
        if file_hash(_native_path()) != runtime_hash:
            raise ValueError("Native compiler changed while preparing a static library")
        checkpoint(
            _static_receipt(destination),
            {
                "source_sha256": source_hash,
                "runtime_sha256": runtime_hash,
                "library_sha256": file_hash(destination),
                "compact": True,
            },
        )


def _native_path() -> Path:
    path = importlib.import_module("deq_runtime.deq_runtime").__file__
    if path is None:
        raise RuntimeError("Cannot identify the native runtime")
    return Path(path)


def _static_receipt(path: Path) -> Path:
    return path.with_name(path.name + ".metadata.json")


def verify_static_cache(
    source: Path, cached: Path, *, compiler_sha256: str | None = None
) -> None:
    """Verify against the current compiler or an explicitly pinned producer."""
    if compiler_sha256 is not None and (
        not isinstance(compiler_sha256, str)
        or re.fullmatch("[0-9a-f]{64}", compiler_sha256) is None
    ):
        raise ValueError(
            "A supplied static library needs its recorded compiler SHA-256"
        )
    receipt_path = _static_receipt(cached)
    if not receipt_path.is_file():
        raise ValueError(
            f"Unverified static library cache: {cached}; rebuild it in a fresh work directory"
        )
    receipt = json.loads(receipt_path.read_text())
    expected = {
        "source_sha256": file_hash(source),
        "runtime_sha256": (
            compiler_sha256
            if compiler_sha256 is not None
            else file_hash(_native_path())
        ),
        "library_sha256": file_hash(cached),
        "compact": True,
    }
    if receipt != expected:
        raise ValueError(
            f"Static library cache does not match its input, runtime, or receipt: {cached}"
        )


def run_point(case: Path, directory: Path, series: Series, settings: dict) -> dict:
    coordinator_name = settings.get("coordinator", "window")
    if coordinator_name not in ("window", "compiled-window"):
        raise ValueError("Unsupported replay coordinator")
    if coordinator_name == "compiled-window" and not series.static:
        raise ValueError("compiled-window requires a static series")
    worker_sample_rate = settings.get("worker_sample_rate", 1)
    if (
        isinstance(worker_sample_rate, bool)
        or not isinstance(worker_sample_rate, int)
        or worker_sample_rate < 1
    ):
        raise ValueError("worker_sample_rate must be a positive integer")
    numa_policy = settings.get("numa_policy", "inherit")
    if numa_policy not in ("inherit", "local"):
        raise ValueError("numa_policy must be inherit or local")
    policy_command = None
    if numa_policy == "local":
        policy_command = shutil.which("numactl")
        if policy_command is None:
            raise FileNotFoundError(
                "numactl is required for an explicit local NUMA policy"
            )
    started = time.monotonic()

    def guarded(
        command: list[str], log: Path, env: dict[str, str] | None = None
    ) -> dict:
        remaining = settings["task_timeout_seconds"] - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError(
                "Replay point exceeded its preparation and execution time budget"
            )
        budget = settings["memory_gib"] - psutil.Process().memory_info().rss / 2**30
        if budget <= 0:
            raise MemoryError("Replay supervisor consumed the memory budget")
        return run_process(
            command, log, memory_gib=budget, timeout_seconds=remaining, env=env
        )

    manifest = json.loads((case / "samples.json").read_text())
    verify_samples(case, manifest)
    for name in ("circuit.deq", "circuit.deq.jit", "configuration.json"):
        if file_hash(case / name) != manifest["input_sha256"][name]:
            raise ValueError(f"Sampling input changed before replay: {case / name}")
    configuration = json.loads((case / "configuration.json").read_text())
    if "layers" in settings and configuration["layers"] != settings["layers"]:
        raise ValueError("Sampling depth differs from the requested evaluation")
    if (
        "probability" in settings
        and configuration["probability"] != settings["probability"]
    ):
        raise ValueError("Sampling probability differs from the requested evaluation")
    directory.mkdir(parents=True)
    jit_path = case / "circuit.deq.jit"
    (directory / "Brickwall.deq.jit").symlink_to(jit_path.resolve())
    library = JitLibrary.FromString(jit_path.read_bytes())
    counts = {kind.base.gtype: len(kind.base.readouts) for kind in library.gadget_types}
    expected_readouts = {
        instruction.gadget.gid: counts[instruction.gadget.gtype]
        for instruction in library.program
    }
    if len(expected_readouts) != configuration["gadgets"]:
        raise ValueError("Compiled gadget identities do not match the configuration")
    del library
    compilation_resources = None
    supplied_static = settings.get("static_input")
    compiler_sha256 = None
    if supplied_static is not None and not series.static:
        raise ValueError("A supplied static library cannot be used by a JIT series")
    if series.static:
        if supplied_static is not None:
            cached = Path(supplied_static["path"]).resolve()
            compiler_sha256 = supplied_static["compiler_sha256"]
            if compiler_sha256 is None:
                raise ValueError(
                    "A supplied static library needs its recorded compiler SHA-256"
                )
        else:
            cached = directory.parent / "Brickwall.deq.bin"
        if supplied_static is None and not cached.exists():
            compilation_resources = guarded(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--compile-static",
                    str(jit_path),
                    str(cached),
                ],
                directory / "compilation.log",
            )
        verify_static_cache(jit_path, cached, compiler_sha256=compiler_sha256)
        (directory / "Brickwall.deq.bin").symlink_to(
            cached if supplied_static is not None else Path("..") / cached.name
        )
        controller_path = cached
    else:
        controller_path = jit_path

    block = _render_rhai_assertion_block(
        configuration["assertions"], None, include_measurements=False
    )
    script = "\n".join(line.removeprefix("# ") for line in block.splitlines()[1:])
    assertion_path = directory / "assertions.rhai"
    _write_artifact(assertion_path, script.encode())
    warmups = settings["warmup_shots"]
    measured = settings["measured_shots"]
    total = warmups + measured
    sample_files = [
        str((case / sample["file"]).resolve()) for sample in manifest["samples"]
    ]
    decoder_config = (
        {"pools": settings["pools"]}
        if settings["pools"]
        else {"parallel": settings["threads"]}
    )
    if series.decoder == "mwpf":
        decoder_config["cluster_node_limit"] = 10
    simulator = {
        "python_sampler": {
            "sampler": str(HERE / "replay_sampler.py"),
            "py_config": {
                "files": sample_files,
                "sha256": [sample["sha256"] for sample in manifest["samples"]],
            },
        },
        "shots": total,
        "errors": total + 1,
        "seed": 42,
        "strict_timing": True,
        "logical_assert_filepath": str(assertion_path.resolve()),
        "simulator_trace_output": str((directory / "shots.pb").resolve()),
    }
    if settings.get("trim_heap", False):
        simulator["python_sampler"]["py_config"].update(
            trim_heap=True,
            memory_log=str((directory / "allocator.jsonl").resolve()),
        )
    if series.static:
        simulator["program_filepath"] = str(controller_path.resolve())
    else:
        simulator["jit_library_filepath"] = str(controller_path.resolve())
        simulator["registration_lead_seconds"] = 2 * configuration["block_ns"] / 1e9
    coordinator = {
        "buffer_radius": 1,
        "lookahead_radius": 0,
        "persistent_decoder": series.static,
        "automatic_collection": False,
        "assert_parity_factor": True,
    }
    if coordinator_name == "compiled-window":
        coordinator = {
            "worker_threads": settings.get("compiled_worker_threads", 24),
            "batch_size": settings.get("compiled_batch_size", 8),
            "max_templates": settings.get("compiled_max_templates", 2048),
        }
        if any(type(value) is not int or value <= 0 for value in coordinator.values()):
            raise ValueError("Compiled coordinator sizes must be positive integers")
    elif settings.get("system_diagnostics", False):
        coordinator["trace_pipeline"] = True
    controller = {"filepath": str(controller_path.resolve())}
    if "decode_task_batch_size" in settings:
        batch_size = settings["decode_task_batch_size"]
        if not series.static or type(batch_size) is not int or batch_size <= 0:
            raise ValueError(
                "decode_task_batch_size requires a positive integer and a static series"
            )
        controller["decode_task_batch_size"] = batch_size
    command = [
        sys.executable,
        "-m",
        "deq.runtime",
        "server",
        "--addr",
        "127.0.0.1:0",
        "--decoder",
        series.decoder,
        "--decoder-config",
        json.dumps(decoder_config),
        "--coordinator",
        coordinator_name,
        "--coordinator-config",
        json.dumps(coordinator),
        "--controller",
        "static" if series.static else "jit",
        "--controller-config",
        json.dumps(controller),
        "--simulator",
        "static" if series.static else "jit-static",
        "--simulator-config",
        json.dumps(simulator),
        "--trace",
        str((directory / "timing.pb").resolve()),
    ]
    if policy_command is not None:
        command = [policy_command, "--localalloc", *command]
    if settings["frontend_cpus"]:
        command = ["taskset", "--cpu-list", settings["frontend_cpus"], *command]
    environment = {
        "TOKIO_WORKER_THREADS": str(settings["tokio_threads"]),
        "RAYON_NUM_THREADS": str(DEFAULT_RAYON_NUM_THREADS),
        "DEQ_TRACE_MODE": "latency",
        "DEQ_TRACE_DIAGNOSTICS": (
            "1" if settings.get("system_diagnostics", False) else "0"
        ),
        "DEQ_TRACE_WORKER_SAMPLE_RATE": str(worker_sample_rate),
        "DEQ_TRACE_MAX_BYTES": str(MAX_FILE_BYTES),
        "QDK_PYTHON_TELEMETRY": "none",
    }
    if "GLIBC_TUNABLES" in os.environ:
        environment["GLIBC_TUNABLES"] = os.environ["GLIBC_TUNABLES"]
    native = _native_path()
    runtime_hash = file_hash(native)
    checkpoint(
        directory / "command.json",
        {
            "command": command,
            "environment": environment,
            "runtime_sha256": runtime_hash,
            "controller_sha256": file_hash(controller_path),
            "static_compiler_sha256": (
                (compiler_sha256 or runtime_hash) if series.static else None
            ),
            "sample_manifest_sha256": file_hash(case / "samples.json"),
            "supervisor_nice": psutil.Process().nice(),
            "shot_inputs": [
                {
                    "shot": shot,
                    "warmup": shot < warmups,
                    **manifest["samples"][shot % len(sample_files)],
                }
                for shot in range(total)
            ],
        },
    )
    resources = guarded(
        command,
        directory / "runtime.log",
        env=os.environ | environment,
    )
    log = (directory / "runtime.log").read_text()
    if file_hash(native) != runtime_hash:
        raise ValueError("The installed native runtime changed during measurement")
    if "panicked" in log:
        raise ValueError(f"Native panic during replay: {directory}")
    summary = log.rpartition("=== Simulation Complete ===")[2]
    if not summary:
        raise ValueError(f"Native simulation did not finish normally: {directory}")
    reported = {}
    for label in ("Shots", "Logical errors", "Failed shots"):
        match = re.search(rf"^\s*{label}:\s+(\d+)(?:/\d+)?\s*$", summary, re.MULTILINE)
        if match is None:
            raise ValueError(f"Missing native result {label}: {directory}")
        reported[label] = int(match[1])
    if reported["Shots"] != total or reported["Failed shots"]:
        raise ValueError(
            "Failed or incomplete shots cannot enter the warm distributions"
        )
    seen = set()
    logical_errors = warmup_errors = 0
    for data in protobuf_records(directory / "shots.pb"):
        shot = SimulatorShot.FromString(data)
        if shot.shot in seen or not shot.HasField("decode_result"):
            raise ValueError("Failed or duplicate simulator shot")
        seen.add(shot.shot)
        actual = {
            readout.gid: readout.readouts.size for readout in shot.gadget_readouts
        }
        if len(actual) != len(shot.gadget_readouts) or actual != expected_readouts:
            raise ValueError("Missing, duplicate, or incomplete gadget readouts")
        if shot.decode_result.readouts.size != configuration["readouts"]:
            raise ValueError("Incomplete aggregate logical readouts")
        if shot.shot < warmups:
            warmup_errors += int(shot.logical_error)
        else:
            logical_errors += int(shot.logical_error)
    if (
        seen != set(range(total))
        or logical_errors + warmup_errors != reported["Logical errors"]
    ):
        raise ValueError("Native log and simulator shot records disagree")
    samples, coverage = readout_samples(
        {
            "run_directory": str(directory),
            "controller": "static" if series.static else "jit",
            "shots": total,
            "warmup_shots": warmups,
            "require_lossless_trace": True,
        }
    )
    populations = {
        region: population(
            [sample.latency_ms for sample in samples if sample.region == region]
        )
        for region in ("intermediate", "final")
    }
    expected = {
        "final": configuration["width"] * measured,
        "intermediate": (configuration["readouts"] - configuration["width"]) * measured,
    }
    if any(
        populations[region]["samples"] != count for region, count in expected.items()
    ):
        raise ValueError("Warm sample counts do not match the requested protocol")
    point = {
        "series": series.key,
        "width": configuration["width"],
        "shots": measured,
        "warmup_shots": warmups,
        "native_shots": total,
        "execution_settings": settings,
        "compilation_resources": compilation_resources,
        "populations": populations,
        "coverage": coverage,
        "logical_error_shots": logical_errors,
        "warmup_logical_error_shots": warmup_errors,
        "runtime_sha256": runtime_hash,
        "run_directory": str(directory.resolve()),
        "input_kind": "raw_binary_physical_replay",
        "input_provenance": {
            "manifest": str((case / "samples.json").resolve()),
            "sha256": file_hash(case / "samples.json"),
        },
        "registration_lead_seconds": simulator.get("registration_lead_seconds"),
        "latency_origin": "native start_decoding; sampling and static program setup excluded",
        **resources,
    }
    checkpoint(directory / "summary.json", point)
    return point


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compile-static", type=Path, nargs=2, help=argparse.SUPPRESS)
    parser.add_argument("--max-width", type=int, default=8192)
    parser.add_argument("--min-width", type=int, default=1)
    parser.add_argument("--layers", type=int, default=50)
    parser.add_argument("--probability", type=float, default=0.001)
    parser.add_argument(
        "--numa-policy",
        choices=("inherit", "local"),
        default="inherit",
        help="Explicit local-first allocation uses numactl for this process only",
    )
    parser.add_argument(
        "--trim-heap",
        action="store_true",
        help="Reclaim unused glibc pages before each shot; excluded from decoder timing and logged",
    )
    parser.add_argument(
        "--series",
        nargs="+",
        choices=[series.key for series in SERIES],
        default=[series.key for series in SERIES],
    )
    parser.add_argument("--memory-gib", type=float, default=320)
    parser.add_argument("--task-timeout-seconds", type=float, default=14400)
    parser.add_argument("--wait-for-samples", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    arguments = parser.parse_args()
    if arguments.compile_static is not None:
        compile_static_file(*(str(path) for path in arguments.compile_static))
        return
    if arguments.samples is None or arguments.output is None:
        parser.error("--samples and --output are required for an evaluation")
    if not 0 <= arguments.probability < 0.5:
        raise ValueError("probability must be in [0, 0.5)")
    if not 1 <= arguments.max_width <= 8192 or arguments.max_width & (
        arguments.max_width - 1
    ):
        raise ValueError("Maximum width must be a power of two through 8192")
    if (
        not 1 <= arguments.min_width <= arguments.max_width
        or arguments.min_width & (arguments.min_width - 1)
        or arguments.layers < 1
    ):
        raise ValueError(
            "Minimum width must be a power of two through maximum width; layers must be positive"
        )
    if any(
        not math.isfinite(value) or value <= 0
        for value in (arguments.memory_gib, arguments.task_timeout_seconds)
    ):
        raise ValueError("Memory and timeout budgets must be finite and positive")
    if not set(range(120)) <= os.sched_getaffinity(0):
        raise ValueError("This evaluation requires the qualified 120-CPU topology")
    arguments.output.mkdir(parents=True, exist_ok=True)
    with interrupt_on_termination(), exclusive_output(arguments.output / ".active"):
        _evaluate(arguments)


def _evaluate(arguments: argparse.Namespace) -> None:
    if arguments.wait_for_samples:
        print("Waiting for every physical sampling batch before decoding.", flush=True)
    deadline = time.monotonic() + 36000
    while True:
        sampling = json.loads((arguments.samples / "sampling.json").read_text())
        if sampling["status"] == "complete":
            break
        if sampling["status"] != "running" or not arguments.wait_for_samples:
            raise ValueError(
                "All physical samples must complete before decoding begins"
            )
        if time.monotonic() > deadline:
            raise TimeoutError("Physical preparation did not finish within ten hours")
        time.sleep(10)
    print(
        "Physical sampling is complete; starting sequential warm-only evaluation.",
        flush=True,
    )
    expected_widths = [
        1 << power
        for power in range(
            arguments.min_width.bit_length() - 1, arguments.max_width.bit_length()
        )
    ]
    chosen_series = [series for series in SERIES if series.key in arguments.series]
    if not set(expected_widths) <= set(sampling["widths"]):
        raise ValueError("Sampling campaign does not contain every requested width")
    if sampling["shots_per_width"] != 10:
        raise ValueError("This warm-only protocol requires ten raw shots per width")
    if sampling.get("layers", 50) != arguments.layers:
        raise ValueError("Sampling campaign uses a different circuit depth")
    if sampling.get("probability", 0.001) != arguments.probability:
        raise ValueError(
            "Sampling campaign uses a different physical error probability"
        )
    for width in expected_widths:
        verify_samples(arguments.samples / f"w{width}", sampling["points"][str(width)])
    settings = {
        "max_width": arguments.max_width,
        "min_width": arguments.min_width,
        "series": [series.key for series in chosen_series],
        "nice": psutil.Process().nice(),
        "numa_policy": arguments.numa_policy,
        "trim_heap": arguments.trim_heap,
        "memory_gib": arguments.memory_gib,
        "task_timeout_seconds": arguments.task_timeout_seconds,
        "stop_mean_ms": 1000,
        "warmup_shots": 2,
        "measured_shots": 10,
        "threads": 72,
        "frontend_cpus": "0-29",
        "tokio_threads": 24,
        "rayon_threads": DEFAULT_RAYON_NUM_THREADS,
        "pools": POOLS,
        "automatic_collection": False,
        "compact_static_library": True,
        "latency_trace_transport": "lossless",
        "latency_clock": "shared monotonic anchor",
        "static_compiler_threads": DEFAULT_TOKIO_WORKER_THREADS,
        "distance": 7,
        "probability": arguments.probability,
        "layers": arguments.layers,
        "seed": 42,
        "interval_ms": 70,
    }
    identity = provenance()
    identity["replay_sources"] = {
        name: file_hash(HERE / name)
        for name in ("warm_replay.py", "replay_sampler.py", "readout_latency.py")
    }
    identity["sampling_manifest_sha256"] = file_hash(
        arguments.samples / "sampling.json"
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    output = arguments.output / "results.json"
    if output.exists():
        if not arguments.resume:
            raise FileExistsError(output)
        document = json.loads(output.read_text())
        if document["settings"] != settings or document["provenance"] != identity:
            raise ValueError(
                "Resume requires identical settings, samples, code, and binaries"
            )
        if "failure" in document:
            document.setdefault("failure_history", []).append(document.pop("failure"))
    else:
        document = {
            "schema_version": 3,
            "status": "running",
            "settings": settings,
            "input_protocol": "Warm-only binary physical replay: 2 warm-ups, then 10 measured shots; files cycle",
            "series": [asdict(series) for series in chosen_series],
            "minimum_samples": {
                "intermediate": 3
                * ((arguments.layers + 1) // 2)
                * settings["measured_shots"],
                "final": settings["measured_shots"],
            },
            "latency_definition": "Readout-owning window commitment minus its last required scheduled PROGRAM input; only declared warm-up shots excluded",
            "provenance": identity,
            "points": [],
            "stopped": {},
        }
    checkpoint(output, document)
    try:
        for width in expected_widths:
            for series in chosen_series:
                if series.key in document["stopped"] or any(
                    point["width"] == width and point["series"] == series.key
                    for point in document["points"]
                ):
                    continue
                directory = arguments.output / "work" / f"w{width}" / series.key
                completed = directory / "summary.json"
                if completed.exists():
                    point = json.loads(completed.read_text())
                    if point["runtime_sha256"] != identity["binary_sha256"][
                        "python_runtime"
                    ] or point["input_provenance"]["sha256"] != file_hash(
                        arguments.samples / f"w{width}" / "samples.json"
                    ):
                        raise ValueError(
                            "Completed point has incompatible runtime or samples"
                        )
                else:
                    if directory.exists():
                        if not arguments.retry_failed:
                            raise ValueError(
                                f"Incomplete replay point: {directory}; use --resume --retry-failed to archive and retry"
                            )
                        failed = (
                            arguments.output
                            / "failed"
                            / f"w{width}-{series.key}-{time.time_ns()}"
                        )
                        failed.parent.mkdir(exist_ok=True)
                        directory.rename(failed)
                    point = run_point(
                        arguments.samples / f"w{width}", directory, series, settings
                    )
                if (
                    point["runtime_sha256"]
                    != identity["binary_sha256"]["python_runtime"]
                ):
                    raise ValueError("Native runtime changed between evaluation points")
                if (
                    point["shots"] != settings["measured_shots"]
                    or point["warmup_shots"] != settings["warmup_shots"]
                    or point["native_shots"]
                    != settings["measured_shots"] + settings["warmup_shots"]
                ):
                    raise ValueError(
                        "Completed point uses a different warm-up or shot protocol"
                    )
                if (
                    point.get("execution_settings") != settings
                    or point["series"] != series.key
                    or point["width"] != width
                ):
                    raise ValueError(
                        "Completed point uses different execution settings or workload"
                    )
                document["points"].append(point)
                crossed = overloaded(point, settings["stop_mean_ms"])
                if crossed or width == settings["max_width"]:
                    document["stopped"][series.key] = {
                        "width": width,
                        "reason": "threshold" if crossed else "width_limit",
                        "mean_ms": {
                            region: group["mean_ms"]
                            for region, group in point["populations"].items()
                        },
                    }
                checkpoint(output, document)
                print(
                    f"{series.key} W={width}: "
                    + str(
                        {
                            region: group["mean_ms"]
                            for region, group in point["populations"].items()
                        }
                    ),
                    flush=True,
                )
        document["status"] = "complete"
        checkpoint(output, document)
        write_summary(document, arguments.output / "performance-data.json")
        plot(
            arguments.output / "performance-data.json",
            arguments.output / "performance.png",
        )
        if (
            len(document["points"]) == 1
            and document["points"][0]["series"] == "persistent"
        ):
            from plot_latency_distribution import generate_report

            generate_report(
                Path(document["points"][0]["run_directory"]),
                arguments.output / "intermediate-latency.png",
            )
    except BaseException as error:
        document.update(
            status="stopped",
            failure={"error": repr(error), "width": width, "series": series.key},
        )
        checkpoint(output, document)
        raise


if __name__ == "__main__":
    main()
