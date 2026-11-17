"""Measure decoder configurations, stopping each at overload."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DEQ = HERE.parents[3]
JIT_EXAMPLE = HERE.parent / "jit-basics"
sys.path.insert(0, str(JIT_EXAMPLE))

from lattice_surgery import compile_case
from lattice_surgery_latency import run_case
from study_utils import run_process, sampling_plan, save_json
from summarize import summary_path, write_summary


@dataclass(frozen=True)
class Series:
    key: str
    label: str
    decoder: str
    static: bool = False


BASIS = "Z"
INPUT_PROTOCOL = "fresh physical sampling at every width; native PROGRAM timing"
SERIES = (
    Series("mwpf", "MWPF c=10", "mwpf"),
    Series("mwpm", "MWPM", "mwpm"),
    Series("persistent", "Static + persistent MWPM", "mwpm", static=True),
    Series("persistent_mwpf", "Static + persistent MWPF c=10", "mwpf", static=True),
)


@dataclass(frozen=True)
class Settings:
    threads: int = 24
    memory_gib: float = 24
    distance: int = 7
    probability: float = 0.001
    layers: int = 50
    interval_ms: float = 70
    seed: int = 42
    stop_mean_ms: float = 1_000
    max_width: int | None = None
    task_timeout_seconds: float = 3600

    def validate(self) -> None:
        if self.max_width is not None and (
            self.max_width < 1 or self.max_width & (self.max_width - 1)
        ):
            raise ValueError("maximum width must be a positive power of two")
        if (
            self.threads < 1
            or self.layers < 1
            or self.distance < 3
            or self.distance % 2 == 0
        ):
            raise ValueError(
                "positive workers/layers and an odd distance >= 3 are required"
            )
        for value in (
            self.memory_gib, self.interval_ms, self.stop_mean_ms, self.task_timeout_seconds
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "memory, interval, stopping latency, and task timeout must be positive and finite"
                )
        if not 0 <= self.probability < 0.5:
            raise ValueError("physical error probability must be in [0, 0.5)")


def overloaded(point: dict, threshold_ms: float) -> bool:
    return any(
        group["mean_ms"] is not None and group["mean_ms"] > threshold_ms
        for group in point["populations"].values()
    )


def prepare_point(result: dict, series: Series, expected: dict) -> dict:
    if not result["timings_qualified"] or result["failed_shots"]:
        raise ValueError("failed or censored runs cannot become evaluation points")
    groups = result["populations"]
    for region, group in groups.items():
        if group["samples"] != expected[region]:
            raise ValueError(f"{region} sample count differs from the requested plan")
    return {
        "series": series.key,
        "width": result["width"],
        "shots": result["shots"],
        "populations": groups,
        "logical_error_shots": result["logical_errors"],
        "peak_rss_gib": result["peak_rss_gib"],
        "wall_seconds": result["wall_seconds"],
        "run_directory": result["run_directory"],
        "runtime_sha256": result["runtime_sha256"],
        "input_kind": result["input_kind"],
        "input_provenance": result.get("input_provenance"),
        "registration_lead_seconds": result["registration_lead_seconds"],
        "latency_origin": result["latency_origin"],
    }


def task_timeout(job: dict) -> float:
    settings = Settings(**job["settings"])
    plan = sampling_plan([job["width"]], settings.layers)[0]
    scheduled = (3 * settings.layers + 2) * settings.interval_ms / 1000
    return max(
        settings.task_timeout_seconds,
        4 * plan["shots"] * (scheduled + settings.stop_mean_ms / 1000),
    )


def perform_task(job: dict) -> dict:
    settings = Settings(**job["settings"])
    settings.validate()
    case = Path(job["case"])
    width = job["width"]
    if job["task"] == "prepare":
        return compile_case(
            case,
            distance=settings.distance,
            probability=settings.probability,
            width=width,
            layers=settings.layers,
            basis=BASIS,
            seed=settings.seed,
            block_seconds=settings.interval_ms / 1000,
            export_physical=False,
        )
    series = next(item for item in SERIES if item.key == job["task"])
    shots = job["plan"]["shots"]
    timeout = task_timeout(job)
    result = run_case(
        case,
        shots=shots,
        threads=settings.threads,
        noiseless=False,
        simulator="static" if series.static else "jit-static",
        decoder=series.decoder,
        decoder_config={"cluster_node_limit": 10} if series.decoder == "mwpf" else {},
        memory_gib=settings.memory_gib,
        timeout_seconds=timeout,
        registration_lead_seconds=(
            None if series.static else 2 * settings.interval_ms / 1000
        ),
    )
    return prepare_point(result, series, job["plan"])


def archive_failed_task(job: dict, request: Path) -> None:
    case = Path(job["case"])
    task = job["task"]
    if task == "prepare":
        candidates = [case] if case.exists() else []
    else:
        series = next(item for item in SERIES if item.key == task)
        run = case / f"{'static' if series.static else 'jit-static'}-{series.decoder}"
        candidates = [run] if run.exists() else []
    archive = request.parent / "failed" / f"{task}-{time.time_ns()}"
    archive.mkdir(parents=True)
    for path in [request, request.with_suffix(".log"), *candidates]:
        if path.exists():
            path.rename(archive / path.name)
    tqdm.write(
        f"Retrying {task} at W={job['width']}; failed attempt retained in {archive}"
    )


def run_worker(
    job: dict,
    directory: Path,
    *,
    retry_failed: bool = False,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    request = directory / f"{job['task']}.json"
    response = request.with_suffix(".result.json")
    if response.exists() and not request.exists():
        raise ValueError(f"result has no matching task request: {response}")
    if request.exists():
        previous = json.loads(request.read_text())
        if previous != job:
            raise ValueError(
                f"incomplete or incompatible task: {request}; use a fresh results directory"
            )
        if response.exists():
            return json.loads(response.read_text())
        if not retry_failed:
            raise ValueError(
                f"incomplete task: {request}; use --resume --retry-failed to archive and retry"
            )
        archive_failed_task(job, request)
    save_json(request, job)
    run_process(
        [sys.executable, str(Path(__file__).resolve()), "--worker", str(request)],
        request.with_suffix(".log"),
        memory_gib=job["settings"]["memory_gib"],
        timeout_seconds=task_timeout(job) + 300,
    )
    return json.loads(response.read_text())


def provenance() -> dict:
    native = importlib.import_module("deq_runtime.deq_runtime").__file__
    parser = importlib.import_module("deqagram.deqagram").__file__
    if native is None or parser is None:
        raise RuntimeError("cannot identify the installed native runtime and parser")
    binaries = {"python_runtime": Path(native), "parser": Path(parser)}
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "logical_cpus": os.cpu_count(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("deq", "deq-runtime", "stim", "numpy")
        },
        "binary_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in binaries.items()
        },
        "source_sha256": {
            str(path.relative_to(DEQ)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).resolve(),
                DEQ / "deq" / "cli" / "jit.py",
                DEQ / "deq" / "cli" / "simulate.py",
                DEQ / "deq" / "transpiler" / "program_timing.py",
                DEQ / "deq" / "transpiler" / "program_artifacts.py",
                JIT_EXAMPLE / "lattice_surgery.py",
                *(
                    HERE / name
                    for name in (
                        "lattice_surgery_latency.py",
                        "study_utils.py",
                        "readout_latency.py",
                    )
                ),
            )
        },
    }


def validate_resume(document: dict, settings: Settings, identity: dict) -> None:
    if (
        document.get("schema_version") != 3
        or document.get("input_protocol") != INPUT_PROTOCOL
    ):
        raise ValueError("resume requires the native-timing input protocol")
    if document["series"] != [asdict(series) for series in SERIES]:
        raise ValueError(
            "resume requires the same decoder configurations; use a fresh results directory"
        )
    if document["settings"] != asdict(settings):
        raise ValueError("resume requires the same evaluation settings")
    if document["provenance"] != identity:
        raise ValueError(
            "resume requires unchanged environment, native builds, and evaluation sources"
        )


def evaluate(
    settings: Settings,
    output: Path,
    *,
    resume: bool = False,
    retry_failed: bool = False,
) -> dict:
    settings.validate()
    identity = provenance()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not resume:
            raise FileExistsError(
                f"refusing to overwrite {output}; use --resume or a fresh results directory"
            )
        document = json.loads(output.read_text())
        validate_resume(document, settings, identity)
        if "failure" in document:
            document.setdefault("failure_history", []).append(document.pop("failure"))
        if document["status"] != "complete":
            document["status"] = "running"
        save_json(output, document)
    else:
        if (output.parent / "work").exists():
            raise FileExistsError(
                "raw work already exists; resume its results file or choose a fresh directory"
            )
        document = {
            "schema_version": 3,
            "input_protocol": INPUT_PROTOCOL,
            "settings": asdict(settings),
            "provenance": identity,
            "series": [asdict(series) for series in SERIES],
            "points": [],
            "stopped": {},
            "status": "running",
            "minimum_samples": {"intermediate": 1001, "final": 100},
            "latency_definition": (
                "Readout-owning window commit minus its last required native gadget "
                "completion, relative to the shot's start_decoding event; physical "
                "sampling and static program setup excluded."
            ),
        }
        save_json(output, document)
    width_limit = (
        f" or reaches W={settings.max_width}"
        if settings.max_width is not None else " (no width limit)"
    )
    print(
        "Widths double until each series first exceeds "
        f"{settings.stop_mean_ms:g} ms mean latency{width_limit}. "
        "Tasks run sequentially.",
        flush=True,
    )
    width = 1
    try:
        while len(document["stopped"]) < len(SERIES) and (
            settings.max_width is None or width <= settings.max_width
        ):
            existing = {
                point["series"]: point
                for point in document["points"]
                if point["width"] == width
            }
            active = [
                series
                for series in SERIES
                if series.key not in document["stopped"] and series.key not in existing
            ]
            plan = sampling_plan([width], settings.layers)[0]
            case = output.parent / "work" / f"w{width}"
            tasks = (["prepare"] if active and not existing else []) + [
                series.key for series in active
            ]
            with tqdm(
                total=len(tasks),
                desc=f"W={width}",
                unit="task",
                dynamic_ncols=True,
                bar_format="{desc}: {bar} {n_fmt}/{total_fmt} tasks [{elapsed}] {postfix}",
            ) as progress:
                for task in tasks:
                    progress.set_postfix_str(task)
                    job = {
                        "task": task,
                        "settings": asdict(settings),
                        "width": width,
                        "case": str(case.resolve()),
                        "plan": plan,
                    }
                    result = run_worker(
                        job,
                        case.parent / "tasks" / case.name,
                        retry_failed=retry_failed,
                    )
                    if task != "prepare":
                        document["points"].append(result)
                        existing[task] = result
                        crossed = overloaded(result, settings.stop_mean_ms)
                        if crossed or width == settings.max_width:
                            document["stopped"][task] = {
                                "width": width,
                                "reason": "threshold" if crossed else "width_limit",
                                "mean_ms": {
                                    key: value["mean_ms"]
                                    for key, value in result["populations"].items()
                                },
                            }
                        document["status"] = "running"
                        document.pop("failure", None)
                        save_json(output, document)
                    progress.update(1)
            width *= 2
        document["status"] = "complete"
        document.pop("failure", None)
        write_summary(document, summary_path(output))
        save_json(output, document)
    except (Exception, KeyboardInterrupt) as error:
        document["status"] = "failed"
        document["failure"] = {
            "width": width,
            "type": type(error).__name__,
            "message": str(error),
        }
        save_json(output, document)
        raise
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results.json")
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--memory-gib", type=float, default=24)
    parser.add_argument(
        "--max-width", type=int,
        help="Optional power-of-two width cap; by default only latency stops a series",
    )
    parser.add_argument(
        "--task-timeout-seconds", type=float, default=3600,
        help="Minimum timeout per preparation or measurement task (default: 3600 seconds)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Archive incomplete task outputs and retry them when resuming",
    )
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.retry_failed and not arguments.resume:
        parser.error("--retry-failed requires --resume")
    if arguments.worker:
        save_json(
            arguments.worker.with_suffix(".result.json"),
            perform_task(json.loads(arguments.worker.read_text())),
        )
    else:
        evaluate(
            Settings(
                threads=arguments.threads,
                memory_gib=arguments.memory_gib,
                max_width=arguments.max_width,
                task_timeout_seconds=arguments.task_timeout_seconds,
            ),
            arguments.results.resolve(),
            resume=arguments.resume,
            retry_failed=arguments.retry_failed,
        )


if __name__ == "__main__":
    main()
