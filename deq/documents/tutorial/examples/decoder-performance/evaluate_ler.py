"""Evaluate fresh physical samples with `deq simulate ler` and plot whole-program LER."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "jit-basics"))

from lattice_surgery import circuit_source
from study_utils import run_process, save_json

SERIES = {"mwpm": "MWPM", "mwpf": "MWPF c=10"}
WORKLOAD = dict(distance=7, probability=0.001, layers=50, basis="Z", seed=42)
TARGET_ERRORS = 20
STOP_RATE = 0.8
MAX_SHOTS = 10_000_000


def parse_result(log: str) -> dict:
    _, marker, summary = log.rpartition("=== Simulation Results ===")
    if not marker:
        raise ValueError("CLI did not report completed simulation results")
    counts = {}
    for key, label in (
        ("shots", "Shots"),
        ("errors", "Logical errors"),
        ("failed", "Failed shots"),
    ):
        match = re.search(rf"^\s*{label}:\s+(\d+)\s*$", summary, re.MULTILINE)
        if match is None:
            raise ValueError(f"CLI result is missing {label}")
        counts[key] = int(match[1])
    if counts["failed"] or not 0 < counts["errors"] <= counts["shots"]:
        raise ValueError(f"failed or invalid simulation counts: {counts}")
    if counts["errors"] < TARGET_ERRORS:
        raise ValueError(f"error target was not reached: {counts}")
    return {**counts, "rate": counts["errors"] / counts["shots"]}


def identity() -> dict:
    native = importlib.import_module("deq_runtime.deq_runtime").__file__
    parser = importlib.import_module("deqagram.deqagram").__file__
    if native is None or parser is None:
        raise RuntimeError("cannot identify the native runtime and parser")
    deq = HERE.parents[3]
    paths = [
        Path(native),
        Path(parser),
        Path(__file__),
        HERE / "study_utils.py",
        HERE.parent / "jit-basics/lattice_surgery.py",
        deq / "deq/cli/jit.py",
        deq / "deq/cli/simulate.py",
        deq / "deq/transpiler/program_timing.py",
        deq / "deq/transpiler/program_artifacts.py",
    ]
    return {
        "python": sys.version,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("deq", "deq-runtime", "stim")
        },
        "sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
    }


def evaluate(
    output: Path, *, jobs: int, threads: int, memory_gib: float, resume: bool
) -> dict:
    if jobs < 1 or threads < 1:
        raise ValueError("jobs and threads must be positive")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = {
        **WORKLOAD,
        "jobs": jobs,
        "threads_per_job": threads,
        "memory_gib": memory_gib,
        "target_errors": TARGET_ERRORS,
        "stop_rate": STOP_RATE,
        "max_shots": MAX_SHOTS,
        "buffer_radius": 1,
        "lookahead_radius": 0,
        "persistent_decoder": False,
    }
    provenance = identity()
    if output.exists():
        if not resume:
            raise FileExistsError(f"use --resume or a new results path: {output}")
        result = json.loads(output.read_text())
        if result["settings"] != settings or result["provenance"] != provenance:
            raise ValueError("resume requires identical code, binaries and settings")
    else:
        result = {
            "input_protocol": "fresh physical sampling; native completion-ordered DEQ",
            "settings": settings,
            "provenance": provenance,
            "points": [],
            "status": "running",
        }
    save_json(output, result)
    width = 1
    try:
        while True:
            stopped = {
                point["decoder"]
                for point in result["points"]
                if point["rate"] > STOP_RATE
            }
            if stopped == set(SERIES):
                break
            case = output.parent / "ler-work" / f"w{width}"
            case.mkdir(parents=True, exist_ok=True)
            source = circuit_source(width=width, **WORKLOAD)
            source_path = case / "circuit.deq"
            if source_path.exists() and source_path.read_text() != source:
                raise ValueError(f"workload changed: {source_path}")
            source_path.write_text(source)
            for decoder in SERIES:
                if decoder in stopped or any(
                    p["width"] == width and p["decoder"] == decoder
                    for p in result["points"]
                ):
                    continue
                run = case / decoder
                run.mkdir()  # A failed attempt must be inspected, not silently overwritten.
                decoder_config = {"parallel": threads}
                if decoder == "mwpf":
                    decoder_config["cluster_node_limit"] = 10
                command = [
                    sys.executable,
                    "-m",
                    "deq",
                    "simulate",
                    "ler",
                    str(source_path),
                    "--program",
                    "Brickwall",
                    "--errors",
                    str(TARGET_ERRORS),
                    "--shots",
                    str(MAX_SHOTS),
                    "--batch-size",
                    "100",
                    "--jobs",
                    str(jobs),
                    "--decoder",
                    decoder,
                    "--decoder-config",
                    json.dumps(decoder_config),
                    "--coordinator",
                    "window",
                    "--coordinator-config",
                    json.dumps(
                        {
                            "buffer_radius": 1,
                            "lookahead_radius": 0,
                            "persistent_decoder": False,
                            "assert_parity_factor": True,
                        }
                    ),
                    "--simulator",
                    "jit-static",
                    "--loss-model",
                    "none",
                    "--seed",
                    str(WORKLOAD["seed"]),
                    "--save",
                    str(run),
                    "--timeout",
                    "36000",
                    "--skip-mako-warning",
                ]
                save_json(run / "command.json", command)
                print(
                    f"W={width} {decoder}: collecting at least {TARGET_ERRORS} errors",
                    flush=True,
                )
                resources = run_process(
                    command,
                    run / "cli.log",
                    memory_gib=memory_gib,
                    timeout_seconds=36_000,
                    cwd=HERE.parents[3],
                )
                point = {
                    "width": width,
                    "decoder": decoder,
                    **parse_result((run / "cli.log").read_text()),
                    **resources,
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                }
                result["points"].append(point)
                result["status"] = "running"
                result.pop("failure", None)
                save_json(output, result)
                print(
                    f"  {point['errors']}/{point['shots']} = {point['rate']:.4%}",
                    flush=True,
                )
            width *= 2
        result["status"] = "complete"
        result.pop("failure", None)
        save_json(output, result)
    except (Exception, KeyboardInterrupt) as error:
        result.update(status="failed", failure={"width": width, "message": str(error)})
        save_json(output, result)
        raise
    return result


def wilson(errors: int, shots: int) -> tuple[float, float]:
    rate = errors / shots
    z = 1.959963984540054
    scale = 1 + z * z / shots
    center = (rate + z * z / (2 * shots)) / scale
    half = z * math.sqrt(rate * (1 - rate) / shots + z * z / (4 * shots**2)) / scale
    return center - half, center + half


def plot(document: dict, output: Path, *, allow_incomplete: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if document["status"] not in {"complete", "failed", "stopped"}:
        raise ValueError("evaluation must finish or stop before publication")
    if document["status"] != "complete" and not allow_incomplete:
        raise ValueError("only completed evaluations can be published")
    curves = []
    for decoder, color in (("mwpm", "#e67e00"), ("mwpf", "#8b45b4")):
        points = sorted(
            (p for p in document["points"] if p["decoder"] == decoder),
            key=lambda p: p["width"],
        )
        if [p["width"] for p in points] != [
            2**index for index in range(len(points))
        ] or not points:
            raise ValueError("each curve must contain consecutive powers of two")
        for index, point in enumerate(points):
            if (
                point["failed"]
                or point["errors"] < TARGET_ERRORS
                or not 0 < point["errors"] <= point["shots"]
                or point["rate"] != point["errors"] / point["shots"]
            ):
                raise ValueError("invalid or insufficiently sampled LER point")
            if point["rate"] > STOP_RATE and index != len(points) - 1:
                raise ValueError("each curve must end at its first rate above 80%")
        incomplete = points[-1]["rate"] <= STOP_RATE
        if incomplete and (not allow_incomplete or document["status"] == "complete"):
            raise ValueError("each completed curve must end above 80%")
        curves.append((decoder, color, points, incomplete))
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for decoder, color, points, incomplete in curves:
        bounds = [wilson(p["errors"], p["shots"]) for p in points]
        axis.errorbar(
            [p["width"] for p in points],
            [p["rate"] for p in points],
            yerr=[
                [max(0, p["rate"] - bound[0]) for p, bound in zip(points, bounds)],
                [max(0, bound[1] - p["rate"]) for p, bound in zip(points, bounds)],
            ],
            fmt="o-",
            color=color,
            label=SERIES[decoder] + (" (incomplete)" if incomplete else ""),
            capsize=3,
        )
    widths = sorted({point["width"] for point in document["points"]})
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(widths, widths)
    axis.set_ylim(top=1.1)
    axis.set_xlabel("Data patches W")
    axis.set_ylabel("Whole-program logical error rate")
    axis.axhline(
        STOP_RATE,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="80% stopping threshold",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    if any(incomplete for _, _, _, incomplete in curves):
        figure.text(
            0.5,
            0.01,
            "Incomplete curves end at measured points, not at an 80% crossing.",
            ha="center",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.04, 1, 1))
    else:
        figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "ler-results.json")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--memory-gib", type=float, default=24)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Publish stopped results with incomplete curves explicitly labeled; requires --plot-only",
    )
    args = parser.parse_args()
    if args.allow_incomplete and not args.plot_only:
        parser.error("--allow-incomplete requires --plot-only")
    document = (
        json.loads(args.results.read_text())
        if args.plot_only
        else evaluate(
            args.results,
            jobs=args.jobs,
            threads=args.threads,
            memory_gib=args.memory_gib,
            resume=args.resume,
        )
    )
    plot(
        document,
        args.results.with_name("logical-error-rate.png"),
        allow_incomplete=args.allow_incomplete,
    )


if __name__ == "__main__":
    main()
