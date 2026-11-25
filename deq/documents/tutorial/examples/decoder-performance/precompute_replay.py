"""Prepare raw physical shots with descending-width, SIMD-batched worker tasks."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

import psutil
import stim

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jit-basics"))
from lattice_surgery import compile_case
import lattice_surgery
from replay_qsharp import MAX_FILE_BYTES, build_recipe
from replay_stim import sample_batch
from study_utils import exclusive_output, interrupt_on_termination, terminate_processes


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint(path: Path, value: dict) -> None:
    encoded = (json.dumps(value, indent=2) + "\n").encode()
    if len(encoded) > MAX_FILE_BYTES:
        raise ValueError(f"{path} exceeds the 10 GB file guard")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def verify_samples(directory: Path, report: dict) -> None:
    shots, measurements = report["shots"], report["measurements"]
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (shots, measurements)
    ):
        raise ValueError("Shot and measurement counts must be integers")
    if shots < 1 or measurements < 0:
        raise ValueError("Invalid sampling dimensions")
    if report["status"] != "complete" or len(report["samples"]) != shots:
        raise ValueError("Incomplete sampling manifest")
    expected = (measurements + 7) // 8
    if expected > MAX_FILE_BYTES:
        raise ValueError("Raw sample exceeds the 10 GB guard")
    seen = set()
    for index, sample in enumerate(report["samples"]):
        name = sample["file"]
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in (".", "..")
        ):
            raise ValueError(
                "Sample names must identify files inside the sample directory"
            )
        path = (directory / name).resolve(strict=True)
        if path.parent != directory.resolve() or path in seen:
            raise ValueError(
                "Sample paths must be distinct and remain inside their directory"
            )
        seen.add(path)
        if (
            type(sample["batch_instance"]) is not int
            or type(sample["bytes"]) is not int
            or sample["batch_instance"] != index
            or sample["bytes"] != expected
        ):
            raise ValueError("Sample ordering or byte counts do not match the manifest")
        if path.stat().st_size != expected or file_hash(path) != sample["sha256"]:
            raise ValueError(f"Raw sample changed or is incomplete: {path}")
        if measurements % 8:
            with path.open("rb") as stream:
                stream.seek(-1, os.SEEK_END)
                if stream.read(1)[0] >> (measurements % 8):
                    raise ValueError(f"Raw sample has nonzero padding: {path}")


def expected_measurements(width: int, layers: int) -> int:
    """Count d=7 physical measurements without expanding the brickwall."""
    if width < 1 or layers < 1:
        raise ValueError("Width and layers must be positive")
    cnots = ((layers + 1) // 2) * (width // 2) + (layers // 2) * ((width - 1) // 2)
    idle_blocks = 3 * (width * layers - 2 * cnots)
    return 673 * width + 3199 * cnots + 336 * idle_blocks


def prepare_width(
    root: str, width: int, shots: int, layers: int = 50, probability: float = 0.001
) -> dict:
    if not 0 <= probability < 0.5:
        raise ValueError("probability must be in [0, 0.5)")
    directory = Path(root) / f"w{width}"
    directory.mkdir(exist_ok=True)
    with exclusive_output(directory / ".active"):
        return _prepare_width(root, width, shots, layers, probability)


def _prepare_width(
    root: str, width: int, shots: int, layers: int, probability: float
) -> dict:
    if (expected_measurements(width, layers) + 7) // 8 > MAX_FILE_BYTES:
        raise ValueError("Planned raw shot exceeds the 10 GB file guard")
    directory = Path(root) / f"w{width}"
    directory.mkdir(exist_ok=True)
    manifest = directory / "samples.json"
    started = time.monotonic()
    report = {
        "status": "preparing",
        "width": width,
        "layers": layers,
        "probability": probability,
        "shots": shots,
        "batch_seed": 42 + width * 104729,
        "reference_seed": 71 + width * 130363,
        "sampler": "streaming Stim frames with CSS reference",
        "stim_version": stim.__version__,
        "format": "headerless LSB-first packed measurement bits",
        "samples": [],
        "sampler_source_sha256": {
            path.name: file_hash(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("replay_qsharp.py"),
                Path(__file__).with_name("replay_stim.py"),
                Path(__file__).with_name("study_utils.py"),
                Path(lattice_surgery.__file__),
            )
        },
    }
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous.get("layers", 50) != layers:
            raise ValueError(f"Incompatible existing sampling depth: {manifest}")
        if previous.get("probability", 0.001) != probability:
            raise ValueError(f"Incompatible existing sampling probability: {manifest}")
        if any(
            previous[key] != report[key]
            for key in (
                "width",
                "shots",
                "batch_seed",
                "reference_seed",
                "stim_version",
                "sampler_source_sha256",
            )
        ):
            raise ValueError(f"Incompatible existing sampling manifest: {manifest}")
        if previous["status"] == "complete":
            verify_samples(directory, previous)
            for name, digest in previous["input_sha256"].items():
                if file_hash(directory / name) != digest:
                    raise ValueError(f"Sampling input changed: {directory / name}")
            return previous
    checkpoint(manifest, report)
    with (directory / "sampling.log").open("a", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                configuration_path = directory / "configuration.json"
                if configuration_path.exists():
                    configuration = json.loads(configuration_path.read_text())
                else:
                    configuration = compile_case(
                        directory,
                        distance=7,
                        probability=probability,
                        width=width,
                        layers=layers,
                        basis="Z",
                        seed=42,
                        block_seconds=0.07,
                        export_physical=False,
                        max_file_bytes=MAX_FILE_BYTES,
                    )
                expected_settings = {
                    "distance": 7,
                    "probability": probability,
                    "width": width,
                    "layers": layers,
                    "basis": "Z",
                    "seed": 42,
                    "block_ns": 70_000_000,
                    "physical_circuit_exported": False,
                }
                if any(
                    configuration[key] != value
                    for key, value in expected_settings.items()
                ):
                    raise ValueError(
                        "Prepared workload settings differ from the replay protocol"
                    )
                if (
                    file_hash(directory / "circuit.deq")
                    != configuration["source_sha256"]
                ):
                    raise ValueError("Prepared source does not match its configuration")
                recipe = build_recipe(
                    directory / "circuit.deq",
                    directory / "circuit.deq.jit",
                    probability,
                )
                if recipe.total_measurements != configuration[
                    "measurements"
                ] or recipe.total_measurements != expected_measurements(width, layers):
                    raise ValueError(
                        "Compiled measurement count differs from the preflight"
                    )
                report.update(
                    {
                        "status": "sampling",
                        "measurements": recipe.total_measurements,
                        "qubits": recipe.qubits,
                        "bytes_per_shot": recipe.shot_bytes,
                        "input_sha256": {
                            name: file_hash(directory / name)
                            for name in (
                                "circuit.deq",
                                "circuit.deq.jit",
                                "configuration.json",
                            )
                        },
                        "preparation_seconds": time.monotonic() - started,
                    }
                )
                checkpoint(manifest, report)
                last_checkpoint = time.monotonic()

                def progress(gadgets: int, measurements: int) -> None:
                    nonlocal last_checkpoint
                    if gadgets % 256 or time.monotonic() - last_checkpoint < 10:
                        return
                    report.update(
                        {
                            "physical_gadgets_processed": gadgets,
                            "measurements_processed": measurements,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    )
                    checkpoint(manifest, report)
                    print(
                        f"W={width}: {measurements}/{recipe.total_measurements} measurements",
                        flush=True,
                    )
                    last_checkpoint = time.monotonic()

                for index, data in enumerate(
                    sample_batch(
                        recipe,
                        probability,
                        shots=shots,
                        seed=report["batch_seed"],
                        reference_seed=report["reference_seed"],
                        progress=progress,
                    )
                ):
                    path = directory / f"shot-{index:02}.b8"
                    digest = hashlib.sha256(data).hexdigest()
                    if len(data) > MAX_FILE_BYTES:
                        raise ValueError("Raw shot exceeds the 10 GB guard")
                    if path.exists():
                        if (
                            path.stat().st_size != len(data)
                            or file_hash(path) != digest
                        ):
                            raise ValueError(
                                f"Existing shot does not match its deterministic rerun: {path}"
                            )
                    else:
                        temporary = path.with_suffix(".b8.tmp")
                        temporary.write_bytes(data)
                        temporary.replace(path)
                    report["samples"].append(
                        {
                            "file": path.name,
                            "batch_instance": index,
                            "bytes": len(data),
                            "sha256": digest,
                        }
                    )
                    checkpoint(manifest, report)
                report["status"] = "complete"
                report["elapsed_seconds"] = time.monotonic() - started
                verify_samples(directory, report)
                checkpoint(manifest, report)
                return report
            except BaseException as error:
                report.update(status="failed", error=repr(error))
                checkpoint(manifest, report)
                raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-width", type=int, default=8192)
    parser.add_argument("--min-width", type=int, default=1)
    parser.add_argument("--layers", type=int, default=50)
    parser.add_argument("--probability", type=float, default=0.001)
    parser.add_argument("--shots", type=int, default=10)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--memory-gib", type=float, default=320)
    parser.add_argument("--timeout-seconds", type=float, default=36000)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if not 0 <= arguments.probability < 0.5:
        raise ValueError("probability must be in [0, 0.5)")
    if not 1 <= arguments.max_width <= 8192 or arguments.max_width & (
        arguments.max_width - 1
    ):
        raise ValueError("max-width must be a power of two between 1 and 8192")
    if (
        not 1 <= arguments.min_width <= arguments.max_width
        or arguments.min_width & (arguments.min_width - 1)
        or arguments.layers < 1
    ):
        raise ValueError(
            "min-width must be a power of two through max-width; layers must be positive"
        )
    if arguments.shots < 10 or arguments.workers < 1:
        raise ValueError("At least ten shots and one worker are required")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (arguments.memory_gib, arguments.timeout_seconds)
    ):
        raise ValueError("Resource budgets must be finite and positive")
    if arguments.output.exists() and not arguments.resume:
        raise FileExistsError("Output exists; use --resume to validate and continue it")
    campaign = arguments.output / "sampling.json"
    if campaign.exists():
        previous = json.loads(campaign.read_text())
        if previous.get("probability", 0.001) != arguments.probability:
            raise ValueError("Existing sampling campaign uses a different probability")
    arguments.output.mkdir(parents=True, exist_ok=True)
    required_disk = sum(
        ((expected_measurements(1 << power, arguments.layers) + 7) // 8)
        * arguments.shots
        + (1 << power) * 1_000_000 * max(1, arguments.layers / 50)
        for power in range(
            arguments.min_width.bit_length() - 1, arguments.max_width.bit_length()
        )
    )
    if shutil.disk_usage(arguments.output).free < 2 * required_disk:
        raise OSError(
            "Insufficient local disk headroom for raw shots and compiled metadata"
        )
    with interrupt_on_termination(), exclusive_output(arguments.output / ".active"):
        _run_campaign(arguments)


def _run_campaign(arguments: argparse.Namespace) -> None:
    widths = [
        1 << power
        for power in range(
            arguments.min_width.bit_length() - 1, arguments.max_width.bit_length()
        )
    ][::-1]
    started = time.monotonic()
    report = {
        "status": "running",
        "widths": widths,
        "layers": arguments.layers,
        "probability": arguments.probability,
        "shots_per_width": arguments.shots,
        "memory_gib": arguments.memory_gib,
        "points": {},
        "peak_rss_gib": 0,
        "seed_protocol": "One deterministic batch seed per width; independent SIMD columns identify shots",
    }
    for key in ("TOKIO_WORKER_THREADS", "RAYON_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["QDK_PYTHON_TELEMETRY"] = "none"
    executor = ProcessPoolExecutor(
        max_workers=min(arguments.workers, len(widths)),
        mp_context=multiprocessing.get_context("spawn"),
    )
    try:
        pending = {
            executor.submit(
                prepare_width,
                str(arguments.output.resolve()),
                width,
                arguments.shots,
                arguments.layers,
                arguments.probability,
            ): width
            for width in widths
        }
        while pending:
            completed, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            root = psutil.Process()
            memory = root.memory_info().rss
            for child in root.children(recursive=True):
                try:
                    memory += child.memory_info().rss
                except psutil.NoSuchProcess:
                    continue
            report["peak_rss_gib"] = max(report["peak_rss_gib"], memory / 2**30)
            if memory > arguments.memory_gib * 2**30:
                raise MemoryError("Sampling pool exceeded its aggregate memory budget")
            if time.monotonic() - started > arguments.timeout_seconds:
                raise TimeoutError("Sampling pool exceeded its time budget")
            for future in completed:
                width = pending.pop(future)
                report["points"][str(width)] = future.result()
                print(
                    f"W={width} complete; {len(report['points'])}/{len(widths)} widths",
                    flush=True,
                )
            report["elapsed_seconds"] = time.monotonic() - started
            checkpoint(arguments.output / "sampling.json", report)
        report["status"] = "complete"
        checkpoint(arguments.output / "sampling.json", report)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        try:
            checkpoint(arguments.output / "sampling.json", report)
        finally:
            terminate_processes(psutil.Process().children(recursive=True))
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
