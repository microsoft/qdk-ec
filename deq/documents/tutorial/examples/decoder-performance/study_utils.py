"""Sample quotas, checkpoints, and guarded subprocess execution."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import time

import psutil


def population(values: list[float]) -> dict:
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("observed latency must be finite and positive")
    return {
        "latency_ms": values,
        "samples": len(values),
        "mean_ms": math.fsum(values) / len(values) if values else None,
    }


def sampling_plan(widths: list[int], layers: int) -> list[dict[str, int]]:
    if layers < 1 or not widths or any(width < 1 for width in widths):
        raise ValueError("the study needs positive layers and widths >= 1")
    if len(set(widths)) != len(widths):
        raise ValueError("study widths must be distinct")
    rows = []
    for width in widths:
        intermediate = 3 * sum((width - layer % 2) // 2 for layer in range(layers))
        shots = max(
            math.ceil(100 / width),
            math.ceil(1001 / intermediate) if intermediate else 0,
        )
        rows.append(
            {
                "width": width,
                "shots": shots,
                "intermediate": shots * intermediate,
                "final": shots * width,
            }
        )
    return rows


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_process(
    command: list[str],
    log: Path,
    *,
    memory_gib: float,
    timeout_seconds: float,
    cwd: Path | None = None,
) -> dict[str, float]:
    """Run one task, monitoring its whole process tree and cleaning up on failure."""
    if any(
        not math.isfinite(value) or value <= 0
        for value in (memory_gib, timeout_seconds)
    ):
        raise ValueError("memory and timeout limits must be positive and finite")
    started = time.monotonic()
    peak = 0
    tracked: set[psutil.Process] = set()
    with log.open("w") as stream, subprocess.Popen(
        command, stdout=stream, stderr=subprocess.STDOUT, cwd=cwd
    ) as process:
        try:
            root = psutil.Process(process.pid)
        except psutil.NoSuchProcess:
            process.wait()
            root = None
        if root is not None:
            tracked.add(root)
        try:
            while process.poll() is None:
                try:
                    if root is not None:
                        tracked.update(root.children(recursive=True))
                except psutil.NoSuchProcess:
                    pass  # The root can exit between poll() and the process-tree query.
                resident = 0
                for child in tuple(tracked):
                    try:
                        resident += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        tracked.discard(child)
                peak = max(peak, resident)
                if resident > memory_gib * 2**30:
                    raise MemoryError(
                        f"process tree exceeded {memory_gib:g} GiB; see {log}"
                    )
                if time.monotonic() - started > timeout_seconds:
                    raise TimeoutError(
                        f"task exceeded {timeout_seconds:g} seconds; see {log}"
                    )
                time.sleep(0.1)
            if process.returncode:
                raise RuntimeError(f"task exited with {process.returncode}; see {log}")
        finally:
            for child in tracked:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(tracked, timeout=5)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(alive, timeout=5)
            process.wait()
    return {"peak_rss_gib": peak / 2**30, "wall_seconds": time.monotonic() - started}
