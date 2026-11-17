"""Format Criterion's JIT-only results and enforce the per-epoch compiler budget."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import zlib

ROOT = Path(__file__).resolve().parents[5]
FIXTURE_DIRECTORY = Path(__file__).resolve().parent / "fixtures"
CACHE_MODES = ("cache_enabled", "cache_disabled")


def check_estimate(
    estimate: dict, *, epochs: int, maximum_mean_epoch_ms: float
) -> dict:
    if (
        epochs <= 0
        or not math.isfinite(maximum_mean_epoch_ms)
        or maximum_mean_epoch_ms <= 0
    ):
        raise ValueError("epochs and compiler budget must be positive")
    interval = estimate["confidence_interval"]
    confidence = float(interval["confidence_level"])
    lower = float(interval["lower_bound"])
    upper = float(interval["upper_bound"])
    mean = float(estimate["point_estimate"])
    if not all(math.isfinite(value) for value in (confidence, lower, mean, upper)):
        raise ValueError("timing estimates must be finite")
    if not 0.95 <= confidence < 1 or not 0 < lower <= mean <= upper:
        raise ValueError("require a valid mean interval with at least 95% confidence")
    upper_epoch_ms = upper / epochs / 1e6
    return {
        "mean_seconds": mean / 1e9,
        "mean_epoch_ms": mean / epochs / 1e6,
        "upper_mean_epoch_ms": upper_epoch_ms,
        "maximum_mean_epoch_ms": maximum_mean_epoch_ms,
        "confidence_level": confidence,
        "budget_passed": upper_epoch_ms <= maximum_mean_epoch_ms,
    }


def cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def collect(criterion: Path) -> dict:
    provenance_path = criterion / "jit_compiler_run.json"
    provenance = json.loads(provenance_path.read_text())
    manifest_bytes = (FIXTURE_DIRECTORY / "manifest.json").read_bytes()
    if zlib.crc32(manifest_bytes) != provenance["fixture_manifest_crc32"]:
        raise ValueError("fixture manifest changed since the benchmark run")
    manifest = json.loads(manifest_bytes)
    if provenance["schema_version"] != 4 or not provenance["optimized"]:
        raise ValueError(
            "run the current optimized, ratio-based Rust benchmark before formatting a baseline"
        )
    if provenance["compiler_threads"] != manifest["compiler_threads"]:
        raise ValueError("compiler worker count differs from the fixture contract")
    if provenance["maximum_mean_epoch_ms"] != manifest["maximum_mean_epoch_ms"]:
        raise ValueError("compiled performance budget differs from the fixture")
    capacity = provenance["error_template_cache_capacity"]
    ratio = provenance["cache_limit_ratio"]
    gadget_types = provenance["registered_gadget_types"]
    if any(type(value) is not int or value <= 0 for value in (capacity, ratio, gadget_types)):
        raise ValueError("cache-enabled ratio, type count, and capacity must be positive integers")
    if capacity != ratio * gadget_types:
        raise ValueError("cache capacity differs from the registered-type ratio")
    sources = {}
    for relative, expected_crc in provenance["compiled_source_crc32"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(
                "benchmark source provenance must stay inside the repository"
            )
        content = path.read_bytes()
        if zlib.crc32(content) != expected_crc:
            raise ValueError(f"source changed since benchmark compilation: {relative}")
        sources[relative] = hashlib.sha256(content).hexdigest()
    expected_runs = {
        f"{mode}/{case['name']}"
        for case in manifest["cases"]
        for mode in CACHE_MODES
    }
    completed = set(provenance["executed"])
    timed_out = set(provenance["timed_out"])
    if completed & timed_out or completed | timed_out != expected_runs:
        raise ValueError("run all JIT benchmark cases before recording a baseline")
    if any(not name.startswith("cache_disabled/") for name in timed_out):
        raise ValueError("cache-enabled cases must complete before recording a baseline")
    timeout_seconds = float(provenance["timeout_seconds"])
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout must be positive and finite")
    rows = []
    for case in manifest["cases"]:
        payload = gzip.decompress((FIXTURE_DIRECTORY / case["file"]).read_bytes())
        if hashlib.sha256(payload).hexdigest() != case["protobuf_sha256"]:
            raise ValueError("fixture no longer matches the recorded workload")
        for mode in CACHE_MODES:
            row = {
                "case": case["name"],
                "mode": mode,
                "width": case["width"],
                "layers": case["layers"],
                "epochs": case["epochs"],
                "gadgets": case["gadgets"],
                "checks": case["checks"],
                "errors": case["errors"],
            }
            if f"{mode}/{case['name']}" in timed_out:
                rows.append(
                    {
                        **row,
                        "status": "timed_out",
                        "timeout_seconds": timeout_seconds,
                    }
                )
                continue
            directory = criterion / "jit_compiler" / mode / case["name"] / "new"
            for filename in ("estimates.json", "sample.json"):
                path = directory / filename
                if path.stat().st_mtime < provenance["started_unix_seconds"]:
                    raise ValueError(
                        f"stale Criterion results from an earlier run: {path}"
                    )
            estimates = json.loads((directory / "estimates.json").read_text())
            samples = json.loads((directory / "sample.json").read_text())
            iterations = samples["iters"]
            times = samples["times"]
            if len(iterations) != len(times) or len(times) < 10:
                raise ValueError("at least ten complete timing samples are required")
            if any(
                not math.isfinite(value) or value <= 0
                for value in [*iterations, *times]
            ):
                raise ValueError(
                    "Criterion samples must have positive finite durations and iterations"
                )
            observed_mean = statistics.mean(
                elapsed / count for elapsed, count in zip(times, iterations)
            )
            if not math.isclose(
                observed_mean, estimates["mean"]["point_estimate"], rel_tol=1e-7
            ):
                raise ValueError("raw samples and Criterion's mean estimate disagree")
            checked = check_estimate(
                estimates["mean"],
                epochs=case["epochs"],
                maximum_mean_epoch_ms=manifest["maximum_mean_epoch_ms"],
            )
            rows.append(
                {
                    **row,
                    "status": "complete",
                    "samples": samples,
                    "estimates": estimates,
                    **checked,
                }
            )
    return {
        "schema_version": 4,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(
            ["git", "--no-pager", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty_worktree": bool(
            subprocess.check_output(
                ["git", "--no-pager", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        "source_sha256": sources,
        "rustc": subprocess.check_output(["rustc", "-vV"], text=True).strip(),
        "platform": platform.platform(),
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "available_cpus": (
            len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        ),
        "provenance": provenance,
        "fixture": manifest,
        "measurements": rows,
        "budget_passed": all(
            row["budget_passed"] for row in rows if row["mode"] == "cache_enabled"
        ),
    }


def markdown(report: dict) -> str:
    lines = [
        "# JIT compiler benchmark baseline",
        "",
        f"Recorded: {report['recorded_utc']}",
        f"Base revision: `{report['git_head']}`; dirty working tree: `{report['dirty_worktree']}`.",
        "Raw samples and run provenance remain in the local Criterion output; only this summary belongs in git.",
        "",
        f"- CPU: {report['cpu_model'] or 'not available'}; available logical CPUs: {report['available_cpus']}.",
        f"- Rust: `{report['rustc'].splitlines()[0]}`.",
        f"- Flags: `{report['provenance']['rustflags']}`; compiler worker threads: {report['provenance']['compiler_threads']}.",
        f"- Cache ratio: {report['provenance']['cache_limit_ratio']}; registered JIT gadget types: {report['provenance']['registered_gadget_types']}.",
        f"- Error-template LRU capacity: {report['provenance']['error_template_cache_capacity']} entries (enabled), 0 (disabled).",
        "- Profile: optimized Cargo bench; workspace release LTO and codegen settings unchanged.",
        "",
        "## Scope",
        "",
        report["fixture"]["scope"],
        "",
        "**Excluded:** " + report["fixture"]["excluded"],
        "",
        "Every iteration starts with a fresh compiler. Cache-enabled runs start with an empty",
        "error-template cache and reuse templates within that program; cache-disabled runs rebuild",
        "those templates instead. Input-check-map and terminal-error-model caches remain enabled in",
        "both modes. There is no retained warmup pass. Every completed iteration checks gadget,",
        "check, and error-model counts and requires the final JIT live-state count to be zero.",
        "Completed rows are means from at least ten Criterion samples. A timeout row is only a",
        "lower bound from an interrupted run, not a measured mean or confidence interval.",
        "",
        "## Results",
        "",
        "| Workload | Error-template cache | Full-program mean / bound (s) | Mean / epoch or bound (ms) | Upper confidence bound / epoch (ms) | Budget |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in report["measurements"]:
        cache = row["mode"].removeprefix("cache_")
        if row["status"] == "timed_out":
            seconds = row["timeout_seconds"]
            lines.append(
                f"| {row['case']} | {cache} | >{seconds:.3f} (timeout) | "
                f">{seconds * 1000 / row['epochs']:.3f} | - | Control; timed out |"
            )
            continue
        budget = "Control; not gated"
        if row["mode"] == "cache_enabled":
            budget = "PASS" if row["budget_passed"] else "FAIL"
        lines.append(
            f"| {row['case']} | {cache} | "
            f"{row['mean_seconds']:.6f} | {row['mean_epoch_ms']:.3f} | "
            f"{row['upper_mean_epoch_ms']:.3f} | {budget} |"
        )
    lines.extend(
        [
            "",
            "## Performance requirement",
            "",
            "The upper confidence bound (at least 95%) for mean full-program compilation time,",
            "divided by its number of d-round epochs, must not exceed **390 ms**. In particular,",
            "The whole-program budget is the fixture's recorded epoch count multiplied by 390 ms.",
            "The budget applies to cache-enabled operation; the disabled cache is an experimental control.",
            "This is a JIT throughput requirement, not a",
            "claim about per-epoch worst-case latency or an end-to-end decoder guarantee.",
            "",
            "Measurements are host-dependent. Run without competing decoder studies or builds.",
            "Criterion's own comparisons detect smaller changes; this absolute gate enforces the workload budget.",
            "See [reproduction instructions](../README.md).",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "jit_compiler_baseline.md").write_text(markdown(report))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--criterion", type=Path, default=ROOT / "target/criterion")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "target/jit-benchmark-report"
    )
    arguments = parser.parse_args()
    report = collect(arguments.criterion)
    write_report(report, arguments.output)
    print(markdown(report))
    if not report["budget_passed"]:
        raise SystemExit(
            "JIT compiler performance budget exceeded; report retained for inspection."
        )


if __name__ == "__main__":
    main()
