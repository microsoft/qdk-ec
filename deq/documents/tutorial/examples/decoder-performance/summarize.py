"""Condense raw decoder-performance results into the compact data shown in the figure.

The full ``results.json`` keeps every latency sample so that evaluations can be
resumed and audited. The figure only needs each population's distribution and
median, so this module writes a small, human-readable JSON file holding exact
summary statistics and a fixed number of representative quantiles.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
SUMMARY_NAME = "performance-data.json"
QUANTILE_COUNT = 256
SIGNIFICANT_DIGITS = 4
REGIONS = ("intermediate", "final")


def validate_results(document: dict) -> dict:
    if document["schema_version"] != 1 or document["status"] != "complete":
        raise ValueError("summaries require a complete version-1 evaluation")
    series = [item["key"] for item in document["series"]]
    current_series = {"mwpf", "mwpm", "persistent"}
    if len(series) != len(set(series)) or set(series) not in (
        current_series,
        current_series | {"compiled"},
    ):
        raise ValueError(
            "expected the three current or four historical decoder configurations"
        )
    for key in series:
        points = series_points(document, key)
        if not points or [point["width"] for point in points] != [
            2**index for index in range(len(points))
        ]:
            raise ValueError(
                "each series must contain successive powers of two from W=1"
            )
        for index, point in enumerate(points):
            crossed = False
            for region in REGIONS:
                group = point["populations"][region]
                values = group["latency_ms"]
                if point["width"] == 1 and region == "intermediate" and values:
                    raise ValueError("W=1 has no intermediate surgery readouts")
                minimum = (
                    0
                    if point["width"] == 1 and region == "intermediate"
                    else document["minimum_samples"][region]
                )
                if len(values) != group["samples"] or len(values) < minimum:
                    raise ValueError("missing required readout samples")
                if not values:
                    if minimum or group["mean_ms"] is not None:
                        raise ValueError("only W=1 intermediate readouts may be absent")
                    continue
                if any(not math.isfinite(value) or value <= 0 for value in values):
                    raise ValueError(
                        "logarithmic plots require finite positive latencies"
                    )
                average = math.fsum(values) / len(values)
                if not math.isclose(average, group["mean_ms"], rel_tol=1e-12):
                    raise ValueError("saved means disagree with the latency samples")
                crossed |= average > document["settings"]["stop_mean_ms"]
            if crossed != (index == len(points) - 1):
                raise ValueError(
                    "a series must end at its first threshold-crossing point"
                )
        if document["stopped"][key]["width"] != points[-1]["width"]:
            raise ValueError("stopping record disagrees with the saved points")
    if any(point["series"] not in series for point in document["points"]):
        raise ValueError("unknown series in results")
    return document


def load_results(path: Path) -> dict:
    return validate_results(json.loads(path.read_text()))


def series_points(document: dict, key: str) -> list[dict]:
    return sorted(
        (point for point in document["points"] if point["series"] == key),
        key=lambda point: point["width"],
    )


def rounded(value: float) -> float:
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def quantile(ordered: list[float], level: float) -> float:
    position = level * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_population(values: list[float], quantile_count: int) -> dict:
    if not values:
        return {"samples": 0, "mean_ms": None, "quantiles_ms": []}
    ordered = sorted(values)
    count = min(len(ordered), quantile_count)
    return {
        "samples": len(ordered),
        "mean_ms": math.fsum(ordered) / len(ordered),
        "median_ms": quantile(ordered, 0.5),
        "p99_ms": quantile(ordered, 0.99),
        "quantiles_ms": [
            rounded(quantile(ordered, index / (count - 1))) if count > 1 else ordered[0]
            for index in range(count)
        ],
    }


def summarize(document: dict, quantile_count: int = QUANTILE_COUNT) -> dict:
    if quantile_count < 2:
        raise ValueError("at least two quantiles are needed to describe a population")
    validate_results(document)
    return {
        "schema_version": 1,
        "description": (
            "Compact decoder-performance figure data. Each quantiles_ms list holds "
            "n = min(samples, quantile_count) latencies at levels i / (n - 1), "
            "linearly interpolated and rounded to "
            f"{SIGNIFICANT_DIGITS} significant digits. The first and last entries are "
            "the minimum and maximum; populations no larger than quantile_count list "
            "every sorted sample. Means, medians, and p99 use the full samples."
        ),
        "quantile_count": quantile_count,
        "settings": document["settings"],
        "provenance": document["provenance"],
        "latency_definition": document["latency_definition"],
        "series": document["series"],
        "stopped": document["stopped"],
        "points": [
            {
                "series": series["key"],
                "width": point["width"],
                "shots": point["shots"],
                "populations": {
                    region: summarize_population(
                        point["populations"][region]["latency_ms"], quantile_count
                    )
                    for region in REGIONS
                },
            }
            for series in document["series"]
            for point in series_points(document, series["key"])
        ],
    }


def dumps(summary: dict) -> str:
    text = json.dumps(summary, indent=2)
    number = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    return (
        re.sub(
            rf"\[\s*({number}(?:,\s*{number})*)\s*\]",
            lambda match: "[" + re.sub(r",\s+", ", ", match.group(1)) + "]",
            text,
        )
        + "\n"
    )


def write_summary(
    document: dict, output: Path, quantile_count: int = QUANTILE_COUNT
) -> dict:
    summary = summarize(document, quantile_count)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(dumps(summary))
    temporary.replace(output)
    return summary


def summary_path(results: Path) -> Path:
    return results.with_name(SUMMARY_NAME)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results.json")
    parser.add_argument(
        "--output",
        type=Path,
        help=f"defaults to {SUMMARY_NAME} beside the results file",
    )
    parser.add_argument("--quantiles", type=int, default=QUANTILE_COUNT)
    arguments = parser.parse_args()
    write_summary(
        load_results(arguments.results),
        arguments.output or summary_path(arguments.results),
        arguments.quantiles,
    )


if __name__ == "__main__":
    main()
