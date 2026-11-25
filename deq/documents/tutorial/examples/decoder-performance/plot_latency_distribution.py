"""Plot validated intermediate-readout tails and latency versus circuit layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from precompute_replay import checkpoint, file_hash
from readout_latency import ReadoutSample, readout_samples

PERCENTILES = (50, 95, 99, 99.9)
READOUT_NAMES = ("MZZ", "MXX", "MeasureZ")


def population(values: np.ndarray) -> dict:
    if not len(values):
        return {
            "samples": 0,
            "mean_ms": None,
            "maximum_ms": None,
            "percentiles_ms": {},
            "ccdf": [],
        }
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Latencies must be finite and positive")
    ordered = np.sort(values)
    positions = np.unique(
        np.concatenate(
            (
                np.linspace(0, len(values) - 1, min(len(values), 2000)),
                len(values) - np.geomspace(1, len(values), min(len(values), 2000)),
            )
        ).astype(np.int64)
    )
    knots = np.unique(ordered[positions])
    survival = (len(values) - np.searchsorted(ordered, knots, side="left")) / len(
        values
    )
    return {
        "samples": len(values),
        "mean_ms": float(np.mean(values)),
        "maximum_ms": float(ordered[-1]),
        "percentiles_ms": {
            str(percentile): float(value)
            for percentile, value in zip(
                PERCENTILES, np.percentile(ordered, PERCENTILES)
            )
        },
        "ccdf": [
            [float(value), float(probability)]
            for value, probability in zip(knots, survival)
        ],
    }


def summarize(
    samples: list[ReadoutSample], configuration: dict, measured_shots: int
) -> dict:
    width, layers = configuration["width"], configuration["layers"]
    epochs = configuration["epochs"]
    names = configuration["gadget_names"]
    intermediate = [sample for sample in samples if sample.region == "intermediate"]
    values = np.fromiter(
        (sample.latency_ms for sample in intermediate), dtype=np.float64
    )
    layer_indices = np.fromiter(
        ((epochs[str(sample.gid)] - 2) // 3 for sample in intermediate), dtype=np.int64
    )
    if np.any(layer_indices < 0) or np.any(layer_indices >= layers):
        raise ValueError("Intermediate readout has an invalid brickwall layer")
    observed = np.bincount(layer_indices, minlength=layers)
    expected = np.array(
        [3 * ((width - layer % 2) // 2) * measured_shots for layer in range(layers)]
    )
    if not np.array_equal(observed, expected):
        raise ValueError(
            "Intermediate readout coverage differs from the layer schedule"
        )
    kinds = np.fromiter(
        (READOUT_NAMES.index(names[str(sample.gid)]) for sample in intermediate),
        dtype=np.uint8,
    )
    ordered_by_layer = values[np.argsort(layer_indices, kind="stable")]
    boundaries = np.concatenate(([0], np.cumsum(observed)))
    by_layer = []
    for layer in range(layers):
        group = ordered_by_layer[boundaries[layer] : boundaries[layer + 1]]
        by_layer.append(
            {
                "layer": layer + 1,
                "samples": len(group),
                "percentiles_ms": (
                    {
                        str(percentile): float(value)
                        for percentile, value in zip(
                            PERCENTILES, np.percentile(group, PERCENTILES)
                        )
                    }
                    if len(group)
                    else {}
                ),
                "maximum_ms": float(np.max(group)) if len(group) else None,
            }
        )
    report = {
        "intermediate": population(values),
        "by_readout": {
            name: population(values[kinds == index])
            for index, name in enumerate(READOUT_NAMES)
        },
        "by_layer": by_layer,
        "final": population(
            np.fromiter(
                (sample.latency_ms for sample in samples if sample.region == "final"),
                dtype=np.float64,
            )
        ),
        "interpretation": "Readout-weighted observations, not independent trials; plotting knots are sampled, statistics use every observation.",
    }
    if all(sample.own_gadget_latency_ms is not None for sample in intermediate):
        report["own_gadget_latency"] = population(
            np.asarray(
                [sample.own_gadget_latency_ms for sample in intermediate], dtype=float
            )
        )
    return report


def render(report: dict, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    for name, group in [
        ("All intermediate", report["intermediate"]),
        *report["by_readout"].items(),
    ]:
        if not group["samples"]:
            continue
        knots = np.asarray(group["ccdf"])
        axes[0].plot(knots[:, 0], knots[:, 1], label=name)
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Readout latency (ms)",
        ylabel="Fraction with latency >= x",
    )
    axes[0].legend()
    for percentile in PERCENTILES:
        axes[1].plot(
            [row["layer"] for row in report["by_layer"]],
            [row["percentiles_ms"].get(str(percentile)) for row in report["by_layer"]],
            label=f"p{percentile:g}",
        )
    axes[1].set(
        yscale="log",
        xlabel="Brickwall layer",
        ylabel="Intermediate readout latency (ms)",
    )
    axes[1].legend()
    for axis in axes:
        axis.grid(True, which="both", alpha=0.2)
    axes[0].axvline(1000, color="gray", linestyle=":", linewidth=1)
    axes[1].axhline(1000, color="gray", linestyle=":", linewidth=1)
    probability = report.get(
        "probability", report.get("execution_settings", {}).get("probability")
    )
    probability_label = f", p={probability:g}" if probability is not None else ""
    figure.suptitle(
        f"W={report['width']}, depth={report['layers']}{probability_label}: static/persistent MWPM\n"
        f"{report['measured_shots']} measured shots; {report['warmup_shots']} warm-ups excluded"
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def generate_report(run: Path, output: Path) -> None:
    point = json.loads((run / "summary.json").read_text())
    if point["series"] != "persistent":
        raise ValueError("This distribution report requires static/persistent MWPM")
    point.pop("populations")
    manifest = Path(point["input_provenance"]["manifest"])
    if file_hash(manifest) != point["input_provenance"]["sha256"]:
        raise ValueError("Sampling manifest changed after evaluation")
    configuration_path = manifest.parent / "configuration.json"
    sampling = json.loads(manifest.read_text())
    if file_hash(configuration_path) != sampling["input_sha256"]["configuration.json"]:
        raise ValueError("Circuit configuration changed after sampling")
    configuration = json.loads(configuration_path.read_text())
    if configuration["blocks_per_layer"] != 3:
        raise ValueError(
            "Distribution report requires the three-block brickwall schedule"
        )
    samples, coverage = readout_samples(
        {
            "run_directory": str(run),
            "controller": "static",
            "shots": point["native_shots"],
            "warmup_shots": point["warmup_shots"],
            "require_lossless_trace": True,
        }
    )
    if coverage != point["coverage"]:
        raise ValueError("Readout coverage changed after evaluation")
    report = summarize(samples, configuration, point["shots"])
    report.update(
        width=configuration["width"],
        layers=configuration["layers"],
        probability=configuration["probability"],
        measured_shots=point["shots"],
        warmup_shots=point["warmup_shots"],
        coverage=coverage,
        runtime_sha256=point["runtime_sha256"],
        execution_settings=point["execution_settings"],
        trace_sha256=file_hash(run / "timing.pb"),
        input_provenance=point["input_provenance"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint(output.with_suffix(".json"), report)
    render(report, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    generate_report(arguments.run, arguments.output)


if __name__ == "__main__":
    main()
