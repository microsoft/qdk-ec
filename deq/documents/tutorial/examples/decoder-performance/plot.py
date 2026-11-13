"""Plot the compact decoder-performance summary without importing or running DEQ."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = {
    "mwpf": "#8b45b4",
    "mwpm": "#e67e00",
    "compiled": "#0089a8",
    "persistent": "#229849",
}


def load_data(path: Path) -> dict:
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1 or "quantile_count" not in document:
        raise ValueError(
            f"{path} is not a version-1 figure summary; create one with summarize.py"
        )
    keys = [series["key"] for series in document["series"]]
    if len(keys) != len(set(keys)) or not set(keys) <= set(COLORS):
        raise ValueError("unknown or repeated decoder configurations")
    for point in document["points"]:
        if point["series"] not in keys:
            raise ValueError("unknown series in figure data")
        for group in point["populations"].values():
            values = group["quantiles_ms"]
            if len(values) != min(group["samples"], document["quantile_count"]):
                raise ValueError("quantile count disagrees with the sample count")
            if any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError("logarithmic plots require finite positive latencies")
    return document


def plot(data: Path, output: Path) -> None:
    document = load_data(data)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    widths = sorted({point["width"] for point in document["points"]})
    extremes = []
    for axis, region, label in zip(
        axes,
        ("intermediate", "final"),
        ("Intermediate readout latency (ms)", "Final readout latency (ms)"),
    ):
        for series_index, series in enumerate(document["series"]):
            color = COLORS[series["key"]]
            positions, medians = [], []
            points = sorted(
                (
                    point
                    for point in document["points"]
                    if point["series"] == series["key"]
                ),
                key=lambda point: point["width"],
            )
            for point in points:
                group = point["populations"][region]
                if not group["samples"]:
                    continue
                values = np.asarray(group["quantiles_ms"])
                extremes.extend((values[0], values[-1]))
                position = (
                    math.log2(point["width"])
                    + (series_index - (len(document["series"]) - 1) / 2) * 0.16
                )
                logarithms = np.log10(values)
                if len(set(logarithms)) > 1:
                    violin = axis.violinplot(
                        [logarithms],
                        positions=[position],
                        widths=0.14,
                        showextrema=False,
                        points=100,
                    )
                    for body in violin["bodies"]:
                        for path in body.get_paths():
                            path.vertices[:, 0] = np.exp2(path.vertices[:, 0])
                            path.vertices[:, 1] = np.power(10, path.vertices[:, 1])
                        body.set_facecolor(color)
                        body.set_edgecolor(color)
                        body.set_alpha(0.22)
                positions.append(2**position)
                medians.append(group["median_ms"])
            axis.plot(
                positions,
                medians,
                "o-",
                color=color,
                markersize=3,
                markerfacecolor="white",
                linewidth=1,
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_xticks(widths, widths)
        axis.set_xlim(2**-0.45, max(widths) * 2**0.45)
        axis.set_xlabel("Circuit width W")
        axis.set_ylabel(label)
        axis.grid(axis="y", which="major", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(
        10 ** math.floor(math.log10(min(extremes))),
        10 ** math.ceil(math.log10(max(extremes))),
    )
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                color=COLORS[series["key"]],
                marker="o",
                markerfacecolor="white",
                markersize=4,
                label=series["label"],
            )
            for series in document["series"]
        ],
        loc="upper center",
        ncol=len(document["series"]),
        frameon=False,
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "performance-data.json")
    parser.add_argument("--output", type=Path, default=HERE / "performance.png")
    arguments = parser.parse_args()
    plot(arguments.data, arguments.output)


if __name__ == "__main__":
    main()
