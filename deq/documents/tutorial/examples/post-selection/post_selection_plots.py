"""Shared plotting for code-capacity and circuit-level post-selection studies."""

import argparse
from dataclasses import asdict
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from simulation import digest, load_trace, write_json

from post_selection_stats import (
    ScoreGroup,
    binomial_interval,
    rejection_curve,
)


CASE_STYLES = {
    "monolithic": ("Monolithic", "#C0C0C0"),
    "capacity-pauli": ("X errors only", "#0072B2"),
    "capacity-mixed": ("X errors + loss", "#009E73"),
    "capacity-pauli-circuit-gap": ("X errors only", "#0072B2"),
    "capacity-mixed-circuit-gap": ("X errors + loss", "#009E73"),
    "window-r0": ("Radius 0", "#0072B2"),
    "window-r1": ("Radius 1", "#D55E00"),
    "window-r2": ("Radius 2", "#009E73"),
    "window-r3": ("Radius 3", "#B45F9B"),
    "window-r4": ("Radius 4", "#56B4E9"),
    "window-r5": ("Radius 5", "#CC9900"),
    "window-r6": ("Radius 6", "#8C6D31"),
}
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#7C858B",
    "axes.labelcolor": "#263238",
    "text.color": "#263238",
    "xtick.color": "#4A5358",
    "ytick.color": "#4A5358",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

DIRECTORY = Path(__file__).resolve().parent
FIGURES = DIRECTORY / "figures"
SELECTION_STATISTICS = {
    "correction_count": "correction-count thresholds",
    "correction_weight": "correction-weight thresholds",
    "syndrome_count": "fired-check-count thresholds",
}
CORRECTION_COUNT_MARKERS = {2: "*", 3: "^"}
CORRECTION_COUNT_REJECTION_LIMIT = 5


def selection_point(
    shots: int, retained: int, errors: int, limit: int | float, *, failed_shots: int = 0
) -> dict:
    if (
        not 0 <= errors <= retained <= shots - failed_shots
        or shots <= 0
        or failed_shots < 0
    ):
        raise ValueError("invalid attempted, retained, or logical-error count")
    return {
        "limit": limit,
        "shots": shots,
        "retained_shots": retained,
        "failed_shots": failed_shots,
        "rejected_shots": shots - retained - failed_shots,
        "logical_errors": errors,
        "rejected_percent": 100 * (shots - retained) / shots,
        "ler": errors / retained if retained else None,
        "confidence_interval_95": list(binomial_interval(errors, retained)),
    }


def precision_summary(groups: list[dict], weight: dict | None) -> dict:
    curve = rejection_curve([ScoreGroup(**group) for group in groups]) if groups else []
    point = (
        min(curve, key=lambda point: abs(point.rejected_percent - 10))
        if curve
        else None
    )
    counts = {
        "gap_at_10_percent": point.expected_errors if point else 0,
        "weight_threshold": weight["logical_errors"] if weight else 0,
    }
    return {
        "target_errors": 400,
        "selected_errors": counts,
        "approximate_relative_standard_error": {
            method: 1 / math.sqrt(errors) if errors else None
            for method, errors in counts.items()
        },
        "meets_error_count_target": all(errors >= 400 for errors in counts.values()),
        "stopping_rule": "fixed shot budget; this precision target does not stop sampling",
    }


def threshold_points(
    groups: list[ScoreGroup], *, attempted_shots: int | None = None,
    limits: tuple[int, ...] | None = None,
) -> list[dict]:
    decoded = sum(group.shots for group in groups)
    total = decoded if attempted_shots is None else attempted_shots
    if limits is not None:
        if not total:
            return []
        if any(not math.isfinite(group.score) for group in groups):
            raise ValueError("selection statistics must be finite")
        return [
            selection_point(
                total,
                sum(group.shots for group in groups if group.score <= limit),
                sum(group.logical_errors for group in groups if group.score <= limit),
                limit,
                failed_shots=total - decoded,
            )
            for limit in limits
        ]
    retained = errors = 0
    points = []
    for group in sorted(groups, key=lambda group: group.score):
        if not math.isfinite(group.score):
            raise ValueError("selection statistics must be finite")
        retained += group.shots
        errors += group.logical_errors
        points.append(
            selection_point(
                total, retained, errors, group.score, failed_shots=total - decoded
            )
        )
    return points


def weight_points(
    groups: list[ScoreGroup], *, attempted_shots: int | None = None
) -> list[dict]:
    if any(group.score < 0 or int(group.score) != group.score for group in groups):
        raise ValueError("correction counts must be nonnegative integers")
    return threshold_points(groups, attempted_shots=attempted_shots)


def figure_metadata(summary: dict) -> dict:
    private = {"worker_environment", "scheduler_address", "source", "source_sha256",
               "figure_output", "final_output", "output_dir", "batch_records", "failure_records",
               "data_file", "failure_index", "stop_reason", "input_files"}

    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items() if key not in private}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return clean(summary)


def gap_threshold_markers(axes, curve: list[dict], color: str, rejection_limit: float, *, zorder: float = 2) -> list[float]:
    thresholds = [
        point for point in curve
        if point.get("exact_threshold") and point["rejected_percent"] <= rejection_limit
    ]
    if not thresholds:
        return []
    rates = np.array([point["rate"] for point in thresholds])
    intervals = np.array([
        binomial_interval(round(point["expected_errors"]), point["retained_shots"])
        for point in thresholds
    ]).T
    upper_limits = np.array([point["expected_errors"] == 0 for point in thresholds])
    values = np.where(upper_limits, intervals[1], rates)
    errors = np.maximum(0, np.vstack((rates - intervals[0], intervals[1] - rates)))
    errors[:, upper_limits] = 0.15 * values[upper_limits]
    axes.errorbar(
        [point["rejected_percent"] for point in thresholds], values,
        yerr=errors, uplims=upper_limits, marker="o", linestyle="none",
        color=color, markerfacecolor="white", markersize=2.5,
        markeredgewidth=0.6, elinewidth=0.6, capsize=1, zorder=zorder,
    )
    return intervals[1].tolist()


def draw(
    summary: dict,
    output: Path,
    *,
    title: str,
    subtitle: str,
    series: list,
    rejection_limit: float = 20,
    exact: bool = False,
    threshold_label: str = "correction-count thresholds",
    conditioned_series: list | None = None,
    threshold_markers: dict[int, str] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with plt.rc_context(STYLE):
        if conditioned_series is None:
            figure, axes = plt.subplots(figsize=(11, 7.2))
            conditional_axes = None
        else:
            figure, (axes, conditional_axes) = plt.subplots(1, 2, figsize=(16, 7.8))
        figure.subplots_adjust(left=0.105, right=0.98, bottom=0.23 if threshold_markers else 0.16, top=0.68 if len(series) > 4 else 0.75)
        figure.suptitle(title, y=0.97, fontsize=15)
        multiline_subtitle = "\n" in subtitle
        figure.text(
            0.5,
            0.916,
            subtitle,
            ha="center",
            va="top" if multiline_subtitle else "baseline",
            fontsize=10,
        )
        handles = []
        visible_rates = []
        for entry in series:
            color = entry["color"]
            linestyle = entry.get("linestyle", "-")
            zorder = entry.get("zorder", 2)
            handles.append(Line2D([], [], color=color, linestyle=linestyle, label=entry["label"]))
            curve = entry["gap"]
            axes.plot(
                [point["rejected_percent"] for point in curve],
                [point["rate"] or np.nan for point in curve],
                color=color,
                linewidth=1.7,
                linestyle=linestyle,
                zorder=zorder,
            )
            axes.plot(
                [point["rejected_percent"] for point in curve],
                [point.get("upper_limit") or np.nan for point in curve],
                color=color,
                linewidth=1,
                linestyle=":",
                zorder=zorder,
            )
            visible_rates.extend(
                point["rate"] or point.get("upper_limit")
                for point in curve
                if point["rejected_percent"] <= rejection_limit
            )
            for point in entry["weight"]:
                rate = point["ler"]
                if rate is None:
                    continue
                interval = point.get("confidence_interval_95")
                plotted = rate if rate or interval is None else interval[1]
                marker = threshold_markers[point["limit"]] if threshold_markers else ("s" if rate or exact else "v")
                upper_bound = bool(threshold_markers and not rate and not exact and interval)
                axes.errorbar(
                    point["rejected_percent"],
                    plotted,
                    yerr=(
                        [[max(0, rate - interval[0])], [max(0, interval[1] - rate)]]
                        if rate and interval
                        else (0.15 * plotted if upper_bound else None)
                    ),
                    uplims=upper_bound,
                    marker=marker,
                    linestyle="none",
                    markerfacecolor="white",
                    color=color,
                    markersize=10 if marker == "*" else 7,
                    capsize=3,
                    zorder=zorder,
                )
                if point["rejected_percent"] <= rejection_limit:
                    visible_rates.append(interval[1] if interval else plotted)
            visible_rates.extend(gap_threshold_markers(axes, curve, color, rejection_limit, zorder=zorder))
        positive = [rate for rate in visible_rates if rate and math.isfinite(rate)]
        if conditional_axes is not None:
            axes.set_title("All attempts (failures unavailable)", fontsize=11)
            hidden = [entry for entry in series if entry["gap"] and min(point["rejected_percent"] for point in entry["gap"]) > rejection_limit]
            for index, entry in enumerate(hidden):
                unavailable = min(point["rejected_percent"] for point in entry["gap"])
                axes.text(0.03, 0.97 - index * 0.07,
                          f"{entry['label'].splitlines()[0]} starts at {unavailable:.1f}% unavailable",
                          transform=axes.transAxes, va="top", fontsize=9, color=entry["color"])
            conditional_rates = []
            for entry in conditioned_series:
                curve = entry["gap"]
                zorder = entry.get("zorder", 2)
                conditional_axes.plot([point["rejected_percent"] for point in curve],
                                      [point["rate"] or np.nan for point in curve], color=entry["color"], linewidth=1.7,
                                      linestyle=entry.get("linestyle", "-"), zorder=zorder)
                conditional_axes.plot([point["rejected_percent"] for point in curve],
                                      [point.get("upper_limit") or np.nan for point in curve],
                                      color=entry["color"], linewidth=1, linestyle=":", zorder=zorder)
                conditional_rates.extend(point["rate"] or point.get("upper_limit") for point in curve if point["rejected_percent"] <= rejection_limit)
                for point in entry["weight"]:
                    if point["ler"] is None:
                        continue
                    lower, upper = point["confidence_interval_95"]
                    rate = point["ler"]
                    marker = threshold_markers[point["limit"]] if threshold_markers else ("s" if rate else "v")
                    upper_bound = bool(threshold_markers and not rate)
                    conditional_axes.errorbar(point["rejected_percent"], rate or upper,
                                              yerr=[[max(0, rate - lower)], [max(0, upper - rate)]] if rate else (0.15 * upper if upper_bound else None),
                                              uplims=upper_bound, marker=marker, color=entry["color"], zorder=zorder,
                                              markerfacecolor="white", markersize=10 if marker == "*" else 7, capsize=2, linestyle="none")
                    if point["rejected_percent"] <= rejection_limit:
                        conditional_rates.extend((rate or upper, upper))
                conditional_rates.extend(gap_threshold_markers(conditional_axes, curve, entry["color"], rejection_limit, zorder=zorder))
            conditional_positive = [rate for rate in conditional_rates if rate and math.isfinite(rate)]
            conditional_axes.set(xlim=(0, rejection_limit), yscale="log",
                                 xlabel="Rejected decoded shots (%)", ylabel="Logical error rate per retained decoded shot",
                                 title="Conditional on successful decoding")
            if conditional_positive:
                conditional_axes.set_ylim(10 ** math.floor(math.log10(min(conditional_positive))) / 2, min(1, 2 * max(conditional_positive)))
            conditional_axes.grid(axis="y", alpha=0.25)
        axes.set(
            xlim=(0, rejection_limit),
            ylim=(
                (
                    10 ** math.floor(math.log10(min(positive))) / 2,
                    min(1, 2 * max(positive)),
                )
                if positive
                else (1e-7, 1)
            ),
            yscale="log",
            xlabel="Unavailable or rejected attempts (%)" if conditioned_series is not None else "Rejected shots (%)",
            ylabel="Logical error rate per retained shot",
        )
        axes.grid(axis="y", alpha=0.25)
        figure.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.86 if multiline_subtitle else 0.88),
            ncol=4 if len(handles) > 6 else 3 if len(handles) > 4 else 2 if len(handles) == 4 else max(1, len(handles)),
            frameon=False,
            fontsize=8,
        )
        if threshold_markers:
            figure.legend(
                handles=[
                    Line2D([], [], color="black", marker=marker, linestyle="none",
                           markerfacecolor="white", markersize=10 if marker == "*" else 7,
                           label=f"Count <= {limit}")
                    for limit, marker in threshold_markers.items()
                ],
                title="Retain shots with maximum corrections per gadget",
                loc="upper center", bbox_to_anchor=(0.5, 0.165), ncol=3,
                frameon=False, fontsize=9, title_fontsize=9,
            )
        figure.text(
            0.5,
            0.035,
                "Lines: forced gap with uniform random ties. Circles: all exact gap thresholds in the displayed range.\n"
                + ("Stars / upward triangles: correction counts <= 2 / <= 3, without interpolation.\n"
                    if threshold_markers else f"Squares: {threshold_label}, without interpolation.\n")
            + (
                "Exact enumeration; no sampling uncertainty."
                if exact
                    else ("Intervals: pointwise 95%. Downward limit arrows and dotted tails: zero-error upper bounds, not zero LER."
                        if threshold_markers else "Intervals: pointwise 95%. Triangles and dotted tails: zero-error upper bounds, not zero LER.")
            ) + ("\nRight panel excludes failed attempts; it does not measure unconditional performance." if conditioned_series is not None else ""),
            ha="center",
            fontsize=8,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        for destination in (output, output.with_suffix(".png")):
            temporary = destination.with_name(
                f".{destination.stem}.{os.getpid()}{destination.suffix}"
            )
            try:
                figure.savefig(temporary, dpi=160)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        write_json(figure_metadata(summary), output.with_suffix(".json"))
        plt.close(figure)


def render_native(
    summary: dict, output: Path, selection_statistic: str = "correction_count"
) -> None:
    surface = summary.get("code") == "surface-code"
    capacity = summary.get("noise_model") == "capacity"
    paired_gaps = capacity and "case_gap_configurations" in summary
    name = "Distance-5 surface code" if surface else "Fire & Ice"
    upper_bounds_only = all(
        not group["logical_errors"]
        for case in summary["cases"]
        for group in case["groups"]["gap"]
    )
    series = []
    conditioned_series = []
    has_failures = False
    count_limits = tuple(CORRECTION_COUNT_MARKERS) if selection_statistic == "correction_count" else None
    rejection_limit = 2 if capacity else CORRECTION_COUNT_REJECTION_LIMIT if count_limits else 20
    for case in summary["cases"]:
        curve = (
            [
                asdict(point)
                for point in rejection_curve(
                    [ScoreGroup(**group) for group in case["groups"]["gap"]]
                )
            ]
            if case["groups"]["gap"]
            else []
        )
        decoded = sum(group["shots"] for group in case["groups"]["gap"])
        failed = case["shots"] - decoded
        has_failures |= failed > 0
        conditional_curve = [point.copy() for point in curve]
        for point in curve:
            point["rejected_percent"] = 100 * (
                1 - point["retained_shots"] / case["shots"]
            )
        logical_errors = sum(group["logical_errors"] for group in case["groups"]["gap"])
        circuit_gap = case["name"].endswith("-circuit-gap")
        gap_label = ("; gap-config2" if circuit_gap else "; gap-config1") if paired_gaps else ""
        series.append(
            {
                "label": (
                    f"{CASE_STYLES[case['name']][0]}{gap_label}\n{case['shots']:,} shared shots; "
                    f"{logical_errors:,} {'error' if logical_errors == 1 else 'errors'}"
                    + (f"\n{failed:,} unavailable ({100 * failed / case['shots']:.1f}%)" if failed else "")
                ),
                "color": CASE_STYLES[case["name"]][1],
                "linestyle": "--" if circuit_gap else "-",
                "zorder": 1 if case["name"] == "monolithic" else 2,
                "gap": curve,
                "weight": threshold_points(
                    [
                        ScoreGroup(**group)
                        for group in case["groups"][selection_statistic]
                    ],
                    attempted_shots=case["shots"],
                    limits=count_limits,
                ) if not circuit_gap else [],
            }
        )
        conditioned_series.append({
            "color": CASE_STYLES[case["name"]][1],
            "linestyle": "--" if circuit_gap else "-",
            "zorder": 1 if case["name"] == "monolithic" else 2,
            "gap": conditional_curve,
            "weight": threshold_points([ScoreGroup(**group) for group in case["groups"][selection_statistic]], limits=count_limits) if not circuit_gap else [],
        })
    program = summary.get("program", "SteaneMemory")
    rounds = (
        "Z EC rounds with verified ancillas"
        if program == "SteaneZMemory"
        else "EC cycles with verified ancillas"
    )
    if surface:
        rounds = "syndrome-extraction rounds"
    if capacity:
        rounds = "data-noise layer; ideal preparation and Z readout"
    loss_fraction = summary.get("loss_fraction", 0.7)
    noise_label = ("SI1000 Pauli noise" if surface else "100% Pauli; no loss" if loss_fraction == 0
                   else f"{100 * (1 - loss_fraction):g}% Pauli / {100 * loss_fraction:g}% correlated loss")
    if capacity:
        noise_label = f"X only vs {100 * (1 - loss_fraction):g}% X / {100 * loss_fraction:g}% native loss"
    decoder_label = summary.get("decoder", "Tesseract")
    if summary.get("gap_decoder"):
        decoder_label += f" hard; {summary['gap_decoder'].removeprefix('black-box-')} approximate gap"
    if capacity and summary.get("gap_decoder") == "black-box-tesseract":
        decoder_label = "Tesseract hard/gap"
    progress = f"{summary['status']}: {sum(case['shots'] for case in summary['cases']):,} {'decoder evaluations' if paired_gaps else 'shots'}"
    if "shots" in summary:
        total_target = summary["shots"] * len(
            summary.get("configurations", summary["cases"])
        )
        progress += f" / {total_target:,} target"
    if summary.get("updated_at"):
        progress += f"; updated {summary['updated_at']}"
    parallelism = summary.get("window_parallelism")
    if parallelism is None and "configurations" in summary:
        parallelism = "sliding" if summary.get("causal_commit_order", False) else "all"
    scope = "final readouts" if summary.get("selection_scope") else "all readouts"
    title = f"{name}: {'code capacity' if capacity else program}, {scope}"
    if parallelism is not None and not capacity:
        title += ", sliding windows" if parallelism == "sliding" else ", full parallelism"
    draw(
        {
            **summary,
            "plotted_statistic": selection_statistic,
            "upper_bounds_only": upper_bounds_only,
            "plot_range_percent": [0, rejection_limit],
            "conditional_panel": has_failures,
            "failure_accounting": "left: all attempts, including failures; right: successful decodes only, failures excluded explicitly",
            "plotted_count_limits": list(count_limits) if count_limits else None,
            "window_parallelism": parallelism,
        },
        output,
        title=title + (" (upper bounds only)" if upper_bounds_only else ""),
        subtitle=f"{summary['rounds']} {rounds}; {decoder_label}; p={summary['physical_error_rate']:g}; {noise_label}\n{progress}",
        series=series,
        conditioned_series=conditioned_series if has_failures else None,
        rejection_limit=rejection_limit,
        threshold_label=SELECTION_STATISTICS[selection_statistic],
        threshold_markers=CORRECTION_COUNT_MARKERS if count_limits else None,
    )



def render_final_readouts(summary: dict, data_dir: Path, output: Path,
                          selection_statistic: str = "correction_count") -> None:
    count = 1 if summary.get("code") == "surface-code" else 2
    final = deepcopy(summary)
    final["selection_scope"] = f"Final {count} asserted logical readout(s)"
    for case in final["cases"]:
        if "final_gap" in case["groups"]:
            case["groups"]["gap"] = case["groups"]["final_gap"]
            continue
        groups = {}
        for filename in case["batch_records"]:
            record_path = data_dir / filename
            record = json.loads(record_path.read_text())
            trace_path = record_path.with_name("shots.pb")
            if digest(trace_path) != record["files"]["shots.pb"]:
                raise ValueError(f"trace checksum changed: {trace_path}")
            trace = load_trace(trace_path, record["identity"]["shots"])
            for shot in trace.shots:
                if not shot.HasField("decode_result"):
                    continue
                scores = shot.decode_result.probabilities
                if len(scores) < count or any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
                    raise ValueError("invalid final-readout scores")
                totals = groups.setdefault(max(scores[-count:]), [0, 0])
                totals[0] += 1
                totals[1] += int(shot.logical_error)
        case["groups"]["gap"] = [vars(ScoreGroup(score, *counts)) for score, counts in sorted(groups.items(), reverse=True)]
    render_native(final, output, selection_statistic)


def main():
    parser = argparse.ArgumentParser(description="Replot saved figure summaries without sampling or decoding.")
    parser.add_argument("summaries", nargs="+", type=Path, help="Figure JSON files produced by the renderer")
    parser.add_argument("--output-dir", type=Path, help="Destination directory (default: beside each input)")
    args = parser.parse_args()
    summaries = [(path, json.loads(path.read_text())) for path in args.summaries]
    for path, summary in summaries:
        output = (args.output_dir / path.name if args.output_dir else path).with_suffix(".pdf")
        render_native(summary, output, summary.get("plotted_statistic", "correction_count"))
        print(f"Replotted: {output} and {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()


def publish_figures(summary, data_dir, output, final_output=None, selection_statistic="correction_count"):
    render_native(summary, output, selection_statistic)
    if final_output is not None:
        render_final_readouts(summary, data_dir, final_output, selection_statistic)
