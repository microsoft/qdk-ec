"""Post-selection score grouping, rejection curves, and confidence intervals."""

from __future__ import annotations

from bisect import bisect_left
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from deq.proto import simulator_pb2
from scipy.stats import beta


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from simulation import DEQ_ROOT


FIXTURE = DEQ_ROOT / "tests/circuit/fixtures/fire_ice.deq"


@dataclass(frozen=True)
class ScoreGroup:
    score: float
    shots: int
    logical_errors: int


def merge_score_groups(batches):
    groups = {}
    for batch in batches:
        for statistic, bins in batch.items():
            pooled = groups.setdefault(statistic, {})
            for group in bins:
                counts = pooled.setdefault(group["score"], [0, 0])
                counts[0] += group["shots"]
                counts[1] += group["logical_errors"]
    return {statistic: [vars(ScoreGroup(score, *counts)) for score, counts in sorted(bins.items(), reverse=True)]
            for statistic, bins in groups.items()}


@dataclass(frozen=True)
class RejectionPoint:
    rejected_percent: float
    retained_shots: int
    expected_errors: float
    rate: float
    upper_limit: float | None
    exact_threshold: bool


def rejection_curve(groups: list[ScoreGroup]) -> list[RejectionPoint]:
    """Average uniform tie-breaking, retaining actual integer sample sizes."""
    if not groups or any(
        group.shots <= 0 or not 0 <= group.logical_errors <= group.shots
        for group in groups
    ):
        raise ValueError("nonempty valid score groups are required")
    pooled: dict[float, list[int]] = {}
    for group in groups:
        if not math.isfinite(group.score):
            raise ValueError("scores must be finite")
        counts = pooled.setdefault(group.score, [0, 0])
        counts[0] += group.shots
        counts[1] += group.logical_errors
    ordered = sorted(pooled.items())
    total = sum(counts[0] for counts in pooled.values())
    cumulative_shots = []
    cumulative_errors = [0]
    for _, (shots, errors) in ordered:
        cumulative_shots.append(
            (cumulative_shots[-1] if cumulative_shots else 0) + shots
        )
        cumulative_errors.append(cumulative_errors[-1] + errors)
    thresholds = set(cumulative_shots)
    retained_counts = thresholds | {
        max(1, total - total * index // 1000) for index in range(1000)
    }
    points = []
    for retained in sorted(retained_counts, reverse=True):
        index = bisect_left(cumulative_shots, retained)
        shots, errors = ordered[index][1]
        selected = retained - (cumulative_shots[index - 1] if index else 0)
        expected = cumulative_errors[index] + (
            errors if selected == shots else selected * errors / shots
        )
        points.append(
            RejectionPoint(
                rejected_percent=100 * (total - retained) / total,
                retained_shots=retained,
                expected_errors=expected,
                rate=expected / retained,
                upper_limit=(
                    binomial_interval(0, retained)[1] if expected == 0 else None
                ),
                exact_threshold=retained in thresholds,
            )
        )
    return points


def count_selection(
    trace: simulator_pb2.SimulatorTrace, limit: int | None = None
) -> list[int]:
    if limit is not None and limit < 0:
        raise ValueError("correction-count thresholds must be nonnegative")
    retained = errors = 0
    for index, shot in enumerate(trace.shots):
        if shot.shot != index:
            raise ValueError("count-selection trace has misaligned shot IDs")
        if shot.HasField("decode_result"):
            if limit is not None:
                if not shot.gadget_readouts:
                    raise ValueError(
                        "trace has no per-gadget statistics; generate a new trace"
                    )
                if (
                    max(gadget.correction_count for gadget in shot.gadget_readouts)
                    > limit
                ):
                    continue
            retained += 1
            errors += int(shot.logical_error)
        elif shot.logical_error:
            raise ValueError("a failed decode cannot have a logical-error label")
    return [len(trace.shots), retained, errors]


def trace_statistics(
    trace: simulator_pb2.SimulatorTrace, *, scored: bool = True, final_readouts: int | None = None
) -> dict[str, list[ScoreGroup]]:
    """Group offline selection statistics, taking a maximum across each shot's gadgets."""
    count_selection(trace)
    statistics = ("correction_count", "correction_weight", "syndrome_count")
    groups: dict[str, dict[float, list[int]]] = {
        name: {} for name in (("gap",) if scored else ()) + statistics
    }
    if scored and final_readouts is not None:
        if final_readouts < 1:
            raise ValueError("final_readouts must be positive")
        groups["final_gap"] = {}
    for shot in trace.shots:
        if not shot.HasField("decode_result"):
            continue
        if not shot.gadget_readouts:
            raise ValueError("trace has no per-gadget statistics; generate a new trace")
        if any(
            not math.isfinite(gadget.correction_weight)
            for gadget in shot.gadget_readouts
        ):
            raise ValueError("expected finite per-gadget correction weights")
        values = {
            name: max(getattr(gadget, name) for gadget in shot.gadget_readouts)
            for name in statistics
        }
        if scored:
            probabilities = shot.decode_result.probabilities
            if not probabilities or any(
                not math.isfinite(score) or not 0 <= score <= 1
                for score in probabilities
            ):
                raise ValueError("expected finite readout scores in [0, 1]")
            values["gap"] = max(probabilities)
            if final_readouts is not None:
                if len(probabilities) < final_readouts:
                    raise ValueError("missing final logical readout scores")
                values["final_gap"] = max(probabilities[-final_readouts:])
        for name, value in values.items():
            counts = groups[name].setdefault(value, [0, 0])
            counts[0] += 1
            counts[1] += int(shot.logical_error)
    return {
        name: [
            ScoreGroup(score, *counts)
            for score, counts in sorted(bins.items(), reverse=True)
        ]
        for name, bins in groups.items()
    }


def binomial_interval(
    errors: int, shots: int, *, alpha: float = 0.05
) -> tuple[float, float]:
    if not 0 <= errors <= shots or not 0 < alpha < 1:
        raise ValueError("invalid binomial counts or confidence level")
    if shots == 0:
        return 0.0, 1.0
    lower = (
        0.0 if errors == 0 else float(beta.ppf(alpha / 2, errors, shots - errors + 1))
    )
    upper = (
        1.0
        if errors == shots
        else float(beta.ppf(1 - alpha / 2, errors + 1, shots - errors))
    )
    return lower, upper
