"""Paired surface-code capacity comparison using exact minimum-weight matching."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pymatching
from scipy.sparse import csc_matrix, eye
from scipy.special import logsumexp
from scipy.stats import binom

from analyze_fire_ice_post_selection import (
    DEQ_ROOT,
    ScoreGroup,
    binomial_interval,
    digest,
    write_json,
)
from deq.circuit.model import CodeDefinition
from deq.circuit.parser import render_and_parse_file
from plot_fire_ice_findings import rejection_curve


FIXTURE = DEQ_ROOT / "tests/circuit/surface_code/surface_code.deq"
REJECTION_BUDGETS = (1, 5, 10)


def surface_code_checks(distance: int) -> tuple[np.ndarray, np.ndarray]:
    if distance < 3 or distance % 2 != 1:
        raise ValueError("distance must be odd and at least three")
    parsed = render_and_parse_file(
        str(FIXTURE), mako_defs={"d": str(distance)}, skip_mako_warning=True
    )
    code = next(item for item in parsed.definitions if isinstance(item, CodeDefinition))
    rows = []
    for product in [*code.stabilizers, code.logicals[0].z_operator]:
        if any(term.pauli != "Z" for term in product.terms):
            continue
        row = np.zeros(code.n, dtype=np.uint8)
        row[[term.index for term in product.terms]] = 1
        rows.append(row)
    checks = np.array(rows, dtype=np.uint8)
    if np.any(checks.sum(axis=0) > 2) or np.any(checks.sum(axis=0) == 0):
        raise ValueError("logical boundary must preserve a graphlike matching problem")
    return checks[:-1], checks[-1]


class CapacityDecoder:
    def __init__(self, distance: int):
        self.checks, self.logical = surface_code_checks(distance)
        self.forced_checks = np.vstack((self.checks, self.logical))
        faults = eye(len(self.logical), format="csc", dtype=np.uint8)
        self.primary = pymatching.Matching.from_check_matrix(
            csc_matrix(self.checks), weights=1.0, faults_matrix=faults
        )
        self.opposite = pymatching.Matching.from_check_matrix(
            csc_matrix(self.forced_checks), weights=1.0, faults_matrix=faults
        )

    def decode(self, errors: np.ndarray) -> dict[str, np.ndarray]:
        syndrome = (errors @ self.checks.T) & 1
        correction, weight = self.primary.decode_batch(syndrome, return_weights=True)
        prediction = (correction @ self.logical) & 1
        forced_syndrome = np.column_stack((syndrome, prediction ^ 1))
        alternative, alternative_weight = self.opposite.decode_batch(
            forced_syndrome, return_weights=True
        )
        if not np.array_equal((correction @ self.checks.T) & 1, syndrome):
            raise ValueError("primary correction does not satisfy the syndrome")
        if not np.array_equal(
            (alternative @ self.forced_checks.T) & 1, forced_syndrome
        ):
            raise ValueError(
                "forced correction does not satisfy the opposite logical class"
            )
        if not np.array_equal(weight, correction.sum(axis=1)) or not np.array_equal(
            alternative_weight, alternative.sum(axis=1)
        ):
            raise ValueError("matching weights do not agree with correction counts")
        if np.any(alternative_weight < weight):
            raise ValueError("primary solution is not minimum weight")
        return {
            "prediction": prediction.astype(np.uint8),
            "correction_weight": weight.astype(np.uint8),
            "gap": (alternative_weight - weight).astype(np.uint8),
            "logical_error": ((errors @ self.logical) & 1) != prediction,
        }


def paired_comparison(result: dict, seed: int, comparisons: int) -> list[dict]:
    errors = result["logical_error"]
    shots = len(errors)
    tie_keys = np.random.default_rng(seed).random(shots)
    orders = {
        "weight": np.lexsort((tie_keys, result["correction_weight"])),
        "gap": np.lexsort((tie_keys, -result["gap"].astype(np.float64))),
    }
    rows = []
    for rejection in REJECTION_BUDGETS:
        retained = shots - round(shots * rejection / 100)
        masks = {}
        counts = {}
        intervals = {}
        for method, order in orders.items():
            mask = np.zeros(shots, dtype=bool)
            mask[order[:retained]] = True
            masks[method] = mask & errors
            counts[method] = int(masks[method].sum())
            intervals[method] = list(binomial_interval(counts[method], retained))
        weight_only = int((masks["weight"] & ~masks["gap"]).sum())
        gap_only = int((masks["gap"] & ~masks["weight"]).sum())
        discordant = weight_only + gap_only
        log_pvalue = (
            min(
                0.0,
                float(
                    np.log(2)
                    + logsumexp(
                        binom.logpmf(
                            np.arange(min(weight_only, gap_only) + 1), discordant, 0.5
                        )
                    )
                ),
            )
            if discordant
            else 0.0
        )
        pvalue = float(np.exp(log_pvalue))
        rows.append(
            {
                "rejected_percent": rejection,
                "retained_shots": retained,
                "logical_errors": counts,
                "logical_error_rate": {
                    method: count / retained for method, count in counts.items()
                },
                "confidence_interval_95": intervals,
                "weight_only_errors": weight_only,
                "gap_only_errors": gap_only,
                "paired_pvalue": pvalue,
                "paired_log10_pvalue": log_pvalue / float(np.log(10)),
                "bonferroni_pvalue": min(1.0, comparisons * pvalue),
                "weight_over_gap": (
                    counts["weight"] / counts["gap"] if counts["gap"] else None
                ),
            }
        )
    return rows


def score_groups(result: dict, method: str) -> list[ScoreGroup]:
    scores = (
        result["correction_weight"]
        if method == "weight"
        else -result["gap"].astype(np.float64)
    )
    return [
        ScoreGroup(
            score.item(),
            int((scores == score).sum()),
            int(result["logical_error"][scores == score].sum()),
        )
        for score in sorted(np.unique(scores), reverse=True)
    ]


def render(summary: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(10, 7.2))
    figure.subplots_adjust(left=0.105, right=0.98, bottom=0.15, top=0.75)
    figure.suptitle(
        f"Distance-{summary['distance']} surface code: code-capacity X noise",
        y=0.97,
        fontsize=15,
    )
    figure.text(
        0.5,
        0.916,
        f"Exact MWPM logical gap vs correction weight; {summary['shots']:,} paired shots per p",
        ha="center",
        fontsize=10,
    )
    for color, case in zip(
        ("#0072B2", "#009E73", "#C64B37")[: len(summary["cases"])],
        summary["cases"],
        strict=True,
    ):
        for method, style in (("weight", "--"), ("gap", "-")):
            curve = rejection_curve(
                [ScoreGroup(**group) for group in case["groups"][method]]
            )
            axes.plot(
                [point.rejected_percent for point in curve],
                [point.rate or float("nan") for point in curve],
                color=color,
                linestyle=style,
                linewidth=1.8,
                label=f"p={case['physical_error_rate']:g}, {'forced gap' if method == 'gap' else 'correction weight'}",
            )
            for row in case["comparison"]:
                errors = row["logical_errors"][method]
                lower, upper = row["confidence_interval_95"][method]
                rate = row["logical_error_rate"][method]
                axes.errorbar(
                    row["rejected_percent"],
                    rate if errors else upper,
                    yerr=[[rate - lower], [upper - rate]] if errors else None,
                    marker=("o" if method == "gap" else "s") if errors else "v",
                    markerfacecolor="white",
                    color=color,
                    markersize=5,
                    capsize=2,
                )
    axes.set(
        xlim=(0, 12),
        ylim=(
            min(1e-7, 0.1 / summary["shots"]),
            max(1e-3, 2 * max(case["raw_ler"] for case in summary["cases"])),
        ),
        yscale="log",
        xlabel="Rejected shots (%)",
        ylabel="Logical error rate per retained shot",
    )
    axes.grid(axis="y", alpha=0.25)
    figure.legend(
        *axes.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=len(summary["cases"]),
        frameon=False,
        fontsize=8.5,
    )
    figure.text(
        0.5,
        0.035,
        "Lines: expected random ties. Markers: fixed seeded ties, 95% intervals; triangles: zero-error upper bounds.",
        fontsize=8,
        ha="center",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=160)
    write_json(summary, output.with_suffix(".json"))
    plt.close(figure)


def run(
    distance: int, shots: int, seed: int, output_dir: Path, figure_output: Path
) -> dict:
    if shots < 100:
        raise ValueError("at least 100 shots are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "manifest.json").exists():
        raise ValueError(
            "use a fresh output directory; existing evaluations are immutable"
        )
    probabilities = (0.01, 0.02)
    sources = {str(path.resolve()): digest(path) for path in (FIXTURE, Path(__file__))}
    manifest = {
        "distance": distance,
        "shots": shots,
        "seed": seed,
        "physical_error_rates": list(probabilities),
        "rejection_budgets": list(REJECTION_BUDGETS),
        "noise": "single independent X-error layer on data; ideal preparation, syndrome and readout",
        "primary_decoder": "PyMatching minimum-weight perfect matching",
        "forced_decoder": "same matching solver with an extra logical-boundary parity check",
        "weight_units": "unit Hamming weights; multiply by log((1-p)/p) for log-odds costs",
        "implementation_scope": "exact matching reference, not the DEQ/Tesseract runtime",
        "tie_rule": "fixed independent random keys, shared across both selection methods",
        "comparisons": len(probabilities) * len(REJECTION_BUDGETS),
        "source_sha256": sources,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("pymatching", "numpy", "scipy", "matplotlib")
        },
    }
    write_json(manifest, output_dir / "manifest.json")
    decoder = CapacityDecoder(distance)
    summary = {**manifest, "status": "running", "cases": []}
    for index, probability in enumerate(probabilities):
        rng = np.random.default_rng(seed + index)
        batches = []
        sampled_errors = []
        for start in range(0, shots, 50_000):
            errors = (
                rng.random((min(50_000, shots - start), distance**2)) < probability
            ).astype(np.uint8)
            batches.append(decoder.decode(errors))
            sampled_errors.append(np.packbits(errors, axis=1))
        result = {
            key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]
        }
        artifact = output_dir / f"p-{probability:g}.npz"
        np.savez_compressed(
            artifact,
            **result,
            sampled_errors=np.concatenate(sampled_errors),
            checks=decoder.checks,
            logical=decoder.logical,
        )
        comparison = paired_comparison(
            result, seed + 1_000_000 + index, manifest["comparisons"]
        )
        groups = {
            method: [vars(group) for group in score_groups(result, method)]
            for method in ("weight", "gap")
        }
        summary["cases"].append(
            {
                "physical_error_rate": probability,
                "seed": seed + index,
                "tie_seed": seed + 1_000_000 + index,
                "logical_errors": int(result["logical_error"].sum()),
                "raw_ler": float(result["logical_error"].mean()),
                "data_file": artifact.name,
                "data_sha256": digest(artifact),
                "comparison": comparison,
                "groups": groups,
            }
        )
        write_json(summary, output_dir / "summary.json")
        print(
            json.dumps(
                {
                    "p": probability,
                    "raw_errors": int(result["logical_error"].sum()),
                    "comparison": comparison,
                }
            ),
            flush=True,
        )
    if any(digest(Path(path)) != fingerprint for path, fingerprint in sources.items()):
        raise ValueError("source changed during evaluation")
    summary["status"] = "complete"
    write_json(summary, output_dir / "summary.json")
    render(summary, figure_output)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=7)
    parser.add_argument("--shots", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results/surface_code_d7_10m_20260915",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path(__file__).parent / "surface_code_d7_post_selection.pdf",
    )
    run(**vars(parser.parse_args()))
