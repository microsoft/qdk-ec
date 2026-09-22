"""Exercise every figure workflow during tutorial refresh without replacing published results."""

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def main():
    from run_post_selection import batch_seed

    parameters = {"seed": 17, "batch_size": 1000, "batch_seed_stride": 16}
    assert batch_seed(parameters, 1000) == batch_seed(parameters, 0) + 16
    example = Path(__file__).resolve().parent
    with TemporaryDirectory(prefix="post-selection-smoke-") as directory:
        root = Path(directory)
        subprocess.run(
            ["make", "all", f"PYTHON={sys.executable}", "SHOTS=10", "CAPACITY_SHOTS=10",
             "JOBS=2", "CAPACITY_JOBS=2", "ROUNDS=2", "FIRE_BATCH_SIZE=10", "SURFACE_BATCH_SIZE=10",
             "PLOT_INTERVAL=1", "SCHEDULER_URL=", "EXTRA_ARGS=",
             f"DATA_DIR={root / 'data'}", f"FIGURES_DIR={root / 'figures'}"],
            cwd=example, check=True, timeout=600,
            env={**os.environ, "MAKEFLAGS": "", "MFLAGS": ""},
        )
        for study in ("capacity", "surface", "fire-ice-sliding", "fire-ice-all"):
            summary = json.loads((root / "data" / study / "summary.json").read_text())
            expected_cases = 4 if study == "capacity" else 8
            if summary["status"] != "complete" or len(summary["cases"]) != expected_cases:
                raise RuntimeError(f"{study}: incomplete smoke evaluation")
            if study == "capacity" and (
                summary.get("program") != "CodeCapacityZMemory" or summary.get("decoder") != "Tesseract"
            ):
                raise RuntimeError("capacity must exercise the native DEQ program and decoder")
            expected_stride = 1 if study == "surface" else (summary["batch_size"] + 63) // 64
            if summary.get("batch_seed_stride") != expected_stride:
                raise RuntimeError(f"{study}: overlapping sampler seed ranges")
            if any(case["shots"] != 10 or case.get("failed_shots", 0) for case in summary["cases"]):
                raise RuntimeError(f"{study}: failed smoke evaluation shots")
            if study == "capacity":
                cases = {case["name"]: case for case in summary["cases"]}
                for name in ("capacity-pauli", "capacity-mixed"):
                    precise, fast = cases[name], cases[name + "-circuit-gap"]
                    if precise["logical_errors"] != fast["logical_errors"] or any(
                        precise["groups"][statistic] != fast["groups"][statistic]
                        for statistic in ("correction_count", "correction_weight", "syndrome_count")
                    ):
                        raise RuntimeError("capacity gap comparison changed hard decoding")
        figures = list((root / "figures").glob("*.pdf"))
        if len(figures) != 5 or any(not path.read_bytes().startswith(b"%PDF") for path in figures):
            raise RuntimeError("expected five generated PDF figures")
        if any(not path.with_suffix(".png").is_file() for path in figures):
            raise RuntimeError("missing generated PNG figure")
    print("Post-selection smoke passed: 10 shots per configuration, all five figures, no decoding failures.")


if __name__ == "__main__":
    main()
