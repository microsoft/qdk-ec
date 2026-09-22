"""Generate and verify the non-Clifford tutorial examples.

Run from this directory with ``python gen_non_clifford.py``. The tutorial
generator runner also discovers this script automatically. Annotation verifies
that the generated files compile equivalently to the source. Interactive
assertions verify adaptive decoding without future gadgets.
"""

import argparse
import asyncio
from pathlib import Path

from deq.cli.annotate import annotate
from interactive_t import run


HERE = Path(__file__).resolve().parent


def main() -> None:
    annotate(str(HERE / "rotations.deq"))
    annotate(str(HERE / "trivial_non_clifford.deq"), mako=["p=0.001"])
    results = asyncio.run(run(argparse.Namespace(
        axis="both", shots=32, noise_shots=128, noise=0.01, seed=144,
        output=None,
    )))
    assert len(results["tomography"]) == 24
    assert len(results["noise"]) == 3
    for row in results["tomography"]:
        assert row["shots"] == 32
        assert row["t_readouts"] == row["gates"] * 32
        assert 0 < row["conditional_s_readouts"] < row["t_readouts"]
        if row["gates"] == 4 and row["basis"] == row["prep"]:
            assert row["ones"] == 32


if __name__ == "__main__":
    main()