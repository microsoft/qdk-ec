"""Generate and verify the non-Clifford tutorial examples.

Run from this directory with ``python gen_non_clifford.py``. The tutorial
generator runner also discovers this script automatically. Annotation verifies
that the generated files compile equivalently to the source. Interactive
assertions verify adaptive decoding and automatic Pauli-frame propagation.
"""

import argparse
import asyncio
from pathlib import Path

from deq.circuit.model import Instruction, PropagateStatement
from deq.cli.annotate import annotate
from interactive_t import make_library, run


HERE = Path(__file__).resolve().parent


def main() -> None:
    annotate(str(HERE / "rotations.deq"))
    annotate(str(HERE / "interactive_t.deq"), mako=["p=0.001"])
    library, gadgets = make_library(0.0)
    assert not any(
        isinstance(statement, PropagateStatement)
        for gadget in gadgets.values() for statement in gadget.body
    )
    types = {gadget.base.name: gadget.base for gadget in library.gadget_types}
    for axis, frame_rows in (("X", (1,)), ("Y", (0, 1)), ("Z", (0,))):
        name = f"Correct{axis}"
        assert not any(isinstance(statement, Instruction) for statement in gadgets[name].body)
        correction = types[name].correction_propagation
        assert set(zip(correction.i, correction.j)) == {(0, 0), (1, 1)} | {(row, 2) for row in frame_rows}
    coupling_entries = {
        "X": {(0, 0), (1, 1), (1, 3), (2, 0), (2, 2), (3, 3)},
        "Z": {(0, 0), (0, 2), (1, 1), (2, 2), (3, 1), (3, 3)},
    }
    for axis in ("X", "Z"):
        propagation = types[f"Couple{axis}"].correction_propagation
        assert set(zip(propagation.i, propagation.j)) == coupling_entries[axis]
        readout = types[f"Read{axis}"].readout_propagation
        assert set(zip(readout.i, readout.j)) == {(0, 0 if axis == "X" else 1)}
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