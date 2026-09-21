"""Generate and verify the non-Clifford tutorial's annotated example.

Run from this directory with ``python gen_non_clifford.py``. The tutorial
generator runner also discovers this script automatically. Annotation verifies
that the generated file compiles equivalently to the source.
"""

from pathlib import Path

from deq.cli.annotate import annotate


HERE = Path(__file__).resolve().parent


def main() -> None:
    annotate(str(HERE / "rotations.deq"))


if __name__ == "__main__":
    main()