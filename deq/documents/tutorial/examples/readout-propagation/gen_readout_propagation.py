"""Generate the annotated fixture referenced by the ``readout-propagation``
tutorial chapter.

``documents/tutorial/chapters/readout-propagation.md`` includes highlighted
snippets from
``tests/circuit/repetition_code/exercise_readout_conditions.annotated.deq``.
That ``.annotated.deq`` file is ``.gitignore``d (``*.annotated.deq`` under
``deq/.gitignore``), so it doesn't ship in the repo — CI must regenerate it
before ``highlight_deq.py`` runs.  This script is discovered by
``documents/tutorial/scripts/run_generators.py`` (which globs
``gen_*.py`` under ``documents/tutorial/examples/``) and produces the
missing file in place.
"""

import os

from deq.cli.annotate import annotate

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# THIS_DIR = deq/documents/tutorial/examples/readout-propagation/, so four
# ``..`` steps land at the ``deq/`` package root.
DEQ_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", ".."))
SOURCE = os.path.join(
    DEQ_ROOT, "tests", "circuit", "repetition_code",
    "exercise_readout_conditions.deq",
)
ANNOTATED = os.path.join(
    DEQ_ROOT, "tests", "circuit", "repetition_code",
    "exercise_readout_conditions.annotated.deq",
)


def main() -> None:
    print(f"Annotating  {os.path.relpath(SOURCE, DEQ_ROOT)}...")
    annotate(SOURCE, out=ANNOTATED, skip_mako_warning=True)


if __name__ == "__main__":
    main()
