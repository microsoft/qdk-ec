"""Generate transpiled, annotated, and snippet outputs for the Floquet tutorial.

Produces, next to ``floquet.deq``:

* ``floquet.deq.jit`` + ``floquet.deq.jit.txt`` (transpiler output for ``MemoryDirect``)
* ``MemoryDirect.stim`` (companion Stim circuit emitted alongside the JIT library)
* ``floquet.annotated.deq`` (round-trip-verified annotated form)
* ``snippet_*.deq`` files referenced from the tutorial chapter
* ``lattice_d3.png`` figure of the 3x3 honeycomb torus geometry

The simulation that backs the chapter's logical error rate section is run
separately by hand (see the chapter for the exact command); reproducing it
on every ``make tutorial`` invocation would be too slow.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from snippet_utils import extract_block, write_snippet  # noqa: E402

from deq.cli.annotate import annotate  # noqa: E402
from deq.cli.jit import transpile  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(THIS_DIR, "floquet.deq")
JIT = os.path.join(THIS_DIR, "floquet.deq.jit")
ANNOTATED = os.path.join(THIS_DIR, "floquet.annotated.deq")
LATTICE_PNG = os.path.join(THIS_DIR, "lattice_d3.png")


def extract_propagate_only(text: str, gadget_name: str) -> str:
    """Return a stub ``GADGET <name> { # other entries omitted; PROPAGATE... }``.

    Pulls the ``PROPAGATE`` lines out of the annotated ``GADGET <name>`` block
    and wraps them in a minimal gadget shell suitable as a tutorial snippet.
    """

    block = extract_block(text, "GADGET", gadget_name)
    propagate_lines = [
        line for line in block.splitlines() if line.lstrip().startswith("PROPAGATE")
    ]
    if not propagate_lines:
        raise ValueError(f"No PROPAGATE lines found in GADGET {gadget_name}")
    body = "\n".join(propagate_lines)
    return f"GADGET {gadget_name} {{\n    # other entries omitted\n{body}\n}}\n"


def main() -> None:
    print(f"Transpiling {os.path.basename(SOURCE)} (program=MemoryDirect)...")
    transpile(SOURCE, out=JIT, program="MemoryDirect", skip_mako_warning=True)

    print(f"Annotating  {os.path.basename(SOURCE)}...")
    annotate(SOURCE, out=ANNOTATED, skip_mako_warning=True, mako=["p=0"])

    with open(SOURCE, encoding="utf-8") as f:
        src = f.read()
    with open(ANNOTATED, encoding="utf-8") as f:
        annotated_src = f.read()

    # Source snippets ----------------------------------------------------
    snippets_from_source = [
        ("snippet_code_red.deq", "CODE", "HoneycombR"),
        ("snippet_code_green.deq", "CODE", "HoneycombG"),
        ("snippet_code_blue.deq", "CODE", "HoneycombB"),
        ("snippet_round_red.deq", "GADGET", "RoundRed"),
        ("snippet_round_green.deq", "GADGET", "RoundGreen"),
        ("snippet_round_blue.deq", "GADGET", "RoundBlue"),
        ("snippet_syndrome.deq", "COMPOSE", "Syndrome"),
        ("snippet_prepare_blue.deq", "GADGET", "PrepareBlue"),
        ("snippet_measure_blue.deq", "GADGET", "MeasureBlue"),
        ("snippet_program_direct.deq", "PROGRAM", "MemoryDirect"),
    ]
    for filename, keyword, name in snippets_from_source:
        write_snippet(
            os.path.join(THIS_DIR, filename), extract_block(src, keyword, name)
        )

    # Centerpiece: the annotated RoundRed gadget showing the dynamic logical
    # Pauli frame propagation (PROPAGATE ... IN<p>.DS<s> M<i> FLIP) and the
    # auto-derived check rows.
    write_snippet(
        os.path.join(THIS_DIR, "snippet_round_red_annotated.deq"),
        extract_block(annotated_src, "GADGET", "RoundRed"),
    )

    # Just the PROPAGATE rows from the annotated Syndrome gadget, wrapped in
    # a stub GADGET block (the full Syndrome body is huge; this snippet shows
    # only the per-round logical-frame propagation distilled from one full
    # R/G/B cycle).
    write_snippet(
        os.path.join(THIS_DIR, "snippet_syndrome_propagate.deq"),
        extract_propagate_only(annotated_src, "Syndrome"),
    )

    print("Done.")


if __name__ == "__main__":
    main()
