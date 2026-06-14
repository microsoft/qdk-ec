"""Generate outputs for the Steane-style EC tutorial chapter.

Runs transpile, annotate, and extracts snippets referenced in
steane-style-ec.md so that breaking changes are caught by ``make tutorial``.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from snippet_utils import extract_block, write_snippet

from deq.cli.jit import transpile
from deq.cli.annotate import annotate

this_dir = os.path.dirname(os.path.abspath(__file__))

# ── Transpile ──────────────────────────────────────────────────────────

source = os.path.join(this_dir, "steane_code.deq")

print("Transpiling steane_code.deq (no phantom)...")
transpile(
    source,
    out=os.path.join(this_dir, "steane_code.deq.jit"),
    program="Minimal",
    mako=["p=0.001"],
)

print("Transpiling steane_code.deq (with phantom)...")
transpile(
    source,
    out=os.path.join(this_dir, "steane_code_phantom.deq.jit"),
    program="Minimal",
    mako=["p=0.001", "has_phantom=1"],
)

# ── Annotate (noiseless for clean check structure) ─────────────────────

print("Annotating steane_code.deq...")
annotate(
    source,
    out=os.path.join(this_dir, "steane_code.annotated.deq"),
    mako=["p=0"],
)

# ── Extract snippets ───────────────────────────────────────────────────

with open(source, encoding="utf-8") as f:
    src = f.read()

with open(os.path.join(this_dir, "steane_code.annotated.deq"), encoding="utf-8") as f:
    annotated = f.read()

print("Extracting snippets...")
write_snippet(
    os.path.join(this_dir, "snippet_code_definition.deq"),
    extract_block(src, "CODE", "SteaneCode"),
)
write_snippet(
    os.path.join(this_dir, "snippet_syndrome_gadget.deq"),
    extract_block(src, "GADGET", "SteaneSyndrome"),
)
write_snippet(
    os.path.join(this_dir, "snippet_measure_phantom.deq"),
    extract_block(src, "GADGET", "MeasureZ"),
)
write_snippet(
    os.path.join(this_dir, "snippet_syndrome_annotated.deq"),
    extract_block(annotated, "GADGET", "SteaneSyndrome"),
)

print("Done.")
