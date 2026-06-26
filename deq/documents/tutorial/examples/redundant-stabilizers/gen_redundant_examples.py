"""Generate annotated output for the redundant-stabilizer tutorial examples."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from snippet_utils import extract_block, write_snippet

from deq.cli.jit import transpile
from deq.cli.annotate import annotate

this_dir = os.path.dirname(os.path.abspath(__file__))

examples = [
    ("01_non_redundant.deq", "Non-redundant (2 stabilizers, 3 ancillae)"),
    ("02_redundant.deq", "Redundant (3 stabilizers, 3 ancillae)"),
]


for filename, label in examples:
    path = os.path.join(this_dir, filename)
    print(f"\n{'='*60}")
    print(f"Processing: {label}")

    # Transpile
    jit_out = os.path.join(this_dir, filename.replace(".deq", ".deq.jit"))
    transpile(path, out=jit_out, program="Simulation", jobs=1)

    # Annotate
    annotated_out = os.path.join(this_dir, filename.replace(".deq", ".annotated.deq"))
    annotate(path, out=annotated_out)
    print(f"  Annotated: {annotated_out}")

    # Also produce a snippet of just the Idle gadget from the annotated file
    with open(annotated_out, encoding="utf-8") as f:
        annotated_text = f.read()
    idle_snippet = extract_block(annotated_text, "GADGET", "Idle")
    snippet_path = os.path.join(
        this_dir, "snippet_" + filename.replace(".deq", "_idle.deq")
    )
    with open(snippet_path, "w", encoding="utf-8") as f:
        f.write(idle_snippet)
    print(f"  Idle snippet: {snippet_path}")


# ── Extract CODE and shared Idle snippets from source files ────────────

with open(os.path.join(this_dir, "01_non_redundant.deq"), encoding="utf-8") as f:
    src_01 = f.read()
write_snippet(
    os.path.join(this_dir, "snippet_code_non_redundant.deq"),
    extract_block(src_01, "CODE", "RepetitionCode"),
)
write_snippet(
    os.path.join(this_dir, "snippet_idle.deq"),
    extract_block(src_01, "GADGET", "Idle"),
)

with open(os.path.join(this_dir, "02_redundant.deq"), encoding="utf-8") as f:
    src_02 = f.read()
write_snippet(
    os.path.join(this_dir, "snippet_code_redundant.deq"),
    extract_block(src_02, "CODE", "RepetitionCode"),
)

print("\nDone.")
