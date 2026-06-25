"""Generate annotated output for the multi-port tutorial examples."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from snippet_utils import extract_block

from deq.cli.annotate import annotate

this_dir = os.path.dirname(os.path.abspath(__file__))

examples = [
    ("01_noiseless.deq", "Transversal CNOT (no noise)"),
    ("02_noisy.deq", "Transversal CNOT (noisy)"),
    ("03_redundant.deq", "Transversal CNOT (redundant stabilizers)"),
]

for filename, label in examples:
    path = os.path.join(this_dir, filename)
    print(f"\n{'='*60}")
    print(f"Processing: {label}")

    # Annotate
    annotated_out = os.path.join(this_dir, filename.replace(".deq", ".annotated.deq"))
    annotate(path, out=annotated_out)
    print(f"  Annotated: {annotated_out}")

    # Extract TransversalCNOT snippet from annotated output
    with open(annotated_out, encoding="utf-8") as f:
        annotated_text = f.read()
    cnot_snippet = extract_block(annotated_text, "GADGET", "TransversalCNOT")
    snippet_path = os.path.join(
        this_dir, "snippet_" + filename.replace(".deq", "_cnot.deq")
    )
    with open(snippet_path, "w", encoding="utf-8") as f:
        f.write(cnot_snippet)
    print(f"  CNOT snippet: {snippet_path}")

# ── Extract shared TransversalCNOT gadget snippet from source ──────────

with open(os.path.join(this_dir, "01_noiseless.deq"), encoding="utf-8") as f:
    src_01 = f.read()
gadget_snippet = extract_block(src_01, "GADGET", "TransversalCNOT")
gadget_path = os.path.join(this_dir, "snippet_cnot_gadget.deq")
with open(gadget_path, "w", encoding="utf-8") as f:
    f.write(gadget_snippet)
print(f"  CNOT gadget snippet: {gadget_path}")

print("\nDone.")
