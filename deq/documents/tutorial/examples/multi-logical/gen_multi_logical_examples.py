"""Generate outputs for the multi-logical tutorial examples.

Runs CLI commands referenced in codes-multi-logical.md so that
breaking changes are caught by ``make tutorial``.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from snippet_utils import extract_block

from deq.cli.jit import transpile

this_dir = os.path.dirname(os.path.abspath(__file__))

source = os.path.join(this_dir, "multi-logical.deq")

# --- Transpile ---
print("Transpiling multi-logical.deq...")
transpile(
    source,
    out=os.path.join(this_dir, "multi-logical.deq.jit"),
    program="MemoryExperiment",
    jobs=1,
)

# --- Extract snippets from source ---
with open(source, encoding="utf-8") as f:
    text = f.read()

snippets = {
    "snippet_code.deq": ("CODE", "QuantumHamming"),
    "snippet_idle.deq": ("GADGET", "Idle"),
    "snippet_measure.deq": ("GADGET", "MeasureZAll"),
    "snippet_program.deq": ("PROGRAM", "MemoryExperiment"),
}

for filename, (keyword, name) in snippets.items():
    block = extract_block(text, keyword, name)
    snippet_path = os.path.join(this_dir, filename)
    with open(snippet_path, "w", encoding="utf-8") as f:
        f.write(block)
    print(f"  Snippet: {filename}")

print("Done.")
