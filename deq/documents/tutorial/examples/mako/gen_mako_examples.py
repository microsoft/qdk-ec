"""Generate outputs for the Mako parametrization tutorial examples.

Runs CLI commands referenced in mako-parametrization.md so that breaking
changes are caught by ``make tutorial``.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from snippet_utils import write_snippet

from deq.cli.jit import transpile
from deq.circuit.mako_support import render_mako

this_dir = os.path.dirname(os.path.abspath(__file__))

# ── 1. Transpile the fixed d=3 example ────────────────────────────────

fixed = os.path.join(this_dir, "01_fixed_d3.deq")
print("Transpiling 01_fixed_d3.deq...")
transpile(
    fixed,
    out=os.path.join(this_dir, "01_fixed_d3.deq.jit"),
    program="MemoryExperiment",
    jobs=1,
)

# ── 2. Transpile the parametrized example at several distances ────────

parametrized = os.path.join(this_dir, "02_parametrized.deq")

for d in [3, 5, 7]:
    print(f"Transpiling 02_parametrized.deq with d={d}...")
    transpile(
        parametrized,
        out=os.path.join(this_dir, f"02_parametrized_d{d}.deq.jit"),
        program="MemoryExperiment",
        mako=[f"d={d}"],
        jobs=1,
    )

# ── 3. Render the template at d=5 so readers can see the expansion ────

print("Rendering 02_parametrized.deq with d=5 → 02_parametrized_d5.deq")
with open(parametrized, encoding="utf-8") as f:
    template_text = f.read()
rendered = render_mako(template_text, {"d": "5", "p": "0.05"})
rendered_path = os.path.join(this_dir, "02_parametrized_d5.deq")
with open(rendered_path, "w", encoding="utf-8") as f:
    f.write(rendered)

# ── 4. Extract snippet files from the template source ─────────────────

print("Extracting snippets from 02_parametrized.deq...")

# Snippet: just the <%...%> parameter block
header_lines = []
in_block = False
for line in template_text.splitlines():
    if line.strip().startswith("<%"):
        in_block = True
    if in_block:
        header_lines.append(line)
    if in_block and line.strip().startswith("%>"):
        break
write_snippet(
    os.path.join(this_dir, "snippet_mako_header.deq"),
    "\n".join(header_lines) + "\n",
)

# Snippet: the parametrized CODE block (first block starting with CODE)
code_lines = []
capturing = False
brace_depth = 0
for line in template_text.splitlines():
    if not capturing and line.strip().startswith("CODE "):
        capturing = True
    if capturing:
        code_lines.append(line)
        brace_depth += line.count("{") - line.count("}")
        if brace_depth <= 0:
            break
write_snippet(
    os.path.join(this_dir, "snippet_mako_code.deq"),
    "\n".join(code_lines) + "\n",
)

# Snippet: the parametrized Syndrome gadget
gadget_lines = []
capturing = False
brace_depth = 0
for line in template_text.splitlines():
    if not capturing and "GADGET Syndrome" in line.strip():
        capturing = True
    if capturing:
        gadget_lines.append(line)
        brace_depth += line.count("{") - line.count("}")
        if brace_depth <= 0:
            break
write_snippet(
    os.path.join(this_dir, "snippet_mako_gadget.deq"),
    "\n".join(gadget_lines) + "\n",
)

# ── 5. Transpile the <%include> example ───────────────────────────────

include_file = os.path.join(this_dir, "03_include.deq")
print("Transpiling 03_include.deq...")
transpile(
    include_file,
    out=os.path.join(this_dir, "03_include.deq.jit"),
    skip_mako_warning=True,
    jobs=1,
)

print("\nDone.")
