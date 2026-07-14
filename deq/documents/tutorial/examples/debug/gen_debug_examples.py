"""Generate all debug tutorial example outputs.

Runs the CLI commands referenced in debug-deq-program.md so that
breaking changes are caught by ``make tutorial``.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from snippet_utils import run_cli, write_snippet  # noqa: E402

this_dir = os.path.dirname(os.path.abspath(__file__))
language_dir = os.path.join(this_dir, "..", "language")

# Input files (produced by gen_language_examples.py)
deq_file = os.path.join(language_dir, "03_with_idle.deq")
jit_file = os.path.join(language_dir, "03_with_idle.deq.jit")

# Auto-run the dependency generator if prerequisite files are missing
if not os.path.isfile(jit_file):
    print("  Prerequisites missing — running gen_language_examples.py first...")
    dep_script = os.path.join(language_dir, "gen_language_examples.py")
    result = subprocess.run(
        [sys.executable, dep_script],
        cwd=language_dir,
    )
    if result.returncode != 0:
        raise RuntimeError("gen_language_examples.py failed")
    if not os.path.isfile(jit_file):
        raise FileNotFoundError(
            f"Prerequisite still missing after running gen_language_examples.py: {jit_file}"
        )


# ── Level 1: annotate ──────────────────────────────────────────────
annotated_path = os.path.join(this_dir, "03_with_idle.annotated.deq")
run_cli("annotate", ["annotate", deq_file, "--out", annotated_path])

# ── Level 3: compile + canonicalize ─────────────────────────────────────────
bin_file = os.path.join(this_dir, "03_with_idle.deq.bin")
run_cli(
    "compile",
    [
        "compile",
        jit_file,
        "--out",
        bin_file,
    ],
)

canonical_file = os.path.join(this_dir, "03_with_idle.canonical.deq.bin")
run_cli(
    "canonicalize",
    [
        "canonicalize",
        bin_file,
        "--out",
        canonical_file,
    ],
)

# ── Level 4: sample + interpret ─────────────────────────────────────
stim_file = os.path.join(language_dir, "03_with_idle.stim")

_, output, _ = run_cli(
    "sample",
    ["sample", stim_file, "--shots", "10", "--seed", "1"],
)
write_snippet(os.path.join(this_dir, "stim_sample_output.txt"), output)

for hex_val, label in [
    ("0x00", "no_error"),
    ("0x80", "ancilla_error"),
    ("0x20", "data_error"),
]:
    _, output, _ = run_cli(
        f"interpret ({label})",
        ["interpret", bin_file, "--measurements", hex_val],
    )
    write_snippet(os.path.join(this_dir, f"interpret_{label}.txt"), output)
