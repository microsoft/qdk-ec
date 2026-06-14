"""Generate all debug tutorial example outputs.

Runs the CLI commands referenced in debug-deq-program.md so that
breaking changes are caught by ``make tutorial``.
"""

import os
import subprocess
import sys

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


def run(description: str, args: list[str]) -> str:
    """Run a CLI command and return stdout."""
    print(f"  {description}...")
    result = subprocess.run(
        [sys.executable, "-m", "deq"] + args,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def write(path: str, content: str) -> None:
    """Write content to a file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"    -> {os.path.basename(path)}")


# ── Level 1: annotate ──────────────────────────────────────────────
annotated_path = os.path.join(this_dir, "03_with_idle.annotated.deq")
run("annotate", ["annotate", deq_file, "--out", annotated_path])

# ── Level 3: compile + canonicalize ─────────────────────────────────────────
bin_file = os.path.join(this_dir, "03_with_idle.deq.bin")
run(
    "compile",
    [
        "compile",
        jit_file,
        "--out",
        bin_file,
    ],
)

canonical_file = os.path.join(this_dir, "03_with_idle.canonical.deq.bin")
run(
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

output = run(
    "sample",
    ["sample", stim_file, "--shots", "10", "--seed", "1"],
)
write(os.path.join(this_dir, "stim_sample_output.txt"), output)

for hex_val, label in [
    ("0x00", "no_error"),
    ("0x80", "ancilla_error"),
    ("0x20", "data_error"),
]:
    output = run(
        f"interpret ({label})",
        ["interpret", bin_file, "--measurements", hex_val],
    )
    write(os.path.join(this_dir, f"interpret_{label}.txt"), output)
