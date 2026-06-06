"""Generate all COMPOSE tutorial example outputs.

Runs `transpile` and `annotate` on all COMPOSE examples so that
breaking changes are caught by ``make tutorial``.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from snippet_utils import extract_block

this_dir = os.path.dirname(os.path.abspath(__file__))

examples = [
    "01_flat_3idle.deq",
    "02_compose_3idle.deq",
    "03_nested_compose.deq",
]


def run_cli(description: str, args: list[str]) -> str:
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


for example in examples:
    path = os.path.join(this_dir, example)
    base = os.path.splitext(example)[0]

    # Transpile — may fail on stim export for COMPOSE examples, but .jit is
    # written before the stim export step, so we allow non-zero exit codes
    out = os.path.join(this_dir, f"{example}.jit")
    print(f"  transpile {example}...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deq",
            "transpile",
            path,
            "--out",
            out,
            "--program",
            "Simulation",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if "cannot export stim" in result.stderr:
            print(
                f"    (stim export skipped — COMPOSE gadgets have no physical circuit)"
            )
        else:
            print(result.stderr)
            raise RuntimeError(f"transpile failed for {example}")

    # Annotate
    annotated_path = os.path.join(this_dir, f"{base}.annotated.deq")
    run_cli(f"annotate {example}", ["annotate", path, "--out", annotated_path])


# ── Extract annotated gadget snippets from annotated output ────────────

# Idle3 from 02_compose_3idle.annotated.deq
with open(
    os.path.join(this_dir, "02_compose_3idle.annotated.deq"), encoding="utf-8"
) as f:
    annotated_02 = f.read()
write(
    os.path.join(this_dir, "snippet_idle3_annotated.deq"),
    extract_block(annotated_02, "GADGET", "Idle3"),
)

# Idle4 from 03_nested_compose.annotated.deq
with open(
    os.path.join(this_dir, "03_nested_compose.annotated.deq"), encoding="utf-8"
) as f:
    annotated_03 = f.read()
write(
    os.path.join(this_dir, "snippet_idle4_annotated.deq"),
    extract_block(annotated_03, "GADGET", "Idle4"),
)


# ── Extract COMPOSE snippets from source files ────────────────────────

with open(os.path.join(this_dir, "02_compose_3idle.deq"), encoding="utf-8") as f:
    src_02 = f.read()
write(
    os.path.join(this_dir, "snippet_compose_idle3.deq"),
    extract_block(src_02, "COMPOSE", "Idle3"),
)

with open(os.path.join(this_dir, "03_nested_compose.deq"), encoding="utf-8") as f:
    src_03 = f.read()
# Both COMPOSE blocks together
write(
    os.path.join(this_dir, "snippet_nested_compose.deq"),
    extract_block(src_03, "COMPOSE", "Idle3")
    + "\n"
    + extract_block(src_03, "COMPOSE", "Idle4"),
)


print("\nDone.")
