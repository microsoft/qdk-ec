#!/usr/bin/env python3
"""Run all gen_*.py scripts under documents/tutorial/examples/.

Each generator script is executed with its parent directory as the working
directory, so it can reference sibling .deq files with relative paths.

Scripts are discovered recursively and executed in sorted order for
determinism.
"""


import subprocess
import sys
from pathlib import Path


EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def main() -> int:
    scripts = sorted(EXAMPLES_DIR.rglob("gen_*.py"))
    if not scripts:
        print("  No gen_*.py scripts found.")
        return 0

    failed = False
    for script in scripts:
        rel = script.relative_to(EXAMPLES_DIR.parent.parent)
        print(f"  running: {rel}")
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(script.parent),
        )
        if result.returncode != 0:
            print(f"  FAILED: {rel} (exit code {result.returncode})", file=sys.stderr)
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
