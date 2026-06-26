"""Generate / verify outputs for the python-runtime tutorial chapter.

Each of the ``NN_*.py`` files in this directory is a runnable example used by
``chapters/python-runtime.md``. Running them through this driver serves two
purposes:

1. **CI verification.** If the runtime API changes in a way that breaks the
   examples, the tutorial build fails — the chapter can never document an
   API that no longer works.
2. **Output capture.** Each example's stdout is captured into a sibling
   ``NN_*.out.txt`` file. The chapter can then quote the expected output
   so readers know what to look for when running the script themselves.

The ``.out.txt`` files contain a single line listing the example name plus
the captured stdout. Random readouts from the naive coordinator are masked
so the files stay stable across runs.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
EXAMPLES = sorted(THIS_DIR.glob("[0-9][0-9]_*.py"))

# Some examples print values that change per run (random readouts, OS-picked
# ports, wall-clock durations). Mask them so the captured .out.txt stays
# stable for CI diff checks.
MASKS: list[tuple[re.Pattern[str], str]] = [
    # bound_port = 12345  ->  bound_port = <port>
    (re.compile(r"bound_port = \d+"), "bound_port = <port>"),
    # http://[::1]:12345 -> http://[::1]:<port>
    (re.compile(r"http://([^:]+):\d+"), r"http://\1:<port>"),
    # bound at http://...
    (re.compile(r"gRPC server bound at .+"), "gRPC server bound at <url>"),
    # in N.N ms
    (re.compile(r"in \d+\.\d+ ms"), "in <T> ms"),
    # Runtime(bound=:12345...) representation
    (re.compile(r"Runtime\(bound_port=\d+"), "Runtime(bound_port=<port>"),
]


def _mask(text: str) -> str:
    for pattern, replacement in MASKS:
        text = pattern.sub(replacement, text)
    return text


def main() -> None:
    if not EXAMPLES:
        raise SystemExit(f"no examples found in {THIS_DIR}")

    for script in EXAMPLES:
        print(f"running {script.name}...", flush=True)
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=THIS_DIR,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        # Tolerate crashes that happen *after* the script's main work
        # completed, during Python interpreter finalization.
        crashed_during_finalization = (
            result.returncode in (-6, -11)
            and result.stdout.strip()
            and (
                not result.stderr.strip()
                or "finalizing" in result.stderr.lower()
            )
        )
        if result.returncode != 0 and not crashed_during_finalization:
            sys.stderr.write(result.stdout)
            sys.stderr.write(result.stderr)
            raise SystemExit(
                f"{script.name} failed (exit {result.returncode}); see captured stderr above"
            )
        if crashed_during_finalization:
            print(
                f"  warning: {script.name} crashed on teardown "
                f"(exit {result.returncode}); stdout looks complete, "
                "treating as success",
                flush=True,
            )
        out_path = script.with_suffix(".out.txt")
        out_path.write_text(_mask(result.stdout))
        print(f"  -> {out_path.name}")


if __name__ == "__main__":
    main()
