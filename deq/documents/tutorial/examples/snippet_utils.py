"""Shared utilities for tutorial generator scripts."""

import os
import subprocess
import sys


def write_snippet(path: str, content: str) -> None:
    """Write a snippet file and print its name."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"    -> {os.path.basename(path)}")


def run_cli(
    description: str,
    args: list[str],
    *,
    allow_failure: bool = False,
    cwd: str | None = None,
) -> tuple[int, str, str]:
    """Run ``python -m deq <args>`` and return ``(returncode, stdout, stderr)``.

    Args:
        description: Human-readable label printed before running.
        args: The deq subcommand and its options (without ``python -m deq``).
        allow_failure: If True, a non-zero exit is returned instead of raising;
            the caller inspects ``returncode`` to decide what to do.
        cwd: Working directory for the subprocess.  If None, inherits the
            parent process's cwd.

    Raises:
        RuntimeError: If the command exits non-zero and ``allow_failure`` is
            False.  The captured stderr is written to ``sys.stderr`` first.
    """
    print(f"  {description}...")
    result = subprocess.run(
        [sys.executable, "-m", "deq"] + args,
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
    )
    if result.returncode != 0 and not allow_failure:
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"command failed: {' '.join(args)}")
    return result.returncode, result.stdout, result.stderr


def extract_block(text: str, keyword: str, name: str) -> str:
    """Extract a top-level block (CODE/GADGET/PROGRAM) by keyword and name.

    Includes any decorator lines (e.g. ``@CHECKS("manual")``) immediately
    preceding the block header.

    Args:
        text: Full source text to search.
        keyword: Block keyword, e.g. ``"CODE"``, ``"GADGET"``, ``"PROGRAM"``.
        name: Block name, e.g. ``"Idle"``, ``"RepetitionCode"``.

    Returns:
        The extracted block as a string (including trailing newline).

    Raises:
        ValueError: If the block is not found.
    """
    lines = text.splitlines()
    result: list[str] = []
    decorator_buffer: list[str] = []
    capturing = False
    brace_depth = 0

    for line in lines:
        stripped = line.strip()

        if not capturing:
            if stripped.startswith("@"):
                decorator_buffer.append(line)
                continue
            if stripped.startswith(keyword) and name in stripped:
                result.extend(decorator_buffer)
                result.append(line)
                capturing = True
                brace_depth = line.count("{") - line.count("}")
                decorator_buffer = []
                if brace_depth <= 0:
                    break
                continue
            decorator_buffer = []
        else:
            result.append(line)
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                break

    if not result:
        raise ValueError(f"Block not found: {keyword} {name}")

    return "\n".join(result) + "\n"
