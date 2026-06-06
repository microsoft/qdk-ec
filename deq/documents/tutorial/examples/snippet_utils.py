"""Shared utilities for tutorial generator scripts."""

import os


def write_snippet(path: str, content: str) -> None:
    """Write a snippet file and print its name."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"    -> {os.path.basename(path)}")


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
