#!/usr/bin/env python3
"""Embed syntax-highlighted .deq code blocks into tutorial Markdown files.

Scans every ``.md`` file under ``documents/tutorial/`` for Markdown links
whose target ends in ``.deq``, optionally followed by a GitHub-style
line-range fragment::

    [caption for the whole file](path/to/file.deq)
    [caption for a slice](path/to/file.deq#L20-L30)
    [caption for one line](path/to/file.deq#L42)

For each such link the script:

1. Reads the referenced ``.deq`` file (path resolved relative to the
   ``.md`` file that contains the link) and, if a range fragment is
   present, slices the file down to that 1-indexed inclusive range
   before highlighting.
2. Generates syntax-highlighted HTML via Shiki (Node.js subprocess) using
   the VS Code *Light+* TextMate theme and the project's own
   ``deq.tmLanguage.json`` grammar.
3. Inserts (or replaces) a fenced HTML block *immediately after* the link
   line, delimited by recognisable HTML comments that include the range
   fragment (if any) so multiple snippets from the same file coexist::

       <!-- deq-highlight-begin: path/to/file.deq#L20-L30 -->
       <pre class="shiki light-plus" ...>...</pre>
       <!-- deq-highlight-end: path/to/file.deq#L20-L30 -->

The delimiters allow the script to be re-run idempotently: stale blocks
are removed and regenerated every time.

Intended to be called from the Makefile / CI pipeline so that reviewers
can verify that committed Markdown always contains up-to-date highlighted
code.  The pipeline step should run this script and then assert
``git diff --exit-code`` to detect uncommitted drift.
"""


import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


# ── paths ──────────────────────────────────────────────────────────────
TUTORIAL_DIR = Path(__file__).resolve().parent.parent  # documents/tutorial
REPO_ROOT = TUTORIAL_DIR.parent.parent  # repo root
GRAMMAR_PATH = (
    REPO_ROOT / "deq" / "circuit" / "vscode-deq" / "syntaxes" / "deq.tmLanguage.json"
)
HIGHLIGHT_SCRIPT = TUTORIAL_DIR / "scripts" / "highlight-deq.mjs"

# ── regex patterns ─────────────────────────────────────────────────────
# Matches a Markdown link whose href ends with .deq (optionally followed
# by a GitHub-style ``#L<start>[-L<end>]`` line-range fragment), but ONLY
# when the link is the entire content of its line (optionally surrounded
# by whitespace).  Inline links embedded mid-sentence are intentionally
# skipped, otherwise injecting a highlighted code block on the next line
# would split a paragraph into nonsense.
DEQ_LINK_RE = re.compile(
    r"^[ \t]*\[(?P<text>[^\]]*)\]"
    r"\((?P<path>[^)#]+\.deq)"
    r"(?P<frag>#L\d+(?:-L\d+)?)?\)"
    r"[ \t]*$",
    re.MULTILINE,
)

# Parses a validated fragment like ``#L20-L30`` or ``#L42`` into (start,
# end) 1-indexed inclusive line numbers.  A single-line fragment collapses
# to (n, n).
FRAGMENT_RE = re.compile(r"^#L(?P<start>\d+)(?:-L(?P<end>\d+))?$")

# Matches any highlight block, regardless of which .deq file (or slice)
# it points at.  Used to strip the entire previous run's output before
# re-injecting, so blocks that no longer have a matching own-line link
# get cleaned up instead of becoming orphans.
ANY_BLOCK_RE = re.compile(
    r"<!-- deq-highlight-begin: [^>]+? -->\n.*?<!-- deq-highlight-end: [^>]+? -->\n?",
    re.DOTALL,
)

BEGIN_COMMENT = "<!-- deq-highlight-begin: {} -->"
END_COMMENT = "<!-- deq-highlight-end: {} -->"


def _parse_fragment(frag: str | None) -> tuple[int | None, int | None]:
    """Return (start, end) 1-indexed inclusive line numbers, or (None, None)."""
    if not frag:
        return (None, None)
    m = FRAGMENT_RE.match(frag)
    if not m:
        raise ValueError(f"Invalid .deq line-range fragment: {frag!r}")
    start = int(m.group("start"))
    end = int(m.group("end")) if m.group("end") else start
    if start < 1 or end < start:
        raise ValueError(f"Invalid .deq line-range fragment: {frag!r}")
    return (start, end)


def highlight_deq(
    deq_file: Path,
    *,
    start: int | None = None,
    end: int | None = None,
) -> str:
    """Return Shiki-highlighted HTML for *deq_file* (Light+ theme).

    When *start* and/or *end* are given, only that 1-indexed inclusive
    line range is highlighted.
    """
    cmd: list[str] = [
        "node",
        str(HIGHLIGHT_SCRIPT),
        str(deq_file),
        "--theme",
        "light",
    ]
    if start is not None:
        cmd.extend(["--start", str(start)])
    if end is not None:
        cmd.extend(["--end", str(end)])
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout


def process_markdown(md_path: Path, *, check_only: bool = False) -> bool:
    """Process a single Markdown file.  Returns True if the file was changed."""
    original = md_path.read_text(encoding="utf-8")

    # 1. Strip every existing highlight block, regardless of which .deq
    # path (or slice) it points at. This guarantees orphan blocks (those
    # whose link was deleted, edited inline, or otherwise no longer
    # qualifies) get cleaned up on every run.
    content = ANY_BLOCK_RE.sub("", original)

    # 2. Find all .deq links that occupy their own line. ``DEQ_LINK_RE``
    # is anchored with MULTILINE so links embedded mid-sentence don't
    # match, and we won't inject highlight blocks that would split the
    # surrounding paragraph.
    links = list(DEQ_LINK_RE.finditer(content))

    # 3. Insert blocks after each link line, bottom-up so earlier match
    # positions stay valid as we mutate later in the string.
    for m in reversed(links):
        deq_rel = m.group("path")
        frag = m.group("frag") or ""
        start, end = _parse_fragment(frag or None)
        deq_abs = (md_path.parent / deq_rel).resolve()

        if not deq_abs.is_file():
            raise FileNotFoundError(
                f"{deq_rel} referenced in {md_path.name} does not exist: {deq_abs}"
            )

        html = highlight_deq(deq_abs, start=start, end=end)
        marker = deq_rel + frag
        block = (
            BEGIN_COMMENT.format(marker)
            + "\n"
            + html
            + "\n"
            + END_COMMENT.format(marker)
            + "\n"
        )

        # Insert the block on the line after the link.
        if "\n" in content[m.end():]:
            line_end = content.index("\n", m.end())
        else:
            line_end = len(content) - 1  # link is on the last line
        content = content[: line_end + 1] + block + content[line_end + 1 :]

    changed = content != original
    if changed and not check_only:
        md_path.write_text(content, encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with code 1 if any file would be changed (for CI).",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Markdown files to process.  Defaults to all .md files under documents/tutorial/.",
    )
    args = parser.parse_args()

    md_files: list[Path] = args.files or sorted(
        p
        for p in TUTORIAL_DIR.rglob("*.md")
        if p.name != "WRITING-TUTORIAL-CHAPTERS.md"
    )
    any_changed = False

    for md in md_files:
        changed = process_markdown(md, check_only=args.check)
        if changed:
            any_changed = True
            verb = "would change" if args.check else "updated"
            print(f"  {verb}: {md.relative_to(REPO_ROOT)}")

    if args.check and any_changed:
        print(
            "\nERROR: Highlighted .deq blocks are out of date.\n"
            "Run 'make tutorial' and commit the changes.",
            file=sys.stderr,
        )
        return 1

    if not any_changed:
        print("  All .deq highlights are up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
