#!/usr/bin/env python3
"""Embed syntax-highlighted .deq code blocks into tutorial Markdown files.

Scans every ``.md`` file under ``documents/tutorial/`` for Markdown links
whose target ends in ``.deq``::

    [description of the code](path/to/file.deq)

For each such link the script:

1. Reads the referenced ``.deq`` file (path resolved relative to the
   ``.md`` file that contains the link).
2. Generates syntax-highlighted HTML via Shiki (Node.js subprocess) using
   the VS Code *Light+* TextMate theme and the project's own
   ``deq.tmLanguage.json`` grammar.
3. Inserts (or replaces) a fenced HTML block *immediately after* the link
   line, delimited by recognisable HTML comments::

       <!-- deq-highlight-begin: path/to/file.deq -->
       <pre class="shiki light-plus" ...>...</pre>
       <!-- deq-highlight-end: path/to/file.deq -->

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
# Matches a Markdown link whose href ends with .deq, but ONLY when the
# link is the entire content of its line (optionally surrounded by
# whitespace). Inline links embedded mid-sentence are intentionally
# skipped, otherwise injecting a highlighted code block on the next
# line would split a paragraph into nonsense.
DEQ_LINK_RE = re.compile(
    r"^[ \t]*\[(?P<text>[^\]]*)\]\((?P<path>[^)]+\.deq)\)[ \t]*$",
    re.MULTILINE,
)

# Matches any highlight block, regardless of which .deq file it points
# at. Used to strip the entire previous run's output before re-injecting,
# so blocks that no longer have a matching own-line link get cleaned up
# instead of becoming orphans.
ANY_BLOCK_RE = re.compile(
    r"<!-- deq-highlight-begin: [^>]+? -->\n.*?<!-- deq-highlight-end: [^>]+? -->\n?",
    re.DOTALL,
)

BEGIN_COMMENT = "<!-- deq-highlight-begin: {} -->"
END_COMMENT = "<!-- deq-highlight-end: {} -->"


def highlight_deq(deq_file: Path) -> str:
    """Return Shiki-highlighted HTML for *deq_file* (Light+ theme)."""
    result = subprocess.run(
        ["node", str(HIGHLIGHT_SCRIPT), str(deq_file), "--theme", "light"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def process_markdown(md_path: Path, *, check_only: bool = False) -> bool:
    """Process a single Markdown file.  Returns True if the file was changed."""
    original = md_path.read_text(encoding="utf-8")

    # 1. Strip every existing highlight block, regardless of which .deq
    # path it points at. This guarantees orphan blocks (those whose link
    # was deleted, edited inline, or otherwise no longer qualifies) get
    # cleaned up on every run.
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
        deq_abs = (md_path.parent / deq_rel).resolve()

        if not deq_abs.is_file():
            raise FileNotFoundError(
                f"{deq_rel} referenced in {md_path.name} does not exist: {deq_abs}"
            )

        html = highlight_deq(deq_abs)
        block = (
            BEGIN_COMMENT.format(deq_rel)
            + "\n"
            + html
            + "\n"
            + END_COMMENT.format(deq_rel)
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
