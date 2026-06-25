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
# Matches a Markdown link whose href ends with .deq
DEQ_LINK_RE = re.compile(r"\[(?P<text>[^\]]*)\]\((?P<path>[^)]+\.deq)\)")

BEGIN_COMMENT = "<!-- deq-highlight-begin: {} -->"
END_COMMENT = "<!-- deq-highlight-end: {} -->"

# Matches an entire existing highlight block (greedy within the two comments)
BLOCK_RE_TEMPLATE = (
    r"<!-- deq-highlight-begin: {} -->\n" r".*?" r"<!-- deq-highlight-end: {} -->\n?"
)


def _block_re(deq_path: str) -> re.Pattern[str]:
    escaped = re.escape(deq_path)
    return re.compile(BLOCK_RE_TEMPLATE.format(escaped, escaped), re.DOTALL)


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
    content = original

    # 1. Find all .deq links in the file (collect first, then mutate)
    links: list[tuple[str, str]] = []  # (link_text, deq_rel_path)
    for m in DEQ_LINK_RE.finditer(content):
        links.append((m.group("text"), m.group("path")))

    if not links:
        return False

    # 2. Remove all existing highlight blocks (so we can regenerate)
    seen_paths: set[str] = set()
    for _, deq_rel in links:
        if deq_rel in seen_paths:
            continue
        seen_paths.add(deq_rel)
        content = _block_re(deq_rel).sub("", content)

    # 3. For each link, generate highlighted HTML and insert after the link line
    #    We process from bottom to top so that insertions don't shift later positions.
    for m in reversed(list(DEQ_LINK_RE.finditer(content))):
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

        # Insert the block on the line after the link
        line_end = (
            content.index("\n", m.end()) if "\n" in content[m.end() :] else len(content)
        )
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
