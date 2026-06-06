#!/usr/bin/env python3
"""Remove all generated (non-git-tracked) files from the examples/ directory.

This script uses ``git ls-files`` to determine which files are tracked and
removes everything else — bringing the examples/ tree back to a clean state
so that ``make tutorial`` can regenerate all derived artifacts from scratch.

Usage:
    python documents/tutorial/scripts/clean_examples.py [--dry-run]
"""


import argparse
import os
import shutil
import subprocess
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def get_tracked_files() -> set[Path]:
    """Return the set of git-tracked files under examples/."""
    repo_root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
            cwd=EXAMPLES_DIR,
        ).stdout.strip()
    )
    result = subprocess.run(
        ["git", "ls-files", str(EXAMPLES_DIR.relative_to(repo_root))],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root,
    )
    return {(repo_root / line).resolve() for line in result.stdout.splitlines() if line}


def clean(*, dry_run: bool = False) -> int:
    """Remove untracked files and empty directories. Returns count of removed items."""
    tracked = get_tracked_files()
    removed = 0

    for root, dirs, files in os.walk(EXAMPLES_DIR, topdown=False):
        root_path = Path(root)

        # Skip .git internals (shouldn't appear, but be safe)
        if ".git" in root_path.parts:
            continue

        for name in files:
            file_path = (root_path / name).resolve()
            if file_path not in tracked:
                if dry_run:
                    print(f"  would remove: {file_path.relative_to(EXAMPLES_DIR)}")
                else:
                    file_path.unlink()
                    print(f"  removed: {file_path.relative_to(EXAMPLES_DIR)}")
                removed += 1

        # Remove __pycache__ directories
        for name in dirs:
            dir_path = root_path / name
            if name == "__pycache__" and dir_path.is_dir():
                if dry_run:
                    print(f"  would remove: {dir_path.relative_to(EXAMPLES_DIR)}/")
                else:
                    shutil.rmtree(dir_path)
                    print(f"  removed: {dir_path.relative_to(EXAMPLES_DIR)}/")
                removed += 1

    return removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without deleting anything.",
    )
    args = parser.parse_args()

    print(f"Cleaning generated files from {EXAMPLES_DIR.relative_to(Path.cwd())}...")

    # Always preview first
    count = clean(dry_run=True)

    if count == 0:
        print("  Already clean — nothing to remove.")
        return 0

    if args.dry_run:
        print(f"\n  Would remove {count} item(s). (dry-run)")
        return 0

    answer = input(f"\nRemove {count} item(s)? [y/N] ").strip().lower()
    if answer != "y":
        print("  Aborted.")
        return 0

    clean(dry_run=False)
    print(f"\n  Removed {count} item(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
