#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

DEFAULT_WIDTH = 110


class IsaDumper(yaml.SafeDumper):
    pass


def _str_presenter(dumper: yaml.SafeDumper, data: str):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


IsaDumper.add_representer(str, _str_presenter)


def process_file(path: Path, width: int, write: bool) -> bool:
    original = path.read_text(encoding="utf-8")
    data = yaml.safe_load(original)
    if data is None:
        return False

    output = yaml.dump(
        data,
        Dumper=IsaDumper,
        sort_keys=False,
        allow_unicode=True,
        width=width,
    )

    changed = output != original
    if changed and write:
        path.write_text(output, encoding="utf-8")
    return changed


def collect_isa_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.name.endswith(".isa.yaml"):
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.isa.yaml")))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize qodec ISA YAML files with readable multiline scalars.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["examples"],
        help="Files/directories to scan (default: examples)",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only report files that would change, without writing.",
    )
    args = parser.parse_args()

    files = collect_isa_files(args.paths)
    changed_files: list[Path] = []

    for file_path in files:
        changed = process_file(file_path, width=args.width, write=not args.check)
        if changed:
            changed_files.append(file_path)

    for file_path in changed_files:
        print(file_path)

    if args.check:
        print(f"{len(changed_files)} files would change")
        return 1 if changed_files else 0

    print(f"{len(changed_files)} files changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
