#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import yaml

DEFAULT_MAX_ITEMS = 8
DEFAULT_MAX_LINE = 110


class FlowList(list):
    """Marker list that should be serialized in YAML flow style."""


def _flow_list_representer(dumper: yaml.SafeDumper, data: FlowList):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", list(data), flow_style=True)


yaml.SafeDumper.add_representer(FlowList, _flow_list_representer)


def should_flow_list(values: list[object], max_items: int, max_line: int) -> bool:
    if len(values) == 0:
        return True
    if len(values) > max_items:
        return False

    for value in values:
        if isinstance(value, dict):
            if len(value) != 1:
                return False
            inner = next(iter(value.values()))
            if isinstance(inner, (dict, list)):
                return False
        elif isinstance(value, list):
            return False

    rendered = yaml.safe_dump(values, default_flow_style=True, width=10_000).strip()
    return len(rendered) <= max_line


def to_preferred_style(node: object, max_items: int, max_line: int) -> object:
    if isinstance(node, dict):
        return {k: to_preferred_style(v, max_items, max_line) for k, v in node.items()}
    if isinstance(node, list):
        converted = [to_preferred_style(v, max_items, max_line) for v in node]
        if should_flow_list(converted, max_items, max_line):
            return FlowList(converted)
        return converted
    return node


def process_file(path: Path, max_items: int, max_line: int, write: bool) -> bool:
    original = path.read_text(encoding="utf-8")
    data = yaml.safe_load(original)
    if data is None:
        return False

    converted = to_preferred_style(data, max_items, max_line)
    output = yaml.safe_dump(
        converted,
        sort_keys=False,
        allow_unicode=False,
        width=max_line,
    )

    changed = output != original
    if changed and write:
        path.write_text(output, encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prefer compact flow-style YAML lists for qodec gadget files.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["examples", "tests"],
        help="Files/directories to scan (defaults: examples tests)",
    )
    parser.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS)
    parser.add_argument("--max-line", type=int, default=DEFAULT_MAX_LINE)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only report files that would change, without writing.",
    )
    args = parser.parse_args()

    files: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_file() and path.name.endswith(".gadget.yaml"):
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.gadget.yaml")))

    changed_files: list[Path] = []
    for file_path in files:
        changed = process_file(
            file_path,
            max_items=args.max_items,
            max_line=args.max_line,
            write=not args.check,
        )
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
