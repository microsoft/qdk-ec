"""Parity harness: deqagram + shim must reproduce deq's ``model.py`` parse.

For every ``.deq`` file in the repository (skipping Mako templates, which are
rendered before parsing), this compares the ``model.py`` model produced by deq's
own parser against the one produced by parsing with deqagram and mapping through
:mod:`deq.circuit.deqagram_shim`.

The shim is an incremental port. Right now it maps ``CODE`` definitions, so this
harness compares ``CODE`` definitions extracted from both parses. As the shim
grows to cover the other definition kinds, this will tighten to whole-file
equality.
"""

from __future__ import annotations

import re
from pathlib import Path

import deqagram
import pytest

from deq.circuit import deqagram_shim as shim
from deq.circuit import model
from deq.circuit import parser as deq_parser

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAKO = re.compile(r"<%|\$\{")


def _deq_files() -> list[Path]:
    """All non-Mako ``.deq`` files under the repo, excluding the venv."""
    files = []
    for path in sorted(_REPO_ROOT.rglob("*.deq")):
        if ".venv" in path.parts:
            continue
        if _MAKO.search(path.read_text()):
            continue
        files.append(path)
    return files


def _codes_by_name(f: model.DeqFile) -> dict[str, model.CodeDefinition]:
    """Extract CODE definitions from a model DeqFile, keyed by name.

    ``source_file``/``source_line`` are diagnostic metadata derived from source
    spans, which the deqagram bindings do not expose; normalize them out so the
    comparison is over semantic content.
    """
    codes = {}
    for definition in f.definitions:
        if isinstance(definition, model.CodeDefinition):
            definition.source_file = None
            definition.source_line = None
            codes[definition.name] = definition
    return codes


@pytest.mark.parametrize(
    "path", _deq_files(), ids=lambda p: str(p.relative_to(_REPO_ROOT))
)
def test_code_definitions_match_deq(path: Path) -> None:
    text = path.read_text()

    try:
        parsed = deq_parser.parse(text)
    except SyntaxError:
        # deq's own parser rejects this file (e.g. duplicate definition names);
        # there is no model to compare against.
        pytest.skip("deq parser rejects this file")

    deq_codes = _codes_by_name(parsed)
    if not deq_codes:
        pytest.skip("no CODE definitions in this file")

    # Map only the CODE definitions from the deqagram parse via the shim.
    attached = deqagram.parse_attached(text)
    shim_codes = {}
    for definition in attached.definitions:
        if isinstance(definition, deqagram.AttachedDefinition.Code):
            code = shim._code_definition(definition.code)
            shim_codes[code.name] = code

    assert shim_codes.keys() == deq_codes.keys()
    for name, deq_code in deq_codes.items():
        assert shim_codes[name] == deq_code, f"CODE {name} differs"
