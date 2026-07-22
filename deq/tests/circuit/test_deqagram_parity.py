"""Parity harness: deqagram + shim must reproduce deq's ``model.py`` parse.

For every ``.deq`` file in the repository (skipping Mako templates, which are
rendered before parsing), this compares the ``model.py`` model produced by deq's
own parser against the one produced by parsing with deqagram and mapping through
:mod:`deq.circuit.deqagram_shim`.

The shim now covers all definition kinds, so this asserts whole-file equality at
the ``model.py`` level (``source_file`` is normalized out — it is the path,
supplied separately from parsing).
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


def _normalize(f: model.DeqFile) -> model.DeqFile:
    """Drop ``source_file`` (the path, supplied separately from parsing)."""
    f.source_file = None
    for definition in f.definitions:
        if hasattr(definition, "source_file"):
            definition.source_file = None
    return f


@pytest.mark.parametrize(
    "path", _deq_files(), ids=lambda p: str(p.relative_to(_REPO_ROOT))
)
def test_file_matches_deq(path: Path) -> None:
    text = path.read_text()

    try:
        expected = deq_parser.parse(text)
    except SyntaxError:
        # deq's own parser rejects this file (e.g. duplicate definition names);
        # there is no model to compare against.
        pytest.skip("deq parser rejects this file")

    actual = shim.parse(text)

    assert _normalize(actual) == _normalize(expected)
