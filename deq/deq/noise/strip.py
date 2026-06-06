"""Strip all noise instructions from a DEQ circuit."""

import re

from .common import INSTRUCTION_RE
from deq.transpiler.stim_constants import (
    MEASUREMENT_INSTRUCTIONS,
    NOISE_INSTRUCTIONS,
)

_ALL_MEASURES = MEASUREMENT_INSTRUCTIONS | {"MPP"}

_PARENS_RE = re.compile(r"\([^)]*\)")


def strip_noise(text: str) -> str:
    """Remove all noise instructions from a DEQ circuit, preserving everything else.

    Lines whose instruction name matches the ``NOISE_INSTRUCTIONS`` set
    (``DEPOLARIZE1``, ``X_ERROR``, ``Z_ERROR``, etc.) are dropped.

    Noisy measurements like ``M(0.01) 0 1 2`` have their probability
    argument stripped, becoming ``M 0 1 2``.

    All other lines — gates, annotations, comments, blank lines,
    structural keywords — are kept verbatim.
    """
    result: list[str] = []
    for line in text.splitlines(keepends=True):
        m = INSTRUCTION_RE.match(line)
        if not m:
            result.append(line)
            continue
        name_upper = m.group("name").upper()
        if name_upper in NOISE_INSTRUCTIONS:
            continue
        if name_upper in _ALL_MEASURES and "(" in m.group("after"):
            cleaned = _PARENS_RE.sub("", line, count=1)
            result.append(cleaned)
            continue
        result.append(line)
    return "".join(result)
