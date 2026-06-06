"""Shared utilities and gate classifications for noise injection.

Noise is injected at the text level (line-by-line) so that comments,
blank lines, and indentation are preserved exactly.
"""

import re

from deq.transpiler.stim_constants import (
    ANNOTATION_INSTRUCTIONS,
    NOISE_INSTRUCTIONS,
    ONE_QUBIT_GATES,
    PAIR_MEASURE_GATES,
    PAULI_PRODUCT_GATES,
    TWO_QUBIT_GATES,
    X_BASIS_MEASURE,
    X_BASIS_RESET,
    Y_BASIS_MEASURE,
    Y_BASIS_RESET,
    Z_BASIS_MEASURE,
    Z_BASIS_RESET,
)

# ── Line parsing ─────────────────────────────────────────────────────

INSTRUCTION_RE = re.compile(
    r"^(?P<indent>\s*)" r"(?P<name>[A-Za-z][A-Za-z0-9_]*)" r"(?P<after>.*?)$",
)


def strip_comment(text: str) -> str:
    """Remove a trailing ``# ...`` comment, preserving everything before it."""
    idx = text.find("#")
    return text[:idx] if idx >= 0 else text


def extract_targets(after: str) -> str | None:
    """Return the qubit-target portion of an instruction line.

    Strips the optional ``[tag]`` and ``(args)`` leaving only the
    space-separated targets.  Returns *None* when no targets remain.
    """
    s = strip_comment(after)
    s = re.sub(r"\[[^\]]*\]", "", s)  # remove tag
    s = re.sub(r"\([^)]*\)", "", s)  # remove parenthesized arguments
    s = s.strip()
    return s if s else None


def strip_parens(text: str) -> str:
    """Remove the first parenthesized group from *text*."""
    return re.sub(r"\([^)]*\)", "", text, count=1)


def fmt_prob(p: float) -> str:
    """Format a probability as a compact string (e.g. ``0.001``)."""
    if p == int(p):
        return str(int(p))
    return f"{p:g}"


# ── Block-context tracking ───────────────────────────────────────────
#
# Noise is only injected inside GADGET blocks (including REPEAT sub-blocks
# within a GADGET).  COMPOSE, PROGRAM, and CODE blocks are left untouched
# because their instructions operate at the logical level.

BLOCK_OPEN_RE = re.compile(
    r"^\s*(?:@\w+(?:\([^)]*\))?\s*)*"  # optional decorators
    r"(?P<kind>GADGET|COMPOSE|PROGRAM|CODE|REPEAT)\b",
)


def count_braces(line: str) -> tuple[int, int]:
    """Count ``{`` and ``}`` in *line* after stripping comments."""
    s = strip_comment(line)
    return s.count("{"), s.count("}")


# ── DEQ-specific keywords ──────────────────────────────────────────

DEQ_KEYWORDS = frozenset(
    {
        "INPUT",
        "OUTPUT",
        "READOUT",
        "CHECK",
        "CONDITIONAL",
        "VIRTUAL",
        "REPEAT",
        "PRESELECT",
    }
)

# The union of all known instruction names inside a GADGET block.
ALL_KNOWN_GADGET_INSTRUCTIONS = (
    ONE_QUBIT_GATES
    | TWO_QUBIT_GATES
    | Z_BASIS_MEASURE
    | X_BASIS_MEASURE
    | Y_BASIS_MEASURE
    | Z_BASIS_RESET
    | X_BASIS_RESET
    | Y_BASIS_RESET
    | PAIR_MEASURE_GATES
    | PAULI_PRODUCT_GATES
    | NOISE_INSTRUCTIONS
    | ANNOTATION_INSTRUCTIONS
    | DEQ_KEYWORDS
)
