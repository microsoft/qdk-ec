"""Shared utilities and gate classifications for noise injection.

Noise is injected at the text level (line-by-line) so that comments,
blank lines, and indentation are preserved exactly.
"""

import re

from deq.transpiler.stim_constants import (
    ANNOTATION_INSTRUCTIONS,
    NOISE_INSTRUCTIONS_ALL,
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


_CLASSICAL_REF_RE = re.compile(r"^!?(?:rec\[|sweep\[)")


def split_two_qubit_gate_targets(
    after: str,
) -> tuple[str | None, str | None]:
    """For a 2-qubit gate, split targets into qubit-qubit pairs and singles.

    Two-qubit gate targets are consumed in sequential ``(control, target)``
    pairs.  When the control is a classical reference (``rec[-N]`` or
    ``sweep[N]``), the gate is classically-controlled: only the target qubit
    is subject to noise, and only single-qubit noise is appropriate.

    Returns ``(qubit_pairs_str, single_qubits_str)`` where:

    * *qubit_pairs_str* — space-separated qubit indices suitable for
      ``DEPOLARIZE2`` noise, or ``None`` if there are no qubit-qubit pairs.
    * *single_qubits_str* — space-separated qubit indices suitable for
      ``DEPOLARIZE1`` noise (their partner was a classical reference), or
      ``None`` if none.
    """
    s = strip_comment(after)
    s = re.sub(r"\([^)]*\)", "", s)  # remove parenthesized arguments
    raw_tokens = s.split()

    def _process(tok: str) -> tuple[str, bool]:
        """Return ``(token, is_classical_ref)``."""
        if _CLASSICAL_REF_RE.match(tok):
            return tok, True
        return re.sub(r"\[[^\]]*\]", "", tok), False

    processed = [_process(t) for t in raw_tokens if t]

    qubit_pairs: list[str] = []
    single_qubits: list[str] = []

    for i in range(0, len(processed) - 1, 2):
        (tok_a, a_is_ref), (tok_b, b_is_ref) = processed[i], processed[i + 1]
        if not a_is_ref and not b_is_ref:
            qubit_pairs.extend([tok_a, tok_b])
        elif a_is_ref and not b_is_ref:
            # Classical control (e.g. rec[-1]) drives a qubit target.
            single_qubits.append(tok_b)
        elif not a_is_ref and b_is_ref:
            # Qubit drives a classical target (unusual, but handled gracefully).
            single_qubits.append(tok_a)
        # both classical refs → no qubit noise

    return (
        " ".join(qubit_pairs) if qubit_pairs else None,
        " ".join(single_qubits) if single_qubits else None,
    )


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
    | NOISE_INSTRUCTIONS_ALL
    | ANNOTATION_INSTRUCTIONS
    | DEQ_KEYWORDS
)
