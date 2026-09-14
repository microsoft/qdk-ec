"""Action values for :attr:`qodec.Instruction.action`.

Pauli inputs accept strings or :class:`qodec.codes.PauliExpression` objects;
getters return strings. List and dictionary getters return copies.
``condition`` is keyword-only and defaults to ``None`` (unconditional)
on every action that accepts it. :class:`Observe` has no condition.
"""
from ._native import (
    Clifford,
    Condition,
    Observe,
    Pauli,
    Rotate,
    Stabilize,
)

__all__ = [
    "Clifford",
    "Condition",
    "Observe",
    "Pauli",
    "Rotate",
    "Stabilize",
]
