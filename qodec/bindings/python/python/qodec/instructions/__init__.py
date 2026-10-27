"""Instruction definitions and parsed calls; actions live in :mod:`qodec.actions`."""
from enum import Enum

from .._native import (
    Block,
    BlockOperand,
    Instruction,
    InstructionCall,
    InstructionSet,
    Parameter,
)


class Kind(Enum):
    """The declared type of a classical :class:`Parameter`.

    A ``BIT`` parameter is a runtime classical input usable in action guards.
    Parameters of the other kinds, including ``BOOLEAN``, take
    compile-time literal arguments.

    Returned by :attr:`Parameter.kind`. Accepted (alongside the equivalent
    lowercase string) by the :class:`Parameter` constructor.
    """

    BIT = "bit"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    STRING = "string"
    PAULI = "pauli"


# Reached as `Parameter.Kind`; the qualname keeps that path picklable.
Kind.__qualname__ = "Parameter.Kind"
Parameter.Kind = Kind
del Kind


__all__ = [
    "Block",
    "BlockOperand",
    "Instruction",
    "InstructionCall",
    "InstructionSet",
    "Parameter",
]
