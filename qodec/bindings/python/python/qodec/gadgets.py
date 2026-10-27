"""Circuits, encodings, and parity equations for :class:`qodec.Gadget`.

Use :class:`qodec.Reference` for path syntax, :class:`Circuit` for parsed calls
and their output bits, and :class:`Readout` for gadget output equations.
"""
from collections.abc import Mapping, Sequence
from typing import Literal, Union
from . import Reference as _Reference, ReferenceLike as _ReferenceLike

from ._native import (
    Circuit,
    Encoding,
    Flag,
    Gadget,
    Outcome,
    Readout,
)

#: Runtime alias for an immutable parity equation returned by :attr:`qodec.Gadget.checks`.
Check = tuple[Union[_Reference, Literal[0, 1]], ...]

#: Runtime alias for an input readout: a returned :class:`Readout`, a parity
#: sequence, or a single-key ``{name: equation}`` mapping.
ReadoutLike = Union[Readout, Sequence[Union[_ReferenceLike, Literal[0, 1]]], Mapping[str, Sequence[Union[_ReferenceLike, Literal[0, 1]]]]]


__all__ = [
    "Check",
    "Circuit",
    "Encoding",
    "Flag",
    "Gadget",
    "Outcome",
    "Readout",
    "ReadoutLike",
]
