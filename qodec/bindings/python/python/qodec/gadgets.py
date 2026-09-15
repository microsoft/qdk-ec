"""Circuits, encodings, and parity references for :class:`qodec.Gadget`.

Use :class:`Reference` for path syntax, :class:`Circuit` for parsed calls
and their output bits, and :class:`Readout` for gadget output equations.
"""
from collections.abc import Mapping, Sequence
from typing import Literal, Union

from ._native import (
    Circuit,
    Encoding,
    Flag,
    Gadget,
    Outcome,
    Readout,
    Reference,
)


#: Runtime alias for an input reference: a path string or :class:`Reference`.
ReferenceLike = Union[Reference, str]

#: Runtime alias for an immutable parity equation returned by :attr:`qodec.Gadget.checks`.
Check = tuple[Union[Reference, Literal[0, 1]], ...]

#: Runtime alias for an input readout: a returned :class:`Readout`, a parity
#: sequence, or a single-key ``{name: equation}`` mapping.
ReadoutLike = Union[Readout, Sequence[Union[ReferenceLike, Literal[0, 1]]], Mapping[str, Sequence[Union[ReferenceLike, Literal[0, 1]]]]]


__all__ = [
    "Check",
    "Circuit",
    "Encoding",
    "Flag",
    "Gadget",
    "Outcome",
    "Readout",
    "ReadoutLike",
    "Reference",
    "ReferenceLike",
]
