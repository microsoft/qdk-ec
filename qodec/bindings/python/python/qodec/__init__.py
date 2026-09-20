"""Build, inspect, and save quantum error-correction protocols.

``Qodec`` and ``Layer`` organize a protocol. ``Code``, ``InstructionSet``,
``Instruction`` and ``Gadget`` describe its artifacts. ``Reference`` and
``Node`` address their declarations. The remaining types
live in ``qodec.codes``, ``qodec.gadgets``, ``qodec.instructions`` and
``qodec.actions``. See each accessor for which objects are shared or copied.
"""
from ._native import (
    Code,
    Qodec,
    QodecLoadError,
    QodecSaveError,
    Gadget,
    Instruction,
    InstructionSet,
    Layer,
    QodecError,
    Reference,
    register,
)

ReferenceLike = Reference | str

from . import _reference
from . import actions, codes, gadgets, instructions
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from ._nodes import Node, SourceLocation
from . import _parsers

try:
    __version__ = _pkg_version("qodec")
except PackageNotFoundError:  # pragma: no cover - running from a source checkout
    # A sentinel, not a guess: a hardcoded real-looking version here is a second
    # source of truth, and it silently drifted from the crate version before.
    __version__ = "0+unknown"
del _pkg_version, PackageNotFoundError


QodecError.__module__ = "qodec"
QodecLoadError.__module__ = "qodec"
QodecSaveError.__module__ = "qodec"


__all__ = [
    # Version
    "__version__",
    "register",
    # Submodules
    "actions",
    "codes",
    "gadgets",
    "instructions",
    # Top-level container types
    "Qodec",
    "Layer",
    "Node",
    "SourceLocation",
    "Reference",
    "ReferenceLike",
    # Top-level code types
    "Code",
    # Top-level instruction set types
    "InstructionSet",
    "Instruction",
    # Top-level gadget types
    "Gadget",
    # Exception hierarchy
    "QodecError",
    "QodecLoadError",
    "QodecSaveError",
]
