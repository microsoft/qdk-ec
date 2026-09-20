from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, final

if TYPE_CHECKING:
    from . import Action, Code, Gadget, Instruction, InstructionSet, Layer, Qodec, Reference, ReferenceLike
    from .actions import Condition
    from .gadgets import Circuit, Encoding, Readout
    from .instructions import Block, BlockOperand, Parameter

_Value = TypeVar("_Value")

@final
class SourceLocation:
    """A point in the actual source file read by the loader.

    Output-only. Bundle locations name the outer bundle, not an internal key.
    Locations refer to the loaded revision; later filesystem edits are not tracked.
    """
    @property
    def path(self) -> Path: ...
    @property
    def line(self) -> int:
        """One-based line in the loaded source file."""
        ...

@final
class Node:
    """A live model path returned by Qodec.resolve or Gadget.resolve.

    Each access follows the current model and raises LookupError if the path
    disappeared. Reordering a sequence may change its target. Equality and hashing
    compare owner identity and canonical path, not contents or source locations.
    value(expected), as_action(), as_sequence(), and as_mapping() require matching
    types and raise TypeError on mismatch.
    Mutable objects retain the sharing rules of their normal qodec getters.
    """
    @property
    def path(self) -> str:
        """Canonical root-relative model path. The empty string names the root."""
        ...
    @property
    def source_location(self) -> SourceLocation | None:
        """Exact source position, with no implicit parent fallback.

        None for standalone gadgets, constructed models, slices, unlocated fields, or a model differing
        from its loaded snapshot. Changing the file on disk is not detected.
        """
        ...
    @property
    def is_none(self) -> bool: ...
    def resolve(self, path: ReferenceLike) -> Node:
        """Follow a path relative to this occurrence; return a root-relative node."""
        ...
    def value(self, expected: type[_Value]) -> _Value:
        """A scalar or model object, required to be an instance of ``expected``.

        ``node.value(Gadget)`` returns the live gadget; ``node.value(int)``
        returns an integer and rejects a bool, which is an ``int`` subclass.
        Collections require ``as_sequence()`` or ``as_mapping()``. Raises
        ``TypeError`` for a collection or when the value has another type.
        """
        ...
    def as_action(self) -> Action: ...
    def as_sequence(self) -> tuple[Node, ...]:
        """All sequence or selection entries in order, retaining paths and duplicates."""
        ...
    def as_mapping(self) -> Mapping[str, Node]:
        """Read-only snapshot of all mapping entries, keyed by literal strings.

        Total for this collection; its values are live child nodes.
        """
        ...
    def __bool__(self) -> bool:
        """Raise TypeError: test the stored value explicitly."""
        ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __ne__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...