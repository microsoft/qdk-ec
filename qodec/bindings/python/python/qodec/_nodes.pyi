from collections.abc import Mapping
from pathlib import Path
from typing import Any, TYPE_CHECKING, Protocol, TypeVar, final, overload

if TYPE_CHECKING:
    from . import ReferenceLike

_Value = TypeVar("_Value")
_Result = TypeVar("_Result", covariant=True)

class _ExpectedType(Protocol[_Result]):
    def __call__(self, *args: Any, **kwargs: Any) -> _Result: ...
    def __instancecheck__(self, instance: Any, /) -> bool: ...

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
    value() returns ordinary getter values, including live collection views.
    An optional positional type checks the result with isinstance, without conversion.
    sequence_nodes() and mapping_nodes() return child nodes instead of values.
    Their collection kind must match the target, else they raise TypeError.
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

    def resolve(self, path: ReferenceLike) -> Node:
        """Follow a path relative to this occurrence; return a root-relative node."""
        ...

    @overload
    def value(self, expected: type[_Value] | _ExpectedType[_Value]) -> _Value:
        """Return the value if ``isinstance(value, expected)``, else raise TypeError.

        The type check does not convert, copy, or inspect elements. Use runtime
        types such as tuple or collections.abc.Sequence, not parameterized types.
        """
        ...

    @overload
    def value(self, expected: type[object] = object) -> object:
        """Return the ordinary field value with its normal ownership rules.

        A selection returns a tuple of selected values, retaining order and
        duplicates. Each access follows the current model. Previously returned
        values retain the normal getter's ownership, not the node's path tracking.
        """
        ...

    def sequence_nodes(self) -> tuple[Node, ...]:
        """All sequence or selection entries in order, retaining paths and duplicates."""
        ...

    def mapping_nodes(self) -> Mapping[str, Node]:
        """Read-only snapshot of all mapping entries, keyed by literal strings.

        Total for this collection; its values are live child nodes.
        """
        ...

    def __bool__(self) -> bool:
        """Raise TypeError: inspect value() explicitly."""
        ...

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __ne__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...
