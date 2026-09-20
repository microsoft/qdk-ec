"""Typed, occurrence-aware navigation of the resolved model."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, TYPE_CHECKING, Mapping, TypeVar, final

from . import _native, actions

if TYPE_CHECKING:
    from . import Action, Gadget, Qodec, ReferenceLike

_Value = TypeVar("_Value")


@final
class SourceLocation:
    """A file and one-based line in the source revision read by the loader."""

    __slots__ = ("_path", "_line")
    __module__ = "qodec"
    _path: Path
    _line: int

    def __new__(cls) -> SourceLocation:
        raise TypeError("SourceLocation values are returned by Node.source_location")

    @classmethod
    def _create(cls, path: Path, line: int) -> SourceLocation:
        location = object.__new__(cls)
        location._path, location._line = Path(path), line
        return location

    @property
    def path(self) -> Path:
        return self._path

    @property
    def line(self) -> int:
        return self._line


@final
class Node:
    """A live path in one Qodec or standalone Gadget.

    Equality and hashing compare owner identity and canonical path, not model
    contents. Mutable objects follow the normal sharing rules of qodec getters.
    """

    __slots__ = ("_owner", "_path", "_parsed")
    __module__ = "qodec"
    _owner: Qodec | Gadget
    _path: str
    _parsed: Any

    def __new__(cls) -> Node:
        raise TypeError("Node values are returned by Qodec.resolve or Gadget.resolve")

    @classmethod
    def _create(cls, owner: Qodec | Gadget, path: ReferenceLike) -> Node:
        node = cls._at_path(owner, _native._ModelPath(path))
        node._query("exists")
        return cls._at_path(owner, node._parsed._canonical())

    @classmethod
    def _at_path(cls, owner: Qodec | Gadget, path: Any) -> Node:
        node = object.__new__(cls)
        node._owner, node._path, node._parsed = owner, str(path), path
        return node

    def _query(self, request: str) -> Any:
        return _native._node_query(self._owner, self._parsed, request)

    def value(self, expected: type[_Value]) -> _Value:
        """A scalar or model object, required to be an instance of ``expected``.

        ``node.value(Gadget)`` returns the live gadget; ``node.value(int)``
        returns an integer and rejects a bool, which is an ``int`` subclass.
        Collections require ``as_sequence()`` or ``as_mapping()``. Raises
        ``TypeError`` for a collection or when the value has another type.
        """
        value = self._query("value")
        if not isinstance(value, expected) or (expected in (int, float, bool, str) and type(value) is not expected):
            raise TypeError(f"{self.path!r} contains {type(value).__name__}, expected {expected.__name__}")
        return value

    @property
    def path(self) -> str:
        """Canonical root-relative model path; empty for the root."""
        return self._path

    @property
    def source_location(self) -> SourceLocation | None:
        """Exact loaded source location, or None when unavailable or modified."""
        from . import Qodec

        self._query("exists")
        if not isinstance(self._owner, Qodec):
            return None
        location = self._owner._node_source_location(self._path)
        return None if location is None else SourceLocation._create(*location)

    @property
    def is_none(self) -> bool:
        return bool(self._query("is_none"))

    def resolve(self, path: ReferenceLike) -> Node:
        node = Node._at_path(self._owner, self._parsed._resolve(path))
        node._query("exists")
        return Node._at_path(self._owner, node._parsed._canonical())













    def as_action(self) -> Action:
        value = self._query("value")
        if isinstance(value, (actions.Stabilize, actions.Clifford, actions.Pauli, actions.Observe, actions.Rotate)):
            return value
        raise TypeError(f"{self.path!r} does not contain an action")




    def as_sequence(self) -> tuple[Node, ...]:
        return tuple(Node._at_path(self._owner, self._parsed._index(index)) for index in range(self._query("length")))

    def as_mapping(self) -> Mapping[str, Node]:
        return MappingProxyType({
            key: Node._at_path(self._owner, self._parsed._key(key))
            for key in self._query("keys")
        })





    def __bool__(self) -> bool:
        raise TypeError("Node has no truth value; use value(bool), is_none, or a typed collection accessor")

    def __str__(self) -> str:
        return self._path

    def __repr__(self) -> str:
        try:
            kind = self._query("kind")
        except LookupError:
            kind = "missing"
        return f"Node(path={self.path!r}, type={kind!r})"

    def __eq__(self, other: object, /) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self._owner is other._owner and self._path == other._path

    def __ne__(self, other: object, /) -> bool:
        equal = self.__eq__(other)
        return NotImplemented if equal is NotImplemented else not equal

    def __hash__(self) -> int:
        return hash((id(self._owner), self._path))