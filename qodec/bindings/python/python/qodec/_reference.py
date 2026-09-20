"""Immutable structural values exposed only through Reference."""

from dataclasses import dataclass
from typing import final

from . import Reference


def _position(value: int) -> None:
    if type(value) is not int:
        raise TypeError("sequence positions must be integers, not booleans")
    if value < 0:
        raise ValueError("sequence positions must be nonnegative")


@final
@dataclass(frozen=True, slots=True)
class Field:
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("field name must be a string")
        if not self.name.isascii() or not self.name.isidentifier():
            raise ValueError("field name must be an ASCII identifier")


@final
@dataclass(frozen=True, slots=True)
class Key:
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise TypeError("mapping key must be a string")


@final
@dataclass(frozen=True, slots=True)
class Index:
    value: int

    def __post_init__(self) -> None:
        _position(self.value)


@final
@dataclass(frozen=True, slots=True)
class Slice:
    start: int
    stop: int
    step: int = 1

    def __post_init__(self) -> None:
        for value in (self.start, self.stop, self.step):
            _position(value)
        Reference(f"[{self.start}:{self.stop}:{self.step}]")


@final
@dataclass(frozen=True, slots=True)
class Union:
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.indices, tuple):
            raise TypeError("union indices must be an immutable tuple")
        if len(self.indices) < 2:
            raise ValueError("a union requires at least two positions")
        for value in self.indices:
            _position(value)


for _segment in (Field, Key, Index, Slice, Union):
    _segment.__module__ = "qodec"
    _segment.__qualname__ = f"Reference.{_segment.__name__}"
    setattr(Reference, _segment.__name__, _segment)
del _segment