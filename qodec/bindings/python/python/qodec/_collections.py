"""Owner-bound collection protocols; native setters own conversion and guards."""

from collections.abc import Iterable, Iterator, Mapping, MutableMapping, MutableSequence
from itertools import chain
from typing import Any, overload

from .instructions import InstructionCall


def _plain(value: Any, memo: dict[int, Any] | None = None) -> Any:
    if isinstance(value, _View):
        return value._read()
    if not isinstance(value, (Mapping, list)):
        return value
    if memo is None:
        memo = {}
    if id(value) in memo:
        return memo[id(value)]
    result: Any = {} if isinstance(value, Mapping) else []
    memo[id(value)] = result
    if isinstance(value, Mapping):
        result.update((key, _plain(item, memo)) for key, item in value.items())
    else:
        result.extend(_plain(item, memo) for item in value)
    return result


class _View:
    def __init__(self, owner: Any, field: str, path: tuple[Any, ...] = ()) -> None:
        self._owner = owner
        self._field = field
        self._path = path

    def _root(self) -> Any:
        return getattr(self._owner, "_get_" + self._field)()

    def _read(self) -> Any:
        value = self._root()
        for key in self._path:
            value = value[key]
        return value

    def _write(self, value: Any) -> None:
        value = _plain(value)
        if not self._path:
            setattr(self._owner, self._field, value)
            return
        root = self._root()
        parent = root
        for key in self._path[:-1]:
            parent = parent[key]
        parent[self._path[-1]] = value
        setattr(self._owner, self._field, root)

    def _item(self, key: Any, value: Any) -> Any:
        if isinstance(self._owner, InstructionCall) and self._field in ("arguments", "operands"):
            return value
        if isinstance(value, dict):
            return _Mapping(self._owner, self._field, (*self._path, key))
        if isinstance(value, list):
            return _Sequence(self._owner, self._field, (*self._path, key))
        return value

    def __len__(self) -> int:
        return len(self._read())

    def __repr__(self) -> str:
        return repr(self._read())

    def __eq__(self, other: object) -> bool:
        if isinstance(self, _Sequence) and isinstance(other, (list, tuple, _Sequence)):
            return list(self._read()) == list(other)
        return bool(self._read() == _plain(other))

    def __copy__(self) -> Any:
        return dict(self._read()) if isinstance(self, Mapping) else list(self._read())

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        from copy import deepcopy

        result: Any = {} if isinstance(self, Mapping) else []
        memo[id(self)] = result
        if isinstance(self, Mapping):
            result.update((key, deepcopy(value, memo)) for key, value in self._read().items())
        else:
            result.extend(deepcopy(value, memo) for value in self._read())
        return result


class _ReadOnlyMapping(_View, Mapping[str, Any]):
    def __getitem__(self, key: str) -> Any:
        return self._read()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._read())


class _Mapping(_ReadOnlyMapping, MutableMapping[str, Any]):
    def _key(self, key: str, values: Mapping[str, Any]) -> str:
        if self._field != "frames" or self._path:
            return key
        from . import Reference

        reference = self._owner._validate_frame_key(key)
        return next((stored for stored in values if Reference(stored) == reference), reference.path)

    def __getitem__(self, key: str) -> Any:
        values = self._read()
        key = self._key(key, values)
        return self._item(key, values[key])

    def __setitem__(self, key: str, value: Any) -> None:
        values = self._read()
        values[self._key(key, values)] = value
        self._write(values)

    def __delitem__(self, key: str) -> None:
        values = self._read()
        del values[self._key(key, values)]
        self._write(values)

    def update(self, *args: Any, **kwargs: Any) -> None:
        values = self._read()
        if self._field == "frames" and not self._path:
            from . import Reference

            if len(args) > 1:
                raise TypeError(f"update expected at most 1 argument, got {len(args)}")
            incoming = args[0] if args else ()
            entries = ((key, incoming[key]) for key in incoming.keys()) if hasattr(incoming, "keys") else incoming
            stored_keys = {Reference(key): key for key in values}
            for key, value in chain(entries, kwargs.items()):
                reference = self._owner._validate_frame_key(key)
                stored = stored_keys.setdefault(reference, reference.path)
                values[stored] = value
        else:
            values.update(*args, **kwargs)
        self._write(values)

    def clear(self) -> None:
        self._write({})

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, key: str, *default: Any) -> Any:
        values = self._read()
        result = values.pop(self._key(key, values), *default)
        self._write(values)
        return result

    def popitem(self) -> tuple[str, Any]:
        values = self._read()
        result: tuple[str, Any] = values.popitem()
        self._write(values)
        return result


class _Sequence(_View, MutableSequence[Any]):
    @overload
    def __getitem__(self, index: int) -> Any: ...
    @overload
    def __getitem__(self, index: slice) -> list[Any]: ...
    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return list(self._read()[index])
        return self._item(index, self._read()[index])

    def __iter__(self) -> Iterator[Any]:
        return (self._item(index, value) for index, value in enumerate(self._read()))

    def __setitem__(self, index: int | slice, value: Any) -> None:
        values = list(self._read())
        values[index] = value
        self._write(values)

    def __delitem__(self, index: int | slice) -> None:
        values = list(self._read())
        del values[index]
        self._write(values)

    def insert(self, index: int, value: Any) -> None:
        values = list(self._read())
        values.insert(index, value)
        self._write(values)

    def extend(self, values: Iterable[Any]) -> None:
        self._write([*self._read(), *values])

    def clear(self) -> None:
        self._write([])

    def reverse(self) -> None:
        self._write(list(reversed(self._read())))

    def pop(self, index: int = -1) -> Any:
        values = list(self._read())
        result = values.pop(index)
        self._write(values)
        return result
