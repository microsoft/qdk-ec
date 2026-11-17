"""Validated, tool-owned annotations shared by source expansion and lowering."""

from collections.abc import Sequence
import math
from typing import Union

from deqagram import Decorator, RepeatBlock

MetadataValue = Union[None, bool, str, int, float, list["MetadataValue"], dict[str, "MetadataValue"]]


def _value(value: object) -> MetadataValue:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        if abs(value) > 2**53:
            raise ValueError("@METADATA integers beyond 2**53 must be quoted to preserve precision")
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _value(item) for key, item in value.items()}
    raise ValueError("@METADATA requires finite JSON-compatible values")


def metadata_from(decorators: Sequence[Decorator]) -> dict[str, MetadataValue]:
    result: dict[str, MetadataValue] = {}
    for decorator in decorators:
        if decorator.name != "METADATA":
            continue
        if len(decorator.arguments) != 2:
            raise ValueError('@METADATA requires exactly two positional arguments: ("key", data)')
        key, value = decorator.arguments
        if not isinstance(key, str) or not key:
            raise ValueError("@METADATA requires a nonempty string key")
        if key.startswith("deq.") or key in ("loss_strategy", "private_gadgets"):
            raise ValueError(f"@METADATA key {key!r} is reserved for DEQ tooling")
        if key in result:
            raise ValueError(f"duplicate @METADATA key {key!r} on the same target")
        result[key] = _value(value)
    return result


def inherited_decorators(
    decorators: Sequence[Decorator], defaults: dict[str, MetadataValue]
) -> list[Decorator]:
    values = defaults | metadata_from(decorators)
    return [
        *(decorator for decorator in decorators if decorator.name != "METADATA"),
        *(Decorator("METADATA", (key, value)) for key, value in values.items()),
    ]


def has_metadata(body: Sequence[object]) -> bool:
    return any(
        any(decorator.name == "METADATA" for decorator in getattr(item, "decorators", ()))
        or isinstance(item, RepeatBlock) and has_metadata(item.body)
        for item in body
    )
