"""Validated, tool-owned annotations shared by source expansion and lowering."""

from collections.abc import Sequence
from decimal import Decimal
import math
from typing import Union
from deqagram import Decorator, RepeatBlock
from deqagram import DecoratorArgValue
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct

MetadataValue = Union[None, bool, str, int, float, list["MetadataValue"], dict[str, "MetadataValue"]]


def _value(value: object) -> MetadataValue:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("@METADATA requires finite JSON-compatible values")
        if value == value.to_integral_value():
            if value.copy_abs() > 2**53:
                raise ValueError("@METADATA integers beyond 2**53 must be quoted to preserve precision")
            return int(value)
        converted = float(value)
        if not math.isfinite(converted) or (value != 0 and converted == 0):
            raise ValueError("@METADATA number is outside the finite binary64 range")
        return converted
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


def _literal(value: MetadataValue) -> DecoratorArgValue:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, list):
        return [_literal(item) for item in value]
    return {key: _literal(item) for key, item in value.items()}


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
        *(Decorator("METADATA", (key, _literal(value))) for key, value in values.items()),
    ]


def has_metadata(body: Sequence[object]) -> bool:
    return any(
        any(decorator.name == "METADATA" for decorator in getattr(item, "decorators", ()))
        or isinstance(item, RepeatBlock) and has_metadata(item.body)
        for item in body
    )


def decorators_from_metadata(metadata: Struct) -> list[Decorator]:
    return [Decorator("METADATA", (key, _literal(value))) for key, value in MessageToDict(metadata).items()]
