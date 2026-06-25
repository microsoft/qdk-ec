from typing import Any, Type, Callable
import typing
from urllib.parse import parse_qs
import inspect
import types
import json
from enum import Enum
from google.protobuf.json_format import MessageToDict
import deq.proto.deq_bin_pb2 as pb

# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint


def kwargs_of_qs(qs: str) -> dict[str, str]:
    """
    parsing a custom querystring format into a dictionary.
    The format is normal querystring format, but with '&' replaced by ',' or ';'.
    Thus, the value cannot contain ',' or ';', which is usually fine in the context of this project.
    Optionally, one can use '@' as an alias to '=' to avoid issues when '='
    has special meanings (e.g. in papermill).
    """
    querystring = qs.replace(",", "&").replace(";", "&").replace("@", "=")
    dict_of_qs = parse_qs(querystring)
    assert len(dict_of_qs) > 0 or qs == "", f"querystring '{qs}' is not valid"
    for key in dict_of_qs:
        assert key.isidentifier(), f"key '{key}' is not a valid identifier"
    return {key: value[0] for key, value in dict_of_qs.items()}


def named_kwargs_of(input_str: str) -> tuple[str, dict[str, str]]:
    """
    expecting a format of 'name(a=1,b=2)'
    """
    if "(" in input_str:
        assert (
            input_str[-1] == ")"
        ), f"input '{input_str}' is not a valid format, consider using ; in lieu of ,"
        split_index = input_str.index("(")
        name = input_str[:split_index]
        assert name.isidentifier(), f"name '{name}' is not a valid identifier"
        kwargs = kwargs_of_qs(input_str[split_index + 1 : -1])
        return name, kwargs
    name = input_str
    assert name.isidentifier(), f"name '{name}' is not a valid identifier"
    return name, {}


def params_of_func_or_cls(func: Any) -> dict[str, Any]:
    """
    the decorated class must have an initialization function that accepts
    str, int or float KEYWORD input.
    or it could be a function that accepts str, int or float KEYWORD input.
    All other types of arguments must be convertible from str, i.e., cls(str) must work
    """
    signature = inspect.signature(func)
    params = {}
    for param in list(signature.parameters.values()):
        if (
            isinstance(param.annotation, types.UnionType)
            or typing.get_origin(param.annotation) == typing.Union
        ):
            args = [arg for arg in param.annotation.__args__ if arg != type(None)]
            assert len(args) == 1, "only support Union[TYPE, None] for now"
            assert param.default is None, (
                f"default value of {param.name} must be None for "
                + "Union[TYPE, None] in {func.__name__}"
            )
            params[param.name] = args[0]
        elif param.annotation == bool:
            params[param.name] = bool_constructor
        elif inspect.isclass(param.annotation) and issubclass(param.annotation, Enum):
            params[param.name] = enum_constructor_of(param.annotation)
        else:
            params[param.name] = param.annotation
    return params


def bool_constructor(name: str) -> bool:
    if name.lower() == "true" or name == "1":
        return True
    if name.lower() == "false" or name == "0":
        return False
    raise ValueError(
        f"cannot convert '{name}' to bool, only 'true', 'false', '1', '0' are supported"
    )


def enum_constructor_of(enum_class: Type[Enum]) -> Callable[[str], Enum]:
    supported_names = {e.name: e for e in enum_class}

    def constructor(name: str) -> Enum:
        if name not in supported_names:
            raise ValueError(
                f"enum name {name} not in supported names: {list(supported_names.keys())}"
            )
        return supported_names[name]

    return constructor


def pretty_print_json(
    obj: Any, max_depth: int = 2, current_depth: int = 0, indent: int = 4
) -> str:
    if current_depth >= max_depth or not isinstance(obj, (dict, list)):
        return json.dumps(obj)
    if isinstance(obj, dict):
        return (
            "{\n"
            + ",\n".join(
                f'{" " * (current_depth + 1) * indent}"{k}": '
                + f"{pretty_print_json(v, max_depth, current_depth + 1)}"
                for k, v in obj.items()
            )
            + f'\n{" " * current_depth * indent}}}'
        )
    # obj is a list
    return (
        "[\n"
        + ",\n".join(
            f'{" " * (current_depth + 1) * indent}'
            + f"{pretty_print_json(v, max_depth, current_depth + 1)}"
            for v in obj
        )
        + f'\n{" " * current_depth * indent}]'
    )


def pretty_print_library(
    lib: pb.Library, max_depth: int = 2, current_depth: int = 0, indent: int = 4
) -> str:
    return pretty_print_json(MessageToDict(lib), max_depth, current_depth, indent)


class StdColor(Enum):
    Unknown = "#cfcfcf"
    Prepare = "#d6938e"
    Idle = "#92abc0"
    Measure = "#aac8a4"
    CNOT = "#bca9c1"
    UNKNOWN1 = "#dab786"
    UNKNOWN2 = "#dbdbaa"
    UNKNOWN3 = "#c2b69c"
    UNKNOWN4 = "#d9b7c9"


# ── Bit / hex conversions (BitVector MSB-first convention) ───────────


def hex_to_bits(hex_str: str, num_bits: int) -> list[int]:
    """Convert a hex string to a list of bits.

    MSB-first within each byte: bit 0 = 0x80, bit 1 = 0x40, …, bit 7 = 0x01.
    """
    raw = bytes.fromhex(hex_str)
    bits: list[int] = []
    for byte in raw:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    if len(bits) < num_bits:
        raise ValueError(
            f"Hex string '{hex_str}' provides {len(bits)} bits, "
            f"but {num_bits} measurements expected"
        )
    return bits[:num_bits]


def parse_bits(value: str, num_bits: int) -> list[int]:
    """Parse a bit-vector from a string with ``0x`` or ``0b`` prefix.

    Supported formats:

    - ``0b01010`` — binary literal, MSB first.  Must provide at least
      *num_bits* digits.
    - ``0xa0`` — hex literal (same encoding as :func:`hex_to_bits`).

    Returns a list of *num_bits* ints (each 0 or 1).
    """
    if value.startswith(("0b", "0B")):
        digits = value[2:]
        if not all(c in "01" for c in digits):
            raise ValueError(f"Invalid binary string: {value!r}")
        bits = [int(c) for c in digits]
        if len(bits) < num_bits:
            raise ValueError(
                f"Binary string '{value}' provides {len(bits)} bits, "
                f"but {num_bits} measurements expected"
            )
        return bits[:num_bits]
    if value.startswith(("0x", "0X")):
        return hex_to_bits(value[2:], num_bits)
    raise ValueError(f"Bit-vector string must start with '0x' or '0b', got: {value!r}")


def bits_to_hex(bits: list[int]) -> str:
    """Convert a list of bits to a ``0x``-prefixed hex string.

    MSB-first within each byte: bit 0 = 0x80, bit 1 = 0x40, …, bit 7 = 0x01.
    """
    padded = bits + [0] * ((8 - len(bits) % 8) % 8)
    result = bytearray()
    for i in range(0, len(padded), 8):
        byte = 0
        for j in range(8):
            byte |= padded[i + j] << (7 - j)
        result.append(byte)
    return "0x" + result.hex()


def bits_to_str(bits: list[int]) -> str:
    """Format a list of bits as a ``0b``-prefixed binary string like ``'0b01010'``."""
    return "0b" + "".join(str(b) for b in bits)
