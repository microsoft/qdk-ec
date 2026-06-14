from enum import Enum
import pytest
from deq.cli.util import (
    kwargs_of_qs,
    named_kwargs_of,
    params_of_func_or_cls,
    bool_constructor,
    enum_constructor_of,
)


def test_kwargs_of() -> None:
    expected = {"a": "1", "b": "2", "c": "3"}

    # different separators
    assert kwargs_of_qs("a=1,b=2,c=3") == expected
    assert kwargs_of_qs("a=1;b=2;c=3") == expected
    assert kwargs_of_qs("a=1&b=2&c=3") == expected

    #
    assert named_kwargs_of("MWPF(c=100)") == ("MWPF", {"c": "100"})
    assert named_kwargs_of("MWPF") == ("MWPF", {})


def test_params_of_func() -> None:

    # pylint: disable=unused-argument
    def example(
        a: int,
        b: str = "default",
        c: float = 1.0,
        d: float | None = None,
    ) -> None: ...

    expected = {"a": int, "b": str, "c": float, "d": float}
    assert params_of_func_or_cls(example) == expected


def test_bool_constructor() -> None:
    assert bool_constructor("true") is True
    assert bool_constructor("True") is True
    assert bool_constructor("1") is True
    assert bool_constructor("false") is False
    assert bool_constructor("False") is False
    assert bool_constructor("0") is False

    with pytest.raises(
        ValueError,
        match=r"cannot convert 'yes' to bool",
    ):
        bool_constructor("yes")

    # pylint: disable=unused-argument
    def example(
        flag: bool,
    ) -> None: ...

    params = params_of_func_or_cls(example)
    assert params["flag"]("true") is True
    assert params["flag"]("0") is False


def test_enum_constructor() -> None:

    class ColorEnum(Enum):
        red = "red"
        green = "green"
        blue = "blue"

    constructor = enum_constructor_of(ColorEnum)
    assert constructor("red") == ColorEnum.red
    assert constructor("green") == ColorEnum.green
    assert constructor("blue") == ColorEnum.blue

    with pytest.raises(
        ValueError,
        match=r"enum name yes not in supported names",
    ):
        constructor("yes")

    # pylint: disable=unused-argument
    def example(
        color: ColorEnum,
    ) -> None: ...

    params = params_of_func_or_cls(example)
    assert params["color"]("red") == ColorEnum.red
    assert params["color"]("blue") == ColorEnum.blue
