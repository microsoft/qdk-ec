"""Code definitions and text helpers for Pauli strings."""

from typing import Any

from ._native import Code


class PauliExpression:
    """Pauli text with ``*`` for joining terms with a space.

    Construction stores ``str(value)`` without checking Pauli syntax.
    Multiplication joins text; it does not simplify operators or compute
    phases. For example, ``pauli("X_0") * "X_1" == "X_0 X_1"``.

    This is not a ``str`` subclass. Equality and hashing use its text.
    Pauli-valued qodec inputs accept this object or a plain string.
    """

    __slots__ = ("_text",)

    def __init__(self, value: object) -> None:
        self._text = str(value)

    @property
    def text(self) -> str:
        """The operator's raw string form."""
        return self._text

    def __str__(self) -> str:
        return self._text

    def __repr__(self) -> str:
        return f"PauliExpression({self._text!r})"

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        text = repr(self) if cycle else str(self)
        for index, line in enumerate(text.split("\n")):
            if index:
                printer.break_()
            printer.text(line)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PauliExpression):
            return self._text == other._text
        if isinstance(other, str):
            return self._text == other
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self) -> int:
        return hash(self._text)

    def __mul__(self, other: object) -> "PauliExpression":
        if not isinstance(other, (str, PauliExpression)):
            return NotImplemented
        return PauliExpression(f"{self} {other}")

    def __rmul__(self, other: object) -> "PauliExpression":
        if not isinstance(other, (str, PauliExpression)):
            return NotImplemented
        return PauliExpression(f"{other} {self}")


def pauli(*tokens: str) -> PauliExpression:
    """Join tokens with spaces and return a :class:`PauliExpression`.

    Raises ``ValueError`` if no tokens are supplied. Does not validate syntax.
    """
    if not tokens:
        raise ValueError("pauli() requires at least one token")
    return PauliExpression(" ".join(tokens))


__all__ = ["Code", "PauliExpression", "pauli"]
