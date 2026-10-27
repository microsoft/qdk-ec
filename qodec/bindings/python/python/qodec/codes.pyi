"""Define error-correcting codes and write the Pauli strings they use."""

from typing import TYPE_CHECKING, Any

from . import Code as Code

if TYPE_CHECKING:
    # A type-only alias: `qodec.PauliString` has no runtime counterpart here.
    from . import PauliString as PauliString

__all__ = ["Code", "PauliExpression", "pauli"]

class PauliExpression:
    """Write Pauli strings by joining terms with ``*``.

    Construction stores ``str(value)`` without checking Pauli syntax.
    For example, ``pauli("X_0") * "X_1"`` has the text ``"X_0 X_1"``.
    Here ``*`` joins text with a space. It does not simplify operators or
    compute phases.

    This is not a ``str`` subclass. Equality and hashing use its text.
    Pauli-valued qodec inputs accept this object or a plain string.
    """

    def __init__(self, value: object) -> None: ...
    @property
    def text(self) -> str:
        """The operator's raw string form."""
        ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def _repr_pretty_(self, printer: Any, cycle: bool) -> None: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __mul__(self, other: object) -> "PauliExpression": ...
    def __rmul__(self, other: object) -> "PauliExpression": ...

def pauli(*tokens: str) -> PauliExpression:
    """Build a :class:`PauliExpression` from space-separated terms.

    ``pauli("X_0", "Z_1")`` joins the tokens into ``"X_0 Z_1"``.

    Raises ``ValueError`` if no tokens are supplied. Does not validate syntax.
    """
    ...
