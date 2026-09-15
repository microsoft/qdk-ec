"""Declare the blocks and parameters an instruction uses, then supply them in a call.

An instruction's steps are described by :mod:`qodec.actions`.
"""

from typing import TYPE_CHECKING, Any, final
from enum import Enum

from .. import Instruction as Instruction
from .. import InstructionSet as InstructionSet
if TYPE_CHECKING:
    from .. import Metadata, PauliString
from typing_extensions import Self

__all__ = ["Block", "BlockOperand", "Instruction", "InstructionCall", "InstructionSet", "Parameter"]

@final
class Block:
    """Declare a kind of quantum block in an instruction set architecture (ISA).

    This describes a block type, not a particular block in a circuit.
    ``encodes`` is the number of logical qubits in each block of this type.
    Instances are read-only; construct a new block to change either value.
    """

    def __new__(cls, name: str, encodes: int) -> Self: ...

    def __str__(self) -> str: ...
    def _repr_pretty_(self, printer: Any, cycle: bool) -> None: ...

    @property
    def name(self) -> str: ...

    @property
    def encodes(self) -> int:
        """Number of logical qubits this block type encodes."""
        ...
    def __eq__(self, other: object, /) -> bool:
        """Value equality, comparing fields rather than identity. Instances are unhashable."""
        ...
    def __repr__(self) -> str: ...

@final
class BlockOperand:
    """Declare a quantum block an instruction takes as input or produces as output.

    A block operand is an entry in the instruction's ``in:`` or ``out:``
    list. It names a block type; a call supplies the actual block by position.
    Each entry contributes its block type's ``encodes`` logical indices,
    in list order. Actions use those flat indices, as in ``X_0`` or ``Z_3``.
    """

    def __new__(
        cls,
        block: str,
        *,
        is_variadic: bool = False,
    ) -> Self:
        """Construct a block operand; ``is_variadic=False`` by default.

        A variadic operand stands for a variable number of blocks.
        A trailing ``...`` is removed from ``block`` and sets ``is_variadic=True``.
        The suffix takes precedence even when ``is_variadic=False`` is supplied.
        """
        ...

    @property
    def block(self) -> str:
        """Name of the block-type declaration this entry references."""
        ...

    @property
    def is_variadic(self) -> bool: ...

    def __str__(self) -> str: ...
    def _repr_pretty_(self, printer: Any, cycle: bool) -> None: ...
    def __eq__(self, other: object, /) -> bool:
        """Value equality, comparing fields rather than identity. Instances are unhashable."""
        ...
    def __repr__(self) -> str: ...

@final
class Parameter:
    """Declare the name and kind of a classical input to an instruction.

    A parameter says what the instruction expects. An argument is the
    value supplied for that parameter in an :class:`InstructionCall`.
    """

    class Kind(Enum):
        """The declared type of a classical :class:`Parameter`.

        A ``BIT`` parameter is a runtime classical input that can control
        whether an action step runs, through :class:`qodec.actions.Condition`.
        Parameters of the other kinds, including ``BOOLEAN``, take
        compile-time literal arguments.
        """
        BIT = "bit"
        NUMBER = "number"
        INTEGER = "integer"
        BOOLEAN = "boolean"
        STRING = "string"
        PAULI = "pauli"

    def __new__(
        cls, name: str, kind: "str | Parameter.Kind") -> Self:
        """Accept a :class:`Parameter.Kind` member or its lowercase string value.

        An unknown kind string raises ``ValueError``.
        """
        ...

    @property
    def name(self) -> str: ...

    @property
    def kind(self) -> "Parameter.Kind":
        """The declared kind as an enum member, even when constructed from a string."""
        ...

    def __str__(self) -> str: ...
    def _repr_pretty_(self, printer: Any, cycle: bool) -> None: ...
    def __eq__(self, other: object, /) -> bool:
        """Value equality, comparing fields rather than identity. Instances are unhashable."""
        ...
    def __repr__(self) -> str: ...

@final
class InstructionCall:
    """Specify one use of an instruction, with its blocks and classical inputs.

    ``operands`` supplies quantum blocks by position. ``arguments`` supplies
    values for the instruction's declared classical parameters by name.
    For an instruction named ``RZ`` with one block operand and a numeric
    parameter ``theta``, ``InstructionCall("RZ", operands=["data"],
    arguments={"theta": 0.5})`` supplies block ``data`` and argument ``0.5``.

    Calls returned by :meth:`qodec.gadgets.Circuit.calls` are parsed
    values; editing them does not change the circuit source.
    """

    Argument = int | float | bool | str | list[int] | list[str]
    """Type-only alias, not available at runtime: a bound call argument.

    Parsed readout references are strings of the form ``circuit.readouts[i]``.
    """

    def __new__(
        cls,
        mnemonic: str,
        *,
        operands: list[int | str] | None = ...,
        arguments: dict[str, "InstructionCall.Argument"] | None = None,
        select: list[dict[str, int]] | None = None,
    ) -> Self:
        """Store a call without checking it against an instruction set.

        All optional arguments default to ``None``, producing empty
        collections. ``select`` values must be integers 0 or 1. Booleans
        raise ``TypeError``; invalid integers raise ``ValueError`` or ``OverflowError``.
        """
        ...

    @property
    def mnemonic(self) -> str: ...

    @property
    def operands(self) -> list[int | str]:
        """The blocks this call acts on, in the instruction's declared order.

        An entry is a block index (``int``) or label (``str``); an integer
        becomes its decimal text in :attr:`qodec.gadgets.Circuit.blocks`.
        The returned list is a new container.
        """
        ...

    @property
    def arguments(self) -> dict[str, "InstructionCall.Argument"]:
        """Supplied classical values, keyed by the instruction's parameter names.

        Parsed ``bit`` references use ``"circuit.readouts[i]"``, where ``i``
        is the absolute, zero-based bit position across prior calls.
        Literal values are ``int``, ``float``, ``bool``, ``str``, ``list[int]``,
        or ``list[str]``. Boolean literals remain ``bool``, distinct from 0 and 1.
        Parser callbacks require scalar integers to fit in a signed 64-bit integer.
        The returned dictionary is new, but its values are shared with the
        call, including nested lists.
        """
        ...

    @property
    def select(self) -> list[dict[str, int]]:
        """Accepted flag patterns for this call, returned as a copy.

        Each sparse ``{flag: 0|1}`` dictionary requires all its entries to
        match; the list accepts any matching pattern. Keys are declared
        flag names or ``flags[i]``, where ``i`` indexes the called
        instruction's flag list. Empty means no selection is applied.
        """
        ...
    def __eq__(self, other: object, /) -> bool:
        """Structural equality of fields, including literal kinds inside lists.

        Boolean, integer, and floating-point literals are distinct even when
        Python compares their values equal. Instances are unhashable.
        """
        ...
    def __repr__(self) -> str: ...
