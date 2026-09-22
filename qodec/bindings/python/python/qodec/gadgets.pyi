"""Describe a gadget's circuit, how its blocks are encoded, and how its bits relate.

These types supply the parts of a :class:`qodec.Gadget`.
"""

from collections.abc import Callable, Mapping, MutableSequence, Sequence
from typing import TYPE_CHECKING, Any, Literal, final
from . import Gadget as Gadget
from . import Reference as _Reference, ReferenceLike as _ReferenceLike
if TYPE_CHECKING:
    from . import Code, Instruction, InstructionSet, Metadata, PauliString
    from .instructions import InstructionCall
from typing_extensions import Self

__all__ = [
    "Check", "Circuit", "Encoding", "Flag", "Gadget", "Outcome",
    "Readout", "ReadoutLike",
]

@final
class Circuit:
    """Hold circuit source text and the instruction set it calls.

    ``format`` optionally identifies the source format. The instruction set
    supplies the instruction definitions.

    Direct construction does not parse the source. Call :meth:`calls`,
    or read :attr:`blocks` or :attr:`readouts` to interpret it. Each access
    parses the current source again. Built-in parsing raises ``ValueError``
    for an unavailable format, invalid source, or an unknown instruction.
    Custom parser exceptions propagate unchanged.
    Loading and saving a protocol preserve the source without checking its
    calls. These accessors can therefore fail on a loadable draft.
    YAML is registered in Rust. The supplied Stim adapter is registered in
    Python at module load when Stim is installed. Use ``qodec[parsers]`` to install
    that dependency.
    Other formats need :func:`qodec.register` or an explicit parser.
    """

    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Self: ...
    def __replace__(self, **changes: Any) -> Self: ...

    def __str__(self) -> str:
        """A YAML snippet containing verbatim source and its effective format.

        The containing layer supplies the instruction set. Display does not
        parse or validate source, including explicitly tagged YAML source.
        """
        ...
    def _repr_pretty_(self, printer: Any, cycle: bool) -> None: ...

    def __new__(
        cls,
        instruction_set: "InstructionSet",
        source: str,
        *,
        format: str | None = ...,
    ) -> Self:
        """Store source text and a shared instruction set; no file is read.

        ``format`` defaults to ``None``, which infers the format from the
        source text when needed. Construction does not parse the source.
        """
        ...

    @property
    def instruction_set(self) -> "InstructionSet":
        """The shared instruction set the source calls; changes are visible through all references."""
        ...
    @instruction_set.setter
    def instruction_set(self, value: "InstructionSet") -> None: ...

    @property
    def source(self) -> str:
        """The stored source text, not a file path."""
        ...
    @source.setter
    def source(self, value: str) -> None: ...

    @property
    def format(self) -> str | None:
        """An explicit format tag, or ``None`` to infer it from the source text."""
        ...
    @format.setter
    def format(self, value: str | None) -> None: ...

    @property
    def effective_format(self) -> str:
        """The format tag if set, otherwise the one inferred from the source."""
        ...

    def calls(
        self, *, parser: Callable[[str, InstructionSet], Sequence[InstructionCall]] | None = None
    ) -> list["InstructionCall"]:
        """A new list of parsed calls, in program order.

        An explicit parser overrides format-based registration for this call.
        Otherwise use the shared Rust registry for the effective format.
        A missing registration raises ValueError, as does a call naming an
        instruction the instruction set does not declare. The callback receives source
        and an isolated target instruction set snapshot as positional arguments. All call
        values are copied through the Rust representation; editing the returned
        calls or snapshot does not change :attr:`source` or :attr:`instruction_set`.
        """
        ...

    @property
    def blocks(self) -> list[str]:
        """Distinct block labels this circuit's calls name, in first-appearance order.

        Uses the same registered parser as :meth:`calls`.
        Multi-qubit blocks are not expanded into individual qubits.
        These are the labels used by :attr:`Encoding.support`, not physical
        addresses or a simulator size.
        """
        ...

    @property
    def readouts(self) -> list["Outcome | Flag"]:
        """One :class:`Outcome` or :class:`Flag` per circuit output bit, in record order.

        The list index is the ``i`` of a ``circuit.readouts[i]`` reference.
        Each call contributes its ``observe`` outcomes first, then its
        declared flags. Entries carry no position of their own, so use
        ``enumerate`` rather than filtering first.
        Uses the same registered parser as :meth:`calls`.

        Raises ``ValueError`` if a call names an instruction absent from
        the instruction set, in addition to the parsing errors described on :class:`Circuit`.
        """
        ...
    def __eq__(self, other: object, /) -> bool:
        """Value equality, comparing fields rather than identity. Instances are unhashable."""
        ...
    def __repr__(self) -> str: ...

@final
class Outcome:
    """A measurement-result bit from a circuit call's ``observe`` action."""

    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Self: ...

    @property
    def instruction(self) -> int:
        """Index into the result of :meth:`Circuit.calls` that produced this bit."""
        ...

    @property
    def observable(self) -> str:
        """The observable measured, exactly as the called instruction declares it.

        It is not rewritten with the call's circuit labels. The call is
        ``circuit.calls()[instruction]``.
        """
        ...

    def __eq__(self, other: object, /) -> bool: ...
    def __repr__(self) -> str: ...
@final
class Flag:
    """A named output bit declared by a circuit call's instruction.

    A flag reports a bit; the caller decides how to use it.
    """

    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Self: ...

    @property
    def instruction(self) -> int:
        """Index into the result of :meth:`Circuit.calls` that produced this bit."""
        ...

    @property
    def name(self) -> str:
        """The flag's declared name on the called instruction."""
        ...

    def __eq__(self, other: object, /) -> bool: ...
    def __repr__(self) -> str: ...
@final
class Encoding:
    """Say which code and circuit labels represent one gadget block operand.

    A block operand is a quantum block in the instruction's input or output
    list. The labels are ordered to match the code's qubits.
    Encodings align with the implemented instruction's input and output
    operand lists. :attr:`qodec.Gadget.inputs` and :attr:`qodec.Gadget.outputs`
    return live sequences of shared encoding objects. Changing an encoding
    is visible to every gadget referencing it; its code is shared too.
    """

    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Self: ...
    def __replace__(self, **changes: Any) -> Self: ...

    def __new__(
        cls,
        code: "Code",
        *,
        support: Sequence[str] = ...,
        block_types: Sequence[str] = ...,
    ) -> Self:
        """Store a shared code; omitted support and block types are empty lists."""
        ...

    @property
    def code(self) -> "Code":
        """The shared code; changes are visible to every encoding referencing it."""
        ...
    @code.setter
    def code(self, value: "Code") -> None: ...

    @property
    def support(self) -> MutableSequence[str]:
        """Circuit labels in code-qubit order, returned as a live sequence.

        With ``support=["left", "right"]``, code qubit 0 uses label ``left``
        and code qubit 1 uses ``right``.
        """
        ...
    @support.setter
    def support(self, value: Sequence[str]) -> None: ...

    @property
    def block_types(self) -> MutableSequence[str]:
        """Optional block-type names parallel to :attr:`support`, returned as a live sequence.

        Empty when not supplied. Names refer to the circuit's instruction set. Explicit
        lists must match the support length. A circuit label must have the
        same type throughout one gadget input or output boundary, but its
        types on the two boundaries may differ. These rules are checked by
        :meth:`qodec.Qodec.validate` and when saving.
        """
        ...
    @block_types.setter
    def block_types(self, value: Sequence[str]) -> None: ...

    def __eq__(self, other: object, /) -> bool:
        """Value equality, comparing fields rather than identity. Instances are unhashable."""
        ...
    def __repr__(self) -> str: ...


#: Runtime alias for an immutable parity equation returned by :attr:`qodec.Gadget.checks`.
Check = tuple[_Reference | Literal[0, 1], ...]

#: Runtime alias for an input readout: a returned :class:`Readout`, a parity
#: sequence, or a single-key ``{name: equation}`` mapping.
ReadoutLike = Readout | Sequence[_ReferenceLike | Literal[0, 1]] | Mapping[str, Sequence[_ReferenceLike | Literal[0, 1]]]


@final
class Readout:
    """Describe one gadget output bit by the parity equation that produces it.

    Carries its position, optional name, and flag role. Returned by
    :attr:`qodec.Gadget.readouts` and accepted by the gadget constructor and
    setter alongside parity sequences and named mappings. Only its name
    and equation are copied; its position and flag role are recomputed for
    the destination gadget. This value is an immutable snapshot.
    """

    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Self: ...

    @property
    def position(self) -> int:
        """Index in :attr:`qodec.Gadget.readouts`, addressed as ``readouts[i]``."""
        ...

    @property
    def name(self) -> str | None:
        """The authored name, or ``None`` if the entry is anonymous."""
        ...

    @property
    def is_flag(self) -> bool:
        """Whether this entry realizes one of the instruction's declared flags.

        Outcomes come first, then flags, so this is fixed by position against
        the implemented instruction rather than by anything in the equation.
        """
        ...

    @property
    def equation(self) -> Check:
        """The parity terms as immutable references and integer bits 0 or 1.

        To edit, supply a new equation to the gadget's ``readouts`` setter.
        """
        ...

    def __eq__(self, other: object, /) -> bool: ...
    def __str__(self) -> str:
        """The authored parity list or named dictionary, using quoted path strings.

        The result is a Python literal, also valid YAML, without position
        or role fields. An empty anonymous equation prints as ``[]``.
        """
        ...
    def __repr__(self) -> str:
        """Show position, name, flag role, and the complete equation.

        This is a diagnostic description, not a constructor expression.
        """
        ...
    def _repr_pretty_(self, printer: Any, cycle: bool) -> None: ...
