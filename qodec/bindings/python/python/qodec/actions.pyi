"""Describe what an instruction does, one action step at a time.

These values make up :attr:`qodec.Instruction.action`.

Pauli inputs accept strings or :class:`qodec.codes.PauliExpression` objects;
getters return strings. List and dictionary getters return copies.
``condition`` is keyword-only and defaults to ``None`` (unconditional)
on every action that accepts it. :class:`Observe` has no condition.
"""

from typing import TYPE_CHECKING, Any, final
from typing_extensions import Self

if TYPE_CHECKING:
    from . import PauliLike, PauliString

__all__ = ["Clifford", "Condition", "Observe", "Pauli", "Rotate", "Stabilize"]

@final
class Condition:
    """Choose when an action step runs, using classical bits.

    Each string in ``predicates`` names a declared ``bit`` parameter or an
    ``outcomes[i]`` measurement result within the instruction. ``i`` counts
    observe outcomes across the action list, starting at zero. Flags and
    ``readouts[i]`` cannot guard an action step.

    ``invert=False`` runs the step when their XOR is 1 (``if:``);
    ``invert=True`` runs it when their XOR is 0 (``unless:``).
    XOR is 1 when an odd number of the bits are 1. For example,
    ``Condition(["outcomes[0]"])`` runs on a first measurement result of 1;
    adding ``invert=True`` runs on 0 instead.

    Construction stores the strings without resolving their names.
    """

    def __new__(
        cls, predicates: list[str], *, invert: bool = False) -> Self: ...
    @property
    def predicates(self) -> list[str]: ...
    @property
    def invert(self) -> bool: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def _repr_pretty_(self, printer: Any, cycle: bool) -> None: ...

@final
class Stabilize:
    """Prepare or reset into each operator's +1 eigenspace, in list order.

    No measurement outcome is emitted. Without a condition, this step can
    introduce temporary qubits outside the instruction's input/output index
    range. For a one-qubit input and output, ``Stabilize(["Z_1"])`` prepares
    temporary qubit 1 in the zero state. Later action steps can use it.

    Only named nonidentity indices are introduced, not intervening gaps.
    Temporary qubits are local to the instruction invocation and are traced
    out at its end. A conditional stabilization can reset available qubits
    but cannot introduce temporary ones.

    A partial eigenspace constraint does not choose a state or recovery
    within the unconstrained space. These semantic qubits do not prescribe
    the physical ancillas used by a gadget.
    """

    def __new__(
        cls, operators: list[PauliLike], *, condition: Condition | None = ...) -> Self: ...
    @property
    def operators(self) -> list[PauliString]: ...
    @property
    def condition(self) -> Condition | None: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def _repr_pretty_(self, printer: Any, cycle: bool) -> None: ...

@final
class Clifford:
    """Apply a Clifford operation by specifying where it sends Pauli generators.

    ``generators`` is a ``{input_generator: output_image}`` dictionary:
    each key is a Pauli generator before the operation, and its value is
    the Pauli operator it becomes.
    """

    def __new__(
        cls, generators: dict[PauliLike, PauliLike], *, condition: Condition | None = ...) -> Self: ...
    @property
    def generators(self) -> dict[PauliString, PauliString]: ...
    @property
    def condition(self) -> Condition | None: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def _repr_pretty_(self, printer: Any, cycle: bool) -> None: ...

@final
class Pauli:
    """Apply a single Pauli operator."""

    def __new__(
        cls, operator: PauliLike, *, condition: Condition | None = ...) -> Self: ...
    @property
    def operator(self) -> PauliString: ...
    @property
    def condition(self) -> Condition | None: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def _repr_pretty_(self, printer: Any, cycle: bool) -> None: ...

@final
class Observe:
    """Measure Pauli observables, producing one outcome per entry.

    Each observable is a Pauli operator to measure. Outcomes are numbered
    across the instruction's action list. To condition an action step on a
    result, refer to it as ``outcomes[i]`` in a :class:`Condition`.
    This action takes no ``condition``.
    """

    def __new__(
        cls,
        observables: list[PauliLike],
    ) -> Self: ...
    @property
    def observables(self) -> list[PauliString]: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def _repr_pretty_(self, printer: Any, cycle: bool) -> None: ...

@final
class Rotate:
    """Apply a rotation about ``pauli`` by ``angle`` radians.

    ``angle`` is a number or a parameter name, for example
    ``Rotate("Z_0", "theta")``. The getter returns a ``float`` for a numeric
    input and a ``str`` for a parameter name.
    """

    def __new__(
        cls, pauli: PauliLike, angle: float | str, *, condition: Condition | None = ...) -> Self: ...
    @property
    def pauli(self) -> PauliString: ...
    @property
    def angle(self) -> float | str: ...
    @property
    def condition(self) -> Condition | None: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def _repr_pretty_(self, printer: Any, cycle: bool) -> None: ...
