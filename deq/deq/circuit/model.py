"""Semantic model for DEQ files using Python dataclasses.

Reuses Stim-level types (Instruction, Circuit, targets) for physical-level
operations inside GADGET blocks.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

# ── Decorator metadata ───────────────────────────────────────────────


@dataclass(frozen=True)
class KeywordArg:
    """A key=value argument in a decorator, e.g. ``level="high"``."""

    key: str
    value: str | int | float

    def __str__(self) -> str:
        v = f'"{self.value}"' if isinstance(self.value, str) else str(self.value)
        return f"{self.key}={v}"


DecoratorArgValue = str | int | float

DecoratorArg = DecoratorArgValue | KeywordArg


@dataclass(frozen=True)
class Decorator:
    """A decorator annotation like ``@CHECKS("manual")``."""

    name: str
    arguments: tuple[DecoratorArg, ...] = ()

    def __str__(self) -> str:
        if not self.arguments:
            return f"@{self.name}"
        args_str = ", ".join(
            (
                str(a)
                if isinstance(a, KeywordArg)
                else f'"{a}"' if isinstance(a, str) else str(a)
            )
            for a in self.arguments
        )
        return f"@{self.name}({args_str})"


# ── Stim-level target types ──────────────────────────────────────────


@dataclass(frozen=True)
class QubitTarget:
    """A qubit target: an optional ``!`` prefix and a non-negative integer index."""

    index: int
    inverted: bool = False

    def __str__(self) -> str:
        prefix = "!" if self.inverted else ""
        return f"{prefix}{self.index}"


@dataclass(frozen=True)
class MeasurementRecordTarget:
    """A measurement record target like ``rec[-3]``."""

    offset: int

    def __str__(self) -> str:
        return f"rec[-{self.offset}]"


@dataclass(frozen=True)
class PhysicalMeasurementTarget:
    """An absolute physical-measurement target like ``M5``.

    Refers to the ``index``-th physical (internal) measurement of the
    enclosing GADGET, 0-based, excluding the virtual measurements that
    INPUT/OUTPUT ports implicitly introduce.  It is the gadget-scoped
    counterpart of the relative ``rec[-k]`` syntax.
    """

    index: int

    def __str__(self) -> str:
        return f"M{self.index}"


@dataclass(frozen=True)
class InputVirtualTarget:
    """An absolute virtual stabilizer measurement on an INPUT port, like
    ``IN0.S2``.

    Refers to the ``stabilizer_index``-th stabilizer of the
    ``port_index``-th INPUT port (both 0-based, in declaration order
    within the gadget).  Each INPUT port implicitly introduces one
    virtual measurement per stabilizer of its code.
    """

    port_index: int
    stabilizer_index: int

    def __str__(self) -> str:
        return f"IN{self.port_index}.S{self.stabilizer_index}"


@dataclass(frozen=True)
class OutputVirtualTarget:
    """An absolute virtual stabilizer measurement on an OUTPUT port, like
    ``OUT2.S4``.

    Refers to the ``stabilizer_index``-th stabilizer of the
    ``port_index``-th OUTPUT port (both 0-based, in declaration order
    within the gadget).  Each OUTPUT port implicitly introduces one
    virtual measurement per stabilizer of its code.
    """

    port_index: int
    stabilizer_index: int

    def __str__(self) -> str:
        return f"OUT{self.port_index}.S{self.stabilizer_index}"


# Union of all measurement-reference forms (relative and absolute).  Any
# of these resolves to the same global measurement index space during
# transpilation; the same context-specific validation rules apply
# regardless of the source form.
MeasurementRefTarget = (
    MeasurementRecordTarget
    | PhysicalMeasurementTarget
    | InputVirtualTarget
    | OutputVirtualTarget
)


@dataclass(frozen=True)
class SweepBitTarget:
    """A sweep bit target like ``sweep[5]``."""

    index: int

    def __str__(self) -> str:
        return f"sweep[{self.index}]"


@dataclass(frozen=True)
class PauliTarget:
    """A Pauli target like ``X1``, ``!Y3``, or ``Z0``."""

    pauli: str  # "X", "Y", or "Z"
    index: int
    inverted: bool = False

    def __str__(self) -> str:
        prefix = "!" if self.inverted else ""
        return f"{prefix}{self.pauli}{self.index}"


@dataclass(frozen=True)
class CombinerTarget:
    """The combiner target ``*``."""

    def __str__(self) -> str:
        return "*"


@dataclass(frozen=True)
class CheckTarget:
    """A check target like ``C0`` — references a previously declared CHECK."""

    index: int

    def __str__(self) -> str:
        return f"C{self.index}"


@dataclass(frozen=True)
class ReadoutTarget:
    """A readout target like ``R0`` — references a previously declared READOUT."""

    index: int

    def __str__(self) -> str:
        return f"R{self.index}"


@dataclass(frozen=True)
class LogicalPauliTarget:
    """A logical Pauli operator like ``LX0`` / ``LY0`` / ``LZ0``.

    By default (``port_kind=None``), the ``index`` is interpreted
    *globally* across all ports of whichever side (input or output) the
    context dictates: READOUT, CHECK, CONDITIONAL and VIRTUAL targets
    use output ports; the LHS of PROPAGATE uses output ports; on the
    RHS of PROPAGATE the bare form selects an input logical.

    When ``port_kind`` and ``port_index`` are provided (``IN<p>.L<P><i>``
    or ``OUT<p>.L<P><i>``), the ``index`` is *port-local* and the
    direction must match the context (e.g. ``IN<p>.L...`` is not
    allowed on the LHS of PROPAGATE).
    """

    pauli: str  # "X", "Y", or "Z"
    index: int
    port_kind: Optional[Literal["IN", "OUT"]] = None
    port_index: Optional[int] = None

    def __str__(self) -> str:
        if self.port_kind is not None and self.port_index is not None:
            return f"{self.port_kind}{self.port_index}.L{self.pauli}{self.index}"
        return f"L{self.pauli}{self.index}"


@dataclass(frozen=True)
class DestabilizerTarget:
    """An INPUT-port destabilizer target like ``IN0.DS3``.

    Refers to the destabilizer of the ``stab_index``-th stabilizer of
    the ``port_index``-th INPUT port: the Pauli operator that
    anticommutes with that one stabilizer and commutes with all the
    others.  Inside ``PROPAGATE`` it contributes the input cp column
    carrying the syndrome bit of stabilizer ``stab_index`` on that port
    (decomposed into generator columns when the stabilizer is
    redundant).

    PROPAGATE never refers to OUTPUT destabilizers, so no
    corresponding ``OUT<p>.DS<s>`` form is supported.
    """

    port_index: int
    stab_index: int

    def __str__(self) -> str:
        return f"IN{self.port_index}.DS{self.stab_index}"


Target = (
    QubitTarget
    | MeasurementRecordTarget
    | PhysicalMeasurementTarget
    | InputVirtualTarget
    | OutputVirtualTarget
    | SweepBitTarget
    | PauliTarget
    | CombinerTarget
)

ErrorTarget = CheckTarget | ReadoutTarget | LogicalPauliTarget | PauliTarget

ReadoutTargetItem = Target | LogicalPauliTarget | DestabilizerTarget


# ── Stim-level instructions and circuit ──────────────────────────────


@dataclass
class Instruction:
    """A single Stim instruction."""

    name: str
    tag: str | None = None
    arguments: list[float] = field(default_factory=list)
    targets: list[Target] = field(default_factory=list)
    decorators: list[Decorator] = field(default_factory=list)

    def __str__(self) -> str:
        parts = [self.name]
        if self.tag is not None:
            escaped = (
                self.tag.replace("\\", "\\B")
                .replace("]", "\\C")
                .replace("\r", "\\r")
                .replace("\n", "\\n")
            )
            parts.append(f"[{escaped}]")
        if self.arguments:
            args_str = ", ".join(_format_float(a) for a in self.arguments)
            parts.append(f"({args_str})")
        if self.targets:
            parts.append(" ")
            # Don't put spaces around CombinerTarget (*) so MPP renders
            # as "Z0*Z1 Z2*Z3" instead of "Z0 * Z1 Z2 * Z3".
            target_parts: list[str] = []
            prev_was_combiner = False
            for t in self.targets:
                is_combiner = isinstance(t, CombinerTarget)
                if is_combiner:
                    target_parts.append(str(t))
                elif prev_was_combiner:
                    target_parts.append(str(t))
                else:
                    if target_parts:
                        target_parts.append(" ")
                    target_parts.append(str(t))
                prev_was_combiner = is_combiner
            parts.append("".join(target_parts))
        return "".join(parts)


@dataclass
class RepeatBlock:
    """A ``REPEAT K { ... }`` block inside a circuit."""

    count: int
    body: list[Any] = field(default_factory=list)
    decorators: list[Decorator] = field(default_factory=list)

    def __str__(self) -> str:
        inner = "\n".join(str(s) for s in self.body)
        if self.body:
            inner += "\n"
        body_str = _indent(inner)
        return f"REPEAT {self.count} {{\n{body_str}}}"


StimStatement = Instruction | RepeatBlock


@dataclass
class Circuit:
    """A sequence of Stim instructions and repeat blocks."""

    statements: list[StimStatement] = field(default_factory=list)

    def __str__(self) -> str:
        return "\n".join(str(s) for s in self.statements) + (
            "\n" if self.statements else ""
        )


# ── CODE definition ──────────────────────────────────────────────────


@dataclass(frozen=True)
class PauliTerm:
    """A single Pauli operator on a qubit, e.g. ``Z0`` or ``X3``."""

    pauli: str  # "I", "X", "Y", or "Z"
    index: int

    def __str__(self) -> str:
        return f"{self.pauli}{self.index}"


@dataclass(frozen=True)
class PauliProduct:
    """A product of Pauli terms, e.g. ``Z0*Z1*Z2*Z3``."""

    terms: tuple[PauliTerm, ...]

    def __str__(self) -> str:
        return "*".join(str(t) for t in self.terms)


@dataclass
class LogicalOperator:
    """A pair of X and Z logical operators for one logical qubit."""

    x_operator: PauliProduct
    z_operator: PauliProduct


@dataclass
class CodeDefinition:
    """A ``CODE Name [[n,k,d]] { ... }`` definition."""

    name: str
    n: int
    k: int
    d: int | None = None
    logicals: list[LogicalOperator] = field(default_factory=list)
    stabilizers: list[PauliProduct] = field(default_factory=list)
    decorators: list[Decorator] = field(default_factory=list)
    source_file: str | None = None
    source_line: int | None = None


# ── Ports ─────────────────────────────────────────────────────────────


@dataclass
class InputPort:
    """An ``INPUT CodeName qubit_indices...`` declaration."""

    code_name: str
    qubit_indices: list[int] = field(default_factory=list)
    decorators: list[Decorator] = field(default_factory=list)

    def __str__(self) -> str:
        decos = "".join(f"{d}\n" for d in self.decorators)
        idx = " ".join(str(i) for i in self.qubit_indices)
        return f"{decos}INPUT {self.code_name} {idx}".rstrip()


@dataclass
class OutputPort:
    """An ``OUTPUT CodeName qubit_indices...`` declaration."""

    code_name: str
    qubit_indices: list[int] = field(default_factory=list)
    decorators: list[Decorator] = field(default_factory=list)

    def __str__(self) -> str:
        decos = "".join(f"{d}\n" for d in self.decorators)
        idx = " ".join(str(i) for i in self.qubit_indices)
        return f"{decos}OUTPUT {self.code_name} {idx}".rstrip()


# ── GADGET-specific statements ────────────────────────────────────────


@dataclass
class ReadoutStatement:
    """A ``READOUT targets... [FLIP]`` declaration (alias for OBSERVABLE_INCLUDE).

    ``flip=True`` marks the readout as naturally flipped: its
    deterministic value is ``1`` in the absence of faults.
    """

    targets: list[ReadoutTargetItem] = field(default_factory=list)
    flip: bool = False
    decorators: list[Decorator] = field(default_factory=list)


@dataclass
class CheckStatement:
    """A ``CHECK targets... [FLIP]`` declaration (alias for DETECTOR).

    ``flip=True`` marks the check as naturally flipped: its expected
    parity is ``1`` in the absence of faults.
    """

    targets: list[Target] = field(default_factory=list)
    flip: bool = False
    decorators: list[Decorator] = field(default_factory=list)


@dataclass
class ErrorStatement:
    """An ``ERROR(p) targets...`` declaration.

    Specifies an error mechanism with probability ``p`` that flips the
    listed targets (checks ``C<i>``, residual Paulis ``X<i>`` etc.,
    readouts ``R<i>``, and/or logical Paulis ``LX<i>``).
    """

    probability: float
    targets: list[ErrorTarget] = field(default_factory=list)
    decorators: list[Decorator] = field(default_factory=list)


@dataclass
class ConditionalStatement:
    """A ``CONDITIONAL R<j> L<P><i>...`` or ``CONDITIONAL rec[-k] L<P><i>...`` declaration.

    When conditioned on a readout (``R<j>``), applies a logical Pauli
    correction via the ``logical_correction`` matrix (post-decoder).

    When conditioned on a measurement record (``rec[-k]`` or one of the
    absolute forms ``M<i>`` / ``IN<p>.S<s>`` / ``OUT<p>.S<s>``), applies
    a pre-decoder feedforward via the logical rows of
    ``physical_correction``.
    """

    condition: ReadoutTarget | MeasurementRefTarget
    targets: list[LogicalPauliTarget] = field(default_factory=list)
    decorators: list[Decorator] = field(default_factory=list)


@dataclass
class VirtualLogicalStatement:
    """A ``VIRTUAL LZ0 LX1 ...`` declaration inside a GADGET.

    Applies unconditional logical Pauli corrections on the output.
    Each logical target flips the constant column of the
    ``correction_propagation`` matrix for the anti-commuting output
    observables.
    """

    targets: list[LogicalPauliTarget] = field(default_factory=list)
    decorators: list[Decorator] = field(default_factory=list)


PropagateTerm = (
    LogicalPauliTarget | DestabilizerTarget | MeasurementRefTarget | ReadoutTarget
)


@dataclass
class PropagateStatement:
    """A ``PROPAGATE LX0 FROM ...`` declaration inside a GADGET.

    Pins one row of the output residual operator the runtime evaluates.
    The target on the left of ``FROM`` names the residual operator
    (e.g. ``PROPAGATE LZ0`` declares the logical-Z residual, which
    flips the ``LX`` observable when non-zero); the RHS is the explicit
    XOR of the listed contributions:

    .. code-block:: text

        residual[row] = correction_propagation[row] · input_observables
                      ⊕ physical_correction[row]    · raw_measurements
                      ⊕ logical_correction[row]     · decoded_readouts

    Each term after ``FROM`` contributes one bit:

    * ``LX<i>`` / ``LZ<i>``: input-frame logical column of
      ``correction_propagation``
    * ``IN<p>.DS<s>``: input-frame destabilizer column of
      ``correction_propagation`` (the syndrome bit of stabilizer
      ``s`` of INPUT port ``p``).  PROPAGATE never refers to OUTPUT
      destabilizers, so no ``OUT<p>.DS<s>`` form.
    * ``rec[-k]`` or the absolute ``M<i>``: internal physical
      measurement column of ``physical_correction``.  Virtual
      stabilizer measurements (``IN<p>.S<s>`` / ``OUT<p>.S<s>``) are
      NOT allowed here.
    * ``R<k>``: decoded-readout column of ``logical_correction`` —
      the readout-conditioned frame correction the runtime XORs on
      top of the natural-Heisenberg residual.

    The optional ``FLIP`` token sets the affine constant column of
    ``correction_propagation``.

    Output residual operators not covered by an explicit ``PROPAGATE``
    fall back to the natural-Heisenberg derivation composed with any
    ``VIRTUAL`` / ``CONDITIONAL`` shortcut contributions.
    """

    target: LogicalPauliTarget
    terms: list[PropagateTerm] = field(default_factory=list)
    flip: bool = False
    decorators: list[Decorator] = field(default_factory=list)


@dataclass
class PreselectStatement:
    """A ``PRESELECT rec[-k] <bit>`` declaration inside a GADGET.

    The condition may use either the relative ``rec[-k]`` form or one
    of the absolute forms (``M<i>`` / ``IN<p>.S<s>`` / ``OUT<p>.S<s>``);
    PRESELECT only accepts targets that resolve to an internal physical
    measurement.

    When the referenced physical measurement does not equal the expected
    bit, the simulator either discards the shot (resample mode, used by
    the static simulators) or replays from the beginning of the gadget
    (retry mode, used by the preselect simulators).
    """

    condition: MeasurementRefTarget
    expected_value: int
    decorators: list[Decorator] = field(default_factory=list)


GadgetStatement = (
    Instruction
    | RepeatBlock
    | InputPort
    | OutputPort
    | ReadoutStatement
    | CheckStatement
    | ErrorStatement
    | ConditionalStatement
    | VirtualLogicalStatement
    | PropagateStatement
    | PreselectStatement
)


@dataclass
class GadgetDefinition:
    """A ``GADGET Name { ... }`` definition."""

    name: str
    body: list[GadgetStatement] = field(default_factory=list)
    decorators: list[Decorator] = field(default_factory=list)
    source_file: str | None = None
    source_line: int | None = None

    @property
    def input_ports(self) -> list[InputPort]:
        return [s for s in self.body if isinstance(s, InputPort)]

    @property
    def output_ports(self) -> list[OutputPort]:
        return [s for s in self.body if isinstance(s, OutputPort)]


@dataclass
class GadgetApplication:
    """A gadget application: ``Name IN(indices) OUT(indices)``."""

    gadget_name: str
    in_indices: list[int] | None = None
    out_indices: list[int] | None = None
    decorators: list[Decorator] = field(default_factory=list)
    source_line: int | None = None

    @property
    def is_shortcut(self) -> bool:
        """True if this uses the shortcut form (IN=OUT inferred from qubit targets)."""
        return self.in_indices is None and self.out_indices is None

    def __str__(self) -> str:
        decos = "".join(f"{d}\n" for d in self.decorators)
        parts: list[str] = [self.gadget_name]
        if self.in_indices is not None:
            indices = " ".join(str(i) for i in self.in_indices)
            parts.append(f"IN({indices})")
        if self.out_indices is not None:
            indices = " ".join(str(i) for i in self.out_indices)
            parts.append(f"OUT({indices})")
        if self.in_indices is None and self.out_indices is None:
            parts.append("()")
        return f"{decos}{' '.join(parts)}"


@dataclass
class AssertStatement:
    """An ``ASSERT_EQ target value`` statement."""

    target: Target
    expected_value: int
    decorators: list[Decorator] = field(default_factory=list)

    def __str__(self) -> str:
        decos = "".join(f"{d}\n" for d in self.decorators)
        return f"{decos}ASSERT_EQ {self.target} {self.expected_value}"


@dataclass
class VirtualCorrection:
    """A ``VIRTUAL X0*Y1 wire`` Pauli correction pseudo-instruction."""

    paulis: list[tuple[str, int]]
    wire: int

    def __str__(self) -> str:
        parts = "*".join(f"{p}{q}" for p, q in self.paulis)
        return f"VIRTUAL {parts} {self.wire}"


@dataclass
class ConditionalCorrection:
    """A ``CONDITIONAL rec[-k] X0*Y1 wire`` conditional Pauli correction.

    Applies the logical Pauli product ``paulis`` to ``wire`` conditioned
    on the ``readout_offset``-th most recent logical readout in
    program/compose order (i.e. ``rec[-readout_offset]``).

    Each ``(pauli_letter, logical_qubit_index)`` entry is a logical Pauli
    on a logical qubit of the code carried by ``wire``.
    """

    readout_offset: int
    paulis: list[tuple[str, int]]
    wire: int

    def __str__(self) -> str:
        parts = "*".join(f"{p}{q}" for p, q in self.paulis)
        return f"CONDITIONAL rec[-{self.readout_offset}] {parts} {self.wire}"


ComposeStatement = (
    GadgetApplication
    | ConditionalCorrection
    | Instruction
    | RepeatBlock
    | InputPort
    | OutputPort
)


ProgramStatement = (
    GadgetApplication
    | AssertStatement
    | VirtualCorrection
    | ConditionalCorrection
    | Instruction
    | RepeatBlock
    | InputPort
    | OutputPort
)


@dataclass
class ComposeDefinition:
    """A ``COMPOSE Name { ... }`` definition."""

    name: str
    body: list[ComposeStatement] = field(default_factory=list)
    decorators: list[Decorator] = field(default_factory=list)
    source_file: str | None = None
    source_line: int | None = None

    @property
    def input_ports(self) -> list[InputPort]:
        return [s for s in self.body if isinstance(s, InputPort)]

    @property
    def output_ports(self) -> list[OutputPort]:
        return [s for s in self.body if isinstance(s, OutputPort)]


@dataclass
class ProgramDefinition:
    """A ``PROGRAM Name { ... }`` definition."""

    name: str
    body: list[ProgramStatement] = field(default_factory=list)
    decorators: list[Decorator] = field(default_factory=list)
    source_file: str | None = None
    source_line: int | None = None


# ── Top-level ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ImportStatement:
    """An ``IMPORT "path"`` statement."""

    path: str

    def __str__(self) -> str:
        return f'IMPORT "{self.path}"'


Definition = CodeDefinition | GadgetDefinition | ComposeDefinition | ProgramDefinition


@dataclass
class DeqFile:
    """A complete ``.deq`` file — a list of definitions."""

    definitions: list[Definition] = field(default_factory=list)
    imports: list[ImportStatement] = field(default_factory=list)
    source_file: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────


def _format_float(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def _indent(text: str) -> str:
    return "".join(f"    {line}\n" for line in text.splitlines())
