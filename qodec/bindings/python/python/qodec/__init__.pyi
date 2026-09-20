"""Build, inspect, and save quantum error-correction protocols.

``Qodec`` and ``Layer`` organize a protocol. ``Code``, ``InstructionSet``,
``Instruction`` and ``Gadget`` describe its artifacts. ``Reference`` and
``Node`` address their declarations. The remaining types
live in ``qodec.codes``, ``qodec.gadgets``, ``qodec.instructions`` and
``qodec.actions``.
"""
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar, final, overload
from collections.abc import Callable, Iterable, Mapping, MutableMapping, MutableSequence, Sequence
from _typeshed import SupportsKeysAndGetItem
from abc import ABCMeta
from typing import Any
from dataclasses import dataclass

from . import actions as actions
from . import codes as codes
from . import gadgets as gadgets
from . import instructions as instructions
from ._nodes import Node as Node, SourceLocation as SourceLocation

if TYPE_CHECKING:
    from .codes import PauliExpression
    from .gadgets import Check, Circuit, Encoding, Readout, ReadoutLike
    from .instructions import Block, BlockOperand, InstructionCall, Parameter
from typing_extensions import Self, TypeAlias

__all__ = [
    "Code", "Gadget", "Instruction", "InstructionSet", "Layer", "Qodec",
    "Node", "SourceLocation", "Reference", "ReferenceLike", "QodecError", "QodecLoadError", "QodecSaveError",
    "__version__", "register", "actions", "codes", "gadgets", "instructions",
]

__version__: str
"""Installed package version (PEP 440), read from package metadata."""

@final
class Reference:
    """An immutable parsed address into a qodec model.

    For example, ``layers[0].gadgets["measure_z"]`` selects a gadget when passed
    to :meth:`Qodec.resolve`. ``segments`` describes the path as fields, literal
    mapping keys, indices, slices, and unions. ``path`` retains its authored text.
    A Reference has no owner or resolved value; a Node supplies those through
    resolution. The same path can be resolved against different model owners.

    Construction accepts only a string or an existing Reference. Other inputs
    raise TypeError; malformed syntax or empty selections raise ValueError.
    Construction never checks target existence or interprets circuit source.
    Indices are zero-based, slice stops exclusive, and steps positive.
    Selections preserve order and duplicates; slices remain compact.

    Parity fields separately restrict which addresses they accept, for example
    ``in[0].z[0]``.

    Equality and hashing use the authored path text, including spelling.
    Segment values compare by structure.
    """

    def __new__(cls, value: ReferenceLike) -> Self: ...
    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Self: ...
    def __replace__(self, **changes: Any) -> Self: ...

    @final
    @dataclass(frozen=True, slots=True)
    class Field:
        """A dotted model field, named by an ASCII identifier."""
        name: str
        def __post_init__(self) -> None: ...

    @final
    @dataclass(frozen=True, slots=True)
    class Key:
        """A literal mapping key, distinct from a model field."""
        value: str
        def __post_init__(self) -> None: ...

    @final
    @dataclass(frozen=True, slots=True)
    class Index:
        """One nonnegative position in a model sequence."""
        value: int
        def __post_init__(self) -> None: ...

    @final
    @dataclass(frozen=True, slots=True)
    class Slice:
        """Sequence positions from start to exclusive stop, with a positive step."""
        start: int
        stop: int
        step: int = 1
        def __post_init__(self) -> None: ...

    @final
    @dataclass(frozen=True, slots=True)
    class Union:
        """At least two nonnegative sequence positions, preserving order and duplicates."""
        indices: tuple[int, ...]
        def __post_init__(self) -> None: ...

    @property
    def path(self) -> str:
        """The original path text, including selector spelling."""
        ...

    @property
    def segments(self) -> tuple[Field | Key | Index | Slice | Union, ...]:
        """Immutable parsed path structure.

        Field and Key are distinct. Slices are compact; unions retain order
        and duplicates. Segment values compare structurally and support matching.
        """
        ...

    def expand(self) -> list[Reference]:
        """Expand the final index selector, preserving order and duplicates.

        With no final index or exactly one selected index, returns ``[self]`` and preserves
        its spelling. Otherwise returns single-index paths. For example,
        ``Reference("layers[0,2]").expand()`` returns references to ``layers[0]``
        and ``layers[2]``. Earlier selectors are retained, not broadcast over.
        """
        ...

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def _repr_pretty_(self, printer: Any, cycle: bool) -> None: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __ne__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...


ReferenceLike = Reference | str
"""A model address supplied as a path string or parsed Reference."""

_ReferenceKey = TypeVar("_ReferenceKey", bound=ReferenceLike)
_Default = TypeVar("_Default")

class _FrameMapping(MutableMapping[str, "Check"], metaclass=ABCMeta):
    def __getitem__(self, key: ReferenceLike) -> Check: ...
    def __setitem__(self, key: ReferenceLike, value: Sequence[ReferenceLike | Literal[0, 1]]) -> None: ...
    def __delitem__(self, key: ReferenceLike) -> None: ...
    @overload
    def get(self, key: ReferenceLike, /) -> Check | None: ...
    @overload
    def get(self, key: ReferenceLike, default: Check, /) -> Check: ...
    @overload
    def get(self, key: ReferenceLike, default: _Default, /) -> Check | _Default: ...
    @overload
    def pop(self, key: ReferenceLike, /) -> Check: ...
    @overload
    def pop(self, key: ReferenceLike, default: Check, /) -> Check: ...
    @overload
    def pop(self, key: ReferenceLike, default: _Default, /) -> Check | _Default: ...
    def setdefault(self, key: ReferenceLike, default: Sequence[ReferenceLike | Literal[0, 1]], /) -> Check: ...
    @overload
    def update(self, value: SupportsKeysAndGetItem[_ReferenceKey, Sequence[ReferenceLike | Literal[0, 1]]], /, **kwargs: Sequence[ReferenceLike | Literal[0, 1]]) -> None: ...
    @overload
    def update(self, value: Iterable[tuple[ReferenceLike, Sequence[ReferenceLike | Literal[0, 1]]]], /, **kwargs: Sequence[ReferenceLike | Literal[0, 1]]) -> None: ...
    @overload
    def update(self, **kwargs: Sequence[ReferenceLike | Literal[0, 1]]) -> None: ...

def register(
    parser: Callable[[str, InstructionSet], Sequence[InstructionCall]], *, format: str
) -> None:
    """Register a source parser for a format tag in the shared Rust registry.

    The callback takes ``(source, instruction_set)`` with an isolated snapshot
    of the target instruction set, and returns a sequence of
    InstructionCall values preserving execution and readout order, or raises
    an exception. It should be deterministic for unchanged inputs and not
    mutate the instruction set.

    Python and native registrations replace each other: the latest wins in
    the same linked Rust instance. Calls in progress keep their selected parser.
    Callbacks and replaced callback destructors run outside the registry lock.
    Invalid tags raise ValueError; non-callables raise TypeError.

    YAML is registered in Rust. The supplied Stim adapter registers on import
    if Stim is available. Python callbacks acquire interpreter access when
    invoked, including from native threads, and require their interpreter to
    remain running. Shutdown releases the callable; later calls return an error.
    Python calls preserve callback exceptions; native callers receive their
    description through the Rust error result. Configuration is not saved model
    data. Use Circuit.calls(parser=...) for a one-call override.
    """
    ...

PauliString: TypeAlias = str
"""Type-only alias, not available at runtime: a Pauli operator string, such
as ``'X_0 Z_1'``, returned by accessors."""

PauliLike: TypeAlias = str | PauliExpression
"""Type-only alias, not available at runtime: the inputs accepted for a Pauli."""

Metadata: TypeAlias = MutableMapping[str, Any]
"""Type-only alias, not available at runtime: a dictionary of annotations.

qodec stores annotations without interpreting their keys. They participate
in structural equality. Getters return live views, including nested containers;
item edits immediately update the owning object.
"""


# ── Exceptions ───────────────────────────────────────────────────────────────

class QodecError(Exception):
    """Base class for every qodec domain error."""

class QodecLoadError(QodecError):
    """Raised when reading, parsing, or validating a qodec or artifact fails."""

class QodecSaveError(QodecError):
    """Raised when preparing, serializing, or writing a qodec or artifact fails."""


# ── Package ──────────────────────────────────────────────────────────────────

@final
class Qodec:
    """Keep a protocol's layers and artifacts together, with references resolved."""

    def _node_source_location(self, path: str) -> tuple[Path, int] | None: ...
    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Self: ...
    def __replace__(self, **changes: Any) -> Self: ...

    def resolve(self, path: ReferenceLike) -> Node:
        """Resolve an exact model path. Empty selects the root.

        Accept a string or parsed Reference, with dotted fields, JSON-quoted
        mapping keys, and zero-based indices, slices, or unions. A selection
        returns a Node whose as_sequence() preserves order and duplicates.
        Invalid syntax raises ValueError; any missing selected target
        raises LookupError. Does not parse circuits, evaluate parity, or run
        analysis. Paths address resolved declarations, not the YAML file layout.
        """
        ...

    def __new__(
        cls,
        layers: Sequence["Layer"],
        *,
        name: str | None = ...,
        description: str | None = ...,
        schema_version: int | None = ...,
        metadata: Mapping[str, Any] | None = None,
    ) -> Self:
        """Build a qodec from already-resolved components.

        Put ``layers`` in order from logical to physical. The layer objects
        are shared, so edits to a layer are visible through this qodec too.
        ``name`` and ``description`` default to ``None`` and read back as
        empty strings. ``metadata`` defaults to an empty dictionary.

        ``schema_version`` defaults to ``None``, which uses the current
        version on save. An explicit unsupported version raises
        ``ValueError``.

        Construction does not check layer consistency. Call :meth:`validate`
        to check the completed protocol or after editing it.
        """
        ...

    @staticmethod
    def load(path: str | os.PathLike[str]) -> "Qodec":
        """Read a protocol from its manifest or multi-document YAML bundle.

        Pass the file's path, not its directory. The file may have any name.
        Directory paths are rejected even when they contain a valid manifest.
        Raises :class:`QodecLoadError` on I/O, parsing, resolution, or
        validation failure.
        """
        ...

    @staticmethod
    def loads(text: str) -> "Qodec":
        """Read a protocol from bundle text already held in memory.

        A self-contained bundle, as returned by :meth:`dumps`, needs no
        filesystem access. External paths resolve relative to the current
        working directory. Raises :class:`QodecLoadError` on parsing,
        resolution, or validation failure.
        """
        ...

    def validate(self) -> None:
        """Check preservation preconditions without file access.

        Raises ``ValueError`` for ambiguous definitions, inconsistent layer
        bindings, unaligned encodings, or derived roles that would change on
        reload. The bottom layer must have no gadgets because it has no target.
        This does not audit the protocol. Short layer lists, unequal logical
        lists, incomplete readouts, and invalid uses inside actions, circuit
        text, and parity equations can be preserved. Interpreting accessors
        and analysis routines check their own preconditions.
        """
        ...

    def dumps(self) -> str:
        """Return the whole protocol as a single-file bundle string.

        Runs :meth:`validate` before serialization.
        Raises :class:`QodecSaveError` if validation or serialization fails or a circuit
        source cannot be inlined. Unlike :meth:`save`, this cannot write
        separate source files.
        """
        ...

    def save(self, destination: str | os.PathLike[str], *, single_file: bool = False) -> Path:
        """Write the protocol to a directory, creating it if needed.

        The default, ``single_file=False``, writes separate artifact files.
        ``single_file=True`` writes a multi-document YAML bundle to
        ``destination/manifest_filename``, inlining circuit sources where
        possible. Sources that cannot be inlined are written alongside it.
        Bundles include current artifact values without reusing external files.

        Directory saves retain references to unchanged files outside the original
        manifest's directory. Modified external artifacts are copied locally;
        external input files are never overwritten. Reused files must still match
        their loaded text. Missing or changed files raise :class:`QodecSaveError`
        before writing. Files inside the original directory are copied normally.

        Returns the written manifest as a :class:`pathlib.Path`, equal to
        ``Path(destination) / self.manifest_filename``. Relative destinations
        remain relative, and ``..`` components are preserved. Pass the returned
        path directly to :meth:`load`.

        Saving uses the current objects and reuses compatible loaded artifact
        paths. Referenced codes unused by gadgets remain while their layer and
        block declaration remain. The manifest filename is preserved; YAML
        formatting may change. Runs :meth:`validate` before writing. Raises
        :class:`QodecSaveError` if validation, serialization, or writing fails.
        """
        ...

    @property
    def name(self) -> str: ...
    @name.setter
    def name(self, value: str) -> None: ...

    @property
    def description(self) -> str: ...
    @description.setter
    def description(self, value: str) -> None: ...

    @property
    def schema_version(self) -> int | None:
        """Declared schema version; ``None`` uses the current version on save.

        Assigning an unsupported version raises ``ValueError``.
        """
        ...
    @schema_version.setter
    def schema_version(self, value: int | None) -> None: ...

    @property
    def manifest_filename(self) -> str:
        """The manifest filename used inside the directory passed to ``save``.

        For a loaded bundle, this is the internal manifest filename, not
        necessarily the bundle's outer filename. Qodecs built through the
        constructor use ``"qodec.yaml"``.
        """
        ...
    @manifest_filename.setter
    def manifest_filename(self, value: str) -> None: ...

    @property
    def metadata(self) -> Metadata:
        """Live manifest annotations; see :data:`Metadata`."""
        ...
    @metadata.setter
    def metadata(self, value: Mapping[str, Any]) -> None: ...

    @property
    def layers(self) -> MutableSequence["Layer"]:
        """A live sequence of shared layers, ordered from logical to physical.

        Assign a sequence to replace the layers, or edit the live sequence.
        """
        ...
    @layers.setter
    def layers(self, value: Sequence["Layer"]) -> None: ...

    @property
    def instruction_sets(self) -> Mapping[str, "InstructionSet"]:
        """A read-only live index of shared layer instruction sets, keyed by name."""
        ...

    @property
    def codes(self) -> Mapping[str, "Code"]:
        """A read-only live index of shared codes, keyed by name.

        Includes layer code bindings and gadget input/output encodings.
        Explicit layer bindings remain after the last gadget is removed.
        """
        ...

    def slice(self, start: int, stop: int) -> "Qodec":
        """Keep a range of layers as a separate qodec.

        Selects layer indices ``start <= index < stop``. For example,
        ``protocol.slice(0, 2)`` keeps its first two layers.

        Bounds must satisfy ``0 <= start <= stop <= len(layers)``; they
        are not clipped. Reversed or oversized bounds raise ``ValueError``;
        negative bounds raise ``OverflowError``. Equal bounds give no layers.

        A slice with fewer than two layers is a partial model that can be
        saved. Audit reports it as an incomplete lowering protocol.

        Retained layers are shared, except the new bottom layer, which has
        no gadgets and shares its instruction set and bound codes. Changes to shared layers, gadgets,
        instruction sets, and codes are visible in both qodecs. Manifest fields are copied.
        """
        ...

    def __str__(self) -> str:
        """A summary of the layers, gadget counts, and referenced codes."""
        ...

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None: ...
    def __eq__(self, other: object, /) -> bool:
        """Value equality, comparing fields rather than identity. Instances are unhashable."""
        ...
    def __repr__(self) -> str: ...



@final
class Layer:
    """Pair an instruction set with its implementations in the next layer.

    The instruction set architecture (ISA) says which instructions this
    layer offers. Its gadgets implement those instructions using the next
    layer's instruction set. The bottom layer has no gadgets. instruction set and gadget objects
    are shared.
    """

    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Self: ...
    def __replace__(self, **changes: Any) -> Self: ...

    def __new__(
        cls,
        instruction_set: "InstructionSet",
        *,
        gadgets: Sequence["Gadget"] | Mapping[str, "Gadget"] | None = None,
        codes: Mapping[str, "Code"] | None = None,
    ) -> Self:
        """Build a layer from an instruction set and its gadgets.

        Omitted or ``None`` gadgets give an empty dictionary. A list is
        keyed by each gadget's ``implements.mnemonic``, the instruction name;
        duplicate mnemonics raise ``ValueError``. A dictionary's keys are
        used as supplied. Construction does not check instruction set membership.

        ``codes`` binds block type names to shared code definitions. It is sparse:
        gadget encodings supply omitted bindings. Explicit bindings survive gadget
        removal while the block remains declared by the instruction set. Bound
        codes and gadget encodings must agree when the protocol is validated.
        """
        ...

    @property
    def instruction_set(self) -> "InstructionSet":
        """The shared instruction set this layer presents."""
        ...
    @instruction_set.setter
    def instruction_set(self, value: "InstructionSet") -> None: ...

    @property
    def codes(self) -> MutableMapping[str, "Code"]:
        """A live mapping of explicit shared codes, keyed by layer block type.

        Gadget encodings can supply bindings omitted here. Assign a mapping
        to replace the explicit bindings, or edit the live mapping. Editing a code changes every shared
        reference to it. Replacing a code requires updating its encodings too.
        """
        ...
    @codes.setter
    def codes(self, value: Mapping[str, "Code"]) -> None: ...

    @property
    def gadgets(self) -> MutableMapping[str, "Gadget"]:
        """A live mapping of shared gadgets, keyed by source instruction set mnemonic.

        Assign a list or dictionary to replace the collection.
        """
        ...
    @gadgets.setter
    def gadgets(self, value: Sequence["Gadget"] | Mapping[str, "Gadget"]) -> None: ...

    def __str__(self) -> str:
        """A summary of the instruction set and gadget names, without expanding them."""
        ...
    def _repr_pretty_(self, printer: Any, cycle: bool) -> None: ...
    def __eq__(self, other: object, /) -> bool:
        """Value equality, comparing fields rather than identity. Instances are unhashable."""
        ...
    def __repr__(self) -> str: ...



# ── Instruction Sets ─────────────────────────────────────────────────────────

@final
class InstructionSet:
    """Describe the quantum instructions available at a layer of a protocol.

    This is the layer's instruction set architecture (ISA).

    Property assignments change this object wherever it is shared.
    Owned collection getters are live. Whole-property assignment replaces their contents.
    """

    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Self: ...
    def __replace__(self, **changes: Any) -> Self: ...

    def __str__(self) -> str:
        """The current declaration as YAML, without validation."""
        ...
    def _repr_pretty_(self, printer: Any, cycle: bool) -> None: ...

    def __new__(
        cls,
        name: str,
        *,
        description: str = ...,
        blocks: Sequence["Block"] = ...,
        instructions: Sequence["Instruction"] | Mapping[str, "Instruction"] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Self:
        """Build an instruction set; structural errors raise ``ValueError``.

        ``description`` defaults to ``""``. Omitted ``blocks`` and
        ``instructions`` are empty; ``instructions=None`` is also empty, but
        ``blocks=None`` raises ``TypeError``.
        ``metadata`` defaults to an empty dictionary.
        """
        ...

    @staticmethod
    def load(path: str | os.PathLike[str]) -> "InstructionSet":
        """Load and validate a standalone instruction-set YAML file.

        Raises :class:`QodecLoadError` if reading, parsing, or validation fails.
        """
        ...

    def save(self, path: str | os.PathLike[str]) -> None:
        """Validate and write a standalone instruction-set YAML file.

        Creates parent directories. Raises :class:`QodecSaveError` if
        validation, serialization, or writing fails.
        """
        ...

    @property
    def name(self) -> str: ...
    @name.setter
    def name(self, value: str) -> None: ...

    @property
    def description(self) -> str: ...
    @description.setter
    def description(self, value: str) -> None: ...

    @property
    def blocks(self) -> MutableSequence["Block"]:
        """Block type declarations."""
        ...
    @blocks.setter
    def blocks(self, value: Sequence["Block"]) -> None: ...

    @property
    def instructions(self) -> MutableMapping[str, "Instruction"]:
        """A live mapping of shared instructions, keyed by their mnemonics.

        The constructor and setter accept a list or dictionary. Dictionary
        keys must match each instruction's mnemonic. Duplicate
        mnemonics raise ``ValueError``.
        """
        ...
    @instructions.setter
    def instructions(self, value: Sequence["Instruction"] | Mapping[str, "Instruction"]) -> None: ...

    @property
    def metadata(self) -> Metadata:
        """Live annotations; see :data:`Metadata`."""
        ...
    @metadata.setter
    def metadata(self, value: Mapping[str, Any]) -> None: ...
    def __eq__(self, other: object, /) -> bool:
        """Value equality, comparing fields rather than identity. Instances are unhashable."""
        ...
    def __repr__(self) -> str: ...



@final
class Instruction:
    """Define an instruction's quantum blocks, classical inputs, and action steps.

    An instruction is shared and mutable; its mnemonic is read-only. Its classical outputs are the measurement
    results of :class:`qodec.actions.Observe` actions, in order, followed by
    named flags. Use :class:`qodec.actions.Condition` to describe when an
    action step runs.

    Actions address the flat logical indices contributed by the input and
    output block operands. An unconditional :class:`qodec.actions.Stabilize`
    can introduce temporary indices outside that range for later steps to
    use. At the end, every qubit absent from ``outputs`` is traced out,
    without emitting an outcome. Temporary qubits do not become operands
    or survive between instruction calls.
    """

    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Self: ...
    def __replace__(self, **changes: Any) -> Self: ...

    def __str__(self) -> str:
        """The instruction's YAML declaration, including action guards, without validation."""
        ...
    def _repr_pretty_(self, printer: Any, cycle: bool) -> None: ...

    def __new__(
        cls,
        mnemonic: str,
        *,
        description: str = ...,
        inputs: Sequence["BlockOperand"] = ...,
        outputs: Sequence["BlockOperand"] = ...,
        flags: Sequence[str] = ...,
        parameters: Sequence["Parameter"] = ...,
        action: Sequence["Action"] = ...,
        metadata: Mapping[str, Any] | None = None,
    ) -> Self:
        """Build an instruction value.

        ``description`` defaults to ``""``; omitted lists are empty.
        ``metadata`` defaults to an empty dictionary. instruction set-level validation
        runs when the instruction is added to a new :class:`InstructionSet`.
        """
        ...

    @property
    def mnemonic(self) -> str: ...

    @property
    def description(self) -> str: ...

    @description.setter
    def description(self, value: str) -> None: ...

    @property
    def inputs(self) -> MutableSequence["BlockOperand"]:
        """Quantum block operands taken as input, in declaration order.

        Each operand names a block type. This list is spelled ``in:`` on disk.
        :attr:`Gadget.inputs` aligns with this list one entry at a time.
        """
        ...

    @inputs.setter
    def inputs(self, value: Sequence["BlockOperand"]) -> None: ...

    @property
    def outputs(self) -> MutableSequence["BlockOperand"]:
        """Quantum block operands produced as output, in declaration order.

        Each operand names a block type. This list is spelled ``out:`` on disk.
        :attr:`Gadget.outputs` aligns with this list one entry at a time.
        """
        ...

    @outputs.setter
    def outputs(self, value: Sequence["BlockOperand"]) -> None: ...

    @property
    def flags(self) -> MutableSequence[str]:
        """Named classical bits the instruction reports alongside its outcomes.

        Their parity equations are declared in :attr:`Gadget.readouts`.
        """
        ...

    @flags.setter
    def flags(self, value: Sequence[str]) -> None: ...

    @property
    def observe_count(self) -> int:
        """Number of outcome bits declared by Observe steps, excluding named flags."""
        ...

    @property
    def parameters(self) -> MutableSequence["Parameter"]:
        """The named classical inputs this instruction expects.

        A parameter declares an input; an argument supplies its value in a call.
        A ``bit`` parameter is a runtime classical input bit, referenced by
        name in conditions; parameters of the remaining kinds take
        compile-time literal arguments at the call site.
        """
        ...

    @parameters.setter
    def parameters(self, value: Sequence["Parameter"]) -> None: ...

    @property
    def action(self) -> MutableSequence["Action"]:
        """The steps that specify what this instruction does, in order."""
        ...

    @action.setter
    def action(self, value: Sequence["Action"]) -> None: ...

    @property
    def metadata(self) -> Metadata:
        """Live annotations; nested mappings and sequences also write through."""
        ...
    @metadata.setter
    def metadata(self, value: Mapping[str, Any]) -> None: ...

    def __eq__(self, other: object, /) -> bool:
        """Value equality: two instructions are equal when all fields match."""
        ...


# ── Actions ──────────────────────────────────────────────────────────────────
#
# Action types (Stabilize, Clifford, Pauli, Observe, Rotate, Condition)
# live in the ``qodec.actions`` submodule. They are imported here
# only to build the ``Action`` type alias used in `Instruction` annotations;
# they are not exposed as top-level ``qodec.*`` names (use a submodule import).

if TYPE_CHECKING:
    from .actions import (
        Clifford,
        Condition,
        Observe,
        Pauli,
        Rotate,
        Stabilize,
    )

Action: TypeAlias = Stabilize | Clifford | Pauli | Observe | Rotate
"""Type-only alias, not available at runtime: an instruction's action step."""


# ── Codes ────────────────────────────────────────────────────────────────────

@final
class Code:
    """Describe an error-correcting code by its stabilizers and logical operators.

    Stabilizers define which states belong to the code. ``x[i]`` and ``z[i]``
    are the paired logical operators for logical qubit ``i``.
    Code operators are unsigned: explicit ``+`` and ``-`` signs are rejected.

    Property assignments change this object wherever it is shared.
    Lists and metadata are live views; assignments and collection edits update the code.
    """

    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Self: ...
    def __replace__(self, **changes: Any) -> Self: ...

    def __str__(self) -> str:
        """The current code declaration as YAML, without validation or analysis."""
        ...
    def _repr_pretty_(self, printer: Any, cycle: bool) -> None: ...

    def __new__(
        cls,
        name: str,
        stabilizers: Sequence[PauliLike],
        x: Sequence[PauliLike],
        z: Sequence[PauliLike],
        *,
        description: str = ...,
        metadata: Mapping[str, Any] | None = None,
    ) -> Self:
        """Check Pauli syntax; malformed tokens raise ``ValueError``.

        ``description`` defaults to ``""`` and ``metadata`` to an empty
        dictionary. ``x`` and ``z`` can be edited independently. Matching
        lengths, commutation, and other requirements are checked by audit,
        not by construction, loading, or saving.
        """
        ...

    @staticmethod
    def load(path: str | os.PathLike[str]) -> "Code":
        """Load and validate a standalone code-definition YAML file.

        Raises :class:`QodecLoadError` if reading, parsing, or validation fails.
        """
        ...

    def save(self, path: str | os.PathLike[str]) -> None:
        """Validate and write a standalone code-definition YAML file.

        Creates parent directories. Raises :class:`QodecSaveError` if
        validation, serialization, or writing fails.
        """
        ...

    @property
    def name(self) -> str: ...
    @name.setter
    def name(self, value: str) -> None: ...

    @property
    def description(self) -> str: ...
    @description.setter
    def description(self, value: str) -> None: ...

    @property
    def stabilizers(self) -> MutableSequence[PauliString]: ...
    @stabilizers.setter
    def stabilizers(self, value: Sequence[PauliLike]) -> None: ...

    @property
    def logical_count(self) -> int:
        """Number of declared logical X operators, equal to ``len(x)``.

        The Z list can differ while drafting; this property does not validate it.
        """
        ...

    @property
    def physical_qubit_count(self) -> int:
        """One more than the highest index in well-formed Pauli tokens.

        Includes identity tokens: ``I_7`` declares eight qubits. Missing
        indices default to zero. Malformed tokens are ignored; an empty code
        or one with no valid tokens counts as zero. This property does not
        validate the code; construction, load, and save reject malformed Paulis.
        Indices must be less than the host's ``usize::MAX`` so the count fits.
        """
        ...

    @property
    def x(self) -> MutableSequence[PauliString]: ...
    @x.setter
    def x(self, value: Sequence[PauliLike]) -> None: ...

    @property
    def z(self) -> MutableSequence[PauliString]: ...
    @z.setter
    def z(self, value: Sequence[PauliLike]) -> None: ...

    @property
    def metadata(self) -> Metadata:
        """Live annotations; see :data:`Metadata`."""
        ...
    @metadata.setter
    def metadata(self, value: Mapping[str, Any]) -> None: ...
    def __eq__(self, other: object, /) -> bool:
        """Value equality, comparing fields rather than identity. Instances are unhashable."""
        ...
    def __repr__(self) -> str: ...



@final
class Gadget:
    """Describe how a circuit implements one instruction.

    Encodings say which code and circuit labels represent each input or
    output block. Parity equations relate circuit output bits and encoding
    signs using XOR, which is 1 when an odd number of terms are 1.

    Equations use :class:`Reference` paths. For example,
    ``in[0].stabilizers[0]`` addresses the first input encoding's first
    stabilizer sign; ``out[0].z[1]`` addresses the first output encoding's
    second logical-Z sign. Load and save gadgets through :class:`Qodec`,
    which resolves their instruction-set and code references.
    """

    def resolve(self, path: ReferenceLike) -> Node:
        """Resolve relative to this gadget without interpreting circuit source.

        Nodes use this gadget's identity, gadget-relative paths, and no source
        locations. Selections return a Node whose as_sequence() retains order
        and duplicates. Any missing selected target raises LookupError.
        """
        ...

    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Self: ...
    def __replace__(self, **changes: Any) -> Self: ...

    def __str__(self) -> str:
        """A layer-relative YAML snippet, not a self-contained bundle.

        The layer supplies the implemented instruction, target instruction set,
        and code bindings. Circuit source is verbatim inline text with its
        effective format. No circuit parsing or validation is performed.
        For an unmatched encoding in a draft, its code name labels the block.
        """
        ...
    def _repr_pretty_(self, printer: Any, cycle: bool) -> None: ...

    def __new__(
        cls,
        implements: "Instruction",
        circuit: "Circuit",
        *,
        inputs: Sequence["Encoding"] = ...,
        outputs: Sequence["Encoding"] = ...,
        checks: Sequence[Sequence[ReferenceLike | Literal[0, 1]]] = ...,
        readouts: Sequence["ReadoutLike"] | None = ...,
        frames: Mapping[_ReferenceKey, Sequence[ReferenceLike | Literal[0, 1]]] | None = None,
        parameter_bindings: dict[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Self:
        """Connect an instruction to a shared circuit and its declared relations.

        Omitted ``inputs``, ``outputs``, ``checks``, and ``readouts`` are
        empty. Input and output encoding counts must match the implemented
        instruction's operands, or construction raises ``ValueError``.
        Checks are optional. Empty ``checks`` and ``readouts`` lists declare
        no check or readout relations.

        Each readout is a returned :class:`qodec.gadgets.Readout`, a parity
        sequence, or a single-key ``{name: equation}`` mapping. Names and
        equations are copied; positions and flag roles are computed against
        this instruction. ``Observe`` outcomes come first, then flags.
        ``readouts=None`` gives an empty tuple. Lists and tuples are accepted
        for both the outer collections and their equations.

        ``parameter_bindings`` maps instruction parameter names to circuit
        source parameter names, for example ``{"theta": "angle"}``.
        It and ``metadata`` default to empty dictionaries.
        """
        ...

    @property
    def implements(self) -> "Instruction":
        """The instruction this gadget implements."""
        ...
    @implements.setter
    def implements(self, value: "Instruction") -> None: ...

    @property
    def circuit(self) -> "Circuit":
        """The shared circuit; changing it is visible on this gadget."""
        ...
    @circuit.setter
    def circuit(self, value: "Circuit") -> None: ...

    @property
    def inputs(self) -> MutableSequence["Encoding"]:
        """Input encodings aligned with :attr:`Instruction.inputs`.

        Entry ``i`` is addressed as ``in[i]`` in parity references.
        Returns a live sequence of shared encoding objects. Both encoding
        changes and sequence edits change this gadget immediately.
        """
        ...
    @inputs.setter
    def inputs(self, value: Sequence["Encoding"]) -> None: ...

    @property
    def outputs(self) -> MutableSequence["Encoding"]:
        """Output encodings aligned with :attr:`Instruction.outputs`.

        Entry ``i`` is addressed as ``out[i]`` in parity references.
        Returns a live sequence of shared encoding objects, as :attr:`inputs` does.
        """
        ...
    @outputs.setter
    def outputs(self, value: Sequence["Encoding"]) -> None: ...

    @property
    def parameter_bindings(self) -> MutableMapping[str, str]:
        """Connect instruction parameter names to circuit source parameter names.

        Returns a live, sparse ``{instruction_parameter: source_parameter}`` map.
        In ``{"theta": "angle"}``, the value supplied for the instruction's
        ``theta`` is passed to ``angle`` in the circuit source. The dictionary
        stores names, not the supplied values. On disk, ``angle`` is written
        ``circuit.source.angle``.
        Assign a dictionary to replace the bindings.
        """
        ...
    @parameter_bindings.setter
    def parameter_bindings(self, value: Mapping[str, str]) -> None: ...

    @property
    def checks(self) -> MutableSequence["Check"]:
        """Relations declared to have zero parity, as a live sequence of immutable tuples.

        Each :data:`qodec.gadgets.Check` equation is declared to XOR to zero
        on a noiseless +1-codeword execution. The default is empty.
        Assign new equations to replace them; returned tuples are snapshots.
        The setter accepts reference strings, :class:`Reference`
        objects, and integer bits 0 or 1, including getter results.
        """
        ...
    @checks.setter
    def checks(self, value: Sequence[Sequence[ReferenceLike | Literal[0, 1]]]) -> None: ...

    @property
    def readouts(self) -> MutableSequence["Readout"]:
        """Output-bit equations as a live sequence of :class:`qodec.gadgets.Readout` values.

        Positions before the instruction's observe count are observables
        (measurement results); later positions are flags. Assign a sequence
        to replace them. Existing ``Readout`` values preserve names and
        equations, with position and role recomputed here. Parity sequences
        and single-key ``{name: equation}`` mappings are also accepted.
        Equation terms may be strings, ``Reference`` objects, or integer bits 0 or 1. Returned
        readouts and their equations are immutable snapshots.
        """
        ...
    @readouts.setter
    def readouts(self, value: Sequence["ReadoutLike"]) -> None: ...

    @property
    def frames(self) -> _FrameMapping:
        """Additional output logical-sign corrections, as a live sparse map.

        Keys select one ``out[entry].x[index]`` or ``out[entry].z[index]`` sign.
        Construction, assignment, and mapping operations accept strings or
        References as keys. Iteration returns their authored path strings.
        Values are XORs of circuit readouts, integer bits 0 or 1, and readout
        aliases resolving entirely to those terms. Input and output encoding
        signs are not permitted, even through aliases. These are correction
        definitions, not zero-parity checks. A Z-sign correction represents a logical X Pauli,
        and conversely. Missing entries and empty equations mean no additional
        correction; they do not reset the incoming frame. Defaults to ``{}``.
        Terms are immutable tuples of References and integer bits 0 or 1.
        Assign a map to replace it. Booleans and other numbers are rejected.
        Audit checks target bounds and whether terms are available classical bits.
        """
        ...
    @frames.setter
    def frames(self, value: Mapping[_ReferenceKey, Sequence[ReferenceLike | Literal[0, 1]]]) -> None: ...

    @property
    def metadata(self) -> Metadata:
        """Live annotations; see :data:`Metadata`."""
        ...
    @metadata.setter
    def metadata(self, value: Mapping[str, Any]) -> None: ...
    def __eq__(self, other: object, /) -> bool:
        """Value equality, comparing fields rather than identity. Instances are unhashable."""
        ...
    def __repr__(self) -> str: ...


# ── Circuit IR ───────────────────────────────────────────────────────────────
