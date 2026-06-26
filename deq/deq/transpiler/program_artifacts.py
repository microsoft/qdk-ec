# pylint: disable=no-member
#   no-member: protobuf generated classes are not detected by pylint
"""Reusable building blocks for the ``.deq`` PROGRAM → JIT + Stim pipeline.

Multiple call sites (the ``Sampler`` wrapper, the ``deq transpile`` and
``deq simulate ler`` CLI commands) need to:

1. Classify a parsed ``.deq`` file's top-level definitions by kind so
   they can look up gadgets / codes / programs / composes by name.
2. Drive :func:`deq.transpiler.jit_library_builder.build_jit_library`,
   :func:`deq.cli.jit.compile_program_for_jit`, and
   :func:`deq.cli.jit.export_program_stim` in the same order to produce
   a :class:`~deq.proto.deq_jit_pb2.JitLibrary`, the compiled program
   instructions, and a Stim source string.
3. Derive the per-gadget measurement partition the runtime expects so
   each shot can be chunked into one ``Outcomes`` per gadget invocation.

This module centralizes (1)-(3) so callers can express the whole flow
as ``transpile_program(deq_file, "MyProgram")`` and pull individual
artifacts off the returned :class:`ProgramArtifacts`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from deq.circuit.model import (
    CodeDefinition,
    ComposeDefinition,
    GadgetDefinition,
    ProgramDefinition,
)
from deq.proto import deq_jit_pb2 as _jit_pb


__all__ = [
    "ProgramArtifacts",
    "ProgramDefinitions",
    "gadget_measurement_partition",
    "transpile_program",
]


@dataclass
class ProgramDefinitions:
    """Indexed view of a parsed ``.deq`` file's top-level definitions.

    Walks ``deq_file.definitions`` once and partitions each definition
    into a per-kind dictionary keyed by name. Avoids the repeated
    ``isinstance``-and-filter pattern that otherwise shows up at every
    call site that wants to look up a gadget or program by name.
    """

    programs: dict[str, ProgramDefinition] = field(default_factory=dict)
    gadgets: dict[str, GadgetDefinition] = field(default_factory=dict)
    codes: dict[str, CodeDefinition] = field(default_factory=dict)
    composes: dict[str, ComposeDefinition] = field(default_factory=dict)

    @classmethod
    def from_deq_file(cls, deq_file: Any) -> "ProgramDefinitions":
        """Classify every top-level definition of ``deq_file`` by kind."""
        result = cls()
        for definition in deq_file.definitions:
            if isinstance(definition, ProgramDefinition):
                result.programs[definition.name] = definition
            elif isinstance(definition, GadgetDefinition):
                result.gadgets[definition.name] = definition
            elif isinstance(definition, CodeDefinition):
                result.codes[definition.name] = definition
            elif isinstance(definition, ComposeDefinition):
                result.composes[definition.name] = definition
        return result

    def get_program(self, name: str) -> ProgramDefinition:
        """Look up a ``PROGRAM`` by name.

        Raises :class:`KeyError` whose message lists the available
        program names, so callers don't need to format their own.
        """
        program = self.programs.get(name)
        if program is None:
            raise KeyError(
                f"program {name!r} not found in .deq file; "
                f"available programs: {sorted(self.programs)!r}"
            )
        return program


@dataclass
class ProgramArtifacts:
    """Bundle of artifacts produced by transpiling one ``PROGRAM`` against
    its ``.deq`` file.

    Attributes
    ----------
    library:
        The :class:`~deq.proto.deq_jit_pb2.JitLibrary` produced by
        :func:`deq.transpiler.jit_library_builder.build_jit_library`,
        with the compiled instructions appended to its ``program``
        field.
    instructions:
        The compiled JIT instructions, in execution order. Equivalent
        to ``list(library.program)`` but cached so callers don't have
        to re-copy the protobuf repeated field.
    circuit:
        The Stim circuit string produced by
        :func:`deq.cli.jit.export_program_stim`.
    partition:
        Per-gadget measurement counts in ``instructions`` order — the
        shape the runtime uses to chunk each shot into one
        ``Outcomes`` message per gadget invocation.
    """

    library: _jit_pb.JitLibrary
    instructions: List[_jit_pb.JitInstruction]
    circuit: str
    partition: List[int]


def gadget_measurement_partition(
    library: _jit_pb.JitLibrary,
    instructions: Optional[List[_jit_pb.JitInstruction]] = None,
) -> List[int]:
    """Per-gadget measurement counts in instruction order.

    Returns the partition the runtime expects: one entry per
    instruction, giving the number of measurement bits that
    instruction's gadget type emits. Pass ``instructions=None`` to use
    ``library.program`` as the instruction sequence.
    """
    measurements_per_gtype = {
        gt.base.gtype: len(gt.base.measurements) for gt in library.gadget_types
    }
    sequence = instructions if instructions is not None else list(library.program)
    return [measurements_per_gtype[instr.gadget.gtype] for instr in sequence]


def transpile_program(
    deq_file: Any,
    program_name: str,
    *,
    decoder_data: bool = True,
) -> ProgramArtifacts:
    """Run the standard ``.deq`` → JIT + Stim pipeline for one program.

    Builds a fresh :class:`~deq.proto.deq_jit_pb2.JitLibrary` from the
    parsed ``deq_file``, compiles the named ``PROGRAM`` into JIT
    instructions, appends them to the library's ``program`` field, and
    exports the matching Stim circuit text.

    Parameters
    ----------
    deq_file:
        Parsed ``.deq`` file (e.g. from
        :func:`deq.circuit.parser.parse` or
        :func:`deq.circuit.parser.parse_file`).
    program_name:
        Name of the ``PROGRAM`` block to compile.
    decoder_data:
        When ``True`` (default), builds a full decoder-capable
        :class:`~deq.proto.deq_jit_pb2.JitLibrary` via
        :func:`deq.transpiler.jit_library_builder.build_jit_library`.
        Use this when the result library will be fed into a decoder /
        coordinator.

        When ``False``, builds a program-only library via
        :func:`deq.transpiler.jit_library_builder.build_jit_program`,
        skipping per-gadget stabilizer simulation, noise propagation,
        and check resolution. Typically more than an order of magnitude
        faster than the full build, and still supports VIRTUAL Pauli
        corrections in PROGRAM bodies.

    ``COMPOSE`` definitions are inlined into synthetic gadgets via
    :func:`deq.transpiler.compose_builder.compose_to_synthetic_gadget`
    so the Stim exporter can find their bodies. Nested COMPOSEs work as
    long as the inner one is declared first.
    """
    from deq.cli.jit import compile_program_for_jit, export_program_stim
    from deq.transpiler.compose_builder import compose_to_synthetic_gadget
    from deq.transpiler.jit_library_builder import (
        build_jit_library,
        build_jit_program,
    )
    from deq.transpiler.jit_transpiler import flatten_body

    definitions = ProgramDefinitions.from_deq_file(deq_file)
    program_def = definitions.get_program(program_name)

    if decoder_data:
        library = build_jit_library(deq_file)
    else:
        library = build_jit_program(deq_file)

    compiled_with_apps, _assertions = compile_program_for_jit(
        library, program_def, definitions.programs, definitions.codes
    )
    for instruction, _application in compiled_with_apps:
        library.program.append(instruction)

    # ``export_program_stim`` needs a GadgetDefinition for every gtype
    # it sees in the program; COMPOSEs only exist as synthetic gadgets
    # built by inlining their bodies. Process them in source order so
    # nested COMPOSEs can reference earlier ones.
    gadgets_for_stim: dict[str, Any] = dict(definitions.gadgets)
    compose_so_far: dict[str, Any] = {}
    for compose in definitions.composes.values():
        synthetic = compose_to_synthetic_gadget(
            compose, gadgets_for_stim, compose_so_far, definitions.codes
        )
        gadgets_for_stim[compose.name] = synthetic
        compose_so_far[compose.name] = compose

    gtype_to_name = {gt.base.gtype: gt.base.name for gt in library.gadget_types}
    program_applications = [
        application for _instruction, application in compiled_with_apps
    ]
    circuit = export_program_stim(
        library,
        gadgets_for_stim,
        gtype_to_name,
        flatten_body,
        program_def,
        program_applications,
    )

    instructions = list(library.program)
    return ProgramArtifacts(
        library=library,
        instructions=instructions,
        circuit=circuit,
        partition=gadget_measurement_partition(library, instructions),
    )
