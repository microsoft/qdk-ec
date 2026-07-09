# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint
import os
import re
from collections.abc import Sequence
from typing import NamedTuple

import arguably
import deq.proto.deq_jit_pb2 as jit_pb
import deq.proto.deq_bin_pb2 as pb
import deq.proto.util_pb2 as util_pb
from deq.compiler.jit_compiler import static_jit_compiler


@arguably.command
def transpile(
    *deq_files: str,
    #: the binary .deq.jit file to output
    out: str | None = None,
    #: name of the PROGRAM block to compile into the .deq.jit's program field;
    #: when set, also writes a sibling .stim file with the concatenated bodies
    program: str | None = None,
    #: number of parallel worker processes for GADGET type construction;
    #: defaults to (logical CPU count - 2), minimum 1
    jobs: int = max((os.cpu_count() or 1) - 2, 1),
    #: register an external check plugin from a .py file (makes the
    #: file's stem name available as a @CHECKS("name") value)
    plugin: list[str] | None = None,
    #: Mako variable definitions, each as key=value
    #: (e.g. --mako d=3 --mako p=0.01); implies --skip-mako-warning
    mako: list[str] | None = None,
    #: suppress the interactive Mako safety prompt
    skip_mako_warning: bool = False,
) -> None:
    """
    Transpile .deq files into a .deq.jit library (new pipeline).

    Reads one or more ``.deq`` files (imports inlined automatically by
    the parser), merges all definitions, and produces a ``JitLibrary``
    via :func:`deq.transpiler.jit_library_builder.build_jit_library`.
    All tags are preserved; use a separate tool to strip them if needed.

    When ``--program NAME`` is given, the named ``PROGRAM`` block is
    compiled into the library's ``program`` field by linking
    ``GadgetApplication`` wires through the gadgets' input/output ports,
    and a ``.stim`` file is emitted next to ``--out`` containing the
    concatenated bodies of the gadgets in invocation order.
    """
    from deq.circuit.model import (
        DeqFile,
    )
    from deq.circuit.parser import render_and_parse_files
    from deq.transpiler.jit_library_builder import build_jit_library
    from deq.circuit.mako_support import parse_mako_vars

    if not deq_files:
        raise ValueError("at least one .deq file is required")

    mako_vars = parse_mako_vars(mako) if mako else None

    if plugin:
        from deq.transpiler.check_plugins import register_plugin_file

        for p in plugin:
            name = register_plugin_file(p)
            print(f"Registered check plugin: {name!r} from {p}")

    merged = render_and_parse_files(
        list(deq_files), mako_defs=mako_vars, skip_mako_warning=skip_mako_warning
    )

    jit_library = build_jit_library(merged, jobs=jobs)

    if out is None:
        base = deq_files[0]
        if base.endswith(".deq"):
            base = base[:-4]
        out = f"{base}.deq.jit"

    jit_compile_program_to_file(jit_library, merged, out, program=program)


def jit_compile_program_to_file(
    jit_library: jit_pb.JitLibrary,
    merged: "DeqFile",  # noqa: F821
    out: str,
    *,
    program: str | None = None,
) -> None:
    """Compile a PROGRAM block into *jit_library* and write output files.

    Finds the named ``PROGRAM`` in *merged*, compiles it into JIT
    instructions (clearing any previous program), appends them to
    *jit_library.program*, and writes the ``.deq.jit``,
    ``.deq.jit.txt``, and companion ``.stim`` files.

    When *program* is ``None`` the library is written as-is (no program
    compilation, no ``.stim`` output).

    This is the public entry point for "recompile a program against an
    existing JitLibrary" — used by ``sample --jit``.
    """
    from deq.circuit.model import (
        CodeDefinition,
        ComposeDefinition,
        GadgetDefinition,
        ProgramDefinition,
    )
    from deq.transpiler.jit_transpiler import flatten_body
    from deq.transpiler.jit_annotate import expand_compose_circuit

    program_def: ProgramDefinition | None = None
    if program is not None:
        del jit_library.program[:]
        for d in merged.definitions:
            if isinstance(d, ProgramDefinition) and d.name == program:
                program_def = d
                break
        if program_def is None:
            available = [
                d.name for d in merged.definitions if isinstance(d, ProgramDefinition)
            ]
            raise ValueError(
                f"no PROGRAM named {program!r} found; available: {available}"
            )
        compiled, assertions = compile_program_for_jit(
            jit_library,
            program_def,
            {d.name: d for d in merged.definitions if isinstance(d, ProgramDefinition)},
            codes={
                d.name: d for d in merged.definitions if isinstance(d, CodeDefinition)
            },
        )
        for instr, _src in compiled:
            jit_library.program.append(instr)

    with open(out, "wb") as f:
        f.write(jit_library.SerializeToString())
    with open(f"{out}.txt", "w", encoding="utf8") as f:
        f.write(str(jit_library))

    print(f"Generated JIT library: {out}")
    print(f"  Port types: {len(jit_library.port_types)}")
    print(f"  Gadget types: {len(jit_library.gadget_types)}")
    if program_def is not None:
        print(f"  Program instructions: {len(jit_library.program)}")

        gadgets_by_name: dict[str, GadgetDefinition] = {
            d.name: d for d in merged.definitions if isinstance(d, GadgetDefinition)
        }
        compose_defs: dict[str, ComposeDefinition] = {
            d.name: d for d in merged.definitions if isinstance(d, ComposeDefinition)
        }
        code_defs: dict[str, CodeDefinition] = {
            d.name: d for d in merged.definitions if isinstance(d, CodeDefinition)
        }
        # Expand each ComposeDefinition into a synthetic GadgetDefinition
        # so the stim exporter can inline its circuit body.
        known_names = set(gadgets_by_name) | set(compose_defs)
        for cname, cdef in compose_defs.items():
            in_ports, circuit_stmts, out_ports = expand_compose_circuit(
                cdef, gadgets_by_name, compose_defs, known_names, code_defs
            )
            gadgets_by_name[cname] = GadgetDefinition(
                name=cname,
                body=list(in_ports) + list(circuit_stmts) + list(out_ports),  # type: ignore[arg-type]
            )
        gtype_to_name = {gt.base.gtype: gt.base.name for gt in jit_library.gadget_types}
        stim_text = export_program_stim(
            jit_library,
            gadgets_by_name,
            gtype_to_name,
            flatten_body,
            program_def,
            [src for _instr, src in compiled],
            assertions,
        )
        stim_out = _stim_path_for(out)
        with open(stim_out, "w", encoding="utf8") as f:
            f.write(stim_text)
        print(f"Generated stim circuit: {stim_out}")


class _WireProducer(NamedTuple):
    """Producer info for a wire in a PROGRAM body.

    *gid* and *port* identify which gadget output port produced the wire;
    *ptype* is that port's port type; *desc* is a human-readable string
    used in error messages (e.g. ``"step 3 (PrepareZ at line 17, output
    port 0)"``).
    """

    gid: int
    port: int
    ptype: int
    desc: str


def _stim_path_for(jit_out_path: str) -> str:
    """Derive a sibling ``.stim`` path from a ``.deq.jit`` output path."""
    base = jit_out_path
    for suffix in (".jit", ".deq"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return f"{base}.stim"


def _format_application(application: object) -> str:
    """Render a ``GadgetApplication`` like the source: ``Name in [-> out]``."""
    from deq.circuit.model import GadgetApplication

    assert isinstance(application, GadgetApplication)
    parts = [application.gadget_name]
    in_idx = application.in_indices or []
    out_idx = application.out_indices or []
    if in_idx == out_idx:
        if in_idx:
            parts.append(" ".join(str(i) for i in in_idx))
    else:
        if in_idx:
            parts.append("IN(" + " ".join(str(i) for i in in_idx) + ")")
        if out_idx:
            parts.append("OUT(" + " ".join(str(i) for i in out_idx) + ")")
    return " ".join(parts)


def _format_source_line(line_no: int | None, program_def: object) -> str:
    """Render a trailing ``  (file.deq:N)`` annotation, or empty string."""
    if line_no is None:
        return ""
    source = getattr(program_def, "source_file", None)
    if source:
        return f"  ({os.path.basename(source)}:{line_no})"
    return f"  (line {line_no})"


def _is_synthesised_identity_gadget(name: str) -> bool:
    """Return ``True`` if *name* is a synthesised identity gadget
    emitted by :func:`emit_conditional_correction_instruction`.

    Such gadgets host a ``remote_conditional_correction`` modifier
    that applies a Pauli frame correction conditioned on a previous
    logical readout.  They have no source ``GadgetDefinition`` (they
    are created on the fly by the COMPOSE / PROGRAM compiler), no
    measurements, and pass each input wire's physical qubits straight
    through to the matching output port.
    """
    return name.startswith("__identity_pt") and name.endswith("__")


def _program_source_lines(
    program_def: object,
    applications: list[object],
) -> list[int | None]:
    """Map each program ``GadgetApplication`` to its source line number.

    Re-reads ``program_def.source_file``, locates the ``PROGRAM <name> {``
    block, and walks its body lines in order. Lines whose first non-
    whitespace token equals an application's gadget name are matched in
    order to the applications. If the source file is missing or the
    block can't be located, returns ``[None] * len(applications)``.
    """
    fallback: list[int | None] = [None] * len(applications)

    name = getattr(program_def, "name", None)
    source = getattr(program_def, "source_file", None)
    if not source or not name:
        return fallback
    try:
        text = open(source, encoding="utf-8").read()
    except OSError:
        return fallback

    lines = text.splitlines()
    header_re = re.compile(rf"^\s*PROGRAM\s+{re.escape(name)}\b")
    start_idx: int | None = None
    for i, line in enumerate(lines):
        if header_re.match(line):
            start_idx = i
            break
    if start_idx is None:
        return fallback

    # Find matching closing brace (assume single-level braces).
    end_idx = len(lines)
    for i in range(start_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped == "}" or stripped.startswith("}"):
            end_idx = i
            break

    result: list[int | None] = []
    cursor = start_idx + 1
    for app in applications:
        target = getattr(app, "gadget_name", None)
        found: int | None = None
        for i in range(cursor, end_idx):
            stripped = lines[i].strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Strip inline comment.
            code = stripped.split("#", 1)[0].strip()
            if not code:
                continue
            first = code.split()[0] if code.split() else ""
            if first == target:
                found = i + 1  # 1-based line number
                cursor = i + 1
                break
        result.append(found)
    return result


def _analyze_program_wires(
    body: Sequence[object],
) -> tuple[list[int], list[int]]:
    """Return ``(input_wires, output_wires)`` for a program body.

    Simulates wire production/consumption sequentially:
    *input_wires* — consumed before being produced (need external supplier).
    *output_wires* — still live (produced but not consumed) after all stmts.
    Both lists are sorted and deduplicated.
    """
    from deq.circuit.model import (
        GadgetApplication,
        RepeatBlock,
    )

    live: set[int] = set()
    external_inputs: set[int] = set()

    def _scan(stmts: Sequence[object]) -> None:
        for s in stmts:
            if isinstance(s, RepeatBlock):
                _scan(s.body)
            elif isinstance(s, GadgetApplication):
                if s.in_indices:
                    for w in s.in_indices:
                        if w in live:
                            live.discard(w)
                        else:
                            external_inputs.add(w)
                if s.out_indices:
                    live.update(s.out_indices)

    _scan(body)
    return sorted(external_inputs), sorted(live)


def _remap_stmt(
    stmt: object,
    wire_map: dict[int, int],
) -> object:
    """Return a copy of *stmt* with wire indices remapped."""
    from deq.circuit.model import (
        AssertStatement,
        GadgetApplication,
        Instruction,
        QubitTarget,
        RepeatBlock,
        VirtualCorrection,
    )

    if isinstance(stmt, GadgetApplication):
        return GadgetApplication(
            gadget_name=stmt.gadget_name,
            in_indices=(
                [wire_map[w] for w in stmt.in_indices]
                if stmt.in_indices
                else stmt.in_indices
            ),
            out_indices=(
                [wire_map[w] for w in stmt.out_indices]
                if stmt.out_indices
                else stmt.out_indices
            ),
            decorators=stmt.decorators,
        )
    if isinstance(stmt, VirtualCorrection):
        return VirtualCorrection(paulis=stmt.paulis, wire=wire_map[stmt.wire])
    if isinstance(stmt, Instruction):
        new_targets = [
            (
                QubitTarget(index=wire_map[t.index])
                if isinstance(t, QubitTarget) and t.index in wire_map
                else t
            )
            for t in stmt.targets
        ]
        return Instruction(
            name=stmt.name,
            tag=stmt.tag,
            arguments=stmt.arguments,
            targets=new_targets,
            decorators=stmt.decorators,
        )
    if isinstance(stmt, RepeatBlock):
        return RepeatBlock(
            count=stmt.count,
            body=[_remap_stmt(s, wire_map) for s in stmt.body],
            decorators=stmt.decorators,
        )
    if isinstance(stmt, AssertStatement):
        return stmt
    return stmt


def _flatten_repeats(
    body: Sequence[object],
) -> list[object]:
    """Unroll all ``RepeatBlock``s in *body* (recursively)."""
    from deq.circuit.model import RepeatBlock

    result: list[object] = []
    for s in body:
        if isinstance(s, RepeatBlock):
            unrolled = _flatten_repeats(s.body)
            for _ in range(s.count):
                result.extend(unrolled)
        else:
            result.append(s)
    return result


def _expand_program_calls(
    body: Sequence[object],
    program_defs: dict[str, object],
    gtype_names: set[str],
    wire_counter: list[int],
    expanding: frozenset[str] = frozenset(),
) -> list[object]:
    """Expand sub-program references in *body*, returning a flat statement list.

    *program_defs* maps program names to their definitions.
    *gtype_names* is the set of known gadget/compose names (not expanded).
    *wire_counter* is a mutable ``[next_id]`` used to allocate fresh wire IDs.
    *expanding* tracks the call stack for cycle detection.
    """
    from deq.circuit.model import (
        GadgetApplication,
        Instruction,
        ProgramDefinition,
        RepeatBlock,
        VirtualCorrection,
    )

    result: list[object] = []
    for stmt in body:
        if isinstance(stmt, RepeatBlock):
            expanded_inner = _expand_program_calls(
                stmt.body, program_defs, gtype_names, wire_counter, expanding
            )
            for _ in range(stmt.count):
                result.extend(expanded_inner)
            continue

        name: str | None = None
        caller_in: list[int] | None = None
        caller_out: list[int] | None = None
        if (
            isinstance(stmt, GadgetApplication)
            and stmt.gadget_name in program_defs
            and stmt.gadget_name not in gtype_names
        ):
            name = stmt.gadget_name
            caller_in = stmt.in_indices or []
            caller_out = stmt.out_indices or []
        elif (
            isinstance(stmt, Instruction)
            and stmt.name in program_defs
            and stmt.name not in gtype_names
        ):
            from deq.circuit.model import QubitTarget

            name = stmt.name
            indices = [t.index for t in stmt.targets if isinstance(t, QubitTarget)]
            sub_prog = program_defs[name]
            assert isinstance(sub_prog, ProgramDefinition)
            sub_in, sub_out = _analyze_program_wires(sub_prog.body)
            caller_in = indices[: len(sub_in)]
            caller_out = indices[: len(sub_out)]

        if name is None:
            result.append(stmt)
            continue

        if name in expanding:
            cycle = " -> ".join([*expanding, name])
            raise ValueError(f"Recursive PROGRAM cycle detected: {cycle}")

        sub_prog = program_defs[name]
        assert isinstance(sub_prog, ProgramDefinition)
        sub_body = _flatten_repeats(sub_prog.body)
        sub_body = _expand_program_calls(
            sub_body,
            program_defs,
            gtype_names,
            wire_counter,
            expanding | {name},
        )

        sub_in, sub_out = _analyze_program_wires(sub_body)
        assert caller_in is not None and caller_out is not None
        if len(caller_in) != len(sub_in):
            raise ValueError(
                f"Sub-program {name!r} expects {len(sub_in)} input wires, "
                f"got {len(caller_in)}"
            )
        if len(caller_out) != len(sub_out):
            raise ValueError(
                f"Sub-program {name!r} expects {len(sub_out)} output wires, "
                f"got {len(caller_out)}"
            )

        wire_map: dict[int, int] = {}
        for sub_w, parent_w in zip(sub_in, caller_in):
            wire_map[sub_w] = parent_w
        for sub_w, parent_w in zip(sub_out, caller_out):
            wire_map[sub_w] = parent_w

        all_sub_wires: set[int] = set()
        for s in sub_body:
            if isinstance(s, GadgetApplication):
                if s.in_indices:
                    all_sub_wires.update(s.in_indices)
                if s.out_indices:
                    all_sub_wires.update(s.out_indices)
            elif isinstance(s, VirtualCorrection):
                all_sub_wires.add(s.wire)
            elif isinstance(s, Instruction):
                from deq.circuit.model import QubitTarget

                all_sub_wires.update(
                    t.index for t in s.targets if isinstance(t, QubitTarget)
                )
        for w in all_sub_wires:
            if w not in wire_map:
                wire_map[w] = wire_counter[0]
                wire_counter[0] += 1

        for s in sub_body:
            result.append(_remap_stmt(s, wire_map))

    return result


def compile_program_for_jit(
    jit_library: jit_pb.JitLibrary,
    program_def: "ProgramDefinition",  # noqa: F821 - imported lazily by caller
    program_defs: dict[str, object] | None = None,
    codes: dict[str, "CodeDefinition"] | None = None,  # noqa: F821
) -> tuple[
    list[tuple[jit_pb.JitInstruction, "GadgetApplication"]],  # noqa: F821
    list[tuple[int, bool, str]],
]:
    """Compile a ``ProgramDefinition`` into a list of ``JitInstruction``s.

    Returns each emitted instruction paired with the
    :class:`GadgetApplication` it was compiled from (the shortcut
    ``Instruction`` form is normalized to ``GadgetApplication`` first),
    so callers can render a faithful header (gadget name + IN/OUT
    indices) when emitting derived artifacts.

    Only ``GadgetApplication`` statements are supported. Wires (the
    integers in ``IN(...)`` / ``OUT(...)``) live in a single program-
    wide namespace: each integer must be produced exactly once (by
    some gadget's output port) before being consumed exactly once (by
    a later gadget's input port).

    Pauli correction pseudo-instructions (``VIRTUAL X0 wire``,
    ``VIRTUAL Z1 wire``, ``VIRTUAL Y2 wire``) toggle the constant
    column of the producer gadget's ``correction_propagation`` matrix
    without consuming the wire.

    Returns ``(instructions_with_app, assertions)`` where ``assertions``
    is a list of ``(readout_index, expected_bool, source)`` tuples
    derived from ``ASSERT_EQ`` statements with ``rec[-k]`` targets.
    The readout index is absolute over the program's flattened logical
    readout stream (``rec[-k]`` in a PROGRAM body refers to the
    ``k``-th most recent logical readout, not a physical measurement).
    """
    from deq.circuit.model import (
        AssertStatement,
        ConditionalCorrection,
        GadgetApplication,
        Instruction,
        MeasurementRecordTarget,
        QubitTarget,
        VirtualCorrection,
    )
    from deq.transpiler.compose_builder import (
        emit_conditional_correction_instruction,
    )

    gtype_of_name: dict[str, int] = {
        gt.base.name: gt.base.gtype for gt in jit_library.gadget_types
    }
    gadget_types_by_gtype: dict[int, jit_pb.JitGadgetType] = {
        gt.base.gtype: gt for gt in jit_library.gadget_types
    }
    port_types_by_ptype: dict[int, jit_pb.JitPortType] = {
        pt.base.ptype: pt for pt in jit_library.port_types
    }

    # ptype → number of logical qubits, read directly from JitPortType.k
    ptype_to_k: dict[int, int] = {pt.base.ptype: pt.k for pt in jit_library.port_types}

    instructions: list[tuple[jit_pb.JitInstruction, GadgetApplication]] = []
    # (absolute_readout_index, expected_value, source_text)
    assertions: list[tuple[int, bool, str]] = []
    # wire_id -> producer (gadget id, output port slot, port type, description)
    wire_producer: dict[int, _WireProducer] = {}
    next_gid = 1
    running_readouts = 0
    gid_to_index: dict[int, int] = {}
    gid_to_gadget_type: dict[int, jit_pb.JitGadgetType] = {}
    # gid -> list of (row, col) toggles for correction_propagation
    pauli_toggles: dict[int, list[tuple[int, int]]] = {}
    # absolute readout index -> (gid, local_readout_index) for resolving rec[-k]
    # in CONDITIONAL statements.
    readout_history: list[tuple[int, int]] = []
    # ptype -> synthesized identity gadget gtype (lazily created on first use)
    identity_gtype_of_ptype: dict[int, int] = {}
    next_synthetic_gtype = (
        max((gt.base.gtype for gt in jit_library.gadget_types), default=0) + 1
    )

    # Pre-expand sub-program calls and REPEAT blocks.
    body: list[object] = list(program_def.body)
    if program_defs:
        gtype_names = set(gtype_of_name)
        all_wires: set[int] = set()
        for s in body:
            if isinstance(s, GadgetApplication):
                if s.in_indices:
                    all_wires.update(s.in_indices)
                if s.out_indices:
                    all_wires.update(s.out_indices)
        wire_counter = [max(all_wires, default=-1) + 1]
        body = _expand_program_calls(body, program_defs, gtype_names, wire_counter)
    body = _flatten_repeats(body)

    for stmt in body:
        # ASSERT_EQ resolves against the running logical readout stream.
        if isinstance(stmt, AssertStatement):
            target = stmt.target
            if not isinstance(target, MeasurementRecordTarget):
                raise ValueError(
                    f"PROGRAM {program_def.name!r}: ASSERT_EQ only supports "
                    f"rec[-k] targets; got {type(target).__name__}"
                )
            if target.offset <= 0 or target.offset > running_readouts:
                raise ValueError(
                    f"PROGRAM {program_def.name!r}: ASSERT_EQ rec[-{target.offset}] "
                    f"out of range (only {running_readouts} readouts so far)"
                )
            abs_index = running_readouts - target.offset
            expected = bool(stmt.expected_value)
            source = f"ASSERT_EQ rec[-{target.offset}] {stmt.expected_value}"
            assertions.append((abs_index, expected, source))
            continue

        # Shortcut form: ``GadgetName q1 q2 ...`` is parsed as ``Instruction``;
        # promote to a GadgetApplication. The qubit list is fed (as a prefix)
        # to both the gadget's input and output ports — gadgets with no
        # inputs (like ``PrepareZ``) get an empty in_indices, gadgets with
        # no outputs (like ``MeasureZ``) get an empty out_indices.
        if isinstance(stmt, Instruction) and stmt.name in gtype_of_name:
            indices = [t.index for t in stmt.targets if isinstance(t, QubitTarget)]
            n_in = len(gadget_types_by_gtype[gtype_of_name[stmt.name]].base.inputs)
            n_out = len(gadget_types_by_gtype[gtype_of_name[stmt.name]].base.outputs)
            expected = max(n_in, n_out)
            if len(indices) != expected:
                raise ValueError(
                    f"PROGRAM {program_def.name!r}: shortcut '{stmt.name}' has "
                    f"{len(indices)} qubit targets but gadget expects "
                    f"{expected} (max of {n_in} inputs, {n_out} outputs)"
                )
            stmt = GadgetApplication(
                gadget_name=stmt.name,
                in_indices=indices[:n_in],
                out_indices=indices[:n_out],
            )

        # VIRTUAL Pauli correction pseudo-instructions: ``VIRTUAL X0*Y1 wire``.
        if isinstance(stmt, VirtualCorrection):
            wire = stmt.wire
            if wire not in wire_producer:
                raise ValueError(
                    f"PROGRAM {program_def.name!r}: {stmt} references "
                    f"wire {wire} which has no producer"
                )
            producer = wire_producer[wire]
            gadget_type = gid_to_gadget_type[producer.gid]

            port_ptype = gadget_type.base.outputs[producer.port].ptype
            if port_ptype in ptype_to_k:
                k = ptype_to_k[port_ptype]
            else:
                raise ValueError(
                    f"PROGRAM {program_def.name!r}: VIRTUAL requires code "
                    f"definitions (pass codes= to compile_program_for_jit)"
                )

            row_offset = sum(
                len(
                    port_types_by_ptype[
                        gadget_type.base.outputs[p].ptype
                    ].base.observables
                )
                for p in range(producer.port)
            )
            const_col = gadget_type.base.correction_propagation.cols - 1

            if producer.gid not in pauli_toggles:
                pauli_toggles[producer.gid] = []
            for pauli_type, qubit_index in stmt.paulis:
                if qubit_index >= k:
                    raise ValueError(
                        f"PROGRAM {program_def.name!r}: {stmt} qubit index "
                        f"{qubit_index} out of range "
                        f"(port on wire {wire} has {k} logical qubits)"
                    )
                if pauli_type in ("X", "Y"):
                    pauli_toggles[producer.gid].append(
                        (row_offset + 2 * qubit_index + 1, const_col)
                    )
                if pauli_type in ("Z", "Y"):
                    pauli_toggles[producer.gid].append(
                        (row_offset + 2 * qubit_index, const_col)
                    )
            continue

        # CONDITIONAL pseudo-instruction: ``CONDITIONAL rec[-k] X0*Y1 wire``.
        # Emits a synthesized "identity" gadget that consumes the wire from
        # its current producer and re-outputs it, carrying a
        # ``remote_conditional_correction`` modifier conditioned on the
        # k-th most recent logical readout.
        if isinstance(stmt, ConditionalCorrection):
            wire = stmt.wire
            if wire not in wire_producer:
                raise ValueError(
                    f"PROGRAM {program_def.name!r}: {stmt} references "
                    f"wire {wire} which has no producer"
                )
            producer = wire_producer[wire]
            if producer.ptype not in port_types_by_ptype:
                raise ValueError(
                    f"PROGRAM {program_def.name!r}: {stmt} references wire "
                    f"{wire} whose port type {producer.ptype} is not in the "
                    f"JIT library"
                )

            gid = next_gid
            next_gid += 1
            instruction, new_identity_gt, next_synthetic_gtype = (
                emit_conditional_correction_instruction(
                    conditional=stmt,
                    error_context=f"PROGRAM {program_def.name!r}",
                    wire_ptype=producer.ptype,
                    wire_source=(producer.gid, producer.port),
                    readout_history=readout_history,
                    port_types_by_ptype=port_types_by_ptype,
                    identity_gtype_of_ptype=identity_gtype_of_ptype,
                    next_synthetic_gtype=next_synthetic_gtype,
                    gid=gid,
                )
            )
            if new_identity_gt is not None:
                jit_library.gadget_types.append(new_identity_gt)
                gadget_types_by_gtype[new_identity_gt.base.gtype] = new_identity_gt

            identity_jit_gt = gadget_types_by_gtype[
                identity_gtype_of_ptype[producer.ptype]
            ]

            # Synthesize a GadgetApplication for the returned tuple (used by
            # downstream rendering/logging). It re-uses ``wire`` as the
            # single in/out binding.
            synthetic_app = GadgetApplication(
                gadget_name=identity_jit_gt.base.name,
                in_indices=[wire],
                out_indices=[wire],
            )
            instructions.append((instruction, synthetic_app))
            gid_to_index[gid] = len(instructions) - 1
            gid_to_gadget_type[gid] = identity_jit_gt

            # Identity gadget has no readouts; do not update readout_history.

            # The identity gadget becomes the new producer of the wire.
            wire_producer[wire] = _WireProducer(
                gid=gid,
                port=0,
                ptype=producer.ptype,
                desc=f"step {gid} (CONDITIONAL {stmt!s}, identity gadget output port 0)",
            )
            continue

        if not isinstance(stmt, GadgetApplication):
            if isinstance(stmt, Instruction):
                raise ValueError(
                    f"PROGRAM {program_def.name!r}: unknown gadget "
                    f"{stmt.name!r}; available: {sorted(gtype_of_name)}"
                )
            raise ValueError(
                f"PROGRAM {program_def.name!r}: unsupported statement "
                f"{type(stmt).__name__}; only gadget applications are supported"
            )
        if stmt.gadget_name not in gtype_of_name:
            raise ValueError(
                f"PROGRAM {program_def.name!r}: unknown gadget "
                f"{stmt.gadget_name!r}; available: {sorted(gtype_of_name)}"
            )

        gtype = gtype_of_name[stmt.gadget_name]
        gadget_type = gadget_types_by_gtype[gtype]
        in_indices = stmt.in_indices or []
        out_indices = stmt.out_indices or []

        expected_in = len(gadget_type.base.inputs)
        if len(in_indices) != expected_in:
            raise ValueError(
                f"PROGRAM {program_def.name!r}: gadget {stmt.gadget_name!r} "
                f"expects {expected_in} input ports, got {len(in_indices)}"
            )
        expected_out = len(gadget_type.base.outputs)
        if len(out_indices) != expected_out:
            raise ValueError(
                f"PROGRAM {program_def.name!r}: gadget {stmt.gadget_name!r} "
                f"expects {expected_out} output ports, got {len(out_indices)}"
            )

        connectors: list[pb.Gadget.Connector] = []
        for slot, wire in enumerate(in_indices):
            if wire not in wire_producer:
                raise ValueError(
                    f"PROGRAM {program_def.name!r}: input wire {wire} of "
                    f"{stmt.gadget_name!r} has no producer"
                )
            producer = wire_producer.pop(wire)
            expected_ptype = gadget_type.base.inputs[slot].ptype
            if producer.ptype != expected_ptype:
                raise ValueError(
                    f"PROGRAM {program_def.name!r}: input wire {wire} of "
                    f"{stmt.gadget_name!r} (slot {slot}) has port type "
                    f"{producer.ptype}, expected {expected_ptype}"
                )
            connectors.append(pb.Gadget.Connector(gid=producer.gid, port=producer.port))

        gid = next_gid
        next_gid += 1
        instructions.append(
            (
                jit_pb.JitInstruction(
                    gadget=pb.Gadget(gtype=gtype, gid=gid, connectors=connectors)
                ),
                stmt,
            )
        )
        gid_to_index[gid] = len(instructions) - 1
        gid_to_gadget_type[gid] = gadget_type
        running_readouts += len(gadget_type.base.readouts)
        for local_r in range(len(gadget_type.base.readouts)):
            readout_history.append((gid, local_r))

        for slot, wire in enumerate(out_indices):
            if wire in wire_producer:
                raise ValueError(
                    f"PROGRAM {program_def.name!r}: output wire {wire} of "
                    f"{stmt.gadget_name!r} (slot {slot}) is already produced "
                    "by an earlier gadget"
                )
            line_suffix = (
                f" at line {stmt.source_line}" if stmt.source_line is not None else ""
            )
            wire_producer[wire] = _WireProducer(
                gid=gid,
                port=slot,
                ptype=gadget_type.base.outputs[slot].ptype,
                desc=(
                    f"step {gid} ({stmt.gadget_name}{line_suffix}, "
                    f"output port {slot})"
                ),
            )

    if wire_producer:
        loc = (
            f" ({program_def.source_file})"
            if program_def.source_file is not None
            else ""
        )
        msg_lines = [
            f"PROGRAM {program_def.name!r}{loc}: dangling output wires "
            "(produced but never consumed). Every output wire of every "
            "gadget application in a PROGRAM must be consumed by a later "
            "gadget application (PROGRAMs are top-level and have no "
            "OUTPUT declaration). Fix by adding a consumer (e.g. a "
            "measurement gadget) for each wire below:",
        ]
        for wire in sorted(wire_producer):
            msg_lines.append(f"    wire {wire}: produced by {wire_producer[wire].desc}")
        raise ValueError("\n".join(msg_lines))

    for toggle_gid, toggles in pauli_toggles.items():
        idx = gid_to_index[toggle_gid]
        instr = instructions[idx][0]
        toggle_gadget_type = gid_to_gadget_type[toggle_gid]

        n_out = toggle_gadget_type.base.correction_propagation.rows
        n_in = toggle_gadget_type.base.correction_propagation.cols - 1

        toggle_set: set[tuple[int, int]] = set()
        for pos in toggles:
            toggle_set ^= {pos}

        if toggle_set:
            rows_list = sorted(r for r, _ in toggle_set)
            cols_list = [c for _, c in sorted(toggle_set)]
            toggle_matrix = util_pb.BitMatrix(
                rows=n_out,
                cols=n_in + 1,
                i=rows_list,
                j=cols_list,
            )
            instr.gadget.modifier.correction_propagation_mod.toggle.CopyFrom(
                toggle_matrix
            )

    return instructions, assertions


def export_program_stim(
    jit_library: jit_pb.JitLibrary,
    gadgets_by_name: dict[str, "GadgetDefinition"],  # noqa: F821
    gtype_to_name: dict[int, str],
    flatten_body: object,
    program_def: "ProgramDefinition",  # noqa: F821
    program_applications: list["GadgetApplication"],  # noqa: F821
    assertions: list[tuple[int, bool, str]] | None = None,
) -> str:
    """Concatenate gadget bodies (in program invocation order) into stim text.

    Each gadget body lives in its own local qubit namespace anchored on
    its ``INPUT`` / ``OUTPUT`` port ``qubit_indices``. To produce a
    single self-consistent stim circuit we:

    * Track per-wire physical qubit ids in a single program-wide
      namespace. The first time a wire is produced (an ``OUTPUT``
      port), fresh physical ids are allocated for it.
    * On each gadget invocation, build a ``local → physical`` map by
      pairing each ``INPUT`` port's ``qubit_indices`` with its incoming
      wire's physical ids. If the same gadget-local qubit also appears
      in an ``OUTPUT`` port (passthrough), the physical id is reused;
      otherwise a fresh physical id is allocated.
    * Ancilla qubits — those referenced by body instructions but not
      bound to any port — get fresh physical ids per invocation.
    * Body ``Instruction``s are emitted with qubit targets remapped
      from local to physical ids. ``REPEAT`` blocks are unrolled by
      ``flatten_body``. Non-``Instruction`` body items (ports,
      READOUT/CHECK/ERROR/ASSERT) are skipped.
    * Any physical id used by the gadget that is *not* exported via
      an ``OUTPUT`` port is reset with a trailing ``R`` instruction
      so the next gadget starts from a clean state.

    A header comment per invocation prints the local→physical map.
    """
    from deq.circuit.model import (
        Instruction,
        PauliTarget,
        PhysicalMeasurementTarget,
        PreselectStatement,
        QubitTarget,
        Target,
    )
    from deq.circuit.model import MeasurementRecordTarget
    from deq.transpiler.stim_constants import instruction_num_measurements

    chunks: list[str] = []
    next_physical = 0
    next_meas_idx = 0
    # (gid, port_index) -> list of physical qubit ids for that output port
    output_physicals: dict[tuple[int, int], list[int]] = {}

    source_lines = _program_source_lines(program_def, program_applications)

    for jit_instr, application, source_line in zip(
        jit_library.program, program_applications, source_lines
    ):
        gid = jit_instr.gadget.gid
        gtype = jit_instr.gadget.gtype
        name = gtype_to_name.get(gtype, f"<gtype={gtype}>")

        # Synthesised identity gadgets host the
        # ``remote_conditional_correction`` modifier emitted from a
        # PROGRAM-level or COMPOSE-level ``CONDITIONAL`` statement
        # (see :func:`emit_conditional_correction_instruction`).
        # They are purely propagation nodes — no measurements, no
        # circuit instructions — and pass each input wire's physical
        # qubits straight through to the matching output port.  The
        # conditional Pauli the modifier applies is a *frame*
        # correction that lives in the JIT propagation matrices, not
        # in the stim circuit, so the exported stim circuit need only
        # forward the physicals.
        if name not in gadgets_by_name and _is_synthesised_identity_gadget(name):
            if len(jit_instr.gadget.connectors) != 1:
                raise ValueError(
                    f"G{gid}/{name}: synthesised identity gadget must "
                    f"have exactly 1 connector, got "
                    f"{len(jit_instr.gadget.connectors)}"
                )
            conn = jit_instr.gadget.connectors[0]
            key = (conn.gid, conn.port)
            if key not in output_physicals:
                raise ValueError(
                    f"G{gid}/{name}: input port references "
                    f"(gid={conn.gid}, port={conn.port}) which has no "
                    "registered output physicals"
                )
            producer_phys = list(output_physicals[key])
            output_physicals[(gid, 0)] = producer_phys
            chunks.append(
                f"# G{gid}: {name} "
                f"(synthesised identity — CONDITIONAL passthrough)"
                f"{_format_source_line(source_line, program_def)}"
            )
            continue

        if name not in gadgets_by_name:
            raise ValueError(
                f"cannot export stim: gadget {name!r} (gtype={gtype}) "
                "has no source GadgetDefinition"
            )
        body = list(gadgets_by_name[name].body)
        flattened = list(flatten_body(body, for_simulate=True))  # type: ignore[operator]

        input_ports = gadgets_by_name[name].input_ports
        output_ports = gadgets_by_name[name].output_ports

        connectors = list(jit_instr.gadget.connectors)
        if len(connectors) != len(input_ports):
            raise ValueError(
                f"G{gid}/{name}: {len(connectors)} connectors but "
                f"{len(input_ports)} INPUT ports"
            )

        local_to_physical: dict[int, int] = {}

        # Bind input ports.
        for port, conn in zip(input_ports, connectors):
            key = (conn.gid, conn.port)
            if key not in output_physicals:
                raise ValueError(
                    f"G{gid}/{name}: input port references "
                    f"(gid={conn.gid}, port={conn.port}) which has no "
                    "registered output physicals"
                )
            producer_phys = output_physicals[key]
            if len(producer_phys) != len(port.qubit_indices):
                raise ValueError(
                    f"G{gid}/{name}: INPUT port has {len(port.qubit_indices)} "
                    f"qubits but producer wire has {len(producer_phys)}"
                )
            for local_q, phys_q in zip(port.qubit_indices, producer_phys):
                local_to_physical.setdefault(local_q, phys_q)

        # Bind output ports — reuse passthrough physicals, allocate fresh
        # ones for output-only qubits.
        for port_index, port in enumerate(output_ports):
            phys_for_port: list[int] = []
            for local_q in port.qubit_indices:
                if local_q not in local_to_physical:
                    local_to_physical[local_q] = next_physical
                    next_physical += 1
                phys_for_port.append(local_to_physical[local_q])
            output_physicals[(gid, port_index)] = phys_for_port

        # Bind ancilla qubits (used in body but not in any port).
        body_qubits: set[int] = set()
        for stmt in flattened:
            if isinstance(stmt, Instruction):
                for t in stmt.targets:
                    if isinstance(t, QubitTarget):
                        body_qubits.add(t.index)
        for local_q in sorted(body_qubits):
            if local_q not in local_to_physical:
                local_to_physical[local_q] = next_physical
                next_physical += 1

        # Compute physicals to reset: every physical touched by this
        # gadget that is not exported via an output port.
        exported = {
            phys
            for port_index in range(len(output_ports))
            for phys in output_physicals[(gid, port_index)]
        }
        used_physicals = set(local_to_physical.values())
        dying = sorted(used_physicals - exported)

        header_lines = [
            f"# G{gid}: {_format_application(application)}{_format_source_line(source_line, program_def)}"
        ]
        if local_to_physical:
            mapping_str = ", ".join(
                f"{local}->{phys}" for local, phys in sorted(local_to_physical.items())
            )
            header_lines.append(f"# qubit map (local->physical): {mapping_str}")

        body_lines: list[str] = []
        has_preselect = any(isinstance(s, PreselectStatement) for s in flattened)
        if has_preselect:
            body_lines.append("#!preselect_begin")
        gadget_start_meas = next_meas_idx
        for stmt in flattened:
            if isinstance(stmt, PreselectStatement):
                condition = stmt.condition
                if isinstance(condition, MeasurementRecordTarget):
                    abs_idx = next_meas_idx - condition.offset
                elif isinstance(condition, PhysicalMeasurementTarget):
                    abs_idx = gadget_start_meas + condition.index
                else:
                    raise ValueError(
                        f"G{gid}/{name}: PRESELECT condition has unsupported "
                        f"target type {type(condition).__name__}"
                    )
                body_lines.append(f"#!preselect_expect {abs_idx} {stmt.expected_value}")
                continue
            if not isinstance(stmt, Instruction):
                continue
            new_targets: list[Target] = []
            for t in stmt.targets:
                if isinstance(t, QubitTarget):
                    new_targets.append(
                        QubitTarget(
                            index=local_to_physical[t.index],
                            inverted=t.inverted,
                        )
                    )
                elif isinstance(t, PauliTarget):
                    new_targets.append(
                        PauliTarget(
                            pauli=t.pauli,
                            index=local_to_physical[t.index],
                            inverted=t.inverted,
                        )
                    )
                else:
                    new_targets.append(t)
            remapped = Instruction(
                name=stmt.name,
                tag=stmt.tag,
                arguments=list(stmt.arguments),
                targets=new_targets,
            )
            body_lines.append(str(remapped))
            next_meas_idx += instruction_num_measurements(str(stmt))

        if dying:
            body_lines.append("R " + " ".join(str(p) for p in dying))

        chunks.append("\n".join(header_lines + body_lines))

    body = "\n\n".join(chunks) + ("\n" if chunks else "")
    if assertions:
        return _render_rhai_assertion_block(assertions, program_def) + "\n" + body
    return body


def _program_assert_source_lines(
    program_def: object,
    count: int,
) -> list[int | None]:
    """Return source line numbers (1-based) for the first ``count``
    ``ASSERT_EQ`` statements inside ``program_def``'s PROGRAM block.

    Re-reads ``program_def.source_file`` and walks the body of
    ``PROGRAM <name> { ... }``. Returns ``[None] * count`` if the file
    is missing or the block can't be located.
    """
    fallback: list[int | None] = [None] * count

    name = getattr(program_def, "name", None)
    source = getattr(program_def, "source_file", None)
    if not source or not name:
        return fallback
    try:
        text = open(source, encoding="utf-8").read()
    except OSError:
        return fallback

    lines = text.splitlines()
    header_re = re.compile(rf"^\s*PROGRAM\s+{re.escape(name)}\b")
    start_idx: int | None = None
    for i, line in enumerate(lines):
        if header_re.match(line):
            start_idx = i
            break
    if start_idx is None:
        return fallback

    end_idx = len(lines)
    for i in range(start_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped == "}" or stripped.startswith("}"):
            end_idx = i
            break

    found: list[int | None] = []
    for i in range(start_idx + 1, end_idx):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            continue
        code = stripped.split("#", 1)[0].strip()
        if code.split()[:1] == ["ASSERT_EQ"]:
            found.append(i + 1)
            if len(found) == count:
                break
    while len(found) < count:
        found.append(None)
    return found


def _render_rhai_assertion_block(
    assertions: list[tuple[int, bool, str]],
    program_def: object,
) -> str:
    """Render ``ASSERT_EQ`` statements as an embedded ``#!rhai`` block.

    Each assertion contributes one check inside the generated
    ``is_logical_error`` function: the named readout bit must equal its
    expected value, otherwise the shot is flagged as a logical error.
    Each check is annotated with the source file and line number of the
    originating ``ASSERT_EQ`` statement to aid debugging. The function
    returns ``true`` on the first failed check.
    """
    src_lines = _program_assert_source_lines(program_def, len(assertions))
    source_file = getattr(program_def, "source_file", None)
    file_label = os.path.basename(source_file) if source_file else None

    lines = [
        "#!rhai",
        "# // Auto-generated from PROGRAM assertion statements.",
        "# fn is_logical_error(shot, readouts, measurements) {",
    ]
    for (abs_index, expected, source), line_no in zip(assertions, src_lines):
        expected_str = "true" if expected else "false"
        if line_no is not None and file_label:
            location = f"  ({file_label}:{line_no})"
        elif line_no is not None:
            location = f"  (line {line_no})"
        else:
            location = ""
        lines.append(f"#     // {source}{location}")
        lines.append(
            f"#     if readouts[{abs_index}] != {expected_str} {{ return true; }}"
        )
    lines.append("#     false")
    lines.append("# }")
    return "\n".join(lines) + "\n"


@arguably.command
def compile_(
    jit_file: str,
    *,
    out: str | None = None,
) -> None:
    """
    Compile a .deq.jit file into a static .deq.bin file.

    Uses the program already embedded in the .deq.jit file (produced by
    transpile with --program).
    """
    # Load the JIT library
    with open(jit_file, "rb") as f:
        jit_library = jit_pb.JitLibrary.FromString(f.read())

    if not jit_library.program:
        raise ValueError(
            f"No program found in {jit_file}. "
            f"Use transpile with --program to embed one."
        )

    # Compile to deq.bin
    deq_bin = static_jit_compiler(jit_library)

    # Determine output path
    if out is None:
        base = os.path.splitext(jit_file)[0]
        if base.endswith(".deq"):
            base = base[:-4]
        out = f"{base}.deq.bin"

    # Write binary output
    with open(out, "wb") as f:
        f.write(deq_bin.SerializeToString())

    # Also write text representation
    with open(f"{out}.txt", "w", encoding="utf-8") as f:
        f.write(str(deq_bin))

    print(f"Compiled JIT program to: {out}")
    print(
        f"  Instructions: {len(deq_bin.program)} (including gadgets, check models and error models)"
    )


def parse_jit_program(
    jit_library: jit_pb.JitLibrary,
    program: str,
    program_defs: dict[str, object] | None = None,
    codes: dict[str, "CodeDefinition"] | None = None,  # noqa: F821
) -> list[jit_pb.JitInstruction]:
    """Parse a ``.deq`` PROGRAM body string and return JIT instructions.

    *program* should contain the **body** of a ``PROGRAM`` block in
    ``.deq`` syntax (one statement per line), e.g.::

        PrepareZ OUT(0)
        Idle IN(0) OUT(0)
        MeasureZ IN(0)

    *program_defs* is an optional mapping of sub-program names to their
    ``ProgramDefinition`` objects.  When a gadget application in the
    program body references a name found in *program_defs* (rather than
    in *jit_library*), the sub-program is inlined with wire indices
    remapped.

    Pauli corrections use the ``VIRTUAL`` keyword, e.g.
    ``VIRTUAL X0 0`` applies an X correction on logical qubit 0
    of the gadget that produced wire 0. Multi-Pauli products
    are supported: ``VIRTUAL X0*Z1 0``.

    This is a public API for programmatic use.
    """
    from deq.circuit.model import ProgramDefinition
    from deq.circuit.parser import parse

    wrapped = f"PROGRAM InlineProgram {{\n{program}\n}}"
    deq_file = parse(wrapped)
    all_program_defs = [
        d for d in deq_file.definitions if isinstance(d, ProgramDefinition)
    ]
    if not all_program_defs:
        raise ValueError("Failed to parse program body as .deq PROGRAM block")
    # Merge any additional program definitions from the parsed text
    # (excluding the inline program itself).
    merged_defs: dict[str, object] = dict(program_defs) if program_defs else {}
    for d in all_program_defs[1:]:
        merged_defs.setdefault(d.name, d)
    compiled, _assertions = compile_program_for_jit(
        jit_library, all_program_defs[0], merged_defs or None, codes=codes
    )
    return [instr for instr, _app in compiled]
