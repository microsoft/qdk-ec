"""Build a composed ``JitGadgetType`` from a ``COMPOSE`` definition.

The compose builder constructs a ``JitLibrary`` with mock boundary gadgets
(to close off dangling ports), runs the Rust JIT compiler to produce a
complete ``Library``, then calls ``merge()`` on only the real (non-mock)
gadgets.  The ``merge()`` function handles observable propagation, check
classification, and error mapping — producing a ``MergedGadget`` that is
converted to a ``JitGadgetType``.
"""

# pylint: disable=no-member


from typing import Callable, Mapping, Sequence

import deq.proto.deq_bin_pb2 as pb
import deq.proto.deq_jit_pb2 as jit_pb
import deq.proto.util_pb2 as util_pb

from deq.circuit.model import (
    CodeDefinition,
    ComposeDefinition,
    ComposeStatement,
    ConditionalCorrection,
    ConditionalStatement,
    GadgetApplication,
    GadgetDefinition,
    GadgetStatement,
    InputPort,
    Instruction,
    MeasurementRecordTarget,
    OutputPort,
    PauliTarget,
    PhysicalMeasurementTarget,
    QubitTarget,
    ReadoutStatement,
    ReadoutTargetItem,
    RepeatBlock,
    Target,
)
from deq.compiler.jit_compiler import static_jit_compiler
from deq.spec.canonical import merge
from deq.transpiler.jit_transpiler import flatten_body, num_frame_columns

# ---------------------------------------------------------------------------
# COMPOSE validation
# ---------------------------------------------------------------------------


def validate_compose(
    compose: ComposeDefinition,
    *,
    gadget_definitions: Mapping[str, GadgetDefinition],
    compose_definitions: Mapping[str, ComposeDefinition],
) -> None:
    """Validate a ``COMPOSE`` definition or raise ``ValueError``.

    Rejects constructs we do not support: non-``@GTYPE`` decorators,
    raw Stim instructions, forward references, and unsupported statement
    types.  Also checks that each gadget/compose application has a
    target count consistent with the sub-definition's port counts.

    ``gadget_definitions`` and ``compose_definitions`` map the *already
    declared* gadget/compose names visible to ``compose``. Names declared
    later in the file are not visible.
    """
    unsupported = [
        d for d in compose.decorators if d.name not in ("GTYPE", "REPROPAGATE")
    ]
    if unsupported:
        names = ", ".join(d.name for d in unsupported)
        raise ValueError(
            f"COMPOSE {compose.name!r}: only @GTYPE and @REPROPAGATE are "
            f"supported on COMPOSE definitions (got @{names})"
        )

    for deco in compose.decorators:
        if deco.name == "REPROPAGATE" and deco.arguments:
            raise ValueError(
                f"COMPOSE {compose.name!r}: @REPROPAGATE takes no arguments "
                f"(got {deco.arguments!r})"
            )

    declared_names = set(gadget_definitions) | set(compose_definitions)

    def _lookup(name: str) -> GadgetDefinition | ComposeDefinition:
        if name in gadget_definitions:
            return gadget_definitions[name]
        return compose_definitions[name]

    def _check_body(items: list, where: str) -> None:
        for stmt in items:
            if isinstance(stmt, GadgetApplication):
                if stmt.gadget_name not in declared_names:
                    raise ValueError(
                        f"COMPOSE {compose.name!r}{where}: unknown gadget "
                        f"{stmt.gadget_name!r}; gadgets/composes must be "
                        f"declared before they are used"
                    )
                _check_explicit_application(compose, stmt, _lookup, where)
                continue
            if isinstance(stmt, RepeatBlock):
                _check_body(stmt.body, where + " (inside REPEAT)")
                continue
            if isinstance(stmt, (InputPort, OutputPort)):
                continue
            if isinstance(stmt, ConditionalCorrection):
                continue
            if isinstance(stmt, Instruction):
                if stmt.name in declared_names:
                    _check_shortcut_application(compose, stmt, _lookup, where)
                    continue
                raise ValueError(
                    f"COMPOSE {compose.name!r}{where}: raw Stim instruction "
                    f"{stmt.name!r} is not supported in a COMPOSE body; "
                    f"COMPOSE may only contain GADGET/COMPOSE applications, "
                    f"REPEAT blocks, INPUT and OUTPUT declarations"
                )
            raise ValueError(
                f"COMPOSE {compose.name!r}{where}: unsupported statement "
                f"{type(stmt).__name__}; COMPOSE may only contain "
                f"GADGET/COMPOSE applications, REPEAT blocks, INPUT and "
                f"OUTPUT declarations"
            )

    _check_body(compose.body, "")


def _format_compose_loc(compose: ComposeDefinition) -> str:
    return f" ({compose.source_file})" if compose.source_file is not None else ""


def _check_shortcut_application(
    compose: ComposeDefinition,
    inst: Instruction,
    lookup: Callable[[str], GadgetDefinition | ComposeDefinition],
    where: str,
) -> None:
    """Validate a shortcut-form gadget application: ``Name 0 1 2 ...``.

    The number of qubit targets must equal ``max(n_in, n_out)``, where
    ``n_in`` and ``n_out`` are the sub-definition's INPUT and OUTPUT
    port counts.  Each port at the application level corresponds to
    exactly one target; under-supplied targets cause silent dropping
    of wires, and over-supplied targets are silently ignored — both
    leading to confusing downstream errors.
    """
    sub_def = lookup(inst.name)
    n_in = len(sub_def.input_ports)
    n_out = len(sub_def.output_ports)
    expected = max(n_in, n_out)
    qubit_targets = [t for t in inst.targets if isinstance(t, QubitTarget)]
    if len(qubit_targets) == expected:
        return
    loc = _format_compose_loc(compose)
    raise ValueError(
        f"COMPOSE {compose.name!r}{loc}{where}: gadget application "
        f"{inst.name!r} has {len(qubit_targets)} target(s), but "
        f"{inst.name!r} declares {n_in} INPUT port(s) and {n_out} "
        f"OUTPUT port(s); the shortcut form requires exactly "
        f"{expected} target(s) (= max(INPUT ports, OUTPUT ports)). "
        f"Use the explicit form '{inst.name} IN(...) OUT(...)' if "
        f"you need different wires for the inputs and outputs."
    )


def _check_explicit_application(
    compose: ComposeDefinition,
    app: GadgetApplication,
    lookup: Callable[[str], GadgetDefinition | ComposeDefinition],
    where: str,
) -> None:
    """Validate an explicit-form gadget application.

    The explicit form is one of ``Name IN(...) OUT(...)``,
    ``Name IN(...)``, ``Name OUT(...)``, or ``Name()``.  The number of
    wires inside each ``IN(...)`` / ``OUT(...)`` clause must equal the
    sub-definition's INPUT / OUTPUT port count.  An omitted clause is
    treated as zero wires (so the corresponding port count must also
    be zero).
    """
    sub_def = lookup(app.gadget_name)
    n_in = len(sub_def.input_ports)
    n_out = len(sub_def.output_ports)
    actual_in = len(app.in_indices) if app.in_indices is not None else 0
    actual_out = len(app.out_indices) if app.out_indices is not None else 0
    if actual_in == n_in and actual_out == n_out:
        return
    loc = _format_compose_loc(compose)
    line_loc = f" line {app.source_line}" if app.source_line is not None else ""
    raise ValueError(
        f"COMPOSE {compose.name!r}{loc}{where}{line_loc}: gadget "
        f"application {app.gadget_name!r} declares {actual_in} IN "
        f"wire(s) and {actual_out} OUT wire(s), but {app.gadget_name!r} "
        f"declares {n_in} INPUT port(s) and {n_out} OUTPUT port(s); "
        f"each IN/OUT clause must list one wire per port."
    )


def _instruction_to_application(
    inst: Instruction,
    *,
    sub_def: GadgetDefinition | ComposeDefinition,
) -> GadgetApplication:
    indices = [t.index for t in inst.targets if isinstance(t, QubitTarget)]
    n_in = len(sub_def.input_ports)
    n_out = len(sub_def.output_ports)
    return GadgetApplication(
        gadget_name=inst.name,
        in_indices=indices[:n_in],
        out_indices=indices[:n_out],
    )


def _expand_compose_body(
    body: list,
    *,
    gadget_definitions: Mapping[str, GadgetDefinition],
    compose_definitions: Mapping[str, ComposeDefinition],
) -> tuple[
    list[InputPort],
    list[OutputPort],
    list[GadgetApplication | ConditionalCorrection],
]:
    """Flatten a compose body into ``(inputs, outputs, ordered_items)``.

    ``ordered_items`` preserves source order across both gadget
    applications and ``CONDITIONAL`` pseudo-instructions, so the
    consumer can interleave synthetic identity gadgets at the right
    program positions.
    """
    inputs: list[InputPort] = []
    outputs: list[OutputPort] = []
    items: list[GadgetApplication | ConditionalCorrection] = []

    def _lookup(name: str) -> GadgetDefinition | ComposeDefinition | None:
        if name in gadget_definitions:
            return gadget_definitions[name]
        if name in compose_definitions:
            return compose_definitions[name]
        return None

    def _walk(stmts: list) -> None:
        for stmt in stmts:
            if isinstance(stmt, InputPort):
                inputs.append(stmt)
            elif isinstance(stmt, OutputPort):
                outputs.append(stmt)
            elif isinstance(stmt, GadgetApplication):
                if stmt.is_shortcut:
                    sub_def = _lookup(stmt.gadget_name)
                    if sub_def is None:
                        items.append(stmt)
                    else:
                        indices = list(stmt.in_indices or [])
                        n_in = len(sub_def.input_ports)
                        n_out = len(sub_def.output_ports)
                        items.append(
                            GadgetApplication(
                                gadget_name=stmt.gadget_name,
                                in_indices=indices[:n_in],
                                out_indices=indices[:n_out],
                            )
                        )
                else:
                    items.append(stmt)
            elif isinstance(stmt, ConditionalCorrection):
                items.append(stmt)
            elif isinstance(stmt, Instruction):
                sub_def = _lookup(stmt.name)
                if sub_def is not None:
                    items.append(_instruction_to_application(stmt, sub_def=sub_def))
            elif isinstance(stmt, RepeatBlock):
                for _ in range(stmt.count):
                    _walk(stmt.body)

    _walk(body)
    return inputs, outputs, items


# ===================================================================
# Port-type compatibility validation
# ===================================================================


def _validate_compose_port_types(
    compose: ComposeDefinition,
    inputs: list[InputPort],
    apps: list[GadgetApplication],
    *,
    gadget_definitions: Mapping[str, GadgetDefinition],
    compose_definitions: Mapping[str, ComposeDefinition],
) -> None:
    """Check that consecutive gadgets in a COMPOSE have matching port types.

    Raises ``ValueError`` with a clear message identifying which gadget
    application has a port-type mismatch and what the expected/actual
    code names are.
    """
    compose_name = compose.name

    def _get_def(
        name: str,
    ) -> GadgetDefinition | ComposeDefinition:
        if name in gadget_definitions:
            return gadget_definitions[name]
        return compose_definitions[name]

    # Map wire index → (code_name, producer_description)
    wire_code: dict[int, tuple[str, str]] = {}

    for inp in inputs:
        for wire in inp.qubit_indices:
            wire_code[wire] = (inp.code_name, f"COMPOSE INPUT ({inp.code_name})")

    for app_idx, app in enumerate(apps):
        sub_def = _get_def(app.gadget_name)
        in_ports = sub_def.input_ports
        in_wires = list(app.in_indices or [])

        for port_idx, wire in enumerate(in_wires):
            if wire not in wire_code:
                continue
            if port_idx >= len(in_ports):
                continue
            expected_code = in_ports[port_idx].code_name
            actual_code, producer = wire_code[wire]
            if expected_code != actual_code:
                loc_parts: list[str] = []
                if compose.source_file is not None:
                    loc_parts.append(compose.source_file)
                if app.source_line is not None:
                    loc_parts.append(f"line {app.source_line}")
                loc = f" ({', '.join(loc_parts)})" if loc_parts else ""
                raise ValueError(
                    f"COMPOSE {compose_name!r}: port type mismatch at "
                    f"step {app_idx + 1} ({app.gadget_name!r}){loc}, "
                    f"input port {port_idx}:\n"
                    f"  expected: {expected_code!r} "
                    f"(required by {app.gadget_name}'s INPUT)\n"
                    f"  got:      {actual_code!r} "
                    f"(produced by {producer})"
                )

        # Update wire_code with this gadget's outputs.
        out_ports = sub_def.output_ports
        out_wires = list(app.out_indices or [])
        for port_idx, wire in enumerate(out_wires):
            if port_idx < len(out_ports):
                wire_code[wire] = (
                    out_ports[port_idx].code_name,
                    f"step {app_idx + 1} ({app.gadget_name})",
                )


# ===================================================================
# Duplicate INPUT/OUTPUT wire detection
# ===================================================================


def _validate_compose_no_duplicate_port_wires(
    compose: ComposeDefinition,
    inputs: list[InputPort],
    outputs: list[OutputPort],
) -> None:
    """Reject COMPOSE blocks that bind the same wire to multiple INPUT or
    OUTPUT ports.

    A duplicate ``OUTPUT`` binding (e.g. ``OUTPUT Rep 0`` declared twice) is
    silently accepted by the dangling-output check (which compares wire
    *sets*), but slips through to the Rust JIT compiler where the second
    consumer of the same source port indexes an already-emptied check
    vector and panics.  Duplicate ``INPUT`` bindings have the analogous
    failure mode.  Reject both cases here with a clear ``ValueError`` so
    the user gets a structured diagnostic rather than a native panic.
    """

    def _check(ports: list, kind: str) -> None:
        seen: dict[int, str] = {}
        duplicates: list[tuple[int, str, str]] = []
        for port in ports:
            for wire in port.qubit_indices:
                if wire in seen:
                    duplicates.append((wire, seen[wire], port.code_name))
                else:
                    seen[wire] = port.code_name
        if not duplicates:
            return
        loc = _format_compose_loc(compose)
        msg_lines = [
            f"COMPOSE {compose.name!r}{loc}: wire bound to multiple "
            f"{kind} ports; each wire may appear in at most one {kind} "
            f"declaration.",
        ]
        for wire, first_code, second_code in duplicates:
            msg_lines.append(
                f"    wire {wire}: declared as {kind} {first_code} "
                f"and again as {kind} {second_code}"
            )
        raise ValueError("\n".join(msg_lines))

    _check(inputs, "INPUT")
    _check(outputs, "OUTPUT")


# ===================================================================
# Dangling-output detection
# ===================================================================


def _validate_compose_no_dangling_outputs(
    compose: ComposeDefinition,
    inputs: list[InputPort],
    outputs: list[OutputPort],
    apps: list[GadgetApplication],
    *,
    gadget_definitions: Mapping[str, GadgetDefinition],
    compose_definitions: Mapping[str, ComposeDefinition],
) -> None:
    """Check that every wire still live after the body is declared as ``OUTPUT``.

    A wire is *live* at the end of the body if its most recent producer
    (an ``INPUT`` declaration or a sub-gadget's output port) has not been
    consumed by any subsequent sub-gadget application.  Live wires must
    match the wires declared in the COMPOSE's ``OUTPUT`` statements.

    If they do not match, raises ``ValueError`` with a clear message
    listing the offending wires.  Without this check, the Rust JIT
    compiler would block forever waiting for a consumer of the dangling
    port.
    """

    def _get_def(name: str) -> GadgetDefinition | ComposeDefinition:
        if name in gadget_definitions:
            return gadget_definitions[name]
        return compose_definitions[name]

    # ``live`` maps a wire to the producer that currently owns it (the
    # most recent INPUT or sub-gadget output that wrote that wire).
    # ``intermediate_dangling`` collects producers that were overwritten
    # before any subsequent gadget consumed them — these correspond to
    # JIT-level output ports that have no downstream connector and would
    # otherwise cause the Rust JIT compiler to block forever waiting for
    # a consumer.
    live: dict[int, str] = {}
    intermediate_dangling: list[tuple[int, str]] = []

    def _produce(wire: int, producer: str) -> None:
        if wire in live:
            intermediate_dangling.append((wire, live[wire]))
        live[wire] = producer

    for inp in inputs:
        for wire in inp.qubit_indices:
            _produce(wire, f"COMPOSE INPUT ({inp.code_name})")

    for app_idx, app in enumerate(apps):
        sub_def = _get_def(app.gadget_name)
        n_in = len(sub_def.input_ports)
        n_out = len(sub_def.output_ports)
        for wire in list(app.in_indices or [])[:n_in]:
            live.pop(wire, None)
        for port_idx, wire in enumerate(list(app.out_indices or [])[:n_out]):
            _produce(
                wire,
                f"step {app_idx + 1} ({app.gadget_name}, output port {port_idx})",
            )

    declared: dict[int, str] = {}
    for out in outputs:
        for wire in out.qubit_indices:
            declared[wire] = out.code_name

    dangling = sorted(set(live) - set(declared))
    missing = sorted(set(declared) - set(live))
    if not dangling and not missing and not intermediate_dangling:
        return

    loc = f" ({compose.source_file})" if compose.source_file is not None else ""
    msg_lines = [
        f"COMPOSE {compose.name!r}{loc}: declared OUTPUT ports do not "
        f"match the wires that are still live at the end of the body.",
    ]
    if intermediate_dangling:
        msg_lines.append(
            "  Dangling wires (produced but overwritten before any "
            "later gadget consumed them — the JIT compiler would block "
            "forever waiting for a consumer of these ports):"
        )
        for wire, producer in intermediate_dangling:
            msg_lines.append(f"    wire {wire}: produced by {producer}")
    if dangling:
        msg_lines.append(
            "  Dangling wires (live at end of body but not declared as "
            "OUTPUT — must be either consumed by a later gadget or "
            "exposed via OUTPUT):"
        )
        for wire in dangling:
            msg_lines.append(f"    wire {wire}: produced by {live[wire]}")
    if missing:
        msg_lines.append(
            "  Declared OUTPUT wires that no gadget produces (or that "
            "were consumed by a later gadget):"
        )
        for wire in missing:
            msg_lines.append(f"    wire {wire}: declared as OUTPUT {declared[wire]}")
    raise ValueError("\n".join(msg_lines))


# ===================================================================
# Compose body expansion (recursive, with dense qubit remapping)
# ===================================================================
#
# These helpers inline a COMPOSE body into a flat
# (input_ports, circuit, output_ports) triple as if it were a single
# GADGET.  They are used both by the annotate tool (to render a COMPOSE
# as an inlined GADGET block) and by the @REPROPAGATE compose path
# (to recompute propagation matrices from circuit flow).


def _flatten_compose_apps_with_bindings(
    body: Sequence[ComposeStatement],
    known_names: set[str],
) -> list[tuple[str, list[int], list[int]]]:
    """Flatten a compose body into ``(name, in_wires, out_wires)`` tuples.

    ``REPEAT`` blocks are unrolled.  Shortcut applications (``Idle 0``)
    are recognized by matching the instruction name against *known_names*.
    """
    result: list[tuple[str, list[int], list[int]]] = []
    for stmt in body:
        if isinstance(stmt, RepeatBlock):
            sub = _flatten_compose_apps_with_bindings(list(stmt.body), known_names)
            for _ in range(stmt.count):
                result.extend(sub)
        elif isinstance(stmt, GadgetApplication):
            in_wires = list(stmt.in_indices) if stmt.in_indices is not None else []
            out_wires = list(stmt.out_indices) if stmt.out_indices is not None else []
            result.append((stmt.gadget_name, in_wires, out_wires))
        elif isinstance(stmt, Instruction) and stmt.name in known_names:
            wires = [t.index for t in stmt.targets if isinstance(t, QubitTarget)]
            result.append((stmt.name, wires, wires))
    return result


def _expand_definition(
    name: str,
    gadget_defs: Mapping[str, GadgetDefinition],
    compose_defs: Mapping[str, ComposeDefinition],
    known_names: set[str],
    codes: Mapping[str, CodeDefinition],
) -> tuple[list[InputPort], list[GadgetStatement], list[OutputPort]]:
    """Expand a single definition into ``(input_ports, circuit, output_ports)``.

    For a ``GADGET``, returns its raw ports, Stim instructions, and
    ``READOUT`` statements (with absolute ``M<i>`` measurement
    references rewritten to relative ``rec[-k]`` references so they
    remain valid when the body is inlined into a larger COMPOSE).
    For a ``COMPOSE``, recursively expands with qubit remapping.
    """
    # Local import to avoid the cycle compose_builder ↔ jit_library_builder.
    from deq.transpiler.jit_library_builder import _measurement_count_of

    if name in gadget_defs:
        gadget = gadget_defs[name]
        flat = flatten_body(list(gadget.body))
        inputs = [s for s in flat if isinstance(s, InputPort)]
        outputs = [s for s in flat if isinstance(s, OutputPort)]
        circuit: list[GadgetStatement] = []
        running = 0
        for s in flat:
            if isinstance(s, Instruction):
                circuit.append(s)
                running += _measurement_count_of(s)
            elif isinstance(s, ReadoutStatement):
                circuit.append(_relativize_readout(s, running))
        return inputs, circuit, outputs
    if name in compose_defs:
        return expand_compose_circuit(
            compose_defs[name], gadget_defs, compose_defs, known_names, codes
        )
    return [], [], []


def _relativize_readout(stmt: ReadoutStatement, running: int) -> ReadoutStatement:
    """Translate ``M<i>`` targets in *stmt* to ``rec[-(running - i)]``.

    *running* is the number of internal physical measurements emitted
    in the enclosing GADGET body strictly before *stmt*.  After the
    rewrite, the readout's measurement references survive inlining
    into a larger COMPOSE body unchanged: ``rec[-k]`` is always
    interpreted relative to the position of the READOUT statement at
    decode time, so it continues to point at the same physical
    measurement event regardless of how much code precedes the sub-
    gadget in the inlined body.
    """
    new_targets: list[ReadoutTargetItem] = []
    for target in stmt.targets:
        if isinstance(target, PhysicalMeasurementTarget):
            new_targets.append(
                MeasurementRecordTarget(offset=running - target.index)
            )
        else:
            new_targets.append(target)
    return ReadoutStatement(
        targets=new_targets,
        flip=stmt.flip,
        decorators=list(stmt.decorators),
    )


def expand_compose_circuit(
    compose: ComposeDefinition,
    gadget_defs: Mapping[str, GadgetDefinition],
    compose_defs: Mapping[str, ComposeDefinition],
    known_names: set[str],
    codes: Mapping[str, CodeDefinition],
) -> tuple[list[InputPort], list[GadgetStatement], list[OutputPort]]:
    """Recursively expand a compose with dense qubit remapping.

    Port data qubits are numbered ``0 .. total_data-1`` (dense).
    Ancilla qubits follow starting at ``total_data``.
    """
    compose_inputs = compose.input_ports
    compose_outputs = compose.output_ports

    # Determine all compose-level wires and their qubit ranges.
    # Wires are discovered from compose INPUT/OUTPUT ports and from
    # sub-gadget application bindings (for composes without explicit ports).
    wire_code: dict[int, str] = {}
    for port in compose_inputs:
        for wire_idx in port.qubit_indices:
            wire_code[wire_idx] = port.code_name
    for port in compose_outputs:
        for wire_idx in port.qubit_indices:
            wire_code.setdefault(wire_idx, port.code_name)

    # Infer wire codes from sub-gadget bindings when compose has no
    # explicit INPUT/OUTPUT for a wire.
    apps = _flatten_compose_apps_with_bindings(list(compose.body), known_names)
    for app_name, in_wires, out_wires in apps:
        sub_def_inputs: list[InputPort] = []
        sub_def_outputs: list[OutputPort] = []
        if app_name in gadget_defs:
            flat = flatten_body(list(gadget_defs[app_name].body))
            sub_def_inputs = [s for s in flat if isinstance(s, InputPort)]
            sub_def_outputs = [s for s in flat if isinstance(s, OutputPort)]
        elif app_name in compose_defs:
            sub_body = compose_defs[app_name].body
            sub_def_inputs = [s for s in sub_body if isinstance(s, InputPort)]
            sub_def_outputs = [s for s in sub_body if isinstance(s, OutputPort)]
        for port_idx, wire_idx in enumerate(in_wires):
            if wire_idx not in wire_code and port_idx < len(sub_def_inputs):
                wire_code[wire_idx] = sub_def_inputs[port_idx].code_name
        for port_idx, wire_idx in enumerate(out_wires):
            if wire_idx not in wire_code and port_idx < len(sub_def_outputs):
                wire_code[wire_idx] = sub_def_outputs[port_idx].code_name

    sorted_wires = sorted(wire_code)
    wire_n = {w: codes[wire_code[w]].n for w in sorted_wires}

    # A wire's code (and therefore its qubit count) can change as
    # sub-gadgets run. Pre-compute the
    # maximum qubit count each wire ever holds so we can allocate
    # enough contiguous dense indices to cover its peak size.
    wire_max_n: dict[int, int] = dict(wire_n)
    for app_name, _in_wires, out_wires in apps:
        sub_def_outputs2: list[OutputPort] = []
        if app_name in gadget_defs:
            flat = flatten_body(list(gadget_defs[app_name].body))
            sub_def_outputs2 = [s for s in flat if isinstance(s, OutputPort)]
        elif app_name in compose_defs:
            sub_def_outputs2 = [
                s for s in compose_defs[app_name].body if isinstance(s, OutputPort)
            ]
        for port_idx, wire_idx in enumerate(out_wires):
            if port_idx < len(sub_def_outputs2) and wire_idx in wire_max_n:
                new_n = codes[sub_def_outputs2[port_idx].code_name].n
                if new_n > wire_max_n[wire_idx]:
                    wire_max_n[wire_idx] = new_n

    wire_offset: dict[int, int] = {}
    cursor = 0
    for w in sorted_wires:
        wire_offset[w] = cursor
        cursor += wire_max_n[w]
    total_data = cursor

    # Build compose-level ports with dense qubit indices.
    dense_inputs: list[InputPort] = []
    for port in compose_inputs:
        wire_idx = port.qubit_indices[0]
        off = wire_offset[wire_idx]
        n = wire_n[wire_idx]
        dense_inputs.append(
            InputPort(code_name=port.code_name, qubit_indices=list(range(off, off + n)))
        )
    dense_outputs: list[OutputPort] = []
    for port in compose_outputs:
        wire_idx = port.qubit_indices[0]
        off = wire_offset[wire_idx]
        n = wire_n[wire_idx]
        dense_outputs.append(
            OutputPort(
                code_name=port.code_name, qubit_indices=list(range(off, off + n))
            )
        )

    if not apps:
        return dense_inputs, [], dense_outputs

    # Track the current dense qubit indices for each wire, updated after
    # each sub-gadget to reflect output port permutations.
    wire_qubits: dict[int, list[int]] = {}
    for w in sorted_wires:
        off = wire_offset[w]
        n = wire_n[w]
        wire_qubits[w] = list(range(off, off + n))

    circuit: list[GadgetStatement] = []
    for app_name, in_wires, out_wires in apps:
        sub_inputs, sub_stmts, sub_outputs = _expand_definition(
            app_name, gadget_defs, compose_defs, known_names, codes
        )

        # Build qubit remapping: port qubits → dense data indices.
        qmap: dict[int, int] = {}
        for port_idx, wire_idx in enumerate(in_wires):
            if port_idx < len(sub_inputs):
                current = wire_qubits[wire_idx]
                for local_i, phys_q in enumerate(sub_inputs[port_idx].qubit_indices):
                    if local_i < len(current):
                        qmap[phys_q] = current[local_i]
        for port_idx, wire_idx in enumerate(out_wires):
            if port_idx < len(sub_outputs):
                out_phys_qs = sub_outputs[port_idx].qubit_indices
                current = wire_qubits[wire_idx]
                # If the sub-gadget's output port carries more qubits
                # than the wire currently holds, extend the wire into its
                # reserved dense block.
                offset = wire_offset[wire_idx]
                while len(current) < len(out_phys_qs):
                    current.append(offset + len(current))
                for local_i, phys_q in enumerate(out_phys_qs):
                    qmap.setdefault(phys_q, current[local_i])

        # Non-port qubits → ancilla indices after data qubits.
        all_qs = _collect_qubit_indices_from_stmts(sub_stmts)
        ancilla_cursor = total_data
        for q in sorted(all_qs):
            if q not in qmap:
                qmap[q] = ancilla_cursor
                ancilla_cursor += 1

        for stmt in sub_stmts:
            if isinstance(stmt, Instruction):
                circuit.append(_remap_instruction(stmt, qmap))
            else:
                circuit.append(stmt)

        # Update wire qubit maps based on output port permutations.
        # The output port's qubit_indices define which sub-gadget-local
        # qubits map to each position in the wire. After remapping
        # through qmap, we get the new dense qubit order for that wire.
        for port_idx, wire_idx in enumerate(out_wires):
            if port_idx < len(sub_outputs):
                new_order: list[int] = []
                for phys_q in sub_outputs[port_idx].qubit_indices:
                    new_order.append(qmap[phys_q])
                wire_qubits[wire_idx] = new_order

    # Rebuild dense_outputs using the final wire qubit order (after
    # all permutations have been applied by sub-gadgets).
    dense_outputs = []
    for port in compose_outputs:
        wire_idx = port.qubit_indices[0]
        dense_outputs.append(
            OutputPort(
                code_name=port.code_name, qubit_indices=list(wire_qubits[wire_idx])
            )
        )

    return dense_inputs, circuit, dense_outputs


def _collect_qubit_indices_from_stmts(
    stmts: Sequence[GadgetStatement],
) -> set[int]:
    """Collect all qubit indices referenced in instructions."""
    indices: set[int] = set()
    for stmt in stmts:
        if isinstance(stmt, Instruction):
            for t in stmt.targets:
                if isinstance(t, (QubitTarget, PauliTarget)):
                    indices.add(t.index)
    return indices


def _remap_instruction(stmt: Instruction, qmap: dict[int, int]) -> Instruction:
    """Return a copy of *stmt* with qubit indices remapped via *qmap*."""
    new_targets: list[Target] = []
    for t in stmt.targets:
        if isinstance(t, QubitTarget):
            new_targets.append(
                QubitTarget(index=qmap.get(t.index, t.index), inverted=t.inverted)
            )
        elif isinstance(t, PauliTarget):
            new_targets.append(
                PauliTarget(
                    pauli=t.pauli, index=qmap.get(t.index, t.index), inverted=t.inverted
                )
            )
        else:
            new_targets.append(t)
    return Instruction(
        name=stmt.name,
        tag=stmt.tag,
        arguments=list(stmt.arguments),
        targets=new_targets,
    )


# ===================================================================
# @REPROPAGATE: inline-circuit compose builder
# ===================================================================


def has_repropagate(compose: ComposeDefinition) -> bool:
    """Return ``True`` if *compose* carries an ``@REPROPAGATE`` decorator."""
    return any(d.name == "REPROPAGATE" for d in compose.decorators)


def _reject_conditionals_under_repropagate(
    compose: ComposeDefinition,
    gadget_definitions: Mapping[str, GadgetDefinition],
    compose_definitions: Mapping[str, ComposeDefinition],
) -> None:
    """Raise :class:`ValueError` if *compose* (or any reachable
    sub-COMPOSE / sub-GADGET) carries a CONDITIONAL frame correction.

    ``@REPROPAGATE`` recomputes propagation matrices from circuit flow
    on the inlined body.  Any ``CONDITIONAL rec[-k]`` (COMPOSE-level
    :class:`ConditionalCorrection`) or ``CONDITIONAL R<j>`` (GADGET-
    level :class:`ConditionalStatement`) is a readout-conditioned frame
    flip that lives in ``logical_correction`` — mixing it with
    circuit-flow re-derivation has unclear semantics: the CONDITIONAL's
    frame flip cannot be reconstructed from the flat inlined body's
    Heisenberg propagation alone.

    Rather than silently drop or fragile-inject those contributions,
    reject the combination up front and direct the user to the plain
    COMPOSE + CONDITIONAL idiom (where the correction is preserved
    verbatim in the merged ``logical_correction`` and evaluated at
    runtime via ``residual ^= lc · readouts``).
    """
    visited: set[str] = set()

    def _reject_in_gadget(gadget: GadgetDefinition) -> None:
        for stmt in flatten_body(list(gadget.body)):
            if isinstance(stmt, ConditionalStatement):
                raise ValueError(
                    f"COMPOSE {compose.name!r} @REPROPAGATE: sub-GADGET "
                    f"{gadget.name!r} contains a CONDITIONAL frame "
                    f"correction, which @REPROPAGATE cannot re-derive "
                    f"from circuit flow. Either drop @REPROPAGATE and "
                    f"rely on merge-based composition, or move the "
                    f"CONDITIONAL out of {gadget.name!r} into a plain "
                    f"(non-@REPROPAGATE) COMPOSE that wraps it."
                )

    def _reject_in_compose(sub: ComposeDefinition, is_root: bool) -> None:
        if sub.name in visited:
            return
        visited.add(sub.name)
        for stmt in flatten_body(list(sub.body)):
            if isinstance(stmt, ConditionalCorrection):
                where = (
                    "its own body"
                    if is_root
                    else f"sub-COMPOSE {sub.name!r}"
                )
                raise ValueError(
                    f"COMPOSE {compose.name!r} @REPROPAGATE: {where} "
                    f"contains a CONDITIONAL frame correction, which "
                    f"@REPROPAGATE cannot re-derive from circuit flow. "
                    f"Either drop @REPROPAGATE and rely on merge-based "
                    f"composition, or restructure to keep CONDITIONAL "
                    f"statements outside any @REPROPAGATE compose."
                )
            if isinstance(stmt, GadgetApplication):
                name = stmt.gadget_name
            elif isinstance(stmt, Instruction):
                name = stmt.name
            else:
                continue
            if name in gadget_definitions:
                _reject_in_gadget(gadget_definitions[name])
            elif name in compose_definitions:
                _reject_in_compose(compose_definitions[name], is_root=False)

    _reject_in_compose(compose, is_root=True)


def compose_to_synthetic_gadget(
    compose: ComposeDefinition,
    gadget_definitions: Mapping[str, GadgetDefinition],
    compose_definitions: Mapping[str, ComposeDefinition],
    codes: Mapping[str, CodeDefinition],
) -> GadgetDefinition:
    """Inline a COMPOSE body into a flat synthetic ``GadgetDefinition``.

    The synthetic gadget has the same name as *compose*; its body is
    ``input_ports + circuit + output_ports`` produced by
    :func:`expand_compose_circuit`.  Decorators are dropped — the caller
    is responsible for re-attaching ``@GTYPE`` / ``@CHECKS`` on the
    pipeline side as needed.

    Used exclusively by the ``@REPROPAGATE`` build/annotate path (see
    :func:`_build_repropagated_compose`), which requires the body to be
    free of any CONDITIONAL frame correction.
    :func:`_reject_conditionals_under_repropagate` runs first at the
    ``@REPROPAGATE`` dispatch site to enforce that invariant, so no
    ``ConditionalStatement`` reconstruction is needed here.
    """
    known_names = set(gadget_definitions) | set(compose_definitions)
    input_ports, circuit, output_ports = expand_compose_circuit(
        compose,
        gadget_definitions,
        compose_definitions,
        known_names,
        codes,
    )
    body: list = [*input_ports, *circuit, *output_ports]
    return GadgetDefinition(
        name=compose.name,
        body=body,
        decorators=[],
        source_file=compose.source_file,
        source_line=compose.source_line,
    )


def _build_repropagated_compose(
    compose: ComposeDefinition,
    *,
    gtype: int,
    gadget_definitions: Mapping[str, GadgetDefinition],
    compose_definitions: Mapping[str, ComposeDefinition],
    jit_gadget_types_by_name: Mapping[str, jit_pb.JitGadgetType],
    codes: Mapping[str, CodeDefinition],
    ptype_of_code: Mapping[str, int],
    port_types: list[jit_pb.JitPortType],
) -> jit_pb.JitGadgetType:
    """Build a JitGadgetType for an ``@REPROPAGATE`` COMPOSE.

    Routes the COMPOSE through *both* pipelines and combines them:

    * The merge() / Rust JIT compiler pipeline produces the
      *structural* output: measurements, finished/unfinished checks,
      and readouts.  These reflect the sub-gadget composition
      (e.g. round-to-round comparison checks across repeated syndrome
      extraction) and must be preserved — otherwise users who add
      ``@REPROPAGATE`` silently lose the check basis their sub-gadgets
      define.
    * The flat-circuit pipeline (inlining the body into a synthetic
      :class:`GadgetDefinition` and running
      :func:`_build_jit_gadget_type`) produces the *propagation*
      output: ``correction_propagation``, ``physical_correction``,
      ``logical_correction``, and the noise-derived ``ERROR`` rows.
      These come from circuit-flow analysis on the inlined body and
      cover conditional logical corrections that matrix composition
      cannot represent (the teleportation case).

    The merge-derived check basis is fed into ``_build_jit_gadget_type``
    via its ``check_override`` parameter so the propagation/error
    derivation references the *same* check indices the merge pipeline
    produces.  This keeps everything self-consistent.

    Note: a deferred import is used to avoid a circular dependency
    between this module and ``jit_library_builder``.
    """
    from deq.transpiler.jit_library_builder import (  # local import: cycle
        _build_jit_gadget_type,
    )

    _reject_conditionals_under_repropagate(
        compose, gadget_definitions, compose_definitions
    )

    merge_jt = _build_merge_compose(
        compose,
        gtype=gtype,
        gadget_definitions=gadget_definitions,
        compose_definitions=compose_definitions,
        jit_gadget_types_by_name=jit_gadget_types_by_name,
        codes=codes,
        ptype_of_code=ptype_of_code,
        port_types=port_types,
    )
    synthetic = compose_to_synthetic_gadget(
        compose, gadget_definitions, compose_definitions, codes
    )
    finished, unfinished = _check_basis_from_jit_gadget_type(
        merge_jt, synthetic, codes
    )
    return _build_jit_gadget_type(
        synthetic,
        gtype,
        dict(ptype_of_code),
        dict(codes),
        check_override=(finished, unfinished),
    )


def _check_basis_from_jit_gadget_type(
    jt: jit_pb.JitGadgetType,
    synthetic: GadgetDefinition,
    codes: Mapping[str, CodeDefinition],
) -> tuple[list[tuple[frozenset[int], bool]], list[tuple[frozenset[int], bool]]]:
    """Recover the ``(members, parity)`` check basis from a JitGadgetType.

    Inverts the encoding done by ``_build_jit_gadget_type._build_check``:
    converts each :class:`JitGadgetType.Check`'s ``PresentMeasurement``
    list back into a ``frozenset`` of global measurement indices, and
    re-adds the implicit output-virtual index for each unfinished check.

    The global indexing matches what ``resolve_gadget_checks`` returns
    for *synthetic*: ``[input-virtual | internal | output-virtual]`` in
    that order, with input-virtual measurements grouped per input port.
    """
    input_ports = synthetic.input_ports
    output_ports = synthetic.output_ports
    input_stab_counts = [len(codes[p.code_name].stabilizers) for p in input_ports]
    iv_count = sum(input_stab_counts)
    internal_count = len(jt.base.measurements)
    ov_start = iv_count + internal_count
    num_ov = sum(len(codes[p.code_name].stabilizers) for p in output_ports)

    def members_of(check: jit_pb.JitGadgetType.Check) -> set[int]:
        members: set[int] = set()
        for m in check.measurements:
            if m.HasField("input_port"):
                global_idx = (
                    sum(input_stab_counts[: m.input_port]) + m.measurement_index
                )
            else:
                global_idx = iv_count + m.measurement_index
            members.add(global_idx)
        return members

    finished: list[tuple[frozenset[int], bool]] = [
        (frozenset(members_of(c)), bool(c.base.naturally_flipped))
        for c in jt.finished_checks
    ]
    unfinished: list[tuple[frozenset[int], bool]] = []
    for k, c in enumerate(jt.unfinished_checks):
        if k >= num_ov:
            raise ValueError(
                f"merge() produced more unfinished checks ({len(jt.unfinished_checks)}) "
                f"than the synthetic gadget has output-virtual measurements ({num_ov})"
            )
        members = members_of(c)
        members.add(ov_start + k)
        unfinished.append((frozenset(members), bool(c.base.naturally_flipped)))
    return finished, unfinished


# ===================================================================
# Public API — JIT-based compose builder
# ===================================================================


def build_compose_jit_gadget_type(
    compose: ComposeDefinition,
    *,
    gtype: int,
    gadget_definitions: Mapping[str, GadgetDefinition],
    compose_definitions: Mapping[str, ComposeDefinition],
    jit_gadget_types_by_name: Mapping[str, jit_pb.JitGadgetType],
    codes: Mapping[str, CodeDefinition],
    ptype_of_code: Mapping[str, int],
    port_types: list[jit_pb.JitPortType],
) -> jit_pb.JitGadgetType:
    """Build a composed JitGadgetType.

    By default uses the merge() / Rust JIT compiler pipeline (see
    :func:`_build_merge_compose`).  When the COMPOSE has the
    ``@REPROPAGATE`` decorator, a hybrid path is taken: the structural
    output (measurements, checks, readouts) still comes from
    merge(), but the propagation matrices and noise-derived ERRORs are
    recomputed from circuit flow on the inlined body.  See
    :func:`_build_repropagated_compose` for details.
    """
    validate_compose(
        compose,
        gadget_definitions=gadget_definitions,
        compose_definitions=compose_definitions,
    )

    if has_repropagate(compose):
        return _build_repropagated_compose(
            compose,
            gtype=gtype,
            gadget_definitions=gadget_definitions,
            compose_definitions=compose_definitions,
            jit_gadget_types_by_name=jit_gadget_types_by_name,
            codes=codes,
            ptype_of_code=ptype_of_code,
            port_types=port_types,
        )

    return _build_merge_compose(
        compose,
        gtype=gtype,
        gadget_definitions=gadget_definitions,
        compose_definitions=compose_definitions,
        jit_gadget_types_by_name=jit_gadget_types_by_name,
        codes=codes,
        ptype_of_code=ptype_of_code,
        port_types=port_types,
    )


def _build_merge_compose(
    compose: ComposeDefinition,
    *,
    gtype: int,
    gadget_definitions: Mapping[str, GadgetDefinition],
    compose_definitions: Mapping[str, ComposeDefinition],
    jit_gadget_types_by_name: Mapping[str, jit_pb.JitGadgetType],
    codes: Mapping[str, CodeDefinition],
    ptype_of_code: Mapping[str, int],
    port_types: list[jit_pb.JitPortType],
) -> jit_pb.JitGadgetType:
    """Build a composed JitGadgetType using mock gadgets + JIT compiler + merge.

    1. Expand the COMPOSE body and validate port-type compatibility
       and dangling outputs.
    2. Construct mock boundary gadgets to close off dangling ports.
    3. Run the Rust JIT compiler to produce a complete Library.
    4. Call ``merge()`` on only the real gadgets (excluding mocks).
    5. Convert the ``MergedGadget`` to a ``JitGadgetType``.

    The caller is expected to have already run :func:`validate_compose`.
    """
    inputs, outputs, items = _expand_compose_body(
        list(compose.body),
        gadget_definitions=gadget_definitions,
        compose_definitions=compose_definitions,
    )
    # The validators only care about real gadget applications; conditional
    # corrections sit between gadgets and do not change the wire's port
    # type or producer/consumer count.
    apps = [it for it in items if isinstance(it, GadgetApplication)]

    # ── Validate port-type compatibility between consecutive gadgets ──
    _validate_compose_port_types(
        compose,
        inputs,
        apps,
        gadget_definitions=gadget_definitions,
        compose_definitions=compose_definitions,
    )

    # ── Validate that no wire is bound to multiple INPUT or OUTPUT ──
    # ports.  Duplicate bindings (especially on OUTPUT) collapse in the
    # set-based dangling-output check below, but cause the Rust JIT
    # compiler to panic when the second consumer of the same source
    # port indexes an already-emptied check vector.
    _validate_compose_no_duplicate_port_wires(compose, inputs, outputs)

    # ── Validate that all live wires are declared as OUTPUT ──
    # Without this, a dangling wire causes the JIT compiler to block
    # forever waiting for a consumer of that port.
    _validate_compose_no_dangling_outputs(
        compose,
        inputs,
        outputs,
        apps,
        gadget_definitions=gadget_definitions,
        compose_definitions=compose_definitions,
    )

    in_ptypes = [ptype_of_code[p.code_name] for p in inputs]
    in_stabs = [len(codes[p.code_name].stabilizers) for p in inputs]
    in_obs = [num_frame_columns(codes[p.code_name]) for p in inputs]

    sub_jits = [jit_gadget_types_by_name[app.gadget_name] for app in apps]
    mock_base = max((jt.base.gtype for jt in sub_jits), default=0) + 100
    out_mock_gt = mock_base + len(inputs)

    # The output mock consumes every wire declared as a COMPOSE OUTPUT.
    # Without this, dangling sub-gadget output ports cause the JIT
    # compiler to block forever waiting for a downstream consumer.
    # Driving the mock from the declared OUTPUT list (rather than from
    # the last sub-gadget's outputs) is what makes "fan-out" patterns —
    # multiple parallel sub-gadgets each contributing a final wire —
    # work; using the last sub-gadget alone would leave earlier
    # sub-gadgets' wires dangling.
    output_wires: list[tuple[int, str]] = []
    for out in outputs:
        for wire in out.qubit_indices:
            output_wires.append((wire, out.code_name))
    out_ptypes = [ptype_of_code[code] for _, code in output_wires]
    out_stabs = [len(codes[code].stabilizers) for _, code in output_wires]
    out_obs = sum(num_frame_columns(codes[code]) for _, code in output_wires)

    gt_map: dict[int, jit_pb.JitGadgetType] = {}
    for jt in sub_jits:
        gt_map[jt.base.gtype] = jt

    has_input_mock = bool(inputs)

    # Use one input mock per compose input port so that each has a unique
    # gid.  This avoids a merge-function limitation where multiple merge
    # input ports sharing the same peer_gid causes incorrect per-port
    # measurement indices.
    if has_input_mock:
        for i, (pt, sc, obs) in enumerate(zip(in_ptypes, in_stabs, in_obs)):
            mock_gt = mock_base + i
            gt_map[mock_gt] = _mk_input_mock(mock_gt, [pt], [sc], obs)
    output_mock = _mk_output_mock(out_mock_gt, out_ptypes, out_stabs, out_obs)
    gt_map[out_mock_gt] = output_mock

    # ── Build JIT program ────────────────────────────────────────────
    prog: list[jit_pb.JitInstruction] = []
    real_gids: set[int] = set()
    gid = 1

    # Track, for each logical wire, which (gid, output_port_index) last
    # wrote to it. Connectors reference this mapping instead of blindly
    # pointing at the previous gadget.
    wire_source: dict[int, tuple[int, int]] = {}  # wire → (gid, port)
    # Track which port type each wire currently carries (so we can build
    # the identity-gadget modifier for a ConditionalCorrection).
    wire_ptype: dict[int, int] = {}
    # Track logical readout history for resolving ``rec[-k]`` in
    # CONDITIONAL statements: absolute_index → (gid, local_readout_index).
    readout_history: list[tuple[int, int]] = []
    # Lazily synthesized identity gadget types, keyed by port type.
    identity_gtype_of_ptype: dict[int, int] = {}
    next_synthetic_gtype = max(gt_map, default=0) + 1

    port_types_by_ptype: dict[int, jit_pb.JitPortType] = {
        pt.base.ptype: pt for pt in port_types
    }

    if has_input_mock:
        for i, inp in enumerate(inputs):
            mock_gt = mock_base + i
            prog.append(jit_pb.JitInstruction(gadget=pb.Gadget(gtype=mock_gt, gid=gid)))
            wire_source[inp.qubit_indices[0]] = (gid, 0)
            wire_ptype[inp.qubit_indices[0]] = in_ptypes[i]
            gid += 1
    for item in items:
        if isinstance(item, ConditionalCorrection):
            wire = item.wire
            if wire not in wire_source:
                raise ValueError(
                    f"COMPOSE {compose.name!r}: {item} references wire "
                    f"{wire} which has no producer"
                )
            wire_pt = wire_ptype[wire]
            if wire_pt not in port_types_by_ptype:
                raise ValueError(
                    f"COMPOSE {compose.name!r}: {item} references wire "
                    f"{wire} whose port type {wire_pt} is unknown"
                )
            instruction, new_identity_gt, next_synthetic_gtype = (
                emit_conditional_correction_instruction(
                    conditional=item,
                    error_context=f"COMPOSE {compose.name!r}",
                    wire_ptype=wire_pt,
                    wire_source=wire_source[wire],
                    readout_history=readout_history,
                    port_types_by_ptype=port_types_by_ptype,
                    identity_gtype_of_ptype=identity_gtype_of_ptype,
                    next_synthetic_gtype=next_synthetic_gtype,
                    gid=gid,
                )
            )
            if new_identity_gt is not None:
                gt_map[new_identity_gt.base.gtype] = new_identity_gt
            prog.append(instruction)
            real_gids.add(gid)
            # Identity gadget preserves the port type, so wire_ptype[wire]
            # stays as is; only the source pointer needs updating.
            wire_source[wire] = (gid, 0)
            gid += 1
            continue

        # item is a GadgetApplication
        app = item
        sub_jit = jit_gadget_types_by_name[app.gadget_name]
        in_wires = list(app.in_indices or [])
        out_wires = list(app.out_indices or [])
        connectors = []
        for wire in in_wires:
            src_gid, src_port = wire_source[wire]
            connectors.append(pb.Gadget.Connector(gid=src_gid, port=src_port))
        prog.append(
            jit_pb.JitInstruction(
                gadget=pb.Gadget(
                    gtype=sub_jit.base.gtype,
                    gid=gid,
                    connectors=connectors,
                )
            )
        )
        real_gids.add(gid)
        for local_r in range(len(sub_jit.base.readouts)):
            readout_history.append((gid, local_r))
        for port_idx, wire in enumerate(out_wires):
            wire_source[wire] = (gid, port_idx)
            wire_ptype[wire] = sub_jit.base.outputs[port_idx].ptype
        gid += 1
    # The output mock consumes one input port per declared compose
    # OUTPUT wire, each connected to whichever sub-gadget last wrote
    # that wire.  This handles fan-out (multiple sub-gadgets each
    # producing one final wire) as well as the linear case.
    connectors_out = [
        pb.Gadget.Connector(gid=wire_source[wire][0], port=wire_source[wire][1])
        for wire, _ in output_wires
    ]
    prog.append(
        jit_pb.JitInstruction(
            gadget=pb.Gadget(gtype=out_mock_gt, gid=gid, connectors=connectors_out)
        )
    )

    jit_lib = jit_pb.JitLibrary(
        gadget_types=list(gt_map.values()), port_types=list(port_types), program=prog
    )

    # ── JIT compile → merge real gadgets ─────────────────────────────
    deq_bin = static_jit_compiler(jit_lib)
    merged = merge(deq_bin, real_gids)
    return merged.to_jit_gadget_type(gtype=gtype, name=compose.name)


# ===================================================================
# Mock gadget builders for JIT compose
# ===================================================================


def _mk_input_mock(
    gtype: int,
    out_ptypes: list[int],
    stab_counts: list[int],
    n_obs: int,
) -> jit_pb.JitGadgetType:
    n = sum(stab_counts)
    return jit_pb.JitGadgetType(
        base=pb.GadgetType(
            gtype=gtype,
            name="__input_mock__",
            measurements=[pb.GadgetType.Measurement()] * n,
            outputs=[pb.GadgetType.Port(ptype=pt) for pt in out_ptypes],
            correction_propagation=util_pb.BitMatrix(rows=n_obs, cols=1),
            readout_propagation=util_pb.BitMatrix(cols=1),
            logical_correction=util_pb.BitMatrix(rows=n_obs),
            physical_correction=util_pb.BitMatrix(rows=n_obs, cols=n),
        ),
        unfinished_checks=[
            jit_pb.JitGadgetType.Check(
                base=pb.CheckModelType.Check(),
                measurements=[
                    jit_pb.JitGadgetType.PresentMeasurement(measurement_index=i)
                ],
            )
            for i in range(n)
        ],
    )


def _mk_output_mock(
    gtype: int,
    in_ptypes: list[int],
    stab_counts: list[int],
    n_obs: int,
) -> jit_pb.JitGadgetType:
    n = sum(stab_counts)
    fin = []
    c = 0
    for pi, ns in enumerate(stab_counts):
        for s in range(ns):
            fin.append(
                jit_pb.JitGadgetType.Check(
                    base=pb.CheckModelType.Check(),
                    measurements=[
                        jit_pb.JitGadgetType.PresentMeasurement(
                            input_port=pi, measurement_index=s
                        ),
                        jit_pb.JitGadgetType.PresentMeasurement(measurement_index=c),
                    ],
                )
            )
            c += 1
    return jit_pb.JitGadgetType(
        base=pb.GadgetType(
            gtype=gtype,
            name="__output_mock__",
            measurements=[pb.GadgetType.Measurement()] * n,
            inputs=[pb.GadgetType.Port(ptype=pt) for pt in in_ptypes],
            correction_propagation=util_pb.BitMatrix(cols=n_obs + 1),
            readout_propagation=util_pb.BitMatrix(cols=n_obs + 1),
            logical_correction=util_pb.BitMatrix(),
            physical_correction=util_pb.BitMatrix(cols=n),
        ),
        finished_checks=fin,
    )


def mk_identity_gadget_type(
    gtype: int,
    ptype: int,
    n_obs: int,
    stab_count: int,
) -> jit_pb.JitGadgetType:
    """Build a ``JitGadgetType`` for a measurement-free identity gadget.

    The identity gadget has a single INPUT port and a single OUTPUT
    port, both of type ``ptype``.  Its ``correction_propagation`` is
    the identity matrix on the port observables (with a zero affine
    column), so the wire frame passes through unchanged.  Other
    propagation matrices are empty.

    It is used as a placeholder host for
    :class:`pb.GadgetModifier.remote_conditional_correction` modifiers
    derived from ``CONDITIONAL`` statements inside ``COMPOSE`` or
    ``PROGRAM`` bodies.  Because the gadget is inserted into the JIT
    program *after* the gadget whose readout it conditions on, the
    program validator's ordering constraint is naturally satisfied.

    The ``stab_count`` output stabilizers each become an unfinished
    check linking the matching input-virtual stabilizer to the implicit
    output-virtual stabilizer — bridging stabilizer measurements across
    the gadget so the JIT compiler can chain them with downstream
    consumers.
    """
    identity_cp = util_pb.BitMatrix(
        rows=n_obs,
        cols=n_obs + 1,
        i=list(range(n_obs)),
        j=list(range(n_obs)),
    )
    return jit_pb.JitGadgetType(
        base=pb.GadgetType(
            gtype=gtype,
            name=f"__identity_pt{ptype}__",
            measurements=[],
            inputs=[pb.GadgetType.Port(ptype=ptype)],
            outputs=[pb.GadgetType.Port(ptype=ptype)],
            correction_propagation=identity_cp,
            readout_propagation=util_pb.BitMatrix(cols=n_obs + 1),
            logical_correction=util_pb.BitMatrix(rows=n_obs),
            physical_correction=util_pb.BitMatrix(rows=n_obs),
        ),
        unfinished_checks=[
            jit_pb.JitGadgetType.Check(
                base=pb.CheckModelType.Check(),
                measurements=[
                    jit_pb.JitGadgetType.PresentMeasurement(
                        input_port=0, measurement_index=s
                    )
                ],
            )
            for s in range(stab_count)
        ],
    )


def _single_column_bitmatrix(
    rows: int, flipped_rows: Sequence[int]
) -> util_pb.BitMatrix:
    """Build a 1-column ``BitMatrix`` that flips the given rows.

    Used for the ``correction`` field of
    :class:`pb.RemoteConditionalCorrection` modifiers, whose layout is
    always a single column (the modifier is conditioned on a single
    readout bit).
    """
    flipped_list = list(flipped_rows)
    return util_pb.BitMatrix(
        rows=rows,
        cols=1,
        i=flipped_list,
        j=[0] * len(flipped_list),
    )


def emit_conditional_correction_instruction(
    *,
    conditional: ConditionalCorrection,
    error_context: str,
    wire_ptype: int,
    wire_source: tuple[int, int],
    readout_history: Sequence[tuple[int, int]],
    port_types_by_ptype: Mapping[int, jit_pb.JitPortType],
    identity_gtype_of_ptype: dict[int, int],
    next_synthetic_gtype: int,
    gid: int,
) -> tuple[jit_pb.JitInstruction, jit_pb.JitGadgetType | None, int]:
    """Build the JIT instruction realising one ``CONDITIONAL`` statement.

    The instruction is a synthesised identity gadget that consumes the
    wire from its current producer (``wire_source``) and re-emits it
    with a :class:`pb.RemoteConditionalCorrection` modifier conditioned
    on ``readout_history[-conditional.readout_offset]``.

    Returns a triple:

    * the new :class:`jit_pb.JitInstruction` (caller appends it to its
      own program stream and bumps its own ``gid`` counter);
    * a newly created :class:`jit_pb.JitGadgetType` to register in the
      library, or ``None`` if the cache (``identity_gtype_of_ptype``)
      already contained an identity gadget for this port type;
    * the updated ``next_synthetic_gtype`` counter.

    Mutates ``identity_gtype_of_ptype`` (registering the gtype on first
    use of each port type).
    """
    from deq.transpiler.jit_library_builder import pauli_to_observable_flips

    if conditional.readout_offset < 1:
        raise ValueError(
            f"{error_context}: {conditional} requires k >= 1 in "
            f"rec[-k]; got rec[-{conditional.readout_offset}]"
        )
    if conditional.readout_offset > len(readout_history):
        raise ValueError(
            f"{error_context}: {conditional} references "
            f"rec[-{conditional.readout_offset}] but only "
            f"{len(readout_history)} logical readout(s) have been "
            f"produced so far"
        )

    remote_gid, remote_local_readout = readout_history[
        len(readout_history) - conditional.readout_offset
    ]

    jit_port_type = port_types_by_ptype[wire_ptype]
    n_obs = len(jit_port_type.base.observables)
    stab_count = len(jit_port_type.stabilizers)
    flip_rows = pauli_to_observable_flips(conditional.paulis, jit_port_type.k)

    newly_created: jit_pb.JitGadgetType | None = None
    if wire_ptype in identity_gtype_of_ptype:
        identity_gtype = identity_gtype_of_ptype[wire_ptype]
    else:
        identity_gtype = next_synthetic_gtype
        next_synthetic_gtype += 1
        identity_gtype_of_ptype[wire_ptype] = identity_gtype
        newly_created = mk_identity_gadget_type(
            gtype=identity_gtype,
            ptype=wire_ptype,
            n_obs=n_obs,
            stab_count=stab_count,
        )

    src_gid, src_port = wire_source
    modifier = pb.GadgetModifier(
        remote_conditional_correction=pb.RemoteConditionalCorrection(
            remote_readouts=[
                pb.RemoteConditionalCorrection.RemoteReadout(
                    gid=remote_gid,
                    readout_index=remote_local_readout,
                )
            ],
            correction=_single_column_bitmatrix(n_obs, flip_rows),
        )
    )
    instruction = jit_pb.JitInstruction(
        gadget=pb.Gadget(
            gtype=identity_gtype,
            gid=gid,
            connectors=[pb.Gadget.Connector(gid=src_gid, port=src_port)],
            modifier=modifier,
        )
    )
    return instruction, newly_created, next_synthetic_gtype
