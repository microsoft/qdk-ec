"""Build a composed ``JitGadgetType`` from a ``COMPOSE`` definition.

The compose builder constructs a ``JitLibrary`` with mock boundary gadgets
(to close off dangling ports), runs the Rust JIT compiler to produce a
complete ``Library``, then calls ``merge()`` on only the real (non-mock)
gadgets.  The ``merge()`` function handles observable propagation, check
classification, and error mapping — producing a ``MergedGadget`` that is
converted to a ``JitGadgetType``.
"""

# pylint: disable=no-member


from typing import Callable, Mapping

import deq.proto.deq_bin_pb2 as pb
import deq.proto.deq_jit_pb2 as jit_pb
import deq.proto.util_pb2 as util_pb

from deq.circuit.model import (
    CodeDefinition,
    ComposeDefinition,
    GadgetApplication,
    GadgetDefinition,
    InputPort,
    Instruction,
    OutputPort,
    QubitTarget,
    RepeatBlock,
)
from deq.compiler.jit_compiler import static_jit_compiler
from deq.spec.canonical import merge
from deq.transpiler.jit_transpiler import num_frame_columns

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
    unsupported = [d for d in compose.decorators if d.name != "GTYPE"]
    if unsupported:
        names = ", ".join(d.name for d in unsupported)
        raise ValueError(
            f"COMPOSE {compose.name!r}: only @GTYPE is supported "
            f"on COMPOSE definitions (got @{names})"
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
) -> tuple[list[InputPort], list[OutputPort], list[GadgetApplication]]:
    inputs: list[InputPort] = []
    outputs: list[OutputPort] = []
    apps: list[GadgetApplication] = []

    def _lookup(name: str) -> GadgetDefinition | ComposeDefinition | None:
        if name in gadget_definitions:
            return gadget_definitions[name]
        if name in compose_definitions:
            return compose_definitions[name]
        return None

    def _walk(items: list) -> None:
        for stmt in items:
            if isinstance(stmt, InputPort):
                inputs.append(stmt)
            elif isinstance(stmt, OutputPort):
                outputs.append(stmt)
            elif isinstance(stmt, GadgetApplication):
                if stmt.is_shortcut:
                    sub_def = _lookup(stmt.gadget_name)
                    if sub_def is None:
                        apps.append(stmt)
                    else:
                        indices = list(stmt.in_indices or [])
                        n_in = len(sub_def.input_ports)
                        n_out = len(sub_def.output_ports)
                        apps.append(
                            GadgetApplication(
                                gadget_name=stmt.gadget_name,
                                in_indices=indices[:n_in],
                                out_indices=indices[:n_out],
                            )
                        )
                else:
                    apps.append(stmt)
            elif isinstance(stmt, Instruction):
                sub_def = _lookup(stmt.name)
                if sub_def is not None:
                    apps.append(_instruction_to_application(stmt, sub_def=sub_def))
            elif isinstance(stmt, RepeatBlock):
                for _ in range(stmt.count):
                    _walk(stmt.body)

    _walk(body)
    return inputs, outputs, apps


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

    live: dict[int, str] = {}
    for inp in inputs:
        for wire in inp.qubit_indices:
            live[wire] = f"COMPOSE INPUT ({inp.code_name})"

    for app_idx, app in enumerate(apps):
        sub_def = _get_def(app.gadget_name)
        n_in = len(sub_def.input_ports)
        n_out = len(sub_def.output_ports)
        for wire in list(app.in_indices or [])[:n_in]:
            live.pop(wire, None)
        for port_idx, wire in enumerate(list(app.out_indices or [])[:n_out]):
            live[wire] = (
                f"step {app_idx + 1} ({app.gadget_name}, " f"output port {port_idx})"
            )

    declared: dict[int, str] = {}
    for out in outputs:
        for wire in out.qubit_indices:
            declared[wire] = out.code_name

    dangling = sorted(set(live) - set(declared))
    missing = sorted(set(declared) - set(live))
    if not dangling and not missing:
        return

    loc = f" ({compose.source_file})" if compose.source_file is not None else ""
    msg_lines = [
        f"COMPOSE {compose.name!r}{loc}: declared OUTPUT ports do not "
        f"match the wires that are still live at the end of the body.",
    ]
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
    """Build a composed JitGadgetType using mock gadgets + JIT compiler + merge.

    1. Validate and expand the COMPOSE body.
    2. Construct mock boundary gadgets to close off dangling ports.
    3. Run the Rust JIT compiler to produce a complete Library.
    4. Call ``merge()`` on only the real gadgets (excluding mocks).
    5. Convert the ``MergedGadget`` to a ``JitGadgetType``.
    """
    validate_compose(
        compose,
        gadget_definitions=gadget_definitions,
        compose_definitions=compose_definitions,
    )
    inputs, outputs, apps = _expand_compose_body(
        list(compose.body),
        gadget_definitions=gadget_definitions,
        compose_definitions=compose_definitions,
    )

    # ── Validate port-type compatibility between consecutive gadgets ──
    _validate_compose_port_types(
        compose,
        inputs,
        apps,
        gadget_definitions=gadget_definitions,
        compose_definitions=compose_definitions,
    )

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

    if has_input_mock:
        for i, inp in enumerate(inputs):
            mock_gt = mock_base + i
            prog.append(jit_pb.JitInstruction(gadget=pb.Gadget(gtype=mock_gt, gid=gid)))
            wire_source[inp.qubit_indices[0]] = (gid, 0)
            gid += 1
    for app_idx, app in enumerate(apps):
        in_wires = list(app.in_indices or [])
        out_wires = list(app.out_indices or [])
        connectors = []
        for wire in in_wires:
            src_gid, src_port = wire_source[wire]
            connectors.append(pb.Gadget.Connector(gid=src_gid, port=src_port))
        prog.append(
            jit_pb.JitInstruction(
                gadget=pb.Gadget(
                    gtype=sub_jits[app_idx].base.gtype,
                    gid=gid,
                    connectors=connectors,
                )
            )
        )
        real_gids.add(gid)
        # Update wire_source: output port i writes to out_wires[i].
        for port_idx, wire in enumerate(out_wires):
            wire_source[wire] = (gid, port_idx)
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
