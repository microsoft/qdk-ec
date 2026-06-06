"""CLI command to compute and print the logical action of DEQ gadgets.

Usage::

    deq logical-action path/to/deq/*.deq
"""

import traceback
from collections.abc import Sequence

import arguably

from errata.codes import SubsystemCode
from errata.circuits.circuit import Location
from errata.simulation.stabilizer.circuit_action import CircuitAction, action_of
from deq.circuit.model import (
    CodeDefinition,
    ComposeDefinition,
    ComposeStatement,
    ConditionalStatement,
    GadgetApplication,
    GadgetDefinition,
    InputPort,
    Instruction,
    OutputPort,
    QubitTarget,
    RepeatBlock,
)
from deq.errata.errata_utils import (
    build_port_code,
    extend_support,
    make_wrt,
)
from deq.errata.stim_to_locations import instructions_to_locations


@arguably.command
def logical_action(
    *deq_files: str,
    gadgets: str | None = None,
    mako: list[str] | None = None,
    skip_mako_warning: bool = False,
) -> None:
    """Compute and print the logical action of each gadget in the given DEQ files.

    By default, only gadgets/composes defined directly in the listed files
    are processed (imported definitions are skipped).  Use ``--gadgets``
    to select specific gadgets by name (searches imports too).

    Example::

        deq logical-action codes/.../deq/*.deq
        deq logical-action codes/.../deq/*.deq --gadgets "CxAll,PrepareZAll"
    """
    from pathlib import Path
    from deq.circuit.parser import render_and_parse_file

    from deq.circuit.mako_support import parse_mako_vars

    # ── Parse all DEQ files ──────────────────────────────────────────
    paths = [Path(f) for f in deq_files]
    if not paths:
        print("No DEQ files specified.")
        return

    # Resolve listed file paths to absolute for source_file comparison
    listed_sources = {str(p.resolve()) for p in paths}

    # Parse the --gadgets filter
    filter_names: set[str] | None = None
    if gadgets is not None:
        filter_names = {name.strip() for name in gadgets.split(",") if name.strip()}

    all_code_defs: dict[str, CodeDefinition] = {}
    all_gadget_defs_ordered: list[GadgetDefinition] = []
    all_gadget_defs_map: dict[str, GadgetDefinition] = {}
    all_compose_defs: list[ComposeDefinition] = []
    all_compose_defs_map: dict[str, ComposeDefinition] = {}

    for fpath in paths:
        mako_vars = parse_mako_vars(mako) if mako else None
        deq_file = render_and_parse_file(
            fpath,
            mako_defs=mako_vars,
            skip_mako_warning=skip_mako_warning,
        )
        for defn in deq_file.definitions:
            if isinstance(defn, CodeDefinition):
                all_code_defs[defn.name] = defn
            elif isinstance(defn, GadgetDefinition):
                all_gadget_defs_ordered.append(defn)
                all_gadget_defs_map[defn.name] = defn
            elif isinstance(defn, ComposeDefinition):
                all_compose_defs.append(defn)
                all_compose_defs_map[defn.name] = defn

    if not all_code_defs:
        print("No CODE definitions found.")
        return

    def _should_process(defn: GadgetDefinition | ComposeDefinition) -> bool:
        """Return True if this definition should be processed."""
        if filter_names is not None:
            return defn.name in filter_names
        # Default: only definitions from listed files (not imports)
        src = getattr(defn, "source_file", None)
        if src is None:
            return True
        return str(Path(src).resolve()) in listed_sources

    # ── Process each GADGET ───────────────────────────────────────────
    for gdef in all_gadget_defs_ordered:
        if _should_process(gdef):
            _process_gadget(gdef, all_code_defs)

    # ── Process each COMPOSE ──────────────────────────────────────────
    for cdef in all_compose_defs:
        if _should_process(cdef):
            _process_compose(
                cdef,
                all_code_defs,
                all_gadget_defs_map,
                all_compose_defs_map,
            )


# ── GADGET processing ────────────────────────────────────────────────


def gadget_action(
    gdef: GadgetDefinition,
    all_code_defs: dict[str, CodeDefinition],
) -> CircuitAction:
    """Compute the logical action of a GADGET definition.

    Supports gadgets with multiple ports using different codes.

    Returns a :class:`CircuitAction` containing the logical mapping,
    measured observables, and output stabilizers.

    Raises :class:`ValueError` if the gadget has no ports or a port's
    code is not found.
    """
    input_ports = [s for s in gdef.body if isinstance(s, InputPort)]
    output_ports = [s for s in gdef.body if isinstance(s, OutputPort)]

    if not input_ports and not output_ports:
        raise ValueError(f"Gadget '{gdef.name}' has no INPUT/OUTPUT ports")

    code_in = build_port_code(input_ports, all_code_defs)
    code_out = build_port_code(output_ports, all_code_defs)

    locations = instructions_to_locations(gdef.body)

    # VIRTUAL logical statements (e.g. VIRTUAL LZ0) are expanded into
    # physical Pauli operations from the output code's logical basis
    # and appended to the circuit. This lets action_of compute the
    # correct logical action naturally.
    from deq.circuit.model import VirtualLogicalStatement
    from deq.circuit.model import LogicalPauliTarget as LPT

    virtual_targets: list[LPT] = []
    for stmt in gdef.body:
        if isinstance(stmt, VirtualLogicalStatement):
            virtual_targets.extend(stmt.targets)

    if virtual_targets and code_out is not None:
        from errata.circuits.clifford import PauliOperation

        # Pre-compute the per-port logical offset for translating
        # port-qualified targets (``OUT<p>.L<P><i>``) to the global
        # logical index used by ``code_out.logical_basis``.
        per_port_logical_offset: list[int] = []
        acc = 0
        for op in output_ports:
            per_port_logical_offset.append(acc)
            acc += len(all_code_defs[op.code_name].logicals)

        max_level = max((loc.level for loc in locations), default=-1)
        level = max_level + 1
        for target in virtual_targets:
            if target.port_kind is not None:
                assert target.port_index is not None
                if target.port_kind != "OUT":
                    raise ValueError(
                        f"Gadget '{gdef.name}': VIRTUAL target {target!s} "
                        f"has direction {target.port_kind!r}; VIRTUAL "
                        f"references OUTPUT port logicals only"
                    )
                if not 0 <= target.port_index < len(output_ports):
                    raise ValueError(
                        f"Gadget '{gdef.name}': VIRTUAL target {target!s}: "
                        f"port index out of range (only {len(output_ports)} "
                        f"OUTPUT port(s))"
                    )
                global_logical_idx = (
                    per_port_logical_offset[target.port_index] + target.index
                )
            else:
                global_logical_idx = target.index
            # LX<i> → logical_basis[2*i], LZ<i> → logical_basis[2*i+1]
            basis_idx = 2 * global_logical_idx + (0 if target.pauli == "X" else 1)
            logical_pauli = code_out.logical_basis[basis_idx]
            for qubit, character in logical_pauli.characters.items():
                locations.append(Location(level, PauliOperation(character, qubit)))
            level += 1

    circuit_qubits: set[int] = set()
    for loc in locations:
        circuit_qubits.update(loc.operation.support)

    if code_in is not None:
        code_in = extend_support(code_in, circuit_qubits)
    if code_out is not None:
        code_out = extend_support(code_out, circuit_qubits)

    wrt = make_wrt(code_in, code_out, circuit_qubits)
    return action_of(locations, with_respect_to=wrt)


def _process_gadget(
    gdef: GadgetDefinition,
    all_code_defs: dict[str, CodeDefinition],
) -> None:
    """Compute and print logical action of a GADGET definition."""

    input_ports = [s for s in gdef.body if isinstance(s, InputPort)]
    output_ports = [s for s in gdef.body if isinstance(s, OutputPort)]

    if not input_ports and not output_ports:
        print(f"\n{'='*60}")
        print(f"Gadget: {gdef.name}")
        print("  (no INPUT/OUTPUT ports — skipping)")
        return

    all_ports = input_ports + output_ports
    port_code_names = sorted({p.code_name for p in all_ports})
    missing = [name for name in port_code_names if name not in all_code_defs]
    if missing:
        print(f"\n{'='*60}")
        print(f"Gadget: {gdef.name}")
        print(f"  (code(s) {missing} not found — skipping)")
        return

    codes_str = ", ".join(
        f"{name} [[{all_code_defs[name].n},{all_code_defs[name].k}]]"
        for name in port_code_names
    )
    locations = instructions_to_locations(gdef.body)

    print(f"\n{'='*60}")
    print(f"Gadget: {gdef.name}")
    print(f"  Ports: {len(input_ports)} in, {len(output_ports)} out")
    print(f"  Code: {codes_str}")
    print(f"  Locations: {len(locations)}")

    if any(isinstance(s, ConditionalStatement) for s in gdef.body):
        print(
            "  WARNING: CONDITIONAL statements are present but not "
            "included in the logical action analysis"
        )

    try:
        act = gadget_action(gdef, all_code_defs)
    except Exception as exc:
        print(f"  ERROR computing action: {exc}")
        traceback.print_exc()
        return

    _print_action(act)


# ── COMPOSE processing ───────────────────────────────────────────────


def _resolve_compose_body(
    body: Sequence[ComposeStatement | Instruction | RepeatBlock],
    gadget_map: dict[str, GadgetDefinition],
    compose_map: dict[str, ComposeDefinition],
    seen: set[str] | None = None,
) -> list[ComposeStatement]:
    """Expand REPEAT blocks, shortcut Instructions, and nested composes.

    The DEQ grammar parses shortcut gadget/compose applications
    (e.g. ``Syndrome 0``) as :class:`Instruction` objects.  This function
    converts them to :class:`GadgetApplication` and recursively expands
    REPEAT blocks.

    When an application refers to a compose (not a gadget), its body is
    recursively expanded and the inner gadget applications are remapped
    to the parent's port indices.
    """
    if seen is None:
        seen = set()
    result: list[ComposeStatement] = []
    for stmt in body:
        if isinstance(stmt, RepeatBlock):
            inner = _resolve_compose_body(stmt.body, gadget_map, compose_map, seen)
            for _ in range(stmt.count):
                result.extend(inner)
        elif isinstance(stmt, Instruction) and stmt.name in gadget_map:
            indices = [t.index for t in stmt.targets if isinstance(t, QubitTarget)]
            result.append(
                GadgetApplication(
                    gadget_name=stmt.name,
                    in_indices=indices,
                    out_indices=indices,
                )
            )
        elif isinstance(stmt, Instruction) and stmt.name in compose_map:
            indices = [t.index for t in stmt.targets if isinstance(t, QubitTarget)]
            app = GadgetApplication(
                gadget_name=stmt.name,
                in_indices=indices,
                out_indices=indices,
            )
            result.extend(_expand_compose_app(app, gadget_map, compose_map, seen))
        elif isinstance(stmt, GadgetApplication) and stmt.gadget_name in compose_map:
            result.extend(_expand_compose_app(stmt, gadget_map, compose_map, seen))
        else:
            result.append(stmt)
    return result


def _expand_compose_app(
    app: GadgetApplication,
    gadget_map: dict[str, GadgetDefinition],
    compose_map: dict[str, ComposeDefinition],
    seen: set[str],
) -> list[ComposeStatement]:
    """Expand a compose application into its constituent gadget applications.

    Builds a port index remapping from the nested compose's internal port
    indices to the parent's port indices, then recursively resolves the
    nested compose's body.
    """
    compose_name = app.gadget_name
    if compose_name in seen:
        raise ValueError(
            f"Circular compose reference detected: '{compose_name}' "
            f"is already being expanded (chain: {seen})"
        )

    cdef = compose_map[compose_name]
    inner_seen = seen | {compose_name}

    # Build port index remapping: inner_port_idx → parent_port_idx
    # The compose's INPUT/OUTPUT declarations define its internal port indices.
    # The application's in_indices/out_indices define the parent's port indices.
    inner_inputs = cdef.input_ports
    inner_outputs = cdef.output_ports
    app_in = app.in_indices or []
    app_out = app.out_indices or []

    port_remap: dict[int, int] = {}
    for port_pos, parent_idx in enumerate(app_in):
        if port_pos < len(inner_inputs):
            for inner_idx in inner_inputs[port_pos].qubit_indices:
                port_remap[inner_idx] = parent_idx
    for port_pos, parent_idx in enumerate(app_out):
        if port_pos < len(inner_outputs):
            for inner_idx in inner_outputs[port_pos].qubit_indices:
                port_remap.setdefault(inner_idx, parent_idx)

    # Recursively resolve the nested compose's body
    resolved = _resolve_compose_body(cdef.body, gadget_map, compose_map, inner_seen)

    # Remap port indices in the expanded gadget applications
    remapped: list[ComposeStatement] = []
    for item in resolved:
        if isinstance(item, GadgetApplication):
            remapped.append(
                GadgetApplication(
                    gadget_name=item.gadget_name,
                    in_indices=[port_remap.get(i, i) for i in (item.in_indices or [])],
                    out_indices=[
                        port_remap.get(i, i) for i in (item.out_indices or [])
                    ],
                    decorators=item.decorators,
                )
            )
        elif isinstance(item, (InputPort, OutputPort)):
            pass  # Drop inner compose's port declarations
        else:
            remapped.append(item)
    return remapped


def _process_compose(
    cdef: ComposeDefinition,
    all_code_defs: dict[str, CodeDefinition],
    gadget_map: dict[str, GadgetDefinition],
    compose_map: dict[str, ComposeDefinition],
) -> None:
    """Compute and print logical action of a COMPOSE definition."""

    resolved_body = _resolve_compose_body(cdef.body, gadget_map, compose_map)

    compose_inputs = [s for s in resolved_body if isinstance(s, InputPort)]
    compose_outputs = [s for s in resolved_body if isinstance(s, OutputPort)]
    gadget_apps = [s for s in resolved_body if isinstance(s, GadgetApplication)]
    correction_stmts = [
        s
        for s in resolved_body
        if isinstance(s, Instruction) and s.name.upper() == "CX"
    ]

    if not compose_inputs and not compose_outputs:
        print(f"\n{'='*60}")
        print(f"Compose: {cdef.name}")
        print("  (no INPUT/OUTPUT ports — skipping)")
        return

    all_compose_ports = compose_inputs + compose_outputs
    port_code_names = sorted({p.code_name for p in all_compose_ports})
    missing = [name for name in port_code_names if name not in all_code_defs]
    if missing:
        print(f"\n{'='*60}")
        print(f"Compose: {cdef.name}")
        print(f"  (code(s) {missing} not found — skipping)")
        return

    if correction_stmts:
        print(
            f"  WARNING: {len(correction_stmts)} correction statement(s) "
            f"(CX rec[...]) ignored — not yet supported by logical-action"
        )

    # Build a mapping from compose port index → n (number of physical qubits).
    # Each compose port is associated with a code via the INPUT/OUTPUT that
    # first declares it; all gadgets using that port must agree on the code.
    port_n: dict[int, int] = {}
    for p in compose_inputs + compose_outputs:
        for idx in p.qubit_indices:
            if idx not in port_n:
                code_def = all_code_defs.get(p.code_name)
                port_n[idx] = code_def.n if code_def else 1

    # Collect all port indices used in the COMPOSE
    all_port_indices: set[int] = set()
    for p in compose_inputs:
        all_port_indices.update(p.qubit_indices)
    for p in compose_outputs:
        all_port_indices.update(p.qubit_indices)
    for app in gadget_apps:
        if app.in_indices:
            all_port_indices.update(app.in_indices)
        if app.out_indices:
            all_port_indices.update(app.out_indices)

    # Assign qubit ranges per port: port_base[p] is the first physical qubit
    # for compose port p, and the range is [port_base[p], port_base[p]+n_p).
    port_base: dict[int, int] = {}
    next_qubit = 0
    for p in sorted(all_port_indices):
        port_base[p] = next_qubit
        n_p = port_n.get(p, 1)
        next_qubit += n_p
    ancilla_base = next_qubit

    all_locations = []
    level_offset = 0

    # Live permutation tracking: after each gadget, code qubit i on compose
    # port p is at compose qubit live_pos[p][i].
    live_pos: dict[int, list[int]] = {
        p: [port_base[p] + i for i in range(port_n.get(p, 1))] for p in all_port_indices
    }

    for app in gadget_apps:
        inner_name = app.gadget_name
        if inner_name not in gadget_map:
            print(f"\n{'='*60}")
            print(f"Compose: {cdef.name}")
            print(f"  (inner gadget '{inner_name}' not found — skipping)")
            return

        inner_gdef = gadget_map[inner_name]
        inner_inputs = [s for s in inner_gdef.body if isinstance(s, InputPort)]
        inner_outputs = [s for s in inner_gdef.body if isinstance(s, OutputPort)]

        in_indices = app.in_indices or []
        if app.out_indices is not None:
            out_indices = app.out_indices
        elif inner_outputs:
            out_indices = in_indices[: len(inner_outputs)]
        else:
            out_indices = []

        # Build qubit remapping: inner gadget qubit → compose qubit
        qubit_remap: dict[int, int] = {}

        # Map input port qubits using the live permutation so that we
        # correctly follow where each code qubit currently sits.
        for port_pos, compose_port_idx in enumerate(in_indices):
            if port_pos < len(inner_inputs):
                inner_port = inner_inputs[port_pos]
                for code_q_idx, inner_q in enumerate(inner_port.qubit_indices):
                    qubit_remap[inner_q] = live_pos[compose_port_idx][code_q_idx]

        # Map output port qubits (only adds mappings for qubits not already
        # covered by INPUT, i.e. prepare-gadget output-only qubits).
        for port_pos, compose_port_idx in enumerate(out_indices):
            if port_pos < len(inner_outputs):
                inner_port = inner_outputs[port_pos]
                for code_q_idx, inner_q in enumerate(inner_port.qubit_indices):
                    qubit_remap.setdefault(
                        inner_q, live_pos[compose_port_idx][code_q_idx]
                    )

        # Map ancilla qubits (not in any port) to unique IDs
        inner_locations = instructions_to_locations(inner_gdef.body)
        for loc in inner_locations:
            for q in loc.operation.support:
                if q not in qubit_remap:
                    qubit_remap[q] = ancilla_base
                    ancilla_base += 1

        # Remap locations and shift levels
        max_inner_level = max((loc.level for loc in inner_locations), default=0)
        for loc in inner_locations:
            remapped_op = loc.operation.relocated_by(qubit_remap)
            all_locations.append(Location(loc.level + level_offset, remapped_op))

        level_offset += max_inner_level + 1

        # Update live_pos for each output port.  After this gadget, code
        # qubit i on compose port p is at qubit_remap[OUTPUT[i]].
        for port_pos, compose_port_idx in enumerate(out_indices):
            if port_pos < len(inner_outputs):
                inner_port = inner_outputs[port_pos]
                new_live = [0] * port_n.get(
                    compose_port_idx, len(inner_port.qubit_indices)
                )
                for code_q_idx, inner_q in enumerate(inner_port.qubit_indices):
                    new_live[code_q_idx] = qubit_remap[inner_q]
                live_pos[compose_port_idx] = new_live

    if compose_inputs:
        synthetic_in_ports: list[InputPort] = []
        for p in compose_inputs:
            for idx in p.qubit_indices:
                n_p = port_n.get(idx, 1)
                synthetic_in_ports.append(
                    InputPort(
                        code_name=p.code_name,
                        qubit_indices=[port_base[idx] + i for i in range(n_p)],
                    )
                )
        code_in = build_port_code(synthetic_in_ports, all_code_defs)
    else:
        code_in = None

    if compose_outputs:
        synthetic_out_ports: list[OutputPort] = []
        for p in compose_outputs:
            for idx in p.qubit_indices:
                synthetic_out_ports.append(
                    OutputPort(
                        code_name=p.code_name,
                        qubit_indices=live_pos[idx],
                    )
                )
        code_out = build_port_code(synthetic_out_ports, all_code_defs)
    else:
        code_out = None

    circuit_qubits: set[int] = set()
    for loc in all_locations:
        circuit_qubits.update(loc.operation.support)

    if code_in is not None:
        code_in = extend_support(code_in, circuit_qubits)
    if code_out is not None:
        code_out = extend_support(code_out, circuit_qubits)

    wrt = make_wrt(code_in, code_out, circuit_qubits)

    compose_code_names = sorted({p.code_name for p in compose_inputs + compose_outputs})
    codes_str = ", ".join(
        f"{name} [[{all_code_defs[name].n},{all_code_defs[name].k}]]"
        for name in compose_code_names
    )
    print(f"\n{'='*60}")
    print(f"Compose: {cdef.name}  ({len(gadget_apps)} gadget(s))")
    print(f"  Ports: {len(compose_inputs)} in, {len(compose_outputs)} out")
    print(f"  Code: {codes_str}")
    print(f"  Gadgets: {', '.join(app.gadget_name for app in gadget_apps)}")
    print(f"  Locations: {len(all_locations)}")

    if any(
        isinstance(s, ConditionalStatement)
        for app in gadget_apps
        if app.gadget_name in gadget_map
        for s in gadget_map[app.gadget_name].body
    ):
        print(
            "  WARNING: CONDITIONAL statements are present in inner "
            "gadget(s) but not included in the logical action analysis"
        )

    _compute_and_print_action(all_locations, wrt)


# ── Shared helpers ───────────────────────────────────────────────────


def _print_action(act: CircuitAction) -> None:
    """Print the logical mapping and observables of a CircuitAction."""
    if act.mapping:
        print("  Logical mapping:")
        for pauli_in, pauli_out in act.mapping.items():
            print(f"    {pauli_in}  →  {pauli_out}")
    else:
        print("  Logical mapping: empty")

    if act.observables and act.observables.generators:
        print(f"  Observables: {act.observables}")


def _compute_and_print_action(
    locations: list[Location],
    wrt: tuple[SubsystemCode, SubsystemCode],
) -> None:
    """Call action_of and print results."""
    try:
        act = action_of(locations, with_respect_to=wrt)
    except Exception as exc:
        print(f"  ERROR computing action: {exc}")
        traceback.print_exc()
        return

    _print_action(act)
