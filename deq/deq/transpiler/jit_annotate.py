"""Annotate a ``.deq`` file with derived check structure and noise errors.

This tool helps users understand how their source code is transpiled into
hardware-level information (the binary ``JitLibrary``).

Transformations applied
=======================

- All imported definitions (resolved by :func:`parse_file`) are inlined
  into the output — no ``IMPORT`` statements are emitted.
- ``CODE`` blocks keep their ``[[n,k,d]]`` parameters and decorators,
  but every ``LOGICAL`` and ``STABILIZER`` Pauli product is replaced
  with the ``_`` identity placeholder. The original Pauli string is
  preserved as a trailing comment for reference.
- ``GADGET`` blocks are forced to ``@CHECKS("manual", verify=0)``. The body is
  rewritten so that:
    - ``REPEAT`` blocks are unrolled (matching the ``.deq.jit``
      view of the gadget);
    - circuit and measurement instructions are kept verbatim;
    - physical noise instructions are marked ``@SIMULATE_ONLY``;
    - noisy measurements receive clean ``@DECODE_ONLY`` counterparts;
    - declared and ordinary noise-derived ``ERROR`` rows are emitted beside
        their source statements in canonical index order;
    - user ``ERROR`` statement spelling is retained;
    - user-written ``CHECK`` and ``READOUT`` statements are emitted
      verbatim;
    - auto-derived ``CHECK`` statements that extend the user-provided
      ones are inserted right after the latest measurement they depend
      on;
    - inferred loss-only rows are emitted at their constituent boundary and
        the authoritative ``LOSS`` graph is emitted explicitly.
- ``COMPOSE`` definitions are emitted as comments.
- ``PROGRAM`` definitions are emitted verbatim.
"""

from typing import Iterable, Sequence

from deq.circuit.model import (
    CheckStatement,
    CodeDefinition,
    ComposeDefinition,
    ConditionalStatement,
    Decorator,
    ErrorStatement,
    VirtualLogicalStatement,
    GadgetDefinition,
    GadgetStatement,
    InputPort,
    Instruction,
    KeywordArg,
    LossStatement,
    OutputPort,
    PauliProduct,
    PhysicalMeasurementTarget,
    PreselectStatement,
    ProgramDefinition,
    PropagateStatement,
    DeqFile,
    ReadoutStatement,
    RepeatBlock,
)
from deq.transpiler.jit_transpiler import (
    Check,
    PortColumnLayout,
    flatten_body,
    select_stabilizer_generators,
)
from deq.transpiler.check_plugins import compute_layout, resolve_gadget_checks
from deq.transpiler.code_validation import validate_code
from deq.transpiler.compose_builder import (
    _check_basis_from_jit_gadget_type,
    compose_to_synthetic_gadget,
    expand_compose_circuit,
    has_repropagate,
)
from deq.transpiler.jit_library_builder import (
    JitGadgetArtifacts,
    build_jit_library_artifacts,
)
from deq.transpiler.jit_noise_builder import compute_implicit_readout_propagation
from deq.transpiler.loss.syntax import loss_model_to_statements
from deq.spec.common import bitmatrix_of
import deq.proto.deq_jit_pb2 as jit_pb
import deq.proto.util_pb2 as util_pb
from deq.transpiler.stim_constants import (
    NOISE_INSTRUCTIONS_ALL,
    NOISY_MEASUREMENT_INSTRUCTIONS,
    instruction_num_measurements,
)


def annotate(qfile: DeqFile) -> str:
    """Render ``qfile`` as annotated ``.deq`` source mirroring its JIT form.

    Parameters
    ----------
    qfile:
        The parsed ``.deq`` file to annotate.

    Undecorated noise is split into its original ``@SIMULATE_ONLY`` physical
    instruction and canonical decode-side metadata. Decode-visible noisy
    measurements retain a clean ``@DECODE_ONLY`` structural instruction.
    Existing visibility decorators are preserved by intent, and measurement
    counts must remain equal between the two views.
    """
    codes: dict[str, CodeDefinition] = {
        d.name: d for d in qfile.definitions if isinstance(d, CodeDefinition)
    }
    for code in codes.values():
        validate_code(code)
    gadget_defs: dict[str, GadgetDefinition] = {
        d.name: d for d in qfile.definitions if isinstance(d, GadgetDefinition)
    }
    compose_defs: dict[str, ComposeDefinition] = {
        d.name: d for d in qfile.definitions if isinstance(d, ComposeDefinition)
    }

    # Always build the JIT library to get stable gtype/ptype assignments
    # and to render COMPOSE definitions as GADGET blocks.
    library_artifacts = build_jit_library_artifacts(qfile)
    library = library_artifacts.jit_library
    stab_count_of_ptype: dict[int, int] = {
        pt.base.ptype: len(pt.stabilizers) for pt in library.port_types
    }
    jit_by_name: dict[str, jit_pb.JitGadgetType] = {
        g.base.name: g for g in library.gadget_types if g.base.name
    }
    ptype_by_name: dict[str, int] = {
        pt.base.name: pt.base.ptype for pt in library.port_types
    }

    # COMPOSE definitions visible up to (and including) each compose,
    # used by ``compose_to_synthetic_gadget`` for nested @REPROPAGATE.
    compose_so_far: dict[str, ComposeDefinition] = {}

    blocks: list[str] = []
    for definition in qfile.definitions:
        if isinstance(definition, CodeDefinition):
            blocks.append(
                _annotate_code(definition, ptype_by_name.get(definition.name))
            )
        elif isinstance(definition, GadgetDefinition):
            blocks.append(
                _annotate_gadget(
                    definition,
                    codes,
                    artifacts=library_artifacts.gadget_artifacts_by_name[
                        definition.name
                    ],
                )
            )
        elif isinstance(definition, ComposeDefinition):
            if has_repropagate(definition):
                # @REPROPAGATE: render via the standard GADGET pipeline so
                # propagation matrices and ERROR rows come from circuit
                # flow on the inlined body, not from sub-gadget matrix
                # composition.  The check basis, however, comes from
                # the merge() pipeline (already grafted onto
                # ``jit_by_name[name]`` by ``build_jit_library``); we
                # extract it and pass it through so the emitted CHECK
                # statements and the internally derived propagation /
                # ERROR rows reference the same check indices.
                synthetic = compose_to_synthetic_gadget(
                    definition, gadget_defs, compose_so_far, codes
                )
                check_override = _check_basis_from_jit_gadget_type(
                    jit_by_name[definition.name], synthetic, codes
                )
                blocks.append(
                    _annotate_gadget(
                        synthetic,
                        codes,
                        artifacts=library_artifacts.gadget_artifacts_by_name[
                            definition.name
                        ],
                        check_override=check_override,
                    )
                )
            else:
                blocks.append(
                    _render_composed_gadget(
                        library_artifacts.gadget_artifacts_by_name[definition.name],
                        stab_count_of_ptype,
                        definition,
                        gadget_defs,
                        compose_defs,
                        codes,
                    )
                )
            compose_so_far[definition.name] = definition
        else:
            assert isinstance(
                definition, ProgramDefinition
            ), f"unsupported top-level definition: {type(definition).__name__}"
            blocks.append(_emit_program(definition))
    return "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------------------
# CODE blocks
# ---------------------------------------------------------------------------


def _annotate_code(code: CodeDefinition, ptype: int | None = None) -> str:

    header_decorators = [str(d) for d in code.decorators if d.name != "PTYPE"]
    if ptype is not None:
        header_decorators.insert(0, f"@PTYPE({ptype})")
    if code.d is not None:
        params = f"[[{code.n},{code.k},{code.d}]]"
    else:
        params = f"[[{code.n},{code.k}]]"
    lines: list[str] = [*header_decorators, f"CODE {code.name} {params} {{"]
    for logical in code.logicals:
        x_str = _render_pauli_product(logical.x_operator)
        z_str = _render_pauli_product(logical.z_operator)
        lines.append(f"    LOGICAL {x_str} {z_str}")
    if code.stabilizers:
        # Show each stabilizer on its own line; generators get a trailing
        # comment with their destabilizer Pauli string.
        sel = select_stabilizer_generators(code)
        gen_map: dict[int, int] = {}  # stabilizer index → generator seq index
        for seq, gi in enumerate(sel.generator_indices):
            gen_map[gi] = seq
        for si, stab in enumerate(code.stabilizers):
            stab_str = _render_pauli_product(stab)
            if si in gen_map:
                dp = sel.destabilizer_paulis[gen_map[si]]
                terms = []
                for q in range(len(dp)):
                    v = dp[q]
                    if v == 1:
                        terms.append(f"X{q}")
                    elif v == 2:
                        terms.append(f"Y{q}")
                    elif v == 3:
                        terms.append(f"Z{q}")
                ds_str = "*".join(terms) if terms else "I"
                lines.append(
                    f"    STABILIZER {stab_str}"
                    f"  # generator S{si}, destabilizer DS{si}={ds_str}"
                )
            else:
                lines.append(f"    STABILIZER {stab_str}")
    lines.append("}")
    return "\n".join(lines)


def _render_pauli_product(product: PauliProduct) -> str:
    if not product.terms:
        return "_"
    return str(product)


# ---------------------------------------------------------------------------
# GADGET blocks
# ---------------------------------------------------------------------------


def _annotate_gadget(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
    *,
    artifacts: JitGadgetArtifacts,
    check_override: (
        tuple[
            list[tuple[frozenset[int], bool]],
            list[tuple[frozenset[int], bool]],
        ]
        | None
    ) = None,
) -> str:
    """Render *gadget* as a ``@CHECKS("manual", verify=0)`` GADGET block.

    When *check_override* is provided as ``(finished, unfinished)``, it
    replaces what :func:`resolve_gadget_checks` would derive from the
    gadget body.  Used by ``@REPROPAGATE`` composes so the emitted
    CHECK statements match the merge-derived check basis (the same
    one the build pipeline grafts onto the flat-circuit propagation /
    error derivation).
    """
    flat_body = flatten_body(list(gadget.body))
    jit_gadget = artifacts.jit_type
    jit_errors = list(jit_gadget.errors)
    loss_model = (
        jit_gadget.base.loss_model if jit_gadget.base.HasField("loss_model") else None
    )
    emit_loss_metadata = loss_model is not None
    # Walk the body once to label every body position with the running
    # measurement count *after* that position. We use these snapshots
    # both for placing auto-derived CHECKs.
    running_counts: list[int] = []
    running = 0
    for stmt in flat_body:
        running = _advance_running_count(stmt, running, codes)
        running_counts.append(running)

    # Use the plugin system to derive the final check basis,
    # respecting the gadget's @CHECKS decorator.  When the caller
    # supplies an override (used by the @REPROPAGATE compose path to
    # keep the merge-derived basis), use that instead.
    if check_override is not None:
        finished = check_override[0]
        unfinished = check_override[1]
    else:
        check_result = resolve_gadget_checks(gadget, codes)
        finished = check_result.finished
        unfinished = check_result.unfinished

    # Emit ALL plugin-derived checks (finished + unfinished).
    # User-written CHECKs are dropped from the body and replaced by
    # the full plugin output to ensure the annotated file reproduces
    # the exact same check basis under @CHECKS("manual", verify=0).
    #
    # Finished checks are emitted right before the first OUTPUT (they
    # don't reference output-virtual indices).  Unfinished checks are
    # emitted right after the last OUTPUT (each references exactly one
    # output-virtual index).  This preserves the plugin's ordering so
    # the manual(verify=0) plugin reproduces the same check indices.
    first_output_pos: int | None = None
    last_output_pos: int | None = None
    for idx, stmt in enumerate(flat_body):
        if isinstance(stmt, OutputPort):
            if first_output_pos is None:
                first_output_pos = idx
            last_output_pos = idx

    finished_at_position: int | None = None
    unfinished_at_position: int | None = None
    if first_output_pos is not None:
        # Emit finished checks just before the first OUTPUT.
        finished_at_position = first_output_pos - 1 if first_output_pos > 0 else 0
        # Emit unfinished checks just after the last OUTPUT.
        unfinished_at_position = last_output_pos
    else:
        # No OUTPUT ports: emit all checks at the end.
        finished_at_position = len(flat_body) - 1
        unfinished_at_position = len(flat_body) - 1

    decorators = _gadget_decorators_with_manual_checks(
        gadget.decorators, gtype=jit_gadget.base.gtype
    )
    lines: list[str] = [*[str(d) for d in decorators], f"GADGET {gadget.name} {{"]
    noise_error_indices_by_body: dict[int, list[int]] = {}
    for origin in artifacts.noise_error_origins:
        noise_error_indices_by_body.setdefault(origin.body_index, []).append(
            origin.error_index
        )
    declared_error_index_by_body = {
        origin.body_index: origin.error_index
        for origin in artifacts.declared_error_origins
        if isinstance(flat_body[origin.body_index], ErrorStatement)
    }
    num_finished = len(jit_gadget.finished_checks)
    cp_pb = jit_gadget.base.correction_propagation
    pc_pb = jit_gadget.base.physical_correction
    lc_pb = jit_gadget.base.logical_correction

    # Compute column layouts for output and input ports.
    output_ports = gadget.output_ports
    output_col_layout = PortColumnLayout(output_ports, codes)

    # Compute readout propagation for annotating READOUT lines.
    input_ports = gadget.input_ports
    input_col_layout = PortColumnLayout(input_ports, codes)
    layout = compute_layout(gadget, codes)
    input_virtual_count = layout.input_virtual_count
    propagation = jit_gadget.base.readout_propagation

    input_port_stab_counts = [len(codes[p.code_name].stabilizers) for p in input_ports]
    output_port_stab_counts = [
        len(codes[p.code_name].stabilizers) for p in output_ports
    ]
    propagate_lines = _format_propagate_statements(
        cp_pb,
        pc_pb,
        lc_pb,
        input_layout=input_col_layout,
        output_layout=output_col_layout,
    )

    readout_counter = 0
    pre_running = 0
    source_loss_lines: list[str] = []
    input_loss_lines: list[str] = []
    loss_error_counter = 0
    if emit_loss_metadata:
        source_losses, input_losses = loss_model_to_statements(
            loss_model,
            input_ports=input_ports,
            output_ports=output_ports,
            codes=codes,
            gadget_name=gadget.name,
        )
        source_loss_lines = [
            f"    {statement}  # L{loss_index}"
            for loss_index, statement in enumerate(source_losses)
        ]
        input_loss_lines = [f"    {statement}" for statement in input_losses]
    simulation_only_by_boundary = _simulation_only_instructions_by_decode_boundary(
        gadget.body
    )
    physical_running = 0
    for body_index, stmt in enumerate(flat_body):
        for simulation_instruction in simulation_only_by_boundary.get(body_index, ()):
            lines.extend(_render_decorated_instruction(simulation_instruction))
        if isinstance(stmt, ReadoutStatement):
            comment = _format_propagation_comment(
                propagation,
                readout_counter,
                layout=input_col_layout,
            )
            lines.append(f"    {_render_readout_statement(stmt, comment)}")
            readout_counter += 1
        elif isinstance(stmt, ErrorStatement):
            error_index = declared_error_index_by_body[body_index]
            lines.append(f"    {_render_error_statement(stmt)}  # E{error_index}")
        else:
            for line in _render_body_statement(stmt, physical_running=physical_running):
                lines.append(line)
            # Emit each source loss's LOSS(...) line right after its
            # commented-out LOSS_ERROR so the loss model is legible in place;
            # the trailing ``# L<i>`` on that line labels the loss for
            # ``child_losses`` references elsewhere.
            if (
                isinstance(stmt, Instruction)
                and stmt.name.upper() == "LOSS_ERROR"
                and loss_error_counter < len(source_loss_lines)
            ):
                lines.append(source_loss_lines[loss_error_counter])
                loss_error_counter += 1
        for error_index in noise_error_indices_by_body.get(body_index, ()):
            lines.append(
                "    "
                + _render_jit_error_to_source(
                    jit_errors[error_index],
                    num_finished=num_finished,
                    layout=output_col_layout,
                )
                + f"  # E{error_index}"
            )
        pre_running = running_counts[body_index]
        if isinstance(stmt, Instruction):
            physical_running += instruction_num_measurements(str(stmt))
        if body_index == finished_at_position:
            for check in finished:
                assert check[0], "checks should never be empty, bug in check plugin"
                lines.append(
                    _render_auto_check(
                        check,
                        pre_running,
                        iv_count=input_virtual_count,
                        internal_count=layout.internal_count,
                        input_port_stab_counts=input_port_stab_counts,
                        output_port_stab_counts=output_port_stab_counts,
                    )
                )
        if body_index == unfinished_at_position:
            for check in unfinished:
                assert check[0], "checks should never be empty, bug in check plugin"
                lines.append(
                    _render_auto_check(
                        check,
                        pre_running,
                        iv_count=input_virtual_count,
                        internal_count=layout.internal_count,
                        input_port_stab_counts=input_port_stab_counts,
                        output_port_stab_counts=output_port_stab_counts,
                    )
                )
    for simulation_instruction in simulation_only_by_boundary.get(len(flat_body), ()):
        lines.extend(_render_decorated_instruction(simulation_instruction))

    # Each PROPAGATE row below is the complete XOR formula the runtime
    # evaluates for that output observable.
    lines.extend(propagate_lines)

    if artifacts.appended_error_origins:
        lines.append("")
        for origin in artifacts.appended_error_origins:
            lines.append(
                "    "
                + _render_jit_error_to_source(
                    jit_errors[origin.error_index],
                    num_finished=num_finished,
                    layout=output_col_layout,
                )
                + f"  # E{origin.error_index}"
            )
    if emit_loss_metadata:
        trailing_loss_lines = list(source_loss_lines[loss_error_counter:])
        trailing_loss_lines.extend(input_loss_lines)
        if trailing_loss_lines:
            lines.append("")
            lines.extend(trailing_loss_lines)

    # Statistics summary
    lines.append("")
    lines.extend(
        _format_stats_comment(
            [len(members) for members, _ in finished],
            [len(members) for members, _ in unfinished],
            [len(e.finished_checks) + len(e.unfinished_checks) for e in jit_errors],
        )
    )

    lines.append("}")
    return "\n".join(lines)


def _gadget_decorators_with_manual_checks(
    decorators: Iterable[Decorator],
    *,
    gtype: int | None = None,
) -> list[Decorator]:
    """Return the gadget's decorators with ``@CHECKS`` forced to ``"manual", verify=0``.

    If *gtype* is given, ``@GTYPE(gtype)`` is prepended (replacing any
    existing ``@GTYPE``).
    """
    out: list[Decorator] = []
    if gtype is not None:
        out.append(Decorator(name="GTYPE", arguments=(gtype,)))
    for decorator in decorators:
        if decorator.name in ("CHECKS", "GTYPE"):
            continue
        out.append(decorator)
    out.append(
        Decorator(
            name="CHECKS",
            arguments=("manual", KeywordArg(key="verify", value=0)),
        )
    )
    return out


def _advance_running_count(
    stmt: GadgetStatement,
    running: int,
    codes: dict[str, CodeDefinition],
) -> int:
    """Return the running measurement count *after* ``stmt`` is processed."""
    if isinstance(stmt, InputPort):
        return running + len(codes[stmt.code_name].stabilizers)
    if isinstance(stmt, OutputPort):
        return running + len(codes[stmt.code_name].stabilizers)
    if isinstance(stmt, Instruction):
        return running + instruction_num_measurements(str(stmt))
    return running


def _render_jit_error_to_source(
    error_row: jit_pb.JitGadgetType.Error,
    *,
    num_finished: int,
    layout: PortColumnLayout,
) -> str:
    """Render one ``JitGadgetType.Error`` as a source ``ERROR(p) ...`` line.

    Residual columns are filtered to logical only and rendered with
    correct multi-port observable indices.  Stabilizer generator columns
    are omitted — their effect is fully determined by unfinished check
    references.
    """
    targets: list[str] = []
    for index in error_row.finished_checks:
        targets.append(f"C{index}")
    for index in error_row.unfinished_checks:
        targets.append(f"C{num_finished + index}")
    for index in error_row.base.readout_flips:
        targets.append(f"R{index}")

    residual_indices = set(error_row.base.residual) & layout.logical_columns
    targets.extend(layout.render_logical_labels(residual_indices))

    probability = error_row.base.probability
    suffix = " " + " ".join(targets) if targets else ""
    return f"ERROR({probability}){suffix}"


def _render_body_statement(
    stmt: GadgetStatement,
    *,
    physical_running: int = 0,
) -> list[str]:
    """Render a single body statement as one or more lines (already indented).

    Noise instructions are emitted under ``@SIMULATE_ONLY``. Noisy
    measurements also emit a clean ``@DECODE_ONLY`` twin so simulation and
    decoding retain identical measurement structure. Canonical ERROR rows are
    rendered separately.

    ``physical_running`` is the running count of physical measurements
    produced by preceding statements; it is used to translate
    absolute ``M<i>`` PRESELECT targets into relative ``rec[-k]``.
    """
    if isinstance(stmt, InputPort):
        return [_render_input_or_output(stmt, "INPUT")]
    if isinstance(stmt, OutputPort):
        return [_render_input_or_output(stmt, "OUTPUT")]
    if isinstance(stmt, CheckStatement):
        # User-written CHECKs are redundant — the full plugin-derived
        # check set is emitted separately. Drop silently.
        return []
    if isinstance(stmt, PropagateStatement):
        # User-written PROPAGATEs are redundant — the full PROPAGATE
        # block is regenerated from the cp/pc matrices. Drop silently.
        return []
    if isinstance(stmt, LossStatement):
        # User-written LOSS statements are regenerated from the binary loss
        # model as a dedicated block, so drop them here to avoid duplication.
        return []
    if isinstance(stmt, Instruction):
        return _render_instruction(stmt)
    if isinstance(stmt, ConditionalStatement):
        # CONDITIONAL R<j>/rec[-k]/M<i> statements are absorbed into
        # the PROPAGATE block: readout targets appear as ``R<k>`` terms
        # (via ``logical_correction``), measurement targets appear as
        # ``M<i>`` terms (via ``physical_correction``).
        return []
    if isinstance(stmt, VirtualLogicalStatement):
        # VIRTUAL adds a constant flip to the affine column of
        # ``correction_propagation``; it appears in the PROPAGATE
        # block as the trailing ``FLIP`` keyword.
        return []
    if isinstance(stmt, PreselectStatement):
        return [_render_preselect(stmt, physical_running)]
    raise TypeError(f"unhandled gadget statement: {type(stmt).__name__}")


def _render_instruction(
    stmt: Instruction,
) -> list[str]:
    """Split physical noise from its noiseless decode-side structure."""
    name = stmt.name.upper()
    simulate_only = any(
        decorator.name == "SIMULATE_ONLY" for decorator in stmt.decorators
    )
    decode_only = any(decorator.name == "DECODE_ONLY" for decorator in stmt.decorators)
    is_noise_channel = name in NOISE_INSTRUCTIONS_ALL
    is_noisy_measurement = (
        stmt.arguments
        and stmt.arguments[0] != 0
        and name in NOISY_MEASUREMENT_INSTRUCTIONS
    )
    if not is_noise_channel and not is_noisy_measurement:
        if stmt.decorators:
            return _render_decorated_instruction(stmt)
        return [f"    {stmt}"]

    simulation_visible = not decode_only
    decode_visible = not simulate_only
    lines: list[str] = []
    if simulation_visible:
        lines.extend(("    @SIMULATE_ONLY", f"    {stmt}"))
    if decode_visible and is_noisy_measurement:
        clean = Instruction(
            name=stmt.name,
            tag=stmt.tag,
            arguments=[],
            targets=list(stmt.targets),
        )
        lines.extend(("    @DECODE_ONLY", f"    {clean}"))
    elif decode_only and is_noise_channel:
        lines.extend(
            (
                *(f"    # {decorator}" for decorator in stmt.decorators),
                f"    # {stmt}",
            )
        )
    return lines


def _render_decorated_instruction(stmt: Instruction) -> list[str]:
    return [
        *(f"    {decorator}" for decorator in stmt.decorators),
        f"    {stmt}",
    ]


def _simulation_only_instructions_by_decode_boundary(
    statements: list[GadgetStatement],
) -> dict[int, list[Instruction]]:
    """Group decode-hidden instructions by the next decode-body position."""
    grouped: dict[int, list[Instruction]] = {}
    decode_boundary = 0

    def visit(items: list[GadgetStatement]) -> None:
        nonlocal decode_boundary
        for statement in items:
            if isinstance(statement, RepeatBlock):
                for _ in range(statement.count):
                    visit(statement.body)
            elif isinstance(statement, Instruction) and any(
                decorator.name == "SIMULATE_ONLY" for decorator in statement.decorators
            ):
                grouped.setdefault(decode_boundary, []).append(statement)
            else:
                decode_boundary += 1

    visit(statements)
    return grouped


def _render_preselect(
    stmt: PreselectStatement,
    physical_running: int,
) -> str:
    """Render a ``PRESELECT`` line, translating absolute ``M<i>`` targets
    into relative ``rec[-k]`` where ``physical_running`` is the running
    count of physical measurements produced *before or including* the
    current position.

    Preferring the relative form keeps the annotated output stable
    against re-numbering of a synthetic gadget's measurement stream
    (as happens when a COMPOSE inlines sub-gadgets whose PRESELECTs
    used sub-gadget-local ``M<i>`` targets).  Existing ``rec[-k]``
    targets pass through unchanged.
    """
    tokens: list[str] = []
    for cond in stmt.conditions:
        if isinstance(cond, PhysicalMeasurementTarget):
            offset = physical_running - cond.index
            tokens.append(f"rec[-{offset}]")
        else:
            tokens.append(str(cond))
    return f"    PRESELECT {' '.join(tokens)} {stmt.expected_value}"


def _weight_distribution(weights: Sequence[int]) -> str:
    """Format a weight distribution as ``{weight:count, ...}``."""
    dist: dict[int, int] = {}
    for w in weights:
        dist[w] = dist.get(w, 0) + 1
    items = sorted(dist.items())
    return "{ " + ", ".join(f"{w}:{c}" for w, c in items) + " }"


def _format_stats_comment(
    finished_weights: Sequence[int],
    unfinished_weights: Sequence[int],
    error_check_weights: Sequence[int],
) -> list[str]:
    """Return statistics lines as ``# ...`` comments (indented)."""
    lines = ["    # --- statistics ---"]
    lines.append(f"    # finished checks: {len(finished_weights)}")
    if finished_weights:
        lines.append(
            f"    #   weight distribution: {_weight_distribution(finished_weights)}"
        )
    lines.append(f"    # unfinished checks: {len(unfinished_weights)}")
    if unfinished_weights:
        lines.append(
            f"    #   weight distribution: {_weight_distribution(unfinished_weights)}"
        )
    lines.append(f"    # errors: {len(error_check_weights)}")
    if error_check_weights:
        lines.append(
            f"    #   check-weight distribution: {_weight_distribution(error_check_weights)}"
        )
    return lines


def _format_propagate_statements(
    cp_pb: util_pb.BitMatrix,
    pc_pb: util_pb.BitMatrix,
    lc_pb: util_pb.BitMatrix | None = None,
    *,
    input_layout: PortColumnLayout,
    output_layout: PortColumnLayout,
) -> list[str]:
    """Render output-logical-row cp/pc/lc data as ``PROPAGATE`` source lines.

    For every output logical row, emit a line of the form

    .. code-block:: text

        PROPAGATE LZ0 FROM LZ0 IN0.DS2 M3 R0 FLIP

    The right-hand side is the XOR of:

    * input-frame logical columns (rendered with the same
      X-column-as-``LZ`` convention used by ``ERROR`` rows);
    * input-frame destabilizer columns, labelled ``IN<p>.DS<s>``
      for the destabilizer of stabilizer ``s`` of INPUT port ``p``
      (port-explicit form);
    * internal physical measurement outcomes, labelled ``M<i>``
      (the i-th internal/physical measurement of the gadget,
      gadget-scoped, 0-based);
    * decoded readouts, labelled ``R<k>`` — the readout-conditioned
      frame correction the runtime XORs on top of the natural-Heisenberg
      residual (rendered when ``lc_pb`` has entries for this row);
    * the affine ``FLIP`` constant absorbed by the last column of
      ``correction_propagation`` (appended as the trailing keyword).

    Rows with no terms and no ``FLIP`` are emitted as ``PROPAGATE LZ0
    FROM`` (the grammar accepts an empty term list).  Rows with only
    a ``FLIP`` are emitted as ``PROPAGATE LZ0 FROM FLIP``.
    """
    if not output_layout.logical_columns:
        return []

    cp_mat = bitmatrix_of(cp_pb)
    pc_mat = bitmatrix_of(pc_pb)
    lc_mat = bitmatrix_of(lc_pb) if lc_pb is not None and lc_pb.cols > 0 else None
    affine_col = cp_pb.cols - 1

    lines: list[str] = []
    for out_row in sorted(output_layout.logical_columns):
        out_label = output_layout.render_logical_labels({out_row})[0]
        cp_cols = set(cp_mat.rows[out_row].support)
        pc_cols = set(pc_mat.rows[out_row].support)
        lc_cols = set(lc_mat.rows[out_row].support) if lc_mat is not None else set()
        has_flip = affine_col in cp_cols

        in_obs_cols = cp_cols & input_layout.logical_columns
        in_stab_cols = sorted(
            c
            for c in cp_cols
            if c != affine_col and c not in input_layout.logical_columns
        )

        terms: list[str] = []
        terms.extend(
            input_layout.render_logical_labels(in_obs_cols, combine_xz_to_y=False)
        )
        for c in in_stab_cols:
            port_idx, stab_idx_in_port = input_layout.generator_map[c]
            terms.append(f"IN{port_idx}.DS{stab_idx_in_port}")
        for j in sorted(pc_cols):
            terms.append(f"M{j}")
        for k in sorted(lc_cols):
            terms.append(f"R{k}")

        suffix = " FLIP" if has_flip else ""
        body = " " + " ".join(terms) if terms else ""
        lines.append(f"    PROPAGATE {out_label} FROM{body}{suffix}")

    return lines


def _render_input_or_output(port: InputPort | OutputPort, keyword: str) -> str:
    indices = " ".join(str(i) for i in port.qubit_indices)
    if indices:
        return f"    {keyword} {port.code_name} {indices}"
    return f"    {keyword} {port.code_name}"


def _render_readout_statement(
    stmt: ReadoutStatement,
    propagation_comment: str = "",
) -> str:
    targets = " ".join(str(t) for t in stmt.targets)
    suffix = " FLIP" if stmt.flip else ""
    comment = f"  {propagation_comment}" if propagation_comment else ""
    return f"READOUT {targets}{suffix}{comment}"


def _format_propagation_comment(
    propagation: util_pb.BitMatrix,
    row_index: int,
    layout: PortColumnLayout,
) -> str:
    """Format a ``# LX0 ...`` comment for one readout row.

    Shows which input-frame bits flip this readout in addition to what
    is already on the READOUT line — semantically, the readout value is
    the XOR of everything on the line and everything in this comment.

    *layout* provides the column-to-observable mapping and stabilizer
    generator indices for correct multi-port rendering.
    """
    affine_col = propagation.cols - 1
    row_cols = set(bitmatrix_of(propagation).rows[row_index].support)
    has_affine = affine_col in row_cols
    cols_set = row_cols - {affine_col}

    parts: list[str] = []

    # Logical columns
    log_cols = cols_set & layout.logical_columns
    parts.extend(layout.render_logical_labels(log_cols, combine_xz_to_y=False))

    # Stabilizer generator columns (rendered with port-explicit syntax
    # so the destabilizer reference matches what PROPAGATE accepts).
    stab_cols = sorted(c for c in cols_set if c not in layout.logical_columns)
    for c in stab_cols:
        port_idx, stab_idx = layout.generator_map[c]
        parts.append(f"IN{port_idx}.DS{stab_idx}")

    if has_affine:
        parts.append("FLIP")
    if not parts:
        return ""
    return "# " + " ".join(parts)


def _render_error_statement(stmt: ErrorStatement) -> str:
    targets = " ".join(str(t) for t in stmt.targets)
    return f"ERROR({stmt.probability}) {targets}".rstrip()


def _format_measurement_ref(
    global_index: int,
    *,
    iv_count: int,
    internal_count: int,
    input_port_stab_counts: list[int],
    output_port_stab_counts: list[int],
) -> str:
    """Render a global measurement index as ``M<i>``, ``IN<p>.S<s>`` or ``OUT<p>.S<s>``.

    Gadget measurement regions, in order:
      ``[input-virtual | internal/physical | output-virtual]``
    """
    if global_index < 0:
        raise ValueError(f"negative global measurement index {global_index}")
    if global_index < iv_count:
        offset = global_index
        for port_idx, count in enumerate(input_port_stab_counts):
            if offset < count:
                return f"IN{port_idx}.S{offset}"
            offset -= count
        raise ValueError(
            f"global index {global_index} falls in input-virtual region "
            f"(iv_count={iv_count}) but does not map to any input port "
            f"(stab counts={input_port_stab_counts})"
        )
    if global_index < iv_count + internal_count:
        return f"M{global_index - iv_count}"
    offset = global_index - iv_count - internal_count
    for port_idx, count in enumerate(output_port_stab_counts):
        if offset < count:
            return f"OUT{port_idx}.S{offset}"
        offset -= count
    raise ValueError(
        f"global index {global_index} is out of range "
        f"(iv_count={iv_count}, internal_count={internal_count}, "
        f"output total={sum(output_port_stab_counts)})"
    )


def _render_auto_check(
    check: Check,
    running: int,
    *,
    iv_count: int,
    internal_count: int,
    input_port_stab_counts: list[int],
    output_port_stab_counts: list[int],
) -> str:
    members, parity = check
    # Sort by ascending rec offset (most-recent first); matches the
    # original ``rec[-k]`` rendering order.
    sorted_global = sorted(members, key=lambda idx: running - idx)
    tokens = " ".join(
        _format_measurement_ref(
            idx,
            iv_count=iv_count,
            internal_count=internal_count,
            input_port_stab_counts=input_port_stab_counts,
            output_port_stab_counts=output_port_stab_counts,
        )
        for idx in sorted_global
    )
    suffix = " FLIP" if parity else ""
    return f"    CHECK {tokens}{suffix}"


# ---------------------------------------------------------------------------
# COMPOSE → GADGET rendering
# ---------------------------------------------------------------------------


def _render_composed_gadget(
    artifacts: JitGadgetArtifacts,
    stab_count_of_ptype: dict[int, int],
    compose: ComposeDefinition,
    gadget_defs: dict[str, GadgetDefinition],
    compose_defs: dict[str, ComposeDefinition],
    codes: dict[str, CodeDefinition],
) -> str:
    """Render a composed ``JitGadgetType`` as a ``GADGET`` block.

    Instead of opaque placeholders, the actual circuit of each sub-gadget is
    inlined with physical noise separated into the simulation view. Port qubits
    are densely numbered starting at 0; ancilla qubits follow.

    Physical noise is retained under ``@SIMULATE_ONLY``. Declared and ordinary
    noise-derived rows are emitted at mapped source positions; inferred
    loss-only rows are emitted at constituent boundaries. Noisy measurements
    receive clean ``@DECODE_ONLY`` twins, and the merged loss DAG is emitted
    explicitly.
    """
    gadget = artifacts.jit_type
    base = gadget.base
    name = base.name or f"AnonymousGadget{base.gtype}"
    loss_model = base.loss_model if base.HasField("loss_model") else None
    emit_loss_metadata = loss_model is not None
    input_stab_counts = [stab_count_of_ptype[p.ptype] for p in base.inputs]
    output_stab_counts = [stab_count_of_ptype[p.ptype] for p in base.outputs]
    iv_count = sum(input_stab_counts)
    internal_count = len(base.measurements)

    lines: list[str] = [
        f"@GTYPE({base.gtype})",
        '@CHECKS("manual", verify=0)',
        f"GADGET {name} {{",
    ]

    # Expand compose into (input_ports, circuit_stmts, output_ports)
    # with dense qubit remapping.
    known = set(gadget_defs) | set(compose_defs)
    input_ports, circuit_stmts, output_ports = expand_compose_circuit(
        compose, gadget_defs, compose_defs, known, codes
    )
    output_col_layout = PortColumnLayout(output_ports, codes)
    noise_errors_by_position: dict[int, list[int]] = {}
    for origin in artifacts.noise_error_origins:
        noise_errors_by_position.setdefault(origin.body_index, []).append(
            origin.error_index
        )
    boundary_errors_by_position: dict[int, list[int]] = {}
    for origin in (
        *artifacts.declared_error_origins,
        *artifacts.appended_error_origins,
    ):
        boundary_errors_by_position.setdefault(origin.body_index, []).append(
            origin.error_index
        )
    for error_indices in noise_errors_by_position.values():
        error_indices.sort()
    for error_indices in boundary_errors_by_position.values():
        error_indices.sort()
    localized_error_indices = {
        origin.error_index
        for origin in (
            *artifacts.noise_error_origins,
            *artifacts.declared_error_origins,
            *artifacts.appended_error_origins,
        )
    }
    expected_error_indices = set(range(len(gadget.errors)))
    if localized_error_indices != expected_error_indices:
        raise AssertionError(
            f"GADGET {name!r} error provenance is incomplete: got "
            f"{sorted(localized_error_indices)}, expected "
            f"{sorted(expected_error_indices)}"
        )
    source_loss_lines: list[str] = []
    input_loss_lines: list[str] = []
    if emit_loss_metadata:
        source_losses, input_losses = loss_model_to_statements(
            loss_model,
            input_ports=input_ports,
            output_ports=output_ports,
            codes=codes,
            gadget_name=name,
        )
        source_loss_lines = [
            f"    {statement}  # L{loss_index}"
            for loss_index, statement in enumerate(source_losses)
        ]
        input_loss_lines = [f"    {statement}" for statement in input_losses]

    # INPUT lines from sub-gadgets' port declarations.
    for port in input_ports:
        lines.append(_render_input_or_output(port, "INPUT"))

    # Circuit body: inline sub-gadget instructions, noise commented out.
    # Track the running physical-measurement count so that any
    # PRESELECT statements inherited from sub-gadgets can have their
    # absolute ``M<i>`` targets translated into relative ``rec[-k]``.
    physical_running = 0
    num_finished = len(gadget.finished_checks)

    def emit_error(error_index: int) -> None:
        lines.append(
            "    "
            + _render_jit_error_to_source(
                gadget.errors[error_index],
                num_finished=num_finished,
                layout=output_col_layout,
            )
            + f"  # E{error_index}"
        )

    decode_position = 0

    for stmt in circuit_stmts:
        simulate_only = isinstance(stmt, Instruction) and any(
            decorator.name == "SIMULATE_ONLY" for decorator in stmt.decorators
        )
        if not simulate_only:
            for error_index in boundary_errors_by_position.get(decode_position, ()):
                emit_error(error_index)
        if isinstance(stmt, Instruction):
            lines.extend(_render_instruction(stmt))
            if not simulate_only:
                physical_running += instruction_num_measurements(str(stmt))
        elif isinstance(stmt, PreselectStatement):
            lines.append(_render_preselect(stmt, physical_running))
        if not simulate_only:
            for error_index in noise_errors_by_position.get(decode_position, ()):
                emit_error(error_index)
            decode_position += 1
    for error_index in boundary_errors_by_position.get(decode_position, ()):
        emit_error(error_index)

    # OUTPUT lines from sub-gadgets' port declarations.
    for port in output_ports:
        lines.append(_render_input_or_output(port, "OUTPUT"))

    # CHECK statements.
    for check in gadget.finished_checks:
        lines.append(
            "    "
            + _format_composed_check(
                check,
                None,
                input_stab_counts,
                output_stab_counts,
                iv_count,
                internal_count,
            )
        )
    for k, check in enumerate(gadget.unfinished_checks):
        ov_global = iv_count + internal_count + k
        lines.append(
            "    "
            + _format_composed_check(
                check,
                ov_global,
                input_stab_counts,
                output_stab_counts,
                iv_count,
                internal_count,
            )
        )

    # READOUT statements.
    #
    # Each readout's ``base.readout_propagation`` row encodes which
    # input-observable columns flip it (matrix-composed semantics from
    # the binary).  When the re-parsed annotated body's Heisenberg
    # walker (:func:`compute_implicit_readout_propagation`) gives the
    # same rp row, no extra tokens are needed: walker output alone
    # suffices.  When the walker differs (e.g. chained-teleportation
    # cumulative readouts whose input-observable parity cancels across
    # hops in the inlined body), we emit the *diff* as explicit
    # ``IN<p>.L<P><i>`` tokens.  ``_build_readout_propagation`` XORs
    # walker-implicit columns with explicit-logical columns, so
    # walker_cols XOR diff = binary_cols on re-parse.
    prop = base.readout_propagation
    input_col_layout = PortColumnLayout(input_ports, codes)
    affine_col = prop.cols - 1
    binary_rp_cols_by_row: dict[int, set[int]] = {}
    for r, c in zip(prop.i, prop.j):
        binary_rp_cols_by_row.setdefault(r, set()).add(c)

    decode_circuit_stmts = flatten_body(circuit_stmts)
    synth_for_walker = GadgetDefinition(
        name=base.name,
        body=[*input_ports, *decode_circuit_stmts, *output_ports],
        decorators=[],
    )
    walker_implicit = compute_implicit_readout_propagation(
        synth_for_walker,
        codes,
        input_ports=input_ports,
        readout_measurement_sets=[set(r.measurement_indices) for r in base.readouts],
    )

    for row_index, readout in enumerate(base.readouts):
        rec_refs = [f"M{mi}" for mi in readout.measurement_indices]
        binary_cols = binary_rp_cols_by_row.get(row_index, set())
        binary_observable_cols = binary_cols - {affine_col}
        walker_cols = (
            walker_implicit[row_index] if row_index < len(walker_implicit) else set()
        )
        diff_cols = binary_observable_cols ^ walker_cols
        if diff_cols:
            diff_logical = diff_cols & input_col_layout.logical_columns
            diff_destab = sorted(
                c for c in diff_cols if c in input_col_layout.generator_map
            )
            rec_refs.extend(
                input_col_layout.render_logical_labels(
                    diff_logical, combine_xz_to_y=False
                )
            )
            for c in diff_destab:
                port_idx, stab_idx = input_col_layout.generator_map[c]
                rec_refs.append(f"IN{port_idx}.DS{stab_idx}")
        # The affine flip has no source in the flattened body (it originates
        # from a dropped VIRTUAL or a CONDITIONAL correction), so emit it as
        # an explicit ``FLIP`` token to keep the readout row round-tripping.
        if affine_col in binary_cols:
            rec_refs.append("FLIP")
        assert rec_refs, f"GADGET {name!r} readout R{row_index} has no source"
        comment = _format_propagation_comment(
            prop,
            row_index,
            layout=input_col_layout,
        )
        suffix = f"  {comment}" if comment else ""
        lines.append("    READOUT " + " ".join(rec_refs) + suffix)

    # PROPAGATE emission.  Emit binary cp/pc/lc verbatim: each row is
    # authoritative and describes the complete XOR formula the runtime
    # evaluates for that output observable.  ``VIRTUAL`` and
    # ``CONDITIONAL`` are intentionally dropped from the annotated body
    # since their contributions already live in cp/pc/lc (VIRTUAL adds a
    # ``FLIP`` bit in ``cp``'s affine column; CONDITIONAL populates ``lc``
    # entries) and the PROPAGATE rows below re-emit them as ``FLIP``
    # keywords and ``R<k>`` terms.
    lines.extend(
        _format_propagate_statements(
            base.correction_propagation,
            base.physical_correction,
            base.logical_correction,
            input_layout=input_col_layout,
            output_layout=output_col_layout,
        )
    )

    if emit_loss_metadata:
        trailing_loss_lines = list(source_loss_lines)
        trailing_loss_lines.extend(input_loss_lines)
        assert trailing_loss_lines, f"GADGET {name!r} has an empty loss model"
        lines.append("")
        lines.extend(trailing_loss_lines)

    # Statistics summary
    lines.append("")
    lines.extend(
        _format_stats_comment(
            [len(c.measurements) for c in gadget.finished_checks],
            [len(c.measurements) for c in gadget.unfinished_checks],
            [len(e.finished_checks) + len(e.unfinished_checks) for e in gadget.errors],
        )
    )

    lines.append("}")
    return "\n".join(lines)


def _format_composed_check(
    check: jit_pb.JitGadgetType.Check,
    ov_index: int | None,
    input_stab_counts: list[int],
    output_stab_counts: list[int],
    iv_count: int,
    internal_count: int,
) -> str:
    """Format a single check from a composed JitGadgetType."""
    global_indices: list[int] = []
    for m in check.measurements:
        if m.HasField("input_port"):
            port_offset = sum(input_stab_counts[: m.input_port])
            global_indices.append(port_offset + m.measurement_index)
        else:
            global_indices.append(iv_count + m.measurement_index)
    if ov_index is not None:
        global_indices.append(ov_index)

    tokens = " ".join(
        _format_measurement_ref(
            idx,
            iv_count=iv_count,
            internal_count=internal_count,
            input_port_stab_counts=input_stab_counts,
            output_port_stab_counts=output_stab_counts,
        )
        for idx in global_indices
    )
    suffix = " FLIP" if check.base.naturally_flipped else ""
    return f"CHECK {tokens}{suffix}"


# ---------------------------------------------------------------------------
# PROGRAM blocks (emitted verbatim — part of .deq.jit)
# ---------------------------------------------------------------------------


def _emit_program(definition: ProgramDefinition) -> str:
    lines = [*[str(d) for d in definition.decorators], f"PROGRAM {definition.name} {{"]
    for stmt in definition.body:
        for sub_line in str(stmt).splitlines() or [""]:
            lines.append(f"    {sub_line}")
    lines.append("}")
    return "\n".join(lines)
