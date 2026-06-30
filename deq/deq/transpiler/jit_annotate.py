"""Annotate a ``.deq`` file with derived check structure and noise errors.

This tool helps users understand how their source code is compiled into
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
    - noise instructions and user ``ERROR`` statements are commented out;
    - user-written ``CHECK`` and ``READOUT`` statements are emitted
      verbatim;
    - auto-derived ``CHECK`` statements that extend the user-provided
      ones are inserted right after the latest measurement they depend
      on;
    - computed ``ERROR`` statements (from noise propagation) are
      inserted after each noise instruction.
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
    OutputPort,
    PauliProduct,
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
    build_jit_library,
    build_readouts,
    collect_physical_conditionals,
    conditional_flipped_rows,
)
from deq.transpiler.jit_noise_builder import (
    compute_correction_propagation,
    compute_physical_correction,
    iter_noise_errors_with_origin,
)
from deq.spec.common import bitmatrix_of
import deq.proto.deq_jit_pb2 as jit_pb
import deq.proto.util_pb2 as util_pb
from deq.transpiler.stim_constants import (
    MEASUREMENT_INSTRUCTIONS,
    NOISE_INSTRUCTIONS_ALL,
    NOISY_MEASUREMENT_INSTRUCTIONS,
    PASSTHROUGH_NOISE_INSTRUCTIONS,
    TWO_QUBIT_MEASUREMENT_INSTRUCTIONS,
    mpp_measurement_count,
)
from deq.transpiler.stim_constants import qubit_indices as _qubit_indices


def annotate(qfile: DeqFile, *, keep_noise: bool = False) -> str:
    """Render ``qfile`` as annotated ``.deq`` source mirroring its JIT form.

    Parameters
    ----------
    qfile:
        The parsed ``.deq`` file to annotate.
    keep_noise:
        When ``True``, noise instructions (and noise on noisy
        measurements) are emitted verbatim into the annotated output
        and the ``ERROR(p) ...`` rows derived from those noise
        instructions are *not* emitted.  Re-transpilation of the
        annotated file re-derives the same ERRORs from the kept noise
        instructions.  When ``False`` (default), noise instructions
        are commented out and the corresponding ERROR rows are emitted
        explicitly.
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
    library = build_jit_library(qfile)
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
            gtype = (
                jit_by_name[definition.name].base.gtype
                if definition.name in jit_by_name
                else None
            )
            blocks.append(
                _annotate_gadget(
                    definition, codes, gtype=gtype, keep_noise=keep_noise
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
                gtype = (
                    jit_by_name[definition.name].base.gtype
                    if definition.name in jit_by_name
                    else None
                )
                check_override = _check_basis_from_jit_gadget_type(
                    jit_by_name[definition.name], synthetic, codes
                )
                blocks.append(
                    _annotate_gadget(
                        synthetic,
                        codes,
                        gtype=gtype,
                        keep_noise=keep_noise,
                        check_override=check_override,
                    )
                )
            else:
                blocks.append(
                    _render_composed_gadget(
                        jit_by_name[definition.name],
                        stab_count_of_ptype,
                        definition,
                        gadget_defs,
                        compose_defs,
                        codes,
                        keep_noise=keep_noise,
                    )
                )
            compose_so_far[definition.name] = definition
        elif isinstance(definition, ProgramDefinition):
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
    gtype: int | None = None,
    keep_noise: bool = False,
    check_override: tuple[
        list[tuple[frozenset[int], bool]],
        list[tuple[frozenset[int], bool]],
    ]
    | None = None,
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

    decorators = _gadget_decorators_with_manual_checks(gadget.decorators, gtype=gtype)
    lines: list[str] = [*[str(d) for d in decorators], f"GADGET {gadget.name} {{"]

    (
        noise_errors_at,
        num_finished,
        cp_pb,
        pc_pb,
        input_virtual_count,
    ) = _compute_gadget_runtime_data(
        gadget, codes, check_override=check_override
    )

    # Compute column layouts for output and input ports.
    output_ports = gadget.output_ports
    output_col_layout = PortColumnLayout(output_ports, codes)

    # Compute readout propagation for annotating READOUT lines.
    input_ports = gadget.input_ports
    input_col_layout = PortColumnLayout(input_ports, codes)
    layout = compute_layout(gadget, codes)
    _readouts_pb, propagation, _readouts_info = build_readouts(
        gadget,
        codes,
        layout.input_virtual_count,
        input_ports,
        output_ports,
        layout.internal_count,
    )

    input_port_stab_counts = [len(codes[p.code_name].stabilizers) for p in input_ports]
    output_port_stab_counts = [
        len(codes[p.code_name].stabilizers) for p in output_ports
    ]
    propagate_lines = _format_propagate_statements(
        cp_pb,
        pc_pb,
        input_layout=input_col_layout,
        output_layout=output_col_layout,
    )

    readout_counter = 0
    pre_running = 0
    for body_index, stmt in enumerate(flat_body):
        if isinstance(stmt, ReadoutStatement):
            comment = _format_propagation_comment(
                propagation,
                readout_counter,
                layout=input_col_layout,
            )
            lines.append(f"    {_render_readout_statement(stmt, comment)}")
            readout_counter += 1
        else:
            for line in _render_body_statement(stmt, keep_noise=keep_noise):
                lines.append(line)
        # When ``keep_noise`` is set, the noise instructions are emitted
        # verbatim, so re-transpilation will re-derive the same ERROR
        # rows from circuit flow. Emitting them here as well would
        # duplicate them.
        if not keep_noise:
            for error_row in noise_errors_at.get(body_index, []):
                lines.append(
                    "    "
                    + _render_jit_error_to_source(
                        error_row,
                        num_finished=num_finished,
                        layout=output_col_layout,
                    )
                )
        pre_running = running_counts[body_index]
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
            lines.extend(propagate_lines)

    # Statistics summary
    all_errors = [e for errs in noise_errors_at.values() for e in errs]
    lines.append("")
    lines.extend(
        _format_stats_comment(
            [len(members) for members, _ in finished],
            [len(members) for members, _ in unfinished],
            [len(e.finished_checks) + len(e.unfinished_checks) for e in all_errors],
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
        name = stmt.name.upper()
        if name in MEASUREMENT_INSTRUCTIONS:
            return running + len(_qubit_indices(stmt))
        if name in TWO_QUBIT_MEASUREMENT_INSTRUCTIONS:
            return running + len(_qubit_indices(stmt)) // 2
        if name == "MPP":
            return running + mpp_measurement_count(list(stmt.targets))
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
    keep_noise: bool = False,
) -> list[str]:
    """Render a single body statement as one or more lines (already indented).

    When ``keep_noise`` is ``True``, noise instructions are emitted
    verbatim and noisy measurements keep their probability arguments,
    so re-transpilation re-derives the original ERROR rows from
    circuit flow.
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
    if isinstance(stmt, ReadoutStatement):
        return [f"    {_render_readout_statement(stmt)}"]
    if isinstance(stmt, ErrorStatement):
        return [f"    # {_render_error_statement(stmt)}"]
    if isinstance(stmt, Instruction):
        name = stmt.name.upper()
        if name in NOISE_INSTRUCTIONS_ALL:
            # Passthrough noise (e.g. LOSS_ERROR) has no equivalent
            # ERROR-row representation that re-transpilation could
            # reconstruct, so it must be emitted verbatim even when the
            # caller asked to comment out regular noise.
            if keep_noise or name in PASSTHROUGH_NOISE_INSTRUCTIONS:
                return [f"    {stmt}"]
            return [f"    # {stmt}"]
        # Noisy measurement: comment out original, emit clean version.
        if (
            stmt.arguments
            and stmt.arguments[0] != 0
            and name in NOISY_MEASUREMENT_INSTRUCTIONS
        ):
            if keep_noise:
                return [f"    {stmt}"]
            clean = Instruction(
                name=stmt.name, tag=stmt.tag, arguments=[], targets=list(stmt.targets)
            )
            return [f"    # {stmt}", f"    {clean}"]
        # Circuit and measurement instructions: keep verbatim.
        return [f"    {stmt}"]
    if isinstance(stmt, RepeatBlock):
        # flatten_body should have already unrolled these; defensive.
        return [f"    # REPEAT {stmt.count} {{ ... }} (unexpected — not unrolled)"]
    if isinstance(stmt, ConditionalStatement):
        targets = " ".join(str(t) for t in stmt.targets)
        return [f"    CONDITIONAL {stmt.condition} {targets}"]
    if isinstance(stmt, VirtualLogicalStatement):
        targets = " ".join(str(t) for t in stmt.targets)
        return [f"    VIRTUAL {targets}"]
    if isinstance(stmt, PreselectStatement):
        return [f"    PRESELECT {stmt.condition} {stmt.expected_value}"]
    raise TypeError(f"unhandled gadget statement: {type(stmt).__name__}")


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
    *,
    input_layout: PortColumnLayout,
    output_layout: PortColumnLayout,
) -> list[str]:
    """Render output-logical-row pc/cp data as ``PROPAGATE`` source lines.

    For every output logical row, emit a line of the form

    .. code-block:: text

        PROPAGATE LZ0 FROM LZ0 IN0.DS2 M3 FLIP

    The right-hand side is the XOR of:

    * input-frame logical columns (rendered with the same
      X-column-as-``LZ`` convention used by ``ERROR`` rows);
    * input-frame destabilizer columns, labelled ``IN<p>.DS<s>``
      for the destabilizer of stabilizer ``s`` of INPUT port ``p``
      (port-explicit form);
    * internal physical measurement outcomes, labelled ``M<i>``
      (the i-th internal/physical measurement of the gadget,
      gadget-scoped, 0-based);
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
    affine_col = cp_pb.cols - 1

    lines: list[str] = []
    for out_row in sorted(output_layout.logical_columns):
        out_label = output_layout.render_logical_labels({out_row})[0]
        cp_cols = set(cp_mat.rows[out_row].support)
        pc_cols = set(pc_mat.rows[out_row].support)
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

        suffix = " FLIP" if has_flip else ""
        body = " " + " ".join(terms) if terms else ""
        lines.append(f"    PROPAGATE {out_label} FROM{body}{suffix}")

    return lines


def _render_input_or_output(port: InputPort | OutputPort, keyword: str) -> str:
    indices = " ".join(str(i) for i in port.qubit_indices)
    if indices:
        return f"    {keyword} {port.code_name} {indices}"
    return f"    {keyword} {port.code_name}"


def _render_check_statement(stmt: CheckStatement) -> str:
    targets = " ".join(str(t) for t in stmt.targets)
    suffix = " FLIP" if stmt.flip else ""
    return f"CHECK {targets}{suffix}"


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
    """Format a ``# flipped by: LX0 ...`` comment for one readout row.

    Shows which input correction flips this readout.

    *layout* provides the column-to-observable mapping and stabilizer
    generator indices for correct multi-port rendering.
    """
    affine_col = propagation.cols - 1 if propagation.cols > 0 else -1
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
    return "# flipped by: " + " ".join(parts)


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
# Noise-instruction propagation (forward Pauli walk)
# ---------------------------------------------------------------------------


def _compute_gadget_runtime_data(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
    *,
    check_override: tuple[
        list[tuple[frozenset[int], bool]],
        list[tuple[frozenset[int], bool]],
    ]
    | None = None,
) -> tuple[
    dict[int, list["jit_pb.JitGadgetType.Error"]],
    int,
    util_pb.BitMatrix,
    util_pb.BitMatrix,
    int,
]:
    """Return runtime data needed to annotate a gadget's noise and flows.

    Returns ``(noise_errors_by_position, num_finished_checks, cp_pb,
    pc_pb, input_virtual_count)``:

    * ``noise_errors_by_position`` — propagated noise mechanisms keyed
      by the originating noise instruction's index in the flattened
      gadget body.  Empty if the gadget has no noise.
    * ``num_finished_checks`` — needed by
      :func:`_render_jit_error_to_source` to renumber unfinished
      checks and by the output-logical-deps comment.
    * ``cp_pb`` — ``correction_propagation`` matrix
      (output rows × input cols + FLIP).
    * ``pc_pb`` — ``physical_correction`` matrix
      (output rows × internal-measurement cols).
    * ``input_virtual_count`` — number of input virtual measurements,
      used to convert pc column indices to global measurement indices.
    """
    input_ports = gadget.input_ports
    output_ports = gadget.output_ports
    layout = compute_layout(gadget, codes)

    if check_override is not None:
        finished = check_override[0]
        unfinished = check_override[1]
    else:
        check_result = resolve_gadget_checks(gadget, codes)
        finished = check_result.finished
        unfinished = check_result.unfinished

    _readouts_pb, _propagation, readouts_info = build_readouts(
        gadget,
        codes,
        layout.input_virtual_count,
        input_ports,
        output_ports,
        layout.internal_count,
    )

    cp_pb, logical_physical_entries = compute_correction_propagation(
        gadget,
        codes,
        input_ports=input_ports,
        output_ports=output_ports,
        unfinished_checks=unfinished,
        input_virtual_count=layout.input_virtual_count,
    )
    physical_conditionals_raw = collect_physical_conditionals(
        gadget,
        codes,
        layout.input_virtual_count,
        input_ports,
        output_ports,
        layout.internal_count,
    )
    resolved_physical_conditionals: list[tuple[int, list[int]]] = []
    for pc in physical_conditionals_raw:
        flipped: list[int] = []
        for target in pc.targets:
            flipped.extend(conditional_flipped_rows(target, output_ports, codes))
        resolved_physical_conditionals.append((pc.internal_meas_index, flipped))
    pc_pb = compute_physical_correction(
        codes,
        output_ports=output_ports,
        unfinished_checks=unfinished,
        input_virtual_count=layout.input_virtual_count,
        ov_start=layout.ov_start,
        physical_conditionals=resolved_physical_conditionals,
        logical_physical_entries=logical_physical_entries,
    )

    by_position: dict[int, list[jit_pb.JitGadgetType.Error]] = {}
    for body_index, error_row in iter_noise_errors_with_origin(
        gadget,
        codes,
        output_ports=output_ports,
        input_virtual_count=layout.input_virtual_count,
        finished_checks=finished,
        unfinished_checks=unfinished,
        ov_start=layout.ov_start,
        readouts_info=readouts_info,
        physical_correction=pc_pb,
    ):
        by_position.setdefault(body_index, []).append(error_row)

    return by_position, len(finished), cp_pb, pc_pb, layout.input_virtual_count


# ---------------------------------------------------------------------------
# COMPOSE → GADGET rendering
# ---------------------------------------------------------------------------


def _render_composed_gadget(
    gadget: jit_pb.JitGadgetType,
    stab_count_of_ptype: dict[int, int],
    compose: ComposeDefinition,
    gadget_defs: dict[str, GadgetDefinition],
    compose_defs: dict[str, ComposeDefinition],
    codes: dict[str, CodeDefinition],
    *,
    keep_noise: bool = False,
) -> str:
    """Render a composed ``JitGadgetType`` as a ``GADGET`` block.

    Instead of opaque placeholders, the actual circuit of each
    sub-gadget is inlined (noise instructions are commented out).  Port
    qubits are densely numbered starting at 0; ancilla qubits follow.

    When ``keep_noise`` is ``True``, the noise instructions are emitted
    verbatim and the merge-derived ``ERROR`` rows are skipped, so
    re-transpilation re-derives them from the inlined circuit.  This
    is orthogonal to whether the COMPOSE has ``@REPROPAGATE``: it
    only changes the noise rendering, not the propagation matrices.
    """
    base = gadget.base
    name = base.name or f"AnonymousGadget{base.gtype}"

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

    # INPUT lines from sub-gadgets' port declarations.
    for port in input_ports:
        lines.append(_render_input_or_output(port, "INPUT"))

    # Circuit body: inline sub-gadget instructions, noise commented out.
    for stmt in circuit_stmts:
        if isinstance(stmt, Instruction):
            name = stmt.name.upper()
            if name in NOISE_INSTRUCTIONS_ALL:
                # Passthrough noise (LOSS_ERROR) has no ERROR-row
                # equivalent; keep verbatim regardless of `keep_noise`.
                if keep_noise or name in PASSTHROUGH_NOISE_INSTRUCTIONS:
                    lines.append(f"    {stmt}")
                else:
                    lines.append(f"    # {stmt}")
            elif stmt.arguments and name in NOISY_MEASUREMENT_INSTRUCTIONS:
                if keep_noise:
                    lines.append(f"    {stmt}")
                else:
                    # Strip noise arguments from measurements (e.g. M(0.01) → M)
                    # so re-transpilation doesn't generate extra noise errors.
                    clean = Instruction(
                        name=stmt.name,
                        targets=stmt.targets,
                    )
                    lines.append(f"    {clean}")
            else:
                lines.append(f"    {stmt}")

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
    prop = base.readout_propagation
    input_col_layout = PortColumnLayout(input_ports, codes)
    for row_index, readout in enumerate(base.readouts):
        rec_refs = [f"M{mi}" for mi in readout.measurement_indices]
        if rec_refs:
            comment = _format_propagation_comment(
                prop,
                row_index,
                layout=input_col_layout,
            )
            suffix = f"  {comment}" if comment else ""
            lines.append("    READOUT " + " ".join(rec_refs) + suffix)

    # PROPAGATE statements pin every output logical row to the cp/pc
    # representative the COMPOSE pipeline picked, so re-transpilation
    # of the rendered GADGET produces a byte-identical
    # ``correction_propagation`` and ``physical_correction``.
    output_col_layout = PortColumnLayout(output_ports, codes)
    propagate_lines = _format_propagate_statements(
        base.correction_propagation,
        base.physical_correction,
        input_layout=input_col_layout,
        output_layout=output_col_layout,
    )
    lines.extend(propagate_lines)

    # ERROR statements.  When ``keep_noise`` is set, the noise
    # instructions above are emitted verbatim, so re-transpilation
    # re-derives the same ERROR rows from circuit flow — emitting
    # them here as well would duplicate them.
    if not keep_noise:
        num_finished = len(gadget.finished_checks)
        for error_row in gadget.errors:
            lines.append(
                "    "
                + _render_jit_error_to_source(
                    error_row,
                    num_finished=num_finished,
                    layout=output_col_layout,
                )
            )

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
