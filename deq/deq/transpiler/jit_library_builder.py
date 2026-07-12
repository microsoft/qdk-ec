"""Build a ``JitLibrary`` protobuf from a parsed ``.deq`` file.

This turns parsed gadget/code definitions + their derived parity-check
structure (see :mod:`deq.transpiler.jit_transpiler`) into the
serialisable ``deq.jit.JitLibrary`` protobuf consumed by the deq
runtime / JIT compiler.

Each ``JitGadgetType`` is fully populated: ``finished_checks``,
``unfinished_checks``, and ``errors`` (from both explicit ``ERROR``
statements and automatic noise-channel propagation).

``@GTYPE(n)`` and ``@PTYPE(n)`` decorators may pin a specific
globally-unique id on ``GADGET`` / ``CODE`` / ``COMPOSE`` definitions;
remaining definitions are auto-assigned sequentially starting at the
smallest available id.
"""

from dataclasses import dataclass
from typing import Sequence

import deq.proto.deq_bin_pb2 as pb
import deq.proto.deq_jit_pb2 as jit_pb
import deq.proto.util_pb2 as util_pb
from deq.circuit.model import (
    CheckTarget,
    CodeDefinition,
    ComposeDefinition,
    ConditionalStatement,
    Decorator,
    DestabilizerTarget,
    ErrorStatement,
    GadgetDefinition,
    InputPort,
    InputVirtualTarget,
    Instruction,
    LogicalPauliTarget,
    MeasurementRecordTarget,
    MeasurementRefTarget,
    OutputPort,
    OutputVirtualTarget,
    PauliTarget,
    PhysicalMeasurementTarget,
    DeqFile,
    PropagateStatement,
    QubitTarget,
    ReadoutStatement,
    ReadoutTarget,
)
from deq.transpiler.compose_builder import (
    build_compose_jit_gadget_type,
    compose_to_synthetic_gadget,
)
from deq.transpiler.jit_transpiler import (
    flatten_body,
    num_frame_columns,
    resolve_measurement_ref_global,
    x_column,
    z_column,
    select_stabilizer_generators,
    PortColumnLayout,
)
from deq.transpiler.check_plugins import (
    compute_layout,
    resolve_gadget_checks,
    warn_unrecognized_decorators,
)
from deq.transpiler.jit_noise_builder import (
    _resolve_logical_target_to_columns,
    _resolve_ds_to_input_cols,
    compute_correction_propagation,
    compute_implicit_readout_propagation,
    compute_noise_errors,
    compute_physical_correction,
    resolve_propagations,
)
import stim

from deq.transpiler.code_validation import validate_code
from deq.transpiler.stim_constants import qubit_indices as _qubit_indices
from deq.transpiler.stim_constants import (
    PASSTHROUGH_NOISE_INSTRUCTIONS,
    mpp_measurement_count,
    split_mpp_targets,
)


def _measurement_tags_of(inst: Instruction) -> list[str]:
    """Return one human-readable tag per measurement produced by *inst*."""
    name = inst.name.upper()
    if name in PASSTHROUGH_NOISE_INSTRUCTIONS:
        # ``LOSS_ERROR`` (and other QDK-style passthrough extensions) are
        # unknown to upstream Stim; they produce no measurement bits.
        return []
    gate = stim.gate_data(name)
    if not gate.produces_measurements:
        return []
    if gate.takes_pauli_targets:
        # MPP / SPP — one measurement per Pauli-product group.
        # The first PauliTarget in a group may be inverted (``!``).
        tags: list[str] = []
        for group in split_mpp_targets(list(inst.targets)):
            prefix = "!" if group[0].inverted else ""
            paulis = "*".join(f"{pt.pauli}{pt.index}" for pt in group)
            tags.append(f"{name} {prefix}{paulis}")
        return tags
    qubits = _qubit_indices(inst)
    if gate.is_two_qubit_gate:
        # MXX, MYY, MZZ — one measurement per pair.
        return [f"{name} {qubits[i]} {qubits[i + 1]}" for i in range(0, len(qubits), 2)]
    # Single-qubit measurements (M, MX, MR, etc.) and MPAD.
    single_tags: list[str] = []
    for t in inst.targets:
        if isinstance(t, QubitTarget):
            prefix = "!" if t.inverted else ""
            single_tags.append(f"{name} {prefix}{t.index}")
    return single_tags


def _measurement_count_of(inst: Instruction) -> int:
    """Return the number of measurements produced by *inst*."""
    name = inst.name.upper()
    if name in PASSTHROUGH_NOISE_INSTRUCTIONS:
        return 0
    gate = stim.gate_data(name)
    if not gate.produces_measurements:
        return 0
    if gate.takes_pauli_targets:
        return mpp_measurement_count(list(inst.targets))
    if gate.is_two_qubit_gate:
        return len(_qubit_indices(inst)) // 2
    return len(_qubit_indices(inst))


def build_jit_library(
    qfile: DeqFile,
    *,
    jobs: int = 1,
) -> jit_pb.JitLibrary:
    """Build a :class:`JitLibrary` from a parsed deq file.

    Parameters
    ----------
    qfile:
        Parsed ``.deq`` file containing CODE, GADGET, and COMPOSE
        definitions.
    jobs:
        Number of parallel worker processes for GADGET type construction.
        ``1`` (default) runs sequentially with no subprocess overhead.
        Values > 1 use :class:`~concurrent.futures.ProcessPoolExecutor`.
    """
    scaffold = _build_library_scaffold(qfile)

    if jobs > 1 and len(scaffold.gadgets) > 1:
        gadget_types = _build_gadget_types_parallel(
            scaffold.gadgets,
            scaffold.gtype_of_gadget,
            scaffold.ptype_of_code,
            scaffold.code_by_name,
            jobs,
        )
    else:
        gadget_types = [
            _build_jit_gadget_type(
                gadget,
                scaffold.gtype_of_gadget[gadget.name],
                scaffold.ptype_of_code,
                scaffold.code_by_name,
            )
            for gadget in scaffold.gadgets
        ]

    # Process COMPOSE definitions in source order. Each one becomes a new
    # JitGadgetType visible to subsequent COMPOSEs (so nested COMPOSE works
    # automatically as long as the inner one is declared first).
    jit_by_name: dict[str, jit_pb.JitGadgetType] = {
        g.name: jt for g, jt in zip(scaffold.gadgets, gadget_types)
    }
    compose_so_far: dict[str, ComposeDefinition] = {}
    for compose in scaffold.composes:
        composed_jit = build_compose_jit_gadget_type(
            compose,
            gtype=scaffold.gtype_of_compose[compose.name],
            gadget_definitions=scaffold.gadget_by_name,
            compose_definitions=compose_so_far,
            jit_gadget_types_by_name=jit_by_name,
            codes=scaffold.code_by_name,
            ptype_of_code=scaffold.ptype_of_code,
            port_types=scaffold.port_types,
        )
        gadget_types.append(composed_jit)
        jit_by_name[compose.name] = composed_jit
        compose_so_far[compose.name] = compose

    return jit_pb.JitLibrary(
        port_types=sorted(scaffold.port_types, key=lambda p: p.base.ptype),
        gadget_types=sorted(gadget_types, key=lambda g: g.base.gtype),
    )


def build_jit_program(qfile: DeqFile) -> jit_pb.JitLibrary:
    """Build a program-only :class:`JitLibrary` (fast path).

    Same shape as :func:`build_jit_library`, but populates only the
    metadata :func:`deq.cli.jit.compile_program_for_jit` and
    :func:`deq.cli.jit.export_program_stim` read — port types,
    per-gadget input/output ptypes, measurement/readout counts, and
    the ``correction_propagation`` matrix shape (needed for VIRTUAL
    Pauli corrections in PROGRAM bodies). ``COMPOSE`` definitions are
    inlined into synthetic gadgets via
    :func:`deq.transpiler.compose_builder.compose_to_synthetic_gadget`,
    so they're treated uniformly with ordinary ``GADGET`` blocks.

    Skips the per-gadget stabilizer simulation, noise propagation,
    and check resolution, so it's typically more than an order of
    magnitude faster than :func:`build_jit_library`. The result is
    **not** decoder-compatible.
    """
    scaffold = _build_library_scaffold(qfile)

    gadget_types = [
        _build_jit_program_gadget_type(
            gadget,
            scaffold.gtype_of_gadget[gadget.name],
            scaffold.ptype_of_code,
            scaffold.obs_count_of_ptype,
            codes=scaffold.code_by_name,
        )
        for gadget in scaffold.gadgets
    ]

    # Process COMPOSE definitions in source order. Each one becomes a
    # synthetic GadgetDefinition (via the same inliner the CLI uses) so
    # the lite gadget builder handles it uniformly with regular gadgets.
    # Nested COMPOSEs work as long as the inner one is declared first.
    augmented_gadgets: dict[str, GadgetDefinition] = dict(scaffold.gadget_by_name)
    compose_so_far: dict[str, ComposeDefinition] = {}
    for compose in scaffold.composes:
        synthetic = compose_to_synthetic_gadget(
            compose, augmented_gadgets, compose_so_far, scaffold.code_by_name
        )
        gadget_types.append(
            _build_jit_program_gadget_type(
                synthetic,
                scaffold.gtype_of_compose[compose.name],
                scaffold.ptype_of_code,
                scaffold.obs_count_of_ptype,
                codes=scaffold.code_by_name,
            )
        )
        augmented_gadgets[compose.name] = synthetic
        compose_so_far[compose.name] = compose

    return jit_pb.JitLibrary(
        port_types=sorted(scaffold.port_types, key=lambda p: p.base.ptype),
        gadget_types=sorted(gadget_types, key=lambda g: g.base.gtype),
    )


@dataclass
class _LibraryScaffold:
    """Common setup state shared by every ``JitLibrary`` builder.

    Holds the parsed definitions classified by kind, their assigned
    ``ptype`` / ``gtype`` ids, the built ``JitPortType``s, and a cached
    per-``ptype`` observable count (number of logical observables plus
    selected stabilizer-generator columns; used to size the
    ``correction_propagation`` matrix without re-walking ``port_types``).
    Per-gadget construction — the expensive part that varies between
    full decoder-capable and program-only builds — happens after this
    in the caller-specific way.
    """

    codes: list[CodeDefinition]
    gadgets: list[GadgetDefinition]
    composes: list[ComposeDefinition]
    code_by_name: dict[str, CodeDefinition]
    gadget_by_name: dict[str, GadgetDefinition]
    ptype_of_code: dict[str, int]
    gtype_of_gadget: dict[str, int]
    gtype_of_compose: dict[str, int]
    port_types: list[jit_pb.JitPortType]
    obs_count_of_ptype: dict[int, int]


def _build_library_scaffold(qfile: DeqFile) -> _LibraryScaffold:
    """Classify definitions, run early validation, assign ids, build ports.

    GADGETs and COMPOSEs share the ``gtype`` namespace: ``@GTYPE(n)`` pins
    are honoured for both, and unpinned definitions auto-assign from the
    smallest available id.
    """
    codes: list[CodeDefinition] = [
        d for d in qfile.definitions if isinstance(d, CodeDefinition)
    ]
    gadgets: list[GadgetDefinition] = [
        d for d in qfile.definitions if isinstance(d, GadgetDefinition)
    ]
    composes: list[ComposeDefinition] = [
        d for d in qfile.definitions if isinstance(d, ComposeDefinition)
    ]

    for code in codes:
        validate_code(code)
    for code in codes:
        warn_unrecognized_decorators(code)
    for gadget in gadgets:
        warn_unrecognized_decorators(gadget)
    for compose in composes:
        warn_unrecognized_decorators(compose)

    ptype_of_code = _assign_ids(codes, "PTYPE")
    all_gtypes = _assign_ids(list(gadgets) + list(composes), "GTYPE")
    gadget_names = {g.name for g in gadgets}
    gtype_of_gadget = {n: t for n, t in all_gtypes.items() if n in gadget_names}
    gtype_of_compose = {n: t for n, t in all_gtypes.items() if n not in gadget_names}

    port_types = [
        _build_jit_port_type(code, ptype_of_code[code.name]) for code in codes
    ]
    obs_count_of_ptype = {
        pt.base.ptype: len(pt.base.observables) for pt in port_types
    }

    return _LibraryScaffold(
        codes=codes,
        gadgets=gadgets,
        composes=composes,
        code_by_name={c.name: c for c in codes},
        gadget_by_name={g.name: g for g in gadgets},
        ptype_of_code=ptype_of_code,
        gtype_of_gadget=gtype_of_gadget,
        gtype_of_compose=gtype_of_compose,
        port_types=port_types,
        obs_count_of_ptype=obs_count_of_ptype,
    )


def _build_jit_program_gadget_type(
    gadget: GadgetDefinition,
    gtype: int,
    ptype_of_code: dict[str, int],
    obs_count_of_ptype: dict[int, int],
    *,
    codes: dict[str, CodeDefinition],
) -> jit_pb.JitGadgetType:
    """Lean ``JitGadgetType`` for the program-only fast path.

    Populates only the fields :func:`deq.cli.jit.compile_program_for_jit`
    reads — name / gtype / input + output ptypes, measurement and
    readout counts, and the ``correction_propagation`` matrix shape
    (rows / cols only, no entries) so VIRTUAL Pauli corrections still
    work. No tags, no checks, no propagation entries, no parity
    validation: callers who need any of that should use
    :func:`build_jit_library`.

    Measurement count comes from the simulate view, matching what Stim
    actually emits — so a partition derived from ``base.measurements``
    lines up with the bits Stim produces.
    """
    for port in gadget.input_ports:
        _validate_port_qubit_count(port, codes, gadget.name, "INPUT")
    for port in gadget.output_ports:
        _validate_port_qubit_count(port, codes, gadget.name, "OUTPUT")

    measurement_count = sum(
        _measurement_count_of(statement)
        for statement in flatten_body(list(gadget.body), for_simulate=True)
        if isinstance(statement, Instruction)
    )
    readout_count = sum(
        1
        for statement in flatten_body(list(gadget.body))
        if isinstance(statement, ReadoutStatement)
    )

    input_observable_count = sum(
        obs_count_of_ptype[ptype_of_code[port.code_name]]
        for port in gadget.input_ports
    )
    output_observable_count = sum(
        obs_count_of_ptype[ptype_of_code[port.code_name]]
        for port in gadget.output_ports
    )
    correction_propagation_shape = util_pb.BitMatrix(
        rows=output_observable_count,
        cols=input_observable_count + 1,
    )

    return jit_pb.JitGadgetType(
        base=pb.GadgetType(
            gtype=gtype,
            name=gadget.name,
            inputs=[
                pb.GadgetType.Port(ptype=ptype_of_code[port.code_name])
                for port in gadget.input_ports
            ],
            outputs=[
                pb.GadgetType.Port(ptype=ptype_of_code[port.code_name])
                for port in gadget.output_ports
            ],
            measurements=[
                pb.GadgetType.Measurement() for _ in range(measurement_count)
            ],
            readouts=[pb.GadgetType.Readout() for _ in range(readout_count)],
            correction_propagation=correction_propagation_shape,
        ),
    )


def _build_gadget_types_parallel(
    gadgets: list[GadgetDefinition],
    gtype_of_gadget: dict[str, int],
    ptype_of_code: dict[str, int],
    code_by_name: dict[str, CodeDefinition],
    jobs: int,
) -> list[jit_pb.JitGadgetType]:
    """Build gadget types in parallel using worker processes.

    Protobuf messages are not picklable, so each worker serializes its
    result as bytes and the main process deserializes.
    """
    from concurrent.futures import ProcessPoolExecutor

    args = [(g, gtype_of_gadget[g.name], ptype_of_code, code_by_name) for g in gadgets]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        result_bytes = list(pool.map(_build_jit_gadget_type_bytes, args))

    return [jit_pb.JitGadgetType.FromString(b) for b in result_bytes]


def _build_jit_gadget_type_bytes(
    args: tuple[GadgetDefinition, int, dict[str, int], dict[str, CodeDefinition]],
) -> bytes:
    """Worker entry point: build a JitGadgetType and return serialized bytes."""
    g, gtype, ptype_of_code, code_by_name = args
    result = _build_jit_gadget_type(g, gtype, ptype_of_code, code_by_name)
    return result.SerializeToString()


# ---------------------------------------------------------------------------
# Id assignment
# ---------------------------------------------------------------------------


def _pinned_id(decorators: Sequence[Decorator], decorator_name: str) -> int | None:
    pinned: int | None = None
    for deco in decorators:
        if deco.name != decorator_name:
            continue
        if len(deco.arguments) != 1:
            raise ValueError(
                f"@{decorator_name} expects exactly one integer argument; "
                f"got {deco.arguments!r}"
            )
        arg = deco.arguments[0]
        if not isinstance(arg, int) or isinstance(arg, bool) or arg <= 0:
            raise ValueError(
                f"@{decorator_name} argument must be a positive integer; "
                f"got {arg!r}"
            )
        if pinned is not None and pinned != arg:
            raise ValueError(f"multiple conflicting @{decorator_name}(...) decorators")
        pinned = arg
    return pinned


def _assign_ids(
    definitions: Sequence[CodeDefinition | GadgetDefinition | ComposeDefinition],
    decorator_name: str,
) -> dict[str, int]:
    result: dict[str, int] = {}
    taken: set[int] = set()
    pending: list[CodeDefinition | GadgetDefinition | ComposeDefinition] = []

    for definition in definitions:
        pinned = _pinned_id(definition.decorators, decorator_name)
        if pinned is None:
            pending.append(definition)
            continue
        if pinned in taken:
            raise ValueError(
                f"@{decorator_name}({pinned}) on {definition.name!r} "
                f"conflicts with an earlier pin"
            )
        taken.add(pinned)
        result[definition.name] = pinned

    next_id = 1
    for definition in pending:
        while next_id in taken:
            next_id += 1
        result[definition.name] = next_id
        taken.add(next_id)
        next_id += 1

    return result


# ---------------------------------------------------------------------------
# Port types
# ---------------------------------------------------------------------------


def _validate_port_qubit_count(
    port: InputPort | OutputPort,
    codes: dict[str, CodeDefinition],
    gadget_name: str,
    kind: str,
) -> None:
    """Raise if the port declares a different number of qubits than the code's ``n``."""
    if port.code_name not in codes:
        known = sorted(codes)
        known_str = ", ".join(repr(name) for name in known) if known else "(none)"
        raise ValueError(
            f"{kind} port in GADGET {gadget_name!r} references undefined "
            f"CODE {port.code_name!r}. Known CODE names: {known_str}.\n"
            f"  Hint: define a 'CODE {port.code_name} [[n,k,d]] {{ ... }}' "
            f"block, or change the {kind} port to reference an existing code."
        )
    code = codes[port.code_name]
    if len(port.qubit_indices) != code.n:
        raise ValueError(
            f"{kind} port '{port.code_name}' in GADGET {gadget_name!r} declares "
            f"{len(port.qubit_indices)} qubit(s), but code "
            f"'{code.name}' has n={code.n}"
        )


def _build_jit_port_type(code: CodeDefinition, ptype: int) -> jit_pb.JitPortType:
    observables: list[pb.PortType.Observable] = []
    # Logical observables: LX_i, LZ_i per logical qubit
    for logical in code.logicals:
        observables.append(pb.PortType.Observable(tag=str(logical.x_operator)))
        observables.append(pb.PortType.Observable(tag=str(logical.z_operator)))
    # Stabilizer generator columns (one per selected generator)
    sel = select_stabilizer_generators(code)
    for gen_idx in sel.generator_indices:
        stab = code.stabilizers[gen_idx]
        observables.append(pb.PortType.Observable(tag=f"S{gen_idx}:{stab}"))

    stabilizers = [
        jit_pb.JitPortType.Stabilizer(tag=str(stab)) for stab in code.stabilizers
    ]
    base = pb.PortType(
        ptype=ptype,
        name=code.name,
        observables=observables,
    )
    return jit_pb.JitPortType(base=base, k=code.k, stabilizers=stabilizers)


# ---------------------------------------------------------------------------
# Gadget types
# ---------------------------------------------------------------------------


def _build_jit_gadget_type(
    gadget: GadgetDefinition,
    gtype: int,
    ptype_of_code: dict[str, int],
    codes: dict[str, CodeDefinition],
    *,
    check_override: tuple[
        list[tuple[frozenset[int], bool]],
        list[tuple[frozenset[int], bool]],
    ]
    | None = None,
) -> jit_pb.JitGadgetType:
    """Build a ``JitGadgetType`` from a ``GadgetDefinition``.

    When *check_override* is provided as ``(finished, unfinished)``, it
    replaces what :func:`resolve_gadget_checks` would derive from the
    gadget body.  The propagation matrices and noise-derived ERROR
    rows are then computed against this externally supplied check basis
    so all downstream check indices remain self-consistent.  This is
    used by the ``@REPROPAGATE`` compose path to graft the merge()
    pipeline's check structure onto a flat-circuit propagation /
    error derivation.
    """
    input_ports = gadget.input_ports
    output_ports = gadget.output_ports

    for port in input_ports:
        _validate_port_qubit_count(port, codes, gadget.name, "INPUT")
    for port in output_ports:
        _validate_port_qubit_count(port, codes, gadget.name, "OUTPUT")

    layout = compute_layout(gadget, codes)
    input_virtual_count = layout.input_virtual_count
    ov_start = layout.ov_start

    input_stabilizer_offsets: list[int] = []
    running = 0
    for port in input_ports:
        input_stabilizer_offsets.append(running)
        running += len(codes[port.code_name].stabilizers)

    measurement_tags: list[str] = []
    for stmt in flatten_body(list(gadget.body)):
        if isinstance(stmt, Instruction):
            measurement_tags.extend(_measurement_tags_of(stmt))
    internal_count = len(measurement_tags)

    # Validate that decode and simulate views produce the same number of
    # physical measurements.  A mismatch means the user put @SIMULATE_ONLY
    # or @DECODE_ONLY on a measurement instruction without a matching
    # counterpart.
    sim_meas_count = sum(
        _measurement_count_of(s)
        for s in flatten_body(list(gadget.body), for_simulate=True)
        if isinstance(s, Instruction)
    )
    if internal_count != sim_meas_count:
        raise ValueError(
            f"GADGET {gadget.name!r} has mismatched measurement counts between "
            f"decode view ({internal_count}) and simulate view ({sim_meas_count}). "
            f"Every @SIMULATE_ONLY measurement must be paired with a "
            f"@DECODE_ONLY measurement (and vice versa) so both views "
            f"produce the same number of measurement records."
        )

    base_measurements = [pb.GadgetType.Measurement(tag=tag) for tag in measurement_tags]

    base_inputs = [
        pb.GadgetType.Port(ptype=ptype_of_code[p.code_name]) for p in input_ports
    ]
    base_outputs = [
        pb.GadgetType.Port(ptype=ptype_of_code[p.code_name]) for p in output_ports
    ]

    if check_override is not None:
        finished, unfinished = check_override
    else:
        check_result = resolve_gadget_checks(gadget, codes)
        finished = check_result.finished
        unfinished = check_result.unfinished
    total = (
        input_virtual_count
        + internal_count
        + sum(len(codes[p.code_name].stabilizers) for p in output_ports)
    )
    num_output_virtual = sum(len(codes[p.code_name].stabilizers) for p in output_ports)
    if total - ov_start != num_output_virtual:
        raise ValueError(
            f"internal error: output-virtual count mismatch for gadget "
            f"{gadget.name!r}: expected {num_output_virtual}, got {total - ov_start}"
        )

    def _to_present(index: int) -> jit_pb.JitGadgetType.PresentMeasurement:
        if index < input_virtual_count:
            port_index, offset = _input_port_of(index, input_stabilizer_offsets)
            return jit_pb.JitGadgetType.PresentMeasurement(
                input_port=port_index, measurement_index=offset
            )
        if index < ov_start:
            return jit_pb.JitGadgetType.PresentMeasurement(
                measurement_index=index - input_virtual_count
            )
        raise ValueError(
            f"internal error: output-virtual index {index} leaked into a "
            f"present-measurement list"
        )

    def _build_check(
        members: frozenset[int], parity: bool, drop_ov: int | None
    ) -> jit_pb.JitGadgetType.Check:
        present: list[jit_pb.JitGadgetType.PresentMeasurement] = []
        for idx in sorted(members):
            if drop_ov is not None and idx == drop_ov:
                continue
            present.append(_to_present(idx))
        base = pb.CheckModelType.Check(
            tag=_check_tag(members, parity),
            naturally_flipped=parity,
        )
        return jit_pb.JitGadgetType.Check(base=base, measurements=present)

    finished_pb: list[jit_pb.JitGadgetType.Check] = [
        _build_check(members, parity, drop_ov=None) for members, parity in finished
    ]

    unfinished_pb: list[jit_pb.JitGadgetType.Check] = []
    for k, (members, parity) in enumerate(unfinished):
        ov_index = ov_start + k
        if ov_index not in members:
            raise ValueError(
                f"internal error: unfinished check #{k} does not contain "
                f"its output-virtual measurement {ov_index}"
            )
        unfinished_pb.append(_build_check(members, parity, drop_ov=ov_index))

    readouts_pb, readout_propagation_pb, readouts_info = build_readouts(
        gadget, codes, input_virtual_count, input_ports, output_ports, internal_count
    )
    num_output_observables = sum(
        num_frame_columns(codes[p.code_name]) for p in output_ports
    )
    input_layout = PortColumnLayout(input_ports, codes)
    propagations = resolve_propagations(
        gadget,
        codes,
        input_ports=input_ports,
        output_ports=output_ports,
        input_layout=input_layout,
        input_virtual_count=input_virtual_count,
        ov_start=ov_start,
    )

    # CONDITIONAL R<j> statements and PROPAGATE R<k> terms in the body
    # are both translated faithfully into the ``logical_correction``
    # matrix (see :func:`_build_logical_correction` below).  PROPAGATE
    # rows are authoritative: whatever the user declares is installed
    # verbatim into ``correction_propagation`` / ``physical_correction``
    # for that output row, with no basis-freedom check against the
    # natural-Heisenberg derivation.

    correction_propagation_pb, logical_physical_entries = (
        compute_correction_propagation(
            gadget,
            codes,
            input_ports=input_ports,
            output_ports=output_ports,
            unfinished_checks=unfinished,
            finished_checks=finished,
            input_virtual_count=input_virtual_count,
            ov_start=ov_start,
            propagations=propagations,
        )
    )
    logical_correction_pb = _build_logical_correction(
        gadget,
        num_output_observables,
        len(readouts_pb),
        output_ports,
        codes,
    )

    physical_conditionals_raw = collect_physical_conditionals(
        gadget, codes, input_virtual_count, input_ports, output_ports, internal_count
    )
    resolved_physical_conditionals: list[tuple[int, list[int]]] = []
    for pc in physical_conditionals_raw:
        flipped: list[int] = []
        for target in pc.targets:
            flipped.extend(conditional_flipped_rows(target, output_ports, codes))
        resolved_physical_conditionals.append((pc.internal_meas_index, flipped))
    physical_correction_pb = compute_physical_correction(
        codes,
        output_ports=output_ports,
        unfinished_checks=unfinished,
        input_virtual_count=input_virtual_count,
        ov_start=ov_start,
        physical_conditionals=resolved_physical_conditionals,
        logical_physical_entries=logical_physical_entries,
    )

    errors_pb = _build_errors(
        gadget,
        codes,
        output_ports,
        num_finished=len(finished_pb),
        num_unfinished=len(unfinished_pb),
        num_readouts=len(readouts_pb),
    )
    errors_pb.extend(
        compute_noise_errors(
            gadget,
            codes,
            output_ports=output_ports,
            input_virtual_count=input_virtual_count,
            finished_checks=finished,
            unfinished_checks=unfinished,
            ov_start=ov_start,
            readouts_info=readouts_info,
            physical_correction=physical_correction_pb,
        )
    )

    base = pb.GadgetType(
        gtype=gtype,
        name=gadget.name,
        measurements=base_measurements,
        inputs=base_inputs,
        outputs=base_outputs,
        readouts=readouts_pb,
        readout_propagation=readout_propagation_pb,
        correction_propagation=correction_propagation_pb,
        logical_correction=logical_correction_pb,
        physical_correction=physical_correction_pb,
    )
    return jit_pb.JitGadgetType(
        base=base,
        finished_checks=finished_pb,
        unfinished_checks=unfinished_pb,
        errors=errors_pb,
    )


def _input_port_of(
    index: int, input_stabilizer_offsets: Sequence[int]
) -> tuple[int, int]:
    for port_index in range(len(input_stabilizer_offsets) - 1, -1, -1):
        start = input_stabilizer_offsets[port_index]
        if index >= start:
            return port_index, index - start
    raise ValueError(f"internal error: index {index} has no containing input port")


def _check_tag(members: frozenset[int], parity: bool) -> str:
    sorted_members = sorted(members)
    suffix = " FLIP" if parity else ""
    return "CHECK " + " ".join(f"m{m}" for m in sorted_members) + suffix


# ---------------------------------------------------------------------------
# Conditional correction (logical feedforward)
# ---------------------------------------------------------------------------


@dataclass
class _PhysicalConditional:
    """A resolved ``CONDITIONAL rec[-k] L<P><i>`` entry.

    ``internal_meas_index`` is the 0-based index into the gadget's
    internal measurements (i.e. column in ``physical_correction``).
    """

    internal_meas_index: int
    targets: list[LogicalPauliTarget]


def collect_physical_conditionals(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
    input_virtual_count: int,
    input_ports: list[InputPort],
    output_ports: list[OutputPort],
    internal_count: int,
) -> list[_PhysicalConditional]:
    """Walk the gadget body, resolve ``CONDITIONAL`` references to internal
    measurement indices, and return the list of physical conditionals.

    Each ``CONDITIONAL`` may use any of the four measurement-reference
    forms (``rec[-k]``, ``M<i>``, ``IN<p>.S<s>``, ``OUT<p>.S<s>``).  The
    resolved target must lie in the internal/physical region; virtual
    stabilizer references are rejected.
    """
    ov_start = input_virtual_count + internal_count
    result: list[_PhysicalConditional] = []
    running = 0
    for stmt in flatten_body(list(gadget.body)):
        if isinstance(stmt, InputPort):
            running += len(codes[stmt.code_name].stabilizers)
        elif isinstance(stmt, OutputPort):
            running += len(codes[stmt.code_name].stabilizers)
        elif isinstance(stmt, Instruction):
            running += _measurement_count_of(stmt)
        elif isinstance(stmt, ConditionalStatement):
            cond = stmt.condition
            if not isinstance(
                cond,
                (
                    MeasurementRecordTarget,
                    PhysicalMeasurementTarget,
                    InputVirtualTarget,
                    OutputVirtualTarget,
                ),
            ):
                continue
            global_index = resolve_measurement_ref_global(
                cond,
                running=running,
                input_ports=input_ports,
                output_ports=output_ports,
                codes=codes,
                internal_count=internal_count,
                gadget_name=gadget.name,
            )
            if global_index < input_virtual_count:
                raise ValueError(
                    f"in GADGET {gadget.name!r}: CONDITIONAL {cond!s} "
                    f"references an input-virtual stabilizer measurement"
                )
            if global_index >= ov_start:
                raise ValueError(
                    f"in GADGET {gadget.name!r}: CONDITIONAL {cond!s} "
                    f"references an output-virtual stabilizer measurement"
                )
            internal_index = global_index - input_virtual_count
            result.append(
                _PhysicalConditional(
                    internal_meas_index=internal_index,
                    targets=stmt.targets,
                )
            )
    return result


def _build_logical_correction(
    gadget: GadgetDefinition,
    num_output_observables: int,
    num_readouts: int,
    output_ports: list[OutputPort],
    codes: dict[str, CodeDefinition],
) -> util_pb.BitMatrix:
    """Build the ``logical_correction`` matrix from CONDITIONAL and PROPAGATE.

    Two source-level constructs write to ``logical_correction``:

    1. ``CONDITIONAL R<j> L<P><i>`` — a logical Pauli correction
       conditioned on readout *j*.  The correction flips all
       anti-commuting output observables.

    2. ``PROPAGATE OUT<p>.LX<i> FROM ... R<j> ...`` — a readout term
       inside a ``PROPAGATE`` row.  Each ``R<j>`` term XORs
       ``logical_correction[target_row, j] = 1`` for every row the
       PROPAGATE target flips.  Non-readout terms in the same
       ``PROPAGATE`` are handled by ``_resolve_propagate_statements``
       and route into ``correction_propagation`` /
       ``physical_correction`` instead.

    Frame conventions:

    * In the **logical** frame (2 observables per logical qubit,
      interleaved X then Z), ``LX<i>`` flips ``LZ<i>`` and vice versa.
    * In the **physical** frame (2 observables per physical qubit),
      the logical operator is expanded into its physical Pauli string
      and each physical qubit's anti-commuting observable is flipped
      individually.

    Contributions from both sources XOR into the matrix.  Writing both
    a ``CONDITIONAL R<j> OUT.LP<i>`` and a ``PROPAGATE OUT.LP<i> FROM
    ... R<j> ...`` in the same gadget therefore cancels.
    """
    entries: set[tuple[int, int]] = set()
    num_logicals = sum(len(codes[p.code_name].logicals) for p in output_ports)

    def _xor_entry(row: int, readout_col: int) -> None:
        entries.symmetric_difference_update({(row, readout_col)})

    for stmt in flatten_body(list(gadget.body)):
        if isinstance(stmt, ConditionalStatement):
            if not isinstance(stmt.condition, ReadoutTarget):
                continue
            readout_col = stmt.condition.index
            if readout_col >= num_readouts:
                raise ValueError(
                    f"CONDITIONAL in gadget {gadget.name!r}: readout index "
                    f"R{readout_col} out of range (only {num_readouts} readouts "
                    f"declared)"
                )
            for target in stmt.targets:
                if target.port_kind is None and target.index >= num_logicals:
                    raise ValueError(
                        f"CONDITIONAL in gadget {gadget.name!r}: logical index "
                        f"L{target.pauli}{target.index} out of range (only "
                        f"{num_logicals} output logical qubits)"
                    )
                for row in conditional_flipped_rows(target, output_ports, codes):
                    _xor_entry(row, readout_col)

        elif isinstance(stmt, PropagateStatement):
            readout_terms = [t for t in stmt.terms if isinstance(t, ReadoutTarget)]
            if not readout_terms:
                continue
            target_rows = _resolve_logical_target_to_columns(
                stmt.target, list(output_ports), codes, expected_kind="OUT"
            )
            for term in readout_terms:
                if term.index >= num_readouts:
                    raise ValueError(
                        f"PROPAGATE in gadget {gadget.name!r}: readout index "
                        f"R{term.index} out of range (only {num_readouts} "
                        f"readouts declared)"
                    )
                for row in target_rows:
                    _xor_entry(row, term.index)

    sorted_entries = sorted(entries)
    rows_list = [r for r, _ in sorted_entries]
    cols_list = [c for _, c in sorted_entries]
    return util_pb.BitMatrix(
        rows=num_output_observables,
        cols=num_readouts,
        i=rows_list,
        j=cols_list,
    )


def pauli_to_observable_flips(
    paulis: list[tuple[str, int]],
    num_logical_qubits: int,
) -> list[int]:
    """Compute the column indices of a port's observable layout that flip
    when applying the logical Pauli product ``paulis``.

    Each entry ``(pauli_letter, logical_qubit_index)`` is a logical Pauli
    on one logical qubit of the port's code (``X``, ``Y``, or ``Z``).
    The flips follow the standard symplectic-pair convention (matching
    :func:`conditional_flipped_rows` and the layout described in
    :mod:`deq.transpiler.jit_transpiler`):

    * ``X`` on logical ``k`` flips the Z column (``z_column(k) = 2*k + 1``)
    * ``Z`` on logical ``k`` flips the X column (``x_column(k) = 2*k``)
    * ``Y`` on logical ``k`` flips both columns

    Stabilizer-generator columns are never flipped by a logical Pauli
    (logical operators commute with all stabilizers by construction).

    Multiple Paulis compose by XOR: ``X1 * X1`` cancels out, etc. The
    return value is a sorted list of column indices.
    """
    flips: set[int] = set()
    for pauli_letter, logical_idx in paulis:
        pauli = pauli_letter.upper()
        if pauli not in ("X", "Y", "Z"):
            raise ValueError(
                f"unsupported Pauli letter {pauli_letter!r}; "
                f"expected 'X', 'Y', or 'Z'"
            )
        if not 0 <= logical_idx < num_logical_qubits:
            raise ValueError(
                f"logical qubit index {logical_idx} out of range; "
                f"port has {num_logical_qubits} logical qubit(s)"
            )
        if pauli in ("X", "Y"):
            flips ^= {z_column(logical_idx)}
        if pauli in ("Z", "Y"):
            flips ^= {x_column(logical_idx)}
    return sorted(flips)


def conditional_flipped_rows(
    target: LogicalPauliTarget,
    output_ports: list[OutputPort],
    codes: dict[str, CodeDefinition],
) -> list[int]:
    """Return the observable rows flipped by applying logical Pauli *target*.

    Resolves the logical index to a specific output port, then computes
    the flipped rows.  When ``target`` is port-qualified
    (``OUT<p>.L<P><i>``), ``target.port_kind`` must be ``"OUT"`` and
    ``target.index`` is interpreted as logical-within-port.
    """
    if target.port_kind is not None:
        assert target.port_index is not None
        if target.port_kind != "OUT":
            raise ValueError(
                f"logical target {target!s}: only OUT-side logicals are "
                f"meaningful here (CONDITIONAL / READOUT / VIRTUAL all "
                f"reference output observables)"
            )
        if not 0 <= target.port_index < len(output_ports):
            raise ValueError(
                f"logical target {target!s}: port index out of range "
                f"(only {len(output_ports)} OUTPUT port(s))"
            )
        obs_offset = 0
        for p in output_ports[: target.port_index]:
            obs_offset += num_frame_columns(codes[p.code_name])
        port_code = codes[output_ports[target.port_index].code_name]
        n_logicals = len(port_code.logicals)
        if not 0 <= target.index < n_logicals:
            raise ValueError(
                f"logical target {target!s}: logical index out of range "
                f"(port has {n_logicals} logical observable(s))"
            )
        logical_idx = target.index
    else:
        # Resolve target.index to (port, logical_within_port, obs_offset).
        obs_offset = 0
        remaining = target.index
        matched_port: OutputPort | None = None
        logical_idx = 0
        for p in output_ports:
            code = codes[p.code_name]
            n_logicals = len(code.logicals)
            if remaining < n_logicals:
                matched_port = p
                logical_idx = remaining
                break
            remaining -= n_logicals
            obs_offset += num_frame_columns(code)

        assert matched_port is not None
    pauli = target.pauli.upper()

    # Unified frame: 2 columns per logical qubit (X at 2i, Z at 2i+1),
    # followed by stabilizer generator columns.
    # A logical Pauli LX_i flips the Z column, LZ_i flips the X column.
    rows: list[int] = []
    if pauli in ("X", "Y"):
        rows.append(obs_offset + z_column(logical_idx))  # flips Z
    if pauli in ("Z", "Y"):
        rows.append(obs_offset + x_column(logical_idx))  # flips X
    return rows


# ---------------------------------------------------------------------------
# Logical readouts
# ---------------------------------------------------------------------------


def build_readouts(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
    input_virtual_count: int,
    input_ports: list[InputPort],
    output_ports: list[OutputPort],
    internal_count: int,
) -> tuple[list[pb.GadgetType.Readout], util_pb.BitMatrix, list["_ReadoutInfo"]]:
    """Extract READOUT statements and build the ``readouts`` / propagation.

    ``GadgetType.Readout.measurement_indices`` indexes the gadget's
    physical (real) measurements only — it cannot reference input-
    virtual or output-virtual stabilizer measurements.  Each
    measurement target (``rec[-k]`` or ``M<i>``) is resolved to a
    global measurement index, validated to lie in the internal /
    physical region, and translated to a real-only index.

    The ``readout_propagation`` matrix is sized
    ``|readouts| x (|input_observables| + 1)``. Each row records:

    - measurement contributions are folded into the readout's
      ``measurement_indices`` (XOR-deduplicated);
    - the trailing affine/constant column reflects ``FLIP``.

    Observable columns are populated automatically by Clifford-circuit
    propagation (see :func:`compute_implicit_readout_propagation`).
    """
    num_input_observables = sum(
        num_frame_columns(codes[p.code_name]) for p in input_ports
    )

    readouts_info: list[_ReadoutInfo] = []
    running = 0
    output_virtual_indices: set[int] = set()
    for stmt in flatten_body(list(gadget.body)):
        if isinstance(stmt, InputPort):
            running += len(codes[stmt.code_name].stabilizers)
        elif isinstance(stmt, OutputPort):
            count = len(codes[stmt.code_name].stabilizers)
            for k in range(count):
                output_virtual_indices.add(running + k)
            running += count
        elif isinstance(stmt, Instruction):
            running += _measurement_count_of(stmt)
        elif isinstance(stmt, ReadoutStatement):
            readouts_info.append(
                _parse_readout(
                    stmt,
                    running,
                    input_virtual_count,
                    output_virtual_indices,
                    input_ports,
                    output_ports,
                    codes,
                    internal_count,
                    gadget.name,
                )
            )

    if not readouts_info:
        propagation = _empty_bit_matrix(0, num_input_observables + 1)
        return [], propagation, []

    readouts_pb: list[pb.GadgetType.Readout] = []
    for info in readouts_info:
        tag = _readout_tag(info.measurement_indices, info.affine_flip)
        readouts_pb.append(
            pb.GadgetType.Readout(tag=tag, measurement_indices=info.measurement_indices)
        )

    implicit = compute_implicit_readout_propagation(
        gadget,
        codes,
        input_ports=input_ports,
        readout_measurement_sets=[
            set(info.measurement_indices) for info in readouts_info
        ],
    )
    propagation = _build_readout_propagation(
        readouts_info, num_input_observables, implicit
    )
    return readouts_pb, propagation, readouts_info


@dataclass
class _ReadoutInfo:
    measurement_indices: list[int]
    affine_flip: bool
    explicit_logical_cols: set[int]
    explicit_destab_cols: set[int]


def _parse_readout(
    stmt: ReadoutStatement,
    running: int,
    input_virtual_count: int,
    output_virtual_indices: set[int],
    input_ports: list[InputPort],
    output_ports: list[OutputPort],
    codes: dict[str, CodeDefinition],
    internal_count: int,
    gadget_name: str,
) -> _ReadoutInfo:
    """Translate a ``ReadoutStatement`` into a :class:`_ReadoutInfo`.

    Accepts physical measurement references (``rec[-k]``, ``M<i>``),
    input-side logical Pauli targets (``IN<p>.L<P><i>`` / bare
    ``L<P><i>``), and input-side destabilizer targets
    (``IN<p>.DS<s>``) which explicitly encode ``readout_propagation``
    bits.

    Logical and destabilizer targets are XOR-combined with the
    implicit walker-derived columns at
    :func:`_build_readout_propagation` time so the rendered annotated
    form can override walker output for readouts whose
    matrix-composed rp differs from the inlined-body Heisenberg walk
    (e.g. chained teleportation's cumulative readouts, or nested
    composes that physically reset qubits carrying an input
    destabilizer's Pauli representative before the readout's
    measurements).  Raises ``ValueError`` if a ``rec[-k]`` reference
    resolves to a virtual stabilizer measurement.
    """
    measurement_indices: list[int] = []
    explicit_logical_cols: set[int] = set()
    explicit_destab_cols: set[int] = set()
    affine_flip = stmt.flip

    input_layout: PortColumnLayout | None = None

    for target in stmt.targets:
        if isinstance(
            target,
            (
                MeasurementRecordTarget,
                PhysicalMeasurementTarget,
            ),
        ):
            measurement_indices.append(
                _resolve_measurement_target(
                    target,
                    stmt,
                    running,
                    input_virtual_count,
                    output_virtual_indices,
                    input_ports,
                    output_ports,
                    codes,
                    internal_count,
                    gadget_name,
                )
            )
            continue
        if isinstance(target, LogicalPauliTarget):
            for col in _resolve_logical_target_to_columns(
                target, input_ports, codes, expected_kind="IN"
            ):
                explicit_logical_cols.symmetric_difference_update([col])
            continue
        if isinstance(target, DestabilizerTarget):
            if input_layout is None:
                input_layout = PortColumnLayout(input_ports, codes)
            for col in _resolve_ds_to_input_cols(
                target, input_layout, input_ports, codes
            ):
                explicit_destab_cols.symmetric_difference_update([col])
            continue
        raise ValueError(
            f"in GADGET {gadget_name!r}: {_render_readout(stmt)}: "
            f"unsupported target {target!r}; only physical measurement "
            f"references (rec[-k], M<i>), input logical Paulis "
            f"(IN<p>.L<P><i>), input destabilizers (IN<p>.DS<s>), "
            f"or FLIP are supported in READOUT statements"
        )

    return _ReadoutInfo(
        measurement_indices=sorted(_xor_deduplicate(measurement_indices)),
        affine_flip=affine_flip,
        explicit_logical_cols=explicit_logical_cols,
        explicit_destab_cols=explicit_destab_cols,
    )


def _resolve_measurement_target(
    target: MeasurementRefTarget,
    stmt: ReadoutStatement,
    running: int,
    input_virtual_count: int,
    output_virtual_indices: set[int],
    input_ports: list[InputPort],
    output_ports: list[OutputPort],
    codes: dict[str, CodeDefinition],
    internal_count: int,
    gadget_name: str,
) -> int:
    """Translate a physical measurement-reference target to a
    real-measurement index.

    The target is either ``rec[-k]`` (``MeasurementRecordTarget``) or
    ``M<i>`` (``PhysicalMeasurementTarget``).  ``M<i>`` always names a
    physical measurement by construction; ``rec[-k]`` may resolve to
    a virtual stabilizer slot depending on where the READOUT sits in
    the body, and is rejected in that case — readouts must reference
    physical measurements only.
    """
    global_index = resolve_measurement_ref_global(
        target,
        running=running,
        input_ports=input_ports,
        output_ports=output_ports,
        codes=codes,
        internal_count=internal_count,
        gadget_name=gadget_name,
    )
    if global_index < input_virtual_count:
        raise ValueError(
            f"in GADGET {gadget_name!r}: {_render_readout(stmt)}: "
            f"{target!s} references an input-virtual "
            f"stabilizer measurement (global index {global_index}); "
            f"logical readouts must refer to physical (real) "
            f"measurements only"
        )
    if global_index in output_virtual_indices:
        raise ValueError(
            f"in GADGET {gadget_name!r}: {_render_readout(stmt)}: "
            f"{target!s} references an output-virtual "
            f"stabilizer measurement (global index {global_index}); "
            f"logical readouts must refer to physical (real) "
            f"measurements only"
        )
    return global_index - input_virtual_count


def _render_readout(stmt: ReadoutStatement) -> str:
    targets = " ".join(str(t) for t in stmt.targets)
    suffix = " FLIP" if stmt.flip else ""
    return f"READOUT {targets}{suffix}".rstrip()


def _xor_deduplicate(indices: list[int]) -> list[int]:
    seen: dict[int, int] = {}
    for idx in indices:
        seen[idx] = seen.get(idx, 0) + 1
    return [idx for idx, count in seen.items() if count % 2 == 1]


def _readout_tag(measurement_indices: list[int], flip: bool) -> str:
    body = " ".join(f"m{m}" for m in measurement_indices)
    suffix = " FLIP" if flip else ""
    return f"READOUT {body}{suffix}".strip()


def _build_readout_propagation(
    readouts_info: list[_ReadoutInfo],
    num_input_observables: int,
    implicit_columns: list[set[int]],
) -> util_pb.BitMatrix:
    rows = len(readouts_info)
    cols = num_input_observables + 1
    row_idx: list[int] = []
    col_idx: list[int] = []
    for index, info in enumerate(readouts_info):
        effective_cols = (
            implicit_columns[index]
            ^ info.explicit_logical_cols
            ^ info.explicit_destab_cols
        )
        for col in sorted(effective_cols):
            row_idx.append(index)
            col_idx.append(col)
        if info.affine_flip:
            row_idx.append(index)
            col_idx.append(num_input_observables)
    return util_pb.BitMatrix(rows=rows, cols=cols, i=row_idx, j=col_idx)


def _empty_bit_matrix(rows: int, cols: int) -> util_pb.BitMatrix:
    return util_pb.BitMatrix(rows=rows, cols=cols)


# ---------------------------------------------------------------------------
# Error mechanisms
# ---------------------------------------------------------------------------


def _build_errors(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
    output_ports: list[OutputPort],
    *,
    num_finished: int,
    num_unfinished: int,
    num_readouts: int,
) -> list[jit_pb.JitGadgetType.Error]:
    """Translate ``ERROR(p) ...`` statements into JIT error rows.

    ``ERROR`` is a *footprint declaration*: each statement directly names
    the checks (``C<i>``), readouts (``R<i>``), and output observables
    (``LX<i>``/``LZ<i>``/``LY<i>``) that the mechanism
    flips. We translate names to indices and emit one
    ``JitGadgetType.Error`` per statement. No Pauli propagation happens
    here — propagation of physical noise sources into footprints is the
    responsibility of the ``annotate`` tool.

    Indexing conventions for ``C<i>``: ``i`` indexes the concatenated
    ``[finished_checks, unfinished_checks]`` array of the gadget. We
    split internally into ``finished_checks`` / ``unfinished_checks``.
    """

    output_layout = PortColumnLayout(output_ports, codes)

    error_statements = [
        s for s in flatten_body(list(gadget.body)) if isinstance(s, ErrorStatement)
    ]
    errors_pb: list[jit_pb.JitGadgetType.Error] = []
    for stmt in error_statements:
        errors_pb.append(
            _parse_error(
                stmt,
                gadget_name=gadget.name,
                num_finished=num_finished,
                num_unfinished=num_unfinished,
                num_readouts=num_readouts,
                logical_qubit_columns=output_layout.logical_qubit_columns,
                unfinished_to_column=output_layout.stab_to_column,
                output_ports=output_ports,
                codes=codes,
            )
        )
    return errors_pb


def _parse_error(
    stmt: ErrorStatement,
    *,
    gadget_name: str,
    num_finished: int,
    num_unfinished: int,
    num_readouts: int,
    logical_qubit_columns: list[tuple[int, int]],
    unfinished_to_column: list[int | None],
    output_ports: list[OutputPort],
    codes: dict[str, CodeDefinition],
) -> jit_pb.JitGadgetType.Error:
    """Translate a single ``ErrorStatement`` into a ``JitGadgetType.Error``."""
    if not 0.0 <= stmt.probability <= 1.0:
        raise ValueError(
            f"in GADGET {gadget_name!r}: {_render_error(stmt)}: "
            f"probability {stmt.probability!r} must be in [0, 1]"
        )

    finished_set: set[int] = set()
    unfinished_set: set[int] = set()
    residual_set: set[int] = set()
    readout_set: set[int] = set()
    total_checks = num_finished + num_unfinished

    for target in stmt.targets:
        if isinstance(target, CheckTarget):
            if target.index < 0 or target.index >= total_checks:
                raise ValueError(
                    f"in GADGET {gadget_name!r}: {_render_error(stmt)}: "
                    f"check target {target} is out of range; the gadget has "
                    f"{num_finished} finished + {num_unfinished} unfinished "
                    f"= {total_checks} checks (C0..C{total_checks - 1})"
                )
            if target.index < num_finished:
                _xor_toggle(finished_set, target.index)
            else:
                _xor_toggle(unfinished_set, target.index - num_finished)
            continue
        if isinstance(target, ReadoutTarget):
            if target.index < 0 or target.index >= num_readouts:
                raise ValueError(
                    f"in GADGET {gadget_name!r}: {_render_error(stmt)}: "
                    f"readout target {target} is out of range; the gadget "
                    f"has {num_readouts} readout(s) (R0..R{num_readouts - 1})"
                )
            _xor_toggle(readout_set, target.index)
            continue
        if isinstance(target, LogicalPauliTarget):
            if target.port_kind is not None:
                assert target.port_index is not None
                if target.port_kind != "OUT":
                    raise ValueError(
                        f"in GADGET {gadget_name!r}: {_render_error(stmt)}: "
                        f"observable {target} has direction "
                        f"{target.port_kind!r}, but ERROR observables refer "
                        f"to OUTPUT port logicals only"
                    )
                if not 0 <= target.port_index < len(output_ports):
                    raise ValueError(
                        f"in GADGET {gadget_name!r}: {_render_error(stmt)}: "
                        f"observable {target}: port index out of range "
                        f"(only {len(output_ports)} OUTPUT port(s))"
                    )
                port_code = codes[output_ports[target.port_index].code_name]
                n_logicals = len(port_code.logicals)
                if not 0 <= target.index < n_logicals:
                    raise ValueError(
                        f"in GADGET {gadget_name!r}: {_render_error(stmt)}: "
                        f"observable {target}: logical index out of range "
                        f"(port has {n_logicals} logical observable(s))"
                    )
                global_idx = sum(
                    len(codes[p.code_name].logicals)
                    for p in output_ports[: target.port_index]
                ) + target.index
            else:
                if target.index >= len(logical_qubit_columns):
                    raise ValueError(
                        f"in GADGET {gadget_name!r}: {_render_error(stmt)}: "
                        f"observable {target} is out of range; the gadget has "
                        f"only {len(logical_qubit_columns)} logical qubit(s) "
                        f"across its output ports"
                    )
                global_idx = target.index
            x_col, z_col = logical_qubit_columns[global_idx]
            upper = target.pauli.upper()
            if upper == "X":
                _xor_toggle(residual_set, z_col)
            elif upper == "Z":
                _xor_toggle(residual_set, x_col)
            elif upper == "Y":
                _xor_toggle(residual_set, x_col)
                _xor_toggle(residual_set, z_col)
            else:
                raise ValueError(
                    f"in GADGET {gadget_name!r}: {_render_error(stmt)}: "
                    f"unsupported Pauli {target.pauli!r} in observable "
                    f"target {target}"
                )
            continue
        if isinstance(target, PauliTarget):
            raise ValueError(
                f"in GADGET {gadget_name!r}: {_render_error(stmt)}: "
                f"physical observable {target} is not supported; "
                f"use L{target.pauli}{target.index} for the logical "
                f"observable instead"
            )
        raise ValueError(
            f"in GADGET {gadget_name!r}: {_render_error(stmt)}: "
            f"unsupported target {target!r}; expected C<i>, R<i>, "
            f"LX/LY/LZ<i>, or X/Y/Z<i>"
        )

    # Set stabilizer generator residual columns from unfinished check
    # triggers: an error's stabilizer residual is fully determined by
    # whether it triggers the corresponding unfinished check.
    for uc_idx in unfinished_set:
        col = unfinished_to_column[uc_idx]
        if col is not None:
            _xor_toggle(residual_set, col)

    base = pb.ErrorModelType.Error(
        tag=_render_error(stmt),
        residual=sorted(residual_set),
        readout_flips=sorted(readout_set),
        probability=stmt.probability,
    )
    return jit_pb.JitGadgetType.Error(
        base=base,
        finished_checks=sorted(finished_set),
        unfinished_checks=sorted(unfinished_set),
    )


def _xor_toggle(bag: set[int], index: int) -> None:
    if index in bag:
        bag.remove(index)
    else:
        bag.add(index)


def _render_error(stmt: ErrorStatement) -> str:
    targets = " ".join(str(t) for t in stmt.targets)
    return f"ERROR({stmt.probability}) {targets}".rstrip()
