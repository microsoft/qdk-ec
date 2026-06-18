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
    QubitTarget,
    ReadoutStatement,
    ReadoutTarget,
)
from deq.transpiler.compose_builder import build_compose_jit_gadget_type
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
    compute_correction_propagation,
    compute_implicit_readout_propagation,
    compute_noise_errors,
    compute_physical_correction,
    resolve_propagations,
)
import stim

from deq.transpiler.code_validation import validate_code
from deq.transpiler.stim_constants import qubit_indices as _qubit_indices
from deq.transpiler.stim_constants import mpp_measurement_count, split_mpp_targets


def _measurement_tags_of(inst: Instruction) -> list[str]:
    """Return one human-readable tag per measurement produced by *inst*."""
    name = inst.name.upper()
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
    codes: list[CodeDefinition] = [
        d for d in qfile.definitions if isinstance(d, CodeDefinition)
    ]
    gadgets: list[GadgetDefinition] = [
        d for d in qfile.definitions if isinstance(d, GadgetDefinition)
    ]
    composes: list[ComposeDefinition] = [
        d for d in qfile.definitions if isinstance(d, ComposeDefinition)
    ]
    code_by_name = {c.name: c for c in codes}
    gadget_by_name = {g.name: g for g in gadgets}

    # Validate code definitions early.
    for c in codes:
        validate_code(c)

    # Warn on unrecognized decorators early.
    for c in codes:
        warn_unrecognized_decorators(c)
    for g in gadgets:
        warn_unrecognized_decorators(g)
    for comp in composes:
        warn_unrecognized_decorators(comp)

    ptype_of_code = _assign_ids(codes, "PTYPE")
    # GADGETs and COMPOSEs share the gtype namespace.  @GTYPE(n) pins
    # are honoured for both; unpinned definitions auto-assign from 1.
    all_gtypes = _assign_ids(list(gadgets) + list(composes), "GTYPE")
    gadget_names = {g.name for g in gadgets}
    gtype_of_gadget = {n: t for n, t in all_gtypes.items() if n in gadget_names}
    gtype_of_compose = {n: t for n, t in all_gtypes.items() if n not in gadget_names}

    port_types = [
        _build_jit_port_type(code, ptype_of_code[code.name]) for code in codes
    ]

    if jobs > 1 and len(gadgets) > 1:
        gadget_types = _build_gadget_types_parallel(
            gadgets, gtype_of_gadget, ptype_of_code, code_by_name, jobs
        )
    else:
        gadget_types = [
            _build_jit_gadget_type(
                g, gtype_of_gadget[g.name], ptype_of_code, code_by_name
            )
            for g in gadgets
        ]

    # Process COMPOSE definitions in source order. Each one becomes a new
    # JitGadgetType visible to subsequent COMPOSEs (so nested COMPOSE works
    # automatically as long as the inner one is declared first).
    jit_by_name: dict[str, jit_pb.JitGadgetType] = {
        g.name: jt for g, jt in zip(gadgets, gadget_types)
    }
    compose_so_far: dict[str, ComposeDefinition] = {}
    for compose in composes:
        composed_jit = build_compose_jit_gadget_type(
            compose,
            gtype=gtype_of_compose[compose.name],
            gadget_definitions=gadget_by_name,
            compose_definitions=compose_so_far,
            jit_gadget_types_by_name=jit_by_name,
            codes=code_by_name,
            ptype_of_code=ptype_of_code,
            port_types=port_types,
        )
        gadget_types.append(composed_jit)
        jit_by_name[compose.name] = composed_jit
        compose_so_far[compose.name] = compose

    return jit_pb.JitLibrary(
        port_types=sorted(port_types, key=lambda p: p.base.ptype),
        gadget_types=sorted(gadget_types, key=lambda g: g.base.gtype),
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
        gadget, num_output_observables, len(readouts_pb), output_ports, codes
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
    """Build the ``logical_correction`` matrix from CONDITIONAL statements.

    Each ``CONDITIONAL R<j> L<P><i>`` applies a logical Pauli correction
    conditioned on readout *j*.  The correction flips all anti-commuting
    output observables.

    In the **logical** frame (2 observables per logical qubit, interleaved
    X then Z), ``LX<i>`` flips ``LZ<i>`` and vice versa.

    In the **physical** frame (2 observables per physical qubit), the
    logical operator is expanded into its physical Pauli string and each
    physical qubit's anti-commuting observable is flipped individually.

    Multiple CONDITIONAL statements XOR into the matrix.
    """
    entries: set[tuple[int, int]] = set()

    for stmt in flatten_body(list(gadget.body)):
        if not isinstance(stmt, ConditionalStatement):
            continue
        if not isinstance(stmt.condition, ReadoutTarget):
            continue
        readout_col = stmt.condition.index
        if readout_col >= num_readouts:
            raise ValueError(
                f"CONDITIONAL in gadget {gadget.name!r}: readout index "
                f"R{readout_col} out of range (only {num_readouts} readouts "
                f"declared)"
            )

        num_logicals = sum(len(codes[p.code_name].logicals) for p in output_ports)

        for target in stmt.targets:
            if target.port_kind is None and target.index >= num_logicals:
                raise ValueError(
                    f"CONDITIONAL in gadget {gadget.name!r}: logical index "
                    f"L{target.pauli}{target.index} out of range (only "
                    f"{num_logicals} output logical qubits)"
                )
            flipped = conditional_flipped_rows(target, output_ports, codes)
            for row in flipped:
                entries.symmetric_difference_update({(row, readout_col)})

    sorted_entries = sorted(entries)
    rows_list = [r for r, _ in sorted_entries]
    cols_list = [c for _, c in sorted_entries]
    return util_pb.BitMatrix(
        rows=num_output_observables,
        cols=num_readouts,
        i=rows_list,
        j=cols_list,
    )


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
    virtual or output-virtual stabilizer measurements. Each target
    (``rec[-k]``, ``M<i>``, ``IN<p>.S<s>``, or ``OUT<p>.S<s>``) is
    resolved to a global measurement index, validated to lie in the
    internal/physical region, and translated to a real-only index.

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

    Accepts any of the four measurement-reference forms (``rec[-k]``,
    ``M<i>``, ``IN<p>.S<s>``, ``OUT<p>.S<s>``). Raises ``ValueError`` if
    the resolved target references a virtual stabilizer measurement.
    """
    measurement_indices: list[int] = []
    affine_flip = stmt.flip

    for target in stmt.targets:
        if isinstance(
            target,
            (
                MeasurementRecordTarget,
                PhysicalMeasurementTarget,
                InputVirtualTarget,
                OutputVirtualTarget,
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
        raise ValueError(
            f"in GADGET {gadget_name!r}: {_render_readout(stmt)}: "
            f"unsupported target {target!r}; only measurement references "
            f"(rec[-k], M<i>, IN<p>.S<s>, OUT<p>.S<s>) or FLIP are "
            f"supported in READOUT statements"
        )

    return _ReadoutInfo(
        measurement_indices=sorted(_xor_deduplicate(measurement_indices)),
        affine_flip=affine_flip,
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
    """Translate a measurement-reference target to a real-measurement index.

    The target may be any of the four forms (``rec[-k]``, ``M<i>``,
    ``IN<p>.S<s>``, ``OUT<p>.S<s>``).  Virtual targets are rejected.
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
    implicit_columns: list[set[int]] | None = None,
) -> util_pb.BitMatrix:
    rows = len(readouts_info)
    cols = num_input_observables + 1
    row_idx: list[int] = []
    col_idx: list[int] = []
    for index, info in enumerate(readouts_info):
        if implicit_columns is not None:
            for col in sorted(implicit_columns[index]):
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
