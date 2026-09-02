"""Transpile ``LOSS_ERROR`` instructions into decoder loss metadata.

This reuses the noise-error frame propagator: each loss-induced Pauli generator
is projected exactly like a ``DEPOLARIZE1``/``X_ERROR`` fault, so the gadget's
checks, readouts, and output observables are inferred by the existing tools.
The resulting footprints become probability-0 ``errors`` (activated by the loss
herald at runtime) and each declared loss links to the generator indices.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import stim

import deq.proto.deq_bin_pb2 as bin_pb
import deq.proto.deq_jit_pb2 as jit_pb
import deq.proto.util_pb2 as util_pb
from deq.circuit.model import CodeDefinition, GadgetDefinition, OutputPort
from deq.transpiler.fault_propagation import (
    build_decomposed_body,
    build_error_projection_context,
    build_error_row_from_flips,
    build_port_paulis,
    propagate_pauli_mechanisms,
)
from deq.transpiler.jit_transpiler import (
    flatten_body,
    max_qubit_index,
)
from deq.transpiler.loss.analysis import analyze_loss_events
from deq.transpiler.loss.api import LossModel
from deq.transpiler.loss.syntax import PhysicalPortLayout
from deq.transpiler.stim_constants import single_pauli_to_stim


@dataclass(frozen=True)
class LossModelArtifacts:
    """Runtime loss metadata and newly projected errors for one gadget."""

    model: bin_pb.GadgetType.LossModel | None
    added_errors: tuple[jit_pb.JitGadgetType.Error, ...] = ()


def _error_footprint(error_row: jit_pb.JitGadgetType.Error) -> tuple:
    """Return the deduplication key (footprint) of an error row.

    Two errors with the same footprint flip the same checks, readouts, and
    output residual, so a loss generator may reuse either interchangeably.
    """
    return (
        tuple(error_row.finished_checks),
        tuple(error_row.unfinished_checks),
        tuple(error_row.base.residual),
        tuple(error_row.base.readout_flips),
    )


def transpile_inferred_loss_model(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
    *,
    output_ports: list[OutputPort],
    input_ports,
    input_virtual_count: int,
    finished_checks,
    unfinished_checks,
    ov_start: int,
    readouts: Sequence[bin_pb.GadgetType.Readout],
    physical_correction: util_pb.BitMatrix,
    existing_errors: Sequence[jit_pb.JitGadgetType.Error],
    loss_model: LossModel,
) -> LossModelArtifacts:
    """Analyze a gadget and transpile its inferred loss metadata.

    The library builder calls this only when the library contains loss and this
    gadget has no explicit ``LOSS`` model. A gadget without a local
    ``LOSS_ERROR`` may still produce ``input_losses`` for loss entering through
    its input ports. The result has no model only when neither a local nor an
    entering loss event can exist.

    A loss generator whose footprint already matches one of ``existing_errors``
    reuses that error's index instead of appending a duplicate; only genuinely
    new footprints are returned (to be appended after ``existing_errors``).
    """
    analysis = analyze_loss_events(gadget, loss_model)
    table = analysis.graph
    exit_qubits_by_event = analysis.exit_qubits_by_event
    input_event_id_by_qubit = analysis.input_event_id_by_qubit
    if not table.events:
        return LossModelArtifacts(model=None)
    input_layout = PhysicalPortLayout(input_ports, codes)
    output_layout = PhysicalPortLayout(output_ports, codes)
    input_event_ids = frozenset(input_event_id_by_qubit.values())
    fresh_events = [
        event for event in table.events if event.event_id not in input_event_ids
    ]
    fresh_position = {event.event_id: index for index, event in enumerate(fresh_events)}

    body_flat = flatten_body(list(gadget.body))
    num_qubits = max(max_qubit_index(list(gadget.body)) + 1, 0)
    decomposed = build_decomposed_body(body_flat)
    output_stabilizer_paulis, frame_column_paulis = build_port_paulis(
        output_ports, codes, num_qubits
    )
    context = build_error_projection_context(
        output_ports=output_ports,
        codes=codes,
        input_virtual_count=input_virtual_count,
        finished_checks=finished_checks,
        unfinished_checks=unfinished_checks,
        output_virtual_start=ov_start,
        readouts=readouts,
        physical_correction=physical_correction,
    )

    # Collect one projection mechanism per (event, kind, generator).
    specs: list[tuple[int, str, str]] = []
    mechanisms: list[tuple[int, stim.PauliString]] = []
    for event in table.events:
        for kind, insertions in (
            ("source", event.source_pauli_insertions),
            ("continuation", event.continuation_pauli_insertions),
        ):
            for insertion in insertions:
                for generator in insertion.generators:
                    specs.append((event.event_id, kind, f"loss{event.event_id}"))
                    mechanisms.append(
                        (
                            insertion.boundary,
                            single_pauli_to_stim(
                                generator, insertion.qubit, num_qubits
                            ),
                        )
                    )

    flips = propagate_pauli_mechanisms(
        mechanisms,
        decomposed,
        num_qubits,
        output_stabilizer_paulis,
        frame_column_paulis,
    )

    errors: list[jit_pb.JitGadgetType.Error] = []
    existing_error_count = len(existing_errors)
    # Seed the footprint map with the caller's regular errors so a loss
    # generator that matches one reuses its index instead of duplicating it.
    # First occurrence wins, matching a linear scan over ``existing_errors``.
    index_by_footprint: dict[tuple, int] = {}
    for existing_index, existing_error in enumerate(existing_errors):
        index_by_footprint.setdefault(_error_footprint(existing_error), existing_index)
    source_error_indices: dict[int, set[int]] = defaultdict(set)
    continuation_error_indices: dict[int, set[int]] = defaultdict(set)
    for (
        (event_id, kind, site_name),
        (
            _walk_start,
            pauli,
        ),
        flip,
    ) in zip(specs, mechanisms, flips):
        error_row = build_error_row_from_flips(
            site_name=site_name,
            site_pauli=pauli,
            probability=0.0,
            flips=flip,
            context=context,
        )
        if error_row is None:
            continue
        footprint = _error_footprint(error_row)
        error_index = index_by_footprint.get(footprint)
        if error_index is None:
            error_index = existing_error_count + len(errors)
            index_by_footprint[footprint] = error_index
            errors.append(error_row)
        # An entering loss (input_losses) has no source in this gadget: all of its
        # generators are continuation effects, so route its "source" insertions to
        # the continuation set too.
        is_continuation = kind != "source" or event_id in input_event_ids
        target = continuation_error_indices if is_continuation else source_error_indices
        target[event_id].add(error_index)

    # Flat output-port position of each output physical qubit, so a loss that is
    # still active on an output qubit at the gadget end can be recorded as
    # leaving on that port position (``child_output_qubits``).
    output_qubit_to_slot = output_layout.qubit_slots()

    def _exit_slots(event_id: int) -> list[int]:
        return sorted(
            output_qubit_to_slot[qubit]
            for qubit in exit_qubits_by_event.get(event_id, ())
            if qubit in output_qubit_to_slot
        )

    # Fresh (in-gadget) losses become ``losses``; entering losses become
    # ``input_losses``. ``child_losses`` index into the fresh ``losses`` list, so
    # build the fresh-position map first and remap successors through it.
    events_by_id = {event.event_id: event for event in table.events}
    successors_by_event: dict[int, tuple[int, ...]] = {
        event.event_id: table.successor_event_ids[index]
        for index, event in enumerate(table.events)
    }

    def _child_losses(event_id: int) -> list[int]:
        return sorted(
            fresh_position[successor]
            for successor in successors_by_event[event_id]
            if successor in fresh_position
        )

    losses_pb: list[bin_pb.GadgetType.LossModel.Loss] = []
    for event in fresh_events:
        losses_pb.append(
            bin_pb.GadgetType.LossModel.Loss(
                probability=event.loss_probability,
                source_errors=sorted(source_error_indices[event.event_id]),
                continuation_errors=sorted(continuation_error_indices[event.event_id]),
                child_losses=_child_losses(event.event_id),
                child_output_qubits=_exit_slots(event.event_id),
                loss_measurements=sorted(event.loss_measurements),
            )
        )

    # Flat input-port slot of each input physical qubit; entering-loss generators
    # are all continuation (the loss did not originate in this gadget).
    input_qubit_to_slot = input_layout.qubit_slots()
    input_losses = [
        bin_pb.GadgetType.LossModel.InputLoss() for _ in range(input_layout.size)
    ]
    for qubit, event_id in input_event_id_by_qubit.items():
        input_losses[input_qubit_to_slot[qubit]] = (
            bin_pb.GadgetType.LossModel.InputLoss(
                continuation_errors=sorted(continuation_error_indices[event_id]),
                child_losses=_child_losses(event_id),
                child_output_qubits=_exit_slots(event_id),
                loss_measurements=sorted(events_by_id[event_id].loss_measurements),
            )
        )
    loss_model = bin_pb.GadgetType.LossModel(
        losses=losses_pb, input_losses=input_losses
    )
    return LossModelArtifacts(
        model=loss_model,
        added_errors=tuple(errors),
    )
