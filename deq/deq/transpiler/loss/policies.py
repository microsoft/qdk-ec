"""Reusable gate-level mechanics for physical loss models."""

from __future__ import annotations

from deq.transpiler.loss.api import (
    GateLossPolicy,
    LossAnalysisState,
    LossGate,
    UnsupportedLossModelError,
)


def _has_lost_operand(gate: LossGate, state: LossAnalysisState) -> bool:
    return any(state.active_event_ids(qubit) for qubit in gate.qubits)


def handle_loss_source(event_id: int, state: LossAnalysisState) -> None:
    """Model the reset accompanying physical loss as a full Pauli envelope."""

    state.add_source_pauli_insertion(event_id)


def handle_measurement(gate: LossGate, state: LossAnalysisState) -> None:
    """Record a loss herald and the unresolved value of a lost measurement."""

    if len(gate.qubits) != 1 or len(gate.measurement_indices) != 1:
        raise UnsupportedLossModelError(
            f"M at body index {gate.body_index} requires one qubit and result"
        )
    qubit = gate.qubits[0]
    state.add_continuation_pauli_insertion(qubit, gate.boundary_before)
    state.record_loss_measurement(qubit, gate.measurement_indices[0])


def handle_reset(gate: LossGate, state: LossAnalysisState) -> None:
    """Reload every reset qubit, terminating its active loss branches."""

    for qubit in gate.qubits:
        state.clear_loss(qubit)


def handle_skip(gate: LossGate, state: LossAnalysisState) -> None:
    """Apply the exact SKIP envelope rules supported by the current loss IR."""

    if not _has_lost_operand(gate, state):
        return
    if gate.name == "H":
        for qubit in gate.qubits:
            state.add_continuation_pauli_insertion(qubit, gate.boundary_after)
        return
    if gate.name in {"S", "CZ"}:
        return
    if gate.name in {"SQRT_X", "SQRT_X_DAG"}:
        for qubit in gate.qubits:
            state.add_continuation_pauli_insertion(qubit, gate.boundary_after)
        return
    if gate.name == "CX":
        control, target = gate.qubits
        for event_id in state.active_event_ids(target):
            state.add_event_continuation_pauli_insertion(
                event_id,
                lost_qubit=target,
                error_qubit=target,
                boundary=gate.boundary_after,
            )
            if not state.event_has_active_loss(event_id, control):
                state.add_event_continuation_pauli_insertion(
                    event_id,
                    lost_qubit=target,
                    error_qubit=control,
                    boundary=gate.boundary_after,
                    paulis=("I", "X"),
                )
        return
    raise UnsupportedLossModelError(
        f"exact SKIP envelope for gate {gate.source_name} is not implemented"
    )


def handle_propagate(gate: LossGate, state: LossAnalysisState) -> None:
    """Propagate each event with a lost operand to every gate operand."""

    active_sources: dict[int, int] = {}
    for qubit in gate.qubits:
        for event_id in state.active_event_ids(qubit):
            active_sources.setdefault(event_id, qubit)
    for event_id, lost_qubit in active_sources.items():
        for new_qubit in gate.qubits:
            state.propagate_loss(
                event_id,
                lost_qubit=lost_qubit,
                new_qubit=new_qubit,
                boundary=gate.boundary_after,
            )


def handle_apply_anyway_swap(gate: LossGate, state: LossAnalysisState) -> None:
    """Apply a physical SWAP by relocating each event's loss flag."""

    if gate.name != "SWAP":
        raise UnsupportedLossModelError(
            f"QDK APPLY_ANYWAY is supported only for SWAP, not {gate.source_name}"
        )
    first, second = gate.qubits
    event_ids = set(state.active_event_ids(first)) | set(state.active_event_ids(second))
    for event_id in event_ids:
        state.swap_losses(event_id, first, second, gate.boundary_after)


def handle_degrade(gate: LossGate, state: LossAnalysisState) -> None:
    """Reject DEGRADE until conditional survivor unitaries are represented exactly."""

    if not _has_lost_operand(gate, state):
        return
    raise UnsupportedLossModelError(
        f"exact DEGRADE envelope for gate {gate.source_name} is not implemented"
    )


def handle_residual_s_dagger(gate: LossGate, state: LossAnalysisState) -> None:
    """Add the ``{I, Z}`` Pauli envelope of S-dagger to each survivor."""

    if not _has_lost_operand(gate, state):
        return
    active_sources: dict[int, int] = {}
    for qubit in gate.qubits:
        for event_id in state.active_event_ids(qubit):
            active_sources.setdefault(event_id, qubit)
    for event_id, lost_qubit in active_sources.items():
        for error_qubit in gate.qubits:
            if not state.event_has_active_loss(event_id, error_qubit):
                state.add_event_continuation_pauli_insertion(
                    event_id,
                    lost_qubit=lost_qubit,
                    error_qubit=error_qubit,
                    boundary=gate.boundary_after,
                    paulis=("I", "Z"),
                )


def handle_gate_policy(
    policy: GateLossPolicy, gate: LossGate, state: LossAnalysisState
) -> None:
    """Apply one QDK gate policy through the exact loss-analysis helpers."""

    handlers = {
        GateLossPolicy.SKIP: handle_skip,
        GateLossPolicy.PROPAGATE: handle_propagate,
        GateLossPolicy.DEGRADE: handle_degrade,
        GateLossPolicy.RESIDUAL_S_DAGGER: handle_residual_s_dagger,
        GateLossPolicy.APPLY_ANYWAY: handle_apply_anyway_swap,
    }
    handlers[policy](gate, state)
