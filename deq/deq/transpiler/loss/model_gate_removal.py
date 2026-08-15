"""Built-in persistent gate-removal loss model."""

from __future__ import annotations

from deq.transpiler.loss.api import (
    LossAnalysisState,
    LossGate,
    LossGateHandler,
    UnsupportedLossModelError,
)


class GateRemovalGateHandler(LossGateHandler):
    """Per-gadget handler for persistent loss and removed gates."""

    native_gate_names = frozenset({"CZ", "S", "SQRT_X", "SQRT_X_DAG", "SWAP"})

    def handle_loss_source(
        self, event_id: int, state: LossAnalysisState
    ) -> None:
        state.add_source_pauli_insertion(event_id)

    def handle_h(self, gate: LossGate, state: LossAnalysisState) -> None:
        for qubit in gate.qubits:
            state.add_continuation_pauli_insertion(qubit, gate.boundary_after)

    def handle_s(self, gate: LossGate, state: LossAnalysisState) -> None:
        del gate, state

    def handle_cx(self, gate: LossGate, state: LossAnalysisState) -> None:
        control, target = gate.qubits  # splitted by ./analysis.py
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

    def handle_m(self, gate: LossGate, state: LossAnalysisState) -> None:
        if len(gate.qubits) != 1 or len(gate.measurement_indices) != 1:
            raise UnsupportedLossModelError(
                f"M at body index {gate.body_index} requires one qubit and result"
            )
        qubit = gate.qubits[0]
        state.add_continuation_pauli_insertion(qubit, gate.boundary_before)
        state.record_loss_measurement(qubit, gate.measurement_indices[0])

    def handle_r(self, gate: LossGate, state: LossAnalysisState) -> None:
        for qubit in gate.qubits:
            state.clear_loss(qubit)

    def handle_native_gate(self, gate: LossGate, state: LossAnalysisState) -> None:
        assert gate.name in self.native_gate_names  # guaranteed by LossGateHandler
        if gate.name == "SWAP":
            # A SWAP is a physical atom relabelling in neutral-atom hardware: it
            # carries a lost site (and its Pauli envelope) to the partner site
            # without injecting any new error. Move each active loss to the
            # partner qubit; no continuation Pauli insertion is added.
            first, second = gate.qubits
            event_ids = set(state.active_event_ids(first)) | set(
                state.active_event_ids(second)
            )
            for event_id in event_ids:
                state.swap_losses(event_id, first, second, gate.boundary_after)
            return
        if gate.name in {"SQRT_X", "SQRT_X_DAG"}:
            for qubit in gate.qubits:
                state.add_continuation_pauli_insertion(qubit, gate.boundary_after)


class GateRemovalLossModel:
    """Persistent loss with removal of gates touching a lost operand."""

    def create_handler(self) -> LossGateHandler:
        """Create independent state for one gadget traversal."""

        return GateRemovalGateHandler()