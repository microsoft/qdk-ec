"""Persistent loss model that emits no Pauli-envelope errors."""

from __future__ import annotations

from deq.transpiler.loss.api import (
    LossAnalysisState,
    LossGate,
    LossGateHandler,
    UnsupportedLossModelError,
)
from deq.transpiler.loss.model_gate_removal import GateRemovalGateHandler


class IgnoreLossGateHandler(GateRemovalGateHandler):
    """Track physical loss lifetimes and heralds without adding errors."""

    def handle_loss_source(
        self, event_id: int, state: LossAnalysisState
    ) -> None:
        del event_id, state

    def handle_h(self, gate: LossGate, state: LossAnalysisState) -> None:
        del gate, state

    def handle_cx(self, gate: LossGate, state: LossAnalysisState) -> None:
        del gate, state

    def handle_m(self, gate: LossGate, state: LossAnalysisState) -> None:
        if len(gate.qubits) != 1 or len(gate.measurement_indices) != 1:
            raise UnsupportedLossModelError(
                f"M at body index {gate.body_index} requires one qubit and result"
            )
        state.record_loss_measurement(gate.qubits[0], gate.measurement_indices[0])

    def handle_native_gate(self, gate: LossGate, state: LossAnalysisState) -> None:
        if gate.name == "SWAP":
            super().handle_native_gate(gate, state)


class IgnoreLossModel:
    """Preserve loss metadata while omitting every Pauli-envelope error."""

    def create_handler(self) -> LossGateHandler:
        """Create independent state for one gadget traversal."""

        return IgnoreLossGateHandler()
