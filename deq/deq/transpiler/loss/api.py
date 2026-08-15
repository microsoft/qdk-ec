"""Public protocols and immutable inputs for physical loss models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class UnsupportedLossModelError(ValueError):
    """Raised when a circuit is outside a loss model's supported scope."""


@dataclass(frozen=True)
class LossGate:
    """One gate occurrence passed to a loss-model handler.

    Multi-target Stim instructions are atomized before dispatch, so ``qubits``
    contains exactly the operands of one gate application. Boundaries are in
    the current loss-analysis operation stream.
    """

    name: str
    source_name: str
    arguments: tuple[float, ...]
    qubits: tuple[int, ...]
    measurement_indices: tuple[int, ...]
    body_index: int
    boundary_before: int
    boundary_after: int
    produces_measurement: bool
    resets_qubits: bool
    is_native: bool


@runtime_checkable
class LossAnalysisState(Protocol):
    """Constrained mutation surface available to individual gate handlers."""

    def active_event_ids(self, qubit: int) -> tuple[int, ...]:
        """Return source-event worlds with an active branch on ``qubit``."""

        ...

    def event_has_active_loss(self, event_id: int, qubit: int) -> bool:
        """Whether ``event_id`` currently loses ``qubit``."""

        ...

    def add_continuation_pauli_insertion(
        self,
        qubit: int,
        boundary: int,
        paulis: tuple[str, ...] = ("I", "X", "Y", "Z"),
    ) -> None:
        """Add an inheritable Pauli set to every active branch on ``qubit``."""

        ...

    def add_event_continuation_pauli_insertion(
        self,
        event_id: int,
        *,
        lost_qubit: int,
        error_qubit: int,
        boundary: int,
        paulis: tuple[str, ...] = ("I", "X", "Y", "Z"),
    ) -> None:
        """Add an inheritable Pauli set in one source-event world."""

        ...

    def add_source_pauli_insertion(
        self,
        event_id: int,
        paulis: tuple[str, ...] = ("I", "X", "Y", "Z"),
    ) -> None:
        """Add a Pauli set at one loss event's source boundary."""

        ...

    def record_loss_measurement(self, qubit: int, measurement_index: int) -> None:
        """Associate a measurement result with every active branch."""

        ...

    def clear_loss(self, qubit: int) -> None:
        """Terminate all active branches occupying ``qubit``."""

        ...

    def propagate_loss(
        self,
        event_id: int,
        *,
        lost_qubit: int,
        new_qubit: int,
        boundary: int,
    ) -> None:
        """Extend one active loss-event world onto another qubit."""

        ...

    def swap_losses(
        self, event_id: int, first_qubit: int, second_qubit: int, boundary: int
    ) -> None:
        """Swap active loss flags by ending and recreating event branches."""

        ...

@runtime_checkable
class LossGateHandler(Protocol):
    """Stateful per-gadget handler that receives one gate at a time."""

    native_gate_names: frozenset[str]

    def handle_loss_source(
        self, event_id: int, state: LossAnalysisState
    ) -> None:
        """Handle a newly created physical loss event."""

        ...

    def handle_h(self, gate: LossGate, state: LossAnalysisState) -> None:
        """Handle one primitive Hadamard occurrence."""

        ...

    def handle_s(self, gate: LossGate, state: LossAnalysisState) -> None:
        """Handle one primitive square-root-of-Z occurrence."""

        ...

    def handle_cx(self, gate: LossGate, state: LossAnalysisState) -> None:
        """Handle one primitive controlled-X occurrence."""

        ...

    def handle_m(self, gate: LossGate, state: LossAnalysisState) -> None:
        """Handle one primitive Z-measurement occurrence."""

        ...

    def handle_r(self, gate: LossGate, state: LossAnalysisState) -> None:
        """Handle one primitive Z-reset occurrence."""

        ...

    def handle_native_gate(self, gate: LossGate, state: LossAnalysisState) -> None:
        """Handle an opted-in source gate without Stim decomposition."""

        ...


@runtime_checkable
class LossModel(Protocol):
    """Configured physical loss model shared across gadget analyses."""

    def create_handler(self) -> LossGateHandler:
        """Create fresh mutable handler state for one gadget traversal."""

        ...
