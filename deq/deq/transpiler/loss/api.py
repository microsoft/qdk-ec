"""Public protocols and immutable inputs for physical loss models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class UnsupportedLossModelError(ValueError):
    """Raised when a circuit is outside a loss model's supported scope."""


class GateLossPolicy(StrEnum):
    """QDK behavior for a gate with at least one lost operand."""

    SKIP = "SKIP"
    PROPAGATE = "PROPAGATE"
    DEGRADE = "DEGRADE"
    RESIDUAL_S_DAGGER = "RESIDUAL_S_DAGGER"
    APPLY_ANYWAY = "APPLY_ANYWAY"


@dataclass(frozen=True)
class QdkLossConfig:
    """JSON-safe QDK policy overrides keyed by gate name."""

    gate_policies: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        normalized: list[tuple[str, str]] = []
        seen: set[str] = set()
        for raw_gate, raw_policy in self.gate_policies:
            gate = raw_gate.lower()
            policy = str(raw_policy)
            if gate in seen:
                raise ValueError(f"duplicate QDK loss-policy override for {gate!r}")
            seen.add(gate)
            normalized.append((gate, policy))
        object.__setattr__(self, "gate_policies", tuple(sorted(normalized)))

    def policy_for(self, gate: str) -> str:
        """Return the configured policy for a QDK multi-qubit gate."""

        normalized = gate.lower()
        try:
            return dict(self.gate_policies)[normalized]
        except KeyError:
            raise ValueError(
                f"no QDK loss policy configured for gate {normalized!r}"
            ) from None

    def to_json_object(self) -> dict[str, str]:
        """Return the canonical JSON-compatible representation."""

        return dict(self.gate_policies)

    def to_json(self) -> str:
        """Serialize this configuration deterministically."""

        return json.dumps(self.to_json_object(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> QdkLossConfig:
        """Parse and validate a serialized configuration."""

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid QDK loss config JSON: {error}") from error
        return cls.from_json_object(parsed)

    @classmethod
    def from_json_object(cls, value: object) -> QdkLossConfig:
        """Parse and validate a JSON-compatible representation."""

        if not isinstance(value, dict):
            raise ValueError("QDK loss config must be a JSON object")
        return cls(
            gate_policies=tuple(
                (str(gate), str(policy)) for gate, policy in value.items()
            ),
        )


@dataclass(frozen=True)
class LossGate:
    """One gate occurrence passed to a loss-model handler.

    Multi-target Stim instructions are atomized before dispatch, so ``qubits``
    contains exactly the physical operands of one gate application. A
    measurement-record control is instead resolved to an absolute gadget index
    in ``control_measurement_index``. ``measurement_index`` is the absolute
    index produced by a measurement gate. Each is ``None`` when absent.
    Boundaries are in the current loss-analysis operation stream.
    """

    name: str
    source_name: str
    arguments: tuple[float, ...]
    qubits: tuple[int, ...]
    measurement_index: int | None
    control_measurement_index: int | None
    body_index: int
    boundary_before: int
    boundary_after: int
    produces_measurement: bool
    resets_qubits: bool
    is_source_gate: bool


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
        generators: tuple[str, ...] = ("X", "Z"),
    ) -> None:
        """Add Pauli generators to every active branch on ``qubit``."""

        ...

    def add_event_continuation_pauli_insertion(
        self,
        event_id: int,
        *,
        branch_qubit: int,
        qubit: int,
        boundary: int,
        generators: tuple[str, ...] = ("X", "Z"),
    ) -> None:
        """Add generators on ``qubit`` in one event's ``branch_qubit`` branch."""

        ...

    def add_source_pauli_insertion(
        self,
        event_id: int,
        generators: tuple[str, ...] = ("X", "Z"),
    ) -> None:
        """Add Pauli generators at one loss event's source boundary."""

        ...

    def add_loss_controlled_pauli_insertion(
        self,
        measurement_index: int,
        qubit: int,
        boundary: int,
        generators: tuple[str, ...],
    ) -> None:
        """Add generators when ``measurement_index`` is a loss herald."""

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

    native_gates: frozenset[str]

    def handle_loss_source(self, event_id: int, state: LossAnalysisState) -> None:
        """Handle a newly created physical loss event."""

        ...

    def handle_gate(self, gate: LossGate, state: LossAnalysisState) -> None:
        """Handle a source-level or decomposed primitive gate."""

        ...


@runtime_checkable
class LossModel(Protocol):
    """Configured physical loss model shared across gadget analyses."""

    config: QdkLossConfig

    def create_handler(self) -> LossGateHandler:
        """Create fresh mutable handler state for one gadget traversal."""

        ...
