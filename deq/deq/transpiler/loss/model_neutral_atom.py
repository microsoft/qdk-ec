"""Neutral-atom model for native CZ and its compiled CX/CY aliases.

QDK lowers CX and CY to local gates around CZ. When an operand is absent,
skipping CZ leaves cancelling local wrappers, so all three source gates have the
same effective loss behavior.
"""

from __future__ import annotations

from deq.transpiler.loss.api import (
    GateLossPolicy,
    LossAnalysisState,
    LossGate,
    LossGateHandler,
    QdkLossConfig,
)
from deq.transpiler.loss.policies import (
    handle_gate_policy,
    handle_loss_source,
    handle_measurement,
    handle_reset,
    handle_skip,
)

_QDK_TABLE_BY_SOURCE_GATE = {
    "CX": "cx",
    "CY": "cy",
    "CZ": "cz",
    "SWAP": "swap",
}

_NEUTRAL_ATOM_CONFIG = QdkLossConfig(
    gate_policies=(
        ("cx", GateLossPolicy.SKIP),
        ("cy", GateLossPolicy.SKIP),
        ("cz", GateLossPolicy.SKIP),
        ("swap", GateLossPolicy.APPLY_ANYWAY),
    ),
)


class NeutralAtomLossGateHandler(LossGateHandler):
    """Use skipped lost-operand gates with physical SWAP relocation."""

    native_gates = frozenset(
        {
            *_QDK_TABLE_BY_SOURCE_GATE,
            "S",
            "SQRT_X",
            "SQRT_X_DAG",
        }
    )

    def __init__(self, config: QdkLossConfig = _NEUTRAL_ATOM_CONFIG) -> None:
        self.config = config

    def handle_loss_source(self, event_id: int, state: LossAnalysisState) -> None:
        handle_loss_source(event_id, state)

    def handle_gate(self, gate: LossGate, state: LossAnalysisState) -> None:
        if gate.name == "M":
            handle_measurement(gate, state)
            return
        if gate.name == "R":
            handle_reset(gate, state)
            return
        qdk_table = _QDK_TABLE_BY_SOURCE_GATE.get(gate.name)
        if qdk_table is not None:
            handle_gate_policy(
                GateLossPolicy(self.config.policy_for(qdk_table)), gate, state
            )
            return
        handle_skip(gate, state)


class NeutralAtomLossModel:
    """Neutral-atom platform model: SKIP gates and relocate atoms on SWAP."""

    config = _NEUTRAL_ATOM_CONFIG

    def create_handler(self) -> LossGateHandler:
        """Create independent state for one gadget traversal."""

        return NeutralAtomLossGateHandler(self.config)


def create_loss_model() -> NeutralAtomLossModel:
    """Create this model when the module is loaded as a plugin file."""

    return NeutralAtomLossModel()
