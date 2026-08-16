"""Residual-phase approximation for MS-native trapped-ion hardware."""

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


_TRAPPED_ION_CONFIG = QdkLossConfig(
    gate_policies=(
        ("cx", GateLossPolicy.RESIDUAL_S_DAGGER),
        ("cy", GateLossPolicy.RESIDUAL_S_DAGGER),
        ("cz", GateLossPolicy.RESIDUAL_S_DAGGER),
        ("swap", GateLossPolicy.APPLY_ANYWAY),
    ),
)


class TrappedIonLossGateHandler(LossGateHandler):
    """Apply QDK's residual-S-dagger policy to controlled gates."""

    source_gate_names = frozenset(
        {
            *_QDK_TABLE_BY_SOURCE_GATE,
            "S",
            "SQRT_X",
            "SQRT_X_DAG",
        }
    )

    def __init__(self, config: QdkLossConfig = _TRAPPED_ION_CONFIG) -> None:
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


class TrappedIonLossModel:
    """Effective trapped-ion model with residual phase on surviving operands."""

    config = _TRAPPED_ION_CONFIG

    def create_handler(self) -> LossGateHandler:
        """Create independent state for one gadget traversal."""

        return TrappedIonLossGateHandler(self.config)
