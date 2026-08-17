"""Residual-phase approximation for MS-native trapped-ion hardware."""

from __future__ import annotations

from deq.transpiler.loss.api import (
    GateLossPolicy,
    LossAnalysisState,
    LossGate,
    LossGateHandler,
    QdkLossConfig,
    UnsupportedLossModelError,
)
from deq.transpiler.loss.policies import (
    handle_gate_policy,
    handle_loss_source,
    handle_measurement,
    handle_reset,
    handle_skip,
)

_QDK_TABLE_BY_SOURCE_GATE = {
    "CZ": "cz",
    "SWAP": "swap",
}
_UNSUPPORTED_CONTROLLED_GATES = frozenset({"CX", "CY"})


_TRAPPED_ION_CONFIG = QdkLossConfig(
    gate_policies=(
        ("cz", GateLossPolicy.RESIDUAL_S_DAGGER),
        ("swap", GateLossPolicy.APPLY_ANYWAY),
    ),
)


class TrappedIonLossGateHandler(LossGateHandler):
    """Apply one explicit compiled-CZ residual-phase approximation."""

    native_gates = frozenset(
        {
            *_QDK_TABLE_BY_SOURCE_GATE,
            *_UNSUPPORTED_CONTROLLED_GATES,
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
        if gate.name in _UNSUPPORTED_CONTROLLED_GATES and gate.control_measurement_index is None:
            raise UnsupportedLossModelError(
                "trapped-ion residual-phase model supports CZ only; "
                f"{gate.source_name} requires an explicit device-specific "
                "decomposition or custom loss model"
            )
        qdk_table = _QDK_TABLE_BY_SOURCE_GATE.get(gate.name)
        if qdk_table is not None:
            handle_gate_policy(
                GateLossPolicy(self.config.policy_for(qdk_table)), gate, state
            )
            return
        handle_skip(gate, state)


class TrappedIonLossModel:
    """Effective trapped-ion model for one specified CZ compilation."""

    config = _TRAPPED_ION_CONFIG

    def create_handler(self) -> LossGateHandler:
        """Create independent state for one gadget traversal."""

        return TrappedIonLossGateHandler(self.config)


def create_loss_model() -> TrappedIonLossModel:
    """Create this model when the module is loaded as a plugin file."""

    return TrappedIonLossModel()
