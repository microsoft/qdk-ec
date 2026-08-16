"""Trapped-ion loss model for Mølmer-Sørensen-native hardware."""

from __future__ import annotations

from deq.transpiler.loss.api import GateLossPolicy, LossGateHandler, QdkLossConfig
from deq.transpiler.loss.model_configured import ConfiguredLossGateHandler


_TRAPPED_ION_QDK_CONFIG = QdkLossConfig(
    gate_policies=(
        ("cx", GateLossPolicy.RESIDUAL_S_DAGGER),
        ("cy", GateLossPolicy.RESIDUAL_S_DAGGER),
        ("cz", GateLossPolicy.RESIDUAL_S_DAGGER),
        ("swap", GateLossPolicy.SKIP),
    ),
)


class TrappedIonLossGateHandler(ConfiguredLossGateHandler):
    """Retain the local S-dagger fixup when loss removes the MS interaction."""

    def __init__(self) -> None:
        super().__init__(_TRAPPED_ION_QDK_CONFIG)


class TrappedIonLossModel:
    """CX-native trapped ions whose lost-operand interaction leaves S-dagger."""

    name = "trapped-ion"
    source_config = _TRAPPED_ION_QDK_CONFIG
    qdk_config = _TRAPPED_ION_QDK_CONFIG

    def create_handler(self) -> LossGateHandler:
        """Create independent state for one gadget traversal."""

        return TrappedIonLossGateHandler()
