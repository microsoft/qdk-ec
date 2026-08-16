"""Trapped-ion loss model for Mølmer-Sørensen-native hardware."""

from __future__ import annotations

from deq.transpiler.loss.api import GateLossPolicy, LossGateHandler, QdkLossConfig
from deq.transpiler.loss.model_configured import ConfiguredLossGateHandler


_TRAPPED_ION_CONFIG = QdkLossConfig(
    gate_policies=(
        ("cx", GateLossPolicy.RESIDUAL_S_DAGGER),
        ("cy", GateLossPolicy.RESIDUAL_S_DAGGER),
        ("cz", GateLossPolicy.RESIDUAL_S_DAGGER),
        ("swap", GateLossPolicy.SKIP),
    ),
)


class TrappedIonLossGateHandler(ConfiguredLossGateHandler):
    """Retain the local S-dagger fixup when loss removes the MS interaction."""

    def __init__(self, config: QdkLossConfig = _TRAPPED_ION_CONFIG) -> None:
        super().__init__(config)


class TrappedIonLossModel:
    """Trapped-ion controlled gates whose lost interaction leaves S-dagger."""

    config = _TRAPPED_ION_CONFIG

    def create_handler(self) -> LossGateHandler:
        """Create independent state for one gadget traversal."""

        return TrappedIonLossGateHandler(self.config)
