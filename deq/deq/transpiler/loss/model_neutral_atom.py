"""Neutral-atom loss model matching the QDK simulator configuration."""

from __future__ import annotations

from deq.transpiler.loss.api import (
    GateLossPolicy,
    LossGateHandler,
    QdkLossConfig,
)
from deq.transpiler.loss.model_configured import ConfiguredLossGateHandler

_NEUTRAL_ATOM_CONFIG = QdkLossConfig(
    gate_policies=(
        ("cx", GateLossPolicy.SKIP),
        ("cy", GateLossPolicy.SKIP),
        ("cz", GateLossPolicy.SKIP),
        ("swap", GateLossPolicy.APPLY_ANYWAY),
    ),
)


class NeutralAtomLossGateHandler(ConfiguredLossGateHandler):
    """Use skipped lost-operand gates with physical SWAP relocation."""

    def __init__(self, config: QdkLossConfig = _NEUTRAL_ATOM_CONFIG) -> None:
        super().__init__(config)


class NeutralAtomLossModel:
    """Neutral-atom platform model: SKIP gates and relocate atoms on SWAP."""

    config = _NEUTRAL_ATOM_CONFIG

    def create_handler(self) -> LossGateHandler:
        """Create independent state for one gadget traversal."""

        return NeutralAtomLossGateHandler(self.config)
