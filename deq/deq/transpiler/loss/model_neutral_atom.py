"""Neutral-atom loss model matching the QDK simulator configuration."""

from __future__ import annotations

from deq.transpiler.loss.api import (
    GateLossPolicy,
    LossGateHandler,
    QdkLossConfig,
)
from deq.transpiler.loss.model_configured import ConfiguredLossGateHandler

_NEUTRAL_ATOM_QDK_CONFIG = QdkLossConfig(
    gate_policies=(
        ("cx", GateLossPolicy.SKIP),
        ("cy", GateLossPolicy.SKIP),
        ("cz", GateLossPolicy.SKIP),
        ("swap", GateLossPolicy.APPLY_ANYWAY),
    ),
)


class NeutralAtomLossGateHandler(ConfiguredLossGateHandler):
    """Use skipped lost-operand gates with physical SWAP relocation."""

    def __init__(self) -> None:
        super().__init__(_NEUTRAL_ATOM_QDK_CONFIG)


class NeutralAtomLossModel:
    """Neutral-atom platform model: SKIP gates and relocate atoms on SWAP."""

    name = "neutral-atom"
    source_config = _NEUTRAL_ATOM_QDK_CONFIG
    qdk_config = _NEUTRAL_ATOM_QDK_CONFIG

    def create_handler(self) -> LossGateHandler:
        """Create independent state for one gadget traversal."""

        return NeutralAtomLossGateHandler()
