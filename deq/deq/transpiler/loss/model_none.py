"""Opt-out selector that compiles a circuit as if no qubit could be lost."""

from __future__ import annotations

from deq.transpiler.loss.api import (
    LossAnalysisState,
    LossGate,
    QdkLossConfig,
    UnsupportedLossModelError,
)

_NO_LOSS_CONFIG = QdkLossConfig(gate_policies=())

_DISABLED = (
    "the 'none' loss model never analyzes loss; select a platform model to "
    "compile LOSS_ERROR into decoder metadata"
)


class NoLossModel:
    """Disable loss entirely instead of describing how a lost qubit behaves.

    Selecting this model leaves every gadget without loss metadata and drops
    ``LOSS_ERROR`` from the exported Stim circuit, so the decoder and the
    simulator agree that loss never happens. Use it when a circuit declares
    ``LOSS_ERROR`` for other backends, or when its gates fall outside every
    built-in platform model's supported scope.
    """

    config = _NO_LOSS_CONFIG
    native_gates: frozenset[str] = frozenset()

    def handle_loss_source(self, event_id: int, state: LossAnalysisState) -> None:
        raise UnsupportedLossModelError(_DISABLED)

    def handle_gate(self, gate: LossGate, state: LossAnalysisState) -> None:
        raise UnsupportedLossModelError(_DISABLED)


def create_loss_model() -> NoLossModel:
    """Create this model when the module is loaded as a plugin file."""

    return NoLossModel()
