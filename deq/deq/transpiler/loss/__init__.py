"""Physical loss-event analysis and built-in platform models.

Loss analysis discovers source events and their propagated physical branches.
Projection of the resulting Pauli generators into decoder error rows is owned
by :mod:`deq.transpiler.loss.transpiler`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

from deq.transpiler.loss.analysis import LossAnalysisResult, analyze_loss_events
from deq.transpiler.loss.api import (
    GateLossPolicy,
    LossAnalysisState,
    LossGate,
    LossModel,
    QdkLossConfig,
    UnsupportedLossModelError,
)
from deq.transpiler.loss.model_neutral_atom import NeutralAtomLossModel
from deq.transpiler.loss.model_trapped_ion import TrappedIonLossModel
from deq.transpiler.loss.loss_graph import (
    LossBranch,
    LossEvent,
    LossEventGraph,
    PauliInsertion,
    build_loss_event_graph,
)

LOSS_MODEL_NAMES = ("neutral-atom", "trapped-ion")


@lru_cache(maxsize=None)
def _load_loss_model_file(path: str) -> LossModel:
    """Load and validate the model returned by a Python plugin file."""

    plugin_path = Path(path)
    digest = hashlib.sha256(path.encode()).hexdigest()[:16]
    module_name = f"deq_loss_model_{plugin_path.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, plugin_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load loss model from {plugin_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    factory = getattr(module, "create_loss_model", None)
    if not callable(factory):
        raise ValueError(
            f"loss model file {plugin_path} does not define a callable "
            "create_loss_model()"
        )
    model = factory()
    if not isinstance(model, LossModel):
        raise ValueError(
            f"create_loss_model() in {plugin_path} did not return a LossModel"
        )
    if not isinstance(model.config, QdkLossConfig):
        raise ValueError(
            f"loss model from {plugin_path} has config of type "
            f"{type(model.config).__name__}, expected QdkLossConfig"
        )
    return model


@dataclass(frozen=True)
class _FileLossModel:
    """Pickle-safe proxy that reloads a user model in worker processes."""

    path: str
    config: QdkLossConfig
    native_gates: frozenset[str]

    @cached_property
    def _model(self) -> LossModel:
        model = _load_loss_model_file(self.path)
        if model.config != self.config:
            raise ValueError(f"loss model file changed after loading: {self.path}")
        if model.native_gates != self.native_gates:
            raise ValueError(f"loss model file changed after loading: {self.path}")
        return model

    def handle_loss_source(self, event_id: int, state: LossAnalysisState) -> None:
        self._model.handle_loss_source(event_id, state)

    def handle_gate(self, gate: LossGate, state: LossAnalysisState) -> None:
        self._model.handle_gate(gate, state)

    def __reduce__(
        self,
    ) -> tuple[
        type[_FileLossModel],
        tuple[str, QdkLossConfig, frozenset[str]],
    ]:
        return type(self), (self.path, self.config, self.native_gates)


def create_loss_model(selector: str | Path) -> LossModel:
    """Create a built-in model by name or load one from a Python file."""

    constructors = {
        "neutral-atom": NeutralAtomLossModel,
        "trapped-ion": TrappedIonLossModel,
    }
    value = str(selector)
    if value in constructors:
        return constructors[value]()

    path = Path(value).expanduser()
    if path.suffix.lower() == ".py":
        if not path.is_file():
            raise ValueError(f"loss model file does not exist: {path}")
        resolved = str(path.resolve())
        model = _load_loss_model_file(resolved)
        return _FileLossModel(
            path=resolved,
            config=model.config,
            native_gates=model.native_gates,
        )

    supported = ", ".join(LOSS_MODEL_NAMES)
    raise ValueError(
        f"unknown loss model {value!r}; expected one of: {supported}, "
        "or a path to a .py file"
    )


__all__ = [
    "NeutralAtomLossModel",
    "TrappedIonLossModel",
    "GateLossPolicy",
    "QdkLossConfig",
    "LossBranch",
    "LossEvent",
    "LossEventGraph",
    "LossAnalysisState",
    "LossAnalysisResult",
    "LossGate",
    "LossModel",
    "LOSS_MODEL_NAMES",
    "PauliInsertion",
    "UnsupportedLossModelError",
    "analyze_loss_events",
    "build_loss_event_graph",
    "create_loss_model",
]
