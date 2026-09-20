from deq.transpiler.loss import GateLossPolicy, QdkLossConfig
from deq.transpiler.loss.model_neutral_atom import NeutralAtomLossModel


class PropagatingCzLossModel(NeutralAtomLossModel):
    config = QdkLossConfig(
        gate_policies=(
            ("cx", GateLossPolicy.SKIP),
            ("cy", GateLossPolicy.SKIP),
            ("cz", GateLossPolicy.PROPAGATE),
            ("swap", GateLossPolicy.APPLY_ANYWAY),
        )
    )


def create_loss_model() -> PropagatingCzLossModel:
    return PropagatingCzLossModel()
