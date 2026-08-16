"""QDK sampler platform loss-configuration tests."""

import importlib.util
from pathlib import Path

import pytest
from qdk.simulation import LossPolicy, NoiseConfig

from deq.transpiler.loss import NeutralAtomLossModel, TrappedIonLossModel


_SAMPLER_PATH = (
    Path(__file__).resolve().parents[2]
    / "deq_runtime"
    / "src"
    / "simulator"
    / "qdk_sampler.py"
)
_SPEC = importlib.util.spec_from_file_location("qdk_sampler_for_test", _SAMPLER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_SAMPLER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SAMPLER)


def test_neutral_atom_config_skips_gates_and_relocates_swap() -> None:
    noise = NoiseConfig()

    _SAMPLER._configure_loss(noise, NeutralAtomLossModel.config.to_json_object())

    for (
        table_name,
        policy_name,
    ) in NeutralAtomLossModel.config.to_json_object().items():
        expected = getattr(LossPolicy, policy_name)
        assert getattr(noise, table_name).on_loss == expected


def test_missing_config_leaves_qdk_defaults_unchanged() -> None:
    noise = NoiseConfig()
    defaults = {
        gate: getattr(noise, gate).on_loss
        for gate in NeutralAtomLossModel.config.to_json_object()
    }

    _SAMPLER._configure_loss(noise, None)

    assert {gate: getattr(noise, gate).on_loss for gate in defaults} == defaults


def test_trapped_ion_config_sets_only_supported_gate_policies() -> None:
    noise = NoiseConfig()

    _SAMPLER._configure_loss(noise, TrappedIonLossModel.config.to_json_object())

    for (
        table_name,
        policy_name,
    ) in TrappedIonLossModel.config.to_json_object().items():
        assert getattr(noise, table_name).on_loss == getattr(LossPolicy, policy_name)


@pytest.mark.parametrize("lost_qubit", [0, 1])
def test_trapped_ion_qdk_sampler_applies_cz_residual_s_dagger(
    lost_qubit: int,
) -> None:
    survivor = 1 - lost_qubit
    sampler = _SAMPLER.Sampler(
        f"H {survivor}\nS {survivor}\nLOSS_ERROR(1) {lost_qubit}\n"
        f"CZ 0 1\nH {survivor}\nM 0 1\n",
        {
            "seed": 7,
            "batch_size": 1,
            "loss_config": TrappedIonLossModel.config.to_json_object(),
        },
    )

    assert sampler.sample() == ("-0" if lost_qubit == 0 else "0-")


def test_config_applies_only_explicit_gate_overrides() -> None:
    noise = NoiseConfig()
    original_cx = noise.cx.on_loss

    _SAMPLER._configure_loss(
        noise,
        {"cz": "RESIDUAL_S_DAGGER"},
    )

    assert noise.cz.on_loss == LossPolicy.RESIDUAL_S_DAGGER
    assert noise.cx.on_loss == original_cx


@pytest.mark.parametrize(
    ("config", "error_type", "message"),
    [
        ([], ValueError, "loss_config must be a JSON object"),
        ({"unknown": "SKIP"}, AttributeError, "unknown"),
        ({"cx": "UNKNOWN"}, AttributeError, "UNKNOWN"),
        ({"cx": "APPLY_ANYWAY"}, AttributeError, "only supports"),
    ],
)
def test_invalid_qdk_sampler_loss_config_is_rejected(
    config: object, error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        _SAMPLER._configure_loss(NoiseConfig(), config)
