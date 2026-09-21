"""QDK sampler non-Clifford and platform loss-configuration tests."""

import importlib.util
from collections import Counter
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


@pytest.mark.parametrize("kind", ["clifford", "cpu"])
@pytest.mark.parametrize("gate,basis", [
    ("T", "X"), ("T_DAG", "X"), ("R_X(0.25)", "Z"),
    ("R_Y(0.25)", "Z"), ("R_Z(0.25)", "X"),
    ("R_X(-0.25)", "Z"), ("R_Y(-0.25)", "Z"),
])
def test_non_clifford_rotation_probability(kind, gate, basis):
    sampler = _SAMPLER.Sampler(
        f"R{basis} 0\n{gate} 0\nM{basis} 0",
        {"seed": 42, "batch_size": 2000, "type": kind, "num_measurements": 1},
    )
    frequency = sum(sampler.sample() == "1" for _ in range(2000)) / 2000
    assert frequency == pytest.approx(0.1464466094, abs=0.025)


@pytest.mark.parametrize("kind", ["clifford", "cpu"])
@pytest.mark.parametrize("gate,expected", [
    ("T", 0.1464466094), ("T_DAG", 0.8535533906),
    ("R_Z(0.5)", 0.0), ("R_Z(-0.5)", 1.0),
])
def test_non_clifford_rotation_sign(kind, gate, expected):
    sampler = _SAMPLER.Sampler(
        f"RX 0\n{gate} 0\nMY 0", {"seed": 78, "batch_size": 2000, "type": kind}
    )
    frequency = sum(sampler.sample() == "1" for _ in range(2000)) / 2000
    assert frequency == pytest.approx(expected, abs=0.025)


@pytest.mark.parametrize("kind", ["clifford", "cpu"])
@pytest.mark.parametrize("source,expected", [
    ("RX 0\nREPEAT 4 {\nT 0\n}\nMX 0", {"1"}),
    ("RX 0\nT 0\nT_DAG 0\nMX 0", {"0"}),
    ("R 0\nR_X(0.13) 0\nR_X(-0.13) 0\nM 0", {"0"}),
    ("R 0\nR_Y(0.7) 0\nR_Y(-0.7) 0\nM 0", {"0"}),
    ("RX 0\nR_Z(0.23) 0\nR_Z(-0.23) 0\nMX 0", {"0"}),
    ("RX 0\nM 0\nT 0\nCX rec[-1] 0\nM 0", {"00", "10"}),
    ("R 0\nM !0\nT 0\nCX rec[-1] 0\nM 0", {"11"}),
    ("RX 0 1\nT 0 1\nT_DAG 0 1\nMX !0 1", {"10"}),
    ("RX 0\nREPEAT 4 {\nT 0\n}\nZ_ERROR(1) 0\nMX 0", {"0"}),
    ("RX 0\nT 0\nLOSS_ERROR(1) 0\nMX 0", {"-"}),
    ("SELECT {\nRX 0\nT 0\nMX 0\nREQUIRE rec[-1]\n}\nMX 0", {"00"}),
])
def test_non_clifford_sampling_workflows(kind, source, expected):
    sampler = _SAMPLER.Sampler(source, {"seed": 7, "batch_size": 64, "type": kind})
    assert {sampler.sample() for _ in range(64)} == expected


def test_non_clifford_seed_replay_and_skip_across_refills():
    source = "RX 0\nT 0\nMX 0"
    config = {"seed": 42, "batch_size": 13}
    first = _SAMPLER.Sampler(source, config)
    replay = _SAMPLER.Sampler(source, config)
    skipped = _SAMPLER.Sampler(source, {**config, "skip_shots": 17})
    shots = [first.sample() for _ in range(80)]
    assert shots == [replay.sample() for _ in range(80)]
    assert shots[17:] == [skipped.sample() for _ in range(63)]


def test_non_clifford_branching_with_many_qubits():
    targets = " ".join(str(qubit) for qubit in range(65))
    sampler = _SAMPLER.Sampler(
        f"RX {targets}\nT 64\nT_DAG 64\nMX {targets}",
        {"seed": 42, "batch_size": 8, "num_measurements": 65},
    )
    assert {sampler.sample() for _ in range(8)} == {"0" * 65}


@pytest.mark.parametrize("gate,rotation,measurement,probability", [
    ("TX", "R_X(0.25)", "MY", 0.8535533906),
    ("TX_DAG", "R_X(-0.25)", "MY", 0.1464466094),
    ("TY", "R_Y(0.25)", "MX", 0.1464466094),
    ("TY_DAG", "R_Y(-0.25)", "MX", 0.8535533906),
])
def test_deq_alias_export_runs_on_qdk(tmp_path, gate, rotation, measurement, probability):
    from deq.circuit.parser import parse
    from deq.cli.jit import jit_compile_program_to_file
    from deq.transpiler.jit_library_builder import build_jit_library

    source = parse(f"""
        GADGET G {{
            R 7 9
            {gate}[phase] 7 9
            {measurement} 7 9
        }}
        PROGRAM Run {{ G }}
    """)
    jit_compile_program_to_file(
        build_jit_library(source), source, str(tmp_path / "run.deq.jit"), program="Run"
    )
    circuit = (tmp_path / "run.stim").read_text()
    name, arguments = rotation.split("(", 1)
    assert f"{name}[phase]({arguments} 0 1" in circuit
    sampler = _SAMPLER.Sampler(circuit, {"seed": 42, "batch_size": 2000, "num_measurements": 2})
    shots = [sampler.sample() for _ in range(2000)]
    for target in range(2):
        frequency = sum(shot[target] == "1" for shot in shots) / len(shots)
        assert frequency == pytest.approx(probability, abs=0.03)


def test_non_clifford_independent_targets_and_inverted_measurements():
    sampler = _SAMPLER.Sampler(
        "RX 0 1\nT 0 1\nMX !0 1", {"seed": 21, "batch_size": 1000}
    )
    assert {sampler.sample() for _ in range(1000)} == {"00", "01", "10", "11"}


def test_non_clifford_zero_measurement_circuit():
    sampler = _SAMPLER.Sampler("T 0", {"seed": 7, "batch_size": 1, "num_measurements": 0})
    assert sampler.sample() == ""


def test_non_clifford_tutorial_runs_through_qdk_and_tesseract(tmp_path, monkeypatch, capsys):
    from deq.cli.simulate import simulate__ler

    deq_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(deq_root)
    simulate__ler(
        str(deq_root / "documents/tutorial/examples/non-clifford/rotations.deq"),
        program="FourTExperiment", simulator="qdk", decoder="black-box-tesseract",
        shots=100, errors=100, batch_size=50, jobs=1, seed=42, save=str(tmp_path),
    )
    output = capsys.readouterr().out
    assert "Shots:          100" in output
    assert "Logical errors: 0" in output
    assert "Failed shots:   0" in output
    assert (tmp_path / "FourTExperiment.stim").read_text().splitlines().count("T 0") == 4


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


def test_qdk_correlated_loss_branches_have_equal_marginal_probabilities():
    sampler = _SAMPLER.Sampler(
        "R 0 1\nCORRELATED_ERROR(0.1) L0\n"
        "ELSE_CORRELATED_ERROR(0.1111111111111111) L1\n"
        "ELSE_CORRELATED_ERROR(0.125) L0 L1\nM 0 1\n",
        {"seed": 351, "batch_size": 10000, "loss_config": NeutralAtomLossModel.config.to_json_object()},
    )
    counts = Counter(sampler.sample() for _ in range(10000))
    assert set(counts) == {"00", "-0", "0-", "--"}
    for outcome in ("-0", "0-", "--"):
        assert counts[outcome] / 10000 == pytest.approx(0.1, abs=0.015)


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


@pytest.mark.parametrize(
    ("control_setup", "expected"),
    [
        ("LOSS_ERROR(1) 1", "-0"),
        ("X 1", "11"),
    ],
)
def test_qdk_sampler_record_control_skips_loss_and_applies_one(
    control_setup: str, expected: str
) -> None:
    sampler = _SAMPLER.Sampler(
        f"R 0 1\n{control_setup}\nM 1\nCX rec[-1] 0\nM 0\n",
        {
            "seed": 7,
            "batch_size": 1,
            "loss_config": NeutralAtomLossModel.config.to_json_object(),
        },
    )

    assert sampler.sample() == expected


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
