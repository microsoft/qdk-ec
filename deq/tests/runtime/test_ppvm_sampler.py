"""Physical non-Clifford sampling through the QuEra PPVM adapter."""

import importlib.util
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

try:
    distribution("ppvm")
except PackageNotFoundError:
    pytest.skip("QuEra PPVM is optional", allow_module_level=True)

_PATH = Path(__file__).resolve().parents[2] / "deq_runtime/src/simulator/ppvm_sampler.py"
_SPEC = importlib.util.spec_from_file_location("ppvm_sampler_for_test", _PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
Sampler = _MODULE.Sampler


@pytest.mark.parametrize("failure", ["missing_package", "missing_extension", "missing_loader"])
def test_import_errors_include_install_command(monkeypatch, failure):
    def find_distribution(name):
        assert name == "ppvm"
        if failure == "missing_package":
            raise PackageNotFoundError(name)
        return SimpleNamespace(locate_file=lambda path: Path("/missing") / path)

    monkeypatch.setattr(_MODULE, "distribution", find_distribution)
    monkeypatch.setattr(_MODULE, "PathFinder", SimpleNamespace(
        find_spec=lambda *args: SimpleNamespace(loader=None) if failure == "missing_loader" else None
    ))
    with pytest.raises(ImportError) as caught:
        _MODULE._load_core.__wrapped__()

    message = str(caught.value)
    assert f'"{sys.executable}" -m pip install --no-deps --force-reinstall ' in message
    assert (
        '"ppvm @ git+https://github.com/QuEraComputing/ppvm.git@'
        '731e59fc98489f206767822f8dc8798ba23a5551#subdirectory=ppvm-python"'
    ) in message
    assert "requires Git and a Rust toolchain" in message
    assert "PyPI's ppvm is unrelated" in message


@pytest.mark.parametrize("gate,basis", [
    ("T", "X"), ("T_DAG", "X"), ("TX", "Z"), ("TX_DAG", "Z"),
    ("TY", "Z"), ("TY_DAG", "Z"), ("R_X(0.25)", "Z"),
    ("R_Y(0.25)", "Z"), ("R_Z(0.25)", "X"),
])
def test_rotation_probability(gate, basis):
    sampler = Sampler(f"R{basis} 0\n{gate} 0\nM{basis} 0", {"seed": 41})
    frequency = sum(sampler.sample() == "1" for _ in range(2000)) / 2000
    assert frequency == pytest.approx(0.1464466094, abs=0.025)


@pytest.mark.parametrize("gate", ["R_NEW", "NEW", "NEW_DAG"])
def test_new_non_clifford_gate_requires_explicit_angle_rule(monkeypatch, gate):
    from deq.circuit.model import Instruction, QubitTarget

    instruction = Instruction(name=gate, arguments=[0.125], targets=[QubitTarget(0)])
    monkeypatch.setattr(_MODULE, "parse", lambda text: SimpleNamespace(
        definitions=[SimpleNamespace(body=[instruction])]
    ))
    monkeypatch.setattr(_MODULE, "NON_CLIFFORD_AXES", {
        **_MODULE.NON_CLIFFORD_AXES, gate: "Z",
    })
    with pytest.raises(ValueError, match=f"PPVM does not support non-Clifford gate {gate}"):
        Sampler(f"{gate} 0", {"seed": 41})


def test_four_t_gates_physically_flip_x_measurement():
    sampler = Sampler("RX 0\nREPEAT 4 { T 0 }\nMX 0", {"seed": 2})
    assert {sampler.sample() for _ in range(30)} == {"1"}


def test_record_control_across_rotation():
    sampler = Sampler("RX 0\nM 0\nT 0\nCX rec[-1] 0\nM 0", {"seed": 3})
    shots = [sampler.sample() for _ in range(40)]
    assert set(shots) == {"00", "10"}


def test_seed_replay_and_skip_shots():
    source = "RX 0\nT 0\nMX 0"
    first = Sampler(source, {"seed": 42})
    replay = Sampler(source, {"seed": 42})
    skipped = Sampler(source, {"seed": 42, "skip_shots": 17})
    shots = [first.sample() for _ in range(80)]
    assert shots == [replay.sample() for _ in range(80)]
    assert shots[17:] == [skipped.sample() for _ in range(63)]


@pytest.mark.parametrize("gate,expected", [
    ("T", 0.1464466094), ("T_DAG", 0.8535533906),
    ("R_Z(0.5)", 0.0), ("R_Z(-0.5)", 1.0),
])
def test_rotation_sign(gate, expected):
    sampler = Sampler(f"RX 0\n{gate} 0\nMY 0", {"seed": 78})
    frequency = sum(sampler.sample() == "1" for _ in range(2000)) / 2000
    assert frequency == pytest.approx(expected, abs=0.025)


@pytest.mark.parametrize("config,message", [
    ({"seed": -1}, "seed"), ({"skip_shots": -1}, "skip_shots"),
    ({"num_measurements": 3}, "num_measurements"),
    ({"min_abs_coeff": -0.1}, "min_abs_coeff"),
    ({"min_abs_coeff": float("nan")}, "min_abs_coeff"),
    ({"loss_config": {}}, "loss_config"),
])
def test_invalid_sampler_config(config, message):
    with pytest.raises(ValueError, match=message):
        Sampler("RX 0\nT 0\nMX 0", config)


def test_independent_targets_and_inverted_measurements():
    sampler = Sampler("RX 0 1\nT 0 1\nMX !0 1", {"seed": 21})
    assert set(sampler.sample() for _ in range(1000)) == {"00", "01", "10", "11"}


@pytest.mark.parametrize("gate,inverse,basis", [
    ("T", "T_DAG", "X"), ("TX", "TX_DAG", "Z"), ("TY", "TY_DAG", "Z"),
    ("R_X(0.13)", "R_X(-0.13)", "Z"),
    ("R_Y(0.7)", "R_Y(-0.7)", "Z"),
    ("R_Z(0.23)", "R_Z(-0.23)", "X"),
])
def test_physical_inverse_pairs_cancel(gate, inverse, basis):
    sampler = Sampler(f"R{basis} 0\n{gate} 0\n{inverse} 0\nM{basis} 0", {"seed": 24})
    assert {sampler.sample() for _ in range(50)} == {"0"}


def test_inverted_record_control_uses_reported_bit():
    sampler = Sampler("R 0\nM !0\nT 0\nCX rec[-1] 0\nM 0", {"seed": 7})
    assert sampler.sample() == "11"


def test_preselection_is_explicitly_unsupported():
    with pytest.raises(ValueError, match="preselection is not supported"):
        Sampler("SELECT { R 0\nM 0\nREQUIRE rec[-1] }", {"seed": 7})


def test_zero_measurement_circuit():
    assert Sampler("T 0", {"seed": 7, "num_measurements": 0}).sample() == ""


def test_tutorial_runs_through_embedded_sampler_and_tesseract(tmp_path, monkeypatch, capsys):
    from deq.cli.simulate import simulate__ler

    deq_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(deq_root)
    simulate__ler(
        str(deq_root / "documents/tutorial/examples/non-clifford/rotations.deq"),
        program="FourTExperiment", simulator="ppvm", decoder="black-box-tesseract",
        shots=100, errors=100, batch_size=50, jobs=1, seed=42, save=str(tmp_path),
    )
    output = capsys.readouterr().out
    assert "Shots:          100" in output
    assert "Logical errors: 0" in output
    assert "Failed shots:   0" in output
    assert (tmp_path / "FourTExperiment.stim").read_text().splitlines().count("T 0") == 4
