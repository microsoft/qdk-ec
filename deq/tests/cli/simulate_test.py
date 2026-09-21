# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint
"""End-to-end smoke test for `deq simulate ler` across simulator backends.

Regression test for the bug where `--simulator jit-static` panicked with
`Array index 0 out of bounds: array is empty` while `--simulator static`
worked on the same program.
"""

from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
import json
import math

import pytest

from deq.proto import simulator_pb2 as simulator_pb
from deq.cli.simulate import (
    _LerResult,
    _batch_trace_path,
    _configure_loss_imputation,
    _merge_simulator_traces,
    _parse_server_output,
    _run_batch,
    simulate__ler,
)


@pytest.fixture
def require_ppvm():
    try:
        distribution("ppvm")
    except PackageNotFoundError:
        pytest.skip("QuEra PPVM is optional")


@pytest.mark.usefixtures("require_ppvm")
def test_ppvm_backend_uses_embedded_python_sampler(monkeypatch):
    from types import SimpleNamespace

    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(
            returncode=0, stdout="Simulation Complete\nShots: 2/2\nLogical errors: 0/2\n", stderr=""
        )

    monkeypatch.setattr("deq.cli.simulate.subprocess.run", run)
    _run_batch("input.bin", "input.stim", "input.jit", 2, 10,
               "black-box-tesseract", None, "monolithic", None, 17, None,
               simulator="ppvm")
    command = commands[0]
    assert command[command.index("--simulator") + 1] == "python"
    config = json.loads(command[command.index("--simulator-config") + 1])
    assert config["sampler"] == "@ppvm_sampler"
    assert config["seed"] == 17
    assert config["py_config"] == {}


@pytest.mark.usefixtures("require_ppvm")
def test_ppvm_rejects_qdk_loss_policy_override():
    with pytest.raises(ValueError, match="PPVM does not support .*loss"):
        simulate__ler("unused.deq", program="Unused", simulator="ppvm", simulation_loss_model="{}")


@pytest.mark.parametrize(
    "gap_decoder,gap_config",
    [
        (None, None),
        ("black-box-relay-bp", None),
        ("black-box-relay-bp", '{"seed":17}'),
        (None, "{}"),
    ],
)
def test_gap_decoder_options_are_forwarded(monkeypatch, gap_decoder, gap_config):
    from types import SimpleNamespace

    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout="Simulation Complete\nShots: 1/1\nLogical errors: 0/1\n",
            stderr="",
        )

    monkeypatch.setattr("deq.cli.simulate.subprocess.run", run)
    result = _run_batch(
        "input.bin",
        "input.stim",
        "input.jit",
        1,
        2,
        "black-box-tesseract",
        '{"parallel":1}',
        "monolithic",
        None,
        17,
        None,
        gap_decoder=gap_decoder,
        gap_decoder_config=gap_config,
    )
    assert result["shots"] == 1
    command = commands[0]
    assert command[command.index("--decoder") + 1] == "black-box-tesseract"
    for flag, value in (
        ("--gap-decoder", gap_decoder),
        ("--gap-decoder-config", gap_config),
    ):
        if value is None:
            assert flag not in command
        else:
            assert command[command.index(flag) + 1] == value


@pytest.mark.parametrize("parallel", [0, 1, 2, None, "auto"])
@pytest.mark.parametrize("gap_decoder", [None, "black-box-relay-bp"])
def test_simulate_rejects_gap_pool_size_before_compilation(
    tmp_path, parallel, gap_decoder
):
    with pytest.raises(ValueError, match="parallel.*--decoder-config"):
        simulate__ler(
            str(tmp_path / "not_compiled.deq"),
            program="TestProgram",
            decoder_config='{"parallel":1}',
            gap_decoder=gap_decoder,
            gap_decoder_config=json.dumps({"parallel": parallel}),
        )


def test_failed_shots_are_reported_separately_from_logical_errors() -> None:
    parsed = _parse_server_output(
        "Shots: 10/10\nLogical errors: 1/5\nFailed shots: 8\n"
    )
    assert parsed == {"shots": 10, "logical_errors": 1, "failed_shots": 8}
    result = _LerResult(**parsed)
    assert result.retained_shots == 2
    assert result.error_rate == 0.5
    assert _LerResult(shots=10, failed_shots=10).error_rate == 0


_TEST_PROGRAM_DEQ = """\
CODE TrivialCode [[1,1]] {
    LOGICAL X0 Z0
}

GADGET PrepareZ {
    R 0
    OUTPUT TrivialCode 0
}

GADGET Idle {
    INPUT TrivialCode 0
    OUTPUT TrivialCode 1
}

GADGET MeasureZ {
    INPUT TrivialCode 0
    M 0
    READOUT rec[-1]
}

PROGRAM TestProgram {
    PrepareZ 0
    Idle 0
    MeasureZ 0
    ASSERT_EQ rec[-1] 0
}
"""


@pytest.mark.parametrize("simulator", ["static", "jit-static"])
def test_simulate_ler_does_not_panic(tmp_path: Path, simulator: str) -> None:
    """Both simulators must run the same program to completion without panicking."""
    deq_path = tmp_path / "trivial.deq"
    deq_path.write_text(_TEST_PROGRAM_DEQ)

    output_dir = tmp_path / f"out_{simulator}"
    simulate__ler(
        str(deq_path),
        program="TestProgram",
        save=str(output_dir),
        shots=10,
        errors=1,
        batch_size=10,
        jobs=1,
        simulator=simulator,
        seed=42,
    )
    assert not list(output_dir.glob(".simulator-trace-*.pb"))


@pytest.mark.parametrize("simulator", ["static", "jit-static"])
@pytest.mark.parametrize("gap_decoder", [None, "black-box-relay-bp"])
def test_simulate_ler_forced_gap_outputs_readout_probability(
    tmp_path: Path, simulator: str, gap_decoder
) -> None:
    deq_path = tmp_path / "logical_error.deq"
    deq_path.write_text(
        _TEST_PROGRAM_DEQ.replace("    M 0\n", "    X_ERROR(0.1) 0\n    M 0\n")
    )
    probabilities_path = tmp_path / "probabilities.pb"

    simulate__ler(
        str(deq_path),
        program="TestProgram",
        save=str(tmp_path / "out"),
        shots=10,
        errors=11,
        batch_size=3,
        jobs=2,
        simulator=simulator,
        decoder="black-box-tesseract",
        decoder_config='{"parallel":1}',
        gap_decoder=gap_decoder,
        gap_decoder_config='{"seed":17}' if gap_decoder else None,
        coordinator_config='{"forced_gap":true}',
        simulator_trace_output=str(probabilities_path),
        seed=42,
    )

    output = simulator_pb.SimulatorTrace.FromString(probabilities_path.read_bytes())
    assert len(output.shots) == 10
    assert [record.shot for record in output.shots] == list(range(10))
    assert all(
        record.decode_result.probabilities == pytest.approx([0.1])
        for record in output.shots
    )
    assert not list((tmp_path / "out").glob(".simulator-trace-*.pb"))


def test_simulator_trace_output_can_record_hard_only_trace(tmp_path: Path) -> None:
    deq_path = tmp_path / "trivial.deq"
    deq_path.write_text(_TEST_PROGRAM_DEQ)
    output_path = tmp_path / "hard-only.pb"

    simulate__ler(
        str(deq_path),
        program="TestProgram",
        save=str(tmp_path / "out"),
        shots=10,
        errors=11,
        batch_size=10,
        jobs=1,
        simulator_trace_output=str(output_path),
        seed=42,
    )

    output = simulator_pb.SimulatorTrace.FromString(output_path.read_bytes())
    assert len(output.shots) == 10
    assert all(not record.decode_result.probabilities for record in output.shots)
    assert not list((tmp_path / "out").glob(".simulator-trace-*.pb"))


@pytest.mark.parametrize("simulator", ["static", "jit-static", "preselect"])
@pytest.mark.parametrize("coordinator", ["window", "monolithic"])
def test_simulator_trace_preserves_statistics_for_postprocessing(
    tmp_path: Path, capsys, simulator: str, coordinator: str
) -> None:
    deq_path = tmp_path / "correction_statistics.deq"
    deq_path.write_text(
        _TEST_PROGRAM_DEQ.replace(
            "    M 0\n",
            "    R 1\n    X_ERROR(0.25) 1\n    M 1\n    CHECK rec[-1]\n    M 0\n",
        )
    )
    output = tmp_path / "shots.pb"
    config = {"merge_hyperedges": False}
    if coordinator == "window":
        config["buffer_radius"] = 0
    simulate__ler(
        str(deq_path),
        program="TestProgram",
        save=str(tmp_path / "out"),
        shots=32,
        errors=33,
        batch_size=16,
        jobs=1,
        simulator=simulator,
        coordinator=coordinator,
        coordinator_config=json.dumps(config),
        decoder="black-box-tesseract",
        simulator_trace_output=str(output),
        seed=42,
    )
    trace = simulator_pb.SimulatorTrace.FromString(output.read_bytes())
    assert len(trace.shots) == 32
    assert all(shot.HasField("decode_result") for shot in trace.shots)
    assert all(not shot.logical_error for shot in trace.shots)
    counts = []
    for shot in trace.shots:
        assert len(shot.gadget_readouts) == 3
        assert len({gadget.gid for gadget in shot.gadget_readouts}) == 3
        assert any(gadget.readouts.size == 0 for gadget in shot.gadget_readouts)
        for gadget in shot.gadget_readouts:
            assert gadget.syndrome_count == gadget.correction_count
            assert gadget.correction_weight == pytest.approx(
                gadget.correction_count * math.log(3)
            )
        counts.append(max(gadget.correction_count for gadget in shot.gadget_readouts))
        assert shot.decode_result.correction_count == sum(
            gadget.correction_count for gadget in shot.gadget_readouts
        )
    assert 0 < sum(count <= 0 for count in counts) < 32
    assert sum(count <= 1 for count in counts) == 32
    assert "  Failed shots:   0" in capsys.readouterr().out


def test_simulator_trace_batches_merge_in_order_with_global_shot_ids(
    tmp_path: Path,
) -> None:
    for batch_id in range(3):
        trace = simulator_pb.SimulatorTrace()
        for local_shot in range(2):
            trace.shots.add(
                shot=local_shot,
                logical_error=bool(batch_id % 2),
            )
        Path(_batch_trace_path(str(tmp_path), batch_id)).write_bytes(
            trace.SerializeToString()
        )

    output_path = tmp_path / "merged.pb"
    _merge_simulator_traces(str(tmp_path), 3, str(output_path), 6)

    merged = simulator_pb.SimulatorTrace.FromString(output_path.read_bytes())
    assert [record.shot for record in merged.shots] == list(range(6))
    assert [record.logical_error for record in merged.shots] == [
        False,
        False,
        True,
        True,
        False,
        False,
    ]


def test_failed_trace_merge_preserves_existing_output(tmp_path: Path) -> None:
    batch = simulator_pb.SimulatorTrace()
    batch.shots.add(shot=0)
    Path(_batch_trace_path(str(tmp_path), 0)).write_bytes(batch.SerializeToString())
    output = tmp_path / "merged.pb"
    output.write_bytes(b"existing trace")

    with pytest.raises(RuntimeError, match="1 records for 2 shots"):
        _merge_simulator_traces(str(tmp_path), 1, str(output), 2)

    assert output.read_bytes() == b"existing trace"
    assert not list(tmp_path.glob("merged.pb.tmp-*"))


def test_simulation_seed_defaults_loss_imputation_seed() -> None:
    assert (
        _configure_loss_imputation(
            "window",
            '{"forced_gap":true}',
            42,
        )
        == '{"forced_gap":true,"loss_random_imputation_seed":42}'
    )
    assert (
        _configure_loss_imputation(
            "window",
            None,
            42,
        )
        == '{"loss_random_imputation_seed":42}'
    )


@pytest.mark.parametrize("seed", [42, 43])
def test_explicit_loss_imputation_seed_is_preserved(seed: int) -> None:
    assert (
        _configure_loss_imputation(
            "monolithic",
            '{"loss_random_imputation_seed":7}',
            seed,
        )
        == '{"loss_random_imputation_seed":7}'
    )
    assert _configure_loss_imputation("window", "", 42) == ""
    assert _configure_loss_imputation("mock", None, 42) is None
    assert _configure_loss_imputation("window", None, None) is None
