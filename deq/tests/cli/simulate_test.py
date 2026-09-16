# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint
"""End-to-end smoke test for `deq simulate ler` across simulator backends.

Regression test for the bug where `--simulator jit-static` panicked with
`Array index 0 out of bounds: array is empty` while `--simulator static`
worked on the same program.
"""

from pathlib import Path
import json

import pytest

from deq.proto import simulator_pb2 as simulator_pb
from deq.cli.simulate import (
    _LerResult,
    _batch_trace_path,
    _configure_loss_imputation,
    _merge_simulator_traces,
    _parse_server_output,
    simulate__ler,
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
def test_simulate_ler_forced_gap_outputs_readout_probability(
    tmp_path: Path, simulator: str
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
def test_commit_error_limit_reports_failed_shots_without_aborting(
    tmp_path: Path, capsys, simulator: str, coordinator: str
) -> None:
    deq_path = tmp_path / "count_limit.deq"
    deq_path.write_text(
        _TEST_PROGRAM_DEQ.replace(
            "    M 0\n",
            "    R 1\n    X_ERROR(0.25) 1\n    M 1\n    CHECK rec[-1]\n    M 0\n",
        )
    )
    for limit in (0, 1):
        output = tmp_path / f"limit_{limit}.pb"
        config = {"max_commit_errors": limit, "merge_hyperedges": False}
        if coordinator == "window":
            config["buffer_radius"] = 0
        simulate__ler(
            str(deq_path),
            program="TestProgram",
            save=str(tmp_path / f"out_{limit}"),
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
        failed = sum(not shot.HasField("decode_result") for shot in trace.shots)
        assert (0 < failed < 32) if limit == 0 else failed == 0
        assert all(not shot.logical_error for shot in trace.shots)
        assert f"  Failed shots:   {failed}" in capsys.readouterr().out


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
