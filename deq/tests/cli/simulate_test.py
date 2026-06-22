# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint
"""Tests for `deq.cli.simulate` covering the jit-static simulator wiring."""

import json
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

import deq.proto.deq_jit_pb2 as jit_pb
from deq.cli.simulate import _run_batch, simulate__ler


def _fake_completed_proc() -> SimpleNamespace:
    """Return a SimpleNamespace mimicking the relevant subprocess.run result."""
    return SimpleNamespace(
        returncode=0,
        stdout=(
            "Simulation Complete\n"
            "Shots: 10/10\n"
            "Logical errors: 0/10\n"
            "Retries: 0\n"
        ),
        stderr="",
    )


class TestRunBatchControllerSelection:
    """`_run_batch` must pick the controller that matches the simulator.

    Regression test for the bug where `--simulator jit-static` always launched
    `deq_runtime` with `--controller static`, causing per-gadget execute/decode
    RPCs to fail because they are not implemented on the static controller.
    """

    @staticmethod
    def _extract_controller(cmd: list[str]) -> tuple[str, dict]:
        idx = cmd.index("--controller")
        controller_name = cmd[idx + 1]
        cfg_idx = cmd.index("--controller-config")
        controller_config = json.loads(cmd[cfg_idx + 1])
        return controller_name, controller_config

    @staticmethod
    def _extract_simulator_config(cmd: list[str]) -> dict:
        idx = cmd.index("--simulator-config")
        return json.loads(cmd[idx + 1])

    def test_static_simulator_uses_static_controller(self, tmp_path) -> None:
        bin_path = str(tmp_path / "prog.deq.bin")
        stim_path = str(tmp_path / "prog.stim")
        jit_path = str(tmp_path / "prog.deq.jit")

        with patch(
            "deq.cli.simulate.subprocess.run",
            return_value=_fake_completed_proc(),
        ) as mock_run:
            _run_batch(
                bin_path=bin_path,
                stim_path=stim_path,
                jit_path=jit_path,
                batch_size=10,
                max_errors=1,
                decoder="black-box-tesseract",
                decoder_config=None,
                coordinator="monolithic",
                coordinator_config=None,
                seed=42,
                debug_dir=None,
                simulator="static",
            )

        cmd = mock_run.call_args.args[0]
        controller_name, controller_config = self._extract_controller(cmd)
        assert controller_name == "static"
        assert controller_config == {"filepath": bin_path}
        # jit-static-only field must not leak into the static simulator config.
        sim_cfg = self._extract_simulator_config(cmd)
        assert "jit_library_filepath" not in sim_cfg

    def test_jit_static_simulator_uses_jit_controller(self, tmp_path) -> None:
        bin_path = str(tmp_path / "prog.deq.bin")
        stim_path = str(tmp_path / "prog.stim")
        jit_path = str(tmp_path / "prog.deq.jit")

        with patch(
            "deq.cli.simulate.subprocess.run",
            return_value=_fake_completed_proc(),
        ) as mock_run:
            _run_batch(
                bin_path=bin_path,
                stim_path=stim_path,
                jit_path=jit_path,
                batch_size=10,
                max_errors=1,
                decoder="black-box-tesseract",
                decoder_config=None,
                coordinator="window",
                coordinator_config='{"buffer_radius":0}',
                seed=42,
                debug_dir=None,
                simulator="jit-static",
            )

        cmd = mock_run.call_args.args[0]
        controller_name, controller_config = self._extract_controller(cmd)
        assert controller_name == "jit"
        assert controller_config == {"filepath": jit_path}
        sim_cfg = self._extract_simulator_config(cmd)
        assert sim_cfg["jit_library_filepath"] == jit_path


_REPETITION_CODE_DEQ = """\
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


class _SyncFuture(Future):
    """A pre-resolved Future used by the synchronous fake executor below."""

    def __init__(self, result):
        super().__init__()
        self.set_result(result)


class _SyncPool:
    """Drop-in for ProcessPoolExecutor that runs submitted callables inline."""

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, /, *args, **kwargs):
        return _SyncFuture(fn(*args, **kwargs))


class TestSimulateLerSerializationOrdering:
    """Regression test for bug #1: `.deq.jit` was serialized before the
    program block was appended, so the on-disk file contained an empty
    program. The jit-static simulator reloads this file and iterates over
    its program, so the empty program caused it to produce no readouts.
    """

    def test_jit_file_contains_program_after_simulate_ler(self, tmp_path) -> None:
        deq_path = tmp_path / "trivial.deq"
        deq_path.write_text(_REPETITION_CODE_DEQ)
        out_dir = tmp_path / "out"

        # Make _run_batch return enough errors to terminate after one batch.
        fake_batch_result = {
            "shots": 1,
            "logical_errors": 1,
            "decode_time_per_shot": 0.0,
            "retries": 0,
        }

        with patch(
            "deq.cli.simulate._run_batch", return_value=fake_batch_result
        ), patch(
            "concurrent.futures.ProcessPoolExecutor", _SyncPool
        ), patch(
            "deq.compiler.jit_compiler.static_jit_compiler",
            return_value=SimpleNamespace(SerializeToString=lambda: b""),
        ):
            simulate__ler(
                str(deq_path),
                program="TestProgram",
                save=str(out_dir),
                shots=1,
                errors=1,
                batch_size=1,
                jobs=1,
                simulator="jit-static",
            )

        jit_path = out_dir / "TestProgram.deq.jit"
        assert jit_path.exists(), "simulate ler should write a .deq.jit file"
        library = jit_pb.JitLibrary.FromString(jit_path.read_bytes())
        assert len(library.program) > 0, (
            "Serialized .deq.jit must contain the program instructions; "
            "otherwise the jit-static simulator decodes an empty program "
            "and returns no readouts."
        )
