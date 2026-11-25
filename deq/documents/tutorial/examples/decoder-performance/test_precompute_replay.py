"""Verify raw-sample manifests cannot disguise incomplete or duplicated inputs."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from precompute_replay import verify_samples
from study_utils import exclusive_output


@pytest.mark.parametrize(("width", "layers"), [(1, 3), (3, 2)])
def test_depth_preflight_matches_compiled_measurements(tmp_path, width, layers):
    from lattice_surgery import compile_case
    from precompute_replay import expected_measurements

    configuration = compile_case(
        tmp_path,
        width=width,
        layers=layers,
        distance=7,
        probability=0.001,
        basis="Z",
        seed=42,
        block_seconds=0.07,
        export_physical=False,
    )
    assert configuration["measurements"] == expected_measurements(width, layers)


def test_preparation_rejects_reusing_a_different_depth(tmp_path):
    from precompute_replay import prepare_width

    case = tmp_path / "w1"
    case.mkdir()
    (case / "samples.json").write_text(json.dumps({"layers": 50}))
    with pytest.raises(ValueError, match="sampling depth"):
        prepare_width(str(tmp_path), 1, 10, layers=100)


def test_low_probability_reaches_compiler_and_sampler(tmp_path, monkeypatch):
    import precompute_replay

    probability = 1e-5
    compile_case = precompute_replay.compile_case
    sample_batch = precompute_replay.sample_batch
    calls = []

    def compile_at_probability(*args, **kwargs):
        assert kwargs["probability"] == probability
        calls.append("compile")
        return compile_case(*args, **kwargs)

    def sample_at_probability(recipe, actual_probability, **kwargs):
        assert actual_probability == probability
        calls.append("sample")
        return sample_batch(recipe, actual_probability, **kwargs)

    monkeypatch.setattr(precompute_replay, "compile_case", compile_at_probability)
    monkeypatch.setattr(precompute_replay, "sample_batch", sample_at_probability)
    report = precompute_replay.prepare_width(
        str(tmp_path), 1, 10, layers=1, probability=probability
    )
    assert calls == ["compile", "sample"]
    assert report["status"] == "complete" and report["probability"] == probability
    case = tmp_path / "w1"
    assert (
        json.loads((case / "configuration.json").read_text())["probability"]
        == probability
    )
    verify_samples(case, report)
    manifest = (case / "samples.json").read_bytes()
    with pytest.raises(ValueError, match="sampling probability"):
        precompute_replay.prepare_width(str(tmp_path), 1, 10, layers=1)
    assert (case / "samples.json").read_bytes() == manifest


@pytest.mark.parametrize("probability", [-1, 0.5, float("nan"), float("inf")])
def test_invalid_probability_is_rejected_before_preparation(tmp_path, probability):
    from precompute_replay import prepare_width

    with pytest.raises(ValueError, match="probability"):
        prepare_width(str(tmp_path), 1, 10, probability=probability)
    assert not (tmp_path / "w1").exists()


def test_resume_preserves_a_campaign_with_another_probability(tmp_path, monkeypatch):
    import precompute_replay

    manifest = tmp_path / "sampling.json"
    original = json.dumps({"status": "complete", "probability": 0.001})
    manifest.write_text(original)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "precompute_replay.py",
            "--output",
            str(tmp_path),
            "--max-width",
            "1",
            "--probability",
            "1e-5",
            "--resume",
        ],
    )
    with pytest.raises(ValueError, match="different probability"):
        precompute_replay.main()
    assert manifest.read_text() == original


def test_preparation_rejects_oversized_shots_before_compilation(tmp_path):
    from precompute_replay import prepare_width

    with pytest.raises(ValueError, match="10 GB"):
        prepare_width(str(tmp_path), 1024, 10, layers=100_000)
    assert not (tmp_path / "w1024" / "circuit.deq").exists()


@pytest.fixture
def manifest(tmp_path: Path) -> dict:
    samples = []
    for index, data in enumerate((b"\x01\x00", b"\x00\x01")):
        path = tmp_path / f"shot-{index}.b8"
        path.write_bytes(data)
        samples.append(
            {
                "file": path.name,
                "batch_instance": index,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return {"status": "complete", "shots": 2, "measurements": 9, "samples": samples}


@pytest.mark.parametrize(
    "mutation", ["duplicate", "instance", "padding", "escape", "count", "bytes"]
)
def test_manifest_rejects_invalid_samples(
    tmp_path: Path, manifest: dict, mutation: str
):
    verify_samples(tmp_path, manifest)
    report = copy.deepcopy(manifest)
    if mutation == "duplicate":
        report["samples"][1].update(file=report["samples"][0]["file"])
    elif mutation == "instance":
        report["samples"][1]["batch_instance"] = 0
    elif mutation == "padding":
        data = b"\x00\x02"
        (tmp_path / report["samples"][1]["file"]).write_bytes(data)
        report["samples"][1]["sha256"] = hashlib.sha256(data).hexdigest()
    elif mutation == "escape":
        report["samples"][1]["file"] = "../other.b8"
    elif mutation == "count":
        report["measurements"] = True
    else:
        report["samples"][1]["bytes"] = 1
    with pytest.raises(ValueError):
        verify_samples(tmp_path, report)


def test_executor_startup_failure_releases_the_output_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import precompute_replay

    def fail(**kwargs):
        raise RuntimeError("executor startup failed")

    for key in (
        "TOKIO_WORKER_THREADS",
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "QDK_PYTHON_TELEMETRY",
    ):
        monkeypatch.setenv(key, "1")
    monkeypatch.setattr(precompute_replay, "ProcessPoolExecutor", fail)
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        ["precompute_replay.py", "--output", str(output), "--max-width", "1"],
    )
    with pytest.raises(RuntimeError, match="executor startup"):
        precompute_replay.main()
    with exclusive_output(output / ".active"):
        pass


def test_failed_checkpoint_does_not_leave_workers_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from concurrent.futures import Future
    import precompute_replay

    workers = []

    class FailedExecutor:
        def __init__(self, **kwargs):
            pass

        def submit(self, *args):
            workers.append(
                subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
            )
            future = Future()
            future.set_exception(RuntimeError("worker failed"))
            return future

        def shutdown(self, **kwargs):
            assert all(
                worker.poll() is not None for worker in workers
            ), "worker leaked during error reporting"

    def full_disk(*args):
        raise OSError("disk full")

    for key in (
        "TOKIO_WORKER_THREADS",
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "QDK_PYTHON_TELEMETRY",
    ):
        monkeypatch.setenv(key, "1")
    monkeypatch.setattr(precompute_replay, "ProcessPoolExecutor", FailedExecutor)
    monkeypatch.setattr(precompute_replay, "checkpoint", full_disk)
    monkeypatch.setattr(
        sys,
        "argv",
        ["precompute_replay.py", "--output", str(tmp_path / "run"), "--max-width", "1"],
    )
    try:
        with pytest.raises(OSError, match="disk full"):
            precompute_replay.main()
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
                worker.wait(timeout=5)
