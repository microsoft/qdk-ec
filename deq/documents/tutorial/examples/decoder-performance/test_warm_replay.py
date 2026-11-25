"""Exercise packed replay and compact timing through all four native paths."""

import hashlib
import ctypes
import json
from pathlib import Path
import shutil
import sys

import pytest

from deq.proto.deq_bin_pb2 import Library
from replay_qsharp import build_recipe
from replay_stim import sample_batch
from run import SERIES
from warm_replay import run_point


@pytest.mark.parametrize(
    ("series", "diagnostics", "compiled"),
    [(series, False, False) for series in SERIES]
    + [(next(series for series in SERIES if series.key == "persistent"), True, False)]
    + [
        (
            next(series for series in SERIES if series.key == "persistent"),
            diagnostic,
            True,
        )
        for diagnostic in (False, True)
    ],
    ids=[series.key for series in SERIES]
    + ["persistent-diagnostics", "compiled", "compiled-noisy"],
)
def test_native_warm_replay_requires_no_stim_file(
    tmp_path: Path, series, diagnostics, compiled, monkeypatch
):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jit-basics"))
    from lattice_surgery import compile_case

    if diagnostics:
        monkeypatch.setenv("GLIBC_TUNABLES", "glibc.malloc.mxfast=0")
    case = tmp_path / "case"
    probability = 0.01 if diagnostics else 0
    configuration = compile_case(
        case,
        distance=3,
        probability=probability,
        width=2,
        layers=2,
        basis="Z",
        seed=5,
        block_seconds=0.02 if diagnostics else 0.002,
        export_physical=False,
    )
    recipe = build_recipe(case / "circuit.deq", case / "circuit.deq.jit", probability)
    samples = []
    for index, data in enumerate(
        sample_batch(recipe, probability, shots=10, seed=42, reference_seed=71)
    ):
        path = case / f"shot-{index}.b8"
        path.write_bytes(data)
        samples.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "batch_instance": index,
                "bytes": len(data),
            }
        )
    (case / "samples.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "shots": 10,
                "measurements": configuration["measurements"],
                "samples": samples,
                "input_sha256": {
                    name: hashlib.sha256((case / name).read_bytes()).hexdigest()
                    for name in ("circuit.deq", "circuit.deq.jit", "configuration.json")
                },
            }
        )
    )
    trim_heap = (
        series.key == "persistent"
        and sys.platform == "linux"
        and hasattr(ctypes.CDLL(None), "malloc_trim")
    )
    static_input = None
    if diagnostics:
        from warm_replay import _native_path, compile_static_file, file_hash

        fixed = tmp_path / "fixed-program.bin"
        compile_static_file(str(case / "circuit.deq.jit"), str(fixed))
        static_input = {
            "path": str(fixed),
            "compiler_sha256": file_hash(_native_path()),
        }
    result = run_point(
        case,
        tmp_path / "runs" / series.key,
        series,
        {
            "warmup_shots": 1,
            "measured_shots": 2,
            "pools": [],
            "threads": 1,
            "tokio_threads": 2,
            "frontend_cpus": None,
            "memory_gib": 8,
            "task_timeout_seconds": 60,
            "trim_heap": trim_heap,
            "probability": probability,
            "system_diagnostics": diagnostics,
            "coordinator": "compiled-window" if compiled else "window",
            "compiled_worker_threads": 2,
            "numa_policy": (
                "local" if diagnostics and shutil.which("numactl") else "inherit"
            ),
            **({"static_input": static_input} if static_input is not None else {}),
            **({"decode_task_batch_size": 1} if diagnostics and not compiled else {}),
        },
    )
    if not diagnostics:
        assert (
            result["logical_error_shots"] == result["warmup_logical_error_shots"] == 0
        )
    assert result["native_shots"] == 3 and result["shots"] == 2
    assert result["populations"]["final"]["samples"] == 4
    assert result["populations"]["intermediate"]["samples"] == 6
    assert all(row["expected"] == row["observed"] for row in result["coverage"])
    assert not list(tmp_path.rglob("*.stim"))
    if diagnostics:
        from deq.proto.fastrace_pb2 import SpanRecord
        from readout_latency import protobuf_records

        records = [
            SpanRecord.FromString(data)
            for data in protobuf_records(tmp_path / "runs" / series.key / "timing.pb")
        ]
        expected_records = {"input_delivery", "static_batch", "decoder_job"}
        expected_records.add("compiled_readout" if compiled else "frame_operation")
        assert expected_records <= {record.name for record in records}
        for record in records:
            if record.name == "frame_operation":
                properties = {value.key: value.value for value in record.properties}
                offsets = [
                    int(properties[f"{stage}_offset_ns"])
                    for stage in (
                        "submission_started",
                        "worker_started",
                        "worker_finished",
                        "caller_resumed",
                    )
                ]
                assert offsets == sorted(offsets)
            if record.name != "decoder_job":
                continue
            properties = {value.key: value.value for value in record.properties}
            assert properties["timing_transport"] == "lossless"
            assert properties["solver_success"] == "true"
            offsets = [
                int(properties[f"{stage}_offset_ns"])
                for stage in (
                    "admitted",
                    "worker_started",
                    "initialized",
                    "solver_started",
                    "solver_returned",
                    "solve_finished",
                    "worker_finished",
                )
            ]
            assert offsets == sorted(offsets)
        command_metadata = json.loads(
            (tmp_path / "runs" / series.key / "command.json").read_text()
        )
        assert (
            command_metadata["environment"]["GLIBC_TUNABLES"] == "glibc.malloc.mxfast=0"
        )
        command = command_metadata["command"]
        if not compiled:
            assert (
                json.loads(command[command.index("--controller-config") + 1])[
                    "decode_task_batch_size"
                ]
                == 1
            )
        for record in records:
            if record.name == "gadget_polls":
                properties = {value.key: value.value for value in record.properties}
                assert properties["completed"] == "true"
                assert int(properties["polls"]) > 0
                assert int(properties["wake_delay_ns"]) >= int(
                    properties["max_wake_delay_ns"]
                )
        if shutil.which("numactl"):
            assert command[1] == "--localalloc"
    if series.static:
        library = Library.FromString(
            (tmp_path / "runs" / series.key / "Brickwall.deq.bin").read_bytes()
        )
        assert len(library.check_model_types) < configuration["gadgets"]
        assert len(library.error_model_types) < configuration["gadgets"]
    if series.key == "persistent":
        from plot_latency_distribution import generate_report

        figure = tmp_path / "intermediate-latency.png"
        generate_report(tmp_path / "runs" / series.key, figure)
        assert figure.read_bytes().startswith(b"\x89PNG")
        distribution = json.loads(figure.with_suffix(".json").read_text())
        assert distribution["probability"] == probability
        assert distribution["intermediate"]["samples"] == 6
        assert distribution["final"]["samples"] == 4
        if trim_heap:
            records = [
                json.loads(line)
                for line in (tmp_path / "runs" / series.key / "allocator.jsonl")
                .read_text()
                .splitlines()
            ]
            assert [row["sample_call"] for row in records] == [0, 1, 2]


@pytest.mark.parametrize("mismatched_settings", [False, True])
@pytest.mark.parametrize("single_depth", [False, True])
def test_complete_driver_writes_only_matching_warm_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatched_settings: bool,
    single_depth: bool,
):
    import warm_replay
    import plot_latency_distribution

    distributions = []
    monkeypatch.setattr(
        plot_latency_distribution,
        "generate_report",
        lambda run, output: distributions.append((run, output)),
    )

    inputs = tmp_path / "samples"
    inputs.mkdir()
    (inputs / "sampling.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "shots_per_width": 10,
                "widths": [2, 1],
                "points": {"1": {}, "2": {}},
                "layers": 1000 if single_depth else 50,
                **({"probability": 1e-5} if single_depth else {}),
            }
        )
    )
    monkeypatch.setattr(warm_replay, "verify_samples", lambda *_: None)
    monkeypatch.setattr(
        warm_replay, "provenance", lambda: {"binary_sha256": {"python_runtime": "test"}}
    )
    monkeypatch.setattr(
        warm_replay.os, "sched_getaffinity", lambda _: set(range(120)), raising=False
    )

    def point(case, directory, series, settings):
        assert settings["layers"] == (1000 if single_depth else 50)
        assert settings["probability"] == (1e-5 if single_depth else 0.001)
        assert settings["numa_policy"] == ("local" if single_depth else "inherit")
        width = int(case.name[1:])
        groups = {
            "final": [2.0] * (10 * width),
            "intermediate": (
                []
                if width == 1
                else [3.0]
                * (3 * ((settings["layers"] + 1) // 2) * settings["measured_shots"])
            ),
        }
        return {
            "series": series.key,
            "width": width,
            "shots": 10,
            "warmup_shots": 2,
            "native_shots": 12,
            "execution_settings": (
                {**settings, "threads": 1} if mismatched_settings else settings
            ),
            "runtime_sha256": "test",
            "run_directory": str(directory),
            "populations": {
                name: warm_replay.population(values) for name, values in groups.items()
            },
        }

    monkeypatch.setattr(warm_replay, "run_point", point)
    output = tmp_path / "evaluation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "warm_replay.py",
            "--samples",
            str(inputs),
            "--output",
            str(output),
            "--max-width",
            "2",
            *(
                [
                    "--min-width",
                    "2",
                    "--layers",
                    "1000",
                    "--probability",
                    "1e-5",
                    "--series",
                    "persistent",
                    "--numa-policy",
                    "local",
                ]
                if single_depth
                else []
            ),
        ],
    )
    if mismatched_settings:
        with pytest.raises(ValueError, match="different execution settings"):
            warm_replay.main()
        assert not (output / "performance.png").exists()
        assert json.loads((output / "results.json").read_text())["status"] == "stopped"
        return
    warm_replay.main()
    document = json.loads((output / "performance-data.json").read_text())
    assert document["status"] == "complete" and len(document["points"]) == (
        1 if single_depth else 8
    )
    assert document["settings"]["warmup_shots"] == 2
    assert document["settings"]["measured_shots"] == 10
    assert all(stop["reason"] == "width_limit" for stop in document["stopped"].values())
    assert (output / "performance.png").read_bytes().startswith(b"\x89PNG")
    assert distributions == (
        [(output / "work/w2/persistent", output / "intermediate-latency.png")]
        if single_depth
        else []
    )


def test_replay_rejects_a_different_sampling_probability(tmp_path, monkeypatch):
    import warm_replay

    (tmp_path / "sampling.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "shots_per_width": 10,
                "widths": [1],
                "probability": 0.001,
            }
        )
    )
    monkeypatch.setattr(
        warm_replay.os, "sched_getaffinity", lambda _: set(range(120)), raising=False
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "warm_replay.py",
            "--samples",
            str(tmp_path),
            "--output",
            str(tmp_path / "output"),
            "--max-width",
            "1",
            "--probability",
            "1e-5",
        ],
    )
    with pytest.raises(ValueError, match="different physical error probability"):
        warm_replay.main()
    assert not (tmp_path / "output" / "work").exists()


def test_explicit_numa_policy_never_silently_falls_back(tmp_path, monkeypatch):
    import warm_replay

    monkeypatch.setattr(warm_replay.shutil, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="numactl"):
        run_point(tmp_path, tmp_path / "output", SERIES[2], {"numa_policy": "local"})
    with pytest.raises(ValueError, match="numa_policy"):
        run_point(tmp_path, tmp_path / "output", SERIES[2], {"numa_policy": "unknown"})


def test_static_cache_is_bound_to_source_and_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import warm_replay
    from deq.proto.deq_jit_pb2 import JitLibrary

    monkeypatch.setenv("TOKIO_WORKER_THREADS", "4")
    monkeypatch.setenv("RAYON_NUM_THREADS", "2")
    source = tmp_path / "program.jit"
    source.write_bytes(JitLibrary().SerializeToString())
    cached = tmp_path / "program.bin"
    warm_replay.compile_static_file(str(source), str(cached))
    warm_replay.verify_static_cache(source, cached)
    producer = warm_replay.file_hash(warm_replay._native_path())
    with monkeypatch.context() as consumer:
        other_runtime = tmp_path / "other-runtime"
        other_runtime.write_bytes(b"other consumer runtime")
        consumer.setattr(warm_replay, "_native_path", lambda: other_runtime)
        with pytest.raises(ValueError, match="Static library cache"):
            warm_replay.verify_static_cache(source, cached)
        warm_replay.verify_static_cache(source, cached, compiler_sha256=producer)
        with pytest.raises(ValueError, match="Static library cache"):
            warm_replay.verify_static_cache(source, cached, compiler_sha256="0" * 64)
    original = source.read_bytes()
    source.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="Static library cache"):
        warm_replay.verify_static_cache(source, cached)
    source.write_bytes(original)
    cached.write_bytes(cached.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Static library cache"):
        warm_replay.verify_static_cache(source, cached)
