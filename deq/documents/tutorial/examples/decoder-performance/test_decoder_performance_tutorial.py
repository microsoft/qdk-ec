"""Exercise adaptive stopping, standalone plotting, and current native configurations."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

from study_utils import population

EXAMPLE = Path(__file__).resolve().parent


def load_example(name: str):
    spec = importlib.util.spec_from_file_location(
        f"performance_{name}", EXAMPLE / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load_example("run")
plotter = load_example("plot")
summarizer = load_example("summarize")


def test_default_measurement_contract():
    settings = runner.Settings()
    assert (
        settings.threads,
        settings.probability,
        settings.distance,
        settings.layers,
    ) == (
        24,
        0.001,
        7,
        50,
    )
    assert settings.stop_mean_ms == 1_000
    assert settings.max_width is None
    assert len(runner.SERIES) == 4


def test_tutorial_deq_examples_compile_with_the_evaluated_surgery_library():
    from generate_circuits import compiled_program
    from lattice_surgery import surgery_library

    chapter = EXAMPLE.parents[1] / "chapters" / "decoder-performance.md"
    examples = re.findall(r"```deq\n(.*?)\n```", chapter.read_text(), re.DOTALL)
    assert len(examples) == 2
    brickwall, cnot = examples
    library = surgery_library(distance=7, probability=0.001, block_seconds=0.07)
    reference = compiled_program(library + "\n" + brickwall)
    library_without_cnot = library[: library.index("\nPROGRAM CNOT {")]
    assert reference
    assert (
        compiled_program(library_without_cnot + "\n" + cnot + "\n" + brickwall)
        == reference
    )


@pytest.fixture
def adaptive_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(runner, "provenance", lambda: {"test": True})
    scales = {"mwpf": 2, "mwpm": 1, "persistent": 0.25, "persistent_mwpf": 0.5}
    calls = []

    def worker(job, directory, *, retry_failed=False):
        calls.append((job["width"], job["task"]))
        if job["task"] == "prepare":
            assert "source_run" not in job
            return {}
        values = {
            region: population(
                [1_000 * job["width"] * scales[job["task"]]] * job["plan"][region]
            )
            for region in ("intermediate", "final")
        }
        return {
            "series": job["task"],
            "width": job["width"],
            "populations": values,
            "shots": job["plan"]["shots"],
            "wall_seconds": 1,
            "run_directory": str(tmp_path / f"absent-{job['width']}-{job['task']}"),
        }

    monkeypatch.setattr(runner, "run_worker", worker)
    output = tmp_path / "results.json"
    settings = runner.Settings(threads=2)
    result = runner.evaluate(settings, output)
    assert result["status"] == "complete"
    assert {key: value["width"] for key, value in result["stopped"].items()} == {
        "mwpf": 1,
        "mwpm": 2,
        "persistent": 8,
        "persistent_mwpf": 4,
    }
    for series, stop in result["stopped"].items():
        assert [width for width, task in calls if task == series][-1] == stop["width"]
    assert max(width for width, _ in calls) == 8
    assert runner.evaluate(settings, output, resume=True) == result
    with pytest.raises(FileExistsError):
        runner.evaluate(settings, output)
    return output


def test_adaptive_stopping_and_json_only_plot(adaptive_result: Path, tmp_path: Path):
    document = summarizer.load_results(adaptive_result)
    assert len(document["points"]) == 10
    data = summarizer.summary_path(adaptive_result)
    summary = json.loads(data.read_text())
    assert summary == summarizer.summarize(document)
    assert summary["provenance"] == document["provenance"]
    assert [(point["series"], point["width"]) for point in summary["points"]] == [
        ("mwpf", 1),
        ("mwpm", 1),
        ("mwpm", 2),
        ("persistent", 1),
        ("persistent", 2),
        ("persistent", 4),
        ("persistent", 8),
        ("persistent_mwpf", 1),
        ("persistent_mwpf", 2),
        ("persistent_mwpf", 4),
    ]
    assert data.stat().st_size < adaptive_result.stat().st_size
    plotter.plot(data, tmp_path / "figure.png")
    assert (tmp_path / "figure.png").stat().st_size > 1000
    assert not (tmp_path / "figure.svg").exists()
    assert not any(tmp_path.glob("absent-*"))
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import runpy, sys; sys.modules.update(deq=None, deq_runtime=None); "
            "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')",
            str(EXAMPLE / "plot.py"),
            "--data",
            str(data),
            "--output",
            str(tmp_path / "without-deq.png"),
        ],
        check=True,
        timeout=60,
    )
    assert (tmp_path / "without-deq.png").stat().st_size > 1000


def test_width_limit_stops_sampling_without_legend_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "provenance", lambda: {"test": True})
    widths = []

    def worker(job, directory, *, retry_failed=False):
        widths.append(job["width"])
        if job["task"] == "prepare":
            return {}
        return {
            "series": job["task"],
            "width": job["width"],
            "shots": job["plan"]["shots"],
            "populations": {
                region: population([1.0] * job["plan"][region])
                for region in ("intermediate", "final")
            },
        }

    monkeypatch.setattr(runner, "run_worker", worker)
    output = tmp_path / "results.json"
    document = runner.evaluate(runner.Settings(max_width=2), output)
    assert document["status"] == "complete"
    assert max(widths) == 2
    assert len(document["points"]) == 8
    assert all(stop["reason"] == "width_limit" for stop in document["stopped"].values())
    data = summarizer.summary_path(output)
    labels = []
    legend = plotter.plt.Figure.legend

    def capture_legend(figure, *args, **kwargs):
        assert kwargs["ncol"] == 2
        labels.extend(handle.get_label() for handle in kwargs["handles"])
        return legend(figure, *args, **kwargs)

    monkeypatch.setattr(plotter.plt.Figure, "legend", capture_legend)
    plotter.plot(data, tmp_path / "capped.png")
    assert len(labels) == 4
    assert set(labels) == {series["label"] for series in document["series"]}
    assert all("MWPF" in label for label in labels[::2])
    assert all("MWPM" in label for label in labels[1::2])
    document["stopped"]["persistent"]["reason"] = "threshold"
    with pytest.raises(ValueError, match="stopping record"):
        summarizer.summarize(document)


def test_uncapped_run_passes_1024_and_stops_only_above_latency_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "provenance", lambda: {"test": True})
    widths = []

    def worker(job, directory, *, retry_failed=False):
        widths.append(job["width"])
        if job["task"] == "prepare":
            return {}
        mean = 1000 if job["width"] <= 1024 else 1001
        return {
            "series": job["task"],
            "width": job["width"],
            "shots": job["plan"]["shots"],
            "populations": {
                region: population([float(mean)] * job["plan"][region])
                for region in ("intermediate", "final")
            },
        }

    monkeypatch.setattr(runner, "run_worker", worker)
    output = tmp_path / "results.json"
    settings = runner.Settings(threads=120, memory_gib=320, layers=1)
    document = runner.evaluate(settings, output)
    assert max(widths) == 2048
    assert document["status"] == "complete"
    assert all(
        stop["width"] == 2048 and stop["reason"] == "threshold"
        for stop in document["stopped"].values()
    )
    assert summarizer.load_results(output) == document
    assert plotter.load_data(summarizer.summary_path(output))["settings"]["max_width"] is None


def test_large_machine_cli_has_no_implicit_width_cap(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(runner, "evaluate", lambda settings, output, **kwargs: captured.append(settings))
    monkeypatch.setattr(sys, "argv", [
        "run.py", "--threads", "120", "--memory-gib", "320",
        "--task-timeout-seconds", "14400", "--results", str(tmp_path / "results.json"),
    ])
    runner.main()
    assert captured == [runner.Settings(threads=120, memory_gib=320, task_timeout_seconds=14400)]


@pytest.mark.parametrize("width", [0, 3])
def test_width_limit_requires_a_positive_power_of_two(width):
    with pytest.raises(ValueError, match="maximum width"):
        runner.Settings(max_width=width).validate()


@pytest.mark.parametrize(
    "mutation", ["incomplete", "mean", "extra-point", "missing-samples"]
)
def test_plot_rejects_incomplete_or_inconsistent_data(
    adaptive_result: Path, mutation: str
):
    document = json.loads(adaptive_result.read_text())
    if mutation == "incomplete":
        document["status"] = "running"
    elif mutation == "mean":
        document["points"][0]["populations"]["final"]["mean_ms"] = 1
    elif mutation == "extra-point":
        document["points"].append(document["points"][0])
    else:
        document["points"][0]["populations"]["final"]["latency_ms"] = []
    adaptive_result.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        summarizer.load_results(adaptive_result)


def test_summary_quantiles_span_the_samples_and_list_small_populations():
    values = [float(value) for value in range(1, 1001)]
    large = summarizer.summarize_population(values[::-1], 11)
    assert large["samples"] == 1000
    assert large["mean_ms"] == 500.5
    assert large["median_ms"] == 500.5
    assert large["quantiles_ms"][0] == 1 and large["quantiles_ms"][-1] == 1000
    assert len(large["quantiles_ms"]) == 11
    small = summarizer.summarize_population([3.0, 1.0, 2.0], 11)
    assert small["quantiles_ms"] == [1.0, 2.0, 3.0]
    assert summarizer.summarize_population([], 11)["quantiles_ms"] == []
    with_quantiles = {"points": [{"q": [1.0, 2.5e-05]}], "q": ["a", "b"]}
    text = summarizer.dumps(with_quantiles)
    assert '"q": [1.0, 2.5e-05]' in text and json.loads(text) == with_quantiles


def test_plot_rejects_raw_results(adaptive_result: Path, tmp_path: Path):
    with pytest.raises(ValueError, match="summarize.py"):
        plotter.plot(adaptive_result, tmp_path / "figure.png")


def test_explicit_partial_publication_labels_unfinished_curves(
    adaptive_result: Path, tmp_path: Path
):
    document = json.loads(adaptive_result.read_text())
    document["points"] = [
        p
        for p in document["points"]
        if not (p["series"] == "persistent" and p["width"] == 8)
    ]
    del document["stopped"]["persistent"]
    document["status"] = "stopped"
    with pytest.raises(ValueError, match="complete"):
        summarizer.summarize(document)
    output = tmp_path / "partial.json"
    summary = summarizer.write_summary(document, output, allow_incomplete=True)
    assert summary["incomplete_series"] == ["persistent"]
    assert (
        max(p["width"] for p in summary["points"] if p["series"] == "persistent") == 4
    )
    plotter.plot(output, tmp_path / "partial.png")
    document["points"][0]["populations"]["final"]["samples"] = 0
    with pytest.raises(ValueError, match="samples"):
        summarizer.summarize(document, allow_incomplete=True)


def test_evaluation_requires_all_four_configurations(adaptive_result: Path):
    document = json.loads(adaptive_result.read_text())
    document["series"].pop()
    with pytest.raises(ValueError, match="four decoder configurations"):
        summarizer.validate_results(document)
    summary = json.loads(summarizer.summary_path(adaptive_result).read_text())
    summary["series"].pop()
    path = adaptive_result.with_name("missing-series.json")
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="configurations"):
        plotter.load_data(path)


def test_summary_failure_does_not_leave_a_complete_checkpoint(
    adaptive_result: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = runner.Settings(**json.loads(adaptive_result.read_text())["settings"])
    write_summary = runner.write_summary
    before = json.loads(adaptive_result.read_text())

    def reject_summary(*args):
        raise ValueError("invalid final observations")

    monkeypatch.setattr(runner, "write_summary", reject_summary)
    with pytest.raises(ValueError, match="invalid final observations"):
        runner.evaluate(settings, adaptive_result, resume=True)
    result = json.loads(adaptive_result.read_text())
    assert result["status"] == "failed"
    assert result["failure"]["type"] == "ValueError"
    assert result["points"] == before["points"]
    monkeypatch.setattr(runner, "write_summary", write_summary)
    monkeypatch.setattr(
        runner,
        "run_worker",
        lambda *args, **kwargs: pytest.fail("completed points must not rerun"),
    )
    restored = runner.evaluate(settings, adaptive_result, resume=True)
    assert restored["status"] == "complete"
    assert restored["points"] == before["points"]


def test_evaluation_errors_are_persisted_and_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(runner, "provenance", lambda: {})

    def fail(*args, **kwargs):
        raise MemoryError("test resource guard")

    monkeypatch.setattr(runner, "run_worker", fail)
    output = tmp_path / "results.json"
    with pytest.raises(MemoryError):
        runner.evaluate(runner.Settings(threads=2), output)
    document = json.loads(output.read_text())
    assert document["status"] == "failed"
    assert document["failure"]["type"] == "MemoryError"
    assert document["points"] == []


def test_current_configurations_run_from_generated_inputs(tmp_path: Path):
    settings = runner.Settings(threads=2, distance=3, layers=2, interval_ms=2)
    job = {
        "settings": asdict(settings),
        "task": "prepare",
        "width": 4,
        "case": str(tmp_path / "circuit"),
    }
    configuration = runner.perform_task(job)
    job["plan"] = {
        "width": 4,
        "shots": 1,
        "final": 4,
        "intermediate": 3 * configuration["cnot_count"],
    }
    for series in runner.SERIES:
        job["task"] = series.key
        result = runner.perform_task(job)
        assert result["series"] == series.key
        assert result["populations"]["final"]["samples"] == 4
        assert result["input_kind"] == "physical_stim_sampling"
        command = json.loads(
            (Path(result["run_directory"]) / "command.json").read_text()
        )
        assert command[1:5] == ["-m", "deq", "simulate", "ler"]
        assert command[command.index("--jobs") + 1] == "1"
        assert command[command.index("--shots") + 1] == "1"
        assert command[command.index("--batch-size") + 1] == "1"
        assert command[command.index("--errors") + 1] == "2"
        assert "--strict-timing" in command
        assert "--trace" in command and "--simulator-trace-output" in command
        assert "--stim" not in command and "--jit" not in command


def test_prepare_worker_is_isolated_and_resumable(tmp_path: Path):
    settings = runner.Settings(threads=2, distance=3, layers=1)
    job = {
        "settings": asdict(settings),
        "task": "prepare",
        "width": 2,
        "case": str(tmp_path / "circuit"),
    }
    result = runner.run_worker(job, tmp_path / "tasks")
    assert result["width"] == 2
    assert runner.run_worker(job, tmp_path / "tasks") == result
    job["width"] = 4
    with pytest.raises(ValueError, match="incomplete or incompatible"):
        runner.run_worker(job, tmp_path / "tasks")


def test_task_timeout_covers_sampling_and_scheduled_shots():
    settings = runner.Settings(threads=24)
    job = {
        "settings": asdict(settings),
        "width": 128,
        "task": "mwpm",
    }
    assert runner.task_timeout(job) >= 3600
    job["width"] = 1
    assert (
        runner.task_timeout(job)
        > 100 * (3 * settings.layers + 2) * settings.interval_ms / 1000
    )
    job["settings"]["task_timeout_seconds"] = 14_400
    job["width"] = 2048
    assert runner.task_timeout(job) == 14_400


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_task_timeout_is_rejected(seconds):
    with pytest.raises(ValueError, match="task timeout"):
        runner.Settings(task_timeout_seconds=seconds).validate()


def test_retry_preserves_failed_outputs_and_restarts_only_that_task(tmp_path: Path):
    settings = runner.Settings(threads=2, distance=3, layers=1)
    case = tmp_path / "circuit"
    case.mkdir()
    (case / "partial.txt").write_text("failed preparation")
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    job = {
        "settings": asdict(settings),
        "task": "prepare",
        "width": 2,
        "case": str(case),
    }
    (tasks / "prepare.json").write_text(json.dumps(job))
    (tasks / "prepare.log").write_text("original failure")
    with pytest.raises(ValueError, match="incomplete task"):
        runner.run_worker(job, tasks)
    result = runner.run_worker(job, tasks, retry_failed=True)
    assert result["width"] == 2
    (archive,) = (tasks / "failed").iterdir()
    assert (archive / "circuit/partial.txt").read_text() == "failed preparation"
    assert (archive / "prepare.log").read_text() == "original failure"
    assert not (case / "partial.txt").exists()


def test_resume_requires_identical_sources_and_builds():
    import copy

    settings = runner.Settings(threads=2)
    key = str(Path(runner.__file__).resolve().relative_to(runner.DEQ))
    original = {
        "binary_sha256": {"runtime": "same"},
        "source_sha256": {key: "old", "workload.py": "unchanged"},
    }
    document = {
        "schema_version": 3,
        "input_protocol": runner.INPUT_PROTOCOL,
        "settings": asdict(settings),
        "series": [asdict(series) for series in runner.SERIES],
        "provenance": original,
        "points": [{"latency": 7}],
    }
    updated = copy.deepcopy(original)
    updated["source_sha256"][key] = "new"
    runner.validate_resume(document, settings, original)
    with pytest.raises(ValueError, match="unchanged"):
        runner.validate_resume(document, settings, updated)
    invalid = copy.deepcopy(updated)
    invalid["source_sha256"]["workload.py"] = "changed"
    with pytest.raises(ValueError, match="unchanged"):
        runner.validate_resume(document, settings, invalid)
    assert document["points"] == [{"latency": 7}]
    assert document["provenance"] == original
    document["schema_version"] = 1
    with pytest.raises(ValueError, match="native-timing input protocol"):
        runner.validate_resume(document, settings, original)
