"""Check the manual JIT benchmark's fixture generation and budget calculation."""

import gzip
import hashlib
import json
import math
from pathlib import Path
from shutil import copyfile
import subprocess
import sys
import zlib

import pytest
from deq.transpiler.program_timing import gadget_timing

import report_jit_benchmark
from generate_jit_benchmark_fixture import FIXTURES, compile_fixture
from report_jit_benchmark import FIXTURE_DIRECTORY, ROOT, check_estimate, collect, write_report


def test_relocated_tools_find_the_same_fixtures(tmp_path: Path):
    assert FIXTURES == FIXTURE_DIRECTORY == Path(__file__).resolve().parent / "fixtures"
    assert (ROOT / "deq/deq_runtime/Cargo.toml").is_file()
    library, metadata = compile_fixture(tmp_path / "small", 4, 2)
    repeated, repeated_metadata = compile_fixture(tmp_path / "repeated", 4, 2)
    assert library.SerializeToString(deterministic=True) == repeated.SerializeToString(deterministic=True)
    assert metadata == repeated_metadata
    assert metadata["gadgets"] == 47
    assert metadata["epochs"] == 8
    assert metadata["connected_circuit_verification"]["cnots"] == 3
    assert all(gadget_timing(instruction.gadget) is not None for instruction in library.program)
    completions = [
        gadget_timing(instruction.gadget).end
        for instruction in library.program
    ]
    assert completions == sorted(completions)
    assert max(completions) == metadata["epochs"] * 70_000_000
    source = (tmp_path / "small" / "circuit.deq").read_text()
    assert "@START(at=0.06,duration=0.22)\n    CNOT" in source
    assert "@START(at=0,duration=0.01)\n    PrepareXNonFT 2" in source


def test_generated_artifacts_are_ignored_and_separate_from_rust_sources():
    benches = ROOT / "deq/deq_runtime/benches"
    assert not list(benches.glob("*.py"))
    assert not (benches / "fixtures").exists()
    assert not (benches / "results").exists()
    artifacts = [
        str(FIXTURES / "manifest.json"),
        str(Path(__file__).with_name("results") / "jit_compiler_baseline.json"),
    ]
    ignored = subprocess.run(
        ["git", "check-ignore", "--", *artifacts],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert ignored.stdout.splitlines() == artifacts


def test_fixture_generation_needs_only_the_jit_example(tmp_path: Path):
    for name in ("generate_jit_benchmark_fixture.py", "lattice_surgery.py"):
        copyfile(Path(__file__).with_name(name), tmp_path / name)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path\n"
            "import sys\n"
            "from generate_jit_benchmark_fixture import compile_fixture\n"
            "library, metadata = compile_fixture(Path('case'), 4, 2)\n"
            "assert len(library.program) == metadata['gadgets'] == 47\n"
            "assert metadata['connected_circuit_verification']['cnots'] == 3\n"
            "assert not {'lattice_surgery_latency', 'circuit_validation'} & sys.modules.keys()\n",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=120,
    )


def test_report_writes_only_a_concise_text_summary(tmp_path: Path):
    report = {
        "recorded_utc": "test",
        "git_head": "test",
        "dirty_worktree": False,
        "cpu_model": "test",
        "available_cpus": 24,
        "rustc": "test",
        "provenance": {
            "rustflags": "",
            "compiler_threads": 24,
            "cache_limit_ratio": 10,
            "registered_gadget_types": 24,
            "error_template_cache_capacity": 240,
        },
        "fixture": {"scope": "compiler", "excluded": "decoding"},
        "measurements": [],
    }
    write_report(report, tmp_path)
    assert [path.name for path in tmp_path.iterdir()] == ["jit_compiler_baseline.md"]
    assert "local Criterion output" in (tmp_path / "jit_compiler_baseline.md").read_text()


def test_report_rejects_replaced_fixture_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "manifest.json").write_text("{}")
    (tmp_path / "jit_compiler_run.json").write_text(json.dumps({"fixture_manifest_crc32": 0}))
    monkeypatch.setattr(report_jit_benchmark, "FIXTURE_DIRECTORY", fixtures)
    with pytest.raises(ValueError, match="fixture manifest changed"):
        collect(tmp_path)


@pytest.fixture
def criterion_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    payload = b"fixture"
    (fixtures / "case.jit.gz").write_bytes(gzip.compress(payload))
    case = {
        "name": "case",
        "file": "case.jit.gz",
        "protobuf_sha256": hashlib.sha256(payload).hexdigest(),
        "width": 200,
        "layers": 100,
        "epochs": 302,
        "gadgets": 1,
        "checks": 1,
        "errors": 1,
    }
    manifest = {
        "compiler_threads": 24,
        "maximum_mean_epoch_ms": 390,
        "cases": [case],
        "scope": "test",
        "excluded": "test",
    }
    manifest_bytes = json.dumps(manifest).encode()
    (fixtures / "manifest.json").write_bytes(manifest_bytes)
    provenance = {
        "schema_version": 4,
        "optimized": True,
        "started_unix_seconds": 0,
        "compiler_threads": 24,
        "maximum_mean_epoch_ms": 390,
        "fixture_manifest_crc32": zlib.crc32(manifest_bytes),
        "compiled_source_crc32": {},
        "executed": ["cache_enabled/case", "cache_disabled/case"],
        "timed_out": [],
        "timeout_seconds": 360,
        "cache_limit_ratio": 10,
        "registered_gadget_types": 24,
        "error_template_cache_capacity": 240,
        "rustflags": "",
    }
    (tmp_path / "jit_compiler_run.json").write_text(json.dumps(provenance))
    for mode in report_jit_benchmark.CACHE_MODES:
        directory = tmp_path / "jit_compiler" / mode / "case/new"
        directory.mkdir(parents=True)
        value = estimate(1e9 if mode == "cache_enabled" else 1e12)
        (directory / "estimates.json").write_text(json.dumps({"mean": value}))
        (directory / "sample.json").write_text(
            json.dumps({"iters": [1] * 10, "times": [value["point_estimate"]] * 10})
        )
    monkeypatch.setattr(report_jit_benchmark, "FIXTURE_DIRECTORY", fixtures)
    return tmp_path


def test_only_cache_enabled_operation_is_budget_gated(criterion_results: Path):
    report = collect(criterion_results)
    assert report["budget_passed"]
    assert [row["budget_passed"] for row in report["measurements"]] == [True, False]
    assert "Control; not gated" in report_jit_benchmark.markdown(report)


def test_uncached_timeout_is_a_lower_bound_not_an_estimate(criterion_results: Path):
    path = criterion_results / "jit_compiler_run.json"
    provenance = json.loads(path.read_text())
    provenance["executed"] = ["cache_enabled/case"]
    provenance["timed_out"] = ["cache_disabled/case"]
    path.write_text(json.dumps(provenance))
    report = collect(criterion_results)
    assert report["budget_passed"]
    timeout = report["measurements"][1]
    assert timeout["status"] == "timed_out"
    assert timeout["timeout_seconds"] == 360
    assert "mean_seconds" not in timeout
    assert "estimates" not in timeout
    assert ">360.000 (timeout)" in report_jit_benchmark.markdown(report)


@pytest.mark.parametrize(
    "invalid",
    ["incomplete", "overlap", "enabled-timeout", "invalid-timeout", "zero-capacity", "zero-ratio", "mismatched-capacity"],
)
def test_incomplete_or_inconsistent_runs_are_rejected(
    criterion_results: Path, invalid: str
):
    path = criterion_results / "jit_compiler_run.json"
    provenance = json.loads(path.read_text())
    if invalid == "incomplete":
        provenance["executed"].pop()
    elif invalid == "overlap":
        provenance["timed_out"] = ["cache_disabled/case"]
    elif invalid == "enabled-timeout":
        provenance["executed"] = ["cache_disabled/case"]
        provenance["timed_out"] = ["cache_enabled/case"]
    elif invalid == "invalid-timeout":
        provenance["timeout_seconds"] = 0
    elif invalid == "zero-capacity":
        provenance["error_template_cache_capacity"] = 0
    elif invalid == "zero-ratio":
        provenance["cache_limit_ratio"] = 0
    else:
        provenance["error_template_cache_capacity"] = 128
    path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError):
        collect(criterion_results)


@pytest.mark.parametrize("name", ["generate_jit_benchmark_fixture", "report_jit_benchmark"])
def test_tool_entry_points_work_outside_the_checkout(tmp_path: Path, name: str):
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name(f"{name}.py")), "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=30,
    )


def estimate(upper_ns: float) -> dict:
    return {
        "point_estimate": upper_ns * 0.9,
        "confidence_interval": {
            "confidence_level": 0.95,
            "lower_bound": upper_ns * 0.8,
            "upper_bound": upper_ns,
        },
    }


def test_exact_two_hundred_width_budget_boundary():
    limit_ns = 302 * 390 * 1e6
    result = check_estimate(estimate(limit_ns), epochs=302, maximum_mean_epoch_ms=390)
    assert result["budget_passed"]
    assert result["upper_mean_epoch_ms"] == 390
    result = check_estimate(
        estimate(limit_ns + 1), epochs=302, maximum_mean_epoch_ms=390
    )
    assert not result["budget_passed"]


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -1, 0])
def test_invalid_estimates_are_not_performance_successes(invalid: float):
    with pytest.raises(ValueError):
        check_estimate(estimate(invalid), epochs=302, maximum_mean_epoch_ms=390)


def test_low_confidence_and_invalid_epoch_counts_are_rejected():
    value = estimate(1000)
    value["confidence_interval"]["confidence_level"] = 0.90
    with pytest.raises(ValueError, match="95%"):
        check_estimate(value, epochs=302, maximum_mean_epoch_ms=390)
    with pytest.raises(ValueError, match="positive"):
        check_estimate(estimate(1000), epochs=0, maximum_mean_epoch_ms=390)
