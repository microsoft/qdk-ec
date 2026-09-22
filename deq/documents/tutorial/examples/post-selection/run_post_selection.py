"""Run capacity or circuit post-selection studies with local or remote Dask."""

import argparse
from datetime import datetime, timezone
import fcntl
from functools import wraps
import json
import os
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from post_selection_stats import (
    FIXTURE,
    ScoreGroup,
    count_selection,
    merge_score_groups,
    trace_statistics,
)
from deq.circuit.mako_support import read_and_render_file
from deq.transpiler.loss import NeutralAtomLossModel
from post_selection_plots import (
    FIGURES,
    SELECTION_STATISTICS,
    precision_summary,
    render_native,
    publish_figures,
    weight_points,
)
from simulation import (
    DEQ_ROOT,
    WORKER_ENVIRONMENT,
    compile_circuit,
    digest,
    load_trace,
    run_logged,
    sample_batches,
    source_identity,
    worker_function,
    write_json,
)


PROGRAM = "SteaneZMemory"
CASES = ("monolithic", "window-r0", "window-r1", "window-r2", "window-r3", "window-r4", "window-r5", "window-r6")
CAPACITY_PROGRAM = "CodeCapacityZMemory"
CAPACITY_CASES = ("capacity-pauli", "capacity-mixed")
BACKENDS = {"tesseract": "black-box-tesseract"}
DEFAULT_PHYSICAL_ERROR_RATE = 0.007
DEFAULT_SHOTS = 100_000
DEFAULT_BATCH_SIZE = 25
DEFAULT_JOBS = min(20, os.cpu_count() or 1)
TESSERACT_CONFIG = {
    "parallel": 1,
    "pqlimit": 200_000,
    "det_beam": 5,
    "beam_climbing": False,
    "det_penalty": 0.0,
}
GAP_CONFIG = {"det_penalty": 30, "pqlimit": 2000, "det_beam": 2, "beam_climbing": False}
SURFACE_FIXTURE = DEQ_ROOT / "tests/circuit/surface_code/surface_code.deq"


def prepare_surface_circuit(rounds: int, physical_error_rate: float, output: Path) -> Path:
    from deq.noise import inject_si1000

    if rounds < 1 or not 0 <= physical_error_rate < 0.5:
        raise ValueError("positive rounds and physical_error_rate in [0, 0.5) are required")
    source = read_and_render_file(str(SURFACE_FIXTURE), mako_defs={"d": "5", "rounds": str(rounds)})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(inject_si1000(source, physical_error_rate))
    return output


def case_configuration(name: str, scored: bool = True, *, window_parallelism: str = "sliding") -> tuple[str, dict]:
    if name not in (*CASES, *CAPACITY_CASES):
        raise ValueError(f"unknown coordinator case: {name}")
    if window_parallelism not in ("sliding", "fully_parallel"):
        raise ValueError(f"unknown window parallelism: {window_parallelism}")
    config = {
        "forced_gap": scored,
        "merge_hyperedges": True,
        "assert_parity_factor": True,
        "loss_random_imputation_seed": 17,
        "loss_strategy": "reweight",
        "loss_config": {"weight_fraction": 0.5, "scale": "local"},
    }
    if name == "monolithic" or name in CAPACITY_CASES:
        return "monolithic", config
    config.update(
        window_parallelism=window_parallelism,
        buffer_radius=int(name.removeprefix("window-r")),
        lookahead_radius=0,
        forced_gap_strategy="lazy",
    )
    return "window", config


def compilation_directory(root: Path, name: str, *, capacity: bool = False) -> Path:
    directory = root / "setup/compile"
    return directory / name if capacity else directory


def prepare_circuit(
    rounds: int,
    physical_error_rate: float,
    output: Path,
    *,
    loss_fraction: float = 0.7,
) -> Path:
    if rounds < 1:
        raise ValueError("positive rounds are required")
    if not 0 <= physical_error_rate < 0.5:
        raise ValueError("physical_error_rate must be in [0, 0.5)")
    if not 0 <= loss_fraction <= 1:
        raise ValueError("loss_fraction must be in [0, 1]")
    circuit = read_and_render_file(
        str(FIXTURE),
        mako_defs={
            "p": str(physical_error_rate),
            "rounds": str(rounds),
            "loss_fraction": str(loss_fraction),
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(circuit)
    return output


def decoder_configuration(backend: str = "tesseract", config: dict | None = None) -> dict:
    if backend != "tesseract":
        raise ValueError("the post-selection evaluation requires Tesseract")
    if config is not None and not isinstance(config, dict):
        raise ValueError("decoder_config must be a JSON object")
    return {**TESSERACT_CONFIG, **(config or {})}


def gap_configuration(backend: str, decoder: str | None, config: dict | None) -> dict:
    if decoder is None and config is None:
        return {}
    if config is not None and not isinstance(config, dict):
        raise ValueError("gap_decoder_config must be a JSON object")
    if config is not None and "parallel" in config:
        raise ValueError(
            "gap_decoder_config must not contain 'parallel'; "
            "the shared thread pool is configured by the hard decoder"
        )
    return {
        "gap_decoder": decoder or BACKENDS[backend],
        "gap_decoder_config": {} if config is None else config.copy(),
    }


def load_batch(directory: Path, shots: int, *, final_readouts: int = 2) -> dict:
    trace = load_trace(directory / "shots.pb", shots)
    return {
        "counts": count_selection(trace),
        "groups": {
            name: [vars(group) for group in groups]
            for name, groups in trace_statistics(trace, final_readouts=final_readouts).items()
        },
    }


def serialized_batch(function):
    @wraps(function)
    def locked(root, build, name, start, *arguments, **options):
        lock_path = root / name / "locks" / f"{start:09d}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return function(root, build, name, start, *arguments, _lock_fd=lock.fileno(), **options)

    return locked


@serialized_batch
def run_batch(
    root: Path,
    build: Path,
    name: str,
    start: int,
    shots: int,
    seed: int,
    rounds: int,
    *,
    backend: str = "tesseract",
    timeout_seconds: int = 3600,
    sampling: dict | None = None,
    gap_decoder: str | None = None,
    gap_decoder_config: dict | None = None,
    program: str = PROGRAM,
    _lock_fd: int | None = None,
    accept_failure: bool = False,
    _compiled_sha256: dict | None = None,
    decoder_config: dict | None = None,
    window_parallelism: str = "sliding",
    code: str = "fire-ice",
    _recorded_coordinator: tuple[str, dict] | None = None,
) -> tuple[dict, dict]:
    decoder_config = decoder_configuration(backend, decoder_config)
    gap = gap_configuration(backend, gap_decoder, gap_decoder_config)
    simulator = "static" if code == "surface-code" else "python"
    sampler_config = {} if code == "surface-code" else {
        "sampler": "@qdk_sampler",
        "py_config": {
            "batch_size": min(shots, 64),
            "loss_config": NeutralAtomLossModel.config.to_json_object(),
        },
    }
    directory = root / name / "chunks" / f"{start:09d}"
    directory.mkdir(parents=True, exist_ok=True)
    coordinator, config = _recorded_coordinator or case_configuration(name, window_parallelism=window_parallelism)
    controller_config = {
        "filepath": str(build / f"{program}.deq.bin"),
        "reset_decoder_service": coordinator == "window",
    }
    identity = {
        "trace_schema": "per-gadget-statistics-v1",
        "code": code,
        "program": program,
        "simulator": simulator,
        "sampler_config": sampler_config,
        "case": name,
        "start": start,
        "shots": shots,
        "seed": seed,
        "rounds": rounds,
        "compiled_sha256": _compiled_sha256 if _compiled_sha256 is not None else {
            suffix: digest(build / f"{program}{suffix}")
            for suffix in (".stim", ".deq.bin")
        },
        "coordinator": coordinator,
        "coordinator_config": config,
        "controller_config": controller_config,
        "worker_address_space_bytes": 4 * 1024**3,
        "worker_timeout_seconds": timeout_seconds,
        "decoder": BACKENDS[backend],
        "decoder_config": decoder_config,
        **gap,
    }
    record_path = directory / "record.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record["identity"] != identity or any(
            digest(directory / name) != value for name, value in record["files"].items()
        ):
            raise ValueError("batch identity or data changed")
        return record, load_batch(directory, shots, final_readouts=1 if code == "surface-code" else 2)
    failure_path = directory / "failure.json"
    if accept_failure and failure_path.exists():
        failure = json.loads(failure_path.read_text())
        if failure["identity"] != identity or not failure.get("returncode"):
            raise ValueError("failed batch identity changed")
        outcome_path = directory / "outcome.json"
        fingerprints = {
            filename: digest(directory / filename)
            for filename in ("failure.json", "runtime.log")
        }
        record = {
            "identity": identity,
            "seconds": failure.get("seconds", timeout_seconds),
            "execution_failure": {
                "returncode": failure["returncode"],
                "kind": "timeout" if failure["returncode"] == 124 else "native_exit",
                "seed": seed,
                "log": str(directory / "runtime.log"),
            },
            "files": fingerprints,
        }
        if outcome_path.exists():
            saved = json.loads(outcome_path.read_text())
            if saved != record:
                raise ValueError("failed batch evidence changed")
        else:
            write_json(record, outcome_path)
        return record, {
            "counts": [shots, 0, 0],
            "groups": {statistic: [] for statistic in ("gap", "final_gap", *SELECTION_STATISTICS)},
        }
    if _recorded_coordinator is not None:
        raise ValueError("recorded batch is missing; recovery must not run decoders")
    if any(directory.iterdir()):
        interrupted = root / name / "interrupted" / f"{start:09d}-{time.time_ns()}"
        interrupted.parent.mkdir(parents=True, exist_ok=True)
        directory.rename(interrupted)
        directory.mkdir()
        print(
            f"Archived uncommitted batch to {interrupted}; retrying seed {seed}.",
            flush=True,
        )
    options = {
        "addr": "[::]:0",
        "decoder": BACKENDS[backend],
        "decoder-config": decoder_config,
        **{name.replace("_", "-"): value for name, value in gap.items()},
        "coordinator": coordinator,
        "coordinator-config": config,
        "controller": "static",
        "controller-config": controller_config,
        "simulator": simulator,
        "simulator-config": {
            **sampler_config,
            "filepath": str(build / f"{program}.stim"),
            "shots": shots,
            "errors": shots + 1,
            "seed": seed,
            "simulator_trace_output": str(directory / "shots.pb"),
        },
    }
    command = [
        "prlimit",
        f"--as={identity['worker_address_space_bytes']}",
        "--",
        "timeout",
        "--kill-after=10s",
        str(timeout_seconds),
        sampling["python"] if sampling else sys.executable,
        "-m",
        "deq.runtime",
        "server",
    ]
    command.extend(
        value
        for key, item in options.items()
        for value in (f"--{key}", json.dumps(item) if isinstance(item, dict) else item)
    )
    started = time.monotonic()
    execution = (
        {"cwd": sampling["cwd"], "environment": sampling["environment"]}
        if sampling
        else {}
    )
    returncode = run_logged(
        command, directory / "runtime.log",
        pass_fds=(_lock_fd,) if _lock_fd is not None else (), **execution,
    )
    if returncode:
        write_json(
            {"identity": identity, "returncode": returncode,
             "seconds": time.monotonic() - started, "command": command,
             "cwd": str(execution.get("cwd", DEQ_ROOT)),
             "environment": execution.get("environment", WORKER_ENVIRONMENT),
             "host": socket.gethostname()}, directory / "failure.json"
        )
        raise RuntimeError(f"native batch failed with exit {returncode}: {directory}")
    result = load_batch(directory, shots, final_readouts=1 if code == "surface-code" else 2)
    record = {
        "identity": identity,
        "seconds": time.monotonic() - started,
        "host": socket.gethostname(),
        "python": sys.executable,
        "files": {"shots.pb": digest(directory / "shots.pb")},
    }
    write_json(record, record_path)
    return record, result


def render(
    summary: dict, output: Path, selection_statistic: str = "correction_count", *,
    data_dir: Path | None = None, final_output: Path | None = None,
) -> None:
    publish_figures(summary, data_dir, output, final_output, selection_statistic)



def summarize(
    root: Path,
    parameters: dict,
    batches: dict,
    records: dict,
    complete: bool,
    *,
    status: str | None = None,
) -> dict:
    summary = {
        **parameters,
        "trace_schema": "per-gadget-statistics-v1",
        "status": status or ("complete" if complete else "running"),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases": [],
    }
    for name, completed in batches.items():
        if not completed:
            continue
        attempted, retained, errors = [
            sum(batch["counts"][index] for batch in completed.values())
            for index in range(3)
        ]
        grouped = merge_score_groups(batch["groups"] for batch in completed.values())
        points = weight_points(
            [ScoreGroup(**group) for group in grouped["correction_count"]],
            attempted_shots=attempted,
        )
        execution_failures = {
            start: records[name, start]
            for start in completed if records[name, start].get("execution_failure")
        }
        execution_failed_shots = sum(record["identity"]["shots"] for record in execution_failures.values())
        case = {
            "name": name,
            "shots": attempted,
            "decoded_shots": retained,
            "failed_shots": attempted - retained,
            "execution_failed_shots": execution_failed_shots,
            "decode_failed_shots": attempted - retained - execution_failed_shots,
            "logical_errors": errors,
            "raw_ler": errors / retained if retained else None,
            "groups": grouped,
            "count_thresholds": points,
            "batch_records": [
                str(Path(name) / "chunks" / f"{start:09d}" / "record.json")
                for start in sorted(completed) if start not in execution_failures
            ],
            "failure_records": [
                str(Path(name) / "chunks" / f"{start:09d}" / "outcome.json")
                for start in sorted(execution_failures)
            ],
            "worker_seconds": sum(
                records[name, start]["seconds"] for start in completed
            ),
        }
        if "case_loss_fractions" in parameters:
            case["loss_fraction"] = parameters["case_loss_fractions"][name]
        point = (
            min(points, key=lambda point: abs(point["rejected_percent"] - 10))
            if points
            else None
        )
        case["precision"] = precision_summary(grouped["gap"], point)
        summary["cases"].append(case)
    write_json(summary, root / "summary.json")
    return summary


def load_completed_batches(root: Path, parameters: dict) -> tuple[dict, dict]:
    batches = {name: {} for name in parameters["configurations"]}
    records = {}
    compiled_hashes = {}
    for name in batches:
        build = compilation_directory(root, name, capacity=parameters.get("noise_model") == "capacity") / "build"
        chunks = root / name / "chunks"
        paths = {path.parent: path for path in chunks.glob("*/failure.json")}
        paths.update({path.parent: path for path in chunks.glob("*/record.json")})
        for path in sorted(paths.values()):
            if build not in compiled_hashes:
                compiled_hashes[build] = {
                    suffix: digest(build / f"{parameters.get('program', PROGRAM)}{suffix}")
                    for suffix in (".stim", ".deq.bin")
                }
            identity = json.loads(path.read_text())["identity"]
            start = int(path.parent.name)
            shots = identity["shots"]
            if (
                start < 0
                or start % parameters["batch_size"]
                or not 0 < shots <= parameters["batch_size"]
            ):
                raise ValueError(f"invalid batch checkpoint: {path}")
            record, result = run_batch(
                root,
                build,
                name,
                start,
                shots,
                parameters["seed"] + start // parameters["batch_size"],
                parameters["rounds"],
                timeout_seconds=identity["worker_timeout_seconds"],
                gap_decoder=parameters.get("gap_decoder"),
                gap_decoder_config=parameters.get("gap_decoder_config"),
                program=parameters.get("program", PROGRAM),
                accept_failure=True,
                _compiled_sha256=compiled_hashes[build],
                decoder_config=parameters.get("decoder_config"),
                _recorded_coordinator=parameters["configurations"][name],
                code=parameters.get("code", "fire-ice"),
            )
            records[name, start] = record
            batches[name][start] = result
    return batches, records


def recover_summary(root: Path) -> dict:
    """Rebuild stopped progress from committed traces without running decoders."""
    root = root.resolve()
    with (root / "runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return json.loads((root / "summary.json").read_text())
        parameters = json.loads((root / "manifest.json").read_text())["parameters"]
        options = root / "run-options.json"
        if options.exists():
            parameters.update(json.loads(options.read_text()))
        batches, records = load_completed_batches(root, parameters)
        if not records:
            raise ValueError("no completed batches are available to plot yet")
        previous_path = root / "summary.json"
        previous = (
            json.loads(previous_path.read_text()) if previous_path.exists() else {}
        )
        complete = all(
            sum(batch["counts"][0] for batch in completed.values())
            >= parameters["shots"]
            for completed in batches.values()
        )
        status = "complete" if complete else previous.get("status", "paused")
        if not complete and status not in {"paused", "stopped"}:
            status = "paused"
        if previous.get("stop_reason"):
            parameters["stop_reason"] = previous["stop_reason"]
        return summarize(root, parameters, batches, records, complete, status=status)


def run(
    output_dir: Path,
    figure_output: Path,
    shots: int | None = None,
    jobs: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    rounds: int = 10,
    physical_error_rate: float | None = None,
    selection_statistic: str = "correction_count",
    cases: tuple[str, ...] | None = None,
    backend: str = "tesseract",
    max_seconds: float | None = None,
    gap_decoder: str | None = None,
    gap_decoder_config: dict | None = None,
    program: str | None = None,
    scheduler_address: str | None = None,
    plot_interval: float = 30.0,
    queue_factor: int = 2,
    decoder_config: dict | None = None,
    window_parallelism: str = "sliding",
    study: str = "fire-ice-circuit",
    loss_fraction: float = 0.7,
    final_output: Path | None = None,
) -> dict:
    if study not in ("fire-ice-capacity", "fire-ice-circuit", "surface-code"):
        raise ValueError("unknown post-selection study")
    capacity = study == "fire-ice-capacity"
    shots = shots if shots is not None else 10_000_000 if capacity else DEFAULT_SHOTS
    seed = seed if seed is not None else 20260915 if capacity else 10226091600
    jobs = jobs if jobs is not None else 2 if capacity else DEFAULT_JOBS
    batch_size = batch_size if batch_size is not None else min(1000, shots) if capacity else DEFAULT_BATCH_SIZE
    cases = cases if cases is not None else CAPACITY_CASES if capacity else CASES
    program = program or (CAPACITY_PROGRAM if capacity else PROGRAM)
    if capacity:
        rounds = 1
    code = "surface-code" if study == "surface-code" else "fire-ice"
    if physical_error_rate is None:
        physical_error_rate = 0.02 if capacity else 0.00345 if code == "surface-code" else DEFAULT_PHYSICAL_ERROR_RATE
    if not 0 <= loss_fraction <= 1:
        raise ValueError("loss_fraction must be in [0, 1]")
    pauli_fraction = round(1 - loss_fraction, 15)
    if code == "surface-code":
        program = "MemoryExperiment"
    if min(shots, jobs, batch_size) < 1 or seed < 0:
        raise ValueError(
            "positive shot/worker/batch sizes and a nonnegative seed are required"
        )
    if shots % batch_size:
        raise ValueError("shots must be a multiple of batch_size for resumable batches")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    if plot_interval <= 0:
        raise ValueError("plot_interval must be positive")
    if queue_factor < 1:
        raise ValueError("queue_factor must be positive")
    if selection_statistic not in SELECTION_STATISTICS:
        raise ValueError("unknown selection statistic")
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("distinct coordinator cases are required")
    for name in cases:
        if name not in (CAPACITY_CASES if capacity else CASES):
            raise ValueError(f"case {name} does not belong to {study}")
        case_configuration(name)
    hard_config = decoder_configuration(backend, decoder_config)
    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    WORKER_ENVIRONMENT["TOKIO_WORKER_THREADS"] = "1"
    WORKER_ENVIRONMENT["PYTHONPATH"] = os.pathsep.join(
        filter(
            None, (str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH"))
        )
    )
    parameters = dict(
        code=code,
        program=program,
        distance=5 if code == "surface-code" else 6,
        rounds=rounds,
        shots=shots,
        jobs=jobs,
        batch_size=batch_size,
        seed=seed,
        physical_error_rate=physical_error_rate,
        noise=(f"{100 * pauli_fraction:g}% Pauli + {100 * loss_fraction:g}% loss at two-qubit gates; "
               + ("equal first/second/both loss branches; " if loss_fraction else "")
               + "Pauli-only measurement faults; verified-ancilla retries"),
        pauli_fraction=pauli_fraction,
        loss_fraction=loss_fraction,
        measurement_pauli_probability=pauli_fraction * physical_error_rate,
        measurement_loss_probability=0.0,
        sampler="QDK.stim Clifford simulator with neutral-atom SKIP policy",
        two_qubit_loss_branches=("equal marginal probabilities p_loss/3; ELSE arguments p_loss/(3-p_loss) and p_loss/(3-2*p_loss)"
                    if loss_fraction else "none"),
        decoder="Tesseract",
        decoder_config=hard_config,
        configurations={name: case_configuration(name, window_parallelism=window_parallelism) for name in cases},
        window_parallelism=window_parallelism,
        statistic_aggregation="maximum per gadget; full gadget replies retained in the trace",
        comparison="all selection methods use the same shots and hard corrections from one trace",
        trace_schema="per-gadget-statistics-v1",
        window_decoder_cache="reset between shots to bound memory",
        worker_address_space_bytes=4 * 1024**3,
        worker_environment=WORKER_ENVIRONMENT.copy(),
        **gap_configuration(backend, gap_decoder, gap_decoder_config),
    )
    if code == "surface-code":
        for name in ("pauli_fraction", "loss_fraction", "measurement_pauli_probability",
                     "measurement_loss_probability", "two_qubit_loss_branches"):
            parameters.pop(name)
        parameters.update(noise="SI1000 circuit-level Pauli noise; no loss",
                          sampler="Stim static Clifford simulator")
    if capacity:
        parameters.pop("two_qubit_loss_branches")
        parameters.update(
            noise_model="capacity",
            noise="one independent data-qubit DEPOLARIZE1/loss layer; ideal preparation and Z readout",
            case_loss_fractions={name: 0.0 if name == "capacity-pauli" else loss_fraction for name in cases},
            measurement_pauli_probability=0.0,
            logical_error_definition="either of the two final logical Z assertions fails",
            selection_scope="Final 2 asserted logical Z readouts",
        )
    with (root / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if "parameters" not in manifest:
                raise ValueError("saved results use the retired Python capacity decoder; use a fresh directory for DEQ")
            requested = json.loads(json.dumps(parameters))
            if any(
                manifest["parameters"].get(key) != value
                for key, value in requested.items()
                if key not in {"shots", "jobs", "worker_environment"}
            ) or any(
                manifest["parameters"].get(key) != requested.get(key)
                for key in ("gap_decoder", "gap_decoder_config")
            ):
                raise ValueError(
                    "experiment identity changed; use a fresh output directory"
                )
            source = manifest["source"]
            if not any(Path(path).name == Path(__file__).name for path in source["files"]):
                raise ValueError("saved run uses an older runner; use its frozen source to resume, or --plot-only")
            parameters = manifest["parameters"].copy()
        else:
            source = source_identity(
                Path(__file__), Path(render_native.__code__.co_filename),
                Path(trace_statistics.__code__.co_filename),
                SURFACE_FIXTURE if code == "surface-code" else FIXTURE,
                packages=("deq-runtime", "numpy", "scipy", "protobuf", "qdk"),
            )
            source["working_directory"] = str(DEQ_ROOT)
            write_json({"parameters": parameters, "source": source}, manifest_path)

        def verify_source():
            for path, fingerprint in source["files"].items():
                if digest(Path(path)) != fingerprint:
                    raise ValueError(f"recorded sampling source changed: {path}")

        verify_source()
        sampling_root = Path(
            source.get("working_directory")
            or next(
                (
                    str(Path(path).parent.parent)
                    for path in source["files"]
                    if path.endswith("/deq/__init__.py")
                ),
                str(DEQ_ROOT),
            )
        )
        sampling = {
            "python": source.get("interpreter", sys.executable),
            "cwd": sampling_root,
            "environment": parameters["worker_environment"],
        }
        options = {
            "shots": shots,
            "jobs": jobs,
            "max_seconds": max_seconds,
            "scheduler_address": scheduler_address,
            "plot_interval": plot_interval,
            "queue_factor": queue_factor,
            "figure_output": str(figure_output.resolve()),
            "final_output": str(final_output.resolve()) if final_output else None,
        }
        parameters.update(options)
        batches, records = load_completed_batches(root, parameters)
        completed_shots = {
            name: sum(batch["counts"][0] for batch in completed.values())
            for name, completed in batches.items()
        }
        if any(
            start + record["identity"]["shots"] > shots
            or record["identity"]["shots"] != batch_size
            for (_, start), record in records.items()
        ):
            raise ValueError("shot target must cover all completed full batches")
        write_json(options, root / "run-options.json")
        print(
            f"Results: {root}\nLive figure: {figure_output.resolve()}\n"
            f"Sampling source: {sampling_root}\n"
            "Ctrl+C pauses after active batches; repeat with the same data directory to resume.",
            flush=True,
        )

        def publish(status):
            verify_source()
            remaining_seconds = 0.0
            observed = {}
            for name, completed in batches.items():
                attempted = sum(batch["counts"][0] for batch in completed.values())
                seconds = sum(records[name, start]["seconds"] for start in completed)
                per_shot = seconds / attempted if attempted else None
                observed[name] = per_shot
                if per_shot is not None:
                    remaining_seconds += max(0, shots - attempted) * per_shot
            estimated_seconds = remaining_seconds / jobs if all(value is not None for value in observed.values()) else None
            parameters["progress"] = {
                "estimated_remaining_seconds": estimated_seconds,
                "seconds_per_attempt": observed,
                "estimate_basis": "completed batch timings; unavailable until each case completes a batch; excludes queueing and unfinished tails",
                "execution_slots": jobs,
                "submitted_limit": jobs * queue_factor,
            }
            failures = []
            for name, completed in batches.items():
                for start, result in completed.items():
                    if result["counts"][0] == result["counts"][1]:
                        continue
                    record = records[name, start]
                    directory = root / name / "chunks" / f"{start:09d}"
                    failures.append({
                        "case": name, "start": start,
                        "seed": record["identity"]["seed"],
                        "failed_shots": result["counts"][0] - result["counts"][1],
                        "kind": record.get("execution_failure", {}).get("kind", "decode_or_score"),
                        "returncode": record.get("execution_failure", {}).get("returncode", 0),
                        "record": str(directory / ("outcome.json" if record.get("execution_failure") else "record.json")),
                        "runtime_log": str(directory / "runtime.log"),
                    })
            write_json({"failures": failures, "updated_at": datetime.now(timezone.utc).isoformat()},
                       root / "failure-index.json")
            parameters["failure_index"] = "failure-index.json"
            parameters["has_unresolved_failures"] = bool(failures)
            summary = summarize(
                root, parameters, batches, records, status == "complete", status=status
            )
            if summary["cases"]:
                render(summary, figure_output, selection_statistic, data_dir=root, final_output=final_output)
            print(
                json.dumps(
                    {
                        "status": status,
                        "target_per_case": shots,
                        "estimated_remaining_hours": estimated_seconds / 3600 if estimated_seconds is not None else None,
                        "queue": parameters.get("queue", {}),
                        "cases": {
                            case["name"]: {
                                "shots": case["shots"],
                                "errors": case["logical_errors"],
                                "failed_shots": case["failed_shots"],
                                "execution_failed_shots": case["execution_failed_shots"],
                            }
                            for case in summary["cases"]
                        },
                    }
                ),
                flush=True,
            )
            return summary

        def is_complete():
            return all(count >= shots for count in completed_shots.values())

        if records:
            summary = publish("complete" if is_complete() else "running")
            if is_complete():
                return summary
        if not records:
            prepare = prepare_surface_circuit if code == "surface-code" else prepare_circuit
            for name in cases if capacity else ("monolithic",):
                fraction = parameters.get("case_loss_fractions", {}).get(name, loss_fraction)
                circuit = prepare(
                    rounds, physical_error_rate, root / f"{name if capacity else code}.deq",
                    **({"loss_fraction": fraction} if code == "fire-ice" else {}),
                )
                compile_circuit(circuit, compilation_directory(root, name, capacity=capacity), program,
                                case_configuration("monolithic", False), hard_config, decoder=BACKENDS[backend],
                                simulator="static" if code == "surface-code" else "qdk",
                                mako={"p": physical_error_rate, "rounds": rounds})
        builds = {name: compilation_directory(root, name, capacity=capacity) / "build" for name in cases}
        tasks = iter(
            (name, start)
            for start in range(0, shots, batch_size)
            for name in cases
            if start not in batches[name]
        )
        recorded_runner = next(
            (path for path in source["files"] if Path(path).name == Path(__file__).name),
            __file__,
        )
        example_directory = str(Path(recorded_runner).resolve().parent)
        execute_batch = worker_function(run_batch, directory=example_directory, paths=[str(sampling["cwd"])])

        def submit(pool, task):
            name, start = task
            return pool.submit(
                execute_batch, root, builds[name], name, start, batch_size, seed + start // batch_size,
                rounds, backend=backend, sampling=sampling, gap_decoder=gap_decoder,
                gap_decoder_config=gap_decoder_config, program=program, decoder_config=hard_config,
                window_parallelism=window_parallelism, code=code,
            )

        def collect(task, future):
            name, start = task
            try:
                record, result = future.result()
            except RuntimeError:
                if not (root / name / "chunks" / f"{start:09d}" / "failure.json").exists():
                    raise
                record, result = run_batch(
                    root, builds[name], name, start, batch_size, seed + start // batch_size,
                    rounds, backend=backend, sampling=sampling, gap_decoder=gap_decoder,
                    gap_decoder_config=gap_decoder_config, program=program, accept_failure=True,
                    decoder_config=hard_config, window_parallelism=window_parallelism, code=code,
                )
                print(json.dumps({"event": "execution_failure", "case": name,
                                  "start": start, **record["execution_failure"]}), flush=True)
            records[name, start] = record
            batches[name][start] = result
            completed_shots[name] += result["counts"][0]

        return sample_batches(tasks, submit, collect, publish, is_complete, parameters)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("fire-ice-capacity", "fire-ice-circuit", "surface-code"),
                        default="fire-ice-circuit")
    parser.add_argument("--samples", "--shots", dest="shots", type=int, help="Target shots per configuration")
    parser.add_argument("--data-dir", "--output-dir", dest="output_dir", type=Path)
    parser.add_argument("--output", "--figure-output", dest="figure_output", type=Path)
    parser.add_argument("--final-output", type=Path, help="Live final-readout figure path for Fire & Ice circuits")
    parser.add_argument("--jobs", type=int, help="Total concurrent tasks, not per configuration")
    parser.add_argument("--scheduler-address", "--scheduler-url", help="Dask scheduler (default: local cluster)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--plot-interval", type=float, default=30)
    parser.add_argument("--queue-factor", type=int, default=2)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--program")
    parser.add_argument("--physical-error-rate", type=float)
    parser.add_argument("--loss-fraction", type=float)
    parser.add_argument("--cases", nargs="+", choices=(*CASES, *CAPACITY_CASES))
    parser.add_argument("--window-parallelism", choices=("sliding", "fully_parallel"))
    parser.add_argument("--decoder-config", type=json.loads)
    parser.add_argument("--gap-decoder")
    parser.add_argument("--gap-decoder-config", type=json.loads)
    args = parser.parse_args()
    capacity = args.study == "fire-ice-capacity"
    surface = args.study == "surface-code"
    root = args.output_dir or Path(__file__).parent / "results" / args.study
    saved = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else {}
    saved = saved.get("parameters", saved)
    if (root / "run-options.json").exists():
        saved.update(json.loads((root / "run-options.json").read_text()))

    def setting(name, default):
        value = getattr(args, name)
        return saved.get(name, default) if value is None else value

    parallelism = setting("window_parallelism", "sliding" if saved.get("causal_commit_order", not bool(saved)) else "fully_parallel")
    loss_fraction = setting("loss_fraction", 0.7)
    if not 0 <= loss_fraction <= 1:
        parser.error("--loss-fraction must be in [0, 1]")
    prefix = "fire_ice_pauli" if loss_fraction == 0 else "fire_ice" if loss_fraction == 0.7 else f"fire_ice_loss_{loss_fraction:g}"
    filename = ("fire_ice_capacity_post_selection" if capacity else "surface_code_post_selection" if surface
                else f"{prefix}_{parallelism}_all_readouts")
    output = args.figure_output or FIGURES / f"{filename}.pdf"
    final_output = None
    if not capacity and not surface:
        final_output = args.final_output or (output.with_stem(output.stem + "_final") if args.figure_output
                                            else FIGURES / f"{prefix}_{parallelism}_final_readouts.pdf")
    elif args.final_output:
        parser.error("--final-output applies only to Fire & Ice circuits")
    circuit_options = dict(
        batch_size=setting("batch_size", None), rounds=setting("rounds", 1 if capacity else 10),
        program=setting("program", CAPACITY_PROGRAM if capacity else PROGRAM),
        physical_error_rate=setting("physical_error_rate", 0.02 if capacity else 0.00345 if surface else DEFAULT_PHYSICAL_ERROR_RATE),
        loss_fraction=loss_fraction, cases=tuple(args.cases or saved.get("configurations", CAPACITY_CASES if capacity else CASES)),
        window_parallelism=parallelism, decoder_config=setting("decoder_config", None),
        gap_decoder=setting("gap_decoder", None if saved else "black-box-tesseract"),
        gap_decoder_config=setting("gap_decoder_config", None if saved else GAP_CONFIG),
    )
    if args.plot_only:
        if capacity and (saved.get("decoder") != "Tesseract" or saved.get("program") != CAPACITY_PROGRAM):
            parser.error("capacity plotting requires native DEQ results; use frozen tools for the retired Python reference")
        summary = (json.loads((root / "summary.json").read_text())
               if not (root / "manifest.json").exists() else recover_summary(root))
        if summary.get("code", "fire-ice") != ("surface-code" if surface else "fire-ice") or summary.get("decoder") != "Tesseract":
            parser.error("saved data does not match the requested DEQ study")
        publish_figures(summary, root, output, final_output)
    else:
        summary = run(root, output, study=args.study, shots=setting("shots", 10_000_000 if capacity else DEFAULT_SHOTS),
                      seed=setting("seed", 20260915 if capacity else 10226091600), jobs=setting("jobs", 2 if capacity else DEFAULT_JOBS),
                      scheduler_address=args.scheduler_address, max_seconds=args.max_seconds,
                      plot_interval=args.plot_interval, queue_factor=args.queue_factor, final_output=final_output,
                      **circuit_options)
    print(f"Figure: {output.resolve()} ({summary.get('status', 'saved')})")
    if summary.get("has_unresolved_failures"):
        parser.exit(1, "Evaluation contains execution or decoding/scoring failures; inspect failure-index.json.\n")


if __name__ == "__main__":
    main()


