"""Reusable simulation execution and artifact helpers for tutorial examples."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, wait
from contextlib import contextmanager
import hashlib
import importlib.metadata
import importlib.util
from itertools import islice
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Event
import time

from deq.proto import deq_bin_pb2, deq_jit_pb2, simulator_pb2


DEQ_ROOT = Path(__file__).resolve().parents[3]
WORKER_ENVIRONMENT = {
    "TOKIO_WORKER_THREADS": "2",
    "RAYON_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
}


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(document: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def compiled_identity(build_dir: Path, program: str) -> dict[str, str]:
    result = {".stim": digest(build_dir / f"{program}.stim")}
    for suffix, message_type in ((".deq.bin", deq_bin_pb2.Library), (".deq.jit", deq_jit_pb2.JitLibrary)):
        message = message_type.FromString((build_dir / f"{program}{suffix}").read_bytes())
        result[suffix] = hashlib.sha256(message.SerializeToString(deterministic=True)).hexdigest()
    return result


def load_trace(path: Path, expected_shots: int) -> simulator_pb2.SimulatorTrace:
    trace = simulator_pb2.SimulatorTrace.FromString(path.read_bytes())
    if len(trace.shots) != expected_shots or any(shot.shot != index for index, shot in enumerate(trace.shots)):
        raise ValueError(f"{path}: incomplete or misordered shot trace")
    return trace


@contextmanager
def batch_executor(jobs: int, scheduler_address: str | None = None):
    from distributed import Client, LocalCluster

    cluster = None
    if scheduler_address is None:
        cluster = LocalCluster(
            n_workers=1, threads_per_worker=jobs, processes=False, memory_limit=0,
            resources={"MEMORY_GB": 5 * jobs}, dashboard_address=":0",
        )
    try:
        with Client(cluster or scheduler_address, timeout="60s") as client:
            workers = client.scheduler_info()["workers"]
            memory_resources = all("MEMORY_GB" in worker.get("resources", {}) for worker in workers.values())
            slots = sum(
                min(worker["nthreads"], int(worker["resources"]["MEMORY_GB"] // 5)) if memory_resources else worker["nthreads"]
                for worker in workers.values()
            )
            if slots < jobs:
                raise ValueError(f"requested {jobs} jobs but Dask has {slots} available task slots")
            print(f"Dask: {client.scheduler.address}; {len(workers)} workers; {slots} task slots", flush=True)
            with client.get_executor(pure=False, **({"resources": {"MEMORY_GB": 5}} if memory_resources else {})) as executor:
                yield executor
    finally:
        if cluster is not None:
            cluster.close()


def sample_batches(tasks, submit, collect, publish, is_complete, parameters):
    """Run sampling jobs with callbacks, draining assigned work before pausing."""
    tasks = iter(tasks)
    jobs = parameters["jobs"]
    pending_limit = jobs * parameters.get("queue_factor", 2)
    plot_interval = parameters.get("plot_interval", 30)
    max_seconds = parameters.get("max_seconds")
    if jobs < 1 or pending_limit < 1 or plot_interval <= 0 or (max_seconds is not None and max_seconds <= 0):
        raise ValueError("positive concurrency, queue factor, plot interval and session limit are required")
    pause_requested = Event()
    started = time.monotonic()
    last_published = started - plot_interval

    def request_pause(*_args):
        if not pause_requested.is_set():
            print("Pause requested; finishing active batches and saving progress.", flush=True)
            pause_requested.set()

    previous_handlers = {signum: signal.signal(signum, request_pause) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        with batch_executor(jobs, parameters.get("scheduler_address")) as pool:
            futures = {}
            try:
                while True:
                    if not pause_requested.is_set():
                        for task in islice(tasks, pending_limit - len(futures)):
                            futures[submit(pool, task)] = task
                    if not futures:
                        return publish("complete" if is_complete() else "paused")
                    completed, _ = wait(futures, timeout=1.0, return_when=FIRST_COMPLETED)
                    for future in completed:
                        collect(futures.pop(future), future)
                    if max_seconds is not None and time.monotonic() - started >= max_seconds:
                        request_pause()
                    parameters["queue"] = {"submitted": len(futures), "limit": pending_limit, "execution_slots": jobs}
                    status = "complete" if is_complete() else "pausing" if pause_requested.is_set() else "running"
                    if not futures and pause_requested.is_set() and status != "complete":
                        status = "paused"
                    if status in ("complete", "paused") or time.monotonic() - last_published >= plot_interval:
                        summary = publish(status)
                        last_published = time.monotonic()
                    if status in ("complete", "paused"):
                        return summary
            except BaseException as error:
                if futures:
                    wait(futures)
                for future, task in list(futures.items()):
                    if not future.cancelled() and future.exception() is None:
                        collect(task, future)
                parameters["stop_reason"] = f"{type(error).__name__}: {error}"
                publish("stopped")
                raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def worker_function(function, *, directory=None, paths=()):
    """Load recorded modules on workers without relying on prior imports."""
    module_path = Path(function.__code__.co_filename).resolve()
    directory = str(directory or module_path.parent)
    module_name, function_name = module_path.stem, function.__name__
    import_paths = [directory, str(Path(__file__).resolve().parent), str(DEQ_ROOT),
                    *paths, *os.environ.get("PYTHONPATH", "").split(os.pathsep)]

    def execute(*arguments, **options):
        import importlib
        from pathlib import Path
        import sys

        for path in reversed(import_paths):
            if path and path not in sys.path:
                sys.path.insert(0, path)
        module = importlib.import_module(module_name)
        if Path(module.__file__).resolve().parent != Path(directory):
            raise RuntimeError("Dask worker cached a different evaluation source; restart that worker before resuming")
        return getattr(module, function_name)(*arguments, **options)

    return execute


def run_logged(command: list[str], log_path: Path, *, cwd: Path = DEQ_ROOT,
               environment: dict[str, str] | None = None, pass_fds: tuple[int, ...] = ()) -> int:
    with log_path.open("w") as output, subprocess.Popen(
        command, cwd=cwd, env={**os.environ, **WORKER_ENVIRONMENT, **(environment or {})},
        stdout=output, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=pass_fds,
    ) as process:
        try:
            return process.wait()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise


def compile_circuit(fixture, directory, program, coordinator_settings, decoder_config, *,
                    decoder, simulator="qdk", mako=None, seed=17, timeout_seconds=600):
    """Compile a circuit and check one native shot, retaining artifact hashes."""
    build = directory / "build"
    chunks = directory / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    trace_path = chunks / "000000000.pb"
    record_path = trace_path.with_suffix(".json")
    log_path = trace_path.with_suffix(".log")
    coordinator, config = coordinator_settings
    options = dict(program=program, simulator=simulator, shots=1, errors=2,
                   **{"batch-size": 1, "seed": seed, "jobs": 1, "decoder": decoder,
                      "decoder-config": json.dumps(decoder_config), "coordinator": coordinator,
                      "coordinator-config": json.dumps(config), "simulator-trace-output": trace_path, "save": build})
    command = ["timeout", "--kill-after=10s", str(timeout_seconds), sys.executable, "-m", "deq", "simulate", "ler", str(fixture)]
    for name, value in (mako or {}).items():
        command.extend(("--mako", f"{name}={value}"))
    command.extend(value for key, item in options.items() for value in (f"--{key}", str(item)))
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record["command"] != command or any(digest(directory / path) != value for path, value in record["files"].items()):
            raise ValueError("compilation inputs or artifacts changed")
        return
    if trace_path.exists():
        raise ValueError(f"{trace_path}: uncommitted trace requires explicit inspection")
    returncode = run_logged(command, log_path)
    if returncode:
        write_json(dict(command=command, returncode=returncode), trace_path.with_suffix(".failure.json"))
        log_tail = log_path.read_text(errors="replace")[-8192:]
        raise RuntimeError(f"compilation failed with exit {returncode}; see {log_path}\n{log_tail}")
    if not load_trace(trace_path, 1).shots[0].HasField("decode_result"):
        raise RuntimeError(f"{trace_path}: compilation check failed to decode")
    files = [trace_path, fixture, *(build / f"{program}{suffix}" for suffix in (".stim", ".deq.bin", ".deq.jit"))]
    write_json(dict(command=command, files={str(path.resolve()): digest(path) for path in files},
                    compiled_semantic_sha256=compiled_identity(build, program)), record_path)


def source_identity(*source_files: Path, packages=("deq-runtime", "protobuf")) -> dict[str, object]:
    """Fingerprint the installed runtime, frontend, and caller-supplied study files."""
    files = [Path(__file__), *source_files, *sorted((DEQ_ROOT / "deq").rglob("*.py"))]
    for package in ("deq_runtime", "binar", "paulimer", "qodec", "deqagram"):
        specification = importlib.util.find_spec(package)
        if specification is not None and specification.origin:
            files.append(Path(specification.origin))
            for directory in specification.submodule_search_locations or ():
                files.extend(sorted(Path(directory).rglob("*.so")))
    if not any(path.suffix == ".so" for path in files):
        raise RuntimeError("cannot identify the loaded native runtime")
    return {
        "interpreter": sys.executable,
        "python_version": sys.version,
        "files": {str(path.resolve()): digest(path) for path in files},
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }
