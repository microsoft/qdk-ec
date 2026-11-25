"""Run either native simulator through the source CLI and validate every readout."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import re
import sys

import stim

from deq.cli.jit import _annotate_stim_with_detectors
from deq.proto.deq_jit_pb2 import JitLibrary
from deq.proto.simulator_pb2 import SimulatorTrace
from lattice_surgery import circuit_source
from readout_latency import readout_samples
from study_utils import population, run_process, save_json


def validate_distance(directory: Path) -> dict:
    configuration = json.loads((directory / "configuration.json").read_text())
    if configuration["input_kind"] != "physical_stim_sampling":
        raise ValueError("distance validation requires the original physical circuit")
    library = JitLibrary.FromString((directory / "circuit.deq.jit").read_bytes())
    annotated = directory / "detectors.stim"
    annotated.write_text((directory / "circuit.stim").read_text())
    _annotate_stim_with_detectors(library, str(annotated), configuration["assertions"])
    circuit = stim.Circuit.from_file(annotated)
    witness = circuit.shortest_graphlike_error()
    result = {
        "requested_distance": configuration["distance"],
        "graphlike_circuit_distance": len(witness),
        "observables": circuit.num_observables,
    }
    if len(witness) != configuration["distance"]:
        raise ValueError(f"unexpected circuit distance: {result}")
    return result


def run_case(
    directory: Path,
    *,
    shots: int,
    threads: int,
    noiseless: bool,
    simulator: str = "jit-static",
    decoder: str = "mwpm",
    decoder_config: dict | None = None,
    memory_gib: float = 24,
    timeout_seconds: float = 3600,
    registration_lead_seconds: float | None = None,
) -> dict:
    if shots < 1 or threads < 1:
        raise ValueError("shots and threads must be positive")
    if simulator not in ("static", "jit-static"):
        raise ValueError("latency requires a native static or jit-static simulator")
    parameters = dict(decoder_config or {})
    if "parallel" in parameters and parameters["parallel"] != threads:
        raise ValueError("decoder parallel setting must match threads")
    parameters["parallel"] = threads
    configuration = json.loads((directory / "configuration.json").read_text())
    if simulator == "static" and registration_lead_seconds is not None:
        raise ValueError("registration lead applies only to jit-static")
    run = directory / f"{simulator}-{decoder}"
    run.mkdir()
    source_path = directory / "circuit.deq"
    if noiseless:
        source_path = run / "circuit.deq"
        source_path.write_text(
            circuit_source(
                **{
                    key: configuration[key]
                    for key in ("distance", "width", "layers", "basis", "seed")
                },
                probability=0,
                block_seconds=configuration["block_ns"] / 1_000_000_000,
                use_3aux=configuration.get("use_3aux", False),
            )
        )
    command = [
        sys.executable,
        "-m",
        "deq",
        "simulate",
        "ler",
        str(source_path.resolve()),
        "--program",
        "Brickwall",
        "--jobs",
        "1",
        "--shots",
        str(shots),
        "--batch-size",
        str(shots),
        "--errors",
        str(shots + 1),
        "--decoder",
        decoder,
        "--decoder-config",
        json.dumps(parameters),
        "--coordinator",
        "window",
        "--coordinator-config",
        json.dumps(
            {
                "buffer_radius": 1,
                "lookahead_radius": 0,
                "persistent_decoder": simulator == "static",
                "assert_parity_factor": True,
            }
        ),
        "--simulator",
        simulator,
        "--strict-timing",
        "--loss-model",
        "none",
        "--seed",
        str(configuration["seed"]),
        "--save",
        str(run.resolve()),
        "--debug-dir",
        str((run / "debug").resolve()),
        "--timeout",
        str(timeout_seconds),
        "--trace",
        str((run / "timing.pb").resolve()),
        "--simulator-trace-output",
        str((run / "shots.pb").resolve()),
        "--skip-mako-warning",
    ]
    if registration_lead_seconds is not None:
        command.extend(["--registration-lead-seconds", str(registration_lead_seconds)])
    save_json(run / "command.json", command)
    resources = run_process(
        command,
        run / "worker.log",
        memory_gib=memory_gib,
        timeout_seconds=timeout_seconds,
        cwd=Path(__file__).resolve().parents[4],
    )
    native = importlib.import_module("deq_runtime.deq_runtime").__file__
    if native is None:
        raise RuntimeError("cannot identify the native runtime")
    save_json(
        run / "execution.json",
        {
            **resources,
            "shots": shots,
            "threads": threads,
            "decoder": decoder,
            "decoder_config": parameters,
            "interval_ms": configuration["block_ns"] / 1_000_000,
            "noiseless": noiseless,
            "registration_lead_seconds": registration_lead_seconds,
            "controller": "static" if simulator == "static" else "jit",
            "simulator": simulator,
            "persistent_decoder": simulator == "static",
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "runtime_sha256": hashlib.sha256(Path(native).read_bytes()).hexdigest(),
        },
    )
    return analyze_completed_run(run)


def analyze_completed_run(run: Path) -> dict:
    execution = json.loads((run / "execution.json").read_text())
    configuration = json.loads((run.parent / "configuration.json").read_text())
    results = SimulatorTrace.FromString((run / "shots.pb").read_bytes())
    if [shot.shot for shot in results.shots] != list(range(execution["shots"])):
        raise ValueError(f"missing or duplicate shots; see {run}")
    log = (run / "worker.log").read_text()
    native_log = (run / "debug" / "runtime_output.txt").read_text()
    if "panicked" in log or "panicked" in native_log:
        raise ValueError(f"native panic during evaluation; see {run}")
    summary = log.rpartition("=== Simulation Results ===")[2]
    counts = {}
    for label in ("Shots", "Logical errors", "Failed shots"):
        match = re.search(rf"^\s*{label}:\s+(\d+)\s*$", summary, re.MULTILINE)
        if match is None:
            raise ValueError(f"missing CLI result {label}; see {run}")
        counts[label] = int(match[1])
    if counts["Shots"] != execution["shots"] or counts["Failed shots"]:
        raise ValueError(f"incomplete CLI shot results; see {run}")
    for shot in results.shots:
        if not shot.HasField("decode_result"):
            raise ValueError(f"failed decoding shot {shot.shot}; see {run}")
        if [readout.gid for readout in shot.gadget_readouts] != list(
            range(1, configuration["gadgets"] + 1)
        ) or shot.decode_result.readouts.size != configuration["readouts"]:
            raise ValueError(f"incomplete decoding shot {shot.shot}; see {run}")
    failures = sum(shot.logical_error for shot in results.shots)
    if failures != counts["Logical errors"]:
        raise ValueError(f"CLI and simulator trace logical errors disagree; see {run}")
    if execution["noiseless"] and failures:
        raise ValueError(f"{failures} noiseless logical failures; see {run}")
    result = {
        **execution,
        **{
            key: configuration[key]
            for key in (
                "width",
                "layers",
                "distance",
                "probability",
                "seed",
            )
        },
        "input_kind": "physical_stim_sampling",
        "input_provenance": configuration.get("input_provenance"),
        "run_directory": str(run.resolve()),
        "logical_errors": failures,
        "circuit_sha256": execution["source_sha256"],
        "includes_jit_compilation": execution["controller"] == "jit",
        "latency_origin": "native start_decoding event; static program setup is excluded",
        "timings_qualified": True,
        "failed_shots": 0,
    }
    samples, _ = readout_samples(result)
    result["populations"] = {
        region: population(
            [sample.latency_ms for sample in samples if sample.region == region]
        )
        for region in ("intermediate", "final")
    }
    save_json(run / "summary.json", result)
    return result
