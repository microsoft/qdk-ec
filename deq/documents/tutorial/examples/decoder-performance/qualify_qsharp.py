"""Compare physical Q# and Stim sampling with the same deq window decoder."""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
import psutil
import stim

from deq.proto import coordinator_pb2, deq_bin_pb2, util_pb2
from deq.runtime import Runtime
from replay_qsharp import QSharpSampler, build_recipe
from study_utils import (
    exclusive_output,
    interrupt_on_termination,
    save_json,
    terminate_processes,
)

_state = None


def initialize_worker(
    case_name: str, probability: float, window_parallelism: str = "sliding"
) -> None:
    global _state
    case = Path(case_name)
    configuration = json.loads((case / "configuration.json").read_text())
    library = deq_bin_pb2.Library.FromString((case / "circuit.deq.bin").read_bytes())
    kinds = {kind.gtype: kind for kind in library.gadget_types}
    gadgets = [
        instruction.gadget
        for instruction in library.program
        if instruction.WhichOneof("create") == "gadget"
    ]
    circuit = stim.Circuit.from_file(case / "circuit.stim")
    recipe = build_recipe(case / "circuit.deq", case / "circuit.deq.jit", probability)
    if (
        circuit.num_measurements != recipe.total_measurements
        or circuit.num_qubits != recipe.qubits
    ):
        raise ValueError("Q# and Stim workload dimensions disagree")
    runtime = Runtime(
        decoder="mwpm",
        decoder_config={"parallel": 1},
        coordinator="window",
        coordinator_config={
            "buffer_radius": 1,
            "lookahead_radius": 0,
            "persistent_decoder": True,
            "automatic_collection": False,
            "assert_parity_factor": True,
            "window_parallelism": window_parallelism,
        },
    )
    asyncio.run(runtime.coordinator.load_library(library))
    _state = (
        configuration,
        library,
        kinds,
        gadgets,
        circuit,
        circuit.reference_sample(bit_packed=True),
        QSharpSampler(recipe, probability),
        runtime,
    )


async def decode_shot(packed: bytes) -> bool:
    configuration, library, kinds, gadgets, _, _, _, runtime = _state
    coordinator = runtime.coordinator
    await coordinator.reset()
    for instruction in library.program:
        assigned = await coordinator.execute(instruction)
        name = instruction.WhichOneof("create")
        requested = getattr(
            getattr(instruction, name),
            {"gadget": "gid", "check_model": "cid", "error_model": "eid"}[name],
        )
        if requested and assigned != requested:
            raise ValueError("Static program IDs changed during qualification")
    bits = np.unpackbits(
        np.frombuffer(packed, dtype=np.uint8),
        bitorder="little",
        count=configuration["measurements"],
    )
    requests = []
    offset = 0
    for gadget in gadgets:
        count = len(kinds[gadget.gtype].measurements)
        data = np.packbits(bits[offset : offset + count], bitorder="big").tobytes()
        requests.append(
            coordinator.decode(
                coordinator_pb2.Outcomes(
                    gid=gadget.gid,
                    outcomes=util_pb2.BitVector(size=count, data=data),
                )
            )
        )
        offset += count
    if offset != len(bits):
        raise ValueError("Physical measurement partition mismatch")
    decoded = await asyncio.gather(*requests)
    readouts = []
    for gadget, response in zip(gadgets, decoded):
        expected = len(kinds[gadget.gtype].readouts)
        if response.gid != gadget.gid or response.readouts.size != expected:
            raise ValueError("Incomplete decoder response in qualification")
        readouts.extend(
            np.unpackbits(
                np.frombuffer(response.readouts.data, dtype=np.uint8),
                bitorder="big",
                count=expected,
            ).tolist()
        )
    if len(readouts) != configuration["readouts"]:
        raise ValueError("Logical readout count mismatch")
    return any(
        bool(sum(readouts[index] for index in indices) % 2) != expected
        for indices, expected, _ in configuration["assertions"]
    )


def evaluate_shot(backend: str, seed: int) -> dict:
    started = time.monotonic()
    configuration, _, _, _, circuit, reference, qsharp, _ = _state
    if backend == "qdk":
        packed = qsharp.sample(seed)
    elif backend == "stim":
        packed = (
            circuit.compile_sampler(seed=seed, reference_sample=reference)
            .sample(shots=1, bit_packed=True)[0]
            .tobytes()
        )
    else:
        raise ValueError(f"Unknown backend {backend}")
    expected = (configuration["measurements"] + 7) // 8
    if len(packed) != expected:
        raise ValueError("Incomplete physical shot")
    sampled = time.monotonic()
    error = asyncio.run(decode_shot(packed))
    return {
        "backend": backend,
        "seed": seed,
        "logical_error": error,
        "sample_sha256": hashlib.sha256(packed).hexdigest(),
        "sampling_seconds": sampled - started,
        "total_seconds": time.monotonic() - started,
    }


def wilson(errors: int, shots: int) -> list[float]:
    z = 1.959963984540054
    rate = errors / shots
    denominator = 1 + z * z / shots
    center = (rate + z * z / (2 * shots)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1 - rate) / shots + z * z / (4 * shots * shots))
        / denominator
    )
    return [center - radius, center + radius]


def compare_rates(totals: dict, target_errors: int) -> dict:
    if any(value["errors"] < target_errors for value in totals.values()):
        return {"status": "insufficient_errors"}
    for value in totals.values():
        value["rate"] = value["errors"] / value["shots"]
        value["wilson_95"] = wilson(value["errors"], value["shots"])
    first, second = totals["qdk"], totals["stim"]
    pooled = (first["errors"] + second["errors"]) / (first["shots"] + second["shots"])
    variance = pooled * (1 - pooled) * (1 / first["shots"] + 1 / second["shots"])
    z = abs(first["rate"] - second["rate"]) / math.sqrt(variance) if variance else 0.0
    probability = math.erfc(z / math.sqrt(2))
    return {
        "status": "qualified" if probability >= 0.05 else "disagreement",
        "two_proportion_p_approx": probability,
        "decision_alpha": 0.05,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--target-errors", type=int, default=100)
    parser.add_argument(
        "--shots",
        type=int,
        help="Predeclared fixed shots per backend instead of error-count stopping",
    )
    parser.add_argument("--max-shots", type=int, default=20000)
    parser.add_argument("--memory-gib", type=float, default=320)
    parser.add_argument("--timeout-seconds", type=float, default=14400)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--window-parallelism", choices=("sliding", "serial"), default="sliding"
    )
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    if (
        arguments.workers < 1
        or arguments.target_errors < 1
        or arguments.max_shots < arguments.target_errors
    ):
        raise ValueError("Invalid qualification limits")
    if (
        arguments.shots is not None
        and not arguments.target_errors <= arguments.shots <= arguments.max_shots
    ):
        raise ValueError("Fixed shots must lie between target-errors and max-shots")
    if not 0 <= arguments.seed_start < 2**32 or arguments.max_shots > 2**32:
        raise ValueError("Seed range must not repeat within a backend")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (arguments.memory_gib, arguments.timeout_seconds)
    ):
        raise ValueError("Memory and time budgets must be finite and positive")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with interrupt_on_termination(), exclusive_output(
        arguments.output.with_suffix(".lock")
    ):
        _qualify(arguments)


def _qualify(arguments: argparse.Namespace) -> None:
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    configuration = json.loads((arguments.case / "configuration.json").read_text())
    if configuration["width"] != 10:
        raise ValueError("The statistical qualification must use width 10")
    records = []
    totals = {
        backend: {"shots": 0, "errors": 0, "submitted": 0}
        for backend in ("qdk", "stim")
    }
    os.environ["QDK_PYTHON_TELEMETRY"] = "none"
    os.environ["TOKIO_WORKER_THREADS"] = "1"
    os.environ["RAYON_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    started = time.monotonic()
    report = {
        "status": "running",
        "configuration": configuration,
        "decoder": f"deq static/persistent window MWPM; buffer=1, lookahead=0, policy={arguments.window_parallelism}",
        "seed_start": arguments.seed_start,
        "fixed_shots": arguments.shots,
        "decision_alpha": 0.05,
        "python": sys.executable,
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("qdk", "stim", "deq-runtime")
        },
        "source_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("qualify_qsharp.py", "replay_qsharp.py", "replay_sampler.qs")
        },
        "target_errors": arguments.target_errors,
        "records": records,
        "totals": totals,
        "statistical_note": "Wilson intervals and the two-proportion check are approximate under error-count stopping; agreement is not a proof of equal distributions.",
    }
    executor = ProcessPoolExecutor(
        max_workers=arguments.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker,
        initargs=(
            str(arguments.case.resolve()),
            configuration["probability"],
            arguments.window_parallelism,
        ),
    )
    try:
        pending = {}

        def submit(backend):
            index = totals[backend]["submitted"]
            if index >= (arguments.shots or arguments.max_shots):
                return False
            if (
                arguments.shots is None
                and totals[backend]["errors"] >= arguments.target_errors
            ):
                return False
            seed = (
                42
                + (index + arguments.seed_start) * 104729
                + (0xA511E9B3 if backend == "qdk" else 0)
            ) & 0xFFFFFFFF
            pending[executor.submit(evaluate_shot, backend, seed)] = backend
            totals[backend]["submitted"] += 1
            return True

        for index in range(arguments.workers):
            submit(("qdk", "stim")[index % 2])
        while pending:
            completed, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            if time.monotonic() - started > arguments.timeout_seconds:
                raise TimeoutError("Qualification exceeded its time budget")
            root = psutil.Process()
            memory = root.memory_info().rss
            for child in root.children(recursive=True):
                try:
                    memory += child.memory_info().rss
                except psutil.NoSuchProcess:
                    continue
            if memory > arguments.memory_gib * 2**30:
                raise MemoryError(
                    "Qualification process pool exceeded its memory budget"
                )
            for future in completed:
                backend = pending.pop(future)
                result = future.result()
                records.append(result)
                totals[backend]["shots"] += 1
                totals[backend]["errors"] += int(result["logical_error"])
                if not submit(backend):
                    submit("stim" if backend == "qdk" else "qdk")
            if completed:
                report["elapsed_seconds"] = time.monotonic() - started
                save_json(arguments.output, report)
                print(
                    {
                        key: (value["errors"], value["shots"])
                        for key, value in totals.items()
                    },
                    flush=True,
                )
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        try:
            save_json(arguments.output, report)
        finally:
            terminate_processes(psutil.Process().children(recursive=True), timeout=10)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    report.update(compare_rates(totals, arguments.target_errors))
    report["elapsed_seconds"] = time.monotonic() - started
    save_json(arguments.output, report)
    if report["status"] != "qualified":
        raise RuntimeError(
            f"QDK and Stim qualification ended with {report['status']}; do not start the large sweep"
        )


if __name__ == "__main__":
    main()
