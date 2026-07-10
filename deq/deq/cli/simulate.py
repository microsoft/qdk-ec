# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint
"""End-to-end logical error rate simulation from .deq files.

Pipeline stages:

1. Build JIT library and compile program  (.deq → .deq.jit + .stim)
2. Compile program into static binary     (.deq.jit → .deq.bin)
3. Run deq_runtime server to evaluate LER (.deq.bin + .stim → results)

Usage example::

    deq simulate ler path/to/*.deq \\
        --program MemoryExperiment --shots 10000 --errors 100
"""

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass

import arguably

from deq.circuit.model import (
    CodeDefinition,
    ComposeDefinition,
    GadgetDefinition,
    ProgramDefinition,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SIMULATION_COMPLETE = "Simulation Complete"


def _parse_server_output(text: str) -> dict[str, int | float]:
    """Extract statistics from the Rust server's summary output."""
    result: dict[str, int | float] = {}
    m = re.search(r"Shots:\s+(\d+)/(\d+)", text)
    if m:
        result["shots"] = int(m.group(1))
    m = re.search(r"Logical errors:\s+(\d+)/(\d+)", text)
    if m:
        result["logical_errors"] = int(m.group(1))
    m = re.search(r"\(([\d.eE+\-]+)s per shot\)", text)
    if m:
        result["decode_time_per_shot"] = float(m.group(1))
    m = re.search(r"Retries:\s+(\d+)", text)
    if m:
        result["retries"] = int(m.group(1))
    return result


# ---------------------------------------------------------------------------
# Main CLI command
# ---------------------------------------------------------------------------


@dataclass
class _LerResult:
    """Accumulated result across batches."""

    shots: int = 0
    logical_errors: int = 0
    decode_time_total: float = 0.0
    retries: int = 0

    @property
    def error_rate(self) -> float:
        return self.logical_errors / self.shots if self.shots > 0 else 0.0


@arguably.command
def simulate__ler(
    *deq_files: str,
    program: str,
    save: str | None = None,
    shots: int = 100_000,
    errors: int = 100,
    batch_size: int = 100,
    decoder: str = "black-box-relay-bp",
    decoder_config: str | None = None,
    coordinator: str = "monolithic",
    coordinator_config: str | None = None,
    seed: int | None = None,
    debug_dir: str | None = None,
    jobs: int = max((os.cpu_count() or 1) - 2, 1),
    jit: str | None = None,
    #: Override the auto-generated .stim file (for debugging)
    stim: str | None = None,
    #: Mako variable definitions, each as key=value
    #: (e.g. --mako d=3 --mako p=0.01); implies --skip-mako-warning
    mako: list[str] | None = None,
    #: suppress the interactive Mako safety prompt
    skip_mako_warning: bool = False,
    #: simulator type: "static" (native Stim bulk sampler), "jit-static"
    #: (JIT-controller-driven), "preselect" (retry from gadget start via
    #: TableauSimulator), or "qdk" (Python sampler via the compile-time
    #: embedded ``@qdk_sampler`` adapter; the only path that supports
    #: loss-aware simulation).
    simulator: str = "static",
) -> None:
    """
    Run logical error rate simulation from .deq files.

    End-to-end pipeline: build JIT library, compile program, build simulator,
    then run deq_runtime to evaluate the logical error rate.

    Intermediate files (``.deq.jit``, ``.stim``, ``.deq.bin``) are placed
    in a temporary directory by default and cleaned up after the simulation.
    Pass ``--save DIR`` to persist them for inspection or reuse.

    When ``--jit PATH`` is given, the expensive gadget-type construction
    is skipped: the pre-compiled ``.deq.jit`` library is loaded and
    only the PROGRAM block is recompiled. This is much faster when
    iterating on program structure without changing gadget definitions.

    Example::

        deq simulate ler tests/circuit/repetition_code/*.deq \\
            --program MemoryExperiment --shots 10000

        deq simulate ler circuit.deq --program Sim --save ./output

    Args:
        deq_files: One or more .deq files containing CODE, GADGET,
            COMPOSE, and PROGRAM definitions.
        program: Name of the PROGRAM to simulate (e.g. "MemoryExperiment").
        save: Directory to persist intermediate files (.deq.jit, .stim,
            .deq.bin). When omitted, files are placed in a temporary
            directory and cleaned up automatically.
        shots: Maximum total shots across all batches.
        errors: Target number of logical errors (stop early once reached).
        batch_size: Shots per batch.
        decoder: Decoder to use (default: black-box-relay-bp).
        decoder_config: JSON string with decoder configuration
            (e.g. '{"cluster_node_limit": 100}').
        coordinator: Coordinator type: "monolithic" or "window".
        coordinator_config: JSON string with coordinator configuration
            (e.g. '{"buffer_radius": 2, "lookahead_radius": 0}').
        seed: Random seed for the simulator.
        jobs: Number of parallel worker processes.
        jit: path to a pre-compiled ``.deq.jit`` file to skip transpilation.
        debug_dir: Directory to dump intermediate files for inspection.
    """
    import tempfile
    import shutil
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    from deq.transpiler.jit_library_builder import build_jit_library
    from deq.compiler.jit_compiler import static_jit_compiler
    from deq.cli.jit import compile_program_for_jit, export_program_stim
    from deq.transpiler.jit_transpiler import flatten_body
    from deq.transpiler.jit_annotate import expand_compose_circuit
    from deq.circuit.mako_support import parse_mako_vars
    from deq.circuit.parser import render_and_parse_files
    import deq.proto.deq_jit_pb2 as jit_pb

    if not deq_files:
        raise ValueError("At least one .deq file is required")

    # Use a temp dir unless --save is given.
    tmpdir_ctx = tempfile.TemporaryDirectory() if save is None else None
    out = save if save is not None else tmpdir_ctx.__enter__()  # type: ignore[union-attr]
    try:
        os.makedirs(out, exist_ok=True)

        if debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)

        # --- Parse DEQ files ---
        print("Parsing DEQ files...")

        mako_vars = parse_mako_vars(mako) if mako else None
        merged = render_and_parse_files(
            list(deq_files), mako_defs=mako_vars, skip_mako_warning=skip_mako_warning
        )

        programs = {
            d.name: d for d in merged.definitions if isinstance(d, ProgramDefinition)
        }
        if program not in programs:
            available = list(programs.keys())
            raise ValueError(f"Program '{program}' not found. Available: {available}")

        program_def = programs[program]

        # --- Stage 1+2: Build JIT library and compile program ---
        if jit is not None:
            # Fast path: load pre-compiled JitLibrary, recompile PROGRAM only

            print(f"Loading pre-compiled JIT library from {jit}...")
            with open(jit, "rb") as f:
                jit_library = jit_pb.JitLibrary.FromString(f.read())

            # Sanity check: every gadget in .deq must exist in .deq.jit
            jit_names = {gt.base.name for gt in jit_library.gadget_types}
            deq_gadget_names = {
                d.name for d in merged.definitions if isinstance(d, GadgetDefinition)
            }
            missing = deq_gadget_names - jit_names
            if missing:
                raise ValueError(
                    f"Gadget(s) defined in .deq but missing from "
                    f"{jit!r}: {sorted(missing)}. "
                    f"The .deq.jit file may be stale — "
                    f"rebuild with transpile."
                )
        else:
            print("Building JIT library...")
            jit_library = build_jit_library(merged, jobs=jobs)

        # Compile program into JIT instructions
        print("Compiling program...")
        compiled, assertions = compile_program_for_jit(
            jit_library,
            program_def,
            programs,
            codes={
                d.name: d for d in merged.definitions if isinstance(d, CodeDefinition)
            },
        )
        for instr, _src in compiled:
            jit_library.program.append(instr)

        jit_path = os.path.join(out, f"{program}.deq.jit")
        with open(jit_path, "wb") as f:
            f.write(jit_library.SerializeToString())
        print(f"  JIT library: {jit_path}")

        # Export .stim circuit for the simulator
        gadgets_by_name: dict[str, GadgetDefinition] = {
            d.name: d for d in merged.definitions if isinstance(d, GadgetDefinition)
        }
        compose_defs: dict[str, ComposeDefinition] = {
            d.name: d for d in merged.definitions if isinstance(d, ComposeDefinition)
        }
        code_defs: dict[str, CodeDefinition] = {
            d.name: d for d in merged.definitions if isinstance(d, CodeDefinition)
        }
        # Expand each ComposeDefinition into a synthetic GadgetDefinition
        # so the stim exporter can inline its circuit body.
        known_names = set(gadgets_by_name) | set(compose_defs)
        for cname, cdef in compose_defs.items():
            in_ports, circuit_stmts, out_ports = expand_compose_circuit(
                cdef, gadgets_by_name, compose_defs, known_names, code_defs
            )
            gadgets_by_name[cname] = GadgetDefinition(
                name=cname,
                body=list(in_ports) + list(circuit_stmts) + list(out_ports),  # type: ignore[arg-type]
            )
        gtype_to_name = {gt.base.gtype: gt.base.name for gt in jit_library.gadget_types}
        stim_text = export_program_stim(
            jit_library,
            gadgets_by_name,
            gtype_to_name,
            flatten_body,
            program_def,
            [src for _instr, src in compiled],
            assertions,
        )
        stim_path = os.path.join(out, f"{program}.stim")
        if stim is not None:
            # User-provided stim file overrides the auto-generated one
            shutil.copy(stim, stim_path)
            print(f"  Stim circuit (override): {stim_path} (from {stim})")
        else:
            with open(stim_path, "w", encoding="utf-8") as f:
                f.write(stim_text)
            print(f"  Stim circuit: {stim_path}")

        # Compile to static binary (.deq.bin)
        deq_bin = static_jit_compiler(jit_library)
        bin_path = os.path.join(out, f"{program}.deq.bin")
        with open(bin_path, "wb") as f:
            f.write(deq_bin.SerializeToString())
        print(f"  Static binary: {bin_path}")

        # --- Stage 3: Run deq_runtime ---
        print(
            f"\nRunning LER simulation ({shots} max shots, "
            f"target {errors} errors, {jobs} worker(s))..."
        )

        result = _LerResult()
        next_seed = seed

        pbar = tqdm(
            total=errors,
            unit="err",
            desc="LER",
            bar_format="{desc} {n_fmt}/{total_fmt} [{bar}] {postfix}",
        )
        pbar.set_postfix_str("shots=0 rate=?")

        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {}

            def _submit_batch() -> bool:
                """Submit one batch if budget remains. Returns True if submitted."""
                nonlocal next_seed
                remaining_shots = (
                    shots - result.shots - sum(f_args[0] for f_args in futures.values())
                )
                if remaining_shots <= 0:
                    return False
                if result.logical_errors >= errors:
                    return False
                this_batch = min(batch_size, remaining_shots)
                if this_batch <= 0:
                    return False
                remaining_errors = errors - result.logical_errors
                fut = pool.submit(
                    _run_batch,
                    bin_path=bin_path,
                    stim_path=stim_path,
                    jit_path=jit_path,
                    batch_size=this_batch,
                    max_errors=remaining_errors,
                    decoder=decoder,
                    decoder_config=decoder_config,
                    coordinator=coordinator,
                    coordinator_config=coordinator_config,
                    seed=next_seed,
                    debug_dir=debug_dir,
                    simulator=simulator,
                )
                futures[fut] = (this_batch,)
                if next_seed is not None:
                    next_seed += 1
                return True

            # Prime the pool with initial batches.
            for _ in range(jobs):
                if not _submit_batch():
                    break

            while futures:
                for fut in as_completed(futures):
                    del futures[fut]
                    batch_result = fut.result()

                    batch_shots = int(batch_result.get("shots", 0))
                    batch_errors = int(batch_result.get("logical_errors", 0))
                    result.shots += batch_shots
                    result.logical_errors += batch_errors
                    result.retries += int(batch_result.get("retries", 0))
                    dt = float(batch_result.get("decode_time_per_shot", 0.0))
                    result.decode_time_total += dt * batch_shots

                    pbar.n = min(result.logical_errors, errors)
                    rate_str = f"{result.error_rate:.2e}" if result.shots else "?"
                    pbar.set_postfix_str(f"shots={result.shots} rate={rate_str}")
                    pbar.refresh()

                    # Refill: submit a new batch to replace the completed one.
                    _submit_batch()
                    break  # Re-enter as_completed with updated futures dict.

        pbar.close()

        # --- Report ---
        print("\n=== Simulation Results ===")
        print(f"  Shots:          {result.shots}")
        print(f"  Logical errors: {result.logical_errors}")
        if result.retries > 0:
            total = result.retries + result.shots
            pct = 100.0 * result.retries / max(total, 1)
            print(f"  Retries:        {result.retries} ({pct:.2f}%)")
        if result.shots > 0:
            rate = result.error_rate
            print(f"  Error rate:     {rate:.6e}")
            avg_time = (
                result.decode_time_total / result.shots if result.shots > 0 else 0.0
            )
            print(f"  Avg decode:     {avg_time:.6e} s/shot")
    finally:
        if tmpdir_ctx is not None:
            tmpdir_ctx.cleanup()


def _run_batch(
    bin_path: str,
    stim_path: str,
    jit_path: str,
    batch_size: int,
    max_errors: int,
    decoder: str,
    decoder_config: str | None,
    coordinator: str,
    coordinator_config: str | None,
    seed: int | None,
    debug_dir: str | None,
    simulator: str = "static",
) -> dict[str, int | float]:
    """Spawn one deq_runtime server process for a batch of shots."""
    simulator_config: dict[str, object] = {
        "filepath": stim_path,
        "shots": batch_size,
        "errors": max_errors,
    }
    if seed is not None:
        simulator_config["seed"] = seed
    if simulator == "jit-static":
        simulator_config["jit_library_filepath"] = jit_path
        controller_name = "jit"
        controller_config = {"filepath": jit_path}
        runtime_simulator = simulator
    elif simulator == "qdk":
        simulator_config["sampler"] = "@qdk_sampler"
        controller_name = "static"
        controller_config = {"filepath": bin_path}
        runtime_simulator = "python"
    else:
        controller_name = "static"
        controller_config = {"filepath": bin_path}
        runtime_simulator = simulator
    cmd = [
        sys.executable,
        "-m",
        "deq.runtime",
        "server",
        "--addr",
        "[::]:0",
        "--decoder",
        decoder,
        "--coordinator",
        coordinator,
        "--controller",
        controller_name,
        "--controller-config",
        json.dumps(controller_config),
        "--simulator",
        runtime_simulator,
        "--simulator-config",
        json.dumps(simulator_config),
    ]
    if decoder_config is not None:
        cmd += ["--decoder-config", decoder_config]
    if coordinator_config is not None:
        cmd += ["--coordinator-config", coordinator_config]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=36000,
        env={
            **os.environ,
            "TOKIO_WORKER_THREADS": "4",
            "RAYON_NUM_THREADS": "2",
        },
    )
    combined = proc.stdout + proc.stderr

    if debug_dir is not None:
        with open(
            os.path.join(debug_dir, "runtime_output.txt"),
            "a",
            encoding="utf-8",
        ) as f:
            f.write(f"--- batch (size={batch_size}, seed={seed}) ---\n")
            f.write(combined)
            f.write("\n")

    if proc.returncode != 0 or _SIMULATION_COMPLETE not in combined:
        raise RuntimeError(
            f"deq_runtime failed (exit {proc.returncode}):\n" + combined[:1000]
        )

    return _parse_server_output(combined)
