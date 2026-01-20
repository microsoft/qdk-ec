#!/usr/bin/env python3
"""
Quantum Error Correction Code Benchmarks

Benchmarking paulimer's simulation classes for stabilizer simulation with Pauli noise
across different code families (surface code, honeycomb code).

Based on https://developer.nvidia.com/blog/advanced-large-scale-quantum-simulation-techniques-in-cuquantum-sdk-v25-11/
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import stim

cust = None
try:
    import cuquantum.stabilizer as cust
    HAS_CUSTABILIZER = True
except Exception:
    HAS_CUSTABILIZER = False

import paulimer

print(f"paulimer location: {paulimer.__file__}")

from paulimer import (
    FaultySimulation,
    PauliDistribution,
    PauliFault,
    SparsePauli,
    UnitaryOpcode,
)

STIM_TO_PAULIMER_GATE = {
    "I": UnitaryOpcode.I,
    "H": UnitaryOpcode.Hadamard,
    "X": UnitaryOpcode.X,
    "Y": UnitaryOpcode.Y,
    "Z": UnitaryOpcode.Z,
    "S": UnitaryOpcode.SqrtZ,
    "S_DAG": UnitaryOpcode.SqrtZInv,
    "SQRT_X": UnitaryOpcode.SqrtX,
    "SQRT_X_DAG": UnitaryOpcode.SqrtXInv,
    "SQRT_Y": UnitaryOpcode.SqrtY,
    "SQRT_Y_DAG": UnitaryOpcode.SqrtYInv,
    "CX": UnitaryOpcode.ControlledX,
    "CNOT": UnitaryOpcode.ControlledX,
    "CZ": UnitaryOpcode.ControlledZ,
    "SWAP": UnitaryOpcode.Swap,
}

TWO_QUBIT_GATES = {"CX", "CNOT", "CZ", "SWAP"}

SKIP_INSTRUCTIONS = {"TICK", "QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"}


class CodeFamily(Enum):
    SURFACE = "surface"
    HONEYCOMB = "honeycomb"


def get_cpu_name() -> str:
    """Get the CPU model name."""
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        elif platform.system() == "Windows":
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    return lines[1].strip()
    except Exception:
        pass
    return "CPU"


def get_gpu_name() -> str:
    """Get the NVIDIA GPU model name, or None if not available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
    except FileNotFoundError:
        pass
    return "GPU"


def shorten_cpu_name(full_name: str) -> str:
    """Extract short CPU model name (e.g., 'i7-1370P' from full Intel name)."""
    import re

    if match := re.search(r"(i[3579]-\w+)", full_name):
        return match.group(1)
    if match := re.search(r"(Ryzen \d+ \w+)", full_name):
        return match.group(1)
    if match := re.search(r"(Apple M\d+)", full_name):
        return match.group(1)
    return full_name.split()[-1] if full_name else "CPU"


def shorten_gpu_name(full_name: str) -> str:
    """Extract short GPU model name (e.g., 'RTX A500' from full NVIDIA name)."""
    import re

    full_name = full_name.replace("NVIDIA ", "").replace(" Laptop GPU", "")
    if match := re.search(r"(RTX [A-Z]?\d+[^,]*)", full_name):
        return match.group(1).strip()
    if match := re.search(r"(GTX \d+[^,]*)", full_name):
        return match.group(1).strip()
    return full_name


# ========== Surface Code ==========


def create_surface_code_stim_circuit(distance: int, p_noise: float = 0.0) -> stim.Circuit:
    """Create a rotated surface code memory experiment circuit."""
    kwargs = {"distance": distance, "rounds": distance}
    if p_noise > 0:
        kwargs.update(
            after_clifford_depolarization=p_noise,
            after_reset_flip_probability=p_noise,
            before_measure_flip_probability=p_noise,
        )
    return stim.Circuit.generated("surface_code:rotated_memory_z", **kwargs)


# ========== Honeycomb Code ==========
# Note: This circuit is not an exact match of the honeycomb code.  It is just an approximation.


def create_honeycomb_stim_circuit(distance: int, rounds: Optional[int] = None, p_noise: float = 0.0) -> stim.Circuit:
    """
    Create a stim honeycomb code circuit with weight-2 XX, YY, ZZ measurements.
    Uses gate decomposition for compatibility with stim.
    """
    if rounds is None:
        rounds = distance
    
    rows = distance
    cols = distance
    num_qubits = rows * cols
    
    def qubit_index(row: int, col: int) -> int:
        return row * cols + col
    
    def get_xx_pairs() -> list[tuple[int, int]]:
        pairs = []
        for row in range(rows):
            start = 0 if row % 2 == 0 else 1
            for col in range(start, cols - 1, 2):
                pairs.append((qubit_index(row, col), qubit_index(row, col + 1)))
        return pairs
    
    def get_yy_pairs() -> list[tuple[int, int]]:
        pairs = []
        for row in range(rows - 1):
            start = 1 if row % 2 == 0 else 0
            for col in range(start, cols, 2):
                pairs.append((qubit_index(row, col), qubit_index(row + 1, col)))
        return pairs
    
    def get_zz_pairs() -> list[tuple[int, int]]:
        pairs = []
        for row in range(rows - 1):
            start = 0 if row % 2 == 0 else 1
            for col in range(start, cols, 2):
                pairs.append((qubit_index(row, col), qubit_index(row + 1, col)))
        return pairs
    
    xx_pairs = get_xx_pairs()
    yy_pairs = get_yy_pairs()
    zz_pairs = get_zz_pairs()
    all_qubits = list(range(num_qubits))
    
    circuit = stim.Circuit()
    
    for _ in range(rounds):
        for q1, q2 in xx_pairs:
            circuit.append("H", [q1, q2])
        if p_noise > 0:
            circuit.append("DEPOLARIZE1", all_qubits, p_noise)
        for q1, q2 in xx_pairs:
            circuit.append("CX", [q1, q2])
        if p_noise > 0:
            for q1, q2 in xx_pairs:
                circuit.append("DEPOLARIZE2", [q1, q2], p_noise)
        for q1, q2 in xx_pairs:
            circuit.append("MR", [q2])
        for q1, q2 in xx_pairs:
            circuit.append("CX", [q1, q2])
        if p_noise > 0:
            for q1, q2 in xx_pairs:
                circuit.append("DEPOLARIZE2", [q1, q2], p_noise)
        for q1, q2 in xx_pairs:
            circuit.append("H", [q1, q2])
        circuit.append("TICK")
        
        for q1, q2 in yy_pairs:
            circuit.append("S_DAG", [q1, q2])
            circuit.append("H", [q1, q2])
        if p_noise > 0:
            circuit.append("DEPOLARIZE1", all_qubits, p_noise)
        for q1, q2 in yy_pairs:
            circuit.append("CX", [q1, q2])
        if p_noise > 0:
            for q1, q2 in yy_pairs:
                circuit.append("DEPOLARIZE2", [q1, q2], p_noise)
        for q1, q2 in yy_pairs:
            circuit.append("MR", [q2])
        for q1, q2 in yy_pairs:
            circuit.append("CX", [q1, q2])
        if p_noise > 0:
            for q1, q2 in yy_pairs:
                circuit.append("DEPOLARIZE2", [q1, q2], p_noise)
        for q1, q2 in yy_pairs:
            circuit.append("H", [q1, q2])
            circuit.append("S", [q1, q2])
        circuit.append("TICK")
        
        for q1, q2 in zz_pairs:
            circuit.append("CX", [q1, q2])
        if p_noise > 0:
            for q1, q2 in zz_pairs:
                circuit.append("DEPOLARIZE2", [q1, q2], p_noise)
        for q1, q2 in zz_pairs:
            circuit.append("MR", [q2])
        for q1, q2 in zz_pairs:
            circuit.append("CX", [q1, q2])
        if p_noise > 0:
            for q1, q2 in zz_pairs:
                circuit.append("DEPOLARIZE2", [q1, q2], p_noise)
        circuit.append("TICK")
    
    return circuit


# ========== Stim to paulimer conversion ==========


def stim_circuit_to_faulty_simulation(stim_circuit: stim.Circuit) -> FaultySimulation:
    """Convert a stim circuit to a FaultySimulation."""
    # Pre-allocate capacity based on circuit stats for better performance
    initialized = set()
    qubit_count = stim_circuit.num_qubits
    outcome_count = stim_circuit.num_measurements
    instruction_count = stim_circuit.num_ticks + outcome_count
    sim = FaultySimulation(
        qubit_count=qubit_count,
        outcome_count=outcome_count,
        instruction_count=instruction_count,
    )

    for instruction in stim_circuit.flattened():
        name = instruction.name
        targets = instruction.targets_copy()

        if name in SKIP_INSTRUCTIONS:
            continue

        if name in ("R", "RZ"):
            if len(initialized & set(targets)):
                raise ValueError(f"Mid-circuit reset is not supported in paulimer: {instruction}")
        elif name == "RX":
            if len(initialized & set(targets)):
                raise ValueError(f"Mid-circuit reset is not supported in paulimer: {instruction}")
            for t in targets:
                sim.apply_unitary(UnitaryOpcode.Hadamard, [t.value])
        elif name == "RY":
            if len(initialized & set(targets)):
                raise ValueError(f"Mid-circuit reset is not supported in paulimer: {instruction}")
            for t in targets:
                sim.apply_unitary(UnitaryOpcode.Hadamard, [t.value])
                sim.apply_unitary(UnitaryOpcode.SqrtZ, [t.value])
        elif name in ("MR", "MRZ"):
            for t in targets:
                qubit = t.value
                outcome_id = sim.measure(SparsePauli.z(qubit))
                sim.apply_conditional_pauli(SparsePauli.x(qubit), [outcome_id])
        elif name == "MRX":
            for t in targets:
                qubit = t.value
                outcome_id = sim.measure(SparsePauli.x(qubit))
                sim.apply_conditional_pauli(SparsePauli.z(qubit), [outcome_id])
        elif name == "MRY":
            for t in targets:
                qubit = t.value
                outcome_id = sim.measure(SparsePauli.y(qubit))
                # Z flips between Y eigenstates: Z|+i⟩ = |−i⟩
                sim.apply_conditional_pauli(SparsePauli.z(qubit), [outcome_id])
        elif name == "DEPOLARIZE1":
            p = instruction.gate_args_copy()[0]
            for t in targets:
                sim.apply_fault(PauliFault.depolarizing(p, [t.value]))
        elif name == "DEPOLARIZE2":
            p = instruction.gate_args_copy()[0]
            for i in range(0, len(targets), 2):
                sim.apply_fault(
                    PauliFault.depolarizing(p, [targets[i].value, targets[i + 1].value])
                )
        elif name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
            p = instruction.gate_args_copy()[0]
            pauli_char = name[0]
            for t in targets:
                pauli = getattr(SparsePauli, pauli_char.lower())(t.value)
                sim.apply_fault(PauliFault(p, PauliDistribution.single(pauli)))
        elif name in ("M", "MZ"):
            for t in targets:
                sim.measure(SparsePauli.z(t.value))
        elif name == "MX":
            for t in targets:
                sim.measure(SparsePauli.x(t.value))
        elif name == "MY":
            for t in targets:
                sim.measure(SparsePauli.y(t.value))
        elif name in STIM_TO_PAULIMER_GATE:
            opcode = STIM_TO_PAULIMER_GATE[name]
            if name in TWO_QUBIT_GATES:
                for i in range(0, len(targets), 2):
                    sim.apply_unitary(opcode, [targets[i].value, targets[i + 1].value])
            else:
                for t in targets:
                    sim.apply_unitary(opcode, [t.value])
        else:
            raise ValueError(f"Unsupported stim instruction for paulimer: {instruction}")
        
        initialized |= set(targets)

    return sim


# ========== Benchmark Infrastructure ==========


@dataclass
class BenchmarkResult:
    """Result from benchmarking a single distance."""

    code_family: str
    distance: int
    qubits: int
    faults: int
    detectors: int
    stim_time: float | None = None
    faulty_time: float | None = None
    faulty_build_time: float | None = None
    faulty_sample_time: float | None = None
    custabilizer_time: float | None = None

    def to_dict(self) -> dict:
        return {
            "code_family": self.code_family,
            "distance": self.distance,
            "qubits": self.qubits,
            "faults": self.faults,
            "detectors": self.detectors,
            "stim_time": self.stim_time,
            "faulty_time": self.faulty_time,
            "faulty_build_time": self.faulty_build_time,
            "faulty_sample_time": self.faulty_sample_time,
            "custabilizer_time": self.custabilizer_time,
        }


@dataclass
class SimulatorState:
    """Tracks whether each simulator is enabled and within time limit."""

    stim: bool = True
    faulty: bool = True
    custabilizer: bool = False

    def any_enabled(self) -> bool:
        return self.stim or self.faulty or self.custabilizer


class CodeFamilyBenchmark:
    """Runs benchmarks for a specific code family across multiple simulators."""

    def __init__(
        self,
        code_family: CodeFamily,
        shots: int,
        p_noise: float,
        time_limit: float | None = None,
        skip_stim: bool = False,
        skip_custabilizer: bool = True,
    ):
        self.code_family = code_family
        self.shots = shots
        self.p_noise = p_noise
        self.time_limit = time_limit
        self.enabled = SimulatorState(
            stim=not skip_stim,
            faulty=True,
            custabilizer=not skip_custabilizer and HAS_CUSTABILIZER,
        )
        self.results: list[BenchmarkResult] = []

    def run(self, distances: list[int]) -> list[BenchmarkResult]:
        """Run benchmarks for all distances."""
        self._print_header()

        for distance in distances:
            result = self._benchmark_distance(distance)
            self.results.append(result)
            self._print_result(result)

        print("\nBenchmark complete!")
        return self.results

    def _create_stim_circuit(self, distance: int, p_noise: float) -> stim.Circuit:
        """Create the appropriate stim circuit for this code family."""
        if self.code_family == CodeFamily.SURFACE:
            return create_surface_code_stim_circuit(distance, p_noise)
        elif self.code_family == CodeFamily.HONEYCOMB:
            return create_honeycomb_stim_circuit(distance, p_noise=p_noise)
        else:
            raise ValueError(f"Unknown code family: {self.code_family}")

    def _benchmark_distance(self, distance: int) -> BenchmarkResult:
        """Benchmark all simulators for a single code distance."""
        noisy_circuit = self._create_stim_circuit(distance, self.p_noise)
        noiseless_circuit = self._create_stim_circuit(distance, p_noise=0.0)

        self._warmup(noisy_circuit, noiseless_circuit)

        result = BenchmarkResult(
            code_family=self.code_family.value,
            distance=distance,
            qubits=noisy_circuit.num_qubits,
            faults=stim_circuit_to_faulty_simulation(noiseless_circuit).fault_count,
            detectors=noisy_circuit.num_detectors,
        )

        if self.enabled.stim:
            result.stim_time = self._benchmark_stim(noisy_circuit)
            self._check_time_limit(result.stim_time, "stim")

        if self.enabled.faulty:
            total, build, sample = self._benchmark_faulty(noisy_circuit)
            result.faulty_time = total
            result.faulty_build_time = build
            result.faulty_sample_time = sample
            self._check_time_limit(total, "faulty")

        if self.enabled.custabilizer:
            try:
                result.custabilizer_time = self._benchmark_custabilizer(noisy_circuit)
                self._check_time_limit(result.custabilizer_time, "custabilizer")
            except Exception as e:
                print(f"cuStabilizer sampling failed: {e}")
                result.custabilizer_time = None
                self.enabled.custabilizer = False

        return result

    def _warmup(
        self, noisy_circuit: stim.Circuit, noiseless_circuit: stim.Circuit
    ) -> None:
        """Run warmup iterations for enabled simulators."""
        if self.enabled.stim:
            noisy_circuit.compile_sampler().sample(100, bit_packed=True)

        if self.enabled.faulty:
            stim_circuit_to_faulty_simulation(noisy_circuit).sample(100)

        if self.enabled.custabilizer:
            cu_circuit = cust.Circuit(noisy_circuit)
            sim = cust.FrameSimulator(
                noisy_circuit.num_qubits,
                10,
                noisy_circuit.num_measurements,
                num_detectors=noisy_circuit.num_detectors,
            )
            sim.apply(cu_circuit)
            sim.get_measurement_bits(bit_packed=True)

    def _benchmark_stim(self, circuit: stim.Circuit) -> float:
        """Benchmark stim noisy sampling."""
        start = time.perf_counter()
        sampler = circuit.compile_sampler()
        sampler.sample(self.shots, bit_packed=True)
        return time.perf_counter() - start

    def _benchmark_faulty(
        self, noisy_circuit: stim.Circuit
    ) -> tuple[float, float, float]:
        """Benchmark FaultySimulation. Returns (total, build, sample)."""
        start = time.perf_counter()
        sim = stim_circuit_to_faulty_simulation(noisy_circuit)
        mid = time.perf_counter()
        sim.sample(self.shots)
        end = time.perf_counter()
        return end - start, mid - start, end - mid

    def _benchmark_custabilizer(self, circuit: stim.Circuit) -> float:
        """Benchmark cuStabilizer GPU simulation."""
        start = time.perf_counter()
        cu_circuit = cust.Circuit(circuit)
        sim = cust.FrameSimulator(
            circuit.num_qubits,
            self.shots,
            circuit.num_measurements,
            num_detectors=circuit.num_detectors,
        )
        sim.apply(cu_circuit)
        sim.get_measurement_bits(bit_packed=True)
        return time.perf_counter() - start

    def _check_time_limit(self, elapsed: float | None, simulator: str) -> None:
        """Disable a simulator if it exceeded the time limit."""
        if self.time_limit and elapsed and elapsed > self.time_limit:
            setattr(self.enabled, simulator, False)

    def _print_header(self) -> None:
        """Print the benchmark table header."""
        cpu_name = get_cpu_name()
        gpu_name = get_gpu_name()

        print(f"\nCPU: {cpu_name}")
        print(f"GPU: {gpu_name}")

        print(
            f"\nBenchmarking noisy {self.code_family.value} code sampling with "
            f"{self.shots:,} shots, p_error={self.p_noise}"
        )
        if self.time_limit:
            print(f"Time limit per simulator: {self.time_limit}s")

        columns = ["Distance", "Qubits", "Faults", "stim (s)", "paulimer"]
        if self.enabled.custabilizer:
            columns.append("cuStab")
        columns.append("vs stim")

        header = "".join(f"{col:>12}" for col in columns)
        print(header)
        print("-" * len(header))

    def _print_result(self, result: BenchmarkResult) -> None:
        """Print a single result row."""

        def fmt(val: float | None) -> str:
            return f"{val:>12.4f}" if val is not None else f"{'—':>12}"

        speedup = "—"
        if result.stim_time and result.faulty_time:
            speedup = f"{result.stim_time / result.faulty_time:.1f}x"

        parts = [
            f"{result.distance:>12}",
            f"{result.qubits:>12}",
            f"{result.faults:>12}",
            fmt(result.stim_time),
            fmt(result.faulty_time),
        ]
        if self.enabled.custabilizer or result.custabilizer_time is not None:
            parts.append(fmt(result.custabilizer_time))
        parts.append(f"{speedup:>12}")

        print("".join(parts))

    def save_json(self, path: str) -> None:
        """Save results to a JSON file."""
        data = {
            "cpu": get_cpu_name(),
            "gpu": get_gpu_name(),
            "code_family": self.code_family.value,
            "shots": self.shots,
            "p_noise": self.p_noise,
            "results": [r.to_dict() for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to: {path}")

    def plot(self, output_path: str = "benchmark_results.png") -> None:
        """Generate a plot of the benchmark results."""
        import matplotlib.pyplot as plt

        distances = [r.distance for r in self.results]

        fig, ax = plt.subplots(figsize=(10, 6))

        def plot_series(times: list[float | None], label: str, marker: str, color: str):
            valid = [(d, t) for d, t in zip(distances, times) if t is not None]
            if valid:
                ax.semilogy(
                    [d for d, _ in valid],
                    [t for _, t in valid],
                    f"{marker}-",
                    color=color,
                    label=label,
                    linewidth=2,
                    markersize=8,
                )

        cpu_short = shorten_cpu_name(get_cpu_name())
        gpu_short = shorten_gpu_name(get_gpu_name())

        plot_series([r.stim_time for r in self.results], f"stim ({cpu_short})", "o", "black")
        plot_series(
            [r.faulty_time for r in self.results], f"paulimer ({cpu_short})", "s", "dodgerblue"
        )
        plot_series(
            [r.custabilizer_time for r in self.results],
            f"cuStabilizer ({gpu_short})",
            "^",
            "forestgreen",
        )

        ax.set_xlabel("Code Distance (d)", fontsize=12)
        ax.set_ylabel("Time (seconds)", fontsize=12)
        ax.set_title(
            f"{self.code_family.value.title()} Code simulation\n({self.shots:,} samples, p={self.p_noise})",
            fontsize=14,
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(distances)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark paulimer vs stim for quantum error correction code simulation"
    )
    parser.add_argument(
        "--code",
        type=str,
        choices=["surface", "honeycomb"],
        default="surface",
        help="Code family to benchmark (default: surface)",
    )
    parser.add_argument("--shots", type=int, default=1_000_000, help="Number of shots")
    parser.add_argument("--p-noise", type=float, default=0.001, help="Error probability")
    parser.add_argument(
        "--distances",
        type=int,
        nargs="+",
        default=None,
        help="Code distances to test (default: code-family-specific)",
    )
    parser.add_argument(
        "--skip-stim", action="store_true", help="Skip stim benchmarks"
    )
    parser.add_argument(
        "--skip-custabilizer",
        default=False,
        action="store_true",
        help="Skip cuStabilizer (GPU) benchmarks (default: False)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Time limit per simulator (seconds). Once exceeded, that simulator is skipped.",
    )
    parser.add_argument(
        "--plot", type=str, default="benchmark_results.png", help="Output plot filename"
    )
    parser.add_argument(
        "--save-json", type=str, default=None, help="Save results to JSON file"
    )
    args = parser.parse_args()

    code_family = CodeFamily(args.code)

    if args.distances is None:
        if code_family == CodeFamily.SURFACE:
            distances = [3, 5, 7, 9, 11, 13]
        elif code_family == CodeFamily.HONEYCOMB:
            distances = [5, 9, 13, 17, 21, 25]
        else:
            distances = [3, 5, 7, 9, 11]
    else:
        distances = args.distances

    if not args.skip_custabilizer and not HAS_CUSTABILIZER:
        print(
            "Warning: cuStabilizer (cuquantum.stabilizer) not available. "
            "Skipping GPU benchmarks."
        )

    runner = CodeFamilyBenchmark(
        code_family=code_family,
        shots=args.shots,
        p_noise=args.p_noise,
        time_limit=args.time_limit,
        skip_stim=args.skip_stim,
        skip_custabilizer=args.skip_custabilizer,
    )

    runner.run(distances)

    if args.save_json:
        runner.save_json(args.save_json)

    if args.plot:
        runner.plot(args.plot)


if __name__ == "__main__":
    main()
