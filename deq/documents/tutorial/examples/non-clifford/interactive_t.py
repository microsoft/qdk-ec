"""Stream adaptive trivial-code T injections through QDK and native DEQ."""

import argparse
import asyncio
import json
import math
from pathlib import Path

import numpy as np
from qdk.simulation import Instrument, Operation, StateVectorSimulator

from deq.circuit.model import GadgetDefinition, Instruction
from deq.circuit.parser import render_and_parse_file
from deq.proto import coordinator_pb2 as coord_pb
from deq.proto import deq_bin_pb2 as bin_pb
from deq.proto import deq_jit_pb2 as jit_pb
from deq.proto import util_pb2 as util_pb
from deq.runtime import Runtime
from deq.transpiler.jit_library_builder import build_jit_library

HERE = Path(__file__).resolve().parent
IDENTITY = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.diag([1, -1]).astype(complex)
HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
PHASE = np.diag([1, 1j]).astype(complex)
CNOT = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)
RESET = Operation([[[1, 0], [0, 0]], [[0, 1], [0, 0]]])
MEASURE_Z = Instrument([Operation([[[1, 0], [0, 0]]]), Operation([[[0, 0], [0, 1]]])])
GATES = {name: Operation([matrix.tolist()]) for name, matrix in {
    "H": HADAMARD, "X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z,
    "S": PHASE, "S_DAG": PHASE.conj().T, "CX": CNOT,
}.items()}


def make_library(probability: float):
    parsed = render_and_parse_file(
        str(HERE / "interactive_t.deq"),
        mako_defs={"p": str(probability)},
        skip_mako_warning=True,
    )
    library = build_jit_library(parsed)
    assert all(not gadget.finished_checks and not gadget.unfinished_checks for gadget in library.gadget_types)
    gadgets = {definition.name: definition for definition in parsed.definitions
               if isinstance(definition, GadgetDefinition)}
    return library, gadgets


def pack_bits(bits: list[int]) -> util_pb.BitVector:
    packed = bytearray((len(bits) + 7) // 8)
    for index, bit in enumerate(bits):
        packed[index // 8] |= bit << (7 - index % 8)
    return util_pb.BitVector(size=len(bits), data=bytes(packed))


class StreamingShot:
    def __init__(self, controller, library, gadgets, probability: float, seed: int):
        self.controller = controller
        self.gadgets = gadgets
        self.types = {gadget.base.name: gadget.base.gtype for gadget in library.gadget_types}
        self.physics = StateVectorSimulator(2, seed=seed)
        self.noise = Operation([
            (math.sqrt(1 - probability) * IDENTITY).tolist(),
            *[(math.sqrt(probability / 3) * matrix).tolist() for matrix in (PAULI_X, PAULI_Y, PAULI_Z)],
        ])
        self.producers = {}
        self.t_branches = 0
        self.s_branches = 0

    def apply(self, instruction: Instruction) -> list[int]:
        name = instruction.name
        targets = [target.index for target in instruction.targets]
        if name in GATES:
            self.physics.apply_operation(GATES[name], targets)
        elif name in {"R", "RZ", "RX", "RY"}:
            for target in targets:
                self.physics.apply_operation(RESET, [target])
                if name in {"RX", "RY"}:
                    self.physics.apply_operation(GATES["H"], [target])
                if name == "RY":
                    self.physics.apply_operation(GATES["S"], [target])
        elif name == "R_Z":
            angle = math.pi * instruction.arguments[0]
            rotation = np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])
            self.physics.apply_operation(Operation([rotation.tolist()]), targets)
        elif name == "DEPOLARIZE1":
            for target in targets:
                self.physics.apply_operation(self.noise, [target])
        elif name in {"M", "MZ", "MX", "MY"}:
            outcomes = []
            for target in targets:
                if name == "MY":
                    self.physics.apply_operation(GATES["S_DAG"], [target])
                if name in {"MX", "MY"}:
                    self.physics.apply_operation(GATES["H"], [target])
                outcomes.append(self.physics.sample_instrument(MEASURE_Z, [target]))
                if name in {"MX", "MY"}:
                    self.physics.apply_operation(GATES["H"], [target])
                if name == "MY":
                    self.physics.apply_operation(GATES["S"], [target])
            return outcomes
        else:
            raise ValueError(f"Unsupported streaming instruction: {instruction}")
        return []

    async def step(self, name: str) -> list[int]:
        gadget = self.gadgets[name]
        connectors = [self.producers[tuple(port.qubit_indices)] for port in gadget.input_ports]
        gid = await self.controller.execute(jit_pb.JitInstruction(
            gadget=bin_pb.Gadget(gtype=self.types[name], connectors=connectors)
        ))
        outcomes = []
        for statement in gadget.body:
            if isinstance(statement, Instruction):
                outcomes.extend(self.apply(statement))
        decoded = await asyncio.wait_for(
            self.controller.decode(coord_pb.Outcomes(gid=gid, outcomes=pack_bits(outcomes))), timeout=10
        )
        for port in gadget.input_ports:
            del self.producers[tuple(port.qubit_indices)]
        for index, port in enumerate(gadget.output_ports):
            self.producers[tuple(port.qubit_indices)] = bin_pb.Gadget.Connector(gid=gid, port=index)
        return [(decoded.readouts.data[index // 8] >> (7 - index % 8)) & 1
                for index in range(decoded.readouts.size)]

    async def inject(self, gate: str, axis: str) -> int:
        await self.step(f"Prepare{gate}{axis}")
        await self.step(f"Couple{axis}")
        outcome, = await self.step(f"Read{axis}")
        return outcome

    async def inject_t(self, axis: str, *, inverse: bool = False):
        suffix = "Inv" if inverse else ""
        self.t_branches += 1
        outcome = await self.inject(f"T{suffix}", axis)
        if outcome:
            self.s_branches += 1
            correction_outcome = await self.inject(f"S{suffix}", axis)
            if correction_outcome:
                await self.step(f"Correct{axis}")


async def measure(runtime, library, gadgets, probability, axis, gates, preparation, measurement, shots, seed):
    controller = runtime.jit_controller
    ones = t_branches = s_branches = 0
    for shot_index in range(shots):
        await controller.reset(reset_decoder_service=True)
        shot = StreamingShot(controller, library, gadgets, probability, seed + shot_index)
        await shot.step(f"Prepare{preparation}")
        for _ in range(gates):
            await shot.inject_t(axis)
        outcome, = await shot.step(f"Measure{measurement}")
        ones += outcome
        t_branches += shot.t_branches
        s_branches += shot.s_branches
    return {"shots": shots, "ones": ones, "expectation": 1 - 2 * ones / shots,
            "t_readouts": t_branches, "conditional_s_readouts": s_branches}


async def run(arguments):
    results = {"shots_per_basis": arguments.shots, "seed": arguments.seed, "tomography": [], "noise": []}
    axes = ("Z", "X") if arguments.axis == "both" else (arguments.axis.upper(),)
    for probability in (0.0, arguments.noise):
        library, gadgets = make_library(probability)
        async with Runtime(
            decoder="black-box-tesseract", coordinator="window",
            coordinator_config={"buffer_radius": 0, "lookahead_radius": 0}, controller="jit",
        ) as runtime:
            await runtime.jit_controller.load_library(library)
            for axis_index, axis in enumerate(axes):
                if probability == 0:
                    for gates in range(1, 5):
                        preparation = "X" if axis == "Z" else "Z"
                        angle = gates * math.pi / 4
                        ideal = {"X": math.cos(angle), "Y": math.sin(angle), "Z": 0.0} if axis == "Z" else {
                            "X": 0.0, "Y": -math.sin(angle), "Z": math.cos(angle)
                        }
                        for basis_index, basis in enumerate(("X", "Y", "Z")):
                            seed = arguments.seed + ((axis_index * 4 + gates - 1) * 3 + basis_index) * arguments.shots
                            observed = await measure(runtime, library, gadgets, probability, axis, gates,
                                                     preparation, basis, arguments.shots, seed)
                            expected = ideal[basis]
                            tolerance = 6 * math.sqrt(max(0, 1 - expected**2) / arguments.shots) + 0.01
                            assert abs(observed["expectation"] - expected) <= tolerance, (axis, gates, basis, observed, expected)
                            results["tomography"].append({"axis": axis, "gates": gates, "prep": preparation,
                                                          "basis": basis, "ideal": expected, **observed})
                            print(f"{axis} N={gates} <{basis}>={observed['expectation']:+.5f} ideal={expected:+.5f}", flush=True)
                else:
                    bases = ("X",) if axis == "Z" else ("X", "Z")
                    for basis_index, basis in enumerate(bases):
                        observed = await measure(runtime, library, gadgets, probability, axis, 4, basis, basis,
                                                 arguments.noise_shots, arguments.seed + 1000000 +
                                                 (axis_index * 2 + basis_index) * arguments.noise_shots)
                        expected_bit = int(basis != axis)
                        errors = observed["ones"] if expected_bit == 0 else observed["shots"] - observed["ones"]
                        row = {"axis": axis, "gates": 4, "prep": basis, "basis": basis, "p": probability,
                               "expected_bit": expected_bit, "errors": errors, "ler": errors / observed["shots"], **observed}
                        results["noise"].append(row)
                        print(f"Noisy T_{axis}^4, {basis}->{basis}: {errors}/{observed['shots']} LER={row['ler']:.6g}", flush=True)
    if arguments.output:
        arguments.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=("x", "z", "both"), default="both")
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--noise-shots", type=int, default=10000)
    parser.add_argument("--noise", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=144)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.shots <= 0 or arguments.noise_shots <= 0 or not 0 < arguments.noise < 0.5:
        parser.error("shot counts must be positive and 0 < noise < 0.5")
    asyncio.run(run(arguments))


if __name__ == "__main__":
    main()
