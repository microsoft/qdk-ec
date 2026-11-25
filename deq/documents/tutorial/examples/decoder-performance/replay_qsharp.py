"""Compact streaming Q# sampling of the existing physical gadget workload."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from deq.circuit.parser import parse_file
from deq.cli.jit import _is_synthesised_identity_gadget
from deq.proto.deq_jit_pb2 import JitLibrary
from deq.transpiler.compose_builder import compose_to_synthetic_gadget
from deq.transpiler.jit_transpiler import flatten_body
from deq.transpiler.program_artifacts import ProgramDefinitions
from deqagram import Instruction, PreselectStatement, QubitTarget

QSHARP_SOURCE = Path(__file__).with_name("replay_sampler.qs")
MAX_FILE_BYTES = 10_000_000_000
OPCODES = {
    "R": 0,
    "RZ": 0,
    "RX": 1,
    "X": 2,
    "Y": 3,
    "Z": 4,
    "H": 5,
    "CX": 6,
    "CNOT": 6,
    "CZ": 7,
    "M": 14,
    "MZ": 14,
    "MX": 15,
    "X_ERROR": 10,
    "Z_ERROR": 11,
    "DEPOLARIZE1": 12,
    "DEPOLARIZE2": 13,
}


@dataclass
class Recipe:
    program: list[list[int]]
    inputs: list[list[list[int]]]
    outputs: list[list[list[int]]]
    used: list[list[int]]
    retired: list[list[int]]
    bodies: list[list[list[int]]]
    measurements: list[int]
    virtual: list[bool]
    qubits: int
    total_measurements: int

    def arguments(self) -> tuple[Any, ...]:
        return (
            self.program,
            self.inputs,
            self.outputs,
            self.used,
            self.retired,
            self.bodies,
            self.measurements,
            self.virtual,
            self.qubits,
            self.total_measurements,
        )

    @property
    def shot_bytes(self) -> int:
        return (self.total_measurements + 7) // 8


def build_recipe(source: Path, library_path: Path, probability: float) -> Recipe:
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Noise probability must lie between zero and one")
    for path in (source, library_path):
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"{path} exceeds the 10 GB guard")
    indexed = ProgramDefinitions.from_deq_file(parse_file(source))
    definitions = dict(indexed.gadgets)
    compositions = {}
    for name, definition in indexed.composes.items():
        definitions[name] = compose_to_synthetic_gadget(
            definition, definitions, compositions, indexed.codes
        )
        compositions[name] = definition
    library = JitLibrary.FromString(library_path.read_bytes())
    inputs: list[list[list[int]]] = []
    outputs: list[list[list[int]]] = []
    used: list[list[int]] = []
    retired: list[list[int]] = []
    bodies: list[list[list[int]]] = []
    measurements: list[int] = []
    virtual: list[bool] = []
    kinds = {}
    for kind, definition in enumerate(library.gadget_types):
        base = definition.base
        kinds[base.gtype] = kind
        is_virtual = base.name not in definitions and _is_synthesised_identity_gadget(
            base.name
        )
        if base.name not in definitions and not is_virtual:
            raise ValueError(f"No physical definition for {base.name}")
        virtual.append(is_virtual)
        measurements.append(len(base.measurements))
        if is_virtual:
            if base.measurements:
                raise ValueError("Virtual passthrough has physical measurements")
            inputs.append([])
            outputs.append([])
            used.append([])
            retired.append([])
            bodies.append([])
            continue
        gadget = definitions[base.name]
        input_ports = [list(port.qubit_indices) for port in gadget.input_ports]
        output_ports = [list(port.qubit_indices) for port in gadget.output_ports]
        if len(input_ports) > 2 or len(output_ports) > 2:
            raise ValueError(
                "This workload sampler supports at most two ports per gadget"
            )
        input_qubits = {qubit for port in input_ports for qubit in port}
        output_qubits = {qubit for port in output_ports for qubit in port}
        if sum(map(len, input_ports)) != len(input_qubits) or sum(
            map(len, output_ports)
        ) != len(output_qubits):
            raise ValueError("Physical ports must be disjoint")
        all_qubits = input_qubits | output_qubits
        pending_measurements: set[int] = set()
        body = []
        counted = 0
        for instruction in flatten_body(list(gadget.body), for_simulate=True):
            if isinstance(instruction, PreselectStatement):
                raise ValueError("Postselection is not supported by the replay sampler")
            if not isinstance(instruction, Instruction):
                continue
            name = instruction.name.upper()
            if name == "TICK":
                continue
            if name not in OPCODES:
                raise ValueError(f"Unsupported physical instruction {name}")
            if any(
                not isinstance(target, QubitTarget) or target.inverted
                for target in instruction.targets
            ):
                raise ValueError(f"Unsupported target in {name}")
            targets = [target.index for target in instruction.targets]
            all_qubits.update(targets)
            if name in ("R", "RZ", "RX"):
                pending_measurements.difference_update(targets)
            elif pending_measurements.intersection(targets):
                raise ValueError(
                    "Noisy measurement qubits must be reset before further quantum use"
                )
            opcode = OPCODES[name]
            arguments = list(instruction.arguments)
            if arguments:
                if (
                    opcode not in (10, 11, 12, 13, 14, 15)
                    or len(arguments) != 1
                    or float(arguments[0]) not in (0.0, probability)
                ):
                    raise ValueError(
                        f"Unexpected noise parameters in {name}: {arguments}"
                    )
                if opcode in (14, 15) and float(arguments[0]) != 0:
                    opcode -= 6
                    pending_measurements.update(targets)
                elif opcode in (10, 11, 12, 13) and float(arguments[0]) == 0:
                    continue
            elif opcode in (10, 11, 12, 13):
                raise ValueError(f"Missing probability for {name}")
            if opcode in (6, 7, 13) and len(targets) % 2:
                raise ValueError(f"Unpaired targets in {name}")
            if opcode in (8, 9, 14, 15):
                counted += len(targets)
            body.append([opcode, *targets])
        if pending_measurements & output_qubits:
            raise ValueError(
                "Noisy measurement qubits cannot remain live at the gadget boundary"
            )
        if counted != len(base.measurements):
            raise ValueError(f"Measurement metadata mismatch for {base.name}")
        if not all_qubits:
            raise ValueError(f"Empty physical gadget {base.name}")
        inputs.append(input_ports)
        outputs.append(output_ports)
        used.append(sorted(all_qubits))
        retired.append(sorted(all_qubits - output_qubits))
        bodies.append(body)

    program = []
    total_measurements = 0
    live = peak = 0
    for gid, instruction in enumerate(library.program, 1):
        gadget = instruction.gadget
        if gadget.gid != gid:
            raise ValueError("Compiled program IDs must be sequential")
        kind = kinds[gadget.gtype]
        connectors = [
            [connector.gid, connector.port] for connector in gadget.connectors
        ]
        expected = 1 if virtual[kind] else len(inputs[kind])
        if len(connectors) != expected or any(
            parent < 1 or parent >= gid or port not in (0, 1)
            for parent, port in connectors
        ):
            raise ValueError(f"Invalid program connectors for gadget {gid}")
        connectors += [[0, 0]] * (2 - len(connectors))
        program.append([kind, *connectors[0], *connectors[1]])
        total_measurements += measurements[kind]
        if not virtual[kind]:
            input_qubits = {q for port in inputs[kind] for q in port}
            output_qubits = {q for port in outputs[kind] for q in port}
            peak = max(peak, live + len(set(used[kind]) - input_qubits))
            live += len(output_qubits) - len(input_qubits)
    if live != 0:
        raise ValueError("The sampling workload must consume all output ports")
    recipe = Recipe(
        program,
        inputs,
        outputs,
        used,
        retired,
        bodies,
        measurements,
        virtual,
        peak,
        total_measurements,
    )
    if recipe.shot_bytes > MAX_FILE_BYTES:
        raise ValueError("Raw shot exceeds the 10 GB guard; user direction is required")
    return recipe


class QSharpSampler:
    def __init__(self, recipe: Recipe, probability: float):
        import qdk
        from qdk.simulation import NoiseConfig

        self.recipe = recipe
        self.context = qdk.Context(target_profile=qdk.TargetProfile.Unrestricted)
        self.context.eval(QSHARP_SOURCE.read_text())
        self.noise = NoiseConfig()
        self.noise.rz.set_bitflip(probability)
        self.noise.rx.set_phaseflip(probability)
        self.noise.ry.set_depolarizing(probability)
        self.noise.rxx.set_depolarizing(probability)

    def sample(self, seed: int) -> bytes:
        words = self.context.run(
            self.context.code.ReplaySampler.Sample,
            1,
            *self.recipe.arguments(),
            noise=self.noise,
            seed=seed,
            type="stabilizer",
            num_qubits=self.recipe.qubits,
        )[0]
        expected_words = (self.recipe.total_measurements + 63) // 64
        if len(words) != expected_words:
            raise ValueError("QDK returned an incomplete packed shot")
        packed = np.asarray(words, dtype="<i8").tobytes()[: self.recipe.shot_bytes]
        remainder = self.recipe.total_measurements % 8
        if remainder and packed[-1] >> remainder:
            raise ValueError("QDK returned nonzero measurement padding")
        return packed
