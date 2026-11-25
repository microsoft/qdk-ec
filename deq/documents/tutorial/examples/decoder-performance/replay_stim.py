"""Stream physical CSS gadgets into Stim frames without an expanded circuit file."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import stim

from replay_qsharp import Recipe

_GATES = {
    0: "R",
    1: "RX",
    2: "X",
    3: "Y",
    4: "Z",
    6: "CX",
    8: "M",
    9: "MX",
    10: "X_ERROR",
    11: "Z_ERROR",
    12: "DEPOLARIZE1",
    13: "DEPOLARIZE2",
    14: "M",
    15: "MX",
}


def physical_chunks(
    recipe: Recipe, probability: float
) -> Iterator[tuple[stim.Circuit, stim.Circuit]]:
    """Yield physical and CSS-reference circuits in compiled measurement order.

    The reference transformation mirrors the runtime's stim_reference.rs:
    retain CSS operations, remove physical noise, and represent Pauli gates
    by probability-one frame errors. Reject non-CSS gates instead of
    allocating a global tableau as a fallback.
    """
    if not 0 <= probability <= 1:
        raise ValueError("Noise probability must lie between zero and one")
    for body in recipe.bodies:
        if any(instruction[0] not in _GATES for instruction in body):
            raise ValueError("Streaming reference generation requires a CSS circuit")
    bodies = []
    for body in recipe.bodies:
        instructions = []
        for opcode, *targets in body:
            gate = _GATES[opcode]
            physical = (
                f"{gate}({probability})" if opcode in (8, 9, 10, 11, 12, 13) else gate
            )
            reference = (
                f"{gate}_ERROR(1)"
                if opcode in (2, 3, 4)
                else None if opcode in (10, 11, 12, 13) else gate
            )
            instructions.append((physical, reference, targets))
        bodies.append(instructions)
    live: dict[tuple[int, int], list[int]] = {}
    free: list[int] = []
    allocated = 0
    measured = 0
    for gid, (kind, first, first_port, second, second_port) in enumerate(
        recipe.program, 1
    ):
        if recipe.virtual[kind]:
            live[gid, 0] = live.pop((first, first_port))
            continue
        mapping = [-1] * (recipe.used[kind][-1] + 1)
        for port, parent in enumerate(((first, first_port), (second, second_port))):
            if port >= len(recipe.inputs[kind]):
                break
            physical = live.pop(parent)
            local = recipe.inputs[kind][port]
            if len(physical) != len(local):
                raise ValueError("Physical input port size mismatch")
            for index, qubit in zip(local, physical):
                mapping[index] = qubit
        for local in recipe.used[kind]:
            if mapping[local] < 0:
                if free:
                    mapping[local] = free.pop()
                else:
                    if allocated >= recipe.qubits:
                        raise ValueError("Physical allocation exceeded its bound")
                    mapping[local] = allocated
                    allocated += 1

        labels = [str(qubit) for qubit in mapping]
        physical_lines = []
        reference_lines = []
        for gate, reference_gate, targets in bodies[kind]:
            qubits = " ".join(labels[target] for target in targets)
            physical_lines.append(f"{gate} {qubits}")
            if reference_gate is not None:
                reference_lines.append(f"{reference_gate} {qubits}")
        retired = [mapping[local] for local in recipe.retired[kind]]
        if retired:
            reset = "R " + " ".join(map(str, retired))
            physical_lines.append(reset)
            reference_lines.append(reset)
        physical = stim.Circuit("\n".join(physical_lines))
        reference = stim.Circuit("\n".join(reference_lines))
        free.extend(retired)
        for port, local in enumerate(recipe.outputs[kind]):
            live[gid, port] = [mapping[index] for index in local]
        count = physical.num_measurements
        if count != recipe.measurements[kind] or reference.num_measurements != count:
            raise ValueError("Physical measurement count differs from decoder metadata")
        measured += count
        yield physical, reference
    if live or measured != recipe.total_measurements:
        raise ValueError("Physical stream is incomplete")


def sample_batch(
    recipe: Recipe,
    probability: float,
    *,
    shots: int,
    seed: int,
    reference_seed: int,
    progress: Callable[[int, int], None] | None = None,
) -> Iterator[bytes]:
    """Yield headerless LSB-first shots; one batch seed identifies all columns."""
    if shots < 1:
        raise ValueError("A sampling batch needs at least one shot")
    simulator = stim.FlipSimulator(
        batch_size=shots, num_qubits=recipe.qubits, seed=seed
    )
    reference = stim.FlipSimulator(
        batch_size=1, num_qubits=recipe.qubits, seed=reference_seed
    )
    for index, (physical, noiseless) in enumerate(
        physical_chunks(recipe, probability), 1
    ):
        simulator.do(physical)
        reference.do(noiseless)
        if progress is not None:
            progress(index, simulator.num_measurements)
    if (
        simulator.num_measurements != recipe.total_measurements
        or reference.num_measurements != recipe.total_measurements
    ):
        raise ValueError("Frame sampler did not consume every measurement")
    baseline = reference.get_measurement_flips(instance_index=0, bit_packed=True)
    del reference
    for index in range(shots):
        packed = np.bitwise_xor(
            simulator.get_measurement_flips(instance_index=index, bit_packed=True),
            baseline,
        ).tobytes()
        remainder = recipe.total_measurements % 8
        if len(packed) != recipe.shot_bytes or (remainder and packed[-1] >> remainder):
            raise ValueError("Frame sampler produced invalid packed measurements")
        yield packed
