"""Convert STIM gate instructions in a DEQ gadget body to errata Locations.

Only *gate* instructions are converted.  Noise instructions
(DEPOLARIZE1/2, CORRELATED_ERROR, …) and declarations (INPUT, OUTPUT,
READOUT, CHECK) are skipped — those are handled by other modules.

Gate conversion uses ``stim.Circuit.decomposed()`` to reduce the circuit
to the ``{H, S, CX, M, R, MPAD}`` base gate set before mapping to errata
operations.  This ensures every Stim gate is automatically supported
without maintaining a per-gate dispatch table.

Levels are assigned automatically: each operation gets the earliest
level at which none of its qubits are already occupied.
"""

import warnings
from typing import Sequence

from errata.circuits.circuit import Location
from errata.circuits.clifford import (
    ControlledX,
    Hadamard,
    Measurement,
    PrepareZero,
    SqrtZ,
)
from errata.pauli import Pauli

from deq.circuit.model import GadgetStatement
from deq.transpiler.jit_transpiler import _body_to_stim_circuit, flatten_body


def instructions_to_locations(
    body: Sequence[GadgetStatement],
) -> list[Location]:
    """Convert all gate instructions in a GADGET body to errata Locations.

    The circuit is first decomposed into the ``{H, S, CX, M, R, MPAD}``
    gate set via ``stim.Circuit.decomposed()``, then each decomposed
    instruction is mapped to errata operations.

    Non-gate instructions (noise, TICK, INPUT, OUTPUT, READOUT, CHECK)
    are skipped.

    Each operation is assigned the earliest level at which none of its
    qubits are already occupied.

    Returns the ordered list of ``Location`` objects representing the
    circuit, suitable for passing to ``FaultEvaluator``.
    """
    circuit = _body_to_stim_circuit(flatten_body(list(body)))
    decomposed = circuit.decomposed()

    locations: list[Location] = []
    qubit_next_level: dict[int, int] = {}

    for inst in decomposed:
        name = inst.name
        targets = inst.targets_copy()

        for loc in _decomposed_inst_to_locations(name, targets, 0):
            qubits_used = list(loc.operation.support)
            lvl = max((qubit_next_level.get(q, 0) for q in qubits_used), default=0)
            for q in qubits_used:
                qubit_next_level[q] = lvl + 1
            locations.append(Location(lvl, loc.operation))

    return locations


def _decomposed_inst_to_locations(
    name: str,
    targets: list,
    level: int,
) -> list[Location]:
    """Map a single decomposed instruction to errata Locations."""
    if name == "H":
        return [Location(level, Hadamard(t.value)) for t in targets]
    if name == "S":
        return [Location(level, SqrtZ(t.value)) for t in targets]
    if name == "CX":
        locs = []
        for i in range(0, len(targets), 2):
            ctrl = targets[i]
            tgt = targets[i + 1]
            if ctrl.is_measurement_record_target:
                # Classically-controlled X (feedback) — not a unitary
                # gate; skip for logical-action purposes.
                warnings.warn(
                    f"skipping classically-controlled CX rec[{ctrl.value}] "
                    f"{tgt.value} in logical-action computation; feedback "
                    f"gates are not modelled by errata",
                    stacklevel=3,
                )
                continue
            locs.append(Location(level, ControlledX(ctrl.value, tgt.value)))
        return locs
    if name == "M":
        return [Location(level, Measurement(Pauli({t.value: "Z"}))) for t in targets]
    if name == "R":
        return [Location(level, PrepareZero(t.value)) for t in targets]
    if name == "MPAD":
        return []
    raise ValueError(
        f"Unexpected instruction '{name}' in decomposed circuit. "
        f"stim.Circuit.decomposed() should only produce {{H, S, CX, M, R, MPAD}}."
    )
