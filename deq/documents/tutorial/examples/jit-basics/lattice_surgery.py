"""Build and validate the shared lattice-surgery workload without running decoders."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re

import stim

from deqagram import (
    AssertStatement,
    CodeDefinition,
    ComposeDefinition,
    ConditionalCorrection,
    GadgetApplication,
    GadgetDefinition,
    Instruction,
    ProgramDefinition,
    QubitTarget,
)
from deq.circuit.parser import parse
from deq.cli.jit import compile_program_for_jit, export_program_stim
from deq.proto.deq_jit_pb2 import JitLibrary
from deq.transpiler.compose_builder import expand_compose_circuit
from deq.transpiler.jit_library_builder import build_jit_library
from deq.transpiler.jit_transpiler import flatten_body
from deq.transpiler.loss.model_none import NoLossModel


@dataclass(frozen=True)
class Check:
    basis: str
    row: int
    column: int
    qubits: tuple[int, ...]


def patch_checks(
    data: dict[tuple[int, int], int], rows: int, columns: int
) -> list[Check]:
    checks = []
    for row in range(rows + 1):
        for column in range(columns + 1):
            if row == 0:
                basis = "Z" if 0 < column < columns and column % 2 == 0 else None
            elif row == rows:
                basis = "Z" if 0 < column < columns and column % 2 == 1 else None
            elif column == 0:
                basis = "X" if row % 2 == 1 else None
            elif column == columns:
                basis = "X" if row % 2 == 0 else None
            else:
                basis = "Z" if (row + column) % 2 == 0 else "X"
            if basis is not None:
                qubits = tuple(
                    data[coordinate]
                    for coordinate in (
                        (row - 1, column - 1),
                        (row - 1, column),
                        (row, column - 1),
                        (row, column),
                    )
                    if coordinate in data
                )
                checks.append(Check(basis, row, column, qubits))
    if len(checks) != rows * columns - 1:
        raise ValueError("incorrect rotated-patch stabilizer count")
    return checks


def targets(qubits: Iterable[int | str]) -> str:
    return " ".join(map(str, qubits))


def reset(qubits: list[int], basis: str, probability: float) -> list[str]:
    if not qubits:
        return []
    error = "Z_ERROR" if basis == "X" else "X_ERROR"
    return [f"R{basis} {targets(qubits)}", f"{error}({probability}) {targets(qubits)}"]


def extraction_round(
    data: dict[tuple[int, int], int], checks: list[Check], probability: float
) -> list[str]:
    ancillas = list(range(max(data.values()) + 1, max(data.values()) + 1 + len(checks)))
    lines = []
    for basis in ("Z", "X"):
        lines.extend(
            reset(
                [
                    ancilla
                    for ancilla, check in zip(ancillas, checks)
                    if check.basis == basis
                ],
                basis,
                probability,
            )
        )
    lines.append(f"DEPOLARIZE1({probability}) {targets(data.values())}")
    offsets = {
        "Z": [(-1, -1), (-1, 0), (0, -1), (0, 0)],
        "X": [(-1, -1), (0, -1), (-1, 0), (0, 0)],
    }
    for layer in range(4):
        pairs: list[int] = []
        for ancilla, check in zip(ancillas, checks):
            delta_row, delta_column = offsets[check.basis][layer]
            coordinate = check.row + delta_row, check.column + delta_column
            if coordinate in data:
                qubit = data[coordinate]
                pairs.extend(
                    (qubit, ancilla) if check.basis == "Z" else (ancilla, qubit)
                )
        if len(set(pairs)) != len(pairs):
            raise ValueError("syndrome-extraction layer reuses a qubit")
        lines.extend(
            [f"CX {targets(pairs)}", f"DEPOLARIZE2({probability}) {targets(pairs)}"]
        )
        idle = sorted((set(data.values()) | set(ancillas)) - set(pairs))
        if idle:
            lines.append(f"DEPOLARIZE1({probability}) {targets(idle)}")
        lines.append("TICK")
    for basis in ("Z", "X"):
        lines.append(
            f"M{basis}({probability}) {targets(ancilla for ancilla, check in zip(ancillas, checks) if check.basis == basis)}"
        )
    lines.append(f"DEPOLARIZE1({probability}) {targets(data.values())}")
    return lines


def gadget(
    name: str,
    inputs: list[list[int]],
    body: list[str],
    outputs: list[list[int]],
    propagation: tuple[str, ...] = (),
    *,
    input_code: str = "SurfaceCode",
    output_code: str = "SurfaceCode",
    syndrome_checks: bool = True,
) -> str:
    lines = [
        '@CHECKS("syndrome", metachecks=1)' if syndrome_checks else '@CHECKS("auto")',
        f"GADGET {name} {{",
    ]
    lines.extend(f"    INPUT {input_code} {targets(port)}" for port in inputs)
    lines.extend(f"    {line}" for line in body)
    lines.extend(f"    OUTPUT {output_code} {targets(port)}" for port in outputs)
    lines.extend(f"    {line}" for line in propagation)
    lines.append("}")
    return "\n".join(lines)


def compose(name: str, body: list[str], inputs: int, outputs: int) -> str:
    return "\n".join(
        [f"COMPOSE {name} {{"]
        + [f"    INPUT SurfaceCode {port}" for port in range(inputs)]
        + [f"    {line}" for line in body]
        + [f"    OUTPUT SurfaceCode {port}" for port in range(outputs)]
        + ["}"]
    )


def surgery_library(distance: int, probability: float) -> str:
    if distance < 3 or distance % 2 != 1:
        raise ValueError("distance must be odd and at least 3")
    if not 0 <= probability < 0.5:
        raise ValueError("probability must be in [0, 0.5)")
    count = distance**2
    first = list(range(count))
    second = list(range(count, 2 * count))
    data = {
        (row, column): row * distance + column
        for row in range(distance)
        for column in range(distance)
    }
    checks = patch_checks(data, distance, distance)
    logical_x = "*".join(f"X{column}" for column in range(distance))
    logical_z = "*".join(f"Z{row * distance}" for row in range(distance))
    stabilizers = " ".join(
        "*".join(f"{check.basis}{qubit}" for qubit in check.qubits) for check in checks
    )
    parts = [
        f"CODE SurfaceCode [[{count},1,{distance}]] {{\n"
        f"    LOGICAL {logical_x} {logical_z}\n    STABILIZER {stabilizers}\n}}"
    ]
    round_body = extraction_round(data, checks, probability)
    parts.append(gadget("SyndromeRound", [first], round_body, [first]))
    parts.append(compose("Syndrome", ["SyndromeRound 0"] * distance, 1, 1))
    for basis in ("Z", "X"):
        logical = first[:distance] if basis == "Z" else first[::distance]
        for bit in (0, 1):
            body = reset(first, basis, probability)
            if bit:
                body.append(f"{'X' if basis == 'Z' else 'Z'} {targets(logical)}")
                body.append(f"DEPOLARIZE1({probability}) {targets(logical)}")
            declared_sign = (
                (f"VIRTUAL L{'X' if basis == 'Z' else 'Z'}0",) if bit else ()
            )
            name = f"Prepare{basis}{bit}"
            parts.append(
                gadget(f"{name}Begin", [], body + round_body, [first], declared_sign)
            )
            parts.append(
                compose(
                    name,
                    [f"{name}Begin 0"] + ["SyndromeRound 0"] * (distance - 1),
                    0,
                    1,
                )
            )
        readout = first[::distance] if basis == "Z" else first[:distance]
        body = [
            f"M{basis}({probability}) {targets(first)}",
            f"READOUT {' '.join(f'rec[-{count - qubit}]' for qubit in readout)}",
        ]
        parts.append(
            gadget(f"Measure{basis}End", [first], body, [], syndrome_checks=False)
        )
        parts.append(
            compose(
                f"Measure{basis}",
                ["SyndromeRound 0"] * (distance - 1) + [f"Measure{basis}End 0"],
                1,
                0,
            )
        )

    # The ancilla preparation has a data-patch input, preventing future CNOT
    # ancillas from being scheduled at the beginning of the whole experiment.
    combined_data = data | {
        (row, column + distance + 1): qubit + count
        for (row, column), qubit in data.items()
    }
    combined_checks = checks + [
        Check(
            check.basis,
            check.row,
            check.column + distance + 1,
            tuple(qubit + count for qubit in check.qubits),
        )
        for check in checks
    ]
    parts.append(
        gadget(
            "PrepareAncillaBegin",
            [first],
            reset(second, "X", probability)
            + extraction_round(combined_data, combined_checks, probability),
            [first, second],
        )
    )
    parts.append(
        compose(
            "PrepareAncilla",
            ["PrepareAncillaBegin IN(0) OUT(0 1)"]
            + ["SyndromeRound 0", "SyndromeRound 1"] * (distance - 1),
            1,
            2,
        )
    )

    for basis in ("Z", "X"):
        rows, columns = (
            (distance, 2 * distance + 1)
            if basis == "Z"
            else (2 * distance + 1, distance)
        )
        merged = {}
        seam = []
        for row in range(rows):
            for column in range(columns):
                coordinate = column if basis == "Z" else row
                if coordinate < distance:
                    qubit = row * distance + column
                elif coordinate == distance:
                    qubit = 2 * count + (row if basis == "Z" else column)
                    seam.append(qubit)
                else:
                    qubit = count + (
                        row * distance + column - distance - 1
                        if basis == "Z"
                        else (row - distance - 1) * distance + column
                    )
                merged[row, column] = qubit
        merged_checks = patch_checks(merged, rows, columns)
        code_name = f"Merged{basis}{basis}"
        merged_x = "*".join(f"X{merged[0, column]}" for column in range(columns))
        merged_z = "*".join(f"Z{merged[row, 0]}" for row in range(rows))
        merged_stabilizers = " ".join(
            "*".join(f"{check.basis}{qubit}" for qubit in check.qubits)
            for check in merged_checks
        )
        parts.append(
            f"CODE {code_name} [[{rows * columns},1,{distance}]] {{\n"
            f"    LOGICAL {merged_x} {merged_z}\n    STABILIZER {merged_stabilizers}\n}}"
        )
        measurement_order = [
            check
            for measured_basis in ("Z", "X")
            for check in merged_checks
            if check.basis == measured_basis
        ]
        parity = [
            index
            for index, check in enumerate(measurement_order)
            if check.basis == basis and set(check.qubits) & set(seam)
        ]
        seam_basis = "X" if basis == "Z" else "Z"
        body = reset(seam, seam_basis, probability)
        merged_round = extraction_round(merged, merged_checks, probability)
        body += merged_round
        body += [f"READOUT {targets(f'M{index}' for index in parity)}"]
        correcting_pauli = "X" if basis == "Z" else "Z"
        merged_port = list(range(rows * columns))
        parts.append(
            gadget(
                f"{code_name}Begin",
                [first, second],
                body,
                [merged_port],
                output_code=code_name,
            )
        )
        parts.append(
            gadget(
                f"{code_name}Round",
                [merged_port],
                merged_round,
                [merged_port],
                input_code=code_name,
                output_code=code_name,
            )
        )
        parts.append(
            gadget(
                f"{code_name}End",
                [merged_port],
                [f"M{seam_basis}({probability}) {targets(seam)}"],
                [first, second],
                input_code=code_name,
                syndrome_checks=False,
            )
        )
        parts.append(
            compose(
                f"M{basis}{basis}",
                [
                    f"{code_name}Begin IN(0 1) OUT(0)",
                    *([f"{code_name}Round 0"] * (distance - 1)),
                    f"{code_name}End IN(0) OUT(0 1)",
                    f"CONDITIONAL rec[-1] {correcting_pauli}0 1",
                ],
                2,
                2,
            )
        )

    parts.append("""PROGRAM CNOT {
    PrepareAncilla IN(0) OUT(0 2)
    Syndrome IN(1) OUT(1)
    MZZ IN(0 2) OUT(0 2)
    Syndrome IN(1) OUT(1)
    MXX IN(2 1) OUT(2 1)
    Syndrome IN(0) OUT(0)
    MeasureZ IN(2)
    Syndrome IN(0) OUT(0)
    Syndrome IN(1) OUT(1)
    CONDITIONAL rec[-2] Z0 0
    CONDITIONAL rec[-3] X0 1
    CONDITIONAL rec[-1] X0 1
}""")
    return "\n\n".join(parts) + "\n"


def brickwall_program(
    width: int, layers: int, basis: str, seed: int, wire_stride: int = 1
) -> str:
    """Write the brickwall with data qubit ``i`` on wire ``wire_stride * i``."""
    if width < 1 or layers < 1 or basis not in ("X", "Z") or wire_stride < 1:
        raise ValueError(
            "require width >= 1, layers >= 1, basis X or Z, and wire_stride >= 1"
        )
    rng = random.Random(seed)
    bits = [rng.randrange(2) for _ in range(width)]
    body = [
        f"    Prepare{basis}{bit} {wire * wire_stride}" for wire, bit in enumerate(bits)
    ]
    for layer in range(layers):
        used: set[int] = set()
        for control in range(layer % 2, width - 1, 2):
            target = control + 1
            body.append(f"    CNOT {control * wire_stride} {target * wire_stride}")
            used.update((control, target))
            if basis == "Z":
                bits[target] ^= bits[control]
            else:
                bits[control] ^= bits[target]
        for wire in range(width):
            if wire not in used:
                body.extend([f"    Syndrome {wire * wire_stride}"] * 4)
    body.extend(f"    Measure{basis} {wire * wire_stride}" for wire in range(width))
    body.extend(
        f"    ASSERT_EQ rec[-{width - wire}] {bit}" for wire, bit in enumerate(bits)
    )
    return "PROGRAM Brickwall {\n" + "\n".join(body) + "\n}\n"


def circuit_source(
    distance: int,
    probability: float,
    width: int,
    layers: int,
    basis: str,
    seed: int,
    wire_stride: int = 1,
) -> str:
    return (
        surgery_library(distance, probability)
        + "\n"
        + brickwall_program(width, layers, basis, seed, wire_stride)
    )


def compile_case(
    directory: Path,
    *,
    distance: int,
    probability: float,
    width: int,
    layers: int,
    basis: str,
    seed: int,
    export_physical: bool = True,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    if not export_physical and (directory / "circuit.stim").exists():
        raise ValueError(
            "metadata-only compilation must not leave an existing physical circuit"
        )
    source = circuit_source(distance, probability, width, layers, basis, seed)
    (directory / "circuit.deq").write_text(source)
    parsed = parse(source)
    library = build_jit_library(parsed, jobs=1, loss_model=NoLossModel())
    programs = {
        definition.name: definition
        for definition in parsed.definitions
        if isinstance(definition, ProgramDefinition)
    }
    codes = {
        definition.name: definition
        for definition in parsed.definitions
        if isinstance(definition, CodeDefinition)
    }
    program_defs: dict[str, object] = dict(programs)
    compiled, assertions = compile_program_for_jit(
        library, programs["Brickwall"], program_defs, codes=codes
    )
    by_type = {item.base.gtype: item.base for item in library.gadget_types}
    epochs: dict[int, int] = {}
    for instruction, _ in compiled:
        instance = instruction.gadget
        predecessors = [connector.gid for connector in instance.connectors]
        predecessors.extend(
            readout.gid
            for readout in instance.modifier.remote_conditional_correction.remote_readouts
        )
        epochs[instance.gid] = max(
            (epochs[gid] for gid in predecessors), default=0
        ) + bool(by_type[instance.gtype].measurements)
    compiled.sort(key=lambda item: epochs[item[0].gadget.gid])
    gid_mapping = {
        instruction.gadget.gid: index
        for index, (instruction, _) in enumerate(compiled, start=1)
    }
    epochs = {gid_mapping[gid]: epoch for gid, epoch in epochs.items()}
    for instruction, _ in compiled:
        instance = instruction.gadget
        instance.gid = gid_mapping[instance.gid]
        for connector in instance.connectors:
            connector.gid = gid_mapping[connector.gid]
        for readout in instance.modifier.remote_conditional_correction.remote_readouts:
            readout.gid = gid_mapping[readout.gid]
    library.program.extend(instruction for instruction, _ in compiled)
    (directory / "circuit.deq.jit").write_bytes(library.SerializeToString())
    circuit = None
    if export_physical:
        gadgets = {
            definition.name: definition
            for definition in parsed.definitions
            if isinstance(definition, GadgetDefinition)
        }
        compositions = {
            definition.name: definition
            for definition in parsed.definitions
            if isinstance(definition, ComposeDefinition)
        }
        known_names = set(gadgets) | set(compositions)
        for name, definition in compositions.items():
            inputs, body, outputs = expand_compose_circuit(
                definition, gadgets, compositions, known_names, codes
            )
            gadgets[name] = GadgetDefinition(name=name, body=[*inputs, *body, *outputs])
        physical = export_program_stim(
            library,
            gadgets,
            {key: value.name for key, value in by_type.items()},
            flatten_body,
            programs["Brickwall"],
            [application for _, application in compiled],
            assertions,
        )
        (directory / "circuit.stim").write_text(physical)
        circuit = stim.Circuit(physical)
    configuration = {
        "distance": distance,
        "probability": probability,
        "width": width,
        "layers": layers,
        "basis": basis,
        "seed": seed,
        "epochs": epochs,
        "gadgets": len(compiled),
        "measurements": sum(
            len(by_type[instruction.gadget.gtype].measurements)
            for instruction, _ in compiled
        ),
        "readouts": sum(
            len(by_type[instruction.gadget.gtype].readouts)
            for instruction, _ in compiled
        ),
        "gadget_names": {
            instruction.gadget.gid: by_type[instruction.gadget.gtype].name
            for instruction, _ in compiled
        },
        "assertions": assertions,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "physical_circuit_exported": export_physical,
        "input_kind": "physical_stim_sampling" if export_physical else "metadata_only",
        "cnot_count": sum(
            by_type[instruction.gadget.gtype].name == "MZZ"
            for instruction, _ in compiled
        ),
    }
    if circuit is not None:
        if circuit.num_measurements != configuration["measurements"]:
            raise ValueError("physical and logical measurement counts disagree")
        configuration["qubits"] = circuit.num_qubits
    (directory / "configuration.json").write_text(
        json.dumps(configuration, indent=2) + "\n"
    )
    return configuration


@dataclass(frozen=True, order=True)
class Cnot:
    layer: int
    control: int
    target: int


def load_circuit(case: Path) -> tuple[dict, list[int], list[Cnot], dict]:
    configuration = json.loads((case / "configuration.json").read_text())
    source = (case / "circuit.deq").read_text()
    if hashlib.sha256(source.encode()).hexdigest() != configuration["source_sha256"]:
        raise ValueError("saved logical source does not match its recorded hash")
    programs = {
        item.name: item
        for item in parse(source).definitions
        if isinstance(item, ProgramDefinition)
    }
    macro = programs["CNOT"]
    applications = [
        (item.gadget_name, item.in_indices, item.out_indices)
        for item in macro.body
        if isinstance(item, GadgetApplication)
    ]
    expected_applications = [
        ("PrepareAncilla", [0], [0, 2]),
        ("Syndrome", [1], [1]),
        ("MZZ", [0, 2], [0, 2]),
        ("Syndrome", [1], [1]),
        ("MXX", [2, 1], [2, 1]),
        ("Syndrome", [0], [0]),
        ("MeasureZ", [2], None),
        ("Syndrome", [0], [0]),
        ("Syndrome", [1], [1]),
    ]
    corrections = [
        (item.readout_offset, item.paulis, item.wire)
        for item in macro.body
        if isinstance(item, ConditionalCorrection)
    ]
    if applications != expected_applications or corrections != [
        (2, [("Z", 0)], 0),
        (3, [("X", 0)], 1),
        (1, [("X", 0)], 1),
    ]:
        raise ValueError("the recorded CNOT macro differs from the expected surgery")
    width, layers = configuration["width"], configuration["layers"]
    epochs = [0] * width
    preparations = [-1] * width
    gates = []
    for item in programs["Brickwall"].body:
        if isinstance(item, AssertStatement):
            continue
        if not isinstance(item, Instruction) or not all(
            isinstance(target, QubitTarget) for target in item.targets
        ):
            raise ValueError("unsupported logical program statement")
        wires = [
            target.index for target in item.targets if isinstance(target, QubitTarget)
        ]
        preparation = re.fullmatch(r"Prepare([XZ])([01])", item.name)
        if preparation:
            if len(wires) != 1 or epochs[wires[0]] != 0:
                raise ValueError("unexpected logical preparation")
            preparations[wires[0]] = int(preparation[2])
            epochs[wires[0]] = 1
        elif item.name == "CNOT":
            if len(wires) != 2 or epochs[wires[0]] != epochs[wires[1]]:
                raise ValueError("CNOT inputs are not aligned")
            epoch = epochs[wires[0]]
            if (epoch - 1) % 4:
                raise ValueError("CNOT does not begin on a four-block layer")
            gates.append(Cnot((epoch - 1) // 4, *wires))
            for wire in wires:
                epochs[wire] += 4
        elif item.name in ("Syndrome", "MeasureX", "MeasureZ"):
            if len(wires) != 1:
                raise ValueError("unexpected single-block instruction")
            epochs[wires[0]] += 1
        else:
            raise ValueError(f"unsupported logical operation: {item.name}")
    if epochs != [4 * layers + 2] * width or -1 in preparations:
        raise ValueError("logical blocks do not finish in the recorded final epoch")
    expected = [
        Cnot(layer, control, control + 1)
        for layer in range(layers)
        for control in range(layer % 2, width - 1, 2)
    ]
    if gates != expected:
        raise ValueError("saved program is not the expected parallel brickwall")
    verification = verify_expansion(case, configuration, preparations, gates)
    return configuration, preparations, gates, verification


def verify_expansion(
    case: Path, configuration: dict, preparations: list[int], gates: list[Cnot]
) -> dict:
    library = JitLibrary.FromString((case / "circuit.deq.jit").read_bytes())
    types = {item.base.gtype: item.base for item in library.gadget_types}
    ports: dict[int, list[tuple[str, int]]] = {}
    ancillas: dict[int, dict] = {}
    observed_corrections: Counter[tuple[int, int]] = Counter()
    preparation_index = 0
    final_data = []
    for instruction in library.program:
        gadget = instruction.gadget
        definition = types[gadget.gtype]
        inputs = [ports[peer.gid][peer.port] for peer in gadget.connectors]
        epoch = configuration["epochs"][str(gadget.gid)]
        if definition.name.startswith(("PrepareZ", "PrepareX")):
            bit = preparations[preparation_index]
            if definition.name != f"Prepare{configuration['basis']}{bit}":
                raise ValueError("expanded preparation differs from the source")
            outputs = [("data", preparation_index)]
            preparation_index += 1
        elif definition.name == "PrepareAncilla":
            if len(inputs) != 1 or inputs[0][0] != "data":
                raise ValueError("ancilla preparation has the wrong data input")
            outputs = [inputs[0], ("ancilla", gadget.gid)]
            ancillas[gadget.gid] = {"control": inputs[0][1], "begin_epoch": epoch}
        elif definition.name == "MZZ":
            if len(inputs) != 2 or inputs[0][0] != "data" or inputs[1][0] != "ancilla":
                raise ValueError("MZZ is not between a control and its ancilla")
            ancilla = ancillas[inputs[1][1]]
            if (
                inputs[0][1] != ancilla["control"]
                or epoch != ancilla["begin_epoch"] + 1
            ):
                raise ValueError("MZZ has incorrect connectivity or timing")
            ancilla["mzz"] = gadget.gid
            outputs = inputs
        elif definition.name == "MXX":
            if len(inputs) != 2 or inputs[0][0] != "ancilla" or inputs[1][0] != "data":
                raise ValueError("MXX is not between an ancilla and target")
            ancilla = ancillas[inputs[0][1]]
            if epoch != ancilla["begin_epoch"] + 2:
                raise ValueError("MXX has incorrect timing")
            ancilla.update(target=inputs[1][1], mxx=gadget.gid)
            outputs = inputs
        elif definition.name in ("MeasureX", "MeasureZ"):
            if len(inputs) != 1:
                raise ValueError("destructive readout has multiple inputs")
            if inputs[0][0] == "ancilla":
                ancilla = ancillas[inputs[0][1]]
                if epoch != ancilla["begin_epoch"] + 3:
                    raise ValueError("ancilla readout has incorrect timing")
                ancilla["measure"] = gadget.gid
            else:
                final_data.append(inputs[0][1])
            outputs = []
        elif definition.name == "Syndrome" or definition.name.startswith("__identity_"):
            if len(inputs) != 1:
                raise ValueError("single-block operation has multiple inputs")
            outputs = inputs
        else:
            raise ValueError(f"unsupported expanded gadget: {definition.name}")
        if len(outputs) != len(definition.outputs):
            raise ValueError("expanded gadget output count disagrees with wiring")
        ports[gadget.gid] = outputs
        remote = gadget.modifier.remote_conditional_correction.remote_readouts
        if remote:
            if len(remote) != 1 or len(inputs) != 1 or inputs[0][0] != "data":
                raise ValueError("unexpected remote Pauli correction")
            observed_corrections[remote[0].gid, inputs[0][1]] += 1
    expanded = sorted(
        Cnot((item["begin_epoch"] - 2) // 4, item["control"], item["target"])
        for item in ancillas.values()
    )
    expected_corrections: Counter[tuple[int, int]] = Counter()
    for item in ancillas.values():
        expected_corrections[item["mxx"], item["control"]] += 1
        expected_corrections[item["mzz"], item["target"]] += 1
        expected_corrections[item["measure"], item["target"]] += 1
    if expanded != sorted(gates) or observed_corrections != expected_corrections:
        raise ValueError(
            "expanded CNOT or classical dependencies differ from the source"
        )
    if sorted(final_data) != list(range(configuration["width"])):
        raise ValueError("expanded final readouts do not cover every data block")
    return {
        "source_sha256": configuration["source_sha256"],
        "cnots": len(gates),
        "temporary_ancillas": len(ancillas),
        "remote_pauli_dependencies": sum(observed_corrections.values()),
        "final_data_readouts": len(final_data),
        "epochs": 4 * configuration["layers"] + 2,
        "source_and_expanded_wiring_agree": True,
    }
