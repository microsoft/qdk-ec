"""Build rotated surface-code bundles, including distance-three lattice surgery.

Run this module to write the distance-three example and check distances 2, 3, 5.
Reference: Fowler et al., Phys. Rev. A 86, 032324 (2012).
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

import yaml

Stabilizer = tuple[str, list[int]]
_Artifact = dict[str, Any]


def rotated_surface_code(rows: int, cols: int | None = None) -> tuple[list[Stabilizer], list[int], list[int]]:
    """Return raster-ordered stabilizers, left-column logical X, and top-row logical Z."""
    columns = rows if cols is None else cols
    if rows < 2 or columns < 2:
        raise ValueError(f"the rotated surface code needs rows, cols >= 2; got {rows}x{columns}")
    stabilizers = []
    for row in range(-1, rows):
        for column in range(-1, columns):
            corners = [(row, column), (row, column + 1), (row + 1, column), (row + 1, column + 1)]
            valid = [
                (vertical, horizontal)
                for vertical, horizontal in corners
                if 0 <= vertical < rows and 0 <= horizontal < columns
            ]
            basis = "X" if (row + column) % 2 == 0 else "Z"
            top_or_bottom = row == -1 or row + 1 == rows
            boundary_check = len(valid) == 2 and (
                (top_or_bottom and basis == "X") or (not top_or_bottom and basis == "Z")
            )
            if len(valid) == 4 or boundary_check:
                stabilizers.append((basis, [vertical * columns + horizontal for vertical, horizontal in valid]))
    return stabilizers, [row * columns for row in range(rows)], list(range(columns))


def _extract(basis: str, qubits: list[int], ancilla: int) -> list[str]:
    if basis == "X":
        return (
            [f"R {ancilla}", f"H {ancilla}"]
            + [f"CX {ancilla} {qubit}" for qubit in qubits]
            + [f"H {ancilla}", f"M {ancilla}"]
        )
    return [f"R {ancilla}"] + [f"CX {qubit} {ancilla}" for qubit in qubits] + [f"M {ancilla}"]


def _pauli(basis: str, qubits: list[int]) -> str:
    return " ".join(f"{basis}_{qubit}" for qubit in qubits)


def _circuit(lines: list[str]) -> _Artifact:
    return {"format": "stim", "source": "\n".join(lines)}


def _prepare(distance: int, basis: str, stabilizers: list[Stabilizer]) -> _Artifact:
    qubit_count = distance * distance
    source = ["R " + " ".join(map(str, range(qubit_count)))]
    if basis == "X":
        source.append("H " + " ".join(map(str, range(qubit_count))))
    checks = []
    readout = 0
    for index, (stabilizer_basis, qubits) in enumerate(stabilizers):
        if stabilizer_basis != basis:
            source.extend(_extract(stabilizer_basis, qubits, qubit_count + readout))
            checks.append([f"circuit.readouts[{readout}]", f"out[0].stabilizers[{index}]"])
            readout += 1
    checks.extend(
        [
            [f"out[0].stabilizers[{index}]"]
            for index, (stabilizer_basis, _) in enumerate(stabilizers)
            if stabilizer_basis == basis
        ]
    )
    return {"circuit": _circuit(source), "out": [{"surface": list(range(qubit_count))}], "checks": checks}


def _idle(distance: int, stabilizers: list[Stabilizer]) -> _Artifact:
    source = [
        line
        for index, (basis, qubits) in enumerate(stabilizers)
        for line in _extract(basis, qubits, distance * distance + index)
    ]
    checks = [
        [f"circuit.readouts[{index}]", f"{boundary}[0].stabilizers[{index}]"]
        for boundary in ("in", "out")
        for index in range(len(stabilizers))
    ]
    return {"circuit": _circuit(source), "checks": checks}


def _measure(
    basis: str, qubit_count: int, stabilizers: list[Stabilizer], logical: list[int], *, block: str = "surface"
) -> _Artifact:
    instruction = "M" if basis == "Z" else "MX"
    field = "z" if basis == "Z" else "x"
    checks = []
    for index, (stabilizer_basis, qubits) in enumerate(stabilizers):
        if stabilizer_basis == basis:
            selector = ",".join(map(str, qubits))
            checks.append([f"circuit.readouts[{selector}]", f"in[0].stabilizers[{index}]"])
    selector = ",".join(map(str, logical))
    return {
        "circuit": _circuit([f"{instruction} {' '.join(map(str, range(qubit_count)))}"]),
        "in": [{block: list(range(qubit_count))}],
        "checks": checks,
        "readouts": [[f"circuit.readouts[{selector}]", f"in[0].{field}[0]"]],
    }


_ZZ_READOUT_SUBSET = [1, 3, 4, 6, 7, 9, 10, 12]


def _patch_stabilizer(basis: str, qubits: list[int], stabilizers: list[Stabilizer], offset: int) -> int | None:
    local = [qubit - offset for qubit in qubits]
    if not all(0 <= qubit < 9 for qubit in local):
        return None
    return next((index for index, candidate in enumerate(stabilizers) if candidate == (basis, local)), None)


def _merge_zz() -> _Artifact:
    merged_stabilizers, _, _ = rotated_surface_code(7, 3)
    patch_stabilizers, _, _ = rotated_surface_code(3)
    source = ["R 9 10 11", "H 9 10 11"]
    checks = []
    for index, (basis, qubits) in enumerate(merged_stabilizers):
        source.extend(_extract(basis, qubits, 21 + index))
        for entry, offset in enumerate((0, 12)):
            local_index = _patch_stabilizer(basis, qubits, patch_stabilizers, offset)
            if local_index is not None:
                checks.append([f"circuit.readouts[{index}]", f"in[{entry}].stabilizers[{local_index}]"])
        checks.append([f"circuit.readouts[{index}]", f"out[0].stabilizers[{index}]"])
    readout = [f"circuit.readouts[{index}]" for index in _ZZ_READOUT_SUBSET] + ["in[0].z[0]", "in[1].z[0]"]
    return {
        "circuit": _circuit(source),
        "in": [{"surface": list(range(9))}, {"surface": list(range(12, 21))}],
        "out": [{"merged": list(range(21))}],
        "checks": checks,
        "readouts": [readout],
    }


def _code(
    name: str, stabilizers: list[Stabilizer], logical_x: list[int], logical_z: list[int], description: str
) -> _Artifact:
    return {
        "name": name,
        "description": description,
        "stabilizers": [_pauli(basis, qubits) for basis, qubits in stabilizers],
        "x": [_pauli("X", logical_x)],
        "z": [_pauli("Z", logical_z)],
    }


def _instruction_set(distance: int) -> _Artifact:
    return {
        "name": "surface",
        "description": f"Logical instruction set for the distance-{distance} rotated surface code.",
        "blocks": {"surface": 1},
        "instructions": [
            {
                "mnemonic": "prepare_z",
                "description": "Prepare the logical |0> state.",
                "out": ["surface"],
                "action": [{"stabilize": "Z_0"}],
            },
            {
                "mnemonic": "prepare_x",
                "description": "Prepare the logical |+> state.",
                "out": ["surface"],
                "action": [{"stabilize": "X_0"}],
            },
            {
                "mnemonic": "idle",
                "description": "One full syndrome-extraction round (identity logical action).",
                "in": ["surface"],
                "out": ["surface"],
            },
            {
                "mnemonic": "measure_z",
                "description": "Destructive measurement of the logical Z observable.",
                "in": ["surface"],
                "action": [{"observe": "Z_0"}],
            },
            {
                "mnemonic": "measure_x",
                "description": "Destructive measurement of the logical X observable.",
                "in": ["surface"],
                "action": [{"observe": "X_0"}],
            },
        ],
    }


def _add_surgery(instruction_set: _Artifact, codes: dict[str, _Artifact], gadgets: dict[str, _Artifact]) -> None:
    stabilizers, logical_x, logical_z = rotated_surface_code(7, 3)
    instruction_set["blocks"]["merged"] = 1
    instruction_set["instructions"].extend(
        [
            {
                "mnemonic": "merge_zz",
                "description": (
                    "Lattice-surgery ZZ-merge: consume two surface patches, produce one merged "
                    "patch, measuring the joint logical Z_A (x) Z_B. The observe records the "
                    "joint parity; the clifford transfers the surviving logical frame onto the "
                    "merged block (merged X = X_0 X_1, merged Z = Z_0 == Z_1)."
                ),
                "in": ["surface", "surface"],
                "out": ["merged"],
                "action": [{"observe": "Z_0 Z_1"}, {"clifford": {"X_0": "X_0 X_1", "Z_1": "Z_0 Z_1"}}],
            },
            {
                "mnemonic": "measure_merged_z",
                "description": "Destructive measurement of the merged logical Z observable.",
                "in": ["merged"],
                "action": [{"observe": "Z_0"}],
            },
            {
                "mnemonic": "measure_merged_x",
                "description": "Destructive measurement of the merged logical X observable.",
                "in": ["merged"],
                "action": [{"observe": "X_0"}],
            },
        ]
    )
    codes["merged"] = _code(
        "merged",
        stabilizers,
        logical_x,
        logical_z,
        "The 7x3 rotated surface code ([[21,1,3]]) formed by ZZ-merging two "
        "patches (A=qubits 0-8, B=qubits 12-20) across a fresh seam row "
        "(qubits 9-11). Logical X = left column, Z = top row (= Z_A).",
    )
    gadgets["merge_zz"] = _merge_zz()
    gadgets["measure_merged_z"] = _measure("Z", 21, stabilizers, logical_z, block="merged")
    gadgets["measure_merged_x"] = _measure("X", 21, stabilizers, logical_x, block="merged")


def _manifest(distance: int, codes: dict[str, _Artifact], gadgets: dict[str, _Artifact]) -> _Artifact:
    surgery = (
        (
            " The d=3 member also demonstrates lattice surgery: merge_zz fuses two "
            "patches into a 7x3 merged patch, measuring the joint logical Z_A Z_B."
        )
        if distance == 3
        else ""
    )
    return {
        "name": f"surface_d{distance}",
        "description": (
            f"Distance-{distance} rotated surface code ([[{distance * distance},1,{distance}]]) \u2014 the practical workhorse of "
            "QEC, as a memory experiment (prepare, syndrome rounds with cross-round "
            "detectors, destructive readout). The d=3 member of the surface-code family "
            f"built by surface.py.{surgery}"
        ),
        "layers": [
            {
                "instruction_set": "surface.isa.yaml",
                "codes": {name: f"{name}.code.yaml" for name in codes},
                "gadgets": {name: f"{name}.gadget.yaml" for name in gadgets},
            },
            {"instruction_set": "../stim.isa.yaml"},
        ],
    }


def build_surface_code(d: int) -> str:
    """Build a distance-d bundle; distance three also includes lattice surgery."""
    distance = d
    stabilizers, logical_x, logical_z = rotated_surface_code(distance)
    description = (
        f"Distance-{distance} rotated surface code on a {distance}x{distance} grid ({len(stabilizers)} stabilizers: "
        "weight-4 bulk plaquettes in an X/Z checkerboard plus weight-2 boundary "
        "checks). Logical X = left column, logical Z = top row."
    )
    codes = {"surface": _code("surface", stabilizers, logical_x, logical_z, description)}
    gadgets = {
        "prepare_z": _prepare(distance, "Z", stabilizers),
        "prepare_x": _prepare(distance, "X", stabilizers),
        "idle": _idle(distance, stabilizers),
        "measure_z": _measure("Z", distance * distance, stabilizers, logical_z),
        "measure_x": _measure("X", distance * distance, stabilizers, logical_x),
    }
    instruction_set = _instruction_set(distance)
    if distance == 3:
        _add_surgery(instruction_set, codes, gadgets)
    documents = [{"qodec.yaml": _manifest(distance, codes, gadgets)}, {"surface.isa.yaml": instruction_set}]
    documents.extend({f"{name}.code.yaml": code} for name, code in codes.items())
    documents.extend({f"{name}.gadget.yaml": gadget} for name, gadget in gadgets.items())
    return yaml.safe_dump_all(documents, sort_keys=False, allow_unicode=True)


def main() -> None:
    import qodec

    example_directory = Path(__file__).resolve().parent
    (example_directory / "surface.qodec.yaml").write_text(build_surface_code(3), encoding="utf-8")
    physical_instructions = (example_directory.parent / "stim.isa.yaml").read_text(encoding="utf-8")
    for distance in (2, 3, 5):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "stim.isa.yaml").write_text(physical_instructions, encoding="utf-8")
            (root / "surface").mkdir()
            manifest = root / "surface/surface.qodec.yaml"
            manifest.write_text(build_surface_code(distance), encoding="utf-8")
            protocol = qodec.Qodec.load(manifest)
        assert len(protocol.codes["surface"].stabilizers) == distance * distance - 1
        assert len(protocol.codes["surface"].x) == len(protocol.codes["surface"].z) == 1
        if distance == 3:
            assert len(protocol.codes["merged"].stabilizers) == 20
        print(f"[[{distance * distance},1,{distance}]] rotated surface code: built, loaded, validated")


if __name__ == "__main__":
    main()
