"""Derived accessors on codes, circuits, instructions, and encodings.

Covers the values Python computes rather than stores: qubit counts, the
inferred circuit format, observe counts, and the block types an encoding
carries through a load/save round trip.
"""

from __future__ import annotations

import pathlib
import json
import struct

import pytest

import qodec

EXAMPLES = pathlib.Path(__file__).resolve().parents[3] / "examples"


def load_c4c6() -> qodec.Qodec:
    return qodec.Qodec.load(str(EXAMPLES / "c4c6" / "qodec.yaml"))


def test_code_reports_its_counts() -> None:
    code = qodec.Code(
        name="repetition3",
        stabilizers=["Z_0 Z_1", "Z_1 Z_2"],
        x=["X_0 X_1 X_2"],
        z=["Z_0"],
    )
    assert code.logical_count == 1
    assert code.physical_qubit_count == 3


def test_physical_qubit_count_follows_the_highest_index() -> None:
    """It is inferred from the operators, so a code whose stabilizer reaches
    qubit 7 acts on eight."""
    code = qodec.Code(
        name="c832",
        stabilizers=["X_0 X_1 X_2 X_3 X_4 X_5 X_6 X_7"],
        x=["X_0 X_1 X_2 X_3"],
        z=["Z_0 Z_4"],
    )
    assert code.physical_qubit_count == 8


@pytest.mark.parametrize("field", ["stabilizers", "x", "z"])
@pytest.mark.parametrize("operator", ["-Z_0", "+Z_0", "-", "+", "X_0\n-Z_1"])
def test_signed_code_paulis_are_rejected(field: str, operator: str, tmp_path: pathlib.Path) -> None:
    operators: dict[str, list[str]] = {"stabilizers": [], "x": [], "z": []}
    operators[field] = [operator]
    with pytest.raises(ValueError, match="code Paulis must not have a sign"):
        qodec.Code("signed", stabilizers=[*operators["stabilizers"]], x=[*operators["x"]], z=[*operators["z"]])
    source = tmp_path / "signed.code.yaml"
    source.write_text(json.dumps({"name": "signed", **operators}), encoding="utf-8")
    with pytest.raises(qodec.QodecLoadError, match="code Paulis must not have a sign"):
        qodec.Code.load(source)
    code = qodec.Code("draft", stabilizers=[], x=[], z=[])
    setattr(code, field, [operator])
    with pytest.raises(qodec.QodecSaveError, match="code Paulis must not have a sign"):
        code.save(tmp_path / "rejected.yaml")
    assert not (tmp_path / "rejected.yaml").exists()
    assert qodec.actions.Observe(["-Z_0"]).observables == ("-Z_0",)


def test_unrepresentable_code_dimension_is_rejected(tmp_path: pathlib.Path) -> None:
    maximum = (1 << (8 * struct.calcsize("P"))) - 1
    code = qodec.Code("boundary", stabilizers=[f"Z_{maximum - 1}"], x=[], z=[])
    assert code.physical_qubit_count == maximum
    with pytest.raises(ValueError, match="exceeds the supported code dimension"):
        qodec.Code("boundary", stabilizers=[f"Z_{maximum}"], x=[], z=[])
    code.stabilizers = [f"Z_{maximum}"]
    assert code.physical_qubit_count == 0
    with pytest.raises(qodec.QodecSaveError, match="exceeds the supported code dimension"):
        code.save(tmp_path / "rejected.yaml")
    assert not (tmp_path / "rejected.yaml").exists()


@pytest.mark.parametrize("field", ["stabilizers", "x", "z"])
def test_physical_qubit_count_uses_declared_pauli_indices(field: str) -> None:
    code = qodec.Code("tokens", stabilizers=[], x=["X_0"], z=["Z_0"])
    for expression, expected in [("I_7", 8), ("X_0", 1), ("Z_02", 3)]:
        setattr(code, field, [expression])
        assert code.physical_qubit_count == expected
        assert getattr(code, field) == [expression]


@pytest.mark.parametrize("invalid", ["Q_7", "outer.inner.X_9", "Z_bad"])
def test_qubit_count_ignores_malformed_tokens_but_save_rejects_them(
    invalid: str, tmp_path: pathlib.Path
) -> None:
    code = qodec.Code("tokens", stabilizers=[], x=[], z=[])
    code.stabilizers = [f"X_2 {invalid}"]
    assert code.physical_qubit_count == 3
    with pytest.raises(qodec.QodecSaveError) as error:
        code.save(tmp_path / "invalid.code.yaml")
    assert invalid in str(error.value)
    assert not (tmp_path / "invalid.code.yaml").exists()


def test_circuit_effective_format_falls_back_to_the_inferred_one() -> None:
    gadget = next(iter(load_c4c6().layers[1].gadgets.values()))
    circuit = gadget.circuit
    assert circuit.effective_format == "stim"
    circuit.format = None
    assert circuit.effective_format == "stim", "an unset tag still resolves to a format"
    circuit.format = "stim"
    assert circuit.effective_format == "stim"


def test_encoding_exposes_block_types() -> None:
    gadget = next(iter(load_c4c6().layers[0].gadgets.values()))
    encoding = (gadget.inputs or gadget.outputs)[0]
    assert len(encoding.block_types) == len(encoding.support)
    declared = {block.name for block in gadget.circuit.instruction_set.blocks}
    assert set(encoding.block_types) <= declared, "every block type must be declared by the circuit's ISA"
    assert all(encoding.block_types), "an exposed block type must be named, not blank"


def test_encoding_accepts_block_types_on_construction() -> None:
    code = qodec.Code(name="c4", stabilizers=["X_0 X_1 X_2 X_3"], x=["X_0 X_1"], z=["Z_0 Z_2"])
    encoding = qodec.gadgets.Encoding(code, support=["0", "1"], block_types=["qubit", "qubit"])
    assert encoding.block_types == ["qubit", "qubit"]


def test_manifest_filename_defaults_for_a_constructed_qodec() -> None:
    assert qodec.Qodec(layers=[]).manifest_filename == "qodec.yaml"
