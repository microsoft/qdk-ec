# pylint: disable=no-member
"""Tests for logical action computation on beryllium code gadgets.

Uses the beryllium code .deq files in documents/examples/beryllium_code/
to verify that each gadget implements the expected logical operation.
"""

import importlib
from pathlib import Path

import pytest

errata = pytest.importorskip("errata", reason="errata package not installed")

from deq.circuit.parser import parse_file
from deq.circuit.model import CodeDefinition, GadgetDefinition
from deq.errata.logical_action import gadget_action

BERYLLIUM_DIR = (
    Path(__file__).resolve().parents[2] / "documents" / "examples" / "beryllium_code"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_gadget(name: str) -> tuple[GadgetDefinition, CodeDefinition]:
    """Load a gadget and its code from the beryllium_code directory."""
    code_deq = parse_file(str(BERYLLIUM_DIR / "code.deq"))
    code_def = next(d for d in code_deq.definitions if isinstance(d, CodeDefinition))

    for f in sorted(BERYLLIUM_DIR.glob("*.deq")):
        if f.name == "code.deq":
            continue
        parsed = parse_file(str(f))
        for d in parsed.definitions:
            if isinstance(d, GadgetDefinition) and d.name == name:
                return d, code_def

    raise ValueError(f"Gadget '{name}' not found in {BERYLLIUM_DIR}")


def _compute_action_mapping(gadget_name: str) -> dict[str, str]:
    """Compute the logical action of a named beryllium gadget."""
    gdef, code_def = _load_gadget(gadget_name)
    act = gadget_action(gdef, {code_def.name: code_def})
    return {str(k): str(v) for k, v in act.mapping.items()}


def _assert_identity(mapping: dict[str, str], k: int) -> None:
    """Assert that the mapping is identity on k logical qubits."""
    for qi in range(k):
        assert _normalize_pauli(mapping[f"+{{{qi}:X}}"]) == f"+{{{qi}:X}}"
        assert _normalize_pauli(mapping[f"+{{{qi}:Z}}"]) == f"+{{{qi}:Z}}"


def _normalize_pauli(s: str) -> str:
    """Normalize a Pauli string like '+{10:X, 4:X}' → '+{4:X, 10:X}'.

    Sorts the terms inside the braces by qubit index for deterministic comparison.
    """
    if not s or s[0] not in "+-":
        return s
    sign = s[0]
    inner = s[2:-1]  # strip sign + { ... }
    if not inner:
        return s
    terms = [t.strip() for t in inner.split(",")]
    terms.sort(key=lambda t: int(t.split(":")[0]))
    return f"{sign}{{{', '.join(terms)}}}"


def _assert_pauli_map(
    mapping: dict[str, str],
    expected: dict[str, str],
) -> None:
    """Assert specific Pauli mappings with normalized ordering."""
    for pauli_in, pauli_out in expected.items():
        actual = mapping[pauli_in]
        assert _normalize_pauli(actual) == _normalize_pauli(
            pauli_out
        ), f"Expected {pauli_in} → {pauli_out}, got {actual}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSyndrome:
    """Syndrome gadget should be identity on all 6 logical qubits."""

    def test_identity(self) -> None:
        mapping = _compute_action_mapping("Syndrome")
        _assert_identity(mapping, k=6)


class TestCxAll:
    """Transversal CNOT: X propagates control→target, Z propagates target→control."""

    def test_cnot_action(self) -> None:
        mapping = _compute_action_mapping("CxAll")
        for qi in range(6):
            ctrl = qi
            tgt = qi + 6
            _assert_pauli_map(
                mapping,
                {
                    f"+{{{ctrl}:X}}": f"+{{{ctrl}:X, {tgt}:X}}",
                    f"+{{{ctrl}:Z}}": f"+{{{ctrl}:Z}}",
                    f"+{{{tgt}:X}}": f"+{{{tgt}:X}}",
                    f"+{{{tgt}:Z}}": f"+{{{ctrl}:Z, {tgt}:Z}}",
                },
            )


class TestFoldH9:
    """Fold-H: Hadamard on all 6 logical qubits (with qubit relabeling)."""

    def test_x_to_z(self) -> None:
        mapping = _compute_action_mapping("FoldH9")
        # X → Z for each logical qubit (possibly with permutation)
        # From the CLI output: X0→Z0, Z0→X0, X1→Z4, Z1→X4, etc.
        expected = {
            "+{0:X}": "+{0:Z}",
            "+{0:Z}": "+{0:X}",
            "+{2:X}": "+{2:Z}",
            "+{2:Z}": "+{2:X}",
            "+{3:X}": "+{3:Z}",
            "+{3:Z}": "+{3:X}",
            "+{5:X}": "+{5:Z}",
            "+{5:Z}": "+{5:X}",
            # Qubits 1 and 4 swap: H on logical qubit 1 maps to logical qubit 4
            "+{1:X}": "+{4:Z}",
            "+{1:Z}": "+{4:X}",
            "+{4:X}": "+{1:Z}",
            "+{4:Z}": "+{1:X}",
        }
        _assert_pauli_map(mapping, expected)


class TestFoldS9:
    """Fold-S: S gate on logical qubits (Z unchanged, X→Y or X→-Y)."""

    def test_z_unchanged(self) -> None:
        mapping = _compute_action_mapping("FoldS9")
        for qi in range(6):
            assert _normalize_pauli(mapping[f"+{{{qi}:Z}}"]) == f"+{{{qi}:Z}}"

    def test_x_to_y_variants(self) -> None:
        mapping = _compute_action_mapping("FoldS9")
        _assert_pauli_map(
            mapping,
            {
                "+{0:X}": "-{0:Y}",
                "+{2:X}": "+{2:Y}",
                "+{3:X}": "-{3:Y}",
                "+{5:X}": "+{5:Y}",
            },
        )

    def test_x1_x4_cross(self) -> None:
        mapping = _compute_action_mapping("FoldS9")
        _assert_pauli_map(
            mapping,
            {
                "+{1:X}": "-{1:X, 4:Z}",
                "+{4:X}": "+{1:Z, 4:X}",
            },
        )


class TestPermute0:
    """Permute0: permutation of logical qubits (0↔5, 1↔4, rest unchanged)."""

    def test_permutation(self) -> None:
        mapping = _compute_action_mapping("Permute0")
        expected = {
            "+{0:X}": "+{5:X}",
            "+{0:Z}": "+{5:Z}",
            "+{5:X}": "+{0:X}",
            "+{5:Z}": "+{0:Z}",
            "+{1:X}": "+{4:X}",
            "+{1:Z}": "+{4:Z}",
            "+{4:X}": "+{1:X}",
            "+{4:Z}": "+{1:Z}",
            "+{2:X}": "+{2:X}",
            "+{2:Z}": "+{2:Z}",
            "+{3:X}": "+{3:X}",
            "+{3:Z}": "+{3:Z}",
        }
        _assert_pauli_map(mapping, expected)
