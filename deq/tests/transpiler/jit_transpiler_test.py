"""Tests for the JIT transpiler outcome-code and check-plugin pipeline."""

from pathlib import Path

import pytest

from deq.circuit.model import CodeDefinition, GadgetDefinition
from deq.circuit.parser import parse_file
from deq.transpiler.jit_transpiler import (
    checks_equivalent,
    derive_checks_auto,
)
from deq.transpiler.check_plugins import resolve_gadget_checks

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_DEQ = (
    REPO_ROOT
    / "tests"
    / "circuit"
    / "repetition_code"
    / "repetition_code_d3_expected.deq"
)


def _load_definitions() -> (
    tuple[dict[str, CodeDefinition], dict[str, GadgetDefinition]]
):
    # parse_file resolves IMPORT statements, so parsing the expected
    # file pulls in the original repetition_code_d3.deq gadgets too.
    qfile = parse_file(str(EXPECTED_DEQ))
    codes: dict[str, CodeDefinition] = {}
    gadgets: dict[str, GadgetDefinition] = {}
    for definition in qfile.definitions:
        if isinstance(definition, CodeDefinition):
            codes[definition.name] = definition
        elif isinstance(definition, GadgetDefinition):
            gadgets[definition.name] = definition
    return codes, gadgets


GADGET_PAIRS = [
    ("PrepareZ", "ExpectedPrepareZ"),
    ("MeasureZ", "ExpectedMeasureZ"),
    ("Syndrome", "ExpectedSyndrome"),
]


@pytest.mark.parametrize("original_name, expected_name", GADGET_PAIRS)
def test_check_structure_matches_expected(
    original_name: str, expected_name: str
) -> None:
    codes, gadgets = _load_definitions()
    original = gadgets[original_name]
    expected = gadgets[expected_name]

    derived_checks, derived_total = derive_checks_auto(original, codes)
    expected_checks, expected_total = derive_checks_auto(expected, codes)

    assert derived_total == expected_total, (
        f"{original_name}: measurement count {derived_total} does not match "
        f"{expected_name}'s {expected_total}"
    )
    assert checks_equivalent(derived_checks, expected_checks, derived_total), (
        f"{original_name}: derived checks {derived_checks} are not equivalent "
        f"to {expected_name}'s checks {expected_checks}"
    )


# ── Small standalone sanity tests on hand-crafted gadgets ────────────

TRIVIAL_DEQ = REPO_ROOT / "tests" / "circuit" / "fixtures" / "trivial_gadgets.deq"


def _load_trivial_gadgets() -> dict[str, GadgetDefinition]:
    qfile = parse_file(str(TRIVIAL_DEQ))
    return {d.name: d for d in qfile.definitions if isinstance(d, GadgetDefinition)}


def test_trivial_check() -> None:
    gadgets = _load_trivial_gadgets()
    checks, total = derive_checks_auto(gadgets["Trivial"], {})
    assert total == 1
    assert checks == [(frozenset({0}), False)]
    expected_checks, expected_total = derive_checks_auto(gadgets["ExpectedTrivial"], {})
    assert expected_total == total
    assert checks_equivalent(checks, expected_checks, total)


def test_naturally_one_check() -> None:
    gadgets = _load_trivial_gadgets()
    checks, total = derive_checks_auto(gadgets["NaturallyOne"], {})
    assert total == 1
    assert checks == [(frozenset({0}), True)]
    expected_checks, expected_total = derive_checks_auto(
        gadgets["ExpectedNaturallyOne"], {}
    )
    assert expected_total == total
    assert checks_equivalent(checks, expected_checks, total)


def test_bell_parity_check() -> None:
    gadgets = _load_trivial_gadgets()
    checks, total = derive_checks_auto(gadgets["Bell"], {})
    assert total == 2
    assert checks_equivalent(checks, [(frozenset({0, 1}), False)], total)
    expected_checks, expected_total = derive_checks_auto(gadgets["ExpectedBell"], {})
    assert expected_total == total
    assert checks_equivalent(checks, expected_checks, total)


def test_rep3_parity_checks() -> None:
    gadgets = _load_trivial_gadgets()
    checks, total = derive_checks_auto(gadgets["Rep3"], {})
    assert total == 2
    assert checks_equivalent(
        checks,
        [(frozenset({0}), False), (frozenset({1}), False)],
        total,
    )
    expected_checks, expected_total = derive_checks_auto(gadgets["ExpectedRep3"], {})
    assert expected_total == total
    assert checks_equivalent(checks, expected_checks, total)


# ── Inverted measurement tests ───────────────────────────────────────


def test_inverted_measurement_naturally_flipped() -> None:
    """M !0 on a qubit in |0⟩ should produce a naturally-flipped check."""
    gadgets = _parse_inline("""
        GADGET G {
            R 0
            M !0
        }
        """)
    checks, total = derive_checks_auto(gadgets["G"], {})
    assert total == 1
    assert checks == [(frozenset({0}), True)]


def test_normal_measurement_not_flipped() -> None:
    """M 0 on a qubit in |0⟩ should NOT be naturally-flipped."""
    gadgets = _parse_inline("""
        GADGET G {
            R 0
            M 0
        }
        """)
    checks, total = derive_checks_auto(gadgets["G"], {})
    assert total == 1
    assert checks == [(frozenset({0}), False)]


def test_inverted_measurement_in_gadget_with_output() -> None:
    """M !0 in a gadget with OUTPUT should have a naturally-flipped finished check."""
    from deq.circuit.parser import parse

    qfile = parse("""
        CODE Trivial [[1,1,1]] {
            LOGICAL X0 Z0
        }

        GADGET Prepare {
            R 0
            M !0
            OUTPUT Trivial 0
        }
        """)
    codes = {d.name: d for d in qfile.definitions if isinstance(d, CodeDefinition)}
    gadgets = {d.name: d for d in qfile.definitions if isinstance(d, GadgetDefinition)}
    result = resolve_gadget_checks(gadgets["Prepare"], codes)
    # The finished check from M !0 should be naturally flipped
    assert any(parity for _, parity in result.finished), (
        f"Expected at least one naturally-flipped finished check, "
        f"got: {result.finished}"
    )


# ── Mode-switch tests ────────────────────────────────────────────────


def _parse_inline(source: str) -> dict[str, GadgetDefinition]:
    from deq.circuit.parser import parse

    qfile = parse(source)
    return {d.name: d for d in qfile.definitions if isinstance(d, GadgetDefinition)}


def test_checks_mode_auto_rejects_invalid_user_check_flip() -> None:
    gadgets = _parse_inline("""
        GADGET G {
            R 0
            M 0
            CHECK rec[-1] FLIP
        }
        """)
    with pytest.raises(ValueError, match="not a valid parity check"):
        resolve_gadget_checks(gadgets["G"], {})


def test_checks_mode_manual_skips_simulation() -> None:
    gadgets = _parse_inline("""
        @CHECKS("manual")
        GADGET G {
            R 0 1
            M 0
            M 1
            CHECK rec[-2]
        }
        """)
    result = resolve_gadget_checks(gadgets["G"], {})
    # manual mode: only the user-declared CHECK, no auto-derived checks.
    assert result.finished == [(frozenset({0}), False)]
    assert result.unfinished == []


def test_checks_mode_auto_keeps_user_checks_first() -> None:
    gadgets = _parse_inline("""
        GADGET G {
            R 0 1
            M 0
            M 1
            CHECK rec[-2]
        }
        """)
    result = resolve_gadget_checks(gadgets["G"], {})
    all_checks = result.finished + result.unfinished
    assert (frozenset({0}), False) in all_checks
    assert checks_equivalent(
        all_checks,
        [(frozenset({0}), False), (frozenset({1}), False)],
        2,
    )


def test_checks_mode_auto_rejects_invalid_user_check() -> None:
    gadgets = _parse_inline("""
        GADGET G {
            RX 0
            R 1
            M 0
            M 1
            CHECK rec[-2] rec[-1]
        }
        """)
    with pytest.raises(ValueError, match="not a valid parity check"):
        resolve_gadget_checks(gadgets["G"], {})


def test_checks_mode_default_is_auto() -> None:
    gadgets = _parse_inline("""
        GADGET G {
            R 0
            M 0
        }
        """)
    result = resolve_gadget_checks(gadgets["G"], {})
    all_checks = result.finished + result.unfinished
    assert len(all_checks) == 1
    assert all_checks == [(frozenset({0}), False)]


# ── Regroup (finished / unfinished) tests ────────────────────────────

from deq.transpiler.jit_transpiler import regroup_checks


def test_regroup_prepare_z_all_unfinished() -> None:
    codes, gadgets = _load_definitions()
    result = resolve_gadget_checks(gadgets["PrepareZ"], codes)
    finished, unfinished = result.finished, result.unfinished
    total = len(finished) + len(unfinished)
    # PrepareZ has no inputs, no internal measurements, and one output
    # port with 2 stabilizers → 2 output-virtual measurements total.
    # Both checks must be unfinished — one per output stabilizer.
    assert len(unfinished) == 2
    assert finished == []


def test_regroup_syndrome_mixed_finished_and_unfinished() -> None:
    codes, gadgets = _load_definitions()
    result = resolve_gadget_checks(gadgets["Syndrome"], codes)
    finished, unfinished = result.finished, result.unfinished
    num_ov = 2
    # Exactly one unfinished check per output stabilizer.
    assert len(unfinished) == num_ov
    # Finished checks never contain output-virtual indices.
    # (output-virtual indices are the highest measurement indices)
    from deq.transpiler.jit_transpiler import compute_layout

    layout = compute_layout(gadgets["Syndrome"], codes)
    ov_start = layout.ov_start
    for members, _parity in finished:
        assert all(m < ov_start for m in members)


def test_regroup_rejects_invalid_gadget() -> None:
    source = """
    CODE RepetitionCode [[3,1,1]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }

    GADGET Broken {
        R 0 1 2
        H 0
        OUTPUT RepetitionCode 0 1 2
    }
    """
    from deq.circuit.parser import parse

    qfile = parse(source)
    codes = {d.name: d for d in qfile.definitions if isinstance(d, CodeDefinition)}
    gadgets = {d.name: d for d in qfile.definitions if isinstance(d, GadgetDefinition)}
    with pytest.raises(ValueError, match="cannot be expressed as a linear combination"):
        resolve_gadget_checks(gadgets["Broken"], codes)


def test_regroup_layout_mismatch_raises() -> None:
    codes, gadgets = _load_definitions()
    checks, total = [(frozenset({0}), False)], 99
    with pytest.raises(ValueError, match="measurement layout mismatch"):
        regroup_checks(gadgets["PrepareZ"], codes, checks, total)


def test_render_logical_labels_combine_xz_to_y_default() -> None:
    """By default both X+Z bits on the same observable render as ``LY{i}``.

    This matches the semantics of an ``ERROR`` row, where the residual
    is a single Pauli operator (``Y = XZ`` up to phase) on each
    logical qubit.
    """
    from deq.circuit.model import InputPort
    from deq.transpiler.jit_transpiler import PortColumnLayout

    qfile = parse_file(str(EXPECTED_DEQ))
    codes = {d.name: d for d in qfile.definitions if isinstance(d, CodeDefinition)}
    layout = PortColumnLayout([InputPort("RepetitionCode", (0, 1, 2))], codes)

    x_col, z_col = layout.logical_qubit_columns[0]
    assert layout.render_logical_labels({x_col, z_col}) == ["IN0.LY0"]
    assert layout.render_logical_labels({x_col}) == ["IN0.LZ0"]
    assert layout.render_logical_labels({z_col}) == ["IN0.LX0"]


def test_render_logical_labels_no_combine_emits_split_lx_lz() -> None:
    """``combine_xz_to_y=False`` emits ``LX{i}`` and ``LZ{i}`` separately.

    This matches the semantics of a ``PROPAGATE`` row, where each
    label is one frame-column XOR contribution (not a Pauli operator),
    so collapsing into ``LY{i}`` would be misleading.
    """
    from deq.circuit.model import InputPort
    from deq.transpiler.jit_transpiler import PortColumnLayout

    qfile = parse_file(str(EXPECTED_DEQ))
    codes = {d.name: d for d in qfile.definitions if isinstance(d, CodeDefinition)}
    layout = PortColumnLayout([InputPort("RepetitionCode", (0, 1, 2))], codes)

    x_col, z_col = layout.logical_qubit_columns[0]
    assert layout.render_logical_labels({x_col, z_col}, combine_xz_to_y=False) == [
        "IN0.LX0",
        "IN0.LZ0",
    ]
    assert layout.render_logical_labels({x_col}, combine_xz_to_y=False) == ["IN0.LZ0"]
    assert layout.render_logical_labels({z_col}, combine_xz_to_y=False) == ["IN0.LX0"]
