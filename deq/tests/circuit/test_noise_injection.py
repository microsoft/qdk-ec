"""Tests for deq.noise — SI1000 noise injection."""

from pathlib import Path
from textwrap import dedent

import pytest

from deq.circuit.parser import parse
from deq.noise import inject_si1000
from deq.noise import strip_noise

# ── Basic gate noise ─────────────────────────────────────────────────


def test_one_qubit_gate_depolarize1() -> None:
    src = dedent("""\
        GADGET Foo {
            H 0
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "DEPOLARIZE1(0.001) 0" in got
    # H should still be present
    assert "    H 0\n" in got


def test_two_qubit_gate_depolarize2() -> None:
    src = dedent("""\
        GADGET Foo {
            CX 0 1
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "DEPOLARIZE2(0.001) 0 1" in got


def test_cnot_alias() -> None:
    src = dedent("""\
        GADGET Foo {
            CNOT 2 3
        }
    """)
    got = inject_si1000(src, 0.01)
    assert "DEPOLARIZE2(0.01) 2 3" in got


def test_swap_depolarize2() -> None:
    src = dedent("""\
        GADGET Foo {
            SWAP 0 1
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "DEPOLARIZE2(0.001) 0 1" in got


def test_s_gate_depolarize1() -> None:
    src = dedent("""\
        GADGET Foo {
            S 5
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "DEPOLARIZE1(0.001) 5" in got


def test_sqrt_x_depolarize1() -> None:
    src = dedent("""\
        GADGET Foo {
            SQRT_X 3
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "DEPOLARIZE1(0.001) 3" in got


def test_pauli_i_depolarize1() -> None:
    src = dedent("""\
        GADGET Foo {
            I 0 1 2
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "DEPOLARIZE1(0.001) 0 1 2" in got


def test_pauli_x_depolarize1() -> None:
    src = dedent("""\
        GADGET Foo {
            X 0
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "DEPOLARIZE1(0.001) 0" in got


def test_h_xy_depolarize1() -> None:
    src = dedent("""\
        GADGET Foo {
            H_XY 2
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "DEPOLARIZE1(0.001) 2" in got


def test_sqrt_y_depolarize1() -> None:
    src = dedent("""\
        GADGET Foo {
            SQRT_Y 4
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "DEPOLARIZE1(0.001) 4" in got


def test_c_xyz_depolarize1() -> None:
    src = dedent("""\
        GADGET Foo {
            C_XYZ 1
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "DEPOLARIZE1(0.001) 1" in got


def test_cy_depolarize2() -> None:
    src = dedent("""\
        GADGET Foo {
            CY 0 1
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "DEPOLARIZE2(0.001) 0 1" in got


def test_iswap_depolarize2() -> None:
    src = dedent("""\
        GADGET Foo {
            ISWAP 2 3
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "DEPOLARIZE2(0.001) 2 3" in got


def test_xcz_depolarize2() -> None:
    src = dedent("""\
        GADGET Foo {
            XCZ 0 1
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "DEPOLARIZE2(0.001) 0 1" in got


def test_sqrt_xx_depolarize2() -> None:
    src = dedent("""\
        GADGET Foo {
            SQRT_XX 0 1
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "DEPOLARIZE2(0.001) 0 1" in got


# ── Measurement noise ────────────────────────────────────────────────


def test_z_basis_measure_x_error_before() -> None:
    src = dedent("""\
        GADGET Foo {
            M 0 1 2
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "M(0.001) 0 1 2" in got
    # No separate X_ERROR line for measurement noise
    assert "X_ERROR" not in got


def test_measure_with_existing_probability_replaced() -> None:
    """Pre-existing flip probability on a measurement is replaced."""
    src = dedent("""\
        GADGET Foo {
            M(0.05) 0 1 2
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "M(0.001) 0 1 2" in got
    assert "(0.05)" not in got


def test_mz_gets_x_error() -> None:
    src = dedent("""\
        GADGET Foo {
            MZ 5 6
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "MZ(0.001) 5 6" in got


def test_mx_gets_z_error() -> None:
    src = dedent("""\
        GADGET Foo {
            MX 3 4
        }
    """)
    got = inject_si1000(src, 0.002)
    assert "MX(0.002) 3 4" in got


def test_my_gets_z_error() -> None:
    src = dedent("""\
        GADGET Foo {
            MY 2
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "MY(0.001) 2" in got


def test_mrx_gets_both_measure_and_reset_noise() -> None:
    src = dedent("""\
        GADGET Foo {
            MRX 5
        }
    """)
    got = inject_si1000(src, 0.001)
    lines = got.splitlines()
    # Measurement noise is embedded in the instruction
    assert "MRX(0.001) 5" in got
    # Reset noise is a separate Z_ERROR line after
    mrx_idx = next(i for i, l in enumerate(lines) if "MRX(0.001) 5" in l)
    zerr_lines = [i for i, l in enumerate(lines) if "Z_ERROR(0.001) 5" in l]
    assert len(zerr_lines) == 1, f"Expected 1 Z_ERROR line, got {len(zerr_lines)}"
    assert zerr_lines[0] > mrx_idx


def test_mxx_gets_measurement_noise() -> None:
    src = dedent("""\
        GADGET Foo {
            MXX 0 1
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "MXX(0.001) 0 1" in got


def test_myy_gets_measurement_noise() -> None:
    src = dedent("""\
        GADGET Foo {
            MYY 2 3
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "MYY(0.001) 2 3" in got


def test_mzz_gets_measurement_noise() -> None:
    src = dedent("""\
        GADGET Foo {
            MZZ 0 1 2 3
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "MZZ(0.001) 0 1 2 3" in got


def test_mpp_gets_measurement_noise() -> None:
    src = dedent("""\
        GADGET Foo {
            MPP X0*X1 Z2*Z3
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "MPP(0.001) X0*X1 Z2*Z3" in got


def test_mpp_existing_probability_replaced() -> None:
    src = dedent("""\
        GADGET Foo {
            MPP(0.05) X0*X1
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "MPP(0.001) X0*X1" in got
    assert "(0.05)" not in got


# ── Reset noise ──────────────────────────────────────────────────────


def test_z_basis_reset_x_error_after() -> None:
    src = dedent("""\
        GADGET Foo {
            R 0 1
        }
    """)
    got = inject_si1000(src, 0.001)
    lines = got.splitlines()
    r_idx = next(i for i, l in enumerate(lines) if "R 0 1" in l and "X_ERROR" not in l)
    xerr_idx = next(i for i, l in enumerate(lines) if "X_ERROR(0.001) 0 1" in l)
    assert xerr_idx > r_idx, "X_ERROR must appear after R"


def test_rz_gets_x_error_after() -> None:
    src = dedent("""\
        GADGET Foo {
            RZ 0 1 2
        }
    """)
    got = inject_si1000(src, 0.001)
    lines = got.splitlines()
    rz_idx = next(
        i for i, l in enumerate(lines) if "RZ 0 1 2" in l and "X_ERROR" not in l
    )
    xerr_idx = next(i for i, l in enumerate(lines) if "X_ERROR(0.001) 0 1 2" in l)
    assert xerr_idx > rz_idx


def test_rx_gets_z_error_after() -> None:
    src = dedent("""\
        GADGET Foo {
            RX 7
        }
    """)
    got = inject_si1000(src, 0.001)
    lines = got.splitlines()
    rx_idx = next(i for i, l in enumerate(lines) if "RX 7" in l and "Z_ERROR" not in l)
    zerr_idx = next(i for i, l in enumerate(lines) if "Z_ERROR(0.001) 7" in l)
    assert zerr_idx > rx_idx


def test_ry_gets_z_error_after() -> None:
    src = dedent("""\
        GADGET Foo {
            RY 3
        }
    """)
    got = inject_si1000(src, 0.001)
    lines = got.splitlines()
    ry_idx = next(i for i, l in enumerate(lines) if "RY 3" in l and "Z_ERROR" not in l)
    zerr_idx = next(i for i, l in enumerate(lines) if "Z_ERROR(0.001) 3" in l)
    assert zerr_idx > ry_idx


# ── MR gets both measurement and reset noise ─────────────────────────


def test_mr_gets_both_measure_and_reset_noise() -> None:
    src = dedent("""\
        GADGET Foo {
            MR 1 3
        }
    """)
    got = inject_si1000(src, 0.001)
    lines = got.splitlines()
    # Measurement noise is embedded in the instruction
    assert "MR(0.001) 1 3" in got
    # Reset noise is a separate X_ERROR line after
    mr_idx = next(i for i, l in enumerate(lines) if "MR(0.001) 1 3" in l)
    xerr_lines = [i for i, l in enumerate(lines) if "X_ERROR(0.001) 1 3" in l]
    assert len(xerr_lines) == 1, f"Expected 1 X_ERROR line, got {len(xerr_lines)}"
    assert xerr_lines[0] > mr_idx


# ── Comment and structure preservation ───────────────────────────────


def test_comments_preserved() -> None:
    src = dedent("""\
        # top-level comment
        GADGET Foo {
            # initialize qubits
            R 0 1
            # do some gates
            H 0
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "# top-level comment" in got
    assert "# initialize qubits" in got
    assert "# do some gates" in got


def test_blank_lines_preserved() -> None:
    src = "GADGET Foo {\n\n    R 0\n\n}\n"
    got = inject_si1000(src, 0.001)
    # The blank lines should still be there
    assert "\n\n" in got


def test_indentation_preserved() -> None:
    src = dedent("""\
        GADGET Foo {
            R 0
        }
    """)
    got = inject_si1000(src, 0.001)
    for line in got.splitlines():
        if "X_ERROR" in line:
            assert line.startswith("    "), f"Noise should match indentation: {line!r}"


def test_trailing_comment_on_instruction_preserved() -> None:
    src = dedent("""\
        GADGET Foo {
            R 0 1  # reset data qubits
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "R 0 1  # reset data qubits" in got


# ── COMPOSE and PROGRAM blocks are untouched ─────────────────────────


def test_compose_block_no_noise() -> None:
    src = dedent("""\
        COMPOSE Idle3 {
            INPUT RepetitionCode 0
            REPEAT 3 {
                Idle IN(0) OUT(0)
            }
            OUTPUT RepetitionCode 0
        }
    """)
    got = inject_si1000(src, 0.001)
    assert got == src, "COMPOSE blocks should be untouched"


def test_program_block_no_noise() -> None:
    src = dedent("""\
        PROGRAM MemExp {
            PrepareZ OUT(0)
            MeasureZ IN(0)
            ASSERT_EQ rec[-1] 0
        }
    """)
    got = inject_si1000(src, 0.001)
    assert got == src, "PROGRAM blocks should be untouched"


def test_code_block_no_noise() -> None:
    src = dedent("""\
        CODE RepetitionCode [[3,1,3]] {
            LOGICAL X0 Z0*Z1*Z2
            STABILIZER Z0*Z1 Z1*Z2
        }
    """)
    got = inject_si1000(src, 0.001)
    assert got == src, "CODE blocks should be untouched"


def test_compose_with_logical_cx_no_noise() -> None:
    """CX rec[-1] 0 in COMPOSE is a logical Pauli correction, not a physical gate."""
    src = dedent("""\
        COMPOSE Ejection {
            INPUT RepetitionCode 0
            EjectionBase IN(0) OUT(0)
            CX rec[-1] 0
            Z 0
            OUTPUT RepetitionCode 0
        }
    """)
    got = inject_si1000(src, 0.001)
    assert got == src


def test_cx_with_rec_control_depolarize1() -> None:
    """CX rec[-1] 0 in a GADGET should inject DEPOLARIZE1 on the data qubit."""
    src = dedent("""\
        GADGET Foo {
            M 0
            CX rec[-1] 0
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "DEPOLARIZE1(0.001) 0" in got
    assert "DEPOLARIZE2" not in got
    assert "rec" not in got.replace("CX rec[-1] 0", "")


def test_cx_with_multiple_rec_controls_depolarize1() -> None:
    """CX rec[-1] 0 rec[-2] 1 should inject DEPOLARIZE1 on both data qubits."""
    src = dedent("""\
        GADGET Foo {
            M 0 1
            CX rec[-1] 0 rec[-2] 1
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "DEPOLARIZE1(0.001) 0 1" in got
    assert "DEPOLARIZE2" not in got


def test_cx_with_mixed_rec_and_qubit_pairs() -> None:
    """CX rec[-1] 0 1 2 should inject DEPOLARIZE2 for qubit pairs and DEPOLARIZE1 for rec-controlled qubits."""
    src = dedent("""\
        GADGET Foo {
            M 0
            CX rec[-1] 0 1 2
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "DEPOLARIZE2(0.001) 1 2" in got
    assert "DEPOLARIZE1(0.001) 0" in got


# ── REPEAT inside GADGET gets noise ──────────────────────────────────


def test_repeat_inside_gadget_gets_noise() -> None:
    src = dedent("""\
        GADGET Foo {
            REPEAT 3 {
                H 0
            }
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "DEPOLARIZE1(0.001) 0" in got


def test_repeat_inside_compose_no_noise() -> None:
    src = dedent("""\
        COMPOSE Foo {
            INPUT RepetitionCode 0
            REPEAT 3 {
                Idle 0
            }
            OUTPUT RepetitionCode 0
        }
    """)
    got = inject_si1000(src, 0.001)
    assert got == src


# ── Unknown gate rejection ───────────────────────────────────────────


def test_unknown_gate_rejected() -> None:
    src = dedent("""\
        GADGET Foo {
            FOOBAR 0 1
        }
    """)
    with pytest.raises(ValueError, match="Unknown instruction 'FOOBAR'"):
        inject_si1000(src, 0.001)


def test_unknown_gate_outside_gadget_ignored() -> None:
    """Unknown names in COMPOSE/PROGRAM are gadget applications, not errors."""
    src = dedent("""\
        COMPOSE Foo {
            INPUT RepetitionCode 0
            SomeGadget IN(0) OUT(0)
            OUTPUT RepetitionCode 0
        }
    """)
    got = inject_si1000(src, 0.001)
    assert got == src


# ── Non-gate instructions pass through ───────────────────────────────


def test_tick_passes_through() -> None:
    src = dedent("""\
        GADGET Foo {
            R 0
            TICK
            H 0
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "TICK" in got
    # TICK should not get noise
    lines = got.splitlines()
    tick_idx = next(i for i, l in enumerate(lines) if l.strip() == "TICK")
    # Line before TICK should not be a noise instruction targeting nothing
    assert "DEPOLARIZE" not in lines[tick_idx]


def test_input_output_pass_through() -> None:
    src = dedent("""\
        GADGET Foo {
            INPUT RepetitionCode 0 1 2
            R 0 1 2
            OUTPUT RepetitionCode 0 1 2
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "INPUT RepetitionCode 0 1 2" in got
    assert "OUTPUT RepetitionCode 0 1 2" in got


def test_readout_and_check_pass_through() -> None:
    src = dedent("""\
        GADGET Foo {
            INPUT RepetitionCode 0 1 2
            M 0 1 2
            READOUT rec[-3]
            CHECK rec[-1] rec[-2]
        }
    """)
    got = inject_si1000(src, 0.001)
    assert "READOUT rec[-3]" in got
    assert "CHECK rec[-1] rec[-2]" in got


# ── Probability formatting ───────────────────────────────────────────


def test_integer_probability() -> None:
    src = "GADGET Foo {\n    H 0\n}\n"
    got = inject_si1000(src, 1)
    assert "DEPOLARIZE1(1) 0" in got


def test_scientific_notation_probability() -> None:
    src = "GADGET Foo {\n    H 0\n}\n"
    got = inject_si1000(src, 1e-5)
    assert "DEPOLARIZE1(1e-05) 0" in got


# ── Full file round-trip ─────────────────────────────────────────────


FIXTURE_DIR = Path(__file__).parent / "fixtures"
REP_CODE_DIR = Path(__file__).parent / "repetition_code"
SURFACE_CODE_DIR = Path(__file__).parent / "surface_code"


def test_repetition_code_d3_injects_and_parses() -> None:
    """Inject noise into the rendered repetition code and verify it still parses."""

    src = (REP_CODE_DIR / "repetition_code_d3.deq").read_text(encoding="utf-8")
    noisy = inject_si1000(strip_noise(src), 0.001)

    # Must contain noise instructions
    assert "DEPOLARIZE2" in noisy
    assert "X_ERROR" in noisy

    # Comments must survive
    assert "# prepare all data qubits" in noisy

    # Must still parse
    deq = parse(noisy)
    assert deq is not None


def test_surface_code_d3_injects_and_parses() -> None:
    """Inject noise into the rendered surface code and verify it still parses."""

    src = (SURFACE_CODE_DIR / "surface_code_d3.deq").read_text(encoding="utf-8")
    noisy = inject_si1000(src, 0.001)

    # Should have both X and Z basis noise
    assert "X_ERROR" in noisy
    assert "Z_ERROR" in noisy
    assert "DEPOLARIZE2" in noisy

    # COMPOSE blocks must be untouched
    assert "DEPOLARIZE" not in noisy.split("COMPOSE")[1].split("GADGET")[0]

    deq = parse(noisy)
    assert deq is not None


def test_example_deq_full_roundtrip() -> None:
    """Inject noise into example.deq and verify structure preservation."""

    src = (FIXTURE_DIR / "example.deq").read_text(encoding="utf-8")
    noisy = inject_si1000(src, 0.001)

    # All original comments must be present
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            assert stripped in noisy, f"Comment lost: {stripped!r}"

    # CODE and COMPOSE blocks must be identical
    # (no noise injected outside GADGET blocks)
    deq = parse(noisy)
    assert deq is not None
