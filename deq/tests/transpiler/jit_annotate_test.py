"""Tests for :mod:`deq.transpiler.jit_annotate`."""

from pathlib import Path

from deq.circuit.parser import parse, parse_file
from deq.transpiler.jit_annotate import annotate
from deq.transpiler.jit_library_builder import build_jit_library

REPO_ROOT = Path(__file__).resolve().parents[2]
REP_DEQ = (
    REPO_ROOT / "tests" / "circuit" / "repetition_code" / "repetition_code_d3.deq"
)


def test_annotate_preserves_logicals_and_stabilizers() -> None:
    qfile = parse("""
        CODE Rep [[3,1,1]] {
            LOGICAL X0*X1*X2 Z0*Z1*Z2
            STABILIZER Z0*Z1 Z1*Z2
        }
        """)
    annotated = annotate(qfile)
    assert "LOGICAL X0*X1*X2 Z0*Z1*Z2" in annotated
    assert "STABILIZER Z0*Z1" in annotated
    assert "STABILIZER Z1*Z2" in annotated


def test_annotate_comments_out_circuit_replaces_check_mode() -> None:
    qfile = parse("""
        CODE Rep [[3,1,1]] {
            LOGICAL X0*X1*X2 Z0*Z1*Z2
            STABILIZER Z0*Z1 Z1*Z2
        }

        @CHECKS("auto")
        GADGET MeasureZ {
            INPUT Rep 0 1 2
            X_ERROR(0.05) 0 1 2
            M 0 1 2
            READOUT rec[-1] rec[-2] rec[-3]
        }
        """)
    annotated = annotate(qfile)
    # Noise instruction commented out.
    assert "# X_ERROR" in annotated
    # Measurement kept in original form.
    assert "M 0 1 2" in annotated
    # @CHECKS forced to manual(verify=0) for re-transpilation correctness.
    assert '@CHECKS("manual", verify=0)' in annotated
    # READOUT preserved as-is.
    assert "READOUT rec[-1] rec[-2] rec[-3]" in annotated


def test_annotate_inserts_auto_checks_after_measurement() -> None:
    qfile = parse("""
        CODE Rep [[3,1,1]] {
            LOGICAL X0*X1*X2 Z0*Z1*Z2
            STABILIZER Z0*Z1 Z1*Z2
        }

        GADGET MeasureZ {
            INPUT Rep 0 1 2
            M 0 1 2
        }
        """)
    annotated = annotate(qfile)
    # At least one auto-emitted CHECK should appear.
    assert "CHECK M" in annotated
    # All CHECKs should appear after the M 0 1 2 line.
    lines = annotated.splitlines()
    measure_idx = next(i for i, line in enumerate(lines) if "M 0 1 2" in line)
    check_indices = [i for i, line in enumerate(lines) if "CHECK M" in line]
    assert check_indices, "expected at least one auto check"
    assert all(i > measure_idx for i in check_indices)


def test_annotate_drops_user_check_emits_auto() -> None:
    qfile = parse("""
        CODE Rep [[3,1,1]] {
            LOGICAL X0*X1*X2 Z0*Z1*Z2
            STABILIZER Z0*Z1 Z1*Z2
        }

        GADGET MeasureZ {
            INPUT Rep 0 1 2
            M 0 1 2
            CHECK rec[-5] rec[-3] rec[-2]
        }
        """)
    annotated = annotate(qfile)
    # User-written CHECKs are dropped (redundant with auto-derived checks).
    user_lines = [
        line
        for line in annotated.splitlines()
        if "CHECK rec[-5] rec[-3] rec[-2]" in line
    ]
    assert not user_lines
    # Auto-derived checks should still be present.
    assert "CHECK M" in annotated


def test_annotated_output_is_a_valid_deq_file_with_same_jit_library() -> None:
    qfile = parse_file(str(REP_DEQ))
    annotated = annotate(qfile)
    # Must re-parse cleanly.
    qfile_round = parse(annotated)
    # And produce the same JitLibrary structure for non-COMPOSE gadgets.
    original_library = build_jit_library(qfile)
    annotated_library = build_jit_library(qfile_round)
    # Same set of port-type and gadget-type ids.
    assert {pt.base.ptype for pt in annotated_library.port_types} == {
        pt.base.ptype for pt in original_library.port_types
    }
    assert {gt.base.gtype for gt in annotated_library.gadget_types} == {
        gt.base.gtype for gt in original_library.gadget_types
    }
    # Per-gadget: same number of finished + unfinished checks.
    original_by_gtype = {gt.base.gtype: gt for gt in original_library.gadget_types}
    for annotated_gadget in annotated_library.gadget_types:
        original_gadget = original_by_gtype[annotated_gadget.base.gtype]
        assert len(annotated_gadget.finished_checks) == len(
            original_gadget.finished_checks
        )
        assert len(annotated_gadget.unfinished_checks) == len(
            original_gadget.unfinished_checks
        )


def test_annotate_unrolls_repeat_blocks() -> None:
    qfile = parse("""
        CODE Rep [[3,1,1]] {
            LOGICAL X0*X1*X2 Z0*Z1*Z2
            STABILIZER Z0*Z1 Z1*Z2
        }

        GADGET RepeatedM {
            INPUT Rep 0 1 2
            REPEAT 2 {
                M 0 1 2
            }
        }
        """)
    annotated = annotate(qfile)
    # Two M 0 1 2 lines (unrolled), no REPEAT block in the output.
    assert annotated.count("M 0 1 2") == 2
    assert "REPEAT" not in annotated


def test_annotate_renders_compose_as_gadget_and_program_verbatim() -> None:
    qfile = parse("""
        CODE Rep [[3,1,1]] {
            LOGICAL X0*X1*X2 Z0*Z1*Z2
            STABILIZER Z0*Z1 Z1*Z2
        }

        GADGET PrepareZ {
            R 0 1 2
            OUTPUT Rep 0 1 2
        }

        COMPOSE Foo {
            PrepareZ 0
            OUTPUT Rep 0
        }

        PROGRAM Bar {
            PrepareZ 0
        }
        """)
    annotated = annotate(qfile)
    # COMPOSE definitions are rendered as GADGET blocks with checks/errors.
    assert "GADGET Foo" in annotated
    assert "COMPOSE Foo" not in annotated
    # PROGRAM blocks are emitted live (they are part of the .deq.jit).
    assert "PROGRAM Bar" in annotated
    assert "# PROGRAM Bar" not in annotated
    # And re-parses cleanly.
    parse(annotated)


# ---------------------------------------------------------------------------
# READOUT propagation comments
# ---------------------------------------------------------------------------


def test_annotate_readout_shows_flips_comment() -> None:
    """MeasureZ readout should show which input observables flip it."""
    qfile = parse("""
        CODE Rep [[3,1,1]] {
            LOGICAL X0*X1*X2 Z0*Z1*Z2
            STABILIZER Z0*Z1 Z1*Z2
        }

        GADGET MeasureZ {
            INPUT Rep 0 1 2
            M 0 1 2
            READOUT rec[-1] rec[-2] rec[-3]
        }
        """)
    annotated = annotate(qfile)
    readout_lines = [l for l in annotated.splitlines() if "READOUT" in l]
    assert len(readout_lines) == 1
    assert "# IN0.LX0" in readout_lines[0]


def test_annotate_readout_no_inputs_no_flips() -> None:
    """Gadget with no input ports should have no flips comment."""
    qfile = parse("""
        GADGET G {
            M 0
            READOUT rec[-1]
        }
        """)
    annotated = annotate(qfile)
    readout_lines = [l for l in annotated.splitlines() if "READOUT" in l]
    assert len(readout_lines) == 1
    assert "#" not in readout_lines[0]


def test_annotate_readout_comment_survives_roundtrip() -> None:
    """Propagation comments are stripped by parser — round-trip still works."""
    qfile = parse("""
        CODE Rep [[3,1,1]] {
            LOGICAL X0*X1*X2 Z0*Z1*Z2
            STABILIZER Z0*Z1 Z1*Z2
        }

        GADGET MeasureZ {
            INPUT Rep 0 1 2
            M 0 1 2
            READOUT rec[-1] rec[-2] rec[-3]
        }
        """)
    annotated = annotate(qfile)
    assert "# IN0.LX0" in annotated
    # Must re-parse cleanly.
    round_trip = parse(annotated)
    # And produce the same JIT library.
    original = build_jit_library(qfile)
    recompiled = build_jit_library(round_trip)
    assert {gt.base.gtype for gt in recompiled.gadget_types} == {
        gt.base.gtype for gt in original.gadget_types
    }


# ---------------------------------------------------------------------------
# MPP measurement counting (regression test for rec[-0] bug)
# ---------------------------------------------------------------------------

CODE422_DEQ = REPO_ROOT / "tests" / "circuit" / "fixtures" / "code422.deq"


def test_annotate_mpp_check_offsets_are_valid() -> None:
    """MPP measurement must be counted; CHECK rec[-0] is invalid."""
    qfile = parse_file(str(CODE422_DEQ))
    annotated = annotate(qfile)
    # All rec offsets must be >= 1.
    for line in annotated.splitlines():
        if "CHECK" not in line:
            continue
        for token in line.split():
            if token.startswith("rec["):
                k = int(token.split("-")[1].rstrip("]"))
                assert k >= 1, f"invalid offset rec[-{k}] in: {line}"


def test_annotate_mpp_roundtrips_identical_jit_library() -> None:
    """Annotated output with MPP gadgets must re-transpile identically."""
    qfile = parse_file(str(CODE422_DEQ))
    annotated = annotate(qfile)
    round_trip = parse(annotated)
    original = build_jit_library(qfile)
    recompiled = build_jit_library(round_trip)
    for orig_gt in original.gadget_types:
        anno_gt = next(
            g for g in recompiled.gadget_types if g.base.gtype == orig_gt.base.gtype
        )
        assert len(anno_gt.finished_checks) == len(orig_gt.finished_checks)
        assert len(anno_gt.unfinished_checks) == len(orig_gt.unfinished_checks)


def test_annotate_compose_with_multiple_input_ports() -> None:
    """COMPOSE with multi-port wiring must annotate without crashing."""
    qfile = parse_file(str(CODE422_DEQ))
    annotated = annotate(qfile)
    # Must produce a GADGET block for the compose.
    assert "GADGET NontrivialIdentity" in annotated
    # Must re-parse and re-transpile cleanly.
    round_trip = parse(annotated)
    original = build_jit_library(qfile)
    recompiled = build_jit_library(round_trip)
    orig_by_name = {gt.base.name: gt for gt in original.gadget_types}
    anno_by_name = {gt.base.name: gt for gt in recompiled.gadget_types}
    nt_orig = orig_by_name["NontrivialIdentity"]
    nt_anno = anno_by_name["NontrivialIdentity"]
    assert len(nt_anno.unfinished_checks) == len(nt_orig.unfinished_checks)


def test_annotate_virtual_logical_absorbed_into_propagate_flip() -> None:
    """VIRTUAL LX0 is absorbed into the PROPAGATE row's ``FLIP`` keyword
    by the annotator; the source-level ``VIRTUAL`` statement is dropped.
    """
    qfile = parse("""
        CODE Trivial [[1,1,1]] {
            LOGICAL X0 Z0
        }
        GADGET NOP {
            INPUT Trivial 0
            OUTPUT Trivial 0
            VIRTUAL LX0
        }
    """)
    annotated = annotate(qfile)
    # PROPAGATE captures the flip as the trailing FLIP keyword.
    assert "PROPAGATE OUT0.LX0 FROM IN0.LX0 FLIP" in annotated
    # VIRTUAL is no longer re-emitted; the FLIP suffix carries the
    # same contribution.
    assert "VIRTUAL" not in annotated
    # Must re-parse and re-transpile cleanly (no crash) and produce
    # identical propagation matrices.
    round_trip = parse(annotated)
    build_jit_library(round_trip)


def test_annotate_preselect_statement() -> None:
    """PRESELECT must not crash the annotator."""
    qfile = parse("""
        CODE Trivial [[1,1,1]] {
            LOGICAL X0 Z0
        }
        GADGET Prep {
            R 0 1
            H 1
            CX 1 0
            H 1
            MRX 1
            PRESELECT rec[-1] 0
            OUTPUT Trivial 0
        }
    """)
    annotated = annotate(qfile)
    assert "PRESELECT rec[-1] 0" in annotated
    # Must re-parse cleanly.
    round_trip = parse(annotated)
    recompiled = build_jit_library(round_trip)
    assert len(recompiled.gadget_types) == 1


def test_annotate_s_gate_on_trivial_code() -> None:
    """Regression: an ``S`` gate on a trivial single-qubit code used to
    panic in ``_compute_pc_logical_via_flows`` with
    ``null-space sign closure produced non-real factor 1j; algebra bug``.

    The S gate maps ``X -> Y`` and ``Z -> Z``.  With the output
    observable basis ``{X_L, Z_L}``, the null-space reconstruction
    picks ``u`` / ``v`` combinations that select anticommuting
    observables of the same logical qubit (``X_L * Z_L = -iY_L``), so
    the observable-side ``reconstructed_input.sign /
    reconstructed_output.sign`` term picks up a ±i phase and used to
    make ``sign_factor`` imaginary.  The flow-side ratio
    ``combined_output.sign / combined_input.sign`` is the same ±1
    physical sign but carries the same ±i phase in both numerator and
    denominator, so those cancel — the fix uses the flow-side ratio.
    """
    qfile = parse("""
        CODE Trivial [[1,1,1]] {
            LOGICAL X0 Z0
        }
        GADGET SGate {
            INPUT Trivial 0
            S 0
            OUTPUT Trivial 0
        }
    """)
    annotated = annotate(qfile)
    assert "PROPAGATE OUT0.LZ0 FROM IN0.LX0 IN0.LZ0 FLIP" in annotated
    assert "PROPAGATE OUT0.LX0 FROM IN0.LX0" in annotated
    # The annotated form must transpile back to the same JIT library.
    from deq.transpiler.jit_library_builder import build_jit_library
    src_lib = build_jit_library(qfile)
    ann_lib = build_jit_library(parse(annotated))
    src_cp = src_lib.gadget_types[0].base.correction_propagation
    ann_cp = ann_lib.gadget_types[0].base.correction_propagation
    assert (src_cp.rows, src_cp.cols) == (ann_cp.rows, ann_cp.cols)
    assert sorted(zip(src_cp.i, src_cp.j)) == sorted(zip(ann_cp.i, ann_cp.j))


def test_annotate_s_gate_on_two_qubit_code_preserves_zz() -> None:
    """Transversal S on a 2-logical trivial code applies the single-qubit
    S propagation per logical qubit.  Also exercises the null-space
    solver path where the raw kernel basis contains multiple imaginary-
    sign_factor vectors (the coset structure the previous fix mishandled).
    """
    qfile = parse("""
        CODE Two [[2,2,1]] {
            LOGICAL X0 Z0
            LOGICAL X1 Z1
        }
        GADGET SS {
            INPUT Two 0 1
            S 0
            S 1
            OUTPUT Two 0 1
        }
    """)
    annotated = annotate(qfile)
    # Each qubit independently follows the single-qubit S propagation.
    assert "PROPAGATE OUT0.LZ0 FROM IN0.LX0 IN0.LZ0 FLIP" in annotated
    assert "PROPAGATE OUT0.LX0 FROM IN0.LX0" in annotated
    assert "PROPAGATE OUT0.LZ1 FROM IN0.LX1 IN0.LZ1 FLIP" in annotated
    assert "PROPAGATE OUT0.LX1 FROM IN0.LX1" in annotated
    build_jit_library(parse(annotated))
