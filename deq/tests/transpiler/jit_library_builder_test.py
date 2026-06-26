"""Tests for :mod:`deq.transpiler.jit_library_builder`."""

import warnings
from pathlib import Path

import pytest

from deq.circuit.parser import parse, parse_file
from deq.proto import deq_jit_pb2 as jit_pb
from deq.transpiler.jit_library_builder import build_jit_library

REPO_ROOT = Path(__file__).resolve().parents[2]
REP_DEQ = (
    REPO_ROOT / "tests" / "circuit" / "repetition_code" / "repetition_code_d3.deq"
)


def test_build_library_on_repetition_code_d3() -> None:
    qfile = parse_file(str(REP_DEQ))
    library = build_jit_library(qfile)

    # 1 CODE → 1 port type; 5 GADGET defs (PrepareZ, MeasureZ, Syndrome,
    # NOP) — ComposeDefinitions (FTSyndrome, FTPrepareZ) are not GADGETs.
    port_names = [pt.base.name for pt in library.port_types]
    assert port_names == ["RepetitionCode"]
    gadget_names = [gt.base.name for gt in library.gadget_types]
    assert "PrepareZ" in gadget_names
    assert "MeasureZ" in gadget_names
    assert "Syndrome" in gadget_names
    assert "NOP" in gadget_names

    # Auto-assigned ids must be unique and start from 1.
    ptypes = sorted(pt.base.ptype for pt in library.port_types)
    assert ptypes == list(range(1, len(ptypes) + 1))
    gtypes = sorted(gt.base.gtype for gt in library.gadget_types)
    assert gtypes == list(range(1, len(gtypes) + 1))

    gadget_by_name = {gt.base.name: gt for gt in library.gadget_types}

    prepare_z = gadget_by_name["PrepareZ"]
    assert len(prepare_z.finished_checks) == 0
    assert len(prepare_z.unfinished_checks) == 2
    assert len(prepare_z.base.measurements) == 0
    assert len(prepare_z.base.inputs) == 0
    assert len(prepare_z.base.outputs) == 1
    # Each unfinished check's remaining present measurements are *only*
    # input-virtual/internal — the output stabilizer is implied by index.
    for check in prepare_z.unfinished_checks:
        for m in check.measurements:
            # PrepareZ has no inputs and no internal measurements in the
            # present list after stripping the OV.
            # (The drop leaves 0 measurements because the check consists
            # only of that single output stabilizer.)
            pytest.fail(f"unexpected present measurement {m}")

    syndrome = gadget_by_name["Syndrome"]
    # Syndrome: 2 input virtuals + 2 internal (M 1 3) + 2 output virtuals
    # → 2 unfinished checks, and up to a few finished checks depending on
    # the derived row-space basis.
    assert len(syndrome.base.inputs) == 1
    assert len(syndrome.base.outputs) == 1
    assert len(syndrome.base.measurements) == 2
    assert len(syndrome.unfinished_checks) == 2
    # Every present measurement index must be within bounds and, when
    # targeting the input port, within that port's stabilizer list.
    input_ptype = syndrome.base.inputs[0].ptype
    input_port = next(pt for pt in library.port_types if pt.base.ptype == input_ptype)
    num_input_stabs = len(input_port.stabilizers)
    for check in list(syndrome.finished_checks) + list(syndrome.unfinished_checks):
        for m in check.measurements:
            if m.HasField("input_port"):
                assert m.input_port == 0
                assert m.measurement_index < num_input_stabs
            else:
                assert m.measurement_index < len(syndrome.base.measurements)


def test_build_library_respects_pinned_ids() -> None:
    source = """
    @PTYPE(7)
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }

    @GTYPE(42)
    GADGET PrepareZ {
        R 0 1 2
        OUTPUT Rep 0 1 2
    }

    GADGET Trivial {
        R 0
        M 0
    }
    """
    library = build_jit_library(parse(source))
    by_name_port = {pt.base.name: pt.base.ptype for pt in library.port_types}
    by_name_gadget = {gt.base.name: gt.base.gtype for gt in library.gadget_types}
    assert by_name_port["Rep"] == 7
    assert by_name_gadget["PrepareZ"] == 42
    # Trivial is auto-assigned and must skip 42 if it had been produced
    # before it; here it's produced after PrepareZ, so it gets 1 (the
    # smallest unused id).
    assert by_name_gadget["Trivial"] == 1


def test_build_library_rejects_conflicting_pins() -> None:
    source = """
    @GTYPE(3)
    GADGET A { R 0 M 0 }

    @GTYPE(3)
    GADGET B { R 0 M 0 }
    """
    with pytest.raises(ValueError, match="conflicts with an earlier pin"):
        build_jit_library(parse(source))


def test_build_library_rejects_invalid_pin() -> None:
    source = """
    @GTYPE(0)
    GADGET A { R 0 M 0 }
    """
    with pytest.raises(ValueError, match="positive integer"):
        build_jit_library(parse(source))


def test_unfinished_check_drops_output_virtual_member() -> None:
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }

    GADGET MeasureZ {
        INPUT Rep 0 1 2
        M 0 1 2
        READOUT rec[-1] rec[-2] rec[-3]
    }
    """
    library = build_jit_library(parse(source))
    measure_z = next(gt for gt in library.gadget_types if gt.base.name == "MeasureZ")
    # MeasureZ: 2 input virtuals + 3 internal + 0 output virtuals.
    assert len(measure_z.unfinished_checks) == 0
    # Present measurements in finished checks should only reference
    # valid indices.
    assert len(measure_z.finished_checks) > 0


def test_compose_fan_out_consumes_all_dangling_outputs() -> None:
    """Multiple parallel sub-gadgets, each contributing a final OUTPUT wire.

    Without this regression test, the COMPOSE builder used to wire the
    output mock only to the *last* sub-gadget, so wires produced by
    earlier parallel sub-gadgets dangled and the JIT compiler hung.
    """
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0
        STABILIZER Z0*Z1 Z1*Z2
    }

    GADGET PrepareZ {
        R 0 1 2
        OUTPUT Rep 0 1 2
    }

    COMPOSE Dynamic1 {
        PrepareZ 0
        PrepareZ 1
        PrepareZ 2
        OUTPUT Rep 0
        OUTPUT Rep 1
        OUTPUT Rep 2
    }
    """
    library = build_jit_library(parse(source))
    dynamic1 = next(gt for gt in library.gadget_types if gt.base.name == "Dynamic1")
    assert len(dynamic1.base.outputs) == 3


def test_compose_rejects_duplicate_output_wire() -> None:
    """A COMPOSE that binds the same wire to multiple OUTPUT ports must
    raise a structured ``ValueError`` rather than panicking inside the
    Rust JIT compiler. Regression test for the duplicate-OUTPUT panic in
    ``deq_runtime/src/jit/jit_compiler.rs``.
    """
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0
        STABILIZER Z0*Z1 Z1*Z2
    }

    GADGET Id {
        INPUT Rep 0 1 2
        OUTPUT Rep 0 1 2
    }

    GADGET PrepareZ {
        R 0 1 2
        OUTPUT Rep 0 1 2
    }

    COMPOSE Chain {
        INPUT Rep 0
        Id 0
        OUTPUT Rep 0
        OUTPUT Rep 0
    }
    """
    with pytest.raises(ValueError) as exc_info:
        build_jit_library(parse(source))
    msg = str(exc_info.value)
    assert "COMPOSE 'Chain'" in msg
    assert "OUTPUT" in msg
    assert "wire 0" in msg


def test_compose_rejects_duplicate_input_wire() -> None:
    """A COMPOSE that binds the same wire to multiple INPUT ports must
    raise a structured ``ValueError``.
    """
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0
        STABILIZER Z0*Z1 Z1*Z2
    }

    GADGET Id {
        INPUT Rep 0 1 2
        OUTPUT Rep 0 1 2
    }

    COMPOSE Chain {
        INPUT Rep 0
        INPUT Rep 0
        Id 0
        OUTPUT Rep 0
    }
    """
    with pytest.raises(ValueError) as exc_info:
        build_jit_library(parse(source))
    msg = str(exc_info.value)
    assert "COMPOSE 'Chain'" in msg
    assert "INPUT" in msg
    assert "wire 0" in msg


def test_compose_rejects_dangling_outputs() -> None:
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0
        STABILIZER Z0*Z1 Z1*Z2
    }

    GADGET PrepareZ {
        R 0 1 2
        OUTPUT Rep 0 1 2
    }

    COMPOSE Dangling {
        PrepareZ 0
        PrepareZ 1
    }
    """
    with pytest.raises(ValueError) as exc_info:
        build_jit_library(parse(source))
    msg = str(exc_info.value)
    assert "COMPOSE 'Dangling'" in msg
    assert "Dangling wires" in msg
    assert "wire 0" in msg
    assert "wire 1" in msg


def test_compose_rejects_output_for_consumed_wire() -> None:
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0
        STABILIZER Z0*Z1 Z1*Z2
    }

    GADGET PrepareZ {
        R 0 1 2
        OUTPUT Rep 0 1 2
    }

    GADGET MeasureZ {
        INPUT Rep 0 1 2
        M 0 1 2
        READOUT rec[-1]
    }

    COMPOSE Closed {
        PrepareZ 0
        MeasureZ 0
        OUTPUT Rep 0
    }
    """
    with pytest.raises(ValueError) as exc_info:
        build_jit_library(parse(source))
    msg = str(exc_info.value)
    assert "COMPOSE 'Closed'" in msg
    assert "Declared OUTPUT wires" in msg
    assert "wire 0" in msg


def test_compose_rejects_dangling_input_overwritten_by_gadget() -> None:
    """Regression test for the dangling-INPUT hang (issue #67).

    A COMPOSE that takes an INPUT wire which is then immediately
    overwritten by a sub-gadget that does not consume it must be
    rejected with a clear error.  Without this validation the JIT
    compiler used to block forever inside ``static_jit_compile`` waiting
    for a consumer of the input mock's output port (uninterruptible by
    Ctrl+C — a single-line bad input produced an unkillable hang).
    """
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0
        STABILIZER Z0*Z1 Z1*Z2
    }

    GADGET G {
        OUTPUT Rep 0 1 2
    }

    COMPOSE C {
        INPUT Rep 0
        G 0
        OUTPUT Rep 0
    }
    """
    with pytest.raises(ValueError) as exc_info:
        build_jit_library(parse(source))
    msg = str(exc_info.value)
    assert "COMPOSE 'C'" in msg
    assert "Dangling wires" in msg
    assert "COMPOSE INPUT" in msg
    assert "wire 0" in msg


def test_compose_rejects_shortcut_with_too_many_targets() -> None:
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0
        STABILIZER Z0*Z1 Z1*Z2
    }

    GADGET PrepareZ {
        R 0 1 2
        OUTPUT Rep 0 1 2
    }

    COMPOSE TooMany {
        PrepareZ 0 1 2
        OUTPUT Rep 0
    }
    """
    with pytest.raises(ValueError) as exc_info:
        build_jit_library(parse(source))
    msg = str(exc_info.value)
    assert "COMPOSE 'TooMany'" in msg
    assert "'PrepareZ' has 3 target(s)" in msg
    assert "0 INPUT port(s) and 1 OUTPUT port(s)" in msg
    assert "exactly 1 target(s)" in msg


def test_compose_rejects_shortcut_with_too_few_targets() -> None:
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0
        STABILIZER Z0*Z1 Z1*Z2
    }

    GADGET TransversalCNOT {
        INPUT Rep 0 1 2
        INPUT Rep 3 4 5
        CX 0 3 1 4 2 5
        OUTPUT Rep 0 1 2
        OUTPUT Rep 3 4 5
    }

    GADGET PrepareZ {
        R 0 1 2
        OUTPUT Rep 0 1 2
    }

    COMPOSE TooFew {
        PrepareZ 0
        PrepareZ 1
        TransversalCNOT 0
        OUTPUT Rep 0
        OUTPUT Rep 1
    }
    """
    with pytest.raises(ValueError) as exc_info:
        build_jit_library(parse(source))
    msg = str(exc_info.value)
    assert "COMPOSE 'TooFew'" in msg
    assert "'TransversalCNOT' has 1 target(s)" in msg
    assert "exactly 2 target(s)" in msg


def test_compose_rejects_explicit_application_with_wrong_port_count() -> None:
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0
        STABILIZER Z0*Z1 Z1*Z2
    }

    GADGET PrepareZ {
        R 0 1 2
        OUTPUT Rep 0 1 2
    }

    COMPOSE WrongOut {
        PrepareZ OUT(0 1)
        OUTPUT Rep 0
    }
    """
    with pytest.raises(ValueError) as exc_info:
        build_jit_library(parse(source))
    msg = str(exc_info.value)
    assert "COMPOSE 'WrongOut'" in msg
    assert "'PrepareZ'" in msg
    assert "0 IN wire(s) and 2 OUT wire(s)" in msg
    assert "0 INPUT port(s) and 1 OUTPUT port(s)" in msg


def test_library_is_serialisable() -> None:
    qfile = parse_file(str(REP_DEQ))
    library = build_jit_library(qfile)
    payload = library.SerializeToString()
    round_trip = jit_pb.JitLibrary()
    round_trip.ParseFromString(payload)
    assert len(round_trip.gadget_types) == len(library.gadget_types)
    assert len(round_trip.port_types) == len(library.port_types)


def test_readouts_use_real_measurement_indices() -> None:
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }

    GADGET MeasureZ {
        INPUT Rep 0 1 2
        M 0 1 2
        READOUT rec[-1] rec[-2] rec[-3]
    }
    """
    library = build_jit_library(parse(source))
    measure_z = next(gt for gt in library.gadget_types if gt.base.name == "MeasureZ")
    assert len(measure_z.base.readouts) == 1
    readout = measure_z.base.readouts[0]
    # 3 real measurements, indices 0/1/2 in the real-measurement space
    # (input virtuals are *not* counted here).
    assert sorted(readout.measurement_indices) == [0, 1, 2]
    # readout_propagation shape: |readouts| x (|input_observables|+1).
    # One input port with 1 logical → 2 input observables → cols=3.
    prop = measure_z.base.readout_propagation
    assert prop.rows == 1
    # Unified frame: 2*k + (n-k) = 4 input observables → cols=5.
    assert prop.cols == 5
    # Walking the LX representative (col 0) → recorded at partner col 1 (LZ).
    assert sorted(zip(prop.i, prop.j)) == [(0, 1)]


def test_readouts_flip_sets_affine_column() -> None:
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }

    GADGET MeasureZFlip {
        INPUT Rep 0 1 2
        M 0 1 2
        READOUT rec[-1] rec[-2] rec[-3] FLIP
    }
    """
    library = build_jit_library(parse(source))
    gadget = next(gt for gt in library.gadget_types if gt.base.name == "MeasureZFlip")
    prop = gadget.base.readout_propagation
    # Implicit walk of the LX representative anticommutes with `M 0 1 2`,
    # which flips the LZ observable's bit (partner of col 0 → col 1).
    # Plus the affine/constant column (index = num_input_observables = 2)
    # set by FLIP.
    # Affine/constant column is at index num_input_observables = 4.
    nonzeros = sorted(zip(prop.i, prop.j))
    assert nonzeros == [(0, 1), (0, 4)]


def test_readouts_xor_duplicate_measurements() -> None:
    source = """
    GADGET G {
        M 0 1
        READOUT rec[-1] rec[-1] rec[-2]
    }
    """
    library = build_jit_library(parse(source))
    (gadget,) = library.gadget_types
    (readout,) = gadget.base.readouts
    # rec[-1] appears twice → cancels in GF(2). Only rec[-2] (index 0)
    # remains.
    assert list(readout.measurement_indices) == [0]


def test_readouts_reject_input_virtual_reference() -> None:
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }

    GADGET Bad {
        INPUT Rep 0 1 2
        M 0
        READOUT rec[-3]
    }
    """
    with pytest.raises(ValueError) as exc_info:
        build_jit_library(parse(source))
    message = str(exc_info.value)
    assert "GADGET 'Bad'" in message
    assert "READOUT rec[-3]" in message
    assert "rec[-3]" in message
    assert "input-virtual" in message


def test_readouts_reject_output_virtual_reference() -> None:
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }

    GADGET Bad {
        M 0
        OUTPUT Rep 0 1 2
        READOUT rec[-1]
    }
    """
    with pytest.raises(ValueError) as exc_info:
        build_jit_library(parse(source))
    message = str(exc_info.value)
    assert "GADGET 'Bad'" in message
    assert "READOUT rec[-1]" in message
    assert "output-virtual" in message


def test_logical_correction_shape_matches_observables_and_readouts() -> None:
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }

    GADGET MeasureZ {
        INPUT Rep 0 1 2
        M 0 1 2
        READOUT rec[-1] rec[-2] rec[-3]
    }
    """
    library = build_jit_library(parse(source))
    gadget = next(gt for gt in library.gadget_types if gt.base.name == "MeasureZ")
    cc = gadget.base.logical_correction
    # No outputs → 0 rows; 1 readout → 1 col.
    assert cc.rows == 0
    assert cc.cols == 1


def test_no_readouts_yields_zero_by_one_propagation() -> None:
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }

    GADGET PrepareZ {
        R 0 1 2
        OUTPUT Rep 0 1 2
    }
    """
    library = build_jit_library(parse(source))
    (gadget,) = library.gadget_types
    assert len(gadget.base.readouts) == 0
    prop = gadget.base.readout_propagation
    # Zero readouts → zero rows; cols = num_input_observables + 1 = 0+1.
    assert prop.rows == 0
    assert prop.cols == 1


def test_logical_frame_default_observable_count() -> None:
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }
    """
    library = build_jit_library(parse(source))
    (port,) = library.port_types
    # Unified frame: 2*k + (n-k) = 2*1 + 2 = 4 observables.
    assert len(port.base.observables) == 4


# ---------------------------------------------------------------------------
# ERROR statements
# ---------------------------------------------------------------------------


def _gadget_with_errors(body: str) -> object:
    source = f"""
    CODE Rep [[3,1,3]] {{
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }}

    GADGET G {{
        INPUT Rep 0 1 2
        M 0 1
        READOUT rec[-1]
        READOUT rec[-2]
        {body}
        OUTPUT Rep 0 1 2
    }}
    """
    library = build_jit_library(parse(source))
    (gadget,) = library.gadget_types
    return gadget


def test_error_with_check_readout_and_logical_targets() -> None:
    gadget = _gadget_with_errors("ERROR(0.001) C0 LX0 R1")
    (err,) = gadget.errors
    assert err.base.probability == pytest.approx(0.001)
    # Two finished checks come from input stabilizers; C0 is finished.
    assert list(err.finished_checks) == [0]
    assert list(err.unfinished_checks) == []
    # LX0 is an X-type error → anticommutes with Z observable → Z column = 1.
    assert list(err.base.residual) == [1]
    assert list(err.base.readout_flips) == [1]
    assert "ERROR(0.001) C0 LX0 R1" in err.base.tag


def test_error_C_index_routes_to_unfinished() -> None:
    gadget = _gadget_with_errors("ERROR(0.5) C2")
    (err,) = gadget.errors
    num_finished = len(gadget.finished_checks)
    num_unfinished = len(gadget.unfinished_checks)
    assert num_finished + num_unfinished >= 3
    # C2 should land in unfinished if there are <=2 finished checks.
    if num_finished <= 2:
        assert list(err.finished_checks) == []
        assert list(err.unfinished_checks) == [2 - num_finished]
    else:
        assert list(err.finished_checks) == [2]


def test_error_LY_flips_both_residual_columns() -> None:
    gadget = _gadget_with_errors("ERROR(0.01) LY0")
    (err,) = gadget.errors
    # LY0 → residual indices 0 and 1.
    assert list(err.base.residual) == [0, 1]


def test_error_xor_dedup_cancels_repeated_targets() -> None:
    gadget = _gadget_with_errors("ERROR(0.1) C0 C0 LX0 LX0 R0 R0")
    (err,) = gadget.errors
    assert list(err.finished_checks) == []
    assert list(err.base.residual) == []
    assert list(err.base.readout_flips) == []


def test_error_check_index_out_of_range() -> None:
    with pytest.raises(ValueError, match="check target C99 is out of range"):
        _gadget_with_errors("ERROR(0.001) C99")


def test_error_readout_index_out_of_range() -> None:
    with pytest.raises(ValueError, match="readout target R5 is out of range"):
        _gadget_with_errors("ERROR(0.001) R5")


def test_error_observable_out_of_range() -> None:
    with pytest.raises(ValueError, match="observable LX9 is out of range"):
        _gadget_with_errors("ERROR(0.001) LX9")


def test_error_logical_target_works() -> None:
    gadget = _gadget_with_errors("ERROR(0.001) LX0")
    (err,) = gadget.errors
    # LX0 → flips LZ0 → column 1
    assert list(err.base.residual) == [1]


def test_error_physical_target_is_rejected() -> None:
    with pytest.raises(ValueError) as exc_info:
        _gadget_with_errors("ERROR(0.001) X0")
    msg = str(exc_info.value)
    assert "physical observable X0" in msg
    assert "LX0" in msg


def test_error_lz_target_works() -> None:
    gadget = _gadget_with_errors("ERROR(0.001) LZ0")
    (err,) = gadget.errors
    # LZ0 flips the X observable → stored at X column = 0.
    assert list(err.base.residual) == [0]


def test_error_probability_out_of_range_is_rejected() -> None:
    with pytest.raises(SyntaxError, match=r"probability must be in \[0, 1\]"):
        _gadget_with_errors("ERROR(1.5) C0")


def test_no_error_statements_yields_empty_errors_list() -> None:
    gadget = _gadget_with_errors("")
    assert list(gadget.errors) == []


def test_multiple_error_statements_emit_multiple_rows() -> None:
    gadget = _gadget_with_errors("ERROR(0.001) C0\n        ERROR(0.002) R0 LZ0")
    assert len(gadget.errors) == 2
    assert gadget.errors[0].base.probability == pytest.approx(0.001)
    assert gadget.errors[1].base.probability == pytest.approx(0.002)
    assert list(gadget.errors[1].base.readout_flips) == [0]
    # LZ0 is a Z-type error → anticommutes with X observable → X column = 0.
    assert list(gadget.errors[1].base.residual) == [0]


def test_compose_gtype_pinned() -> None:
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }

    GADGET Syndrome {
        INPUT Rep 0 2 4
        R 1 3
        CX 0 1 2 3
        CX 2 1 4 3
        M 1 3
        OUTPUT Rep 0 2 4
    }

    @GTYPE(10)
    COMPOSE FTSyndrome {
        INPUT Rep 0
        REPEAT 3 {
            Syndrome 0
        }
        OUTPUT Rep 0
    }
    """
    library = build_jit_library(parse(source))
    by_name = {gt.base.name: gt.base.gtype for gt in library.gadget_types}
    assert by_name["FTSyndrome"] == 10
    assert by_name["Syndrome"] == 1


def test_compose_gtype_pin_conflicts_with_gadget_pin() -> None:
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }

    @GTYPE(5)
    GADGET Syndrome {
        INPUT Rep 0 2 4
        R 1 3
        CX 0 1 2 3
        CX 2 1 4 3
        M 1 3
        OUTPUT Rep 0 2 4
    }

    @GTYPE(5)
    COMPOSE FTSyndrome {
        INPUT Rep 0
        Syndrome 0
        OUTPUT Rep 0
    }
    """
    with pytest.raises(ValueError, match="conflicts with an earlier pin"):
        build_jit_library(parse(source))


def test_compose_rejects_non_gtype_decorator() -> None:
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }

    GADGET Syndrome {
        INPUT Rep 0 2 4
        R 1 3
        CX 0 1 2 3
        CX 2 1 4 3
        M 1 3
        OUTPUT Rep 0 2 4
    }

    @CHECKS("manual")
    COMPOSE FTSyndrome {
        INPUT Rep 0
        Syndrome 0
        OUTPUT Rep 0
    }
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(
            ValueError, match="only @GTYPE and @REPROPAGATE are supported"
        ):
            build_jit_library(parse(source))


def test_compose_auto_gtype_skips_pinned_ids() -> None:
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }

    @GTYPE(2)
    GADGET Syndrome {
        INPUT Rep 0 2 4
        R 1 3
        CX 0 1 2 3
        CX 2 1 4 3
        M 1 3
        OUTPUT Rep 0 2 4
    }

    @GTYPE(1)
    COMPOSE FTSyndrome {
        INPUT Rep 0
        Syndrome 0
        OUTPUT Rep 0
    }

    COMPOSE FTSyndrome2 {
        INPUT Rep 0
        Syndrome 0
        OUTPUT Rep 0
    }
    """
    library = build_jit_library(parse(source))
    by_name = {gt.base.name: gt.base.gtype for gt in library.gadget_types}
    assert by_name["Syndrome"] == 2
    assert by_name["FTSyndrome"] == 1
    # Auto-assigned must skip 1 and 2 (both taken).
    assert by_name["FTSyndrome2"] == 3


def test_error_statement_inside_repeat_block_is_expanded() -> None:
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }

    GADGET G {
        INPUT Rep 0 1 2
        REPEAT 3 {
            ERROR(0.001) C0
        }
        OUTPUT Rep 0 1 2
    }
    """
    library = build_jit_library(parse(source))
    (gadget,) = library.gadget_types
    assert len(gadget.errors) == 3


# ---------------------------------------------------------------------------
# Unrecognized decorator warnings
# ---------------------------------------------------------------------------


def test_unrecognized_gadget_decorator_warns() -> None:
    source = """
    CODE C [[1,1]] { LOGICAL X0 Z0 }
    @METACHECKS("false")
    GADGET G {
        R 0
        OUTPUT C 0
    }
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        build_jit_library(parse(source))
    decorator_warnings = [
        x for x in w if "unrecognized decorator @METACHECKS" in str(x.message)
    ]
    assert len(decorator_warnings) == 1


def test_unrecognized_compose_decorator_raises() -> None:
    source = """
    CODE C [[1,1]] { LOGICAL X0 Z0 }
    GADGET Prep { R 0\n OUTPUT C 0 }
    GADGET Meas { INPUT C 0\n M 0\n READOUT rec[-1] }
    @FOO("bar")
    COMPOSE Foo { Prep 0\n Meas 0 }
    PROGRAM P { Foo 0\n ASSERT_EQ rec[-1] 0 }
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(
            ValueError, match="only @GTYPE and @REPROPAGATE are supported"
        ):
            build_jit_library(parse(source))


# ---------------------------------------------------------------------------
# Conditional correction (CONDITIONAL statement)
# ---------------------------------------------------------------------------


def test_conditional_lx_flips_lz() -> None:
    """CONDITIONAL R0 LX0 should set logical_correction[1, 0] = 1."""
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }
    GADGET G {
        INPUT Rep 0 1 2
        M 3
        READOUT rec[-1]
        OUTPUT Rep 0 1 2
        CONDITIONAL R0 LX0
    }
    """
    library = build_jit_library(parse(source))
    gadget = next(gt for gt in library.gadget_types if gt.base.name == "G")
    cc = gadget.base.logical_correction
    assert cc.rows == 4  # unified frame: 2*k + (n-k) = 2+2 = 4
    assert cc.cols == 1  # 1 readout
    entries = set(zip(cc.i, cc.j))
    # LX0 flips LZ0 → row 1, col 0
    assert entries == {(1, 0)}


# ---------------------------------------------------------------------------
# COMPOSE with multi-port non-linear wiring (regression)
# ---------------------------------------------------------------------------


CODE422_DEQ = REPO_ROOT / "tests" / "circuit" / "fixtures" / "code422.deq"


def test_compose_multi_port_non_linear_wiring() -> None:
    """COMPOSE where wires fan out to different gadgets must not panic.

    Previously, all connectors pointed at the immediately preceding
    gadget (``prev``), causing an index-out-of-bounds panic in the
    Rust JIT compiler when non-adjacent gadgets needed connecting.
    """
    library = build_jit_library(parse_file(str(CODE422_DEQ)))
    by_name = {gt.base.name: gt for gt in library.gadget_types}
    composed = by_name["NontrivialIdentity"]
    # The identity should have 2 input ports and 2 output ports.
    assert len(composed.base.inputs) == 2
    assert len(composed.base.outputs) == 2
    # Each unfinished check should reference per-port stabilizer indices.
    for check in composed.unfinished_checks:
        for m in check.measurements:
            if m.HasField("input_port"):
                assert m.measurement_index < 2, (
                    f"measurement_index {m.measurement_index} out of range "
                    f"for port {m.input_port} (Code422 has 2 stabilizers)"
                )


def test_conditional_lz_flips_lx() -> None:
    """CONDITIONAL R0 LZ0 should set logical_correction[0, 0] = 1."""
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }
    GADGET G {
        INPUT Rep 0 1 2
        M 3
        READOUT rec[-1]
        OUTPUT Rep 0 1 2
        CONDITIONAL R0 LZ0
    }
    """
    library = build_jit_library(parse(source))
    gadget = next(gt for gt in library.gadget_types if gt.base.name == "G")
    cc = gadget.base.logical_correction
    entries = set(zip(cc.i, cc.j))
    # LZ0 flips LX0 → row 0, col 0
    assert entries == {(0, 0)}


def test_conditional_ly_flips_both() -> None:
    """CONDITIONAL R0 LY0 should flip both LX0 and LZ0."""
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }
    GADGET G {
        INPUT Rep 0 1 2
        M 3
        READOUT rec[-1]
        OUTPUT Rep 0 1 2
        CONDITIONAL R0 LY0
    }
    """
    library = build_jit_library(parse(source))
    gadget = next(gt for gt in library.gadget_types if gt.base.name == "G")
    cc = gadget.base.logical_correction
    entries = set(zip(cc.i, cc.j))
    assert entries == {(0, 0), (1, 0)}


def test_conditional_multiple_targets() -> None:
    """CONDITIONAL R0 LX0 LZ0 should flip both anti-commuting partners."""
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }
    GADGET G {
        INPUT Rep 0 1 2
        M 3
        READOUT rec[-1]
        OUTPUT Rep 0 1 2
        CONDITIONAL R0 LX0 LZ0
    }
    """
    library = build_jit_library(parse(source))
    gadget = next(gt for gt in library.gadget_types if gt.base.name == "G")
    cc = gadget.base.logical_correction
    entries = set(zip(cc.i, cc.j))
    # LX0 → flips LZ0 (row 1), LZ0 → flips LX0 (row 0)
    assert entries == {(0, 0), (1, 0)}


def test_conditional_no_statement_empty_matrix() -> None:
    """Without CONDITIONAL, matrix should be empty."""
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }
    GADGET G {
        INPUT Rep 0 1 2
        M 3
        READOUT rec[-1]
        OUTPUT Rep 0 1 2
    }
    """
    library = build_jit_library(parse(source))
    gadget = next(gt for gt in library.gadget_types if gt.base.name == "G")
    cc = gadget.base.logical_correction
    assert cc.rows == 4
    assert cc.cols == 1
    assert len(cc.i) == 0
    assert len(cc.j) == 0


def test_conditional_invalid_readout_index() -> None:
    """Readout index out of range should raise ValueError."""
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }
    GADGET G {
        INPUT Rep 0 1 2
        M 3
        READOUT rec[-1]
        OUTPUT Rep 0 1 2
        CONDITIONAL R5 LX0
    }
    """
    with pytest.raises(ValueError, match="R5 out of range"):
        build_jit_library(parse(source))


def test_conditional_invalid_logical_index() -> None:
    """Logical index out of range should raise ValueError."""
    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }
    GADGET G {
        INPUT Rep 0 1 2
        M 3
        READOUT rec[-1]
        OUTPUT Rep 0 1 2
        CONDITIONAL R0 LX5
    }
    """
    with pytest.raises(ValueError, match="LX5 out of range"):
        build_jit_library(parse(source))


# ── build_jit_program ──────────────────────────────────────────────────────


def test_build_jit_program_populates_type_metadata_only() -> None:
    """``build_jit_program`` produces the type / measurement / readout
    metadata downstream PROGRAM compilation needs, and *omits* every
    decoder-side field (checks, errors, propagation matrices)."""
    from deq.transpiler.jit_library_builder import build_jit_program

    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }
    GADGET PrepareZ {
        R 0 1 2
        X_ERROR(0.01) 0 1 2
        OUTPUT Rep 0 1 2
    }
    GADGET MeasureZ {
        INPUT Rep 0 1 2
        M(0.01) 0 1 2
        READOUT rec[-3] rec[-2] rec[-1]
    }
    """
    qfile = parse(source)
    full = build_jit_library(qfile)
    program_lib = build_jit_program(qfile)

    full_by_name = {gt.base.name: gt for gt in full.gadget_types}
    program_by_name = {gt.base.name: gt for gt in program_lib.gadget_types}
    assert program_by_name.keys() == full_by_name.keys()

    for name, program_gt in program_by_name.items():
        full_gt = full_by_name[name]
        assert program_gt.base.gtype == full_gt.base.gtype
        assert len(program_gt.base.inputs) == len(full_gt.base.inputs)
        assert len(program_gt.base.outputs) == len(full_gt.base.outputs)
        assert len(program_gt.base.measurements) == len(full_gt.base.measurements)
        assert len(program_gt.base.readouts) == len(full_gt.base.readouts)
        assert [p.ptype for p in program_gt.base.inputs] == [
            p.ptype for p in full_gt.base.inputs
        ]
        assert [p.ptype for p in program_gt.base.outputs] == [
            p.ptype for p in full_gt.base.outputs
        ]

        assert len(program_gt.finished_checks) == 0
        assert len(program_gt.unfinished_checks) == 0
        assert len(program_gt.errors) == 0
        # ``correction_propagation`` is shape-only — rows/cols sized to
        # support VIRTUAL toggles, but no entries populated.
        assert len(program_gt.base.correction_propagation.i) == 0
        assert len(program_gt.base.correction_propagation.j) == 0


def test_build_jit_program_inlines_compose_as_synthetic_gadget() -> None:
    """COMPOSE definitions are inlined into synthetic gadgets so the
    lite library treats them uniformly with regular GADGETs — same
    gtype namespace, same shape metadata."""
    from deq.transpiler.jit_library_builder import build_jit_program

    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }
    GADGET PrepareZ {
        R 0 1 2
        OUTPUT Rep 0 1 2
    }
    COMPOSE PrepareTrio {
        PrepareZ 0
        PrepareZ 1
        PrepareZ 2
        OUTPUT Rep 0
        OUTPUT Rep 1
        OUTPUT Rep 2
    }
    """
    qfile = parse(source)
    program_lib = build_jit_program(qfile)

    by_name = {gt.base.name: gt for gt in program_lib.gadget_types}
    assert by_name.keys() == {"PrepareZ", "PrepareTrio"}
    trio = by_name["PrepareTrio"]
    assert len(trio.base.outputs) == 3
    assert len(trio.base.inputs) == 0
    # COMPOSE inherits the lite-builder invariants — no decoder data.
    assert len(trio.finished_checks) == 0
    assert len(trio.unfinished_checks) == 0
    assert len(trio.errors) == 0


def test_build_jit_program_drives_compile_program_for_jit() -> None:
    """The lite library has just enough metadata for
    :func:`compile_program_for_jit` to produce a valid program."""
    from deq.cli.jit import compile_program_for_jit
    from deq.circuit.model import ProgramDefinition
    from deq.transpiler.jit_library_builder import build_jit_program

    source = """
    CODE Rep [[3,1,3]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }
    GADGET PrepareZ {
        R 0 1 2
        OUTPUT Rep 0 1 2
    }
    GADGET MeasureZ {
        INPUT Rep 0 1 2
        M 0 1 2
        READOUT rec[-3] rec[-2] rec[-1]
    }
    PROGRAM Run {
        PrepareZ 0
        MeasureZ 0
    }
    """
    qfile = parse(source)
    program_lib = build_jit_program(qfile)
    program_def = next(
        d for d in qfile.definitions if isinstance(d, ProgramDefinition)
    )
    compiled, _assertions = compile_program_for_jit(program_lib, program_def)
    assert len(compiled) == 2
    gtype_of_name = {gt.base.name: gt.base.gtype for gt in program_lib.gadget_types}
    assert compiled[0][0].gadget.gtype == gtype_of_name["PrepareZ"]
    assert compiled[1][0].gadget.gtype == gtype_of_name["MeasureZ"]
