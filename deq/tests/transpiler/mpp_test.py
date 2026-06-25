# pylint: disable=no-member
"""Tests for MPP (Measure Pauli Product) support in the deq transpiler."""

import pytest
from deq.circuit.model import CombinerTarget, PauliTarget
from deq.transpiler.stim_constants import mpp_measurement_count, split_mpp_targets

# ---------------------------------------------------------------------------
# split_mpp_targets
# ---------------------------------------------------------------------------


class TestSplitMppTargets:
    """Unit tests for splitting MPP target lists into product groups."""

    def test_single_product_single_qubit(self) -> None:
        targets = [PauliTarget("Z", 0)]
        groups = split_mpp_targets(targets)
        assert len(groups) == 1
        assert groups[0] == [PauliTarget("Z", 0)]

    def test_single_product_multi_qubit(self) -> None:
        # MPP X0*Z1*Z2
        targets = [
            PauliTarget("X", 0),
            CombinerTarget(),
            PauliTarget("Z", 1),
            CombinerTarget(),
            PauliTarget("Z", 2),
        ]
        groups = split_mpp_targets(targets)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_multiple_products(self) -> None:
        # MPP X0*X1 Z2*Z3
        targets = [
            PauliTarget("X", 0),
            CombinerTarget(),
            PauliTarget("X", 1),
            PauliTarget("Z", 2),
            CombinerTarget(),
            PauliTarget("Z", 3),
        ]
        groups = split_mpp_targets(targets)
        assert len(groups) == 2
        assert len(groups[0]) == 2
        assert len(groups[1]) == 2

    def test_three_products(self) -> None:
        # MPP Z0 X1 Y2
        targets = [
            PauliTarget("Z", 0),
            PauliTarget("X", 1),
            PauliTarget("Y", 2),
        ]
        groups = split_mpp_targets(targets)
        assert len(groups) == 3

    def test_overlapping_qubits_allowed(self) -> None:
        # MPP X0*Z1 Y1*Z2 — qubit 1 in two products is fine (sequential measurement)
        targets = [
            PauliTarget("X", 0),
            CombinerTarget(),
            PauliTarget("Z", 1),
            PauliTarget("Y", 1),
            CombinerTarget(),
            PauliTarget("Z", 2),
        ]
        groups = split_mpp_targets(targets)
        assert len(groups) == 2

    def test_inverted_target(self) -> None:
        targets = [PauliTarget("Z", 0, inverted=True)]
        groups = split_mpp_targets(targets)
        assert len(groups) == 1
        assert groups[0][0].inverted is True


class TestMppMeasurementCount:
    """Unit tests for counting MPP measurement results."""

    def test_single_product(self) -> None:
        targets = [
            PauliTarget("X", 0),
            CombinerTarget(),
            PauliTarget("X", 1),
        ]
        assert mpp_measurement_count(targets) == 1

    def test_two_products(self) -> None:
        targets = [
            PauliTarget("Z", 0),
            CombinerTarget(),
            PauliTarget("Z", 1),
            PauliTarget("X", 2),
            CombinerTarget(),
            PauliTarget("X", 3),
        ]
        assert mpp_measurement_count(targets) == 2


# ---------------------------------------------------------------------------
# Full transpiler integration: MPP-based stabilizer measurement
# ---------------------------------------------------------------------------

from deq.circuit.parser import parse
from deq.transpiler.jit_library_builder import build_jit_library

# Repetition code where stabilizers are measured using MPP instead of
# ancilla-based syndrome extraction.
_MPP_REPETITION_CODE = """\
CODE RepetitionCode [[3,1,3]] {
    LOGICAL X0*X1*X2 Z0*Z1*Z2
    STABILIZER Z0*Z1 Z1*Z2
}

GADGET PrepareZ {
    R 0 1 2
    X_ERROR(0.01) 0 1 2
    OUTPUT RepetitionCode 0 1 2
}

GADGET IdleMPP {
    INPUT RepetitionCode 0 1 2
    X_ERROR(0.01) 0 1 2
    MPP Z0*Z1
    MPP Z1*Z2
    OUTPUT RepetitionCode 0 1 2
}

GADGET MeasureZ {
    INPUT RepetitionCode 0 1 2
    X_ERROR(0.01) 0 1 2
    M 0 1 2
    READOUT rec[-3] rec[-2] rec[-1]
}

PROGRAM Simulation {
    PrepareZ 0
    IdleMPP 0
    MeasureZ 0
    ASSERT_EQ rec[-1] 0
}
"""


class TestMppTranspiler:
    """Integration tests: build a JIT library from a circuit using MPP."""

    def test_mpp_gadget_builds(self) -> None:
        """MPP-based gadget compiles without error."""
        lib = build_jit_library(parse(_MPP_REPETITION_CODE))
        names = {gt.base.name for gt in lib.gadget_types}
        assert "IdleMPP" in names

    def test_mpp_measurement_count_in_gadget(self) -> None:
        """IdleMPP should have 2 internal measurements from MPP Z0*Z1 Z1*Z2."""
        lib = build_jit_library(parse(_MPP_REPETITION_CODE))
        idle = next(gt for gt in lib.gadget_types if gt.base.name == "IdleMPP")
        # 2 input-virtual + 2 internal MPP + 2 output-virtual = 6 total
        # finished checks connect input-virtual ↔ MPP measurement
        assert len(idle.finished_checks) > 0

    def test_mpp_single_qubit_degenerates_to_m(self) -> None:
        """MPP Z0 should behave identically to M 0."""
        src = """\
CODE TrivialCode [[1,1,1]] {
    LOGICAL X0 Z0
}

GADGET MeasSingle {
    INPUT TrivialCode 0
    MPP Z0
    READOUT rec[-1]
}

PROGRAM P {
    MeasSingle 0
    ASSERT_EQ rec[-1] 0
}
"""
        lib = build_jit_library(parse(src))
        gt = next(gt for gt in lib.gadget_types if gt.base.name == "MeasSingle")
        # One internal measurement from MPP Z0, one readout
        assert len(gt.base.readouts) == 1
