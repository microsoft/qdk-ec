"""Tests for the FramePropagator Python binding.

Covers construction/getters, per-shot injection, reset_qubit semantics,
measure-and-reset ordering, out-of-range error handling, and a regression
guard that the S gate propagates errors correctly through the binding.
"""

import pytest
from binar import BitMatrix
from paulimer import FramePropagator, SparsePauli, UnitaryOpcode


class TestBasics:
    def test_getters(self):
        fp = FramePropagator(3, 5, 7)
        assert fp.qubit_count == 3
        # outcome_count grows as measurements are recorded; starts at 0.
        assert fp.outcome_count == 0
        assert fp.shot_count == 7

    def test_outcome_deltas_is_bitmatrix(self):
        fp = FramePropagator(1, 1, 1)
        fp.measure(SparsePauli("Z"))
        assert isinstance(fp.outcome_deltas, BitMatrix)


class TestInjectionAndMeasurement:
    def test_per_shot_injection_is_independent(self):
        # Shot 0: X on control spreads through CNOT to flip both Z measurements.
        # Shot 1: Z on target commutes with both Z measurements -> no flips.
        fp = FramePropagator(2, 2, 2)
        fp.inject_pauli(0, SparsePauli("XI"))
        fp.inject_pauli(1, SparsePauli("IZ"))
        fp.apply_unitary(UnitaryOpcode.ControlledX, [0, 1])
        fp.measure(SparsePauli("ZI"))
        fp.measure(SparsePauli("IZ"))
        d = fp.outcome_deltas
        assert d[0, 0] and d[1, 0]
        assert not d[0, 1] and not d[1, 1]

    def test_s_gate_maps_x_error_to_y(self):
        # Regression for the apply_s/apply_sqrt_x swap: S(X) = Y, so a Z
        # measurement (anticommuting with the X part of Y) must flip.
        fp = FramePropagator(1, 1, 1)
        fp.inject_pauli(0, SparsePauli("X"))
        fp.apply_unitary(UnitaryOpcode.SqrtZ, [0])
        fp.measure(SparsePauli("Z"))
        assert fp.outcome_deltas[0, 0]


class TestReset:
    def test_reset_clears_frame(self):
        fp = FramePropagator(1, 1, 1)
        fp.inject_pauli(0, SparsePauli("Z"))
        fp.reset_qubit(0)
        fp.apply_unitary(UnitaryOpcode.Hadamard, [0])
        fp.measure(SparsePauli("Z"))
        assert not fp.outcome_deltas[0, 0]

    def test_measure_then_reset_records_delta_before_clearing(self):
        fp = FramePropagator(1, 2, 1)
        fp.inject_pauli(0, SparsePauli("X"))
        fp.measure(SparsePauli("Z"))  # pre-reset: X flips Z
        fp.reset_qubit(0)
        fp.apply_unitary(UnitaryOpcode.Hadamard, [0])
        fp.measure(SparsePauli("Z"))  # post-reset: clean
        d = fp.outcome_deltas
        assert d[0, 0] and not d[1, 0]


class TestBounds:
    def test_reset_qubit_out_of_range(self):
        fp = FramePropagator(2, 1, 2)
        with pytest.raises(IndexError):
            fp.reset_qubit(2)

    def test_inject_shot_out_of_range(self):
        fp = FramePropagator(2, 1, 2)
        with pytest.raises(IndexError):
            fp.inject_pauli(2, SparsePauli("XI"))

    def test_inject_qubit_out_of_range(self):
        fp = FramePropagator(2, 1, 2)
        with pytest.raises(IndexError):
            fp.inject_pauli(0, SparsePauli("IIX"))  # X on qubit 2
