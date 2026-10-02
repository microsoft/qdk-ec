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


class TestOutcomeFlip:
    def test_per_shot_flip_leaves_qubit_frames_unchanged(self):
        propagator = FramePropagator(1, 2, 65)
        propagator.inject_pauli(1, SparsePauli.x(0))
        outcome = propagator.measure(SparsePauli.z(0))
        propagator.inject_outcome_flip(64, outcome)
        repeated = propagator.measure(SparsePauli.z(0))
        deltas = propagator.outcome_deltas
        for shot in range(65):
            assert deltas[outcome, shot] == (shot in (1, 64))
            assert deltas[repeated, shot] == (shot == 1)
        assert propagator.qubit_count == 1
        assert propagator.outcome_count == 2

    def test_xor_cancels_an_existing_outcome_error(self):
        propagator = FramePropagator(1, 1, 1)
        propagator.inject_pauli(0, SparsePauli.x(0))
        outcome = propagator.measure(SparsePauli.z(0))
        propagator.inject_outcome_flip(0, outcome)
        assert not propagator.outcome_deltas[outcome, 0]
        propagator.inject_outcome_flip(0, outcome)
        assert propagator.outcome_deltas[outcome, 0]

    @pytest.mark.parametrize("allocated", [False, True])
    def test_qubit_free_outcomes_support_injection(self, allocated):
        propagator = FramePropagator(0, 1, 1)
        outcome = propagator.allocate_random_bit() if allocated else propagator.measure(SparsePauli.identity())
        propagator.inject_outcome_flip(0, outcome)
        assert propagator.outcome_deltas[outcome, 0]
        assert propagator.qubit_count == 0

    def test_chained_feedback_uses_injected_outcome(self):
        propagator = FramePropagator(2, 3, 1)
        outcome = propagator.allocate_random_bit()
        propagator.inject_outcome_flip(0, outcome)
        propagator.apply_conditional_pauli(SparsePauli.x(0), [outcome])
        intermediate = propagator.measure(SparsePauli.z(0))
        propagator.apply_conditional_pauli(SparsePauli.x(1), [intermediate])
        final_outcome = propagator.measure(SparsePauli.z(1))
        assert propagator.outcome_deltas[intermediate, 0]
        assert propagator.outcome_deltas[final_outcome, 0]

    def test_invalid_indices_leave_existing_outcomes_unchanged(self):
        propagator = FramePropagator(0, 8, 2)
        outcome = propagator.allocate_random_bit()
        for shot, invalid_outcome in ((2, outcome), (0, 1), (0, 8)):
            with pytest.raises(IndexError):
                propagator.inject_outcome_flip(shot, invalid_outcome)
        assert not propagator.outcome_deltas[outcome, 0]
        assert not propagator.outcome_deltas[outcome, 1]
        assert propagator.outcome_count == 1


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
