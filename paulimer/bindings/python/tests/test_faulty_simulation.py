"""Tests for FaultySimulation.

This module contains:
- Basic functionality tests (creation, sampling, shape)
- Noiseless determinism tests
- Noise behavior tests
- Property-based tests
- Large circuit scaling tests
"""

import numpy as np
import pytest
from hypothesis import given
from paulimer import (
    SparsePauli,
    UnitaryOpcode,
    FaultySimulation,
    PauliFault,
)
from binar import BitMatrix

from tests.conftest import (
    simulation_strategy,
    make_bell_circuit,
    make_ghz_circuit,
    make_repetition_code_circuit,
)


class TestSimulationBasics:
    """Basic functionality tests for FaultySimulation."""

    def test_circuit_builder(self):
        sim = make_bell_circuit()
        assert sim.outcome_count == 2

    def test_sample_shape(self):
        sim = make_bell_circuit()
        outcomes = sim.sample(100)
        assert isinstance(outcomes, BitMatrix)
        assert outcomes.shape == (100, 2)

    def test_sample_noiseless_bell(self):
        sim = make_bell_circuit()
        outcomes = sim.sample(1000)
        arr = np.asarray(outcomes.rows)
        assert np.all(arr == 0), "Bell state stabilizers should all measure 0"

    def test_sample_with_noise(self):
        sim = make_repetition_code_circuit(5, 3)
        sim.apply_fault(PauliFault.depolarizing(0.1, list(range(5))))
        outcomes_noisy = sim.sample(10000)
        arr_noisy = np.array(outcomes_noisy)
        # With noise, we should see some non-zero outcomes
        diff = np.sum(arr_noisy != 0)
        assert diff > 0, "Noisy simulation should have some non-zero outcomes"

    def test_ghz_even(self):
        for n in [2, 4, 6]:
            sim = make_ghz_circuit(n)
            outcomes = sim.sample(100)
            arr = np.asarray(outcomes.rows)
            assert np.all(arr == 0), f"GHZ-{n} ZZ...Z should measure 0"

    def test_empty_circuit(self):
        sim = FaultySimulation()
        assert sim.outcome_count == 0
        outcomes = sim.sample(100)
        assert outcomes.shape == (100, 0)

    def test_zero_shots(self):
        sim = make_bell_circuit()
        outcomes = sim.sample(0)
        assert outcomes.shape[0] == 0

    def test_single_measurement(self):
        sim = FaultySimulation()
        sim.measure(SparsePauli("Z"))
        assert sim.outcome_count == 1
        outcomes = sim.sample(100)
        arr = np.asarray(outcomes.rows)
        assert np.all(arr == 0), "Z on |0⟩ should measure 0"

    def test_repr(self):
        sim = make_bell_circuit()
        repr_str = repr(sim)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0

    def test_error_rate_is_sensible(self):
        # Use a circuit with gates that can have noise
        sim = FaultySimulation()
        distance = 5
        rounds = 2
        for _ in range(rounds):
            # Apply some gates (this is where noise would affect things)
            for i in range(distance):
                sim.apply_unitary(UnitaryOpcode.Hadamard, [i])
                sim.apply_unitary(UnitaryOpcode.Hadamard, [i])  # H^2 = I
            # Apply noise after gates
            sim.apply_fault(PauliFault.depolarizing(0.01, list(range(distance))))
            # Measure syndromes
            for i in range(distance - 1):
                pauli_str = "I" * i + "ZZ" + "I" * (distance - i - 2)
                sim.measure(SparsePauli(pauli_str))
        num_shots = 100000
        outcomes = np.asarray(sim.sample(num_shots).rows)
        rate = np.mean(outcomes)
        assert 0.002 < rate < 0.1, f"Error rate {rate} out of sensible range"

    def test_sample_returns_bitmatrix(self):
        sim = make_bell_circuit()
        outcomes = sim.sample(10)
        assert isinstance(outcomes, BitMatrix)

    def test_bitmatrix_has_shape(self):
        sim = make_bell_circuit()
        outcomes = sim.sample(100)
        assert hasattr(outcomes, "shape")
        assert outcomes.shape == (100, 2)

    def test_zero_error_deterministic(self):
        sim = make_repetition_code_circuit(4, 2)
        outcomes_1 = np.array(sim.sample(100))
        outcomes_2 = np.array(sim.sample(100))
        assert np.array_equal(outcomes_1, outcomes_2)

    def test_seeded_sampling_reproducible(self):
        sim = make_repetition_code_circuit(5, 2)
        sim.apply_fault(PauliFault.depolarizing(0.05, list(range(5))))
        outcomes_1 = np.array(sim.sample(500, seed=12345))
        outcomes_2 = np.array(sim.sample(500, seed=12345))
        assert np.array_equal(outcomes_1, outcomes_2)

    def test_different_seeds_differ(self):
        # Need a circuit with noise that actually gets applied
        sim = FaultySimulation()
        distance = 5
        for i in range(distance):
            sim.apply_unitary(UnitaryOpcode.Hadamard, [i])
            sim.apply_unitary(UnitaryOpcode.Hadamard, [i])
        sim.apply_fault(PauliFault.depolarizing(0.1, list(range(distance))))
        for i in range(distance - 1):
            pauli_str = "I" * i + "ZZ" + "I" * (distance - i - 2)
            sim.measure(SparsePauli(pauli_str))
        outcomes_1 = np.array(sim.sample(1000, seed=11111))
        outcomes_2 = np.array(sim.sample(1000, seed=22222))
        assert not np.array_equal(outcomes_1, outcomes_2)


class TestLargeCircuits:
    """Scaling tests for larger circuits."""

    def test_large_repetition_code(self):
        distance = 21
        rounds = 5
        sim = make_repetition_code_circuit(distance, rounds)
        sim.apply_fault(PauliFault.depolarizing(0.01, list(range(distance))))
        expected_outcomes = rounds * (distance - 1)
        assert sim.outcome_count == expected_outcomes
        outcomes = sim.sample(1000)
        assert outcomes.shape == (1000, expected_outcomes)

    def test_many_qubits(self):
        n = 50
        sim = FaultySimulation()
        for i in range(n):
            sim.apply_unitary(UnitaryOpcode.Hadamard, [i])
        for i in range(n):
            pauli_str = "I" * i + "Z" + "I" * (n - i - 1)
            sim.measure(SparsePauli(pauli_str))
        assert sim.outcome_count == n
        outcomes = sim.sample(100)
        assert outcomes.shape == (100, n)


@given(sim=simulation_strategy(min_qubits=1, max_qubits=5, max_instructions=10))
def test_shape_matches_circuit(sim):
    """Property: sample() output shape matches (shots, outcome_count)."""
    outcomes = sim.sample(10)
    assert outcomes.shape == (10, sim.outcome_count)


@given(sim=simulation_strategy(min_qubits=1, max_qubits=5, max_instructions=10))
def test_repr_returns_string(sim):
    """Property: __repr__ returns a string."""
    assert isinstance(repr(sim), str)
