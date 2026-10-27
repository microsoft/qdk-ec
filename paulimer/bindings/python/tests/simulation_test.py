from pathlib import Path

import pytest
from binar import BitMatrix, BitVector

from paulimer import (
    CliffordUnitary,
    OutcomeCompleteSimulation,
    OutcomeFreeSimulation,
    OutcomeSpecificSimulation,
    SparsePauli,
    UnitaryOpcode,
)

SIMULATION_CLASSES = [
    OutcomeCompleteSimulation,
    OutcomeFreeSimulation,
    OutcomeSpecificSimulation,
]


class TestSimulationConstruction:

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_default_construction(self, sim_class):
        sim = sim_class()
        assert isinstance(sim.qubit_count, int)

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_construction_with_qubit_count(self, sim_class):
        sim = sim_class(5)
        assert isinstance(sim.qubit_count, int)
        assert sim.qubit_capacity >= 5

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_with_capacity(self, sim_class):
        sim = sim_class.with_capacity(3, 10, 5)
        assert isinstance(sim.qubit_count, int)
        assert sim.outcome_capacity >= 10
        assert sim.branch_bit_capacity >= 5


class TestSimulationProperties:

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_qubit_count_is_int(self, sim_class):
        sim = sim_class(3)
        assert isinstance(sim.qubit_count, int)
        assert sim.qubit_count == 3

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_qubit_capacity_is_int(self, sim_class):
        sim = sim_class(3)
        assert isinstance(sim.qubit_capacity, int)
        assert sim.qubit_capacity >= 3

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_outcome_count_is_int(self, sim_class):
        sim = sim_class(3)
        assert isinstance(sim.outcome_count, int)

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_outcome_capacity_is_int(self, sim_class):
        sim = sim_class(3)
        assert isinstance(sim.outcome_capacity, int)

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_branch_bit_count_is_int(self, sim_class):
        sim = sim_class(3)
        assert isinstance(sim.branch_bit_count, int)

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_branch_bit_capacity_is_int(self, sim_class):
        sim = sim_class(3)
        assert isinstance(sim.branch_bit_capacity, int)

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_outcome_introduces_branch_bit_is_bitvector(self, sim_class):
        sim = sim_class(3)
        assert isinstance(sim.outcome_introduces_branch_bit, BitVector)

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_branch_bits_and_outcomes_have_distinct_counts(self, sim_class):
        sim = sim_class(1)
        assert sim.measure(SparsePauli("Z")) == 0
        assert sim.allocate_random_bit() == 1
        assert sim.outcome_count == 2
        assert sim.branch_bit_count == 1
        assert list(sim.outcome_introduces_branch_bit) == [False, True]
        with pytest.warns(DeprecationWarning, match="branch_bit_count") as warnings:
            assert sim.random_bit_count == sim.branch_bit_count
        assert Path(warnings[0].filename) == Path(__file__)
        with pytest.warns(DeprecationWarning, match="branch_bit_count"):
            assert sim.random_outcome_count == sim.branch_bit_count
        with pytest.warns(DeprecationWarning, match="branch_bit_capacity"):
            assert sim.random_outcome_capacity == sim.branch_bit_capacity
        with pytest.warns(DeprecationWarning, match="outcome_introduces_branch_bit"):
            assert sim.random_outcome_indicator == sim.outcome_introduces_branch_bit

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_released_capacity_keywords_remain_supported(self, sim_class):
        sim = sim_class.with_capacity(num_qubits=1, num_outcomes=4, num_random_outcomes=2)
        sim.reserve_outcomes(new_outcome_capacity=8, new_random_outcome_capacity=4)
        positional = sim_class.with_capacity(1, 4, 2)
        positional.reserve_outcomes(8, 4)
        assert sim.outcome_capacity == positional.outcome_capacity
        assert sim.branch_bit_capacity == positional.branch_bit_capacity

    def test_zero_samples_preserve_deterministic_negative_outcome(self):
        sim = OutcomeSpecificSimulation.with_zero_samples(1)
        assert sim.measure(SparsePauli("-Z")) == 0
        assert sim.allocate_random_bit() == 1
        assert list(sim.outcome_vector) == [True, False]

    def test_deprecated_sample_constructors_preserve_behavior(self):
        with pytest.warns(DeprecationWarning, match="with_zero_samples") as warnings:
            zero = OutcomeSpecificSimulation.with_zero_outcomes(1)
        assert Path(warnings[0].filename) == Path(__file__)
        zero.measure(SparsePauli("-Z"))
        zero.allocate_random_bit()
        assert list(zero.outcome_vector) == [True, False]
        with pytest.warns(DeprecationWarning, match="with_seeded_random_samples"):
            old = OutcomeSpecificSimulation.new_with_seeded_random_outcomes(1, seed=42)
        new = OutcomeSpecificSimulation.with_seeded_random_samples(1, seed=42)
        for _ in range(16):
            old.allocate_random_bit()
            new.allocate_random_bit()
        assert old.outcome_vector == new.outcome_vector


class TestSimulationOperations:

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_apply_unitary_hadamard(self, sim_class):
        sim = sim_class(1)
        sim.apply_unitary(UnitaryOpcode.Hadamard, [0])

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_apply_unitary_cnot(self, sim_class):
        sim = sim_class(2)
        sim.apply_unitary(UnitaryOpcode.ControlledX, [0, 1])

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_apply_pauli_exp(self, sim_class):
        sim = sim_class(2)
        observable = SparsePauli("XY")
        sim.apply_pauli_exp(observable)

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_apply_pauli_without_control(self, sim_class):
        sim = sim_class(2)
        observable = SparsePauli("XY")
        sim.apply_pauli(observable)

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_apply_pauli_with_control(self, sim_class):
        sim = sim_class(2)
        observable = SparsePauli("IX")  # IX and ZI commute
        control = SparsePauli("ZI")
        sim.apply_pauli(observable, controlled_by=control)

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_apply_conditional_pauli(self, sim_class):
        sim = sim_class(2)
        sim.measure(SparsePauli("ZI"))
        observable = SparsePauli("IX")
        sim.apply_conditional_pauli(observable, outcomes=[0], parity=False)

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_apply_permutation_with_support(self, sim_class):
        sim = sim_class(3)
        sim.apply_permutation([1, 0], supported_by=[0, 1])

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_apply_permutation_full(self, sim_class):
        sim = sim_class(3)
        sim.apply_permutation([2, 0, 1])

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_apply_clifford_with_support(self, sim_class):
        sim = sim_class(3)
        hadamard = CliffordUnitary.from_name("Hadamard", [0], 1)
        sim.apply_clifford(hadamard, supported_by=[1])

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_apply_clifford_full(self, sim_class):
        sim = sim_class(2)
        cnot = CliffordUnitary.from_name("ControlledX", [0, 1], 2)
        sim.apply_clifford(cnot)


class TestSimulationMeasurement:

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_measure_returns_int(self, sim_class):
        sim = sim_class(1)
        outcome_id = sim.measure(SparsePauli("Z"))
        assert isinstance(outcome_id, int)

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_measure_increments_outcome_count(self, sim_class):
        sim = sim_class(2)
        initial_count = sim.outcome_count
        sim.measure(SparsePauli("ZI"))
        assert sim.outcome_count == initial_count + 1

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_measure_with_hint(self, sim_class):
        sim = sim_class(2)
        sim.apply_unitary(UnitaryOpcode.Hadamard, [0])
        observable = SparsePauli("ZI")
        hint = SparsePauli("XI")
        outcome_id = sim.measure(observable, hint=hint)
        assert isinstance(outcome_id, int)

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_is_stabilizer_returns_bool(self, sim_class):
        sim = sim_class(1)
        result = sim.is_stabilizer(SparsePauli("Z"))
        assert isinstance(result, bool)

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_is_stabilizer_with_ignore_sign(self, sim_class):
        sim = sim_class(1)
        result = sim.is_stabilizer(SparsePauli("Z"), ignore_sign=True)
        assert isinstance(result, bool)

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_initial_state_stabilized_by_z(self, sim_class):
        sim = sim_class(1)
        assert sim.is_stabilizer(SparsePauli("Z")) is True


class TestSimulationCapacity:

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_allocate_random_bit_returns_int(self, sim_class):
        sim = sim_class(1)
        bit_id = sim.allocate_random_bit()
        assert isinstance(bit_id, int)

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_reserve_qubits(self, sim_class):
        sim = sim_class(2)
        sim.reserve_qubits(10)
        assert sim.qubit_capacity >= 10

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_reserve_outcomes(self, sim_class):
        sim = sim_class(2)
        sim.reserve_outcomes(20, 10)
        assert sim.outcome_capacity >= 20
        assert sim.branch_bit_capacity >= 10


class TestOutcomeCompleteSimulationSpecific:

    def test_clifford_returns_clifford_unitary(self):
        sim = OutcomeCompleteSimulation(2)
        clifford = sim.clifford
        assert isinstance(clifford, CliffordUnitary)

    def test_sign_matrix_returns_bit_matrix(self):
        sim = OutcomeCompleteSimulation(2)
        matrix = sim.sign_matrix
        assert isinstance(matrix, BitMatrix)

    def test_outcome_matrix_returns_bit_matrix(self):
        sim = OutcomeCompleteSimulation(2)
        matrix = sim.outcome_matrix
        assert isinstance(matrix, BitMatrix)

    def test_outcome_shift_returns_bit_vector(self):
        sim = OutcomeCompleteSimulation(2)
        shift = sim.outcome_shift
        assert isinstance(shift, BitVector)


class TestOutcomeFreeSimulationSpecific:

    def test_clifford_returns_clifford_unitary(self):
        sim = OutcomeFreeSimulation(2)
        clifford = sim.clifford
        assert isinstance(clifford, CliffordUnitary)


class TestOutcomeSpecificSimulationSpecific:

    def test_clifford_returns_clifford_unitary(self):
        sim = OutcomeSpecificSimulation(2)
        clifford = sim.clifford
        assert isinstance(clifford, CliffordUnitary)

    def test_outcome_vector_returns_bit_vector(self):
        sim = OutcomeSpecificSimulation(2)
        sim.measure(SparsePauli("ZI"))
        vector = sim.outcome_vector
        assert isinstance(vector, BitVector)

    def test_with_zero_samples(self):
        sim = OutcomeSpecificSimulation.with_zero_samples(3)
        assert sim.qubit_count == 3

    def test_with_seeded_random_samples(self):
        sim = OutcomeSpecificSimulation.with_seeded_random_samples(3, seed=42)
        assert sim.qubit_count == 3
