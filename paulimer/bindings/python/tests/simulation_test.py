
from pathlib import Path

import pytest
from binar import BitMatrix, BitVector

from paulimer import (
    CliffordUnitary,
    OutcomeCompleteSimulation,
    OutcomeFreeSimulation,
    OutcomeSpecificSimulation,
    PhasedOutcomeCompleteSimulation,
    SparsePauli,
    UnitaryOpcode,
)

SIMULATION_CLASSES = [
    OutcomeCompleteSimulation,
    OutcomeFreeSimulation,
    OutcomeSpecificSimulation,
    PhasedOutcomeCompleteSimulation,
]


@pytest.mark.parametrize("simulation_class", SIMULATION_CLASSES)
def test_legacy_simulation_names_warn_and_capacity_keywords_work(simulation_class):
    simulation = simulation_class.with_capacity(
        num_qubits=1, num_outcomes=2, num_random_outcomes=1
    )
    simulation.reserve_outcomes(new_outcome_capacity=3, new_random_outcome_capacity=2)
    for old, new in [
        ("random_outcome_count", "branch_bit_count"),
        ("random_bit_count", "branch_bit_count"),
        ("random_outcome_capacity", "branch_bit_capacity"),
        ("random_outcome_indicator", "outcome_introduces_branch_bit"),
    ]:
        with pytest.warns(DeprecationWarning, match=f"{old} is deprecated") as warnings:
            assert getattr(simulation, old) == getattr(simulation, new)
        assert Path(warnings[0].filename).resolve() == Path(__file__).resolve()


def test_zero_samples_preserve_deterministic_negative_measurements():
    with pytest.warns(DeprecationWarning, match="with_zero_outcomes is deprecated"):
        simulation = OutcomeSpecificSimulation.with_zero_outcomes(1)
    outcome = simulation.measure(SparsePauli("-Z_0"))
    assert simulation.outcome_vector[outcome]
    with pytest.warns(DeprecationWarning, match="new_with_seeded_random_outcomes is deprecated"):
        seeded = OutcomeSpecificSimulation.new_with_seeded_random_outcomes(1, seed=42)
    assert seeded.qubit_count == 1
    assert OutcomeSpecificSimulation.with_random_samples(1).qubit_count == 1

# PhasedOutcomeCompleteSimulation tracks the exact global phase, which a phaseless CliffordUnitary
# does not determine, so apply_clifford raises NotImplementedError on it; exclude it from the
# apply_clifford tests below while keeping it in the shared list for everything else.
CLIFFORD_CAPABLE_SIMULATION_CLASSES = [
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
    def test_deprecated_random_bit_count_is_int(self, sim_class):
        sim = sim_class(3)
        with pytest.warns(DeprecationWarning, match="random_bit_count is deprecated"):
            assert isinstance(sim.random_bit_count, int)

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_outcome_introduces_branch_bit_is_bitvector(self, sim_class):
        sim = sim_class(3)
        assert isinstance(sim.outcome_introduces_branch_bit, BitVector)


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

    @pytest.mark.parametrize("sim_class", CLIFFORD_CAPABLE_SIMULATION_CLASSES)
    def test_apply_clifford_with_support(self, sim_class):
        sim = sim_class(3)
        hadamard = CliffordUnitary.from_name("Hadamard", [0], 1)
        sim.apply_clifford(hadamard, supported_by=[1])

    @pytest.mark.parametrize("sim_class", CLIFFORD_CAPABLE_SIMULATION_CLASSES)
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


class TestPhasedOutcomeCompleteSimulationSpecific:

    def test_default_construction(self):
        sim = PhasedOutcomeCompleteSimulation()
        assert isinstance(sim.qubit_count, int)

    def test_construction_with_qubit_count(self):
        sim = PhasedOutcomeCompleteSimulation(4)
        assert sim.qubit_count == 4
        assert sim.qubit_capacity >= 4

    def test_with_capacity(self):
        sim = PhasedOutcomeCompleteSimulation.with_capacity(3, 10, 5)
        assert sim.outcome_capacity >= 10
        assert sim.branch_bit_capacity >= 5

    def test_clifford_returns_clifford_unitary(self):
        sim = PhasedOutcomeCompleteSimulation(2)
        assert isinstance(sim.clifford, CliffordUnitary)

    def test_phase_matrices_return_expected_types(self):
        sim = PhasedOutcomeCompleteSimulation(2)
        assert isinstance(sim.sign_matrix, BitMatrix)
        assert isinstance(sim.quadratic_phase_matrix, BitMatrix)
        assert isinstance(sim.outcome_matrix, BitMatrix)
        assert isinstance(sim.outcome_shift, BitVector)
        assert isinstance(sim.linear_i_phase, BitVector)
        assert isinstance(sim.linear_sign_phase, BitVector)

    def test_operations_and_measurement(self):
        sim = PhasedOutcomeCompleteSimulation(2)
        sim.apply_unitary(UnitaryOpcode.Hadamard, [0])
        sim.apply_unitary(UnitaryOpcode.ControlledX, [0, 1])
        outcome = sim.measure(SparsePauli("X_0"))
        assert isinstance(outcome, int)
        assert sim.branch_bit_count == 1

    def test_output_phase_exponent(self):
        sim = PhasedOutcomeCompleteSimulation(1)
        sim.measure(SparsePauli("X_0"))
        exponent = sim.output_phase_exponent([True])
        assert isinstance(exponent, int)
        assert 0 <= exponent < 8
        # The trivial assignment never contributes a phase.
        assert sim.output_phase_exponent([False]) == 0
