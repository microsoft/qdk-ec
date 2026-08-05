import pytest
import random
from binar import BitMatrix, BitVector
from hypothesis import given, strategies as st
from paulimer import (
    CliffordUnitary,
    SparsePauli,
    UnitaryOpcode,
    OutcomeCompleteSimulation,
    OutcomeFreeSimulation,
    OutcomeSpecificSimulation,
    PhasedCircuitAction,
    PhasedOutcomeCompleteSimulation,
)

SIMULATION_CLASSES = [
    OutcomeCompleteSimulation,
    OutcomeFreeSimulation,
    OutcomeSpecificSimulation,
    PhasedOutcomeCompleteSimulation,
]

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
        assert sim.random_outcome_capacity >= 5


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
    def test_random_outcome_count_is_int(self, sim_class):
        sim = sim_class(3)
        assert isinstance(sim.random_outcome_count, int)

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_random_outcome_capacity_is_int(self, sim_class):
        sim = sim_class(3)
        assert isinstance(sim.random_outcome_capacity, int)

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_random_bit_count_is_int(self, sim_class):
        sim = sim_class(3)
        assert isinstance(sim.random_bit_count, int)

    @pytest.mark.parametrize("sim_class", SIMULATION_CLASSES)
    def test_random_outcome_indicator_is_bitvector(self, sim_class):
        sim = sim_class(3)
        assert isinstance(sim.random_outcome_indicator, BitVector)


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
        assert sim.random_outcome_capacity >= 10


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

    def test_with_zero_outcomes(self):
        sim = OutcomeSpecificSimulation.with_zero_outcomes(3)
        assert sim.qubit_count == 3

    def test_new_with_seeded_random_outcomes(self):
        sim = OutcomeSpecificSimulation.new_with_seeded_random_outcomes(3, seed=42)
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
        assert sim.random_outcome_capacity >= 5

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
        assert sim.random_outcome_count == 1

    def test_output_phase_exponent(self):
        sim = PhasedOutcomeCompleteSimulation(1)
        sim.measure(SparsePauli("X_0"))
        exponent = sim.output_phase_exponent([True])
        assert isinstance(exponent, int)
        assert 0 <= exponent < 8
        # The trivial assignment never contributes a phase.
        assert sim.output_phase_exponent([False]) == 0

def _choi_action(build_gadget, n=1):
    """Phased Choi action of a symbolic-rotation gadget on ``n`` system qubits.

    Bell-pairs every system qubit ``q`` in ``0..n`` with its reference ``q + n``, allocates
    one symbolic-angle random bit, applies the gadget to the system qubits, and returns the
    resulting :class:`PhasedCircuitAction`.
    """
    sim = PhasedOutcomeCompleteSimulation(2 * n)
    for q in range(n):
        sim.apply_unitary(UnitaryOpcode.PrepareBell, [q, q + n])
    angle = sim.allocate_symbolic_angle()
    build_gadget(sim, angle)
    return sim.phased_action(list(range(n)), list(range(n)))


class TestPhasedCircuitAction:

    def test_phased_action_returns_action(self):
        action = _choi_action(lambda sim, a: sim.apply_symbolic_pauli_exp(SparsePauli("Z_0"), a))
        assert isinstance(action, PhasedCircuitAction)

    def test_choi_state_stabilizers_are_sparse_paulis(self):
        action = _choi_action(lambda sim, a: sim.apply_symbolic_pauli_exp(SparsePauli("Z_0"), a))
        stabilizers = action.choi_state_stabilizers
        assert len(stabilizers) == 2
        assert all(isinstance(stabilizer, SparsePauli) for stabilizer in stabilizers)

    def test_entangling_rotation_equivalence(self):
        def zz_direct(sim, a):
            sim.apply_symbolic_pauli_exp(SparsePauli("Z_0 Z_1"), a)

        def zz_via_cnot(sim, a):
            sim.apply_unitary(UnitaryOpcode.ControlledX, [0, 1])
            sim.apply_symbolic_pauli_exp(SparsePauli("Z_1"), a)
            sim.apply_unitary(UnitaryOpcode.ControlledX, [0, 1])

        direct = _choi_action(zz_direct, n=2)
        via_cnot = _choi_action(zz_via_cnot, n=2)
        assert direct.is_equivalent(via_cnot)
        assert via_cnot.is_equivalent(direct)

    def test_dropping_conjugation_breaks_equivalence(self):
        direct = _choi_action(lambda sim, a: sim.apply_symbolic_pauli_exp(SparsePauli("Z_0 Z_1"), a), n=2)
        bare = _choi_action(lambda sim, a: sim.apply_symbolic_pauli_exp(SparsePauli("Z_1"), a), n=2)
        assert not direct.is_equivalent(bare)

    def test_opposite_signs_distinguished_only_by_phase(self):
        positive = _choi_action(lambda sim, a: sim.apply_symbolic_pauli_exp(SparsePauli("Z_0"), a))
        negative = _choi_action(lambda sim, a: sim.apply_symbolic_pauli_exp(SparsePauli("-Z_0"), a))
        assert positive.is_equivalent_up_to_signs(negative)
        assert not positive.is_equivalent(negative)

    def test_action_is_self_equivalent(self):
        action = _choi_action(lambda sim, a: sim.apply_symbolic_pauli_exp(SparsePauli("Z_0 Z_1"), a), n=2)
        assert action.is_equivalent(action)

    def test_global_phase_is_zeta8_exponent(self):
        action = _choi_action(lambda sim, a: sim.apply_symbolic_pauli_exp(SparsePauli("Z_0"), a))
        assert isinstance(action.global_phase, int)
        assert 0 <= action.global_phase < 8

    def test_global_phase_distinguishes_y_from_xz(self):
        def apply_pauli(pauli):
            sim = PhasedOutcomeCompleteSimulation(2)
            sim.apply_unitary(UnitaryOpcode.PrepareBell, [0, 1])
            for term in pauli:
                sim.apply_pauli(SparsePauli(term))
            return sim.phased_action([0], [0])

        y_action = apply_pauli(["Y_0"])
        xz_action = apply_pauli(["X_0", "Z_0"])
        assert y_action.is_equivalent(xz_action)
        assert not y_action.is_equivalent_with_global_phase(xz_action)
        assert y_action.is_equivalent_with_global_phase(y_action)


def _direct_z_action(n, angle_supports):
    """Phased action of symbolic Z-rotations applied directly to ``n`` system qubits.

    Layout: system qubits ``0..n``, reference qubits ``n..2n``. Each entry of
    ``angle_supports`` is a list of system-qubit indices naming one symbolic rotation
    ``e^{i alpha Z...}`` about the tensor product of ``Z`` on those qubits.
    """
    sim = PhasedOutcomeCompleteSimulation(2 * n)
    for q in range(n):
        sim.apply_unitary(UnitaryOpcode.PrepareBell, [q, n + q])
    for support in angle_supports:
        pauli = SparsePauli(" ".join(f"Z_{q}" for q in support))
        angle = sim.allocate_symbolic_angle()
        sim.apply_symbolic_pauli_exp(pauli, angle)
    return sim.phased_action(list(range(n)), list(range(n)))


def _z_ejection_action(n, angle_supports):
    """Phased action of the same symbolic Z-rotations executed remotely via ejection.

    Layout: system qubits ``0..n``, reference qubits ``n..2n``, ancillas ``2n..3n``. The
    system qubits drive transversal CNOTs onto the ancillas, the symbolic rotations act on
    the ancillas, each ancilla is measured in the X basis (a *true* random bit), and a
    ``-`` outcome triggers a conditional Z correction on the system qubit. Mixing the virtual
    angle bits with the true measurement bits, the action must equal :func:`_direct_z_action`.
    """
    sim = PhasedOutcomeCompleteSimulation(3 * n)
    for q in range(n):
        sim.apply_unitary(UnitaryOpcode.PrepareBell, [q, n + q])
    for q in range(n):
        sim.apply_unitary(UnitaryOpcode.ControlledX, [q, 2 * n + q])
    for support in angle_supports:
        pauli = SparsePauli(" ".join(f"Z_{2 * n + q}" for q in support))
        angle = sim.allocate_symbolic_angle()
        sim.apply_symbolic_pauli_exp(pauli, angle)
    for q in range(n):
        outcome = sim.measure(SparsePauli(f"X_{2 * n + q}"))
        sim.apply_conditional_pauli(SparsePauli(f"Z_{q}"), [outcome])
    return sim.phased_action(list(range(n)), list(range(n)))


class TestPhasedEjection:
    """Measurement-based "ejection" of a Z-diagonal channel, mixing virtual and true bits."""

    @pytest.mark.parametrize(
        "n, angle_supports",
        [
            (1, [[0]]),
            (2, [[0]]),
            (2, [[0, 1]]),
            (2, [[0], [1], [0, 1]]),
        ],
    )
    def test_ejection_matches_direct(self, n, angle_supports):
        direct = _direct_z_action(n, angle_supports)
        ejection = _z_ejection_action(n, angle_supports)
        assert direct.is_equivalent(ejection)
        assert ejection.is_equivalent(direct)

def _distinct_z_supports(n):
    """All non-trivial Z-product supports on ``n`` qubits (distinct, independent)."""
    return [[bit for bit in range(n) if mask & (1 << bit)] for mask in range(1, 1 << n)]


def _signed_direct_z_action(n, supports, signs):
    """:func:`_direct_z_action` where the ``k``-th rotation is negated iff ``signs[k]``."""
    sim = PhasedOutcomeCompleteSimulation(2 * n)
    for q in range(n):
        sim.apply_unitary(UnitaryOpcode.PrepareBell, [q, n + q])
    for support, negate in zip(supports, signs):
        body = " ".join(f"Z_{q}" for q in support)
        pauli = SparsePauli(("-" if negate else "") + body)
        sim.apply_symbolic_pauli_exp(pauli, sim.allocate_symbolic_angle())
    return sim.phased_action(list(range(n)), list(range(n)))


def _permuted_direct_z_action(n, supports, perm):
    """:func:`_direct_z_action` where allocated angle ``k`` drives ``supports[perm[k]]``."""
    sim = PhasedOutcomeCompleteSimulation(2 * n)
    for q in range(n):
        sim.apply_unitary(UnitaryOpcode.PrepareBell, [q, n + q])
    for target in perm:
        pauli = SparsePauli(" ".join(f"Z_{q}" for q in supports[target]))
        sim.apply_symbolic_pauli_exp(pauli, sim.allocate_symbolic_angle())
    return sim.phased_action(list(range(n)), list(range(n)))


class TestPhasedNegativeChecks:
    """Sign flips and angle permutations must break an otherwise exact phased equivalence."""

    @given(
        n=st.integers(min_value=2, max_value=3),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    def test_sign_flip_is_pure_relative_phase(self, n, seed):
        supports = _distinct_z_supports(n)
        rng = random.Random(seed)
        signs = [rng.random() < 0.5 for _ in supports]
        baseline = _signed_direct_z_action(n, supports, [False] * len(supports))
        flipped = _signed_direct_z_action(n, supports, signs)
        assert baseline.is_equivalent_up_to_signs(flipped)
        assert baseline.is_equivalent(flipped) == (not any(signs))

    @given(
        n=st.integers(min_value=2, max_value=3),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    def test_permuted_angles_match_iff_identity(self, n, seed):
        supports = _distinct_z_supports(n)
        identity = list(range(len(supports)))
        perm = identity[:]
        random.Random(seed).shuffle(perm)
        base = _permuted_direct_z_action(n, supports, identity)
        permuted = _permuted_direct_z_action(n, supports, perm)
        assert base.is_equivalent(permuted) == (perm == identity)
