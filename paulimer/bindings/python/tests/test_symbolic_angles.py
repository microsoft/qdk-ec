import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from paulimer import (
    PhasedCircuitAction,
    PhasedOutcomeCompleteSimulation,
    SparsePauli,
    UnitaryOpcode,
)


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
