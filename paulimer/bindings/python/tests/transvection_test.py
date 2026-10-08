"""Tests for the Clifford -> transvection decomposition bindings (arXiv:2102.11380).

The decomposition reproduces a Clifford's symplectic (conjugation) action with a linear number of
pi/4 Pauli exponents, ignoring Pauli-image signs and the global phase.
"""

import pytest
from binar import BitMatrix, rank
from hypothesis import given, settings
from hypothesis import strategies as st

from paulimer import CliffordUnitary, SparsePauli, UnitaryOpcode


def _rebuild_from_transvections(transvections, qubit_count):
    rebuilt = CliffordUnitary.identity(qubit_count)
    for pauli in transvections:
        rebuilt.left_mul_pauli_exp(pauli)
    return rebuilt


def _residue_rank(clifford):
    """The residue rank ``rank(I + F)``, computed independently of ``fixed_space()``."""
    residue = clifford.symplectic_matrix ^ BitMatrix.identity(2 * clifford.qubit_count)
    return rank(residue)


def _is_conjugation_fixed(clifford, pauli):
    image = SparsePauli.from_dense(clifford.image_of(pauli))
    return (image * pauli).weight == 0


def _assert_valid_decomposition(clifford):
    qubit_count = clifford.qubit_count
    transvections = clifford.to_transvections()

    rebuilt = _rebuild_from_transvections(transvections, qubit_count)
    assert rebuilt.is_valid
    assert rebuilt.symplectic_matrix == clifford.symplectic_matrix

    for pauli in transvections:
        assert pauli.weight > 0
        assert pauli.phase == 1, f"factors must be positive Hermitian Paulis, got {pauli}"

    lower_bound = _residue_rank(clifford)
    assert len(transvections) >= lower_bound
    assert len(transvections) <= 4 * qubit_count + 2


def test_identity_has_no_transvections():
    for qubit_count in range(5):
        identity = CliffordUnitary.identity(qubit_count)
        assert identity.to_transvections() == []
        assert len(identity.fixed_space()) == 2 * qubit_count


def test_invalid_transvection_tableau_is_value_error():
    with pytest.raises(ValueError, match="the Clifford tableau is invalid"):
        CliffordUnitary.zero(1).to_transvections()
    assert CliffordUnitary.identity(0).to_transvections() == []


def test_single_qubit_gate_lengths():
    s_gate = CliffordUnitary.from_name("SqrtZ", [0], 1)
    _assert_valid_decomposition(s_gate)
    assert len(s_gate.to_transvections()) == 1

    hadamard = CliffordUnitary.from_name("Hadamard", [0], 1)
    _assert_valid_decomposition(hadamard)
    assert len(hadamard.to_transvections()) == 1


def test_fixed_space_of_pauli_x_includes_z():
    clifford = CliffordUnitary.from_name("X", [0], 1)
    pauli_z = SparsePauli("Z_0")
    assert pauli_z in clifford.fixed_space()
    assert SparsePauli.from_dense(clifford.image_of(pauli_z)) == -pauli_z


def test_swap_hyperbolic_branch():
    swap = CliffordUnitary.from_name("Swap", [0, 1], 2)
    _assert_valid_decomposition(swap)
    assert _residue_rank(swap) == 2
    assert len(swap.to_transvections()) == 3
    assert len(swap.fixed_space()) == 2


def test_two_qubit_gates():
    for name in ("ControlledX", "ControlledZ"):
        clifford = CliffordUnitary.from_name(name, [0, 1], 2)
        _assert_valid_decomposition(clifford)


def test_fixed_space_generators_are_conjugation_fixed():
    clifford = CliffordUnitary.identity(3)
    clifford.left_mul(UnitaryOpcode.Hadamard, [0])
    clifford.left_mul(UnitaryOpcode.ControlledX, [0, 1])
    clifford.left_mul(UnitaryOpcode.SqrtZ, [2])
    fixed_space = clifford.fixed_space()
    assert all(_is_conjugation_fixed(clifford, pauli) for pauli in fixed_space)
    assert all(pauli.weight > 0 for pauli in fixed_space)
    assert all(pauli.phase == 1 for pauli in fixed_space)
    assert len(fixed_space) == 2 * clifford.qubit_count - _residue_rank(clifford)


def test_fixed_space_generators_of_a_y_axis_rotation_are_hermitian():
    clifford = CliffordUnitary.from_name("SqrtY", [0], 1)
    fixed_space = clifford.fixed_space()
    assert len(fixed_space) == 1
    assert _is_conjugation_fixed(clifford, fixed_space[0])
    assert fixed_space[0].phase == 1


_SINGLE_QUBIT_GATES = ["Hadamard", "SqrtZ", "SqrtX", "X", "Y", "Z"]
_TWO_QUBIT_GATES = ["ControlledX", "ControlledZ", "Swap"]


@st.composite
def _random_clifford(draw, max_qubits=5):
    qubit_count = draw(st.integers(min_value=1, max_value=max_qubits))
    gate_count = draw(st.integers(min_value=0, max_value=3 * qubit_count))
    clifford = CliffordUnitary.identity(qubit_count)
    for _ in range(gate_count):
        if qubit_count >= 2 and draw(st.booleans()):
            name = draw(st.sampled_from(_TWO_QUBIT_GATES))
            first = draw(st.integers(min_value=0, max_value=qubit_count - 1))
            second = draw(
                st.integers(min_value=0, max_value=qubit_count - 1).filter(lambda q: q != first)
            )
            clifford.left_mul(getattr(UnitaryOpcode, name), [first, second])
        else:
            name = draw(st.sampled_from(_SINGLE_QUBIT_GATES))
            qubit = draw(st.integers(min_value=0, max_value=qubit_count - 1))
            clifford.left_mul(getattr(UnitaryOpcode, name), [qubit])
    return clifford


@settings(max_examples=200)
@given(_random_clifford())
def test_random_cliffords_reproduce_symplectic_action(clifford):
    _assert_valid_decomposition(clifford)


@settings(max_examples=200)
@given(_random_clifford())
def test_random_fixed_spaces_are_conjugation_fixed(clifford):
    for generator in clifford.fixed_space():
        assert _is_conjugation_fixed(clifford, generator)
        assert generator.weight > 0
