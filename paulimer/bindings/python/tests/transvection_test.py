"""Tests for the Clifford -> transvection decomposition bindings (arXiv:2102.11380).

The greedy list reproduces the unsigned action. The minimal list plus its final Pauli reproduces
the signed tableau. Neither API reconstructs global phase.
"""

import json
import sys
import threading
from pathlib import Path

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


def _assert_valid_minimal_decomposition(clifford):
    qubit_count = clifford.qubit_count
    transvections, correction = clifford.to_transvections_minimal()
    assert isinstance(transvections, list)
    assert isinstance(correction, SparsePauli)
    assert correction.phase == 1

    rebuilt = _rebuild_from_transvections(transvections, qubit_count)
    assert rebuilt.is_valid
    assert rebuilt.symplectic_matrix == clifford.symplectic_matrix
    rebuilt.left_mul_pauli(correction)
    assert rebuilt == clifford

    for pauli in transvections:
        assert pauli.weight > 0
        assert pauli.phase == 1, f"factors must be positive Hermitian Paulis, got {pauli}"

    rank = _residue_rank(clifford)
    assert len(transvections) in (rank, rank + 1)
    assert len(transvections) <= len(clifford.to_transvections())


def test_identity_has_no_transvections():
    for qubit_count in range(5):
        identity = CliffordUnitary.identity(qubit_count)
        assert identity.to_transvections() == []
        factors, correction = identity.to_transvections_minimal()
        assert factors == []
        assert correction.weight == 0
        assert correction.phase == 1
        _assert_valid_minimal_decomposition(identity)
        assert len(identity.fixed_space()) == 2 * qubit_count


@pytest.mark.parametrize("method", ["to_transvections", "to_transvections_minimal"])
def test_invalid_transvection_tableau_is_value_error(method):
    with pytest.raises(ValueError, match="the Clifford tableau is invalid"):
        getattr(CliffordUnitary.zero(1), method)()
    result = getattr(CliffordUnitary.identity(0), method)()
    assert result == ([] if method == "to_transvections" else ([], SparsePauli("I")))


def test_single_qubit_gate_lengths():
    s_gate = CliffordUnitary.from_name("SqrtZ", [0], 1)
    _assert_valid_decomposition(s_gate)
    _assert_valid_minimal_decomposition(s_gate)
    assert len(s_gate.to_transvections()) == 1
    assert len(s_gate.to_transvections_minimal()[0]) == 1

    hadamard = CliffordUnitary.from_name("Hadamard", [0], 1)
    _assert_valid_decomposition(hadamard)
    _assert_valid_minimal_decomposition(hadamard)
    assert len(hadamard.to_transvections()) == 1
    assert hadamard.to_transvections_minimal() == ([SparsePauli("Y")], SparsePauli("Z"))


def test_fixed_space_of_pauli_x_includes_z():
    clifford = CliffordUnitary.from_name("X", [0], 1)
    pauli_z = SparsePauli("Z_0")
    assert pauli_z in clifford.fixed_space()
    assert SparsePauli.from_dense(clifford.image_of(pauli_z)) == -pauli_z


def test_swap_hyperbolic_branch():
    swap = CliffordUnitary.from_name("Swap", [0, 1], 2)
    _assert_valid_decomposition(swap)
    _assert_valid_minimal_decomposition(swap)
    assert _residue_rank(swap) == 2
    assert len(swap.to_transvections()) == 3
    assert len(swap.to_transvections_minimal()[0]) == 3
    assert len(swap.fixed_space()) == 2


def test_two_qubit_gates():
    for name in ("ControlledX", "ControlledZ"):
        clifford = CliffordUnitary.from_name(name, [0, 1], 2)
        _assert_valid_decomposition(clifford)
        _assert_valid_minimal_decomposition(clifford)


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
                st.integers(min_value=0, max_value=qubit_count - 1).filter(
                    lambda qubit, excluded=first: qubit != excluded
                )
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
def test_random_cliffords_minimal_reproduce_signed_action(clifford):
    _assert_valid_minimal_decomposition(clifford)


@settings(max_examples=200)
@given(_random_clifford())
def test_random_fixed_spaces_are_conjugation_fixed(clifford):
    for generator in clifford.fixed_space():
        assert _is_conjugation_fixed(clifford, generator)
        assert generator.weight > 0


@pytest.mark.skipif(
    sys.platform == "emscripten",
    reason="threading is unavailable under Emscripten/Pyodide",
)
def test_minimal_decomposition_releases_the_gil():
    """Give the observer time to run on one CPU during a sub-second native call.

    The long switch interval prevents an ordinary Python switch before the call.
    """
    qubit_count = 7
    clifford = CliffordUnitary.identity(qubit_count)
    for first in range(0, 6, 2):
        second = first + 1
        for pauli in [
            SparsePauli(f"X_{first}"),
            SparsePauli(f"X_{second}"),
            SparsePauli(f"X_{first} X_{second}"),
            SparsePauli(f"Z_{first}"),
        ]:
            clifford.left_mul_pauli_exp(pauli)
    clifford.left_mul_pauli_exp(SparsePauli("Z_4 X_6"))

    ready = threading.Event()
    start = threading.Event()
    observed = threading.Event()

    def observer():
        ready.set()
        if start.wait(timeout=10):
            observed.set()

    interval = sys.getswitchinterval()
    sys.setswitchinterval(60)
    worker = threading.Thread(target=observer)
    try:
        worker.start()
        assert ready.wait(timeout=10), "the observer did not start"
        start.set()
        factors, correction = clifford.to_transvections_minimal()
        observed_during_call = observed.is_set()
    finally:
        sys.setswitchinterval(interval)
        start.set()
        worker.join(timeout=10)

    assert not worker.is_alive()
    assert len(factors) == 10
    assert observed_during_call, "the search held the GIL"
    rebuilt = _rebuild_from_transvections(factors, qubit_count)
    assert rebuilt.is_valid
    assert rebuilt.symplectic_matrix == clifford.symplectic_matrix
    rebuilt.left_mul_pauli(correction)
    assert rebuilt == clifford


@pytest.mark.parametrize("name", ["X", "Y", "Z"])
def test_pauli_needs_only_final_correction(name):
    clifford = CliffordUnitary.from_name(name, [0], 1)
    assert clifford.to_transvections() == []
    assert clifford.to_transvections_minimal() == ([], SparsePauli(name))
    _assert_valid_minimal_decomposition(clifford)


def test_every_single_qubit_signed_clifford_replays():
    reached = [CliffordUnitary.identity(1)]
    for clifford in reached:
        _assert_valid_minimal_decomposition(clifford)
        for gate in (UnitaryOpcode.Hadamard, UnitaryOpcode.SqrtZ):
            candidate = clifford * CliffordUnitary.identity(1)
            candidate.left_mul(gate, [0])
            if candidate not in reached:
                reached.append(candidate)
    assert len(reached) == 24


@settings(max_examples=100)
@given(_random_clifford(max_qubits=3), st.booleans())
def test_signed_y_rotation_and_final_y_replay(clifford, negative):
    pauli = SparsePauli("Y0")
    clifford.left_mul_pauli_exp(-pauli if negative else pauli)
    clifford.left_mul_pauli(pauli)
    _assert_valid_minimal_decomposition(clifford)


def test_transvection_notebook_signed_replay_and_outputs(capsys):
    path = Path(__file__).resolve().parents[1] / "examples" / "clifford-transvection-decomposition.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        exec(  # noqa: S102 - Execute trusted, checked-in notebook cells only.
            compile("".join(cell["source"]), f"{path}:{cell['id']}", "exec"),
            namespace,
        )
        expected = "".join(
            "".join(output["text"]) for output in cell["outputs"] if output.get("name") == "stdout"
        )
        assert capsys.readouterr().out == expected, cell["id"]
