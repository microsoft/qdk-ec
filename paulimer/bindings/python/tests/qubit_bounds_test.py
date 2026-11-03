from typing import Literal

import pytest
from paulimer import CliffordUnitary, DensePauli, SparsePauli, UnitaryOpcode, encoding_clifford_of


@pytest.mark.parametrize(
    ("label", "qubit_count"),
    [(0, 0), (1, 1), (512, 1)],
)
@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_dense_constructors_reject_out_of_range_positions(
    label: int, qubit_count: int, axis: Literal["X", "Y", "Z"]
) -> None:
    with pytest.raises(ValueError, match="out of bounds"):
        getattr(DensePauli, axis.lower())(label, qubit_count)
    with pytest.raises(ValueError, match="out of bounds"):
        DensePauli.from_sparse(SparsePauli({label: axis}), qubit_count)


@pytest.mark.parametrize(
    ("text", "qubit_count"), [("Y_511", 512), ("-iZ_512", 513), ("iI", 0)]
)
def test_dense_conversion_preserves_valid_positions_and_phase(text: str, qubit_count: int) -> None:
    sparse = SparsePauli(text)
    dense = DensePauli.from_sparse(sparse, qubit_count)
    assert dense.size == qubit_count
    assert dense.support == sparse.support
    assert SparsePauli.from_dense(dense) == sparse


@pytest.mark.parametrize("dense_input", [False, True])
@pytest.mark.parametrize("method", ["image_of", "preimage_of", "left_mul_pauli", "left_mul_pauli_exp"])
def test_clifford_pauli_operations_reject_out_of_range_support(method: str, dense_input: bool) -> None:
    sparse = SparsePauli("X_512")
    pauli = DensePauli.from_sparse(sparse, 513) if dense_input else sparse
    clifford = CliffordUnitary.identity(1)
    with pytest.raises(ValueError, match="out of bounds"):
        getattr(clifford, method)(pauli)
    assert clifford.is_identity


@pytest.mark.parametrize("dense_input", [False, True])
@pytest.mark.parametrize("invalid_control", [False, True])
def test_controlled_pauli_rejects_either_out_of_range_operand(
    dense_input: bool, invalid_control: bool
) -> None:
    sparse_control = SparsePauli({512 if invalid_control else 0: "Z"})
    sparse_target = SparsePauli({1 if invalid_control else 512: "X"})
    control = DensePauli.from_sparse(sparse_control, 513) if dense_input else sparse_control
    target = DensePauli.from_sparse(sparse_target, 513) if dense_input else sparse_target
    clifford = CliffordUnitary.identity(2)
    with pytest.raises(ValueError, match="out of bounds"):
        clifford.left_mul_controlled_pauli(control, target)
    assert clifford.is_identity


def test_encoding_clifford_rejects_out_of_range_support() -> None:
    generators = [SparsePauli("Z_512")]
    for construct in [encoding_clifford_of, CliffordUnitary.group_encoding_clifford_of]:
        with pytest.raises(ValueError, match="out of bounds"):
            construct(generators, 1)


def test_clifford_operations_preserve_valid_support_with_trailing_identities() -> None:
    clifford = CliffordUnitary.identity(2)
    pauli = DensePauli("IXII")
    expected = SparsePauli("X_1")
    assert SparsePauli.from_dense(clifford.image_of(pauli)) == expected
    assert SparsePauli.from_dense(clifford.preimage_of(pauli)) == expected
    clifford.left_mul_pauli_exp(pauli)
    assert clifford.is_valid
    clifford.left_mul_pauli(pauli)
    assert clifford.is_valid


@pytest.mark.parametrize("method", ["image_x", "image_z", "preimage_x", "preimage_z"])
def test_clifford_basis_access_rejects_out_of_range_positions(method: str) -> None:
    clifford = CliffordUnitary.identity(1)
    with pytest.raises(ValueError, match="out of bounds"):
        getattr(clifford, method)(512)


def test_clifford_mutations_validate_all_support_before_mutating() -> None:
    operations = [
        ("left_mul", (UnitaryOpcode.ControlledX, [0, 512])),
        ("left_mul_clifford", (CliffordUnitary.identity(2), [0, 512])),
        ("left_mul_permutation", ([1, 0], [0, 512])),
    ]
    for method, arguments in operations:
        clifford = CliffordUnitary.identity(2)
        with pytest.raises(ValueError, match="out of bounds"):
            getattr(clifford, method)(*arguments)
        assert clifford.is_identity
    with pytest.raises(ValueError, match="out of bounds"):
        CliffordUnitary.from_name("ControlledX", [0, 512], 2)


@pytest.mark.parametrize("method", ["from_images", "from_preimages"])
def test_clifford_construction_validates_inferred_extent(method: str) -> None:
    operators = [DensePauli.x(512, 513), DensePauli.z(512, 513)]
    with pytest.raises(ValueError, match="out of bounds"):
        getattr(CliffordUnitary, method)(operators)

    clifford = getattr(CliffordUnitary, method)([DensePauli("XII"), DensePauli("ZII")])
    assert clifford.qubit_count == 1
    assert clifford.is_identity
