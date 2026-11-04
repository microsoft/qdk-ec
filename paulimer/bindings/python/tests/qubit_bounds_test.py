import pytest
from paulimer import CliffordUnitary, DensePauli, SparsePauli, UnitaryOpcode, encoding_clifford_of


@pytest.mark.parametrize("label", [1, 512])
def test_dense_conversion_rejects_out_of_range_support(label: int) -> None:
    with pytest.raises(ValueError, match="out of bounds"):
        DensePauli.from_sparse(SparsePauli({label: "X"}), 1)


def test_dense_axis_constructors_reject_out_of_range_positions() -> None:
    for construct in [DensePauli.x, DensePauli.y, DensePauli.z]:
        with pytest.raises(ValueError, match="out of bounds"):
            construct(1, 1)


@pytest.mark.parametrize("method", ["image_of", "preimage_of", "left_mul_pauli", "left_mul_pauli_exp"])
def test_clifford_pauli_operations_reject_out_of_range_support(method: str) -> None:
    clifford = CliffordUnitary.identity(1)
    with pytest.raises(ValueError, match="out of bounds"):
        getattr(clifford, method)(SparsePauli("X_512"))
    assert clifford.is_identity


def test_controlled_pauli_rejects_either_out_of_range_operand() -> None:
    clifford = CliffordUnitary.identity(2)
    valid = SparsePauli("Z_0")
    invalid = DensePauli.x(512, 513)
    for invalid_control in [True, False]:
        control = invalid if invalid_control else valid
        target = valid if invalid_control else invalid
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


def test_clifford_basis_access_rejects_out_of_range_positions() -> None:
    clifford = CliffordUnitary.identity(1)
    for method in [clifford.image_x, clifford.image_z, clifford.preimage_x, clifford.preimage_z]:
        with pytest.raises(ValueError, match="out of bounds"):
            method(512)


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


def test_clifford_construction_validates_inferred_extent() -> None:
    operators = [DensePauli.x(512, 513), DensePauli.z(512, 513)]
    for construct in [CliffordUnitary.from_images, CliffordUnitary.from_preimages]:
        with pytest.raises(ValueError, match="out of bounds"):
            construct(operators)
