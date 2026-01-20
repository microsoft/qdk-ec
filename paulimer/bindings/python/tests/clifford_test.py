from paulimer import DensePauli, SparsePauli, CliffordUnitary, UnitaryOpcode
from paulimer import (
    is_diagonal_resource_encoder,
    unitary_from_diagonal_resource_state,
    split_qubit_cliffords_and_css,
    split_phased_css,
    encoding_clifford_of,
)


SINGLE_QUBIT_OPCODES = [
    UnitaryOpcode.I,
    UnitaryOpcode.X,
    UnitaryOpcode.Y,
    UnitaryOpcode.Z,
    UnitaryOpcode.SqrtX,
    UnitaryOpcode.SqrtXInv,
    UnitaryOpcode.SqrtY,
    UnitaryOpcode.SqrtYInv,
    UnitaryOpcode.SqrtZ,
    UnitaryOpcode.SqrtZInv,
    UnitaryOpcode.Hadamard,
]

SINGLE_QUBIT_GATE_NAMES = [
    "I",
    "X",
    "Y",
    "Z",
    "SqrtX",
    "SqrtXInv",
    "SqrtY",
    "SqrtYInv",
    "SqrtZ",
    "SqrtZInv",
    "Hadamard",
]

TWO_QUBIT_OPCODES = [
    UnitaryOpcode.Swap,
    UnitaryOpcode.ControlledX,
    UnitaryOpcode.ControlledZ,
    UnitaryOpcode.PrepareBell,
]

TWO_QUBIT_GATE_NAMES = [
    "Swap",
    "ControlledX",
    "ControlledZ",
    "PrepareBell",
]


def test_hadamard_from_preimages():
    pauli_x = DensePauli("X")
    pauli_z = DensePauli("Z")

    hadamard = CliffordUnitary.from_preimages([pauli_z, pauli_x])

    assert hadamard.is_valid
    assert not hadamard.is_identity
    assert hadamard.qubit_count == 1
    assert hadamard.preimage_of(pauli_x) == pauli_z
    assert hadamard.preimage_of(pauli_z) == pauli_x
    assert (hadamard * hadamard).is_identity


def test_from_images_creates_correct_clifford():
    image_of_x = DensePauli("Z")
    image_of_z = DensePauli("X")

    hadamard = CliffordUnitary.from_images([image_of_x, image_of_z])

    assert hadamard.is_valid
    assert hadamard.image_x(0) == image_of_x
    assert hadamard.image_z(0) == image_of_z


def test_from_string_parses_clifford():
    hadamard = CliffordUnitary.from_string("Z0:X, X0:Z")

    assert hadamard.is_valid
    assert hadamard.qubit_count == 1
    assert hadamard.image_x(0) == DensePauli("Z")
    assert hadamard.image_z(0) == DensePauli("X")


def test_from_name_all_single_qubit_gates():
    for gate_name in SINGLE_QUBIT_GATE_NAMES:
        clifford = CliffordUnitary.from_name(gate_name, [0], 1)
        assert clifford.is_valid, f"{gate_name} should produce valid Clifford"
        assert clifford.qubit_count == 1


def test_from_name_all_two_qubit_gates():
    for gate_name in TWO_QUBIT_GATE_NAMES:
        clifford = CliffordUnitary.from_name(gate_name, [0, 1], 2)
        assert clifford.is_valid, f"{gate_name} should produce valid Clifford"
        assert clifford.qubit_count == 2


def test_from_name_specific_gate_actions():
    hadamard = CliffordUnitary.from_name("Hadamard", [0], 1)
    assert hadamard.image_x(0) == DensePauli("Z")
    assert hadamard.image_z(0) == DensePauli("X")

    cnot = CliffordUnitary.from_name("ControlledX", [0, 1], 2)
    assert cnot.image_x(0) == DensePauli("XX")
    assert cnot.image_z(1) == DensePauli("ZZ")

    cz = CliffordUnitary.from_name("ControlledZ", [0, 1], 2)
    assert cz.image_x(0) == DensePauli("XZ")
    assert cz.image_x(1) == DensePauli("ZX")


def test_identity_has_trivial_action():
    identity = CliffordUnitary.identity(3)

    assert identity.is_valid
    assert identity.is_identity
    assert identity.qubit_count == 3
    assert identity.image_x(0) == DensePauli("XII")
    assert identity.image_z(0) == DensePauli("ZII")
    assert identity.image_x(1) == DensePauli("IXI")
    assert identity.image_z(2) == DensePauli("IIZ")


def test_composition_is_associative():
    clifford_a = CliffordUnitary.from_name("Hadamard", [0], 2)
    clifford_b = CliffordUnitary.from_name("ControlledX", [0, 1], 2)
    clifford_c = CliffordUnitary.from_name("Hadamard", [1], 2)

    left_grouped = (clifford_a * clifford_b) * clifford_c
    right_grouped = clifford_a * (clifford_b * clifford_c)

    for qubit in range(2):
        assert left_grouped.image_x(qubit) == right_grouped.image_x(qubit)
        assert left_grouped.image_z(qubit) == right_grouped.image_z(qubit)


def test_inverse_reverses_composition():
    hadamard = CliffordUnitary.from_name("Hadamard", [0], 1)
    sqrt_z = CliffordUnitary.from_name("SqrtZ", [0], 1)
    clifford = hadamard * sqrt_z

    composed_with_inverse = clifford * clifford.inverse()

    assert composed_with_inverse.is_identity


def test_double_inverse_is_identity():
    clifford = CliffordUnitary.from_name("SqrtZ", [0], 1)

    double_inverse = clifford.inverse().inverse()

    assert double_inverse.image_x(0) == clifford.image_x(0)
    assert double_inverse.image_z(0) == clifford.image_z(0)


def test_hadamard_squared_is_identity():
    hadamard = CliffordUnitary.from_name("Hadamard", [0], 1)

    result = hadamard * hadamard

    assert result.is_identity


def test_sqrt_z_fourth_power_is_identity():
    sqrt_z = CliffordUnitary.from_name("SqrtZ", [0], 1)

    result = sqrt_z * sqrt_z * sqrt_z * sqrt_z

    assert result.is_identity


def test_pow_positive_exponent():
    sqrt_z = CliffordUnitary.from_name("SqrtZ", [0], 1)

    # sqrt_z ** 4 should be identity
    result = sqrt_z ** 4
    assert result.is_identity

    # sqrt_z ** 2 should be Z
    result2 = sqrt_z ** 2
    z_gate = CliffordUnitary.from_name("Z", [0], 1)
    assert result2 == z_gate


def test_pow_zero_exponent():
    hadamard = CliffordUnitary.from_name("Hadamard", [0], 1)
    
    result = hadamard ** 0
    assert result.is_identity


def test_pow_negative_exponent():
    sqrt_z = CliffordUnitary.from_name("SqrtZ", [0], 1)
    sqrt_z_inv = CliffordUnitary.from_name("SqrtZInv", [0], 1)
    
    # sqrt_z ** -1 should equal sqrt_z_inv
    result = sqrt_z ** -1
    assert result == sqrt_z_inv
    
    # sqrt_z ** -2 should equal Z inverse = Z
    result2 = sqrt_z ** -2
    z_gate = CliffordUnitary.from_name("Z", [0], 1)
    assert result2 == z_gate


def test_pow_one_exponent():
    hadamard = CliffordUnitary.from_name("Hadamard", [0], 1)
    
    result = hadamard ** 1
    assert result == hadamard


def test_equality_same_clifford():
    hadamard1 = CliffordUnitary.from_name("Hadamard", [0], 1)
    hadamard2 = CliffordUnitary.from_name("Hadamard", [0], 1)
    
    assert hadamard1 == hadamard2
    assert not (hadamard1 != hadamard2)


def test_equality_different_cliffords():
    hadamard = CliffordUnitary.from_name("Hadamard", [0], 1)
    sqrt_z = CliffordUnitary.from_name("SqrtZ", [0], 1)
    
    assert hadamard != sqrt_z
    assert not (hadamard == sqrt_z)


def test_equality_identity():
    identity1 = CliffordUnitary.identity(3)
    identity2 = CliffordUnitary.identity(3)
    
    assert identity1 == identity2


def test_equality_composed_vs_direct():
    # H * H should equal identity
    hadamard = CliffordUnitary.from_name("Hadamard", [0], 1)
    composed = hadamard * hadamard
    identity = CliffordUnitary.identity(1)
    
    assert composed == identity


def test_left_mul_all_single_qubit_opcodes():
    for opcode in SINGLE_QUBIT_OPCODES:
        clifford = CliffordUnitary.identity(1)
        clifford.left_mul(opcode, [0])
        assert clifford.is_valid, f"left_mul with {opcode.name} should produce valid Clifford"


def test_left_mul_all_two_qubit_opcodes():
    for opcode in TWO_QUBIT_OPCODES:
        clifford = CliffordUnitary.identity(2)
        clifford.left_mul(opcode, [0, 1])
        assert clifford.is_valid, f"left_mul with {opcode.name} should produce valid Clifford"


def test_left_mul_builds_same_clifford_as_composition():
    composed = CliffordUnitary.identity(2)
    composed = CliffordUnitary.from_name("Hadamard", [0], 2) * composed
    composed = CliffordUnitary.from_name("ControlledX", [0, 1], 2) * composed

    mutated = CliffordUnitary.identity(2)
    mutated.left_mul(UnitaryOpcode.Hadamard, [0])
    mutated.left_mul(UnitaryOpcode.ControlledX, [0, 1])

    for qubit in range(2):
        assert mutated.image_x(qubit) == composed.image_x(qubit)
        assert mutated.image_z(qubit) == composed.image_z(qubit)


def test_left_mul_pauli_with_dense_pauli():
    clifford = CliffordUnitary.identity(2)
    pauli = DensePauli("XZ")

    clifford.left_mul_pauli(pauli)

    assert clifford.is_valid


def test_left_mul_pauli_with_sparse_pauli():
    clifford_dense = CliffordUnitary.identity(2)
    clifford_sparse = CliffordUnitary.identity(2)
    pauli_dense = DensePauli("XZ")
    pauli_sparse = SparsePauli("XZ")

    clifford_dense.left_mul_pauli(pauli_dense)
    clifford_sparse.left_mul_pauli(pauli_sparse)

    for qubit in range(2):
        assert clifford_dense.image_x(qubit) == clifford_sparse.image_x(qubit)
        assert clifford_dense.image_z(qubit) == clifford_sparse.image_z(qubit)


def test_left_mul_pauli_exp_with_dense_pauli():
    clifford = CliffordUnitary.identity(2)
    pauli = DensePauli("ZI")

    clifford.left_mul_pauli_exp(pauli)

    assert clifford.is_valid


def test_left_mul_pauli_exp_with_sparse_pauli():
    clifford_dense = CliffordUnitary.identity(2)
    clifford_sparse = CliffordUnitary.identity(2)
    pauli_dense = DensePauli("ZI")
    pauli_sparse = SparsePauli({0: "Z"})

    clifford_dense.left_mul_pauli_exp(pauli_dense)
    clifford_sparse.left_mul_pauli_exp(pauli_sparse)

    for qubit in range(2):
        assert clifford_dense.image_x(qubit) == clifford_sparse.image_x(qubit)
        assert clifford_dense.image_z(qubit) == clifford_sparse.image_z(qubit)


def test_left_mul_controlled_pauli_with_dense_paulis():
    clifford = CliffordUnitary.identity(2)
    control = DensePauli("ZI")
    target = DensePauli("IX")

    clifford.left_mul_controlled_pauli(control, target)

    assert clifford.is_valid
    assert not clifford.is_identity


def test_left_mul_controlled_pauli_with_sparse_paulis():
    clifford_dense = CliffordUnitary.identity(2)
    clifford_sparse = CliffordUnitary.identity(2)

    clifford_dense.left_mul_controlled_pauli(DensePauli("ZI"), DensePauli("IX"))
    clifford_sparse.left_mul_controlled_pauli(
        SparsePauli({0: "Z"}), SparsePauli({1: "X"})
    )

    for qubit in range(2):
        assert clifford_dense.image_x(qubit) == clifford_sparse.image_x(qubit)
        assert clifford_dense.image_z(qubit) == clifford_sparse.image_z(qubit)


def test_left_mul_controlled_pauli_with_mixed_types():
    clifford_1 = CliffordUnitary.identity(2)
    clifford_2 = CliffordUnitary.identity(2)

    clifford_1.left_mul_controlled_pauli(DensePauli("ZI"), SparsePauli({1: "X"}))
    clifford_2.left_mul_controlled_pauli(SparsePauli({0: "Z"}), DensePauli("IX"))

    for qubit in range(2):
        assert clifford_1.image_x(qubit) == clifford_2.image_x(qubit)
        assert clifford_1.image_z(qubit) == clifford_2.image_z(qubit)


def test_left_mul_clifford_is_equivalent_to_composition():
    target_clifford = CliffordUnitary.identity(2)
    hadamard = CliffordUnitary.from_name("Hadamard", [0], 1)

    target_clifford.left_mul_clifford(hadamard, [0])

    expected = CliffordUnitary.from_name("Hadamard", [0], 2)
    assert target_clifford.image_x(0) == expected.image_x(0)
    assert target_clifford.image_z(0) == expected.image_z(0)


def test_left_mul_permutation_swaps_qubits():
    clifford = CliffordUnitary.identity(2)

    clifford.left_mul_permutation([1, 0], [0, 1])

    assert clifford.is_valid
    assert clifford.image_x(0) == DensePauli("IX")
    assert clifford.image_x(1) == DensePauli("XI")


def test_preimage_of_accepts_dense_pauli():
    hadamard = CliffordUnitary.from_name("Hadamard", [0], 1)

    result = hadamard.preimage_of(DensePauli("X"))

    assert result == DensePauli("Z")


def test_preimage_of_accepts_sparse_pauli():
    hadamard = CliffordUnitary.from_name("Hadamard", [0], 1)

    result = hadamard.preimage_of(SparsePauli("X"))

    assert result == DensePauli("Z")


def test_image_of_accepts_dense_pauli():
    hadamard = CliffordUnitary.from_name("Hadamard", [0], 1)

    result = hadamard.image_of(DensePauli("X"))

    assert result == DensePauli("Z")


def test_image_of_accepts_sparse_pauli():
    hadamard = CliffordUnitary.from_name("Hadamard", [0], 1)

    result = hadamard.image_of(SparsePauli("X"))

    assert result == DensePauli("Z")


def test_image_and_preimage_are_inverses():
    clifford = CliffordUnitary.from_name("SqrtZ", [0], 2)
    clifford.left_mul(UnitaryOpcode.Hadamard, [1])

    original = DensePauli("XZ")
    image = clifford.image_of(original)
    recovered = clifford.preimage_of(image)

    assert abs(recovered) == abs(original)


def test_preimage_x_and_preimage_z_match_preimage_of():
    hadamard = CliffordUnitary.from_name("Hadamard", [0], 1)

    assert hadamard.preimage_x(0) == hadamard.preimage_of(DensePauli("X"))
    assert hadamard.preimage_z(0) == hadamard.preimage_of(DensePauli("Z"))


def test_image_x_and_image_z_match_image_of():
    hadamard = CliffordUnitary.from_name("Hadamard", [0], 1)

    assert hadamard.image_x(0) == hadamard.image_of(DensePauli("X"))
    assert hadamard.image_z(0) == hadamard.image_of(DensePauli("Z"))


def test_tensor_combines_qubit_counts():
    hadamard = CliffordUnitary.from_name("Hadamard", [0], 1)
    cnot = CliffordUnitary.from_name("ControlledX", [0, 1], 2)

    tensor_product = hadamard.tensor(cnot)

    assert tensor_product.qubit_count == 3
    assert tensor_product.is_valid


def test_tensor_acts_independently():
    hadamard_1 = CliffordUnitary.from_name("Hadamard", [0], 1)
    hadamard_2 = CliffordUnitary.from_name("Hadamard", [0], 1)

    tensor_product = hadamard_1.tensor(hadamard_2)

    assert tensor_product.image_x(0) == DensePauli("ZI")
    assert tensor_product.image_z(0) == DensePauli("XI")
    assert tensor_product.image_x(1) == DensePauli("IZ")
    assert tensor_product.image_z(1) == DensePauli("IX")


def test_is_css_for_various_cliffords():
    identity = CliffordUnitary.identity(2)
    assert identity.is_css

    cnot = CliffordUnitary.from_name("ControlledX", [0, 1], 2)
    assert cnot.is_css

    hadamard = CliffordUnitary.from_name("Hadamard", [0], 1)
    assert not hadamard.is_css


def test_is_diagonal_for_identity():
    identity = CliffordUnitary.identity(2)

    assert identity.is_diagonal("X")
    assert identity.is_diagonal("Z")


def test_is_diagonal_for_hadamard():
    hadamard = CliffordUnitary.from_name("Hadamard", [0], 1)

    assert not hadamard.is_diagonal("X")
    assert not hadamard.is_diagonal("Z")


def test_symplectic_matrix():
    clifford = CliffordUnitary.identity(2)
    matrix = clifford.symplectic_matrix
    import binar
    assert type(matrix) is binar.BitMatrix


def test_str_and_repr_return_strings():
    clifford = CliffordUnitary.identity(1)

    assert isinstance(str(clifford), str)
    assert len(str(clifford)) > 0
    assert isinstance(repr(clifford), str)
    assert len(repr(clifford)) > 0


def test_is_diagonal_resource_encoder_returns_bool():
    identity = CliffordUnitary.identity(2)

    result = is_diagonal_resource_encoder(identity, "Z")

    assert isinstance(result, bool)


def test_unitary_from_diagonal_resource_state_returns_optional():
    identity = CliffordUnitary.identity(2)

    result = unitary_from_diagonal_resource_state(identity, "Z")

    assert result is None or isinstance(result, CliffordUnitary)


def test_split_qubit_cliffords_and_css_decomposes_clifford():
    clifford = CliffordUnitary.identity(2)

    result = split_qubit_cliffords_and_css(clifford)

    if result is not None:
        left, right = result
        assert left.is_valid
        assert right.is_valid
        composed = left * right
        for qubit in range(2):
            assert composed.image_x(qubit) == clifford.image_x(qubit)
            assert composed.image_z(qubit) == clifford.image_z(qubit)


def test_split_phased_css_decomposes_clifford():
    clifford = CliffordUnitary.identity(2)

    result = split_phased_css(clifford)

    if result is not None:
        left, right = result
        assert left.is_valid
        assert right.is_valid
        composed = left * right
        for qubit in range(2):
            assert composed.image_x(qubit) == clifford.image_x(qubit)
            assert composed.image_z(qubit) == clifford.image_z(qubit)


def test_encoding_clifford_of_with_sparse_generators():
    generators = [SparsePauli("ZI"), SparsePauli("IZ")]

    result = encoding_clifford_of(generators, 2)

    assert isinstance(result, CliffordUnitary)
    assert result.is_valid
    assert result.qubit_count == 2


def test_encoding_clifford_of_with_dense_generators():
    generators = [DensePauli("ZI"), DensePauli("IZ")]

    result = encoding_clifford_of(generators, 2)

    assert isinstance(result, CliffordUnitary)
    assert result.is_valid


def test_encoding_clifford_of_with_mixed_generators():
    generators = [SparsePauli("ZI"), DensePauli("IZ")]

    result = encoding_clifford_of(generators, 2)

    assert isinstance(result, CliffordUnitary)
    assert result.is_valid


def test_pickle_roundtrip():
    import pickle

    hadamard = CliffordUnitary.from_string("Z0:X, X0:Z")

    serialized = pickle.dumps(hadamard)
    restored = pickle.loads(serialized)

    assert isinstance(restored, CliffordUnitary)
    assert restored.is_valid
    assert str(restored) == str(hadamard)
    assert restored.image_x(0) == hadamard.image_x(0)
    assert restored.image_z(0) == hadamard.image_z(0)


def test_pickle_larger_clifford():
    import pickle

    # Two-qubit CNOT
    cnot = CliffordUnitary.from_name("ControlledX", [0, 1], 2)

    serialized = pickle.dumps(cnot)
    restored = pickle.loads(serialized)

    assert isinstance(restored, CliffordUnitary)
    assert restored.is_valid
    assert restored.qubit_count == 2
    assert str(restored) == str(cnot)


def test_symplectic_matrix_type_identity():
    """Test that symplectic_matrix returns the same BitMatrix type as binar"""
