from paulimer import DensePauli, SparsePauli, indexed_anticommutators_of


def test_known_anticommutations():
    x = DensePauli("X")
    z = DensePauli("Z")
    y = DensePauli("Y")
    i = DensePauli("I")
    result = indexed_anticommutators_of(x, [x, z, y, i])
    assert sorted(result) == [1, 2]


def test_all_commute():
    x = DensePauli("X")
    result = indexed_anticommutators_of(x, [x, DensePauli("I")])
    assert result == []


def test_empty_paulis():
    x = DensePauli("X")
    result = indexed_anticommutators_of(x, [])
    assert result == []


def test_sparse_pauli_input():
    x = SparsePauli("X_0")
    z = SparsePauli("Z_0")
    i = SparsePauli("I")
    result = indexed_anticommutators_of(x, [z, i, x])
    assert result == [0]


def test_mixed_input_types():
    x_dense = DensePauli("X")
    z_sparse = SparsePauli("Z_0")
    result = indexed_anticommutators_of(x_dense, [z_sparse])
    assert result == [0]
