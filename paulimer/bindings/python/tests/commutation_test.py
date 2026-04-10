from paulimer import DensePauli, SparsePauli


def test_dense_indexed_anti_commutators_of():
    x = DensePauli("X")
    z = DensePauli("Z")
    y = DensePauli("Y")
    i = DensePauli("I")
    result = x.indexed_anti_commutators_of([x, z, y, i])
    assert sorted(result) == [1, 2]


def test_dense_indexed_anti_commutators_of_all_commute():
    x = DensePauli("X")
    result = x.indexed_anti_commutators_of([x, DensePauli("I")])
    assert result == []


def test_dense_indexed_anti_commutators_of_empty():
    x = DensePauli("X")
    result = x.indexed_anti_commutators_of([])
    assert result == []


def test_dense_indexed_commutators_of():
    x = DensePauli("X")
    z = DensePauli("Z")
    y = DensePauli("Y")
    i = DensePauli("I")
    result = x.indexed_commutators_of([x, z, y, i])
    assert sorted(result) == [0, 3]


def test_dense_indexed_commutators_of_all_anticommute():
    x = DensePauli("X")
    z = DensePauli("Z")
    y = DensePauli("Y")
    result = x.indexed_commutators_of([z, y])
    assert result == []


def test_dense_indexed_commutators_of_empty():
    x = DensePauli("X")
    result = x.indexed_commutators_of([])
    assert result == []


def test_sparse_indexed_anti_commutators_of():
    x = SparsePauli("X_0")
    z = SparsePauli("Z_0")
    i = SparsePauli("I")
    result = x.indexed_anti_commutators_of([z, i, x])
    assert result == [0]


def test_sparse_indexed_commutators_of():
    x = SparsePauli("X_0")
    z = SparsePauli("Z_0")
    i = SparsePauli("I")
    result = x.indexed_commutators_of([z, i, x])
    assert sorted(result) == [1, 2]


def test_dense_with_sparse_args_indexed_commutators_of():
    x = DensePauli("X")
    z = SparsePauli("Z_0")
    i = SparsePauli("I")
    result = x.indexed_commutators_of([z, i])
    assert result == [1]


def test_dense_with_sparse_args_indexed_anti_commutators_of():
    x = DensePauli("X")
    z = SparsePauli("Z_0")
    i = SparsePauli("I")
    result = x.indexed_anti_commutators_of([z, i])
    assert result == [0]


def test_sparse_with_dense_args_indexed_commutators_of():
    x = SparsePauli("X_0")
    z = DensePauli("Z")
    i = DensePauli("I")
    result = x.indexed_commutators_of([z, i])
    assert result == [1]


def test_sparse_with_dense_args_indexed_anti_commutators_of():
    x = SparsePauli("X_0")
    z = DensePauli("Z")
    i = DensePauli("I")
    result = x.indexed_anti_commutators_of([z, i])
    assert result == [0]
