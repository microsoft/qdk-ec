from binar import BitMatrix, BitVector


def test_init_from_int_lists():
    matrix = BitMatrix([[1, 0, 1], [0, 1, 0]])
    assert matrix.row_count == 2
    assert matrix.column_count == 3
    assert matrix[0, 0] is True
    assert matrix[0, 1] is False
    assert matrix[1, 1] is True


def test_init_from_strings():
    matrix = BitMatrix(['101', '010', '110'])
    assert matrix.row_count == 3
    assert matrix.column_count == 3
    assert matrix[0, 0] is True
    assert matrix[0, 1] is False
    assert matrix[2, 2] is False


def test_init_from_bool_lists():
    matrix = BitMatrix([[True, False], [False, True]])
    assert matrix.row_count == 2
    assert matrix.column_count == 2
    assert matrix[0, 0] is True
    assert matrix[1, 1] is True


def test_init_from_bitvectors():
    v1 = BitVector('101')
    v2 = BitVector('010')
    matrix = BitMatrix([v1, v2])
    assert matrix.row_count == 2
    assert matrix.column_count == 3
    assert matrix[0, 0] is True
    assert matrix[0, 2] is True
    assert matrix[1, 1] is True


def test_init_from_mixed_types():
    v = BitVector('11')
    matrix = BitMatrix([v, '01', [True, False]])
    assert matrix.row_count == 3
    assert matrix.column_count == 2
    assert matrix[0, 0] is True
    assert matrix[0, 1] is True
    assert matrix[1, 0] is False
    assert matrix[2, 0] is True


def test_identity():
    matrix = BitMatrix.identity(3)
    assert matrix.row_count == 3
    assert matrix.column_count == 3


def test_zeros():
    matrix = BitMatrix.zeros(2, 5)
    assert matrix.row_count == 2
    assert matrix.column_count == 5


def test_ones():
    matrix = BitMatrix.ones(4, 3)
    assert matrix.row_count == 4
    assert matrix.column_count == 3


def test_row_count():
    matrix = BitMatrix.zeros(7, 3)
    assert isinstance(matrix.row_count, int)
    assert matrix.row_count == 7


def test_column_count():
    matrix = BitMatrix.zeros(2, 9)
    assert isinstance(matrix.column_count, int)
    assert matrix.column_count == 9


def test_shape():
    matrix = BitMatrix.zeros(5, 6)
    shape = matrix.shape
    assert isinstance(shape, tuple)
    assert len(shape) == 2
    assert shape[0] == 5
    assert shape[1] == 6


def test_transposed():
    matrix = BitMatrix.zeros(3, 4)
    transposed = matrix.T
    assert isinstance(transposed, BitMatrix)
    assert transposed.row_count == 4
    assert transposed.column_count == 3


def test_submatrix():
    matrix = BitMatrix.zeros(5, 5)
    rows = [0, 2, 4]
    columns = [1, 3]
    submat = matrix.submatrix(rows, columns)
    assert isinstance(submat, BitMatrix)
    assert submat.row_count == 3
    assert submat.column_count == 2


def test_echelonize():
    matrix = BitMatrix.zeros(3, 3)
    pivots = matrix.echelonize()
    assert isinstance(pivots, list)
    assert all(isinstance(p, int) for p in pivots)


def test_getitem():
    matrix = BitMatrix.zeros(3, 3)
    value = matrix[0, 0]
    assert isinstance(value, bool)


def test_setitem():
    matrix = BitMatrix.zeros(3, 3)
    matrix[1, 2] = True
    assert matrix[1, 2] is True


def test_getitem_setitem_roundtrip():
    matrix = BitMatrix.zeros(2, 2)
    matrix[0, 1] = True
    assert matrix[0, 1] is True
    matrix[0, 1] = False
    assert matrix[0, 1] is False


def test_eq():
    matrix1 = BitMatrix.zeros(2, 2)
    matrix2 = BitMatrix.zeros(2, 2)
    result = matrix1 == matrix2
    assert isinstance(result, bool)


def test_ne():
    matrix1 = BitMatrix.zeros(2, 2)
    matrix2 = BitMatrix.ones(2, 2)
    result = matrix1 != matrix2
    assert isinstance(result, bool)


def test_eq_with_non_bitmatrix():
    matrix = BitMatrix.zeros(2, 2)
    result = matrix == "not a matrix"
    assert isinstance(result, bool)
    assert result is False


def test_add():
    matrix1 = BitMatrix.zeros(2, 2)
    matrix2 = BitMatrix.zeros(2, 2)
    result = matrix1 + matrix2
    assert isinstance(result, BitMatrix)
    assert result.shape == (2, 2)


def test_iadd():
    matrix1 = BitMatrix.zeros(2, 2)
    matrix2 = BitMatrix.zeros(2, 2)
    matrix1 += matrix2
    assert isinstance(matrix1, BitMatrix)


def test_mul():
    # * is now element-wise AND (not matrix multiplication)
    matrix1 = BitMatrix.ones(2, 3)
    matrix2 = BitMatrix.zeros(2, 3)
    result = matrix1 * matrix2
    assert isinstance(result, BitMatrix)
    assert result.shape == (2, 3)
    # Element-wise AND with zeros = all zeros
    for i in range(2):
        for j in range(3):
            assert result[i, j] is False


def test_xor():
    matrix1 = BitMatrix.zeros(2, 2)
    matrix2 = BitMatrix.ones(2, 2)
    result = matrix1 ^ matrix2
    assert isinstance(result, BitMatrix)
    assert result.shape == (2, 2)


def test_ixor():
    matrix1 = BitMatrix.zeros(2, 2)
    matrix2 = BitMatrix.ones(2, 2)
    matrix1 ^= matrix2
    assert isinstance(matrix1, BitMatrix)


def test_and():
    matrix1 = BitMatrix.ones(2, 2)
    matrix2 = BitMatrix.zeros(2, 2)
    result = matrix1 & matrix2
    assert isinstance(result, BitMatrix)
    assert result.shape == (2, 2)


def test_iand():
    matrix1 = BitMatrix.ones(2, 2)
    matrix2 = BitMatrix.zeros(2, 2)
    matrix1 &= matrix2
    assert isinstance(matrix1, BitMatrix)


def test_str():
    matrix = BitMatrix.zeros(2, 2)
    result = str(matrix)
    assert isinstance(result, str)
    assert len(result) > 0


def test_repr():
    matrix = BitMatrix.zeros(2, 2)
    result = repr(matrix)
    assert isinstance(result, str)
    assert len(result) > 0


def test_single_element_matrix():
    matrix = BitMatrix.zeros(1, 1)
    assert matrix.row_count == 1
    assert matrix.column_count == 1
    assert matrix.shape == (1, 1)


def test_empty_submatrix_selection():
    matrix = BitMatrix.zeros(3, 3)
    submat = matrix.submatrix([], [])
    assert isinstance(submat, BitMatrix)
    assert submat.row_count == 0
    assert submat.column_count == 0


def test_large_matrix_creation():
    matrix = BitMatrix.zeros(100, 50)
    assert matrix.row_count == 100
    assert matrix.column_count == 50


def test_identity_single_dimension():
    matrix = BitMatrix.identity(1)
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] is True


def test_constructor_returns_bitmatrix():
    matrix = BitMatrix([[0, 0], [0, 0]])
    assert isinstance(matrix, BitMatrix)


def test_static_methods_return_bitmatrix():
    identity = BitMatrix.identity(2)
    zeros = BitMatrix.zeros(2, 2)
    ones = BitMatrix.ones(2, 2)
    
    assert isinstance(identity, BitMatrix)
    assert isinstance(zeros, BitMatrix)
    assert isinstance(ones, BitMatrix)


def test_method_chaining():
    matrix = BitMatrix.identity(3)
    result = matrix.T.T
    assert isinstance(result, BitMatrix)
    assert result.shape == (3, 3)


def test_matmul_matrix_matrix():
    """Test @ operator for matrix-matrix multiplication"""
    # Create simple test matrices
    # [1 0]   [1 1]   [1 1]
    # [0 1] @ [0 1] = [0 1]
    m1 = BitMatrix.identity(2)
    m2 = BitMatrix([[1, 1], [0, 1]])
    result = m1 @ m2
    
    assert isinstance(result, BitMatrix)
    assert result.shape == (2, 2)
    assert result[0, 0] is True
    assert result[0, 1] is True
    assert result[1, 0] is False
    assert result[1, 1] is True


def test_matmul_matrix_vector():
    """Test @ operator for matrix-vector multiplication"""
    # [1 0 1]   [1]   [1]  (row 0: 1*1 ⊕ 0*1 ⊕ 1*0 = 1)
    # [0 1 0] @ [1] = [1]  (row 1: 0*1 ⊕ 1*1 ⊕ 0*0 = 1)
    # [1 1 0]   [0]   [0]  (row 2: 1*1 ⊕ 1*1 ⊕ 0*0 = 0)
    matrix = BitMatrix([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    vector = BitVector([1, 1, 0])
    result = matrix @ vector
    
    assert isinstance(result, BitVector)
    assert len(result) == 3
    assert result[0] is True   # 1⊕0⊕0 = 1
    assert result[1] is True   # 0⊕1⊕0 = 1
    assert result[2] is False  # 1⊕1⊕0 = 0


def test_matmul_returns_correct_types():
    """Verify return types are correct"""
    m = BitMatrix.identity(3)
    v = BitVector.ones(3)
    
    mat_result = m @ m
    vec_result = m @ v
    
    assert isinstance(mat_result, BitMatrix)
    assert isinstance(vec_result, BitVector)


def test_matmul_shape_compatibility():
    """Test that @ enforces shape constraints"""
    m1 = BitMatrix.zeros(2, 3)
    m2 = BitMatrix.zeros(3, 4)
    v = BitVector.zeros(3)
    
    result_mm = m1 @ m2
    assert result_mm.shape == (2, 4)
    
    result_mv = m1 @ v
    assert len(result_mv) == 2


def test_mul_now_elementwise():
    """Test that * is now element-wise AND (not matrix multiplication)"""
    m1 = BitMatrix([[1, 1], [0, 1]])
    m2 = BitMatrix([[1, 0], [0, 1]])
    
    result = m1 * m2
    
    # Should be element-wise AND
    assert result[0, 0] is True   # 1 & 1 = 1
    assert result[0, 1] is False  # 1 & 0 = 0
    assert result[1, 0] is False  # 0 & 0 = 0
    assert result[1, 1] is True   # 1 & 1 = 1
    
    # Compare with old behavior (should now be different from @)
    matmul_result = m1 @ m2
    # [1 1]   [1 0]   [1 1]
    # [0 1] @ [0 1] = [0 1]
    assert matmul_result[0, 0] is True
    assert matmul_result[0, 1] is True
    assert matmul_result[1, 0] is False
    assert matmul_result[1, 1] is True


def test_matmul_identity():
    """Test that I @ A = A @ I = A"""
    m = BitMatrix([[1, 0, 1], [0, 1, 0]])
    identity = BitMatrix.identity(3)
    
    result = m @ identity
    assert result.shape == m.shape
    for i in range(m.row_count):
        for j in range(m.column_count):
            assert result[i, j] == m[i, j]


def test_dot_still_works():
    """Ensure the .dot() method still works for backward compatibility"""
    m1 = BitMatrix.identity(2)
    m2 = BitMatrix([[1, 1], [0, 1]])
    
    dot_result = m1.dot(m2)
    matmul_result = m1 @ m2
    
    assert dot_result.shape == matmul_result.shape
    for i in range(2):
        for j in range(2):
            assert dot_result[i, j] == matmul_result[i, j]


def test_pickle_roundtrip():
    """Test that BitMatrix can be pickled and unpickled correctly"""
    import pickle

    matrix = BitMatrix([[1, 0, 1], [0, 1, 0], [1, 1, 1]])

    serialized = pickle.dumps(matrix)
    restored = pickle.loads(serialized)

    assert isinstance(restored, BitMatrix)
    assert restored.shape == matrix.shape
    for i in range(matrix.row_count):
        for j in range(matrix.column_count):
            assert restored[i, j] == matrix[i, j]


def test_pickle_identity_matrix():
    """Test pickle with identity matrix"""
    import pickle

    matrix = BitMatrix.identity(5)

    restored = pickle.loads(pickle.dumps(matrix))

    assert restored == matrix


def test_pickle_large_matrix():
    """Test pickle with a larger matrix"""
    import pickle

    # 100x100 matrix
    matrix = BitMatrix.identity(100)

    restored = pickle.loads(pickle.dumps(matrix))

    assert restored.shape == (100, 100)
    assert restored == matrix


def test_bytes_roundtrip():
    """Test that _to_bytes/_from_bytes work correctly"""
    matrix = BitMatrix([[1, 0, 1], [0, 1, 0]])

    data = matrix._to_bytes()
    restored = BitMatrix._from_bytes(matrix.row_count, matrix.column_count, data)

    assert isinstance(restored, BitMatrix)
    assert restored.shape == matrix.shape
    for i in range(matrix.row_count):
        for j in range(matrix.column_count):
            assert restored[i, j] == matrix[i, j]


def test_bytes_large_matrix():
    """Test bytes serialization with a larger matrix"""
    matrix = BitMatrix.identity(100)

    data = matrix._to_bytes()
    restored = BitMatrix._from_bytes(matrix.row_count, matrix.column_count, data)

    assert restored == matrix


def test_capsule_roundtrip():
    matrix = BitMatrix([[1, 0, 1], [0, 1, 0]])

    capsule = matrix._as_capsule()
    assert capsule is not None


def test_capsule_multiple_calls():
    matrix = BitMatrix.identity(50)

    # Multiple capsules from same matrix should work without error
    capsule1 = matrix._as_capsule()
    capsule2 = matrix._as_capsule()

    assert capsule1 is not None
    assert capsule2 is not None
