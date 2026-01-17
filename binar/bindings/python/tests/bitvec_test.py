from binar import BitVector


def test_initializers():
    for initializer in [
        "101",
        [True, False, True],
        [1, 0, 1],
        [1, 0, True],
    ]:
        bitvec = BitVector(initializer)
        for index, value in enumerate(initializer):
            assert bitvec[index] is bool(int(value)), (index, initializer)


def test_init_from_string_invalid():
    try:
        BitVector("102")
        assert False, "Should have raised an exception"
    except ValueError:
        pass


def test_init_from_invalid_iterable():
    try:
        BitVector([1, 0, 2])  # integers instead of booleans
        assert False, "Should have raised an exception"
    except ValueError:
        pass


def test_zeros():
    bitvec = BitVector.zeros(5)
    assert len(bitvec) == 5
    assert bitvec.is_zero is True


def test_ones():
    bitvec = BitVector.ones(3)
    assert len(bitvec) == 3
    assert bitvec.weight == 3


def test_weight():
    bitvec = BitVector([True, True, False, True])
    assert isinstance(bitvec.weight, int)
    assert bitvec.weight == 3


def test_parity():
    bitvec = BitVector([True, True, False])
    assert isinstance(bitvec.parity, bool)


def test_is_zero():
    zero_vec = BitVector.zeros(5)
    non_zero_vec = BitVector([True, False])
    assert isinstance(zero_vec.is_zero, bool)
    assert isinstance(non_zero_vec.is_zero, bool)
    assert zero_vec.is_zero is True
    assert non_zero_vec.is_zero is False


def test_support():
    bitvec = BitVector([True, False, True, False, True])
    support = bitvec.support
    assert isinstance(support, list)
    assert all(isinstance(idx, int) for idx in support)


def test_resize():
    bitvec = BitVector("101")
    bitvec.resize(10)
    assert len(bitvec) == 10


def test_clear():
    bitvec = BitVector([True, True, True])
    bitvec.clear()
    assert bitvec.is_zero is True


def test_negate_index():
    bitvec = BitVector.zeros(3)
    bitvec.negate_index(1)
    assert bitvec[1] is True


def test_dot():
    vec1 = BitVector([True, False, True])
    vec2 = BitVector([True, True, False])
    result = vec1.dot(vec2)
    assert isinstance(result, bool)


def test_and_weight():
    vec1 = BitVector([True, True, False])
    vec2 = BitVector([True, False, True])
    result = vec1.and_weight(vec2)
    assert isinstance(result, int)


def test_or_weight():
    vec1 = BitVector([True, False, False])
    vec2 = BitVector([False, True, False])
    result = vec1.or_weight(vec2)
    assert isinstance(result, int)


def test_getitem():
    bitvec = BitVector([True, False, True])
    value = bitvec[0]
    assert isinstance(value, bool)
    assert value is True


def test_setitem():
    bitvec = BitVector.zeros(3)
    bitvec[1] = True
    assert bitvec[1] is True


def test_getitem_setitem_roundtrip():
    bitvec = BitVector.zeros(3)
    bitvec[2] = True
    assert bitvec[2] is True
    bitvec[2] = False
    assert bitvec[2] is False


def test_len():
    bitvec = BitVector("1010101")
    result = len(bitvec)
    assert isinstance(result, int)
    assert result == 7


def test_eq():
    vec1 = BitVector.zeros(3)
    vec2 = BitVector.zeros(3)
    result = vec1 == vec2
    assert isinstance(result, bool)


def test_ne():
    vec1 = BitVector.zeros(3)
    vec2 = BitVector([True, False, False])
    result = vec1 != vec2
    assert isinstance(result, bool)


def test_eq_with_non_bitvec():
    bitvec = BitVector.zeros(3)
    result = bitvec == "not a bitvec"
    assert isinstance(result, bool)
    assert result is False


def test_xor():
    vec1 = BitVector([True, False, True])
    vec2 = BitVector([False, True, True])
    result = vec1 ^ vec2
    assert isinstance(result, BitVector)
    assert len(result) == 3


def test_ixor():
    vec1 = BitVector([True, False, True])
    vec2 = BitVector([False, True, False])
    vec1 ^= vec2
    assert isinstance(vec1, BitVector)


def test_and():
    vec1 = BitVector([True, True, False])
    vec2 = BitVector([True, False, True])
    result = vec1 & vec2
    assert isinstance(result, BitVector)
    assert len(result) == 3


def test_iand():
    vec1 = BitVector([True, True, False])
    vec2 = BitVector([True, False, True])
    vec1 &= vec2
    assert isinstance(vec1, BitVector)


def test_or():
    vec1 = BitVector([True, False, False])
    vec2 = BitVector([False, True, False])
    result = vec1 | vec2
    assert isinstance(result, BitVector)
    assert len(result) == 3


def test_ior():
    vec1 = BitVector([True, False, False])
    vec2 = BitVector([False, True, False])
    vec1 |= vec2
    assert isinstance(vec1, BitVector)


def test_iter():
    bitvec = BitVector([True, False, True])
    bits = list(bitvec)
    assert isinstance(bits, list)
    assert all(isinstance(bit, bool) for bit in bits)
    assert len(bits) == 3


def test_str():
    bitvec = BitVector([True, False])
    result = str(bitvec)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "[10]" == result


def test_repr():
    bitvec = BitVector([True, False])
    result = repr(bitvec)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "BitVector" in result

def test_empty_bitvec():
    bitvec = BitVector([])
    assert len(bitvec) == 0


def test_single_bit_operations():
    bitvec = BitVector("1")
    assert bitvec[0] is True
    assert bitvec.weight == 1


def test_large_bitvec_creation():
    bitvec = BitVector.zeros(1000)
    assert len(bitvec) == 1000
    assert bitvec.is_zero is True


def test_constructor_returns_bitvec():
    bitvec = BitVector("101")
    assert isinstance(bitvec, BitVector)


def test_static_methods_return_bitvec():
    zeros = BitVector.zeros(3)
    ones = BitVector.ones(2)
    
    assert isinstance(zeros, BitVector)
    assert isinstance(ones, BitVector)


def test_method_chaining_compatibility():
    bitvec = BitVector([True, False, True])
    result = bitvec ^ bitvec
    assert isinstance(result, BitVector)
    assert result.is_zero is True


def test_resize_preserves_data():
    bitvec = BitVector([True, False])
    original_length = len(bitvec)
    bitvec.resize(5)
    assert len(bitvec) == 5
    assert len(bitvec) > original_length


def test_support_indices():
    bitvec = BitVector([True, False, True, False, True])
    support = bitvec.support
    assert len(support) == bitvec.weight
    for idx in support:
        assert bitvec[idx] is True


def test_slice_basic():
    bitvec = BitVector([True, False, True, True, False])
    sliced = bitvec[1:4]
    assert isinstance(sliced, BitVector)
    assert len(sliced) == 3
    assert sliced[0] is False
    assert sliced[1] is True
    assert sliced[2] is True


def test_slice_from_start():
    bitvec = BitVector([True, False, True])
    sliced = bitvec[:2]
    assert isinstance(sliced, BitVector)
    assert len(sliced) == 2
    assert sliced[0] is True
    assert sliced[1] is False


def test_slice_to_end():
    bitvec = BitVector([True, False, True])
    sliced = bitvec[1:]
    assert isinstance(sliced, BitVector)
    assert len(sliced) == 2
    assert sliced[0] is False
    assert sliced[1] is True


def test_slice_empty():
    bitvec = BitVector([True, False, True])
    sliced = bitvec[2:2]
    assert isinstance(sliced, BitVector)
    assert len(sliced) == 0


def test_slice_full():
    bitvec = BitVector([True, False, True])
    sliced = bitvec[:]
    assert isinstance(sliced, BitVector)
    assert len(sliced) == 3
    assert sliced[0] is True
    assert sliced[1] is False
    assert sliced[2] is True


def test_slice_step_2():
    bitvec = BitVector([True, False, True, False, True])
    sliced = bitvec[::2]
    assert isinstance(sliced, BitVector)
    assert len(sliced) == 3
    assert sliced[0] is True
    assert sliced[1] is True
    assert sliced[2] is True


def test_slice_step_3():
    bitvec = BitVector([True, False, True, False, True, False])
    sliced = bitvec[::3]
    assert isinstance(sliced, BitVector)
    assert len(sliced) == 2
    assert sliced[0] is True
    assert sliced[1] is False


def test_slice_negative_step():
    bitvec = BitVector([True, False, True])
    sliced = bitvec[::-1]
    assert isinstance(sliced, BitVector)
    assert len(sliced) == 3
    assert sliced[0] is True
    assert sliced[1] is False
    assert sliced[2] is True


def test_slice_negative_step_2():
    bitvec = BitVector([True, False, True, False, True])
    sliced = bitvec[::-2]
    assert isinstance(sliced, BitVector)
    assert len(sliced) == 3
    assert sliced[0] is True
    assert sliced[1] is True
    assert sliced[2] is True


def test_slice_with_start_stop_step():
    bitvec = BitVector([True, False, True, False, True, False])
    sliced = bitvec[1:5:2]
    assert isinstance(sliced, BitVector)
    assert len(sliced) == 2
    assert sliced[0] is False
    assert sliced[1] is False


def test_pickle_roundtrip():
    """Test that BitVector can be pickled and unpickled correctly"""
    import pickle

    bitvec = BitVector([True, False, True, False, True])

    serialized = pickle.dumps(bitvec)
    restored = pickle.loads(serialized)

    assert isinstance(restored, BitVector)
    assert len(restored) == len(bitvec)
    for i in range(len(bitvec)):
        assert restored[i] == bitvec[i]


def test_pickle_zeros():
    """Test pickle with zeros vector"""
    import pickle

    bitvec = BitVector.zeros(10)

    restored = pickle.loads(pickle.dumps(bitvec))

    assert restored == bitvec
    assert restored.is_zero is True


def test_pickle_large_vector():
    """Test pickle with a larger vector"""
    import pickle

    # 1000-bit vector
    bitvec = BitVector.ones(1000)

    restored = pickle.loads(pickle.dumps(bitvec))

    assert len(restored) == 1000
    assert restored == bitvec


def test_bytes_roundtrip():
    """Test that _to_bytes/_from_bytes work correctly"""
    bitvec = BitVector([True, False, True, False, True])

    data = bitvec._to_bytes()
    restored = BitVector._from_bytes(len(bitvec), data)

    assert isinstance(restored, BitVector)
    assert len(restored) == len(bitvec)
    for i in range(len(bitvec)):
        assert restored[i] == bitvec[i]


def test_bytes_large_vector():
    """Test bytes serialization with a larger vector"""
    bitvec = BitVector.ones(1000)

    data = bitvec._to_bytes()
    restored = BitVector._from_bytes(len(bitvec), data)

    assert restored == bitvec


def test_capsule_roundtrip():
    """Test that _as_capsule works correctly for borrowing"""
    bitvec = BitVector([True, False, True, False, True])

    capsule = bitvec._as_capsule()
    assert capsule is not None


def test_capsule_multiple_calls():
    """Test that multiple capsule calls from same vector work"""
    bitvec = BitVector.ones(100)

    # Multiple capsules from same vector should work without error
    capsule1 = bitvec._as_capsule()
    capsule2 = bitvec._as_capsule()

    assert capsule1 is not None
    assert capsule2 is not None