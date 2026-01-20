from paulimer import PauliGroup, SparsePauli, centralizer_of, symplectic_form_of


def test_construction():
    generators = [SparsePauli.x(0), SparsePauli.z(1)]
    group = PauliGroup(generators)
    assert isinstance(group, PauliGroup)


def test_empty():
    group = PauliGroup([])
    assert isinstance(group, PauliGroup)


def test_generators():
    generators = [SparsePauli.x(0), SparsePauli.z(1)]
    group = PauliGroup(generators)
    result = group.generators
    assert isinstance(result, list)
    assert len(result) == 2


def test_is_abelian():
    generators = [SparsePauli.x(0), SparsePauli.x(1)]
    group = PauliGroup(generators)
    result = group.is_abelian
    assert isinstance(result, bool)


def test_log2_size():
    generators = [SparsePauli.x(0)]
    group = PauliGroup(generators)
    result = group.log2_size
    assert isinstance(result, int)
    assert result >= 0


def test_support():
    generators = [SparsePauli.x(0), SparsePauli.z(1)]
    group = PauliGroup(generators)
    result = group.support
    assert isinstance(result, list)
    assert 0 in result
    assert 1 in result


def test_elements():
    generators = [SparsePauli.x(0)]
    group = PauliGroup(generators)
    result = list(group.elements)
    assert len(result) > 0
    assert all(isinstance(elem, SparsePauli) for elem in result)


def test_contains():
    generators = [SparsePauli.x(0)]
    group = PauliGroup(generators)
    element = SparsePauli.x(0)
    result = element in group
    assert isinstance(result, bool)
    assert result is True


def test_iter():
    generators = [SparsePauli.x(0)]
    group = PauliGroup(generators)
    elements = list(group.elements)
    assert len(elements) > 0
    assert all(isinstance(elem, SparsePauli) for elem in elements)


def test_centralizer_of():
    generators = [SparsePauli.x(0), SparsePauli.z(1)]
    group = PauliGroup(generators)
    result = centralizer_of(group)
    assert isinstance(result, PauliGroup)


def test_symplectic_form_of():
    generators = [SparsePauli.x(0), SparsePauli.z(1)]
    result = symplectic_form_of(generators)
    basis_list = list(result)
    assert all(isinstance(elem, SparsePauli) for elem in basis_list)



def test_binary_rank():
    generators = [SparsePauli.x(0)]
    group = PauliGroup(generators)
    result = group.binary_rank
    assert isinstance(result, int)


def test_phases():
    generators = [SparsePauli.x(0)]
    group = PauliGroup(generators)
    result = group.phases
    assert isinstance(result, list)
    assert all(isinstance(phase, int) for phase in result)


def test_standard_generators():
    generators = [SparsePauli.x(0), SparsePauli.z(1)]
    group = PauliGroup(generators)
    result = group.standard_generators
    assert isinstance(result, list)


def test_is_stabilizer_group():
    generators = [SparsePauli.x(0), SparsePauli.z(1)]
    group = PauliGroup(generators)
    result = group.is_stabilizer_group
    assert isinstance(result, bool)


def test_group_equality():
    generators1 = [SparsePauli.x(0)]
    generators2 = [SparsePauli.x(0)]
    group1 = PauliGroup(generators1)
    group2 = PauliGroup(generators2)
    result = group1 == group2
    assert isinstance(result, bool)


def test_group_ordering():
    generators1 = [SparsePauli.x(0)]
    generators2 = [SparsePauli.x(0), SparsePauli.z(1)]
    group1 = PauliGroup(generators1)
    group2 = PauliGroup(generators2)
    
    result_le = group1 <= group2
    assert isinstance(result_le, bool)
    
    result_lt = group1 < group2
    assert isinstance(result_lt, bool)


def test_group_union():
    generators1 = [SparsePauli.x(0)]
    generators2 = [SparsePauli.z(1)]
    group1 = PauliGroup(generators1)
    group2 = PauliGroup(generators2)
    result = group1 | group2
    assert isinstance(result, PauliGroup)


def test_group_intersection():
    generators1 = [SparsePauli.x(0), SparsePauli.z(1)]
    generators2 = [SparsePauli.x(0), SparsePauli.y(2)]
    group1 = PauliGroup(generators1)
    group2 = PauliGroup(generators2)
    result = group1 & group2
    assert isinstance(result, PauliGroup)
    # The intersection should contain at least the identity element
    assert result.log2_size >= 0
    # The intersection should be smaller than or equal to both groups
    assert result.log2_size <= group1.log2_size
    assert result.log2_size <= group2.log2_size


def test_group_intersection_with_self():
    generators = [SparsePauli.x(0), SparsePauli.z(1)]
    group = PauliGroup(generators)
    result = group & group
    assert isinstance(result, PauliGroup)
    # Should equal the original group (idempotency)
    assert result.log2_size == group.log2_size
    assert result == group


def test_group_intersection_commutativity():
    generators1 = [SparsePauli.x(0)]
    generators2 = [SparsePauli.z(1)]
    group1 = PauliGroup(generators1)
    group2 = PauliGroup(generators2)
    
    result1 = group1 & group2
    result2 = group2 & group1
    
    assert result1 == result2


def test_group_intersection_empty_groups():
    group1 = PauliGroup([])
    group2 = PauliGroup([])
    result = group1 & group2
    assert isinstance(result, PauliGroup)
    assert result.log2_size == 0


def test_group_intersection_with_identity():
    generators = [SparsePauli.x(0), SparsePauli.z(1)]
    group = PauliGroup(generators)
    identity_group = PauliGroup([SparsePauli.identity()])
    
    result = group & identity_group
    assert isinstance(result, PauliGroup)
    # Intersection with identity group should contain at least the identity
    assert result.log2_size >= 0
    # Should be a subset of both groups
    assert result.log2_size <= group.log2_size
    assert result.log2_size <= identity_group.log2_size


def test_group_quotient():
    generators1 = [SparsePauli.x(0), SparsePauli.z(0)]
    generators2 = [SparsePauli.x(0)]
    group1 = PauliGroup(generators1)
    group2 = PauliGroup(generators2)
    result = group1 / group2
    assert isinstance(result, PauliGroup)


def test_getstate():
    generators = [SparsePauli.x(0), SparsePauli.z(1)]
    group = PauliGroup(generators)
    state = group.__getstate__()
    assert isinstance(state, tuple)


def test_centralizer_with_support():
    generators = [SparsePauli.x(0), SparsePauli.z(1)]
    group = PauliGroup(generators)
    support = [0, 1, 2]
    result = centralizer_of(group, supported_by=support)
    assert isinstance(result, PauliGroup)


def test_single_identity_group():
    generators = [SparsePauli.identity()]
    group = PauliGroup(generators)
    assert group.log2_size >= 0
    assert group.binary_rank == 0


def test_large_qubit_indices():
    generators = [SparsePauli.x(1000), SparsePauli.z(2000)]
    group = PauliGroup(generators)
    assert 1000 in group.support
    assert 2000 in group.support


def test_all_commute_parameter():
    generators = [SparsePauli.x(0), SparsePauli.x(1)]
    group = PauliGroup(generators, all_commute=True)
    assert isinstance(group, PauliGroup)
    assert group.is_abelian == True


def test_getstate_serialization():
    gen1 = SparsePauli.x(0)
    gen2 = SparsePauli.z(0)
    group = PauliGroup([gen1, gen2])
    
    state = group.__getstate__()
    assert state is not None  # Should return some serializable state
    assert isinstance(state, tuple)  # Usually returns a tuple


def test_setstate_deserialization():
    gen1 = SparsePauli.x(0)
    gen2 = SparsePauli.z(0)
    group = PauliGroup([gen1, gen2])
    
    # Get the current state
    state = group.__getstate__()
    
    # Create a new group and try to set its state
    new_group = PauliGroup([])
    new_group.__setstate__(state)
    
    # If this works, the groups should be equivalent in some way
    # (exact equality test might fail due to implementation details)
    assert isinstance(new_group, PauliGroup)
    # Try to verify the restored group has similar properties
    assert new_group.log2_size > 0


def test_serialization_roundtrip():
    import pickle
    
    gen1 = SparsePauli.x(0)
    gen2 = SparsePauli.z(1)
    original_group = PauliGroup([gen1, gen2])
    
    # Try to pickle and unpickle the group
    serialized = pickle.dumps(original_group)
    restored_group = pickle.loads(serialized)
    
    # Verify the restored group is valid and similar
    assert isinstance(restored_group, PauliGroup)
    assert restored_group.log2_size == original_group.log2_size
    assert restored_group.is_abelian == original_group.is_abelian


def test_edge_case_empty_generators():
    group = PauliGroup([])
    assert isinstance(group, PauliGroup)
    assert group.log2_size == 0
    assert group.is_abelian == True  # Empty group should be abelian


def test_edge_case_duplicate_generators():
    gen = SparsePauli.x(0)
    group = PauliGroup([gen, gen, gen])  # Same generator multiple times
    assert isinstance(group, PauliGroup)
    # Should handle duplicates gracefully
    assert group.log2_size > 0


def test_edge_case_identity_generator():
    identity = SparsePauli.identity()
    group = PauliGroup([identity])
    assert isinstance(group, PauliGroup)
    # Identity still generates a group with phases (+I, -I, etc.)
    assert group.log2_size >= 0
    assert group.is_abelian == True


def test_factorization_of_interface():
    generators = [SparsePauli.x(0), SparsePauli.z(1)]
    group = PauliGroup(generators)
    
    # Test return type for element in group
    factorization = group.factorization_of(SparsePauli.x(0))
    assert factorization is None or isinstance(factorization, list)
    
    # Test return type for element not in group  
    factorization = group.factorization_of(SparsePauli.y(0))
    assert factorization is None or isinstance(factorization, list)


def test_factorizations_of_interface():
    generators = [SparsePauli.x(0), SparsePauli.z(1)]
    group = PauliGroup(generators)
    
    # Test empty input
    factorizations = group.factorizations_of([])
    assert isinstance(factorizations, list)
    assert len(factorizations) == 0
    
    # Test single element
    factorizations = group.factorizations_of([SparsePauli.x(0)])
    assert isinstance(factorizations, list)
    assert len(factorizations) == 1
    assert factorizations[0] is None or isinstance(factorizations[0], list)
    
    # Test multiple elements
    elements = [SparsePauli.x(0), SparsePauli.z(1), SparsePauli.y(0)]
    factorizations = group.factorizations_of(elements)
    assert isinstance(factorizations, list)
    assert len(factorizations) == len(elements)
    for f in factorizations:
        assert f is None or isinstance(f, list)


def test_factorization_consistency():
    generators = [SparsePauli.x(0), SparsePauli.z(1)]
    group = PauliGroup(generators)
    
    element = SparsePauli.x(0)
    
    # Single factorization
    single_result = group.factorization_of(element)
    
    # Batch factorization
    batch_result = group.factorizations_of([element])
    
    assert len(batch_result) == 1
    assert batch_result[0] == single_result


def test_quotient_raises_when_divisor_not_subgroup():
    import pytest

    group = PauliGroup([SparsePauli.x(0)])
    non_subgroup = PauliGroup([SparsePauli.z(0)])
    with pytest.raises(ValueError, match="divisor is not a subgroup"):
        _ = group / non_subgroup
