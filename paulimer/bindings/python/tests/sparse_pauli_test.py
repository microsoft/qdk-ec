import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Callable, Any
import hypothesis
import pytest
from hypothesis import strategies, given
from paulimer import SparsePauli
import tests.paulilike_assertions as paulilike
from tests.paulilike_assertions import pauli_strings

SparsePhase = complex

def sparse_phases() -> strategies.SearchStrategy[SparsePhase]:
    return strategies.sampled_from([1,-1,1.j, -1.j])

def exponents() -> strategies.SearchStrategy[int]:
    return strategies.sampled_from(range(3))


@strategies.composite
def sparse_pauli_elements(  # pylint: disable=too-many-arguments
    draw_from: Callable,
    size: Optional[int] = None,
    min_weight: int = 0,
    max_weight: Optional[int] = None,
    exponent_strategy: strategies.SearchStrategy = exponents(),
    qubit_strategy: strategies.SearchStrategy = strategies.integers(min_value=0, max_value=2**64 - 1),
) -> SparsePauli:
    character_string = draw_from(
        pauli_strings(size=size, min_weight=min_weight, max_weight=max_weight)
    )
    qubits = draw_from(
        strategies.lists(
            qubit_strategy,
            min_size=len(character_string),
            max_size=len(character_string),
            unique=True,
        )
    )
    characters = dict(zip(qubits, character_string))
    exponent = draw_from(exponent_strategy)
    return SparsePauli(characters, exponent=exponent)


@strategies.composite
def equal_length_sparse_pauli_elements(
    draw_from: Callable,
    count: int = 2,
    max_length: int = 100,
    exponent_strategy: strategies.SearchStrategy[int] = strategies.integers(0,3),
) -> tuple[SparsePauli, ...]:
    size = draw_from(strategies.integers(min_value=0, max_value=max_length))
    element_stategy = sparse_pauli_elements(size=size, exponent_strategy=exponent_strategy)
    elements = draw_from(
        strategies.lists(element_stategy, min_size=count, max_size=count)
    )
    return tuple(elements)


@strategies.composite
def distinct_length_sparse_pauli_elements(
    draw_from: Callable,
) -> tuple[SparsePauli, SparsePauli]:
    size_strategy = strategies.lists(
        strategies.integers(min_value=0, max_value=100),
        max_size=2,
        min_size=2,
        unique=True,
    )
    left_size, right_size = draw_from(size_strategy)
    left = draw_from(sparse_pauli_elements(size=left_size))
    right = draw_from(sparse_pauli_elements(size=right_size))
    return (left, right)


@given(pauli_strings())
def test_pauli_characters_match_construction_argument(characters: str) -> None:
    element = SparsePauli(characters)
    element_characters = "".join(element[index] for index in range(len(characters)))
    assert element_characters.upper() == characters.upper()


@given(exponents())
def test_pauli_exponent_matches_construction_argument(exponent: int) -> None:
    assert SparsePauli({}, exponent=exponent).exponent == exponent


@given(
    strategies.text(strategies.sampled_from("ABCDEFGHJKLMNOPQRSTUVW!@#$%^&*()"), min_size=1)
)
def test_pauli_construction_raises_on_invalid_character(characters: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        SparsePauli(characters)


@given(pauli_strings())
def test_pauli_weight_equals_number_of_xyz_characters(characters: str) -> None:
    element = SparsePauli(characters)
    expected_weight = sum(1 for character in characters.upper() if character in "XYZ")
    assert element.weight == expected_weight


@given(pauli_strings())
def test_pauli_support_matches_xyz_characters(characters: str) -> None:
    element = SparsePauli(characters)
    paulilike.assert_support_matches_xyz_characters(element, characters)


def test_single_qubit_pauli_multiplication_is_cyclic() -> None:
    for permutation in itertools.permutations(
        [
            SparsePauli("X"),
            SparsePauli("Y"),
            SparsePauli("Z"),
        ]
    ):
        paulilike.assert_product_is_cyclic(permutation)


@hypothesis.settings(deadline=2000, max_examples=5)
@given(pauli_strings(min_weight=1))
def test_is_persistent(string: str) -> None:
    executor = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
    characters = {index: char for index, char in enumerate(string)}
    persisted = executor.submit(_as_pauli_set, characters).result()
    local = _as_pauli_set(characters)
    assert local == persisted


def test_large_y() -> None:
    weight = 1025
    pauli = SparsePauli({index: "Y" for index in range(weight)})
    assert pauli.weight == weight


def _as_pauli_set(characters: dict[Any, str]) -> set[SparsePauli]:
    element = SparsePauli(characters)
    return {element}


# Additional tests for API compliance

@given(pauli_strings())
def test_characters_property(characters: str) -> None:
    """Test that characters property returns the Pauli string in support order"""
    element = SparsePauli(characters)
    expected_chars = "".join(
        char.upper() for char in characters if char.upper() in "XYZ"
    )
    assert element.characters == expected_chars


def test_identity() -> None:
    """Test identity constructor"""
    identity = SparsePauli.identity()
    assert identity.weight == 0
    assert identity.exponent == 0
    assert identity.phase == 1
    assert len(identity.support) == 0


def test_single_qubit_x_constructor() -> None:
    index = 5
    x_pauli = SparsePauli.x(index)
    assert x_pauli.weight == 1
    assert x_pauli[index] == "X"
    assert list(x_pauli.support) == [index]


def test_single_qubit_y_constructor() -> None:
    index = 3
    y_pauli = SparsePauli.y(index)
    assert y_pauli.weight == 1
    assert y_pauli[index] == "Y"
    assert list(y_pauli.support) == [index]


def test_single_qubit_z_constructor() -> None:
    index = 7
    z_pauli = SparsePauli.z(index)
    assert z_pauli.weight == 1
    assert z_pauli[index] == "Z"
    assert list(z_pauli.support) == [index]


@given(sparse_pauli_elements())
def test_copy(element: SparsePauli) -> None:
    copied = element.copy()
    assert element == copied


def test_abs_has_positive_phase() -> None:
    """Test that abs returns a Pauli with positive phase"""
    # Test with simple cases
    p1 = SparsePauli("XYZ")
    assert abs(p1).phase == 1
    
    p2 = SparsePauli({0: "X"}, exponent=2)  # phase = -1
    assert abs(p2).phase == 1
    
    p3 = SparsePauli({0: "Y", 1: "Z"}, exponent=1)
    assert abs(p3).phase == 1


@given(sparse_pauli_elements())
def test_neg(element: SparsePauli) -> None:
    negated = -element
    assert negated.phase == -element.phase
    assert negated.exponent == (element.exponent + 2) % 4
    assert negated.characters == element.characters


def test_phase_property() -> None:
    """Test phase property for different exponents"""
    assert SparsePauli("I").phase == 1
    assert SparsePauli("X").phase == 1
    assert SparsePauli("Z").phase == 1
    assert SparsePauli("Y").phase == 1


def test_exponent_property() -> None:
    assert SparsePauli("I").exponent == 0
    assert SparsePauli("X").exponent == 0
    assert SparsePauli("Z").exponent == 0
    assert SparsePauli("Y").exponent == 1


def test_str_and_repr() -> None:
    """Test string representations"""
    p = SparsePauli("XYZ")
    str_rep = str(p)
    repr_rep = repr(p)
    assert len(str_rep) > 0
    assert len(repr_rep) > 0


@given(equal_length_sparse_pauli_elements(count=2))
def test_imul(pair: tuple[SparsePauli, SparsePauli]) -> None:
    left, right = pair
    product = left * right
    left *= right
    assert left == product


@given(equal_length_sparse_pauli_elements(count=2))
def test_commutation_is_symmetric(pair: tuple[SparsePauli, SparsePauli]) -> None:
    left, right = pair
    assert left.commutes_with(right) == right.commutes_with(left)


# Exception tests

def test_invalid_comparison_operators_raise() -> None:
    """Comparison operators other than == and != should raise NotImplementedError"""
    p1 = SparsePauli("X")
    p2 = SparsePauli("Y")
    
    with pytest.raises(NotImplementedError):
        p1 < p2
    
    with pytest.raises(NotImplementedError):
        p1 <= p2
    
    with pytest.raises(NotImplementedError):
        p1 > p2
    
    with pytest.raises(NotImplementedError):
        p1 >= p2


def test_invalid_pauli_strings() -> None:
    """Test various invalid Pauli strings"""
    invalid_strings = [
        "A",  # Invalid character
        "XYZ!",  # Special character
        "X@Y",  # Special character
        "XYZAB",  # Mixed valid and invalid
    ]
    
    for invalid in invalid_strings:
        with pytest.raises((ValueError, Exception)):
            SparsePauli(invalid)


def test_equality_and_inequality() -> None:
    """Test equality and inequality operators work correctly"""
    p1 = SparsePauli("XYZ")
    p2 = SparsePauli("XYZ")
    p3 = SparsePauli("ZYX")
    
    # Equality
    assert p1 == p2
    assert not (p1 != p2)
    
    # Inequality
    assert p1 != p3
    assert not (p1 == p3)


def test_support_is_sorted() -> None:
    """Test that support is returned in sorted order"""
    # Create with non-sorted indices
    pauli = SparsePauli({5: "X", 2: "Y", 8: "Z"})
    assert pauli.support == [2, 5, 8]


def test_string_constructor() -> None:
    """Test that string constructor works"""
    p1 = SparsePauli("XYZ")
    assert p1.weight == 3
    assert p1.characters == "XYZ"


def test_dict_constructor() -> None:
    """Test that dict constructor works"""
    p1 = SparsePauli({0: "X", 1: "Y", 2: "Z"})
    assert p1.weight == 3
    assert p1.characters == "XYZ"


def test_empty_constructor() -> None:
    """Test that empty constructor creates identity"""
    p1 = SparsePauli()
    assert p1.weight == 0
    assert p1.phase == 1


def test_dict_constructor_with_exponent() -> None:
    """Test that dict constructor with exponent works"""
    p1 = SparsePauli({0: "X"}, exponent=2)
    assert p1.exponent == 2
    assert p1.phase == -1
