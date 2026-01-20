import itertools
import pickle
from typing import Optional, Callable
import hypothesis
import pytest
from hypothesis import strategies, given
from paulimer import DensePauli
import tests.paulilike_assertions as paulilike
from tests.paulilike_assertions import pauli_strings


@strategies.composite
def dense_pauli_strings(
    draw_from: Callable,
    size: Optional[int] = None,
    min_weight: int = 0,
    max_weight: Optional[int] = None,
) -> str:
    """Generate pauli strings that are at least size 1 (DensePauli needs non-empty strings)"""
    if size is None:
        size = draw_from(strategies.integers(min_value=max(1, min_weight), max_value=100))
    else:
        size = max(1, size)  # Ensure at least size 1
    return draw_from(pauli_strings(size=size, min_weight=min_weight, max_weight=max_weight))


@strategies.composite
def dense_pauli_elements(
    draw_from: Callable,
    size: Optional[int] = None,
    min_weight: int = 0,
    max_weight: Optional[int] = None,
) -> DensePauli:
    character_string = draw_from(dense_pauli_strings(size=size, min_weight=min_weight, max_weight=max_weight))
    return DensePauli(character_string)


@strategies.composite
def equal_size_dense_pauli_elements(
    draw_from: Callable,
    count: int = 2,
    max_size: int = 50,
) -> tuple[DensePauli, ...]:
    size = draw_from(strategies.integers(min_value=1, max_value=max_size))
    elements = []
    for _ in range(count):
        character_string = draw_from(pauli_strings(size=size))
        elements.append(DensePauli(character_string))
    return tuple(elements)


@given(dense_pauli_strings())
def test_characters_property(characters: str) -> None:
    element = DensePauli(characters)
    assert element.characters == characters.upper()


@given(dense_pauli_strings())
def test_getitem(characters: str) -> None:
    element = DensePauli(characters)
    for index, character in enumerate(characters):
        assert element[index] == character.upper()


@given(strategies.integers(min_value=1, max_value=2000))
def test_identity(size: int) -> None:
    identity = DensePauli.identity(size)
    assert identity.exponent == 0
    assert identity.phase == 1
    assert identity.weight == 0
    assert identity.size == size


@given(
    strategies.text(
        strategies.characters(
            blacklist_characters="ixyzIXYZ-+ ",
            blacklist_categories=("Zs", "Zl", "Zp", "Cc", "Cf"),
        ),
        min_size=1,
    )
)
def test_raises_on_invalid_character(characters: str) -> None:
    with pytest.raises((ValueError, TypeError, RuntimeError, Exception)):
        DensePauli(characters)


@given(dense_pauli_strings())
def test_weight(characters: str) -> None:
    element = DensePauli(characters)
    expected_weight = sum(1 for character in characters.upper() if character in "XYZ")
    assert element.weight == expected_weight


@given(dense_pauli_strings())
def test_support(characters: str) -> None:
    element = DensePauli(characters)
    expected_support = [
        index
        for index, character in enumerate(characters.upper())
        if character in "XYZ"
    ]
    assert element.support == expected_support


@given(dense_pauli_strings())
def test_size(characters: str) -> None:
    element = DensePauli(characters)
    assert element.size == len(characters)


@given(dense_pauli_strings())
def test_copy(characters: str) -> None:
    element = DensePauli(characters)
    copied = element.copy()
    assert element == copied
    for index in range(len(characters)):
        element *= DensePauli.x(index, len(characters))
        assert element != copied 


@given(dense_pauli_elements())
def test_abs_has_positive_phase(element: DensePauli) -> None:
    absolute = abs(element)
    assert absolute.phase == 1


@given(dense_pauli_elements())
def test_neg(element: DensePauli) -> None:
    negated = -element
    assert negated.phase == -element.phase
    assert negated.exponent == (element.exponent + 2) % 4
    assert negated.characters == element.characters


@given(dense_pauli_elements())
def test_identity_multiplication_invariance(element: DensePauli) -> None:
    identity = DensePauli.identity(element.size)
    assert element * identity == element
    assert identity * element == element


def test_single_qubit_pauli_multiplication_is_cyclic() -> None:
    for permutation in itertools.permutations(
        [
            DensePauli("X"),
            DensePauli("Y"),
            DensePauli("Z"),
        ]
    ):
        paulilike.assert_product_is_cyclic(permutation)


def test_specific_single_qubit_products() -> None:
    """Test specific Pauli products: X*Y = iZ, etc."""
    x = DensePauli("X")
    y = DensePauli("Y")
    z = DensePauli("Z")
    
    xy = x * y
    assert xy.characters == "Z"
    assert xy.phase == 1.0j
    
    zx = z * x
    assert zx.characters == "Y"
    assert zx.phase == 1.0j, zx


@given(equal_size_dense_pauli_elements(count=2))
def test_multiplication_preserves_size(pair: tuple[DensePauli, DensePauli]) -> None:
    left, right = pair
    product = left * right
    assert product.size == left.size == right.size


@given(equal_size_dense_pauli_elements(count=2))
def test_commutation_is_symmetric(pair: tuple[DensePauli, DensePauli]) -> None:
    left, right = pair
    assert left.commutes_with(right) == right.commutes_with(left)


def test_single_qubit_x_constructor() -> None:
    size = 5
    index = 2
    x_pauli = DensePauli.x(index, size)
    assert x_pauli.size == size
    assert x_pauli[index] == "X"
    assert x_pauli.weight == 1
    assert list(x_pauli.support) == [index]


def test_single_qubit_y_constructor() -> None:
    size = 5
    index = 3
    y_pauli = DensePauli.y(index, size)
    assert y_pauli.size == size
    assert y_pauli[index] == "Y"
    assert y_pauli.weight == 1
    assert list(y_pauli.support) == [index]


def test_single_qubit_z_constructor() -> None:
    size = 5
    index = 1
    z_pauli = DensePauli.z(index, size)
    assert z_pauli.size == size
    assert z_pauli[index] == "Z"
    assert z_pauli.weight == 1
    assert list(z_pauli.support) == [index]


def test_y_phase_accounting() -> None:
    y = DensePauli("Y")
    assert y.phase  == 1
    
    yy = DensePauli("YY")
    assert yy.phase == 1


def test_abs_with_y_operators() -> None:
    y = DensePauli("Y")
    abs_y = abs(y)
    assert abs_y.phase == 1
    assert abs_y.characters == "Y"
    
    yy = DensePauli("YY")
    abs_yy = abs(yy)
    assert abs_yy.phase == 1
    assert abs_yy.characters == "YY"


@given(dense_pauli_elements(), dense_pauli_elements())
def test_tensor(left: DensePauli, right: DensePauli) -> None:
    tensored = left + right
    assert tensored.size == left.size + right.size
    for index in range(left.size):
        assert tensored[index] == left[index]
    for index in range(right.size):
        assert tensored[left.size + index] == right[index]


@given(equal_size_dense_pauli_elements(count=2))
def test_imul(pair: tuple[DensePauli, DensePauli]) -> None:
    left, right = pair
    product = left * right
    left *= right
    assert left == product


@hypothesis.settings(deadline=2000, max_examples=5)
@given(dense_pauli_strings(min_weight=1))
def test_is_persistent(string: str) -> None:
    """DensePauli should be pickleable"""
    original = DensePauli(string)
    dump = pickle.dumps(original)
    loaded = pickle.loads(dump)
    assert loaded == DensePauli(string)


def test_large() -> None:
    weight = 1025
    pauli = DensePauli("Y" * weight)
    assert pauli.weight == weight
    assert pauli.size == weight


def test_str_and_repr() -> None:
    p = DensePauli("XYZ")
    str_rep = str(p)
    repr_rep = repr(p)
    assert len(str_rep) > 0
    assert len(repr_rep) > 0


def test_exponent() -> None:
    assert DensePauli("I").exponent == 0
    assert DensePauli("X").exponent == 0
    assert DensePauli("Z").exponent == 0
    assert DensePauli("Y").exponent == 1


def test_phase() -> None:
    assert DensePauli("I").phase == 1
    assert DensePauli("X").phase == 1
    assert DensePauli("Z").phase == 1
    assert DensePauli("Y").phase == 1

# Exception tests
def test_multiplication_different_sizes_raises() -> None:
    p1 = DensePauli("X")
    p2 = DensePauli("XX")
    with pytest.raises(ValueError, match="Cannot multiply DensePaulis of different sizes"):
        p1 * p2


def test_imul_different_sizes_raises() -> None:
    p1 = DensePauli("X")
    p2 = DensePauli("XX")
    with pytest.raises(ValueError, match="Cannot multiply DensePaulis of different sizes"):
        p1 *= p2


def test_invalid_comparison_operators_raise() -> None:
    p1 = DensePauli("X")
    p2 = DensePauli("Y")
    
    with pytest.raises(NotImplementedError):
        p1 < p2
    
    with pytest.raises(NotImplementedError):
        p1 <= p2
    
    with pytest.raises(NotImplementedError):
        p1 > p2
    
    with pytest.raises(NotImplementedError):
        p1 >= p2


def test_getitem_out_of_bounds_raises() -> None:
    p = DensePauli("XYZ")
    # Valid indices
    assert p[0] == "X"
    assert p[2] == "Z"
    
    # Out of bounds should raise
    with pytest.raises((IndexError, Exception)):
        _ = p[3]
    
    with pytest.raises((IndexError, Exception)):
        _ = p[100]


def test_empty_string_constructor() -> None:
    p = DensePauli("")
    assert p.size == 0
    assert p.weight == 0


def test_invalid_pauli_strings() -> None:
    invalid_strings = [
        "A",  # Invalid character
        "XYZ!",  # Special character
        "X@Y",  # Special character
        "XYZABC",  # Mixed valid and invalid
    ]
    
    for invalid in invalid_strings:
        with pytest.raises((ValueError, Exception)):
            DensePauli(invalid)


def test_equality_and_inequality() -> None:
    p1 = DensePauli("XYZ")
    p2 = DensePauli("XYZ")
    p3 = DensePauli("ZYX")
    
    # Equality
    assert p1 == p2
    assert not (p1 != p2)
    
    # Inequality
    assert p1 != p3
    assert not (p1 == p3)
