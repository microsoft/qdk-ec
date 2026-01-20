import operator
from functools import reduce
from typing import Iterable, Hashable, Callable, Optional, Type, Any
from hypothesis import strategies
from more_itertools import all_equal, circular_shifts

PauliLike = Any


def pauli_characters() -> strategies.SearchStrategy[str]:
    return strategies.sampled_from("IXYZixyz")


@strategies.composite
def pauli_strings(
    draw_from: Callable,
    size: Optional[int] = None,
    min_weight: int = 0,
    max_weight: Optional[int] = None,
) -> str:
    if size is None:
        size = draw_from(strategies.integers(min_value=min_weight, max_value=100))
    if size < min_weight:
        raise ValueError(f"Size {size} is less than minimum weight {min_weight}.")
    if size == 0:
        return ""
    if max_weight is None:
        max_weight = size
    max_weight = min(size, max_weight)
    weight = draw_from(strategies.integers(min_value=min_weight, max_value=max_weight))
    support = draw_from(
        strategies.lists(
            strategies.integers(min_value=0, max_value=size - 1),
            min_size=weight,
            max_size=weight,
            unique=True,
        )
    )
    support_string = draw_from(
        strategies.text("XYZxyz", min_size=weight, max_size=weight)
    )
    characters = ["I"] * size
    for index, character in zip(support, support_string):
        characters[index] = character
    return "".join(characters)


def assert_scalar_multiplication_is_phase_multiplication(element: PauliLike) -> None:
    for scalar in [1, -1, 1.0j, -1.0j]:
        product = scalar * element
        assert product == element * scalar
        assert product.phase == scalar * element.phase
        assert abs(product) == abs(element)


def assert_product_is_cyclic(elements: Iterable[PauliLike]) -> None:
    cyclic_products = [product_of(shifted) for shifted in circular_shifts(elements)]
    assert all_equal(cyclic_products)


def assert_multiplication_is_invertible(
    left_multiplicand: PauliLike, right_multiplicand: PauliLike
) -> None:
    product = left_multiplicand * right_multiplicand
    right_phase = left_multiplicand.phase**2
    left_phase = right_multiplicand.phase**2
    assert left_multiplicand * product == right_multiplicand * right_phase
    assert product * right_multiplicand == left_multiplicand * left_phase


def assert_hashable_multiplication_is_conditional_phase_multiplication(
    element: PauliLike, label: Hashable
) -> None:
    product = label * element
    assert product == element * label
    assert product.phase == element.phase * label
    assert abs(product) == abs(element)


def assert_support_matches_xyz_characters(element: PauliLike, characters: str) -> None:
    expected_support = [
        index
        for index, character in enumerate(characters.upper())
        if character in "XYZ"
    ]
    assert list(element.support) == expected_support


def assert_construction_is_case_insensitive(
    class_: Type[PauliLike], characters: str
) -> None:
    as_given = class_(characters)
    from_upper = class_(characters.upper())
    from_lower = class_(characters.lower())
    assert as_given == from_upper
    assert as_given == from_lower
    assert from_upper == from_lower


def product_of(elements: Iterable[PauliLike]) -> PauliLike:
    return reduce(operator.mul, elements)
