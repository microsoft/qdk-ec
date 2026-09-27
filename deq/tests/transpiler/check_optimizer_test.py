"""Unit tests for :mod:`deq.transpiler.check_optimizer`."""

import pytest

from deq.transpiler.check_optimizer import (
    RowSpaceTester,
    checks_to_bitmatrix,
    filter_input_virtual_only,
    optimize_checks,
    rref_reduce_with_priority,
)
from deq.transpiler.jit_transpiler import Check


# Helpers --------------------------------------------------------------


def C(*indices: int, parity: bool = False) -> Check:
    return frozenset(indices), parity


def _row_in_span(
    row: Check,
    basis: list[Check],
) -> bool:
    """Brute-force check that ``row`` is in the GF(2) span of ``basis``."""
    n = len(basis)
    for mask in range(1 << n):
        members: set[int] = set()
        parity = False
        for i in range(n):
            if mask & (1 << i):
                members ^= basis[i][0]
                parity ^= basis[i][1]
        if frozenset(members) == row[0] and parity == row[1]:
            return True
    return False


# Tests ----------------------------------------------------------------


def test_no_op_when_already_minimal() -> None:
    # iv = [0,1], physical = [2,3], ov = [4,5].
    finished = [C(0, 2)]
    unfinished = [C(0, 2, 4), C(1, 3, 5)]
    finished_out, unfinished_out = optimize_checks(
        finished, unfinished, input_virtual_count=2, ov_start=4
    )
    # The optimizer can replace unfinished[0] with (4,) because
    # (0,2,4) XOR (0,2) = (4,). That IS strictly less weight, so this
    # case actually exercises elimination too.
    assert unfinished_out[0] == C(4)
    assert unfinished_out[1] == C(1, 3, 5)
    assert finished_out == finished


def test_unfinished_iv_eliminated_by_xor_with_finished() -> None:
    # iv=[0], physical=[1], ov=[2]. Finished ties iv to physical, so
    # the unfinished row can eliminate its iv member via XOR.
    finished = [C(0, 1)]
    unfinished = [C(0, 2)]
    finished_out, unfinished_out = optimize_checks(
        finished, unfinished, input_virtual_count=1, ov_start=2
    )
    # Unfinished now has just (1, 2) -- iv eliminated.
    assert unfinished_out == [C(1, 2)]
    assert finished_out == finished


def test_finished_basis_minimized_pairwise() -> None:
    # iv=[0,1], physical=[2,3], no ov.
    # Two finished rows where row1 XOR row0 has lower physical weight.
    finished = [C(0, 2), C(0, 2, 3)]
    unfinished: list[Check] = []
    finished_out, _ = optimize_checks(
        finished, unfinished, input_virtual_count=2, ov_start=4
    )
    weights = sorted(
        (
            sum(1 for x in m if x < 2),
            sum(1 for x in m if 2 <= x < 4),
        )
        for m, _ in finished_out
    )
    # Ideal: rows become C(0, 2) and C(3,) -- iv 1+0, physical 1+1.
    assert weights == [(0, 1), (1, 1)]


def test_ov_pivot_preserved() -> None:
    # iv=[0], physical=[1,2], ov=[3,4].
    finished = [C(0, 1, 2)]
    unfinished = [C(0, 1, 3), C(2, 4)]
    _, unfinished_out = optimize_checks(
        finished, unfinished, input_virtual_count=1, ov_start=3
    )
    # First unfinished must still pivot ov=3, second ov=4.
    assert 3 in unfinished_out[0][0] and 4 not in unfinished_out[0][0]
    assert 4 in unfinished_out[1][0] and 3 not in unfinished_out[1][0]


def test_row_space_preserved() -> None:
    finished = [C(0, 1), C(0, 2, 3)]
    unfinished = [C(0, 1, 4), C(1, 2, 5)]
    finished_out, unfinished_out = optimize_checks(
        finished, unfinished, input_virtual_count=2, ov_start=4
    )
    new_basis = list(finished_out) + list(unfinished_out)
    for original in list(finished) + list(unfinished):
        assert _row_in_span(original, new_basis), f"{original} no longer in span"


def test_parity_bit_propagates_through_xor() -> None:
    # iv=[0], physical=[1], ov=[2].
    finished = [(frozenset({0, 1}), True)]
    unfinished = [(frozenset({0, 2}), False)]
    _, unfinished_out = optimize_checks(
        finished, unfinished, input_virtual_count=1, ov_start=2
    )
    # Unfinished XORed with finished gets parity True.
    assert unfinished_out == [(frozenset({1, 2}), True)]


def test_no_unfinished_no_iv_in_finished_is_noop() -> None:
    finished = [C(2), C(3)]
    finished_out, unfinished_out = optimize_checks(
        finished, [], input_virtual_count=2, ov_start=4
    )
    assert finished_out == finished
    assert unfinished_out == []


@pytest.mark.parametrize(
    "input_virtual_count,ov_start",
    [(0, 3), (3, 3)],
)
def test_handles_empty_basis(input_virtual_count: int, ov_start: int) -> None:
    finished_out, unfinished_out = optimize_checks(
        [], [], input_virtual_count=input_virtual_count, ov_start=ov_start
    )
    assert finished_out == []
    assert unfinished_out == []


def test_checks_to_bitmatrix_maps_and_filters_columns() -> None:
    columns = {0: 2, 1: None, 2: -1, 3: 4}
    matrix = checks_to_bitmatrix(
        [C(0, 1, 2, 3, parity=True), C(0)],
        4,
        col_of=columns.get,
        parity_col=3,
    )

    assert set(matrix.rows[0].support) == {2, 3}
    assert set(matrix.rows[1].support) == {2}


def test_empty_finished_donor_is_ignored() -> None:
    finished = [C(), C(0, 1)]

    finished_out, unfinished_out = optimize_checks(
        finished, [C(0, 2)], input_virtual_count=1, ov_start=2
    )

    assert finished_out == finished
    assert unfinished_out == [C(1, 2)]


def test_rref_finds_multi_donor_reduction_missed_by_pairwise_descent() -> None:
    finished = [C(0, 3), C(1, 3)]
    unfinished = [C(0, 1, 2, 4)]

    finished_out, unfinished_out = optimize_checks(
        finished, unfinished, input_virtual_count=4, ov_start=4
    )

    assert finished_out == finished
    assert unfinished_out == [C(2, 4)]


def test_rref_priority_remaps_columns_and_propagates_parity() -> None:
    parity_col = 3
    marker_col = 4
    result = rref_reduce_with_priority(
        [0, (1 << 2) | (1 << parity_col)],
        (1 << marker_col) | (1 << 2) | (1 << 0),
        col_order=[2, 0, 1],
        parity_col=parity_col,
        marker_col=marker_col,
    )

    assert result == (1 << 0) | (1 << parity_col)


def test_filter_input_virtual_only_checks() -> None:
    checks = [C(), C(0, 1), C(1, 2), C(3)]

    assert filter_input_virtual_only(checks, input_virtual_count=0) == checks
    assert filter_input_virtual_only(checks, input_virtual_count=2) == [
        C(1, 2),
        C(3),
    ]


def test_row_space_tester_handles_dependencies_parity_and_combinations() -> None:
    tester = RowSpaceTester(
        [C(0, 1), C(0, 1), C(2, parity=True)],
        total_measurements=3,
    )

    assert tester.test(C())
    assert tester.test(C(0, 1))
    assert tester.test(C(0, 1, 2, parity=True))
    assert not tester.test(C(2))
    assert not tester.test(C(0, 1, parity=True))
