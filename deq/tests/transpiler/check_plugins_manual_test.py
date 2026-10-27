"""Unit tests for the manual check plugin."""

import logging

import pytest

from deq.circuit.model import GadgetDefinition
from deq.transpiler.check_optimizer import RowSpaceTester
from deq.transpiler.check_plugins import CheckPluginInput
from deq.transpiler.check_plugins.manual import (
    _suggest_closest_check,
    classify_manual_checks,
    resolve_checks,
)
from deq.transpiler.jit_transpiler import Check, MeasurementLayout


def check(*indices: int, parity: bool = False) -> Check:
    return frozenset(indices), parity


def plugin_input(
    manual_checks: list[Check],
    auto_checks: list[Check],
    *,
    plugin_kwargs: dict[str, str | int | float] | None = None,
    total_measurements: int = 2,
) -> CheckPluginInput:
    inp = CheckPluginInput(
        gadget=GadgetDefinition("G"),
        codes={},
        manual_checks=manual_checks,
        total_measurements=total_measurements,
        layout=MeasurementLayout(0, total_measurements),
        plugin_kwargs=dict(plugin_kwargs or {}),
    )
    inp.__dict__["auto_checks"] = auto_checks
    return inp


def test_verify_zero_does_not_request_auto_checks() -> None:
    inp = CheckPluginInput(
        gadget=GadgetDefinition("G"),
        codes={},
        manual_checks=[check(0)],
        total_measurements=1,
        layout=MeasurementLayout(0, 1),
        plugin_kwargs={"verify": 0},
    )

    result = resolve_checks(inp)

    assert result.finished == [check(0)]
    assert result.unfinished == []
    assert "auto_checks" not in inp.__dict__


def test_resolve_checks_rejects_unexpected_plugin_kwargs() -> None:
    inp = plugin_input([], [], plugin_kwargs={"verify": 0, "unknown": 1})

    with pytest.raises(AssertionError, match="unexpected plugin kwargs"):
        resolve_checks(inp)


def test_resolve_checks_reports_wrong_parity_and_invalid_rows() -> None:
    inp = plugin_input(
        [check(0), check(0, 1)],
        [check(0, parity=True)],
    )

    with pytest.raises(ValueError) as exc_info:
        resolve_checks(inp)

    message = str(exc_info.value)
    assert "1 manual CHECK(s) have the wrong parity" in message
    assert "CHECK #0: CHECK m0" in message
    assert "1 manual CHECK(s) are not in the auto-derived check space" in message
    assert "CHECK #1: CHECK m0 m1" in message
    assert "closest valid check: CHECK m0 FLIP" in message
    assert "- remove m1" in message
    assert "- add FLIP" in message


def test_resolve_checks_logs_missing_manual_rank(
    caplog: pytest.LogCaptureFixture,
) -> None:
    inp = plugin_input([check(0)], [check(0), check(1)])

    with caplog.at_level(logging.INFO, logger="deq.transpiler.check_plugins.manual"):
        result = resolve_checks(inp)

    assert result.finished == [check(0)]
    assert "manual checks span rank 1 / 2" in caplog.text
    assert "1 independent checks are missing" in caplog.text


def test_classify_manual_checks_orders_unfinished_by_output_index() -> None:
    result = classify_manual_checks(
        "G",
        [check(4, 6, parity=True), check(1), check(0, 5)],
        ov_start=5,
        num_ov=2,
    )

    assert result.finished == [check(1)]
    assert result.unfinished == [check(0, 5), check(4, 6, parity=True)]


def test_classify_manual_checks_rejects_duplicate_output_index() -> None:
    with pytest.raises(ValueError, match="already covered by an earlier CHECK"):
        classify_manual_checks("G", [check(5), check(1, 5)], ov_start=5, num_ov=1)


def test_classify_manual_checks_rejects_multiple_output_indices() -> None:
    with pytest.raises(ValueError, match="multiple output-virtual indices"):
        classify_manual_checks("G", [check(5, 6)], ov_start=5, num_ov=2)


def test_classify_manual_checks_rejects_missing_output_index() -> None:
    with pytest.raises(ValueError, match=r"indices have no CHECK: \[1\]"):
        classify_manual_checks("G", [check(5)], ov_start=5, num_ov=2)


def test_suggest_closest_check_returns_none_for_valid_or_disjoint_check() -> None:
    tester = RowSpaceTester([check(0, 1)], total_measurements=3)

    assert _suggest_closest_check(tester, check(0, 1)) is None
    assert _suggest_closest_check(tester, check(2)) is None


def test_suggest_closest_check_describes_added_and_removed_measurements() -> None:
    tester = RowSpaceTester([check(0, 1)], total_measurements=3)

    suggestion = _suggest_closest_check(tester, check(0, 2))

    assert suggestion is not None
    assert "closest valid check: CHECK m0 m1" in suggestion
    assert "- remove m2" in suggestion
    assert "- add m1" in suggestion
