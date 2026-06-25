"""``manual`` check plugin.

Uses only the explicit ``CHECK`` statements written in the gadget body.
Auto-derived checks are ignored except for a verbose-level log comparing
the rank of the manual set against the auto set.

Unlike the ``auto`` plugin, this plugin classifies user-defined checks
into finished/unfinished by direct inspection of output-virtual indices
rather than Gaussian elimination, so the original check structure is
preserved exactly as the user wrote it.

When ``verify=0`` is passed (e.g. ``@CHECKS("manual", verify=0)``),
the paulimer auto-check derivation is skipped entirely, making
transpilation significantly faster for gadgets whose checks are already
known to be correct (e.g. annotated files produced by ``deq annotate``).

Usage::

    @CHECKS("manual")              # default: verify=1
    @CHECKS("manual", verify=0)    # skip auto-check derivation
"""

import logging

from binar import rank

from deq.transpiler.check_optimizer import RowSpaceTester, checks_to_bitmatrix
from deq.transpiler.check_plugins import (
    Check,
    CheckPluginInput,
    CheckPluginOutput,
)

logger = logging.getLogger(__name__)


def resolve_checks(inp: CheckPluginInput) -> CheckPluginOutput:
    verify = int(inp.plugin_kwargs.pop("verify", 1))
    assert inp.plugin_kwargs == {}, f"unexpected plugin kwargs: {inp.plugin_kwargs}"

    if verify and inp.auto_checks:
        # Verify every manual check is in the auto-derived row space.
        tester = RowSpaceTester(inp.auto_checks, inp.total_measurements)
        invalid: list[tuple[int, Check]] = []
        sign_flipped: list[tuple[int, Check]] = []
        for idx, check in enumerate(inp.manual_checks):
            if not tester.test(check):
                members, parity = check
                if tester.test((members, not parity)):
                    sign_flipped.append((idx, check))
                else:
                    invalid.append((idx, check))
        errors: list[str] = []
        if sign_flipped:
            errors.extend(
                _format_invalid_checks(
                    inp.gadget.name,
                    sign_flipped,
                    tester,
                    header=f"{len(sign_flipped)} manual CHECK(s) have the wrong parity (flip the sign):",
                )
            )
        if invalid:
            errors.extend(
                _format_invalid_checks(
                    inp.gadget.name,
                    invalid,
                    tester,
                    header=f"{len(invalid)} manual CHECK(s) are not in the auto-derived check space:",
                )
            )
        if errors:
            raise ValueError("\n".join(errors))

        # Also warn about missing rank.
        auto_rank = _check_rank(inp.auto_checks, inp.total_measurements)
        manual_rank = _check_rank(inp.manual_checks, inp.total_measurements)
        if manual_rank < auto_rank:
            logger.info(
                "%s: manual checks span rank %d / %d (auto rank); "
                "%d independent checks are missing",
                inp.gadget.name,
                manual_rank,
                auto_rank,
                auto_rank - manual_rank,
            )

    return classify_manual_checks(
        inp.gadget.name,
        inp.manual_checks,
        inp.ov_start,
        inp.num_ov,
        check_label="manual CHECK",
    )


def classify_manual_checks(
    gadget_name: str,
    manual_checks: list[Check],
    ov_start: int,
    num_ov: int,
    *,
    check_label: str = "CHECK",
) -> CheckPluginOutput:
    """Partition *manual_checks* into finished / unfinished by OV membership.

    A check is *finished* if it contains no output-virtual (OV) indices,
    and *unfinished* if it contains exactly one.  Raises :class:`ValueError`
    if a check references multiple OV indices, if two checks share the same
    OV index, or if any OV index is uncovered.

    *check_label* is used in error messages (e.g. ``"manual CHECK"``).
    """
    if num_ov == 0:
        return CheckPluginOutput(finished=list(manual_checks), unfinished=[])

    finished: list[Check] = []
    unfinished_by_ov: list[Check | None] = [None] * num_ov

    for check_idx, (members, parity) in enumerate(manual_checks):
        ov_members = {m for m in members if m >= ov_start}
        if len(ov_members) == 0:
            finished.append((members, parity))
        elif len(ov_members) == 1:
            ov_idx = next(iter(ov_members)) - ov_start
            if unfinished_by_ov[ov_idx] is not None:
                raise ValueError(
                    f"gadget {gadget_name!r}: {check_label} #{check_idx} "
                    f"assigns output-virtual index {ov_idx} "
                    f"(measurement {ov_start + ov_idx}), but that index "
                    f"is already covered by an earlier {check_label}"
                )
            unfinished_by_ov[ov_idx] = (members, parity)
        else:
            raise ValueError(
                f"gadget {gadget_name!r}: {check_label} #{check_idx} "
                f"references multiple output-virtual indices "
                f"{sorted(ov_members)}; each unfinished check must "
                f"reference exactly one output-virtual measurement"
            )

    missing_ovs = [k for k, c in enumerate(unfinished_by_ov) if c is None]
    if missing_ovs:
        raise ValueError(
            f"gadget {gadget_name!r}: the following output-virtual "
            f"indices have no {check_label}: {missing_ovs}"
        )

    unfinished = [c for c in unfinished_by_ov if c is not None]
    return CheckPluginOutput(finished=finished, unfinished=unfinished)


def _fmt_check(members: frozenset[int], parity: bool) -> str:
    """Format a check using absolute measurement indices."""
    parts = [f"m{m}" for m in sorted(members)]
    if parity:
        parts.append("FLIP")
    return "CHECK " + " ".join(parts)


def _suggest_closest_check(
    tester: RowSpaceTester,
    check: Check,
) -> str | None:
    """Find the closest valid check by GF(2) row reduction.

    Reduces the candidate check against the auto-derived RREF pivots.
    The residual after reduction is the "error"; XORing the original
    with the residual yields the closest valid check (in the row-space
    sense).

    Returns a human-readable suggestion string, or None if the check
    is already valid or the residual equals the original (no overlap).
    """
    members, parity = check

    # Build bitmask for the candidate.
    original = 0
    for idx in members:
        original |= 1 << idx
    if parity:
        original |= 1 << tester._total

    # Reduce through the RREF pivots to get the residual.
    residual = original
    for pivot_bit, pivot_mask in tester._pivot_rows:
        if residual & pivot_bit:
            residual ^= pivot_mask

    if residual == 0:
        return None  # already valid

    # closest valid check = original XOR residual
    valid_bitmask = original ^ residual

    if valid_bitmask == 0:
        return None  # no useful suggestion

    # Decode bitmask back to members + parity.
    valid_members: set[int] = set()
    for bit in range(tester._total):
        if valid_bitmask & (1 << bit):
            valid_members.add(bit)
    valid_parity = bool(valid_bitmask & (1 << tester._total))

    if not valid_members:
        return None  # degenerate: only parity bit, not useful
    valid_check = (frozenset(valid_members), valid_parity)

    # Compute the diff.
    added = sorted(valid_members - members)
    removed = sorted(members - valid_members)
    parity_changed = valid_parity != parity

    diff_parts: list[str] = []
    if removed:
        diff_parts.append("- remove " + ", ".join(f"m{m}" for m in removed))
    if added:
        diff_parts.append("- add " + ", ".join(f"m{m}" for m in added))
    if parity_changed:
        diff_parts.append("- add FLIP" if valid_parity else "- remove FLIP")

    if not diff_parts:
        return None

    return (
        f"  -> closest valid check: {_fmt_check(valid_check[0], valid_check[1])}\n"
        f"     suggestion:\n" + "\n".join(f"       {p}" for p in diff_parts)
    )


def _format_invalid_checks(
    gadget_name: str,
    invalid: list[tuple[int, Check]],
    tester: RowSpaceTester,
    *,
    header: str,
) -> list[str]:
    """Format error lines for invalid checks, with closest-match suggestions."""
    lines = [f"gadget {gadget_name!r}: {header}"]
    for idx, (members, parity) in invalid:
        lines.append(f"  CHECK #{idx}: {_fmt_check(members, parity)}")
        suggestion = _suggest_closest_check(tester, (members, parity))
        if suggestion:
            lines.append(suggestion)
    return lines


def _check_rank(checks: list[Check], width: int) -> int:
    return rank(checks_to_bitmatrix(checks, width + 1))
