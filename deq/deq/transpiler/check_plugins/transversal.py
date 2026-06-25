"""``transversal`` check plugin.

Designed for gadgets whose unfinished checks can each be expressed as
a single OV index paired with a small number of non-OV indices.  This
is usually the case for transversal gates on CSS codes.

For each output-virtual stabilizer, enumerates candidate checks at
increasing non-OV weight (0, 1, 2, …) until a valid check is found,
guaranteeing the true minimum weight.  An optional ``max_weight``
parameter caps the total check weight (including the OV index).

Usage::

    @CHECKS("transversal")            # default: search up to full weight
    @CHECKS("transversal", 3)         # each check has ≤ 3 members total
    @CHECKS("transversal", max_weight=3)
"""

from itertools import combinations

from deq.transpiler.check_optimizer import (
    RowSpaceTester,
    filter_input_virtual_only,
    optimize_checks,
)
from deq.transpiler.check_plugins import Check, CheckPluginInput, CheckPluginOutput
from deq.transpiler.jit_transpiler import regroup_checks


def resolve_checks(inp: CheckPluginInput) -> CheckPluginOutput:
    # Parse optional max_weight from positional or keyword args.
    # max_weight is the total check weight including the OV index.
    max_weight: int | None = None
    if "max_weight" in inp.plugin_kwargs:
        max_weight = int(inp.plugin_kwargs.pop("max_weight"))
    elif inp.plugin_args:
        max_weight = int(inp.plugin_args[0])
    assert inp.plugin_kwargs == {}, f"unexpected plugin kwargs: {inp.plugin_kwargs}"

    if inp.num_ov == 0:
        return CheckPluginOutput(finished=list(inp.auto_checks), unfinished=[])

    ov_start = inp.ov_start
    iv_count = inp.input_virtual_count
    num_ov = inp.num_ov
    total = inp.total_measurements

    # Non-OV weight limit: max_weight includes the OV index itself.
    nonov_limit = (max_weight - 1) if max_weight is not None else ov_start

    tester = RowSpaceTester(inp.auto_checks, total)

    # Build unfinished: for each OV stabilizer, find the minimum-weight
    # check by enumerating candidates at increasing non-OV weight.
    unfinished: list[Check] = []
    for k in range(num_ov):
        ov_idx = ov_start + k
        found = _search_min_weight(ov_idx, ov_start, tester, nonov_limit)
        if found is None:
            raise ValueError(
                f"transversal plugin: output stabilizer {k} of gadget "
                f"{inp.gadget.name!r} cannot be reduced to weight "
                f"≤ {nonov_limit} non-OV indices. "
                f"Use a different check plugin or increase max_weight."
            )
        unfinished.append(found)

    # Build finished via standard regroup, then optimize.
    finished, _ = regroup_checks(inp.gadget, inp.codes, inp.auto_checks, total)
    finished, _ = optimize_checks(
        finished,
        unfinished,
        input_virtual_count=iv_count,
        ov_start=ov_start,
    )
    finished = filter_input_virtual_only(finished, input_virtual_count=iv_count)

    return CheckPluginOutput(finished=finished, unfinished=unfinished)


def _search_min_weight(
    ov_idx: int,
    ov_start: int,
    tester: RowSpaceTester,
    max_nonov_weight: int,
) -> Check | None:
    """Find a check ``{ov_idx, ...}`` with minimum non-OV weight.

    Enumerates all non-OV index subsets at weight 0, 1, 2, …, up to
    *max_nonov_weight*, returning the first valid check found.
    """
    all_nonov = list(range(ov_start))
    for w in range(max_nonov_weight + 1):
        for combo in combinations(all_nonov, w):
            base = frozenset({ov_idx}) | frozenset(combo)
            for parity in (False, True):
                if tester.test((base, parity)):
                    return (base, parity)
    return None
