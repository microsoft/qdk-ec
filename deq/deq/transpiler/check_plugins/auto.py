"""``auto`` check plugin — the default.

When the gadget has no user-written ``CHECK`` statements, uses the
paulimer-derived auto checks directly.  When user ``CHECK`` statements
are present, emits them first (verifying each is in the auto-derived
row space) and extends the basis with auto-derived rows that are
linearly independent from the manual set.

The merged set is regrouped into finished/unfinished, optimized via
greedy pairwise XOR, and stripped of input-virtual-only finished checks.
"""

from deq.transpiler.check_plugins import CheckPluginInput, CheckPluginOutput
from deq.transpiler.check_optimizer import (
    filter_input_virtual_only,
    optimize_checks,
)
from deq.transpiler.jit_transpiler import merge_checks_with_manual, regroup_checks


def resolve_checks(inp: CheckPluginInput) -> CheckPluginOutput:
    assert inp.plugin_kwargs == {}, f"unexpected plugin kwargs: {inp.plugin_kwargs}"

    if not inp.manual_checks:
        checks = inp.auto_checks
    else:
        checks = merge_checks_with_manual(
            inp.auto_checks, inp.manual_checks, inp.total_measurements
        )

    finished, unfinished = regroup_checks(
        inp.gadget, inp.codes, checks, inp.total_measurements
    )
    finished, unfinished = optimize_checks(
        finished,
        unfinished,
        input_virtual_count=inp.input_virtual_count,
        ov_start=inp.ov_start,
    )
    finished = filter_input_virtual_only(
        finished, input_virtual_count=inp.input_virtual_count
    )
    return CheckPluginOutput(finished=finished, unfinished=unfinished)
