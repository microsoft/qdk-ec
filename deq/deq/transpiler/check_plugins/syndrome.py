"""``syndrome`` check plugin.

For syndrome-extraction gadgets and similar circuits where each
stabilizer check should involve as few measurements as possible.

**Finished checks** (no OV indices): for each input-virtual stabilizer,
search for a minimal-weight set of physical (internal) measurements
that forms a valid check with it.  If none exists, no finished check
is emitted for that stabilizer.

**Unfinished checks** (one OV index each): for each output-virtual
stabilizer, first try to find a check using only physical measurements.
If that fails, search for a check using weight-1 then weight-2
combinations of IV indices (plus any physical).  Raises ``ValueError``
if no check is found at weight ≤ 2 on the IV side.

**Metachecks** (``metachecks=1``): additionally emit a basis of
*metachecks* — finished (no-OV) parity checks reflecting redundancy
among the physical (internal) measurements.  Use
``@CHECKS("syndrome", metachecks=1)`` to enable.

Usage::

    @CHECKS("syndrome")                  # default: no metachecks
    @CHECKS("syndrome", metachecks=1)    # with metachecks
"""

from itertools import combinations

from deq.transpiler.check_optimizer import (
    RowSpaceTester,
    checks_to_bitmatrix,
    greedy_minimize_checks,
    rref_reduce_check_targets,
)
from deq.transpiler.check_plugins import Check, CheckPluginInput, CheckPluginOutput
from deq.transpiler.jit_transpiler import regroup_checks


def resolve_checks(inp: CheckPluginInput) -> CheckPluginOutput:
    """Build minimal-weight syndrome checks for *inp*."""
    metachecks_level = int(inp.plugin_kwargs.pop("metachecks", 0))
    assert inp.plugin_kwargs == {}, f"unexpected plugin kwargs: {inp.plugin_kwargs}"

    tester = RowSpaceTester(inp.auto_checks, inp.total_measurements)

    if inp.num_ov == 0:
        finished = _build_finished(inp, tester)
        if metachecks_level >= 1:
            finished = _add_metachecks(inp, finished)
        return CheckPluginOutput(finished=finished, unfinished=[])

    # Standard regroup to get the OV pivot assignment.
    _, unfinished_base = regroup_checks(
        inp.gadget, inp.codes, inp.auto_checks, inp.total_measurements
    )

    # Build finished: per-IV minimal physical checks.
    finished = _build_finished(inp, tester)

    # Build unfinished: per-OV minimal checks.
    unfinished = _build_unfinished(inp, unfinished_base, tester)

    if metachecks_level >= 1:
        finished = _add_metachecks(inp, finished)

    return CheckPluginOutput(finished=finished, unfinished=unfinished)


def _build_finished(
    inp: CheckPluginInput,
    tester: RowSpaceTester,
) -> list[Check]:
    """For each IV stabilizer, find a minimal-weight physical-only check."""
    iv_count = inp.input_virtual_count
    phys_start = iv_count
    phys_end = inp.ov_start

    finished: list[Check] = []
    for iv in range(iv_count):
        found = _find_physical_check_for(iv, phys_start, phys_end, tester)
        if found is not None:
            finished.append(found)
    return finished


def _build_unfinished(
    inp: CheckPluginInput,
    unfinished_base: list[Check],
    tester: RowSpaceTester,
) -> list[Check]:
    """For each OV stabilizer, find minimal check."""
    ov_start = inp.ov_start
    iv_count = inp.input_virtual_count

    result: list[Check] = []
    for k in range(len(unfinished_base)):
        ov_idx = ov_start + k
        # Try physical-only first.
        found = _find_physical_check_for(ov_idx, iv_count, ov_start, tester)
        if found is not None:
            result.append(found)
            continue

        # Try weight-1 IV.
        found = _find_with_iv(ov_idx, iv_count, ov_start, tester, 1)
        if found is not None:
            result.append(found)
            continue

        # Try weight-2 IV.
        found = _find_with_iv(ov_idx, iv_count, ov_start, tester, 2)
        if found is not None:
            result.append(found)
            continue

        raise ValueError(
            f"syndrome plugin: cannot find a check for output stabilizer {k} "
            f"of gadget {inp.gadget.name!r} with ≤2 IV indices. "
            f"Use a different check plugin."
        )
    return result


def _find_physical_check_for(
    anchor: int,
    phys_start: int,
    phys_end: int,
    tester: RowSpaceTester,
) -> Check | None:
    """Find a minimal check ``{anchor, phys...}`` using only physical indices."""
    # Weight 0 on physical side.
    for parity in (False, True):
        candidate = (frozenset({anchor}), parity)
        if tester.test(candidate):
            return candidate

    # Weight 1.
    for p in range(phys_start, phys_end):
        for parity in (False, True):
            candidate = (frozenset({anchor, p}), parity)
            if tester.test(candidate):
                return candidate

    # Weight 2.
    for pa, pb in combinations(range(phys_start, phys_end), 2):
        for parity in (False, True):
            candidate = (frozenset({anchor, pa, pb}), parity)
            if tester.test(candidate):
                return candidate

    return None


def _find_with_iv(
    ov_idx: int,
    iv_count: int,
    ov_start: int,
    tester: RowSpaceTester,
    iv_weight: int,
) -> Check | None:
    """Find a check ``{ov_idx, iv..., phys...}`` with exactly *iv_weight* IVs."""
    phys_range = range(iv_count, ov_start)
    for iv_combo in combinations(range(iv_count), iv_weight):
        base = frozenset({ov_idx}) | frozenset(iv_combo)
        # Try with zero physical indices.
        for parity in (False, True):
            if tester.test((base, parity)):
                return (base, parity)
        # Try with one physical index.
        for p in phys_range:
            for parity in (False, True):
                candidate = (base | frozenset({p}), parity)
                if tester.test(candidate):
                    return candidate
    return None


# ---------------------------------------------------------------------------
# Metacheck support
# ---------------------------------------------------------------------------


def _add_metachecks(
    inp: CheckPluginInput,
    finished: list[Check],
) -> list[Check]:
    """Enumerate and minimize metachecks, appending to *finished*."""
    metachecks = _enumerate_metachecks(inp, finished)
    if metachecks:
        _minimize_metachecks(metachecks, inp)
        finished = finished + [
            (frozenset(members), parity) for members, parity in metachecks if members
        ]
    return finished


def _enumerate_metachecks(
    inp: CheckPluginInput,
    existing_finished: list[Check],
) -> list[tuple[set[int], bool]]:
    """Return a basis of physical-only relations not already in
    ``existing_finished``'s row space.

    Each entry is ``(members, parity)`` where ``members`` is a set of
    indices in ``[input_virtual_count, ov_start)`` (physical block).
    """
    iv_count = inp.input_virtual_count
    ov_start = inp.ov_start
    total = inp.total_measurements
    num_virtual = iv_count + (total - ov_start)  # IV + OV

    if not inp.auto_checks:
        return []

    # Column layout: [IV cols | OV cols | physical cols | parity].
    # Mapping from original index -> column:
    #   IV  i in [0, iv_count)            -> col i
    #   OV  i in [ov_start, total)        -> col iv_count + (i - ov_start)
    #   PHY i in [iv_count, ov_start)     -> col num_virtual + (i - iv_count)
    phys_count = ov_start - iv_count
    width = num_virtual + phys_count + 1
    parity_col = width - 1

    def col_of(idx: int) -> int:
        if idx < iv_count:
            return idx
        if idx >= ov_start:
            return iv_count + (idx - ov_start)
        return num_virtual + (idx - iv_count)

    matrix = checks_to_bitmatrix(inp.auto_checks, width, col_of=col_of)
    matrix.echelonize()

    # After echelonization with IV+OV columns first, rows whose IV+OV
    # entries are all zero form a basis of the physical-only subspace
    # (a.k.a. the metacheck space).
    candidates: list[tuple[set[int], bool]] = []
    for row_idx in range(matrix.row_count):
        if any(matrix[row_idx, c] for c in range(num_virtual)):
            continue
        meta_members: set[int] = set()
        for col in range(num_virtual, width - 1):
            if matrix[row_idx, col]:
                meta_members.add(iv_count + (col - num_virtual))
        meta_parity = bool(matrix[row_idx, parity_col])
        if not meta_members and not meta_parity:
            continue
        candidates.append((meta_members, meta_parity))

    if not candidates or not existing_finished:
        return candidates

    # Skip any candidate already in the row space of existing_finished.
    fin_tester = RowSpaceTester(existing_finished, total)
    return [
        (members, parity)
        for members, parity in candidates
        if not fin_tester.test((frozenset(members), parity))
    ]


def _minimize_metachecks(
    metachecks: list[tuple[set[int], bool]],
    inp: CheckPluginInput,
) -> None:
    """In-place deterministic weight minimization of *metachecks*.

    Metachecks are physical-only rows, so IV-carrying finished checks
    can never reduce their weight (XOR would add IV bits).  We reduce
    metachecks against each other only, with ``forbid_self=True`` to
    prevent self-cancellation.
    """
    iv_count = inp.input_virtual_count
    ov_start = inp.ov_start

    greedy_minimize_checks(
        metachecks,
        donors=metachecks,
        input_virtual_count=iv_count,
        ov_start=ov_start,
        forbid_self=True,
    )
    rref_reduce_check_targets(
        metachecks,
        donors=metachecks,
        input_virtual_count=iv_count,
        ov_start=ov_start,
    )
