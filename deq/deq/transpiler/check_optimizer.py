"""Greedy check-basis optimizer.

After :func:`deq.transpiler.jit_transpiler.regroup_checks` produces
the ``(finished, unfinished)`` partition of a gadget's outcome code,
the resulting basis rows often contain unnecessarily many input-virtual
or physical (real / internal) measurements: any XOR with another row
in the same row space yields an equivalent check, so we are free to
substitute lower-weight representatives.

This module implements a small greedy descent that, for each row,
tries every legal XOR-donor and accepts strict lexicographic
decreases of ``(input_virtual_count, physical_count)``.

Index layout (matches the rest of the JIT pipeline):

- input-virtual measurements: ``[0, input_virtual_count)``
- physical / real / internal measurements:
  ``[input_virtual_count, ov_start)``
- output-virtual measurements: ``[ov_start, total_measurements)``

Constraints honoured by the optimizer:

- **Row-space invariant**: rows are only XOR-combined, never replaced
  by anything outside the original GF(2) row space.
- **OV pivot invariant**: ``unfinished[k]`` must contain the
  output-virtual column ``ov_start + k`` and **no other** OV column.
  So an unfinished row may only receive XOR donations from finished
  rows (which contain zero OV columns by definition); never from
  another unfinished row (that would inject a second OV bit).
"""

from typing import Callable, Sequence

from binar import BitMatrix

from deq.transpiler.jit_transpiler import Check


def checks_to_bitmatrix(
    checks: Sequence[Check],
    width: int,
    *,
    col_of: Callable[[int], int | None] = lambda idx: idx,
    parity_col: int | None = None,
) -> BitMatrix:
    """Build a ``BitMatrix`` of shape ``(len(checks), width)`` from *checks*.

    Each member index ``idx`` of each check is mapped through ``col_of``;
    if the result is ``None`` or outside ``[0, width)`` the bit is skipped.
    The parity bit is placed at ``parity_col`` (default ``width - 1``).

    All call sites that build per-check incidence matrices should funnel
    through this helper so that column-ordering policies stay obvious
    and don't get reimplemented.
    """
    if parity_col is None:
        parity_col = width - 1
    mat = BitMatrix.zeros(len(checks), width)
    for ri, (members, parity) in enumerate(checks):
        for idx in members:
            col = col_of(idx)
            if col is None or col < 0 or col >= width:
                continue
            mat[ri, col] = True
        if parity:
            mat[ri, parity_col] = True
    return mat


def optimize_checks(
    finished: Sequence[Check],
    unfinished: Sequence[Check],
    *,
    input_virtual_count: int,
    ov_start: int,
) -> tuple[list[Check], list[Check]]:
    """Return an equivalent ``(finished, unfinished)`` basis with
    minimized ``(iv_count, physical_count)`` per row.

    The OV-pivot ordering of ``unfinished`` is preserved: the output
    rows come back in the same order, with ``unfinished_out[k]``
    pivoting the same OV column as ``unfinished[k]``.
    """
    finished_rows: list[tuple[set[int], bool]] = [
        (set(members), parity) for members, parity in finished
    ]
    unfinished_rows: list[tuple[set[int], bool]] = [
        (set(members), parity) for members, parity in unfinished
    ]

    # Phase A: optimize finished basis using only finished donors.
    greedy_minimize_checks(
        finished_rows,
        donors=finished_rows,
        input_virtual_count=input_virtual_count,
        ov_start=ov_start,
        forbid_self=True,
    )

    # Phase B: optimize unfinished rows using only finished donors
    # (a finished donor has zero OV bits, so the OV pivot invariant
    # of the unfinished target is preserved).
    greedy_minimize_checks(
        unfinished_rows,
        donors=finished_rows,
        input_virtual_count=input_virtual_count,
        ov_start=ov_start,
        forbid_self=False,
    )

    # Phase C: per-target RREF reduction against finished donors.
    # For each unfinished row that still has weight > (2, 0), build
    # the finished donor space as int bitmasks and RREF-reduce the
    # target with its set columns placed first.  This finds the
    # optimal multi-donor combination in O(n * width) per target.
    rref_reduce_check_targets(
        unfinished_rows,
        donors=finished_rows,
        input_virtual_count=input_virtual_count,
        ov_start=ov_start,
    )

    return (
        [(frozenset(members), parity) for members, parity in finished_rows],
        [(frozenset(members), parity) for members, parity in unfinished_rows],
    )


def greedy_minimize_checks(
    targets: list[tuple[set[int], bool]],
    *,
    donors: list[tuple[set[int], bool]],
    input_virtual_count: int,
    ov_start: int,
    forbid_self: bool,
) -> None:
    """In-place pairwise-XOR descent on ``targets`` using ``donors``.

    Each iteration scans every target and tries XOR-ing it with every
    donor, accepting the first strict lexicographic decrease of the
    target's weight tuple. Loops until no target improves.

    Public API: deterministic — iterates targets/donors in their list
    order with no RNG, so output is byte-identical across runs.
    """
    improved = True
    while improved:
        improved = False
        for t_idx, (t_members, t_parity) in enumerate(targets):
            t_weight = _weight(t_members, input_virtual_count, ov_start)
            for d_idx, (d_members, d_parity) in enumerate(donors):
                if forbid_self and donors is targets and d_idx == t_idx:
                    continue
                if not d_members:
                    continue
                new_members = t_members ^ d_members
                new_weight = _weight(new_members, input_virtual_count, ov_start)
                if new_weight < t_weight:
                    targets[t_idx] = (new_members, t_parity ^ d_parity)
                    improved = True
                    break


def rref_reduce_with_priority(
    donor_masks: Sequence[int],
    target_mask: int,
    *,
    col_order: Sequence[int],
    parity_col: int,
    marker_col: int,
) -> int | None:
    """RREF-reduce *target_mask* against *donor_masks*, prioritizing columns
    listed in *col_order* as pivots.

    Bit ``col_order[i]`` of every input mask is moved to position ``i``
    before RREF; ``parity_col`` and ``marker_col`` are positional bits
    that are NOT in *col_order* (they pass through unchanged).  The
    returned mask is back in the original column space (target's
    marker bit stripped), or ``None`` if the target reduces to zero.

    Used by both :func:`rref_reduce_check_targets` (non-OV donors only)
    and the transversal plugin (full donors with OV-bit filtering done
    by the caller).
    """
    remap = [0] * (max(col_order) + 1 if col_order else 0)
    for pos, orig in enumerate(col_order):
        remap[orig] = pos

    def _remap(mask: int) -> int:
        out = mask & ((1 << parity_col) | (1 << marker_col))
        for orig in col_order:
            if mask & (1 << orig):
                out |= 1 << remap[orig]
        return out

    remapped_donors = [_remap(m) for m in donor_masks if m]
    result = rref_reduce_with_marker(
        remapped_donors, _remap(target_mask), 1 << marker_col
    )
    if result is None:
        return None

    # Unmap: bit at position `pos` came from original col `col_order[pos]`.
    out = result & ((1 << parity_col) | (1 << marker_col))
    out &= ~(1 << marker_col)  # strip marker
    for pos, orig in enumerate(col_order):
        if result & (1 << pos):
            out |= 1 << orig
    return out


def rref_reduce_with_marker(
    donor_masks: list[int],
    target_mask: int,
    marker_bit: int,
) -> int | None:
    """RREF-reduce a *target* row against *donor_masks* using GF(2) elimination.

    Returns the reduced bitmask of the target row (with the marker bit
    stripped) if the target row survives elimination, or ``None`` if the
    target is in the row space of the donors (and thus reduces to zero
    after stripping the marker).

    ``target_mask`` must have ``marker_bit`` set.  Donors must **not**
    have ``marker_bit`` set.

    The caller is responsible for column remapping (to control column
    ordering) and for interpreting the result bitmask back into
    domain-specific indices.
    """
    rows = list(donor_masks) + [target_mask]
    pivots: dict[int, int] = {}
    for ri, _ in enumerate(rows):
        r = rows[ri]
        if r == 0:
            continue
        for col in sorted(pivots):
            if r & (1 << col):
                r ^= rows[pivots[col]]
        if r == 0:
            rows[ri] = 0
            continue
        lead = (r & -r).bit_length() - 1
        rows[ri] = r
        for col, pri in list(pivots.items()):
            if rows[pri] & (1 << lead):
                rows[pri] ^= r
        pivots[lead] = ri

    for row in rows:
        if row & marker_bit:
            return row & ~marker_bit
    return None


def rref_reduce_check_targets(
    targets: list[tuple[set[int], bool]],
    *,
    donors: list[tuple[set[int], bool]],
    input_virtual_count: int,
    ov_start: int,
) -> None:
    """Per-target RREF reduction using int bitmasks for speed.

    For each target with non-OV weight > (2, 0), builds the donor
    space as bitmasks (restricted to non-OV columns), then RREF-reduces
    the target row with its set columns placed first in the pivot
    ordering.  The result is the minimum-weight representative for that
    particular column ordering.

    Public API: deterministic — column ordering is fixed by sorted
    iteration over the target's set bits.
    """
    if not donors:
        return

    # Pre-build donor bitmasks (non-OV columns only + parity).
    parity_col = ov_start
    marker_col = ov_start + 1
    parity_bit = 1 << parity_col
    donor_masks: list[int] = []
    for d_members, d_parity in donors:
        mask = 0
        for idx in d_members:
            if idx < ov_start:
                mask |= 1 << idx
        if d_parity:
            mask |= parity_bit
        donor_masks.append(mask)

    nonov_cols = list(range(ov_start))

    for t_idx, (t_members, t_parity) in enumerate(targets):
        t_w = _weight(t_members, input_virtual_count, ov_start)
        if t_w <= (2, 0):
            continue

        t_ov = {idx for idx in t_members if idx >= ov_start}
        t_nonov = sorted(idx for idx in t_members if idx < ov_start)

        # Column ordering: target's set non-OV bits first, then the rest.
        rest = [idx for idx in nonov_cols if idx not in t_members]
        col_order = t_nonov + rest

        t_mask = 1 << marker_col
        for idx in t_nonov:
            t_mask |= 1 << idx
        if t_parity:
            t_mask |= parity_bit

        result_mask = rref_reduce_with_priority(
            donor_masks,
            t_mask,
            col_order=col_order,
            parity_col=parity_col,
            marker_col=marker_col,
        )
        if result_mask is not None:
            new_nonov = {idx for idx in nonov_cols if result_mask & (1 << idx)}
            new_parity = bool(result_mask & parity_bit)
            new_members = new_nonov | t_ov
            new_w = _weight(new_members, input_virtual_count, ov_start)
            if new_w < t_w:
                targets[t_idx] = (new_members, new_parity)


def _weight(
    members: set[int], input_virtual_count: int, ov_start: int
) -> tuple[int, int]:
    """Return ``(input_virtual_count, physical_count)`` for a row.

    OV indices are ignored — they pivot the unfinished rows and are
    never reduced by the greedy step.
    """
    iv = 0
    physical = 0
    for idx in members:
        if idx < input_virtual_count:
            iv += 1
        elif idx < ov_start:
            physical += 1
    return iv, physical


def filter_input_virtual_only(
    finished: Sequence[Check],
    *,
    input_virtual_count: int,
) -> list[Check]:
    """Remove finished checks that contain only input-virtual indices.

    These are CODE-level stabilizer redundancy relations that get
    recreated by regroup echelonization from output-virtual-only rows.
    They should be stripped after optimization so the optimizer can
    use them as donors first.
    """
    if input_virtual_count <= 0:
        return list(finished)
    return [
        (members, parity)
        for members, parity in finished
        if not all(idx < input_virtual_count for idx in members)
    ]


class RowSpaceTester:
    """Efficient repeated membership tests against a fixed GF(2) row space.

    Pre-computes the RREF and stores each pivot row as a Python ``int``
    bitmask.  Each :meth:`test` call reduces a candidate bitmask against
    the stored pivots — a single XOR per pivot, O(num_pivots) total.
    """

    def __init__(
        self,
        row_space: Sequence[Check],
        total_measurements: int,
    ) -> None:
        width = total_measurements + 1
        self._total = total_measurements

        mat = checks_to_bitmatrix(row_space, width)
        mat.echelonize()

        # Store each pivot row as (pivot_bit, bitmask_int).  Reading
        # the row's `support` (set-bit indices) is much faster than
        # probing each column with __getitem__.
        self._pivot_rows: list[tuple[int, int]] = []
        for row in mat.rows:
            support = row.support
            if not support:
                break  # all-zero row → done (echelonized)
            bitmask = 0
            for col in support:
                bitmask |= 1 << col
            self._pivot_rows.append((1 << support[0], bitmask))

    def test(self, candidate: Check) -> bool:
        """Return True if *candidate* is in the pre-computed row space."""
        members, parity = candidate
        bitmask = 0
        for idx in members:
            bitmask |= 1 << idx
        if parity:
            bitmask |= 1 << self._total
        for pivot_bit, pivot_mask in self._pivot_rows:
            if bitmask & pivot_bit:
                bitmask ^= pivot_mask
        return bitmask == 0
