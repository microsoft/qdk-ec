# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint
"""Merge and deduplicate errors in .deq.jit or .deq.bin files.

Removes undetectable errors (those triggering no checks) and merges errors
that trigger the same set of checks into a single error.  When multiple
effects (residual / readout-flips combinations) exist for the same check
set, the effect group with the highest aggregated probability is kept.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

import arguably

import deq.proto.deq_bin_pb2 as pb
import deq.proto.deq_jit_pb2 as jit_pb
from deq.spec.program_identicalness import exclusive_probability_of

JitCheckKey = frozenset[tuple[bool, int]]
BinCheckKey = frozenset[tuple[int, int]]
EffectKey = tuple[frozenset[int], frozenset[int]]


@dataclass
class _EffectGroup:
    """Accumulates errors with the same (checks, effect)."""

    probability: float = 0.0
    tags: list[str] = field(default_factory=list)
    first_residual: list[int] = field(default_factory=list)
    first_readout_flips: list[int] = field(default_factory=list)
    original_indices: list[int] = field(default_factory=list)


@dataclass
class _CheckGroup:
    """All errors sharing the same check key, split by effect."""

    effects: dict[EffectKey, _EffectGroup] = field(default_factory=dict)
    total_probability: float = 0.0
    all_tags: list[str] = field(default_factory=list)
    original_indices: list[int] = field(default_factory=list)


# ── JIT helpers ──────────────────────────────────────────────────────────


def _jit_check_key(error: jit_pb.JitGadgetType.Error) -> JitCheckKey:
    finished = {(True, int(i)) for i in error.finished_checks}
    unfinished = {(False, int(i)) for i in error.unfinished_checks}
    return frozenset(finished | unfinished)


def _jit_effect_key(error: jit_pb.JitGadgetType.Error) -> EffectKey:
    return (
        frozenset(int(r) for r in error.base.residual),
        frozenset(int(r) for r in error.base.readout_flips),
    )


def merge_jit_gadget_type_errors(
    errors: list[jit_pb.JitGadgetType.Error],
) -> tuple[list[jit_pb.JitGadgetType.Error], list[int | None]]:
    """Merge errors for a single JIT gadget type.

    Returns (merged_errors, index_map) where index_map[old] is the new
    index or None if the error was removed.
    """
    groups: dict[JitCheckKey, _CheckGroup] = {}
    removed: set[int] = set()

    for idx, error in enumerate(errors):
        ck = _jit_check_key(error)
        if not ck:
            removed.add(idx)
            continue

        if ck not in groups:
            groups[ck] = _CheckGroup()
        grp = groups[ck]

        p = error.base.probability
        grp.total_probability = exclusive_probability_of(grp.total_probability, p)
        grp.all_tags.append(error.base.tag)
        grp.original_indices.append(idx)

        ek = _jit_effect_key(error)
        if ek not in grp.effects:
            grp.effects[ek] = _EffectGroup(
                first_residual=list(error.base.residual),
                first_readout_flips=list(error.base.readout_flips),
            )
        eg = grp.effects[ek]
        eg.probability = exclusive_probability_of(eg.probability, p)
        eg.tags.append(error.base.tag)
        eg.original_indices.append(idx)

    # Build merged errors
    merged: list[jit_pb.JitGadgetType.Error] = []
    # Map from original index to new index
    index_map: list[int | None] = [None] * len(errors)

    for ck, grp in groups.items():
        best_effect = max(grp.effects.values(), key=lambda eg: eg.probability)

        # Recover finished/unfinished from the check key
        finished = sorted(i for is_fin, i in ck if is_fin)
        unfinished = sorted(i for is_fin, i in ck if not is_fin)

        merged_error = jit_pb.JitGadgetType.Error(
            base=pb.ErrorModelType.Error(
                tag=", ".join(grp.all_tags),
                residual=best_effect.first_residual,
                readout_flips=best_effect.first_readout_flips,
                probability=grp.total_probability,
            ),
            finished_checks=finished,
            unfinished_checks=unfinished,
        )

        new_idx = len(merged)
        for orig_idx in grp.original_indices:
            index_map[orig_idx] = new_idx
        merged.append(merged_error)

    for err in merged:
        if err.base.probability > 0.5:
            tag_preview = err.base.tag[:60]
            print(
                f"  Warning: merged probability {err.base.probability:.6f} > 0.5 "
                f"for error '{tag_preview}...'",
                file=sys.stderr,
            )

    return merged, index_map


# ── BIN helpers ──────────────────────────────────────────────────────────


def _bin_check_key(error: pb.ErrorModelType.Error) -> BinCheckKey:
    checks: set[tuple[int, int]] = set()
    for rc in error.checks:
        rcm = rc.remote_check_model if rc.HasField("remote_check_model") else -1
        checks.add((int(rcm), int(rc.check_index)))
    return frozenset(checks)


def _bin_effect_key(error: pb.ErrorModelType.Error) -> EffectKey:
    return (
        frozenset(int(r) for r in error.residual),
        frozenset(int(r) for r in error.readout_flips),
    )


def merge_bin_error_model_type_errors(
    errors: list[pb.ErrorModelType.Error],
) -> tuple[list[pb.ErrorModelType.Error], list[int | None]]:
    """Merge errors for a single BIN error model type.

    Returns (merged_errors, index_map).
    """
    groups: dict[BinCheckKey, _CheckGroup] = {}
    removed: set[int] = set()

    for idx, error in enumerate(errors):
        ck = _bin_check_key(error)
        if not ck:
            removed.add(idx)
            continue

        if ck not in groups:
            groups[ck] = _CheckGroup()
        grp = groups[ck]

        p = error.probability
        grp.total_probability = exclusive_probability_of(grp.total_probability, p)
        grp.all_tags.append(error.tag)
        grp.original_indices.append(idx)

        ek = _bin_effect_key(error)
        if ek not in grp.effects:
            grp.effects[ek] = _EffectGroup(
                first_residual=list(error.residual),
                first_readout_flips=list(error.readout_flips),
            )
        eg = grp.effects[ek]
        eg.probability = exclusive_probability_of(eg.probability, p)
        eg.tags.append(error.tag)
        eg.original_indices.append(idx)

    merged: list[pb.ErrorModelType.Error] = []
    index_map: list[int | None] = [None] * len(errors)

    for ck, grp in groups.items():
        best_effect = max(grp.effects.values(), key=lambda eg: eg.probability)

        # Rebuild RemoteCheck list from the check key
        remote_checks = []
        for rcm_val, ci in sorted(ck):
            rc = pb.ErrorModelType.RemoteCheck(check_index=ci)
            if rcm_val >= 0:
                rc.remote_check_model = rcm_val
            remote_checks.append(rc)

        merged_error = pb.ErrorModelType.Error(
            tag=", ".join(grp.all_tags),
            checks=remote_checks,
            residual=best_effect.first_residual,
            readout_flips=best_effect.first_readout_flips,
            probability=grp.total_probability,
        )

        new_idx = len(merged)
        for orig_idx in grp.original_indices:
            index_map[orig_idx] = new_idx
        merged.append(merged_error)

    for err in merged:
        if err.probability > 0.5:
            tag_preview = err.tag[:60]
            print(
                f"  Warning: merged probability {err.probability:.6f} > 0.5 "
                f"for error '{tag_preview}...'",
                file=sys.stderr,
            )

    return merged, index_map


# ── Top-level merge functions ────────────────────────────────────────────


def merge_jit_library(
    jit_library: jit_pb.JitLibrary,
) -> tuple[jit_pb.JitLibrary, dict[int, list[int | None]]]:
    """Merge errors across all gadget types in a JIT library.

    Returns (merged_library, map) where map keys are gadget type IDs
    (``gtype``).
    """
    result = jit_pb.JitLibrary()
    result.CopyFrom(jit_library)
    index_maps: dict[int, list[int | None]] = {}

    total_before = 0
    total_after = 0
    total_removed = 0

    for gtype in result.gadget_types:
        gtype_id = int(gtype.base.gtype)
        name = gtype.base.name or f"gtype:{gtype_id}"
        original_errors = list(gtype.errors)
        total_before += len(original_errors)

        merged, idx_map = merge_jit_gadget_type_errors(original_errors)
        total_after += len(merged)
        total_removed += sum(1 for v in idx_map if v is None)

        del gtype.errors[:]
        gtype.errors.extend(merged)
        index_maps[gtype_id] = idx_map

        print(
            f"  {name}: {len(original_errors)} -> {len(merged)} errors "
            f"({sum(1 for v in idx_map if v is None)} removed)"
        )

    print(f"Total: {total_before} -> {total_after} errors ({total_removed} removed)")
    return result, index_maps


def merge_bin_library(
    library: pb.Library,
) -> tuple[pb.Library, dict[int, list[int | None]]]:
    """Merge errors across all error model types in a BIN library.

    Returns (merged_library, map) where map keys are error model type IDs
    (``etype``).
    """
    result = pb.Library()
    result.CopyFrom(library)
    index_maps: dict[int, list[int | None]] = {}

    total_before = 0
    total_after = 0
    total_removed = 0

    for etype in result.error_model_types:
        etype_id = int(etype.etype)
        name = etype.name or f"etype:{etype_id}"
        original_errors = list(etype.errors)
        total_before += len(original_errors)

        merged, idx_map = merge_bin_error_model_type_errors(original_errors)
        total_after += len(merged)
        total_removed += sum(1 for v in idx_map if v is None)

        del etype.errors[:]
        etype.errors.extend(merged)
        index_maps[etype_id] = idx_map

        print(
            f"  {name}: {len(original_errors)} -> {len(merged)} errors "
            f"({sum(1 for v in idx_map if v is None)} removed)"
        )

    print(f"Total: {total_before} -> {total_after} errors ({total_removed} removed)")
    return result, index_maps


# ── CLI ──────────────────────────────────────────────────────────────────


@arguably.command
def merge_errors(
    file: str,
    *,
    out: str | None = None,
    map_: str | None = None,
) -> None:
    """Merge and deduplicate errors in a .deq.jit or .deq.bin file.

    Removes undetectable errors (those that trigger no checks) and merges
    errors that trigger the same set of checks.  When multiple effects
    exist for the same check set, the effect with the highest aggregated
    probability is kept.

    The merged tag is the comma-separated concatenation of the original
    error tags.

    Use ``--map`` to write a JSON index mapping from old to new error
    indices (``null`` for removed errors).
    """
    is_jit = file.endswith(".deq.jit")
    is_bin = file.endswith(".deq.bin")
    if not is_jit and not is_bin:
        print(
            f"Error: cannot determine file type for {file!r}. "
            f"Expected .deq.jit or .deq.bin extension.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if is_jit:
        with open(file, "rb") as f:
            jit_library = jit_pb.JitLibrary.FromString(f.read())

        merged_lib, index_maps = merge_jit_library(jit_library)

        if out is None:
            base = file[: -len(".deq.jit")]
            out = f"{base}.merged.deq.jit"

        with open(out, "wb") as f:
            f.write(merged_lib.SerializeToString())
        with open(f"{out}.txt", "w", encoding="utf-8") as f:
            f.write(str(merged_lib))

    else:
        with open(file, "rb") as f:
            library = pb.Library.FromString(f.read())

        merged_lib_bin, index_maps = merge_bin_library(library)

        if out is None:
            base = file[: -len(".deq.bin")]
            out = f"{base}.merged.deq.bin"

        with open(out, "wb") as f:
            f.write(merged_lib_bin.SerializeToString())
        with open(f"{out}.txt", "w", encoding="utf-8") as f:
            f.write(str(merged_lib_bin))

    print(f"Output: {out}")

    if map_ is not None:
        with open(map_, "w", encoding="utf-8") as f:
            json.dump(index_maps, f, indent=2)
        print(f"Index map: {map_}")
