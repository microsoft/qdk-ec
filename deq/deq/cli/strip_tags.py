# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint
"""Strip all tag fields from .deq.jit or .deq.bin files.

Clears every ``tag`` string field while preserving ``name`` and
``description`` fields.  Useful for reducing file size and removing
debug annotations before distribution.
"""

from __future__ import annotations

import sys

import arguably

import deq.proto.deq_bin_pb2 as pb
import deq.proto.deq_jit_pb2 as jit_pb


def _strip_gadget_type_tags(gt: pb.GadgetType) -> int:
    """Clear tags on a GadgetType and its children. Returns count cleared."""
    count = 0
    for m in gt.measurements:
        if m.tag:
            m.tag = ""
            count += 1
    for p in gt.inputs:
        if p.tag:
            p.tag = ""
            count += 1
    for p in gt.outputs:
        if p.tag:
            p.tag = ""
            count += 1
    for r in gt.readouts:
        if r.tag:
            r.tag = ""
            count += 1
    return count


def _strip_port_type_tags(pt: pb.PortType) -> int:
    """Clear tags on a PortType and its children."""
    count = 0
    for obs in pt.observables:
        if obs.tag:
            obs.tag = ""
            count += 1
    return count


def _strip_check_model_type_tags(cmt: pb.CheckModelType) -> int:
    """Clear tags on a CheckModelType and its children."""
    count = 0
    for rg in cmt.remote_gadgets:
        if rg.tag:
            rg.tag = ""
            count += 1
    for check in cmt.checks:
        if check.tag:
            check.tag = ""
            count += 1
    return count


def _strip_error_model_type_tags(emt: pb.ErrorModelType) -> int:
    """Clear tags on an ErrorModelType and its children."""
    count = 0
    for rcm in emt.remote_check_models:
        if rcm.tag:
            rcm.tag = ""
            count += 1
    for error in emt.errors:
        if error.tag:
            error.tag = ""
            count += 1
    return count


def strip_bin_library(library: pb.Library) -> tuple[pb.Library, int]:
    """Strip all tag fields from a BIN library.

    Returns (stripped_library, count_of_tags_cleared).
    """
    result = pb.Library()
    result.CopyFrom(library)
    count = 0

    for gt in result.gadget_types:
        count += _strip_gadget_type_tags(gt)
    for pt in result.port_types:
        count += _strip_port_type_tags(pt)
    for cmt in result.check_model_types:
        count += _strip_check_model_type_tags(cmt)
    for emt in result.error_model_types:
        count += _strip_error_model_type_tags(emt)

    for instr in result.program:
        if instr.HasField("gadget") and instr.gadget.tag:
            instr.gadget.tag = ""
            count += 1
        if instr.HasField("check_model") and instr.check_model.tag:
            instr.check_model.tag = ""
            count += 1
        if instr.HasField("error_model") and instr.error_model.tag:
            instr.error_model.tag = ""
            count += 1

    return result, count


def strip_jit_library(
    jit_library: jit_pb.JitLibrary,
) -> tuple[jit_pb.JitLibrary, int]:
    """Strip all tag fields from a JIT library.

    Returns (stripped_library, count_of_tags_cleared).
    """
    result = jit_pb.JitLibrary()
    result.CopyFrom(jit_library)
    count = 0

    for gtype in result.gadget_types:
        count += _strip_gadget_type_tags(gtype.base)
        for check in gtype.finished_checks:
            if check.base.tag:
                check.base.tag = ""
                count += 1
        for check in gtype.unfinished_checks:
            if check.base.tag:
                check.base.tag = ""
                count += 1
        for error in gtype.errors:
            if error.base.tag:
                error.base.tag = ""
                count += 1

    for ptype in result.port_types:
        count += _strip_port_type_tags(ptype.base)
        for stab in ptype.stabilizers:
            if stab.tag:
                stab.tag = ""
                count += 1

    return result, count


# ── CLI ──────────────────────────────────────────────────────────────────


@arguably.command
def strip_tags(
    file: str,
    *,
    out: str | None = None,
) -> None:
    """Strip all tag fields from a .deq.jit or .deq.bin file.

    Clears every ``tag`` string field while preserving ``name`` and
    ``description`` fields.  Produces a smaller binary file suitable
    for distribution.
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

        stripped, count = strip_jit_library(jit_library)

        if out is None:
            base = file[: -len(".deq.jit")]
            out = f"{base}.min.deq.jit"

        with open(out, "wb") as f:
            f.write(stripped.SerializeToString())

    else:
        with open(file, "rb") as f:
            library = pb.Library.FromString(f.read())

        stripped_bin, count = strip_bin_library(library)

        if out is None:
            base = file[: -len(".deq.bin")]
            out = f"{base}.min.deq.bin"

        with open(out, "wb") as f:
            f.write(stripped_bin.SerializeToString())

    print(f"Stripped {count} tags -> {out}")
