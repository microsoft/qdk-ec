"""Tests that annotate + re-transpile produces a byte-identical JIT library.

After tag stripping (``deq.cli.strip_tags.strip_jit_library``), the
serialized protobuf bytes of the original and re-transpiled annotated
library must match exactly.  This works because:

* Explicit ``PROPAGATE`` statements emitted by annotate pin every
  output logical row of ``correction_propagation`` /
  ``physical_correction`` to the representative the original transpile
  chose (no GF(2) basis-freedom slack).
* Intra-check measurement ordering is reproducible across the
  parser/transpiler.
* Error probabilities are computed deterministically.

The remaining variation (debug ``tag`` strings) is removed by
``strip_jit_library`` before comparison.
"""

from pathlib import Path

import deq.proto.deq_jit_pb2 as jit_pb
from deq.circuit.parser import parse as parse_deq, render_and_parse_file
from deq.cli.strip_tags import strip_jit_library
from deq.transpiler.jit_annotate import annotate as annotate_impl
from deq.transpiler.jit_library_builder import build_jit_library

# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint

CIRCUIT_DIR = Path(__file__).parent


def _assert_annotate_roundtrip(deq_path: Path) -> None:
    """Verify that annotating a .deq file preserves transpilation output."""
    qfile = render_and_parse_file(str(deq_path), mako_defs=None, skip_mako_warning=True)
    orig_lib = build_jit_library(qfile)
    rendered = annotate_impl(qfile)
    anno_lib = build_jit_library(parse_deq(rendered))
    _assert_stripped_bytes_equal(orig_lib, anno_lib, deq_path.name)


def _assert_annotate_roundtrip_mako(deq_path: Path, mako_defs: dict[str, str]) -> None:
    """Verify annotate roundtrip for Mako-templated .deq files."""
    qfile = render_and_parse_file(
        str(deq_path), mako_defs=mako_defs, skip_mako_warning=True
    )
    orig_lib = build_jit_library(qfile)
    rendered = annotate_impl(qfile)
    anno_lib = build_jit_library(parse_deq(rendered))
    _assert_stripped_bytes_equal(orig_lib, anno_lib, f"{deq_path.name} {mako_defs}")


def _assert_stripped_bytes_equal(
    orig_lib: jit_pb.JitLibrary,
    anno_lib: jit_pb.JitLibrary,
    label: str,
) -> None:
    """Assert that ``orig_lib`` and ``anno_lib`` are byte-identical
    after stripping debug tag fields."""
    orig_stripped, _ = strip_jit_library(orig_lib)
    anno_stripped, _ = strip_jit_library(anno_lib)
    orig_bytes = orig_stripped.SerializeToString()
    anno_bytes = anno_stripped.SerializeToString()
    assert orig_bytes == anno_bytes, (
        f"{label}: stripped JIT library bytes differ"
        f" ({len(orig_bytes)} vs {len(anno_bytes)} bytes)"
    )


def test_annotate_code422() -> None:
    _assert_annotate_roundtrip(CIRCUIT_DIR / "fixtures" / "code422.deq")


def test_annotate_repetition_code_d3() -> None:
    _assert_annotate_roundtrip(
        CIRCUIT_DIR / "repetition_code" / "repetition_code_d3.deq"
    )


def test_annotate_surface_code_d3() -> None:
    _assert_annotate_roundtrip(CIRCUIT_DIR / "surface_code" / "surface_code_d3.deq")


def test_annotate_surface_code_d3_noisy() -> None:
    _assert_annotate_roundtrip(
        CIRCUIT_DIR / "surface_code" / "surface_code_d3_noisy.deq"
    )


def test_annotate_repetition_code_mako() -> None:
    _assert_annotate_roundtrip_mako(
        CIRCUIT_DIR / "repetition_code" / "repetition_code.deq",
        {"d": "3", "p": "0.05"},
    )
    _assert_annotate_roundtrip_mako(
        CIRCUIT_DIR / "repetition_code" / "repetition_code.deq",
        {"d": "7", "p": "0.05"},
    )


def test_annotate_surface_code_mako() -> None:
    _assert_annotate_roundtrip_mako(
        CIRCUIT_DIR / "surface_code" / "surface_code.deq",
        {"d": "3", "p": "0.001"},
    )
    _assert_annotate_roundtrip_mako(
        CIRCUIT_DIR / "surface_code" / "surface_code.deq",
        {"d": "7", "p": "0.004"},
    )


def test_annotate_trivial_gadgets() -> None:
    _assert_annotate_roundtrip(CIRCUIT_DIR / "fixtures" / "trivial_gadgets.deq")


def test_annotate_trivial_surgery() -> None:
    _assert_annotate_roundtrip(CIRCUIT_DIR / "fixtures" / "trivial_surgery.deq")


def test_annotate_floquet666() -> None:
    _assert_annotate_roundtrip(CIRCUIT_DIR / "fixtures" / "floquet666.deq")


def test_annotate_teleportation_d3() -> None:
    _assert_annotate_roundtrip(CIRCUIT_DIR / "surface_code" / "teleportation_d3.deq")


def test_annotate_lattice_surgery_d3() -> None:
    _assert_annotate_roundtrip(CIRCUIT_DIR / "surface_code" / "lattice_surgery_d3.deq")


def test_annotate_chained_conditional_same_row() -> None:
    qfile = render_and_parse_file(
        str(CIRCUIT_DIR / "surface_code" / "teleportation_d3.deq"),
        mako_defs=None,
        skip_mako_warning=True,
    )
    orig_lib = build_jit_library(qfile)
    annotated = annotate_impl(qfile)
    anno_lib = build_jit_library(parse_deq(annotated))
    _assert_stripped_bytes_equal(
        orig_lib, anno_lib, "DoubleTeleportConditional + TripleTeleportConditional"
    )
    # Annotated compose GADGETs never emit ``CONDITIONAL``: step-9
    # absorption clears ``logical_correction`` on every merged gadget,
    # so the annotator has no readout-conditioned flip to re-emit.
    for name in (
        "DoubleTeleportConditional",
        "TripleTeleportConditional",
    ):
        block = annotated.split(f"GADGET {name} {{", 1)[1].split("\n}", 1)[0]
        assert "\n    CONDITIONAL " not in block, (
            f"unexpected CONDITIONAL line in {name}: annotator should have "
            f"dropped every source CONDITIONAL (they are absorbed into "
            f"cp/pc by canonical.merge step 9):\n{block}"
        )


def test_annotate_exercise_readout_conditions_destab_readout() -> None:
    """``ExerciseReadoutConditions`` from ``exercise_readout_conditions.deq``
    triggers the case where a compose's ``readout_propagation`` row has
    entries in **destabilizer** columns of the input frame (not just
    logical observable columns).
    """
    fixture = CIRCUIT_DIR / "repetition_code" / "exercise_readout_conditions.deq"
    qfile = render_and_parse_file(str(fixture), mako_defs=None, skip_mako_warning=True)
    orig_lib = build_jit_library(qfile)
    annotated = annotate_impl(qfile)
    anno_lib = build_jit_library(parse_deq(annotated))
    _assert_stripped_bytes_equal(orig_lib, anno_lib, fixture.name)

    block = annotated.split("ExerciseReadoutConditions {", 1)[1].split("\n}", 1)[0]
    readout_lines = [
        line.strip()
        for line in block.splitlines()
        if line.lstrip().startswith("READOUT ")
    ]
    has_destab_token = any(".DS" in line.split("#", 1)[0] for line in readout_lines)
    assert has_destab_token, (
        "expected at least one READOUT line in ExerciseReadoutConditions "
        "to carry an IN<p>.DS<s> destabilizer token bridging the "
        "walker/binary rp mismatch, but found none.  READOUT lines:\n"
        + "\n".join(readout_lines)
    )
