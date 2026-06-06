# pylint: disable=no-member
"""Tests for the merge-errors CLI tool."""

import json
import tempfile
from pathlib import Path

import deq.proto.deq_bin_pb2 as pb
import deq.proto.deq_jit_pb2 as jit_pb
from deq.cli.merge_errors import (
    merge_jit_gadget_type_errors,
    merge_bin_error_model_type_errors,
    merge_jit_library,
    merge_bin_library,
    merge_errors,
)

# ── JIT tests ────────────────────────────────────────────────────────────


def _jit_error(
    probability: float,
    finished: list[int] | None = None,
    unfinished: list[int] | None = None,
    residual: list[int] | None = None,
    readout_flips: list[int] | None = None,
    tag: str = "",
) -> jit_pb.JitGadgetType.Error:
    return jit_pb.JitGadgetType.Error(
        base=pb.ErrorModelType.Error(
            tag=tag,
            residual=residual or [],
            readout_flips=readout_flips or [],
            probability=probability,
        ),
        finished_checks=finished or [],
        unfinished_checks=unfinished or [],
    )


def test_jit_remove_undetectable() -> None:
    """Errors with no checks are removed."""
    errors = [
        _jit_error(0.01, finished=[], unfinished=[], tag="E0"),
        _jit_error(0.02, finished=[0], tag="E1"),
    ]
    merged, idx_map = merge_jit_gadget_type_errors(errors)

    assert len(merged) == 1
    assert idx_map[0] is None
    assert idx_map[1] == 0
    assert merged[0].base.tag == "E1"


def test_jit_merge_same_checks_same_effect() -> None:
    """Two errors with same checks and same effect merge into one."""
    errors = [
        _jit_error(0.01, finished=[0, 1], residual=[0], tag="E0"),
        _jit_error(0.02, finished=[0, 1], residual=[0], tag="E1"),
    ]
    merged, idx_map = merge_jit_gadget_type_errors(errors)

    assert len(merged) == 1
    assert idx_map[0] == 0
    assert idx_map[1] == 0
    expected_p = 0.01 + 0.02 - 2 * 0.01 * 0.02
    assert abs(merged[0].base.probability - expected_p) < 1e-12
    assert merged[0].base.tag == "E0, E1"
    assert list(merged[0].base.residual) == [0]


def test_jit_merge_same_checks_different_effects() -> None:
    """Same checks but different effects: picks highest-probability group."""
    errors = [
        _jit_error(0.01, finished=[0], residual=[0], tag="low-res0"),
        _jit_error(0.001, finished=[0], residual=[0], tag="low-res0-2"),
        _jit_error(0.05, finished=[0], residual=[1], tag="high-res1"),
    ]
    merged, idx_map = merge_jit_gadget_type_errors(errors)

    assert len(merged) == 1
    # All three share the same check key, so all map to index 0
    assert all(idx_map[i] == 0 for i in range(3))
    # Effect should be residual=[1] (the high-probability group)
    assert list(merged[0].base.residual) == [1]
    # Tag includes all three
    assert "low-res0" in merged[0].base.tag
    assert "high-res1" in merged[0].base.tag


def test_jit_distinct_check_keys_not_merged() -> None:
    """Errors with different check keys remain separate."""
    errors = [
        _jit_error(0.01, finished=[0], tag="E0"),
        _jit_error(0.02, finished=[1], tag="E1"),
        _jit_error(0.03, unfinished=[0], tag="E2"),
    ]
    merged, idx_map = merge_jit_gadget_type_errors(errors)

    assert len(merged) == 3
    assert all(v is not None for v in idx_map)


def test_jit_finished_vs_unfinished_distinct() -> None:
    """finished=[0] and unfinished=[0] are different check keys."""
    errors = [
        _jit_error(0.01, finished=[0], tag="finished"),
        _jit_error(0.02, unfinished=[0], tag="unfinished"),
    ]
    merged, _ = merge_jit_gadget_type_errors(errors)
    assert len(merged) == 2


def test_jit_all_undetectable() -> None:
    """All errors undetectable → empty result."""
    errors = [
        _jit_error(0.01, tag="E0"),
        _jit_error(0.02, tag="E1"),
    ]
    merged, idx_map = merge_jit_gadget_type_errors(errors)
    assert len(merged) == 0
    assert all(v is None for v in idx_map)


def test_jit_single_error_passthrough() -> None:
    """A single detectable error passes through unchanged."""
    errors = [_jit_error(0.03, finished=[1, 2], residual=[0], tag="only")]
    merged, idx_map = merge_jit_gadget_type_errors(errors)
    assert len(merged) == 1
    assert idx_map[0] == 0
    assert merged[0].base.probability == 0.03
    assert merged[0].base.tag == "only"


# ── BIN tests ────────────────────────────────────────────────────────────


def _bin_error(
    probability: float,
    checks: list[tuple[int | None, int]] | None = None,
    residual: list[int] | None = None,
    readout_flips: list[int] | None = None,
    tag: str = "",
) -> pb.ErrorModelType.Error:
    remote_checks = []
    for rcm, ci in checks or []:
        rc = pb.ErrorModelType.RemoteCheck(check_index=ci)
        if rcm is not None:
            rc.remote_check_model = rcm
        remote_checks.append(rc)
    return pb.ErrorModelType.Error(
        tag=tag,
        checks=remote_checks,
        residual=residual or [],
        readout_flips=readout_flips or [],
        probability=probability,
    )


def test_bin_remove_undetectable() -> None:
    errors = [
        _bin_error(0.01, checks=[], tag="E0"),
        _bin_error(0.02, checks=[(None, 0)], tag="E1"),
    ]
    merged, idx_map = merge_bin_error_model_type_errors(errors)
    assert len(merged) == 1
    assert idx_map[0] is None
    assert idx_map[1] == 0


def test_bin_merge_same_checks_same_effect() -> None:
    errors = [
        _bin_error(0.01, checks=[(0, 1)], residual=[0], tag="E0"),
        _bin_error(0.02, checks=[(0, 1)], residual=[0], tag="E1"),
    ]
    merged, _ = merge_bin_error_model_type_errors(errors)
    assert len(merged) == 1
    expected_p = 0.01 + 0.02 - 2 * 0.01 * 0.02
    assert abs(merged[0].probability - expected_p) < 1e-12
    assert merged[0].tag == "E0, E1"


def test_bin_different_remote_check_model_not_merged() -> None:
    """Same check_index but different remote_check_model → distinct."""
    errors = [
        _bin_error(0.01, checks=[(0, 1)], tag="rcm0"),
        _bin_error(0.02, checks=[(1, 1)], tag="rcm1"),
    ]
    merged, _ = merge_bin_error_model_type_errors(errors)
    assert len(merged) == 2


def test_bin_merge_picks_highest_probability_effect() -> None:
    errors = [
        _bin_error(0.001, checks=[(None, 0)], residual=[0], tag="low"),
        _bin_error(0.1, checks=[(None, 0)], residual=[1], tag="high"),
    ]
    merged, _ = merge_bin_error_model_type_errors(errors)
    assert len(merged) == 1
    assert list(merged[0].residual) == [1]


# ── Library-level tests ──────────────────────────────────────────────────


def test_merge_jit_library_smoke() -> None:
    gtype = jit_pb.JitGadgetType(
        base=pb.GadgetType(gtype=1, name="TestGadget"),
        errors=[
            _jit_error(0.01, finished=[0], tag="E0"),
            _jit_error(0.02, finished=[0], tag="E1"),
            _jit_error(0.03, tag="undetectable"),
        ],
    )
    lib = jit_pb.JitLibrary(gadget_types=[gtype])
    merged_lib, maps = merge_jit_library(lib)

    assert len(merged_lib.gadget_types[0].errors) == 1
    assert 1 in maps
    m = maps[1]
    assert m[0] == 0
    assert m[1] == 0
    assert m[2] is None


def test_merge_bin_library_smoke() -> None:
    etype = pb.ErrorModelType(
        etype=1,
        name="TestEtype",
        errors=[
            _bin_error(0.01, checks=[(None, 0)], tag="E0"),
            _bin_error(0.02, checks=[(None, 0)], tag="E1"),
            _bin_error(0.03, checks=[], tag="undetectable"),
        ],
    )
    lib = pb.Library(error_model_types=[etype])
    merged_lib, maps = merge_bin_library(lib)

    assert len(merged_lib.error_model_types[0].errors) == 1
    assert 1 in maps
    m = maps[1]
    assert m[0] == 0
    assert m[1] == 0
    assert m[2] is None


# ── CLI integration test ─────────────────────────────────────────────────


def test_cli_jit_roundtrip() -> None:
    gtype = jit_pb.JitGadgetType(
        base=pb.GadgetType(gtype=1, name="G"),
        errors=[
            _jit_error(0.01, finished=[0], tag="E0"),
            _jit_error(0.02, finished=[0], tag="E1"),
        ],
    )
    lib = jit_pb.JitLibrary(gadget_types=[gtype])

    with tempfile.TemporaryDirectory() as tmp:
        infile = str(Path(tmp) / "test.deq.jit")
        outfile = str(Path(tmp) / "test.merged.deq.jit")
        mapfile = str(Path(tmp) / "map.json")

        with open(infile, "wb") as f:
            f.write(lib.SerializeToString())

        merge_errors(infile, out=outfile, map_=mapfile)

        with open(outfile, "rb") as f:
            result = jit_pb.JitLibrary.FromString(f.read())
        assert len(result.gadget_types[0].errors) == 1

        with open(mapfile, encoding="utf-8") as f:
            maps = json.load(f)
        assert maps["1"] == [0, 0]


def test_cli_bin_roundtrip() -> None:
    etype = pb.ErrorModelType(
        etype=1,
        name="E",
        errors=[
            _bin_error(0.01, checks=[(None, 0)], tag="E0"),
            _bin_error(0.03, checks=[], tag="removed"),
        ],
    )
    lib = pb.Library(error_model_types=[etype])

    with tempfile.TemporaryDirectory() as tmp:
        infile = str(Path(tmp) / "test.deq.bin")
        outfile = str(Path(tmp) / "test.merged.deq.bin")
        mapfile = str(Path(tmp) / "map.json")

        with open(infile, "wb") as f:
            f.write(lib.SerializeToString())

        merge_errors(infile, out=outfile, map_=mapfile)

        with open(outfile, "rb") as f:
            result = pb.Library.FromString(f.read())
        assert len(result.error_model_types[0].errors) == 1

        with open(mapfile, encoding="utf-8") as f:
            maps = json.load(f)
        assert maps["1"] == [0, None]
