"""Tests for the offline headless renderer."""

import struct
import pathlib
import pytest

import deq.proto.deq_bin_pb2 as pb2
import deq.proto.visualizer_pb2 as vis_pb
from deq.visual.render import render_to_png


def _make_box_library() -> pb2.Library:
    """Create a minimal library with one orange box gadget."""
    lib = pb2.Library()
    gt = lib.gadget_types.add()
    gt.name = "TestGadget"
    m = gt.mesh.add()
    m.geometry.CopyFrom(vis_pb.Geometry(type="box", size=[1, 1, 1]))
    m.material.CopyFrom(vis_pb.Material(type="standard", color="#FF8F20"))
    m.relative.CopyFrom(vis_pb.Position(t=0.5))
    instr = lib.program.add()
    instr.gadget.CopyFrom(
        pb2.Gadget(gtype=0, gid=1, position=vis_pb.Position(t=0, i=0, j=0))
    )
    return lib


def _png_dimensions(data: bytes) -> tuple[int, int]:
    """Extract width and height from a PNG file header."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "Not a valid PNG"
    return struct.unpack(">II", data[16:24])


def test_basic_render(tmp_path: pathlib.Path) -> None:
    lib = _make_box_library()
    output = tmp_path / "output.png"
    png = render_to_png(lib, output, width=400, height=300, background="white")

    assert output.exists()
    assert len(png) > 0
    w, h = _png_dimensions(png)
    assert w == 400
    assert h == 300


def test_no_output_path() -> None:
    lib = _make_box_library()
    png = render_to_png(lib, width=400, height=300, background="white")

    assert len(png) > 0
    w, h = _png_dimensions(png)
    assert w == 400
    assert h == 300


def test_empty_library() -> None:
    png = render_to_png(pb2.Library(), width=200, height=150, background="white")

    w, h = _png_dimensions(png)
    assert w == 200
    assert h == 150


def test_box_differs_from_empty() -> None:
    box_png = render_to_png(
        _make_box_library(), width=400, height=300, background="white"
    )
    empty_png = render_to_png(pb2.Library(), width=400, height=300, background="white")

    assert box_png != empty_png


def test_camera_position() -> None:
    lib = _make_box_library()

    png1 = render_to_png(lib, width=400, height=300, background="white")
    png2 = render_to_png(
        lib,
        width=400,
        height=300,
        camera_position={"x": 0, "y": 0, "z": 3},
        background="white",
    )

    assert png1 != png2


def test_bytes_input() -> None:
    lib_bytes = _make_box_library().SerializeToString()
    png = render_to_png(lib_bytes, width=400, height=300, background="white")

    w, h = _png_dimensions(png)
    assert w == 400
    assert h == 300
    assert len(png) > 0


def test_invalid_library_type(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="library must be"):
        render_to_png(123)  # type: ignore[arg-type]


def test_camera_loop_distinct() -> None:
    lib = _make_box_library()
    cameras: list[dict[str, float]] = [
        {"x": 0, "y": 0, "z": 10},
        {"x": 0, "y": 0, "z": 5},
        {"x": 0, "y": 10, "z": 0.1},
        {"x": 8, "y": 2, "z": 0},
        {"x": 5, "y": 5, "z": 5},
    ]

    pngs: list[bytes] = []
    for cam in cameras:
        png = render_to_png(
            lib,
            width=400,
            height=300,
            camera_position=cam,
            background="white",
        )
        assert len(png) > 0
        pngs.append(png)

    for i in range(len(pngs)):
        for j in range(i + 1, len(pngs)):
            assert (
                pngs[i] != pngs[j]
            ), f"camera {cameras[i]} and {cameras[j]} produced identical PNGs"
