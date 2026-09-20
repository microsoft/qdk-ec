"""Container-level API: slicing, summaries, and the resolved maps.

These take a loaded qodec and are the entry points a Python caller reaches
for first.
"""

from __future__ import annotations

import pathlib

import pytest

import qodec

EXAMPLES = pathlib.Path(__file__).resolve().parents[3] / "examples"


@pytest.fixture(scope="module")
def repetition3() -> qodec.Qodec:
    return qodec.Qodec.load(str(EXAMPLES / "repetition3" / "repetition3.qodec.yaml"))


@pytest.mark.parametrize("example", ["repetition3", "c4c6"])
def test_load_rejects_directory_with_valid_manifest(example: str) -> None:
    with pytest.raises(qodec.QodecLoadError, match="expected a manifest file path, got directory"):
        qodec.Qodec.load(EXAMPLES / example)


@pytest.mark.parametrize("first_filename", ["qodec.yaml", "first.qodec.yaml"])
def test_load_selects_explicit_file_with_multiple_manifests(
    repetition3: qodec.Qodec, tmp_path: pathlib.Path, first_filename: str
) -> None:
    codec = qodec.Qodec.loads(repetition3.dumps())
    first_path = tmp_path / first_filename
    second_path = tmp_path / "second.qodec.yaml"

    codec.name = "first"
    first_path.write_text(codec.dumps(), encoding="utf-8")
    codec.name = "second"
    second_path.write_text(codec.dumps(), encoding="utf-8")

    first = qodec.Qodec.load(first_path)
    second = qodec.Qodec.load(second_path)
    assert (first.name, second.name) == ("first", "second")
    assert first.manifest_filename == second.manifest_filename == "qodec.yaml"


def test_resolved_maps_are_keyed_by_name(repetition3: qodec.Qodec) -> None:
    assert sorted(repetition3.instruction_sets) == ["repetition3", "stim+rz"]
    assert sorted(repetition3.codes) == ["repetition3"]

    instruction_set = repetition3.instruction_sets["repetition3"]
    assert instruction_set.name == "repetition3"
    assert "prepare_z" in instruction_set.instructions

    code = repetition3.codes["repetition3"]
    assert code.name == "repetition3"
    assert len(code.x) == len(code.z), "one logical X per logical Z"


def test_slice_keeps_the_named_layers_and_clears_the_new_bottom(repetition3: qodec.Qodec) -> None:
    top = repetition3.slice(0, 1)
    assert [layer.instruction_set.name for layer in top.layers] == ["repetition3"]
    # The bottom layer of a slice has nothing left to lower to.
    assert not top.layers[0].gadgets

    whole = repetition3.slice(0, 2)
    assert [layer.instruction_set.name for layer in whole.layers] == ["repetition3", "stim+rz"]
    assert whole.layers[0].gadgets, "an interior layer keeps its gadgets"

    assert not repetition3.slice(1, 1).layers, "an empty half-open range yields no layers"


def test_slice_rejects_an_out_of_range_or_inverted_span(repetition3: qodec.Qodec) -> None:
    for start, stop in [(0, 99), (5, 6)]:
        with pytest.raises(ValueError, match="layer"):
            repetition3.slice(start, stop)
    with pytest.raises(ValueError, match=r"start index 1 must be <= stop index 0"):
        repetition3.slice(1, 0)
    with pytest.raises(OverflowError):
        repetition3.slice(-1, 1)


def test_summary_names_the_qodec_and_its_layers(repetition3: qodec.Qodec) -> None:
    rendered = str(repetition3)
    assert repetition3.name in rendered
    for layer in repetition3.layers:
        assert layer.instruction_set.name in rendered, f"summary omits layer {layer.instruction_set.name}"
    assert rendered.count("\n") >= len(repetition3.layers), "summary should be multi-line"


def test_manifest_fields_are_readable(repetition3: qodec.Qodec) -> None:
    assert repetition3.name == "repetition3"
    assert repetition3.description
    assert isinstance(repetition3.metadata, dict)
    assert repetition3.schema_version is None or isinstance(repetition3.schema_version, int)


def test_code_fields_allow_independent_logical_list_edits() -> None:
    code = qodec.Code(name="c", stabilizers=["Z_0 Z_1"], x=["X_0 X_1"], z=["Z_0"])

    code.name = "renamed"
    code.description = "a description"
    code.stabilizers = ["Z_0 Z_1", "Z_1 Z_2"]
    assert (code.name, code.description) == ("renamed", "a description")
    assert code.stabilizers == ["Z_0 Z_1", "Z_1 Z_2"]

    code.x = ["X_0 X_1 X_2"]
    code.z = ["Z_2"]
    assert (code.x, code.z) == (["X_0 X_1 X_2"], ["Z_2"])

    for field, value in [("x", ["X_0", "X_1"]), ("z", ["Z_0", "Z_1"])]:
        setattr(code, field, value)
        assert getattr(code, field) == value
    code.z = []
    assert code.x == ["X_0", "X_1"] and code.z == []
