"""Validate streaming physical frames against the original detector model."""

from pathlib import Path
import sys

import numpy as np
import pytest
import stim

from deq.cli.jit import _annotate_stim_with_detectors
from deq.proto.deq_jit_pb2 import JitLibrary
from replay_qsharp import build_recipe
from replay_stim import physical_chunks, sample_batch
from test_replay_qsharp import write_case


def test_streaming_frames_preserve_compiled_noiseless_parity(tmp_path: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jit-basics"))
    from lattice_surgery import compile_case

    configuration = compile_case(
        tmp_path, distance=3, probability=0.001, width=4, layers=3, basis="Z", seed=5
    )
    library_path = tmp_path / "circuit.deq.jit"
    physical_path = tmp_path / "circuit.stim"
    _annotate_stim_with_detectors(
        JitLibrary.FromString(library_path.read_bytes()),
        str(physical_path),
        configuration["assertions"],
    )
    circuit = stim.Circuit.from_file(physical_path)
    converter = circuit.compile_m2d_converter()
    recipe = build_recipe(tmp_path / "circuit.deq", library_path, 0.001)
    shots = list(sample_batch(recipe, 0, shots=10, seed=42, reference_seed=71))
    for packed in shots:
        data = np.frombuffer(packed, dtype=np.uint8).reshape(1, -1)
        assert not np.any(
            converter.convert(
                measurements=data, bit_packed=True, append_observables=True
            )
        )


@pytest.mark.parametrize(
    "body",
    [
        "REPEAT 17 { R 0 X 0 M 0 }",
        "REPEAT 17 { R 0 M(1) 0 }",
        "REPEAT 17 { RX 0 Z_ERROR(1) 0 MX 0 }",
    ],
)
def test_reference_signs_noise_and_packed_boundaries(tmp_path: Path, body: str):
    recipe = build_recipe(*write_case(tmp_path, body, 17, repetitions=5), 1)
    shots = list(sample_batch(recipe, 1, shots=10, seed=42, reference_seed=71))
    assert shots == [b"\xff" * 10 + b"\x1f"] * 10


def test_non_css_input_has_no_tableau_fallback(tmp_path: Path):
    recipe = build_recipe(*write_case(tmp_path, "R 0 H 0 M 0", 1), 0)
    with pytest.raises(ValueError, match="requires a CSS circuit"):
        list(physical_chunks(recipe, 0))


def test_preparation_file_guard_rejects_before_writing(tmp_path: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jit-basics"))
    from lattice_surgery import compile_case

    with pytest.raises(ValueError, match="file limit"):
        compile_case(
            tmp_path,
            distance=3,
            probability=0.001,
            width=2,
            layers=1,
            basis="Z",
            seed=42,
            export_physical=False,
            max_file_bytes=1,
        )
    assert not list(tmp_path.iterdir())
