"""Check physical translation, packed output, and qualification failure gates."""

from pathlib import Path
import sys

import numpy as np
import pytest
import stim

from deq.cli.jit import _annotate_stim_with_detectors
from deq.proto.deq_jit_pb2 import JitLibrary
from qualify_qsharp import compare_rates
from replay_qsharp import QSharpSampler, build_recipe


def write_case(
    directory: Path, body: str, measurements: int, repetitions: int = 1
) -> tuple[Path, Path]:
    source = directory / "circuit.deq"
    source.write_text(f"GADGET Probe {{ {body} }}")
    library = JitLibrary()
    definition = library.gadget_types.add().base
    definition.gtype = 1
    definition.name = "Probe"
    for _ in range(measurements):
        definition.measurements.add()
    for gid in range(1, repetitions + 1):
        gadget = library.program.add().gadget
        gadget.gid = gid
        gadget.gtype = 1
    binary = directory / "circuit.deq.jit"
    binary.write_bytes(library.SerializeToString())
    return source, binary


def test_packed_output_crosses_signed_word_and_partial_byte_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("QDK_PYTHON_TELEMETRY", "none")
    paths = write_case(tmp_path, "REPEAT 17 { R 0 X 0 M 0 }", 17, repetitions=5)
    sampler = QSharpSampler(build_recipe(*paths, probability=0), probability=0)
    assert sampler.sample(42) == b"\xff" * 10 + b"\x1f"


def test_compiled_port_mapping_preserves_noiseless_detectors_and_observables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jit-basics"))
    from lattice_surgery import compile_case

    monkeypatch.setenv("QDK_PYTHON_TELEMETRY", "none")
    configuration = compile_case(
        tmp_path, distance=3, probability=0.001, width=4, layers=2, basis="Z", seed=5
    )
    library_path = tmp_path / "circuit.deq.jit"
    physical_path = tmp_path / "circuit.stim"
    _annotate_stim_with_detectors(
        JitLibrary.FromString(library_path.read_bytes()),
        str(physical_path),
        configuration["assertions"],
    )
    circuit = stim.Circuit.from_file(physical_path)
    assert circuit.num_detectors > 0 and circuit.num_observables == 4
    converter = circuit.compile_m2d_converter()
    recipe = build_recipe(tmp_path / "circuit.deq", library_path, probability=0.001)
    assert recipe.total_measurements == circuit.num_measurements
    sampler = QSharpSampler(recipe, probability=0)
    for seed in (0, 1, 42):
        data = np.frombuffer(sampler.sample(seed), dtype=np.uint8).reshape(1, -1)
        detections = converter.convert(
            measurements=data, bit_packed=True, append_observables=True
        )
        assert not np.any(detections)


@pytest.mark.parametrize(
    "body,measurements",
    [
        ("R 0 M(1) 0", 1),
        ("RX 0 MX(1) 0", 1),
        ("R 0 X_ERROR(1) 0 M 0", 1),
        ("RX 0 Z_ERROR(1) 0 MX 0", 1),
        ("R 0 M(1) 0 R 0 X 0 M 0", 2),
    ],
)
def test_measurement_and_pauli_noise_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str, measurements: int
):
    monkeypatch.setenv("QDK_PYTHON_TELEMETRY", "none")
    paths = write_case(tmp_path, body, measurements)
    sampler = QSharpSampler(build_recipe(*paths, probability=1), probability=1)
    assert sampler.sample(42) == bytes([(1 << measurements) - 1])


@pytest.mark.parametrize(
    "body,measurements,reason",
    [
        ("R 0 M(0.001) 0 H 0 M 0", 2, "reset before further quantum use"),
        ("R 0 M 0", 2, "Measurement metadata mismatch"),
        ("R 0 S 0 M 0", 1, "Unsupported physical instruction"),
        ("R 0 X_ERROR(0.002) 0 M 0", 1, "Unexpected noise parameters"),
        ("R 0 M 0 PRESELECT rec[-1]", 1, "Postselection"),
    ],
)
def test_unsupported_or_inconsistent_sources_fail(
    tmp_path: Path, body: str, measurements: int, reason: str
):
    with pytest.raises(ValueError, match=reason):
        build_recipe(*write_case(tmp_path, body, measurements), probability=0.001)


def test_raw_file_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import replay_qsharp

    paths = write_case(tmp_path, "REPEAT 1000 { R 0 M 0 }", 1000, repetitions=100)
    recipe = build_recipe(*paths, probability=0)
    limit = max(path.stat().st_size for path in paths)
    assert recipe.shot_bytes > limit
    monkeypatch.setattr(replay_qsharp, "MAX_FILE_BYTES", limit)
    with pytest.raises(ValueError, match="Raw shot exceeds the 10 GB guard"):
        build_recipe(*paths, probability=0)


@pytest.mark.parametrize(
    "counts,expected",
    [
        ((109, 918, 105, 649), "disagreement"),
        ((109, 738, 104, 787), "qualified"),
        ((99, 900, 104, 787), "insufficient_errors"),
    ],
)
def test_comparison_does_not_accept_tension_or_insufficient_errors(
    counts: tuple[int, int, int, int], expected: str
):
    qdk_errors, qdk_shots, stim_errors, stim_shots = counts
    totals = {
        "qdk": {"errors": qdk_errors, "shots": qdk_shots},
        "stim": {"errors": stim_errors, "shots": stim_shots},
    }
    assert compare_rates(totals, target_errors=100)["status"] == expected
