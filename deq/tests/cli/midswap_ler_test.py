"""Fixed-seed logical-error-rate regression for the mid-SWAP surface code."""

from __future__ import annotations

import json
import math
from pathlib import Path

import deq.proto.deq_jit_pb2 as jit_pb
from deq.cli.jit import transpile
from deq.cli.simulate import _run_batch
from deq.compiler.jit_compiler import static_jit_compiler


MIDSWAP_DEQ = (
    Path(__file__).resolve().parents[1]
    / "circuit"
    / "surface_code"
    / "mid_swap_surface_code.deq"
)
PROGRAM = "MidSwapMemory"
SHOTS = 10_000


def _build_mid_swap(tmp_path: Path) -> tuple[Path, Path, Path]:
    jit_path = tmp_path / f"{PROGRAM}.deq.jit"
    stim_path = tmp_path / f"{PROGRAM}.stim"
    bin_path = tmp_path / f"{PROGRAM}.deq.bin"
    transpile(
        str(MIDSWAP_DEQ),
        out=str(jit_path),
        program=PROGRAM,
        jobs=1,
        mako=["d=3", "rounds=3", "p_loss=0.005", "p=0.001"],
        skip_mako_warning=True,
    )
    library = jit_pb.JitLibrary.FromString(jit_path.read_bytes())
    bin_path.write_bytes(static_jit_compiler(library).SerializeToString())
    return jit_path, stim_path, bin_path


def _measure(
    jit_path: Path,
    stim_path: Path,
    bin_path: Path,
    coordinator_config: dict[str, object],
) -> tuple[int, int]:
    result = _run_batch(
        bin_path=str(bin_path),
        stim_path=str(stim_path),
        jit_path=str(jit_path),
        batch_size=SHOTS,
        max_errors=SHOTS + 1,
        decoder="black-box-tesseract",
        decoder_config=None,
        coordinator="monolithic",
        coordinator_config=json.dumps(coordinator_config),
        seed=1,
        debug_dir=None,
        simulator="qdk",
        timeout=120,
    )
    return int(result["shots"]), int(result["logical_errors"])


def _pooled_z_score(baseline: tuple[int, int], envelope: tuple[int, int]) -> float:
    baseline_shots, baseline_errors = baseline
    envelope_shots, envelope_errors = envelope
    pooled = (baseline_errors + envelope_errors) / (baseline_shots + envelope_shots)
    standard_error = math.sqrt(
        pooled * (1.0 - pooled) * (1.0 / baseline_shots + 1.0 / envelope_shots)
    )
    return (
        baseline_errors / baseline_shots - envelope_errors / envelope_shots
    ) / standard_error


def test_mid_swap_envelope_matching_beats_random_imputation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Protect the fixed-seed 5-sigma envelope advantage at d=3.

    Each arm runs exactly 10,000 shots. The initial measurement on the current
    implementation was 70 baseline errors versus 5 envelope errors (7.52 sigma),
    leaving margin above the asserted threshold without making exact RNG output
    part of the contract.
    """
    for variable in (
        "RAYON_NUM_THREADS",
        "TOKIO_WORKER_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        monkeypatch.setenv(variable, "1")
    jit_path, stim_path, bin_path = _build_mid_swap(tmp_path)
    common = {
        "loss_random_imputation": True,
        "loss_random_imputation_seed": 7,
        "merge_hyperedges": True,
        "persistent_decoder": True,
    }
    baseline = _measure(
        jit_path,
        stim_path,
        bin_path,
        {**common, "loss_strategy": "ignore"},
    )
    envelope = _measure(
        jit_path,
        stim_path,
        bin_path,
        {
            **common,
            "loss_strategy": "reweight",
            "loss_config": {"weight_fraction": 0.5, "scale": "local"},
        },
    )

    assert baseline[0] == envelope[0] == SHOTS
    assert baseline[1] > envelope[1], (
        f"expected fewer envelope errors; baseline={baseline[1]}, "
        f"envelope={envelope[1]}"
    )
    z_score = _pooled_z_score(baseline, envelope)
    assert z_score >= 5.0, (
        f"expected envelope matching to beat random imputation by at least 5 sigma; "
        f"baseline={baseline[1]}/{baseline[0]}, "
        f"envelope={envelope[1]}/{envelope[0]}, z={z_score:.2f}"
    )


def test_window_loss_compilation_accepts_an_incoming_boundary_connector(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A radius-1 interior window may omit a retained gadget's predecessor."""
    monkeypatch.setenv("RAYON_NUM_THREADS", "1")
    monkeypatch.setenv("TOKIO_WORKER_THREADS", "2")
    jit_path = tmp_path / f"{PROGRAM}.deq.jit"
    stim_path = tmp_path / f"{PROGRAM}.stim"
    bin_path = tmp_path / f"{PROGRAM}.deq.bin"
    transpile(
        str(MIDSWAP_DEQ),
        out=str(jit_path),
        program=PROGRAM,
        jobs=1,
        mako=[
            "d=3",
            "rounds=5",
            "p_loss=0.1",
            "p=0",
            "layout=windowed",
        ],
        skip_mako_warning=True,
    )
    library = jit_pb.JitLibrary.FromString(jit_path.read_bytes())
    bin_path.write_bytes(static_jit_compiler(library).SerializeToString())

    result = _run_batch(
        bin_path=str(bin_path),
        stim_path=str(stim_path),
        jit_path=str(jit_path),
        batch_size=1,
        max_errors=2,
        decoder="black-box-tesseract",
        decoder_config=None,
        coordinator="window",
        coordinator_config=json.dumps(
            {
                "buffer_radius": 1,
                "lookahead_radius": 0,
                "loss_strategy": "reweight",
                "loss_random_imputation_seed": 7,
            }
        ),
        seed=1,
        debug_dir=None,
        simulator="qdk",
        timeout=120,
    )

    assert result["shots"] == 1
