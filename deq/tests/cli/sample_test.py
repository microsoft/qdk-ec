import subprocess
import sys

import pytest

import deq.cli.sample as sample_cli
from deq.cli.util import parse_bits

_preselect_deq = """
CODE Trivial [[1,1,1]] {
    LOGICAL X0 Z0
}

GADGET Prep {
    R 0 1
    MX 0
    MX 1
    PRESELECT rec[-1] rec[-2] 1
    OUTPUT Trivial 0
}

GADGET Measure {
    INPUT Trivial 0
    M 0
    READOUT rec[-1]
}

PROGRAM Simulation {
    Prep OUT(0)
    Measure IN(0)
}
"""


def test_sample_deq_with_preselect(tmp_path) -> None:
    deq_file = tmp_path / "preselect.deq"
    deq_file.write_text(_preselect_deq, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deq",
            "sample",
            str(deq_file),
            "--program",
            "Simulation",
            "--shots",
            "30",
            "--seed",
            "42",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    samples = result.stdout.splitlines()
    assert len(samples) == 30
    num_measurements = 3
    for sample in samples:
        measurements = parse_bits(sample, num_measurements)
        assert measurements[0] ^ measurements[1] == 1


def test_sample_stops_after_preselect_attempt_limit(monkeypatch) -> None:
    monkeypatch.setattr(sample_cli, "_max_preselect_attempts", 2)
    stim_text = """
PREPARE {
    R 0
    M 0
    REQUIRE !rec[-1]
}
"""

    with pytest.raises(RuntimeError, match="not satisfied after 2 attempts"):
        sample_cli._sample_stim_text(stim_text, shots=1, seed=42)
