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


@pytest.mark.parametrize(
    ("targets", "candidate", "expected"),
    [
        ("rec[-1]", [False], True),
        ("rec[-1] 0", [False], True),
        ("rec[-1] 1", [True], True),
        ("rec[-1] 1", [False], False),
        ("0", [False], True),
        ("1", [False], False),
    ],
)
def test_sample_parses_optional_constant_require_targets(
    targets, candidate, expected
) -> None:
    stim_text = f"""
PREPARE {{
    M 0
    REQUIRE {targets}
}}
"""
    _, requires = sample_cli._strip_preselect_directives(stim_text)

    assert sample_cli._require_is_satisfied(candidate, requires[0]) is expected


def test_sample_expands_repeat_blocks() -> None:
    stim_text = """
R 0 1 2
REPEAT 3 {
    CX 0 1
    M 1
    TICK
}
M 0 2
"""

    samples = sample_cli._sample_stim_text(stim_text, shots=2, seed=0)

    assert len(samples) == 2
    num_measurements = 3 + 2
    assert all(
        parse_bits(sample, num_measurements) == [0] * num_measurements
        for sample in samples
    )


def test_nested_repeat_blocks_with_prepare_directive() -> None:
    stim_text = """
REPEAT 2 {
    PREPARE {
        REPEAT 2 {
            M 0
        }
        REQUIRE rec[-1]
    }
}
"""

    expanded = sample_cli._expand_repeat_blocks(stim_text)
    samples = sample_cli._sample_stim_text(stim_text, shots=1, seed=0)

    assert expanded.count("M 0") == 4
    assert expanded.count("PREPARE {") == 2
    assert expanded.count("REQUIRE rec[-1]") == 2
    assert parse_bits(samples[0], 4) == [0] * 4


@pytest.mark.parametrize(
    ("stim_text", "message"),
    [
        ("REPEAT x {", "invalid REPEAT block header"),
        ("REPEAT 0 {\n}", "REPEAT count must be at least 1"),
        ("REPEAT 2 {", "unclosed block"),
        ("}", "unexpected closing brace"),
    ],
)
def test_expand_repeat_blocks_rejects_malformed_input(
    stim_text: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        sample_cli._expand_repeat_blocks(stim_text)
