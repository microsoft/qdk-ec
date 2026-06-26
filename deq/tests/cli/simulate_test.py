# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint
"""End-to-end smoke test for `deq simulate ler` across simulator backends.

Regression test for the bug where `--simulator jit-static` panicked with
`Array index 0 out of bounds: array is empty` while `--simulator static`
worked on the same program.
"""

from pathlib import Path

import pytest

from deq.cli.simulate import simulate__ler


_TEST_PROGRAM_DEQ = """\
CODE TrivialCode [[1,1]] {
    LOGICAL X0 Z0
}

GADGET PrepareZ {
    R 0
    OUTPUT TrivialCode 0
}

GADGET Idle {
    INPUT TrivialCode 0
    OUTPUT TrivialCode 1
}

GADGET MeasureZ {
    INPUT TrivialCode 0
    M 0
    READOUT rec[-1]
}

PROGRAM TestProgram {
    PrepareZ 0
    Idle 0
    MeasureZ 0
    ASSERT_EQ rec[-1] 0
}
"""


@pytest.mark.parametrize("simulator", ["static", "jit-static"])
def test_simulate_ler_does_not_panic(tmp_path: Path, simulator: str) -> None:
    """Both simulators must run the same program to completion without panicking."""
    deq_path = tmp_path / "trivial.deq"
    deq_path.write_text(_TEST_PROGRAM_DEQ)

    simulate__ler(
        str(deq_path),
        program="TestProgram",
        save=str(tmp_path / f"out_{simulator}"),
        shots=10,
        errors=1,
        batch_size=10,
        jobs=1,
        simulator=simulator,
        seed=42,
    )
