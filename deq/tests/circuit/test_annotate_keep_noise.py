"""Tests for the ``--keep-noise`` flag of ``deq annotate``.

When the flag is on, noise instructions (``X_ERROR``, ``DEPOLARIZE1/2``,
noisy measurements, etc.) are emitted verbatim in the annotated output
and the corresponding ``ERROR(p) ...`` rows are *not* emitted.
Re-transpilation re-derives the same ERROR rows from the kept noise
instructions, so the JIT library round-trips byte-equivalently.
"""

import pytest

from deq.cli.strip_tags import strip_jit_library
from deq.circuit.parser import parse
from deq.transpiler.jit_annotate import annotate as render_annotated
from deq.transpiler.jit_library_builder import build_jit_library


_NOISY_GADGET_SRC = """
CODE C[[1,1,1]] {
    LOGICAL X0 Z0
    STABILIZER
}

GADGET Prep {
    R 0
    X_ERROR(0.05) 0
    OUTPUT C 0
}

GADGET Idle {
    INPUT C 0
    DEPOLARIZE1(0.01) 0
    OUTPUT C 0
}

GADGET Meas {
    INPUT C 0
    M(0.02) 0
    READOUT M0
}
"""


class TestKeepNoiseGadget:
    """``--keep-noise`` keeps noise verbatim and skips noise-origin ERRORs."""

    def test_default_comments_noise_and_emits_errors(self) -> None:
        rendered = render_annotated(parse(_NOISY_GADGET_SRC))
        assert "# X_ERROR(0.05) 0" in rendered
        assert "# DEPOLARIZE1(0.01) 0" in rendered
        # Default mode emits explicit ERROR(p) rows.
        assert "ERROR(0.05)" in rendered

    def test_keep_noise_keeps_verbatim_and_no_explicit_errors(self) -> None:
        rendered = render_annotated(parse(_NOISY_GADGET_SRC), keep_noise=True)
        # Noise instructions appear uncommented.
        assert "\n    X_ERROR(0.05) 0" in rendered
        assert "\n    DEPOLARIZE1(0.01) 0" in rendered
        # The noisy measurement keeps its probability argument.
        assert "M(0.02) 0" in rendered
        # No standalone ``ERROR(p) ...`` lines are emitted (substring
        # matches like ``X_ERROR(0.05)`` are excluded by checking that
        # no line *starts* with ``ERROR(`` after stripping indent).
        for line in rendered.splitlines():
            stripped = line.lstrip()
            assert not stripped.startswith("ERROR("), (
                f"unexpected explicit ERROR line under --keep-noise: {line!r}"
            )

    def test_keep_noise_round_trips_byte_equivalent(self) -> None:
        qfile = parse(_NOISY_GADGET_SRC)
        rendered = render_annotated(qfile, keep_noise=True)
        orig_lib = build_jit_library(qfile)
        anno_lib = build_jit_library(parse(rendered))
        orig_stripped, _ = strip_jit_library(orig_lib)
        anno_stripped, _ = strip_jit_library(anno_lib)
        assert (
            orig_stripped.SerializeToString()
            == anno_stripped.SerializeToString()
        )


_REPROPAGATE_TELEPORT_SRC = """
CODE Code[[4,1,2]] {
    LOGICAL X0*X2 Z0*Z1
    STABILIZER Z0*Z2 Z1*Z3 X0*X1*X2*X3
}

GADGET PrepareZero {
    R 0 1 2 3
    X_ERROR(0.05) 0 1 2 3
    MPP X0*X1*X2*X3
    OUTPUT Code 0 1 2 3
}

GADGET CNOT {
    INPUT Code 0 1 2 3
    INPUT Code 4 5 6 7
    CX 0 4 1 5 2 6 3 7
    DEPOLARIZE2(0.01) 0 4 1 5 2 6 3 7
    OUTPUT Code 0 1 2 3
    OUTPUT Code 4 5 6 7
}

GADGET MeasureX {
    INPUT Code 0 1 2 3
    MX(0.02) 0 1 2 3
    READOUT M0 M2
}

@REPROPAGATE
COMPOSE Teleport {
    INPUT Code 0
    PrepareZero 1
    CNOT 0 1
    MeasureX 0
    OUTPUT Code 1
}
"""


class TestKeepNoiseRepropagateCompose:
    """``--keep-noise`` combined with ``@REPROPAGATE`` on the user's
    teleportation example."""

    def test_keep_noise_repropagate_round_trips(self) -> None:
        qfile = parse(_REPROPAGATE_TELEPORT_SRC)
        rendered = render_annotated(qfile, keep_noise=True)
        # The composed gadget is rendered as a flat GADGET block.
        assert "GADGET Teleport {" in rendered
        # Noise instructions are present verbatim.
        assert "X_ERROR(0.05)" in rendered
        assert "DEPOLARIZE2(0.01)" in rendered
        assert "MX(0.02)" in rendered
        # Re-transpile and compare.
        orig_lib = build_jit_library(qfile)
        anno_lib = build_jit_library(parse(rendered))
        orig_stripped, _ = strip_jit_library(orig_lib)
        anno_stripped, _ = strip_jit_library(anno_lib)
        assert (
            orig_stripped.SerializeToString()
            == anno_stripped.SerializeToString()
        )

    def test_default_mode_repropagate_round_trips(self) -> None:
        """Without ``--keep-noise``, an @REPROPAGATE compose still
        round-trips: noise is commented and ERROR rows are recomputed
        from the new propagation matrix."""
        qfile = parse(_REPROPAGATE_TELEPORT_SRC)
        rendered = render_annotated(qfile, keep_noise=False)
        assert "GADGET Teleport {" in rendered
        # Noise commented out.
        assert "# X_ERROR(0.05)" in rendered
        assert "# DEPOLARIZE2(0.01)" in rendered
        # ERROR rows are present (recomputed from flat circuit).
        assert "ERROR(" in rendered
        # Round-trip verification.
        orig_lib = build_jit_library(qfile)
        anno_lib = build_jit_library(parse(rendered))
        orig_stripped, _ = strip_jit_library(orig_lib)
        anno_stripped, _ = strip_jit_library(anno_lib)
        assert (
            orig_stripped.SerializeToString()
            == anno_stripped.SerializeToString()
        )


_NON_REPROPAGATE_NOISY_COMPOSE_SRC = """
CODE C[[1,1,1]] {
    LOGICAL X0 Z0
    STABILIZER
}

GADGET Idle {
    INPUT C 0
    DEPOLARIZE1(0.01) 0
    OUTPUT C 0
}

COMPOSE Sequence {
    INPUT C 0
    Idle 0
    OUTPUT C 0
}
"""


class TestKeepNoiseNonRepropagateCompose:
    """``--keep-noise`` is orthogonal to ``@REPROPAGATE``.

    A COMPOSE whose merge-derived propagation is already consistent
    with the flat-circuit flow (i.e. no teleportation-style conditional
    correction) round-trips byte-equivalently with or without
    ``--keep-noise`` — even though the COMPOSE has noise instructions
    and lacks ``@REPROPAGATE``.
    """

    @pytest.mark.parametrize("keep_noise", [False, True])
    def test_round_trips(self, keep_noise: bool) -> None:
        qfile = parse(_NON_REPROPAGATE_NOISY_COMPOSE_SRC)
        rendered = render_annotated(qfile, keep_noise=keep_noise)
        orig_lib = build_jit_library(qfile)
        anno_lib = build_jit_library(parse(rendered))
        orig_stripped, _ = strip_jit_library(orig_lib)
        anno_stripped, _ = strip_jit_library(anno_lib)
        assert (
            orig_stripped.SerializeToString()
            == anno_stripped.SerializeToString()
        )
