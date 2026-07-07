"""Equivalence test for the batched FramePropagator noise-error builder.

``iter_noise_errors_with_origin`` propagates every noise mechanism through a
gadget body in a single :class:`~paulimer.FramePropagator` pass
(:func:`_batched_mechanism_flips`) instead of one :func:`walk_pauli_forward`
per mechanism.  The two must produce identical JIT error models.

Each fixture's error model is built twice -- once with the batched pass and
once with the retained per-mechanism walk (:func:`_walk_mechanism_flips`)
monkeypatched in -- and every gadget's ``Error`` rows are asserted to match.
The fixtures include resets and manual ``@CHECKS``
(``floquet666``, ``example``), so the walk's retain-``Z``-through-reset vs the
batched pass's Stim reset is exercised and shown to be invisible to the rows.
The fixtures are small, so the per-mechanism walk is cheap here (the quadratic
cost only bites on large gadgets, which is the point of the batched pass).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deq.circuit.parser import render_and_parse_files
from deq.transpiler import jit_noise_builder
from deq.transpiler.jit_library_builder import build_jit_library

_FIXTURES = Path(__file__).resolve().parents[1] / "circuit" / "fixtures"
_FIXTURE_FILES = [
    "bacon-shor.deq",
    "code422.deq",
    "example.deq",
    "floquet666.deq",
    "teleportation.deq",
    "trivial_gadgets.deq",
]


def _error_models(deq_file: Path):
    """Map gadget name -> sorted list of canonical error-row tuples."""
    qf = render_and_parse_files([str(deq_file)], skip_mako_warning=True)
    lib = build_jit_library(qf, jobs=1)
    models = {}
    for gt in lib.gadget_types:
        models[gt.base.name] = sorted(
            (
                tuple(sorted(e.base.residual)),
                tuple(sorted(e.base.readout_flips)),
                e.base.probability,
                tuple(sorted(e.finished_checks)),
                tuple(sorted(e.unfinished_checks)),
            )
            for e in gt.errors
        )
    return models


@pytest.mark.parametrize("fixture", _FIXTURE_FILES)
def test_batched_error_model_matches_walk(fixture, monkeypatch):
    batched = _error_models(_FIXTURES / fixture)
    monkeypatch.setattr(
        jit_noise_builder, "_batched_mechanism_flips", jit_noise_builder._walk_mechanism_flips
    )
    walked = _error_models(_FIXTURES / fixture)
    assert batched == walked, (
        f"{fixture}: batched FramePropagator error model diverges from the "
        f"per-mechanism walk"
    )
