"""Tests for ``deq transpile --detectors``: annotate the companion ``.stim``
with detectors/observables.

deq's standard Stim export emits only the physical circuit; ``--detectors``
additionally annotates ``DETECTOR`` / ``OBSERVABLE_INCLUDE`` derived from the
canonical (manual + auto) check model.  The key correctness property is that
every exported detector is *deterministic* under noiseless execution (which
Stim verifies when building a detector error model), so a misaligned
measurement record (the one thing that could go wrong in the local-to-global
record rebasing) is caught by ``detector_error_model()`` raising.
"""

import tempfile
from pathlib import Path

import stim

from deq.cli.jit import transpile

_FIXTURES = Path(__file__).resolve().parents[1] / "circuit"
_REP_D3 = _FIXTURES / "repetition_code" / "repetition_code_d3.deq"
# A fixture built entirely from ``@CHECKS("manual")`` gadgets, so exporting it
# exercises the hand-crafted-detector path (the rep-code fixture uses auto
# checks and would not catch a regression there).
_FLOQUET = _FIXTURES / "fixtures" / "floquet666.deq"


def _transpile_with_detectors(deq_file: Path, program: str) -> stim.Circuit:
    with tempfile.TemporaryDirectory() as tmpdir:
        out = str(Path(tmpdir) / "library.deq.jit")
        transpile(
            str(deq_file),
            out=out,
            program=program,
            jobs=1,
            skip_mako_warning=True,
            detectors=True,
        )
        return stim.Circuit.from_file(str(Path(tmpdir) / "library.stim"))


def test_memory_experiment_has_detectors_and_observable():
    circuit = _transpile_with_detectors(_REP_D3, "MemoryExperiment")
    assert circuit.num_detectors > 0
    # The repetition code protects one logical qubit.
    assert circuit.num_observables == 1


def test_detectors_are_deterministic():
    # detector_error_model() with allow_gauge_detectors=False raises if any
    # detector is non-deterministic noiselessly -- i.e. if the local->global
    # measurement-record rebasing misaligned the detectors.
    circuit = _transpile_with_detectors(_REP_D3, "MemoryExperiment")
    model = circuit.detector_error_model(
        decompose_errors=False,
        approximate_disjoint_errors=True,
        allow_gauge_detectors=False,
    )
    assert model.num_detectors == circuit.num_detectors
    assert model.num_observables == circuit.num_observables
    assert model.num_errors > 0


def test_manual_checks_are_exported():
    # floquet666's gadgets are all @CHECKS("manual"): the exported detectors
    # must be the hand-crafted checks themselves (resolved to global records),
    # deterministic under noiseless execution.  A regression that dropped or
    # re-derived manual checks would change the count or break determinism.
    circuit = _transpile_with_detectors(_FLOQUET, "Memory")
    model = circuit.detector_error_model(
        decompose_errors=False,
        approximate_disjoint_errors=True,
        allow_gauge_detectors=False,
    )
    assert circuit.num_detectors == 26
    assert circuit.num_observables == 2
    assert model.num_detectors == 26


def test_detectors_requires_program():
    # --detectors without --program has no compiled check model to annotate,
    # so it must be rejected up front.
    with tempfile.TemporaryDirectory() as tmpdir:
        out = str(Path(tmpdir) / "library.deq.jit")
        try:
            transpile(
                str(_REP_D3),
                out=out,
                program=None,
                jobs=1,
                skip_mako_warning=True,
                detectors=True,
            )
        except ValueError as exc:
            assert "--detectors requires --program" in str(exc)
        else:  # pragma: no cover - guard must fire
            raise AssertionError("expected a --detectors requires --program ValueError")
