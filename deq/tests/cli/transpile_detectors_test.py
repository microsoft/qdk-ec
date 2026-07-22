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

import pytest
import stim

from deq.cli.jit import transpile

_FIXTURES = Path(__file__).resolve().parents[1] / "circuit"
_REP_D3 = _FIXTURES / "repetition_code" / "repetition_code_d3.deq"
# A fixture built entirely from ``@CHECKS("manual")`` gadgets, so exporting it
# exercises the hand-crafted-detector path (the rep-code fixture uses auto
# checks and would not catch a regression there).
_FLOQUET = _FIXTURES / "fixtures" / "floquet666.deq"
# Surface-code (d=3) logical teleportation fixture.  Every PROGRAM here is a
# memory experiment routed through a Bell-pair teleportation, so each emits
# several logical readouts of which only the final, ``ASSERT_EQ``'d one is
# deterministic; the intermediate Bell-basis readouts are individually random.
_TELEPORT_D3 = _FIXTURES / "surface_code" / "teleportation_d3.deq"
# The teleportation PROGRAMs, covering both the ``@REPROPAGATE`` and the
# explicit-``CONDITIONAL`` correction pathways, and both COMPOSE-level and
# PROGRAM-level ``CONDITIONAL`` placement, in the Z and X logical bases.
_TELEPORT_PROGRAMS = [
    "TeleportRepropagateMemoryZ",
    "TeleportConditionalMemoryZ",
    "TeleportRepropagateMemoryX",
    "TeleportConditionalMemoryX",
    "TeleportProgramConditionalMemoryZ",
    "TeleportProgramConditionalMemoryX",
]


def _transpile_detectors_stim_text(deq_file: Path, program: str) -> str:
    """Return the raw ``.stim`` text produced by ``transpile --detectors``.
    """
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
        return Path(tmpdir, "library.stim").read_text(encoding="utf8")


def _transpile_with_detectors(deq_file: Path, program: str) -> stim.Circuit:
    stim_text = _transpile_detectors_stim_text(deq_file, program)
    return stim.Circuit(stim_text)


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


# A program that prepares but never measures is not a closed circuit. Such a
# program must be rejected with a clear ValueError (never a bare
# AssertionError from the canonicalizer's internal invariants).
def test_detectors_on_open_program_raises_value_error():
    open_program = """
CODE RepetitionCode [[3,1,3]] {
    LOGICAL X0*X1*X2 Z0
    STABILIZER Z0*Z1 Z1*Z2
}

GADGET PrepareZ {
    R 0 1 2
    OUTPUT RepetitionCode 0 1 2
}

PROGRAM Open {
    PrepareZ 0
}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        deq_file = Path(tmpdir) / "open.deq"
        deq_file.write_text(open_program, encoding="utf-8")
        out = str(Path(tmpdir) / "library.deq.jit")
        # An uncaught AssertionError here fails the test, which is exactly the
        # regression we guard against: an open program must surface as a clear
        # ValueError, whether from the compiler (dangling wires) or the
        # --detectors canonicalize backstop (closed-program requirement).
        try:
            transpile(
                str(deq_file),
                out=out,
                program="Open",
                jobs=1,
                skip_mako_warning=True,
                detectors=True,
            )
        except ValueError as exc:
            assert str(exc)
        else:  # pragma: no cover - guard must fire
            raise AssertionError("expected a ValueError for an open program")


@pytest.mark.parametrize("program", _TELEPORT_PROGRAMS)
def test_teleportation_stim_passes_validation(program: str):
    # detector_error_model(allow_gauge_detectors=False) raises "The circuit
    # contains non-deterministic observables." if a random Bell-basis readout
    # leaked into an OBSERVABLE_INCLUDE (the exact panic this feature avoids),
    # and likewise rejects misaligned detectors.  Building it without error is
    # precisely the "passes stim validation" property requested.
    circuit = _transpile_with_detectors(_TELEPORT_D3, program)
    assert circuit.num_observables == 1
    assert circuit.num_detectors > 0
    model = circuit.detector_error_model(
        decompose_errors=False,
        approximate_disjoint_errors=True,
        allow_gauge_detectors=False,
    )
    assert model.num_observables == 1
    assert model.num_detectors == circuit.num_detectors


def test_annotation_preserves_comments_and_records_assert_source():
    # The annotator appends detectors/observables as text instead of
    # round-tripping through stim.Circuit, so the exported .stim keeps the
    # body's per-gadget header comments and gains a ``# ASSERT_EQ ...``
    # comment before each OBSERVABLE_INCLUDE for traceability.
    text = _transpile_detectors_stim_text(
        _TELEPORT_D3, "TeleportProgramConditionalMemoryZ"
    )
    lines = text.splitlines()
    # A per-gadget header comment from the body survived the annotation.
    assert any(line.startswith("# G1:") for line in lines)
    # The ASSERT_EQ source is emitted as the comment directly above its
    # OBSERVABLE_INCLUDE line.
    obs_index = next(
        i for i, line in enumerate(lines) if line.startswith("OBSERVABLE_INCLUDE(0)")
    )
    assert lines[obs_index - 1] == "# ASSERT_EQ rec[-1] 0"
