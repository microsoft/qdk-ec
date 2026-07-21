# pylint: disable=no-member
"""Tests for check plugins on gadgets with redundant stabilizers.

Uses the repetition code [[3,1,1]] with 3 ancillae measuring Z0*Z1, Z1*Z2,
and the redundant Z0*Z2.  Verifies that the ``auto`` plugin keeps the
metacheck (weight-3 finished check) while the ``syndrome`` plugin produces
minimal per-stabilizer checks without metachecks.
"""

from deq.circuit.parser import parse
from deq.transpiler.jit_library_builder import build_jit_library

# Shared circuit: 3 ancillae measuring Z0*Z1, Z1*Z2, Z0*Z2
_REDUNDANT_BASE = """\
CODE RepetitionCode [[3,1,1]] {{
    LOGICAL X0*X1*X2 Z0*Z1*Z2
    STABILIZER Z0*Z1 Z1*Z2 Z0*Z2
}}

GADGET PrepareZ {{
    R 0 1 2
    X_ERROR(0.01) 0 1 2
    OUTPUT RepetitionCode 0 1 2
}}

{checks_decorator}
GADGET Idle {{
    INPUT RepetitionCode 0 2 4
    X_ERROR(0.01) 0 2 4
    R 1 3 5
    CX 0 1 2 3 4 5
    CX 2 1 4 3 0 5
    X_ERROR(0.01) 1 3 5
    M 1 3 5
    OUTPUT RepetitionCode 0 2 4
}}

GADGET MeasureZ {{
    INPUT RepetitionCode 0 1 2
    X_ERROR(0.01) 0 1 2
    M 0 1 2
    READOUT rec[-3] rec[-2] rec[-1]
}}

PROGRAM Simulation {{
    PrepareZ 0
    Idle 0
    MeasureZ 0
    ASSERT_EQ rec[-1] 0
}}
"""


def _build(checks: str | None = None) -> dict:
    """Build JIT library and return a dict of gadget info keyed by name."""
    if checks is None:
        deco = ""
    else:
        deco = f'@CHECKS("{checks}")'
    lib = build_jit_library(parse(_REDUNDANT_BASE.format(checks_decorator=deco)))
    result = {}
    for gt in lib.gadget_types:
        result[gt.base.name] = gt
    return result


class TestAutoPlugin:
    """@CHECKS("auto") (or no decorator): metachecks are included."""

    def test_default_includes_metacheck(self) -> None:
        gadgets = _build()
        idle = gadgets["Idle"]
        assert len(idle.finished_checks) == 4
        assert len(idle.unfinished_checks) == 3

    def test_explicit_auto_includes_metacheck(self) -> None:
        gadgets = _build("auto")
        idle = gadgets["Idle"]
        assert len(idle.finished_checks) == 4
        assert len(idle.unfinished_checks) == 3

    def test_metacheck_is_weight_3(self) -> None:
        gadgets = _build("auto")
        idle = gadgets["Idle"]
        # The metacheck involves all 3 physical measurements, no virtuals
        metachecks = [c for c in idle.finished_checks if len(c.measurements) == 3]
        assert len(metachecks) == 1
        for m in metachecks[0].measurements:
            assert not m.HasField(
                "input_port"
            ), "Metacheck measurement should be physical, not virtual"

    def test_measurement_errors_are_hyperedges(self) -> None:
        gadgets = _build("auto")
        idle = gadgets["Idle"]
        for e in idle.errors:
            total_checks = len(e.finished_checks) + len(e.unfinished_checks)
            # Ancilla (measurement) errors trigger unfinished checks;
            # with auto checks they become weight-3 hyperedges.
            if e.unfinished_checks:
                assert (
                    total_checks == 3
                ), f"Expected weight 3 for measurement error, got {total_checks}"


class TestSyndromePlugin:
    """@CHECKS("syndrome"): minimal per-stabilizer checks, no metachecks."""

    def test_no_metacheck(self) -> None:
        gadgets = _build("syndrome")
        idle = gadgets["Idle"]
        assert len(idle.finished_checks) == 3
        assert len(idle.unfinished_checks) == 3

    def test_all_finished_checks_weight_2(self) -> None:
        gadgets = _build("syndrome")
        idle = gadgets["Idle"]
        for c in idle.finished_checks:
            assert (
                len(c.measurements) == 2
            ), f"Expected weight 2, got {len(c.measurements)}"

    def test_all_unfinished_checks_weight_1(self) -> None:
        gadgets = _build("syndrome")
        idle = gadgets["Idle"]
        for c in idle.unfinished_checks:
            assert len(c.measurements) == 1

    def test_each_finished_check_has_one_virtual(self) -> None:
        gadgets = _build("syndrome")
        idle = gadgets["Idle"]
        for c in idle.finished_checks:
            virtuals = [m for m in c.measurements if m.HasField("input_port")]
            physicals = [m for m in c.measurements if not m.HasField("input_port")]
            assert len(virtuals) == 1
            assert len(physicals) == 1

    def test_no_hyperedges(self) -> None:
        gadgets = _build("syndrome")
        idle = gadgets["Idle"]
        for e in idle.errors:
            total_checks = len(e.finished_checks) + len(e.unfinished_checks)
            assert (
                total_checks <= 2
            ), f"Hyperedge found: {e.base.tag} triggers {total_checks} checks"

    def test_measurement_errors_weight_2(self) -> None:
        gadgets = _build("syndrome")
        idle = gadgets["Idle"]
        for e in idle.errors:
            # Ancilla (measurement) errors trigger unfinished checks
            if e.unfinished_checks:
                fc = len(e.finished_checks)
                uc = len(e.unfinished_checks)
                assert fc == 1 and uc == 1, (
                    f"Expected 1 fin + 1 unf for measurement error, "
                    f"got {fc} fin + {uc} unf"
                )

    def test_data_errors_weight_2(self) -> None:
        gadgets = _build("syndrome")
        idle = gadgets["Idle"]
        for e in idle.errors:
            # Data errors don't trigger unfinished checks
            if not e.unfinished_checks:
                fc = len(e.finished_checks)
                uc = len(e.unfinished_checks)
                assert fc == 2 and uc == 0, (
                    f"Expected 2 fin + 0 unf for data error, "
                    f"got {fc} fin + {uc} unf"
                )

    def test_error_count_unchanged(self) -> None:
        """Syndrome plugin should not change the number of errors."""
        gadgets_auto = _build("auto")
        gadgets_syndrome = _build("syndrome")
        assert len(gadgets_auto["Idle"].errors) == len(gadgets_syndrome["Idle"].errors)


class TestNonRedundantUnaffected:
    """auto vs syndrome should produce same results on non-redundant codes."""

    def test_non_redundant_same_auto_and_syndrome(self) -> None:
        deq_auto = """\
CODE RepetitionCode [[3,1,1]] {
    LOGICAL X0*X1*X2 Z0*Z1*Z2
    STABILIZER Z0*Z1 Z1*Z2
}

GADGET Idle {
    INPUT RepetitionCode 0 2 4
    X_ERROR(0.01) 0 2 4
    R 1 3
    CX 0 1 2 3
    CX 2 1 4 3
    X_ERROR(0.01) 1 3
    M 1 3
    OUTPUT RepetitionCode 0 2 4
}
"""
        deq_syndrome = deq_auto.replace(
            "GADGET Idle", '@CHECKS("syndrome")\nGADGET Idle'
        )
        lib_auto = build_jit_library(parse(deq_auto))
        lib_syndrome = build_jit_library(parse(deq_syndrome))

        idle_auto = next(g for g in lib_auto.gadget_types if g.base.name == "Idle")
        idle_syndrome = next(
            g for g in lib_syndrome.gadget_types if g.base.name == "Idle"
        )

        # Non-redundant code: no metachecks to differ
        assert len(idle_auto.finished_checks) == len(idle_syndrome.finished_checks)
        assert len(idle_auto.unfinished_checks) == len(idle_syndrome.unfinished_checks)
