"""Tests for the ``@REPROPAGATE`` decorator on COMPOSE definitions.

The decorator switches the COMPOSE build path from "compose sub-gadget
matrices via merge()" to "inline body into a flat circuit and run it
through the standard GADGET pipeline".  This lets composes whose
effective input→output Pauli flow includes conditional logical
corrections (e.g. teleportation) get propagation matrices that match
what circuit-flow analysis derives from the inlined circuit.
"""

import pytest

from deq.cli.strip_tags import strip_jit_library
from deq.circuit.parser import parse
from deq.transpiler.compose_builder import (
    compose_to_synthetic_gadget,
    has_repropagate,
)
from deq.transpiler.jit_library_builder import build_jit_library


_TELEPORTATION_SOURCE = """
CODE Code[[4,1,2]] {
    LOGICAL X0*X2 Z0*Z1
    STABILIZER Z0*Z2 Z1*Z3 X0*X1*X2*X3
}

GADGET PrepareZero {
    R 0 1 2 3
    MPP X0*X1*X2*X3
    OUTPUT Code 0 1 2 3
}

GADGET CNOT {
    INPUT Code 0 1 2 3
    INPUT Code 4 5 6 7
    CX 0 4 1 5 2 6 3 7
    OUTPUT Code 0 1 2 3
    OUTPUT Code 4 5 6 7
}

GADGET MeasureX {
    INPUT Code 0 1 2 3
    MX 0 1 2 3
    READOUT M0 M2
}
"""


class TestHasRepropagate:
    def test_present(self) -> None:
        src = (
            _TELEPORTATION_SOURCE
            + """
            @REPROPAGATE
            COMPOSE C {
                INPUT Code 0
                PrepareZero 1
                CNOT 0 1
                MeasureX 0
                OUTPUT Code 1
            }
            """
        )
        compose = next(
            d for d in parse(src).definitions if d.__class__.__name__ == "ComposeDefinition"
        )
        assert has_repropagate(compose)

    def test_absent(self) -> None:
        src = (
            _TELEPORTATION_SOURCE
            + """
            COMPOSE C {
                INPUT Code 0
                PrepareZero 1
                CNOT 0 1
                MeasureX 0
                OUTPUT Code 1
            }
            """
        )
        compose = next(
            d for d in parse(src).definitions if d.__class__.__name__ == "ComposeDefinition"
        )
        assert not has_repropagate(compose)


class TestRepropagateRejectsArguments:
    def test_decorator_with_args_rejected(self) -> None:
        src = (
            _TELEPORTATION_SOURCE
            + """
            @REPROPAGATE("yes")
            COMPOSE C {
                INPUT Code 0
                PrepareZero 1
                CNOT 0 1
                MeasureX 0
                OUTPUT Code 1
            }
            """
        )
        with pytest.raises(ValueError, match="@REPROPAGATE takes no arguments"):
            build_jit_library(parse(src))


class TestRepropagateBuildPath:
    """``build_jit_library`` runs the GADGET pipeline for @REPROPAGATE composes."""

    def test_teleportation_compose_builds(self) -> None:
        src = (
            _TELEPORTATION_SOURCE
            + """
            @REPROPAGATE
            COMPOSE Teleport {
                INPUT Code 0
                PrepareZero 1
                CNOT 0 1
                MeasureX 0
                OUTPUT Code 1
            }
            """
        )
        lib = build_jit_library(parse(src))
        names = {g.base.name for g in lib.gadget_types}
        assert "Teleport" in names

    def test_repropagate_compose_matches_handwritten_flat_gadget(self) -> None:
        """The @REPROPAGATE COMPOSE produces the same JitGadgetType (after
        tag stripping and gtype normalization) as a hand-written flat GADGET
        with the same inlined body."""
        compose_src = (
            _TELEPORTATION_SOURCE
            + """
            @REPROPAGATE
            COMPOSE Teleport {
                INPUT Code 0
                PrepareZero 1
                CNOT 0 1
                MeasureX 0
                OUTPUT Code 1
            }
            """
        )
        # Hand-written equivalent: same circuit, written as a flat GADGET.
        # The body was derived by tracing the COMPOSE inlining by hand:
        # qubit 0 = input wire 0 (4 qubits dense: 0..3),
        # qubit 1 = wire 1 (4 qubits: 4..7).
        flat_src = (
            _TELEPORTATION_SOURCE
            + """
            GADGET Teleport {
                INPUT Code 0 1 2 3
                R 4 5 6 7
                MPP X4*X5*X6*X7
                CX 0 4 1 5 2 6 3 7
                MX 0 1 2 3
                READOUT M0 M2
                OUTPUT Code 4 5 6 7
            }
            """
        )
        lib_compose = build_jit_library(parse(compose_src))
        lib_flat = build_jit_library(parse(flat_src))

        compose_gt = next(g for g in lib_compose.gadget_types if g.base.name == "Teleport")
        flat_gt = next(g for g in lib_flat.gadget_types if g.base.name == "Teleport")

        # Compare structural fields (gtype is allowed to differ since
        # the two libraries are independent).
        assert compose_gt.base.measurements == flat_gt.base.measurements
        assert list(compose_gt.base.inputs) == list(flat_gt.base.inputs)
        assert list(compose_gt.base.outputs) == list(flat_gt.base.outputs)
        assert (
            compose_gt.base.correction_propagation
            == flat_gt.base.correction_propagation
        )
        assert (
            compose_gt.base.physical_correction
            == flat_gt.base.physical_correction
        )
        assert list(compose_gt.finished_checks) == list(flat_gt.finished_checks)
        assert list(compose_gt.unfinished_checks) == list(flat_gt.unfinished_checks)


class TestComposeToSyntheticGadget:
    def test_synthetic_gadget_has_compose_name_and_no_decorators(self) -> None:
        src = (
            _TELEPORTATION_SOURCE
            + """
            @REPROPAGATE
            @GTYPE(7)
            COMPOSE T {
                INPUT Code 0
                PrepareZero 1
                CNOT 0 1
                MeasureX 0
                OUTPUT Code 1
            }
            """
        )
        qfile = parse(src)
        from deq.circuit.model import (
            CodeDefinition,
            ComposeDefinition,
            GadgetDefinition,
        )

        codes = {d.name: d for d in qfile.definitions if isinstance(d, CodeDefinition)}
        gadgets = {
            d.name: d for d in qfile.definitions if isinstance(d, GadgetDefinition)
        }
        composes = {
            d.name: d for d in qfile.definitions if isinstance(d, ComposeDefinition)
        }
        synthetic = compose_to_synthetic_gadget(composes["T"], gadgets, {}, codes)
        assert synthetic.name == "T"
        assert synthetic.decorators == []
        assert synthetic.input_ports
        assert synthetic.output_ports


class TestRepropagatePreservesMergeChecks:
    """``@REPROPAGATE`` recomputes propagation matrices and ERRORs from
    the inlined circuit, but the *check structure* (which measurements
    each finished/unfinished check XORs together, the parities, the
    finished/unfinished split) must match what the normal
    ``merge()``-based COMPOSE pipeline produces.  Otherwise users
    relying on a specific check basis from their sub-gadgets would see
    it silently change as soon as they add ``@REPROPAGATE``.
    """

    @staticmethod
    def _strip_check_tag(check):
        out = type(check)()
        out.CopyFrom(check)
        out.base.tag = ""
        return out

    @classmethod
    def _checks_equal(cls, a, b) -> bool:
        return [
            cls._strip_check_tag(c).SerializeToString() for c in a
        ] == [cls._strip_check_tag(c).SerializeToString() for c in b]

    def _assert_checks_match(self, base: str, compose_body: str, name: str) -> None:
        lib_merge = build_jit_library(parse(base + compose_body))
        lib_repro = build_jit_library(
            parse(base + "@REPROPAGATE\n" + compose_body)
        )
        gt_merge = next(g for g in lib_merge.gadget_types if g.base.name == name)
        gt_repro = next(g for g in lib_repro.gadget_types if g.base.name == name)

        assert len(gt_merge.finished_checks) == len(gt_repro.finished_checks)
        assert len(gt_merge.unfinished_checks) == len(gt_repro.unfinished_checks)
        assert self._checks_equal(
            gt_merge.finished_checks, gt_repro.finished_checks
        )
        assert self._checks_equal(
            gt_merge.unfinished_checks, gt_repro.unfinished_checks
        )

    def test_teleportation_checks_match(self) -> None:
        compose = (
            "COMPOSE Teleport {\n"
            "    INPUT Code 0\n"
            "    PrepareZero 1\n"
            "    CNOT 0 1\n"
            "    MeasureX 0\n"
            "    OUTPUT Code 1\n"
            "}\n"
        )
        self._assert_checks_match(_TELEPORTATION_SOURCE, compose, "Teleport")

    def test_simple_cycle_checks_match(self) -> None:
        base = """
        CODE C[[3,1,3]] {
            LOGICAL X0*X1*X2 Z0*Z1*Z2
            STABILIZER Z0*Z1 Z1*Z2
        }
        GADGET Idle {
            INPUT C 0 1 2
            OUTPUT C 0 1 2
        }
        GADGET Syndrome {
            INPUT C 0 1 2
            CX 0 3 1 3 1 4 2 4
            M 3 4
            OUTPUT C 0 1 2
        }
        """
        compose = (
            "COMPOSE Cycle {\n"
            "    INPUT C 0\n"
            "    Idle 0\n"
            "    Syndrome 0\n"
            "    Idle 0\n"
            "    OUTPUT C 0\n"
            "}\n"
        )
        self._assert_checks_match(base, compose, "Cycle")

    def test_repeated_syndrome_rounds_preserve_round_to_round_checks(self) -> None:
        """Multi-round syndrome extraction (the FTPrepareZ / repetition-code
        memory case from the user's repro): merge() derives weight-2
        round-to-round comparisons, while the auto plugin on the flat
        circuit derives weight-1 single-measurement checks.

        @REPROPAGATE must preserve the merge-derived weight-2 structure
        — otherwise users who add @REPROPAGATE silently lose the
        round-to-round comparison checks that decoders rely on.
        """
        base = """
        CODE C[[3,1,3]] {
            LOGICAL X0*X1*X2 Z0
            STABILIZER Z0*Z1 Z1*Z2
        }
        GADGET PrepareZ {
            R 0 1 2
            OUTPUT C 0 1 2
        }
        GADGET Syndrome {
            INPUT C 0 2 4
            R 1 3
            CX 0 1 2 3
            CX 2 1 4 3
            M 1 3
            OUTPUT C 0 2 4
        }
        """
        compose = (
            "COMPOSE FTPrepareZ {\n"
            "    PrepareZ 0\n"
            "    REPEAT 3 { Syndrome 0 }\n"
            "    OUTPUT C 0\n"
            "}\n"
        )
        # First check the property the test is meant to lock in: merge()
        # produces the round-to-round structure.  This guards against a
        # regression in the merge() path itself silently making this
        # test trivially pass.
        lib_merge = build_jit_library(parse(base + compose))
        gt_merge = next(
            g for g in lib_merge.gadget_types if g.base.name == "FTPrepareZ"
        )
        merge_finished_weights = sorted(
            len(c.measurements) for c in gt_merge.finished_checks
        )
        assert merge_finished_weights == [1, 1, 2, 2, 2, 2], (
            "regression: merge() pipeline no longer produces round-to-round "
            f"checks; got weights {merge_finished_weights!r}"
        )

        # Now the actual property: @REPROPAGATE preserves them.
        self._assert_checks_match(base, compose, "FTPrepareZ")


class TestRepropagateAnnotateRoundtrip:
    """``deq annotate`` on @REPROPAGATE composes round-trips successfully."""

    def test_annotate_then_retranspile_byte_equivalent(self) -> None:
        from deq.transpiler.jit_annotate import annotate as render_annotated

        src = (
            _TELEPORTATION_SOURCE
            + """
            @REPROPAGATE
            COMPOSE Teleport {
                INPUT Code 0
                PrepareZero 1
                CNOT 0 1
                MeasureX 0
                OUTPUT Code 1
            }
            """
        )
        qfile = parse(src)
        rendered = render_annotated(qfile)
        # The annotated COMPOSE is rendered as a regular GADGET block.
        assert "GADGET Teleport {" in rendered
        # Re-transpile and compare (after tag-stripping).
        orig_lib = build_jit_library(qfile)
        anno_lib = build_jit_library(parse(rendered))
        orig_stripped, _ = strip_jit_library(orig_lib)
        anno_stripped, _ = strip_jit_library(anno_lib)
        assert (
            orig_stripped.SerializeToString()
            == anno_stripped.SerializeToString()
        )
