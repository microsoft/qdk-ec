"""Tests for the simulation/decoder split emitted by ``deq annotate``.

Physical noise is retained under ``@SIMULATE_ONLY`` and the canonical
``ERROR(p) ...`` rows are emitted explicitly. Noisy measurements receive clean
``@DECODE_ONLY`` twins so both views have the same records.
"""

import pytest

from deq.cli.strip_tags import strip_jit_library
from deq.circuit.model import GadgetDefinition, Instruction
from deq.circuit.parser import parse
from deq.transpiler.jit_annotate import annotate as render_annotated
from deq.transpiler.jit_library_builder import build_jit_library
from deq.transpiler.jit_transpiler import flatten_body

_NOISY_GADGET_SRC = """
CODE C[[1,1,1]] {
    LOGICAL X0 Z0
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


class TestSplitViewGadget:
    """Annotation separates simulation noise from decoder metadata."""

    def test_splits_simulation_noise_and_decoder_errors(self) -> None:
        rendered = render_annotated(parse(_NOISY_GADGET_SRC))
        assert "@SIMULATE_ONLY\n    X_ERROR(0.05) 0" in rendered
        assert "@SIMULATE_ONLY\n    DEPOLARIZE1(0.01) 0" in rendered
        assert (
            "@SIMULATE_ONLY\n    M(0.02) 0\n"
            "    @DECODE_ONLY\n    M 0"
        ) in rendered
        assert "\n    ERROR(" in rendered

        prep = rendered.split("GADGET Prep {", 1)[1].split("\n}", 1)[0]
        assert prep.index("X_ERROR(0.05) 0") < prep.index("ERROR(0.05)")
        assert prep.index("ERROR(0.05)") < prep.index("OUTPUT C 0")

    def test_declared_and_noise_errors_stay_at_source_positions(self) -> None:
        rendered = render_annotated(
            parse(
                """
                CODE C[[1,1,1]] { LOGICAL X0 Z0 }
                GADGET G {
                    INPUT C 0
                    X_ERROR(0.01) 0
                    ERROR(0.02) LX0
                    H 0
                    OUTPUT C 0
                }
                """
            )
        )
        gadget = rendered.split("GADGET G {", 1)[1].split("\n}", 1)[0]

        positions = [
            gadget.index("X_ERROR(0.01) 0"),
            gadget.index("ERROR(0.01)"),
            gadget.index("ERROR(0.02) LX0  # E1"),
            gadget.index("H 0"),
        ]
        assert positions == sorted(positions)

    def test_round_trips_byte_equivalent(self) -> None:
        qfile = parse(_NOISY_GADGET_SRC)
        rendered = render_annotated(qfile)
        orig_lib = build_jit_library(qfile)
        anno_lib = build_jit_library(parse(rendered))
        orig_stripped, _ = strip_jit_library(orig_lib)
        anno_stripped, _ = strip_jit_library(anno_lib)
        assert orig_stripped.SerializeToString() == anno_stripped.SerializeToString()

    def test_annotation_is_idempotent(self) -> None:
        qfile = parse(_NOISY_GADGET_SRC)
        rendered_once = render_annotated(qfile)
        rendered_twice = render_annotated(parse(rendered_once))

        assert rendered_twice.count("X_ERROR(0.05) 0") == 1
        assert rendered_twice.count("DEPOLARIZE1(0.01) 0") == 1
        assert rendered_twice.count("M(0.02) 0") == 1

        original, _ = strip_jit_library(build_jit_library(qfile))
        rebuilt, _ = strip_jit_library(build_jit_library(parse(rendered_twice)))
        assert original.SerializeToString() == rebuilt.SerializeToString()

    def test_trailing_simulation_only_instruction_is_preserved(self) -> None:
        rendered = render_annotated(
            parse(
                """
                CODE C[[1,1,1]] { LOGICAL X0 Z0 }
                GADGET G {
                    INPUT C 0
                    @SIMULATE_ONLY
                    X_ERROR(0.01) 0
                }
                """
            )
        )
        gadget = rendered.split("GADGET G {", 1)[1].split("\n}", 1)[0]
        assert gadget.index("INPUT C 0") < gadget.index("@SIMULATE_ONLY")
        assert gadget.index("@SIMULATE_ONLY") < gadget.index("X_ERROR(0.01) 0")

        annotated = parse(rendered)
        rendered_twice = render_annotated(annotated)
        assert rendered_twice.count("X_ERROR(0.01) 0") == 1

    def test_error_index_comments_survive_second_annotation(self) -> None:
        qfile = parse(
            """
            CODE C[[1,1,1]] { LOGICAL X0 Z0 }
            GADGET G {
                INPUT C 0
                X_ERROR(0.01) 0
                H 0
                ERROR(0.02) LX0
                OUTPUT C 0
            }
            """
        )

        rendered_once = render_annotated(qfile)
        rendered_twice = render_annotated(parse(rendered_once))
        once_errors = [
            line.strip()
            for line in rendered_once.splitlines()
            if line.lstrip().startswith("ERROR(")
        ]
        twice_errors = [
            line.strip()
            for line in rendered_twice.splitlines()
            if line.lstrip().startswith("ERROR(")
        ]

        assert once_errors == twice_errors
        assert once_errors[0].endswith("# E0")
        assert once_errors[1].endswith("# E1")

    def test_noisy_measurement_has_one_instruction_in_each_view(self) -> None:
        rendered = render_annotated(parse(_NOISY_GADGET_SRC))
        qfile = parse(rendered)
        gadget = next(
            definition
            for definition in qfile.definitions
            if isinstance(definition, GadgetDefinition) and definition.name == "Meas"
        )

        decode_measurements = [
            str(statement)
            for statement in flatten_body(gadget.body)
            if isinstance(statement, Instruction) and statement.name.upper() == "M"
        ]
        simulation_measurements = [
            str(statement)
            for statement in flatten_body(gadget.body, for_simulate=True)
            if isinstance(statement, Instruction) and statement.name.upper() == "M"
        ]
        assert decode_measurements == ["M 0"]
        assert simulation_measurements == ["M(0.02) 0"]

    def test_existing_noise_decorators_preserve_view_intent(self) -> None:
        qfile = parse(
            """
            CODE C[[1,1,1]] { LOGICAL X0 Z0 }
            GADGET G {
                INPUT C 0
                @SIMULATE_ONLY
                X_ERROR(0.01) 0
                @DECODE_ONLY
                Z_ERROR(0.02) 0
                OUTPUT C 0
            }
            """
        )

        rendered = render_annotated(qfile)
        annotated = parse(rendered)
        gadget = next(
            definition
            for definition in annotated.definitions
            if isinstance(definition, GadgetDefinition)
        )
        decode_instructions = [
            str(statement)
            for statement in flatten_body(gadget.body)
            if isinstance(statement, Instruction)
        ]
        simulation_instructions = [
            str(statement)
            for statement in flatten_body(gadget.body, for_simulate=True)
            if isinstance(statement, Instruction)
        ]

        assert "X_ERROR(0.01) 0" in simulation_instructions
        assert "X_ERROR(0.01) 0" not in decode_instructions
        assert "Z_ERROR(0.02) 0" not in simulation_instructions
        assert "Z_ERROR(0.02) 0" not in decode_instructions
        assert "ERROR(0.02) OUT0.LZ0  # E0" in rendered

        original, _ = strip_jit_library(build_jit_library(qfile))
        rebuilt, _ = strip_jit_library(build_jit_library(annotated))
        assert original.SerializeToString() == rebuilt.SerializeToString()

    def test_decorated_noisy_measurements_preserve_each_view(self) -> None:
        qfile = parse(
            """
            CODE C[[1,1,1]] { LOGICAL X0 Z0 }
            GADGET G {
                INPUT C 0
                @SIMULATE_ONLY
                M(0.03) 0
                @DECODE_ONLY
                M(0.04) 0
                READOUT M0
            }
            """
        )

        rendered = render_annotated(qfile)
        annotated = parse(rendered)
        gadget = next(
            definition
            for definition in annotated.definitions
            if isinstance(definition, GadgetDefinition)
        )
        decode_measurements = [
            str(statement)
            for statement in flatten_body(gadget.body)
            if isinstance(statement, Instruction) and statement.name.upper() == "M"
        ]
        simulation_measurements = [
            str(statement)
            for statement in flatten_body(gadget.body, for_simulate=True)
            if isinstance(statement, Instruction) and statement.name.upper() == "M"
        ]

        assert decode_measurements == ["M 0"]
        assert simulation_measurements == ["M(0.03) 0"]
        assert "ERROR(0.04) R0  # E0" in rendered

        original, _ = strip_jit_library(build_jit_library(qfile))
        rebuilt, _ = strip_jit_library(build_jit_library(annotated))
        assert original.SerializeToString() == rebuilt.SerializeToString()

    def test_decode_only_noisy_measurement_stays_decode_only_and_clean(self) -> None:
        qfile = parse(
            """
            CODE C[[1,1,1]] { LOGICAL X0 Z0 }
            GADGET G {
                INPUT C 0
                @SIMULATE_ONLY
                M 0
                @DECODE_ONLY
                M(0.04) 0
                READOUT M0
            }
            """
        )

        rendered = render_annotated(qfile)
        annotated = parse(rendered)
        gadget = next(
            definition
            for definition in annotated.definitions
            if isinstance(definition, GadgetDefinition)
        )
        decode_measurements = [
            str(statement)
            for statement in flatten_body(gadget.body)
            if isinstance(statement, Instruction) and statement.name.upper() == "M"
        ]
        simulation_measurements = [
            str(statement)
            for statement in flatten_body(gadget.body, for_simulate=True)
            if isinstance(statement, Instruction) and statement.name.upper() == "M"
        ]

        assert decode_measurements == ["M 0"]
        assert simulation_measurements == ["M 0"]
        assert "@SIMULATE_ONLY\n    M(0.04) 0" not in rendered
        assert "@DECODE_ONLY\n    M 0" in rendered
        assert "ERROR(0.04) R0  # E0" in rendered
        assert rendered.index("@DECODE_ONLY\n    M 0") < rendered.index("# E0")
        assert rendered.index("# E0") < rendered.index("READOUT M0")

        original, _ = strip_jit_library(build_jit_library(qfile))
        rebuilt, _ = strip_jit_library(build_jit_library(annotated))
        assert original.SerializeToString() == rebuilt.SerializeToString()

    def test_unpaired_decode_only_measurement_is_rejected(self) -> None:
        qfile = parse(
            """
            CODE C[[1,1,1]] { LOGICAL X0 Z0 }
            GADGET G {
                INPUT C 0
                @DECODE_ONLY
                M(0.04) 0
                READOUT M0
            }
            """
        )

        with pytest.raises(ValueError, match="mismatched measurement counts"):
            render_annotated(qfile)

    def test_existing_non_noise_decorators_are_preserved(self) -> None:
        qfile = parse(
            """
            CODE C[[1,1,1]] { LOGICAL X0 Z0 }
            GADGET G {
                INPUT C 0
                @SIMULATE_ONLY
                H 0
                @DECODE_ONLY
                S 0
                OUTPUT C 0
            }
            """
        )

        rendered = render_annotated(qfile)
        assert "@SIMULATE_ONLY\n    H 0" in rendered
        assert "@DECODE_ONLY\n    S 0" in rendered

        original, _ = strip_jit_library(build_jit_library(qfile))
        rebuilt, _ = strip_jit_library(build_jit_library(parse(rendered)))
        assert original.SerializeToString() == rebuilt.SerializeToString()

    def test_program_stim_export_uses_only_simulation_noise(self) -> None:
        from deq.cli.jit import compile_program_for_jit, export_program_stim
        from deq.circuit.model import ProgramDefinition

        qfile = parse(
            """
            CODE C[[1,1,1]] { LOGICAL X0 Z0 }
            GADGET Prep {
                R 0
                X_ERROR(0.01) 0
                LOSS_ERROR(0.1) 0
                OUTPUT C 0
            }
            GADGET Meas {
                INPUT C 0
                M(0.02) 0
                READOUT M0
            }
            PROGRAM Run {
                Prep OUT(0)
                Meas IN(0)
            }
            """
        )
        annotated = parse(render_annotated(qfile))
        library = build_jit_library(annotated)
        program = next(
            definition
            for definition in annotated.definitions
            if isinstance(definition, ProgramDefinition)
        )
        gadgets = {
            definition.name: definition
            for definition in annotated.definitions
            if isinstance(definition, GadgetDefinition)
        }
        compiled, assertions = compile_program_for_jit(library, program)
        for instruction, _ in compiled:
            library.program.append(instruction)

        stim_text = export_program_stim(
            library,
            gadgets,
            {gadget.base.gtype: gadget.base.name for gadget in library.gadget_types},
            flatten_body,
            program,
            [application for _, application in compiled],
            assertions,
        )

        assert stim_text.count("X_ERROR(0.01)") == 1
        assert stim_text.count("LOSS_ERROR(0.1)") == 1
        assert stim_text.count("M(0.02)") == 1
        assert "\nM 0\n" not in stim_text

    def test_annotated_compose_stim_export_preserves_decorated_views(
        self, tmp_path
    ) -> None:
        from deq.cli.jit import jit_compile_program_to_file

        qfile = parse(
            """
            CODE C[[1,1,1]] { LOGICAL X0 Z0 }
            GADGET Prep {
                R 0
                @SIMULATE_ONLY
                X_ERROR(0.01) 0
                @DECODE_ONLY
                Z_ERROR(0.02) 0
                OUTPUT C 0
            }
            COMPOSE Chain {
                Prep 0
                OUTPUT C 0
            }
            GADGET Sink {
                INPUT C 0
                M 0
                READOUT M0
            }
            PROGRAM Run {
                Chain OUT(0)
                Sink IN(0)
            }
            """
        )
        annotated = parse(render_annotated(qfile))
        output_path = tmp_path / "run.deq.jit"
        jit_compile_program_to_file(
            build_jit_library(annotated),
            annotated,
            str(output_path),
            program="Run",
        )
        stim_text = (tmp_path / "run.stim").read_text(encoding="utf-8")

        assert stim_text.count("X_ERROR(0.01)") == 1
        assert "Z_ERROR(0.02)" not in stim_text


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


class TestSplitViewRepropagateCompose:
    """Split-view annotation combined with ``@REPROPAGATE``."""

    def test_repropagate_round_trips(self) -> None:
        qfile = parse(_REPROPAGATE_TELEPORT_SRC)
        rendered = render_annotated(qfile)
        # The composed gadget is rendered as a flat GADGET block.
        assert "GADGET Teleport {" in rendered
        # Noise instructions are present verbatim.
        assert "X_ERROR(0.05)" in rendered
        assert "DEPOLARIZE2(0.01)" in rendered
        assert "MX(0.02)" in rendered
        teleport = rendered.split("GADGET Teleport {", 1)[1].split("\n}", 1)[0]
        assert teleport.index("X_ERROR(0.05)") < teleport.index("ERROR(0.05)")
        # Re-transpile and compare.
        orig_lib = build_jit_library(qfile)
        anno_lib = build_jit_library(parse(rendered))
        orig_stripped, _ = strip_jit_library(orig_lib)
        anno_stripped, _ = strip_jit_library(anno_lib)
        assert orig_stripped.SerializeToString() == anno_stripped.SerializeToString()

    def test_repropagate_preserves_decorated_measurement_views(self) -> None:
        qfile = parse(
            """
            CODE C[[1,1,1]] { LOGICAL X0 Z0 }
            GADGET Measure {
                INPUT C 0
                @SIMULATE_ONLY
                M(0.03) 0
                @DECODE_ONLY
                M(0.04) 0
                READOUT M0
                OUTPUT C 0
            }
            @REPROPAGATE
            COMPOSE Chain {
                INPUT C 0
                Measure 0
                OUTPUT C 0
            }
            """
        )

        rendered = render_annotated(qfile)
        chain = rendered.split("GADGET Chain {", 1)[1]
        assert "@SIMULATE_ONLY\n    M(0.03) 0" in chain
        assert "@DECODE_ONLY\n    M 0" in chain
        assert "M(0.04) 0" not in chain
        assert "ERROR(0.04) R0 OUT0.LX0  # E0" in chain

        original, _ = strip_jit_library(build_jit_library(qfile))
        rebuilt, _ = strip_jit_library(build_jit_library(parse(rendered)))
        assert original.SerializeToString() == rebuilt.SerializeToString()

_NON_REPROPAGATE_NOISY_COMPOSE_SRC = """
CODE C[[1,1,1]] {
    LOGICAL X0 Z0
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


class TestSplitViewNonRepropagateCompose:
    """A normal COMPOSE retains simulation noise and canonical errors."""

    def test_round_trips(self) -> None:
        qfile = parse(_NON_REPROPAGATE_NOISY_COMPOSE_SRC)
        rendered = render_annotated(qfile)
        orig_lib = build_jit_library(qfile)
        anno_lib = build_jit_library(parse(rendered))
        orig_stripped, _ = strip_jit_library(orig_lib)
        anno_stripped, _ = strip_jit_library(anno_lib)
        assert orig_stripped.SerializeToString() == anno_stripped.SerializeToString()

    def test_missing_provenance_is_rejected(
        self, monkeypatch
    ) -> None:
        import deq.transpiler.jit_annotate as annotate_module
        from deq.transpiler.jit_library_builder import JitGadgetArtifacts

        qfile = parse(_NON_REPROPAGATE_NOISY_COMPOSE_SRC)
        original_build = annotate_module.build_jit_library_artifacts

        def without_compose_provenance(qfile):
            artifacts = original_build(qfile)
            composed = artifacts.gadget_artifacts_by_name["Sequence"]
            artifacts.gadget_artifacts_by_name["Sequence"] = JitGadgetArtifacts(
                jit_type=composed.jit_type
            )
            return artifacts

        monkeypatch.setattr(
            annotate_module,
            "build_jit_library_artifacts",
            without_compose_provenance,
        )
        with pytest.raises(AssertionError, match="error provenance is incomplete"):
            annotate_module.annotate(qfile)

    def test_preserves_existing_split_view_noise(self) -> None:
        qfile = parse(
            """
            CODE C[[1,1,1]] { LOGICAL X0 Z0 }
            GADGET Split {
                INPUT C 0
                @SIMULATE_ONLY
                X_ERROR(0.01) 0
                @DECODE_ONLY
                Z_ERROR(0.02) 0
                OUTPUT C 0
            }
            COMPOSE Chain {
                INPUT C 0
                Split 0
                OUTPUT C 0
            }
            """
        )

        rendered = render_annotated(qfile)
        chain = rendered.split("GADGET Chain {", 1)[1]
        assert "@SIMULATE_ONLY\n    X_ERROR(0.01) 0" in chain
        assert "@SIMULATE_ONLY\n    Z_ERROR(0.02) 0" not in chain
        assert "\n    ERROR(0.02) OUT0.LZ0  # E0" in chain

        original, _ = strip_jit_library(build_jit_library(qfile))
        rebuilt, _ = strip_jit_library(build_jit_library(parse(rendered)))
        assert original.SerializeToString() == rebuilt.SerializeToString()

    def test_simulation_only_child_does_not_shift_later_error(self) -> None:
        qfile = parse(
            """
            CODE C[[1,1,1]] { LOGICAL X0 Z0 }
            GADGET SimulationOnly {
                INPUT C 0
                @SIMULATE_ONLY
                X_ERROR(0.01) 0
                OUTPUT C 0
            }
            GADGET Noisy {
                INPUT C 0
                Z_ERROR(0.02) 0
                OUTPUT C 0
            }
            COMPOSE Chain {
                INPUT C 0
                SimulationOnly 0
                Noisy 0
                OUTPUT C 0
            }
            """
        )

        rendered = render_annotated(qfile)
        chain = rendered.split("GADGET Chain {", 1)[1]
        assert "@SIMULATE_ONLY\n    X_ERROR(0.01) 0" in chain
        assert chain.index("Z_ERROR(0.02) 0") < chain.index("# E0")
        assert chain.index("# E0") < chain.index("OUTPUT C 0")

        original, _ = strip_jit_library(build_jit_library(qfile))
        rebuilt, _ = strip_jit_library(build_jit_library(parse(rendered)))
        assert original.SerializeToString() == rebuilt.SerializeToString()

    def test_nested_compose_preserves_existing_split_view_noise(self) -> None:
        qfile = parse(
            """
            CODE C[[1,1,1]] { LOGICAL X0 Z0 }
            GADGET Split {
                INPUT C 0
                @SIMULATE_ONLY
                X_ERROR(0.01) 0
                @DECODE_ONLY
                Z_ERROR(0.02) 0
                OUTPUT C 0
            }
            COMPOSE Inner {
                INPUT C 0
                Split 0
                OUTPUT C 0
            }
            COMPOSE Outer {
                INPUT C 0
                Inner 0
                OUTPUT C 0
            }
            """
        )

        rendered = render_annotated(qfile)
        outer = rendered.split("GADGET Outer {", 1)[1]
        assert "@SIMULATE_ONLY\n    X_ERROR(0.01) 0" in outer
        assert "@SIMULATE_ONLY\n    Z_ERROR(0.02) 0" not in outer
        assert "ERROR(0.02) OUT0.LZ0  # E0" in outer
        assert outer.index("Z_ERROR(0.02) 0") < outer.index("# E0")
        assert outer.index("# E0") < outer.index("OUTPUT C 0")

        original, _ = strip_jit_library(build_jit_library(qfile))
        rebuilt, _ = strip_jit_library(build_jit_library(parse(rendered)))
        assert original.SerializeToString() == rebuilt.SerializeToString()


_PASSTHROUGH_LOSS_SRC = """
CODE C[[1,1,1]] {
    LOGICAL X0 Z0
}

GADGET Prep {
    R 0
    X_ERROR(0.05) 0
    LOSS_ERROR(0.5) 0
    OUTPUT C 0
}

GADGET Meas {
    INPUT C 0
    LOSS_ERROR(0.3) 0
    M 0
    READOUT M0
}
"""


_PASSTHROUGH_LOSS_COMPOSE_SRC = """
CODE C[[1,1,1]] {
    LOGICAL X0 Z0
}

GADGET Idle {
    INPUT C 0
    DEPOLARIZE1(0.01) 0
    LOSS_ERROR(0.4) 0
    OUTPUT C 0
}

COMPOSE Sequence {
    INPUT C 0
    Idle 0
    OUTPUT C 0
}
"""


_DECLARED_LOSS_NOISY_COMPOSE_SRC = """
CODE C[[1,1,1]] {
    LOGICAL X0 Z0
}

GADGET Declared {
    INPUT C 0
    OUTPUT C 0
    ERROR(0) LX0
    LOSS(0.1) SE0 CE0 OUT0.L0
    LOSS(IN0.L0) CE0 L0
}

GADGET Noisy {
    INPUT C 0
    X_ERROR(0.01) 0
    OUTPUT C 0
}

COMPOSE Chain {
    INPUT C 0
    Declared 0
    Noisy 0
    OUTPUT C 0
}
"""


class TestSplitViewPassthroughLoss:
    """``LOSS_ERROR`` remains physical while explicit loss metadata decodes."""

    def test_loss_error_is_split_from_decoder_metadata(self) -> None:
        rendered = render_annotated(parse(_PASSTHROUGH_LOSS_SRC))
        assert "@SIMULATE_ONLY\n    LOSS_ERROR(0.5) 0" in rendered
        assert "@SIMULATE_ONLY\n    LOSS_ERROR(0.3) 0" in rendered
        assert "\n    LOSS(0.5)" in rendered
        assert "ERROR(0.05)" in rendered

    def test_empty_composed_loss_model_is_rejected(self, monkeypatch) -> None:
        import deq.transpiler.jit_annotate as annotate_module

        qfile = parse(_PASSTHROUGH_LOSS_COMPOSE_SRC)
        original_build = annotate_module.build_jit_library_artifacts

        def with_empty_loss_model(qfile):
            artifacts = original_build(qfile)
            loss_model = artifacts.gadget_artifacts_by_name[
                "Sequence"
            ].jit_type.base.loss_model
            loss_model.Clear()
            loss_model.SetInParent()
            return artifacts

        monkeypatch.setattr(
            annotate_module,
            "build_jit_library_artifacts",
            with_empty_loss_model,
        )
        with pytest.raises(AssertionError, match="has an empty loss model"):
            annotate_module.annotate(qfile)

    def test_loss_error_round_trips_byte_equivalent(self) -> None:
        qfile = parse(_PASSTHROUGH_LOSS_SRC)
        rendered = render_annotated(qfile)
        orig_lib = build_jit_library(qfile)
        anno_lib = build_jit_library(parse(rendered))
        orig_stripped, _ = strip_jit_library(orig_lib)
        anno_stripped, _ = strip_jit_library(anno_lib)
        assert orig_stripped.SerializeToString() == anno_stripped.SerializeToString()

    def test_loss_annotation_is_idempotent(self) -> None:
        qfile = parse(_PASSTHROUGH_LOSS_SRC)
        rendered_once = render_annotated(qfile)
        rendered_twice = render_annotated(parse(rendered_once))

        assert rendered_twice.count("LOSS_ERROR(0.5) 0") == 1
        assert rendered_twice.count("LOSS_ERROR(0.3) 0") == 1

        original, _ = strip_jit_library(build_jit_library(qfile))
        rebuilt, _ = strip_jit_library(build_jit_library(parse(rendered_twice)))
        assert original.SerializeToString() == rebuilt.SerializeToString()

    def test_decorated_loss_errors_preserve_view_intent(self) -> None:
        qfile = parse(
            """
            CODE C[[1,1,1]] { LOGICAL X0 Z0 }
            GADGET G {
                INPUT C 0
                @SIMULATE_ONLY
                LOSS_ERROR(0.11) 0
                @DECODE_ONLY
                LOSS_ERROR(0.22) 0
                OUTPUT C 0
            }
            """
        )

        rendered = render_annotated(qfile)
        annotated = parse(rendered)
        gadget = next(
            definition
            for definition in annotated.definitions
            if isinstance(definition, GadgetDefinition)
        )
        decode_instructions = [
            str(statement)
            for statement in flatten_body(gadget.body)
            if isinstance(statement, Instruction)
        ]
        simulation_instructions = [
            str(statement)
            for statement in flatten_body(gadget.body, for_simulate=True)
            if isinstance(statement, Instruction)
        ]

        assert "LOSS_ERROR(0.11) 0" in simulation_instructions
        assert "LOSS_ERROR(0.22) 0" not in simulation_instructions
        assert not any(instruction.startswith("LOSS_ERROR") for instruction in decode_instructions)
        assert "LOSS(0.22)" in rendered

        original, _ = strip_jit_library(build_jit_library(qfile))
        rebuilt, _ = strip_jit_library(build_jit_library(annotated))
        assert original.SerializeToString() == rebuilt.SerializeToString()

    def test_loss_error_is_split_inside_compose(self) -> None:
        qfile = parse(_PASSTHROUGH_LOSS_COMPOSE_SRC)
        rendered = render_annotated(qfile)
        assert "GADGET Sequence {" in rendered
        sequence_block = rendered.split("GADGET Sequence {", 1)[1]
        assert "\n    LOSS(0.4)" in sequence_block
        assert "@SIMULATE_ONLY\n    DEPOLARIZE1(0.01) 0" in sequence_block
        assert "@SIMULATE_ONLY\n    LOSS_ERROR(0.4) 0" in sequence_block

        original = build_jit_library(qfile)
        rebuilt = build_jit_library(parse(rendered))
        original_stripped, _ = strip_jit_library(original)
        rebuilt_stripped, _ = strip_jit_library(rebuilt)
        assert (
            original_stripped.SerializeToString()
            == rebuilt_stripped.SerializeToString()
        )

    def test_compose_loss_indices_survive_later_noise(self) -> None:
        qfile = parse(
            """
            CODE C[[1,1,1]] { LOGICAL X0 Z0 }

            GADGET First {
                INPUT C 0
                X_ERROR(0.01) 0
                LOSS_ERROR(0.1) 0
                OUTPUT C 0
            }

            GADGET Second {
                INPUT C 0
                Z_ERROR(0.02) 0
                OUTPUT C 0
            }

            COMPOSE Chain {
                INPUT C 0
                First 0
                Second 0
                OUTPUT C 0
            }
            """
        )

        rendered = render_annotated(qfile)
        chain = rendered.split("GADGET Chain {", 1)[1]
        assert "@SIMULATE_ONLY\n    X_ERROR(0.01) 0" in chain
        assert "@SIMULATE_ONLY\n    LOSS_ERROR(0.1) 0" in chain
        assert "@SIMULATE_ONLY\n    Z_ERROR(0.02) 0" in chain
        assert "\n    LOSS(0.1)" in chain
        positions = [
            chain.index("X_ERROR(0.01) 0"),
            chain.index("# E0"),
            chain.index("LOSS_ERROR(0.1) 0"),
            chain.index("# E1"),
            chain.index("Z_ERROR(0.02) 0"),
            chain.index("# E2"),
            chain.index("# E3"),
            chain.index("OUTPUT C 0"),
        ]
        assert positions == sorted(positions)

        original, _ = strip_jit_library(build_jit_library(qfile))
        rebuilt, _ = strip_jit_library(build_jit_library(parse(rendered)))
        assert original.SerializeToString() == rebuilt.SerializeToString()

    def test_compose_loss_indices_do_not_deduplicate_by_footprint(self) -> None:
        qfile = parse(
            """
            CODE C[[1,1,1]] { LOGICAL X0 Z0 }

            GADGET First {
                INPUT C 0
                X_ERROR(0.01) 0
                OUTPUT C 0
            }

            GADGET Second {
                INPUT C 0
                LOSS_ERROR(0.1) 0
                OUTPUT C 0
            }

            COMPOSE Chain {
                INPUT C 0
                First 0
                Second 0
                OUTPUT C 0
            }
            """
        )

        rendered = render_annotated(qfile)
        chain = rendered.split("GADGET Chain {", 1)[1]
        assert "@SIMULATE_ONLY\n    X_ERROR(0.01) 0" in chain
        assert "@SIMULATE_ONLY\n    LOSS_ERROR(0.1) 0" in chain
        assert all(f"# E{index}" in chain for index in range(4))

        original, _ = strip_jit_library(build_jit_library(qfile))
        rebuilt, _ = strip_jit_library(build_jit_library(parse(rendered)))
        assert original.SerializeToString() == rebuilt.SerializeToString()

    def test_declared_loss_metadata_coexists_with_preserved_compose_noise(
        self,
    ) -> None:
        qfile = parse(_DECLARED_LOSS_NOISY_COMPOSE_SRC)
        rendered = render_annotated(qfile)
        chain = rendered.split("GADGET Chain {", 1)[1]

        assert "\n    ERROR(0.0) OUT0.LX0  # E0" in chain
        assert "@SIMULATE_ONLY\n    X_ERROR(0.01) 0" in chain
        assert "\n    ERROR(0.01) OUT0.LX0  # E1" in chain
        assert "\n    ERROR(0.0) OUT0.LZ0  # E2" in chain
        assert "\n    LOSS(0.1)" in chain
        assert chain.index("# E0") < chain.index("X_ERROR(0.01) 0")
        assert chain.index("X_ERROR(0.01) 0") < chain.index("# E1")
        assert chain.index("# E1") < chain.index("# E2")

        original, _ = strip_jit_library(build_jit_library(qfile))
        rebuilt, _ = strip_jit_library(build_jit_library(parse(rendered)))
        assert original.SerializeToString() == rebuilt.SerializeToString()

    def test_mixed_compose_loss_forms_use_merged_metadata(self) -> None:
        qfile = parse(
            """
            CODE C[[1,1,1]] { LOGICAL X0 Z0 }

            GADGET Inferred {
                INPUT C 0
                LOSS_ERROR(0.1) 0
                H 0
                OUTPUT C 0
            }

            GADGET Declared {
                INPUT C 0
                M 0
                OUTPUT C 0
                ERROR(0) LX0
                LOSS(0.2) SE0 M0
                LOSS(IN0.L0) CE0 L0 M0
            }

            COMPOSE Chain {
                INPUT C 0
                Inferred 0
                Declared 0
                OUTPUT C 0
            }
            """
        )

        rendered = render_annotated(qfile)
        chain = rendered.split("GADGET Chain {", 1)[1]
        assert "@SIMULATE_ONLY\n    LOSS_ERROR(0.1) 0" in chain
        assert "\n    LOSS(0.1)" in chain
        assert "\n    LOSS(0.2)" in chain

        original, _ = strip_jit_library(build_jit_library(qfile))
        rebuilt, _ = strip_jit_library(build_jit_library(parse(rendered)))
        assert original.SerializeToString() == rebuilt.SerializeToString()
