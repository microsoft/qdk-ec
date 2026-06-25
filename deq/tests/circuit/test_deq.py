"""Tests that parse the repetition_code.deq example and verify all definitions."""

from pathlib import Path

import pytest

from deq.circuit.parser import parse
from deq.circuit.model import (
    DeqFile,
    CodeDefinition,
    GadgetDefinition,
    ComposeDefinition,
    ProgramDefinition,
    GadgetApplication,
    AssertStatement,
    InputPort,
    OutputPort,
    ReadoutStatement,
    CheckStatement,
    Instruction,
    RepeatBlock,
    PauliProduct,
    PauliTerm,
    QubitTarget,
    MeasurementRecordTarget,
    Decorator,
    KeywordArg,
    ConditionalStatement,
    DestabilizerTarget,
    PropagateStatement,
    ReadoutTarget,
    LogicalPauliTarget,
)

DEQ_FILE = Path(__file__).parent / "fixtures" / "example.deq"


@pytest.fixture(scope="module")
def deq() -> DeqFile:
    return parse(DEQ_FILE.read_text(encoding="utf-8"))


# ── Top-level structure ──────────────────────────────────────────────


class TestTopLevel:
    def test_parses_successfully(self, deq: DeqFile):
        assert isinstance(deq, DeqFile)

    def test_definition_count(self, deq: DeqFile):
        assert len(deq.definitions) == 9

    def test_definition_types(self, deq: DeqFile):
        types = [type(d).__name__ for d in deq.definitions]
        assert types == [
            "CodeDefinition",
            "GadgetDefinition",
            "GadgetDefinition",
            "GadgetDefinition",
            "ComposeDefinition",
            "ComposeDefinition",
            "GadgetDefinition",
            "ProgramDefinition",
            "ProgramDefinition",
        ]

    def test_definition_names(self, deq: DeqFile):
        names = [d.name for d in deq.definitions]
        assert names == [
            "RepetitionCode",
            "PrepareZ",
            "MeasureZ",
            "Idle",
            "Idle3",
            "Idle4",
            "Ejection",
            "Idle16",
            "MemoryExperiment",
        ]


# ── CODE definition ──────────────────────────────────────────────────


class TestCodeDefinition:
    @pytest.fixture()
    def code(self, deq: DeqFile) -> CodeDefinition:
        return deq.definitions[0]

    def test_name(self, code: CodeDefinition):
        assert code.name == "RepetitionCode"

    def test_params(self, code: CodeDefinition):
        assert code.n == 3
        assert code.k == 1
        assert code.d == 3

    def test_logical_count(self, code: CodeDefinition):
        assert len(code.logicals) == 1

    def test_logical_x_operator(self, code: CodeDefinition):
        x_op = code.logicals[0].x_operator
        assert x_op == PauliProduct(
            terms=(
                PauliTerm("X", 0),
                PauliTerm("X", 1),
                PauliTerm("X", 2),
            )
        )

    def test_logical_z_operator(self, code: CodeDefinition):
        z_op = code.logicals[0].z_operator
        assert z_op == PauliProduct(
            terms=(
                PauliTerm("Z", 0),
                PauliTerm("Z", 1),
                PauliTerm("Z", 2),
            )
        )

    def test_stabilizer_count(self, code: CodeDefinition):
        assert len(code.stabilizers) == 2

    def test_first_stabilizer(self, code: CodeDefinition):
        assert code.stabilizers[0] == PauliProduct(
            terms=(
                PauliTerm("Z", 0),
                PauliTerm("Z", 1),
            )
        )

    def test_last_stabilizer(self, code: CodeDefinition):
        assert code.stabilizers[1] == PauliProduct(
            terms=(
                PauliTerm("Z", 1),
                PauliTerm("Z", 2),
            )
        )


# ── GADGET PrepareZ ──────────────────────────────────────────────────


class TestGadgetPrepareZ:
    @pytest.fixture()
    def gadget(self, deq: DeqFile) -> GadgetDefinition:
        return deq.definitions[1]

    def test_name(self, gadget: GadgetDefinition):
        assert gadget.name == "PrepareZ"

    def test_has_instruction(self, gadget: GadgetDefinition):
        instrs = [s for s in gadget.body if isinstance(s, Instruction)]
        assert len(instrs) == 1
        assert instrs[0].name == "R"
        assert instrs[0].targets == [QubitTarget(i) for i in range(3)]

    def test_has_output(self, gadget: GadgetDefinition):
        outputs = [s for s in gadget.body if isinstance(s, OutputPort)]
        assert len(outputs) == 1
        assert outputs[0].code_name == "RepetitionCode"
        assert outputs[0].qubit_indices == [0, 1, 2]


# ── GADGET MeasureZ ──────────────────────────────────────────────────


class TestGadgetMeasureZ:
    @pytest.fixture()
    def gadget(self, deq: DeqFile) -> GadgetDefinition:
        return deq.definitions[2]

    def test_has_input(self, gadget: GadgetDefinition):
        inputs = [s for s in gadget.body if isinstance(s, InputPort)]
        assert len(inputs) == 1
        assert inputs[0].code_name == "RepetitionCode"
        assert inputs[0].qubit_indices == [0, 1, 2]

    def test_has_readout(self, gadget: GadgetDefinition):
        readouts = [s for s in gadget.body if isinstance(s, ReadoutStatement)]
        assert len(readouts) == 1
        assert readouts[0].targets == [
            MeasurementRecordTarget(3),
            MeasurementRecordTarget(2),
            MeasurementRecordTarget(1),
        ]


# ── GADGET Idle ──────────────────────────────────────────────────────


class TestGadgetIdle:
    @pytest.fixture()
    def gadget(self, deq: DeqFile) -> GadgetDefinition:
        return deq.definitions[3]

    def test_name(self, gadget: GadgetDefinition):
        assert gadget.name == "Idle"

    def test_has_input_and_output(self, gadget: GadgetDefinition):
        inputs = [s for s in gadget.body if isinstance(s, InputPort)]
        outputs = [s for s in gadget.body if isinstance(s, OutputPort)]
        assert len(inputs) == 1
        assert len(outputs) == 1
        assert inputs[0].qubit_indices == [0, 2, 4]
        assert outputs[0].qubit_indices == [0, 2, 4]

    def test_check_count(self, gadget: GadgetDefinition):
        checks = [s for s in gadget.body if isinstance(s, CheckStatement)]
        assert len(checks) == 4  # 2 before OUTPUT + 2 after OUTPUT

    def test_check_targets(self, gadget: GadgetDefinition):
        checks = [s for s in gadget.body if isinstance(s, CheckStatement)]
        assert checks[0].targets == [
            MeasurementRecordTarget(2),
            MeasurementRecordTarget(4),
        ]

    def test_stim_instructions(self, gadget: GadgetDefinition):
        instrs = [s for s in gadget.body if isinstance(s, Instruction)]
        names = [i.name for i in instrs]
        assert names == ["R", "CNOT", "CNOT", "MR"]

    def test_has_decorator(self, gadget: GadgetDefinition):
        assert len(gadget.decorators) == 1
        assert gadget.decorators[0].name == "CHECKS"
        assert gadget.decorators[0].arguments == ("manual",)


# ── COMPOSE Idle3 ────────────────────────────────────────────────────


class TestComposeIdle3:
    @pytest.fixture()
    def compose(self, deq: DeqFile) -> ComposeDefinition:
        return deq.definitions[4]

    def test_name(self, compose: ComposeDefinition):
        assert compose.name == "Idle3"

    def test_has_input_output(self, compose: ComposeDefinition):
        inputs = [s for s in compose.body if isinstance(s, InputPort)]
        outputs = [s for s in compose.body if isinstance(s, OutputPort)]
        assert len(inputs) == 1
        assert len(outputs) == 1
        assert inputs[0].code_name == "RepetitionCode"
        assert inputs[0].qubit_indices == [0]

    def test_has_repeat_block(self, compose: ComposeDefinition):
        repeats = [s for s in compose.body if isinstance(s, RepeatBlock)]
        assert len(repeats) == 1
        assert repeats[0].count == 3

    def test_repeat_body_has_gadget_application(self, compose: ComposeDefinition):
        repeat = next(s for s in compose.body if isinstance(s, RepeatBlock))
        stmts = repeat.body
        assert len(stmts) == 1
        assert isinstance(stmts[0], GadgetApplication)
        assert stmts[0].gadget_name == "Idle"
        assert stmts[0].in_indices == [0]
        assert stmts[0].out_indices == [0]


# ── COMPOSE Idle4 (with shortcut form) ───────────────────────────────


class TestComposeIdle4:
    @pytest.fixture()
    def compose(self, deq: DeqFile) -> ComposeDefinition:
        return deq.definitions[5]

    def test_name(self, compose: ComposeDefinition):
        assert compose.name == "Idle4"

    def test_explicit_application(self, compose: ComposeDefinition):
        apps = [s for s in compose.body if isinstance(s, GadgetApplication)]
        assert len(apps) == 1
        assert apps[0].gadget_name == "Idle3"
        assert apps[0].in_indices == [0]
        assert apps[0].out_indices == [0]

    def test_shortcut_application(self, compose: ComposeDefinition):
        # Shortcut "Idle 0" parses as an Instruction
        instrs = [s for s in compose.body if isinstance(s, Instruction)]
        assert len(instrs) == 1
        assert instrs[0].name == "Idle"
        assert instrs[0].targets == [QubitTarget(0)]


# ── GADGET Ejection ──────────────────────────────────────────────────


class TestGadgetEjection:
    @pytest.fixture()
    def gadget(self, deq: DeqFile) -> GadgetDefinition:
        return deq.definitions[6]

    def test_name(self, gadget: GadgetDefinition):
        assert gadget.name == "Ejection"

    def test_has_readout(self, gadget: GadgetDefinition):
        readouts = [s for s in gadget.body if isinstance(s, ReadoutStatement)]
        assert len(readouts) == 1

    def test_has_conditional(self, gadget: GadgetDefinition):
        conditionals = [s for s in gadget.body if isinstance(s, ConditionalStatement)]
        assert len(conditionals) == 1

    def test_conditional_readout(self, gadget: GadgetDefinition):
        cond = next(s for s in gadget.body if isinstance(s, ConditionalStatement))
        assert cond.condition == ReadoutTarget(index=0)

    def test_conditional_targets(self, gadget: GadgetDefinition):
        cond = next(s for s in gadget.body if isinstance(s, ConditionalStatement))
        assert cond.targets == [LogicalPauliTarget(pauli="X", index=0)]


# ── PROGRAM Idle16 ───────────────────────────────────────────────────


class TestProgramIdle16:
    @pytest.fixture()
    def program(self, deq: DeqFile) -> ProgramDefinition:
        return deq.definitions[7]

    def test_name(self, program: ProgramDefinition):
        assert program.name == "Idle16"

    def test_gadget_application_count(self, program: ProgramDefinition):
        apps = [s for s in program.body if isinstance(s, GadgetApplication)]
        assert len(apps) == 4
        assert all(a.gadget_name == "Idle4" for a in apps)


# ── PROGRAM MemoryExperiment ─────────────────────────────────────────


class TestProgramMemoryExperiment:
    @pytest.fixture()
    def program(self, deq: DeqFile) -> ProgramDefinition:
        return deq.definitions[8]

    def test_name(self, program: ProgramDefinition):
        assert program.name == "MemoryExperiment"

    def test_has_no_input_output(self, program: ProgramDefinition):
        inputs = [s for s in program.body if isinstance(s, InputPort)]
        outputs = [s for s in program.body if isinstance(s, OutputPort)]
        assert len(inputs) == 0
        assert len(outputs) == 0

    def test_gadget_applications(self, program: ProgramDefinition):
        apps = [s for s in program.body if isinstance(s, GadgetApplication)]
        assert len(apps) == 3
        assert apps[0].gadget_name == "PrepareZ"
        assert apps[0].in_indices is None
        assert apps[0].out_indices == [0]

        assert apps[1].gadget_name == "Idle16"
        assert apps[1].in_indices == [0]
        assert apps[1].out_indices == [0]

        assert apps[2].gadget_name == "MeasureZ"
        assert apps[2].in_indices == [0]
        assert apps[2].out_indices is None

    def test_assert_statement(self, program: ProgramDefinition):
        asserts = [s for s in program.body if isinstance(s, AssertStatement)]
        assert len(asserts) == 1
        assert asserts[0].target == MeasurementRecordTarget(1)
        assert asserts[0].expected_value == 0


# ── Unit tests for individual constructs ─────────────────────────────


class TestCodeParsing:
    def test_code_without_distance(self):
        deq = parse("""
CODE MyCode [[3,1]] {
    LOGICAL X0*X1*X2 Z0*Z1*Z2
    STABILIZER Z0*Z1
}
""")
        code = deq.definitions[0]
        assert isinstance(code, CodeDefinition)
        assert code.n == 3
        assert code.k == 1
        assert code.d is None

    def test_multiple_logicals(self):
        text = """
CODE C [[4,2]] {
    LOGICAL X0 Z0*Z1
    LOGICAL X2 Z2*Z3
}
"""
        deq = parse(text)
        code = deq.definitions[0]
        assert len(code.logicals) == 2
        assert code.logicals[0].x_operator == PauliProduct(terms=(PauliTerm("X", 0),))
        assert code.logicals[1].x_operator == PauliProduct(terms=(PauliTerm("X", 2),))


class TestGadgetParsing:
    def test_stim_instruction_with_args(self):
        text = """GADGET G {
    DEPOLARIZE1(0.001) 0 1 2
}
"""
        deq = parse(text)
        gadget = deq.definitions[0]
        instr = gadget.body[0]
        assert isinstance(instr, Instruction)
        assert instr.arguments == [0.001]


class TestComposeParsing:
    def test_pauli_correction(self):
        text = """COMPOSE C {
    INPUT Code 0
    Z 0
    OUTPUT Code 0
}
"""
        deq = parse(text)
        compose = deq.definitions[0]
        instrs = [s for s in compose.body if isinstance(s, Instruction)]
        assert len(instrs) == 1
        assert instrs[0].name == "Z"


class TestProgramParsing:
    def test_simple_program(self):
        text = """PROGRAM P {
    Prep OUT(0)
    Meas IN(0)
    ASSERT_EQ rec[-1] 0
}
"""
        deq = parse(text)
        prog = deq.definitions[0]
        assert isinstance(prog, ProgramDefinition)
        apps = [s for s in prog.body if isinstance(s, GadgetApplication)]
        assert len(apps) == 2
        asserts = [s for s in prog.body if isinstance(s, AssertStatement)]
        assert asserts[0].expected_value == 0


class TestEmptyFile:
    def test_empty(self):
        deq = parse("")
        assert deq.definitions == []

    def test_comments_only(self):
        deq = parse("# just comments\n# more comments\n")
        assert deq.definitions == []


class TestEmptyStabilizer:
    def test_empty_stabilizer_with_logicals(self):
        text = (
            "CODE C [[3, 1]] {\n"
            "    LOGICAL X0*X1*X2 Z0*Z1*Z2\n"
            "    STABILIZER\n"
            "}\n"
        )
        deq = parse(text)
        code = deq.definitions[0]
        assert isinstance(code, CodeDefinition)
        assert len(code.logicals) == 1
        assert code.stabilizers == []

    def test_no_stabilizer_declaration(self):
        text = "CODE C [[3, 1]] {\n" "    LOGICAL X0*X1*X2 Z0*Z1*Z2\n" "}\n"
        deq = parse(text)
        code = deq.definitions[0]
        assert isinstance(code, CodeDefinition)
        assert len(code.logicals) == 1
        assert code.stabilizers == []


# ── Repeat block restriction tests ───────────────────────────────────


class TestRepeatBlockRestrictions:
    def test_input_in_gadget_repeat_is_invalid(self):
        text = "GADGET G {\n    REPEAT 3 {\n        INPUT a 0\n    }\n}\n"
        with pytest.raises(SyntaxError):
            parse(text)

    def test_output_in_gadget_repeat_is_invalid(self):
        text = "GADGET G {\n    REPEAT 3 {\n        OUTPUT a 0\n    }\n}\n"
        with pytest.raises(SyntaxError):
            parse(text)

    def test_input_in_compose_repeat_is_invalid(self):
        text = "COMPOSE C {\n    REPEAT 3 {\n        INPUT a 0\n    }\n}\n"
        with pytest.raises(SyntaxError):
            parse(text)

    def test_output_in_program_repeat_is_invalid(self):
        text = "PROGRAM P {\n    REPEAT 3 {\n        OUTPUT a 0\n    }\n}\n"
        with pytest.raises(SyntaxError):
            parse(text)

    def test_instruction_in_gadget_repeat_is_valid(self):
        text = "GADGET G {\n    REPEAT 3 {\n        H 0\n    }\n}\n"
        deq = parse(text)
        gadget = deq.definitions[0]
        assert isinstance(gadget, GadgetDefinition)
        assert len(gadget.body) == 1
        assert isinstance(gadget.body[0], RepeatBlock)


# ── Decorator tests ──────────────────────────────────────────────────


class TestDecoratorNoArgs:
    def test_decorator_without_arguments(self):
        text = "@marker\nGADGET G {\n}\n"
        deq = parse(text)
        gadget = deq.definitions[0]
        assert len(gadget.decorators) == 1
        assert gadget.decorators[0] == Decorator(name="marker")
        assert gadget.decorators[0].arguments == ()


class TestDecoratorPositionalArgs:
    def test_string_arg(self):
        text = '@CHECKS("manual")\nGADGET G {\n}\n'
        deq = parse(text)
        gadget = deq.definitions[0]
        assert gadget.decorators[0] == Decorator(name="CHECKS", arguments=("manual",))

    def test_integer_arg(self):
        text = "@priority(42)\nGADGET G {\n}\n"
        deq = parse(text)
        gadget = deq.definitions[0]
        assert gadget.decorators[0] == Decorator(name="priority", arguments=(42,))

    def test_float_arg(self):
        text = "@noise(0.001)\nGADGET G {\n}\n"
        deq = parse(text)
        gadget = deq.definitions[0]
        assert gadget.decorators[0] == Decorator(name="noise", arguments=(0.001,))

    def test_multiple_positional_args(self):
        text = '@config("fast", 42, 3.14)\nGADGET G {\n}\n'
        deq = parse(text)
        gadget = deq.definitions[0]
        assert gadget.decorators[0] == Decorator(
            name="config", arguments=("fast", 42, 3.14)
        )


class TestDecoratorKeywordArgs:
    def test_keyword_args(self):
        text = '@tag(level="high", order=1)\nGADGET G {\n}\n'
        deq = parse(text)
        gadget = deq.definitions[0]
        dec = gadget.decorators[0]
        assert dec.name == "tag"
        assert dec.arguments == (
            KeywordArg(key="level", value="high"),
            KeywordArg(key="order", value=1),
        )

    def test_mixed_positional_and_keyword(self):
        text = '@tag("test", verbose=1)\nGADGET G {\n}\n'
        deq = parse(text)
        gadget = deq.definitions[0]
        dec = gadget.decorators[0]
        assert dec.arguments == ("test", KeywordArg(key="verbose", value=1))


class TestDecoratorStacked:
    def test_multiple_decorators(self):
        text = '@CHECKS("manual")\n@priority(1)\nGADGET G {\n}\n'
        deq = parse(text)
        gadget = deq.definitions[0]
        assert len(gadget.decorators) == 2
        assert gadget.decorators[0].name == "CHECKS"
        assert gadget.decorators[1].name == "priority"


class TestDecoratorOnStatements:
    def test_decorator_on_instruction(self):
        text = "GADGET G {\n    @noise(0.01)\n    CNOT 0 1\n}\n"
        deq = parse(text)
        gadget = deq.definitions[0]
        instr = gadget.body[0]
        assert isinstance(instr, Instruction)
        assert len(instr.decorators) == 1
        assert instr.decorators[0] == Decorator(name="noise", arguments=(0.01,))

    def test_decorator_on_port(self):
        text = 'GADGET G {\n    @label("in")\n    INPUT Code 0\n}\n'
        deq = parse(text)
        gadget = deq.definitions[0]
        port = gadget.body[0]
        assert isinstance(port, InputPort)
        assert port.decorators[0].name == "label"

    def test_decorator_on_gadget_application(self):
        text = "COMPOSE C {\n    INPUT Code 0\n    @round(1)\n    G IN(0) OUT(0)\n    OUTPUT Code 0\n}\n"
        deq = parse(text)
        compose = deq.definitions[0]
        apps = [s for s in compose.body if isinstance(s, GadgetApplication)]
        assert len(apps) == 1
        assert apps[0].decorators[0] == Decorator(name="round", arguments=(1,))

    def test_decorator_with_empty_args(self):
        text = "@marker()\nGADGET G {\n}\n"
        deq = parse(text)
        gadget = deq.definitions[0]
        assert gadget.decorators[0] == Decorator(name="marker", arguments=())

    def test_decorator_on_compose_definition(self):
        text = "@parallel\nCOMPOSE C {\n    INPUT Code 0\n    OUTPUT Code 0\n}\n"
        deq = parse(text)
        compose = deq.definitions[0]
        assert isinstance(compose, ComposeDefinition)
        assert compose.decorators[0].name == "parallel"

    def test_decorator_on_code_definition(self):
        text = """
@version(2)
CODE C [[3,1]] {
    LOGICAL X0*X1*X2 Z0*Z1*Z2
    STABILIZER Z0*Z1
}
"""
        deq = parse(text)
        code = deq.definitions[0]
        assert isinstance(code, CodeDefinition)
        assert code.decorators[0] == Decorator(name="version", arguments=(2,))

    def test_decorator_string_escapes(self):
        text = '@label("hello\\nworld")\nGADGET G {\n}\n'
        deq = parse(text)
        gadget = deq.definitions[0]
        assert gadget.decorators[0].arguments == ("hello\nworld",)


class TestDecoratorStr:
    def test_str_no_args(self):
        d = Decorator(name="marker")
        assert str(d) == "@marker"

    def test_str_with_args(self):
        d = Decorator(name="CHECKS", arguments=("manual",))
        assert str(d) == '@CHECKS("manual")'

    def test_str_keyword_arg(self):
        ka = KeywordArg(key="level", value="high")
        assert str(ka) == 'level="high"'

    def test_str_keyword_int(self):
        ka = KeywordArg(key="order", value=1)
        assert str(ka) == "order=1"


# ── CONDITIONAL statement parsing ────────────────────────────────────


class TestConditionalParsing:
    def test_single_target(self):
        source = """
        CODE C [[3,1,3]] { LOGICAL X0*X1*X2 Z0*Z1*Z2\n STABILIZER Z0*Z1 Z1*Z2 }
        GADGET G {
            INPUT C 0 1 2
            M 3
            READOUT rec[-1]
            OUTPUT C 0 1 2
            CONDITIONAL R0 LX0
        }
        """
        deq = parse(source)
        gadget = deq.definitions[1]
        assert isinstance(gadget, GadgetDefinition)
        conditionals = [s for s in gadget.body if isinstance(s, ConditionalStatement)]
        assert len(conditionals) == 1
        assert conditionals[0].condition == ReadoutTarget(index=0)
        assert conditionals[0].targets == [LogicalPauliTarget(pauli="X", index=0)]

    def test_multiple_targets(self):
        source = """
        CODE C [[3,1,3]] { LOGICAL X0*X1*X2 Z0*Z1*Z2\n STABILIZER Z0*Z1 Z1*Z2 }
        GADGET G {
            INPUT C 0 1 2
            M 3
            READOUT rec[-1]
            OUTPUT C 0 1 2
            CONDITIONAL R0 LX0 LZ0
        }
        """
        deq = parse(source)
        gadget = deq.definitions[1]
        assert isinstance(gadget, GadgetDefinition)
        cond = next(s for s in gadget.body if isinstance(s, ConditionalStatement))
        assert cond.condition == ReadoutTarget(index=0)
        assert cond.targets == [
            LogicalPauliTarget(pauli="X", index=0),
            LogicalPauliTarget(pauli="Z", index=0),
        ]

    def test_multiple_statements(self):
        source = """
        CODE C [[3,1,3]] { LOGICAL X0*X1*X2 Z0*Z1*Z2\n STABILIZER Z0*Z1 Z1*Z2 }
        GADGET G {
            INPUT C 0 1 2
            M 3 4
            READOUT rec[-2]
            READOUT rec[-1]
            OUTPUT C 0 1 2
            CONDITIONAL R0 LX0
            CONDITIONAL R1 LZ0
        }
        """
        deq = parse(source)
        gadget = deq.definitions[1]
        assert isinstance(gadget, GadgetDefinition)
        conditionals = [s for s in gadget.body if isinstance(s, ConditionalStatement)]
        assert len(conditionals) == 2
        assert conditionals[0].condition == ReadoutTarget(index=0)
        assert conditionals[1].condition == ReadoutTarget(index=1)

    def test_conditional_before_output_rejected(self):
        source = """
        CODE C [[3,1,3]] { LOGICAL X0*X1*X2 Z0*Z1*Z2\n STABILIZER Z0*Z1 Z1*Z2 }
        GADGET G {
            INPUT C 0 1 2
            M 3
            READOUT rec[-1]
            CONDITIONAL R0 LX0
            OUTPUT C 0 1 2
        }
        """
        with pytest.raises(
            SyntaxError, match="CONDITIONAL must appear after all OUTPUT"
        ):
            parse(source)


# ── PROPAGATE statement parsing ──────────────────────────────────────


class TestPropagateParsing:
    _CODE_PREAMBLE = (
        "CODE C [[3,1,3]] { LOGICAL X0*X1*X2 Z0*Z1*Z2\n" " STABILIZER Z0*Z1 Z1*Z2 }\n"
    )

    def test_logical_only(self):
        source = self._CODE_PREAMBLE + """
        GADGET G {
            INPUT C 0 1 2
            OUTPUT C 0 1 2
            PROPAGATE LX0 FROM LZ0
        }
        """
        deq = parse(source)
        gadget = deq.definitions[1]
        assert isinstance(gadget, GadgetDefinition)
        propagates = [s for s in gadget.body if isinstance(s, PropagateStatement)]
        assert len(propagates) == 1
        assert propagates[0].target == LogicalPauliTarget(pauli="X", index=0)
        assert propagates[0].terms == [LogicalPauliTarget(pauli="Z", index=0)]
        assert propagates[0].flip is False

    def test_mixed_terms_with_flip(self):
        source = self._CODE_PREAMBLE + """
        GADGET G {
            INPUT C 0 1 2
            M 3
            OUTPUT C 0 1 2
            PROPAGATE LX0 FROM LZ0 IN0.DS0 rec[-1] FLIP
        }
        """
        deq = parse(source)
        gadget = deq.definitions[1]
        propagate = next(s for s in gadget.body if isinstance(s, PropagateStatement))
        assert propagate.target == LogicalPauliTarget(pauli="X", index=0)
        assert propagate.terms == [
            LogicalPauliTarget(pauli="Z", index=0),
            DestabilizerTarget(port_index=0, stab_index=0),
            MeasurementRecordTarget(offset=1),
        ]
        assert propagate.flip is True

    def test_multiple_statements(self):
        source = self._CODE_PREAMBLE + """
        GADGET G {
            INPUT C 0 1 2
            OUTPUT C 0 1 2
            PROPAGATE LX0 FROM LZ0
            PROPAGATE LZ0 FROM LX0 FLIP
        }
        """
        deq = parse(source)
        gadget = deq.definitions[1]
        propagates = [s for s in gadget.body if isinstance(s, PropagateStatement)]
        assert len(propagates) == 2
        assert propagates[0].target == LogicalPauliTarget(pauli="X", index=0)
        assert propagates[0].flip is False
        assert propagates[1].target == LogicalPauliTarget(pauli="Z", index=0)
        assert propagates[1].flip is True

    def test_flip_only_no_terms_accepted(self):
        source = self._CODE_PREAMBLE + """
        GADGET G {
            INPUT C 0 1 2
            OUTPUT C 0 1 2
            PROPAGATE LX0 FROM FLIP
        }
        """
        deq = parse(source)
        gadget = deq.definitions[1]
        propagate = next(s for s in gadget.body if isinstance(s, PropagateStatement))
        assert propagate.target == LogicalPauliTarget(pauli="X", index=0)
        assert propagate.terms == []
        assert propagate.flip is True

    def test_no_terms_no_flip_accepted(self):
        source = self._CODE_PREAMBLE + """
        GADGET G {
            INPUT C 0 1 2
            OUTPUT C 0 1 2
            PROPAGATE LX0 FROM
        }
        """
        deq = parse(source)
        gadget = deq.definitions[1]
        propagate = next(s for s in gadget.body if isinstance(s, PropagateStatement))
        assert propagate.target == LogicalPauliTarget(pauli="X", index=0)
        assert propagate.terms == []
        assert propagate.flip is False

    def test_missing_from_keyword_rejected(self):
        source = self._CODE_PREAMBLE + """
        GADGET G {
            INPUT C 0 1 2
            OUTPUT C 0 1 2
            PROPAGATE LX0 LZ0
        }
        """
        with pytest.raises(SyntaxError):
            parse(source)

    def test_propagate_before_output_rejected(self):
        source = self._CODE_PREAMBLE + """
        GADGET G {
            INPUT C 0 1 2
            PROPAGATE LX0 FROM LZ0
            OUTPUT C 0 1 2
        }
        """
        with pytest.raises(SyntaxError, match="PROPAGATE must appear after all OUTPUT"):
            parse(source)

    def test_port_qualified_logical_parses(self):
        source = self._CODE_PREAMBLE + """
        GADGET G {
            INPUT C 0 1 2
            INPUT C 3 4 5
            OUTPUT C 0 1 2
            OUTPUT C 3 4 5
            PROPAGATE OUT0.LX0 FROM IN0.LZ0
            PROPAGATE OUT1.LZ0 FROM IN1.LX0
        }
        """
        deq = parse(source)
        gadget = deq.definitions[1]
        assert isinstance(gadget, GadgetDefinition)
        propagates = [s for s in gadget.body if isinstance(s, PropagateStatement)]
        assert len(propagates) == 2
        assert propagates[0].target == LogicalPauliTarget(
            pauli="X", index=0, port_kind="OUT", port_index=0
        )
        assert propagates[0].terms == [
            LogicalPauliTarget(pauli="Z", index=0, port_kind="IN", port_index=0)
        ]
        assert propagates[1].target == LogicalPauliTarget(
            pauli="Z", index=0, port_kind="OUT", port_index=1
        )

    def test_port_qualified_logical_transpiles(self):
        source = self._CODE_PREAMBLE + """
        GADGET G {
            INPUT C 0 1 2
            INPUT C 3 4 5
            OUTPUT C 0 1 2
            OUTPUT C 3 4 5
            PROPAGATE OUT0.LX0 FROM IN0.LX0
            PROPAGATE OUT0.LZ0 FROM IN0.LZ0
            PROPAGATE OUT1.LX0 FROM IN1.LX0
            PROPAGATE OUT1.LZ0 FROM IN1.LZ0
        }
        """
        from deq.transpiler.jit_library_builder import build_jit_library

        deq = parse(source)
        # Should transpile without errors and produce one gadget type.
        lib = build_jit_library(deq)
        assert len(lib.gadget_types) == 1

    def test_port_qualified_wrong_side_rejected(self):
        source = self._CODE_PREAMBLE + """
        GADGET G {
            INPUT C 0 1 2
            OUTPUT C 0 1 2
            PROPAGATE IN0.LX0 FROM LZ0
        }
        """
        from deq.transpiler.jit_library_builder import build_jit_library

        deq = parse(source)
        with pytest.raises(ValueError, match="direction"):
            build_jit_library(deq)


# ── Stim alias tests ─────────────────────────────────────────────────


class TestStimAliases:
    def test_detector_parses_as_check(self):
        text = "GADGET G {\n    M 0\n    DETECTOR rec[-1]\n}\n"
        deq = parse(text)
        gadget = deq.definitions[0]
        checks = [s for s in gadget.body if isinstance(s, CheckStatement)]
        assert len(checks) == 1
        assert checks[0].targets == [MeasurementRecordTarget(1)]

    def test_detector_with_flip(self):
        text = "GADGET G {\n    M 0\n    DETECTOR rec[-1] FLIP\n}\n"
        deq = parse(text)
        gadget = deq.definitions[0]
        checks = [s for s in gadget.body if isinstance(s, CheckStatement)]
        assert len(checks) == 1
        assert checks[0].flip is True

    def test_observable_include_parses_as_readout(self):
        text = "GADGET G {\n    M 0\n    OBSERVABLE_INCLUDE rec[-1]\n}\n"
        deq = parse(text)
        gadget = deq.definitions[0]
        readouts = [s for s in gadget.body if isinstance(s, ReadoutStatement)]
        assert len(readouts) == 1
        assert readouts[0].targets == [MeasurementRecordTarget(1)]

    def test_mixed_check_and_detector(self):
        text = "GADGET G {\n    M 0 1\n    CHECK rec[-2]\n    DETECTOR rec[-1]\n}\n"
        deq = parse(text)
        gadget = deq.definitions[0]
        checks = [s for s in gadget.body if isinstance(s, CheckStatement)]
        assert len(checks) == 2
