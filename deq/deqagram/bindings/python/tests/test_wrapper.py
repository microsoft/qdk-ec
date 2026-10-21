"""Tests for deqagram's Python API: ``parse`` returns the dataclasses ``deqagram`` exports."""

from __future__ import annotations

import math

import pytest

import deqagram

_SOURCE = """
IMPORT "other.deq"

CODE RepetitionCode [[3,1,1]] {
LOGICAL X0*X1*X2 Z0*Z1*Z2
STABILIZER Z0*Z1 Z1*Z2
}

GADGET PrepareZ {
@GTYPE(1)
R 0 1 2
REPEAT 2 {
@DEEP
X_ERROR(0.03) 0 1 2
}
INPUT RepetitionCode 0 1 2
CHECK rec[-1] rec[-2] FLIP
}

PROGRAM Sim {
PrepareZ 0
ASSERT_EQ rec[-1] 0
}
"""


def test_file_shape() -> None:
    file = deqagram.parse(_SOURCE)
    assert file.imports == [deqagram.ImportStatement("other.deq")]
    assert [type(d) for d in file.definitions] == [
        deqagram.CodeDefinition,
        deqagram.GadgetDefinition,
        deqagram.ProgramDefinition,
    ]
    assert [d.source_line for d in file.definitions] == [4, 9, 20]


def test_code_fields() -> None:
    code = deqagram.parse(_SOURCE).definitions[0]
    assert (code.name, code.n, code.k, code.d) == ("RepetitionCode", 3, 1, 1)
    assert str(code.logicals[0].x_operator) == "X0*X1*X2"
    assert [str(s) for s in code.stabilizers] == ["Z0*Z1", "Z1*Z2"]


def test_decorators_attach_through_repeat() -> None:
    gadget = deqagram.parse(_SOURCE).definitions[1]
    reset, repeat = gadget.body[:2]
    assert reset.decorators == [deqagram.Decorator("GTYPE", (1,))]
    assert isinstance(repeat, deqagram.RepeatBlock)
    assert (repeat.count, repeat.decorators, repeat.source_line) == (2, [], 12)
    assert repeat.body == [
        deqagram.Instruction(
            name="X_ERROR",
            arguments=[0.03],
            targets=[deqagram.QubitTarget(0), deqagram.QubitTarget(1), deqagram.QubitTarget(2)],
            decorators=[deqagram.Decorator("DEEP")],
            source_line=14,
        )
    ]


def test_check_statement_flip() -> None:
    check = deqagram.parse(_SOURCE).definitions[1].body[-1]
    assert check == deqagram.CheckStatement(
        targets=[deqagram.MeasurementRecordTarget(1), deqagram.MeasurementRecordTarget(2)], flip=True
    )


def test_statements_render_as_deq() -> None:
    program = deqagram.parse(_SOURCE).definitions[2]
    assert [str(s) for s in program.body] == ["PrepareZ 0", "ASSERT_EQ rec[-1] 0"]


def test_value_equality() -> None:
    assert deqagram.parse(_SOURCE) == deqagram.parse(_SOURCE)


def test_decorator_arguments() -> None:
    gadget = deqagram.parse('@A(1, 2.5, "s", key="v")\nGADGET G {\n}\n').definitions[0]
    assert gadget.decorators == [deqagram.Decorator("A", (1, 2.5, "s", deqagram.KeywordArg("key", "v")))]


@pytest.mark.parametrize(
    "source,reason",
    [
        ("GADGET G {\nR 0\n@FOO\n}\n", "at end of block have no target"),
        ("COMPOSE C {\n@FOO\nCONDITIONAL rec[-1] X0 0\n}\n", "on a CONDITIONAL correction"),
        ("PROGRAM P {\n@FOO\nCONDITIONAL rec[-1] X0 0\n}\n", "on a CONDITIONAL correction"),
        ("PROGRAM P {\n@FOO\nVIRTUAL X0 0\n}\n", "on a VIRTUAL correction"),
    ],
)
def test_unstorable_decorator_warns(source: str, reason: str) -> None:
    with pytest.warns(UserWarning, match=f"@FOO {reason}"):
        definition = deqagram.parse(source).definitions[0]
    assert len(definition.body) == 1


def test_identity_product_has_no_terms() -> None:
    code = deqagram.parse("CODE C [[1,1]] {\nLOGICAL _ Z0\n}\n").definitions[0]
    assert code.logicals[0].x_operator == deqagram.PauliProduct(terms=())


def test_parse_error_raises_value_error() -> None:
    with pytest.raises(ValueError):
        deqagram.parse("CODE oops {")


def test_rotation_radians_normalize_to_half_turns() -> None:
    instruction = deqagram.parse("GADGET G { U3(0.5rad, 0.25, -1rad) 0 }").definitions[0].body[0]
    assert instruction.arguments == pytest.approx([0.5 / math.pi, 0.25, -1 / math.pi])


@pytest.mark.parametrize("instruction", ["X_ERROR(0.1rad) 0", "T(0.25rad) 0"])
def test_radian_suffix_requires_rotation(instruction: str) -> None:
    with pytest.raises(ValueError, match="only supported for rotation angles"):
        deqagram.parse(f"GADGET G {{ {instruction} }}")


def test_all_exports_every_public_class() -> None:
    from deqagram import _nodes

    classes = {
        name
        for name, value in vars(_nodes).items()
        if isinstance(value, type) and value.__module__ == _nodes.__name__ and not name.startswith("_")
    }
    assert classes <= set(deqagram.__all__)
