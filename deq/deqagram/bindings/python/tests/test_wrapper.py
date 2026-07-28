"""Regression tests for the PyO3 wrapper around the deqagram parser.

`parse` returns a fully typed `DeqFile` tree; these walk it and check the
definition kinds, nested/recursive statements, value equality, and the error
path.
"""

from __future__ import annotations

import pytest

import deqagram

# NOT a normal .deq file: it exercises parser edge cases. In normal use `@GTYPE`
# precedes the `GADGET` keyword so it attaches to the gadget; here it is placed
# inside the body on purpose, where it parses as a standalone decorator
# statement instead (see `test_recursive_repeat_and_decorator`).
_SOURCE = """
IMPORT "other.deq"

CODE RepetitionCode [[3,1,1]] {
LOGICAL X0*X1*X2 Z0*Z1*Z2
STABILIZER Z0*Z1 Z1*Z2
}

GADGET PrepareZ {
@GTYPE(1)
R 0 1 2
REPEAT 2 { X_ERROR(0.03) 0 1 2 }
INPUT RepetitionCode 0 1 2
CHECK rec[-1] rec[-2] FLIP
}

PROGRAM Sim {
PrepareZ 0
ASSERT_EQ rec[-1] 0
}
"""


def test_deqfile_shape() -> None:
    f = deqagram.parse(_SOURCE)
    assert f.imports == ["other.deq"]
    kinds = [type(d).__qualname__ for d in f.definitions]
    assert kinds == [
        "Definition_Code",
        "Definition_Gadget",
        "Definition_Program",
    ]


def test_code_fields() -> None:
    f = deqagram.parse(_SOURCE)
    code = f.definitions[0].code
    assert isinstance(code, deqagram.CodeDefinition)
    assert (code.name, code.n, code.k, code.d) == ("RepetitionCode", 3, 1, 1)
    log = code.logicals[0]
    assert isinstance(log.x_operator, deqagram.PauliProduct.Terms)
    assert [(t.pauli, t.index) for t in log.x_operator.terms] == [
        (deqagram.Pauli.X, 0),
        (deqagram.Pauli.X, 1),
        (deqagram.Pauli.X, 2),
    ]
    assert len(code.stabilizers) == 2


def test_recursive_repeat_and_decorator() -> None:
    gad = deqagram.parse(_SOURCE).definitions[1].gadget
    assert gad.decorators == []  # body-level @GTYPE is a standalone statement
    dec = next(s for s in gad.body if isinstance(s, deqagram.GadgetStatement.Decorator))
    assert dec.decorator.name == "GTYPE"
    rep = next(s for s in gad.body if isinstance(s, deqagram.GadgetStatement.Repeat))
    assert rep.count == 2
    inner = rep.body[0]
    assert isinstance(inner, deqagram.GadgetStatement.Instruction)
    assert inner.instruction.name == "X_ERROR"
    assert inner.instruction.arguments == [0.03]
    assert all(isinstance(t, deqagram.Target.Qubit) for t in inner.instruction.targets)


def test_check_statement_flip() -> None:
    gad = deqagram.parse(_SOURCE).definitions[1].gadget
    chk = next(s for s in gad.body if isinstance(s, deqagram.GadgetStatement.Check))
    assert chk.check.flip is True
    assert isinstance(chk.check.targets[0], deqagram.Target.MeasurementRecord)
    assert chk.check.targets[0].offset == 1


def test_value_equality() -> None:
    assert deqagram.parse(_SOURCE) == deqagram.parse(_SOURCE)


def test_variant_construction() -> None:
    # Complex-enum variants are constructible and compare by value.
    assert deqagram.Target.Qubit(inverted=False, index=2) == deqagram.Target.Qubit(
        inverted=False, index=2
    )


def test_parse_error_raises_value_error() -> None:
    with pytest.raises(ValueError):
        deqagram.parse("CODE oops {")


_ATTACHED_SOURCE = """
GADGET G {
@GTYPE(1)
@CHECKS("syndrome")
R 0 1 2
REPEAT 2 {
@DEEP
M 0
}
@DANGLING
}
"""


def test_parse_attached_folds_decorators() -> None:
    f = deqagram.parse_attached(_ATTACHED_SOURCE)
    gadget = f.definitions[0]
    assert isinstance(gadget, deqagram.AttachedDefinition.Gadget)
    assert gadget.name == "G"

    # First statement (R) carries the two decorators that preceded it.
    first = gadget.body[0]
    assert isinstance(first.statement, deqagram.AttachedGadget.Statement)
    assert [d.name for d in first.decorators] == ["GTYPE", "CHECKS"]

    # The REPEAT block itself has no decorators; its inner M carries @DEEP.
    repeat = gadget.body[1].statement
    assert isinstance(repeat, deqagram.AttachedGadget.Repeat)
    assert repeat.count == 2
    inner = repeat.body[0]
    assert [d.name for d in inner.decorators] == ["DEEP"]

    # The trailing @DANGLING has no statement to attach to.
    assert [d.name for d in gadget.dangling] == ["DANGLING"]


def test_parse_attached_code_reuses_flat_definition() -> None:
    f = deqagram.parse_attached("CODE C [[2,0]] {\nSTABILIZER X0*X1\n}\n")
    code = f.definitions[0]
    assert isinstance(code, deqagram.AttachedDefinition.Code)
    assert code.code.name == "C"


def test_parse_attached_exposes_spans() -> None:
    src = "CODE C [[2,0]] {\nSTABILIZER X0*X1\n}\n\nGADGET G {\nR 0 1\n}\n"
    f = deqagram.parse_attached(src)
    code, gadget = f.definitions
    # Definition spans resolve to 1-based (line, column) against the source.
    assert code.span.line_col(src) == (1, 1)
    assert gadget.span.line_col(src) == (5, 1)
    # Statement spans are exposed too, for precise diagnostics.
    stmt = gadget.body[0]
    assert stmt.span.line_col(src) == (6, 1)
    assert stmt.span.start < stmt.span.end
