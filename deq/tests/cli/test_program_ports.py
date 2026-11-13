"""PROGRAM interfaces are declared, ordered, and checked rather than inferred."""

import pytest

from deq.circuit.parser import parse
from deq.cli.jit import compile_program_for_jit
from deq.transpiler.jit_library_builder import build_jit_program
from deqagram import ProgramDefinition

LIBRARY = """
CODE C [[1,1]] { LOGICAL X0 Z0 }
CODE D [[1,1]] { LOGICAL X0 Z0 }
GADGET Prepare { R 0 OUTPUT C 0 }
GADGET PrepareD { R 0 OUTPUT D 0 }
GADGET Idle { INPUT C 0 I 0 OUTPUT C 0 }
GADGET Measure { INPUT C 0 M 0 READOUT M0 }
GADGET MeasureD { INPUT D 0 M 0 READOUT M0 }
"""


def compile_source(programs: str):
    source = parse(LIBRARY + programs)
    definitions = {
        item.name: item
        for item in source.definitions
        if isinstance(item, ProgramDefinition)
    }
    return compile_program_for_jit(
        build_jit_program(source), definitions["Main"], definitions
    )[0]


@pytest.mark.parametrize(
    "body,call,error",
    [
        ("Idle 0", "Sub 0", "declare INPUT"),
        ("INPUT C 0 Idle 0", "Sub 0", "OUTPUT declarations"),
        ("Prepare 0", "Sub 0", "OUTPUT declarations"),
        ("OUTPUT C 0", "Sub 0", "have no producer"),
        ("INPUT C 0 INPUT C 0 OUTPUT C 0", "Sub 0", "duplicate INPUT"),
        ("INPUT C 0 OUTPUT C 0 OUTPUT C 0", "Sub 0", "duplicate OUTPUT"),
        ("INPUT C 0 1", "Sub 0 1", "exactly one logical wire"),
        ("INPUT Missing 0 OUTPUT Missing 0", "Sub 0", "unknown port code"),
        ("INPUT D 0 Idle 0 OUTPUT C 0", "Sub 0", "incompatible port code"),
        ("INPUT C 0 Idle 0 OUTPUT D 0", "Sub 0", "incompatible port code"),
        ("INPUT C 0 Idle 0 OUTPUT C 0", "Sub 0 1", "wire arguments"),
        ("INPUT C 0 Idle 0 OUTPUT C 0", "Sub IN(0) OUT(1)", "conflicting"),
    ],
)
def test_invalid_program_interfaces_fail(body, call, error):
    with pytest.raises(ValueError, match=error):
        compile_source(
            f"PROGRAM Sub {{ {body} }} PROGRAM Main {{ Prepare 0 {call} Measure 0 }}"
        )


def test_declaration_order_not_numeric_wire_order():
    compiled = compile_source("""
PROGRAM Read { INPUT C 8 INPUT C 3 Measure 8 Measure 3 }
PROGRAM Main { Prepare 0 Prepare 1 Read 0 1 }
""")
    assert [item.gadget.connectors[0].gid for item, _ in compiled[2:]] == [1, 2]


def test_closed_programs_need_no_port_declarations():
    compiled = compile_source("""
PROGRAM Unit { Prepare 9 Measure 9 }
PROGRAM Main { Unit() Unit() }
""")
    assert len(compiled) == 4
    assert [compiled[index][0].gadget.connectors[0].gid for index in (1, 3)] == [1, 3]


def test_noop_subprogram_still_checks_declared_code():
    with pytest.raises(ValueError, match="port code"):
        compile_source("""
PROGRAM Pass { INPUT C 0 OUTPUT C 0 }
PROGRAM Main { PrepareD 0 Pass 0 MeasureD 0 }
""")


def test_entry_program_requires_a_closed_wrapper():
    with pytest.raises(ValueError, match="closed entry program"):
        compile_source("PROGRAM Main { INPUT C 0 Measure 0 }")


@pytest.mark.parametrize("count", [0, 1, 2])
def test_programmatic_repeat_cannot_declare_ports(count):
    from deqagram import InputPort, RepeatBlock

    program = ProgramDefinition(
        name="Main", body=[RepeatBlock(count=count, body=[InputPort("C", [0])])]
    )
    with pytest.raises(SyntaxError, match="inside a REPEAT"):
        compile_program_for_jit(build_jit_program(parse(LIBRARY)), program)
