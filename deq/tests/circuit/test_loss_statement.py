"""Parsing and rendering tests for the ``LOSS(...)`` statement."""

import pytest

from deq.circuit.model import Instruction, LossStatement
from deq.circuit.parser import parse


def _gadget_body(src: str) -> list:
    qfile = parse(src)
    gadget = next(
        d for d in qfile.definitions if d.__class__.__name__ == "GadgetDefinition"
    )
    return gadget.body


_HEADER = "CODE C[[3,1,1]] { LOGICAL X0 Z0 }\n"


def test_source_loss_parses_all_target_kinds() -> None:
    body = _gadget_body(
        _HEADER + "GADGET G { LOSS(0.1) SE0 SE3 CE4 CE5 L3 L4 OUT0.L2 M4 }\n"
    )
    (loss,) = [s for s in body if isinstance(s, LossStatement)]
    assert not loss.is_input
    assert loss.probability == 0.1
    assert loss.source_errors == [0, 3]
    assert loss.continuation_errors == [4, 5]
    assert loss.child_losses == [3, 4]
    assert loss.output_qubits == [(0, 2)]
    assert loss.measurement_indices == [4]


def test_input_loss_parses_without_probability_or_source_errors() -> None:
    body = _gadget_body(
        _HEADER + "GADGET G { LOSS(IN0.L1) CE0 CE1 L1 L2 OUT0.L2 M1 }\n"
    )
    (loss,) = [s for s in body if isinstance(s, LossStatement)]
    assert loss.is_input
    assert loss.probability is None
    assert loss.input_port == 0
    assert loss.input_qubit == 1
    assert loss.source_errors == []
    assert loss.continuation_errors == [0, 1]
    assert loss.child_losses == [1, 2]
    assert loss.output_qubits == [(0, 2)]
    assert loss.measurement_indices == [1]


@pytest.mark.parametrize(
    "line",
    [
        "LOSS(0.1) SE0 SE3 CE4 CE5 L3 L4 OUT0.L2 M4",
        "LOSS(IN0.L1) CE0 CE1 L1 L2 OUT0.L2 M1",
        "LOSS(0.05)",
        "LOSS(IN2.L7)",
    ],
)
def test_loss_statement_round_trips_through_str(line: str) -> None:
    body = _gadget_body(_HEADER + f"GADGET G {{ {line} }}\n")
    (loss,) = [s for s in body if isinstance(s, LossStatement)]
    assert str(loss) == line


def test_loss_error_instruction_is_unaffected_by_loss_keyword() -> None:
    body = _gadget_body(_HEADER + "GADGET G { LOSS_ERROR(0.1) 0 }\n")
    (instruction,) = [s for s in body if isinstance(s, Instruction)]
    assert instruction.name.upper() == "LOSS_ERROR"


def test_input_loss_rejects_source_error_target() -> None:
    with pytest.raises(SyntaxError):
        parse(_HEADER + "GADGET G { LOSS(IN0.L0) SE0 }\n")


def test_source_loss_rejects_zero_probability() -> None:
    # Every loss node is a declared site; zero-probability nodes are never
    # created, so LOSS(0) is rejected at parse time.
    with pytest.raises(SyntaxError):
        parse(_HEADER + "GADGET G { LOSS(0) M0 }\n")
