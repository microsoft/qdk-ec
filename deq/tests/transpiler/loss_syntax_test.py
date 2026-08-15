"""Tests for the bidirectional ``LOSS`` syntax/runtime metadata codec."""

import pytest

from deq.circuit.model import CodeDefinition, GadgetDefinition
from deq.circuit.parser import parse
from deq.transpiler.jit_library_builder import build_jit_library
from deq.transpiler.loss.syntax import loss_model_to_statements

_HEADER = """
CODE C[[3,1,1]] {
    LOGICAL X0 Z0
}
"""


def _loss_model(gadget_src: str):
    lib = build_jit_library(parse(_HEADER + gadget_src))
    return lib.gadget_types[0].base.loss_model


def test_source_and_input_losses_are_packed_faithfully() -> None:
    loss_model = _loss_model(
        """
        GADGET G {
            INPUT C 0 1 2
            M 0
            M 1
            OUTPUT C 0 1 2
            ERROR(0) LX0
            ERROR(0) LZ0
            ERROR(0) LY0
            LOSS(0.1) SE0 CE1 L1 OUT0.L2 M1
            LOSS(0.2) SE1 CE2 M0
            LOSS(IN0.L1) CE0 L0 OUT0.L0 M0
        }
        """
    )

    assert len(loss_model.losses) == 2
    assert len(loss_model.input_losses) == 3

    first = loss_model.losses[0]
    assert first.probability == pytest.approx(0.1)
    assert list(first.source_errors) == [0]
    assert list(first.continuation_errors) == [1]
    assert list(first.child_losses) == [1]
    assert list(first.child_output_qubits) == [2]
    assert list(first.loss_measurements) == [1]

    entered = loss_model.input_losses[1]
    assert list(entered.continuation_errors) == [0]
    assert list(entered.child_losses) == [0]
    assert list(entered.child_output_qubits) == [0]
    assert list(entered.loss_measurements) == [0]

    assert loss_model.input_losses[0].SerializeToString() == b""
    assert loss_model.input_losses[2].SerializeToString() == b""


def test_input_losses_flatten_across_multiple_ports() -> None:
    loss_model = _loss_model(
        """
        GADGET G {
            INPUT C 0 1 2
            INPUT C 3 4 5
            M 0
            OUTPUT C 0 1 2
            OUTPUT C 3 4 5
            LOSS(IN1.L2) OUT1.L0 M0
        }
        """
    )
    # Second port starts at flat offset 3; qubit 2 -> slot 5.
    assert len(loss_model.input_losses) == 6
    assert loss_model.input_losses[5].SerializeToString() != b""
    # OUT1.L0 -> output offset 3 + 0 = 3.
    assert list(loss_model.input_losses[5].child_output_qubits) == [3]


def test_runtime_model_decodes_to_canonical_loss_statements() -> None:
    source = (
        _HEADER
        + """
        GADGET G {
            INPUT C 0 1 2
            INPUT C 3 4 5
            M 0
            OUTPUT C 0 1 2
            OUTPUT C 3 4 5
            ERROR(0) LX0
            LOSS(0.1) SE0 OUT1.L2 M0
            LOSS(IN1.L2) CE0 L0 OUT1.L0 M0
        }
        """
    )
    qfile = parse(source)
    codes = {
        definition.name: definition
        for definition in qfile.definitions
        if isinstance(definition, CodeDefinition)
    }
    gadget = next(
        definition
        for definition in qfile.definitions
        if isinstance(definition, GadgetDefinition)
    )
    loss_model = build_jit_library(qfile).gadget_types[0].base.loss_model

    source_losses, input_losses = loss_model_to_statements(
        loss_model,
        input_ports=gadget.input_ports,
        output_ports=gadget.output_ports,
        codes=codes,
        gadget_name=gadget.name,
    )

    assert [str(statement) for statement in source_losses] == [
        "LOSS(0.1) SE0 OUT1.L2 M0"
    ]
    assert [str(statement) for statement in input_losses] == [
        "LOSS(IN1.L2) CE0 L0 OUT1.L0 M0"
    ]


def test_loss_free_gadget_has_no_loss_model() -> None:
    lib = build_jit_library(
        parse(
            _HEADER
            + """
            GADGET G {
                INPUT C 0 1 2
                M 0
                OUTPUT C 0 1 2
            }
            """
        )
    )
    assert not lib.gadget_types[0].base.HasField("loss_model")


def test_loss_and_loss_error_together_is_rejected() -> None:
    with pytest.raises(ValueError, match="mixes"):
        build_jit_library(
            parse(
                _HEADER
                + """
                GADGET G {
                    INPUT C 0 1 2
                    LOSS_ERROR(0.1) 0
                    M 0
                    OUTPUT C 0 1 2
                    LOSS(0.1) M0
                }
                """
            )
        )


def test_out_of_range_error_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="error index"):
        build_jit_library(
            parse(
                _HEADER
                + """
                GADGET G {
                    INPUT C 0 1 2
                    M 0
                    OUTPUT C 0 1 2
                    LOSS(0.1) SE5 M0
                }
                """
            )
        )


def test_out_of_range_output_qubit_is_rejected() -> None:
    with pytest.raises(ValueError, match="physical qubit"):
        build_jit_library(
            parse(
                _HEADER
                + """
                GADGET G {
                    INPUT C 0 1 2
                    M 0
                    OUTPUT C 0 1 2
                    LOSS(0.1) OUT0.L9 M0
                }
                """
            )
        )


@pytest.mark.parametrize(
    "losses",
    [
        "LOSS(0.1) L0\nLOSS(0.1)",
        "LOSS(0.1)\nLOSS(0.1) L0",
    ],
)
def test_source_loss_children_must_point_forward(losses: str) -> None:
    with pytest.raises(ValueError, match="children must have greater indices"):
        build_jit_library(
            parse(
                _HEADER
                + f"""
                GADGET G {{
                    INPUT C 0 1 2
                    M 0
                    OUTPUT C 0 1 2
                    {losses}
                }}
                """
            )
        )


@pytest.mark.parametrize(
    "loss_statement, duplicate",
    [
        ("LOSS(0.1) SE0 SE0", "source-error reference SE0"),
        ("LOSS(0.1) CE0 CE0", "continuation-error reference CE0"),
        ("LOSS(0.1) L1 L1", "child-loss reference L1"),
        ("LOSS(0.1) OUT0.L0 OUT0.L0", "output-qubit reference OUT0.L0"),
        ("LOSS(0.1) M0 M0", "measurement reference M0"),
        ("LOSS(IN0.L0) CE0 CE0", "continuation-error reference CE0"),
        ("LOSS(IN0.L0) L0 L0", "child-loss reference L0"),
        ("LOSS(IN0.L0) OUT0.L0 OUT0.L0", "output-qubit reference OUT0.L0"),
        ("LOSS(IN0.L0) M0 M0", "measurement reference M0"),
    ],
)
def test_explicit_loss_rejects_duplicate_references(
    loss_statement: str, duplicate: str
) -> None:
    with pytest.raises(ValueError, match=duplicate):
        build_jit_library(
            parse(
                _HEADER
                + f"""
                GADGET G {{
                    INPUT C 0 1 2
                    M 0
                    OUTPUT C 0 1 2
                    ERROR(0.0) OUT0.LX0
                    {loss_statement}
                    LOSS(0.1)
                }}
                """
            )
        )
