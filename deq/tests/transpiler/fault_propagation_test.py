"""Tests for shared circuit lowering and the circuit-fault propagation timeline."""

from typing import cast

import pytest
import stim

from deq.circuit.model import GadgetDefinition, GadgetStatement, Instruction
from deq.circuit.parser import parse
from deq.transpiler.circuit_lowering import build_decomposed_body, flatten_body
from deq.transpiler.fault_propagation import (
    ErrorProjectionContext,
    MechanismFlips,
    propagate_pauli_mechanisms,
)
from deq.transpiler.jit_transpiler import (
    checks_equivalent,
    derive_checks_auto,
)


def _raw_body(source: str):
    qfile = parse(source)
    gadget = next(
        definition
        for definition in qfile.definitions
        if isinstance(definition, GadgetDefinition)
    )
    return list(gadget.body)


def _body(source: str):
    return flatten_body(_raw_body(source))


def test_loss_error_uses_the_following_gate_boundary() -> None:
    timeline = build_decomposed_body(
        _body("GADGET G { H 0 LOSS_ERROR(0.1) 0 MX 0 }")
    )

    assert timeline.body_start_at == (0, 1, 1)
    assert [instruction.name for instruction in timeline.instructions] == [
        "H",
        "H",
        "M",
        "H",
    ]
    assert timeline.measurement_start_at == (0, 0, 0, 1)
    assert timeline.total_measurements == 1


def test_adjacent_source_gates_keep_distinct_boundaries() -> None:
    timeline = build_decomposed_body(_body("GADGET G { H 0 H 1 }"))

    assert timeline.body_start_at == (0, 1)
    assert [str(instruction) for instruction in timeline.instructions] == [
        "H 0",
        "H 1",
    ]


def test_user_tick_does_not_participate_in_boundary_mapping() -> None:
    timeline = build_decomposed_body(_body("GADGET G { H 0 TICK MX 1 }"))

    assert timeline.body_start_at == (0, 1, 1)
    assert [str(instruction) for instruction in timeline.instructions] == [
        "H 0",
        "H 1",
        "M 1",
        "H 1",
    ]


def test_unflattened_repeat_block_is_rejected() -> None:
    with pytest.raises(ValueError, match="call flatten_body"):
        build_decomposed_body(
            _raw_body("GADGET G { REPEAT 2 { M 0 } }")
        )


def test_unknown_body_block_is_rejected() -> None:
    class UnknownBlock:
        body: list[object] = []

    with pytest.raises(TypeError, match="UnknownBlock"):
        build_decomposed_body(
            [cast(GadgetStatement, UnknownBlock())]
        )


def test_measurement_count_uses_stim_instruction_metadata() -> None:
    timeline = build_decomposed_body(
        _body("GADGET G { MPAD 0 H 0 }")
    )

    assert timeline.measurement_start_at == (0, 1)
    assert timeline.total_measurements == 1


@pytest.mark.parametrize("gate", [
    "T", "TX", "TY", "T_DAG", "TX_DAG", "TY_DAG",
    "R_X(0.125)", "R_Y(-0.375)", "R_Z(0.3)",
])
@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_checks_and_flows_use_equivalent_lowered_circuit(gate, basis):
    gadget = parse(f"GADGET G {{ R{basis} 0 {gate} 0 M{basis} 0 M{basis} 0 }}").definitions[0]
    timeline = build_decomposed_body(flatten_body(gadget.body))
    circuit = stim.Circuit()
    for instruction in timeline.instructions:
        circuit.append(instruction)
    flow_checks = [
        (frozenset(flow.measurements_copy()), flow.input_copy().sign != flow.output_copy().sign)
        for flow in circuit.flow_generators()
        if not any(flow.input_copy()) and not any(flow.output_copy())
    ]
    checks, total = derive_checks_auto(gadget, {})
    assert total == timeline.total_measurements == circuit.num_measurements == 2
    assert timeline.qubit_count == 2
    assert checks_equivalent(checks, flow_checks, total)


def test_repeated_rotations_have_independent_dephasing():
    gadget = parse("GADGET G { RX 0 REPEAT 2 { T 0 MX 0 } }").definitions[0]
    assert derive_checks_auto(gadget, {}) == ([], 2)


def test_lowering_reserves_private_ancilla_beyond_port_qubits():
    gadget = parse("""
        CODE C [[1,1,1]] { LOGICAL X0 Z0 }
        GADGET G {
            INPUT C 9
            RX 0
            T 0
            MX 0
            OUTPUT C 9
        }
    """).definitions[1]
    source = [str(statement) for statement in gadget.body]
    timeline = build_decomposed_body(flatten_body(gadget.body))
    assert timeline.qubit_count == 11
    touched = {
        target.value for instruction in timeline.instructions
        for target in instruction.targets_copy() if target.is_qubit_target
    }
    assert touched == {0, 10}
    assert [str(statement) for statement in gadget.body] == source
    assert "T 0" in source


def test_lowering_preserves_noise_and_measurement_boundaries():
    body = _body("GADGET G { RX 0 Z_ERROR(0.1) 0 T 0 X_ERROR(0.2) 0 MX 0 }")
    source = [str(statement) for statement in body]
    timeline = build_decomposed_body(body)
    rotation_start = timeline.body_start_at[2]
    rotation_end = timeline.body_start_at[4]
    assert timeline.body_start_at[1] == rotation_start
    assert timeline.body_start_at[3] == rotation_end
    assert rotation_start < rotation_end
    assert set(timeline.measurement_start_at[rotation_start:rotation_end]) == {0}
    assert {instruction.name for instruction in timeline.instructions} <= {"H", "S", "CX", "M", "R", "MPAD"}
    assert timeline.total_measurements == 1
    assert [str(statement) for statement in body] == source


def test_lowering_respects_decoder_visibility():
    gadget = parse("""
        GADGET G {
            RX 0
            @SIMULATE_ONLY
            T 0
            @DECODE_ONLY
            Z 0
            MX 0
        }
    """).definitions[0]
    assert derive_checks_auto(gadget, {}) == ([(frozenset({0}), True)], 1)
    timeline = build_decomposed_body(flatten_body(gadget.body))
    assert timeline.qubit_count == 1
    physical = [statement.name for statement in flatten_body(gadget.body, for_simulate=True)
                if isinstance(statement, Instruction)]
    assert physical == ["RX", "T", "MX"]


@pytest.mark.parametrize("rotation", ["", "T 0", "REPEAT 4 { T 0 }"])
def test_propagation_uses_decoder_capacity_with_padded_paulis(rotation):
    body = build_decomposed_body(_body(f"""
        GADGET G {{
            R 0
            @SIMULATE_ONLY
            X 9
            X_ERROR(0.1) 0
            {rotation}
            M 0
        }}
    """))
    assert body.qubit_count == (2 if rotation else 1)
    fault = stim.PauliString("X_________")
    observable = stim.PauliString("Z_________")
    flips = propagate_pauli_mechanisms(
        [(body.body_start_at[1], fault)], body, [observable], [observable]
    )
    assert flips == [MechanismFlips(
        flipped_real={0}, output_stabilizer_flips=[True], frame_column_flips=[True]
    )]


def test_propagation_includes_qubits_referenced_only_by_ports():
    body = build_decomposed_body(_body("""
        CODE C [[1,1,1]] { LOGICAL X0 Z0 }
        GADGET G { INPUT C 9 OUTPUT C 9 }
    """))
    assert body.qubit_count == 10
    assert body.instructions == ()
    fault = stim.PauliString("_________X")
    observable = stim.PauliString("_________Z")
    flips = propagate_pauli_mechanisms([(0, fault)], body, [observable], [observable])
    assert flips == [MechanismFlips(
        flipped_real=set(), output_stabilizer_flips=[True], frame_column_flips=[True]
    )]


def test_output_stabilizer_measurement_index_uses_offset() -> None:
    context = ErrorProjectionContext(
        input_virtual_count=0,
        finished_member_lists=(),
        unfinished_member_lists=(),
        output_stabilizer_measurement_offset=7,
        readout_measurement_sets=(),
        logical_columns=set(),
        unfinished_to_column=(),
        physical_correction_by_logical={},
    )

    assert context.output_stabilizer_measurement_index(3) == 10