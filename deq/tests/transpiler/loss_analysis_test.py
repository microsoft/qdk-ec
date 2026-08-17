"""Tests for physical loss-event analysis."""

from __future__ import annotations

import pytest

from deq.circuit.model import GadgetDefinition, Instruction, QubitTarget
from deq.circuit.parser import parse
from deq.transpiler.loss import LossEventGraph, PauliInsertion, analyze_loss_events
from deq.transpiler.loss.analysis import _split_source_occurrences
from deq.transpiler.loss.api import (
    GateLossPolicy,
    QdkLossConfig,
    UnsupportedLossModelError,
)
from deq.transpiler.loss.model_neutral_atom import NeutralAtomLossModel
from deq.transpiler.loss.model_trapped_ion import TrappedIonLossModel


def _gadget(source: str) -> GadgetDefinition:
    qfile = parse(source)
    return next(
        definition
        for definition in qfile.definitions
        if isinstance(definition, GadgetDefinition)
    )


def _discover(source: str) -> LossEventGraph:
    return analyze_loss_events(_gadget(source), NeutralAtomLossModel()).graph


class _PropagatingNeutralAtomLossModel(NeutralAtomLossModel):
    config = QdkLossConfig(gate_policies=(("cz", GateLossPolicy.PROPAGATE),))


def _instruction(source: str) -> Instruction:
    gadget = _gadget(f"GADGET G {{ {source} }}")
    return next(
        statement for statement in gadget.body if isinstance(statement, Instruction)
    )


def _reachable_event_ids(graph: LossEventGraph, event_id: int) -> tuple[int, ...]:
    event_index = {event.event_id: index for index, event in enumerate(graph.events)}
    visited: set[int] = set()
    pending = [event_id]
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        pending.extend(graph.successor_event_ids[event_index[current]])
    return tuple(sorted(visited))


def _complete_loss_measurements(
    graph: LossEventGraph, event_id: int
) -> tuple[int, ...]:
    events = {event.event_id: event for event in graph.events}
    return tuple(
        sorted(
            {
                measurement
                for reachable in _reachable_event_ids(graph, event_id)
                for measurement in events[reachable].loss_measurements
            }
        )
    )


def _complete_pauli_insertions(
    graph: LossEventGraph, event_id: int
) -> tuple[PauliInsertion, ...]:
    events = {event.event_id: event for event in graph.events}
    insertions = set(events[event_id].source_pauli_insertions)
    for reachable in _reachable_event_ids(graph, event_id):
        insertions.update(events[reachable].continuation_pauli_insertions)
    return tuple(sorted(insertions))


class _RecordingHandler:
    native_gates = frozenset()

    def __init__(self) -> None:
        self.gates = []

    def handle_loss_source(self, event_id, state) -> None:
        state.add_source_pauli_insertion(event_id)

    def handle_gate(self, gate, state) -> None:
        del state
        self.gates.append(gate)


def test_paper_data_loss_cnot_circuit_produces_expected_boundaries() -> None:
    table = _discover("""
        GADGET G {
            LOSS_ERROR(0.1) 0
            CX 1 0
            LOSS_ERROR(0.1) 0
            CX 0 2
            LOSS_ERROR(0.1) 0
            CX 0 3
            LOSS_ERROR(0.1) 0
            CX 4 0
            LOSS_ERROR(0.1) 0
            M 0
        }
        """)
    events = table.events
    source_boundaries = [event.source_boundary for event in events]

    assert table.measurement_count == 1
    assert [_complete_loss_measurements(table, event.event_id) for event in events] == [
        (0,)
    ] * 5
    assert [_complete_pauli_insertions(table, event.event_id) for event in events] == [
        (
            PauliInsertion(source_boundaries[0], 0),
            PauliInsertion(source_boundaries[1], 0),
            PauliInsertion(source_boundaries[4], 0),
        ),
        (
            PauliInsertion(source_boundaries[1], 0),
            PauliInsertion(source_boundaries[4], 0),
        ),
        (
            PauliInsertion(source_boundaries[2], 0),
            PauliInsertion(source_boundaries[4], 0),
        ),
        (
            PauliInsertion(source_boundaries[3], 0),
            PauliInsertion(source_boundaries[4], 0),
        ),
        (PauliInsertion(source_boundaries[4], 0),),
    ]
    assert table.successor_event_ids == ((1,), (2,), (3,), (4,), ())


def test_consecutive_cnot_targets_share_the_later_suffix() -> None:
    table = _discover("""
        GADGET G {
            LOSS_ERROR(0.1) 0
            CX 1 0
            LOSS_ERROR(0.1) 0
            CX 2 0
            M 0
        }
        """)
    first, second = table.events

    assert _complete_pauli_insertions(table, first.event_id) == (
        PauliInsertion(first.source_boundary, 0),
        PauliInsertion(1, 0),
        PauliInsertion(2, 0),
    )
    assert _complete_pauli_insertions(table, second.event_id) == (
        PauliInsertion(second.source_boundary, 0),
        PauliInsertion(2, 0),
    )


def test_native_cz_and_s_do_not_insert_but_sx_does() -> None:
    (event,) = _discover("""
        GADGET G {
            LOSS_ERROR(0.1) 0
            CZ 0 1  # H 1; CX 0 1; H 1
            S 0
            SQRT_X 0  # H 0; S 0 ; H 0
            S 0
            M 0
        }
        """).events

    assert event.local_pauli_insertions == (
        PauliInsertion(0, 0),
        PauliInsertion(7, 0),
        PauliInsertion(8, 0),
    )


def test_neutral_atom_model_moves_loss_through_physical_swap() -> None:
    (event,) = analyze_loss_events(
        _gadget("""
            GADGET G {
                LOSS_ERROR(0.1) 0
                SWAP 0 1
                M 0 1
            }
            """),
        NeutralAtomLossModel(),
    ).graph.events

    assert event.loss_measurements == (1,)
    assert event.source_pauli_insertions == (PauliInsertion(0, 0),)
    assert event.continuation_pauli_insertions == (PauliInsertion(1, 1),)


@pytest.mark.parametrize("lost_qubit", [0, 1])
def test_propagate_policy_branches_loss_to_every_gate_operand(
    lost_qubit: int,
) -> None:
    survivor = 1 - lost_qubit
    (event,) = analyze_loss_events(
        _gadget(f"""
            GADGET G {{
                LOSS_ERROR(0.1) {lost_qubit}
                CZ 0 1
                CZ 0 1
                M 0 1
            }}
            """),
        _PropagatingNeutralAtomLossModel(),
    ).graph.events

    branches = {branch.qubit: branch for branch in event.branches}
    assert event.affected_qubits == (0, 1)
    assert len(event.branches) == 2
    assert branches[lost_qubit].loss_boundary == 0
    assert branches[lost_qubit].continuation_pauli_insertions == (
        PauliInsertion(6, lost_qubit),
    )
    assert branches[survivor].loss_boundary == 3
    assert branches[survivor].continuation_pauli_insertions == (
        PauliInsertion(3, survivor),
        PauliInsertion(6, survivor),
    )
    assert branches[0].loss_measurements == (0,)
    assert branches[1].loss_measurements == (1,)


@pytest.mark.parametrize("gate", ["CX", "CY", "CZ"])
@pytest.mark.parametrize("lost_qubit", [0, 1])
def test_neutral_atom_supports_compiled_controlled_pauli_aliases(
    gate: str, lost_qubit: int
) -> None:
    (event,) = _discover(f"""
        GADGET G {{
            LOSS_ERROR(0.1) {lost_qubit}
            {gate} 0 1
            M 0 1
        }}
        """).events

    assert event.affected_qubits == (lost_qubit,)
    assert event.loss_measurements == (lost_qubit,)


@pytest.mark.parametrize("gate", ["CX", "CY"])
def test_neutral_atom_lost_target_inserts_only_on_lost_qubit(gate: str) -> None:
    (event,) = _discover(f"""
        GADGET G {{
            LOSS_ERROR(0.1) 1
            {gate} 0 1
            M 0 1
        }}
        """).events

    assert [
        (insertion.qubit, insertion.generators)
        for insertion in event.continuation_pauli_insertions
    ] == [(1, ("X", "Z"))]


@pytest.mark.parametrize("model", [NeutralAtomLossModel(), TrappedIonLossModel()])
@pytest.mark.parametrize(
    ("gate", "generator"),
    [("CX", "X"), ("CY", "Y"), ("CZ", "Z")],
)
def test_lost_measurement_skips_classically_controlled_pauli(
    model, gate: str, generator: str
) -> None:
    (event,) = analyze_loss_events(
        _gadget(f"""
            GADGET G {{
                LOSS_ERROR(0.1) 1
                M 1
                {gate} rec[-1] 0
                M 0
            }}
            """),
        model,
    ).graph.events

    assert event.loss_measurements == (0,)
    assert any(
        insertion.qubit == 0 and insertion.generators == (generator,)
        for insertion in event.continuation_pauli_insertions
    )


def test_nonlost_measurement_control_adds_no_pauli_insertion() -> None:
    (event,) = _discover("""
        GADGET G {
            LOSS_ERROR(0.1) 2
            M 1
            CX rec[-1] 0
            M 2
        }
        """).events

    assert all(
        insertion.qubit != 0 for insertion in event.continuation_pauli_insertions
    )


def test_classical_control_resolves_older_measurement_record() -> None:
    (event,) = _discover("""
        GADGET G {
            LOSS_ERROR(0.1) 2
            M 2
            M 1
            CX rec[-2] 0
        }
        """).events

    assert any(
        insertion.qubit == 0 and insertion.generators == ("X",)
        for insertion in event.continuation_pauli_insertions
    )


def test_neutral_atom_config_round_trips_canonical_json() -> None:
    config = NeutralAtomLossModel.config

    assert config.policy_for("cx") == "SKIP"
    assert config.policy_for("cy") == "SKIP"
    assert config.policy_for("cz") == "SKIP"
    assert config.policy_for("swap") == "APPLY_ANYWAY"
    assert QdkLossConfig.from_json_object(config.to_json_object()) == config
    assert config.to_json() == (
        '{"cx":"SKIP","cy":"SKIP","cz":"SKIP","swap":"APPLY_ANYWAY"}'
    )


def test_qdk_loss_config_normalizes_direct_string_policies() -> None:
    config = QdkLossConfig(
        gate_policies=tuple(
            (gate, policy)
            for gate, policy in NeutralAtomLossModel.config.to_json_object().items()
        ),
    )

    assert config == NeutralAtomLossModel.config


def test_trapped_ion_config_matches_platform_rules() -> None:
    config = TrappedIonLossModel.config

    assert config.policy_for("cz") == "RESIDUAL_S_DAGGER"
    assert config.policy_for("swap") == "APPLY_ANYWAY"
    assert config.to_json() == '{"cz":"RESIDUAL_S_DAGGER","swap":"APPLY_ANYWAY"}'


@pytest.mark.parametrize("lost_qubit", [0, 1])
def test_trapped_ion_cz_adds_s_dagger_envelope_to_partner(lost_qubit: int) -> None:
    (event,) = analyze_loss_events(
        _gadget(f"""
            GADGET G {{
                LOSS_ERROR(0.1) {lost_qubit}
                CZ 0 1
                M 0 1
            }}
            """),
        TrappedIonLossModel(),
    ).graph.events

    assert event.affected_qubits == (lost_qubit,)
    assert event.loss_measurements == (lost_qubit,)
    assert any(
        insertion.qubit == 1 - lost_qubit and insertion.generators == ("Z",)
        for insertion in event.continuation_pauli_insertions
    )


@pytest.mark.parametrize("gate", ["CX", "CY"])
def test_trapped_ion_rejects_controlled_gate_without_supported_decomposition(
    gate: str,
) -> None:
    with pytest.raises(
        UnsupportedLossModelError,
        match=rf"supports CZ only; {gate} requires",
    ):
        analyze_loss_events(
            _gadget(f"""
                GADGET G {{
                    LOSS_ERROR(0.1) 0
                    {gate} 0 1
                    M 0 1
                }}
                """),
            TrappedIonLossModel(),
        )


def test_lost_cx_control_adds_no_cx_insertion() -> None:
    (event,) = _discover("""
        GADGET G {
            LOSS_ERROR(0.1) 0
            CX 0 1
            M 0 1
        }
        """).events

    assert all(insertion.qubit == 0 for insertion in event.local_pauli_insertions)
    assert all(
        insertion.generators == ("X", "Z") for insertion in event.local_pauli_insertions
    )


def test_hadamards_add_pauli_boundaries_after_each_gate() -> None:
    (event,) = _discover("""
        GADGET G {
            LOSS_ERROR(0.1) 0
            H 0
            S 0
            H 0
            M 0
        }
        """).events

    assert tuple(insertion.boundary for insertion in event.local_pauli_insertions) == (
        0,
        1,
        3,
    )


def test_measurement_indices_include_padding_measurements() -> None:
    table = _discover("""
        GADGET G {
            MPAD 0
            LOSS_ERROR(0.1) 0 1
            M 1 0
        }
        """)

    assert table.measurement_count == 3
    assert [event.source_qubit for event in table.events] == [0, 1]
    assert [event.loss_measurements for event in table.events] == [(2,), (1,)]


def test_measurement_does_not_terminate_loss_lifetime() -> None:
    table = _discover("""
        GADGET G {
            REPEAT 2 {
                LOSS_ERROR(0.1) 0
                M 0
            }
        }
        """)

    assert [
        _complete_loss_measurements(table, event.event_id) for event in table.events
    ] == [
        (0, 1),
        (1,),
    ]
    assert table.successor_event_ids == ((1,), ())


def test_nested_repeats_preserve_loss_and_measurement_order() -> None:
    table = _discover("""
        GADGET G {
            REPEAT 2 {
                REPEAT 2 {
                    LOSS_ERROR(0.1) 0
                    M 0
                }
            }
        }
        """)

    assert table.measurement_count == 4
    assert [event.loss_measurements for event in table.events] == [
        (0,),
        (1,),
        (2,),
        (3,),
    ]
    assert table.successor_event_ids == ((1,), (2,), (3,), ())


def test_suffix_sharing_excludes_later_source_insertion() -> None:
    table = _discover("""
        GADGET G {
            LOSS_ERROR(0.1) 0
            S 0
            LOSS_ERROR(0.1) 0
            H 0
            M 0
        }
        """)

    assert table.successor_event_ids == ((1,), ())
    assert table.events[0].source_pauli_insertions == (PauliInsertion(0, 0),)
    assert table.events[0].continuation_pauli_insertions == ()
    assert table.events[1].source_pauli_insertions == (PauliInsertion(1, 0),)
    assert table.events[1].continuation_pauli_insertions == (PauliInsertion(2, 0),)
    assert _complete_pauli_insertions(table, 0) == (
        PauliInsertion(0, 0),
        PauliInsertion(2, 0),
    )


def test_deep_loss_chain_stores_each_suffix_once() -> None:
    depth = 100
    body = "\n".join("LOSS_ERROR(0.1) 0\nH 0" for _ in range(depth))
    table = _discover(f"GADGET G {{ {body}\nM 0 }}")

    assert len(table.events) == depth
    assert sum(len(event.local_pauli_insertions) for event in table.events) <= 2 * depth
    assert table.successor_event_ids[:-1] == tuple(
        (event_id + 1,) for event_id in range(depth - 1)
    )
    assert _complete_loss_measurements(table, 0) == (0,)


def test_measure_reset_resolves_loss_before_starting_fresh_lifetime() -> None:
    table = _discover("""
        GADGET G {
            LOSS_ERROR(0.1) 0
            MR 0
            LOSS_ERROR(0.1) 0
            M 0
        }
        """)

    assert [event.loss_measurements for event in table.events] == [(0,), (1,)]
    assert [event.source_boundary for event in table.events] == [0, 2]


@pytest.mark.parametrize("reset", ["RX", "RY"])
def test_basis_reset_terminates_loss_without_heralding(reset: str) -> None:
    (event,) = _discover(f"GADGET G {{ LOSS_ERROR(0.1) 0 {reset} 0 M 0 }}").events

    assert event.loss_measurements == ()


@pytest.mark.parametrize("measure_reset", ["MRX", "MRY", "MRZ"])
def test_basis_measure_reset_heralds_then_terminates_loss(
    measure_reset: str,
) -> None:
    table = _discover(f"GADGET G {{ LOSS_ERROR(0.1) 0 {measure_reset} 0 M 0 }}")
    (event,) = table.events

    assert table.measurement_count == 2
    assert event.loss_measurements == (0,)


@pytest.mark.parametrize(
    "measurement, expected_boundaries",
    [("M", (0,)), ("MX", (0, 1, 3)), ("MY", (0, 2, 4))],
)
def test_single_qubit_measurement_bases_resolve_loss(
    measurement: str, expected_boundaries: tuple[int, ...]
) -> None:
    table = _discover(f"""
        GADGET G {{
            LOSS_ERROR(0.1) 0
            {measurement} 0
        }}
        """)
    (event,) = table.events

    assert event.loss_measurements == (0,)
    assert (
        tuple(insertion.boundary for insertion in event.local_pauli_insertions)
        == expected_boundaries
    )


def test_zero_probability_loss_location_is_omitted() -> None:
    table = _discover("""
        GADGET G {
            LOSS_ERROR(0) 0
            M 0
        }
        """)

    assert table.events == ()
    assert table.measurement_count == 1


@pytest.mark.parametrize("target", ["rec[-1]", "sweep[0]", "X1"])
def test_loss_error_rejects_non_qubit_targets(target: str) -> None:
    with pytest.raises(ValueError, match="accepts only qubit targets"):
        _discover(f"GADGET G {{ M 0 LOSS_ERROR(0.1) 0 {target} M 0 }}")


def test_no_loss_events_still_count_general_measurements() -> None:
    table = _discover("""
        GADGET G {
            MXX 0 1
            MPP Z0*Z1 X2
        }
        """)

    assert table.events == ()
    assert table.measurement_count == 3


def test_reset_can_terminate_loss_without_a_loss_measurement() -> None:
    table = _discover("""
        GADGET G {
            LOSS_ERROR(0.1) 0
            R 0
            M 0
        }
        """)
    (event,) = table.events

    assert event.loss_measurements == ()


def test_loss_at_gadget_end_can_be_unheralded() -> None:
    table = _discover("""
        GADGET G {
            LOSS_ERROR(0.1) 0
            H 0
        }
        """)
    (event,) = table.events

    assert event.loss_measurements == ()
    assert event.local_pauli_insertions == (
        PauliInsertion(boundary=0, qubit=0),
        PauliInsertion(boundary=1, qubit=0),
    )


def test_analysis_result_maps_entering_loss_to_gadget_exit() -> None:
    result = analyze_loss_events(
        _gadget("""
            CODE C [[1,1,1]] { LOGICAL X0 Z0 }
            GADGET G {
                INPUT C 0
                H 0
                OUTPUT C 0
            }
            """),
        NeutralAtomLossModel(),
    )

    assert result.input_event_id_by_qubit == {0: 0}
    assert result.exit_qubits_by_event == {0: (0,)}
    assert result.graph.events[0].source_qubit == 0


def test_loss_in_gadget_with_ports_is_analyzed() -> None:
    gadget = _gadget("""
        CODE C [[1,1,1]] {
            LOGICAL X0 Z0
        }

        GADGET G {
            INPUT C 0
            LOSS_ERROR(0.1) 0
            M 0
            OUTPUT C 0
        }
        """)

    result = analyze_loss_events(gadget, NeutralAtomLossModel())
    assert len(result.graph.events) == 2
    assert result.input_event_id_by_qubit == {0: 1}
    assert _complete_loss_measurements(result.graph, 0) == (0,)
    assert _complete_loss_measurements(result.graph, 1) == (0,)


def test_pair_measurement_uses_stim_decomposition_fallback() -> None:
    gadget = _gadget("""
        GADGET G {
            LOSS_ERROR(0.1) 0
            MXX 0 1
        }
        """)

    table = analyze_loss_events(gadget, NeutralAtomLossModel()).graph

    assert table.measurement_count == 1
    assert table.events[0].loss_measurements == (0,)


def test_overlapping_mpp_products_preserve_decomposed_boundaries() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.handler = _RecordingHandler()

        def create_handler(self):
            return self.handler

    model = RecordingModel()
    analyze_loss_events(
        _gadget("""
            GADGET G {
                LOSS_ERROR(0.1) 1
                MPP X0*X1 Z1*Z2
            }
            """),
        model,
    )

    assert [
        (
            gate.name,
            gate.qubits,
            gate.measurement_index,
            gate.boundary_before,
            gate.boundary_after,
        )
        for gate in model.handler.gates
    ] == [
        ("H", (0,), None, 0, 1),
        ("H", (1,), None, 0, 1),
        ("CX", (1, 0), None, 1, 2),
        ("M", (0,), 0, 2, 3),
        ("CX", (1, 0), None, 3, 4),
        ("H", (0,), None, 4, 5),
        ("H", (1,), None, 4, 5),
        ("CX", (2, 1), None, 5, 6),
        ("M", (1,), 1, 6, 7),
        ("CX", (2, 1), None, 7, 8),
    ]


def test_loss_model_handler_receives_individual_gate_occurrences() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.handler = _RecordingHandler()

        def create_handler(self):
            return self.handler

    model = RecordingModel()
    analyze_loss_events(
        _gadget("""
            GADGET G {
                LOSS_ERROR(0.1) 0
                H 0 1
                CX 0 1 2 3
                M 0 1
                R 0
            }
            """),
        model,
    )

    assert [(gate.name, gate.qubits) for gate in model.handler.gates] == [
        ("H", (0,)),
        ("H", (1,)),
        ("CX", (0, 1)),
        ("CX", (2, 3)),
        ("M", (0,)),
        ("M", (1,)),
        ("R", (0,)),
    ]
    assert [
        gate.measurement_index
        for gate in model.handler.gates
        if gate.produces_measurement
    ] == [0, 1]


def test_non_native_gate_uses_stim_decomposition() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.handler = _RecordingHandler()

        def create_handler(self):
            return self.handler

    model = RecordingModel()
    analyze_loss_events(_gadget("GADGET G { LOSS_ERROR(0.1) 0 CZ 0 1 M 0 }"), model)

    assert [(gate.name, gate.qubits) for gate in model.handler.gates[:3]] == [
        ("H", (1,)),
        ("CX", (0, 1)),
        ("H", (1,)),
    ]
    assert all(gate.source_name == "CZ" for gate in model.handler.gates[:3])


def test_classical_control_uses_stim_decomposition_fallback() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.handler = _RecordingHandler()

        def create_handler(self):
            return self.handler

    model = RecordingModel()
    analyze_loss_events(
        _gadget("""
            GADGET G {
                LOSS_ERROR(0.1) 1
                M 1
                CX rec[-1] 0
            }
            """),
        model,
    )

    classical_gate = model.handler.gates[1]
    assert classical_gate.name == "CX"
    assert classical_gate.qubits == (0,)
    assert classical_gate.control_measurement_index == 0


def test_source_gate_override_bypasses_stim_decomposition() -> None:
    class RecordingHandler(_RecordingHandler):
        native_gates = frozenset({"CZ"})

    class RecordingModel:
        def __init__(self) -> None:
            self.handler = RecordingHandler()

        def create_handler(self):
            return self.handler

    model = RecordingModel()
    analyze_loss_events(_gadget("GADGET G { LOSS_ERROR(0.1) 0 CZ 0 1 M 0 }"), model)

    assert (model.handler.gates[0].name, model.handler.gates[0].qubits) == (
        "CZ",
        (0, 1),
    )


@pytest.mark.parametrize(
    "source, expected",
    [
        ("MPAD 0 1", ["MPAD 0", "MPAD 1"]),
        ("MPP X0*Y1 Z2", ["MPP X0*Y1", "MPP Z2"]),
        ("H 0 1", ["H 0", "H 1"]),
        ("CX 0 1 2 3", ["CX 0 1", "CX 2 3"]),
        ("CX rec[-1] 0 rec[-2] 1", ["CX rec[-1] 0", "CX rec[-2] 1"]),
    ],
)
def test_split_source_occurrences_preserves_gate_groups(
    source: str, expected: list[str]
) -> None:
    occurrences = _split_source_occurrences(_instruction(source))

    assert [str(occurrence) for occurrence in occurrences] == expected


def test_split_source_occurrences_rejects_non_qubit_targets() -> None:
    with pytest.raises(UnsupportedLossModelError, match="with non-qubit targets"):
        _split_source_occurrences(_instruction("CX sweep[0] 0"))


def test_split_source_occurrences_rejects_incomplete_gate_group() -> None:
    with pytest.raises(ValueError, match="requires target groups of size 2"):
        _split_source_occurrences(Instruction(name="CX", targets=[QubitTarget(0)]))


def test_classical_control_rejects_measurement_before_gadget() -> None:
    with pytest.raises(ValueError, match="before the gadget"):
        _discover("GADGET G { LOSS_ERROR(0.1) 1 CX rec[-1] 0 }")


@pytest.mark.parametrize(
    "statement, message",
    [
        ("LOSS_ERROR 0", "requires exactly one probability"),
        ("LOSS_ERROR(0.1, 0.2) 0", "requires exactly one probability"),
        ("LOSS_ERROR(-0.1) 0", r"must be in \[0, 1\]"),
        ("LOSS_ERROR(1.1) 0", r"must be in \[0, 1\]"),
        ("LOSS_ERROR(0.1)", "requires at least one qubit target"),
        ("LOSS_ERROR(0.1) 0 0", "contains a duplicate qubit target"),
        ("LOSS_ERROR(0.1) !0", "qubit targets cannot be inverted"),
    ],
)
def test_loss_error_rejects_invalid_source(statement: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _discover(f"GADGET G {{ {statement} }}")


def test_loss_model_must_return_complete_handler() -> None:
    class InvalidModel:
        def create_handler(self) -> object:
            return object()

    with pytest.raises(TypeError, match="does not implement loss-source and gate"):
        analyze_loss_events(_gadget("GADGET G { M 0 }"), InvalidModel())
