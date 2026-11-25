"""Validate the physical 3aux instrument and its compiled surgery readouts."""

import hashlib
from pathlib import Path

import pytest
import stim

from deq.cli.jit import _annotate_stim_with_detectors
from deq.proto.deq_jit_pb2 import JitLibrary
from deq.transpiler.program_timing import gadget_timing
from lattice_surgery import (
    _extraction_round_3aux,
    compile_case,
    extraction_round,
    load_circuit,
    patch_checks,
    surgery_library,
)


def test_default_extraction_is_unchanged():
    source = surgery_library(3, 0.001)
    assert source == surgery_library(3, 0.001, use_3aux=False)
    assert hashlib.sha256(source.encode()).hexdigest() == (
        "f5b4810f430097ffa3d79fcb664fa5d38dbaa46bb2a47ba01993f9e56e27e06f"
    )
    assert "CX " in source
    assert "CX " not in surgery_library(3, 0.001, use_3aux=True)


@pytest.mark.parametrize("check_index", range(8))
def test_3aux_measures_only_the_requested_bulk_or_boundary_parity(check_index: int):
    data = {(row, column): 3 * row + column for row in range(3) for column in range(3)}
    check = patch_checks(data, 3, 3)[check_index]
    lines, syndromes = _extraction_round_3aux(data, [check], 0)
    circuit = stim.Circuit("\n".join(lines))
    stabilizer = stim.PauliString(9)
    for qubit in check.qubits:
        stabilizer[qubit] = check.basis
    assert circuit.has_all_flows(
        [
            stim.Flow(input=stabilizer, output=stabilizer),
            stim.Flow(input=stabilizer, measurements=syndromes[0]),
            stim.Flow(output=stabilizer, measurements=syndromes[0]),
        ],
        unsigned=False,
    )
    assert circuit.num_measurements == (12 if len(check.qubits) == 4 else 4)
    assert len(syndromes[0]) == (6 if len(check.qubits) == 4 else 2)

    conjugate = "X" if check.basis == "Z" else "Z"
    for qubit in check.qubits:
        commuting = stim.PauliString(9)
        commuting[qubit] = check.basis
        assert circuit.has_flow(stim.Flow(input=commuting, output=commuting))
        anticommuting = stim.PauliString(9)
        anticommuting[qubit] = conjugate
        assert circuit.solve_flow_measurements(
            [stim.Flow(input=anticommuting, output=anticommuting)]
        ) == [None]
        for other in check.qubits:
            if other != qubit:
                commuting_pair = anticommuting.copy()
                commuting_pair[other] = conjugate
                assert circuit.solve_flow_measurements(
                    [stim.Flow(input=commuting_pair, output=commuting_pair)]
                )[0] is not None


@pytest.mark.parametrize("rows,columns", [(3, 3), (5, 5), (7, 7), (3, 7), (7, 3)])
@pytest.mark.parametrize("first_basis", ["X", "Z"])
def test_3aux_round_schedule_noise_and_stabilizer_flows(
    rows: int, columns: int, first_basis: str
):
    probability = 0.002
    data = {
        (row, column): row * columns + column
        for row in range(rows)
        for column in range(columns)
    }
    checks = patch_checks(data, rows, columns)
    lines, syndromes = _extraction_round_3aux(
        data, checks, probability, first_basis=first_basis
    )
    circuit = stim.Circuit("\n".join(lines))
    bulk = (rows - 1) * (columns - 1)
    boundary = rows + columns - 2
    assert circuit.num_qubits == rows * columns + 3 * bulk + boundary
    assert circuit.num_measurements == 12 * bulk + 4 * boundary
    assert circuit.num_ticks == 8
    if first_basis == "Z":
        assert lines == extraction_round(data, checks, probability, use_3aux=True)

    used: set[int] = set()
    noised: list[int] = []
    for index, instruction in enumerate(circuit):
        assert instruction.name in {
            "R", "X_ERROR", "M", "MX", "MXX", "MZZ",
            "DEPOLARIZE1", "DEPOLARIZE2", "TICK",
        }
        if instruction.name == "TICK":
            assert sorted(noised) == list(range(circuit.num_qubits))
            used.clear()
            noised.clear()
        elif instruction.name in {"DEPOLARIZE1", "DEPOLARIZE2"}:
            noised.extend(target.value for target in instruction.targets_copy())
        elif instruction.name in {"M", "MX", "MXX", "MZZ"}:
            support = {target.value for target in instruction.targets_copy()}
            assert not used & support
            used.update(support)
            assert instruction.gate_args_copy() == [probability]
            noise = circuit[index + 1]
            assert noise.name == f"DEPOLARIZE{len(support)}"
            assert noise.gate_args_copy() == [probability]
            assert noise.targets_copy()[:len(support)] == instruction.targets_copy()
            if len(support) == 1:
                assert support.isdisjoint(data.values())

    noiseless = circuit.without_noise()
    for check, measurements in zip(checks, syndromes):
        stabilizer = stim.PauliString(rows * columns)
        for qubit in check.qubits:
            stabilizer[qubit] = check.basis
        flow = (
            stim.Flow(input=stabilizer, measurements=measurements)
            if check.basis == first_basis
            else stim.Flow(output=stabilizer, measurements=measurements)
        )
        assert noiseless.has_flow(flow)
        assert all(
            measurements is not None
            for measurements in noiseless.solve_flow_measurements(
                [stim.Flow(input=stabilizer), stim.Flow(output=stabilizer)]
            )
        )


@pytest.mark.parametrize("distance,seed", [(3, 42), (3, 4), (3, 7), (3, 5), (5, 5)])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_3aux_compiled_cnot_has_deterministic_readouts_and_requested_distance(
    tmp_path: Path, distance: int, seed: int, basis: str
):
    configuration = compile_case(
        tmp_path, distance=distance, probability=0.001, width=2, layers=1,
        basis=basis, seed=seed, use_3aux=True,
    )
    assert configuration["use_3aux"] is True
    _, _, _, verification = load_circuit(tmp_path)
    assert verification["source_and_expanded_wiring_agree"]
    library = JitLibrary.FromString((tmp_path / "circuit.deq.jit").read_bytes())
    _annotate_stim_with_detectors(
        library, str(tmp_path / "circuit.stim"), configuration["assertions"]
    )
    circuit = stim.Circuit.from_file(tmp_path / "circuit.stim")
    assert circuit.num_measurements == configuration["measurements"]
    assert circuit.num_detectors > 0
    assert circuit.num_observables == 2
    circuit.detector_error_model()
    assert len(circuit.shortest_graphlike_error()) == distance
    assert not circuit.without_noise().compile_detector_sampler(seed=42).sample(
        128, append_observables=True
    ).any()
    assert circuit.compile_detector_sampler(seed=42).sample(128).any()


def test_3aux_metadata_only_brickwall_preserves_wiring_and_timing(tmp_path: Path):
    configuration = compile_case(
        tmp_path, distance=7, probability=0.001, width=4, layers=2,
        basis="X", seed=42, export_physical=False, use_3aux=True,
    )
    loaded, _, _, verification = load_circuit(tmp_path)
    assert loaded["use_3aux"] is True
    assert not (tmp_path / "circuit.stim").exists()
    assert configuration["input_kind"] == "metadata_only"
    assert verification["cnots"] == 3
    assert verification["remote_pauli_dependencies"] == 9
    library = JitLibrary.FromString((tmp_path / "circuit.deq.jit").read_bytes())
    by_name = {item.base.name: item.base for item in library.gadget_types}
    measurements_per_round = 12 * 6**2 + 4 * 12
    assert len(by_name["PrepareXNonFT"].measurements) == measurements_per_round
    assert len(by_name["Syndrome"].measurements) == 7 * measurements_per_round
    timings = [gadget_timing(item.gadget) for item in library.program]
    assert all(timing is not None for timing in timings)
    assert max(timing.end for timing in timings if timing is not None) == 8 * 70_000_000


@pytest.mark.parametrize("distance,probability", [(2, 0.001), (3, -0.001), (3, 0.5)])
def test_3aux_retains_library_validation(distance: int, probability: float):
    with pytest.raises(ValueError):
        surgery_library(distance, probability, use_3aux=True)
