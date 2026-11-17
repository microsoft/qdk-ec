"""Check surgery distance, logical feed-forward, and measurement timing."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jit-basics"))

import pytest
import stim

from deq.proto.deq_jit_pb2 import JitLibrary
from deq.proto.fastrace_pb2 import Property, SpanRecord
from deq.transpiler.program_timing import gadget_timing
from lattice_surgery import (
    brickwall_program,
    compile_case,
    patch_checks,
    surgery_library,
)
from lattice_surgery_latency import (
    run_case,
    validate_distance,
)
from readout_latency import window_event_times


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_cnot_has_requested_circuit_distance(tmp_path: Path, distance: int, basis: str):
    compile_case(
        tmp_path,
        distance=distance,
        probability=0.001,
        width=2,
        layers=1,
        basis=basis,
        seed=5,
    )
    result = validate_distance(tmp_path)
    assert result["graphlike_circuit_distance"] == distance
    assert result["observables"] == 2


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("seed", [42, 4, 7, 5])
@pytest.mark.parametrize("registration_lead_seconds", [None, 0.002])
def test_cnot_noiseless_truth_tables(
    tmp_path: Path, basis: str, seed: int, registration_lead_seconds: float | None
):
    compile_case(
        tmp_path,
        distance=3,
        probability=0.001,
        width=2,
        layers=1,
        basis=basis,
        seed=seed,
        block_seconds=0.001,
        export_physical=False,
    )
    result = run_case(
        tmp_path,
        shots=8,
        threads=2,
        noiseless=True,
        registration_lead_seconds=registration_lead_seconds,
    )
    assert result["logical_errors"] == 0
    assert result["shots"] == 8
    assert result["input_kind"] == "physical_stim_sampling"
    assert not (tmp_path / "circuit.stim").exists()


@pytest.mark.parametrize("distance", [3, 5])
def test_brickwall_is_reusable_and_scheduled_by_parallel_rounds(
    tmp_path: Path, distance: int
):
    result = compile_case(
        tmp_path,
        distance=distance,
        probability=0.001,
        width=4,
        layers=3,
        basis="X",
        seed=17,
        block_seconds=0.001,
    )
    assert result["cnot_count"] == 5
    assert max(result["epochs"].values()) == 11
    library = JitLibrary.FromString((tmp_path / "circuit.deq.jit").read_bytes())
    assert [item.gadget.gid for item in library.program] == list(
        range(1, len(library.program) + 1)
    )
    seen: set[int] = set()
    corrections = []
    for instruction in library.program:
        instance = instruction.gadget
        dependencies = {connector.gid for connector in instance.connectors}
        remote = instance.modifier.remote_conditional_correction.remote_readouts
        dependencies.update(readout.gid for readout in remote)
        assert dependencies <= seen
        seen.add(instance.gid)
        if remote:
            corrections.append(instance)
    assert len(corrections) == 3 * result["cnot_count"]
    assert validate_distance(tmp_path)["graphlike_circuit_distance"] == distance
    decoded = run_case(
        tmp_path,
        shots=8,
        threads=2,
        noiseless=True,
    )
    assert decoded["logical_errors"] == 0


def test_native_export_follows_source_completion_and_measurement_order(tmp_path: Path):
    import re
    from decimal import Decimal

    configuration = compile_case(
        tmp_path, distance=3, probability=0.001, width=4, layers=2,
        basis="Z", seed=42, block_seconds=0.001,
    )
    library = JitLibrary.FromString((tmp_path / "circuit.deq.jit").read_bytes())
    types = {item.base.gtype: item.base for item in library.gadget_types}
    physical = (tmp_path / "circuit.stim").read_text()
    completions = [
        gadget_timing(item.gadget).end
        for item in library.program
    ]
    assert completions == sorted(completions)
    assert all(gadget_timing(item.gadget) is not None for item in library.program)
    elapsed = 0
    observed = []
    for line in physical.splitlines():
        if line.startswith("#!delay "):
            elapsed += int(Decimal(line.split()[1]) * 1_000_000_000)
        match = re.match(r"# G(\d+):", line)
        if match:
            gid = int(match[1])
            observed.append(gid)
            assert elapsed == completions[gid - 1]
    assert observed == list(range(1, configuration["gadgets"] + 1))
    measurements = [
        stim.Circuit(part).num_measurements
        for part in re.split(r"# G\d+:[^\n]*\n", physical)[1:]
    ]
    assert measurements == [
        len(types[item.gadget.gtype].measurements) for item in library.program
    ]
    assert sum(measurements) == configuration["measurements"]


@pytest.mark.parametrize("distance", [3, 5, 7])
def test_surface_check_generators_commute(distance: int):
    data = {
        (row, column): row * distance + column
        for row in range(distance)
        for column in range(distance)
    }
    checks = patch_checks(data, distance, distance)
    stabilizers = []
    for check in checks:
        pauli = stim.PauliString(distance**2)
        for qubit in check.qubits:
            pauli[qubit] = check.basis
        stabilizers.append(pauli)
    assert len(stabilizers) == distance**2 - 1
    assert all(left.commutes(right) for left in stabilizers for right in stabilizers)


@pytest.mark.parametrize("distance", [3, 5, 7])
def test_ancilla_preparation_is_one_overlapped_round(tmp_path: Path, distance: int):
    from deq.circuit.parser import parse
    from deqagram import GadgetDefinition

    configuration = compile_case(
        tmp_path,
        distance=distance,
        probability=0.001,
        width=2,
        layers=1,
        basis="X",
        seed=42,
        export_physical=False,
    )
    library = JitLibrary.FromString((tmp_path / "circuit.deq.jit").read_bytes())
    types = {item.base.gtype: item.base for item in library.gadget_types}
    by_name = {item.name: item for item in types.values()}
    checks_per_round = distance**2 - 1
    assert len(by_name["PrepareXNonFT"].measurements) == checks_per_round
    assert len(by_name["PrepareXNonFT"].inputs) == 0
    assert len(by_name["PrepareXNonFT"].outputs) == 1
    assert "PrepareAncilla" not in by_name
    assert "PrepareAncillaBegin" not in by_name
    for name in ("PrepareX0", "PrepareX1", "PrepareZ0", "PrepareZ1", "Syndrome"):
        assert len(by_name[name].measurements) == distance * checks_per_round
    source = parse((tmp_path / "circuit.deq").read_text())
    preparation = next(
        item
        for item in source.definitions
        if isinstance(item, GadgetDefinition) and item.name == "PrepareXNonFT"
    )
    assert preparation.input_ports == []
    assert len(preparation.output_ports) == 1
    ancilla = next(
        item.gadget
        for item in library.program
        if types[item.gadget.gtype].name == "PrepareXNonFT"
    )
    assert not ancilla.connectors
    assert gadget_timing(ancilla) is not None
    assert gadget_timing(ancilla).duration == configuration["block_ns"] // distance
    merge = next(
        item.gadget
        for item in library.program
        if types[item.gadget.gtype].name == "MZZ"
    )
    assert merge.connectors[1].gid == ancilla.gid
    assert gadget_timing(ancilla).start == configuration["block_ns"] - gadget_timing(ancilla).duration
    assert gadget_timing(merge).start == configuration["block_ns"]
    assert gadget_timing(merge).duration == configuration["block_ns"]
    assert (
        configuration["epochs"][ancilla.gid] == configuration["epochs"][merge.gid] - 1
    )
    assert max(configuration["epochs"].values()) == 5


def test_invalid_parameters_fail_explicitly():
    with pytest.raises(ValueError, match="distance"):
        surgery_library(4, 0.001)
    with pytest.raises(ValueError, match="probability"):
        surgery_library(5, -0.001)
    with pytest.raises(ValueError, match="width"):
        brickwall_program(0, 50, "Z", 42)
    with pytest.raises(ValueError, match="distance"):
        brickwall_program(1, 1, "Z", 42, distance=0)


def test_cadence_retains_exact_nanosecond_boundaries(tmp_path: Path):
    configuration = compile_case(
        tmp_path, distance=7, probability=0.001, width=2, layers=1,
        basis="Z", seed=42, block_seconds=0.070000001, export_physical=False,
    )
    assert configuration["block_ns"] == 70_000_001
    library = JitLibrary.FromString((tmp_path / "circuit.deq.jit").read_bytes())
    assert max(
        gadget_timing(instruction.gadget).end
        for instruction in library.program
    ) == 5 * 70_000_001
    for instruction in library.program:
        gadget = instruction.gadget
        if gadget.modifier.remote_conditional_correction.remote_readouts:
            assert gadget_timing(gadget) is not None
            assert gadget_timing(gadget).duration == 0


@pytest.mark.parametrize("seconds", [float("nan"), float("inf"), -0.1, 1e-10, 0])
def test_invalid_cadence_is_rejected(seconds: float):
    with pytest.raises(ValueError):
        brickwall_program(1, 1, "Z", 42, block_seconds=seconds)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_width_one_is_a_noiseless_memory_control(tmp_path: Path, basis: str):
    from readout_latency import readout_samples

    configuration = compile_case(
        tmp_path, distance=3, probability=0.001, width=1, layers=2, basis=basis,
        seed=42, block_seconds=0.001,
    )
    assert configuration["cnot_count"] == 0
    assert configuration["readouts"] == 1
    assert configuration["gadgets"] == 8
    assert max(configuration["epochs"].values()) == 8
    result = run_case(
        tmp_path,
        shots=2,
        threads=2,
        noiseless=True,
        registration_lead_seconds=0.002,
    )
    assert result["logical_errors"] == 0
    samples, coverage = readout_samples(result)
    assert len(samples) == 2 and all(sample.region == "final" for sample in samples)
    assert all(row["expected"] == row["observed"] for row in coverage)


def test_monotonic_window_offsets_preserve_missing_event_boundaries():
    names = (
        "outcomes_ready",
        "inputs_ready",
        "graph_slot_acquired",
        "decoding",
        "decoded",
        "pauli_frame_updated",
    )
    window = SpanRecord(
        begin_time_unix_ns=1000,
        properties=[
            Property(key=f"{name}_offset_ns", value=str(10 * index))
            for index, name in enumerate(names, start=1)
        ],
    )
    assert window_event_times(window) == {
        name: 1000 + 10 * index for index, name in enumerate(names, start=1)
    }
    window.properties[-1].value = "1"
    with pytest.raises(ValueError, match="invalid monotonic window"):
        window_event_times(window)
    del window.properties[-1]
    with pytest.raises(ValueError, match="incomplete monotonic window"):
        window_event_times(window)
