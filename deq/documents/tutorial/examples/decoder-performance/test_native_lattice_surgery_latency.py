"""Exercise the same source CLI contract for static and JIT execution."""

import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jit-basics"))

import pytest

from deq.proto.deq_bin_pb2 import Library
from deq.proto.deq_jit_pb2 import JitLibrary
from deq.proto.fastrace_pb2 import TraceFile
from deq.transpiler.program_timing import gadget_timing
from lattice_surgery import compile_case
from lattice_surgery_latency import analyze_completed_run, run_case
from readout_latency import readout_samples


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("decoder", ["mwpm", "mwpf"])
@pytest.mark.parametrize("simulator", ["static", "jit-static"])
def test_native_source_simulators_preserve_noiseless_corrections(
    tmp_path: Path, basis: str, decoder: str, simulator: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(Path(__file__).resolve().parents[5])
    configuration = compile_case(
        tmp_path, distance=3, probability=0, width=4, layers=2,
        basis=basis, seed=5, block_seconds=0.002,
    )
    result = run_case(
        tmp_path, shots=2, threads=2, noiseless=True,
        decoder=decoder, simulator=simulator, timeout_seconds=60,
        decoder_config={"cluster_node_limit": 10} if decoder == "mwpf" else {},
    )
    assert result["logical_errors"] == 0
    assert result["persistent_decoder"] == (simulator == "static")
    assert "includes_code_startup" not in result
    assert "static program setup is excluded" in result["latency_origin"]
    assert result["includes_jit_compilation"] == (simulator == "jit-static")
    samples, coverage = readout_samples(result)
    assert len(samples) == 2 * configuration["readouts"]
    assert all(row["expected"] == row["observed"] for row in coverage)
    run = Path(result["run_directory"])
    trace = TraceFile.FromString((run / "timing.pb").read_bytes())
    expected_span = "static_simulator" if simulator == "static" else "jit_static_simulator"
    assert sum(span.name == expected_span for span in trace.spans) == 2
    assert not any(span.name == "static_controller_driver" for span in trace.spans)
    native_jit = JitLibrary.FromString((run / "Brickwall.deq.jit").read_bytes())
    reference = JitLibrary.FromString((tmp_path / "circuit.deq.jit").read_bytes())
    assert native_jit == reference
    exported = re.sub(
        r"(?m)^(#.*)  \(circuit\.deq:\d+\)$",
        r"\1",
        (run / "Brickwall.stim").read_text(),
    )
    assert exported == (tmp_path / "circuit.stim").read_text()
    assert not (run / "timed.stim").exists()
    if simulator == "static":
        binary = Library.FromString((run / "Brickwall.deq.bin").read_bytes())
        gadgets = [
            instruction.gadget for instruction in binary.program
            if instruction.WhichOneof("create") == "gadget"
        ]
        assert len(gadgets) == configuration["gadgets"]
        assert all(gadget_timing(gadget) is not None for gadget in gadgets)
        assert [gadget_timing(gadget) for gadget in gadgets] == [
            gadget_timing(instruction.gadget) for instruction in native_jit.program
        ]
        assert len(binary.check_model_types) < configuration["gadgets"]
    command = json.loads((run / "command.json").read_text())
    assert sum(argument.endswith(".deq") for argument in command) == 1
    assert command[command.index("--simulator") + 1] == simulator
    assert analyze_completed_run(run) == result
    with pytest.raises(FileExistsError):
        run_case(tmp_path, shots=2, threads=2, noiseless=True,
                 decoder=decoder, simulator=simulator)


def test_static_rejects_jit_registration_lead_before_creating_outputs(tmp_path: Path):
    compile_case(
        tmp_path, distance=3, probability=0, width=1, layers=1, basis="Z", seed=42,
    )
    with pytest.raises(ValueError, match="registration lead"):
        run_case(
            tmp_path, shots=1, threads=1, noiseless=True,
            simulator="static", registration_lead_seconds=0.1,
        )
    assert not (tmp_path / "static-mwpm").exists()
