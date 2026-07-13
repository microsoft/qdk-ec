"""Integration tests for `deq.runtime.Sampler` — the deq-native sampler.

Exercises the standalone Sampler: parses a .deq file (or in-memory source),
compiles its named PROGRAM into JIT instructions, transpiles to Stim, and
returns per-gadget-partitioned shots ready for the coordinator's `decode`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deq.proto import simulator_pb2 as sim_pb
from deq.runtime import RawSampler, Sampler


# ── fixtures ────────────────────────────────────────────────────────────────


# A self-contained 3-qubit repetition-code memory experiment we can inline
# so the tests don't depend on a specific file on disk.
_DEQ_SOURCE = """
CODE RepetitionCode [[3,1,1]] {
    LOGICAL X0*X1*X2 Z0*Z1*Z2
    STABILIZER Z0*Z1 Z1*Z2
}

GADGET PrepareZ {
    R 0 1 2
    X_ERROR(0.01) 0 1 2
    OUTPUT RepetitionCode 0 1 2
}

GADGET MeasureZ {
    INPUT RepetitionCode 0 1 2
    M(0.01) 0 1 2
    READOUT rec[-3] rec[-2] rec[-1]
}

PROGRAM Simulation {
    PrepareZ 0
    MeasureZ 0
    ASSERT_EQ rec[-1] 0
}
"""


@pytest.fixture
def sampler() -> Sampler:
    return Sampler.from_source(_DEQ_SOURCE, program="Simulation", seed=42)


# ── tests ───────────────────────────────────────────────────────────────────


def test_re_exported_raw_class_is_rust_pyclass():
    import deq_runtime

    assert RawSampler is deq_runtime.Sampler


def test_from_source_compiles_and_samples(sampler: Sampler):
    shots = sampler.sample(4)

    assert len(shots) == 4
    for shot in shots:
        assert isinstance(shot, sim_pb.ShotSample)
        # PrepareZ has 0 measurements, MeasureZ has 3 → flat size = 3.
        assert shot.outcomes.size == 3


def test_partition_and_split_outcomes(sampler: Sampler):
    """``Sampler.partition`` exposes per-gadget measurement counts; pair
    it with :meth:`split_outcomes` (or slice manually) to recover the
    per-gadget chunks the coordinator's ``decode`` expects."""
    assert sampler.partition == [0, 3]

    shot = sampler.sample(1)[0]
    chunks = sampler.split_outcomes(shot.outcomes)
    assert len(chunks) == 2
    assert chunks[0].size == 0  # PrepareZ
    assert chunks[1].size == 3  # MeasureZ
    # Sum of chunk sizes equals the flat record.
    assert sum(c.size for c in chunks) == shot.outcomes.size


def test_from_file_reads_a_real_deq(tmp_path: Path):
    deq_path = tmp_path / "experiment.deq"
    deq_path.write_text(_DEQ_SOURCE)

    sampler = Sampler(deq_path, program="Simulation", seed=42)
    shots = sampler.sample(2)
    assert len(shots) == 2


def test_unknown_program_raises_key_error_with_available_list():
    with pytest.raises(KeyError, match="available programs: \\['Simulation'\\]"):
        Sampler.from_source(_DEQ_SOURCE, program="Bogus")


def test_program_and_instructions_accessors(sampler: Sampler):
    assert sampler.program == "Simulation"

    assert len(sampler.instructions) == 2
    prep, meas = sampler.instructions
    # Gids are auto-assigned 1, 2 in execution order.
    assert prep.gadget.gid == 1
    assert meas.gadget.gid == 2
    # Distinct gtype ids for the two distinct GADGET definitions.
    assert prep.gadget.gtype != meas.gadget.gtype


def test_library_accessor_includes_program(sampler: Sampler):
    lib = sampler.library
    assert len(lib.program) == 2
    assert {gt.base.name for gt in lib.gadget_types} == {"PrepareZ", "MeasureZ"}


def test_library_accessor_returns_full_decoder_capable_library(sampler: Sampler):
    """``Sampler.library`` is lazy but must produce the full library —
    decoder-side fields (checks, errors, propagation matrices) populated —
    so it can be fed to the coordinator / static_jit_compiler path."""
    lib = sampler.library
    for gt in lib.gadget_types:
        assert len(gt.errors) > 0, (
            f"{gt.base.name} has no errors; lazy library must build the full "
            "decoder-capable library, not the program-only shim"
        )
    measure_z = next(gt for gt in lib.gadget_types if gt.base.name == "MeasureZ")
    assert len(measure_z.finished_checks) > 0


def test_library_accessor_is_cached(sampler: Sampler):
    """Repeated ``.library`` access returns the same cached object."""
    assert sampler.library is sampler.library


def test_sampler_supports_program_with_virtual_pauli_corrections():
    """VIRTUAL is decoder-only annotation — it doesn't change the Stim
    circuit — so the program-only fast path must still accept it.
    The lite library populates the ``correction_propagation`` matrix
    shape that ``compile_program_for_jit`` reads to record VIRTUAL
    toggles, even though the matrix entries stay empty."""
    src = """
    CODE Rep [[3,1,1]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }
    GADGET PrepareZ { R 0 1 2 OUTPUT Rep 0 1 2 }
    GADGET IdleZ { INPUT Rep 0 1 2 OUTPUT Rep 0 1 2 }
    GADGET MeasureZ {
        INPUT Rep 0 1 2
        M 0 1 2
        READOUT rec[-3] rec[-2] rec[-1]
    }
    PROGRAM WithVirtual {
        PrepareZ 0
        IdleZ 0
        VIRTUAL X0 0
        MeasureZ 0
    }
    """
    sampler = Sampler.from_source(src, program="WithVirtual", seed=0)
    shots = sampler.sample(2)
    assert len(shots) == 2
    # IdleZ should carry a correction_propagation modifier reflecting the
    # VIRTUAL X toggle.
    idle_instr = sampler.instructions[1]
    assert idle_instr.gadget.HasField("modifier")


def test_sampler_supports_program_with_compose():
    """COMPOSE definitions are inlined into synthetic gadgets, so the
    Sampler accepts ``.deq`` files that build up programs with COMPOSE
    blocks — the same way the CLI's transpile pipeline does."""
    src = """
    CODE Rep [[3,1,1]] {
        LOGICAL X0*X1*X2 Z0*Z1*Z2
        STABILIZER Z0*Z1 Z1*Z2
    }
    GADGET PrepareZ {
        R 0 1 2
        OUTPUT Rep 0 1 2
    }
    GADGET MeasureZ {
        INPUT Rep 0 1 2
        M 0 1 2
        READOUT rec[-3] rec[-2] rec[-1]
    }
    COMPOSE PrepThenMeas {
        PrepareZ 0
        MeasureZ 0
    }
    PROGRAM Simulation {
        PrepThenMeas
    }
    """
    sampler = Sampler.from_source(src, program="Simulation", seed=0)
    shots = sampler.sample(2)
    assert len(shots) == 2
    # PrepThenMeas appears as a single (synthetic) gadget application.
    assert len(sampler.instructions) == 1
    assert "R 0 1 2" in sampler.circuit
    assert "M 0 1 2" in sampler.circuit


def test_circuit_property_returns_compiled_stim(sampler: Sampler):
    src = sampler.circuit
    # The PrepareZ body contains an R + X_ERROR for qubits 0/1/2.
    assert "R 0 1 2" in src
    assert "X_ERROR(0.01)" in src
    # And the MeasureZ body contains the measurement.
    assert "M(0.01)" in src


def test_raw_opt_out_returns_bytes(sampler: Sampler):
    shots = sampler.sample(2, raw=True)
    assert len(shots) == 2
    for blob in shots:
        assert isinstance(blob, bytes)
        # Round-trip parse must succeed.
        sim_pb.ShotSample.FromString(blob)


def test_fixed_seed_is_deterministic_across_samplers():
    a = Sampler.from_source(_DEQ_SOURCE, program="Simulation", seed=12345)
    b = Sampler.from_source(_DEQ_SOURCE, program="Simulation", seed=12345)
    assert a.sample(10, raw=True) == b.sample(10, raw=True)


def test_skip_shots_advances_the_stream():
    full = Sampler.from_source(_DEQ_SOURCE, program="Simulation", seed=7).sample(20, raw=True)
    skipped = Sampler.from_source(
        _DEQ_SOURCE, program="Simulation", seed=7, skip_shots=5
    ).sample(15, raw=True)
    assert skipped == full[5:]


def test_repr_surfaces_program_and_gadget_count(sampler: Sampler):
    assert repr(sampler) == "Sampler(program='Simulation', gadgets=2)"


def test_raw_accessor_returns_underlying_pyclass(sampler: Sampler):
    raw = sampler.raw
    blobs = raw.sample(2)
    assert len(blobs) == 2
    assert all(isinstance(b, bytes) for b in blobs)


# ── backend selection ──────────────────────────────────────────────────────


def test_preselect_backend_produces_shots():
    """The ``"preselect"`` backend (tableau-based) accepts any Stim circuit
    and produces shots with the same shape as the default ``"stim"`` one."""
    sampler = Sampler.from_source(
        _DEQ_SOURCE, program="Simulation", simulator="preselect", seed=42
    )
    shots = sampler.sample(4)
    assert len(shots) == 4
    for shot in shots:
        assert shot.outcomes.size == 3


def test_simulator_config_accepts_dict_and_json_string():
    """``simulator_config`` accepts a Python mapping or a JSON string —
    both must be threaded through to the Rust backend without error."""
    for cfg in ({"preselect_max_attempts": 100}, '{"preselect_max_attempts": 100}'):
        sampler = Sampler.from_source(
            _DEQ_SOURCE,
            program="Simulation",
            simulator="preselect",
            simulator_config=cfg,
            seed=0,
        )
        assert len(sampler.sample(2)) == 2


def test_unknown_simulator_name_raises_value_error():
    with pytest.raises(ValueError, match="unknown simulator"):
        Sampler.from_source(_DEQ_SOURCE, program="Simulation", simulator="bogus")


def test_invalid_simulator_config_raises_value_error():
    with pytest.raises(ValueError, match="invalid simulator_config"):
        Sampler.from_source(
            _DEQ_SOURCE,
            program="Simulation",
            simulator_config="{not valid json",
        )


def test_simulator_config_rejects_unknown_field():
    """Each backend has its own config struct with ``deny_unknown_fields``,
    so a typo'd key under one backend surfaces a clear, backend-tagged
    error pointing at the keys that backend actually accepts."""
    with pytest.raises(ValueError, match="invalid stim simulator_config"):
        Sampler.from_source(
            _DEQ_SOURCE,
            program="Simulation",
            simulator_config={"max_attempts_typo": 100},
        )
    with pytest.raises(ValueError, match="invalid preselect simulator_config"):
        Sampler.from_source(
            _DEQ_SOURCE,
            program="Simulation",
            simulator="preselect",
            simulator_config={"max_attempts_typo": 100},
        )

