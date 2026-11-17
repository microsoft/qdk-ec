"""Timed programs preserve source references while serializing by completion time."""

import pytest
import stim

from deq.circuit.parser import parse
from deq.cli.jit import compile_program_for_jit, export_program_stim
from deq.transpiler.jit_library_builder import build_jit_program
from deq.transpiler.jit_transpiler import flatten_body
from deq.transpiler.program_timing import TIMING_METADATA_KEY, Timing, gadget_timing, nanoseconds, set_gadget_timing
from deqagram import GadgetDefinition, ProgramDefinition

LIBRARY = """
CODE C [[1,1]] { LOGICAL X0 Z0 }
GADGET Prepare { R 0 OUTPUT C 0 }
GADGET One { R 0 X 0 OUTPUT C 0 VIRTUAL LX0 }
GADGET Idle { INPUT C 0 I 0 OUTPUT C 0 }
GADGET Read { INPUT C 0 M 0 READOUT M0 }
"""


def compile_timed(source):
    parsed = parse(LIBRARY + source)
    programs = {
        item.name: item
        for item in parsed.definitions
        if isinstance(item, ProgramDefinition)
    }
    library = build_jit_program(parsed)
    compiled, assertions = compile_program_for_jit(library, programs["Main"], programs)
    library.program.extend(item for item, _ in compiled)
    text = export_program_stim(
        library,
        {
            item.name: item
            for item in parsed.definitions
            if isinstance(item, GadgetDefinition)
        },
        {item.base.gtype: item.base.name for item in library.gadget_types},
        flatten_body,
        programs["Main"],
        [app for _, app in compiled],
        assertions,
    )
    return library, assertions, text


def times(library):
    return [
        (gadget_timing(item.gadget).start, gadget_timing(item.gadget).duration)
        for item in library.program
    ]


def test_timing_metadata_is_exact_and_preserves_custom_annotations():
    from deq.proto.deq_bin_pb2 import Gadget

    gadget = Gadget(metadata={"custom.tool": {"values": [True, None, "tag"]}})
    set_gadget_timing(gadget, Timing(9_007_199_254_740_993, 17))
    decoded = Gadget.FromString(gadget.SerializeToString())
    assert gadget_timing(decoded) == Timing(9_007_199_254_740_993, 17)
    assert decoded.metadata["custom.tool"] == gadget.metadata["custom.tool"]
    assert decoded.metadata[TIMING_METADATA_KEY]["start_ns"] == "9007199254740993"
    del decoded.metadata[TIMING_METADATA_KEY]["duration_ns"]
    assert gadget_timing(decoded) == Timing(9_007_199_254_740_993)
    del decoded.metadata[TIMING_METADATA_KEY]
    assert gadget_timing(decoded) is None


@pytest.mark.parametrize("value", [
    None, "invalid", {}, {"start_ns": 10}, {"start_ns": True},
    {"start_ns": "-1"}, {"start_ns": "1.5"}, {"start_ns": "١"},
    {"start_ns": str(1 << 63)}, {"start_ns": "0", "duration_ns": 0},
    {"start_ns": str((1 << 63) - 1), "duration_ns": "1"},
    {"start_ns": "0", "unknown": "1"},
])
def test_invalid_simulator_metadata_is_rejected(value):
    from deq.proto.deq_bin_pb2 import Gadget

    with pytest.raises(ValueError):
        gadget_timing(Gadget(metadata={TIMING_METADATA_KEY: value}))


def test_completion_order_remaps_assertions_and_measurements():
    library, assertions, text = compile_timed("""
PROGRAM Main {
    Prepare 0
    One 1
    @START(at=0.003,duration=0.001)
    Read 0
    ASSERT_EQ rec[-1] 0
    @START(at=0.001,duration=0.001)
    Read 1
    ASSERT_EQ rec[-1] 1
}
""")
    assert times(library) == [
        (0, 0),
        (0, 0),
        (1_000_000, 1_000_000),
        (3_000_000, 1_000_000),
    ]
    assert [indices for indices, _, _ in assertions] == [[1], [0]]
    assert [i.gadget.connectors[0].gid for i in library.program[2:]] == [2, 1]
    assert [line for line in text.splitlines() if line.startswith("#!delay")] == [
        "#!delay 0.002000000",
        "#!delay 0.002000000",
    ]
    assert (
        stim.Circuit(text).compile_sampler(seed=42).sample(5).tolist()
        == [[True, False]] * 5
    )


def test_nested_origins_and_delay_do_not_leak_to_parent():
    library, _, _ = compile_timed("""
PROGRAM Sub {
    @START(at=0.001,duration=0.001)
    Prepare 0
    @DELAY(0.002)
    Read 0
}
PROGRAM Main {
    @START(at=0.010)
    Sub()
    @START(at=0.020,duration=0.010)
    Sub()
}
""")
    assert times(library) == [
        (11_000_000, 1_000_000),
        (13_000_000, 0),
        (21_000_000, 1_000_000),
        (23_000_000, 0),
    ]


def test_delay_accumulates_in_repeat_and_start_resets_cursor():
    library, _, _ = compile_timed("""
PROGRAM Main {
    REPEAT 2 {
        @DELAY(0.001)
        Prepare 0
        Read 0
    }
    @START(at=0.003)
    Prepare 1
    Read 1
}
""")
    assert (
        times(library)
        == [(1_000_000, 0)] * 2 + [(2_000_000, 0)] * 2 + [(3_000_000, 0)] * 2
    )


def test_repeat_start_is_program_relative_and_applied_once():
    library, _, _ = compile_timed("""
PROGRAM Hold {
    INPUT C 0
    @START(at=0.1)
    Idle 0
    @START(at=0.28)
    REPEAT 3 {
        @DELAY(0.07)
        Idle 0
    }
    @DELAY(0.01)
    Idle 0
    OUTPUT C 0
}
PROGRAM Main {
    Prepare 0
    @START(at=1,duration=0.5)
    Hold 0
    @START(at=1.5)
    Read 0
}
""")
    assert times(library) == [
        (0, 0),
        (1_100_000_000, 0),
        (1_350_000_000, 0),
        (1_420_000_000, 0),
        (1_490_000_000, 0),
        (1_500_000_000, 0),
        (1_500_000_000, 0),
    ]


def test_unannotated_repeat_inherits_and_advances_program_cursor():
    library, _, _ = compile_timed("""
PROGRAM Main {
    @START(at=0.28)
    Prepare 0
    REPEAT 3 {
        @DELAY(0.07)
        Idle 0
    }
    @DELAY(0.01)
    Read 0
}
""")
    assert times(library) == [
        (280_000_000, 0),
        (350_000_000, 0),
        (420_000_000, 0),
        (490_000_000, 0),
        (500_000_000, 0),
    ]


def test_delayed_repeat_preserves_explicit_duration_delivery():
    prefix = """
GADGET Syndrome { INPUT C 0 M 0 OUTPUT C 0 }
PROGRAM Main { Prepare 0
"""
    suffix = "@START(at=0.49,duration=0.07) Read 0 }"
    explicit, _, explicit_stim = compile_timed(prefix + """
    @START(at=0.28,duration=0.07)
    Syndrome 0
    @START(at=0.35,duration=0.07)
    Syndrome 0
    @START(at=0.42,duration=0.07)
    Syndrome 0
""" + suffix)
    repeated, _, repeated_stim = compile_timed(prefix + """
    @START(at=0.28)
    REPEAT 3 {
        @DELAY(0.07)
        Syndrome 0
    }
""" + suffix)
    assert [
        gadget_timing(item.gadget).end for item in repeated.program
    ] == [
        gadget_timing(item.gadget).end for item in explicit.program
    ] == [0, 350_000_000, 420_000_000, 490_000_000, 560_000_000]
    assert [
        line for line in repeated_stim.splitlines() if line.startswith("#!delay")
    ] == [
        line for line in explicit_stim.splitlines() if line.startswith("#!delay")
    ]
    assert stim.Circuit(repeated_stim) == stim.Circuit(explicit_stim)


def test_repeat_annotation_without_program_registry_is_not_dropped():
    source = parse(
        LIBRARY + "PROGRAM Main { @START(at=0.003) REPEAT 2 { Prepare 0 Read 0 } }"
    )
    library = build_jit_program(source)
    main = next(
        item for item in source.definitions if isinstance(item, ProgramDefinition)
    )
    instructions, _ = compile_program_for_jit(library, main)
    assert all(
        gadget_timing(instruction.gadget).start == 3_000_000
        for instruction, _ in instructions
    )


def test_ties_keep_reading_order_even_when_start_times_differ():
    library, _, _ = compile_timed("""
PROGRAM Main {
    @START(at=0.003,duration=0.001)
    Prepare 0
    @START(at=0.002,duration=0.002)
    Prepare 1
    @START(at=0.004)
    Read 0
    Read 1
}
""")
    assert times(library) == [
        (3_000_000, 1_000_000),
        (2_000_000, 2_000_000),
        (4_000_000, 0),
        (4_000_000, 0),
    ]


@pytest.mark.parametrize(
    "order", [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
)
def test_all_three_readout_permutations_preserve_source_assertions(order):
    body = "Prepare 0\nOne 1\nPrepare 2\n"
    for wire in range(3):
        body += f"@START(at={0.001 * (order.index(wire) + 1):.3f})\nRead {wire}\nASSERT_EQ rec[-1] {wire % 2}\n"
    library, assertions, text = compile_timed("PROGRAM Main {\n" + body + "}")
    sample = stim.Circuit(text).compile_sampler(seed=42).sample(1)[0]
    assert all(
        bool(sample[indices[0]]) == expected for indices, expected, _ in assertions
    )
    assert [i.gadget.gid for i in library.program] == list(range(1, 7))


def test_conditionals_retain_readout_identity_after_reordering():
    library, _, text = compile_timed("""
PROGRAM Main {
    Prepare 0
    One 1
    Prepare 2
    @START(at=0.004)
    Read 0
    @START(at=0.002)
    Read 1
    CONDITIONAL rec[-2] X0 2
    @START(at=0.005)
    Read 2
}
""")
    conditional = next(
        i.gadget
        for i in library.program
        if i.gadget.modifier.HasField("remote_conditional_correction")
    )
    reference = conditional.modifier.remote_conditional_correction.remote_readouts[0]
    assert reference.gid == 5  # Source Read 0, not the newly preceding Read 1.
    assert gadget_timing(conditional).start == 4_000_000
    assert text.isascii()


@pytest.mark.parametrize(
    "annotation",
    [
        "@START(at=-1)",
        "@START(at=1e100)",
        '@START(at="1")',
        "@START(at=0.0000000001)",
        "@START(t=1)",
        "@START(at=1,at=2)",
        "@DELAY(-1)",
        "@DELAY(seconds=1)",
        "@START(at=1) @DELAY(1)",
    ],
)
def test_invalid_annotations_are_rejected(annotation):
    with pytest.raises((ValueError, SyntaxError)):
        compile_timed(f"PROGRAM Main {{ {annotation} Prepare 0 Read 0 }}")


def test_overlapping_wire_use_is_rejected():
    with pytest.raises(ValueError, match="before predecessor"):
        compile_timed("""
PROGRAM Main {
    @START(at=0,duration=0.002) Prepare 0
    @START(at=0.001,duration=0.001) Read 0
}
""")


def test_subprogram_envelope_is_checked_without_rescaling():
    with pytest.raises(ValueError, match="invocation duration"):
        compile_timed("""
PROGRAM Sub { @START(at=0.002) Prepare 0 Read 0 }
PROGRAM Main { @START(at=0,duration=0.001) Sub() }
""")


def test_untimed_programs_remain_untimed():
    library, _, text = compile_timed("PROGRAM Main { Prepare 0 Read 0 }")
    assert all(gadget_timing(item.gadget) is None for item in library.program)
    assert "#!delay" not in text


def test_zero_measurement_gadgets_keep_their_own_timing():
    library, _, text = compile_timed("""
PROGRAM Main {
    @START(at=0.001,duration=0.002) Prepare 0
    @START(at=0.003,duration=0.002) Idle 0
    @START(at=0.005) Read 0
}
""")
    assert times(library) == [
        (1_000_000, 2_000_000),
        (3_000_000, 2_000_000),
        (5_000_000, 0),
    ]
    assert [line for line in text.splitlines() if line.startswith("#!delay")] == [
        "#!delay 0.003000000",
        "#!delay 0.002000000",
    ]


def test_nanoseconds_are_exact_and_finite():
    assert nanoseconds(0.003) == 3_000_000
    for value in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            nanoseconds(value)


@pytest.mark.parametrize(
    "source",
    [
        "@START(at=0) GADGET Bad { R 0 }",
        "GADGET Bad { @DELAY(1) R 0 }",
        "PROGRAM Main { Prepare 0 @START(at=1) VIRTUAL X0 0 Read 0 }",
        "PROGRAM Main { Prepare 0 Read 0 @DELAY(1) ASSERT_EQ rec[-1] 0 }",
    ],
)
def test_timing_annotations_cannot_be_silently_ignored(source):
    with pytest.raises((SyntaxError, ValueError)):
        compile_timed(source)


def test_explicit_zero_subprogram_envelope_is_not_unspecified():
    with pytest.raises(ValueError, match="invocation duration"):
        compile_timed("""
PROGRAM Sub { @START(at=0.001) Prepare 0 Read 0 }
PROGRAM Main { @START(at=0,duration=0) Sub() }
""")


def test_dependency_adjusted_correction_cannot_escape_invocation():
    with pytest.raises(ValueError, match="invocation duration"):
        compile_timed("""
PROGRAM Correct {
    INPUT C 0
    CONDITIONAL rec[-1] X0 0
    OUTPUT C 0
}
PROGRAM Main {
    Prepare 0
    Prepare 1
    @START(at=0.010) Read 1
    @START(at=0,duration=0.001) Correct 0
    @START(at=0.011) Read 0
}
""")


@pytest.mark.parametrize("port", ["@DELAY(1) INPUT C 0", "@START(at=1) OUTPUT C 0"])
def test_timing_on_subprogram_ports_is_rejected(port):
    with pytest.raises(ValueError, match="port declarations"):
        compile_timed(f"""
PROGRAM Sub {{ {port} }}
PROGRAM Main {{ Sub() }}
""")


def test_exact_decimal_literals_survive_parsing_and_compilation():
    library, _, _ = compile_timed("""
PROGRAM Main {
    @START(at=16777216.000000001) Prepare 0
    Read 0
}
""")
    assert times(library) == [(16_777_216_000_000_001, 0)] * 2
    for literal in (
        "0.100000000000000001",
        "0.1000000000000000000000000000000001",
        "1e-400",
    ):
        with pytest.raises(ValueError, match="nanoseconds"):
            compile_timed(f"PROGRAM Main {{ @START(at={literal}) Prepare 0 Read 0 }}")


@pytest.mark.parametrize("simulator", ["static", "jit-static"])
def test_native_cli_obeys_timing_and_reordered_assertions(tmp_path, simulator):
    import json
    from deq.cli.simulate import simulate__ler
    from deq.proto.fastrace_pb2 import TraceFile
    from deq.proto.simulator_pb2 import SimulatorTrace

    source = tmp_path / "timed.deq"
    source.write_text(LIBRARY + """
PROGRAM Main {
    Prepare 0
    One 1
    @START(at=0.060,duration=0.020) Read 0
    ASSERT_EQ rec[-1] 0
    @START(at=0.010,duration=0.020) Read 1
    ASSERT_EQ rec[-1] 1
}
""")
    trace_path = tmp_path / "timing.pb"
    shots_path = tmp_path / "shots.pb"
    simulate__ler(
        str(source),
        program="Main",
        shots=2,
        errors=3,
        batch_size=2,
        jobs=1,
        simulator=simulator,
        strict_timing=True,
        trace=str(trace_path),
        simulator_trace_output=str(shots_path),
        loss_model="none",
        seed=42,
        coordinator="window",
        coordinator_config=json.dumps({"buffer_radius": 0, "lookahead_radius": 0}),
        save=str(tmp_path / "artifacts"),
    )
    shots = SimulatorTrace.FromString(shots_path.read_bytes())
    assert len(shots.shots) == 2
    assert all(
        shot.HasField("decode_result") and not shot.logical_error
        for shot in shots.shots
    )
    trace = TraceFile.FromString(trace_path.read_bytes())
    simulators = [
        span
        for span in trace.spans
        if span.name in ("static_simulator", "jit_static_simulator")
    ]
    assert len(simulators) == 2
    for span in simulators:
        events = {event.name: event.timestamp_unix_ns for event in span.events}
        assert events["process_result"] - events["start_decoding"] >= 80_000_000
