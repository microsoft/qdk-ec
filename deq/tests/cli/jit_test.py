# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint
"""Tests for the JIT CLI commands, specifically compile."""

from pathlib import Path

import pytest
from deq.circuit.parser import parse
from deq.transpiler.jit_library_builder import build_jit_library
from deq.compiler.jit_compiler import static_jit_compiler
from deq.cli.jit import parse_jit_program
from deq.spec.physical_validator import is_valid_and_physical
from deq.spec.program_equivalence import are_programs_equivalent
import deq.proto.deq_bin_pb2 as pb
import deq.proto.deq_jit_pb2 as jit_pb

# Minimal 3-logical-qubit trivial code to provide 6 observables for Pauli tests.
_TRIVIAL_CODE_K3_DEQ = """\
CODE ThreeQubitCode [[3,3]] {
    LOGICAL X0 Z0
    LOGICAL X1 Z1
    LOGICAL X2 Z2
}

GADGET PrepareZ {
    R 0 1 2
    X_ERROR(0.01) 0 1 2
    OUTPUT ThreeQubitCode 0 1 2
}

GADGET Idle {
    INPUT ThreeQubitCode 0 1 2
    X_ERROR(0.01) 0 1 2
    M 3
    OUTPUT ThreeQubitCode 4 5 6
}

GADGET MeasureZ {
    INPUT ThreeQubitCode 0 1 2
    X_ERROR(0.01) 0 1 2
    M 0 1 2
    READOUT rec[-1] rec[-2] rec[-3]
}
"""


@pytest.fixture
def trivial_code_k3_jit_library() -> jit_pb.JitLibrary:
    """JIT library with PrepareZ, Idle, MeasureZ gadgets for trivial [[3,3]] code."""
    return build_jit_library(parse(_TRIVIAL_CODE_K3_DEQ))


@pytest.fixture
def trivial_code_k3_codes() -> dict[str, object]:
    """Code definitions for the trivial [[3,3]] code."""
    from deq.circuit.model import CodeDefinition
    qfile = parse(_TRIVIAL_CODE_K3_DEQ)
    return {d.name: d for d in qfile.definitions if isinstance(d, CodeDefinition)}


@pytest.fixture
def named_jit_library() -> jit_pb.JitLibrary:
    """JIT library with named gadgets — same as trivial_code_k3_jit_library."""
    return build_jit_library(parse(_TRIVIAL_CODE_K3_DEQ))


class TestCompileProgram:
    """Test compiling full programs using .deq PROGRAM body syntax."""

    def test_simple_program(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Test a simple prepare-idle-measure program using shortcut notation."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ 0\nIdle 0\nMeasureZ 0",
        )

        assert len(instructions) == 3

        # Check PrepareZ (gtype=1)
        assert instructions[0].gadget.gtype == 1
        assert instructions[0].gadget.gid == 1
        assert len(instructions[0].gadget.connectors) == 0

        # Check Idle (gtype=2)
        assert instructions[1].gadget.gtype == 2
        assert instructions[1].gadget.gid == 2
        assert len(instructions[1].gadget.connectors) == 1
        assert instructions[1].gadget.connectors[0].gid == 1
        assert instructions[1].gadget.connectors[0].port == 0

        # Check MeasureZ (gtype=3)
        assert instructions[2].gadget.gtype == 3
        assert instructions[2].gadget.gid == 3
        assert len(instructions[2].gadget.connectors) == 1
        assert instructions[2].gadget.connectors[0].gid == 2
        assert instructions[2].gadget.connectors[0].port == 0

    def test_shortcut_form(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Test shortcut form: PrepareZ 0 (infers IN/OUT from gadget ports)."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ 0\nIdle 0\nMeasureZ 0",
        )

        assert len(instructions) == 3
        assert instructions[0].gadget.gtype == 1
        assert instructions[1].gadget.gtype == 2
        assert instructions[2].gadget.gtype == 3

    def test_chained_idles(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Test multiple chained idle gadgets."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ 0\nIdle 0\nIdle 0\nIdle 0\nMeasureZ 0",
        )

        assert len(instructions) == 5

        # Each instruction should connect to the previous one
        for i in range(1, 5):
            assert instructions[i].gadget.connectors[0].gid == i

    def test_undefined_wire_error(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Test error when using undefined wire."""
        with pytest.raises(ValueError, match="has no producer"):
            parse_jit_program(
                trivial_code_k3_jit_library,
                "MeasureZ IN(5)",
            )

    def test_wrong_input_count_error(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Test error when providing wrong number of inputs."""
        with pytest.raises(ValueError, match="expects 1 input ports"):
            parse_jit_program(
                trivial_code_k3_jit_library,
                "Idle OUT(0)",
            )

    def test_wrong_output_count_error(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Test error when providing wrong number of outputs."""
        with pytest.raises(ValueError, match="expects 1 output ports"):
            parse_jit_program(
                trivial_code_k3_jit_library,
                "PrepareZ OUT(0 1)",
            )

    def test_unknown_gadget_error(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Test error for unknown gadget type."""
        with pytest.raises(ValueError, match="unknown gadget"):
            parse_jit_program(
                trivial_code_k3_jit_library,
                "UnknownGadget OUT(0)",
            )

    def test_dangling_output_error(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Test error when program has unconnected output wires."""
        with pytest.raises(ValueError, match="dangling output wires"):
            parse_jit_program(
                trivial_code_k3_jit_library,
                "PrepareZ 0",
            )

    def test_dangling_output_lists_each_producer(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Dangling-wire error names every offending wire and its producer."""
        with pytest.raises(ValueError) as exc_info:
            parse_jit_program(
                trivial_code_k3_jit_library,
                "PrepareZ 0\nPrepareZ 1\nPrepareZ 2",
            )
        msg = str(exc_info.value)
        assert "dangling output wires" in msg
        for wire in (0, 1, 2):
            assert f"wire {wire}" in msg
            assert "PrepareZ" in msg

    def test_dangling_output_with_idle(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Test error when Idle leaves dangling output."""
        with pytest.raises(ValueError, match="dangling output wires"):
            parse_jit_program(
                trivial_code_k3_jit_library,
                "PrepareZ 0\nIdle 0",
            )

    def test_no_dangling_output_with_measure(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Test that MeasureZ properly consumes output (no dangling)."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ 0\nMeasureZ 0",
        )
        assert len(instructions) == 2


class TestEndToEndCompilation:
    """Test full end-to-end compilation with validation."""

    def test_compile_and_validate(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Test that compiled program passes validation."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ 0\nIdle 0\nMeasureZ 0",
        )

        # Add instructions to library
        jit_library = jit_pb.JitLibrary()
        jit_library.CopyFrom(trivial_code_k3_jit_library)
        jit_library.ClearField("program")
        for instr in instructions:
            jit_library.program.append(instr)

        # Compile to deq.bin
        deq_bin = static_jit_compiler(jit_library)

        # Validate
        assert is_valid_and_physical(deq_bin)

    def test_compile_and_validate_shortcut(
        self, named_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Test that compiled program with shortcut form passes validation."""
        instructions = parse_jit_program(
            named_jit_library,
            "PrepareZ 0\nIdle 0\nMeasureZ 0",
        )

        # Add instructions to library
        jit_library = jit_pb.JitLibrary()
        jit_library.CopyFrom(named_jit_library)
        jit_library.ClearField("program")
        for instr in instructions:
            jit_library.program.append(instr)

        # Compile to deq.bin
        deq_bin = static_jit_compiler(jit_library)

        # Validate
        assert is_valid_and_physical(deq_bin)

    def test_compile_matches_direct_program(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Test that parsed program produces same result as direct construction."""
        # Parse program
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ 0\nIdle 0\nMeasureZ 0",
        )

        # Create library with parsed instructions
        jit_library_parsed = jit_pb.JitLibrary()
        jit_library_parsed.CopyFrom(trivial_code_k3_jit_library)
        jit_library_parsed.ClearField("program")
        for instr in instructions:
            jit_library_parsed.program.append(instr)

        # Create library with direct instructions (like in jit_transpiler_test.py)
        jit_library_direct = jit_pb.JitLibrary()
        jit_library_direct.CopyFrom(trivial_code_k3_jit_library)
        jit_library_direct.ClearField("program")
        jit_library_direct.program.append(
            jit_pb.JitInstruction(gadget=pb.Gadget(gtype=1))
        )
        jit_library_direct.program.append(
            jit_pb.JitInstruction(
                gadget=pb.Gadget(
                    gtype=2, connectors=[pb.Gadget.Connector(gid=1, port=0)]
                )
            )
        )
        jit_library_direct.program.append(
            jit_pb.JitInstruction(
                gadget=pb.Gadget(
                    gtype=3, connectors=[pb.Gadget.Connector(gid=2, port=0)]
                )
            )
        )

        # Compile both
        deq_bin_parsed = static_jit_compiler(jit_library_parsed)
        deq_bin_direct = static_jit_compiler(jit_library_direct)

        # Should be equivalent
        assert are_programs_equivalent(deq_bin_parsed, deq_bin_direct)


class TestParseJitProgramAPI:
    """Test the public parse_jit_program API."""

    def test_parse_jit_program(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Test the public API function with explicit IN/OUT."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ 0\nIdle 0\nMeasureZ 0",
        )
        assert len(instructions) == 3
        assert instructions[0].gadget.gtype == 1
        assert instructions[1].gadget.gtype == 2
        assert instructions[2].gadget.gtype == 3

    def test_parse_jit_program_shortcut(
        self, named_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Test the public API function with shortcut notation."""
        instructions = parse_jit_program(
            named_jit_library,
            "PrepareZ 0\nIdle 0\nMeasureZ 0",
        )
        assert len(instructions) == 3
        assert instructions[0].gadget.gtype == 1
        assert instructions[1].gadget.gtype == 2
        assert instructions[2].gadget.gtype == 3


class TestPauliCorrections:
    """Test Pauli correction pseudo-instructions (VIRTUAL X0, Z1, Y2, etc.)."""

    def test_x0_toggles_z0(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """VIRTUAL X0 0 should toggle Z0 (row 1) in the constant column."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ 0\nVIRTUAL X0 0\nMeasureZ 0",
        )
        assert len(instructions) == 2
        toggle = instructions[0].gadget.modifier.correction_propagation_mod.toggle
        assert toggle.rows == 6
        assert toggle.cols == 1
        assert list(toggle.i) == [1]
        assert list(toggle.j) == [0]

    def test_z1_toggles_x1(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """VIRTUAL Z1 0 should toggle X1 (row 2) in the constant column."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ 0\nVIRTUAL Z1 0\nMeasureZ 0",
        )
        toggle = instructions[0].gadget.modifier.correction_propagation_mod.toggle
        assert list(toggle.i) == [2]
        assert list(toggle.j) == [0]

    def test_y0_toggles_both(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """VIRTUAL Y0 0 should toggle both X0 (row 0) and Z0 (row 1)."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ 0\nVIRTUAL Y0 0\nMeasureZ 0",
        )
        toggle = instructions[0].gadget.modifier.correction_propagation_mod.toggle
        assert list(toggle.i) == [0, 1]
        assert list(toggle.j) == [0, 0]

    def test_multiple_paulis_accumulate(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """VIRTUAL X0 and VIRTUAL Z1 on the same wire should toggle rows 1 and 2."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ 0\nVIRTUAL X0 0\nVIRTUAL Z1 0\nMeasureZ 0",
        )
        toggle = instructions[0].gadget.modifier.correction_propagation_mod.toggle
        assert list(toggle.i) == [1, 2]
        assert list(toggle.j) == [0, 0]

    def test_double_pauli_cancels(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Applying VIRTUAL X0 twice should cancel out (XOR semantics)."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ 0\nVIRTUAL X0 0\nVIRTUAL X0 0\nMeasureZ 0",
        )
        assert not instructions[0].gadget.HasField("modifier")

    def test_pauli_does_not_consume_wire(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """VIRTUAL correction should not consume the wire."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ 0\nVIRTUAL X0 0\nIdle 0\nMeasureZ 0",
        )
        assert len(instructions) == 3
        # X0 applied to PrepareZ (gid=1)
        assert instructions[0].gadget.HasField("modifier")
        # Idle and MeasureZ should have no modifier
        assert not instructions[1].gadget.HasField("modifier")

    def test_pauli_on_idle_output(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """VIRTUAL on the output of Idle (which has 1 input port)."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ 0\nIdle 0\nVIRTUAL X2 0\nMeasureZ 0",
        )
        # X2 should be on Idle (gid=2), which has 1 input and 1 output
        toggle = instructions[1].gadget.modifier.correction_propagation_mod.toggle
        # Idle: 6 output observables, 6 input observables -> cols = 6+1 = 7
        assert toggle.rows == 6
        assert toggle.cols == 7
        # X2 -> toggle Z2 (row 5) at constant column (col 6)
        assert list(toggle.i) == [5]
        assert list(toggle.j) == [6]

    def test_pauli_index_out_of_range(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Qubit index >= k should raise ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            parse_jit_program(
                trivial_code_k3_jit_library,
                "PrepareZ 0\nVIRTUAL X3 0\nMeasureZ 0",
            )

    def test_pauli_undefined_wire(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Undefined wire in VIRTUAL correction should raise ValueError."""
        with pytest.raises(ValueError, match="no producer"):
            parse_jit_program(
                trivial_code_k3_jit_library,
                "PrepareZ 0\nVIRTUAL X0 5\nMeasureZ 0",
            )

    def test_pauli_named_gadgets(self, named_jit_library: jit_pb.JitLibrary) -> None:
        """VIRTUAL corrections should work with named gadgets."""
        instructions = parse_jit_program(
            named_jit_library,
            "PrepareZ 0\nVIRTUAL X1 0\nMeasureZ 0",
        )
        toggle = instructions[0].gadget.modifier.correction_propagation_mod.toggle
        # X1 -> toggle Z1 (row 3) at constant column (col 0)
        assert list(toggle.i) == [3]
        assert list(toggle.j) == [0]

    def test_multi_pauli_product(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """VIRTUAL X0*Z1 0 should toggle both Z0 (row 1) and X1 (row 2)."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ 0\nVIRTUAL X0*Z1 0\nMeasureZ 0",
        )
        toggle = instructions[0].gadget.modifier.correction_propagation_mod.toggle
        assert list(toggle.i) == [1, 2]
        assert list(toggle.j) == [0, 0]

    def test_multi_pauli_product_three(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """VIRTUAL X0*Z1*Y2 0 should toggle rows 1, 2, 4, and 5."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ 0\nVIRTUAL X0*Z1*Y2 0\nMeasureZ 0",
        )
        toggle = instructions[0].gadget.modifier.correction_propagation_mod.toggle
        # X0 -> row 1; Z1 -> row 2; Y2 -> rows 4 and 5
        assert list(toggle.i) == [1, 2, 4, 5]
        assert list(toggle.j) == [0, 0, 0, 0]

    def test_multi_pauli_equivalent_to_separate(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """VIRTUAL X0*Z1 should produce same result as separate VIRTUAL X0 + VIRTUAL Z1."""
        instr_product = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ 0\nVIRTUAL X0*Z1 0\nMeasureZ 0",
        )
        instr_separate = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ 0\nVIRTUAL X0 0\nVIRTUAL Z1 0\nMeasureZ 0",
        )
        t1 = instr_product[0].gadget.modifier.correction_propagation_mod.toggle
        t2 = instr_separate[0].gadget.modifier.correction_propagation_mod.toggle
        assert list(t1.i) == list(t2.i)
        assert list(t1.j) == list(t2.j)


class TestConditionalCorrections:
    """Test CONDITIONAL rec[-k] pauli wire in PROGRAM bodies."""

    def test_emits_identity_gadget_with_modifier(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """CONDITIONAL inserts a synthesized identity gadget instance with
        a remote_conditional_correction modifier."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ OUT(0)\n"
            "PrepareZ OUT(1)\n"
            "MeasureZ IN(1)\n"
            "CONDITIONAL rec[-1] X0 0\n"
            "MeasureZ IN(0)",
        )

        # Expected: 5 instructions = 2 PrepareZ + 1 MeasureZ + identity + MeasureZ
        assert len(instructions) == 5
        cond_instr = instructions[3]
        assert cond_instr.gadget.HasField("modifier")
        modifier = cond_instr.gadget.modifier
        assert modifier.HasField("remote_conditional_correction")
        rcc = modifier.remote_conditional_correction
        # Reference is the most recent logical readout (MeasureZ on wire 1
        # emits 1 logical readout = XOR of the 3 physical measurements).
        assert len(rcc.remote_readouts) == 1
        assert rcc.remote_readouts[0].gid == 3
        assert rcc.remote_readouts[0].readout_index == 0
        # X on logical qubit 0 flips the LZ_0 column (= z_column(0) = 1).
        assert list(rcc.correction.i) == [1]
        assert list(rcc.correction.j) == [0]

    def test_identity_gadget_chains_correctly(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """The identity gadget consumes the wire from the previous producer
        and the next gadget consumes from the identity gadget."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ OUT(0)\n"
            "PrepareZ OUT(1)\n"
            "MeasureZ IN(1)\n"
            "CONDITIONAL rec[-1] X0 0\n"
            "MeasureZ IN(0)",
        )

        # gid 1 = PrepareZ (wire 0); gid 2 = PrepareZ (wire 1);
        # gid 3 = MeasureZ on wire 1; gid 4 = identity; gid 5 = MeasureZ on wire 0.
        # The identity gadget (gid 4) must connect to gid 1 (wire 0's producer).
        assert instructions[3].gadget.gid == 4
        assert len(instructions[3].gadget.connectors) == 1
        assert instructions[3].gadget.connectors[0].gid == 1
        assert instructions[3].gadget.connectors[0].port == 0
        # The final MeasureZ (gid 5) must connect to the identity gadget (gid 4).
        assert instructions[4].gadget.gid == 5
        assert len(instructions[4].gadget.connectors) == 1
        assert instructions[4].gadget.connectors[0].gid == 4
        assert instructions[4].gadget.connectors[0].port == 0

    def test_identity_gadget_type_added_to_library(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """The synthesized identity gadget type is appended to the library."""
        # Snapshot the gtypes before compile.
        before = {gt.base.gtype for gt in trivial_code_k3_jit_library.gadget_types}
        parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ OUT(0)\n"
            "MeasureZ IN(0)\n"  # produces readouts so rec[-1] resolves
            "PrepareZ OUT(0)\n"
            "CONDITIONAL rec[-1] X0 0\n"
            "MeasureZ IN(0)",
        )
        after = {gt.base.gtype for gt in trivial_code_k3_jit_library.gadget_types}
        new_gtypes = after - before
        assert len(new_gtypes) == 1
        new_gt = next(
            gt
            for gt in trivial_code_k3_jit_library.gadget_types
            if gt.base.gtype in new_gtypes
        )
        assert new_gt.base.name.startswith("__identity_")
        assert len(new_gt.base.measurements) == 0
        assert len(new_gt.base.inputs) == 1
        assert len(new_gt.base.outputs) == 1
        assert new_gt.base.inputs[0].ptype == new_gt.base.outputs[0].ptype

    def test_identity_gadget_reused_for_same_ptype(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Multiple CONDITIONALs on the same ptype reuse the same identity gtype."""
        before = len(trivial_code_k3_jit_library.gadget_types)
        parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ OUT(0)\n"
            "PrepareZ OUT(1)\n"
            "MeasureZ IN(1)\n"
            "CONDITIONAL rec[-1] X0 0\n"
            "CONDITIONAL rec[-1] Z0 0\n"
            "MeasureZ IN(0)",
        )
        after = len(trivial_code_k3_jit_library.gadget_types)
        # Only ONE new gtype should be added even for multiple CONDITIONALs on
        # the same port type.
        assert after - before == 1

    def test_multi_pauli_product(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """CONDITIONAL rec[-k] X0*Z1 wire flips both LZ_0 and LX_1."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ OUT(0)\n"
            "PrepareZ OUT(1)\n"
            "MeasureZ IN(1)\n"
            "CONDITIONAL rec[-1] X0*Z1 0\n"
            "MeasureZ IN(0)",
        )
        rcc = instructions[3].gadget.modifier.remote_conditional_correction
        # X0 flips LZ_0 (col 1), Z1 flips LX_1 (col 2). Sorted: [1, 2].
        assert list(rcc.correction.i) == [1, 2]
        assert list(rcc.correction.j) == [0, 0]

    def test_y_pauli_flips_both(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """CONDITIONAL rec[-k] Y<i> wire flips both LX_i and LZ_i columns."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ OUT(0)\n"
            "MeasureZ IN(0)\n"
            "PrepareZ OUT(0)\n"
            "CONDITIONAL rec[-1] Y2 0\n"
            "MeasureZ IN(0)",
        )
        rcc = instructions[3].gadget.modifier.remote_conditional_correction
        # Y2 = X2 * Z2; flips LZ_2 (col 5) and LX_2 (col 4). Sorted: [4, 5].
        assert list(rcc.correction.i) == [4, 5]

    def test_pauli_cancellation(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """CONDITIONAL rec[-k] X0*X0 wire has no effect (cancellation)."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ OUT(0)\n"
            "MeasureZ IN(0)\n"
            "PrepareZ OUT(0)\n"
            "CONDITIONAL rec[-1] X0*X0 0\n"
            "MeasureZ IN(0)",
        )
        rcc = instructions[3].gadget.modifier.remote_conditional_correction
        # X0 * X0 = identity; correction matrix is empty.
        assert list(rcc.correction.i) == []
        assert list(rcc.correction.j) == []

    def test_multiple_conditionals_on_same_wire_chain(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Multiple CONDITIONALs on the same wire chain through identity gadgets."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ OUT(0)\n"
            "PrepareZ OUT(1)\n"
            "PrepareZ OUT(2)\n"
            "MeasureZ IN(1)\n"  # readout #0 (gid=4)
            "MeasureZ IN(2)\n"  # readout #1 (gid=5)
            "CONDITIONAL rec[-1] X0 0\n"  # condition on readout #1 (gid=5)
            "CONDITIONAL rec[-2] Z0 0\n"  # condition on readout #0 (gid=4)
            "MeasureZ IN(0)",
        )
        # 8 instructions: 3 Prep + 2 MeasZ + 2 identity + 1 MeasZ.
        assert len(instructions) == 8
        # Identity #1 (gid 6) consumes from gid 1 (PrepareZ for wire 0).
        assert instructions[5].gadget.gid == 6
        assert instructions[5].gadget.connectors[0].gid == 1
        # Identity #2 (gid 7) consumes from gid 6 (Identity #1).
        assert instructions[6].gadget.gid == 7
        assert instructions[6].gadget.connectors[0].gid == 6
        # Final MeasZ (gid 8) consumes from gid 7 (Identity #2).
        assert instructions[7].gadget.gid == 8
        assert instructions[7].gadget.connectors[0].gid == 7
        # Identity #1's remote ref = gid 5's readout 0.
        rcc1 = instructions[5].gadget.modifier.remote_conditional_correction
        assert rcc1.remote_readouts[0].gid == 5
        assert rcc1.remote_readouts[0].readout_index == 0
        # Identity #2's remote ref = gid 4's readout 0.
        rcc2 = instructions[6].gadget.modifier.remote_conditional_correction
        assert rcc2.remote_readouts[0].gid == 4
        assert rcc2.remote_readouts[0].readout_index == 0

    def test_rec_offset_out_of_range_raises(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """CONDITIONAL rec[-k] with k > number of readouts so far raises."""
        with pytest.raises(ValueError, match="readout"):
            parse_jit_program(
                trivial_code_k3_jit_library,
                "PrepareZ OUT(0)\n"
                "CONDITIONAL rec[-1] X0 0\n"  # no readouts yet
                "MeasureZ IN(0)",
            )

    def test_unknown_wire_raises(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """CONDITIONAL on a wire that has no producer raises."""
        with pytest.raises(ValueError, match="wire"):
            parse_jit_program(
                trivial_code_k3_jit_library,
                "PrepareZ OUT(0)\n"
                "MeasureZ IN(0)\n"
                "CONDITIONAL rec[-1] X0 99\n",  # wire 99 has no producer
            )

    def test_logical_qubit_index_out_of_range_raises(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """CONDITIONAL with a logical qubit index >= code.k raises."""
        # ThreeQubitCode has k=3, so logical qubit 99 is out of range.
        with pytest.raises(ValueError, match="logical qubit"):
            parse_jit_program(
                trivial_code_k3_jit_library,
                "PrepareZ OUT(0)\n"
                "MeasureZ IN(0)\n"
                "PrepareZ OUT(0)\n"
                "CONDITIONAL rec[-1] X99 0\n"
                "MeasureZ IN(0)",
            )

    def test_end_to_end_static_jit_compile(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """A PROGRAM with CONDITIONAL compiles cleanly through static_jit_compile."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ OUT(0)\n"
            "PrepareZ OUT(1)\n"
            "MeasureZ IN(1)\n"
            "CONDITIONAL rec[-1] X0 0\n"
            "MeasureZ IN(0)",
        )
        # parse_jit_program already mutated trivial_code_k3_jit_library to add
        # the identity gadget type. Now we can run the full static compile.
        lib = jit_pb.JitLibrary()
        lib.CopyFrom(trivial_code_k3_jit_library)
        lib.ClearField("program")
        for instr in instructions:
            lib.program.append(instr)
        deq_bin = static_jit_compiler(lib)
        # Sanity check: the modifier is preserved in the compiled output.
        found_modifier = False
        for instr in deq_bin.program:
            if instr.HasField("gadget") and instr.gadget.HasField("modifier"):
                mod = instr.gadget.modifier
                if mod.HasField("remote_conditional_correction"):
                    found_modifier = True
                    break
        assert found_modifier, (
            "remote_conditional_correction modifier was lost in static_jit_compile"
        )


class TestRepeatInProgram:
    """Test REPEAT blocks inside PROGRAM bodies."""

    def test_repeat_unrolls_idles(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """REPEAT 3 { Idle } should produce the same as 3 consecutive Idles."""
        flat = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ 0\nIdle 0\nIdle 0\nIdle 0\nMeasureZ 0",
        )
        repeated = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ OUT(0)\nREPEAT 3 {\n    Idle IN(0) OUT(0)\n}\nMeasureZ IN(0)",
        )

        assert len(flat) == len(repeated)
        for a, b in zip(flat, repeated):
            assert a.gadget.gtype == b.gadget.gtype

    def test_repeat_compiles_and_validates(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """REPEAT inside PROGRAM produces a valid compilable program."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ OUT(0)\nREPEAT 2 {\n    Idle IN(0) OUT(0)\n}\nMeasureZ IN(0)",
        )

        jit_library = jit_pb.JitLibrary()
        jit_library.CopyFrom(trivial_code_k3_jit_library)
        jit_library.ClearField("program")
        for instr in instructions:
            jit_library.program.append(instr)

        deq_bin = static_jit_compiler(jit_library)
        assert is_valid_and_physical(deq_bin)


class TestNestedPrograms:
    """Test sub-program inlining (calling a PROGRAM from another PROGRAM)."""

    def test_sub_program_equivalent_to_inline(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Sub-program call should produce identical instructions to inlining."""
        from deq.circuit.model import ProgramDefinition
        from deq.circuit.parser import parse as parse_deq

        sub_deq = parse_deq(
            "PROGRAM PrepareAndIdle {\n"
            "    PrepareZ OUT(0)\n"
            "    Idle IN(0) OUT(0)\n"
            "}\n"
        )
        sub_def = [d for d in sub_deq.definitions if isinstance(d, ProgramDefinition)][
            0
        ]

        inlined = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ 0\nIdle 0\nMeasureZ 0",
        )
        nested = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareAndIdle OUT(0)\nMeasureZ IN(0)",
            program_defs={"PrepareAndIdle": sub_def},
        )

        assert len(inlined) == len(nested)
        for a, b in zip(inlined, nested):
            assert a.gadget.gtype == b.gadget.gtype

    def test_sub_program_shortcut_form(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Sub-program called with shortcut form: SubProgram wire."""
        from deq.circuit.model import ProgramDefinition
        from deq.circuit.parser import parse as parse_deq

        sub_deq = parse_deq(
            "PROGRAM PrepareAndIdle {\n"
            "    PrepareZ OUT(0)\n"
            "    Idle IN(0) OUT(0)\n"
            "}\n"
        )
        sub_def = [d for d in sub_deq.definitions if isinstance(d, ProgramDefinition)][
            0
        ]

        inlined = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ 0\nIdle 0\nMeasureZ 0",
        )
        nested = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareAndIdle 0\nMeasureZ 0",
            program_defs={"PrepareAndIdle": sub_def},
        )

        assert len(inlined) == len(nested)
        for a, b in zip(inlined, nested):
            assert a.gadget.gtype == b.gadget.gtype

    def test_nested_sub_programs(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """A calls B, B calls gadgets — two levels of nesting."""
        from deq.circuit.model import ProgramDefinition
        from deq.circuit.parser import parse as parse_deq

        parsed = parse_deq(
            "PROGRAM IdleAndMeasure {\n"
            "    Idle IN(0) OUT(0)\n"
            "    MeasureZ IN(0)\n"
            "}\n"
            "PROGRAM Full {\n"
            "    PrepareZ OUT(0)\n"
            "    IdleAndMeasure IN(0)\n"
            "}\n"
        )
        defs = {
            d.name: d for d in parsed.definitions if isinstance(d, ProgramDefinition)
        }
        full_def = defs.pop("Full")

        inlined = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ 0\nIdle 0\nMeasureZ 0",
        )
        nested = parse_jit_program(
            trivial_code_k3_jit_library,
            "Full 0",
            program_defs={**defs, "Full": full_def},
        )

        assert len(inlined) == len(nested)
        for a, b in zip(inlined, nested):
            assert a.gadget.gtype == b.gadget.gtype

    def test_cycle_detection(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Mutually recursive programs should raise an error."""
        from deq.circuit.model import ProgramDefinition
        from deq.circuit.parser import parse as parse_deq

        parsed = parse_deq(
            "PROGRAM A {\n"
            "    B IN(0) OUT(0)\n"
            "}\n"
            "PROGRAM B {\n"
            "    A IN(0) OUT(0)\n"
            "}\n"
        )
        defs = {
            d.name: d for d in parsed.definitions if isinstance(d, ProgramDefinition)
        }

        with pytest.raises(ValueError, match="cycle"):
            parse_jit_program(
                trivial_code_k3_jit_library,
                "PrepareZ OUT(0)\nA IN(0) OUT(0)\nMeasureZ IN(0)",
                program_defs=defs,
            )

    def test_wrong_input_count_for_sub_program(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Wrong number of input wires for sub-program should error."""
        from deq.circuit.model import ProgramDefinition
        from deq.circuit.parser import parse as parse_deq

        sub_deq = parse_deq(
            "PROGRAM NeedsInput {\n"
            "    Idle IN(0) OUT(0)\n"
            "    MeasureZ IN(0)\n"
            "}\n"
        )
        sub_def = [d for d in sub_deq.definitions if isinstance(d, ProgramDefinition)][
            0
        ]

        with pytest.raises(ValueError, match="input wires"):
            parse_jit_program(
                trivial_code_k3_jit_library,
                "PrepareZ OUT(0)\nNeedsInput OUT(0)",
                program_defs={"NeedsInput": sub_def},
            )

    def test_wrong_output_count_for_sub_program(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Wrong number of output wires for sub-program should error."""
        from deq.circuit.model import ProgramDefinition
        from deq.circuit.parser import parse as parse_deq

        sub_deq = parse_deq("PROGRAM HasOutput {\n" "    PrepareZ OUT(0)\n" "}\n")
        sub_def = [d for d in sub_deq.definitions if isinstance(d, ProgramDefinition)][
            0
        ]

        with pytest.raises(ValueError, match="output wires"):
            parse_jit_program(
                trivial_code_k3_jit_library,
                "HasOutput OUT(0 1)",
                program_defs={"HasOutput": sub_def},
            )

    def test_sub_program_with_virtual_corrections(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """VIRTUAL corrections inside a sub-program are remapped correctly."""
        from deq.circuit.model import ProgramDefinition
        from deq.circuit.parser import parse as parse_deq

        sub_deq = parse_deq(
            "PROGRAM PrepareWithX0 {\n"
            "    PrepareZ OUT(0)\n"
            "    VIRTUAL X0 0\n"
            "}\n"
        )
        sub_def = [d for d in sub_deq.definitions if isinstance(d, ProgramDefinition)][
            0
        ]

        inlined = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ 0\nVIRTUAL X0 0\nMeasureZ 0",
        )
        nested = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareWithX0 OUT(0)\nMeasureZ IN(0)",
            program_defs={"PrepareWithX0": sub_def},
        )

        assert len(inlined) == len(nested)
        t1 = inlined[0].gadget.modifier.correction_propagation_mod.toggle
        t2 = nested[0].gadget.modifier.correction_propagation_mod.toggle
        assert list(t1.i) == list(t2.i)
        assert list(t1.j) == list(t2.j)

    def test_end_to_end_compile_with_sub_program(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """Sub-program expansion produces a valid compilable program."""
        from deq.circuit.model import ProgramDefinition
        from deq.circuit.parser import parse as parse_deq

        sub_deq = parse_deq(
            "PROGRAM PrepareAndIdle {\n"
            "    PrepareZ OUT(0)\n"
            "    Idle IN(0) OUT(0)\n"
            "}\n"
        )
        sub_def = [d for d in sub_deq.definitions if isinstance(d, ProgramDefinition)][
            0
        ]

        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareAndIdle OUT(0)\nMeasureZ IN(0)",
            program_defs={"PrepareAndIdle": sub_def},
        )

        jit_library = jit_pb.JitLibrary()
        jit_library.CopyFrom(trivial_code_k3_jit_library)
        jit_library.ClearField("program")
        for instr in instructions:
            jit_library.program.append(instr)

        deq_bin = static_jit_compiler(jit_library)
        assert is_valid_and_physical(deq_bin)


# ---------------------------------------------------------------------------
# Stim export — PauliTarget remapping (regression)
# ---------------------------------------------------------------------------

CODE422_DEQ = Path(__file__).resolve().parents[1] / "circuit" / "fixtures" / "code422.deq"


def test_stim_export_remaps_mpp_pauli_targets() -> None:
    """MPP PauliTarget indices must be remapped to physical qubits.

    Previously, PauliTarget was passed through without remapping,
    causing MPP instructions in non-identity qubit maps to reference
    wrong qubits (e.g. ``MPP X0*X1*X2*X3`` instead of ``X4*X5*X6*X7``).
    """
    from deq.circuit.model import (
        GadgetDefinition,
        ProgramDefinition,
    )
    from deq.circuit.parser import parse_file
    from deq.transpiler.jit_library_builder import build_jit_library
    from deq.transpiler.jit_transpiler import flatten_body
    from deq.cli.jit import compile_program_for_jit, export_program_stim

    merged = parse_file(str(CODE422_DEQ))
    jit_library = build_jit_library(merged)

    program_def = next(
        d for d in merged.definitions if isinstance(d, ProgramDefinition)
    )
    gadgets_by_name = {
        d.name: d for d in merged.definitions if isinstance(d, GadgetDefinition)
    }
    gtype_to_name = {
        gt.base.gtype: gt.base.name for gt in jit_library.gadget_types
    }

    compiled, assertions = compile_program_for_jit(jit_library, program_def)
    for instr, _src in compiled:
        jit_library.program.append(instr)

    stim_text = export_program_stim(
        jit_library,
        gadgets_by_name,
        gtype_to_name,
        flatten_body,
        program_def,
        [src for _instr, src in compiled],
        assertions,
    )

    # PrepareZZ is the second gadget -> its qubits are 4,5,6,7.
    # The MPP instruction must use the remapped indices.
    lines = stim_text.splitlines()
    prepare_zz_header = next(
        i for i, l in enumerate(lines) if "PrepareZZ" in l
    )
    mpp_line = next(
        l for l in lines[prepare_zz_header:] if l.strip().startswith("MPP")
    )
    # Must contain physical indices 4,5,6,7 not local indices 0,1,2,3.
    assert "X4" in mpp_line, f"expected remapped indices in: {mpp_line}"
    assert "X0" not in mpp_line, f"local index leaked through in: {mpp_line}"


# ---------------------------------------------------------------------------
# Surface-code logical teleportation (d=3) — end-to-end PROGRAM
# compilation for both @REPROPAGATE and explicit-CONDITIONAL variants.
# ---------------------------------------------------------------------------

TELEPORTATION_D3_DEQ = (
    Path(__file__).resolve().parents[1]
    / "circuit"
    / "surface_code"
    / "teleportation_d3.deq"
)


@pytest.fixture(scope="module")
def teleportation_d3_setup() -> tuple[jit_pb.JitLibrary, dict[str, object]]:
    """Parse ``teleportation_d3.deq`` and build its JIT library.

    Returns ``(jit_library, program_defs_by_name)``.  ``program_defs``
    keys: ``TeleportRepropagateMemoryZ``, ``TeleportConditionalMemoryZ``,
    ``TeleportRepropagateMemoryX``, ``TeleportConditionalMemoryX``.
    """
    from deq.circuit.model import ProgramDefinition
    from deq.circuit.parser import parse_file

    merged = parse_file(str(TELEPORTATION_D3_DEQ))
    jit_library = build_jit_library(merged)
    program_defs = {
        d.name: d for d in merged.definitions if isinstance(d, ProgramDefinition)
    }
    return jit_library, program_defs


class TestTeleportationD3:
    """End-to-end compilation of surface-code logical teleportation PROGRAMs.

    Exercises both new branch features on the same surface-code fixture:

    * ``@REPROPAGATE`` (``TeleportRepropagate*`` programs) infers the
      conditional teleportation correction from the inlined flat circuit.
    * Explicit ``CONDITIONAL`` (``TeleportConditional*`` programs) emits
      synthesized identity gadgets that the canonicalizer's step-9
      absorption folds back into the propagation/correction matrices.

    Both encodings must produce ``static_jit_compile``-able binaries that
    pass the physical validator.
    """

    @pytest.mark.parametrize(
        "program_name",
        [
            "TeleportRepropagateMemoryZ",
            "TeleportConditionalMemoryZ",
            "TeleportRepropagateMemoryX",
            "TeleportConditionalMemoryX",
        ],
    )
    def test_program_compiles_to_valid_binary(
        self,
        teleportation_d3_setup: tuple[jit_pb.JitLibrary, dict[str, object]],
        program_name: str,
    ) -> None:
        from deq.cli.jit import compile_program_for_jit

        jit_library, program_defs = teleportation_d3_setup
        program_def = program_defs[program_name]

        compiled, assertions = compile_program_for_jit(jit_library, program_def)

        # The program must emit at least one ASSERT_EQ (rec[-1] 0) and
        # one JIT instruction per gadget application in the body.
        assert len(assertions) == 1
        assert assertions[0][1] is False  # expected_value=0

        # Build a fresh library that includes the program stream and run
        # the static JIT compiler.  is_valid_and_physical sanity-checks
        # the produced deq.bin.
        lib = jit_pb.JitLibrary()
        lib.CopyFrom(jit_library)
        lib.ClearField("program")
        for instr, _src in compiled:
            lib.program.append(instr)
        deq_bin = static_jit_compiler(lib)
        assert is_valid_and_physical(deq_bin)

    def test_repropagate_and_conditional_emit_same_propagation(
        self,
        teleportation_d3_setup: tuple[jit_pb.JitLibrary, dict[str, object]],
    ) -> None:
        """``TeleportRepropagate`` and ``TeleportConditional`` are
        operationally equivalent but have *structurally* different
        canonical forms:

        * ``TeleportRepropagate`` rebuilds the GADGET from the flat
          inlined circuit, so the ``MeasureBell`` sub-gadget's logical
          readouts are absorbed away (0 readouts on the composed
          GADGET).
        * ``TeleportConditional`` keeps ``MeasureBell``'s 2 readouts
          visible because the ``CONDITIONAL`` statements explicitly
          reference them via ``rec[-1]`` / ``rec[-2]``; the resulting
          conditional-correction contribution lives in
          ``physical_correction`` (via measurement_indices) rather than
          a separate ``logical_correction`` matrix.

        What MUST agree across both variants:

        * input / output port counts (same COMPOSE signature);
        * empty ``logical_correction`` (the canonical absorption pass
          clears it on both paths).
        """
        jit_library, _ = teleportation_d3_setup
        repro = next(
            gt for gt in jit_library.gadget_types if gt.base.name == "TeleportRepropagate"
        )
        cond = next(
            gt for gt in jit_library.gadget_types if gt.base.name == "TeleportConditional"
        )

        # Same compose-level signature.
        assert len(repro.base.inputs) == len(cond.base.inputs) == 1
        assert len(repro.base.outputs) == len(cond.base.outputs) == 1
        assert repro.base.inputs[0].ptype == cond.base.inputs[0].ptype
        assert repro.base.outputs[0].ptype == cond.base.outputs[0].ptype

        # Both must have an empty logical_correction (absorbed away).
        assert len(cond.base.logical_correction.i) == 0
        assert len(repro.base.logical_correction.i) == 0

        # The CONDITIONAL form preserves MeasureBell's 2 logical readouts;
        # the REPROPAGATE form folds them into the flat-circuit analysis.
        assert len(cond.base.readouts) == 2
        assert len(repro.base.readouts) == 0


# ---------------------------------------------------------------------------
# Lattice surgery (d=3 surface code) — true spatial merge-and-split test.
# ---------------------------------------------------------------------------

LATTICE_SURGERY_D3_DEQ = (
    Path(__file__).resolve().parents[1]
    / "circuit"
    / "surface_code"
    / "lattice_surgery_d3.deq"
)


@pytest.fixture(scope="module")
def lattice_surgery_d3_library() -> jit_pb.JitLibrary:
    """Parse ``lattice_surgery_d3.deq`` and build its JIT library."""
    from deq.circuit.parser import parse_file

    return build_jit_library(parse_file(str(LATTICE_SURGERY_D3_DEQ)))


@pytest.fixture(scope="module")
def lattice_surgery_d3_setup() -> tuple[jit_pb.JitLibrary, dict[str, object]]:
    """Parse ``lattice_surgery_d3.deq`` and return library + PROGRAMs.

    Returns ``(jit_library, program_defs_by_name)``.  ``program_defs``
    keys: ``LSMergeCorrectedMemoryZ``, ``LSMergeConditionalMemoryZ``,
    ``LSMergeCorrectedMemoryX``, ``LSMergeConditionalMemoryX``.
    """
    from deq.circuit.model import ProgramDefinition
    from deq.circuit.parser import parse_file

    merged = parse_file(str(LATTICE_SURGERY_D3_DEQ))
    jit_library = build_jit_library(merged)
    program_defs = {
        d.name: d for d in merged.definitions if isinstance(d, ProgramDefinition)
    }
    return jit_library, program_defs


class TestLatticeSurgeryD3:
    """Verify the structural properties of the d=3 lattice-surgery MZZ
    gadgets.

    Unlike the Bell-pair teleportation in ``teleportation_d3.deq``,
    this fixture spatially merges two surface-code patches via an
    intermediate column of |+⟩ data qubits, measures the four new bulk
    plaquettes spanning the seam, and splits the intermediate column
    back out via X-basis measurement.

    Two flavors of the same surgery are exercised:

    * ``MergeMZZ`` — leaves the lattice-surgery Pauli frame correction
      visible as a logical readout (``m_X19 ⊕ m_X20``);
    * ``MergeMZZCorrected`` — applies the frame correction in-circuit
      via Stim's ``CZ rec[-k] q`` classically-conditioned Z, so the
      gadget acts as logical identity on both patches with NO
      measurement-dependent frame leakage.

    The COMPOSE wrappers (``LSMergePassthrough``, ``LSMergeCorrected``)
    wrap each variant through the default merge() pathway and must
    produce gadgets with empty ``logical_correction`` after the
    absorption pass.
    """

    def test_merge_mzz_has_two_input_two_output_ports(
        self,
        lattice_surgery_d3_library: jit_pb.JitLibrary,
    ) -> None:
        """``MergeMZZ`` is a 2-input, 2-output gadget — both patches
        survive the merge-and-split (it is non-destructive on logical
        information except for the joint frame correction)."""
        merge = next(
            gt for gt in lattice_surgery_d3_library.gadget_types if gt.base.name == "MergeMZZ"
        )
        assert len(merge.base.inputs) == 2
        assert len(merge.base.outputs) == 2
        # Both ports are the same SurfaceCode port type.
        assert merge.base.inputs[0].ptype == merge.base.inputs[1].ptype
        assert merge.base.outputs[0].ptype == merge.base.outputs[1].ptype
        assert merge.base.inputs[0].ptype == merge.base.outputs[0].ptype

    def test_merge_mzz_exposes_frame_correction_readout(
        self,
        lattice_surgery_d3_library: jit_pb.JitLibrary,
    ) -> None:
        """``MergeMZZ`` exposes the Pauli frame correction bit as a
        single logical readout (= m_X19 ⊕ m_X20 from the X-basis split
        measurements of the intermediate column)."""
        merge = next(
            gt for gt in lattice_surgery_d3_library.gadget_types if gt.base.name == "MergeMZZ"
        )
        assert len(merge.base.readouts) == 1
        # The readout reads two measurement records (the M5, M6 of the
        # MX 18 19 20 split).
        assert len(merge.base.readouts[0].measurement_indices) == 2

    def test_merge_mzz_corrected_has_no_readouts(
        self,
        lattice_surgery_d3_library: jit_pb.JitLibrary,
    ) -> None:
        """``MergeMZZCorrected`` applies the frame correction
        in-circuit via ``CZ rec`` feedforward, so it has NO logical
        readout — the gadget is logical identity on both patches.
        """
        merge = next(
            gt
            for gt in lattice_surgery_d3_library.gadget_types
            if gt.base.name == "MergeMZZCorrected"
        )
        assert len(merge.base.readouts) == 0

    def test_merge_mzz_corrected_acts_as_identity_on_logicals(
        self,
        lattice_surgery_d3_library: jit_pb.JitLibrary,
    ) -> None:
        """After the in-circuit correction, ``MergeMZZCorrected`` has a
        diagonal correction_propagation matrix on both patches'
        logical observables and no measurement contributions on
        ``physical_correction`` for those rows.
        """
        merge = next(
            gt
            for gt in lattice_surgery_d3_library.gadget_types
            if gt.base.name == "MergeMZZCorrected"
        )
        cp = merge.base.correction_propagation
        pc = merge.base.physical_correction
        # The 4 logical observable rows (LX_A=0, LZ_A=1, LX_B=10, LZ_B=11)
        # should have only the identity entry in cp (diagonal) and no
        # entries in pc.
        cp_pairs = set(zip(cp.i, cp.j))
        pc_pairs = set(zip(pc.i, pc.j))
        for logical_row in (0, 1, 10, 11):
            assert (logical_row, logical_row) in cp_pairs, (
                f"row {logical_row}: missing identity in correction_propagation"
            )
            pc_row = {(r, c) for (r, c) in pc_pairs if r == logical_row}
            assert pc_row == set(), (
                f"row {logical_row}: unexpected pc entries {pc_row}; "
                f"in-circuit correction should fully absorb them"
            )

    def test_compose_pathways_produce_empty_logical_correction(
        self,
        lattice_surgery_d3_library: jit_pb.JitLibrary,
    ) -> None:
        """Both COMPOSE pathways produce gadgets with an empty
        ``logical_correction`` matrix: the merge() absorption pass
        folds any conditional contribution into ``correction_propagation``
        / ``physical_correction``.
        """
        for name in ("LSMergePassthrough", "LSMergeCorrected", "LSMergeConditional"):
            gt = next(
                g for g in lattice_surgery_d3_library.gadget_types if g.base.name == name
            )
            assert len(gt.base.logical_correction.i) == 0, (
                f"{name}: logical_correction should be empty after merge() absorption"
            )

    def test_compose_pathways_have_two_input_two_output_ports(
        self,
        lattice_surgery_d3_library: jit_pb.JitLibrary,
    ) -> None:
        """All three COMPOSE wrappers preserve the 2-in / 2-out
        signature of their underlying merge gadget."""
        for name in ("LSMergePassthrough", "LSMergeCorrected", "LSMergeConditional"):
            gt = next(
                g for g in lattice_surgery_d3_library.gadget_types if g.base.name == name
            )
            assert len(gt.base.inputs) == 2, f"{name} should have 2 inputs"
            assert len(gt.base.outputs) == 2, f"{name} should have 2 outputs"

    def test_ls_merge_conditional_matches_corrected(
        self,
        lattice_surgery_d3_library: jit_pb.JitLibrary,
    ) -> None:
        """``LSMergeConditional`` applies the Pauli frame correction
        via a COMPOSE-level ``CONDITIONAL rec[-1] Z0 0`` rather than
        an in-circuit ``CZ rec[...]``.  After the merge() absorption
        pass the resulting propagation matrices must match the
        in-circuit variant ``LSMergeCorrected``: the logical rows of
        both patches end up with no measurement contributions on
        ``physical_correction`` (the frame correction is fully
        absorbed) and the correction_propagation is the identity on
        logical observables and passthrough stabs.
        """
        conditional = next(
            g
            for g in lattice_surgery_d3_library.gadget_types
            if g.base.name == "LSMergeConditional"
        )
        corrected = next(
            g
            for g in lattice_surgery_d3_library.gadget_types
            if g.base.name == "LSMergeCorrected"
        )
        cond_cp = set(
            zip(
                conditional.base.correction_propagation.i,
                conditional.base.correction_propagation.j,
            )
        )
        corr_cp = set(
            zip(
                corrected.base.correction_propagation.i,
                corrected.base.correction_propagation.j,
            )
        )
        cond_pc = set(
            zip(
                conditional.base.physical_correction.i,
                conditional.base.physical_correction.j,
            )
        )
        corr_pc = set(
            zip(
                corrected.base.physical_correction.i,
                corrected.base.physical_correction.j,
            )
        )
        assert cond_cp == corr_cp, (
            "LSMergeConditional.correction_propagation should match "
            "LSMergeCorrected after CONDITIONAL absorption"
        )
        assert cond_pc == corr_pc, (
            "LSMergeConditional.physical_correction should match "
            "LSMergeCorrected after CONDITIONAL absorption"
        )

    def test_ls_merge_conditional_has_readout(
        self,
        lattice_surgery_d3_library: jit_pb.JitLibrary,
    ) -> None:
        """``LSMergeConditional`` preserves the underlying
        ``MergeMZZ`` readout (the COMPOSE-level CONDITIONAL is
        absorbed into ``correction_propagation`` /
        ``physical_correction`` but does not eliminate the readout
        itself — the decoder still needs the measurement bit to apply
        the correction)."""
        conditional = next(
            g
            for g in lattice_surgery_d3_library.gadget_types
            if g.base.name == "LSMergeConditional"
        )
        assert len(conditional.base.readouts) == 1
        assert len(conditional.base.readouts[0].measurement_indices) == 2


class TestLatticeSurgeryD3Programs:
    """End-to-end compilation of lattice-surgery memory PROGRAMs.

    These programs are the lattice-surgery analogues of the
    ``Teleport*Memory*`` programs in ``TestTeleportationD3``: they
    verify that both the in-circuit feedforward variant
    (``LSMergeCorrected``) and the COMPOSE-level ``CONDITIONAL`` variant
    (``LSMergeConditional``) compile to a valid binary that the static
    JIT compiler / physical validator accept.

    Each program prepares two surface-code patches in ``|0_L⟩`` (or
    ``|+_L⟩``), applies the lattice-surgery merge, then measures each
    patch in the matching basis.  The merge is logical identity on
    both patches once the frame correction is applied, so each
    ``MeasureZ`` (or ``MeasureX``) outcome must read ``0``
    deterministically — encoded as two ``ASSERT_EQ rec[-k] 0``
    statements.
    """

    @pytest.mark.parametrize(
        "program_name",
        [
            "LSMergeCorrectedMemoryZ",
            "LSMergeConditionalMemoryZ",
            "LSMergeProgramConditionalMemoryZ",
        ],
    )
    def test_program_compiles_to_valid_binary(
        self,
        lattice_surgery_d3_setup: tuple[jit_pb.JitLibrary, dict[str, object]],
        program_name: str,
    ) -> None:
        from deq.cli.jit import compile_program_for_jit

        jit_library, program_defs = lattice_surgery_d3_setup
        program_def = program_defs[program_name]

        compiled, assertions = compile_program_for_jit(jit_library, program_def)

        # Each memory program asserts both ``MeasureZ`` readouts equal 0.
        assert len(assertions) == 2
        for assertion in assertions:
            # ``compile_program_for_jit`` returns ``(abs_index, expected,
            # source)`` tuples; we only care that both are ``ASSERT_EQ ... 0``.
            assert assertion[1] is False

        # Re-run the static JIT compiler with the program stream to make
        # sure the produced deq.bin is physically valid (no dangling
        # measurements, no missing CONDITIONAL absorption, etc.).
        lib = jit_pb.JitLibrary()
        lib.CopyFrom(jit_library)
        lib.ClearField("program")
        for instr, _src in compiled:
            lib.program.append(instr)
        deq_bin = static_jit_compiler(lib)
        assert is_valid_and_physical(deq_bin)

    def test_corrected_and_conditional_programs_have_same_assertions(
        self,
        lattice_surgery_d3_setup: tuple[jit_pb.JitLibrary, dict[str, object]],
    ) -> None:
        """All three Z-basis variants — in-circuit ``CZ rec[...]``
        (``LSMergeCorrectedMemoryZ``), COMPOSE-level ``CONDITIONAL``
        (``LSMergeConditionalMemoryZ``), and PROGRAM-level
        ``CONDITIONAL`` (``LSMergeProgramConditionalMemoryZ``) — reach
        the same logical state by different routes and therefore
        produce the same number of ``ASSERT_EQ rec[-k] 0`` assertions
        with the same expected values.

        Neither absolute measurement offsets nor JIT instruction
        counts are compared: the in-circuit variant folds the merge
        readout away via feedforward, the COMPOSE-level CONDITIONAL
        preserves the readout but absorbs into the COMPOSE matrices
        (no extra JIT instruction), and the PROGRAM-level CONDITIONAL
        emits an extra synthesised identity gadget instruction.  The
        end-to-end behaviour (deterministic ``MeasureZ = 0``) is the
        same for all three, verified by the sample/simulate tests in
        ``TestConditionalEndToEnd``.
        """
        from deq.cli.jit import compile_program_for_jit

        jit_library, program_defs = lattice_surgery_d3_setup
        program_names = [
            "LSMergeCorrectedMemoryZ",
            "LSMergeConditionalMemoryZ",
            "LSMergeProgramConditionalMemoryZ",
        ]
        results = [
            compile_program_for_jit(jit_library, program_defs[name])
            for name in program_names
        ]
        assertion_counts = [len(asserts) for _, asserts in results]
        assertion_values = [
            tuple(a[1] for a in asserts) for _, asserts in results
        ]
        assert assertion_counts == [2, 2, 2], (
            f"expected 2 assertions per variant; got "
            f"{dict(zip(program_names, assertion_counts))}"
        )
        assert len(set(assertion_values)) == 1, (
            f"assertion expected values differ across variants: "
            f"{dict(zip(program_names, assertion_values))}"
        )


# ---------------------------------------------------------------------------
# End-to-end ``deq sample`` + ``deq simulate ler`` smoke tests for both
# COMPOSE-level and PROGRAM-level ``CONDITIONAL`` correction pathways.
# ---------------------------------------------------------------------------


CONDITIONAL_E2E_PROGRAMS: list[tuple[str, Path]] = [
    # (program_name, .deq source file).  All listed programs encode a
    # logical-memory experiment whose ``ASSERT_EQ rec[-k] 0`` statements
    # must hold on every noiseless sample.  The mid-circuit measurement
    # outcomes (Bell-pair / lattice-surgery merge readouts) are
    # individually random; ``ASSERT_EQ`` checks the *corrected* logical
    # readout, which the CONDITIONAL pathway must fold into the readout's
    # measurement set.
    ("TeleportConditionalMemoryZ", TELEPORTATION_D3_DEQ),
    ("TeleportConditionalMemoryX", TELEPORTATION_D3_DEQ),
    ("TeleportProgramConditionalMemoryZ", TELEPORTATION_D3_DEQ),
    ("TeleportProgramConditionalMemoryX", TELEPORTATION_D3_DEQ),
    ("LSMergeConditionalMemoryZ", LATTICE_SURGERY_D3_DEQ),
    ("LSMergeProgramConditionalMemoryZ", LATTICE_SURGERY_D3_DEQ),
]


def _evaluate_assertions_on_sample(
    deq_file: Path,
    program_name: str,
    *,
    shots: int,
    seed: int,
) -> tuple[int, int]:
    """Compile *program_name* from *deq_file*, sample *shots* shots of
    its noiseless stim circuit, and evaluate the program's
    ``ASSERT_EQ`` statements on every shot.

    Returns ``(total_assertions, failed_assertions)``.  A passing
    program has ``failed_assertions == 0``.
    """
    import tempfile

    from deq.cli.jit import compile_program_for_jit
    from deq.cli.sample import (
        _compile_deq_to_stim_and_bin,
        _sample_stim_text,
        _strip_noise_text,
    )
    from deq.cli.util import parse_bits
    from deq.circuit.model import ProgramDefinition
    from deq.circuit.parser import parse_file
    from deq.spec.canonical import canonicalize
    import deq.proto.deq_bin_pb2 as pb

    with tempfile.TemporaryDirectory() as tmpdir:
        stim_path, bin_path = _compile_deq_to_stim_and_bin(
            (str(deq_file),),
            tmpdir,
            program=program_name,
            jit=None,
            jobs=1,
            plugin=None,
            mako=None,
            skip_mako_warning=True,
        )
        with open(stim_path, encoding="utf-8") as f:
            stim_text = _strip_noise_text(f.read())
        with open(bin_path, "rb") as f:
            lib = pb.Library.FromString(f.read())

    hex_samples = _sample_stim_text(stim_text, shots, seed)
    canonical_form = canonicalize(lib)
    gt = canonical_form.gadget_type
    num_meas = len(gt.measurements)

    # The canonical readout_propagation's last column is the affine
    # (constant) column: a 1 entry there means the readout is
    # deterministically flipped (e.g. from a VIRTUAL Pauli correction).
    # ``interpret_measurements`` applies this when computing readout
    # values; we mirror it here so the sample-check matches the
    # decoder's interpretation.
    rp = gt.readout_propagation
    affine_col = rp.cols - 1 if rp.cols > 0 else -1
    readout_affine: list[bool] = [False] * len(gt.readouts)
    for r, c in zip(rp.i, rp.j):
        if c == affine_col:
            readout_affine[r] = not readout_affine[r]

    parsed = parse_file(str(deq_file))
    program_defs = {
        d.name: d
        for d in parsed.definitions
        if isinstance(d, ProgramDefinition)
    }
    jit_lib = build_jit_library(parsed)
    _, assertions = compile_program_for_jit(jit_lib, program_defs[program_name])

    if not assertions:
        raise AssertionError(
            f"PROGRAM {program_name!r} has no ASSERT_EQ statements — "
            f"the sample-check test would vacuously pass"
        )

    total = 0
    failed = 0
    for hex_meas in hex_samples:
        bits = parse_bits(hex_meas, num_meas)
        readout_values = []
        for idx, r in enumerate(gt.readouts):
            parity = 0
            for mi in r.measurement_indices:
                parity ^= bits[mi]
            if readout_affine[idx]:
                parity ^= 1
            readout_values.append(parity)
        for abs_index, expected_value, _src in assertions:
            total += 1
            actual = readout_values[abs_index]
            if actual != (1 if expected_value else 0):
                failed += 1
    return total, failed


class TestConditionalEndToEnd:
    """``deq sample`` + ``deq simulate ler`` end-to-end smoke tests for
    every CONDITIONAL correction pathway exercised in this branch.

    Each program is run through:

    * :func:`_evaluate_assertions_on_sample` — pulls 20 noiseless
      samples from the program's stim circuit, evaluates the canonical
      readout values, and asserts that every ``ASSERT_EQ rec[-k] 0``
      statement holds on every shot.  This validates that the *deq*
      side of the pipeline (transpilation, compose canonicalisation,
      program-level remote-conditional-correction absorption) folds
      the CONDITIONAL contribution into the readout's measurement
      set, so the deterministic logical bit comes out as expected
      despite the random mid-circuit-measurement values.
    * ``deq simulate ler`` (invoked as a subprocess) — runs the same
      program through the full *deq_runtime* decoder for 20 shots and
      asserts zero logical errors.  This validates that the *deq
      runtime* side of the pipeline (decoder + classical correction
      application) is consistent with the canonicalisation that
      ``deq sample`` exercises.
    """

    @pytest.mark.parametrize(
        "program_name,deq_file",
        CONDITIONAL_E2E_PROGRAMS,
        ids=lambda v: v if isinstance(v, str) else v.stem,
    )
    def test_sample_20_shots_all_assertions_pass(
        self,
        program_name: str,
        deq_file: Path,
    ) -> None:
        total, failed = _evaluate_assertions_on_sample(
            deq_file, program_name, shots=20, seed=42
        )
        assert failed == 0, (
            f"{program_name}: {failed}/{total} ASSERT_EQ checks failed "
            f"across 20 noiseless samples — the CONDITIONAL correction "
            f"is not folded into the canonical readout"
        )

    @pytest.mark.parametrize(
        "program_name,deq_file",
        CONDITIONAL_E2E_PROGRAMS,
        ids=lambda v: v if isinstance(v, str) else v.stem,
    )
    def test_simulate_ler_20_shots_zero_logical_errors(
        self,
        program_name: str,
        deq_file: Path,
    ) -> None:
        import re
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "deq",
                "simulate",
                "ler",
                str(deq_file),
                "--program",
                program_name,
                "--shots",
                "20",
                "--errors",
                "100",
                "--batch-size",
                "20",
                "--seed",
                "42",
                "--jobs",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"{program_name}: 'deq simulate ler' exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        m_shots = re.search(r"Shots:\s+(\d+)", result.stdout)
        m_errs = re.search(r"Logical errors:\s+(\d+)", result.stdout)
        assert m_shots is not None and m_errs is not None, (
            f"{program_name}: could not parse simulator output:\n"
            f"{result.stdout}"
        )
        assert int(m_shots.group(1)) == 20, (
            f"{program_name}: expected 20 shots, got {m_shots.group(1)}"
        )
        assert int(m_errs.group(1)) == 0, (
            f"{program_name}: expected 0 logical errors over 20 noiseless "
            f"shots, got {m_errs.group(1)}"
        )


_FLIP_READOUT_FIXTURE_SOURCE = """\
CODE TrivialCode [[1,1]] {
    LOGICAL X0 Z0
}

# Prepare |+> and measure in the X basis.  Raw MX outcome is
# deterministically 0; ``FLIP`` marks the readout as naturally
# flipped, so its canonical value is 1.  The ASSERT_EQ in
# ``TestFlippedReadout`` then verifies that the canonical readout
# evaluation applies the affine bit correctly.
GADGET PreparePlusMeasureXFlipped {
    RX 0
    MX 0
    READOUT rec[-1] FLIP
}

PROGRAM TestFlippedReadout {
    PreparePlusMeasureXFlipped
    ASSERT_EQ rec[-1] 1
}
"""


class TestReadoutAffineFlip:
    """Regression tests for the readout affine-flip
    (``READOUT ... FLIP``) handling in
    :func:`_evaluate_assertions_on_sample`.

    The canonical readout's last ``readout_propagation`` column is the
    affine bit: when set, the readout's value is deterministically
    flipped before any decoder correction.  The sample-check helper
    mirrors ``deq.cli.interpret.interpret_measurements`` and must XOR
    that bit into the computed readout value; otherwise ``ASSERT_EQ
    rec[-k] 1`` against a FLIP'd readout would always look like a
    bit-flip error to the sample checker.
    """

    def test_flipped_readout_assertion_passes_on_all_shots(
        self, tmp_path: Path
    ) -> None:
        """``TestFlippedReadout`` asserts ``rec[-1] == 1`` against a
        readout whose raw bit is deterministically 0 and whose canonical
        value is flipped to 1 by the ``READOUT ... FLIP`` marker.  Every
        sampled shot must pass the ASSERT_EQ check."""
        deq_path = tmp_path / "flip_readout_fixture.deq"
        deq_path.write_text(_FLIP_READOUT_FIXTURE_SOURCE, encoding="utf-8")

        total, failed = _evaluate_assertions_on_sample(
            deq_path, "TestFlippedReadout", shots=20, seed=42
        )
        assert total == 20, f"expected 20 assertion evaluations, got {total}"
        assert failed == 0, (
            f"FLIP readout assertion failed on {failed}/{total} shots — "
            f"the affine bit handling in the sample-check helper is "
            f"missing or wrong"
        )
