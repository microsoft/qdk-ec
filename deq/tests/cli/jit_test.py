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
import deq.proto.util_pb2 as util_pb

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

    def test_synthesizes_identity_gadget(
        self, trivial_code_k3_jit_library: jit_pb.JitLibrary
    ) -> None:
        """CONDITIONAL inserts a synthesized identity gadget that:

        1. Bears a ``remote_conditional_correction`` modifier pointing
           at the most recent logical readout.
        2. Chains into the wire's producer and hands the wire on to the
           next consumer.
        3. Introduces a new ``__identity_...`` gtype into the library.
        """
        # Snapshot gadget types so we can check that exactly one identity
        # gtype is appended.
        before = {gt.base.gtype for gt in trivial_code_k3_jit_library.gadget_types}

        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ OUT(0)\n"
            "PrepareZ OUT(1)\n"
            "MeasureZ IN(1)\n"
            "CONDITIONAL rec[-1] X0 0\n"
            "MeasureZ IN(0)",
        )

        # (1) Program shape: 5 instructions = 2 PrepareZ + 1 MeasureZ +
        # identity + MeasureZ.  The identity carries the modifier.
        assert len(instructions) == 5
        cond_instr = instructions[3]
        assert cond_instr.gadget.HasField("modifier")
        rcc = cond_instr.gadget.modifier.remote_conditional_correction
        # References the MeasureZ on wire 1 (gid 3, readout 0).
        assert len(rcc.remote_readouts) == 1
        assert rcc.remote_readouts[0].gid == 3
        assert rcc.remote_readouts[0].readout_index == 0
        # X on logical qubit 0 flips the LZ_0 column (row 1).
        assert list(rcc.correction.i) == [1]
        assert list(rcc.correction.j) == [0]

        # (2) Identity gadget (gid 4) chains through — consumes wire 0's
        # producer (gid 1) and the following MeasureZ (gid 5) consumes it.
        assert cond_instr.gadget.gid == 4
        assert [c.gid for c in cond_instr.gadget.connectors] == [1]
        assert instructions[4].gadget.gid == 5
        assert [c.gid for c in instructions[4].gadget.connectors] == [4]

        # (3) A single new __identity_... gtype was appended to the library.
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
        assert (
            len(new_gt.base.inputs) == len(new_gt.base.outputs) == 1
        )
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

    @pytest.mark.parametrize(
        "pauli,expected_i",
        [
            # X0*Z1 flips LZ_0 (col 1) and LX_1 (col 2).
            ("X0*Z1", [1, 2]),
            # Y2 = X2 * Z2 flips LX_2 (col 4) and LZ_2 (col 5).
            ("Y2", [4, 5]),
            # X0 * X0 = identity; correction matrix is empty.
            ("X0*X0", []),
        ],
    )
    def test_correction_pauli(
        self,
        trivial_code_k3_jit_library: jit_pb.JitLibrary,
        pauli: str,
        expected_i: list[int],
    ) -> None:
        """``CONDITIONAL rec[-k] <pauli> wire`` builds the correction
        matrix by flipping the LZ_i / LX_i columns of every Pauli
        factor; identical factors cancel via XOR."""
        instructions = parse_jit_program(
            trivial_code_k3_jit_library,
            "PrepareZ OUT(0)\n"
            "PrepareZ OUT(1)\n"
            "MeasureZ IN(1)\n"
            f"CONDITIONAL rec[-1] {pauli} 0\n"
            "MeasureZ IN(0)",
        )
        rcc = instructions[3].gadget.modifier.remote_conditional_correction
        assert list(rcc.correction.i) == expected_i

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

    @pytest.mark.parametrize(
        "program,error_pattern",
        [
            # rec[-1] with no readouts emitted yet.
            (
                "PrepareZ OUT(0)\n"
                "CONDITIONAL rec[-1] X0 0\n"
                "MeasureZ IN(0)",
                "readout",
            ),
            # Wire 99 has no producer.
            (
                "PrepareZ OUT(0)\n"
                "MeasureZ IN(0)\n"
                "CONDITIONAL rec[-1] X0 99\n",
                "wire",
            ),
            # Logical qubit 99 is out of range for a k=3 code.
            (
                "PrepareZ OUT(0)\n"
                "MeasureZ IN(0)\n"
                "PrepareZ OUT(0)\n"
                "CONDITIONAL rec[-1] X99 0\n"
                "MeasureZ IN(0)",
                "logical qubit",
            ),
        ],
    )
    def test_invalid_conditional_raises(
        self,
        trivial_code_k3_jit_library: jit_pb.JitLibrary,
        program: str,
        error_pattern: str,
    ) -> None:
        """Common failure modes: rec offset out of range, unknown wire,
        logical-qubit index >= code.k."""
        with pytest.raises(ValueError, match=error_pattern):
            parse_jit_program(trivial_code_k3_jit_library, program)

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
# Merge-time residual helpers — shared across conditional-equivalence
# tests over the teleportation, lattice-surgery, and trivial-surgery
# fixtures.  Every test that compares two ``CONDITIONAL`` placements
# canonicalising to runtime-equivalent gadgets uses these helpers.
# ---------------------------------------------------------------------------


def _compute_zero_measurement_residual(
    jit_library: jit_pb.JitLibrary,
    gadget_name: str,
    input_obs: list[int],
) -> "np.ndarray":
    """Merge-time residual of *gadget_name* for the given
    input-observable pattern, with zero raw measurements and zero
    decoded correction.

    Isolates the ``cp · input`` and ``lc · (rp · input)`` contributions
    — which is where the merge-time propagator's correctness matters —
    from the physical-measurement and decoder-correction paths.  The
    runtime formula being evaluated (see ``pauli_frame_tracker.rs``)::

        readouts = raw + decoded.readouts + rp · input
        residual = cp · input + pc · raw + lc · readouts + decoded.residual
    """
    import numpy as np

    ptype_by_id = {pt.base.ptype: pt.base for pt in jit_library.port_types}
    gt = next(g for g in jit_library.gadget_types if g.base.name == gadget_name)
    base = gt.base
    n_in = sum(len(ptype_by_id[p.ptype].observables) for p in base.inputs)
    n_out = sum(len(ptype_by_id[p.ptype].observables) for p in base.outputs)
    assert len(input_obs) == n_in, (
        f"{gadget_name}: expected {n_in} input observables, "
        f"got {len(input_obs)}"
    )

    def dense(bm: util_pb.BitMatrix, rows: int, cols: int) -> "np.ndarray":
        m = np.zeros((rows, cols), dtype=np.uint8)
        for i, j in zip(bm.i, bm.j):
            m[i, j] = 1
        return m

    input_ext = np.array(list(input_obs) + [1], dtype=np.uint8)
    cp = dense(base.correction_propagation, n_out, n_in + 1)
    pc = dense(base.physical_correction, n_out, len(base.measurements))
    lc = dense(base.logical_correction, n_out, len(base.readouts))
    rp = dense(base.readout_propagation, len(base.readouts), n_in + 1)
    raw = np.zeros(len(base.measurements), dtype=np.uint8)
    raw_readouts = np.zeros(len(base.readouts), dtype=np.uint8)
    for c, r in enumerate(base.readouts):
        for m in r.measurement_indices:
            raw_readouts[c] ^= raw[m]
    decoded_readouts = np.zeros(len(base.readouts), dtype=np.uint8)
    decoded_residual = np.zeros(n_out, dtype=np.uint8)
    readouts = (raw_readouts + decoded_readouts + rp @ input_ext) % 2
    residual = (
        cp @ input_ext + pc @ raw + lc @ readouts + decoded_residual
    ) % 2
    return residual


def _assert_gadgets_runtime_equivalent(
    jit_library: jit_pb.JitLibrary,
    name_a: str,
    name_b: str,
) -> None:
    """
    Assert two gadgets produce identical merge-time residuals.
    """
    import numpy as np

    ptype_by_id = {pt.base.ptype: pt.base for pt in jit_library.port_types}
    gt_a = next(g for g in jit_library.gadget_types if g.base.name == name_a)
    gt_b = next(g for g in jit_library.gadget_types if g.base.name == name_b)
    n_in = sum(
        len(ptype_by_id[p.ptype].observables) for p in gt_a.base.inputs
    )
    assert n_in == sum(
        len(ptype_by_id[p.ptype].observables) for p in gt_b.base.inputs
    ), f"{name_a} and {name_b} have different input widths"

    basis_inputs: list[list[int]] = [[0] * n_in]
    for bit in range(n_in):
        vec = [0] * n_in
        vec[bit] = 1
        basis_inputs.append(vec)

    for inp in basis_inputs:
        res_a = _compute_zero_measurement_residual(jit_library, name_a, inp)
        res_b = _compute_zero_measurement_residual(jit_library, name_b, inp)
        assert np.array_equal(res_a, res_b), (
            f"{name_a} vs {name_b} disagree for input {inp}: "
            f"a={res_a.tolist()} b={res_b.tolist()}"
        )


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
    """Structural + runtime equivalence of the two logical-teleportation
    encodings in ``teleportation_d3.deq``:

    * ``@REPROPAGATE`` (``TeleportRepropagate*``) infers the conditional
      teleportation correction from the inlined flat circuit.
    * Explicit ``CONDITIONAL`` (``TeleportConditional*``) emits
      synthesised identity gadgets that the canonicaliser's step-9
      absorption folds back into the propagation/correction matrices.

    End-to-end binary validity + assertion + LER checks on the four
    ``TeleportConditional*``/``TeleportProgramConditional*`` PROGRAMs
    live in :class:`TestConditionalEndToEnd`.
    """

    def test_repropagate_and_conditional_emit_same_propagation(
        self,
        teleportation_d3_setup: tuple[jit_pb.JitLibrary, dict[str, object]],
    ) -> None:
        """``TeleportRepropagate`` and ``TeleportConditional`` are
        canonically equivalent: both compose pathways must expose the
        same compose-level signature, preserve ``MeasureBell``'s two
        logical readouts, and produce an empty ``logical_correction``
        (the canonical absorption pass clears it on both paths).

        The two pathways differ only in how the conditional Pauli
        frame correction is *expressed*:

        * ``TeleportConditional`` writes the correction as explicit
          ``CONDITIONAL rec[-k] <pauli> <wire>`` statements at the
          COMPOSE level; ``merge()`` then absorbs them into the
          composed matrices.
        * ``TeleportRepropagate`` inlines the sub-gadget body and
          re-derives the propagation from the flat circuit's
          Heisenberg flow, naturally folding the conditional Pauli
          updates into ``correction_propagation`` /
          ``physical_correction``.

        Both encodings must inherit the inlined ``MeasureBell``'s
        two readouts (``m_XX`` and ``m_ZZ``) so downstream code can
        still address those classical bits.
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

        # Both pathways must preserve MeasureBell's two logical
        # readouts (m_XX, m_ZZ) at identical measurement indices.
        assert len(cond.base.readouts) == 2
        assert len(repro.base.readouts) == 2
        assert [list(r.measurement_indices) for r in repro.base.readouts] == [
            list(r.measurement_indices) for r in cond.base.readouts
        ]


    @pytest.mark.parametrize(
        "cond_name,repro_name",
        [
            ("TeleportConditional", "TeleportRepropagate"),
            ("DoubleTeleportConditional", "DoubleTeleportRepropagate"),
            ("TripleTeleportConditional", "TripleTeleportRepropagate"),
        ],
    )
    def test_conditional_and_repropagate_runtime_equivalent(
        self,
        teleportation_d3_setup: tuple[jit_pb.JitLibrary, dict[str, object]],
        cond_name: str,
        repro_name: str,
    ) -> None:
        """The ``CONDITIONAL`` and ``@REPROPAGATE`` encodings must
        produce the same runtime residual for every input observable
        pattern, at every nesting depth.
        """
        jit_library, _ = teleportation_d3_setup
        _assert_gadgets_runtime_equivalent(jit_library, cond_name, repro_name)


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


class TestLatticeSurgeryD3:
    """Structural properties of the d=3 lattice-surgery joint-Z merge
    gadget in ``lattice_surgery_d3.deq``.

    The fixture spatially merges two surface-code patches via an
    intermediate column of |+⟩ data qubits.  ``MZZ`` measures four
    bulk plaquettes and two Z-type boundary 2-bodies spanning the seam,
    then destructively measures the intermediate column in the X basis.
    The product of the four Z-type outcomes equals the joint
    ``LZ_A · LZ_B`` parity, exposed via ``READOUT M0 M3 M4 M5``.
    ``ComposeMZZ`` wraps ``MZZ`` in a COMPOSE block so the
    canonicaliser folds the ``CONDITIONAL R0 OUT1.LX0`` byproduct into
    ``correction_propagation``.  Compilation validity plus the actual
    ASSERT_EQ semantics for ``ComposeMZZMemoryZ`` live in
    :class:`TestConditionalEndToEnd`.
    """

    @pytest.mark.parametrize(
        "gadget_name,expected_lc_rows",
        [
            # Base MZZ retains the CONDITIONAL R0 OUT1.LX0 byproduct
            # as a single logical_correction row.
            ("MZZ", 1),
            # ComposeMZZ absorbs it into correction_propagation.
            ("ComposeMZZ", 0),
        ],
    )
    def test_merge_shape(
        self,
        lattice_surgery_d3_library: jit_pb.JitLibrary,
        gadget_name: str,
        expected_lc_rows: int,
    ) -> None:
        """Each merge presentation must be a 2-in / 2-out gadget on the
        surface-code port type, expose the four-measurement joint-parity
        readout, and carry the expected number of ``logical_correction``
        rows."""
        gt = next(
            g
            for g in lattice_surgery_d3_library.gadget_types
            if g.base.name == gadget_name
        ).base
        assert len(gt.inputs) == len(gt.outputs) == 2
        assert (
            gt.inputs[0].ptype
            == gt.inputs[1].ptype
            == gt.outputs[0].ptype
            == gt.outputs[1].ptype
        )
        assert len(gt.readouts) == 1
        assert len(gt.readouts[0].measurement_indices) == 4
        assert len(gt.logical_correction.i) == expected_lc_rows
        if expected_lc_rows == 1:
            # Byproduct is driven by R0 (the joint-parity readout).
            assert gt.logical_correction.cols == 1
            assert list(gt.logical_correction.j) == [0]


# ---------------------------------------------------------------------------
# Trivial [[1,1,1]] code — same MZZ merge behavior on a single physical
# qubit per patch, no COMPOSE wrapper (the joint merge gadget is used
# directly inside PROGRAMs).
# ---------------------------------------------------------------------------

TRIVIAL_SURGERY_DEQ = (
    Path(__file__).resolve().parents[1]
    / "circuit"
    / "fixtures"
    / "trivial_surgery.deq"
)


@pytest.fixture(scope="module")
def trivial_surgery_library() -> jit_pb.JitLibrary:
    """Parse ``trivial_surgery.deq`` and build its JIT library."""
    from deq.circuit.parser import render_and_parse_file

    return build_jit_library(
        render_and_parse_file(
            str(TRIVIAL_SURGERY_DEQ), mako_defs=None, skip_mako_warning=True
        )
    )


class TestTrivialTwoMZZ:
    """Structural + runtime-equivalence properties of the trivial-code
    joint-Z merge gadgets in ``trivial_surgery.deq`` — the [[1,1,1]]
    analogues of the surface-code ``MZZ``/``ComposeMZZ`` merges.

    Two single-qubit patches (qubits 0 and 2) are joined by a
    ``|+⟩`` ancilla on qubit 1; ``MPP Z0*Z1`` + ``MPP Z1*Z2`` extract
    the joint ``LZ_A · LZ_B`` parity as ``READOUT M0 M1`` and the
    ancilla is split back out with ``MX 1``.  Two equivalent
    presentations:

    * ``TwoMZZ`` — raw joint-Z merge with an inline
      ``CONDITIONAL R0 OUT1.LX0`` byproduct, deferring absorption to
      the runtime decoder (one ``logical_correction`` row on the base).
    * ``TwoMZZCompose`` — ``TwoMerge`` + ``TwoSplit`` + post-split
      ``CONDITIONAL rec[-1] X0 1`` in a COMPOSE block, where
      canonicalisation absorbs the byproduct (empty
      ``logical_correction``).

    End-to-end assertion + LER validation of the memory programs
    lives in :class:`TestConditionalEndToEnd`.
    """

    @pytest.mark.parametrize(
        "merge_name,expected_lc_rows",
        [
            # Inline byproduct — raw TwoMZZ carries one logical_correction row.
            ("TwoMZZ", 1),
            # COMPOSE-level CONDITIONAL is absorbed into readout_propagation.
            ("TwoMZZCompose", 0),
        ],
    )
    def test_merge_shape(
        self,
        trivial_surgery_library: jit_pb.JitLibrary,
        merge_name: str,
        expected_lc_rows: int,
    ) -> None:
        """Each merge presentation must be a 2-in / 2-out gadget on the
        ``One`` port type, expose the two-measurement joint-parity
        readout, and carry the expected number of ``logical_correction``
        rows (1 for raw TwoMZZ; 0 after COMPOSE absorption)."""
        merge = next(
            gt
            for gt in trivial_surgery_library.gadget_types
            if gt.base.name == merge_name
        ).base
        assert len(merge.inputs) == len(merge.outputs) == 2
        assert (
            merge.inputs[0].ptype
            == merge.inputs[1].ptype
            == merge.outputs[0].ptype
            == merge.outputs[1].ptype
        )
        assert len(merge.readouts) == 1
        assert len(merge.readouts[0].measurement_indices) == 2
        assert len(merge.logical_correction.i) == expected_lc_rows
        if expected_lc_rows == 1:
            # Byproduct is driven by R0 (the joint-parity readout).
            assert merge.logical_correction.cols == 1
            assert list(merge.logical_correction.j) == [0]

    @pytest.mark.parametrize(
        "leaf_name,composed_name",
        [
            ("TwoMZZ", "TwoMZZCompose"),
            ("TwoMZZExtraCorrMixed", "TwoMZZExtraCorrOuter"),
        ],
    )
    def test_conditional_placement_runtime_equivalent(
        self,
        trivial_surgery_library: jit_pb.JitLibrary,
        leaf_name: str,
        composed_name: str,
    ) -> None:
        """CONDITIONAL placement across the leaf/compose boundary must
        not change runtime behavior.

        Two placement patterns are covered:

        1. **Multi-wire cascade at the leaf/compose boundary**
           (``TwoMZZ`` vs ``TwoMZZCompose``).  ``TwoMZZ`` is a leaf
           GADGET carrying a cross-wire ``CONDITIONAL R0 OUT1.LX0``
           whose driving readout (joint parity ``M0 + M1``) depends
           on flow through *both* input patches — the CONDITIONAL
           lives verbatim in the leaf ``logical_correction``.
           ``TwoMZZCompose`` implements the same operation as
           ``TwoMerge`` + ``TwoSplit`` + an outer
           ``CONDITIONAL rec[-1] X0 1``; ``canonical.merge`` absorbs
           that CONDITIONAL into ``correction_propagation``, so the
           composed base gadget has an empty ``logical_correction``.

        2. **Mixed inner/outer CONDITIONALs**
           (``TwoMZZExtraCorrMixed`` vs ``TwoMZZExtraCorrOuter``).
           ``TwoMZZExtraCorrMixed`` wraps ``TwoMZZ`` (whose inner
           GADGET-level CONDITIONAL survives as an
           ``logical_correction`` entry on the sub-gadget) and adds an
           *outer* COMPOSE-level ``CONDITIONAL rec[-1] Z0 0`` that
           references the sub-gadget's readout — so merge must
           compose an inner ``lc`` row with an outer CONDITIONAL
           targeting the same readout.  ``TwoMZZExtraCorrOuter``
           implements the same operation with both CONDITIONALs at
           the outer COMPOSE level (no inner CONDITIONAL).

        In both pairs the two encodings must produce identical
        merge-time residuals on the ``n_in + 1`` basis inputs — which,
        by affine-map linearity, implies agreement on every input
        observable pattern.
        """
        _assert_gadgets_runtime_equivalent(
            trivial_surgery_library, leaf_name, composed_name
        )

    def test_mixed_inner_outer_conditional_matrices_byte_identical(
        self,
        trivial_surgery_library: jit_pb.JitLibrary,
    ) -> None:
        """``TwoMZZExtraCorrMixed`` (inner ``TwoMZZ`` CONDITIONAL +
        outer COMPOSE CONDITIONAL) and ``TwoMZZExtraCorrOuter`` (both
        CONDITIONALs at the outer COMPOSE level) must produce
        byte-identical ``correction_propagation`` / ``physical_correction``
        / ``readout_propagation`` matrices after step-9 absorption.

        This is a stronger property than the runtime-residual
        equivalence checked in
        ``test_conditional_placement_runtime_equivalent``: absorption
        canonicalizes both encodings into the same merged form, so
        the serialized matrices agree entry-for-entry, not just
        modulo the runtime formula.
        """
        mixed = next(
            gt
            for gt in trivial_surgery_library.gadget_types
            if gt.base.name == "TwoMZZExtraCorrMixed"
        ).base
        outer = next(
            gt
            for gt in trivial_surgery_library.gadget_types
            if gt.base.name == "TwoMZZExtraCorrOuter"
        ).base

        def entries(bm: util_pb.BitMatrix) -> set[tuple[int, int]]:
            return set(zip(bm.i, bm.j))

        assert entries(mixed.correction_propagation) == entries(
            outer.correction_propagation
        )
        assert entries(mixed.physical_correction) == entries(
            outer.physical_correction
        )
        assert entries(mixed.readout_propagation) == entries(
            outer.readout_propagation
        )
        assert len(mixed.logical_correction.i) == 0
        assert len(outer.logical_correction.i) == 0


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
    ("ComposeMZZMemoryZ", LATTICE_SURGERY_D3_DEQ),
    ("TwoMZZMemoryZ", TRIVIAL_SURGERY_DEQ),
    ("TwoMZZMemoryZCompose", TRIVIAL_SURGERY_DEQ),
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
    from deq.circuit.parser import render_and_parse_file
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

    parsed = render_and_parse_file(
        str(deq_file), mako_defs=None, skip_mako_warning=True
    )
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
                "--skip-mako-warning",
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
