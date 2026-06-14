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
