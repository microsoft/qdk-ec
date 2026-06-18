"""Tests for friendly error messages from CODE/port validation.

These messages exist to help users locate authoring mistakes in
``.deq`` files (e.g. ``CODE Foo[[1,1,1]]`` whose stabilizers actually
reference qubit indices 1 and 2). Without them, the underlying
problems surface as deep ``IndexError`` / ``KeyError`` exceptions
from the JIT transpiler that give no hint at the offending CODE,
operator, or GADGET.
"""

import pytest

from deq.circuit.parser import parse
from deq.transpiler.code_validation import validate_code
from deq.transpiler.jit_library_builder import build_jit_library


class TestQubitIndexRangeValidation:
    """``validate_code`` rejects stabilizer/logical qubit indices >= n."""

    def test_stabilizer_index_out_of_range_message(self) -> None:
        source = """
        CODE RepetitionCode[[1,1,1]] {
            LOGICAL X0 Z0
            STABILIZER Z0*Z1 Z1*Z2
        }
        """
        qfile = parse(source)
        code = qfile.definitions[0]
        with pytest.raises(ValueError) as exc:
            validate_code(code)
        msg = str(exc.value)
        assert "RepetitionCode" in msg
        assert "STABILIZER" in msg
        assert "out of range" in msg
        assert "[0, 1)" in msg
        assert "n >= 3" in msg

    def test_logical_index_out_of_range_message(self) -> None:
        source = """
        CODE WrongN[[1,1,1]] {
            LOGICAL X0*X1*X2 Z0
            STABILIZER
        }
        """
        qfile = parse(source)
        code = qfile.definitions[0]
        with pytest.raises(ValueError) as exc:
            validate_code(code)
        msg = str(exc.value)
        assert "WrongN" in msg
        assert "LOGICAL X0" in msg
        assert "out of range" in msg
        assert "n >= 3" in msg

    def test_qubit_index_range_check_raised_via_build_jit_library(self) -> None:
        source = """
        CODE RepetitionCode[[1,1,1]] {
            LOGICAL X0*X1*X2 Z0
            STABILIZER Z0*Z1 Z1*Z2
        }
        """
        qfile = parse(source)
        with pytest.raises(ValueError, match="out of range"):
            build_jit_library(qfile)

    def test_in_range_indices_pass(self) -> None:
        source = """
        CODE RepetitionCode[[3,1,1]] {
            LOGICAL X0*X1*X2 Z0
            STABILIZER Z0*Z1 Z1*Z2
        }
        """
        qfile = parse(source)
        code = qfile.definitions[0]
        validate_code(code)

    def test_error_message_mentions_source_line(self) -> None:
        source = "\n\n\nCODE Tiny[[1,1,1]] {\n    LOGICAL X0 Z0\n    STABILIZER Z0*Z3\n}\n"
        qfile = parse(source)
        code = qfile.definitions[0]
        with pytest.raises(ValueError) as exc:
            validate_code(code)
        msg = str(exc.value)
        assert "line" in msg


class TestUnknownPortCodeName:
    """``_validate_port_qubit_count`` reports a friendly error for unknown CODE."""

    def test_unknown_input_port_lists_known_codes(self) -> None:
        source = """
        CODE Real[[1,1,1]] {
            LOGICAL X0 Z0
            STABILIZER
        }
        GADGET G {
            INPUT Trivial 0
            OUTPUT Real 0
        }
        """
        qfile = parse(source)
        with pytest.raises(ValueError) as exc:
            build_jit_library(qfile)
        msg = str(exc.value)
        assert "INPUT" in msg
        assert "GADGET 'G'" in msg
        assert "'Trivial'" in msg
        assert "Known CODE names" in msg
        assert "'Real'" in msg

    def test_unknown_output_port_message(self) -> None:
        source = """
        CODE Real[[1,1,1]] {
            LOGICAL X0 Z0
            STABILIZER
        }
        GADGET G {
            INPUT Real 0
            OUTPUT Missing 0
        }
        """
        qfile = parse(source)
        with pytest.raises(ValueError) as exc:
            build_jit_library(qfile)
        msg = str(exc.value)
        assert "OUTPUT" in msg
        assert "'Missing'" in msg
        assert "Known CODE names" in msg
