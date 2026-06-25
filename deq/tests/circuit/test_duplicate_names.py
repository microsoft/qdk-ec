"""Tests for duplicate-name detection in CODE/GADGET/COMPOSE/PROGRAM definitions."""

from pathlib import Path

import pytest

from deq.circuit.parser import parse, parse_file

# ── Single-file duplicates (via parse) ───────────────────────────────


class TestSingleFileDuplicates:
    def test_duplicate_gadget_raises(self) -> None:
        text = """\
GADGET PrepareZZ {
    R 0
}

GADGET PrepareZZ {
    R 0
}
"""
        with pytest.raises(
            SyntaxError, match="GADGET 'PrepareZZ' is defined more than once"
        ):
            parse(text)

    def test_duplicate_code_raises(self) -> None:
        text = """\
CODE Rep [[3,1]] {
    LOGICAL X0*X1*X2 Z0
    STABILIZER Z0*Z1
    STABILIZER Z1*Z2
}

CODE Rep [[3,1]] {
    LOGICAL X0*X1*X2 Z0
    STABILIZER Z0*Z1
    STABILIZER Z1*Z2
}
"""
        with pytest.raises(SyntaxError, match="CODE 'Rep' is defined more than once"):
            parse(text)

    def test_duplicate_compose_raises(self) -> None:
        text = """\
GADGET G {
    R 0
}

COMPOSE C {
    G
}

COMPOSE C {
    G
}
"""
        with pytest.raises(SyntaxError, match="COMPOSE 'C' is defined more than once"):
            parse(text)

    def test_duplicate_program_raises(self) -> None:
        text = """\
GADGET G {
    R 0
}

PROGRAM P {
    G
}

PROGRAM P {
    G
}
"""
        with pytest.raises(SyntaxError, match="PROGRAM 'P' is defined more than once"):
            parse(text)

    def test_same_name_across_different_kinds_raises(self) -> None:
        # All definition kinds share a single global namespace.
        text = """\
CODE Foo [[1,1]] {
    LOGICAL X0 Z0
}

GADGET Foo {
    R 0
}
"""
        with pytest.raises(SyntaxError) as excinfo:
            parse(text)
        message = str(excinfo.value)
        assert "'Foo'" in message
        assert "CODE" in message
        assert "GADGET" in message
        assert "defined more than once" in message

    def test_three_duplicates_reports_first_pair(self) -> None:
        text = """\
GADGET G {
    R 0
}

GADGET G {
    R 1
}

GADGET G {
    R 2
}
"""
        with pytest.raises(SyntaxError) as excinfo:
            parse(text)
        message = str(excinfo.value)
        # Must report the first redefinition (line 5), not a later one.
        assert "line 1" in message and "line 5" in message
        assert "line 9" not in message

    def test_error_message_includes_line_numbers(self) -> None:
        text = """\
GADGET A {
    R 0
}

GADGET A {
    R 0
}
"""
        with pytest.raises(SyntaxError) as excinfo:
            parse(text)
        message = str(excinfo.value)
        assert "line 1" in message
        assert "line 5" in message

    def test_unique_names_parse_successfully(self) -> None:
        text = """\
GADGET A {
    R 0
}

GADGET B {
    R 0
}
"""
        result = parse(text)
        assert len(result.definitions) == 2


# ── Multi-file duplicates (via parse_file with IMPORT) ───────────────


class TestImportDuplicates:
    def test_duplicate_gadget_across_imports_raises(self, tmp_path: Path) -> None:
        (tmp_path / "lib.deq").write_text(
            "GADGET Shared {\n    R 0\n}\n", encoding="utf-8"
        )
        main = tmp_path / "main.deq"
        main.write_text(
            'IMPORT "lib.deq"\n\nGADGET Shared {\n    R 0\n}\n',
            encoding="utf-8",
        )
        with pytest.raises(SyntaxError) as excinfo:
            parse_file(main)
        message = str(excinfo.value)
        assert "GADGET 'Shared' is defined more than once" in message
        # Cross-file dups should include both source paths.
        assert "lib.deq" in message
        assert "main.deq" in message

    def test_unique_definitions_across_imports_parse(self, tmp_path: Path) -> None:
        (tmp_path / "lib.deq").write_text(
            "GADGET Helper {\n    R 0\n}\n", encoding="utf-8"
        )
        main = tmp_path / "main.deq"
        main.write_text(
            'IMPORT "lib.deq"\n\nGADGET Main {\n    R 0\n}\n',
            encoding="utf-8",
        )
        result = parse_file(main)
        assert len(result.definitions) == 2
        names = {d.name for d in result.definitions}
        assert names == {"Helper", "Main"}
