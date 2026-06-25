"""Tests for IMPORT statement parsing and file resolution."""

from pathlib import Path

import pytest

from deq.circuit.parser import parse, parse_file, render_and_parse_files
from deq.circuit.model import (
    ImportStatement,
    CodeDefinition,
    GadgetDefinition,
)

FIXTURES = Path(__file__).parent / "fixtures" / "imports"


# ── parse() — AST-level import tests (no resolution) ────────────────


class TestParseImportStatements:
    """Test that parse() builds ImportStatement nodes without resolving them."""

    def test_single_import(self) -> None:
        result = parse('IMPORT "foo.deq"\n')
        assert len(result.imports) == 1
        assert result.imports[0] == ImportStatement(path="foo.deq")

    def test_multiple_imports(self) -> None:
        text = 'IMPORT "a.deq"\nIMPORT "b.deq"\n'
        result = parse(text)
        assert len(result.imports) == 2
        assert result.imports[0].path == "a.deq"
        assert result.imports[1].path == "b.deq"

    def test_import_with_subdirectory(self) -> None:
        result = parse('IMPORT "codes/rep.deq"\n')
        assert result.imports[0].path == "codes/rep.deq"

    def test_import_does_not_create_definitions(self) -> None:
        result = parse('IMPORT "foo.deq"\n')
        assert len(result.definitions) == 0

    def test_import_before_definitions(self) -> None:
        text = """\
IMPORT "codes.deq"

GADGET G {
    R 0
}
"""
        result = parse(text)
        assert len(result.imports) == 1
        assert len(result.definitions) == 1
        assert isinstance(result.definitions[0], GadgetDefinition)

    def test_import_str(self) -> None:
        imp = ImportStatement(path="foo.deq")
        assert str(imp) == 'IMPORT "foo.deq"'


# ── parse_file() — import resolution ────────────────────────────────


class TestParseFileImportResolution:
    """Test parse_file() with actual fixture files."""

    def test_simple_file_no_imports(self) -> None:
        result = parse_file(FIXTURES / "codes.deq")
        assert len(result.definitions) == 1
        assert isinstance(result.definitions[0], CodeDefinition)
        assert result.definitions[0].name == "RepetitionCode"

    def test_single_import(self) -> None:
        result = parse_file(FIXTURES / "gadgets.deq")
        # codes.deq defines 1, gadgets.deq defines 1 = 2 total
        assert len(result.definitions) == 2
        names = [d.name for d in result.definitions]
        assert names == ["RepetitionCode", "PrepareZ"]

    def test_transitive_imports(self) -> None:
        result = parse_file(FIXTURES / "main.deq")
        # codes.deq (1) + gadgets.deq (1) + main.deq (1) = 3
        names = [d.name for d in result.definitions]
        assert names == ["RepetitionCode", "PrepareZ", "Main"]

    def test_include_guard_no_duplicates(self) -> None:
        """main.deq imports both codes.deq and gadgets.deq.
        gadgets.deq also imports codes.deq.
        RepetitionCode should appear only once."""
        result = parse_file(FIXTURES / "main.deq")
        rep_count = sum(1 for d in result.definitions if d.name == "RepetitionCode")
        assert rep_count == 1

    def test_source_file_tracking(self) -> None:
        result = parse_file(FIXTURES / "main.deq")
        sources = {d.name: d.source_file for d in result.definitions}
        codes_path = str((FIXTURES / "codes.deq").resolve())
        gadgets_path = str((FIXTURES / "gadgets.deq").resolve())
        main_path = str((FIXTURES / "main.deq").resolve())
        assert sources["RepetitionCode"] == codes_path
        assert sources["PrepareZ"] == gadgets_path
        assert sources["Main"] == main_path

    def test_deqfile_source_file(self) -> None:
        result = parse_file(FIXTURES / "main.deq")
        assert result.source_file == str((FIXTURES / "main.deq").resolve())

    def test_circular_imports(self) -> None:
        """Circular imports should not cause infinite recursion."""
        result = parse_file(FIXTURES / "circular_a.deq")
        names = [d.name for d in result.definitions]
        # circular_a imports circular_b which imports circular_a (skipped)
        # So: B (from circular_b) then A (from circular_a)
        assert "A" in names
        assert "B" in names
        assert len(result.definitions) == 2

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_file(FIXTURES / "nonexistent.deq")

    def test_missing_import_target(self, tmp_path: Path) -> None:
        """An IMPORT pointing to a file that doesn't exist."""
        main = tmp_path / "main.deq"
        main.write_text('IMPORT "does_not_exist.deq"\n')
        with pytest.raises(FileNotFoundError, match="does_not_exist.deq"):
            parse_file(main)

    def test_import_order_depth_first(self) -> None:
        """Imports are resolved depth-first: imported defs come before importer's."""
        result = parse_file(FIXTURES / "gadgets.deq")
        names = [d.name for d in result.definitions]
        # codes.deq defs first, then gadgets.deq defs
        assert names.index("RepetitionCode") < names.index("PrepareZ")


# ── Edge cases ───────────────────────────────────────────────────────


class TestImportEdgeCases:
    def test_empty_file_with_import(self, tmp_path: Path) -> None:
        """Import a file that has no definitions."""
        empty = tmp_path / "empty.deq"
        empty.write_text("")
        main = tmp_path / "main.deq"
        main.write_text('IMPORT "empty.deq"\n\nGADGET G {\n    R 0\n}\n')
        result = parse_file(main)
        assert len(result.definitions) == 1
        assert result.definitions[0].name == "G"

    def test_diamond_import(self, tmp_path: Path) -> None:
        """Diamond dependency: A imports B and C, both import D."""
        d = tmp_path / "d.deq"
        d.write_text("GADGET D {\n    R 0\n}\n")
        b = tmp_path / "b.deq"
        b.write_text('IMPORT "d.deq"\n\nGADGET B {\n    R 1\n}\n')
        c = tmp_path / "c.deq"
        c.write_text('IMPORT "d.deq"\n\nGADGET C {\n    R 2\n}\n')
        a = tmp_path / "a.deq"
        a.write_text('IMPORT "b.deq"\nIMPORT "c.deq"\n\nGADGET A {\n    R 3\n}\n')

        result = parse_file(a)
        names = [defn.name for defn in result.definitions]
        # D should appear only once
        assert names.count("D") == 1
        # D before B, B before C, C before A (depth-first)
        assert names == ["D", "B", "C", "A"]


class TestRenderAndParseFiles:
    """Test the multi-file render_and_parse_files entry point."""

    def test_multiple_files_parsed(self, tmp_path: Path) -> None:
        a = tmp_path / "a.deq"
        a.write_text("GADGET A {\n    R 0\n}\n")
        b = tmp_path / "b.deq"
        b.write_text("GADGET B {\n    R 1\n}\n")
        result = render_and_parse_files([a, b])
        names = [defn.name for defn in result.definitions]
        assert "A" in names
        assert "B" in names

    def test_temp_file_cleaned_up(self, tmp_path: Path) -> None:
        """Regression: the virtual import temp file must be closed before
        parsing and removed afterward on every platform."""
        a = tmp_path / "a.deq"
        a.write_text("GADGET A {\n    R 0\n}\n")
        b = tmp_path / "b.deq"
        b.write_text("GADGET B {\n    R 1\n}\n")
        before = set(tmp_path.iterdir())
        render_and_parse_files([a, b])
        after = set(tmp_path.iterdir())
        assert before == after, f"leftover temp files: {after - before}"
