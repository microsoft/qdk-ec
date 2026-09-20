from html.parser import HTMLParser
from pathlib import Path
import unittest

from sphinx.util.inventory import InventoryFile

import qodec


HTML = Path(__file__).resolve().parents[3] / "target" / "python-docs" / "html"


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


class DocumentationTests(unittest.TestCase):
    def test_read_the_docs_theme(self) -> None:
        for page in ("index.html", "usage.html", "autoapi/qodec/index.html"):
            with self.subTest(page=page):
                source = (HTML / page).read_text()
                self.assertIn("_static/css/theme.css", source)
                self.assertIn("_static/qodec.css", source)
                self.assertIn('class="wy-nav-side"', source)
                self.assertIn('class="wy-nav-content"', source)
                self.assertNotIn("alabaster.css", source)
                self.assertNotIn("Copyright .", " ".join(source.split()))
        self.assertTrue((HTML / "_static/qodec.css").is_file())

    def test_public_top_level_classes(self) -> None:
        with (HTML / "objects.inv").open("rb") as stream:
            inventory = InventoryFile.load(stream, "", lambda base, path: path)
        documented = {
            name
            for kind in ("py:class", "py:exception")
            for name in inventory[kind]
            if name.count(".") == 1
        }
        expected = {
            f"qodec.{name}"
            for name in qodec.__all__
            if isinstance(getattr(qodec, name), type)
        }
        self.assertEqual(documented, expected)

    def test_class_reexports_have_one_documented_home(self) -> None:
        with (HTML / "objects.inv").open("rb") as stream:
            inventory = InventoryFile.load(stream, "", lambda base, path: path)
        documented = set(inventory["py:class"]) | set(inventory["py:exception"])
        for module in (qodec, qodec.actions, qodec.codes, qodec.gadgets, qodec.instructions):
            for name in module.__all__:
                value = getattr(module, name)
                if not isinstance(value, type):
                    continue
                canonical = f"{value.__module__}.{value.__qualname__}"
                exported = f"{module.__name__}.{name}"
                with self.subTest(exported=exported):
                    self.assertIn(canonical, documented)
                    if exported != canonical:
                        self.assertNotIn(exported, documented)

    def test_typed_constructors_and_stub_prose(self) -> None:
        source = (HTML / "autoapi/qodec/index.html").read_text()
        parser = _Text()
        parser.feed(source)
        text = " ".join(" ".join(parser.parts).split())
        self.assertIn("metadata", text)
        self.assertIn("Mapping", text)
        self.assertIn("stabilizers", text)
        self.assertEqual(text.count("Build a qodec from already-resolved components."), 1)
        self.assertIn('id="qodec.Code.__new__"', source)
        self.assertIn('id="qodec.Instruction.observe_count"', source)

    def test_reference_expansion(self) -> None:
        with (HTML / "objects.inv").open("rb") as stream:
            inventory = InventoryFile.load(stream, "", lambda base, path: path)
        self.assertIn("qodec.Reference.expand", inventory["py:method"])
        self.assertNotIn("qodec.gadgets.Reference", inventory["py:class"])
        for name in ("Field", "Key", "Index", "Slice", "Union"):
            self.assertIn(f"qodec.Reference.{name}", inventory["py:class"])
        self.assertIn("qodec.gadgets.Circuit.__new__", inventory["py:method"])
        parser = _Text()
        parser.feed((HTML / "autoapi/qodec/index.html").read_text())
        text = " ".join(" ".join(parser.parts).split())
        self.assertIn("Expand the final index selector, preserving order and duplicates.", text)

    def test_type_only_aliases(self) -> None:
        parser = _Text()
        parser.feed((HTML / "autoapi/qodec/index.html").read_text())
        text = " ".join(" ".join(parser.parts).split())
        aliases = {
            "Action": "an instruction's action step.",
            "Metadata": "a dictionary of annotations.",
            "PauliLike": "the inputs accepted for a Pauli.",
            "PauliString": "a Pauli operator string, such",
        }
        with (HTML / "objects.inv").open("rb") as stream:
            inventory = InventoryFile.load(stream, "", lambda base, path: path)
        type_only = {
            name.removeprefix("qodec.")
            for name in inventory["py:type"]
            if name.count(".") == 1 and not hasattr(qodec, name.removeprefix("qodec."))
        }
        self.assertEqual(type_only, set(aliases))
        for name, description in aliases.items():
            self.assertFalse(hasattr(qodec, name))
            self.assertIn(f"Type-only alias, not available at runtime: {description}", text)
        self.assertIn("Stabilize|Clifford|Pauli|Observe|Rotate", "".join(text.split()))

    def test_gadget_reference_examples(self) -> None:
        source = (HTML / "autoapi/qodec/index.html").read_text()
        self.assertNotIn("in.block.stabilizers", source)
        self.assertNotIn("out.block.z", source)
        for path in ("in[0].stabilizers[0]", "out[0].z[1]"):
            self.assertEqual(qodec.Reference(path).path, path)


if __name__ == "__main__":
    unittest.main()