import json
from pathlib import Path
import tempfile
import unittest

import yaml

from format_gadget_yaml import process_file


class GadgetFormatterTests(unittest.TestCase):
    def test_support_metadata_retains_scalar_types_and_keys(self):
        values = ["true", "false", "null", "007", "1.0", "yes", "a\nb", '"quoted"', "", True, False, None, 7]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.gadget.yaml"
            for value in values:
                with self.subTest(value=value):
                    expected = {
                        "circuit": {"format": "stim", "source": "I 0\n"},
                        "metadata": {"support": [{"true": value}, {"007": value}]},
                    }
                    path.write_text(yaml.safe_dump(expected), encoding="utf-8")
                    process_file(path, max_items=8, max_line=110, write=True)
                    actual = yaml.safe_load(path.read_text(encoding="utf-8"))
                    self.assertEqual(json.dumps(actual, sort_keys=True), json.dumps(expected, sort_keys=True))

    def test_check_mode_does_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.gadget.yaml"
            original = 'metadata:\n  support:\n    - label: "true"\n'
            path.write_text(original, encoding="utf-8")
            self.assertTrue(process_file(path, max_items=8, max_line=110, write=False))
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_preferred_layout_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.gadget.yaml"
            expected = {"checks": [["circuit.readouts[0]", 1]], "metadata": {"support": [{"label": "007"}]}}
            path.write_text(yaml.safe_dump(expected), encoding="utf-8")
            self.assertTrue(process_file(path, max_items=8, max_line=110, write=True))
            self.assertIn('support: [{label:', path.read_text(encoding="utf-8"))
            self.assertFalse(process_file(path, max_items=8, max_line=110, write=True))
            self.assertEqual(yaml.safe_load(path.read_text(encoding="utf-8")), expected)


if __name__ == "__main__":
    unittest.main()