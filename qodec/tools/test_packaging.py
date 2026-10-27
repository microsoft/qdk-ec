"""Build and import release artifacts without changing the selected installation."""

from email.parser import BytesParser
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import unittest
from zipfile import ZipFile

from packaging.version import Version

from check_wheel import PACKAGE_PROBE


ROOT = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_qodec_release_group_has_matching_versions(self):
        version = tomllib.loads((ROOT / "Cargo.toml").read_text())["package"]["version"]
        for manifest in ("Cargo.toml", "bindings/python/Cargo.toml", "bindings/c/Cargo.toml"):
            with self.subTest(manifest=manifest):
                package = tomllib.loads((ROOT / manifest).read_text())["package"]
                self.assertEqual(package["version"], version)
                release = package["metadata"]["release"]
                self.assertEqual(release["shared-version"], "qodec")
                self.assertTrue(release["release"])
                self.assertFalse(release["tag"])

    def run_command(self, command, cwd):
        result = subprocess.run(
            command, cwd=cwd, text=True, capture_output=True,
            env=dict(
                os.environ,
                PYO3_PYTHON=sys.executable,
                PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
            ),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def assert_release_contents(self, names):
        self.assertEqual(len(names), len(set(names)), "duplicate archive paths")
        for name in names:
            parts = PurePosixPath(name).parts
            self.assertFalse(PurePosixPath(name).is_absolute(), name)
            self.assertNotIn("..", parts, name)
            self.assertNotIn(".vscode", parts, name)
            self.assertNotIn(".github", parts, name)
            self.assertNotIn("__pycache__", parts, name)
            self.assertFalse(name.endswith((".pyc", ".pyo")), name)
            self.assertNotIn("proposals", parts, name)
            self.assertNotIn("PRERELEASE.md", parts, name)
            self.assertNotIn("CODE_OF_CONDUCT.md", parts, name)
            self.assertNotIn("SECURITY.md", parts, name)
            self.assertNotIn("SUPPORT.md", parts, name)
            self.assertNotIn("vision.md", parts, name)
            self.assertNotIn("es-metadata.yml", parts, name)
            self.assertNotIn("azure-pipelines.yml", parts, name)
            self.assertNotIn("azure-pipelines-wheels.yml", parts, name)
            self.assertNotIn("patent", name.lower(), name)
            self.assertFalse(name.endswith("-review.html"), name)
            self.assertFalse(name.endswith(".tar.gz"), name)
            self.assertFalse(name.endswith("bindings/python/src/qodec/__init__.py"), name)

    def test_sdist_rebuilds_into_an_importable_wheel(self):
        with tempfile.TemporaryDirectory(prefix="qodec-packaging-") as temporary:
            root = Path(temporary)
            sdist = self.build_sdist(root)
            self.assert_sdist_contents(sdist)
            wheel = self.rebuild_wheel(root, sdist)
            self.assert_wheel_imports(root, wheel)
            self.run_command([sys.executable, str(ROOT / "tools/check_wheel.py"), str(root)], root)

    def test_native_parser_callbacks_in_an_extension_build(self):
        with tempfile.TemporaryDirectory(prefix="qodec-native-callbacks-") as temporary:
            root = Path(temporary)
            self.run_command([
                sys.executable, "-m", "maturin", "build", "--release", "--features", "test-support",
                "--manifest-path", str(ROOT / "bindings/python/Cargo.toml"), "--out", str(root),
            ], ROOT)
            with ZipFile(next(root.glob("*.whl"))) as archive:
                archive.extractall(root / "installed")
            self.assert_native_callbacks(root)

    def assert_native_callbacks(self, root):
        setup = """
import sys
sys.path.insert(0, sys.argv[1])
import atexit
import threading
import qodec
from qodec import _native
from qodec.instructions import InstructionCall
main_thread = threading.get_ident()
def parser(source, target):
    assert threading.get_ident() != main_thread
    assert source == 'authored source' and target.name == 'test'
    return [InstructionCall('idle', operands=[7])]
callbacks = []
register_at_exit = atexit.register
try:
    atexit.register = lambda callback: callbacks.append(callback) or callback
    qodec.register(parser, format='native-test')
finally:
    atexit.register = register_at_exit
"""
        cases = {
            "live navigation without snapshots": """
protocol = qodec.Qodec.loads('entry: {layers: [{instruction_set: target}]}\\n---\\ntarget: {name: logical, blocks: {}, instructions: [{mnemonic: M, description: "", action: [{observe: Z_0, if: [reject]}]}]}')
_native._test_node_snapshots()
node = protocol.resolve('layers[0].instruction_set.instructions["M"].action[0]')
assert node.resolve('condition.invert').value(bool) is False
assert node.resolve('observables').sequence_nodes()[0].value(str) == 'Z_0'
assert node.resolve('observables').value(tuple) == ('Z_0',)
assert node.resolve('condition').value() is not None
assert len(protocol.resolve('layers').sequence_nodes()) == 1
assert _native._test_node_snapshots() == 0
assert node.source_location is None
assert _native._test_node_snapshots() == 1
""",
            "native thread": "assert _native._test_native_calls('native-test')[0].operands == [7]",
            "native replacement": """
assert _native._test_native_calls('native-test')[0].operands == [7]
_native._test_register_native_empty('native-test')
assert _native._test_native_calls('native-test') == []
qodec.register(parser, format='native-test')
assert _native._test_native_calls('native-test')[0].operands == [7]
""",
            "integer arguments": """
for value in (-(1 << 63), -1, 0, 1, (1 << 63) - 1):
    qodec.register(
        lambda source, target: [InstructionCall('idle', arguments={'value': value})],
        format='native-integer-test',
    )
    _native._test_native_calls('native-integer-test', expected_integer=value)
""",
            "closed parser": """
assert _native._test_native_calls('native-test')[0].operands == [7]
callbacks[-1]()
try:
    _native._test_native_calls('native-test')
except ValueError as error:
    assert 'interpreter has shut down' in str(error)
else:
    raise AssertionError('closed callback must fail')
""",
        }
        for name, assertions in cases.items():
            with self.subTest(name=name):
                self.run_command([sys.executable, "-I", "-S", "-c", setup + assertions, str(root / "installed")], root)

    def build_sdist(self, root):
        inventory = self.run_command(
            ["cargo", "package", "--list", "--allow-dirty"], ROOT,
        ).splitlines()
        self.assert_release_contents(inventory)
        self.run_command([
            sys.executable, "-m", "maturin", "sdist", "--manifest-path",
            str(ROOT / "bindings/python/Cargo.toml"), "--out", str(root),
        ], ROOT)
        archives = list(root.glob("*.tar.gz"))
        self.assertEqual(len(archives), 1)
        return archives[0]

    def assert_sdist_contents(self, sdist):
        with tarfile.open(sdist) as archive:
            names = archive.getnames()
            self.assert_release_contents(names)
            prefix = names[0].split("/")[0]
            for filename in ["qodec/README.md", "qodec/LICENSE", "README-python.md", "pyproject.toml"]:
                self.assertIn(f"{prefix}/{filename}", names)

    def rebuild_wheel(self, root, sdist):
        self.run_command([
            sys.executable, "-m", "pip", "wheel", "--no-deps",
            "--no-build-isolation", "--no-cache-dir", "--wheel-dir", str(root), str(sdist),
        ], root)
        wheels = list(root.glob("*.whl"))
        self.assertEqual(len(wheels), 1)
        return wheels[0]

    def assert_wheel_imports(self, root, wheel):
        package_version = tomllib.loads((ROOT / "Cargo.toml").read_text())["package"]["version"]
        extracted = root / "installed"
        with ZipFile(wheel) as archive:
            names = archive.namelist()
            self.assert_release_contents(names)
            self.assertIn("qodec/py.typed", names)
            self.assertIn("qodec/__init__.pyi", names)
            metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
            metadata = BytesParser().parsebytes(archive.read(metadata_name))
            self.assertEqual(metadata["Name"], "qodec")
            self.assertEqual(Version(metadata["Version"]), Version(package_version))
            license_text = (ROOT / "LICENSE").read_text(encoding="utf-8").strip()
            license_files = metadata.get_all("License-File", [])
            self.assertEqual(len(license_files), 1)
            license_path = f"{metadata_name.rsplit('/', 1)[0]}/licenses/{license_files[0]}"
            self.assertEqual(archive.read(license_path).decode("utf-8").strip().splitlines(), license_text.splitlines())
            description = metadata.get_payload()
            readme = (ROOT / "bindings/python/README-python.md").read_text(encoding="utf-8")
            self.assertEqual(description.strip().splitlines(), readme.strip().splitlines())
            self.assertIn("https://github.com/microsoft/qdk-ec/blob/main/qodec/bindings/python/docs/index.rst", description)
            archive.extractall(extracted)
        self.run_command([
            sys.executable, "-I", "-S", "-c", PACKAGE_PROBE, str(extracted), metadata["Version"],
        ], root)


if __name__ == "__main__":
    unittest.main()