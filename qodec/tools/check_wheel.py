"""Install and verify one wheel outside the source checkout."""

from email.parser import BytesParser
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv
from zipfile import ZipFile


PACKAGE_PROBE = """
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import qodec
package = Path(qodec.__file__).resolve().parent
assert package.is_relative_to(root), package
assert qodec.__version__ == sys.argv[2]
assert (package / 'py.typed').is_file()
assert (package / '__init__.pyi').is_file()
assert not hasattr(qodec._native, '_test_native_calls')
assert not hasattr(qodec._native, '_test_register_native_empty')
assert not hasattr(qodec._native, '_test_node_snapshots')
protocol = qodec.Qodec.loads('qodec.yaml: {schema_version: 1, layers: []}')
assert qodec.Qodec.loads(protocol.dumps()) == protocol
code = qodec.Code('test', stabilizers=['Z_0'], x=[], z=[])
assert code.physical_qubit_count == 1
print('Verified wheel:', qodec.__version__, package)
"""


def check_wheel(directory: Path) -> None:
    wheels = list(directory.resolve().glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected one wheel in {directory}, found {len(wheels)}")
    wheel = wheels[0]
    with ZipFile(wheel) as archive:
        metadata_name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        if metadata["Name"] != "qodec":
            raise ValueError(f"Expected qodec metadata in {wheel}")
        version = metadata["Version"]
    with tempfile.TemporaryDirectory(prefix="qodec-wheel-") as temporary:
        root = Path(temporary)
        environment = root / "env"
        venv.create(environment, with_pip=False)
        interpreter = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [sys.executable, "-m", "pip", "--python", str(interpreter), "install", "--no-deps", str(wheel)],
            cwd=root, check=True,
        )
        subprocess.run(
            [str(interpreter), "-I", "-c", PACKAGE_PROBE, str(environment), version],
            cwd=root, check=True,
        )


if __name__ == "__main__":
    check_wheel(Path(sys.argv[1]))