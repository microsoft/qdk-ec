"""
Minimal setup.py — only needed for the custom BuildPyWithProto command.
All metadata lives in pyproject.toml.
"""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyWithProto(build_py):  # type: ignore[misc]
    """Custom build_py that compiles .proto files before building.

    This ensures the generated _pb2.py / _pb2_grpc.py stubs match the
    user's installed grpcio-tools and protobuf versions.
    """

    def run(self) -> None:
        print("BuildPyWithProto: starting proto compilation...")
        # Copy .proto source files into the package so they ship with the
        # sdist and are available at build time.
        proto_src_dir = Path(__file__).parent / "proto"
        bundled_dir = Path(__file__).parent / "deq" / "proto" / "proto_src"
        if proto_src_dir.is_dir() and any(proto_src_dir.glob("*.proto")):
            bundled_dir.mkdir(parents=True, exist_ok=True)
            for proto_file in proto_src_dir.glob("*.proto"):
                shutil.copy2(proto_file, bundled_dir / proto_file.name)

        # Generate protobuf/gRPC Python stubs
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "compile", Path(__file__).parent / "deq" / "proto" / "compile.py"
        )
        assert spec and spec.loader
        compile_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compile_mod)
        compile_mod.compile_protos()

        super().run()


setup(
    cmdclass={"build_py": BuildPyWithProto},
)
