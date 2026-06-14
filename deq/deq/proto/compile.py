import pathlib

import grpc_tools
from grpc_tools import protoc as grpc_protoc


def find_proto_dir() -> pathlib.Path:
    """Locate the .proto source directory.

    Checks two locations in order:
    1. ``<repo_root>/proto/`` — used during development (editable installs,
       CI, ``make interface``).
    2. ``deq/proto/proto_src/`` — bundled .proto files shipped inside the
       package for sdist / wheel installs so that stubs can be regenerated
       against the user's installed grpcio-tools version.
    """
    this_dir = pathlib.Path(__file__).parent

    # Development: repo-root proto/ directory (two levels up from deq/proto/)
    repo_proto = this_dir.parent.parent / "proto"
    if repo_proto.is_dir() and any(repo_proto.glob("*.proto")):
        return repo_proto

    # Packaged: bundled proto sources inside the package
    bundled_proto = this_dir / "proto_src"
    if bundled_proto.is_dir() and any(bundled_proto.glob("*.proto")):
        return bundled_proto

    raise FileNotFoundError(
        "Cannot find .proto source files. Expected either "
        f"{repo_proto} or {bundled_proto} to contain *.proto files."
    )


def compile_protos(proto_dir: pathlib.Path | None = None,
                   output_dir: pathlib.Path | None = None) -> None:
    """Compile .proto files into Python protobuf and gRPC stubs.

    Args:
        proto_dir: Directory containing .proto source files.  Defaults to
            auto-detection via :func:`find_proto_dir`.
        output_dir: Directory to write generated ``_pb2.py`` / ``_pb2_grpc.py``
            files into.  Defaults to the ``deq/proto/`` package directory.
    """
    if proto_dir is None:
        proto_dir = find_proto_dir()
    if output_dir is None:
        output_dir = pathlib.Path(__file__).parent

    # grpc_tools needs access to Google well-known types
    grpc_tools_proto_dir = pathlib.Path(grpc_tools.__file__).parent / "_proto"

    stems: list[str] = []
    for file in sorted(proto_dir.iterdir()):
        if file.suffix == ".proto":
            print(f"Compiling {file.name}...")
            grpc_protoc.main(
                [
                    "grpc_tools.protoc",
                    f"--proto_path={proto_dir}",
                    f"--proto_path={grpc_tools_proto_dir}",
                    f"--python_out={output_dir}",
                    f"--pyi_out={output_dir}",
                    str(file),
                ]
            )
            # fix the type definition
            with open(output_dir / (file.stem + "_pb2.pyi"), "r") as f:
                content = f.read()
            content = content.replace("__slots__ =", "__slots__: list[str] =")
            with open(output_dir / (file.stem + "_pb2.pyi"), "w") as f:
                f.write(content)
            stems.append(file.stem)

    # Generate gRPC service stubs (_pb2_grpc.py)
    print("Generating gRPC service stubs...")
    for file in sorted(proto_dir.iterdir()):
        if file.suffix == ".proto":
            grpc_protoc.main(
                [
                    "grpc_tools.protoc",
                    f"--proto_path={proto_dir}",
                    f"--proto_path={grpc_tools_proto_dir}",
                    f"--grpc_python_out={output_dir}",
                    str(file),
                ]
            )
            print(f"  Generated gRPC stub for {file.name}")

    # Fix imports in generated files to use relative imports
    for stem in stems:
        for suffix in ["_pb2.py", "_pb2.pyi"]:
            file_path = output_dir / (stem + suffix)
            with open(file_path, "r") as f:
                content = f.read()
            for other_stem in stems:
                if other_stem != stem:
                    content = content.replace(
                        f"import {other_stem}_pb2 as",
                        f"from . import {other_stem}_pb2 as",
                    )
            with open(file_path, "w") as f:
                f.write(content)

    # Also fix imports in gRPC stub files (_pb2_grpc.py)
    for stem in stems:
        grpc_file_path = output_dir / (stem + "_pb2_grpc.py")
        if grpc_file_path.exists():
            with open(grpc_file_path, "r") as f:
                content = f.read()
            for other_stem in stems:
                content = content.replace(
                    f"import {other_stem}_pb2 as",
                    f"from . import {other_stem}_pb2 as",
                )
            with open(grpc_file_path, "w") as f:
                f.write(content)
            print(f"Fixed imports in {grpc_file_path.name}")


if __name__ == "__main__":
    compile_protos()
