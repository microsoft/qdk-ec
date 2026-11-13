"""Compile tutorial-generated .deq text into local, git-ignored Rust benchmark inputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zlib

from deq.proto.deq_jit_pb2 import JitLibrary

from lattice_surgery import compile_case, load_circuit

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def gzip_bytes(data: bytes) -> bytes:
    stream = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
        compressed.write(data)
    return stream.getvalue()


def compile_fixture(
    directory: Path, width: int, layers: int
) -> tuple[JitLibrary, dict]:
    configuration = compile_case(
        directory,
        distance=7,
        probability=0.003,
        width=width,
        layers=layers,
        basis="Z",
        seed=42,
        export_physical=False,
    )
    _, _, _, wiring = load_circuit(directory)
    library = JitLibrary.FromString((directory / "circuit.deq.jit").read_bytes())
    types = {definition.base.gtype: definition for definition in library.gadget_types}
    metadata = {
        "name": f"d7_w{width}_l{layers}",
        "width": width,
        "layers": layers,
        "epochs": len(set(configuration["epochs"].values())),
        "gadgets": len(library.program),
        "checks": sum(
            len(types[instruction.gadget.gtype].finished_checks)
            for instruction in library.program
        ),
        "errors": sum(
            len(types[instruction.gadget.gtype].errors)
            for instruction in library.program
        ),
        "source_sha256": configuration["source_sha256"],
        "connected_circuit_verification": wiring,
    }
    return library, metadata


def generate(output: Path, *, check: bool = False) -> dict:
    with TemporaryDirectory(prefix="deq-jit-fixture-") as temporary:
        root = Path(temporary)
        large, large_metadata = compile_fixture(root / "large", 1000, 50)
        small, small_metadata = compile_fixture(root / "small", 4, 2)
        if (
            large.gadget_types != small.gadget_types
            or large.port_types != small.port_types
        ):
            raise ValueError(
                "small and large programs do not share the same type library"
            )
        small_program = JitLibrary(program=small.program)
        payloads = {
            "lattice_surgery_d7_w1000_l50.jit.gz": large.SerializeToString(
                deterministic=True
            ),
            "lattice_surgery_d7_w4_l2.program.jit.gz": small_program.SerializeToString(
                deterministic=True
            ),
        }
        large_metadata["file"] = "lattice_surgery_d7_w1000_l50.jit.gz"
        small_metadata["file"] = "lattice_surgery_d7_w4_l2.program.jit.gz"
        cases = [small_metadata, large_metadata]
        for case in cases:
            data = payloads[case["file"]]
            case["protobuf_crc32"] = zlib.crc32(data)
            case["protobuf_sha256"] = hashlib.sha256(data).hexdigest()
        manifest = {
            "schema_version": 1,
            "distance": 7,
            "probability": 0.003,
            "basis": "Z",
            "seed": 42,
            "compiler_threads": 24,
            "maximum_mean_epoch_ms": 390,
            "scope": "Shared JIT compilation, deferred error resolution, and incremental collection only.",
            "excluded": "Fixture loading, library registration, input cloning, dependency-channel setup, controller, coordinator, decoder, simulation, and intentional sleeps.",
            "cases": cases,
        }
        generated = {name: gzip_bytes(data) for name, data in payloads.items()}
        for case, directory in [(large_metadata, root / "large"), (small_metadata, root / "small")]:
            generated[f"{case['name']}.deq"] = (directory / "circuit.deq").read_bytes()
        generated["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    if check:
        for name, data in generated.items():
            if not (output / name).is_file() or (output / name).read_bytes() != data:
                raise ValueError(
                    f"fixture differs from deterministic regeneration: {name}"
                )
    else:
        output.mkdir(parents=True, exist_ok=True)
        for name, data in generated.items():
            (output / name).write_bytes(data)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIXTURES)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(generate(arguments.output, check=arguments.check), indent=2))


if __name__ == "__main__":
    main()
