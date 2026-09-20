"""Examples reject all audit errors and all but documented unsupported checks."""

from collections import Counter
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import yaml
import qdk.ec
import qodec

from conftest import EXAMPLE_MANIFESTS, EXAMPLES_DIR

INTERLEAVED_ACTIONS = "NotImplementedError: readout verification of interleaved logical actions is not supported"


def _unsupported_rotation(instruction: str) -> str:
    return (
        f"TypeError: unrecognized action type 'Rotate' in instruction '{instruction}'"
    )


def _unsupported_step(index: int, step: str = "Rotate") -> tuple[str, str]:
    return (
        "gadget/unsupported-action-step",
        f"implements.action[{index}] ({step}) is not supported by the action verifier",
    )


UNSUPPORTED_WARNINGS = {
    "c422-c832-arch": {
        **{_unsupported_step(index): 1 for index in range(7)},
        ("gadget/incomplete-output-frame", _unsupported_rotation("T")): 1,
        ("gadget/readout-mismatch", INTERLEAVED_ACTIONS): 2,
    },
    "distillation-15": {
        _unsupported_step(1): 1,
        ("gadget/flag-mismatch", _unsupported_rotation("t_dg")): 1,
    },
    "reed-muller-15": {
        _unsupported_step(0): 1,
        ("gadget/check-mismatch", _unsupported_rotation("Tdg")): 14,
        ("gadget/incomplete-output-frame", _unsupported_rotation("Tdg")): 1,
    },
    "repetition3": {
        _unsupported_step(0): 1,
    },
}


def _unexpected_warnings(
    report: qdk.ec.Report, example: str
) -> Counter[tuple[str, str]]:
    actual = Counter(
        (
            warning.rule,
            (
                warning.summary
                if warning.rule == "gadget/unsupported-action-step"
                else warning.detail.splitlines()[-1]
            ),
        )
        for warning in report.warnings
    )
    return actual - Counter(UNSUPPORTED_WARNINGS.get(example, {}))


def test_every_example_has_an_audit_case() -> None:
    manifests = {
        path.relative_to(EXAMPLES_DIR).as_posix()
        for path in EXAMPLES_DIR.rglob("*qodec.yaml")
    }
    assert len(EXAMPLE_MANIFESTS) == len(set(EXAMPLE_MANIFESTS))
    assert set(EXAMPLE_MANIFESTS) == manifests


def _load_generator(filename: str) -> ModuleType:
    path = EXAMPLES_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_iceberg_builder_reproduces_the_committed_logical_layer() -> None:
    """The builder and the committed bundle agree above the physical layer.

    They differ below it on purpose: the bundle references the shared
    ``../stim.isa.yaml`` while the builder emits its own compact physical set.
    """
    module = _load_generator("iceberg/iceberg.py")
    built = module.build_iceberg(4)
    committed = qodec.Qodec.load(EXAMPLES_DIR / "iceberg" / "iceberg.qodec.yaml")

    def structure(instruction_set: qodec.InstructionSet) -> object:
        # Prose differs: the committed bundle spells out the k = 4 case.
        return (
            instruction_set.name,
            [(block.name, block.encodes) for block in instruction_set.blocks],
            {
                mnemonic: (
                    [operand.block for operand in instruction.inputs],
                    [operand.block for operand in instruction.outputs],
                    instruction.flags,
                    [str(step) for step in instruction.action],
                )
                for mnemonic, instruction in instruction_set.instructions.items()
            },
        )

    assert structure(built.layers[0].instruction_set) == structure(committed.layers[0].instruction_set)
    assert sorted(built.layers[0].gadgets) == sorted(committed.layers[0].gadgets)
    for name, code in built.codes.items():
        reference = committed.codes[name]
        assert (code.stabilizers, code.x, code.z) == (reference.stabilizers, reference.x, reference.z)


@pytest.mark.parametrize(("example", "generator", "function", "arguments"), [
    ("surface", "surface.py", "build_surface_code", (3,)),
    ("honeycomb", "generate_honeycomb.py", "bundle", ()),
])
def test_generators_reproduce_committed_artifact_values(
    example: str, generator: str, function: str, arguments: tuple[int, ...]
) -> None:
    module = _load_generator(f"{example}/{generator}")
    generated = getattr(module, function)(*arguments)
    committed = (EXAMPLES_DIR / example / f"{example}.qodec.yaml").read_text(encoding="utf-8")
    expected = {key: value for document in yaml.safe_load_all(committed) for key, value in document.items()}
    actual = {key: value for document in yaml.safe_load_all(generated) for key, value in document.items()}
    assert actual == expected


@pytest.mark.parametrize("distance", [2, 3, 5])
def test_generated_surface_distances_load(distance: int, tmp_path: Path) -> None:
    generator = _load_generator("surface/surface.py")
    (tmp_path / "stim.isa.yaml").write_text((EXAMPLES_DIR / "stim.isa.yaml").read_text(), encoding="utf-8")
    directory = tmp_path / "surface"
    directory.mkdir()
    manifest = directory / "surface.qodec.yaml"
    manifest.write_text(generator.build_surface_code(distance), encoding="utf-8")
    protocol = qodec.Qodec.load(manifest)
    assert len(protocol.codes["surface"].stabilizers) == distance * distance - 1
    assert protocol.codes["surface"].logical_count == 1
    assert ("merged" in protocol.codes) == (distance == 3)


@pytest.mark.parametrize(
    "manifest", EXAMPLE_MANIFESTS, ids=lambda manifest: Path(manifest).parent.name
)
def test_example_audits_clean(manifest: str) -> None:
    protocol = qodec.Qodec.load(EXAMPLES_DIR / manifest)
    report = qdk.ec.audit(protocol)
    assert report.ok, str(report)
    assert not _unexpected_warnings(report, Path(manifest).parent.name), str(report)


def test_missing_output_relations_are_not_allowed_warnings() -> None:
    warning = qdk.ec.Diagnostic(
        "gadget/incomplete-output-frame",
        qdk.ec.Diagnostic.Severity.WARNING,
        "Missing output relation",
        "layers[0].gadgets['rotate_z']",
        "No noiseless relation was derived.",
    )
    assert _unexpected_warnings(qdk.ec.Report((warning,)), "repetition3")


def test_new_unsupported_warning_counts_are_rejected() -> None:
    rule, summary = _unsupported_step(0)
    warning = qdk.ec.Diagnostic(
        rule,
        qdk.ec.Diagnostic.Severity.WARNING,
        summary,
        "gadget",
        "Logical action not checked.",
    )
    assert not _unexpected_warnings(qdk.ec.Report((warning,)), "repetition3")
    assert _unexpected_warnings(qdk.ec.Report((warning, warning)), "repetition3")
    assert _unexpected_warnings(qdk.ec.Report((warning,)), "steane")
