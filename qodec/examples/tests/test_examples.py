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
    for name, gadget in built.layers[0].gadgets.items():
        reference = committed.layers[0].gadgets[name]
        assert gadget.circuit.source == reference.circuit.source
        assert gadget.checks == reference.checks
        assert gadget.readouts == reference.readouts


@pytest.mark.parametrize("k", [2, 4, 6])
def test_iceberg_gadgets_preserve_code_distance(k: int) -> None:
    protocol = _load_generator("iceberg/iceberg.py").build_iceberg(k)
    report = qdk.ec.audit(protocol)
    assert report.ok and not report.warnings, str(report)
    layer = protocol.layers[0]
    assert {
        name: tuple(instruction.flags)
        for name, instruction in layer.instruction_set.instructions.items()
    } == {
        "prepare_z_all": ("hook_x",),
        "idle": ("detected_x", "detected_z"),
        "measure_z_all": (),
    }
    assert qdk.ec.CodeProfile(layer.codes["iceberg"]).distance() == 2
    for name, gadget in layer.gadgets.items():
        result = qdk.ec.GadgetProfile(gadget).distance()
        assert result == 2, f"{name}: distance {result}; witness: {result.witness}"


@pytest.mark.parametrize("mnemonic", ["prepare_z", "prepare_x"])
def test_steane_preparation_uses_one_verification_ancilla(mnemonic: str) -> None:
    protocol = qodec.Qodec.load(EXAMPLES_DIR / "steane" / "steane.qodec.yaml")
    circuit = protocol.layers[0].gadgets[mnemonic].circuit
    assert len(circuit.blocks) == 8
    assert len(circuit.readouts) == 1


def test_steane_idle_names_preserve_the_same_flagged_round(tmp_path: Path) -> None:
    protocol = qodec.Qodec.load(EXAMPLES_DIR / "steane" / "steane.qodec.yaml")
    reloaded = qodec.Qodec.load(protocol.save(tmp_path, single_file=True))
    for candidate in (protocol, reloaded):
        layer = candidate.layers[0]
        assert set(layer.instruction_set.instructions) == {
            "prepare_z",
            "prepare_x",
            "idle",
            "idle_ft",
            "h",
            "cnot",
            "measure_z",
            "measure_x",
        }
        idle = layer.gadgets["idle"]
        idle_ft = layer.gadgets["idle_ft"]
        for name in ("idle", "idle_ft"):
            instruction = layer.gadgets[name].implements
            assert instruction.mnemonic == name
            assert not instruction.flags
        assert idle.circuit.source == idle_ft.circuit.source
        assert idle.checks == idle_ft.checks
        assert idle.readouts == idle_ft.readouts


def test_teleportation_embeds_the_verified_c832_preparation() -> None:
    protocol = qodec.Qodec.load(EXAMPLES_DIR / "c422-c832-arch" / "qodec.yaml")
    gadgets = protocol.layers[0].gadgets
    preparation = gadgets["prepare_x_all_c832"].circuit.calls()
    for call in preparation:
        call.operands = [int(qubit) + 4 for qubit in call.operands]
    teleportation = gadgets["teleport_c422_to_c832"].circuit.calls()
    assert teleportation[: len(preparation)] == preparation


@pytest.mark.parametrize("mnemonic", ["teleport_c422_to_c832", "teleport_c832_to_c422"])
def test_teleportation_detects_every_single_recorded_bit_flip(mnemonic: str) -> None:
    protocol = qodec.Qodec.load(EXAMPLES_DIR / "c422-c832-arch" / "qodec.yaml")
    gadget = protocol.layers[0].gadgets[mnemonic]
    faults = [
        qdk.ec.FaultEvent.after(i, readout_flips=0)
        for i, call in enumerate(gadget.circuit.calls())
        if call.mnemonic == "M"
    ]
    assert len(faults) == len(gadget.circuit.readouts)
    flag_readouts = {qodec.Reference(f"readouts[{i}]") for i, readout in enumerate(gadget.readouts) if readout.is_flag}
    effects = qdk.ec.GadgetProfile(gadget).effects_of(faults)
    for fault, effect in zip(faults, effects, strict=True):
        assert effect.checks or flag_readouts.intersection(effect.readouts), str(fault)


@pytest.mark.parametrize("color", ["red", "green", "blue"])
def test_honeycomb_rounds_use_one_ancilla_and_two_cnots_per_edge(color: str) -> None:
    protocol = qodec.Qodec.load(EXAMPLES_DIR / "honeycomb" / "honeycomb.qodec.yaml")
    circuit = protocol.layers[0].gadgets[f"round_{color}"].circuit
    assert set(circuit.blocks) == {str(qubit) for qubit in range(7)}
    assert len(circuit.readouts) == 3
    assert sum(call.mnemonic == "CX" for call in circuit.calls()) == 6


def test_reed_muller_idle_reuses_two_ancillas_and_preserves_instruction_names() -> None:
    protocol = qodec.Qodec.load(EXAMPLES_DIR / "reed-muller-15" / "reed-muller-15.qodec.yaml")
    layer = protocol.layers[0]
    assert set(layer.instruction_set.instructions) == {
        "prepare_z",
        "prepare_x",
        "idle",
        "t",
        "measure_z",
        "measure_x",
    }
    gadget = layer.gadgets["idle"]
    assert all(not instruction.flags for instruction in layer.instruction_set.instructions.values())
    assert set(gadget.circuit.blocks) == {str(qubit) for qubit in range(17)}
    assert len(gadget.circuit.readouts) == 28
    assert sum(call.mnemonic == "CX" for call in gadget.circuit.calls()) == 100


@pytest.mark.parametrize(
    ("example", "mnemonic", "flag_records"),
    [
        ("reed-muller-15", "idle", tuple(range(1, 28, 2))),
        ("steane", "idle", tuple(range(1, 12, 2))),
        ("steane", "idle_ft", tuple(range(1, 12, 2))),
        ("iceberg", "idle", (2, 3)),
    ],
)
def test_syndrome_flag_measurements_are_decoder_checks(
    example: str, mnemonic: str, flag_records: tuple[int, ...]
) -> None:
    protocol = qodec.Qodec.load(EXAMPLES_DIR / example / f"{example}.qodec.yaml")
    gadget = protocol.layers[0].gadgets[mnemonic]
    measurements = [i for i, call in enumerate(gadget.circuit.calls()) if call.mnemonic == "M"]
    assert len(measurements) == len(gadget.circuit.readouts)
    faults = [qdk.ec.FaultEvent.after(measurements[record], readout_flips=0) for record in flag_records]
    effects = qdk.ec.GadgetProfile(gadget).effects_of(faults)
    for record, effect in zip(flag_records, effects, strict=True):
        expected = [
            qodec.Reference(f"checks[{i}]")
            for i, check in enumerate(gadget.checks)
            if check == (qodec.Reference(f"circuit.readouts[{record}]"),)
        ]
        assert len(expected) == 1
        assert list(effect) == expected


def test_reed_muller_plus_preparation_uses_the_published_verification_circuit() -> None:
    protocol = qodec.Qodec.load(EXAMPLES_DIR / "reed-muller-15" / "reed-muller-15.qodec.yaml")
    gadget = protocol.layers[0].gadgets["prepare_x"]
    calls = gadget.circuit.calls()
    assert set(gadget.circuit.blocks) == {str(qubit) for qubit in range(20)}
    assert len(gadget.circuit.readouts) == 5
    assert sum(call.mnemonic == "CX" for call in calls) == 42
    assert not gadget.implements.flags
    faults = [qdk.ec.FaultEvent.after(i, readout_flips=0) for i, call in enumerate(calls) if call.mnemonic == "M"]
    effects = qdk.ec.GadgetProfile(gadget).effects_of(faults)
    assert len(effects) == 5
    for i, effect in enumerate(effects):
        assert list(effect) == [qodec.Reference(f"checks[{i}]")]


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


@pytest.mark.parametrize("distance", [3, 5, 7])
def test_surface_memory_gadgets_preserve_code_distance(distance: int, tmp_path: Path) -> None:
    generator = _load_generator("surface/surface.py")
    (tmp_path / "stim.isa.yaml").write_text((EXAMPLES_DIR / "stim.isa.yaml").read_text(), encoding="utf-8")
    directory = tmp_path / "surface"
    directory.mkdir()
    manifest = directory / "surface.qodec.yaml"
    manifest.write_text(generator.build_surface_code(distance), encoding="utf-8")
    protocol = qodec.Qodec.load(manifest)
    report = qdk.ec.audit(protocol)
    assert report.ok and not report.warnings, str(report)
    for name in ("prepare_z", "prepare_x", "idle", "measure_z", "measure_x"):
        profile = qdk.ec.GadgetProfile(protocol.layers[0].gadgets[name])
        result = profile.distance()
        assert result == distance, f"{name}: distance {result}; witness: {result.witness}"
        assert len(result.witness.factors) == distance
        (effect,) = profile.effects_of([result.witness.product])
        assert not effect.checks
        assert effect.frames or effect.readouts


def test_surface_extraction_orders_orient_hooks_across_logical_strings() -> None:
    generator = _load_generator("surface/surface.py")
    corners = [0, 1, 7, 8]
    assert [line for line in generator._extract("X", corners, 49) if line.startswith("CX")] == [
        "CX 49 0", "CX 49 1", "CX 49 7", "CX 49 8"
    ]
    assert [line for line in generator._extract("Z", corners, 49) if line.startswith("CX")] == [
        "CX 0 49", "CX 7 49", "CX 1 49", "CX 8 49"
    ]
    assert corners == [0, 1, 7, 8]
    assert generator._extract("Z", [0, 7], 49) == [
        "R 49", "CX 0 49", "CX 7 49", "M 49"
    ]


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
