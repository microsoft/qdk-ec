"""CLI coverage for selecting inferred physical loss models."""

import pickle
from pathlib import Path

import pytest
from google.protobuf.json_format import MessageToDict

import deq.proto.deq_jit_pb2 as jit_pb
import deq.transpiler.loss.model_neutral_atom as neutral_atom_module
import deq.transpiler.loss.model_trapped_ion as trapped_ion_module
from deq.cli.annotate import annotate as annotate_file
from deq.cli.interpret import _load_library
from deq.cli.jit import transpile
from deq.cli.simulate import _resolve_jit_loss_config, _run_batch, simulate__ler
from deq.transpiler.loss import (
    NeutralAtomLossModel,
    NoLossModel,
    TrappedIonLossModel,
    create_loss_model,
)

_SOURCE = """
CODE Pair [[2,2,1]] {
    LOGICAL X0 Z0
    LOGICAL X1 Z1
}
GADGET G {
    INPUT Pair 0 1
    LOSS_ERROR(0.1) 0
    CZ 0 1
    M 0 1
    OUTPUT Pair 0 1
}
"""

_PROGRAM_SOURCE = """
CODE C [[1,1,1]] { LOGICAL X0 Z0 }
GADGET Prep {
    R 0
    LOSS_ERROR(0.1) 0
    OUTPUT C 0
}
GADGET Meas {
    INPUT C 0
    M 0
    READOUT M0
}
PROGRAM Run {
    Prep 0
    Meas 0
}
"""


def _write_user_loss_model(tmp_path: Path) -> Path:
    plugin = tmp_path / "user_loss.py"
    plugin.write_text(
        """
from deq.transpiler.loss import GateLossPolicy, QdkLossConfig
from deq.transpiler.loss.model_neutral_atom import NeutralAtomLossModel

class UserLossModel(NeutralAtomLossModel):
    config = QdkLossConfig(gate_policies=(
        ("cx", GateLossPolicy.PROPAGATE),
        ("cy", GateLossPolicy.SKIP),
        ("cz", GateLossPolicy.PROPAGATE),
        ("swap", GateLossPolicy.APPLY_ANYWAY),
    ))

def create_loss_model():
    return UserLossModel()
""",
        encoding="utf-8",
    )
    return plugin


def _transpile_with_loss_model(tmp_path: Path, loss_model: str):
    source = tmp_path / f"{loss_model}.deq"
    output = tmp_path / f"{loss_model}.deq.jit"
    source.write_text(_SOURCE, encoding="utf-8")
    transpile(
        str(source),
        out=str(output),
        jobs=1,
        loss_model=loss_model,
        skip_mako_warning=True,
    )
    return jit_pb.JitLibrary.FromString(output.read_bytes()).gadget_types[0]


def test_transpile_accepts_neutral_atom_loss_model(tmp_path: Path) -> None:
    gadget = _transpile_with_loss_model(tmp_path, "neutral-atom")
    (loss,) = gadget.base.loss_model.losses

    assert list(loss.loss_measurements) == [0]
    assert list(loss.source_errors)


def test_create_loss_model_returns_platform_model() -> None:
    assert isinstance(create_loss_model("neutral-atom"), NeutralAtomLossModel)
    assert isinstance(create_loss_model("trapped-ion"), TrappedIonLossModel)
    assert isinstance(create_loss_model("none"), NoLossModel)


@pytest.mark.parametrize(
    ("module", "expected_config"),
    [
        (neutral_atom_module, NeutralAtomLossModel.config),
        (trapped_ion_module, TrappedIonLossModel.config),
    ],
)
def test_builtin_loss_model_file_can_be_loaded_by_path(module, expected_config) -> None:
    model = create_loss_model(Path(module.__file__))

    assert model.config == expected_config
    assert model.native_gates == module.create_loss_model().native_gates


@pytest.mark.parametrize(
    ("module", "expected_config"),
    [
        (neutral_atom_module, NeutralAtomLossModel.config),
        (trapped_ion_module, TrappedIonLossModel.config),
    ],
)
def test_transpile_accepts_builtin_loss_model_file_path(
    tmp_path: Path,
    module,
    expected_config,
) -> None:
    source = tmp_path / "input.deq"
    output = tmp_path / "output.deq.jit"
    source.write_text(_SOURCE, encoding="utf-8")

    transpile(
        str(source),
        out=str(output),
        jobs=1,
        loss_model=module.__file__,
        skip_mako_warning=True,
    )
    library = jit_pb.JitLibrary.FromString(output.read_bytes())

    assert MessageToDict(library.metadata)["loss_strategy"] == (
        expected_config.to_json_object()
    )


def test_create_loss_model_loads_python_file(tmp_path: Path) -> None:
    plugin = _write_user_loss_model(tmp_path)

    model = create_loss_model(plugin)
    loaded_model = getattr(model, "_model")
    assert getattr(model, "_model") is loaded_model
    restored = pickle.loads(pickle.dumps(model))

    assert model.config.policy_for("cx") == "PROPAGATE"
    assert restored.config == model.config
    assert restored.native_gates == model.native_gates
    assert "_model" not in vars(restored)


def test_transpile_accepts_python_loss_model_file(tmp_path: Path) -> None:
    plugin = _write_user_loss_model(tmp_path)
    source = tmp_path / "input.deq"
    output = tmp_path / "output.deq.jit"
    source.write_text(
        _SOURCE + """
GADGET H {
    INPUT Pair 0 1
    LOSS_ERROR(0.2) 1
    CZ 0 1
    M 0 1
    OUTPUT Pair 0 1
}
""",
        encoding="utf-8",
    )

    transpile(
        str(source),
        out=str(output),
        jobs=2,
        loss_model=str(plugin),
        skip_mako_warning=True,
    )
    library = jit_pb.JitLibrary.FromString(output.read_bytes())

    assert MessageToDict(library.metadata)["loss_strategy"]["cz"] == "PROPAGATE"
    assert [
        list(gadget.base.loss_model.losses[0].loss_measurements)
        for gadget in library.gadget_types
    ] == [[0, 1], [0, 1]]


def test_create_loss_model_file_requires_factory(tmp_path: Path) -> None:
    plugin = tmp_path / "invalid.py"
    plugin.write_text("config = {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"callable create_loss_model\(\)"):
        create_loss_model(plugin)


def test_unknown_loss_model_lists_supported_names() -> None:
    with pytest.raises(
        ValueError,
        match="expected one of: neutral-atom, trapped-ion, none",
    ):
        create_loss_model("unknown")


def test_none_loss_model_leaves_gadgets_without_loss_metadata(
    tmp_path: Path,
) -> None:
    gadget = _transpile_with_loss_model(tmp_path, "none")

    assert not gadget.base.HasField("loss_model")
    assert not gadget.errors


def test_none_loss_model_ignores_declared_loss_statements(tmp_path: Path) -> None:
    source = tmp_path / "declared.deq"
    output = tmp_path / "declared.deq.jit"
    source.write_text(
        """
CODE C [[1,1,1]] { LOGICAL X0 Z0 }
GADGET G {
    INPUT C 0
    M 0
    OUTPUT C 0
    ERROR(0) LX0
    LOSS(0.1) SE0 M0
}
""",
        encoding="utf-8",
    )
    transpile(
        str(source),
        out=str(output),
        jobs=1,
        loss_model="none",
        skip_mako_warning=True,
    )
    gadget = jit_pb.JitLibrary.FromString(output.read_bytes()).gadget_types[0]

    assert not gadget.base.HasField("loss_model")


def test_none_loss_model_drops_loss_error_from_stim(tmp_path: Path) -> None:
    source = tmp_path / "program.deq"
    output = tmp_path / "program.deq.jit"
    source.write_text(_PROGRAM_SOURCE, encoding="utf-8")

    transpile(
        str(source),
        out=str(output),
        program="Run",
        jobs=1,
        loss_model="none",
        skip_mako_warning=True,
    )
    stim_text = (tmp_path / "program.stim").read_text(encoding="utf-8")

    assert "LOSS_ERROR" not in stim_text


def test_platform_loss_model_keeps_loss_error_in_stim(tmp_path: Path) -> None:
    source = tmp_path / "program.deq"
    output = tmp_path / "program.deq.jit"
    source.write_text(_PROGRAM_SOURCE, encoding="utf-8")

    transpile(
        str(source),
        out=str(output),
        program="Run",
        jobs=1,
        loss_model="neutral-atom",
        skip_mako_warning=True,
    )
    stim_text = (tmp_path / "program.stim").read_text(encoding="utf-8")

    assert "LOSS_ERROR(0.1)" in stim_text


def test_none_loss_model_annotates_without_loss_statements(tmp_path: Path) -> None:
    source = tmp_path / "input.deq"
    output = tmp_path / "annotated.deq"
    source.write_text(_SOURCE, encoding="utf-8")

    annotate_file(
        str(source),
        out=str(output),
        loss_model="none",
        skip_mako_warning=True,
    )
    annotated = output.read_text(encoding="utf-8")

    assert "LOSS(" not in annotated


def test_annotate_accepts_neutral_atom_loss_model(tmp_path: Path) -> None:
    source = tmp_path / "input.deq"
    output = tmp_path / "annotated.deq"
    source.write_text(_SOURCE, encoding="utf-8")

    annotate_file(
        str(source),
        out=str(output),
        loss_model="neutral-atom",
        skip_mako_warning=True,
    )

    loss_line = next(
        line.strip()
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("LOSS(0.1)")
    )
    assert "M0" in loss_line
    assert " SE" in loss_line


def test_transpile_records_trapped_ion_platform_config(tmp_path: Path) -> None:
    source = tmp_path / "trapped.deq"
    output = tmp_path / "trapped.deq.jit"
    source.write_text(_SOURCE, encoding="utf-8")

    transpile(
        str(source),
        out=str(output),
        jobs=1,
        loss_model="trapped-ion",
        skip_mako_warning=True,
    )
    library = jit_pb.JitLibrary.FromString(output.read_bytes())

    metadata = MessageToDict(library.metadata)
    assert metadata["loss_strategy"] == TrappedIonLossModel.config.to_json_object()


class _StopAfterBuild(Exception):
    pass


def test_simulate_passes_loss_model_to_source_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "simulate.deq"
    source.write_text(_SOURCE + "\nPROGRAM Run {}\n", encoding="utf-8")

    def stop_build(*args, **kwargs):
        assert isinstance(kwargs["loss_model"], NeutralAtomLossModel)
        raise _StopAfterBuild

    monkeypatch.setattr(
        "deq.transpiler.jit_library_builder.build_jit_library", stop_build
    )
    with pytest.raises(_StopAfterBuild):
        simulate__ler(
            str(source),
            program="Run",
            jobs=1,
            loss_model="neutral-atom",
            skip_mako_warning=True,
        )


@pytest.mark.parametrize(
    ("simulation_loss_model", "expected_config"),
    [
        (None, TrappedIonLossModel.config.to_json_object()),
        ('{"cx":"SKIP"}', {"cx": "SKIP"}),
        ("{}", {}),
    ],
)
def test_simulation_loss_json_overrides_decoder_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    simulation_loss_model: str | None,
    expected_config: dict[str, str],
) -> None:
    source = tmp_path / "simulate.deq"
    source.write_text(_SOURCE + "\nPROGRAM Run {}\n", encoding="utf-8")

    class StopPool:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> bool:
            return False

        def submit(self, function, **kwargs):
            assert kwargs["loss_config"] == expected_config
            raise _StopAfterBuild

    monkeypatch.setattr(
        "concurrent.futures.ProcessPoolExecutor",
        StopPool,
    )
    with pytest.raises(_StopAfterBuild):
        simulate__ler(
            str(source),
            program="Run",
            jobs=1,
            loss_model="trapped-ion",
            simulation_loss_model=simulation_loss_model,
            simulator="qdk",
            skip_mako_warning=True,
        )


def test_interpret_uses_neutral_atom_for_source_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "interpret.deq"
    source.write_text(_SOURCE + "\nPROGRAM Run {}\n", encoding="utf-8")

    def stop_build(*args, **kwargs):
        assert isinstance(kwargs["loss_model"], NeutralAtomLossModel)
        raise _StopAfterBuild

    monkeypatch.setattr(
        "deq.transpiler.jit_library_builder.build_jit_library", stop_build
    )
    with pytest.raises(_StopAfterBuild):
        _load_library(
            str(source),
            program="Run",
        )


def test_qdk_batch_passes_loss_config_to_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stop_run(command, **kwargs):
        config_index = command.index("--simulator-config") + 1
        import json

        simulator_config = json.loads(command[config_index])
        assert simulator_config["py_config"]["loss_config"] == (
            NeutralAtomLossModel.config.to_json_object()
        )
        raise _StopAfterBuild

    monkeypatch.setattr("deq.cli.simulate.subprocess.run", stop_run)
    with pytest.raises(_StopAfterBuild):
        _run_batch(
            bin_path="library.deq.bin",
            stim_path="circuit.stim",
            jit_path="library.deq.jit",
            batch_size=10,
            max_errors=1,
            decoder="black-box-naive",
            decoder_config=None,
            coordinator="monolithic",
            coordinator_config=None,
            seed=1,
            debug_dir=None,
            simulator="qdk",
            loss_config=NeutralAtomLossModel.config.to_json_object(),
        )


def test_simulation_loss_model_rejects_non_json_selector(tmp_path: Path) -> None:
    source = tmp_path / "simulate.deq"
    source.write_text(_SOURCE + "\nPROGRAM Run {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid QDK loss config JSON"):
        simulate__ler(
            str(source),
            program="Run",
            simulation_loss_model="neutral-atom",
            skip_mako_warning=True,
        )


def _jit_with_loss_metadata(
    *, config: dict[str, str] | None = None
) -> jit_pb.JitLibrary:
    if config is None:
        config = NeutralAtomLossModel.config.to_json_object()
    return jit_pb.JitLibrary(metadata={"loss_strategy": config})


def test_precompiled_jit_uses_persisted_config() -> None:
    config = _resolve_jit_loss_config(_jit_with_loss_metadata(), None)

    assert config == NeutralAtomLossModel.config


def test_precompiled_jit_accepts_matching_model_assertion() -> None:
    config = _resolve_jit_loss_config(_jit_with_loss_metadata(), "neutral-atom")

    assert config == NeutralAtomLossModel.config


def test_precompiled_jit_uses_trapped_ion_config() -> None:
    config = _resolve_jit_loss_config(
        _jit_with_loss_metadata(
            config=TrappedIonLossModel.config.to_json_object(),
        ),
        "trapped-ion",
    )

    assert config == TrappedIonLossModel.config


def test_precompiled_jit_without_metadata_uses_empty_config() -> None:
    config = _resolve_jit_loss_config(jit_pb.JitLibrary(), None)

    assert config.to_json_object() == {}


def test_precompiled_jit_with_unrelated_metadata_uses_empty_config() -> None:
    library = jit_pb.JitLibrary(metadata={"mock": {"nested": ["value"]}})

    config = _resolve_jit_loss_config(library, None)

    assert config.to_json_object() == {}


def test_precompiled_jit_without_metadata_accepts_model_selector() -> None:
    config = _resolve_jit_loss_config(jit_pb.JitLibrary(), "trapped-ion")

    assert config.to_json_object() == {}


def test_precompiled_jit_accepts_stored_config_without_named_preset() -> None:
    config = {
        "cx": "PROPAGATE",
        "cy": "SKIP",
        "cz": "SKIP",
        "swap": "APPLY_ANYWAY",
    }

    stored_config = _resolve_jit_loss_config(
        _jit_with_loss_metadata(config=config), None
    )

    assert stored_config.to_json_object() == config


def test_precompiled_jit_rejects_mismatched_model_assertion() -> None:
    with pytest.raises(ValueError, match="does not match precompiled JIT"):
        _resolve_jit_loss_config(_jit_with_loss_metadata(), "trapped-ion")
