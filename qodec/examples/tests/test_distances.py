"""Example gadgets must retain the weakest boundary code's Pauli distance."""

import pytest
import qdk.ec as ec
import qodec

from conftest import EXAMPLE_MANIFESTS, EXAMPLES_DIR

_UNSUPPORTED_DISTANCE_CASES = {
    "c4c6/layer0/idle",
    "c4c6/layer0/prepare_x_all",
    "c4c6/layer0/prepare_z_all",
    "c422-c832-arch/layer0/ccz_c832",
    "distillation-15/layer0/distill_t",
    "reed-muller-15/layer0/t",
    "repetition3/layer0/rotate_z",
}
_UNSUPPORTED_DISTANCE_MARK = pytest.mark.xfail(
    reason="distance analysis does not support this gadget",
    strict=True,
)


def _code_distances(layer: qodec.Layer) -> dict[str, int]:
    distances = {}
    for code in layer.codes.values():
        distance = ec.CodeProfile(code).distance()
        value = distance.value
        assert distance.is_exact and value is not None, f"{code.name}: {distance}"
        distances[code.name] = value
    return distances


def _boundary_distance(
    gadget: qodec.Gadget,
    code_distances: dict[str, int],
    identity: str,
) -> int:
    codes = [encoding.code for encoding in (*gadget.inputs, *gadget.outputs)]
    assert codes, f"{identity}: no boundary code to compare"
    return min(code_distances[code.name] for code in codes)


def _gadget_cases():
    for manifest in EXAMPLE_MANIFESTS:
        protocol = qodec.Qodec.load(EXAMPLES_DIR / manifest)
        for layer_index, layer in enumerate(protocol.layers[:-1]):
            code_distances = _code_distances(layer)
            for name, gadget in layer.gadgets.items():
                identity = f"{protocol.name}/layer{layer_index}/{name}"
                expected = _boundary_distance(gadget, code_distances, identity)
                marks = (
                    [_UNSUPPORTED_DISTANCE_MARK]
                    if identity in _UNSUPPORTED_DISTANCE_CASES
                    else []
                )
                yield pytest.param(gadget, expected, id=identity, marks=marks)


@pytest.mark.parametrize("gadget,expected", list(_gadget_cases()))
def test_example_gadget_achieves_code_distance(gadget: qodec.Gadget, expected: int) -> None:
    actual = ec.GadgetProfile(gadget).distance()
    assert actual >= expected, (
        f"{gadget.implements.mnemonic}: gadget distance {actual} < code distance {expected}; "
        f"witness: {actual.witness}"
    )
