"""Example gadgets must retain the weakest boundary code's Pauli distance."""

import pytest
import qdk.ec as ec
import qodec

from conftest import EXAMPLE_MANIFESTS, EXAMPLES_DIR


def _gadget_cases():
    for manifest in EXAMPLE_MANIFESTS:
        protocol = qodec.Qodec.load(EXAMPLES_DIR / manifest)
        for layer_index, layer in enumerate(protocol.layers[:-1]):
            distances = {}
            for code in layer.codes.values():
                distance = ec.CodeProfile(code).distance()
                value = distance.value
                assert distance.is_exact and value is not None, f"{code.name}: {distance}"
                distances[code.name] = value
            for name, gadget in layer.gadgets.items():
                codes = [encoding.code for encoding in (*gadget.inputs, *gadget.outputs)]
                assert codes, f"{protocol.name}/{name}: no boundary code to compare"
                expected = min(distances[code.name] for code in codes)
                identity = f"{protocol.name}/layer{layer_index}/{name}"
                yield pytest.param(gadget, expected, id=identity)


@pytest.mark.parametrize("gadget,expected", list(_gadget_cases()))
def test_example_gadget_achieves_code_distance(gadget: qodec.Gadget, expected: int) -> None:
    actual = ec.GadgetProfile(gadget).distance()
    assert actual >= expected, (
        f"{gadget.implements.mnemonic}: gadget distance {actual} < code distance {expected}; "
        f"witness: {actual.witness}"
    )
