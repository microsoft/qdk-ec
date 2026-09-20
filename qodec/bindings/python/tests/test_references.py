"""Unit tests for the property-path :class:`~qodec.Reference` value class.

These pin the path validation and string-interchange behavior of the reference
wrapper returned by the gadget analytical surface
(``checks`` / ``readouts`` / flag-pattern keys).

Run with::

    pytest bindings/python/tests/test_references.py
"""

from __future__ import annotations

import pytest
from typing import Any
from copy import copy, deepcopy
import pickle

from qodec import Gadget, Instruction, InstructionSet, Reference
from qodec.gadgets import Circuit


def test_encoding_stabilizer_reference() -> None:
    reference = Reference("in[0].stabilizers[1]")
    assert reference.path == "in[0].stabilizers[1]"


def test_logical_references() -> None:
    x_reference = Reference("out[2].x[3]")
    assert x_reference.path == "out[2].x[3]"

    z_reference = Reference("in[1].z[0]")
    assert z_reference.path == "in[1].z[0]"


def test_positional_body_readout_reference() -> None:
    reference = Reference("circuit.readouts[4]")
    assert reference.path == "circuit.readouts[4]"


@pytest.mark.parametrize("path", [
    "circuit.readouts.m_L", "circuit.flags[0]", "checks[0]", "in.stabilizers[0]",
    "out.z[1]", "in[0].stabilizers", "circuit.readouts", "",
])
def test_model_addresses_are_rejected_in_equations(path: str) -> None:
    reference = Reference(path)
    with pytest.raises(ValueError, match="not a parity reference"):
        Gadget(Instruction("draft"), Circuit(InstructionSet("draft"), "opaque", format="unknown"), checks=[[reference]])


def test_readout_reference_term() -> None:
    readout_ref = Reference("readouts[0]")
    assert readout_ref.path == "readouts[0]"


def test_body_and_gadget_readouts_are_distinct_paths() -> None:
    body = Reference("circuit.readouts[2]")
    gadget = Reference("readouts[2]")
    assert body != gadget


def test_slice_selector_accepted() -> None:
    assert Reference("circuit.readouts[0:2]").path == "circuit.readouts[0:2]"


def test_union_selector_accepted() -> None:
    assert Reference("circuit.readouts[0,2]").path == "circuit.readouts[0,2]"


def test_reference_is_not_a_str_subclass() -> None:
    assert not isinstance(Reference("circuit.readouts[0]"), str)


def test_reference_compares_equal_to_its_text_string() -> None:
    reference = Reference("circuit.readouts[0]")
    assert reference == "circuit.readouts[0]"
    assert "circuit.readouts[0]" == reference
    assert reference == Reference("circuit.readouts[0]")
    assert reference != "circuit.readouts[1]"
    assert reference != 42


def test_reference_hashes_like_its_text_string() -> None:
    reference = Reference("circuit.readouts[0]")
    assert hash(reference) == hash("circuit.readouts[0]")
    mapping: dict[object, int] = {reference: 1}
    assert mapping["circuit.readouts[0]"] == 1
    assert {reference: 1} == {"circuit.readouts[0]": 1}


def test_reference_str_and_repr() -> None:
    reference = Reference("in[0].stabilizers[0]")
    assert str(reference) == "in[0].stabilizers[0]"
    assert repr(reference) == "Reference('in[0].stabilizers[0]')"


def test_reference_accepts_a_reference_value() -> None:
    reference = Reference(Reference("circuit.readouts[0]"))
    assert reference.path == "circuit.readouts[0]"


def test_entry_index_identifies_an_encoding_parity_reference() -> None:
    assert Reference("in.stabilizers[0]").segments == (Reference.Field("in"), Reference.Field("stabilizers"), Reference.Index(0))
    assert Reference("in[0].stabilizers[0]").segments == (Reference.Field("in"), Reference.Index(0), Reference.Field("stabilizers"), Reference.Index(0))
    assert Reference("in[0].stabilizers[0]").path == "in[0].stabilizers[0]"


@pytest.mark.parametrize(
    "bad",
    ["bogus..path", "in[0].stabilizers.", "circuit.readouts[", "in.stabilizers()"],
)
def test_malformed_reference_raises_value_error(bad: str) -> None:
    with pytest.raises(ValueError, match=r"unrecognized reference"):
        Reference(bad)


def test_general_paths_expose_typed_segments() -> None:
    assert Reference('layers[00].gadgets["M"].in[0].z[1:4:2]').segments == (
        Reference.Field("layers"), Reference.Index(0), Reference.Field("gadgets"),
        Reference.Key("M"), Reference.Field("in"), Reference.Index(0),
        Reference.Field("z"), Reference.Slice(1, 4, 2),
    )
    assert Reference('metadata["name"]').segments[-1] == Reference.Key("name")
    assert Reference("metadata.name").segments[-1] == Reference.Field("name")
    assert Reference("layers[2,0,2]").segments[-1] == Reference.Union((2, 0, 2))
    assert Reference("").segments == ()
    for name in ("kind", "parity_kind", "boundary", "entry", "encoding_property", "index"):
        assert not hasattr(Reference("in[0].z[0]"), name)


def test_segments_support_pattern_matching_without_parity_assumptions() -> None:
    reference = Reference('layers[0].gadgets["measure_z"]')
    match reference.segments:
        case (Reference.Field("layers"), Reference.Index(layer), Reference.Field("gadgets"), Reference.Key(mnemonic)):
            assert (layer, mnemonic) == (0, "measure_z")
        case _:
            pytest.fail("structural pattern did not match")


@pytest.mark.parametrize("segment", [Reference.Field("name"), Reference.Key("name"), Reference.Index(2),
    Reference.Slice(1, 5, 2), Reference.Union((3, 1, 3))])
def test_segments_are_immutable_hashable_and_picklable(segment: Any) -> None:
    assert copy(segment) == deepcopy(segment) == pickle.loads(pickle.dumps(segment)) == segment
    assert hash(segment) == hash(deepcopy(segment))
    assert type(segment).__module__ == "qodec"
    assert pickle.loads(pickle.dumps(type(segment))) is type(segment)
    for field in segment.__match_args__:
        with pytest.raises((AttributeError, TypeError)):
            setattr(segment, field, "changed")
    assert repr(segment).startswith("Reference.")
    assert Reference.Field("name") != Reference.Key("name")


@pytest.mark.parametrize("constructor, arguments", [
    (Reference.Field, ("a.b",)), (Reference.Field, ("",)), (Reference.Field, ("a-b",)),
    (Reference.Field, (1,)), (Reference.Key, (1,)), (Reference.Index, (True,)),
    (Reference.Index, (-1,)), (Reference.Slice, (0, 3, 0)), (Reference.Slice, (3, 1)),
    (Reference.Slice, (False, 3)), (Reference.Union, ((0,),)), (Reference.Union, ([0, 1],)),
    (Reference.Union, ((0, False),)), (Reference.Union, ((0, -1),)),
])
def test_segment_constructors_reject_invalid_values(constructor: Any, arguments: tuple[Any, ...]) -> None:
    with pytest.raises((TypeError, ValueError)):
        constructor(*arguments)


@pytest.mark.parametrize("value", [None, True, False, Ellipsis, 0, 1.5, [], {}, object()])
def test_reference_requires_text_or_an_existing_reference(value: Any) -> None:
    with pytest.raises(TypeError):
        Reference(value)


def test_reference_does_not_coerce_custom_objects_to_text() -> None:
    class StringLike:
        def __str__(self) -> str:
            raise AssertionError("constructor must not call str")

    value: Any = StringLike()
    with pytest.raises(TypeError):
        Reference(value)


def test_frame_keys_accept_references_for_all_mapping_operations() -> None:
    from collections import UserDict

    first = Reference("out[00].z[0]")
    second = Reference("out[0].x[0]")
    gadget = Gadget(Instruction("draft"), Circuit(InstructionSet("draft"), "opaque", format="unknown"),
        frames=UserDict({first: [0]}))
    assert list(gadget.frames) == [first.path]
    assert all(key.startswith("out[") for key in gadget.frames)
    assert all(key.startswith("out[") for key in gadget.frames.keys())
    assert all(key.startswith("out[") for key, _ in gadget.frames.items())
    gadget.frames[second] = (1,)
    gadget.frames[first] = (1,)
    assert gadget.frames[first.path] == gadget.frames[second] == (1,)
    assert gadget.frames.get(first) == (1,)
    gadget.frames.update({first: (0,)})
    assert gadget.frames.pop(second) == (1,)
    assert gadget.frames.setdefault(second, (0,)) == (0,)
    del gadget.frames[first]
    gadget.frames = {first: [second]}
    assert list(gadget.frames) == [first.path]
    assert gadget.frames[first] == (second,)
    for invalid in [Reference("metadata"), None, True, 3]:
        before = dict(gadget.frames)
        invalid_key: Any = invalid
        with pytest.raises((TypeError, ValueError)):
            gadget.frames.update({second: (1,), invalid_key: ()})
        assert dict(gadget.frames) == before


def test_frame_writes_accept_sequences_and_normalize_read_values() -> None:
    from typing import assert_type
    from qodec.gadgets import Check

    gadget = Gadget(Instruction("draft"), Circuit(InstructionSet("draft"), "opaque", format="unknown"))
    target = Reference("out[0].x[0]")
    term = Reference("circuit.readouts[0]")
    gadget.frames[target] = [0]
    assert gadget.frames[target] == (0,)
    gadget.frames[target] = (term.path,)
    assert gadget.frames[target] == (term,)
    gadget.frames.update({target: [term.path, 1]})
    assert gadget.frames[target] == (term, 1)
    gadget.frames.update([(target, [0])])
    assert gadget.frames[target] == (0,)
    gadget.frames.update(**{target.path: [term.path]})
    assert gadget.frames[target] == (term,)
    del gadget.frames[target]
    assert_type(gadget.frames.setdefault(target, [term.path, 0]), Check)
    assert gadget.frames[target] == (term, 0)
    assert_type(gadget.frames[target], Check)
    assert_type(next(iter(gadget.frames)), str)


def test_a_reference_without_a_selector_expands_to_itself() -> None:
    reference = Reference("in[0].stabilizers[2]")
    assert reference.expand() == [reference]


def test_union_expands_in_selector_order() -> None:
    assert Reference("circuit.readouts[2,0]").expand() == [
        "circuit.readouts[2]",
        "circuit.readouts[0]",
    ]

def test_large_union_expansion_preserves_duplicates() -> None:
    reference = Reference("readouts[" + ",".join(["0", "2"] * 8192) + "]")
    expanded = reference.expand()
    assert len(expanded) == 16384
    assert all(item.path == f"readouts[{2 * (position % 2)}]" for position, item in enumerate(expanded))


def test_slice_expands_stop_exclusive() -> None:
    assert Reference("in[0].stabilizers[1:3]").expand() == [
        "in[0].stabilizers[1]",
        "in[0].stabilizers[2]",
    ]


def test_strided_slice_expands() -> None:
    assert Reference("circuit.readouts[0:5:2]").expand() == [
        "circuit.readouts[0]",
        "circuit.readouts[2]",
        "circuit.readouts[4]",
    ]


def test_selectors_remain_distinct_from_scalar_indices() -> None:
    assert Reference("readouts[0:1]").segments[-1] == Reference.Slice(0, 1)
    assert Reference("readouts[0]").segments[-1] == Reference.Index(0)
    assert Reference("readouts[0,2,4]").segments[-1] == Reference.Union((0, 2, 4))


def test_expansion_is_canonical_and_reparses() -> None:
    """Expanding normalizes each atom to its single-index spelling."""
    for atom in Reference("circuit.readouts[0:2]").expand():
        assert Reference(atom.path) == atom
        assert atom.segments == atom.expand()[0].segments


@pytest.mark.parametrize("path", ["readouts[02]", "out[01].z[03:04]", "circuit.readouts[00:01:2]"])
def test_singleton_expansion_retains_spelling_and_identity(path: str) -> None:
    reference = Reference(path)
    assert reference.expand()[0] is reference
    assert Reference(reference).path == path


def test_union_preserves_spelling_order_and_duplicates() -> None:
    reference = Reference("out[01].z[3, 1,3]")
    assert reference.path == "out[01].z[3, 1,3]"
    assert reference != Reference("out[1].z[3,1,3]")
    assert [term.segments[-1] for term in reference.expand()] == [Reference.Index(3), Reference.Index(1), Reference.Index(3)]
    assert reference.expand() == ["out[1].z[3]", "out[1].z[1]", "out[1].z[3]"]


@pytest.mark.parametrize("attribute", ["path", "segments", "_path"])
def test_reference_is_immutable(attribute: str) -> None:
    reference = Reference("in[0].z[0]")
    with pytest.raises(AttributeError):
        setattr(reference, attribute, "readouts[9]")
    assert reference.path == "in[0].z[0]"
    assert reference.segments[-1] == Reference.Index(0)


@pytest.mark.parametrize("path", ["readouts[0:0]", "circuit.readouts[2:1]", "out[0].z[2:2:3]"])
def test_empty_selectors_are_rejected(path: str) -> None:
    with pytest.raises(ValueError, match="selects no indices"):
        Reference(path)


@pytest.mark.parametrize(("selector", "count"), [
    ("0:1000000", 1000000),
    ("1:1000000:3", 333333),
    ("0:18446744073709551615:18446744073709551614", 2),
])
def test_large_slice_remains_compact_until_expanded(selector: str, count: int) -> None:
    reference = Reference(f"circuit.readouts[{selector}]")
    assert Reference(reference).path == reference.path
    selected = reference.segments[-1]
    assert isinstance(selected, Reference.Slice)
    assert len(range(selected.start, selected.stop, selected.step)) == count


@pytest.mark.parametrize("selector", ["0:1000000000000", "1:1000000000000:3", f"0:{2**64 - 1}"])
def test_a_slice_beyond_the_limit_is_rejected(selector: str) -> None:
    # Every consumer that expands a selector allocates one reference per position.
    with pytest.raises(ValueError, match="more than the limit of"):
        Reference(f"circuit.readouts[{selector}]")
