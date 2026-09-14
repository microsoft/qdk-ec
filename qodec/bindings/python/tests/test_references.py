"""Unit tests for the property-path :class:`~qodec.gadgets.Reference` value class.

These pin the path validation and string-interchange behavior of the reference
wrapper returned by the gadget analytical surface
(``checks`` / ``readouts`` / flag-pattern keys).

Run with::

    pytest bindings/python/tests/test_references.py
"""

from __future__ import annotations

import pytest

from qodec.gadgets import Reference


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


def test_named_readout_reference_rejected() -> None:
    with pytest.raises(ValueError, match=r"unrecognized reference 'circuit\.readouts\.m_L'"):
        Reference("circuit.readouts.m_L")


def test_flag_record_reference_rejected() -> None:
    with pytest.raises(ValueError, match=r"unrecognized reference 'circuit\.flags\[0\]'"):
        Reference("circuit.flags[0]")


def test_readout_reference_term() -> None:
    readout_ref = Reference("readouts[0]")
    assert readout_ref.path == "readouts[0]"


def test_body_and_gadget_readouts_are_distinct_paths() -> None:
    body = Reference("circuit.readouts[2]")
    gadget = Reference("readouts[2]")
    assert body != gadget


def test_check_reference_rejected() -> None:
    with pytest.raises(ValueError, match=r"unrecognized reference 'checks\[0\]'"):
        Reference("checks[0]")


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


def test_entry_index_is_mandatory() -> None:
    # The `[<entry>]` selector is required — the bare-head sugar
    # (`in.stabilizers[i]`) is removed and now raises.
    with pytest.raises(ValueError, match=r"unrecognized reference 'in\.stabilizers\[0\]'"):
        Reference("in.stabilizers[0]")
    with pytest.raises(ValueError, match=r"unrecognized reference 'out\.z\[1\]'"):
        Reference("out.z[1]")
    # The explicit form is accepted.
    assert Reference("in[0].stabilizers[0]").path == "in[0].stabilizers[0]"


@pytest.mark.parametrize(
    "bad",
    ["bogus.path", "in[0].stabilizers", "circuit.readouts", "", "in.stabilizers"],
)
def test_malformed_reference_raises_value_error(bad: str) -> None:
    with pytest.raises(ValueError, match=r"unrecognized reference"):
        Reference(bad)


def test_kind_names_the_three_shapes() -> None:
    assert Reference("circuit.readouts[3]").kind == "circuit_readout"
    assert Reference("readouts[1]").kind == "readout"
    assert Reference("in[0].stabilizers[2]").kind == "encoding"


def test_encoding_parts_are_exposed() -> None:
    reference = Reference("out[1].x[0]")
    assert reference.boundary == "out"
    assert reference.entry == 1
    assert reference.encoding_property == "x"
    assert reference.index == 0


def test_readout_references_have_no_encoding_parts() -> None:
    """A readout addresses a bit, not a code property, so the encoding fields
    are absent rather than defaulted."""
    for path in ("circuit.readouts[3]", "readouts[1]"):
        reference = Reference(path)
        assert reference.boundary is None
        assert reference.entry is None
        assert reference.encoding_property is None


def test_a_reference_without_a_selector_expands_to_itself() -> None:
    reference = Reference("in[0].stabilizers[2]")
    assert reference.expand() == [reference]


def test_union_expands_in_selector_order() -> None:
    assert Reference("circuit.readouts[2,0]").expand() == [
        "circuit.readouts[2]",
        "circuit.readouts[0]",
    ]


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


def test_index_refuses_a_reference_addressing_several() -> None:
    """The obvious wrong answer -- silently taking the first -- would make a
    partial result indistinguishable from a complete one."""
    with pytest.raises(ValueError, match=r"addresses 3 indices"):
        _ = Reference("circuit.readouts[0,2,4]").index


def test_expansion_is_canonical_and_reparses() -> None:
    """Expanding normalizes each atom to its single-index spelling."""
    for atom in Reference("circuit.readouts[0:2]").expand():
        assert Reference(atom.path) == atom
        assert atom.index == atom.expand()[0].index


@pytest.mark.parametrize("path", ["readouts[02]", "out[01].z[03:04]", "circuit.readouts[00:01:2]"])
def test_singleton_expansion_retains_spelling_and_identity(path: str) -> None:
    reference = Reference(path)
    assert reference.expand()[0] is reference
    assert Reference(reference).path == path


def test_union_preserves_spelling_order_and_duplicates() -> None:
    reference = Reference("out[01].z[3, 1,3]")
    assert reference.path == "out[01].z[3, 1,3]"
    assert reference != Reference("out[1].z[3,1,3]")
    assert [term.index for term in reference.expand()] == [3, 1, 3]
    assert reference.expand() == ["out[1].z[3]", "out[1].z[1]", "out[1].z[3]"]


@pytest.mark.parametrize("attribute", ["path", "kind", "boundary", "entry", "encoding_property", "index", "_path"])
def test_reference_is_immutable(attribute: str) -> None:
    reference = Reference("in[0].z[0]")
    with pytest.raises(AttributeError):
        setattr(reference, attribute, "readouts[9]")
    assert reference.path == "in[0].z[0]"
    assert reference.index == 0


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
    assert reference.kind == "circuit_readout"
    assert Reference(reference).path == reference.path
    with pytest.raises(ValueError, match=f"addresses {count} indices"):
        _ = reference.index


@pytest.mark.parametrize("selector", ["0:1000000000000", "1:1000000000000:3", f"0:{2**64 - 1}"])
def test_a_slice_beyond_the_limit_is_rejected(selector: str) -> None:
    # Every consumer that expands a selector allocates one reference per position.
    with pytest.raises(ValueError, match="more than the limit of"):
        Reference(f"circuit.readouts[{selector}]")
