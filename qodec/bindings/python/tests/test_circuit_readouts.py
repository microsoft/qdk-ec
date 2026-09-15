"""``Circuit.readouts``: one entry per measurement-record bit.

A circuit's record is not just its ``observe`` outcomes — a called
instruction's declared flags occupy positions in the same record. These tests
pin that, since counting only observations silently shifts every later
``circuit.readouts[i]`` reference.
"""

from __future__ import annotations

import pathlib

import pytest

import qodec
from qodec.gadgets import Flag, Outcome

EXAMPLES = pathlib.Path(__file__).resolve().parents[3] / "examples"


def _gadgets(protocol: qodec.Qodec) -> list[qodec.Gadget]:
    return [gadget for layer in protocol.layers for gadget in layer.gadgets.values()]


@pytest.fixture(scope="module")
def c4c6() -> qodec.Qodec:
    return qodec.Qodec.load(EXAMPLES / "c4c6" / "qodec.yaml")


def test_readouts_reference_existing_calls(c4c6: qodec.Qodec) -> None:
    for gadget in _gadgets(c4c6):
        call_count = len(gadget.circuit.calls())
        for readout in gadget.circuit.readouts:
            assert 0 <= readout.instruction < call_count


def test_flags_occupy_record_positions(c4c6: qodec.Qodec) -> None:
    """A called instruction's flags are bits of the record, not a side channel."""
    idle = c4c6.layers[0].gadgets["idle"].circuit.readouts
    assert len(idle) == 18
    assert sum(isinstance(readout, Flag) for readout in idle) == 6
    assert {readout.name for readout in idle if isinstance(readout, Flag)} == {"reject"}


def test_a_call_contributes_observations_then_flags(c4c6: qodec.Qodec) -> None:
    for gadget in _gadgets(c4c6):
        seen_flag: set[int] = set()
        for readout in gadget.circuit.readouts:
            if isinstance(readout, Flag):
                seen_flag.add(readout.instruction)
            else:
                assert readout.instruction not in seen_flag


def test_observation_carries_the_declared_pauli(c4c6: qodec.Qodec) -> None:
    for gadget in _gadgets(c4c6):
        for readout in gadget.circuit.readouts:
            if isinstance(readout, Outcome):
                assert readout.observable
                assert set(readout.observable) & set("XYZ")


def test_repetition3_has_no_flags() -> None:
    """The bottom layer measures but declares no flags."""
    protocol = qodec.Qodec.load(EXAMPLES / "repetition3" / "repetition3.qodec.yaml")
    readouts = [
        readout for gadget in _gadgets(protocol) for readout in gadget.circuit.readouts
    ]
    assert readouts
    assert all(isinstance(readout, Outcome) for readout in readouts)
