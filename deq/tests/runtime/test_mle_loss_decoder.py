"""Focused tests for the bundled generator-MILP loss decoder."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _decoder_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "deq_runtime"
        / "src"
        / "decoder"
        / "mle_loss_decoder.py"
    )
    spec = importlib.util.spec_from_file_location("test_mle_loss_decoder_impl", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _hypergraph(*edges):
    return SimpleNamespace(
        vertex_num=1,
        hyperedges=[
            SimpleNamespace(vertices=list(vertices), probability=probability)
            for vertices, probability in edges
        ],
    )


def _site(*, source=(), continuation=(), children=()):
    return SimpleNamespace(
        source_edges=list(source),
        continuation_edges=list(continuation),
        children=list(children),
        probability=0.0,
    )


def test_ordinary_positive_prior_edge_satisfies_syndrome() -> None:
    decoder = _decoder_module().Decoder(_hypergraph(([0], 0.1)))

    assert decoder.decode([0]) == [0]
    assert decoder.decode([]) == []


def test_loss_activates_zero_prior_source_edge() -> None:
    decoder = _decoder_module().Decoder(_hypergraph(([0], 0.0)))
    loss = SimpleNamespace(sites=[_site(source=[0])])

    with pytest.raises(RuntimeError, match="produced no solution"):
        decoder.decode([0])
    assert decoder.decode([0], loss) == [0]


def test_parent_start_enables_child_continuation_edge() -> None:
    decoder = _decoder_module().Decoder(_hypergraph(([0], 0.0)))
    sites = [
        _site(children=[1]),
        _site(continuation=[0]),
    ]

    enabling, loss_edges, components = decoder._loss_structure(sites)

    assert enabling == {0: {0, 1}}
    assert loss_edges == {0}
    assert list(components.values()) == [[0, 1]]
    assert decoder.decode([0], SimpleNamespace(sites=sites)) == [0]


@pytest.mark.parametrize("field", ["source_edges", "continuation_edges"])
def test_out_of_range_loss_edge_is_rejected(field: str) -> None:
    decoder = _decoder_module().Decoder(_hypergraph(([0], 0.0)))
    site = _site()
    setattr(site, field, [1])

    with pytest.raises(ValueError, match=r"edge 1, outside \[0, 1\)"):
        decoder.decode([0], SimpleNamespace(sites=[site]))


def test_out_of_range_child_site_is_rejected() -> None:
    decoder = _decoder_module().Decoder(_hypergraph(([0], 0.0)))

    with pytest.raises(ValueError, match=r"site 1, outside \[0, 1\)"):
        decoder.decode(
            [0],
            SimpleNamespace(sites=[_site(source=[0], children=[1])]),
        )


def test_cyclic_loss_sites_are_rejected() -> None:
    decoder = _decoder_module().Decoder(_hypergraph(([0], 0.0)))
    loss = SimpleNamespace(
        sites=[
            _site(source=[0], children=[1]),
            _site(continuation=[0], children=[0]),
        ]
    )

    with pytest.raises(ValueError, match="children graph contains a cycle"):
        decoder.decode([0], loss)


def test_solver_without_solution_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _decoder_module()
    decoder = module.Decoder(_hypergraph(([0], 0.1)))
    monkeypatch.setattr(
        module,
        "milp",
        lambda **_kwargs: SimpleNamespace(
            x=None,
            status=1,
            message="time limit reached",
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=r"produced no solution \(status=1\): time limit reached",
    ):
        decoder.decode([0])


def test_empty_hypergraph_still_validates_loss_edges() -> None:
    decoder = _decoder_module().Decoder(_hypergraph())

    with pytest.raises(ValueError, match=r"edge 0, outside \[0, 0\)"):
        decoder.decode([], SimpleNamespace(sites=[_site(source=[0])]))