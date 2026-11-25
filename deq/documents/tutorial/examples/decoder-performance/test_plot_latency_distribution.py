"""Check distribution tails and the layer mapping without censoring slow samples."""

import numpy as np
import pytest

from plot_latency_distribution import population, render, summarize
from readout_latency import ReadoutSample


def test_ccdf_preserves_ties_and_the_worst_observation():
    result = population(np.array([2.0, 1000.0, 2.0]))
    assert result["ccdf"] == [[2.0, 1.0], [1000.0, 1 / 3]]
    assert result["maximum_ms"] == 1000
    assert result["percentiles_ms"]["50"] == 2


def test_layer_distribution_keeps_readout_types_and_slow_samples(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    titles = []
    original_title = Figure.suptitle

    def record_title(figure, text, **kwargs):
        titles.append(text)
        return original_title(figure, text, **kwargs)

    monkeypatch.setattr(Figure, "suptitle", record_title)
    samples = []
    for shot, values in enumerate(([1, 2, 3, 7, 8, 9], [4, 5, 600, 10, 11, 12])):
        samples.extend(
            ReadoutSample(shot, gid, "intermediate", float(value))
            for gid, value in enumerate(values, 1)
        )
        samples.extend(ReadoutSample(shot, gid, "final", 13.0) for gid in range(7, 10))
    configuration = {
        "width": 3,
        "layers": 2,
        "epochs": {str(gid): gid + 1 for gid in range(1, 7)},
        "gadget_names": {
            str(gid): ("MZZ", "MXX", "MeasureZ")[(gid - 1) % 3] for gid in range(1, 7)
        },
    }
    report = summarize(samples, configuration, 2)
    assert report["intermediate"]["samples"] == 12
    assert report["final"]["samples"] == 6
    assert report["by_readout"]["MeasureZ"]["maximum_ms"] == 600
    assert report["by_layer"][0]["percentiles_ms"]["50"] == 3.5
    assert report["by_layer"][1]["percentiles_ms"]["50"] == 9.5
    report.update(width=3, layers=2, probability=1e-5, measured_shots=2, warmup_shots=1)
    figure = tmp_path / "latency.png"
    render(report, figure)
    assert figure.read_bytes().startswith(b"\x89PNG")
    assert "p=1e-05" in titles[0]
    with pytest.raises(ValueError, match="coverage"):
        summarize(samples[1:], configuration, 2)
