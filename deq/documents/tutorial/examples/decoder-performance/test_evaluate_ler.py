"""LER publication requires real CLI counts and the requested stopping criteria."""

from pathlib import Path

import pytest

from evaluate_ler import parse_result, plot, wilson


def test_cli_counts_not_progress_updates_determine_the_rate():
    result = parse_result(
        "progress shots=5 rate=0.1\n=== Simulation Results ===\n"
        "  Shots:          200\n  Logical errors: 21\n  Failed shots:   0\n"
    )
    assert result == {"shots": 200, "errors": 21, "failed": 0, "rate": 0.105}


@pytest.mark.parametrize("errors,failed", [(19, 0), (20, 1), (101, 0)])
def test_incomplete_or_failed_points_are_rejected(errors: int, failed: int):
    with pytest.raises(ValueError):
        parse_result(
            f"=== Simulation Results ===\nShots: 100\nLogical errors: {errors}\nFailed shots: {failed}\n"
        )


def test_missing_cli_summary_is_not_a_zero_error_result():
    with pytest.raises(ValueError):
        parse_result("native worker panicked")


def test_plot_requires_each_curve_to_cross_eighty_percent(tmp_path: Path):
    points = [
        {
            "decoder": decoder,
            "width": width,
            "shots": shots,
            "errors": 20,
            "failed": 0,
            "rate": 20 / shots,
        }
        for decoder in ("mwpm", "mwpf")
        for width, shots in ((1, 200), (2, 24))
    ]
    document = {"status": "complete", "points": points}
    plot(document, tmp_path / "ler.png")
    assert (tmp_path / "ler.png").stat().st_size > 1000
    low, high = wilson(20, 200)
    assert 0 < low < 0.1 < high < 1
    document["points"] = points[:-1]
    with pytest.raises(ValueError, match="80%"):
        plot(document, tmp_path / "incomplete.png")
    document["status"] = "stopped"
    with pytest.raises(ValueError, match="completed"):
        plot(document, tmp_path / "not-authorized.png")
    plot(document, tmp_path / "partial.png", allow_incomplete=True)
    assert (tmp_path / "partial.png").stat().st_size > 1000
    points[0]["errors"] = 19
    with pytest.raises(ValueError, match="insufficiently sampled"):
        plot(document, tmp_path / "too-few.png", allow_incomplete=True)
