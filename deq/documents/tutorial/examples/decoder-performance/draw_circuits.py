"""Draw the tutorial's logical brickwall and lattice-surgery circuit as PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle, FancyBboxPatch

HERE = Path(__file__).resolve().parent
INK = "#263449"
PURPLE = "#7646ac"
BLUE = "#e9f1fb"
ORANGE = "#fff0d9"


def gate(
    axis: Axes,
    x: float,
    y: float,
    label: str,
    *,
    height: float = 0.6,
    color: str = BLUE,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x - 0.8, y - height / 2),
            1.6,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=color,
            edgecolor=INK,
            linewidth=1.2,
            zorder=3,
        )
    )
    axis.text(x, y, label, ha="center", va="center", fontsize=11, color=INK, zorder=4)


def draw_brickwall(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 3.8))
    for column, label in ((2, "Layer 1"), (5, "Layer 2")):
        axis.axvspan(column - 1, column + 1, color="#f5f3f9", zorder=0)
        axis.text(
            column, 3.65, label, ha="center", fontsize=12, weight="bold", color=INK
        )
    for patch in range(4):
        y = 3 - patch
        axis.plot([0.3, 6.7], [y, y], color=INK, linewidth=1.3, zorder=1)
        axis.text(
            0.1, y, f"Patch {patch}", ha="right", va="center", fontsize=12, color=INK
        )
    for x, control, target in ((2, 3, 2), (2, 1, 0), (5, 2, 1)):
        axis.plot([x, x], [target, control], color=PURPLE, linewidth=2, zorder=2)
        axis.add_patch(Circle((x, control), 0.08, color=PURPLE, zorder=3))
        axis.add_patch(
            Circle(
                (x, target),
                0.17,
                facecolor="white",
                edgecolor=PURPLE,
                linewidth=2,
                zorder=3,
            )
        )
        axis.plot(
            [x - 0.17, x + 0.17],
            [target, target],
            color=PURPLE,
            linewidth=1.5,
            zorder=4,
        )
        axis.plot(
            [x, x],
            [target - 0.17, target + 0.17],
            color=PURPLE,
            linewidth=1.5,
            zorder=4,
        )
    for y in (0, 3):
        gate(axis, 5, y, "3 SE blocks\nof $d$ rounds", height=0.75)
    axis.annotate(
        "time",
        xy=(6.6, -0.65),
        xytext=(5.4, -0.65),
        arrowprops={"arrowstyle": "->", "color": INK},
        va="center",
        color=INK,
        fontsize=11,
    )
    axis.set(xlim=(-1.1, 7), ylim=(-0.9, 4))
    axis.set_aspect("equal")
    axis.axis("off")
    figure.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def draw_surgery(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 4.5))
    stages = ("Previous block", "ZZ surgery", "XX surgery", "Read ancilla")
    for x, label in zip((2, 5, 8, 11), stages):
        axis.axvspan(x - 1.2, x + 1.2, color="#f6f8fb", zorder=0)
        axis.text(x, 5, label, ha="center", fontsize=12, weight="bold", color=INK)
        duration = "preparation overlaps" if x == 2 else "d rounds"
        axis.text(x, 4.6, duration, ha="center", fontsize=10, color="#627086")
    for y, label in ((4, "Control (0)"), (2, "Ancilla (2)"), (0, "Target (1)")):
        start, end = (2, 11) if y == 2 else (0.3, 12.5)
        axis.plot([start, end], [y, y], color=INK, linewidth=1.3, zorder=1)
        axis.text(0.1, y, label, ha="right", va="center", fontsize=12, color=INK)
    for y in (4, 0):
        gate(axis, 2, y, "Prior work", color="#f0f0f0")
    for x, y in ((5, 0), (8, 4), (11, 4), (11, 0)):
        gate(axis, x, y, "SE\n$d$ rounds", height=0.8)
    gate(axis, 2, 2, "PrepareX\nNonFT\n1 SE round", height=1.15)
    gate(axis, 5, 3, "$M_{ZZ}$\nmerge\n& split", height=2.6, color="#eee5f7")
    gate(axis, 8, 1, "$M_{XX}$\nmerge\n& split", height=2.6, color="#eee5f7")
    gate(axis, 11, 2, "$M_Z$", color=ORANGE)
    axis.text(
        6.5,
        -0.8,
        "Ancilla preparation overlaps the final round of prior work; no extra time block.",
        ha="center",
        fontsize=11,
        color=INK,
    )
    axis.set(xlim=(-1.8, 12.9), ylim=(-1.1, 5.4))
    axis.set_aspect("equal")
    axis.axis("off")
    figure.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def draw(output: Path) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    brickwall = output / "brickwall-circuit.png"
    surgery = output / "lattice-surgery-cnot.png"
    draw_brickwall(brickwall)
    draw_surgery(surgery)
    return brickwall, surgery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    for path in draw(parser.parse_args().output):
        print(path)


if __name__ == "__main__":
    main()
