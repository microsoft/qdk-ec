"""Render the d=3 Floquet honeycomb lattice figure used in the tutorial."""

import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(THIS_DIR, "lattice_d3.png")


# ---------------------------------------------------------------------------
# Lattice figure
# ---------------------------------------------------------------------------
#
# We draw the d=3 honeycomb (666) Floquet code as 9 hexagonal faces tiling a
# 3x3 torus. Faces are tri-colored Red / Green / Blue (one stabilizer of each
# Pauli type per color); each edge color is the Pauli measured by the
# corresponding round (XX = red, YY = green, ZZ = blue). The boundary of a
# color-C face contains only edges of the OTHER two colors (alternating).
#
# Hex-face placement uses axial coordinates (I, J) with cartesian basis
#   e1 = (sqrt(3), 0),   e2 = (sqrt(3)/2, 3/2).
# Tri-coloring is (I - J) mod 3 -> {Red, Green, Blue}. The 3x3 torus has
# period vectors P1 = 3*e1 and P2 = 3*e2; min-image is used to handle wraps.
#
# Each qubit lives at the centroid of its three incident face centers. The
# qubit -> (R-face, G-face, B-face) membership is derived directly from the
# 9 X-loop, 9 Y-loop, and 9 Z-loop plaquette stabilizers in floquet.deq.

SQRT3 = np.sqrt(3.0)
HEX_E1 = np.array([SQRT3, 0.0])
HEX_E2 = np.array([SQRT3 / 2.0, 1.5])
PERIOD1 = 3.0 * HEX_E1
PERIOD2 = 3.0 * HEX_E2

COLORS = {"red": "#d62728", "green": "#2ca02c", "blue": "#1f77b4"}
LIGHT_COLORS = {"red": "#f5cbcb", "green": "#cfe9cf", "blue": "#cfdcf6"}

# Hex (I, J) axial coordinates per face (color, idx).
R_FACE_COORDS = [(1, 1), (0, 0), (2, 2)]  # R0, R1, R2 -> X-loop plaquettes
G_FACE_COORDS = [(1, 0), (2, 1), (0, 2)]  # G0, G1, G2 -> Y-loop plaquettes
B_FACE_COORDS = [(0, 1), (1, 2), (2, 0)]  # B0, B1, B2 -> Z-loop plaquettes

# Qubit -> (R-face idx, G-face idx, B-face idx). Derived from the plaquette
# stabilizers in floquet.deq: each qubit lies on exactly one X-loop, one
# Y-loop and one Z-loop plaquette.
QUBIT_FACES = [
    (0, 0, 0),  # q0
    (1, 0, 0),  # q1
    (1, 1, 0),  # q2
    (2, 1, 0),  # q3
    (2, 2, 0),  # q4
    (0, 2, 0),  # q5
    (1, 0, 1),  # q6
    (2, 0, 1),  # q7
    (2, 1, 1),  # q8
    (0, 1, 1),  # q9
    (0, 2, 1),  # q10
    (1, 2, 1),  # q11
    (2, 0, 2),  # q12
    (0, 0, 2),  # q13
    (0, 1, 2),  # q14
    (1, 1, 2),  # q15
    (1, 2, 2),  # q16
    (2, 2, 2),  # q17
]

# Plaquette (face) -> 6 qubits, taken straight from floquet.deq STABILIZER lines.
FACE_QUBITS = {
    ("red", 0): [0, 5, 9, 10, 13, 14],
    ("red", 1): [1, 2, 6, 11, 15, 16],
    ("red", 2): [3, 4, 7, 8, 12, 17],
    ("green", 0): [0, 1, 6, 7, 12, 13],
    ("green", 1): [2, 3, 8, 9, 14, 15],
    ("green", 2): [4, 5, 10, 11, 16, 17],
    ("blue", 0): [0, 1, 2, 3, 4, 5],
    ("blue", 1): [6, 7, 8, 9, 10, 11],
    ("blue", 2): [12, 13, 14, 15, 16, 17],
}

# Edges from the gadget MPP statements in floquet.deq.
RED_EDGES = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    (12, 13),
    (14, 15),
    (16, 17),
]
GREEN_EDGES = [
    (0, 5),
    (2, 1),
    (4, 3),
    (6, 11),
    (8, 7),
    (10, 9),
    (12, 17),
    (14, 13),
    (16, 15),
]
BLUE_EDGES = [
    (0, 13),
    (2, 15),
    (4, 17),
    (6, 1),
    (8, 3),
    (10, 5),
    (12, 7),
    (14, 9),
    (16, 11),
]


def _hex_pos(coord: tuple[int, int]) -> np.ndarray:
    """Cartesian center of a hex face given its axial (I, J) coordinate."""

    i, j = coord
    return i * HEX_E1 + j * HEX_E2


def _min_image(anchor: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the periodic image of ``target`` closest to ``anchor`` on the 3x3 torus."""

    best = target
    best_d = float(np.linalg.norm(target - anchor))
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            cand = target + di * PERIOD1 + dj * PERIOD2
            d = float(np.linalg.norm(cand - anchor))
            if d < best_d:
                best_d = d
                best = cand
    return best


def _draw_face(
    ax: plt.Axes,
    face_center: np.ndarray,
    face_qubits: list[int],
    qubit_positions: np.ndarray,
    fill_color: str,
) -> None:
    """Fill one hex face as a polygon by sorting its 6 qubits by angle."""

    vertices = np.array(
        [_min_image(face_center, qubit_positions[q]) for q in face_qubits]
    )
    angles = np.arctan2(
        vertices[:, 1] - face_center[1], vertices[:, 0] - face_center[0]
    )
    order = np.argsort(angles)
    polygon = plt.Polygon(
        vertices[order],
        closed=True,
        facecolor=fill_color,
        edgecolor="none",
        alpha=0.55,
        zorder=1,
    )
    ax.add_patch(polygon)


def _draw_edge(
    ax: plt.Axes,
    q_a: int,
    q_b: int,
    color: str,
    qubit_positions: np.ndarray,
    stub_threshold: float = 1.6,
) -> None:
    """Draw an MPP edge between two qubits; emit dashed stubs when it wraps.

    Both qubit positions are assumed to already be centered in the fundamental
    domain. An edge is treated as a wrap iff the *direct* distance between the
    two centered positions exceeds ``stub_threshold`` (natural edge length is
    1.0 in these units).
    """

    pa = qubit_positions[q_a]
    pb = qubit_positions[q_b]
    if float(np.linalg.norm(pb - pa)) <= stub_threshold:
        ax.plot(
            [pa[0], pb[0]],
            [pa[1], pb[1]],
            "-",
            color=color,
            lw=2.5,
            zorder=3,
        )
        return

    pb_image = _min_image(pa, pb)
    pa_image = _min_image(pb, pa)
    stub_len = 0.45
    label_offset = 0.22
    for anchor, target_image, partner in (
        (pa, pb_image, q_b),
        (pb, pa_image, q_a),
    ):
        direction = target_image - anchor
        direction /= float(np.linalg.norm(direction))
        end = anchor + direction * stub_len
        ax.plot(
            [anchor[0], end[0]],
            [anchor[1], end[1]],
            "--",
            color=color,
            lw=2.0,
            alpha=0.8,
            zorder=3,
        )
        text_pos = anchor + direction * (stub_len + label_offset)
        ax.text(
            text_pos[0],
            text_pos[1],
            f"q{partner}",
            color=color,
            ha="center",
            va="center",
            fontsize=8,
            alpha=0.95,
            zorder=4,
        )


def generate_lattice_png(out_path: str = DEFAULT_OUTPUT) -> None:
    """Draw the 3x3 honeycomb (666) Floquet lattice with tri-colored faces."""

    fig, ax = plt.subplots(figsize=(11.0, 9.5))

    face_center: dict[tuple[str, int], np.ndarray] = {}
    for idx, coord in enumerate(R_FACE_COORDS):
        face_center[("red", idx)] = _hex_pos(coord)
    for idx, coord in enumerate(G_FACE_COORDS):
        face_center[("green", idx)] = _hex_pos(coord)
    for idx, coord in enumerate(B_FACE_COORDS):
        face_center[("blue", idx)] = _hex_pos(coord)

    qubit_positions = np.zeros((18, 2))
    for q, (ri, gi, bi) in enumerate(QUBIT_FACES):
        pr = face_center[("red", ri)]
        pg = _min_image(pr, face_center[("green", gi)])
        pb = _min_image(pr, face_center[("blue", bi)])
        qubit_positions[q] = (pr + pg + pb) / 3.0

    figure_center = 0.5 * (PERIOD1 + PERIOD2)
    for q in range(18):
        qubit_positions[q] = _min_image(figure_center, qubit_positions[q])
    for key in list(face_center.keys()):
        face_center[key] = _min_image(figure_center, face_center[key])

    for (color_name, idx), face_q in FACE_QUBITS.items():
        _draw_face(
            ax,
            face_center[(color_name, idx)],
            face_q,
            qubit_positions,
            LIGHT_COLORS[color_name],
        )

    for (color_name, idx), center in face_center.items():
        ax.text(
            center[0],
            center[1],
            f"{color_name[0].upper()}{idx}",
            color=COLORS[color_name],
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            alpha=0.6,
            zorder=2,
        )

    for q_a, q_b in RED_EDGES:
        _draw_edge(ax, q_a, q_b, COLORS["red"], qubit_positions)
    for q_a, q_b in GREEN_EDGES:
        _draw_edge(ax, q_a, q_b, COLORS["green"], qubit_positions)
    for q_a, q_b in BLUE_EDGES:
        _draw_edge(ax, q_a, q_b, COLORS["blue"], qubit_positions)

    for q in range(18):
        p = qubit_positions[q]
        ax.plot(
            p[0],
            p[1],
            "o",
            markersize=20,
            markerfacecolor="white",
            markeredgecolor="black",
            zorder=5,
        )
        ax.text(
            p[0],
            p[1],
            str(q),
            color="black",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            zorder=6,
        )

    handles = [
        mlines.Line2D(
            [], [], color=COLORS["red"], lw=3, label=r"Red edge ($XX$ = MPP $X_a X_b$)"
        ),
        mlines.Line2D(
            [],
            [],
            color=COLORS["green"],
            lw=3,
            label=r"Green edge ($YY$ = MPP $Y_a Y_b$)",
        ),
        mlines.Line2D(
            [],
            [],
            color=COLORS["blue"],
            lw=3,
            label=r"Blue edge ($ZZ$ = MPP $Z_a Z_b$)",
        ),
        mlines.Line2D(
            [], [], color="gray", lw=2.0, ls="--", label="Torus wrap (mod 3)"
        ),
        mpatches.Patch(
            facecolor=LIGHT_COLORS["red"],
            alpha=0.55,
            label=r"Red face: $X$-loop plaquette",
        ),
        mpatches.Patch(
            facecolor=LIGHT_COLORS["green"],
            alpha=0.55,
            label=r"Green face: $Y$-loop plaquette",
        ),
        mpatches.Patch(
            facecolor=LIGHT_COLORS["blue"],
            alpha=0.55,
            label=r"Blue face: $Z$-loop plaquette",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        framealpha=0.95,
        fontsize=9,
    )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "d=3 honeycomb (666) Floquet code on a 3x3 torus (18 qubits, 9 plaquettes)\n"
        "Edge color = MPP type (X/Y/Z); face color = plaquette type (X/Y/Z loop)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    -> {os.path.basename(out_path)}")


if __name__ == "__main__":
    generate_lattice_png()
