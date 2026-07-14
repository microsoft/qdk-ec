"""Render a before/after PNG diagram of the MZZ merge stabilizer group.

The output is committed to the repository (like ``teleport_timeline.png``
in the conditional-correction folder) and referenced by
``chapters/lattice-surgery.md``.  Regenerate manually::

    python stabilizer_flow.py

Uses Stim's ``detslice-svg`` renderer to draw the stabilizer group of
the merged code at two moments in the merge protocol:

* **Before merge** — two independent distance-3 rotated surface-code
  patches with their own 8-stabilizer groups, separated by the seam
  column.
* **After merge** — the same 21 qubits now form one merged code, whose
  20 stabilizers include 4 new bulk plaquettes spanning the seam and
  2 new Z 2-bodies completing the checkerboard at the top and bottom.

Rendering convention: Stim uses XYZ=RGB, so **red = X-stabilizer,
blue = Z-stabilizer**.
"""

import io
import os
import re

import cairosvg
import stim
from PIL import Image, ImageDraw, ImageFont

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(THIS_DIR, "mzz_stabilizer_flow.png")

# ── Qubit layout (matches ``00_lattice_surgery_library.deq``) ──────
#
#     cols   0   1   2   3   4   5   6
#     row 0  q0  q1  q2  q18 q9  q10 q11      patch A: q0..q8
#     row 1  q3  q4  q5  q19 q12 q13 q14      patch B: q9..q17
#     row 2  q6  q7  q8  q20 q15 q16 q17
COORDS: dict[int, tuple[float, float]] = {}
for r in range(3):
    for c in range(3):
        COORDS[r * 3 + c] = (c, r)              # patch A (cols 0-2)
        COORDS[9 + r * 3 + c] = (c + 4, r)      # patch B (cols 4-6)
for r in range(3):
    COORDS[18 + r] = (3, r)                      # seam column

# Stabilizer generators of the ``SurfaceCode [[9,1,3]]`` patch, offset
# for patch A (qubits 0-8) and patch B (qubits 9-17).
PATCH_A_STABS = [
    "Z1*Z2", "X0*X3", "Z0*Z1*Z3*Z4", "X1*X2*X4*X5",
    "X3*X4*X6*X7", "Z4*Z5*Z7*Z8", "X5*X8", "Z6*Z7",
]
PATCH_B_STABS = [
    "Z10*Z11", "X9*X12", "Z9*Z10*Z12*Z13", "X10*X11*X13*X14",
    "X12*X13*X15*X16", "Z13*Z14*Z16*Z17", "X14*X17", "Z15*Z16",
]
# Six new stabilizers measured by the merge: four bulk plaquettes
# spanning the seam plus two Z-type boundary 2-bodies completing the
# merged code's checkerboard.
MERGE_STABS = [
    "Z2*Z5*Z18*Z19",     # M0: bulk Z, rows 0-1 cols 2-3
    "X5*X8*X19*X20",     # M1: bulk X, rows 1-2 cols 2-3
    "X9*X12*X18*X19",    # M2: bulk X, rows 0-1 cols 3-4
    "Z12*Z15*Z19*Z20",   # M3: bulk Z, rows 1-2 cols 3-4
    "Z9*Z18",            # M4: top boundary,    cols 3-4
    "Z8*Z20",            # M5: bottom boundary, cols 2-3
]
# The X 2-bodies ``X5*X8`` (patch A right edge) and ``X9*X12`` (patch B
# left edge) get absorbed into the new bulk plaquettes ``X5*X8*X19*X20``
# and ``X9*X12*X18*X19`` respectively — they are NOT independent
# generators of the merged code.
MERGED_CODE_STABS = [
    s for s in PATCH_A_STABS + PATCH_B_STABS if s not in ("X5*X8", "X9*X12")
] + MERGE_STABS


def mpp_targets(term: str):
    """Convert a Pauli string like ``Z1*Z2`` into Stim MPP targets."""
    out: list = []
    parts = term.split("*")
    for i, p in enumerate(parts):
        p = p.strip()
        pauli, q = p[0], int(p[1:])
        if pauli == "Z":
            out.append(stim.target_z(q))
        elif pauli == "X":
            out.append(stim.target_x(q))
        elif pauli == "Y":
            out.append(stim.target_y(q))
        else:
            raise ValueError(f"Unknown Pauli letter {pauli!r} in {term!r}")
        if i < len(parts) - 1:
            out.append(stim.target_combiner())
    return out


def build_before_circuit() -> stim.Circuit:
    """Circuit for the ``tick=1`` "before merge" snapshot.

    Declares exactly 16 detectors, one per patch-A/B stabilizer.  Each
    detector contributes a small magenta ring around every data qubit
    it touches; keeping the detector set minimal keeps the rings from
    piling up into thick blobs.
    """
    circuit = stim.Circuit()
    for q, (x, y) in sorted(COORDS.items()):
        circuit.append("QUBIT_COORDS", [q], (x, y))
    circuit.append("R", sorted(COORDS.keys()))
    circuit.append("TICK")   # tick=1 (before merge)

    pre_stabs = PATCH_A_STABS + PATCH_B_STABS
    for term in pre_stabs:
        circuit.append("MPP", mpp_targets(term))
    for i in range(len(pre_stabs)):
        circuit.append("DETECTOR", [stim.target_rec(-len(pre_stabs) + i)])
    return circuit


def build_after_circuit() -> stim.Circuit:
    """Circuit for the ``tick=1`` "after merge" snapshot.

    The pre-merge round and merge measurements happen *before* TICK #1
    (undeclared, so they contribute no detectors); a full merged-code
    SE round happens after, declaring exactly 20 detectors — one per
    generator of the merged code, so ``detslice-svg`` renders the
    merged 3x7 code cleanly.
    """
    circuit = stim.Circuit()
    for q, (x, y) in sorted(COORDS.items()):
        circuit.append("QUBIT_COORDS", [q], (x, y))
    circuit.append("R", sorted(COORDS.keys()))
    for term in PATCH_A_STABS + PATCH_B_STABS:
        circuit.append("MPP", mpp_targets(term))
    circuit.append("RX", [18, 19, 20])
    for term in MERGE_STABS:
        circuit.append("MPP", mpp_targets(term))
    circuit.append("TICK")   # tick=1 (after merge)

    for term in MERGED_CODE_STABS:
        circuit.append("MPP", mpp_targets(term))
    for i in range(len(MERGED_CODE_STABS)):
        circuit.append(
            "DETECTOR", [stim.target_rec(-len(MERGED_CODE_STABS) + i)]
        )
    return circuit


def render_detslice_png(circuit: stim.Circuit, tick: int, width: int) -> Image.Image:
    """Render a ``detslice-svg`` at *tick* and rasterise to a PIL Image.

    Two SVG post-processing steps happen before rasterisation:

    * The per-detector magenta rings that Stim overlays on every data
      qubit in a rendered detector's Pauli support are stripped —
      they duplicate the colored polygons and pile up into visual
      noise on qubits shared by many stabilizers.
    * Each qubit-dot ``<circle id="qubit_dot:N:x_y:1"/>`` gets a
      matching ``<text>`` label containing the qubit index ``N``,
      positioned just to the upper-right of the dot.
    """
    svg = str(circuit.diagram("detslice-svg", tick=tick))
    svg = re.sub(r'<circle[^/]*stroke="magenta"[^/]*/>\s*', "", svg)
    svg = _inject_qubit_labels(svg)
    png_bytes = cairosvg.svg2png(bytestring=svg.encode(), output_width=width)
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")


_QUBIT_DOT_RE = re.compile(
    r'<circle\s+id="qubit_dot:(?P<idx>\d+):[^"]*"\s+'
    r'cx="(?P<cx>[-\d.]+)"\s+cy="(?P<cy>[-\d.]+)"[^/]*/>'
)


def _inject_qubit_labels(svg: str) -> str:
    """Insert a small ``<text>`` label with the qubit index next to
    each ``qubit_dot`` circle already present in the Stim SVG."""
    labels: list[str] = []
    for m in _QUBIT_DOT_RE.finditer(svg):
        idx = int(m.group("idx"))
        cx = float(m.group("cx"))
        cy = float(m.group("cy"))
        # Position label just above-right of the dot (SVG units:
        # ~32 units per qubit spacing, so a 6-unit font is unobtrusive).
        labels.append(
            f'<text x="{cx + 3}" y="{cy - 3}" font-family="sans-serif" '
            f'font-size="7" font-weight="bold" fill="black" '
            f'stroke="white" stroke-width="0.4" paint-order="stroke">'
            f'{idx}</text>'
        )
    if not labels:
        return svg
    # Inject labels just before the closing </svg> so they render on top
    # of everything else.
    return svg.replace("</svg>", "\n".join(labels) + "\n</svg>", 1)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a bold sans-serif font, falling back to the default bitmap.
    """
    for candidate in (
        "DejaVuSans-Bold.ttf",                                    # Linux default
        "arialbd.ttf",                                             # Windows: Arial Bold
        "Arial Bold.ttf",                                          # macOS
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def main() -> None:
    panel_width = 900
    title_height = 60
    padding = 30

    before = render_detslice_png(build_before_circuit(), tick=1, width=panel_width)
    after = render_detslice_png(build_after_circuit(), tick=1, width=panel_width)

    # Both panels have the same width; equalise the height by padding the
    # shorter panel so the composed figure has a rectangular canvas.
    max_h = max(before.height, after.height)
    def pad_to_height(img: Image.Image) -> Image.Image:
        if img.height == max_h:
            return img
        canvas = Image.new("RGBA", (img.width, max_h), (255, 255, 255, 255))
        canvas.paste(img, (0, (max_h - img.height) // 2), img)
        return canvas
    before = pad_to_height(before)
    after = pad_to_height(after)

    total_w = padding + panel_width + padding + panel_width + padding
    total_h = padding + title_height + max_h + padding
    canvas = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 255))
    canvas.paste(before, (padding, padding + title_height), before)
    canvas.paste(after, (padding + panel_width + padding, padding + title_height), after)

    draw = ImageDraw.Draw(canvas)
    title_font = load_font(30)
    subtitle_font = load_font(20)

    def centered(x0: int, x1: int, text: str, y: int, font) -> None:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw = right - left
        draw.text(((x0 + x1 - tw) // 2, y), text, fill=(0, 0, 0, 255), font=font)

    centered(padding, padding + panel_width,
             "Before merge", padding // 2, title_font)
    centered(padding, padding + panel_width,
             "two independent d=3 patches (16 stabilizers)",
             padding // 2 + 35, subtitle_font)
    centered(padding * 2 + panel_width, padding * 2 + panel_width * 2,
             "After merge", padding // 2, title_font)
    centered(padding * 2 + panel_width, padding * 2 + panel_width * 2,
             "merged 3x7 code (20 stabilizers, 6 new)",
             padding // 2 + 35, subtitle_font)

    canvas.convert("RGB").save(OUTPUT_PNG, "PNG", optimize=True)
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
