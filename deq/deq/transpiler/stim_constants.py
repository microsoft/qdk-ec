"""Shared Stim instruction name constants and helpers for the deq transpiler.

Classification sets are derived from ``stim.gate_data()`` at import time
so they automatically stay in sync with the installed Stim version.
"""

import stim

_GATE_DATA = stim.gate_data()
_ALL_STIM_NAMES: frozenset[str] = frozenset(
    alias for g in _GATE_DATA.values() for alias in g.aliases
)

# ── Derived from stim.gate_data() ───────────────────────────────────

# Pure noise channels (no measurement side-effects).
# Measurements with optional noise arguments (e.g. M(0.001)) are NOT
# included here — they are in MEASUREMENT_INSTRUCTIONS. Support for
# extracting noise from measurement arguments is a separate concern.
NOISE_INSTRUCTIONS: frozenset[str] = frozenset(
    alias
    for g in _GATE_DATA.values()
    if g.is_noisy_gate and not g.produces_measurements
    for alias in g.aliases
)

# Single-qubit gates that produce measurement results (M, MR, MX, etc.).
# Excludes heralded noise channels (HERALDED_ERASE, etc.) which require
# a probability argument (num_parens_arguments_range starts at > 0).
MEASUREMENT_INSTRUCTIONS: frozenset[str] = frozenset(
    alias
    for g in _GATE_DATA.values()
    if (
        g.produces_measurements
        and g.is_single_qubit_gate
        and g.num_parens_arguments_range.start == 0
    )
    for alias in g.aliases
)

TWO_QUBIT_MEASUREMENT_INSTRUCTIONS: frozenset[str] = frozenset(
    alias
    for g in _GATE_DATA.values()
    if g.produces_measurements and g.is_two_qubit_gate
    for alias in g.aliases
)

# ── Gate classifications (derived from stim.gate_data()) ────────────

# Single-qubit unitary (Clifford) gates — includes Paulis.
ONE_QUBIT_GATES: frozenset[str] = frozenset(
    alias
    for g in _GATE_DATA.values()
    if g.is_unitary and g.is_single_qubit_gate
    for alias in g.aliases
)

# Two-qubit unitary (Clifford) gates.
TWO_QUBIT_GATES: frozenset[str] = frozenset(
    alias
    for g in _GATE_DATA.values()
    if g.is_unitary and g.is_two_qubit_gate
    for alias in g.aliases
)

# Pair measurement gates (two-qubit measurements like MXX, MYY, MZZ).
PAIR_MEASURE_GATES: frozenset[str] = frozenset(
    alias
    for g in _GATE_DATA.values()
    if g.produces_measurements and g.is_two_qubit_gate
    for alias in g.aliases
)

# Generalized Pauli product gates (MPP, SPP, SPP_DAG).
PAULI_PRODUCT_GATES: frozenset[str] = frozenset(
    alias
    for g in _GATE_DATA.values()
    if g.takes_pauli_targets
    and not g.takes_measurement_record_targets
    and (g.is_unitary or g.produces_measurements)
    for alias in g.aliases
)

# ── Measurement/reset basis classification ───────────────────────────

MEASUREMENTS_Z: frozenset[str] = frozenset({"M", "MZ"})
MEASUREMENTS_X: frozenset[str] = frozenset({"MX"})
MEASUREMENTS_Y: frozenset[str] = frozenset({"MY"})
MEASURE_RESETS_Z: frozenset[str] = frozenset({"MR", "MRZ"})
MEASURE_RESETS_X: frozenset[str] = frozenset({"MRX"})
MEASURE_RESETS_Y: frozenset[str] = frozenset({"MRY"})
RESETS_Z: frozenset[str] = frozenset({"R", "RZ"})
RESETS_X: frozenset[str] = frozenset({"RX"})
RESETS_Y: frozenset[str] = frozenset({"RY"})

# Combined basis sets (measure-only ∪ measure-reset, reset-only ∪ measure-reset)
Z_BASIS_MEASURE: frozenset[str] = MEASUREMENTS_Z | MEASURE_RESETS_Z
X_BASIS_MEASURE: frozenset[str] = MEASUREMENTS_X | MEASURE_RESETS_X
Y_BASIS_MEASURE: frozenset[str] = MEASUREMENTS_Y | MEASURE_RESETS_Y
Z_BASIS_RESET: frozenset[str] = RESETS_Z | MEASURE_RESETS_Z
X_BASIS_RESET: frozenset[str] = RESETS_X | MEASURE_RESETS_X
Y_BASIS_RESET: frozenset[str] = RESETS_Y | MEASURE_RESETS_Y

# ── Annotation / control-flow instructions (no-ops for gate walkers) ─

ANNOTATION_INSTRUCTIONS: frozenset[str] = frozenset(
    {"TICK", "QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"}
)

# All instructions that produce measurements and may carry an optional
# noise probability argument (e.g. ``M(0.01)``, ``MPP(0.001)``).
NOISY_MEASUREMENT_INSTRUCTIONS: frozenset[str] = (
    MEASUREMENT_INSTRUCTIONS | TWO_QUBIT_MEASUREMENT_INSTRUCTIONS | frozenset({"MPP"})
)


# ── Target helpers ───────────────────────────────────────────────────

from deq.circuit.model import (
    CombinerTarget,
    Instruction,
    PauliTarget,
    QubitTarget,
    Target,
)


def qubit_indices(inst: Instruction) -> list[int]:
    """Extract qubit index integers from an instruction's targets."""
    return [t.index for t in inst.targets if isinstance(t, QubitTarget)]


def split_mpp_targets(targets: list[Target]) -> list[list[PauliTarget]]:
    """Split an MPP instruction's target list into per-product groups.

    Products are separated at boundaries where two consecutive
    ``PauliTarget`` entries are *not* joined by a ``CombinerTarget``.
    Within a product, ``CombinerTarget`` tokens (``*``) link the terms.
    """
    groups: list[list[PauliTarget]] = [[]]
    prev_was_combiner = False
    for t in targets:
        if isinstance(t, CombinerTarget):
            prev_was_combiner = True
        elif isinstance(t, PauliTarget):
            if not prev_was_combiner and groups[-1]:
                groups.append([])
            groups[-1].append(t)
            prev_was_combiner = False
        else:
            raise ValueError(f"Unexpected target type in MPP instruction: {t!r}")
    return [g for g in groups if g]


def mpp_measurement_count(targets: list[Target]) -> int:
    """Return the number of measurement results an MPP instruction produces."""
    return len(split_mpp_targets(targets))
