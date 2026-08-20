"""Shared Stim constants and conversion helpers for the deq transpiler.

Classification sets are derived from ``stim.gate_data()`` at import time
so they automatically stay in sync with the installed Stim version.
"""

from collections.abc import Iterable

import stim
from paulimer import SparsePauli

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

# Non-Stim noise instructions that deq accepts verbatim inside gadget bodies and
# passes through to the generated ``.stim`` with the usual qubit-target
# relabeling. These instructions:
#
# * are treated the same as :data:`NOISE_INSTRUCTIONS` by every deq
#   transpiler pass that *skips* noise (gate decomposition, hypergraph
#   construction, annotation walks, …),
# * produce **no hyperedges** in the JIT noise builder,
# * are assumed to produce **zero measurement bits** (so any
#   measurement-counting pass returns 0 for them).
#
# ``LOSS_ERROR(p) q...`` is QDK's Stim extension that injects persistent loss at
# that exact circuit location. Adding it here lets users write loss-aware
# circuits directly in ``.deq``; the deq runtime itself does not interpret the
# instruction, but ``qdk.stim`` (driven via ``--simulator python``) does.
PASSTHROUGH_NOISE_INSTRUCTIONS: frozenset[str] = frozenset({"LOSS_ERROR"})

# Union of all instruction names that every deq transpiler pass that
# already skips :data:`NOISE_INSTRUCTIONS` should also skip.  Prefer
# this set in callers that simply want "anything that looks like a
# noise channel" — including QDK-style passthrough extensions.
NOISE_INSTRUCTIONS_ALL: frozenset[str] = (
    NOISE_INSTRUCTIONS | PASSTHROUGH_NOISE_INSTRUCTIONS
)


def instruction_num_measurements(instruction_text: str) -> int:
    """Count measurement bits produced by a single stim instruction.

    Delegates to ``stim.CircuitInstruction(...).num_measurements`` for
    instructions upstream Stim recognizes. For ``LOSS_ERROR``, which upstream
    Stim rejects with ``Gate not found``, returns ``0`` because it contributes no
    measurement bits.

    Use this helper anywhere we used to call
    ``stim.CircuitInstruction(str(stmt)).num_measurements`` on a
    user-authored instruction; otherwise circuits containing
    ``LOSS_ERROR`` will crash the transpiler.
    """
    head = instruction_text.split(None, 1)
    if head:
        name = head[0].split("[", 1)[0].split("(", 1)[0].upper()
        if name in PASSTHROUGH_NOISE_INSTRUCTIONS:
            return 0
    return stim.CircuitInstruction(instruction_text).num_measurements


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
    PauliProduct,
    PauliTarget,
    QubitTarget,
    Target,
)


# ── Pauli conversion helpers ────────────────────────────────────────

_PAULI_TO_INT: dict[str, int] = {"I": 0, "X": 1, "Y": 2, "Z": 3}
_INT_TO_PAULI: tuple[str, ...] = ("I", "X", "Y", "Z")


def pauli_name_to_int(pauli: str) -> int:
    """Return Stim's integer encoding for a Pauli name."""
    return _PAULI_TO_INT[pauli.upper()]


def pauli_terms_to_stim(
    terms: Iterable[tuple[int, str]], num_qubits: int
) -> stim.PauliString:
    """Build a ``stim.PauliString`` from ``(qubit, Pauli)`` terms."""
    result = stim.PauliString(num_qubits)
    for qubit, pauli in terms:
        if qubit < 0 or qubit >= num_qubits:
            raise ValueError(
                f"qubit index {qubit} out of range for gadget with {num_qubits} "
                f"qubit(s) (valid range: 0..{num_qubits - 1})"
            )
        result[qubit] = pauli_name_to_int(pauli)
    return result


def single_pauli_to_stim(
    pauli: str, qubit: int, num_qubits: int
) -> stim.PauliString:
    """Build a ``stim.PauliString`` containing one specified Pauli."""
    return pauli_terms_to_stim(((qubit, pauli),), num_qubits)


def pauli_pair_to_stim(
    first_pauli: str,
    first_qubit: int,
    second_pauli: str,
    second_qubit: int,
    num_qubits: int,
) -> stim.PauliString:
    """Build a ``stim.PauliString`` containing two specified Paulis."""
    return pauli_terms_to_stim(
        ((first_qubit, first_pauli), (second_qubit, second_pauli)),
        num_qubits,
    )


def pauli_product_to_stim(
    product: PauliProduct,
    num_qubits: int,
    local_to_global: dict[int, int] | None = None,
) -> stim.PauliString:
    """Convert a circuit-model ``PauliProduct`` to a ``stim.PauliString``."""
    return pauli_terms_to_stim(
        (
            (
                local_to_global[term.index]
                if local_to_global is not None
                else term.index,
                term.pauli,
            )
            for term in product.terms
        ),
        num_qubits,
    )


def pauli_string_to_symplectic(
    pauli: stim.PauliString, num_qubits: int
) -> list[int]:
    """Encode a Pauli string as a ``[X | Z]`` symplectic bit vector."""
    xs, zs = pauli.to_numpy(bit_packed=False)
    included_qubits = min(len(xs), num_qubits)
    padding = [0] * (num_qubits - included_qubits)
    return (
        [int(bit) for bit in xs[:included_qubits]]
        + padding
        + [int(bit) for bit in zs[:included_qubits]]
        + padding
    )


def pauli_string_to_sparse(pauli: stim.PauliString) -> SparsePauli:
    """Convert a ``stim.PauliString`` to a paulimer ``SparsePauli``."""
    return SparsePauli(
        {
            qubit: _INT_TO_PAULI[pauli[qubit]]
            for qubit in range(len(pauli))
            if pauli[qubit]
        }
    )


def format_pauli_string(pauli: stim.PauliString) -> str:
    """Format a ``stim.PauliString`` as indexed Pauli terms."""
    terms = [
        f"{_INT_TO_PAULI[pauli[qubit]]}{qubit}"
        for qubit in range(len(pauli))
        if pauli[qubit]
    ]
    if not terms:
        return "I"
    sign = "-" if pauli.sign == -1 else ""
    return sign + "*".join(terms)


# ── Target helpers ───────────────────────────────────────────────────


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
