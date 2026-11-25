"""Expand stim noise channels in a gadget body into JIT ``Error`` rows,
and compute the ``correction_propagation`` and ``physical_correction``
matrices that the runtime uses to track logical observables across
gadget boundaries.

A noise channel like ``DEPOLARIZE1(p) 0 1`` is a probability distribution
over Pauli operators; for the JIT decoder we need each independent
mechanism (each Pauli that the channel can sample) as its own ``Error``
entry, with its footprint on the gadget's checks, readouts, and output
observables computed by forward-propagating the Pauli through every
remaining instruction in the body.

Observable residuals use the symplectic column convention defined in
:mod:`deq.transpiler.jit_transpiler`: column ``2k`` is the X observable
of logical qubit ``k``, column ``2k+1`` is the Z observable.  The
``row ^ 1`` idiom selects the symplectic partner for anticommutation
tests.

Two complementary techniques drive the computation:

- **Batched frame propagation**
  (:func:`~deq.transpiler.fault_propagation.propagate_pauli_mechanisms`)
  pushes many Paulis through the body in one pass, recording which real
  measurements each flips.  Used by :func:`compute_noise_errors` (each
  noise mechanism's footprint) and by
  :func:`compute_implicit_readout_propagation` (input-port-observable
  readouts).
- **Choi construction** (:func:`_compute_pc_logical_via_flows`)
  drives **all** logical-row entries of ``correction_propagation`` and
  ``physical_correction``.  One stabilizer simulation, with reference
  qubits for the input logicals, finds every relation between products
  of input and output frame observables and body measurement outcomes;
  each yields one linear constraint on the cp / pc / ``FLIP`` rows.
  The relations are reduced to a canonical form and solved for all
  columns at once through a single echelon form.  This unifies unitary
  bodies, bodies with internal measurements (e.g. Floquet honeycomb
  rounds), and multi-port merges with joint-observable invariants
  (e.g. lattice-surgery :math:`X_A X_B` preservation).

This module provides:

- :func:`enumerate_noise_mechanisms` — given a single noise instruction,
  yield ``(pauli_at_site, probability)`` pairs.
- :func:`compute_correction_propagation` — assemble the cp matrix
  from flow analysis (logical rows), unfinished checks (stab rows),
  and virtual logical statements (FLIP).
- :func:`compute_noise_errors` — orchestrate the above to emit a list
  of ``JitGadgetType.Error`` rows.
"""

from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, Sequence

import stim
from binar import BitMatrix, EchelonForm, null_space
from paulimer import OutcomeCompleteSimulation, SparsePauli

import deq.proto.deq_bin_pb2 as pb
import deq.proto.deq_jit_pb2 as jit_pb
import deq.proto.util_pb2 as util_pb
from deqagram import (
    CodeDefinition,
    DestabilizerTarget,
    GadgetDefinition,
    InputPort,
    Instruction,
    LogicalPauliTarget,
    LossTarget,
    MeasurementRecordTarget,
    OutputPort,
    PhysicalMeasurementTarget,
    PropagateStatement,
    ReadoutTarget,
    VirtualLogicalStatement,
)
from deq.transpiler.circuit_lowering import (
    DecomposedBody,
    build_decomposed_body,
    flatten_body,
    max_qubit_index,
)
from deq.transpiler.fault_propagation import (
    ErrorProjectionContext,
    MechanismFlips,
    build_error_projection_context,
    build_error_row_from_flips,
    build_port_paulis,
    propagate_pauli_mechanisms,
)
from deq.spec.common import bitmatrix_from_sparse
from deq.transpiler.jit_transpiler import (
    Check,
    PortColumnLayout,
    _apply_decomposed_instructions,
    _BuildState,
    resolve_measurement_ref_global,
    select_stabilizer_generators,
)
from deq.transpiler.stim_constants import (
    CorrelatedErrorChain,
    NOISE_INSTRUCTIONS,
    NOISE_INSTRUCTIONS_ALL,
    PASSTHROUGH_NOISE_INSTRUCTIONS,
    instruction_num_measurements,
    pauli_pair_to_stim,
    pauli_product_to_stim,
    pauli_string_to_sparse,
    pauli_terms_to_stim,
    single_pauli_to_stim,
)
from deq.transpiler.stim_constants import qubit_indices as _qubit_indices

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _real_measurement_count(instr: Instruction) -> int:
    """Return the number of real measurements an instruction performs."""
    name = instr.name.upper()
    if name in PASSTHROUGH_NOISE_INSTRUCTIONS:
        return 0
    if name in ("HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"):
        raise ValueError(
            f"Heralded instruction '{name}' is not supported by deq. "
            f"Heralded noise channels produce measurements that require "
            f"erasure decoding, which is not yet implemented."
        )
    return instruction_num_measurements(str(instr))


# ---------------------------------------------------------------------------
# Noise mechanism enumeration
# ---------------------------------------------------------------------------


def enumerate_noise_mechanisms(
    instr: Instruction,
    num_qubits: int,
    *,
    else_chain_remaining: float = 1.0,
) -> list[tuple[stim.PauliString, float]]:
    """Decompose a stim noise instruction into independent Pauli mechanisms.

    Each returned ``(pauli, prob)`` pair represents one independent
    mechanism that, with marginal probability ``prob``, applies ``pauli``
    just before the noise instruction's position. Identity components
    and zero-probability mechanisms are dropped.

    Notes:
    - ``ELSE_CORRELATED_ERROR(p)`` is conditional on the preceding
      ``CORRELATED_ERROR`` / ``ELSE_CORRELATED_ERROR`` chain not having
      fired. The caller passes ``else_chain_remaining`` — the probability
      that no error in the current else-chain has fired so far — and the
      returned marginal probability is ``p * else_chain_remaining``.
      ``CORRELATED_ERROR`` / ``E`` start a new chain and ignore
      ``else_chain_remaining`` (their marginal is the literal ``p``).
    - ``PAULI_CHANNEL_1`` / ``PAULI_CHANNEL_2`` arguments are taken
      directly as independent mechanism probabilities. A general Pauli
      channel has no exact independent-mechanism representation, so this
      is the same independent-error approximation (unlike
      ``DEPOLARIZE1/2``, which are converted exactly).
    - ``I_ERROR`` / ``II_ERROR`` produce no mechanisms.
        - Loss targets in a correlated instruction are handled by the loss pass;
            any accompanying Pauli targets still form one ordinary Pauli mechanism.
    """
    name = instr.name.upper()
    if name in PASSTHROUGH_NOISE_INSTRUCTIONS:
        # Passthrough noise extensions (e.g. ``LOSS_ERROR``) are emitted
        # verbatim in the .stim output but contribute no detector edges
        # to the decoding hypergraph — deq's decoder side does not (yet)
        # consume them.  Surfacing them through the JIT noise builder as
        # "no mechanisms" lets users freely sprinkle them into gadget
        # bodies without breaking hypergraph construction.
        return []
    if name not in NOISE_INSTRUCTIONS:
        raise ValueError(f"{name} is not a recognised noise instruction")

    qubits = _qubit_indices(instr)
    args = list(instr.arguments)

    if name in {"I_ERROR", "II_ERROR"}:
        return []

    if name in {"X_ERROR", "Y_ERROR", "Z_ERROR"}:
        if len(args) != 1:
            raise ValueError(f"{name} expects exactly one probability argument")
        prob = float(args[0])
        pauli = name[0]
        return [
            (single_pauli_to_stim(pauli, q, num_qubits), prob)
            for q in qubits
            if prob > 0
        ]

    if name == "DEPOLARIZE1":
        if len(args) != 1:
            raise ValueError("DEPOLARIZE1 expects exactly one probability argument")
        p = float(args[0])
        if p > 0.75:
            raise ValueError("DEPOLARIZE1 probability must be at most 3/4")
        # Exact conversion to independent mechanisms: three mechanisms of
        # probability q compose (by XOR in the Pauli group) to a depolarizing
        # channel with per-Pauli probability q(1-q); solve q(1-q) = p/3.
        prob = (1.0 - (1.0 - 4.0 * p / 3.0) ** 0.5) / 2.0
        if prob <= 0:
            return []
        out: list[tuple[stim.PauliString, float]] = []
        for q in qubits:
            for pauli in ("X", "Y", "Z"):
                out.append((single_pauli_to_stim(pauli, q, num_qubits), prob))
        return out

    if name == "DEPOLARIZE2":
        if len(args) != 1:
            raise ValueError("DEPOLARIZE2 expects exactly one probability argument")
        if len(qubits) % 2 != 0:
            raise ValueError("DEPOLARIZE2 requires an even number of qubit targets")
        p = float(args[0])
        if p > 15.0 / 16.0:
            raise ValueError("DEPOLARIZE2 probability must be at most 15/16")
        # Exact conversion (same identity as DEPOLARIZE1, over the 15
        # non-identity two-qubit Paulis): solve via the n=2 Walsh inversion.
        prob = (1.0 - (1.0 - 16.0 * p / 15.0) ** 0.125) / 2.0
        if prob <= 0:
            return []
        out = []
        for i in range(0, len(qubits), 2):
            q1, q2 = qubits[i], qubits[i + 1]
            for p1 in ("I", "X", "Y", "Z"):
                for p2 in ("I", "X", "Y", "Z"):
                    if p1 == "I" and p2 == "I":
                        continue
                    out.append(
                        (pauli_pair_to_stim(p1, q1, p2, q2, num_qubits), prob)
                    )
        return out

    if name == "PAULI_CHANNEL_1":
        if len(args) != 3:
            raise ValueError(
                "PAULI_CHANNEL_1 expects exactly three probability arguments "
                "(pX, pY, pZ)"
            )
        probs = {"X": float(args[0]), "Y": float(args[1]), "Z": float(args[2])}
        out = []
        for q in qubits:
            for pauli, prob in probs.items():
                if prob > 0:
                    out.append((single_pauli_to_stim(pauli, q, num_qubits), prob))
        return out

    if name == "PAULI_CHANNEL_2":
        # 15 args, in stim's order: IX, IY, IZ, XI, XX, XY, XZ, YI, YX,
        # YY, YZ, ZI, ZX, ZY, ZZ (all 16 two-qubit Paulis except II).
        if len(args) != 15:
            raise ValueError("PAULI_CHANNEL_2 expects exactly 15 probability arguments")
        if len(qubits) % 2 != 0:
            raise ValueError("PAULI_CHANNEL_2 requires an even number of qubit targets")
        labels = [
            "IX",
            "IY",
            "IZ",
            "XI",
            "XX",
            "XY",
            "XZ",
            "YI",
            "YX",
            "YY",
            "YZ",
            "ZI",
            "ZX",
            "ZY",
            "ZZ",
        ]
        out = []
        for i in range(0, len(qubits), 2):
            q1, q2 = qubits[i], qubits[i + 1]
            for label, raw_prob in zip(labels, args):
                prob = float(raw_prob)
                if prob <= 0:
                    continue
                out.append(
                    (
                        pauli_pair_to_stim(
                            label[0], q1, label[1], q2, num_qubits
                        ),
                        prob,
                    )
                )
        return out

    if name in {"E", "CORRELATED_ERROR", "ELSE_CORRELATED_ERROR"}:
        if len(args) != 1:
            raise ValueError(f"{name} expects exactly one probability argument")
        prob = float(args[0])
        if name == "ELSE_CORRELATED_ERROR":
            prob *= else_chain_remaining
        if prob <= 0:
            return []
        # Targets are Pauli targets like X3 Y4 Z5 (parsed as PauliTarget).
        # Build a single PauliString from them.
        terms: list[tuple[int, str]] = []
        for target in instr.targets:
            if isinstance(target, LossTarget):
                continue
            # PauliTarget has .pauli ("X"/"Y"/"Z") and .index.
            pauli = getattr(target, "pauli", None)
            index = getattr(target, "index", None)
            if pauli is None or index is None:
                raise ValueError(f"{name} target {target!r} is not a Pauli target")
            if index >= num_qubits:
                raise ValueError(
                    f"{name} target {target} references qubit {index} but the "
                    f"gadget body only references qubits 0..{num_qubits - 1}"
                )
            terms.append((index, pauli))
        return [(pauli_terms_to_stim(terms, num_qubits), prob)] if terms else []

    raise ValueError(f"Unsupported noise instruction: {name}")


# ---------------------------------------------------------------------------
# Output-observable relations for measurement-bearing bodies
# ---------------------------------------------------------------------------
#
# When a gadget body contains internal measurements (e.g. a Floquet
# round, a syndrome-extraction sequence, a teleportation gadget), the
# value of an output logical observable :math:`O_\\text{out}` after
# the body is determined by the operator equation
#
# .. math::
#     O_\\text{out}
#         \\;=\\; \\Big(\\bigoplus_{c \\in C_r} I_c\\Big)
#         \\;\\oplus\\; \\bigoplus_{j \\in M_r} m_j
#         \\;\\oplus\\; \\mathrm{FLIP}_r,
#
# where :math:`I_c` ranges over input-frame column observables (both
# logical and stabilizer cols), :math:`m_j` over the body's internal
# measurement outcomes, and :math:`\\mathrm{FLIP}_r` is a constant
# sign offset absorbed by the affine ``FLIP`` column of
# ``correction_propagation``.
#
# A *relation* ``(u, v, w, sigma)`` states that, for every input
# state, the product of the output frame observables selected by
# ``v`` equals the product of the input frame observables selected by
# ``u``, times the measurement outcomes selected by ``w``, with sign
# ``sigma``.  The simulation uses the Choi construction on the logical
# inputs. The body qubits start maximally mixed. For each input logical L,
# measure X_L X_R and Z_L Z_R with its own reference qubit R and record
# both signs. The gadget never acts on R. Input stabilizers are measured
# directly. After the body, a product of output and reference observables
# gives a relation if it stabilizes the final state with a sign determined
# by the recorded input and body outcomes.
#
# Relations are reduced to a canonical form, with pivots on the
# latest output and input columns and on the earliest measurement of
# each check, and solved with free variables set to zero.  Output
# rows that are only jointly determined (e.g. lattice-surgery
# :math:`\\bar X_A \\bar X_B` preservation) thus get a fixed,
# self-consistent anchor.  When an output observable is not
# determined at all (e.g. a freshly prepared logical with no
# deterministic pre-image), its row is left empty; the runtime
# treats the observable's value as the default constant, which is
# correct as long as no downstream gadget consumes the observable's
# specific value.


def _input_column_references(
    input_ports: Sequence[InputPort], codes: dict[str, CodeDefinition]
) -> tuple[list[int | None], int]:
    """Return the reference slot of each input frame column and the slot count.

    Logical qubit ``k`` owns slots ``2k`` (X) and ``2k + 1`` (Z), matching
    the column layout of :func:`build_port_paulis`; stabilizer columns
    have no reference.
    """
    slots: list[int | None] = []
    logical_count = 0
    for port in input_ports:
        code = codes[port.code_name]
        for _ in code.logicals:
            slots.extend((2 * logical_count, 2 * logical_count + 1))
            logical_count += 1
        slots.extend(
            None for _ in select_stabilizer_generators(code).generator_indices
        )
    return slots, logical_count


def _sign_bit(phase: complex) -> int:
    if abs(phase.imag) > 1e-9 or abs(abs(phase.real) - 1) > 1e-9:
        raise RuntimeError(
            f"jit_noise_builder: relation has non-real sign {phase!r}; "
            f"algebra bug."
        )
    return int(phase.real < 0)


def _compute_pc_logical_via_flows(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
    *,
    input_ports: Sequence[InputPort],
    output_ports: Sequence[OutputPort],
    output_layout: PortColumnLayout,
) -> tuple[list[tuple[int, int]], set[tuple[int, int]], set[int]]:
    """Compute logical correction rows from a Choi stabilizer simulation.

    Returns ``(pc_entries, cp_entries, flip_entries)`` covering all
    output logical rows:

    * ``pc_entries`` — list of ``(output_logical_row, body_meas_index)``
      entries to install in ``physical_correction``.
    * ``cp_entries`` — set of ``(output_logical_row, input_col)``
      entries to install in ``correction_propagation``.  Covers all
      input cols (logical and stabilizer alike).
    * ``flip_entries`` — set of ``output_logical_row`` whose ``FLIP``
      (affine) column must be set to absorb the relation's sign.

    Algorithm (see the comment above for the math):

    1. Simulate the body from a maximally mixed input, with reference
       qubits for the input logicals, recording the input frame and
       body measurement outcomes.
    2. Find the products of output frame and reference observables
       that stabilize the final state, and keep the combinations whose
       sign depends only on recorded outcomes.  Each is a relation
       ``(u, v, w)``.
    3. Reduce the relations (latest output and input columns first,
       then earliest measurements) and keep those with an output part
       ``v``.  The reduced form is unique, so the result does not
       depend on how the relations were found.
    4. Solve ``V · [cp | pc | flip] = [U | W | sigma]`` with free
       variables set to zero, signing only the relations that the
       logical rows use.
    """
    body_flat = flatten_body(list(gadget.body))
    decomposed = build_decomposed_body(body_flat)
    num_qubits = decomposed.qubit_count
    if num_qubits == 0:
        return [], set(), set()

    _, input_frame_column_paulis = build_port_paulis(
        list(input_ports), codes, num_qubits
    )
    _, output_frame_column_paulis = build_port_paulis(
        list(output_ports), codes, num_qubits
    )
    n_in = len(input_frame_column_paulis)
    n_out = len(output_frame_column_paulis)
    if n_out == 0:
        return [], set(), set()
    reference_slots, logical_count = _input_column_references(input_ports, codes)

    def reference(slot: int) -> SparsePauli:
        qubit = num_qubits + slot // 2
        return SparsePauli.z(qubit) if slot % 2 else SparsePauli.x(qubit)

    total_qubits = num_qubits + logical_count
    sim = OutcomeCompleteSimulation(total_qubits)
    for qubit in range(num_qubits):
        for twirl in (SparsePauli.x(qubit), SparsePauli.z(qubit)):
            sim.apply_conditional_pauli(twirl, [sim.allocate_random_bit()], parity=True)

    measured_inputs: list[SparsePauli] = []
    input_outcomes: list[int] = []
    for pauli, slot in zip(input_frame_column_paulis, reference_slots):
        observable = pauli_string_to_sparse(pauli)
        if slot is not None:
            observable = observable * reference(slot)
        measured_inputs.append(observable)
        input_outcomes.append(sim.measure(observable))

    state = _BuildState(sim=sim)
    _apply_decomposed_instructions(state, decomposed.instructions)
    measurement_outcomes = state.real_measurements
    n_meas = len(measurement_outcomes)

    # Candidates: output frame observables, then reference X/Z per logical.
    candidates = [pauli_string_to_sparse(p) for p in output_frame_column_paulis]
    candidates += [reference(slot) for slot in range(2 * logical_count)]
    encoder = sim.clifford
    x_parts: list[list[int]] = []
    z_parts: list[list[int]] = []
    for candidate in candidates:
        preimage = encoder.preimage_of(candidate)
        support = preimage.support
        x_parts.append([q for q in support if preimage[q] in "XY"])
        z_parts.append([q for q in support if preimage[q] in "ZY"])
    stabilizing = null_space(
        BitMatrix.from_sparse_columns(x_parts, total_qubits, len(candidates))
    ).sparse_rows()

    sign_rows = sim.sign_matrix.sparse_rows()
    outcome_rows = sim.outcome_matrix.sparse_rows()
    outcome_shift = sim.outcome_shift

    def sign_dependence(candidate_set: Iterable[int]) -> set[int]:
        qubits: set[int] = set()
        for index in candidate_set:
            qubits.symmetric_difference_update(z_parts[index])
        branch_bits: set[int] = set()
        for qubit in qubits:
            branch_bits.symmetric_difference_update(sign_rows[qubit])
        return branch_bits

    tracked = input_outcomes + measurement_outcomes
    dependence_columns = [outcome_rows[outcome] for outcome in tracked]
    dependence_columns += [
        sorted(sign_dependence(combination)) for combination in stabilizing
    ]
    dependencies = null_space(
        BitMatrix.from_sparse_columns(
            dependence_columns, sim.sign_matrix.column_count, len(dependence_columns)
        )
    ).sparse_rows()

    # Relation columns: inputs [0, n_in), outputs [n_in, n_in + n_out),
    # measurements after that.
    measurement_start = n_in + n_out
    logical_input_columns = [
        j for j, slot in enumerate(reference_slots) if slot is not None
    ]
    relations: list[set[int]] = []
    for dependency in dependencies:
        relation = {i for i in dependency if i < n_in}
        relation.update(
            measurement_start + i - n_in
            for i in dependency
            if n_in <= i < len(tracked)
        )
        candidate_set: set[int] = set()
        for k in dependency:
            if k >= len(tracked):
                candidate_set.symmetric_difference_update(
                    stabilizing[k - len(tracked)]
                )
        reference_part = {index - n_out for index in candidate_set if index >= n_out}
        for j in logical_input_columns:
            if (j in relation) != (reference_slots[j] in reference_part):
                raise RuntimeError(
                    "jit_noise_builder: an input logical outcome appears "
                    "without its reference; algebra bug."
                )
        relation.update(n_in + index for index in candidate_set if index < n_out)
        if relation:
            relations.append(relation)

    order = (
        [n_in + i for i in reversed(range(n_out))]
        + list(reversed(range(n_in)))
        + [measurement_start + m for m in range(n_meas)]
    )
    position = {column: rank for rank, column in enumerate(order)}
    reduced = EchelonForm(
        BitMatrix.from_sparse_rows(
            [[position[column] for column in relation] for relation in relations],
            len(relations),
            len(order),
        )
    ).matrix.sparse_rows()

    def relation_sign(u: Sequence[int], v: Sequence[int], w: Sequence[int]) -> int:
        """Sign of one relation; signs are not linear in ``(u, v, w)``."""
        candidate_set = list(v)
        for j in u:
            slot = reference_slots[j]
            if slot is not None:
                candidate_set.append(n_out + slot)
        product = SparsePauli.identity()
        for index in sorted(candidate_set):
            product = product * candidates[index]
        preimage = encoder.preimage_of(abs(product))
        if any(preimage[q] in "XY" for q in preimage.support):
            raise RuntimeError(
                "jit_noise_builder: relation does not stabilize the final "
                "state; algebra bug."
            )
        outcomes = [input_outcomes[j] for j in u]
        outcomes += [measurement_outcomes[m] for m in w]
        expected: set[int] = set()
        for outcome in outcomes:
            expected.symmetric_difference_update(outcome_rows[outcome])
        if sign_dependence(candidate_set) != expected:
            raise RuntimeError(
                "jit_noise_builder: relation sign depends on unrecorded "
                "bits; algebra bug."
            )
        sign = _sign_bit(preimage.phase)
        for outcome in outcomes:
            sign ^= int(outcome_shift[outcome])
        input_product = SparsePauli.identity()
        for j in sorted(u):
            input_product = input_product * measured_inputs[j]
        sign ^= _sign_bit(input_product.phase)
        for m in w:
            sign ^= int(state.real_measurement_inverted[m])
        return sign

    flip_rhs_column = n_in + n_meas
    v_rows: list[list[int]] = []
    rhs_rows: list[set[int]] = []
    row_parts: list[tuple[list[int], list[int], list[int]]] = []
    for row in reduced:
        if not row or row[0] >= n_out:
            continue
        columns = [order[rank] for rank in row]
        u = sorted(c for c in columns if c < n_in)
        v = sorted(c - n_in for c in columns if n_in <= c < measurement_start)
        w = sorted(c - measurement_start for c in columns if c >= measurement_start)
        v_rows.append(v)
        rhs_rows.append(set(u) | {n_in + m for m in w})
        row_parts.append((u, v, w))

    if not v_rows:
        return [], set(), set()

    # The rows are reduced, so V has full row rank and every column of
    # ``V · X = R`` is consistent.  With ``T · V`` in RREF, row ``k`` of
    # ``T · R`` gives the solution at pivot ``k`` (free variables zero).
    v_echelon = EchelonForm(BitMatrix.from_sparse_rows(v_rows, len(v_rows), n_out))
    if len(v_echelon.pivots) != len(v_rows):
        raise RuntimeError(
            "jit_noise_builder: reduced relations are dependent; algebra bug."
        )
    transform_rows = v_echelon.transform.sparse_rows()
    signed: set[int] = set()
    solution_by_output: dict[int, set[int]] = {}
    for k, pivot in enumerate(v_echelon.pivots):
        if pivot not in output_layout.logical_columns:
            continue
        combined: set[int] = set()
        for j in transform_rows[k]:
            if j not in signed:
                signed.add(j)
                if relation_sign(*row_parts[j]):
                    rhs_rows[j].add(flip_rhs_column)
            combined ^= rhs_rows[j]
        solution_by_output[pivot] = combined

    pc_entries: list[tuple[int, int]] = []
    cp_entries: set[tuple[int, int]] = set()
    flip_entries: set[int] = set()
    for i in sorted(output_layout.logical_columns):
        for col in sorted(solution_by_output.get(i, ())):
            if col < n_in:
                cp_entries.add((i, col))
            elif col < flip_rhs_column:
                pc_entries.append((i, col - n_in))
            else:
                flip_entries.add(i)

    return pc_entries, cp_entries, flip_entries


# ---------------------------------------------------------------------------
# Top-level: compute noise errors for a gadget
# ---------------------------------------------------------------------------


def _build_real_meas_start_table(
    body_flat: Sequence[object],
) -> tuple[list[int], int]:
    """Return ``(starts, total_real)`` where ``starts[i]`` is the real-
    measurement index that ``body_flat[i]`` will produce first.
    """
    starts: list[int] = []
    running = 0
    for stmt in body_flat:
        starts.append(running)
        if isinstance(stmt, Instruction):
            running += _real_measurement_count(stmt)
    return starts, running


def compute_noise_errors(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
    *,
    output_ports: list[OutputPort],
    input_virtual_count: int,
    finished_checks: Sequence[Check],
    unfinished_checks: Sequence[Check],
    ov_start: int,
    readouts: Sequence[pb.GadgetType.Readout],
    physical_correction: util_pb.BitMatrix,
) -> list[jit_pb.JitGadgetType.Error]:
    """Expand every noise instruction in the body into JIT ``Error`` rows.

    Parameters mirror the precomputed state available in
    :func:`deq.transpiler.jit_library_builder._build_jit_gadget_type`.
    ``physical_correction`` is the freshly-computed pc matrix and is
    used to subtract out the runtime's automatic Pauli-frame update on
    flipped body measurements (see
    :func:`iter_noise_errors_with_origin` for details).
    """
    errors: list[jit_pb.JitGadgetType.Error] = []
    for _body_index, error_row in iter_noise_errors_with_origin(
        gadget,
        codes,
        output_ports=output_ports,
        input_virtual_count=input_virtual_count,
        finished_checks=finished_checks,
        unfinished_checks=unfinished_checks,
        ov_start=ov_start,
        readouts=readouts,
        physical_correction=physical_correction,
    ):
        errors.append(error_row)
    return errors


@dataclass(frozen=True)
class _NoiseMechanism:
    """A pure-noise mechanism enumerated from the gadget body.

    ``walk_start`` is the decomposed-body index just after the originating noise
    instruction (where the mechanism's Pauli is injected); ``body_index`` is the
    instruction's position in the flattened body, used to interleave emitted
    rows back in body order.
    """

    body_index: int
    walk_start: int
    pauli: stim.PauliString
    probability: float
    site_name: str


def _collect_noise_mechanisms(
    body_flat: Sequence[object],
    num_qubits: int,
    orig_to_decomposed: Sequence[int],
    num_decomposed_instructions: int,
) -> list[_NoiseMechanism]:
    """Enumerate every pure-noise mechanism in body order.

    Tracks Stim's correlated-error else-chain semantics: ``ELSE_CORRELATED_ERROR(p)``
    fires with marginal probability ``p * remaining``, where ``remaining`` is the
    probability that no error in the current chain has fired yet.  ``E`` /
    ``CORRELATED_ERROR`` start a new chain; any other statement breaks it.
    Orphan or interrupted ELSE branches are rejected.
    """
    mechanisms: list[_NoiseMechanism] = []
    chain = CorrelatedErrorChain()
    for body_index, stmt in enumerate(body_flat):
        current_else_remaining = chain.advance(stmt)
        if not isinstance(stmt, Instruction):
            continue
        name = stmt.name.upper()

        if name in NOISE_INSTRUCTIONS_ALL:
            walk_start = (
                orig_to_decomposed[body_index + 1]
                if body_index + 1 < len(orig_to_decomposed)
                else num_decomposed_instructions
            )
            for pauli, probability in enumerate_noise_mechanisms(
                stmt, num_qubits, else_chain_remaining=current_else_remaining
            ):
                mechanisms.append(
                    _NoiseMechanism(body_index, walk_start, pauli, probability, name)
                )
    return mechanisms


def _build_mechanism_rows(
    mechanisms: Sequence[_NoiseMechanism],
    flips: Sequence[MechanismFlips],
    context: ErrorProjectionContext,
) -> list[tuple[int, jit_pb.JitGadgetType.Error | None]]:
    """Build one ``(body_index, error_row)`` per mechanism from its footprint."""
    rows: list[tuple[int, jit_pb.JitGadgetType.Error | None]] = []
    for mechanism, mechanism_flips in zip(mechanisms, flips):
        rows.append(
            (
                mechanism.body_index,
                build_error_row_from_flips(
                    site_name=mechanism.site_name,
                    site_pauli=mechanism.pauli,
                    probability=mechanism.probability,
                    flips=mechanism_flips,
                    context=context,
                ),
            )
        )
    return rows


def iter_noise_errors_with_origin(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
    *,
    output_ports: list[OutputPort],
    input_virtual_count: int,
    finished_checks: Sequence[Check],
    unfinished_checks: Sequence[Check],
    ov_start: int,
    readouts: Sequence[pb.GadgetType.Readout],
    physical_correction: util_pb.BitMatrix,
) -> Iterator[tuple[int, jit_pb.JitGadgetType.Error]]:
    """Yield ``(body_index, error_row)`` for every propagated noise mechanism.

    ``body_index`` is the index of the originating noise instruction in
    the *flattened* gadget body (``flatten_body``). Used by the
    ``annotate`` tool to interleave propagated ``ERROR`` statements
    next to the noise instruction they came from.

    ``physical_correction`` is the gadget's pc matrix.  The runtime
    applies pc to body-measurement outcomes as an automatic Pauli
    frame update; whenever a noise mechanism flips a set of body
    measurements ``M_e`` the runtime over-corrects each output
    observable ``r`` by ``(pc · M_e)[r]``.  This iterator therefore
    XORs ``(pc · M_e)[r]`` into the residual on logical rows so that
    the ``Error`` row stores the *post-runtime* frame error, which is
    what the decoder must learn to undo.
    """
    body_flat = flatten_body(list(gadget.body))
    num_qubits = max(max_qubit_index(list(gadget.body)) + 1, 0)
    real_starts, _total_real = _build_real_meas_start_table(body_flat)
    decomposed = build_decomposed_body(body_flat)
    orig_to_decomposed = decomposed.body_start_at

    output_stabilizer_paulis, frame_column_paulis = build_port_paulis(
        output_ports, codes, num_qubits
    )
    context = build_error_projection_context(
        output_ports=output_ports,
        codes=codes,
        input_virtual_count=input_virtual_count,
        finished_checks=finished_checks,
        unfinished_checks=unfinished_checks,
        output_virtual_start=ov_start,
        readouts=readouts,
        physical_correction=physical_correction,
    )

    mechanisms = _collect_noise_mechanisms(
        body_flat, num_qubits, orig_to_decomposed, len(decomposed.instructions)
    )
    measurement_mechanisms = [
        (body_index, real_starts[body_index] + offset, float(stmt.arguments[0]))
        for body_index, stmt in enumerate(body_flat)
        if isinstance(stmt, Instruction) and stmt.arguments and stmt.arguments[0] != 0
        for offset in range(_real_measurement_count(stmt))
    ]
    flips = propagate_pauli_mechanisms(
        [(m.walk_start, m.pauli) for m in mechanisms],
        decomposed,
        output_stabilizer_paulis,
        frame_column_paulis,
        measurement_flips=[real_index for _, real_index, _ in measurement_mechanisms],
    )
    mechanism_rows = _build_mechanism_rows(mechanisms, flips[:len(mechanisms)], context)
    measurement_rows: dict[int, list[jit_pb.JitGadgetType.Error]] = {}
    for (body_index, real_index, probability), mechanism_flips in zip(measurement_mechanisms, flips[len(mechanisms):]):
        error_row = _build_measurement_flip_error(
            real_index=real_index, probability=probability, flips=mechanism_flips, context=context,
        )
        if error_row is not None:
            measurement_rows.setdefault(body_index, []).append(error_row)

    mechanism_row_index = 0
    for body_index, stmt in enumerate(body_flat):
        if not isinstance(stmt, Instruction):
            continue
        name = stmt.name.upper()

        if name in NOISE_INSTRUCTIONS_ALL:
            while (
                mechanism_row_index < len(mechanism_rows)
                and mechanism_rows[mechanism_row_index][0] == body_index
            ):
                error_row = mechanism_rows[mechanism_row_index][1]
                mechanism_row_index += 1
                if error_row is not None:
                    yield body_index, error_row
            continue

        for error_row in measurement_rows.get(body_index, ()):
            yield body_index, error_row


def _build_measurement_flip_error(
    *,
    real_index: int,
    probability: float,
    flips: MechanismFlips,
    context: ErrorProjectionContext,
) -> jit_pb.JitGadgetType.Error | None:
    """Project a result flip after propagating its downstream feedback effects."""
    error_row = build_error_row_from_flips(
        site_name="M_FLIP",
        site_pauli=stim.PauliString(),
        probability=probability,
        flips=flips,
        context=context,
    )
    if error_row is not None:
        error_row.base.tag = f"M_FLIP m{real_index + context.input_virtual_count}"
    return error_row


# ---------------------------------------------------------------------------
# PROPAGATE statement resolution and validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedPropagation:
    """A ``PROPAGATE`` statement resolved to (cp, pc, flip) entries.

    Attributes
    ----------
    cp_input_cols
        Set of input-frame column indices (logical or stabilizer
        generator) that XOR into the row.
    pc_internal_cols
        Set of internal-measurement column indices that XOR into the
        row.
    flip
        Whether the affine ``FLIP`` column is set for this row.
    statement
        The original :class:`PropagateStatement`, kept for diagnostics.
    """

    cp_input_cols: frozenset[int]
    pc_internal_cols: frozenset[int]
    flip: bool
    statement: PropagateStatement


def _resolve_logical_target_to_columns(
    target: LogicalPauliTarget,
    ports: Sequence[InputPort | OutputPort],
    codes: dict[str, CodeDefinition],
    expected_kind: Literal["IN", "OUT"],
) -> list[int]:
    """Resolve a ``LogicalPauliTarget`` to one or more frame columns.

    Mirrors :func:`deq.transpiler.jit_library_builder.conditional_flipped_rows`
    but is direction-agnostic: the same column convention applies to
    input cp columns and output cp/pc rows (``LX_i`` → Z column,
    ``LZ_i`` → X column, ``LY_i`` → both).

    When the target is port-qualified (``IN<p>.L<P><i>`` /
    ``OUT<p>.L<P><i>``), ``target.port_kind`` must equal
    ``expected_kind``, ``target.port_index`` must be a valid index into
    ``ports``, and ``target.index`` is interpreted as logical-within-port.
    Otherwise (bare ``L<P><i>``), ``target.index`` is interpreted
    globally across all ``ports``.
    """
    if target.port_kind is not None:
        assert target.port_index is not None
        if target.port_kind != expected_kind:
            raise ValueError(
                f"logical target {target!s} has direction "
                f"{target.port_kind!r}, but this context expects "
                f"{expected_kind!r} (e.g. PROPAGATE LHS is OUT-side, "
                f"its RHS logical terms are IN-side)"
            )
        if not 0 <= target.port_index < len(ports):
            raise ValueError(
                f"logical target {target!s}: port index out of range "
                f"(only {len(ports)} {expected_kind} port(s))"
            )
        obs_offset = 0
        for port in ports[: target.port_index]:
            code = codes[port.code_name]
            obs_offset += 2 * len(code.logicals) + len(
                select_stabilizer_generators(code).generator_indices
            )
        matched_port = ports[target.port_index]
        code = codes[matched_port.code_name]
        n_logicals = len(code.logicals)
        if not 0 <= target.index < n_logicals:
            raise ValueError(
                f"logical target {target!s}: logical index out of range "
                f"(port has {n_logicals} logical observable(s))"
            )
        logical_idx = target.index
    else:
        obs_offset = 0
        remaining = target.index
        matched_port = None
        logical_idx = 0
        for port in ports:
            code = codes[port.code_name]
            n_logicals = len(code.logicals)
            if remaining < n_logicals:
                matched_port = port
                logical_idx = remaining
                break
            remaining -= n_logicals
            obs_offset += 2 * n_logicals + len(
                select_stabilizer_generators(code).generator_indices
            )
        if matched_port is None:
            total_logicals = sum(len(codes[p.code_name].logicals) for p in ports)
            raise ValueError(
                f"logical target L{target.pauli}{target.index} out of range "
                f"(only {total_logicals} logical observables across the ports)"
            )
    pauli = target.pauli.upper()
    cols: list[int] = []
    if pauli in ("X", "Y"):
        cols.append(obs_offset + 2 * logical_idx + 1)
    if pauli in ("Z", "Y"):
        cols.append(obs_offset + 2 * logical_idx)
    if not cols:
        raise ValueError(f"unsupported logical Pauli {target.pauli!r}")
    return cols


def _resolve_ds_to_input_cols(
    term: DestabilizerTarget,
    input_layout: PortColumnLayout,
    input_ports: Sequence[InputPort],
    codes: dict[str, CodeDefinition],
) -> set[int]:
    """Resolve an ``IN<p>.DS<s>`` destabilizer target to input cp columns.

    ``IN<p>.DS<s>`` denotes the destabilizer of stabilizer ``s`` of
    INPUT port ``p`` — the operator that anticommutes only with that
    one stabilizer.  The contributed column is the input cp column
    carrying that stabilizer's syndrome bit, decomposed as the XOR of
    generator-stabilizer columns when the stabilizer is redundant.
    """
    port_idx = term.port_index
    if port_idx < 0 or port_idx >= len(input_ports):
        raise ValueError(
            f"IN{port_idx}.DS{term.stab_index} references INPUT port {port_idx} "
            f"but the gadget has only {len(input_ports)} INPUT port(s)"
        )
    port_stab_count = len(codes[input_ports[port_idx].code_name].stabilizers)
    if term.stab_index < 0 or term.stab_index >= port_stab_count:
        raise ValueError(
            f"IN{port_idx}.DS{term.stab_index} out of range "
            f"(INPUT port {port_idx} has only {port_stab_count} stabilizer(s))"
        )
    flat_idx = input_layout.per_port_stab_offsets[port_idx] + term.stab_index
    return set(input_layout.stab_decomposed_columns[flat_idx])


def resolve_propagations(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
    *,
    input_ports: Sequence[InputPort],
    output_ports: Sequence[OutputPort],
    input_layout: PortColumnLayout,
    input_virtual_count: int,
    ov_start: int,
) -> dict[int, ResolvedPropagation]:
    """Resolve every ``PROPAGATE`` statement in *gadget* to row entries.

    Returns a dict mapping output cp/pc row index → resolved entries.
    Each ``PROPAGATE`` produces one or more rows (multiple rows when
    the LHS is ``LY<i>``: both X and Z output rows of that logical
    qubit get the same XOR).

    Raises
    ------
    ValueError
        On duplicate row pinning, out-of-range DS index, or
        ``rec[-k]`` resolving to an input-virtual or output-virtual
        measurement (those have no column in ``physical_correction``).
    """
    body_flat = flatten_body(list(gadget.body))
    result: dict[int, ResolvedPropagation] = {}
    seen_rows: dict[int, PropagateStatement] = {}

    running = 0
    for stmt in body_flat:
        if isinstance(stmt, InputPort):
            running += len(codes[stmt.code_name].stabilizers)
        elif isinstance(stmt, OutputPort):
            running += len(codes[stmt.code_name].stabilizers)
        elif isinstance(stmt, Instruction):
            running += instruction_num_measurements(str(stmt))
        elif isinstance(stmt, PropagateStatement):
            target_rows = _resolve_logical_target_to_columns(
                stmt.target, list(output_ports), codes, expected_kind="OUT"
            )

            cp_cols: set[int] = set()
            pc_cols: set[int] = set()
            for term in stmt.terms:
                if isinstance(term, LogicalPauliTarget):
                    in_cols = _resolve_logical_target_to_columns(
                        term, list(input_ports), codes, expected_kind="IN"
                    )
                    cp_cols ^= set(in_cols)
                elif isinstance(term, DestabilizerTarget):
                    cp_cols ^= _resolve_ds_to_input_cols(
                        term, input_layout, input_ports, codes
                    )
                elif isinstance(term, (MeasurementRecordTarget, PhysicalMeasurementTarget)):
                    internal_count = ov_start - input_virtual_count
                    global_index = resolve_measurement_ref_global(
                        term,
                        running=running,
                        input_ports=list(input_ports),
                        output_ports=list(output_ports),
                        codes=codes,
                        internal_count=internal_count,
                        gadget_name=gadget.name,
                    )
                    if global_index < input_virtual_count:
                        raise ValueError(
                            f"in GADGET {gadget.name!r}: PROPAGATE "
                            f"{term!s} references an input-virtual "
                            f"stabilizer measurement; PROPAGATE may only "
                            f"reference internal physical measurements"
                        )
                    if global_index >= ov_start:
                        raise ValueError(
                            f"in GADGET {gadget.name!r}: PROPAGATE "
                            f"{term!s} references an output-virtual "
                            f"stabilizer measurement; PROPAGATE may only "
                            f"reference internal physical measurements"
                        )
                    pc_cols ^= {global_index - input_virtual_count}
                elif isinstance(term, ReadoutTarget):
                    # ``R<k>`` terms route to ``logical_correction`` via
                    # ``_build_logical_correction``; they do NOT affect
                    # ``correction_propagation`` or ``physical_correction``.
                    continue
                else:
                    raise ValueError(
                        f"in GADGET {gadget.name!r}: unsupported PROPAGATE "
                        f"term {term!r}"
                    )

            for row in target_rows:
                if row in seen_rows:
                    prev = seen_rows[row]
                    raise ValueError(
                        f"in GADGET {gadget.name!r}: duplicate PROPAGATE for "
                        f"output row {row} (target {stmt.target}); a previous "
                        f"PROPAGATE for {prev.target} already pins this row"
                    )
                seen_rows[row] = stmt
                result[row] = ResolvedPropagation(
                    cp_input_cols=frozenset(cp_cols),
                    pc_internal_cols=frozenset(pc_cols),
                    flip=stmt.flip,
                    statement=stmt,
                )
    return result


def _apply_propagations(
    *,
    propagations: dict[int, ResolvedPropagation],
    cp_entries: set[tuple[int, int]],
    logical_physical: list[tuple[int, int]],
    flow_cp_entries: set[tuple[int, int]],
    flow_flip_entries: set[int],
    flip_col: int,
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    """Install each declared ``PROPAGATE`` row verbatim in place of the
    flow-derived row.

    ``PROPAGATE`` is authoritative: the user's declared XOR formula is
    the ground truth for that output row's cp/pc contributions.  For
    rows without an explicit ``PROPAGATE``, the flow-derived
    (natural-Heisenberg + VIRTUAL + measurement-conditioned CONDITIONAL)
    entries are kept.  Readout terms (``R<k>``) are handled by
    :func:`_build_logical_correction` and never touch cp/pc.
    """
    if not propagations:
        return cp_entries, logical_physical

    flow_cp_per_row: dict[int, set[int]] = {}
    for r, c in flow_cp_entries:
        flow_cp_per_row.setdefault(r, set()).add(c)

    for row, resolved in sorted(propagations.items()):
        flow_cp_cols = flow_cp_per_row.get(row, set())
        flow_flip = row in flow_flip_entries

        # Remove flow-derived entries, then add the user-declared entries.
        cp_entries -= {(row, c) for c in flow_cp_cols}
        if flow_flip:
            cp_entries -= {(row, flip_col)}
        cp_entries |= {(row, c) for c in resolved.cp_input_cols}
        if resolved.flip:
            cp_entries |= {(row, flip_col)}

        logical_physical[:] = [(r, c) for (r, c) in logical_physical if r != row]
        for c in sorted(resolved.pc_internal_cols):
            logical_physical.append((row, c))

    return cp_entries, logical_physical


# ---------------------------------------------------------------------------
# Correction propagation (input observables -> output observables)
# ---------------------------------------------------------------------------


def _build_input_port_paulis(
    input_ports: Sequence[InputPort],
    codes: dict[str, CodeDefinition],
    num_qubits: int,
) -> list[stim.PauliString]:
    """Return the anticommuting Pauli for every input frame column.

    For each frame column, returns the operator that anticommutes **only** with
    that column's observable:

    - Logical X column (2i):   returns LZ_i  (anticommutes with LX)
    - Logical Z column (2i+1): returns LX_i  (anticommutes with LZ)
    - Generator column (2k+j): returns D_j   (anticommutes with S_j)

    This means every walk result records at its own column index
    (no symplectic-partner remapping needed).
    """
    paulis: list[stim.PauliString] = []
    for port in input_ports:
        code = codes[port.code_name]
        local_to_global = {
            local_qubit: global_qubit
            for local_qubit, global_qubit in enumerate(port.qubit_indices)
        }
        for logical in code.logicals:
            paulis.append(
                pauli_product_to_stim(
                    logical.z_operator, num_qubits, local_to_global
                )
            )
            paulis.append(
                pauli_product_to_stim(
                    logical.x_operator, num_qubits, local_to_global
                )
            )
        sel = select_stabilizer_generators(code)
        for j in range(len(sel.generator_indices)):
            destab = sel.destabilizer_paulis[j]
            ps = stim.PauliString(num_qubits)
            for q in destab.pauli_indices():
                ps[local_to_global[q]] = destab[q]
            paulis.append(ps)
    return paulis


def compute_correction_propagation(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
    *,
    input_ports: Sequence[InputPort],
    output_ports: Sequence[OutputPort],
    unfinished_checks: Sequence[tuple[frozenset[int], bool]],
    finished_checks: Sequence[tuple[frozenset[int], bool]] = (),
    input_virtual_count: int,
    ov_start: int | None = None,
    propagations: dict[int, ResolvedPropagation] | None = None,
) -> tuple[util_pb.BitMatrix, list[tuple[int, int]]]:
    """Compute the ``correction_propagation`` matrix.

    **Logical rows** (0..2k-1 per port) are computed by
    :func:`_compute_pc_logical_via_flows`, a relation analysis
    by stabilizer simulation that
    expresses each output logical observable as an XOR of
    input-column observables and body measurement outcomes.  The
    same path covers unitary bodies and bodies with internal
    measurements (e.g. Floquet honeycomb rounds).

    **Stabilizer rows** (2k..2k+|S|-1 per port) come from
    ``unfinished_checks``.  Each UC defines
    ``output_stab[k] = XOR(input_stabs) XOR XOR(measurements)``;
    the input-stab references become entries in stabilizer rows,
    decomposed into generator columns via the stabilizer
    decomposition matrix.

    The affine ``FLIP`` column (last col) absorbs constant sign
    offsets from ``VIRTUAL`` logical statements and from the flow
    analysis's signed closure.

    Returns ``(BitMatrix, logical_physical_entries)`` where the second
    element is a list of ``(output_row, internal_meas_col)`` entries
    for the ``physical_correction`` logical rows.  These come from
    the flow analysis's selection of body measurements whose outcomes
    contribute to each output logical observable.
    """
    input_layout = PortColumnLayout(list(input_ports), codes)
    output_layout = PortColumnLayout(list(output_ports), codes)
    num_input_observables = input_layout.num_columns
    num_output_observables = output_layout.num_columns
    cols = num_input_observables + 1
    rows = num_output_observables
    if rows == 0:
        return util_pb.BitMatrix(rows=rows, cols=cols), []

    body_flat = flatten_body(list(gadget.body))

    entries: set[tuple[int, int]] = set()

    # ── Stabilizer rows: from unfinished checks ────────────────────
    for uc_idx, (members, parity) in enumerate(unfinished_checks):
        out_col = output_layout.stab_to_column[uc_idx]
        if out_col is None:
            continue

        for member in members:
            if member < input_virtual_count:
                for in_col in input_layout.stab_decomposed_columns[member]:
                    entries ^= {(out_col, in_col)}

        if parity:
            entries ^= {(out_col, num_input_observables)}

    # VIRTUAL logical statements flip the constant (affine) column.
    constant_col = num_input_observables
    for stmt in body_flat:
        if isinstance(stmt, VirtualLogicalStatement):
            for target in stmt.targets:
                flipped = _resolve_logical_target_to_columns(
                    target, list(output_ports), codes, expected_kind="OUT"
                )
                for row in flipped:
                    entries ^= {(row, constant_col)}

    # ── Compute logical-row pc/cp entries from the body's relations ─
    # Each output logical observable is expressed as an XOR of
    # input-column observables and body measurement outcomes.  This
    # unifies handling of unitary bodies and bodies with internal
    # measurements.
    logical_physical, flow_cp_entries, flip_entries = _compute_pc_logical_via_flows(
        gadget,
        codes,
        input_ports=input_ports,
        output_ports=output_ports,
        output_layout=output_layout,
    )
    entries ^= flow_cp_entries
    for row in flip_entries:
        entries ^= {(row, constant_col)}

    # Separate the combined entries (VIRTUAL + flow + unfinished) into
    # cp-only entries and flip entries so :func:`_apply_propagations`
    # can remove flow contributions on rows the user explicitly pins.
    combined_cp_entries: set[tuple[int, int]] = set()
    combined_flip_entries: set[int] = set()
    for r, c in entries:
        if c == constant_col:
            combined_flip_entries ^= {r}
        else:
            combined_cp_entries.add((r, c))

    if propagations:
        entries, logical_physical = _apply_propagations(
            propagations=propagations,
            cp_entries=entries,
            logical_physical=logical_physical,
            flow_cp_entries=combined_cp_entries,
            flow_flip_entries=combined_flip_entries,
            flip_col=constant_col,
        )

    return (
        bitmatrix_from_sparse(entries, rows=rows, cols=cols),
        logical_physical,
    )


def compute_physical_correction(
    codes: dict[str, CodeDefinition],
    *,
    output_ports: list[OutputPort],
    unfinished_checks: Sequence[tuple[frozenset[int], bool]],
    input_virtual_count: int,
    ov_start: int,
    physical_conditionals: Sequence[tuple[int, list[int]]] = (),
    logical_physical_entries: Sequence[tuple[int, int]] = (),
) -> util_pb.BitMatrix:
    """Compute the ``physical_correction`` matrix.

    Maps internal measurements to output frame columns.

    **Stabilizer rows** (``2k..2k+|S|-1``) are derived from unfinished
    checks that reference internal measurements.

    **Logical rows** (``0..2k-1``) are populated from two sources:

    1. ``CONDITIONAL rec[-k] L<P><i>`` statements via
       *physical_conditionals*: each entry is
       ``(internal_meas_index, flipped_rows)``.

    2. *logical_physical_entries*: ``(output_row, internal_meas_col)``
       pairs from the Pauli forward-walk in
       ``compute_correction_propagation``.  These record which internal
       measurements anti-commuted with input probes and therefore
       contribute to the output logical observable's sign.

    Dimensions: ``|output_observables| rows × |internal_measurements| cols``
    """
    output_layout = PortColumnLayout(output_ports, codes)
    num_output_observables = output_layout.num_columns
    internal_count = ov_start - input_virtual_count

    rows = num_output_observables
    cols = internal_count
    if rows == 0 or cols == 0:
        return util_pb.BitMatrix(rows=rows, cols=cols)

    entries: set[tuple[int, int]] = set()

    for uc_idx, (members, _parity) in enumerate(unfinished_checks):
        out_col = output_layout.stab_to_column[uc_idx]
        if out_col is None:
            continue

        for member in members:
            if member < input_virtual_count:
                continue
            if member >= ov_start:
                continue
            meas_col = member - input_virtual_count
            entries ^= {(out_col, meas_col)}

    for meas_col, flipped_rows in physical_conditionals:
        for row in flipped_rows:
            entries ^= {(row, meas_col)}

    for row, meas_col in logical_physical_entries:
        entries ^= {(row, meas_col)}

    return bitmatrix_from_sparse(entries, rows=rows, cols=cols)


def compute_implicit_readout_propagation(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
    *,
    input_ports: Sequence[InputPort],
    readout_measurement_sets: Sequence[set[int]],
) -> list[set[int]]:
    """Return the set of input-observable column indices that implicitly
    flip each readout via measurement parity.

    For each input observable, we forward-propagate its representative
    Pauli through the gadget body and record which real measurements
    the propagated Pauli flips. A readout is implicitly toggled by an
    input correction whenever the parity of its
    ``measurement_indices`` (real-only) intersected with that flipped
    set is odd.

    Column convention: the contribution is recorded against the
    *symplectic partner* of the walked observable column (``col ^ 1``).
    A measurement-flip on the walked Pauli ``P`` indicates the readout
    value reflects a flip in the partner-basis observable's bit (e.g.
    walking ``LX`` and finding it anticommutes with a ``Z``-basis
    measurement means the readout reports the ``LZ`` observable's
    value flipped). This matches the frame-bit convention used by the
    explicit ``L<P><i>`` / ``<P><i>`` Pauli targets, where ``X<i>``
    contributes column ``2i`` and ``Z<i>`` contributes column ``2i+1``.
    The contributions returned here are XOR-combined with any explicit
    Pauli targets and the affine ``FLIP`` constant to form the final
    ``readout_propagation`` matrix.
    """
    input_layout = PortColumnLayout(list(input_ports), codes)
    num_input_observables = input_layout.num_columns
    if not readout_measurement_sets or num_input_observables == 0:
        return [set() for _ in readout_measurement_sets]

    body_flat = flatten_body(list(gadget.body))
    num_qubits = max(max_qubit_index(list(gadget.body)) + 1, 0)
    decomposed = build_decomposed_body(body_flat)

    input_paulis = _build_input_port_paulis(input_ports, codes, num_qubits)

    contributions: list[set[int]] = [set() for _ in readout_measurement_sets]

    flips_by_column = propagate_pauli_mechanisms(
        [(0, initial) for initial in input_paulis], decomposed, (), ()
    )
    for col, flips in enumerate(flips_by_column):
        for row, meas_set in enumerate(readout_measurement_sets):
            if len(meas_set & flips.flipped_real) % 2 == 1:
                contributions[row].add(col)
    return contributions
