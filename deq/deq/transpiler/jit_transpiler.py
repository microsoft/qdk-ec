"""JIT transpiler — outcome-code computation and shared layout types.

This module derives the parity-check structure ("outcome code") of each
gadget via :mod:`paulimer` stabilizer simulation and defines the shared
type aliases and layout utilities used across the transpiler pipeline.

Measurement layout
==================

Measurements are indexed globally in this order:

1. **input-virtual** stabilizer measurements (one per stabilizer of each
   ``INPUT`` port, in port-declaration order);
2. **internal** measurement results produced by the gadget body (in
   program order);
3. **output-virtual** stabilizer measurements (one per stabilizer of each
   ``OUTPUT`` port, in port-declaration order).

See :class:`MeasurementLayout` and :data:`MeasurementIndex`.

Observable column convention (symplectic pairing)
=================================================

Correction-propagation, readout-propagation, and error-residual matrices
use a column layout per port with two sections:

1. **Logical columns** — each logical qubit contributes two columns
   (X and Z) arranged as consecutive pairs::

       Column 2k     →  X observable of logical qubit k
       Column 2k + 1 →  Z observable of logical qubit k

2. **Stabilizer generator columns** — one column per independent
   stabilizer generator, appended after the logical columns::

       Column 2k + j  →  stabilizer generator j

   where ``k`` is the number of logical qubits and ``j`` ranges over
   the independent generators selected from the code's stabilizer list
   (see :func:`select_stabilizer_generators`).

The trailing column (index ``num_observables``) is the affine **shift**
column: a ``1`` indicates a deterministic flip independent of input state.

The ``col ^ 1`` idiom that appears in :func:`compute_correction_propagation`
and related functions selects the *symplectic partner* — when an X input
observable (column ``2k``) is flipped, we propagate its symplectic
partner Z (column ``2k+1``) through the circuit and record which output
observables are flipped.  The matrix entry is therefore recorded at
column ``2k+1 = symplectic_partner(2k)``.  This matches the convention
in :func:`compute_implicit_readout_propagation`.

See :data:`ObservableColumn`.
"""

from dataclasses import dataclass, field
from typing import Literal, Sequence, cast

import stim
from binar import BitMatrix, rank, vstack

# ---------------------------------------------------------------------------
# Semantic type aliases
# ---------------------------------------------------------------------------

#: Global measurement index within a gadget's measurement layout.
#: Layout: ``[input-virtual | internal | output-virtual]``.
MeasurementIndex = int

#: Column index in a correction/readout propagation matrix.
#: Layout: ``[observable_0, ..., observable_n, shift]`` where observable
#: indices follow the X/Z symplectic pairing: ``X_k = 2k``, ``Z_k = 2k+1``.
ObservableColumn = int

#: A parity check: ``(member_indices, expected_parity)``.
#: Members are :data:`MeasurementIndex` values; parity is True when the
#: check is "naturally flipped" (odd expectation in the noiseless case).
Check = tuple[frozenset[MeasurementIndex], bool]


# ---------------------------------------------------------------------------
# Symplectic observable column layout
# ---------------------------------------------------------------------------
#
# Each observable contributes two columns: the X representative at column
# ``2k`` and the Z representative at column ``2k+1``. The "symplectic
# partner" of a column is the other half of the same qubit's pair —
# anticommutation against the partner determines whether that observable
# is flipped. Use the helpers below instead of writing ``2*k``, ``2*k+1``,
# or ``row ^ 1`` literals so the convention can be changed in one place.

#: Number of columns reserved per observable in propagation matrices.
COLUMNS_PER_OBSERVABLE = 2


def x_column(observable_index: int) -> ObservableColumn:
    """Column index of the X representative of ``observable_index``."""
    return COLUMNS_PER_OBSERVABLE * observable_index


def z_column(observable_index: int) -> ObservableColumn:
    """Column index of the Z representative of ``observable_index``."""
    return COLUMNS_PER_OBSERVABLE * observable_index + 1


def symplectic_partner(column: ObservableColumn) -> ObservableColumn:
    """Return the symplectic partner column (X<->Z of the same observable)."""
    return column ^ 1


def observable_of_column(column: ObservableColumn) -> int:
    """Return the observable index that ``column`` belongs to."""
    return column // COLUMNS_PER_OBSERVABLE


from paulimer import OutcomeCompleteSimulation, PauliGroup, SparsePauli, UnitaryOpcode

from deq.circuit.model import (
    CheckStatement,
    CodeDefinition,
    GadgetDefinition,
    GadgetStatement,
    InputPort,
    InputVirtualTarget,
    Instruction,
    MeasurementRecordTarget,
    MeasurementRefTarget,
    OutputPort,
    OutputVirtualTarget,
    PauliProduct,
    PauliTarget,
    PhysicalMeasurementTarget,
    QubitTarget,
    RepeatBlock,
)
from deq.transpiler.stim_constants import (
    ANNOTATION_INSTRUCTIONS,
    NOISE_INSTRUCTIONS,
)

# ---------------------------------------------------------------------------
# Stim decomposition helpers
# ---------------------------------------------------------------------------


def _body_to_stim_circuit(
    stmts: Sequence[GadgetStatement],
) -> "stim.Circuit":
    """Convert flattened gadget body Instructions to a ``stim.Circuit``.

    Non-Instruction nodes (InputPort, OutputPort, CheckStatement, etc.)
    are skipped.  Tags (``[...]``) are stripped because Stim does not
    recognise them.  The result can be passed to ``.decomposed()`` to
    reduce it to the ``{H, S, CX, M, R, MPAD}`` gate set.
    """
    lines: list[str] = []
    for stmt in stmts:
        if not isinstance(stmt, Instruction):
            continue
        name = stmt.name.upper()
        if name in NOISE_INSTRUCTIONS or name in ANNOTATION_INSTRUCTIONS:
            continue
        # Rebuild the instruction without tag
        inst_copy = Instruction(
            name=stmt.name,
            arguments=stmt.arguments,
            targets=stmt.targets,
        )
        lines.append(str(inst_copy))
    return stim.Circuit("\n".join(lines))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _BuildState:
    """Mutable state threaded through the simulation-building walk."""

    sim: OutcomeCompleteSimulation
    real_measurements: list[int] = field(default_factory=list)
    #: Tracks which real_measurements are inverted (``M !q``).
    #: Indexed in parallel with real_measurements.
    real_measurement_inverted: list[bool] = field(default_factory=list)


def _pauli_product_to_sparse(
    product: PauliProduct, qubit_map: dict[int, int]
) -> SparsePauli:
    """Convert a ``PauliProduct`` (with code-local indices) to a ``SparsePauli``.

    ``qubit_map`` translates code-local (logical-position) indices to
    absolute physical qubit indices, matching the ``INPUT``/``OUTPUT`` port
    declaration.
    """
    terms: dict[int, str] = {}
    for term in product.terms:
        if term.pauli == "I":
            continue
        phys = qubit_map[term.index]
        if phys in terms:
            raise ValueError(f"qubit {phys} appears more than once in PauliProduct")
        terms[phys] = term.pauli
    return SparsePauli(cast(dict, terms))


_KNOWN_INSTRUCTION_DECORATORS = frozenset({"SIMULATE_ONLY", "DECODE_ONLY"})


def _is_simulate_only(stmt: GadgetStatement) -> bool:
    """True if the statement carries an ``@SIMULATE_ONLY`` decorator."""
    return isinstance(stmt, Instruction) and any(
        d.name == "SIMULATE_ONLY" for d in stmt.decorators
    )


def _is_decode_only(stmt: GadgetStatement) -> bool:
    """True if the statement carries a ``@DECODE_ONLY`` decorator."""
    return isinstance(stmt, Instruction) and any(
        d.name == "DECODE_ONLY" for d in stmt.decorators
    )


def _validate_instruction_decorators(stmt: GadgetStatement) -> None:
    """Raise on unrecognized or conflicting instruction-level decorators."""
    if not isinstance(stmt, Instruction) or not stmt.decorators:
        return
    names = set()
    for deco in stmt.decorators:
        if deco.name not in _KNOWN_INSTRUCTION_DECORATORS:
            raise ValueError(
                f"unrecognized instruction decorator @{deco.name} on "
                f"'{stmt.name}'; known instruction decorators are: "
                f"{', '.join(sorted(_KNOWN_INSTRUCTION_DECORATORS))}"
            )
        names.add(deco.name)
    if "SIMULATE_ONLY" in names and "DECODE_ONLY" in names:
        raise ValueError(
            f"instruction '{stmt.name}' has both @SIMULATE_ONLY and "
            f"@DECODE_ONLY; these are mutually exclusive"
        )


def flatten_body(
    statements: Sequence[GadgetStatement],
    *,
    for_simulate: bool = False,
) -> list[GadgetStatement]:
    """Expand ``REPEAT`` blocks inline; filter by decode/simulate view.

    Parameters
    ----------
    for_simulate : bool
        ``False`` (default) — decode view: exclude ``@SIMULATE_ONLY``.
        ``True`` — simulate view: exclude ``@DECODE_ONLY``.
    """
    flat: list[GadgetStatement] = []
    for stmt in statements:
        if isinstance(stmt, RepeatBlock):
            body = list(stmt.body)
            for _ in range(stmt.count):
                flat.extend(flatten_body(body, for_simulate=for_simulate))
        else:
            _validate_instruction_decorators(stmt)
            if not for_simulate and _is_simulate_only(stmt):
                continue
            if for_simulate and _is_decode_only(stmt):
                continue
            flat.append(stmt)
    return flat


def max_qubit_index(statements: Sequence[GadgetStatement]) -> int:
    """Return the largest physical qubit index referenced anywhere in the body."""
    max_idx = -1
    for stmt in statements:
        if isinstance(stmt, Instruction):
            for target in stmt.targets:
                if isinstance(target, QubitTarget):
                    max_idx = max(max_idx, target.index)
                elif isinstance(target, PauliTarget):
                    max_idx = max(max_idx, target.index)
        elif isinstance(stmt, RepeatBlock):
            max_idx = max(max_idx, max_qubit_index(list(stmt.body)))
        elif isinstance(stmt, (InputPort, OutputPort)):
            for q in stmt.qubit_indices:
                max_idx = max(max_idx, q)
    return max_idx


_PAULI_NAME_TO_INT: dict[str, int] = {"I": 0, "X": 1, "Y": 2, "Z": 3}


def pauli_product_to_stim(
    product: PauliProduct,
    num_qubits: int,
    qubit_map: dict[int, int] | None = None,
) -> stim.PauliString:
    """Convert a :class:`PauliProduct` to a ``stim.PauliString``.

    Parameters
    ----------
    product:
        The Pauli product (code-local qubit indices).
    num_qubits:
        Total number of qubits in the target Pauli string.
    qubit_map:
        Optional mapping from code-local to physical qubit indices.
        When ``None``, indices are used as-is (identity mapping).
    """
    ps = stim.PauliString(num_qubits)
    for term in product.terms:
        phys = qubit_map[term.index] if qubit_map is not None else term.index
        ps[phys] = _PAULI_NAME_TO_INT[term.pauli.upper()]
    return ps


# ---------------------------------------------------------------------------
# Measurement layout and code metadata helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeasurementLayout:
    """Measurement index layout for a gadget.

    Global index space: ``[input-virtual | internal | output-virtual]``.
    """

    input_virtual_count: int
    internal_count: int

    @property
    def ov_start(self) -> int:
        """First output-virtual index."""
        return self.input_virtual_count + self.internal_count


def compute_layout(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
) -> MeasurementLayout:
    """Compute the measurement layout for a gadget.

    The global index space is ``[input-virtual | internal | output-virtual]``.
    Measurement counting uses ``stim.Circuit.num_measurements`` to stay
    in sync with all Stim gate types automatically.
    """
    input_virtual_count = 0
    for stmt in gadget.body:
        if isinstance(stmt, InputPort):
            input_virtual_count += len(codes[stmt.code_name].stabilizers)

    internal_count = _body_to_stim_circuit(
        flatten_body(list(gadget.body))
    ).num_measurements

    return MeasurementLayout(
        input_virtual_count=input_virtual_count,
        internal_count=internal_count,
    )


def resolve_measurement_ref_global(
    target: MeasurementRefTarget,
    running: int,
    input_ports: Sequence[InputPort],
    output_ports: Sequence[OutputPort],
    codes: dict[str, CodeDefinition],
    internal_count: int,
    gadget_name: str,
) -> int:
    """Resolve any of the four measurement-reference forms to a global index.

    The global index is into the gadget's full measurement sequence:
    ``[input-virtuals | internal/physical | output-virtuals]``.

    ``running`` is the cumulative count of measurements emitted at the
    source-position of *target* (only used by the ``rec[-k]`` form).
    ``internal_count`` is the total number of internal physical
    measurements in the gadget; used to bounds-check ``M<i>``.
    """
    if isinstance(target, MeasurementRecordTarget):
        if target.offset < 1:
            raise ValueError(
                f"in GADGET {gadget_name!r}: rec[-k] requires k >= 1; "
                f"got rec[-{target.offset}]"
            )
        global_index = running - target.offset
        if global_index < 0:
            raise ValueError(
                f"in GADGET {gadget_name!r}: rec[-{target.offset}] is out "
                f"of range: only {running} measurement(s) have been emitted "
                f"at this point in the gadget body"
            )
        return global_index
    if isinstance(target, PhysicalMeasurementTarget):
        if target.index < 0 or target.index >= internal_count:
            raise ValueError(
                f"in GADGET {gadget_name!r}: M{target.index} out of range; "
                f"gadget has {internal_count} internal physical measurement(s)"
            )
        input_virtual_count = sum(
            len(codes[p.code_name].stabilizers) for p in input_ports
        )
        return input_virtual_count + target.index
    if isinstance(target, InputVirtualTarget):
        if target.port_index < 0 or target.port_index >= len(input_ports):
            raise ValueError(
                f"in GADGET {gadget_name!r}: IN{target.port_index}.S{target.stabilizer_index} "
                f"refers to INPUT port {target.port_index} but the gadget "
                f"declares only {len(input_ports)} INPUT port(s)"
            )
        port = input_ports[target.port_index]
        n_stabs = len(codes[port.code_name].stabilizers)
        if target.stabilizer_index < 0 or target.stabilizer_index >= n_stabs:
            raise ValueError(
                f"in GADGET {gadget_name!r}: IN{target.port_index}.S{target.stabilizer_index} "
                f"out of range; INPUT port {target.port_index} (code "
                f"{port.code_name!r}) has {n_stabs} stabilizer(s)"
            )
        offset = sum(
            len(codes[p.code_name].stabilizers)
            for p in input_ports[: target.port_index]
        )
        return offset + target.stabilizer_index
    if isinstance(target, OutputVirtualTarget):
        if target.port_index < 0 or target.port_index >= len(output_ports):
            raise ValueError(
                f"in GADGET {gadget_name!r}: OUT{target.port_index}.S{target.stabilizer_index} "
                f"refers to OUTPUT port {target.port_index} but the gadget "
                f"declares only {len(output_ports)} OUTPUT port(s)"
            )
        port = output_ports[target.port_index]
        n_stabs = len(codes[port.code_name].stabilizers)
        if target.stabilizer_index < 0 or target.stabilizer_index >= n_stabs:
            raise ValueError(
                f"in GADGET {gadget_name!r}: OUT{target.port_index}.S{target.stabilizer_index} "
                f"out of range; OUTPUT port {target.port_index} (code "
                f"{port.code_name!r}) has {n_stabs} stabilizer(s)"
            )
        input_virtual_count = sum(
            len(codes[p.code_name].stabilizers) for p in input_ports
        )
        offset = (
            input_virtual_count
            + internal_count
            + sum(
                len(codes[p.code_name].stabilizers)
                for p in output_ports[: target.port_index]
            )
        )
        return offset + target.stabilizer_index
    raise TypeError(
        f"in GADGET {gadget_name!r}: unsupported measurement reference: {target!r}"
    )


# ---------------------------------------------------------------------------
# Unified frame: stabilizer generator selection & destabilizer computation
# ---------------------------------------------------------------------------


@dataclass
class SelectedGenerators:
    """Result of selecting independent stabilizer generators from a code.

    Attributes
    ----------
    generator_indices:
        Indices into ``code.stabilizers`` of the selected generators,
        in the order they were selected (preserves declaration order).
    decomposition:
        ``len(code.stabilizers) × len(generator_indices)`` GF(2) matrix.
        Row ``i`` expresses ``code.stabilizers[i]`` as a XOR of the
        selected generators (columns).  A generator row has exactly
        one ``1`` (itself); a redundant stabilizer row has multiple
        ``1``s indicating which generators it decomposes into.
    destabilizer_paulis:
        One ``stim.PauliString`` per selected generator.
        ``destabilizer_paulis[j]`` anticommutes with generator ``j``
        and commutes with all other generators and all logicals.
    """

    generator_indices: list[int]
    decomposition: list[list[int]]
    destabilizer_paulis: list[stim.PauliString]


def select_stabilizer_generators(
    code: CodeDefinition,
) -> SelectedGenerators:
    """Select independent stabilizer generators and compute destabilizers.

    Results are cached on the ``CodeDefinition`` object so repeated
    calls for the same code are free.

    Iterates the code's listed stabilizers in declaration order and
    includes each one that increases the GF(2) rank (greedy selection).
    This ensures selected generators are always user-declared operators
    — never linear combinations — making debugging straightforward.

    The destabilizers are computed via ``stim.Tableau.from_stabilizers``:
    the X output at index ``j`` (within the stabilizer indices) is the
    destabilizer of generator ``j``.

    Parameters
    ----------
    code:
        A code with ``n`` physical qubits, ``k`` logical qubits, and
        a list of stabilizer definitions.

    Returns
    -------
    SelectedGenerators
        Selected generator indices, decomposition matrix, and
        destabilizer Pauli strings.
    """
    cached = getattr(code, "_selected_generators", None)
    if cached is not None:
        return cached

    n = code.n

    # Build binary representation of each stabilizer.
    # A stabilizer is a product of single-qubit Paulis. For rank
    # computation we use a 2n-bit binary vector (X part | Z part).
    def _stab_to_binary(stab: PauliProduct) -> list[int]:
        vec = [0] * (2 * n)
        for term in stab.terms:
            p = term.pauli.upper()
            q = term.index
            if p in ("X", "Y"):
                vec[q] = vec[q] ^ 1
            if p in ("Z", "Y"):
                vec[n + q] = vec[n + q] ^ 1
        return vec

    stab_vecs = [_stab_to_binary(s) for s in code.stabilizers]

    # Greedy selection: iterate in order, include if rank increases
    generator_indices: list[int] = []
    current_rows: list[list[int]] = []
    current_rank = 0

    for idx, vec in enumerate(stab_vecs):
        candidate = current_rows + [vec]
        candidate_matrix = BitMatrix(candidate)
        new_rank = rank(candidate_matrix)
        if new_rank > current_rank:
            generator_indices.append(idx)
            current_rows.append(vec)
            current_rank = new_rank

    num_generators = len(generator_indices)

    # Decompose every listed stabilizer into selected generators
    # using paulimer's PauliGroup factorization.
    identity_map = {i: i for i in range(n)}
    gen_sparse = [
        _pauli_product_to_sparse(code.stabilizers[gi], identity_map)
        for gi in generator_indices
    ]
    group = PauliGroup(gen_sparse, all_commute=True)
    all_stab_sparse = [
        _pauli_product_to_sparse(s, identity_map) for s in code.stabilizers
    ]
    factorizations = group.indexed_factorizations_of(all_stab_sparse)
    neg_factorizations = group.indexed_factorizations_of([-s for s in all_stab_sparse])
    decomposition: list[list[int]] = []
    for stab_idx in range(len(all_stab_sparse)):
        fact = factorizations[stab_idx] or neg_factorizations[stab_idx]
        row = [0] * num_generators
        indices, _phase = fact
        for idx in indices:
            row[idx] = 1
        decomposition.append(row)

    # Compute destabilizers via stim.Tableau.from_stabilizers.
    gen_stim = [
        pauli_product_to_stim(code.stabilizers[i], n) for i in generator_indices
    ]
    logical_x_stim = [pauli_product_to_stim(lg.x_operator, n) for lg in code.logicals]
    logical_z_stim = [pauli_product_to_stim(lg.z_operator, n) for lg in code.logicals]

    # Tableau.from_stabilizers builds a full n-qubit tableau from
    # commuting Pauli operators.  For a standard [[n,k]] code we
    # provide exactly n operators (n-k generators + k logical Z).
    # For a subsystem code we have fewer than n-k stabilizers in total;
    # allow_underconstrained lets stim fill the gauge degrees of
    # freedom automatically.
    all_z_ops = gen_stim + logical_z_stim

    tableau = stim.Tableau.from_stabilizers(all_z_ops, allow_underconstrained=True)

    # Compute destabilizers D_i such that:
    #   {D_i, S_i} ≠ 0, [D_i, S_j] = 0 (j≠i), [D_i, LX_j] = 0, [D_i, LZ_j] = 0
    # Start from the tableau destabilizer D_i (commutes with S_j, LZ_j but
    # may anticommute with LX_j). Fix by multiplying by products of LZ:
    #   If {D_i, LX_j} ≠ 0 then D_i → D_i * LZ_j
    # This works because: LZ_j anticommutes with LX_j (fixes it),
    # LZ_j commutes with LX_m for m≠j, LZ_j commutes with all S_k.
    fixed_destabilizers: list[stim.PauliString] = []
    for i in range(num_generators):
        d = tableau.x_output(i)
        for j, lx in enumerate(logical_x_stim):
            if not d.commutes(lx):
                d = d * logical_z_stim[j]
        fixed_destabilizers.append(d)

    result = SelectedGenerators(
        generator_indices=generator_indices,
        decomposition=decomposition,
        destabilizer_paulis=fixed_destabilizers,
    )
    code._selected_generators = result  # type: ignore[attr-defined]
    return result


def num_frame_columns(code: CodeDefinition) -> int:
    """Return the number of Pauli-frame columns for a code.

    The unified frame has ``2k`` logical columns (X/Z per logical qubit)
    plus one column per selected stabilizer generator.
    """
    sel = select_stabilizer_generators(code)
    return 2 * len(code.logicals) + len(sel.generator_indices)


class PortColumnLayout:
    """Column layout information for a set of input or output ports.

    Computes and caches the mapping between frame column indices and
    their semantic meaning (logical observable or stabilizer generator).

    The frame layout per port is ``[LX0, LZ0, ..., LX_{k-1}, LZ_{k-1},
    S_gen0, ..., S_gen_{g-1}]``, concatenated across ports.

    Attributes
    ----------
    logical_columns : set of int
        Column indices that are logical (not stabilizer).
    generator_map : dict mapping int to (int, int)
        Maps each stabilizer column index to
        ``(port_index, stabilizer_list_index)``.
    col_to_obs : dict mapping int to (int, bool)
        Maps each logical column index to ``(observable_index, is_x)``.
    logical_qubit_columns : list of (int, int)
        One ``(x_column, z_column)`` pair per logical qubit, ordered
        continuously across all ports.  ``logical_qubit_columns[i]``
        gives the frame column indices for global logical qubit ``i``.
    stab_to_column : list of int or None
        One entry per declared stabilizer across all ports (in port
        declaration order).  Generator stabilizers map to their frame
        column index; redundant stabilizers map to ``None``.
    """

    def __init__(
        self,
        ports: list[InputPort] | list[OutputPort],
        codes: dict[str, CodeDefinition],
    ) -> None:
        self.logical_columns: set[int] = set()
        self.generator_map: dict[int, tuple[int, int]] = {}
        self.col_to_obs: dict[int, tuple[int, bool]] = {}
        self.obs_to_port: dict[int, tuple[int, int]] = {}
        self.logical_qubit_columns: list[tuple[int, int]] = []
        self.stab_to_column: list[int | None] = []
        self.stab_decomposed_columns: list[list[int]] = []
        self.per_port_offsets: list[int] = []
        self.per_port_stab_offsets: list[int] = []
        self.port_kind: Literal["IN", "OUT"] | None
        if not ports:
            self.port_kind = None
        else:
            self.port_kind = "IN" if isinstance(ports[0], InputPort) else "OUT"

        offset = 0
        obs_index = 0
        stab_count = 0
        for port_index, port in enumerate(ports):
            self.per_port_offsets.append(offset)
            self.per_port_stab_offsets.append(stab_count)
            code = codes[port.code_name]
            k = len(code.logicals)
            sel = select_stabilizer_generators(code)
            n_gen = len(sel.generator_indices)
            gen_set = set(sel.generator_indices)

            for i in range(2 * k):
                self.logical_columns.add(offset + i)
            for i in range(k):
                self.col_to_obs[offset + 2 * i] = (obs_index + i, True)
                self.col_to_obs[offset + 2 * i + 1] = (obs_index + i, False)
                self.obs_to_port[obs_index + i] = (port_index, i)
                self.logical_qubit_columns.append((offset + 2 * i, offset + 2 * i + 1))
            for seq, gi in enumerate(sel.generator_indices):
                self.generator_map[offset + 2 * k + seq] = (port_index, gi)
            stab_base = offset + 2 * k
            for stab_idx in range(len(code.stabilizers)):
                if stab_idx in gen_set:
                    gen_j = sel.generator_indices.index(stab_idx)
                    self.stab_to_column.append(stab_base + gen_j)
                else:
                    self.stab_to_column.append(None)
                decomp = sel.decomposition[stab_idx]
                self.stab_decomposed_columns.append(
                    [stab_base + gj for gj, c in enumerate(decomp) if c]
                )

            obs_index += k
            offset += 2 * k + n_gen
            stab_count += len(code.stabilizers)

        self.num_columns = offset
        self.num_ports = len(ports)

    def render_logical_labels(
        self,
        columns: set[int],
        *,
        combine_xz_to_y: bool = True,
        port_qualified: bool = True,
    ) -> list[str]:
        """Render logical column indices as ``LX``/``LZ``/``LY`` labels.

        When ``combine_xz_to_y`` is ``True`` (the default), an observable
        with both its X and Z columns present is rendered as a single
        ``LY{i}`` label.  This matches the semantics of an ``ERROR`` row,
        where the residual is a single Pauli operator (``Y = XZ`` up to
        phase) on each logical qubit.

        When ``combine_xz_to_y`` is ``False``, the X and Z columns are
        rendered as two separate ``LZ{i}`` and ``LX{i}`` labels.  This
        matches the semantics of a ``PROPAGATE`` row or a ``# flipped
        by:`` comment, where each label denotes one frame-column
        contribution to an XOR sum and combining them into ``LY{i}``
        would misleadingly suggest a single Pauli operator.

        When ``port_qualified`` is ``True`` (the default) and the layout
        knows its port kind, labels are emitted as
        ``IN<p>.L<P><i>``/``OUT<p>.L<P><i>`` with port-local logical
        indices.  When ``False``, labels use the legacy global
        ``L<P><i>`` form.
        """
        obs_x: set[int] = set()
        obs_z: set[int] = set()
        for c in columns:
            if c in self.col_to_obs:
                obs_idx, is_x = self.col_to_obs[c]
                if is_x:
                    obs_x.add(obs_idx)
                else:
                    obs_z.add(obs_idx)

        def _label(pauli: str, global_obs: int) -> str:
            if port_qualified and self.port_kind is not None:
                port_idx, local_idx = self.obs_to_port[global_obs]
                return f"{self.port_kind}{port_idx}.L{pauli}{local_idx}"
            return f"L{pauli}{global_obs}"

        parts: list[str] = []
        for oi in sorted(obs_x | obs_z):
            x_bit = oi in obs_x
            z_bit = oi in obs_z
            if x_bit and z_bit and combine_xz_to_y:
                parts.append(_label("Y", oi))
            else:
                if z_bit:
                    parts.append(_label("X", oi))
                if x_bit:
                    parts.append(_label("Z", oi))
        return parts


def _apply_decomposed_instructions(
    state: _BuildState,
    stmts: Sequence[GadgetStatement],
) -> None:
    """Run the decomposed circuit through paulimer.

    Converts the gadget body to a ``stim.Circuit``, decomposes it, then
    dispatches the ``{H, S, CX, M, R, MPAD}`` gate types.  This avoids maintaining
    per-gate branches for every Stim gate.
    """
    circuit = _body_to_stim_circuit(stmts)
    decomposed = circuit.decomposed()

    for inst in decomposed:
        name = inst.name
        targets = inst.targets_copy()
        if name == "H":
            for t in targets:
                state.sim.apply_unitary(UnitaryOpcode.Hadamard, [t.value])
        elif name == "S":
            for t in targets:
                state.sim.apply_unitary(UnitaryOpcode.SqrtZ, [t.value])
        elif name == "CX":
            for i in range(0, len(targets), 2):
                ctrl = targets[i]
                tgt = targets[i + 1]
                if ctrl.is_measurement_record_target:
                    # Classically-controlled X: CX rec[-k] qubit
                    rec_idx = len(state.real_measurements) + ctrl.value
                    outcome = state.real_measurements[rec_idx]
                    state.sim.apply_conditional_pauli(
                        SparsePauli.x(tgt.value), [outcome], parity=True
                    )
                else:
                    state.sim.apply_unitary(
                        UnitaryOpcode.ControlledX,
                        [ctrl.value, tgt.value],
                    )
        elif name == "M":
            for t in targets:
                outcome = state.sim.measure(SparsePauli.z(t.value))
                state.real_measurements.append(outcome)
                state.real_measurement_inverted.append(t.is_inverted_result_target)
        elif name == "R":
            for t in targets:
                outcome = state.sim.measure(SparsePauli.z(t.value))
                state.sim.apply_conditional_pauli(
                    SparsePauli.x(t.value), [outcome], parity=True
                )
        elif name == "MPAD":
            for t in targets:
                outcome = state.sim.measure(SparsePauli.identity())
                state.real_measurements.append(outcome)
                state.real_measurement_inverted.append(bool(t.value))
        else:
            raise ValueError(
                f"Unexpected instruction '{name}' in decomposed circuit. "
                f"stim.Circuit.decomposed() should only produce "
                f"{{H, S, CX, M, R, MPAD}}."
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def merge_checks_with_manual(
    auto_checks: Sequence[Check],
    manual_checks: Sequence[Check],
    total_measurements: int,
) -> list[Check]:
    """Emit ``manual_checks`` first, then auto rows that extend the basis.

    Each manual check is verified to lie in the row space spanned by
    ``auto_checks`` (augmented with a shift column) — otherwise a
    ``ValueError`` is raised identifying the invalid check.
    """
    width = total_measurements + 1

    def _fill(matrix: BitMatrix, row_idx: int, check: Check) -> None:
        members, parity = check
        for idx in members:
            assert idx >= 0 and idx < total_measurements
            matrix[row_idx, idx] = True
        if parity:
            matrix[row_idx, total_measurements] = True

    def _single_row(check: Check) -> BitMatrix:
        matrix = BitMatrix.zeros(1, width)
        _fill(matrix, 0, check)
        return matrix

    auto_matrix = BitMatrix.zeros(len(auto_checks), width)
    for row_idx, check in enumerate(auto_checks):
        _fill(auto_matrix, row_idx, check)
    auto_rank = rank(auto_matrix)

    for check in manual_checks:
        combined = vstack([auto_matrix, _single_row(check)])
        if rank(combined) != auto_rank:
            raise ValueError(
                f"manual CHECK {check} is not a valid parity check of the "
                f"gadget's outcome code"
            )

    result: list[Check] = list(manual_checks)
    current = BitMatrix.zeros(len(manual_checks), width)
    for row_idx, check in enumerate(manual_checks):
        _fill(current, row_idx, check)
    current_rank = rank(current)

    for check in auto_checks:
        if current_rank == auto_rank:
            break
        trial = vstack([current, _single_row(check)])
        trial_rank = rank(trial)
        if trial_rank > current_rank:
            result.append(check)
            current = trial
            current_rank = trial_rank

    return result


def regroup_checks(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
    checks: Sequence[Check],
    total_measurements: int,
) -> tuple[
    list[Check],
    list[Check],
]:
    """Split ``checks`` into *finished* and *unfinished* groups.

    The gadget's global measurement layout is

        ``[input-virtual | internal | output-virtual]``,

    where the *output-virtual* block has one index per stabilizer of
    each ``OUTPUT`` port, in port-declaration order.  In the JIT
    runtime every output stabilizer measurement must appear in **one
    and exactly one** parity check.  This function enforces that
    normal form by Gaussian-eliminating the check row space with the
    output-virtual columns prioritised as pivots.

    Returns ``(finished_checks, unfinished_checks)`` where

    - ``finished_checks`` contains no output-virtual indices;
    - ``unfinished_checks`` has length ``num_output_virtual`` and is
      ordered by output stabilizer: ``unfinished_checks[k]`` is the
      parity check whose sole output-virtual member is the ``k``-th
      output stabilizer (global index ``total_measurements -
      num_output_virtual + k``).

    Raises :class:`ValueError` when some output stabilizer cannot be
    written as a linear combination of the input-virtual and internal
    measurements under the check row space — i.e. when the gadget is
    malformed and that stabilizer could not be finalised.  The error
    message identifies each offending ``(OUTPUT port, stabilizer)``
    pair.
    """
    layout = compute_layout(gadget, codes)
    input_virtual_count = layout.input_virtual_count
    internal_count = layout.internal_count

    output_virtual_info: list[tuple[str, int, int, str]] = []
    port_index = 0
    for stmt in gadget.body:
        if isinstance(stmt, OutputPort):
            code = codes[stmt.code_name]
            for stab_index, stab in enumerate(code.stabilizers):
                output_virtual_info.append(
                    (stmt.code_name, port_index, stab_index, str(stab))
                )
            port_index += 1

    num_ov = len(output_virtual_info)
    ov_start = input_virtual_count + internal_count
    if ov_start + num_ov != total_measurements:
        raise ValueError(
            f"measurement layout mismatch in gadget {gadget.name!r}: "
            f"input_virtual={input_virtual_count}, internal={internal_count}, "
            f"output_virtual={num_ov} (sum={ov_start + num_ov}) does not match "
            f"caller-supplied total_measurements={total_measurements}"
        )

    if num_ov == 0:
        return list(checks), []

    # Validate every member index up-front so the matrix-building loop
    # below cannot silently land outside its column space (which would
    # corrupt the echelonization or raise an opaque IndexError).
    for check_idx, (members, _parity) in enumerate(checks):
        for idx in members:
            if idx < 0 or idx >= total_measurements:
                raise ValueError(
                    f"check #{check_idx} references measurement index {idx}, "
                    f"which is outside the gadget's measurement range "
                    f"[0, {total_measurements}); layout: "
                    f"input_virtual={input_virtual_count}, "
                    f"internal={internal_count}, output_virtual={num_ov}"
                )

    # Column layout: [OV cols | other cols | shift].  Placing OV
    # columns first makes echelonize prefer them as pivots.
    num_other = ov_start
    width = num_ov + num_other + 1
    matrix = BitMatrix.zeros(len(checks), width)
    for row_idx, (members, parity) in enumerate(checks):
        for idx in members:
            if idx >= ov_start:
                matrix[row_idx, idx - ov_start] = True
            else:
                matrix[row_idx, num_ov + idx] = True
        if parity:
            matrix[row_idx, num_ov + num_other] = True
    matrix.echelonize()

    finished: list[Check] = []
    unfinished_by_ov: list[Check | None] = [None] * num_ov

    def _row_to_check(row_idx: int) -> Check:
        members: set[int] = set()
        for col in range(num_ov):
            if matrix[row_idx, col]:
                members.add(ov_start + col)
        for col in range(num_other):
            if matrix[row_idx, num_ov + col]:
                members.add(col)
        parity = bool(matrix[row_idx, num_ov + num_other])
        return frozenset(members), parity

    for row_idx in range(matrix.row_count):
        leading_col = -1
        for col in range(width - 1):
            if matrix[row_idx, col]:
                leading_col = col
                break
        if leading_col < 0:
            if matrix[row_idx, num_ov + num_other]:
                raise ValueError(
                    "outcome code is inconsistent: a check has no members "
                    "but a non-zero expected parity"
                )
            continue
        check = _row_to_check(row_idx)
        if leading_col < num_ov:
            unfinished_by_ov[leading_col] = check
        else:
            finished.append(check)

    missing = [
        output_virtual_info[k]
        for k, entry in enumerate(unfinished_by_ov)
        if entry is None
    ]
    if missing:
        lines = [
            f"  OUTPUT port #{port} ({code_name}), stabilizer {stab_idx}: {stab}"
            for code_name, port, stab_idx, stab in missing
        ]
        raise ValueError(
            f"GADGET {gadget.name!r} is invalid: the following output "
            f"stabilizer(s) cannot be expressed as a linear combination "
            f"of input-virtual and internal measurements, so they cannot "
            f"be checked by the gadget's outcome code:\n" + "\n".join(lines)
        )

    unfinished: list[Check] = [cast(Check, entry) for entry in unfinished_by_ov]
    return finished, unfinished


def derive_checks_auto(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
) -> tuple[list[Check], int]:
    """Derive the parity-check structure of a gadget via paulimer."""
    body = list(gadget.body)
    qubit_count = max_qubit_index(body) + 1
    if qubit_count < 0:
        qubit_count = 0

    sim = OutcomeCompleteSimulation(qubit_count)
    state = _BuildState(sim=sim)

    # The check structure of a gadget describes measurement correlations
    # that hold regardless of the input logical state. Paulimer starts
    # every qubit in |0>, which would make individual Z measurements on
    # input qubits deterministic. To recover the input-state-agnostic
    # semantics, randomize each input qubit first by conjugating
    # it with a pair of caller-supplied random bits.
    input_qubits: set[int] = set()
    for stmt in body:
        if isinstance(stmt, InputPort):
            input_qubits.update(stmt.qubit_indices)
    for qubit in sorted(input_qubits):
        rx = sim.allocate_random_bit()
        sim.apply_conditional_pauli(SparsePauli.x(qubit), [rx], parity=True)
        rz = sim.allocate_random_bit()
        sim.apply_conditional_pauli(SparsePauli.z(qubit), [rz], parity=True)

    input_virtual_outcomes: list[int] = []
    for stmt in body:
        if isinstance(stmt, InputPort):
            code = codes[stmt.code_name]
            qubit_map = {i: q for i, q in enumerate(stmt.qubit_indices)}
            for stabilizer in code.stabilizers:
                input_virtual_outcomes.append(
                    sim.measure(_pauli_product_to_sparse(stabilizer, qubit_map))
                )

    _apply_decomposed_instructions(state, flatten_body(body))

    output_virtual_outcomes: list[int] = []
    for stmt in body:
        if isinstance(stmt, OutputPort):
            code = codes[stmt.code_name]
            qubit_map = {i: q for i, q in enumerate(stmt.qubit_indices)}
            for stabilizer in code.stabilizers:
                output_virtual_outcomes.append(
                    sim.measure(_pauli_product_to_sparse(stabilizer, qubit_map))
                )

    sim_outcomes: list[int] = (
        input_virtual_outcomes + state.real_measurements + output_virtual_outcomes
    )
    # Build the inversion mask: False for input-virtual and output-virtual,
    # True/False for real measurements depending on whether M !q was used.
    inversions: list[bool] = (
        [False] * len(input_virtual_outcomes)
        + state.real_measurement_inverted
        + [False] * len(output_virtual_outcomes)
    )
    total_measurements = len(sim_outcomes)
    if total_measurements == 0:
        return [], 0

    outcome_matrix = sim.outcome_matrix
    outcome_shift = sim.outcome_shift
    n_random = outcome_matrix.column_count

    # Build an augmented matrix whose rows correspond to our (global) outcomes:
    #     [ M_row | e_i | shift_i ]
    # Echelonizing it lets us read off parity checks as the rows whose
    # "M_row" portion echelonizes to zero — the tracking columns then
    # identify which outcomes sum to a fixed shift bit.
    aug = BitMatrix.zeros(total_measurements, n_random + total_measurements + 1)
    for row, sim_idx in enumerate(sim_outcomes):
        for col in range(n_random):
            if outcome_matrix[sim_idx, col]:
                aug[row, col] = True
        aug[row, n_random + row] = True
        # XOR the paulimer shift with the inversion flag from M !q
        shift = outcome_shift[sim_idx] ^ inversions[row]
        if shift:
            aug[row, n_random + total_measurements] = True
    aug.echelonize()

    checks: list[Check] = []
    input_virtual_count = len(input_virtual_outcomes)
    for row in range(aug.row_count):
        if any(aug[row, col] for col in range(n_random)):
            continue
        members: set[int] = set()
        for k in range(total_measurements):
            if aug[row, n_random + k]:
                members.add(k)
        if not members:
            continue
        # Skip checks that involve ONLY input-virtual measurements.
        # These are properties of the CODE's stabilizer algebra
        # (e.g. redundant stabilizer relations), not of the gadget.
        if all(idx < input_virtual_count for idx in members):
            continue
        parity = bool(aug[row, n_random + total_measurements])
        checks.append((frozenset(members), parity))

    return checks, total_measurements


def parse_checks_manual(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
) -> tuple[list[Check], int]:
    """Parse explicit ``CHECK`` statements out of a gadget.

    Each ``CHECK`` target may use any of the four measurement-reference
    forms: ``rec[-k]``, ``M<i>``, ``IN<p>.S<s>``, or ``OUT<p>.S<s>``.
    Measurement counting uses ``stim.Circuit.num_measurements`` to
    stay in sync with all Stim gate types automatically.
    ``INPUT``/``OUTPUT`` ports add one virtual measurement per
    stabilizer.
    """
    running = 0
    running_by_order: list[tuple[int, CheckStatement]] = []
    num_input = 0
    num_output = 0
    input_ports: list[InputPort] = []
    output_ports: list[OutputPort] = []
    # Accumulate instructions between CHECKs, flush measurement count
    # via stim when a CHECK is encountered.
    pending_instructions: list[GadgetStatement] = []

    for stmt in flatten_body(list(gadget.body)):
        if isinstance(stmt, InputPort):
            input_ports.append(stmt)
            count = len(codes[stmt.code_name].stabilizers)
            num_input += count
            running += count
        elif isinstance(stmt, OutputPort):
            output_ports.append(stmt)
            count = len(codes[stmt.code_name].stabilizers)
            num_output += count
            running += count
        elif isinstance(stmt, CheckStatement):
            # Flush pending instructions to get their measurement count
            if pending_instructions:
                running += _body_to_stim_circuit(pending_instructions).num_measurements
                pending_instructions = []
            running_by_order.append((running, stmt))
        elif isinstance(stmt, Instruction):
            pending_instructions.append(stmt)

    # Flush remaining instructions after the last CHECK
    if pending_instructions:
        running += _body_to_stim_circuit(pending_instructions).num_measurements

    num_internal = running - num_input - num_output

    checks: list[Check] = []
    for snapshot, stmt in running_by_order:
        members: set[int] = set()
        for target in stmt.targets:
            if not isinstance(
                target,
                (
                    MeasurementRecordTarget,
                    PhysicalMeasurementTarget,
                    InputVirtualTarget,
                    OutputVirtualTarget,
                ),
            ):
                raise ValueError(
                    f"only measurement references (rec[-k], M<i>, IN<p>.S<s>, "
                    f"OUT<p>.S<s>) or FLIP are supported in CHECK statements; "
                    f"got {target!r}"
                )
            resolved = resolve_measurement_ref_global(
                target,
                running=snapshot,
                input_ports=input_ports,
                output_ports=output_ports,
                codes=codes,
                internal_count=num_internal,
                gadget_name=gadget.name,
            )
            if resolved in members:
                raise ValueError(
                    f"CHECK statement in GADGET {gadget.name!r} references "
                    f"measurement index {resolved} more than once ({target!s})"
                )
            members.add(resolved)
        checks.append((frozenset(members), stmt.flip))

    return checks, num_input + num_internal + num_output


def checks_equivalent(
    checks_a: Sequence[Check],
    checks_b: Sequence[Check],
    total_measurements: int,
) -> bool:
    """Return ``True`` iff two check sets span the same affine GF(2) code.

    Each check ``(members, parity)`` is encoded as a row in GF(2) of
    length ``total_measurements + 1`` with a ``1`` in each member column
    and ``parity`` in the final (affine) column. Two check sets are
    equivalent iff their row spaces coincide.
    """

    def _to_matrix(
        checks: Sequence[Check],
    ) -> BitMatrix:
        width = total_measurements + 1
        matrix = BitMatrix.zeros(len(checks), width)
        for row, (members, parity) in enumerate(checks):
            for idx in members:
                assert idx >= 0 and idx < total_measurements
                matrix[row, idx] = True
            if parity:
                matrix[row, total_measurements] = True
        return matrix

    matrix_a = _to_matrix(checks_a)
    matrix_b = _to_matrix(checks_b)
    rank_a = rank(matrix_a)
    rank_b = rank(matrix_b)
    rank_ab = rank(vstack([matrix_a, matrix_b]))
    return rank_a == rank_b == rank_ab
