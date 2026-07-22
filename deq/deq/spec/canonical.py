"""
=====================
Canonicalization
=====================

The canonical form of a deq-bin is a library and a reference program that contains:

* a single global port type, which concatenates all the unconnected output ports of the \
    instantiated gadgets in the reference program. By concatenation we mean that the list of \
    observables are concatenated in the order of the gadgets in the reference program
* a single global gadget type, which concatenates all the measurements of the instantiated gadgets \
    in the reference program. The gadget type is never :code:`transparent`.
* a single global check model, which concatenates all the checks and converts their references to \
    the global measurements of the single gadget type
* a single global error model, which concatenates all the errors, converts their references to the \
    global checks of the single check model, and converts their effects (residual and readout \
    flips) to the global residual and readout flips of the single gadget type
* the reference program that instantiates each of the above types exactly once, bind the check \
    model to the gadget, and attach the error model to the check model, without using any modifier\
    (the effect of the modifiers in the original reference program is reflected in the global types)

Prerequisite: the input deq-bin must satisfy the LibSpec and ProgSpec. By satisfying ProgSpec, \
    we can assure that all the remote gadgets and remote check models can be expanded, i.e., there \
    remains no remote references in the global check model and global error model

Design note: ``logical_correction`` absorption
==============================================

The canonical gadget type produced by :func:`merge` always has an EMPTY
``logical_correction`` matrix.  This is by design.  Conditional output
flips that the runtime would otherwise express as
``residual ^= lc · readouts`` are *absorbed* into ``correction_propagation``
and ``physical_correction`` (and into per-error ``residual``) during the
merge pipeline (see "step 9" inside :func:`merge`).

Motivation: the original encoding had redundancy — the same output flip
could be expressed via ``logical_correction × readout_propagation``
(input side) and ``logical_correction × readout.measurement_indices``
(measurement side) OR via ``correction_propagation`` and
``physical_correction`` directly.  Two byte-different libraries could
encode identical behavior.  Absorbing collapses this redundancy into a
single canonical representation: every static input → output and
measurement → output flip lives in ``cp`` / ``pc``; the merged
``logical_correction`` is reserved as "always empty" so equivalence
checks reduce to byte comparisons of the absorbed matrices.

The ``logical_correction`` field is NOT removed from the proto.  It is
still populated by individual GADGET types (via authoring constructs
like ``CONDITIONAL R<j> L<P><i>``); the absorption only happens when
those GADGETs are merged.  Existing serialized libraries that pre-date
this absorption still load and execute correctly because the runtime
formula ``residual ^= lc · readouts`` is a no-op when ``lc`` is empty.

"""

# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Collection

import deq.proto.deq_bin_pb2 as pb
import deq.proto.deq_jit_pb2 as jit_pb
import deq.proto.util_pb2 as util_pb
from deq.spec.program_validator import ExpandedProgram
from deq.spec.common import (
    Bijection,
    LocalObservableIndex,
    ObservableIndex,
    MeasurementIndex,
    PortIndex,
    ReadoutIndex,
    CheckIndex,
    ErrorIndex,
    OutputPortIndex,
    bitmatrix_from_sparse,
)


@dataclass
class CanonicalForm:
    library: pb.Library = field(default_factory=pb.Library)

    observable_map: "Bijection[ObservableIndex]" = field(default_factory=Bijection)
    measurement_map: "Bijection[MeasurementIndex]" = field(default_factory=Bijection)
    output_map: "Bijection[PortIndex]" = field(default_factory=Bijection)
    readout_map: "Bijection[ReadoutIndex]" = field(default_factory=Bijection)
    check_map: "Bijection[CheckIndex]" = field(default_factory=Bijection)
    error_map: "Bijection[ErrorIndex]" = field(default_factory=Bijection)

    @property
    def port_type(self) -> pb.PortType:
        return self.library.port_types[0]

    @property
    def gadget_type(self) -> pb.GadgetType:
        return self.library.gadget_types[0]

    @property
    def check_model_type(self) -> pb.CheckModelType:
        return self.library.check_model_types[0]

    @property
    def error_model_type(self) -> pb.ErrorModelType:
        return self.library.error_model_types[0]


def canonicalize(lib: pb.Library) -> CanonicalForm:

    program = ExpandedProgram.from_library(lib)
    all_gids = set(program.gadgets.keys())
    merged = merge(lib, all_gids, program=program)
    return merged.to_canonical_form()


def _gids_in_instantiation_order(program: ExpandedProgram) -> list[int]:
    """Return gids sorted by instantiation order, not by gid value."""
    return sorted(
        program.gadgets.keys(), key=lambda gid: program.gadget_instantiation_order[gid]
    )


def _cids_in_instantiation_order(program: ExpandedProgram) -> list[int]:
    """Return cids sorted by instantiation order, not by cid value."""
    return sorted(
        program.check_models.keys(),
        key=lambda cid: program.check_model_instantiation_order[cid],
    )


def _eids_in_instantiation_order(program: ExpandedProgram) -> list[int]:
    """Return eids sorted by instantiation order, not by eid value."""
    return sorted(
        program.error_models.keys(),
        key=lambda eid: program.error_model_instantiation_order[eid],
    )


@dataclass
class ExpandedGadgetMatrices:
    input_observables: list[LocalObservableIndex]
    output_observables: list[LocalObservableIndex]
    # from each input observable to the flipped output observables
    correction_propagation: dict[LocalObservableIndex, set[LocalObservableIndex]]
    naturally_flipped_observables: set[LocalObservableIndex]
    # from each input observable to the flipped readouts
    readout_propagation: dict[LocalObservableIndex, set[int]]
    naturally_flipped_readouts: set[int]
    # from each readout to the flipped output observables
    logical_correction: dict[int, set[LocalObservableIndex]]

    @staticmethod
    def from_gadget_type(
        gadget_type: pb.GadgetType, port_types: dict[int, pb.PortType]
    ) -> "ExpandedGadgetMatrices":

        input_observables: list[LocalObservableIndex] = []
        output_observables: list[LocalObservableIndex] = []
        for ports, observables in [
            (gadget_type.inputs, input_observables),
            (gadget_type.outputs, output_observables),
        ]:
            for port_index, port in enumerate(ports):
                port_type = port_types[port.ptype]
                for observable_index in range(len(port_type.observables)):
                    observables.append(
                        LocalObservableIndex(
                            port=port_index, observable_index=observable_index
                        )
                    )

        correction_propagation: dict[
            LocalObservableIndex, set[LocalObservableIndex]
        ] = {}
        naturally_flipped_observables: set[LocalObservableIndex] = set()

        for i, j in zip(
            gadget_type.correction_propagation.i, gadget_type.correction_propagation.j
        ):
            if j == len(input_observables):
                naturally_flipped_observables.add(output_observables[i])
            else:
                output_observable = output_observables[i]
                input_observable = input_observables[j]
                if input_observable not in correction_propagation:
                    correction_propagation[input_observable] = set()
                correction_propagation[input_observable].add(output_observable)

        readout_propagation: dict[LocalObservableIndex, set[int]] = {}
        naturally_flipped_readouts: set[int] = set()

        for i, j in zip(
            gadget_type.readout_propagation.i, gadget_type.readout_propagation.j
        ):
            if j == len(input_observables):
                naturally_flipped_readouts.add(i)
            else:
                input_observable = input_observables[j]
                if input_observable not in readout_propagation:
                    readout_propagation[input_observable] = set()
                readout_propagation[input_observable].add(i)

        logical_correction: dict[int, set[LocalObservableIndex]] = {}

        for i, j in zip(
            gadget_type.logical_correction.i, gadget_type.logical_correction.j
        ):
            output_observable = output_observables[i]
            if j not in logical_correction:
                logical_correction[j] = set()
            logical_correction[j].add(output_observable)

        return ExpandedGadgetMatrices(
            input_observables=input_observables,
            output_observables=output_observables,
            correction_propagation=correction_propagation,
            naturally_flipped_observables=naturally_flipped_observables,
            readout_propagation=readout_propagation,
            naturally_flipped_readouts=naturally_flipped_readouts,
            logical_correction=logical_correction,
        )


# ===================================================================
# merge() — merge a subset of gadgets into a single MergedGadget
# ===================================================================


@dataclass(frozen=True)
class MergedMeasurementRef:
    """A reference to a measurement in the merged gadget.

    If ``input_port`` is ``None``, this is a real measurement of the merged
    gadget and ``measurement_index`` is the global real-measurement index.

    If ``input_port`` is set, this is an input-virtual measurement:
    ``input_port`` identifies which merge-input port and
    ``measurement_index`` is the stabilizer index within that port's type.
    """

    input_port: int | None = None
    measurement_index: int = 0


@dataclass
class MergedCheck:
    measurements: list[MergedMeasurementRef]
    naturally_flipped: bool = False
    tag: str = ""


@dataclass
class MergedError:
    probability: float
    residual: list[int]
    readout_flips: list[int]
    finished_checks: list[int]
    unfinished_checks: list[int]
    tag: str = ""


@dataclass(frozen=True)
class _MergeInputPort:
    """A merge-boundary input port."""

    merge_gid: int  # the merge-set gadget that receives this input
    port_index: int  # the input port index within that gadget
    ptype: int
    peer_gid: int  # the non-merge gadget that produces this input
    peer_port: int  # the output port index on the peer gadget


@dataclass(frozen=True)
class _MergeOutputPort:
    """A merge-boundary output port."""

    merge_gid: int
    port_index: int
    ptype: int


@dataclass
class MergedGadget:
    """Result of merging a subset of gadgets in a Library.

    This is an intermediate representation that can be converted to either
    a ``JitGadgetType`` (for the compose builder) or a ``CanonicalForm``
    (for the existing canonicalize path).
    """

    input_ptypes: list[int]
    output_ptypes: list[int]
    measurements: list[pb.GadgetType.Measurement]
    readouts: list[pb.GadgetType.Readout]
    correction_propagation: util_pb.BitMatrix
    readout_propagation: util_pb.BitMatrix
    logical_correction: util_pb.BitMatrix
    physical_correction: util_pb.BitMatrix
    finished_checks: list[MergedCheck]
    # Checks that reference a measurement on a non-merge gadget sitting
    # on the merge's output boundary (see ``output_side_gids`` in
    # ``merge()``).
    unfinished_checks: list[MergedCheck]
    errors: list[MergedError]
    # Traceability maps (local → global within the merged gadget)
    measurement_map: "Bijection[MeasurementIndex]"
    observable_map: "Bijection[ObservableIndex]"
    readout_map: "Bijection[ReadoutIndex]"
    check_map: "Bijection[CheckIndex]"
    error_map: "Bijection[ErrorIndex]"

    def to_jit_gadget_type(
        self,
        gtype: int,
        name: str,
    ) -> jit_pb.JitGadgetType:
        """Convert to a ``JitGadgetType`` protobuf."""

        def _to_present(
            ref: MergedMeasurementRef,
        ) -> jit_pb.JitGadgetType.PresentMeasurement:
            if ref.input_port is not None:
                return jit_pb.JitGadgetType.PresentMeasurement(
                    input_port=ref.input_port,
                    measurement_index=ref.measurement_index,
                )
            return jit_pb.JitGadgetType.PresentMeasurement(
                measurement_index=ref.measurement_index,
            )

        def _to_jit_check(mc: MergedCheck) -> jit_pb.JitGadgetType.Check:
            # Sort measurements: input-virtual (has input_port) first,
            # then internal, each sorted by index — matching the order
            # produced by _build_check in jit_library_builder.py.
            sorted_meas = sorted(
                mc.measurements,
                key=lambda m: (
                    0 if m.input_port is not None else 1,
                    m.input_port or 0,
                    m.measurement_index,
                ),
            )
            return jit_pb.JitGadgetType.Check(
                base=pb.CheckModelType.Check(
                    tag=mc.tag,
                    naturally_flipped=mc.naturally_flipped,
                ),
                measurements=[_to_present(m) for m in sorted_meas],
            )

        jit_errors: list[jit_pb.JitGadgetType.Error] = []
        for me in self.errors:
            jit_errors.append(
                jit_pb.JitGadgetType.Error(
                    base=pb.ErrorModelType.Error(
                        tag=me.tag,
                        residual=me.residual,
                        readout_flips=me.readout_flips,
                        probability=me.probability,
                    ),
                    finished_checks=me.finished_checks,
                    unfinished_checks=me.unfinished_checks,
                )
            )

        base = pb.GadgetType(
            gtype=gtype,
            name=name,
            measurements=list(self.measurements),
            inputs=[pb.GadgetType.Port(ptype=pt) for pt in self.input_ptypes],
            outputs=[pb.GadgetType.Port(ptype=pt) for pt in self.output_ptypes],
            readouts=list(self.readouts),
            correction_propagation=self.correction_propagation,
            readout_propagation=self.readout_propagation,
            logical_correction=self.logical_correction,
            physical_correction=self.physical_correction,
        )
        return jit_pb.JitGadgetType(
            base=base,
            finished_checks=[_to_jit_check(c) for c in self.finished_checks],
            unfinished_checks=[_to_jit_check(c) for c in self.unfinished_checks],
            errors=jit_errors,
        )

    def to_canonical_form(self) -> CanonicalForm:
        """Convert to a ``CanonicalForm``.

        This is only valid when the merged gadget has no input ports and no
        unfinished checks (i.e. all gadgets in the circuit were merged).
        """
        assert (
            not self.input_ptypes
        ), "cannot convert to CanonicalForm: merged gadget has input ports"
        assert (
            not self.unfinished_checks
        ), "cannot convert to CanonicalForm: merged gadget has unfinished checks"

        canonical = CanonicalForm()
        canonical.observable_map = self.observable_map
        canonical.measurement_map = self.measurement_map
        canonical.readout_map = self.readout_map
        canonical.check_map = self.check_map
        canonical.error_map = self.error_map

        n_obs = len(self.observable_map)
        canonical.library.port_types.append(
            pb.PortType(
                ptype=1,
                observables=[pb.PortType.Observable()] * n_obs,
            )
        )

        canonical.library.gadget_types.append(
            pb.GadgetType(
                gtype=1,
                measurements=list(self.measurements),
                outputs=[pb.GadgetType.Port(ptype=1)],
                readouts=list(self.readouts),
                correction_propagation=self.correction_propagation,
                readout_propagation=self.readout_propagation,
                logical_correction=self.logical_correction,
                physical_correction=self.physical_correction,
            )
        )

        global_checks: list[pb.CheckModelType.Check] = []
        for mc in self.finished_checks:
            global_checks.append(
                pb.CheckModelType.Check(
                    measurements=sorted(
                        [
                            pb.CheckModelType.RemoteMeasurement(
                                measurement_index=m.measurement_index
                            )
                            for m in mc.measurements
                        ],
                        key=lambda m: m.measurement_index,
                    ),
                    naturally_flipped=mc.naturally_flipped,
                )
            )
        canonical.library.check_model_types.append(
            pb.CheckModelType(ctype=1, gtype=1, checks=global_checks)
        )

        global_errors: list[pb.ErrorModelType.Error] = []
        for me in self.errors:
            checks = [
                pb.ErrorModelType.RemoteCheck(check_index=ci)
                for ci in me.finished_checks
            ]
            global_errors.append(
                pb.ErrorModelType.Error(
                    checks=checks,
                    residual=me.residual,
                    readout_flips=me.readout_flips,
                    probability=me.probability,
                )
            )
        canonical.library.error_model_types.append(
            pb.ErrorModelType(etype=1, ctype=1, errors=global_errors)
        )

        canonical.library.program.extend(canonical_program())

        return canonical


def merge(
    lib: pb.Library,
    merge_gids: Collection[int],
    *,
    program: ExpandedProgram | None = None,
) -> MergedGadget:
    """Merge a subset of gadgets into a single ``MergedGadget``.

    The ``lib`` must be a valid Library (satisfies LibSpec and ProgSpec).
    ``merge_gids`` specifies which gadget instances (by gid) to merge.
    Ports connecting merge-set gadgets to non-merge-set gadgets become the
    input/output ports of the merged gadget.

    An optional pre-computed ``program`` (``ExpandedProgram``) can be
    passed to avoid redundant validation/expansion.
    """
    if program is None:
        program = ExpandedProgram.from_library(lib)
    gid_set = frozenset(merge_gids)

    # ── 1. Classify ports ────────────────────────────────────────────
    input_ports, output_ports = _classify_merge_ports(program, gid_set)

    # ── 2. Assign global indices ─────────────────────────────────────
    measurement_map: Bijection[MeasurementIndex] = Bijection()
    readout_map: Bijection[ReadoutIndex] = Bijection()
    observable_map: Bijection[ObservableIndex] = Bijection()

    ordered_gids = [
        gid for gid in _gids_in_instantiation_order(program) if gid in gid_set
    ]

    for gid in ordered_gids:
        gadget = program.gadgets[gid]
        gadget_type = program.gadget_types[gadget.gtype]

        for mi in range(len(gadget_type.measurements)):
            local_m = MeasurementIndex(gid=gid, measurement_index=mi)
            glob_m = MeasurementIndex(gid=1, measurement_index=len(measurement_map))
            measurement_map.add(local_m, glob_m)

        for ri in range(len(gadget_type.readouts)):
            local_r = ReadoutIndex(gid=gid, readout_index=ri)
            glob_r = ReadoutIndex(gid=1, readout_index=len(readout_map))
            readout_map.add(local_r, glob_r)

    # Assign output observable indices (merge output ports only)
    for op in output_ports:
        port_type = program.port_types[op.ptype]
        for oi in range(len(port_type.observables)):
            local_o = ObservableIndex(
                gid=op.merge_gid, port=op.port_index, observable_index=oi
            )
            glob_o = ObservableIndex(
                gid=1, port=0, observable_index=len(observable_map)
            )
            observable_map.add(local_o, glob_o)

    # ── 3. Observable propagation ────────────────────────────────────
    propagator = _MergePropagator.build(program, gid_set, observable_map, readout_map)

    # Count input observables
    num_input_obs = 0
    input_obs_base: dict[int, int] = {}  # input_port_index → first column
    for ip_idx, ip in enumerate(input_ports):
        input_obs_base[ip_idx] = num_input_obs
        port_type = program.port_types[ip.ptype]
        num_input_obs += len(port_type.observables)
    num_output_obs = len(observable_map)
    num_readouts = len(readout_map)

    # ── 4. Build propagation matrices ────────────────────────────────
    cp_set: set[tuple[int, int]] = set()  # (row, col)
    rp_set: set[tuple[int, int]] = set()  # (row, col)

    # Static (naturally-flipped) contributions
    for gid in ordered_gids:
        matrices = propagator.expanded_matrices[gid]
        for obs_local in matrices.naturally_flipped_observables:
            obs_global = obs_local.to_global(gid)
            for r in propagator.out_to_residual.get(obs_global, set()):
                cp_set ^= {(r, num_input_obs)}
            for r in propagator.out_to_readout.get(obs_global, set()):
                rp_set ^= {(r, num_input_obs)}
        for readout_local in matrices.naturally_flipped_readouts:
            local_ri = ReadoutIndex(gid=gid, readout_index=readout_local)
            if local_ri in readout_map.atob:
                global_ri = readout_map.atob[local_ri].readout_index
                rp_set ^= {(global_ri, num_input_obs)}

    # Input-dependent contributions: trace each input column
    for ip_idx, ip in enumerate(input_ports):
        port_type = program.port_types[ip.ptype]
        matrices = propagator.expanded_matrices[ip.merge_gid]
        col_base = input_obs_base[ip_idx]
        for oi in range(len(port_type.observables)):
            col = col_base + oi
            local_input = LocalObservableIndex(port=ip.port_index, observable_index=oi)
            # Trace through correction_propagation
            if local_input in matrices.correction_propagation:
                for out_local in matrices.correction_propagation[local_input]:
                    out_global = out_local.to_global(ip.merge_gid)
                    for r in propagator.out_to_residual.get(out_global, set()):
                        cp_set ^= {(r, col)}
                    for r in propagator.out_to_readout.get(out_global, set()):
                        rp_set ^= {(r, col)}
            # Trace through readout_propagation
            if local_input in matrices.readout_propagation:
                for readout_local in matrices.readout_propagation[local_input]:
                    local_ri = ReadoutIndex(
                        gid=ip.merge_gid, readout_index=readout_local
                    )
                    if local_ri in readout_map.atob:
                        global_ri = readout_map.atob[local_ri].readout_index
                        rp_set ^= {(global_ri, col)}

    # NOTE: both the ``correction_propagation`` and ``readout_propagation``
    # BitMatrices are built later (step 9) after the absorption pass extends
    # ``cp_set`` / ``rp_set`` with the absorbed contributions from ``cc_set``
    # / ``cc_readout_set``.

    # ── 5a. Logical correction ────────────────────────────────────
    # We build ``cc_set`` here but defer finalization of the
    # ``logical_correction`` matrix until after step 8 (errors), because
    # step 9 absorbs every cc entry into ``cp_set`` / ``pc_set`` and into
    # each error's ``residual`` and then clears ``cc_set``.  The final
    # merged ``logical_correction`` matrix is always empty by design;
    # this removes the matrix-redundancy between (logical_correction ×
    # readout_propagation) and (correction_propagation) and between
    # (logical_correction × readout.measurement_indices) and
    # (physical_correction).
    cc_set: set[tuple[int, int]] = set()
    # Parallel to ``cc_set`` but for conditional corrections whose flipped
    # output observable flows into a downstream *readout* rather than a
    # merged output observable. Entries are (readout_row, conditioning_readout).
    cc_readout_set: set[tuple[int, int]] = set()

    # Local conditional corrections (logical_correction matrix)
    for global_ri_val in range(num_readouts):
        local_ri = readout_map.btoa[ReadoutIndex(gid=1, readout_index=global_ri_val)]
        gid = local_ri.gid
        matrices = propagator.expanded_matrices[gid]
        global_obs_set: set[int] = set()
        global_readout_set: set[int] = set()
        if local_ri.readout_index in matrices.logical_correction:
            for out_local in matrices.logical_correction[local_ri.readout_index]:
                out_global = out_local.to_global(gid)
                global_obs_set ^= propagator.out_to_residual.get(out_global, set())
                global_readout_set ^= propagator.out_to_readout.get(out_global, set())
        for obs_idx in global_obs_set:
            cc_set ^= {(obs_idx, global_ri_val)}
        for readout_idx in global_readout_set:
            cc_readout_set ^= {(readout_idx, global_ri_val)}


    # Remote conditional corrections (from merge-set gadgets only)
    for gid in ordered_gids:
        if gid not in program.expanded_remote_conditional_corrections:
            continue
        expanded_readouts, remote_cc = program.expanded_remote_conditional_corrections[
            gid
        ]
        matrices = propagator.expanded_matrices[gid]

        col_to_global_readout: list[int] = []
        for local_readout in expanded_readouts:
            if local_readout not in readout_map.atob:
                raise ValueError(
                    f"remote_conditional_correction on merge-set gid={gid} "
                    f"references readout {local_readout} on a gadget outside "
                    f"the merge set; the conditional correction would be "
                    f"silently lost in the merged form"
                )
            col_to_global_readout.append(
                readout_map.atob[local_readout].readout_index
            )

        for row, col in zip(remote_cc.correction.i, remote_cc.correction.j):
            global_readout_idx = col_to_global_readout[col]
            out_local = matrices.output_observables[row]
            out_global = out_local.to_global(gid)
            global_obs_set = propagator.out_to_residual.get(out_global, set())
            for obs_idx in global_obs_set:
                cc_set ^= {(obs_idx, global_readout_idx)}
            for readout_idx in propagator.out_to_readout.get(out_global, set()):
                cc_readout_set ^= {(readout_idx, global_readout_idx)}

    # NOTE: ``logical_correction`` BitMatrix is built later (step 9)
    # after the absorption pass clears ``cc_set``.

    # ── 5b. Physical correction ──────────────────────────────────────
    pc_set: set[tuple[int, int]] = set()  # (row=global_obs, col=global_meas)

    for gid in ordered_gids:
        gadget_type = program.modified_gadget_types[gid]
        matrices = propagator.expanded_matrices[gid]
        pc = gadget_type.physical_correction

        for row_local, col_local in zip(pc.i, pc.j):
            # col_local is a local measurement index → remap to global.
            # Every merge-set gadget's own measurements were registered in
            # step 2, so ``local_m`` is always present here.
            local_m = MeasurementIndex(gid=gid, measurement_index=col_local)
            assert local_m in measurement_map.atob
            global_m = measurement_map.atob[local_m].measurement_index

            # row_local is a local output observable index → trace to global
            out_local = matrices.output_observables[row_local]
            out_global = out_local.to_global(gid)
            for global_obs in propagator.out_to_residual.get(out_global, set()):
                pc_set ^= {(global_obs, global_m)}

    num_measurements = len(measurement_map)
    # NOTE: ``physical_correction`` BitMatrix is built later (step 9)
    # after the absorption pass extends ``pc_set`` with the absorbed
    # contributions from ``cc_set``.

    # ── 5c. Track measurement deps per output observable ────────────
    # For each gadget's output observable, track which global measurements
    # contribute to it (via physical_correction and upstream propagation).
    # This is needed to expand readout measurement_indices: a readout that
    # depends on an input observable inherits all the measurement deps of
    # the predecessor's output observable that feeds that input.
    #
    # obs_meas_deps[(gid, output_port, obs_idx)] → set of global meas indices
    obs_meas_deps: dict[tuple[int, int, int], set[int]] = {}

    for gid in ordered_gids:
        gadget = program.gadgets[gid]
        gadget_type = program.modified_gadget_types[gid]
        matrices = propagator.expanded_matrices[gid]

        # 1. Collect input deps from predecessors via connectors
        input_deps: list[set[int]] = []
        for port_idx, connector in enumerate(gadget.connectors):
            pt = program.port_types[gadget_type.inputs[port_idx].ptype]
            for obs_idx in range(len(pt.observables)):
                pred_key = (connector.gid, int(connector.port), obs_idx)
                input_deps.append(set(obs_meas_deps.get(pred_key, set())))

        # 2. Initialize output deps
        for port_idx, port_spec in enumerate(gadget_type.outputs):
            pt = program.port_types[port_spec.ptype]
            for obs_idx in range(len(pt.observables)):
                obs_meas_deps[(gid, port_idx, obs_idx)] = set()

        # 3. Propagate via correction_propagation: input → output
        for input_local, output_locals in matrices.correction_propagation.items():
            # Find input's flat index
            input_flat = matrices.input_observables.index(input_local)
            if input_flat < len(input_deps):
                for out_local in output_locals:
                    key = (gid, out_local.port, out_local.observable_index)
                    obs_meas_deps[key] ^= input_deps[input_flat]

        # 4. Add physical_correction: measurement → output observable
        pc = gadget_type.physical_correction
        for row_local, col_local in zip(pc.i, pc.j):
            local_m = MeasurementIndex(gid=gid, measurement_index=col_local)
            global_m = measurement_map.atob[local_m].measurement_index
            out_local = matrices.output_observables[row_local]
            key = (gid, out_local.port, out_local.observable_index)
            obs_meas_deps[key] ^= {global_m}

        # 5. Add logical_correction: readout → output observable
        for readout_idx, output_locals in matrices.logical_correction.items():
            local_ri = ReadoutIndex(gid=gid, readout_index=readout_idx)
            # Build the readout's full measurement set (original + inherited)
            orig = gadget_type.readouts[readout_idx]
            readout_meas: set[int] = set()
            for local_mi in orig.measurement_indices:
                lm = MeasurementIndex(gid=gid, measurement_index=local_mi)
                if lm in measurement_map.atob:
                    readout_meas ^= {measurement_map.atob[lm].measurement_index}
            # Add inherited deps from input observables via readout_propagation
            for input_local, readout_set in matrices.readout_propagation.items():
                if readout_idx in readout_set:
                    input_flat = matrices.input_observables.index(input_local)
                    if input_flat < len(input_deps):
                        readout_meas ^= input_deps[input_flat]
            # Now XOR readout_meas into output deps
            for out_local in output_locals:
                key = (gid, out_local.port, out_local.observable_index)
                obs_meas_deps[key] ^= readout_meas

        # 6. Add remote_conditional_correction: PROGRAM-/COMPOSE-level
        # ``CONDITIONAL`` synthesises an identity host whose modifier
        # conditionally flips one of its output observables based on a
        # *remote* gadget's readout.  Track the dependency by folding
        # the remote readout's measurement set into the flipped output
        # observable's deps; this lets downstream gadgets — including
        # MeasureZ/MeasureX-style readout-only gadgets — inherit the
        # correction via the usual propagation paths.
        if gid in program.expanded_remote_conditional_corrections:
            expanded_readouts, remote_cc = (
                program.expanded_remote_conditional_corrections[gid]
            )
            col_to_remote_meas: list[set[int]] = []
            for local_readout in expanded_readouts:
                remote_gid = local_readout.gid
                remote_readout_idx = local_readout.readout_index
                remote_gadget = program.gadgets[remote_gid]
                remote_gadget_type = program.gadget_types[remote_gadget.gtype]
                remote_orig = remote_gadget_type.readouts[remote_readout_idx]
                remote_matrices = propagator.expanded_matrices[remote_gid]
                remote_meas_set: set[int] = set()
                for local_mi in remote_orig.measurement_indices:
                    lm = MeasurementIndex(
                        gid=remote_gid, measurement_index=local_mi
                    )
                    if lm in measurement_map.atob:
                        remote_meas_set ^= {
                            measurement_map.atob[lm].measurement_index
                        }
                # Inherited deps from the remote gadget's input
                # observables that feed its readout via
                # ``readout_propagation``.
                for input_local, readout_set in (
                    remote_matrices.readout_propagation.items()
                ):
                    if remote_readout_idx not in readout_set:
                        continue
                    remote_connector = remote_gadget.connectors[input_local.port]
                    pred_key = (
                        remote_connector.gid,
                        int(remote_connector.port),
                        input_local.observable_index,
                    )
                    remote_meas_set ^= obs_meas_deps.get(pred_key, set())
                col_to_remote_meas.append(remote_meas_set)

            for row, col in zip(remote_cc.correction.i, remote_cc.correction.j):
                out_local = matrices.output_observables[row]
                key = (gid, out_local.port, out_local.observable_index)
                obs_meas_deps[key] ^= col_to_remote_meas[col]

    # ── 6. Build measurements and readouts ───────────────────────────
    merged_measurements: list[pb.GadgetType.Measurement] = []
    for gid in ordered_gids:
        gadget = program.gadgets[gid]
        gadget_type = program.gadget_types[gadget.gtype]
        for m in gadget_type.measurements:
            merged_measurements.append(m)

    merged_readouts: list[pb.GadgetType.Readout] = []
    for global_ri_val in range(num_readouts):
        local_ri = readout_map.btoa[ReadoutIndex(gid=1, readout_index=global_ri_val)]
        gid = local_ri.gid
        gadget = program.gadgets[gid]
        gadget_type = program.gadget_types[gadget.gtype]
        matrices = propagator.expanded_matrices[gid]
        orig = gadget_type.readouts[local_ri.readout_index]

        # Start with the original measurement indices (remapped to global)
        meas_set: set[int] = set()
        for local_mi in orig.measurement_indices:
            global_mi = measurement_map.atob[
                MeasurementIndex(gid=gid, measurement_index=local_mi)
            ]
            meas_set ^= {global_mi.measurement_index}

        # XOR in measurement deps from input observables that affect this
        # readout via readout_propagation
        for input_local, readout_set in matrices.readout_propagation.items():
            if local_ri.readout_index in readout_set:
                input_flat = matrices.input_observables.index(input_local)
                connector = gadget.connectors[input_local.port]
                pred_key = (
                    connector.gid,
                    int(connector.port),
                    input_local.observable_index,
                )
                meas_set ^= obs_meas_deps.get(pred_key, set())

        merged_readouts.append(
            pb.GadgetType.Readout(tag=orig.tag, measurement_indices=sorted(meas_set))
        )

    # ── 7. Build checks ──────────────────────────────────────────────
    check_map: Bijection[CheckIndex] = Bijection()
    finished_checks: list[MergedCheck] = []
    unfinished_checks: list[MergedCheck] = []
    # unfinished checks are keyed by (gid, measurement_index) of the
    # output-virtual measurement that made the check unfinished.
    unfinished_by_key: dict[tuple[int, int], int] = {}

    # Build output-side gid set: non-merge gadgets connected to merge output ports.
    output_side_gids: set[int] = set()
    for op in output_ports:
        out_port_instance = OutputPortIndex(gid=op.merge_gid, port_index=op.port_index)
        if out_port_instance in program.peer_input:
            peer_input = program.peer_input[out_port_instance]
            if peer_input.gid not in gid_set:
                output_side_gids.add(peer_input.gid)

    # Track which non-merge gids sit on the input side
    input_side_gids: set[int] = set()
    for ip in input_ports:
        input_side_gids.add(ip.peer_gid)

    def _resolve_measurement_ref(
        gid: int, measurement_index: int
    ) -> tuple[MergedMeasurementRef | None, tuple[int, int] | None]:
        """Resolve a (gid, measurement_index) to a MergedMeasurementRef.

        Returns (ref, None) for real/input-virtual measurements, or
        (None, (out_port_idx, stab_idx)) for output-virtual measurements
        that make the containing check unfinished.
        """
        local_m = MeasurementIndex(gid=gid, measurement_index=measurement_index)
        if local_m in measurement_map.atob:
            # Real measurement in merge set
            return (
                MergedMeasurementRef(
                    measurement_index=measurement_map.atob[local_m].measurement_index
                ),
                None,
            )
        # Non-merge gadget measurement — classify as input or output side
        if gid in input_side_gids:
            # Find which input port this belongs to
            for ip_idx, ip in enumerate(input_ports):
                if ip.peer_gid == gid:
                    return (
                        MergedMeasurementRef(
                            input_port=ip_idx,
                            measurement_index=measurement_index,
                        ),
                        None,
                    )
        # Output-side: find which output port connects to this gadget.
        # This measurement is output-virtual at the merge boundary and
        # makes the containing check unfinished.
        if gid in output_side_gids:
            return None, (gid, measurement_index)
        # Should not reach here in a well-formed circuit
        raise ValueError(  # pragma: no cover
            f"measurement (gid={gid}, idx={measurement_index}) is from a "
            f"non-merge gadget that is neither input-side nor output-side"
        )

    # Process all check models from merge-set gadgets
    external_cids: set[int] = set()
    for eid in _eids_in_instantiation_order(program):
        error_model = program.error_models[eid]
        check_model = program.check_models[error_model.cid]
        if check_model.gid not in gid_set:
            continue
        error_model_type = program.modified_error_model_types[eid]
        remote_cid_vec = program.remote_check_model_cid_vecs[eid]
        for error in error_model_type.errors:
            for c in error.checks:
                if c.HasField("remote_check_model"):
                    remote_cid = remote_cid_vec[c.remote_check_model]
                    if remote_cid is not None:
                        cm = program.check_models[remote_cid]
                        if cm.gid not in gid_set:
                            external_cids.add(remote_cid)

    # Collect all cids to process: merge-set + external
    all_cids: list[int] = []
    for cid in _cids_in_instantiation_order(program):
        cm = program.check_models[cid]
        if cm.gid in gid_set or cid in external_cids or cm.gid in output_side_gids:
            all_cids.append(cid)

    for cid in all_cids:
        check_model = program.check_models[cid]
        check_model_type = program.modified_check_model_types[cid]
        remote_gadget_gid_vec = program.remote_gadget_gid_vecs[cid]

        for check_index, check in enumerate(check_model_type.checks):
            local_ci = CheckIndex(cid=cid, check_index=check_index)

            refs: list[MergedMeasurementRef] = []
            output_virtual_key: tuple[int, int] | None = None

            for m in check.measurements:
                remote_gid: int = check_model.gid
                remote_mi = m.measurement_index
                if m.HasField("remote_gadget"):
                    remote = check_model_type.remote_gadgets[m.remote_gadget]
                    remote_gid = remote_gadget_gid_vec[m.remote_gadget]
                    assert isinstance(remote_gid, int)
                    remote_mi = m.measurement_index + remote.measurement_bias

                ref, ov_key = _resolve_measurement_ref(remote_gid, remote_mi)
                if ref is not None:
                    refs.append(ref)
                else:
                    assert ov_key is not None
                    # Output-virtual: this check becomes an output-boundary
                    # check.  There should be at most one OV measurement per
                    # check.
                    output_virtual_key = ov_key

            mc = MergedCheck(
                measurements=refs,
                naturally_flipped=check.naturally_flipped,
            )
            if output_virtual_key is not None:
                # Unfinished check — keyed by the OV measurement for
                # later lookup; encoded in the check_map with a negative
                # index so step 8's error dispatch can distinguish it
                # from finished checks.
                idx = len(unfinished_checks)
                unfinished_by_key[output_virtual_key] = idx
                unfinished_checks.append(mc)
                global_ci = CheckIndex(cid=1, check_index=-(idx + 1))
                check_map.add(local_ci, global_ci, unique=False)
            else:
                idx = len(finished_checks)
                finished_checks.append(mc)
                global_ci = CheckIndex(cid=1, check_index=idx)
                check_map.add(local_ci, global_ci, unique=False)

    # ── 8. Build errors ──────────────────────────────────────────────
    error_map: Bijection[ErrorIndex] = Bijection()
    merged_errors: list[MergedError] = []

    for eid in _eids_in_instantiation_order(program):
        error_model = program.error_models[eid]
        error_model_type = program.modified_error_model_types[eid]
        remote_cid_vec = program.remote_check_model_cid_vecs[eid]
        check_model = program.check_models[error_model.cid]
        gid = check_model.gid
        if gid not in gid_set:
            continue
        matrices = propagator.expanded_matrices[gid]

        for error_index, error in enumerate(error_model_type.errors):
            local_ei = ErrorIndex(eid=eid, error_index=error_index)

            fr: list[int] = []
            ur: list[int] = []
            for c in error.checks:
                err_remote_cid: int = error_model.cid
                err_remote_check_index = c.check_index
                if c.HasField("remote_check_model"):
                    remote = error_model_type.remote_check_models[c.remote_check_model]
                    err_remote_cid = remote_cid_vec[c.remote_check_model]
                    assert isinstance(err_remote_cid, int)
                    err_remote_check_index = c.check_index + remote.check_bias
                local_ci = CheckIndex(
                    cid=err_remote_cid, check_index=err_remote_check_index
                )
                global_ci = check_map.atob[local_ci]
                if global_ci.check_index >= 0:
                    fr.append(global_ci.check_index)
                else:
                    ur.append(-(global_ci.check_index + 1))

            residual: set[int] = set()
            readout_flips: set[int] = set()
            for residual_index in error.residual:
                out_local = matrices.output_observables[residual_index].to_global(gid)
                residual ^= propagator.out_to_residual.get(out_local, set())
                readout_flips ^= propagator.out_to_readout.get(out_local, set())
            for readout_index in error.readout_flips:
                err_local_ri = ReadoutIndex(gid=gid, readout_index=readout_index)
                if err_local_ri in readout_map.atob:
                    err_global_ri = readout_map.atob[err_local_ri]
                    readout_flips ^= {err_global_ri.readout_index}

            if not fr and not ur and not residual and not readout_flips:
                continue

            global_ei = ErrorIndex(eid=1, error_index=len(error_map))
            error_map.add(local_ei, global_ei)
            merged_errors.append(
                MergedError(
                    probability=error.probability,
                    residual=sorted(residual),
                    readout_flips=sorted(readout_flips),
                    finished_checks=sorted(fr),
                    unfinished_checks=sorted(ur),
                )
            )

    # ── 9. Absorb logical_correction into cp/pc and per-error residual ──
    # The runtime evaluates
    #   readouts[c] = raw[c] ^ decoded.readouts[c] ^ (rp · input)[c]
    #   residual  ^= lc · readouts
    # which factors as
    #   residual += (lc · rp_input) · input
    #            +  (lc · R)        · measurements    # R = readout's measurement_indices
    #            +  lc · decoded.readouts             # decoder-internal channel
    # The first two terms are purely static and identical to populating
    # cp / pc directly.  The third term is routed through per-error
    # residual updates (each error's readout_flips contribution to residual
    # is folded into error.residual).  After this pass the merged
    # logical_correction is always empty by design, eliminating the
    # redundancy between (lc × rp, lc × measurement_indices) and (cp, pc).
    #
    # A conditional correction may instead flip an output observable that
    # flows into a downstream *readout* (e.g. a lattice-surgery ``CONDITIONAL``
    # followed by a ``MeasureZ``).  ``cc_readout_set`` carries those cases: the
    # conditioning readout's ``rp_input`` / ``rp_affine`` dependence is folded
    # into the downstream readout's ``readout_propagation`` row, and its
    # decoded-readout channel into per-error ``readout_flips``.  (The readout's
    # measurement dependence is already carried by ``obs_meas_deps`` when the
    # merged readout's ``measurement_indices`` were built in step 6.)
    if cc_set or cc_readout_set:
        affine_col = num_input_obs

        # readout_index -> {input observable cols}; the affine column is
        # split off so we can apply it at most once per (out_row, readout).
        rp_input_by_readout: dict[int, set[int]] = {}
        rp_affine_readouts: set[int] = set()
        for readout_index, in_col in rp_set:
            if in_col == affine_col:
                rp_affine_readouts.add(readout_index)
            else:
                rp_input_by_readout.setdefault(readout_index, set()).add(in_col)

        # readout_index -> {measurement indices} that the runtime XORs into
        # the readout bit (R in the docstring above).
        readout_to_meas: list[set[int]] = [
            set(readout.measurement_indices) for readout in merged_readouts
        ]

        # Built lazily inside the absorption loop, consumed afterwards by
        # the per-error residual fold-in.
        rows_by_readout: dict[int, set[int]] = {}
        for out_row, readout_index in cc_set:
            rows_by_readout.setdefault(readout_index, set()).add(out_row)
            # (a) absorb lc · rp_input into cp.
            cp_set.symmetric_difference_update(
                (out_row, in_col)
                for in_col in rp_input_by_readout.get(readout_index, ())
            )
            # (a') absorb lc · rp_affine into cp's affine column.
            if readout_index in rp_affine_readouts:
                cp_set.symmetric_difference_update([(out_row, affine_col)])
            # (b) absorb lc · R into pc.
            pc_set.symmetric_difference_update(
                (out_row, meas) for meas in readout_to_meas[readout_index]
            )

        # Same absorption for corrections that flow into downstream readouts,
        # folding into rp instead of cp.  Consumed afterwards by the per-error
        # readout_flips fold-in.
        readout_rows_by_readout: dict[int, set[int]] = {}
        for out_readout, readout_index in cc_readout_set:
            readout_rows_by_readout.setdefault(readout_index, set()).add(out_readout)
            # (a) absorb lc · rp_input into rp.
            rp_set.symmetric_difference_update(
                (out_readout, in_col)
                for in_col in rp_input_by_readout.get(readout_index, ())
            )
            # (a') absorb lc · rp_affine into rp's affine column.
            if readout_index in rp_affine_readouts:
                rp_set.symmetric_difference_update([(out_readout, affine_col)])

        # (c) absorb lc · decoded.readouts via per-error residual / readout_flips.
        for err in merged_errors:
            if not err.readout_flips:
                continue
            residual = set(err.residual)
            readout_flips = set(err.readout_flips)
            for readout_index in err.readout_flips:
                residual.symmetric_difference_update(
                    rows_by_readout.get(readout_index, ())
                )
                readout_flips.symmetric_difference_update(
                    readout_rows_by_readout.get(readout_index, ())
                )
            err.residual = sorted(residual)
            err.readout_flips = sorted(readout_flips)

        cc_set.clear()

    # Build final BitMatrices now that cc_set / cc_readout_set are absorbed.
    correction_propagation = bitmatrix_from_sparse(
        cp_set, rows=num_output_obs, cols=num_input_obs + 1
    )
    readout_propagation = bitmatrix_from_sparse(
        rp_set, rows=num_readouts, cols=num_input_obs + 1
    )
    logical_correction = bitmatrix_from_sparse(
        cc_set, rows=num_output_obs, cols=num_readouts
    )
    physical_correction = bitmatrix_from_sparse(
        pc_set, rows=num_output_obs, cols=num_measurements
    )

    return MergedGadget(
        input_ptypes=[ip.ptype for ip in input_ports],
        output_ptypes=[op.ptype for op in output_ports],
        measurements=merged_measurements,
        readouts=merged_readouts,
        correction_propagation=correction_propagation,
        readout_propagation=readout_propagation,
        logical_correction=logical_correction,
        physical_correction=physical_correction,
        finished_checks=finished_checks,
        unfinished_checks=unfinished_checks,
        errors=merged_errors,
        measurement_map=measurement_map,
        observable_map=observable_map,
        readout_map=readout_map,
        check_map=check_map,
        error_map=error_map,
    )


def _classify_merge_ports(
    program: ExpandedProgram,
    merge_gids: frozenset[int],
) -> tuple[list[_MergeInputPort], list[_MergeOutputPort]]:
    """Classify merge-boundary input and output ports.

    Iterates gadgets in instantiation order.  An input port whose connector
    references a non-merge gadget becomes a merge input port.  An output port
    that is unconnected or connects to a non-merge gadget becomes a merge
    output port.

    Output ports that connect to a non-merge consumer are ordered by the
    consumer's ``(gid, input_port_index)`` pair (consumer gids ordered by
    instantiation order); unconnected outputs follow in producer
    ``(gid, port_index)`` order.  This makes the merged gadget's output
    port ordering match the *consumer's* expected input port layout, so
    callers that build a synthetic consumer (e.g. an ``__output_mock__``
    aggregating compose-level OUTPUT declarations) get back outputs in
    the order they wired the mock's inputs — even when intermediate
    gadgets (such as ``CONDITIONAL`` synthesised identity hosts) push
    some producers to higher gids than others.
    """
    input_ports: list[_MergeInputPort] = []

    # Pre-compute instantiation order for non-merge gids so we can sort
    # connected outputs by their consumer's position in the program.
    gid_order = {
        gid: idx for idx, gid in enumerate(_gids_in_instantiation_order(program))
    }

    connected_outputs: list[tuple[int, int, _MergeOutputPort]] = []
    unconnected_outputs: list[_MergeOutputPort] = []

    for gid in _gids_in_instantiation_order(program):
        if gid not in merge_gids:
            continue
        gadget = program.gadgets[gid]
        gadget_type = program.gadget_types[gadget.gtype]

        for port_idx, (connector, port_spec) in enumerate(
            zip(gadget.connectors, gadget_type.inputs)
        ):
            if connector.gid not in merge_gids:
                input_ports.append(
                    _MergeInputPort(
                        merge_gid=gid,
                        port_index=port_idx,
                        ptype=port_spec.ptype,
                        peer_gid=connector.gid,
                        peer_port=connector.port,
                    )
                )

        for port_idx, port_spec in enumerate(gadget_type.outputs):
            out_instance = OutputPortIndex(gid=gid, port_index=port_idx)
            merge_out = _MergeOutputPort(
                merge_gid=gid,
                port_index=port_idx,
                ptype=port_spec.ptype,
            )
            if out_instance not in program.peer_input:
                unconnected_outputs.append(merge_out)
                continue
            peer = program.peer_input[out_instance]
            if peer.gid in merge_gids:
                continue
            connected_outputs.append(
                (gid_order.get(peer.gid, len(gid_order)), peer.port_index, merge_out)
            )

    connected_outputs.sort(key=lambda triple: (triple[0], triple[1]))
    output_ports: list[_MergeOutputPort] = [
        merge_out for _, _, merge_out in connected_outputs
    ]
    output_ports.extend(unconnected_outputs)

    return input_ports, output_ports


@dataclass
class _MergePropagator:
    """Observable propagator for merge().

    Identical in structure to ``_ObservablePropagator`` but parameterized
    by a merge-set.  Output ports connecting to non-merge gadgets or
    unconnected are treated as merge outputs (identity mapping).  Internal
    connections are recursively expanded.
    """

    expanded_matrices: dict[int, ExpandedGadgetMatrices] = field(default_factory=dict)
    out_to_residual: dict[ObservableIndex, set[int]] = field(default_factory=dict)
    out_to_readout: dict[ObservableIndex, set[int]] = field(default_factory=dict)
    expanded: set[int] = field(default_factory=set)

    @staticmethod
    def build(
        program: ExpandedProgram,
        merge_gids: frozenset[int],
        observable_map: "Bijection[ObservableIndex]",
        readout_map: "Bijection[ReadoutIndex]",
    ) -> "_MergePropagator":
        prop = _MergePropagator()

        # Initialize matrices for all gadgets in the program (needed for
        # chasing through connections)
        for gid in program.gadgets:
            gadget_type = program.modified_gadget_types[gid]
            prop.expanded_matrices[gid] = ExpandedGadgetMatrices.from_gadget_type(
                gadget_type, program.port_types
            )

        # Expand all merge-set gadgets
        for gid in merge_gids:
            prop._expand(gid, merge_gids, program, observable_map, readout_map)

        return prop

    def _expand(
        self,
        gid: int,
        merge_gids: frozenset[int],
        program: ExpandedProgram,
        observable_map: "Bijection[ObservableIndex]",
        readout_map: "Bijection[ReadoutIndex]",
    ) -> None:
        if gid in self.expanded:
            return
        self.expanded.add(gid)

        gadget_type = program.modified_gadget_types[gid]
        for output_index, output in enumerate(gadget_type.outputs):
            out_port = OutputPortIndex(gid=gid, port_index=output_index)
            port_type = program.port_types[output.ptype]

            if (
                out_port in program.peer_input
                and program.peer_input[out_port].gid in merge_gids
            ):
                # Internal connection: expand the next merge-set gadget
                input_port = program.peer_input[out_port]
                next_gid = input_port.gid
                self._expand(next_gid, merge_gids, program, observable_map, readout_map)
                next_matrices = self.expanded_matrices[next_gid]

                for obs_idx in range(len(port_type.observables)):
                    next_input = LocalObservableIndex(
                        port=input_port.port_index, observable_index=obs_idx
                    )

                    residual_set: set[int] = set()
                    readout_set: set[int] = set()
                    if next_input in next_matrices.correction_propagation:
                        for next_out in next_matrices.correction_propagation[
                            next_input
                        ]:
                            global_next = next_out.to_global(next_gid)
                            residual_set ^= self.out_to_residual.get(global_next, set())
                            readout_set ^= self.out_to_readout.get(global_next, set())

                    if next_input in next_matrices.readout_propagation:
                        for readout in next_matrices.readout_propagation[next_input]:
                            local_ri = ReadoutIndex(gid=next_gid, readout_index=readout)
                            if local_ri in readout_map.atob:
                                readout_set ^= {
                                    readout_map.atob[local_ri].readout_index
                                }

                    local_obs = ObservableIndex(
                        gid=gid, port=output_index, observable_index=obs_idx
                    )
                    self.out_to_residual[local_obs] = residual_set
                    self.out_to_readout[local_obs] = readout_set

            else:
                # Merge boundary (unconnected or connects to non-merge):
                # identity mapping to output observables
                for obs_idx in range(len(port_type.observables)):
                    local_obs = ObservableIndex(
                        gid=gid, port=output_index, observable_index=obs_idx
                    )
                    global_obs = observable_map.atob[local_obs]
                    self.out_to_residual[local_obs] = {global_obs.observable_index}
                    self.out_to_readout[local_obs] = set()


def canonical_program() -> list[pb.Instruction]:
    return [
        pb.Instruction(gadget=pb.Gadget(gtype=1, gid=1)),
        pb.Instruction(check_model=pb.CheckModel(ctype=1, gid=1, cid=1)),
        pb.Instruction(error_model=pb.ErrorModel(etype=1, cid=1, eid=1)),
    ]
