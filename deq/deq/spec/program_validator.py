"""
=====================
Program Specification
=====================

1. for every :code:`instruction: Instruction` in :code:`program: Program`:

    1. (ProgSpec 1.1) if :code:`instruction` is a :code:`Gadget`, :code:`gadget.gtype` must be \
        defined in :code:`library.gadget_types`. Each :code:`gadget` is uniquely assigned a \
        :code:`gid` (>= 1). \
        Note that the :code:`gid` is sequentially assigned when running the reference program \
        within the :code:`Library`, but one should not expect it to be sequential when calling \
        the :code:`Execute` RPC interface at runtime (to allow efficient distributed decoding \
        without unnecessary global synchronization), so the user (e.g., logical controller) \
        should not assume the value of :code:`gid` until it's returned by the :code:`Execute` \
        Interface. The same applies below.
    2. (ProgSpec 1.2) if :code:`instruction` is a :code:`CheckModel`, :code:`check_model.ctype`\
        must be defined in :code:`library.check_model_types`
    3. (ProgSpec 1.3) if :code:`instruction` is an :code:`ErrorModel`, :code:`error_model.etype`\
        must be defined in :code:`library.error_model_types`
    4. (ProgSpec 1.4) Other unsupported types of :code:`instruction` are not allowed

2. for every :code:`gadget: Gadget` in :code:`program: Program`:

    1. (ProgSpec 2.1) the number of :code:`gadget.connectors` must match the number of \
        :code:`gadget_type.inputs` where :code:`gadget_type` is the type of the gadget
    2. for every :code:`(connector, port_type)` in :code:`gadget.connectors` and \
        :code:`gadget_type.inputs`:

        1. (ProgSpec 2.2.1) :code:`connector.gid` must be instantiated before the current gadget
        2. (ProgSpec 2.2.2) :code:`connector.port` must be a valid output port of the peer \
            gadget instance referenced by :code:`connector.gid`
        3. (ProgSpec 2.2.3) the port type :code:`ptype` of the current input port and the peer \
            output port must be the same (this is to ensure the Pauli frame propagates correctly)
        4. (ProgSpec 2.2.4) each output port of every gadget instance can be connected only once \
            (one cannot physically pass the same logical qubit to two gadgets)

    3. if :code:`gadget.modifier` is present, for each matrix modifier \
        (:code:`correction_propagation_mod`, :code:`readout_propagation_mod`, \
        :code:`logical_correction_mod`, :code:`physical_correction_mod`):

        1. (ProgSpec 2.3.1) if :code:`toggle` is specified, its dimensions must match the \
            original matrix dimensions in the gadget type
        2. (ProgSpec 2.3.2) if :code:`overwrite` is specified, its dimensions must match the \
            original matrix dimensions in the gadget type

    4. if :code:`gadget.modifier.remote_conditional_correction` is present:

        1. (ProgSpec 2.4.1) the :code:`correction` matrix must have dimensions \
            :code:`|output_observables|` rows x :code:`|remote_readouts|` columns
        2. (ProgSpec 2.4.2) for each :code:`remote_readout` in :code:`remote_readouts`, \
            the :code:`gid` must reference a gadget that was instantiated at or before the \
            current one (i.e., the current gadget can reference its own readouts)
        3. (ProgSpec 2.4.3) for each :code:`remote_readout`, the :code:`readout_index` must be \
            a valid readout index in the referenced gadget type

3. for every :code:`check_model: CheckModel` in :code:`program: Program`:

    1. (ProgSpec 3.1) if :code:`instruction` is a :code:`CheckModel`, :code:`check_model.gid` \
        must be instantiated before the current instruction
    2. (ProgSpec 3.2) if :code:`instruction` is a :code:`CheckModel` and \
        :code:`check_model_type.gtype` is not zero (i.e., not a wildcard), \
        :code:`check_model_type.gtype` must match the :code:`gtype` \
        of the gadget instance referenced by :code:`check_model.gid`
    3. if :code:`check_model.modifier` is present, for every :code:`reroute`:

        1. (ProgSpec 3.3.1) the same :code:`reroute.remote_gadget_index` must appear no more than \
            once in the modifier (although implementation will likely overwrite the existing one,\
            this is likely a user error)
        2. (ProgSpec 3.3.2) if :code:`reroute.remote_gadget_index` is out of range, we allow \
            dynamic extension of the remote gadgets but the index must be less than 65536

    4. (ProgSpec 3.4) the modified :code:`check_model_type` must satisfy the library specifications\
        (LibSpec 3.*), except that we allow placeholders in the rerouted remote gadgets
    5. for every remote gadget :code:`remote` that is not a placeholder, expand it as follows:

        1. (ProgSpec 3.5.1) if :code:`remote.previous_remote_gadget` is present, the corresponding \
            remote gadget must be expandable into a concrete gadget with :code:`previous_gid`; \
            if not presented, the :code:`previous_gid` is set to :code:`check_model.gid`
        2. (ProgSpec 3.5.2) since we have the concrete :code:`previous_gtype` from the \
            :code:`previous_gid`, the :code:`remote.input` or :code:`remote.output`, whichever is \
            present, must be a valid port index. (LibSpec 3.3.4 also checks this if \
            :code:`remote.expecting_gtype` is specified)
        3. (ProgSpec 3.5.3) when :code:`remote.output` is present, it must be connected. \
            Note that we don't have to check the inputs because one cannot instantiate a gadget \
            without connecting all its inputs (ProgSpec 2.1)
        4. (ProgSpec 3.5.4) similar to (LibSpec 3.3.3), when :code:`remote.expecting_gtype`\
            is present, we check if the remote gadget is indeed of the expected :code:`gtype`
        5. (ProgSpec 3.5.5) given the :code:`previous_gtype`, the :code:`remote.measurement_bias` \
            must be strictly smaller than the number of measurements in the previous gadget

    6. for every :code:`check` in :code:`check_model_type.checks` and every :code:`measurement` in \
        :code:`check.measurements`:

        1. (ProgSpec 3.6.1) if :code:`measurement` is a remote measurement, i.e., \
            :code:`measurement.remote_gadget` is specified, it must not refer to a placeholder
        2. (ProgSpec 3.6.2) :code:`measurement.measurement_index` must be a valid index in the \
            expanded remote gadget
        3. (ProgSpec 3.6.3) the measurements, in terms of the tuple \
            :code:`(remote_gid, absolute_measurement_index)`, must not be duplicate


4. for every :code:`error_model: ErrorModel` in :code:`program: Program`:

    1. (ProgSpec 4.1) if :code:`instruction` is an :code:`ErrorModel`, :code:`error_model.cid` \
        must be instantiated before the current instruction
    2. (ProgSpec 4.2) if :code:`instruction` is an :code:`ErrorModel` and \
        :code:`error_model_type.ctype` is not zero (i.e., not a wildcard), \
        :code:`error_model_type.ctype` must match the :code:`ctype` of the check model \
        instance referenced by :code:`error_model.cid`
    3. if :code:`error_model.modifier` is present, 

        1. (ProgSpec 4.3.1) the length of :code:`modifier.probabilities` must be either 0 or \
            equal to the number of errors in the error model, and the values must be in [0, 1]
        2. (ProgSpec 4.3.2) the length of :code:`modifier.sparse_indices` must be equal to \
            the length of :code:`modifier.sparse_probabilities` (they should be pair-wise)
        3. (ProgSpec 4.3.3) the :code:`modifier.sparse_indices` must not contain duplicate indices\
            , all indices must be valid (i.e., less than the number of errors in the error model)\
            and all probabilities must be in [0, 1]
        4. (ProgSpec 4.3.4) for every :code:`reroute`, the same \
            :code:`reroute.remote_check_model_index` must appear no more than \
            once in the modifier (although implementation will likely overwrite the existing one)
        5. (ProgSpec 4.3.5) for every :code:`reroute`, if \
            :code:`reroute.remote_check_model_index` is out of range, we allow \
            dynamic extension of the remote check models but the index must be less than 65536

    4. (ProgSpec 4.4) the modified :code:`error_model_type` must satisfy the library specifications\
        (LibSpec 4.*), except that we allow placeholders in the rerouted remote check models
    5. for every remote check model :code:`remote` that is not a placeholder, expand it as follows:

        1. (ProgSpec 4.5.1) if :code:`remote.previous_remote_check_model` is present, the \
            corresponding remote check model must be expandable into a concrete check model with \
            :code:`previous_cid`; if not presented, the :code:`previous_cid` is set to \
            :code:`error_model.cid`. We also know the binding gadget of :code:`previous_cid`, which\
            is denoted as :code:`previous_gid`. Similarly, we have :code:`previous_ctype` and \
            :code:`previous_gtype`
        2. (ProgSpec 4.5.2) since we have the concrete :code:`previous_gtype` from the \
            :code:`previous_cid`, the :code:`remote.input` or :code:`remote.output`, whichever is \
            present, must be a valid port index. (LibSpec 4.3.4 also checks this if \
            :code:`remote.expecting_ctype` is specified)
        3. (ProgSpec 4.5.3) When :code:`remote.output` is present, it must be connected. \
            Note that we don't have to check the inputs because one cannot instantiate a gadget \
            without connecting all its inputs (ProgSpec 2.1)
        4. (ProgSpec 4.5.4) The remote gadget must be binding to a check model
        5. (ProgSpec 4.5.5) similar to (LibSpec 4.3.3), when :code:`remote.expecting_ctype`\
            is present, we check if the remote check model is indeed of the expected :code:`ctype`
        6. (ProgSpec 4.5.6) given the :code:`previous_ctype`, the :code:`remote.check_bias` \
            must be strictly smaller than the number of checks in the previous check model
            
    6. for every :code:`error` in :code:`error_model_type.errors` and every :code:`check` in \
        :code:`error.checks`:

        1. (ProgSpec 4.6.1) if :code:`check` is a remote check, i.e., \
            :code:`check.remote_check_model` is specified, it must not refer to a placeholder
        2. (ProgSpec 4.6.2) :code:`check.check_index` must be a valid index in the \
            expanded remote check model
        3. (ProgSpec 4.6.3) the checks, in terms of the tuple \
            :code:`(remote_cid, absolute_check_index)`, must not be duplicate

"""

from dataclasses import dataclass, field
from typing import Sequence
from functools import cached_property
import deq.proto.deq_bin_pb2 as pb
import deq.proto.util_pb2 as util_pb
from deq.spec.violations import Violations

WILDCARD = 0
from deq.spec.library_validator import (
    is_library_valid,
    _check_model_validity_single,
    _is_placeholder_remote_gadget,
    _error_model_validity_single,
    _is_placeholder_remote_check_model,
)
from deq.spec.common import (
    InputPortIndex,
    OutputPortIndex,
    MeasurementIndex,
    CheckIndex,
    ErrorIndex,
    ReadoutIndex,
    NoViolations,
    apply_bitmatrix_modifier,
)

# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint


def is_valid(lib: pb.Library) -> "Violations | ExpandedProgram":
    validity = is_library_valid(lib)
    if not validity:
        return validity
    return is_program_valid(lib)


# pylint: disable=too-many-return-statements
def is_program_valid(lib: pb.Library) -> "Violations | ExpandedProgram":
    assert isinstance(lib, pb.Library)

    return ExpandedProgram.from_library_with_check(lib)


def _are_instance_types_all_defined(lib: pb.Library) -> Violations:
    validity = Violations()
    gadget_types = {g.gtype: g for g in lib.gadget_types}
    check_model_types = {c.ctype: c for c in lib.check_model_types}
    error_model_types = {e.etype: e for e in lib.error_model_types}
    for instruction in lib.program:
        instance_type = instruction.WhichOneof("create")
        if instance_type == "gadget":
            gadget = instruction.gadget
            if gadget.gtype not in gadget_types:
                validity += (
                    f"(ProgSpec 1.1) undefined gadget type (gtype={gadget.gtype})"
                )
        elif instance_type == "check_model":
            check_model = instruction.check_model
            if check_model.ctype not in check_model_types:
                validity += f"(ProgSpec 1.2) undefined check model type (ctype={check_model.ctype})"
        elif instance_type == "error_model":
            error_model = instruction.error_model
            if error_model.etype not in error_model_types:
                validity += f"(ProgSpec 1.3) undefined error model type (etype={error_model.etype})"
        else:
            validity += "(ProgSpec 1.4) unsupported instruction type"

    return validity


@dataclass
class ExpandedProgram(NoViolations):
    """
    Expanded program with all modifiers applied to the instances
    """

    #: the original library
    lib: pb.Library
    #: map from :code:`ptype` to (unmodified) :code:`PortType`
    port_types: dict[int, pb.PortType]
    #: map from :code:`gtype` to (unmodified) :code:`GadgetType`
    gadget_types: dict[int, pb.GadgetType]
    #: map from :code:`ctype` to (unmodified) :code:`CheckModelType`
    check_model_types: dict[int, pb.CheckModelType]
    #: map from :code:`etype` to (unmodified) :code:`ErrorModelType`
    error_model_types: dict[int, pb.ErrorModelType]

    #: =====================
    #: Expanded Gadgets
    #: =====================
    #: map from :code:`gid` to :code:`Gadget`
    gadgets: dict[int, pb.Gadget] = field(default_factory=dict)
    #: map from :code:`gid` to instantiation order (0-indexed)
    #: This is needed because user-specified gids may not reflect instantiation order
    gadget_instantiation_order: dict[int, int] = field(default_factory=dict)
    #: map from an output port to its peer input port (if exists)
    peer_input: dict[OutputPortIndex, InputPortIndex] = field(default_factory=dict)
    #: map from an input port to its peer output port (if exists)
    peer_output: dict[InputPortIndex, OutputPortIndex] = field(default_factory=dict)
    #: the check indices associated with the list of measurements
    associated_checks: dict[int, list[set[CheckIndex]]] = field(default_factory=dict)
    #: map from :code:`gid` to the modified :code:`GadgetType` (after applying the modifier)
    modified_gadget_types: dict[int, pb.GadgetType] = field(default_factory=dict)

    #: =====================
    #: Expanded Check Models
    #: =====================
    #: map from :code:`cid` to :code:`CheckModel`
    check_models: dict[int, pb.CheckModel] = field(default_factory=dict)
    #: map from :code:`cid` to instantiation order (0-indexed)
    check_model_instantiation_order: dict[int, int] = field(default_factory=dict)
    #: map from :code:`cid` to the list of remote gadgets' :code:`gid`
    remote_gadget_gid_vecs: dict[int, list[int | None]] = field(default_factory=dict)
    #: map from :code:`cid` to the list of expanded checks
    expanded_checks: dict[int, list[frozenset[MeasurementIndex]]] = field(
        default_factory=dict
    )
    #: map from :code:`cid` to the list of associated errors that would flip the check
    associated_errors: dict[int, list[set[ErrorIndex]]] = field(default_factory=dict)
    #: map from :code:`gid` to the binding :code:`cid`
    binding: dict[int, int] = field(default_factory=dict)
    #: map from :code:`cid` to the modified :code:`CheckModelType` (after applying the modifier)
    modified_check_model_types: dict[int, pb.CheckModelType] = field(
        default_factory=dict
    )

    #: =====================
    #: Expanded Error Models
    #: =====================
    #: map from :code:`eid` to :code:`ErrorModel`
    error_models: dict[int, pb.ErrorModel] = field(default_factory=dict)
    #: map from :code:`eid` to instantiation order (0-indexed)
    error_model_instantiation_order: dict[int, int] = field(default_factory=dict)
    #: from eid to the list of remote check models' cid
    remote_check_model_cid_vecs: dict[int, list[int | None]] = field(
        default_factory=dict
    )
    #: map from :code:`eid` to the list of expanded errors
    expanded_errors: dict[int, list[frozenset[CheckIndex]]] = field(
        default_factory=dict
    )
    #: map from :code:`eid` to the modified :code:`ErrorModelType` (after applying the modifier)
    modified_error_model_types: dict[int, pb.ErrorModelType] = field(
        default_factory=dict
    )

    #: =====================
    #: Expanded Remote Conditional Corrections
    #: =====================
    #: map from :code:`gid` to the expanded remote conditional correction
    #: each entry maps (remote_gid, readout_index) -> column_index in the correction matrix
    expanded_remote_conditional_corrections: dict[
        int, tuple[list[ReadoutIndex], pb.RemoteConditionalCorrection]
    ] = field(default_factory=dict)

    @staticmethod
    def from_library_with_check(lib: pb.Library) -> "ExpandedProgram | Violations":

        # ProgSpec 1.*
        validity = _are_instance_types_all_defined(lib)
        if not validity:
            return validity

        program = ExpandedProgram(
            lib=lib,
            port_types={p.ptype: p for p in lib.port_types},
            gadget_types={g.gtype: g for g in lib.gadget_types},
            check_model_types={c.ctype: c for c in lib.check_model_types},
            error_model_types={e.etype: e for e in lib.error_model_types},
        )

        # checking specs that are order sensitive (e.g. must refer to previously defined instances)
        validity = ExpandedProgram._collect_all_instances(program)
        if not validity:
            return validity

        # ProgSpec 2.*
        validity = ExpandedProgram._collect_gadgets(program)
        if not validity:
            return validity

        validity = ExpandedProgram._collect_check_models(program)
        if not validity:
            return validity

        validity = ExpandedProgram._collect_error_models(program)
        if not validity:
            return validity

        return program

    @staticmethod
    def from_library(lib: pb.Library) -> "ExpandedProgram":
        """
        return a completely expanded program, note that it might panic for invalid programs.
        (use :code:`deq.spec.program_validator.is_valid` to check when in doubt)
        """
        program = ExpandedProgram.from_library_with_check(lib)
        assert not isinstance(program, Violations), "program does not meet ProgSpec"
        return program

    def _collect_all_instances(self) -> Violations:
        validity = Violations()
        # first collect all instances
        for instruction in self.lib.program:
            instance_type = instruction.WhichOneof("create")
            if instance_type == "gadget":
                gid = len(self.gadgets) + 1
                # check if all connectors refer to existing gadgets
                gadget = instruction.gadget
                gadget_type = self.gadget_types[gadget.gtype]
                if len(gadget.connectors) != len(gadget_type.inputs):
                    validity += (
                        "(ProgSpec 2.1) number of connectors is wrong: expecting "
                        + f"{len(gadget_type.inputs)} from gadget instance gid={gid} with "
                        + f"gtype={gadget.gtype} but giving {len(gadget.connectors)} connectors"
                    )
                for port, connector in enumerate(gadget.connectors):
                    peer_gid = connector.gid
                    # gadgets must be instantiated using existing references only
                    if peer_gid not in self.gadgets:
                        validity += (
                            f"(ProgSpec 2.2.1) gadget instance {gid} input port {port} "
                            + f"connects to undefined gadget {peer_gid}"
                        )
                # add the gadget with its instantiation order
                self.gadget_instantiation_order[gid] = len(self.gadgets)
                self.gadgets[gid] = instruction.gadget
            elif instance_type == "check_model":
                cid = len(self.check_models) + 1
                if instruction.check_model.gid not in self.gadgets:
                    validity += (
                        f"(ProgSpec 3.1) check model instance cid={cid} binds to undefined gadget "
                        + f"{instruction.check_model.gid}"
                    )
                    continue
                self.binding[instruction.check_model.gid] = cid
                self.check_model_instantiation_order[cid] = len(self.check_models)
                self.check_models[cid] = instruction.check_model
                # check binding to the correct gadget type
                check_model = instruction.check_model
                check_model_type = self.check_model_types[check_model.ctype]
                if check_model_type.gtype != 0:
                    if check_model_type.gtype != self.gadgets[check_model.gid].gtype:
                        validity += (
                            f"(ProgSpec 3.2) check model (ctype={check_model.ctype}) expects gadget"
                            + f" type {check_model_type.gtype}, but binds to gadget instance "
                            + f"gid={check_model.gid} with gtype="
                            + f"{self.gadgets[check_model.gid].gtype}"
                        )
            elif instance_type == "error_model":
                eid = len(self.error_models) + 1
                if instruction.error_model.cid not in self.check_models:
                    validity += (
                        f"(ProgSpec 4.1) error model instance eid={eid} attaches to "
                        + f"undefined check model {instruction.error_model.cid}"
                    )
                    continue
                self.error_model_instantiation_order[eid] = len(self.error_models)
                self.error_models[eid] = instruction.error_model
                # check attaching to the correct check model type
                error_model = instruction.error_model
                error_model_type = self.error_model_types[error_model.etype]
                if error_model_type.ctype != 0:
                    if (
                        error_model_type.ctype
                        != self.check_models[error_model.cid].ctype
                    ):
                        validity += (
                            f"(ProgSpec 4.2) error model (etype={error_model.etype}) expects check "
                            + f"model type {error_model_type.ctype}, but attaches to check model "
                            + f"instance cid={error_model.cid} with ctype="
                            f"{self.check_models[error_model.cid].ctype}"
                        )
        return validity

    def _collect_gadgets(self) -> Violations:
        validity = Violations()
        for gid in sorted(self.gadgets.keys()):
            gadget = self.gadgets[gid]
            gadget_type = self.gadget_types[gadget.gtype]
            self.associated_checks[gid] = [
                set() for _ in range(len(gadget_type.measurements))
            ]

            # apply the gadget modifier to get the modified gadget type
            modified_gadget_type, modify_validity = self._modify_gadget_type(gid)
            if not modify_validity:
                validity += modify_validity
                continue
            self.modified_gadget_types[gid] = modified_gadget_type

            for port, (connector, port_type) in enumerate(
                zip(gadget.connectors, gadget_type.inputs)
            ):
                peer_gid = connector.gid
                peer_port = connector.port
                peer = self.gadgets[peer_gid]
                # peer.gtype must be defined because we have validated the library part
                peer_type = self.gadget_types[peer.gtype]
                if peer_port >= len(peer_type.outputs):
                    validity += (
                        f"(ProgSpec 2.2.2) gadget instance gid={gid} input port {port} connects to "
                        + f"gadget gid={peer_gid} with overflowed output port {peer_port}"
                    )
                    continue
                if port_type.ptype != peer_type.outputs[peer_port].ptype:
                    validity += (
                        f"(ProgSpec 2.2.3) gadget instance gid={gid} input {port} "
                        + "is incompatible with the output port"
                        + f" {peer_port} from peer gadget gid={peer_gid}"
                    )
                    continue
                # check port not already connected
                output_instance = OutputPortIndex(gid=peer_gid, port_index=peer_port)
                input_instance = InputPortIndex(gid=gid, port_index=port)
                if output_instance in self.peer_input:
                    existing = self.peer_input[output_instance]
                    validity += (
                        f"(ProgSpec 2.2.4) gadget instance gid={gid} input {port} connects to "
                        + f"output port {peer_port} from peer gadget gid={peer_gid}, which is "
                        + f"already connected to input port {existing.port_index} "
                        + f"of gid={existing.gid}"
                    )
                    continue
                self.peer_input[output_instance] = input_instance
                self.peer_output[input_instance] = output_instance

        return validity

    def _collect_check_models(self) -> Violations:
        validity = Violations()

        # check models are not required to instantiate in order
        for cid, check_model in self.check_models.items():
            assert check_model.ctype in self.check_model_types

            # apply the modifier
            check_model_type, modify_validity = self._modify_check_model_type(cid)
            if not modify_validity:
                validity += modify_validity
                continue
            self.modified_check_model_types[cid] = check_model_type

            self.associated_errors[cid] = [
                set() for _ in range(len(check_model_type.checks))
            ]

            # check that the modified check model satisfies the library specifications
            library_validity = _check_model_validity_single(
                check_model_type, self.gadget_types, allow_placeholder=True
            )
            if not library_validity:
                validity += "(ProgSpec 3.4) the modified check model type is invalid"
                validity += library_validity
                # loopy references may block the validator from terminating, skip
                continue

            # also check

            # expand the gid of the remote gadgets
            remote_gadget_gid_vec: list[int | None | Violations] = [None] * len(
                check_model_type.remote_gadgets
            )
            for ri in range(len(check_model_type.remote_gadgets)):
                self._expand_gid(
                    cid,
                    check_model.gid,
                    check_model_type.remote_gadgets,
                    ri,
                    remote_gadget_gid_vec,
                )
            expand_validity: Violations = sum(
                (e for e in remote_gadget_gid_vec if isinstance(e, Violations)),
                start=Violations(),
            )
            if not expand_validity:
                validity += f"(ProgSpec 3.5.*) cannot expand remote gadget {ri}"
                validity += expand_validity
                continue
            self.remote_gadget_gid_vecs[cid] = remote_gadget_gid_vec  # type: ignore

            # validate the checks with concrete references to remote gadgets
            expanded_checks: list[frozenset[MeasurementIndex]] = []
            for ci, check in enumerate(check_model_type.checks):

                measurements: set[MeasurementIndex] = set()

                for m in check.measurements:
                    remote_gid: int = check_model.gid
                    absolute_measurement_index = m.measurement_index
                    if m.HasField("remote_gadget"):
                        remote = check_model_type.remote_gadgets[m.remote_gadget]
                        remote_gid = remote_gadget_gid_vec[m.remote_gadget]
                        if remote_gid is None:
                            validity += "(ProgSpec 3.6.1) remote gadget modified to a placeholder"
                            continue
                        assert isinstance(remote_gid, int), "all checked"
                        absolute_measurement_index = (
                            m.measurement_index + remote.measurement_bias
                        )

                    remote_gadget = self.gadgets[remote_gid]
                    remote_gadget_type = self.gadget_types[remote_gadget.gtype]
                    if absolute_measurement_index >= len(
                        remote_gadget_type.measurements
                    ):
                        validity += (
                            "(ProgSpec 3.6.2) overflowed remote measurement index "
                            + f"{absolute_measurement_index} "
                            + f"in check {ci} of check model (cid={cid})"
                        )

                    key = MeasurementIndex(
                        gid=remote_gid, measurement_index=absolute_measurement_index
                    )
                    if key in measurements:
                        validity += (
                            "(ProgSpec 3.6.3) duplicate measurement in check model "
                            + f"(cid={cid}): remote_gid={remote_gid}, "
                            + f"absolute_measurement_index={absolute_measurement_index}"
                        )
                    measurements.add(key)

                # record the expanded check
                for measurement in measurements:
                    if measurement.measurement_index < len(
                        self.associated_checks[measurement.gid]
                    ):
                        self.associated_checks[measurement.gid][
                            measurement.measurement_index
                        ].add(CheckIndex(cid=cid, check_index=ci))
                expanded_checks.append(frozenset(measurements))

            self.expanded_checks[cid] = expanded_checks

        return validity

    def _modify_gadget_type(self, gid: int) -> tuple[pb.GadgetType, Violations]:
        validity = Violations()

        gadget = self.gadgets[gid]
        origin_gadget_type = self.gadget_types[gadget.gtype]
        gadget_type = pb.GadgetType()
        gadget_type.CopyFrom(origin_gadget_type)

        if gadget.HasField("modifier"):
            modifier = gadget.modifier

            # validate and apply correction_propagation modifier
            if modifier.HasField("correction_propagation_mod"):
                mod = modifier.correction_propagation_mod
                validity += self._validate_bitmatrix_modifier(
                    gid,
                    "correction_propagation",
                    origin_gadget_type.correction_propagation,
                    mod,
                )
                gadget_type.correction_propagation.CopyFrom(
                    apply_bitmatrix_modifier(
                        origin_gadget_type.correction_propagation, mod
                    )
                )

            # validate and apply readout_propagation modifier
            if modifier.HasField("readout_propagation_mod"):
                mod = modifier.readout_propagation_mod
                validity += self._validate_bitmatrix_modifier(
                    gid,
                    "readout_propagation",
                    origin_gadget_type.readout_propagation,
                    mod,
                )
                gadget_type.readout_propagation.CopyFrom(
                    apply_bitmatrix_modifier(
                        origin_gadget_type.readout_propagation, mod
                    )
                )

            # validate and apply logical_correction modifier
            if modifier.HasField("logical_correction_mod"):
                mod = modifier.logical_correction_mod
                validity += self._validate_bitmatrix_modifier(
                    gid,
                    "logical_correction",
                    origin_gadget_type.logical_correction,
                    mod,
                )
                gadget_type.logical_correction.CopyFrom(
                    apply_bitmatrix_modifier(origin_gadget_type.logical_correction, mod)
                )

            # validate and apply physical_correction modifier
            if modifier.HasField("physical_correction_mod"):
                mod = modifier.physical_correction_mod
                validity += self._validate_bitmatrix_modifier(
                    gid,
                    "physical_correction",
                    origin_gadget_type.physical_correction,
                    mod,
                )
                gadget_type.physical_correction.CopyFrom(
                    apply_bitmatrix_modifier(
                        origin_gadget_type.physical_correction, mod
                    )
                )

            # validate remote_conditional_correction
            if modifier.HasField("remote_conditional_correction"):
                validity += self._validate_remote_conditional_correction(
                    gid, gadget_type, modifier.remote_conditional_correction
                )

        return gadget_type, validity

    def _validate_remote_conditional_correction(
        self,
        gid: int,
        gadget_type: pb.GadgetType,
        remote_cc: pb.RemoteConditionalCorrection,
    ) -> Violations:
        validity = Violations()

        num_output_observables = sum(
            len(self.port_types[port.ptype].observables) for port in gadget_type.outputs
        )
        num_remote_readouts = len(remote_cc.remote_readouts)

        if remote_cc.correction.rows != num_output_observables:
            validity += (
                f"(ProgSpec 2.4.1) gadget instance gid={gid} remote_conditional_correction "
                + f"matrix has wrong number of rows: expected {num_output_observables} "
                + f"but got {remote_cc.correction.rows}"
            )
        if remote_cc.correction.cols != num_remote_readouts:
            validity += (
                f"(ProgSpec 2.4.1) gadget instance gid={gid} remote_conditional_correction "
                + f"matrix has wrong number of columns: expected {num_remote_readouts} "
                + f"but got {remote_cc.correction.cols}"
            )

        expanded_readouts: list[ReadoutIndex] = []
        for col_index, remote_readout in enumerate(remote_cc.remote_readouts):
            remote_gid = remote_readout.gid
            readout_index = remote_readout.readout_index

            if remote_gid not in self.gadgets:
                validity += (
                    f"(ProgSpec 2.4.2) gadget instance gid={gid} remote_conditional_correction "
                    + f"references unknown gadget gid={remote_gid} at column {col_index}"
                )
                continue

            current_order = self.gadget_instantiation_order[gid]
            remote_order = self.gadget_instantiation_order[remote_gid]
            if remote_order > current_order:
                validity += (
                    f"(ProgSpec 2.4.2) gadget instance gid={gid} remote_conditional_correction "
                    + f"references gadget gid={remote_gid} which was not instantiated before "
                    + f"or at the current gadget (at column {col_index})"
                )
                continue

            remote_gadget = self.gadgets[remote_gid]
            remote_gadget_type = self.gadget_types[remote_gadget.gtype]
            num_readouts = len(remote_gadget_type.readouts)

            if readout_index >= num_readouts:
                validity += (
                    f"(ProgSpec 2.4.3) gadget instance gid={gid} remote_conditional_correction "
                    + f"references invalid readout_index={readout_index} for gadget gid={remote_gid} "
                    + f"which only has {num_readouts} readouts (at column {col_index})"
                )
            else:
                expanded_readouts.append(
                    ReadoutIndex(gid=remote_gid, readout_index=readout_index)
                )

        if validity:
            self.expanded_remote_conditional_corrections[gid] = (
                expanded_readouts,
                remote_cc,
            )

        return validity

    def _validate_bitmatrix_modifier(
        self,
        gid: int,
        matrix_name: str,
        original: "util_pb.BitMatrix",
        modifier: "pb.BitMatrixModifier",
    ) -> Violations:
        validity = Violations()

        if modifier.HasField("toggle"):
            toggle = modifier.toggle
            if toggle.rows != original.rows or toggle.cols != original.cols:
                validity += (
                    f"(ProgSpec 2.3.1) gadget instance gid={gid} modifier toggle for "
                    + f"{matrix_name} has wrong dimensions: expected "
                    + f"({original.rows}, {original.cols}) but got ({toggle.rows}, {toggle.cols})"
                )

        if modifier.HasField("overwrite"):
            overwrite = modifier.overwrite
            if overwrite.rows != original.rows or overwrite.cols != original.cols:
                validity += (
                    f"(ProgSpec 2.3.2) gadget instance gid={gid} modifier overwrite for "
                    + f"{matrix_name} has wrong dimensions: expected "
                    + f"({original.rows}, {original.cols}) but got ({overwrite.rows}, {overwrite.cols})"
                )

        return validity

    def _modify_check_model_type(
        self, cid: int
    ) -> tuple[pb.CheckModelType, Violations]:
        validity = Violations()

        check_model = self.check_models[cid]
        origin_check_model_type = self.check_model_types[check_model.ctype]
        check_model_type = pb.CheckModelType()
        check_model_type.CopyFrom(origin_check_model_type)

        if check_model.modifier.reroute_remote_gadgets:
            remote_gadgets: list[pb.CheckModelType.RemoteGadget] = list(
                origin_check_model_type.remote_gadgets
            )
            reroute_indices: set[int] = set()

            for reroute in check_model.modifier.reroute_remote_gadgets:
                if reroute.remote_gadget_index in reroute_indices:
                    validity += (
                        f"(ProgSpec 3.3.1) check model instance {cid} modifier reroutes remote "
                        + f"gadget {reroute.remote_gadget_index} multiple times"
                    )
                reroute_indices.add(reroute.remote_gadget_index)

                if reroute.remote_gadget_index >= len(remote_gadgets):
                    if reroute.remote_gadget_index >= 65536:
                        validity += "(ProgSpec 3.3.2) remote gadget index too large"
                        break
                    placeholder = _placeholder_remote_gadget()
                    remote_gadgets.extend(
                        [placeholder]
                        * (reroute.remote_gadget_index + 1 - len(remote_gadgets))
                    )

                remote_gadgets[reroute.remote_gadget_index] = reroute.value

            # clear the remote gadgets and replace with the modified ones
            check_model_type.remote_gadgets.clear()
            check_model_type.remote_gadgets.extend(remote_gadgets)

        return check_model_type, validity

    # pylint: disable=too-many-return-statements
    def _expand_gid(
        self,
        cid: int,
        gid: int,
        remote_gadgets: Sequence[pb.CheckModelType.RemoteGadget],
        ri: int,
        remote_gadget_gid_vec: list[int | None | Violations],
    ) -> None:
        if remote_gadget_gid_vec[ri] is not None:
            return  # already expanded

        remote_gadget = remote_gadgets[ri]
        if _is_placeholder_remote_gadget(remote_gadget):
            return  # no need to expand
        assert (
            remote_gadget.absolute_gid > 0
            or remote_gadget.HasField("input")
            or remote_gadget.HasField("output")
        )

        if remote_gadget.absolute_gid == 0:
            # find the previous gadget instance
            if remote_gadget.HasField("previous_remote_gadget"):
                previous = remote_gadget.previous_remote_gadget
                if remote_gadget_gid_vec[previous] is None:
                    self._expand_gid(
                        cid, gid, remote_gadgets, previous, remote_gadget_gid_vec
                    )
                previous_gid = remote_gadget_gid_vec[previous]
                if previous_gid is None:
                    remote_gadget_gid_vec[ri] = Violations(
                        f"(ProgSpec 3.5.1) cannot expand remote gadget {ri} because "
                        + f"previous_remote_gadget {previous} cannot be expanded "
                        + "(possibly referring to a placeholder)"
                    )
                    return
                if isinstance(previous_gid, Violations):
                    return
            else:
                previous_gid = gid

            previous_gadget = self.gadgets[previous_gid]
            previous_gadget_type = self.gadget_types[previous_gadget.gtype]

            # write the current gadget
            if remote_gadget.HasField("input"):
                if remote_gadget.input >= len(previous_gadget_type.inputs):
                    remote_gadget_gid_vec[ri] = Violations(
                        f"(ProgSpec 3.5.2) invalid input port index {remote_gadget.input} in remote "
                        + f"gadget ri={ri} of check model (cid={cid})"
                    )
                    return
                input_instance = InputPortIndex(
                    gid=previous_gid, port_index=remote_gadget.input
                )
                assert (
                    input_instance in self.peer_output
                ), "all inputs must be connected when created"
                remote_gid = self.peer_output[input_instance].gid
            else:
                if remote_gadget.output >= len(previous_gadget_type.outputs):
                    remote_gadget_gid_vec[ri] = Violations(
                        f"(ProgSpec 3.5.2) invalid output port index {remote_gadget.input} in remote "
                        + f"gadget ri={ri} of check model (cid={cid})"
                    )
                    return
                output_instance = OutputPortIndex(
                    gid=previous_gid, port_index=remote_gadget.output
                )
                if output_instance not in self.peer_input:
                    remote_gadget_gid_vec[ri] = Violations(
                        f"(ProgSpec 3.5.3) remote gadget ri={ri} output port {remote_gadget.output} of"
                        + f" check model (cid={cid}) is not connected"
                    )
                    return
                remote_gid = self.peer_input[output_instance].gid
        else:
            remote_gid = remote_gadget.absolute_gid

        if remote_gadget.expecting_gtype != WILDCARD:
            if self.gadgets[remote_gid].gtype != remote_gadget.expecting_gtype:
                remote_gadget_gid_vec[ri] = Violations(
                    f"(ProgSpec 3.5.4) remote gadget ri={ri} of check model (cid={cid}) "
                    + f"expects gtype {remote_gadget.expecting_gtype}, but got "
                    + f"{self.gadgets[remote_gid].gtype}"
                )
                return

        remote_gadget_gid_vec[ri] = remote_gid

        remote_gadget_instance = self.gadgets[remote_gid]
        remote_gadget_type = self.gadget_types[remote_gadget_instance.gtype]
        if (
            remote_gadget.measurement_bias > 0
            and remote_gadget.measurement_bias >= len(remote_gadget_type.measurements)
        ):
            remote_gadget_gid_vec[ri] = Violations(
                f"(ProgSpec 3.5.5) invalid measurement bias {remote_gadget.measurement_bias} in "
                + f"remote gadget ri={ri} (gtype={remote_gadget_instance.gtype}) "
                + f"of check model (cid={cid})"
            )
            return

    def _collect_error_models(self) -> Violations:
        validity = Violations()

        # error models are not required to instantiate in order
        for eid, error_model in self.error_models.items():
            assert error_model.etype in self.error_model_types

            # apply the modifier
            error_model_type, modify_validity = self._modify_error_model_type(eid)
            if not modify_validity:
                validity += modify_validity
                continue
            self.modified_error_model_types[eid] = error_model_type

            # check that the modified check model satisfies the library specifications
            library_validity = _error_model_validity_single(
                error_model_type,
                self.gadget_types,
                self.port_types,
                self.check_model_types,
                allow_placeholder=True,
            )
            if not library_validity:
                validity += "(ProgSpec 4.4) the modified error model type is invalid"
                validity += library_validity
                # loopy references may block the validator from terminating, skip
                continue

            # expand the cid of the remote check models
            remote_check_model_cid_vec: list[int | None | Violations] = [None] * len(
                error_model_type.remote_check_models
            )
            for ri in range(len(error_model_type.remote_check_models)):
                self._expand_cid(
                    eid,
                    error_model.cid,
                    error_model_type.remote_check_models,
                    ri,
                    remote_check_model_cid_vec,
                )
            expand_validity: Violations = sum(
                (e for e in remote_check_model_cid_vec if isinstance(e, Violations)),
                start=Violations(),
            )
            if not expand_validity:
                validity += expand_validity
                continue
            self.remote_check_model_cid_vecs[eid] = remote_check_model_cid_vec  # type: ignore

            # validate the errors with concrete references to remote check models
            expanded_errors: list[frozenset[CheckIndex]] = []
            for ei, error in enumerate(error_model_type.errors):

                checks: set[CheckIndex] = set()

                for c in error.checks:
                    remote_cid = error_model.cid
                    absolute_check_index = c.check_index
                    if c.HasField("remote_check_model"):
                        remote = error_model_type.remote_check_models[
                            c.remote_check_model
                        ]
                        remote_cid = remote_check_model_cid_vec[c.remote_check_model]
                        if remote_cid is None:
                            validity += (
                                "(ProgSpec 4.6.1) remote check model "
                                + "modified to a placeholder"
                            )
                            continue
                        assert isinstance(remote_cid, int), "all checked"
                        absolute_check_index = c.check_index + remote.check_bias

                    remote_check_model = self.check_models[remote_cid]
                    remote_check_model_type = self.check_model_types[
                        remote_check_model.ctype
                    ]
                    if absolute_check_index >= len(remote_check_model_type.checks):
                        validity += (
                            "(ProgSpec 4.6.2) overflowed remote check index "
                            + f"{absolute_check_index} in error {ei}"
                            + f" of error model (eid={eid})"
                        )

                    key = CheckIndex(cid=remote_cid, check_index=absolute_check_index)
                    if key in checks:
                        validity += (
                            "(ProgSpec 4.6.3) duplicate check in error model "
                            + f"(eid={eid}): remote_cid={remote_cid}, "
                            + f"absolute_check_index={absolute_check_index}"
                        )
                    checks.add(key)

                # record the expanded error
                for check in checks:
                    if check.check_index < len(self.associated_errors[check.cid]):
                        self.associated_errors[check.cid][check.check_index].add(
                            ErrorIndex(eid=eid, error_index=ei)
                        )
                expanded_errors.append(frozenset(checks))

            self.expanded_errors[eid] = expanded_errors

        return validity

    def _modify_error_model_type(
        self, eid: int
    ) -> tuple[pb.ErrorModelType, Violations]:
        validity = Violations()

        error_model = self.error_models[eid]
        origin_error_model_type = self.error_model_types[error_model.etype]
        error_model_type = pb.ErrorModelType()
        error_model_type.CopyFrom(origin_error_model_type)

        modifier = error_model.modifier.probability_modifier

        if len(modifier.probabilities) != 0 and len(modifier.probabilities) != len(
            origin_error_model_type.errors
        ):
            validity += (
                f"(ProgSpec 4.3.1) error model instance {eid} modifier specifies "
                + f"{len(modifier.probabilities)} new probabilities which is neither "
                + f"0 or the number of errors {len(origin_error_model_type.errors)}"
            )
        if validity:  # apply the changes
            for error_index, probability in enumerate(modifier.probabilities):
                if not 0 <= probability <= 1:
                    validity += (
                        "(ProgSpec 4.3.1) invalid modifier probability not in [0, 1]: "
                        + f"invalid probability {probability} for error {error_index}, eid={eid}"
                    )
                else:
                    error_model_type.errors[error_index].probability = probability

        if len(modifier.sparse_indices) != len(modifier.sparse_probabilities):
            validity += (
                f"(ProgSpec 4.3.2) error model instance {eid} modifier specifies unpaired "
                + f"{len(modifier.sparse_indices)} sparse indices but "
                + f"{len(modifier.sparse_probabilities)} sparse probabilities"
            )

        if len(set(modifier.sparse_indices)) != len(modifier.sparse_indices):
            validity += (
                f"(ProgSpec 4.3.3) error model instance {eid} modifier "
                + "has duplicate sparse indices"
            )
        if len(modifier.sparse_indices) > 0:
            max_index = max(modifier.sparse_indices)
            if max_index >= len(origin_error_model_type.errors):
                validity += (
                    f"(ProgSpec 4.3.3) error model instance {eid} modifier "
                    + f"has overflowed sparse index {max_index}"
                )
        if validity:  # apply the changes
            for index, probability in zip(
                modifier.sparse_indices, modifier.sparse_probabilities
            ):
                if not 0 <= probability <= 1:
                    validity += (
                        "(ProgSpec 4.3.3) invalid modifier probability not in [0, 1]: "
                        + f"invalid probability {probability} for error {index}, eid={eid}"
                    )
                else:
                    error_model_type.errors[index].probability = probability

        if error_model.modifier.reroute_remote_check_models:
            remote_check_models: list[pb.ErrorModelType.RemoteCheckModel] = list(
                origin_error_model_type.remote_check_models
            )
            reroute_indices: set[int] = set()

            for reroute in error_model.modifier.reroute_remote_check_models:
                if reroute.remote_check_model_index in reroute_indices:
                    validity += (
                        f"(ProgSpec 4.3.4) error model instance {eid} modifier reroutes remote "
                        + f"check model {reroute.remote_check_model_index} multiple times"
                    )
                reroute_indices.add(reroute.remote_check_model_index)

                if len(remote_check_models) <= reroute.remote_check_model_index:
                    if reroute.remote_check_model_index >= 65536:
                        validity += (
                            "(ProgSpec 4.3.5) remote check model index too large"
                        )
                        break
                    placeholder = _placeholder_remote_check_model()
                    placeholder_round = (
                        reroute.remote_check_model_index + 1 - len(remote_check_models)
                    )
                    remote_check_models.extend([placeholder] * placeholder_round)

                remote_check_models[reroute.remote_check_model_index] = reroute.value

            # clear the remote check models and replace with the modified ones
            error_model_type.remote_check_models.clear()
            error_model_type.remote_check_models.extend(remote_check_models)

        return error_model_type, validity

    # pylint: disable=too-many-return-statements
    def _expand_cid(
        self,
        eid: int,
        cid: int,
        remote_check_models: Sequence[pb.ErrorModelType.RemoteCheckModel],
        ri: int,
        remote_check_model_cid_vec: list[int | None | Violations],
    ) -> None:
        if remote_check_model_cid_vec[ri] is not None:
            return  # already expanded

        remote_check_model = remote_check_models[ri]
        if _is_placeholder_remote_check_model(remote_check_model):
            return  # no need to expand
        assert (
            remote_check_model.absolute_cid > 0
            or remote_check_model.HasField("input")
            or remote_check_model.HasField("output")
        )  # already checked by _error_model_loop_check

        if remote_check_model.absolute_cid == 0:
            # find the previous check model instance
            if remote_check_model.HasField("previous_remote_check_model"):
                previous = remote_check_model.previous_remote_check_model
                if remote_check_model_cid_vec[previous] is None:
                    self._expand_cid(
                        eid,
                        cid,
                        remote_check_models,
                        previous,
                        remote_check_model_cid_vec,
                    )
                previous_cid = remote_check_model_cid_vec[previous]
                if previous_cid is None:
                    remote_check_model_cid_vec[ri] = Violations(
                        f"(ProgSpec 4.5.1) cannot expand remote check model {ri} because "
                        + f"previous_remote_check_model {previous} cannot be expanded "
                        + "(possibly referring to a placeholder)"
                    )
                    return
                if isinstance(previous_cid, Violations):
                    return
            else:
                previous_cid = cid

            previous_check_model = self.check_models[previous_cid]
            previous_gid = previous_check_model.gid
            previous_gadget = self.gadgets[previous_gid]
            previous_gadget_type = self.gadget_types[previous_gadget.gtype]

            # write the current check model
            if remote_check_model.HasField("input"):
                if remote_check_model.input >= len(previous_gadget_type.inputs):
                    remote_check_model_cid_vec[ri] = Violations(
                        f"(ProgSpec 4.5.2) invalid input port index {remote_check_model.input} in "
                        + f"remote gadget ri={ri} (gtype={previous_gadget.gtype}) "
                        + f"of error model (eid={eid})"
                    )
                    return
                input_instance = InputPortIndex(
                    gid=previous_gid, port_index=remote_check_model.input
                )
                assert (
                    input_instance in self.peer_output
                ), "all inputs must be connected when created"
                gid = self.peer_output[input_instance].gid
            else:
                if remote_check_model.output >= len(previous_gadget_type.outputs):
                    remote_check_model_cid_vec[ri] = Violations(
                        f"(ProgSpec 4.5.2) invalid output port index {remote_check_model.output} in "
                        + f"remote gadget ri={ri} (gtype={previous_gadget.gtype}) "
                        + f"of error model (eid={eid})"
                    )
                    return
                output_instance = OutputPortIndex(
                    gid=previous_gid, port_index=remote_check_model.output
                )
                if output_instance not in self.peer_input:
                    remote_check_model_cid_vec[ri] = Violations(
                        f"(ProgSpec 4.5.3) remote gadget ri={ri} output port "
                        + f"{remote_check_model.output} of error model (eid={eid}) is not connected"
                    )
                    return
                gid = self.peer_input[output_instance].gid

            if gid not in self.binding:
                remote_check_model_cid_vec[ri] = Violations(
                    f"(ProgSpec 4.5.4) remote gadget {gid} is not binding to any check model"
                )
                return
            remote_cid = self.binding[gid]
        else:
            remote_cid = remote_check_model.absolute_cid

        if remote_check_model.expecting_ctype != WILDCARD:
            if (
                self.check_models[remote_cid].ctype
                != remote_check_model.expecting_ctype
            ):
                remote_check_model_cid_vec[ri] = Violations(
                    f"(ProgSpec 4.5.5) remote check model ri={ri} of error model (eid={eid}) "
                    + f"expects ctype {remote_check_model.expecting_ctype}, but got "
                    + f"{self.check_models[remote_cid].ctype}"
                )
                return

        remote_check_model_cid_vec[ri] = remote_cid

        remote_check_model_instance = self.check_models[remote_cid]
        remote_check_model_type = self.check_model_types[
            remote_check_model_instance.ctype
        ]
        if remote_check_model.check_bias > 0 and remote_check_model.check_bias >= len(
            remote_check_model_type.checks
        ):
            remote_check_model_cid_vec[ri] = Violations(
                f"(ProgSpec 4.5.6) invalid check bias {remote_check_model.check_bias} in "
                + f"remote check model ri={ri} (ctype={remote_check_model_instance.ctype}) "
                + f"of error model (eid={eid})"
            )
            return

    @cached_property
    def global_measurements(self) -> list[MeasurementIndex]:
        measurements: list[MeasurementIndex] = []
        for gid in sorted(self.gadgets.keys()):
            gadget = self.gadgets[gid]
            gadget_type = self.gadget_types[gadget.gtype]
            for mi in range(len(gadget_type.measurements)):
                measurements.append(MeasurementIndex(gid=gid, measurement_index=mi))
        return measurements

    @cached_property
    def global_measurement_to_idx(self) -> dict[MeasurementIndex, int]:
        mapping: dict[MeasurementIndex, int] = {}
        for idx, measurement in enumerate(self.global_measurements):
            mapping[measurement] = idx
        return mapping

    @cached_property
    def global_checks(self) -> list[CheckIndex]:
        checks: list[CheckIndex] = []
        for cid in sorted(self.check_models.keys()):
            check_model = self.check_models[cid]
            check_model_type = self.check_model_types[check_model.ctype]
            for ci in range(len(check_model_type.checks)):
                checks.append(CheckIndex(cid=cid, check_index=ci))
        return checks

    @cached_property
    def global_check_to_idx(self) -> dict[CheckIndex, int]:
        mapping: dict[CheckIndex, int] = {}
        for idx, check in enumerate(self.global_checks):
            mapping[check] = idx
        return mapping


def _placeholder_remote_gadget() -> pb.CheckModelType.RemoteGadget:
    return pb.CheckModelType.RemoteGadget(tag="placeholder")


def _placeholder_remote_check_model() -> pb.ErrorModelType.RemoteCheckModel:
    return pb.ErrorModelType.RemoteCheckModel(tag="placeholder")
