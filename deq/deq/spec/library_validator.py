"""
=====================
Library Specification
=====================

1. for every :code:`port_type: PortType` in :code:`library.port_types`:

    1. (LibSpec 1.1) :code:`port_type.ptype` is not 0 (which is reserved for wildcard)
    2. (LibSpec 1.2) :code:`port_type.ptype` is unique among :code:`library.port_types`

2. for every :code:`gadget_type: GadgetType` in :code:`library.gadget_types`:

    1. (LibSpec 2.1) :code:`gadget_type.gtype` is not 0 (which is reserved for wildcard)
    2. (LibSpec 2.2) :code:`gadget_type.gtype` is unique among :code:`library.gadget_types`
    3. (LibSpec 2.3) for every :code:`port: GadgetType.Port` in :code:`gadget_type.inputs`,\
        :code:`port.ptype` is defined in :code:`library.port_types`
    4. (LibSpec 2.4) for every :code:`port: GadgetType.Port` in :code:`gadget_type.outputs`,\
        :code:`port.ptype` is defined in :code:`library.port_types`
    5. (LibSpec 2.5) define :code:`input_observables` as the concatenation of those \
        in :code:`gadget_type.inputs`
    6. (LibSpec 2.6) define :code:`output_observables` as the concatenation of those \
        in :code:`gadget_type.outputs`
    7. (LibSpec 2.7) the matrix size of :code:`gadget_type.correction_propagation` is \
        :math:`|`:code:`output_observables`:math:`|`:math:`\\times`\
        :math:`(|`:code:`input_observables`:math:`| + 1)`
    8. for every :code:`readout: GadgetType.Readout` in :code:`gadget_type.readouts`:

        1. (LibSpec 2.8.1) :code:`readout.measurement_indices` does not contain duplicate elements
        2. (LibSpec 2.8.2) every :code:`measurement_index` in :code:`readout.measurement_indices`\
            is a valid index of :code:`gadget_type.measurements`

    9. (LibSpec 2.9) the matrix size of :code:`gadget_type.readout_propagation` is \
        :math:`|`:code:`gadget_type.readouts`:math:`|`:math:`\\times`\
        :math:`(|`:code:`input_observables`:math:`| + 1)`
    10. (LibSpec 2.10) the matrix size of :code:`gadget_type.logical_correction` is \
        :math:`|`:code:`output_observables`:math:`|`:math:`\\times`\
        :math:`|`:code:`gadget_type.readouts`:math:`|`
    11. (LibSpec 2.11) the matrix size of :code:`gadget_type.physical_correction` is \
        :math:`|`:code:`output_observables`:math:`|`:math:`\\times`\
        :math:`|`:code:`gadget_type.measurements`:math:`|`

3. for every :code:`check_model_type: CheckModelType` in :code:`library.check_model_types`:

    1. (LibSpec 3.1) :code:`check_model_type.ctype` is not 0 (which is reserved for wildcard)
    2. (LibSpec 3.2) :code:`check_model_type.ctype` is unique among \
        :code:`library.check_model_types`
    3. for every :code:`remote: CheckModelType.RemoteGadget` in \
        :code:`check_model_type.remote_gadgets`:\

        1. (LibSpec 3.3.1) if :code:`remote.previous_remote_gadget` is set, the value must \
            be a valid index of :code:`check_model_type.remote_gadgets`. When it's not set, the \
            previous gadget is the binding gadget of the current check model
        2. (LibSpec 3.3.2) at least one of the :code:`remote.input` or :code:`remote.output` field \
            is set, which should indicate how to jump from the previous remote gadget (unless \
            :code:`absolute_gid` is non-zero, see below)
        3. (LibSpec 3.3.3) if :code:`remote.expecting_gtype` is not 0, it must be defined in \
            :code:`library.gadget_types` and the :code:`remote.measurement_bias` must be \
            strictly smaller than the number of measurements of the expected gadget type. If the \
            expecting gtype is 0, it is a wildcard and we do not check the measurement bias until \
            we know the gtype at runtime.
        4. (LibSpec 3.3.4) if we do know the gtype of the previous gadget, the \
            :code:`remote.input` or :code:`remote.output`, whichever is present, \
            must be a valid port index of the remote gadget type
        5. (LibSpec 3.3.5) there must not be cyclic references between the remote gadgets
        6. (LibSpec 3.3.6) if :code:`absolute_gid` is non-zero (indicating it's provided), \
            :code:`previous_remote_gadget` and :code:`remote.input` or :code:`remote.output` must not be set

    4. (LibSpec 3.4) if :code:`check_model_type.gtype` is not 0, it must be defined in \
        :code:`library.gadget_types` and we define :code:`gadget_type` as the corresponding \
        gadget type
    5. for every :code:`check` in :code:`check_model_type.checks` and every :code:`measurement` in \
        :code:`check.measurements`:

        1. (LibSpec 3.5.1) if :code:`measurement` is a remote measurement, i.e., \
            :code:`measurement.remote_gadget` is specified, its value must be a valid index in \
            :code:`check_model_type.remote_gadgets`. If we know the gtype of the remote gadget, \
            the value of :code:`measurement.measurement_index + remote_gadget.measurement_bias` \
            (biased measurement index) must be a valid index in the corresponding gadget type
        2. (LibSpec 3.5.2) otherwise :code:`measurement` is a local measurement, so the value of \
            :code:`measurement` must be a valid index of the measurements in :code:`gadget`
        3. (LibSpec 3.5.3) the measurements, in terms of the tuple \
            :code:`(remote_gadget, measurement_index)`, must not be duplicate

4. for every :code:`error_model_type: ErrorModelType` in :code:`library.error_model_types`:

    1. (LibSpec 4.1) :code:`error_model_type.etype` is not 0 (which is reserved for wildcard)
    2. (LibSpec 4.2) :code:`error_model_type.etype` is unique among \
        :code:`library.error_model_types`
    3. for every :code:`remote: ErrorModelType.RemoteCheckModel` in \
        :code:`error_model_type.remote_check_models`:\

        1. (LibSpec 4.3.1) if :code:`remote_check.previous_remote_check_model` is set, the value \
            must be a valid index of :code:`error_model_type.remote_check_models`. When it's not \
            set, the previous check model is the attaching check model of the current error model
        2. (LibSpec 4.3.2) at least one of the :code:`remote.input` or :code:`remote.output` field \
            is set, which should indicate how to jump from the previous remote check model (unless \
            :code:`absolute_cid` is non-zero, see below)
        3. (LibSpec 4.3.3) if :code:`remote.expecting_ctype` is not 0, it must be defined in \
            :code:`library.check_model_types` and the :code:`remote.check_bias` must be \
            strictly smaller than the number of checks of the expected check model type. If the \
            expecting ctype is 0, it is a wildcard and we do not check the check bias until \
            we know the ctype at runtime.
        4. (LibSpec 4.3.4) if we do know the gtype of the previous check model's binding gadget, \
            the :code:`remote.input` or :code:`remote.output`, whichever is present, \
            must be a valid port index of the binding remote gadget type
        5. (LibSpec 4.3.5) there must not be cyclic references between the remote check models
        6. (LibSpec 4.3.6) if :code:`absolute_cid` is non-zero (indicating it's provided), \
            :code:`previous_remote_check_model` and :code:`remote.input` or :code:`remote.output`\
            must not be set

    4. (LibSpec 4.4) if :code:`error_model_type.ctype` is not 0, it must be defined in \
        :code:`library.check_model_types` and we define :code:`check_model_type` as the \
        corresponding check model type. Furthermore, if :code:`check_model_type.gtype` is not 0, \
        it must be defined in :code:`library.gadget_types` and we define :code:`gadget_type` as \
        the corresponding gadget type

    5. for every :code:`error` in :code:`error_model_type.errors`

        1. (LibSpec 4.5.1) :code:`error.probability` must be in the range :math:`0 \\le p \\le 1`
        2. (LibSpec 4.5.2) :code:`error.residual` must be valid indices of the local output \
            observables that the error model attaches to
        3. (LibSpec 4.5.3) :code:`error.readout_flips` must be valid indices of the local readouts \
            in the gadget that the error model attaches to
        4. (LibSpec 4.5.4) :code:`error.residual` must not include duplicate indices
        5. (LibSpec 4.5.5) :code:`error.readout_flips` must not include duplicate indices

    6. for every :code:`error` in :code:`error_model_type.errors` and every :code:`check` in \
        :code:`error.checks`:

        1. (LibSpec 4.6.1) if :code:`check` is a remote check, i.e., \
            :code:`check.remote_check_model` is specified, its value must be a valid index in \
            :code:`error_model_type.remote_check_models`. If we know the ctype of the remote check \
            model, the value of :code:`check.check_index + remote_check_model.check_bias` \
            (biased check index) must be a valid index in the corresponding check model type
        2. (LibSpec 4.6.2) otherwise :code:`check` is a local check, so the value of \
            :code:`check` must be a valid index of the checks in :code:`check_model`
        3. (LibSpec 4.6.3) the checks, in terms of the tuple \
            :code:`(remote_check_model, check_index)`, must not be duplicate, because it is unclear\
            whether such an error flips that check or not

"""

import networkx
import deq.proto.deq_bin_pb2 as pb
import deq.proto.util_pb2 as util_pb
from deq.spec.violations import Violations

WILDCARD = 0

# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint


def is_library_valid(lib: pb.Library) -> Violations:
    assert isinstance(lib, pb.Library)

    validity = _port_validity(lib)
    if not validity:
        return validity

    validity = _gadget_validity(lib)
    if not validity:
        return validity

    validity = _check_model_validity(lib)
    if not validity:
        return validity

    validity = _error_model_validity(lib)
    if not validity:
        return validity

    return Violations()


def _port_validity(lib: pb.Library) -> Violations:
    validity = Violations()
    ptype_set = set()
    for port_type in lib.port_types:
        if port_type.ptype == WILDCARD:
            validity += "(LibSpec 1.1) ptype cannot be 0, it's reserved for wildcard"

        if port_type.ptype in ptype_set:
            validity += f"(LibSpec 1.2) duplicate port type (ptype={port_type.ptype})"
        ptype_set.add(port_type.ptype)

    return validity


def _gadget_validity(lib: pb.Library) -> Violations:
    validity = Violations()
    gtype_set = set()
    port_types = {p.ptype: p for p in lib.port_types}
    for gadget_type in lib.gadget_types:

        if gadget_type.gtype == WILDCARD:
            validity += "(LibSpec 2.1) gtype cannot be 0, it's reserved for wildcard"

        if gadget_type.gtype in gtype_set:
            validity += (
                f"(LibSpec 2.2) duplicate gadget type (gtype={gadget_type.gtype})"
            )
        gtype_set.add(gadget_type.gtype)

        undefined_ports: bool = False
        for port in gadget_type.inputs:
            if port.ptype not in port_types:
                validity += (
                    f"(LibSpec 2.3) undefined input port type (ptype={port.ptype})"
                    + f" in gadget (gtype={gadget_type.gtype})"
                )
                undefined_ports = True
        for port in gadget_type.outputs:
            if port.ptype not in port_types:
                validity += (
                    f"(LibSpec 2.4) undefined output port type (ptype={port.ptype})"
                    + f" in gadget (gtype={gadget_type.gtype})"
                )
                undefined_ports = True
        if undefined_ports:
            continue

        # LibSpec 2.5
        num_input_observables: int = sum(
            len(port_types[port.ptype].observables) for port in gadget_type.inputs
        )
        # LibSpec 2.6
        num_output_observables: int = sum(
            len(port_types[port.ptype].observables) for port in gadget_type.outputs
        )

        # the size of the observable correction propagation matrix must match
        correction_propagation_validity = _bit_matrix_validity(
            gadget_type.correction_propagation,
            rows=num_output_observables,
            cols=num_input_observables + 1,
        )
        if not correction_propagation_validity:
            validity += (
                f"(LibSpec 2.7) correction propagation gtype={gadget_type.gtype}"
            )
            validity += correction_propagation_validity

        num_measurements = len(gadget_type.measurements)
        for readout in gadget_type.readouts:
            if len(set(readout.measurement_indices)) != len(
                readout.measurement_indices
            ):
                validity += (
                    "(LibSpec 2.8.1) duplicate measurement indices in readout "
                    + f"(tag={readout.tag}) of gadget (gtype={gadget_type.gtype})"
                )
            for mi in readout.measurement_indices:
                if mi < 0 or mi >= num_measurements:
                    validity += (
                        f"(LibSpec 2.8.2) invalid measurement index {mi} in readout"
                        + f" (tag={readout.tag}) of gadget (gtype={gadget_type.gtype})"
                    )

        readout_propagation_validity = _bit_matrix_validity(
            gadget_type.readout_propagation,
            rows=len(gadget_type.readouts),
            cols=num_input_observables + 1,
        )
        if not readout_propagation_validity:
            validity += f"(LibSpec 2.9) readout propagation gtype={gadget_type.gtype}"
            validity += readout_propagation_validity

        logical_correction_validity = _bit_matrix_validity(
            gadget_type.logical_correction,
            rows=num_output_observables,
            cols=len(gadget_type.readouts),
        )
        if not logical_correction_validity:
            validity += f"(LibSpec 2.10) logical correction gtype={gadget_type.gtype}"
            validity += logical_correction_validity

        physical_correction_validity = _bit_matrix_validity(
            gadget_type.physical_correction,
            rows=num_output_observables,
            cols=num_measurements,
        )
        if not physical_correction_validity:
            validity += f"(LibSpec 2.11) physical correction gtype={gadget_type.gtype}"
            validity += physical_correction_validity

    return validity


def _check_model_validity(lib: pb.Library) -> Violations:
    validity = Violations()
    ctype_set: set[int] = set()
    gadget_types = {g.gtype: g for g in lib.gadget_types}
    for check_model_type in lib.check_model_types:
        validity += _check_model_validity_single(
            check_model_type, gadget_types, ctype_set
        )
    return validity


def _check_model_validity_single(
    check_model_type: pb.CheckModelType,
    gadget_types: dict[int, pb.GadgetType],
    ctype_set: set[int] | None = None,
    allow_placeholder: bool = False,
) -> Violations:
    validity = Violations()

    if check_model_type.ctype == WILDCARD:
        validity += "(LibSpec 3.1) ctype cannot be 0, it's reserved for wildcard"

    if ctype_set is not None:
        if check_model_type.ctype in ctype_set:
            validity += f"(LibSpec 3.2) duplicate check model type (ctype={check_model_type.ctype})"
        ctype_set.add(check_model_type.ctype)

    # LibSpec 3.3.*
    # check that the remote gadgets references does not contain a loop
    validity += _check_model_loop_check(
        check_model_type, gadget_types, allow_placeholder=allow_placeholder
    )

    gadget_type: pb.GadgetType | None = None
    if check_model_type.gtype != WILDCARD:
        if check_model_type.gtype not in gadget_types:
            validity += (
                f"(LibSpec 3.4) undefined bind gadget type (gtype={check_model_type.gtype}) in "
                + f"check model (ctype={check_model_type.ctype})"
            )
        else:
            gadget_type = gadget_types[check_model_type.gtype]

    for ci, check in enumerate(check_model_type.checks):
        # { (remote_gadget, measurement_index) }
        measurements: set[tuple[int | None, int]] = set()

        for m in check.measurements:
            remote_gadget_index: int | None = None
            if m.HasField("remote_gadget"):
                remote_gadget_index = m.remote_gadget
                if m.remote_gadget >= len(check_model_type.remote_gadgets):
                    validity += (
                        f"(LibSpec 3.5.1) overflowed remote gadget index {m.remote_gadget} in "
                        + f"check {ci} of check model (ctype={check_model_type.ctype})"
                    )
                    continue
                remote = check_model_type.remote_gadgets[m.remote_gadget]
                if (
                    remote.expecting_gtype == WILDCARD
                    or remote.expecting_gtype not in gadget_types
                ):
                    continue
                remote_gadget = gadget_types[remote.expecting_gtype]
                if m.measurement_index + remote.measurement_bias >= len(
                    remote_gadget.measurements
                ):
                    validity += (
                        "(LibSpec 3.5.1) overflowed remote measurement index "
                        + f"{m.measurement_index}+{remote.measurement_bias} in check {ci}"
                        + f" of check model (ctype={check_model_type.ctype})"
                    )
            elif gadget_type is not None:
                if m.measurement_index >= len(gadget_type.measurements):
                    validity += (
                        f"(LibSpec 3.5.2) overflowed measurement index {m.measurement_index} "
                        + f"in check {ci} of check model (ctype={check_model_type.ctype})"
                    )

            key = (remote_gadget_index, m.measurement_index)
            if key in measurements:
                validity += (
                    "(LibSpec 3.5.3) duplicate measurement (remote_gadget="
                    + f"{'None' if remote_gadget_index is None else remote_gadget_index}, "
                    + f"measurement_index={m.measurement_index}) in check {ci} of "
                    + f"check model (ctype={check_model_type.ctype})"
                )
            measurements.add(key)

    return validity


def _check_model_loop_check(
    check_model_type: pb.CheckModelType,
    gadget_types: dict[int, pb.GadgetType],
    allow_placeholder: bool = False,
) -> Violations:
    validity = Violations()
    g = networkx.DiGraph()
    for ri, remote in enumerate(check_model_type.remote_gadgets):
        if allow_placeholder and _is_placeholder_remote_gadget(remote):
            continue  # ignore placeholders

        if remote.HasField("previous_remote_gadget"):
            g.add_edge(ri, remote.previous_remote_gadget)

            if remote.previous_remote_gadget >= len(check_model_type.remote_gadgets):
                validity += (
                    f"(LibSpec 3.3.1) remote gadget {ri} of check model "
                    + f"(ctype={check_model_type.ctype}) refers to an invalid "
                    + f"previous_remote_gadget index of {remote.previous_remote_gadget}"
                )

        if (
            remote.absolute_gid == 0  # not specified
            and not remote.HasField("input")
            and not remote.HasField("output")
        ):
            validity += (
                f"(LibSpec 3.3.2) remote gadget {ri} of check model "
                + f"(ctype={check_model_type.ctype}) must have either input or output specified"
            )

        if remote.expecting_gtype != WILDCARD:
            gtype = remote.expecting_gtype
            if gtype not in gadget_types:
                validity += (
                    f"(LibSpec 3.3.3) undefined gadget type (gtype={gtype}) in "
                    + f"check model (ctype={check_model_type.ctype})"
                )
            else:
                remote_gadget = gadget_types[gtype]
                # sometimes it's essential to allow measurement_bias=0 even if there is no
                # measurement, e.g., as a middle step in the trajectory
                if remote.measurement_bias > 0 and remote.measurement_bias >= len(
                    remote_gadget.measurements
                ):
                    validity += (
                        f"(LibSpec 3.3.3) invalid measurement bias {remote.measurement_bias} "
                        + f"in remote gadget {ri} of check model "
                        + f" (ctype={check_model_type.ctype})"
                    )

        previous_gadget: pb.GadgetType | None = None
        if not remote.HasField("previous_remote_gadget"):
            if check_model_type.gtype != WILDCARD:
                previous_gadget = gadget_types.get(check_model_type.gtype, None)
        elif remote.previous_remote_gadget < len(check_model_type.remote_gadgets):
            prev_remote = check_model_type.remote_gadgets[remote.previous_remote_gadget]
            if prev_remote.expecting_gtype != WILDCARD:
                previous_gadget = gadget_types.get(prev_remote.expecting_gtype, None)
        if previous_gadget is not None:
            if remote.HasField("absolute_gid"):
                pass  # skip because we don't know the absolute gid
            elif remote.HasField("input"):
                if remote.input >= len(previous_gadget.inputs):
                    validity += (
                        f"(LibSpec 3.3.4) invalid input port index {remote.input} in remote "
                        + f"gadget {ri} of check model (ctype={check_model_type.ctype})"
                    )
            else:
                if remote.output >= len(previous_gadget.outputs):
                    validity += (
                        f"(LibSpec 3.3.4) invalid output port index {remote.output} in remote "
                        + f"gadget {ri} of check model (ctype={check_model_type.ctype})"
                    )

        if remote.absolute_gid != 0 and (
            remote.HasField("previous_remote_gadget")
            or remote.HasField("input")
            or remote.HasField("output")
        ):
            validity += (
                f"(LibSpec 3.3.6) remote gadget {ri} of check model "
                + f"(ctype={check_model_type.ctype}) has absolute_gid set, "
                + "so previous_remote_gadget and input/output must not be set"
            )

    loop = next(networkx.simple_cycles(g), None)
    if loop is not None:
        validity += (
            "(LibSpec 3.3.5) remote gadget references in check model "
            + f"(ctype={check_model_type.ctype}) contain a loop {loop}"
        )

    return validity


def _error_model_validity(lib: pb.Library) -> Violations:
    validity = Violations()
    etype_set: set[int] = set()
    gadget_types = {g.gtype: g for g in lib.gadget_types}
    port_types = {p.ptype: p for p in lib.port_types}
    check_model_types = {c.ctype: c for c in lib.check_model_types}
    for error_model_type in lib.error_model_types:
        validity += _error_model_validity_single(
            error_model_type,
            gadget_types,
            port_types,
            check_model_types,
            etype_set,
        )
    return validity


def _error_model_validity_single(
    error_model_type: pb.ErrorModelType,
    gadget_types: dict[int, pb.GadgetType],
    port_types: dict[int, pb.PortType],
    check_model_types: dict[int, pb.CheckModelType],
    etype_set: set[int] | None = None,
    allow_placeholder: bool = False,
) -> Violations:
    validity = Violations()

    if error_model_type.etype == WILDCARD:
        validity += "(LibSpec 4.1) etype cannot be 0, it's reserved for wildcard"

    if etype_set is not None:
        if error_model_type.etype in etype_set:
            validity += f"(LibSpec 4.2) duplicate error model type (etype={error_model_type.etype})"
        etype_set.add(error_model_type.etype)

    # LibSpec 4.3.*
    # check that the remote check models references does not contain a loop
    validity += _error_model_loop_check(
        error_model_type,
        check_model_types,
        gadget_types,
        allow_placeholder=allow_placeholder,
    )

    check_model_type: pb.CheckModelType | None = None
    if error_model_type.ctype != WILDCARD:
        if error_model_type.ctype not in check_model_types:
            validity += (
                "(LibSpec 4.4) undefined attachable check model type "
                + f"(ctype={error_model_type.ctype}) in "
                + f"error model (etype={error_model_type.etype})"
            )
        else:
            check_model_type = check_model_types[error_model_type.ctype]
    gadget_type: pb.GadgetType | None = None
    if (
        check_model_type is not None
        and check_model_type.gtype != WILDCARD
        and check_model_type.gtype in gadget_types
    ):
        gadget_type = gadget_types[check_model_type.gtype]
    num_output_observables = 0
    if gadget_type is not None:
        num_output_observables = sum(
            len(port_types[port.ptype].observables) for port in gadget_type.outputs
        )

    for ei, error in enumerate(error_model_type.errors):

        if not 0 <= error.probability <= 1:
            validity += (
                f"(LibSpec 4.5.1) error probability {error.probability} in error model"
                + f" (etype={error_model_type.etype}) must be in [0, 1]"
            )

        if gadget_type is not None:
            # an error can flip local output observables
            for oi in error.residual:
                if oi >= num_output_observables:
                    validity += (
                        f"(LibSpec 4.5.2) overflowed output observable index {oi} in residual"
                        + f" of errors[{ei}] in error model (etype={error_model_type.etype})"
                    )

            # an error can also flip readouts
            for ri in error.readout_flips:
                if ri >= len(gadget_type.readouts):
                    validity += (
                        f"(LibSpec 4.5.3) overflowed readout index {ri} in readout_flips"
                        + f" of errors[{ei}] in error model (etype={error_model_type.etype})"
                    )

            residual_set = set(error.residual)
            if len(residual_set) != len(error.residual):
                validity += (
                    "(LibSpec 4.5.4) duplicate output observable index in residual"
                    + f" of errors[{ei}] in error model (etype={error_model_type.etype})"
                )

            readout_flip_set = set(error.readout_flips)
            if len(readout_flip_set) != len(error.readout_flips):
                validity += (
                    "(LibSpec 4.5.5) duplicate readout index in readout_flips"
                    + f" of errors[{ei}] in error model (etype={error_model_type.etype})"
                )

        # { (remote_check_model, check_index) }
        checks: set[tuple[int | None, int]] = set()

        for c in error.checks:
            remote_check_model_index: int | None = None
            if c.HasField("remote_check_model"):
                remote_check_model_index = c.remote_check_model
                if c.remote_check_model >= len(error_model_type.remote_check_models):
                    validity += (
                        "(LibSpec 4.6.1) overflowed remote check model index "
                        + f"{c.remote_check_model} in error {ei}"
                        + f" of error model (etype={error_model_type.etype})"
                    )
                    continue
                remote = error_model_type.remote_check_models[c.remote_check_model]
                if (
                    remote.expecting_ctype == WILDCARD
                    or remote.expecting_ctype not in check_model_types
                ):
                    continue
                remote_check_model = check_model_types[remote.expecting_ctype]
                if c.check_index + remote.check_bias >= len(remote_check_model.checks):
                    validity += (
                        f"(LibSpec 4.6.1) overflowed remote check index {c.check_index}+"
                        + f"{remote.check_bias} in error {ei} of error model "
                        + f"(etype={error_model_type.etype})"
                    )
            elif check_model_type is not None:
                if c.check_index >= len(check_model_type.checks):
                    validity += (
                        f"(LibSpec 4.6.2) overflowed check index {c.check_index} in error {ei}"
                        + f" of error model (etype={error_model_type.etype})"
                    )

            key = (remote_check_model_index, c.check_index)
            if key in checks:
                validity += (
                    "(LibSpec 4.6.3) duplicate check (remote_check_model="
                    + f"{'None' if remote_check_model_index is None else remote_check_model_index},"
                    + f" check_index={c.check_index}) in error {ei} of "
                    + f" of error model (etype={error_model_type.etype})"
                )
            checks.add(key)

    return validity


def _error_model_loop_check(
    error_model_type: pb.ErrorModelType,
    check_model_types: dict[int, pb.CheckModelType],
    gadget_types: dict[int, pb.GadgetType],
    allow_placeholder: bool = False,
) -> Violations:
    validity = Violations()

    g = networkx.DiGraph()
    for ri, remote in enumerate(error_model_type.remote_check_models):
        if allow_placeholder and _is_placeholder_remote_check_model(remote):
            continue  # ignore placeholders

        if remote.HasField("previous_remote_check_model"):
            g.add_edge(ri, remote.previous_remote_check_model)

            if remote.previous_remote_check_model >= len(
                error_model_type.remote_check_models
            ):
                validity += (
                    f"(LibSpec 4.3.1) remote check model {ri} of error model "
                    + f"(etype={error_model_type.etype})"
                    + " has invalid previous_remote_check_model index"
                )

        if (
            remote.absolute_cid == 0  # not specified
            and not remote.HasField("input")
            and not remote.HasField("output")
        ):
            validity += (
                f"(LibSpec 4.3.2) remote check model {ri} of error model "
                + f"(etype={error_model_type.etype})"
                + " must have either input or output specified"
            )

        if remote.expecting_ctype != WILDCARD:
            ctype = remote.expecting_ctype
            if ctype not in check_model_types:
                validity += (
                    f"(LibSpec 4.3.3) undefined check model type (ctype={ctype}) in "
                    + f"error model (etype={error_model_type.etype})"
                )
            else:
                remote_check_model = check_model_types[ctype]
                if remote.check_bias > 0 and remote.check_bias >= len(
                    remote_check_model.checks
                ):
                    validity += (
                        f"(LibSpec 4.3.3) invalid check bias {remote.check_bias} "
                        + f"in remote check model {ri} of error model "
                        + f" (etype={error_model_type.etype})"
                    )

        previous_check_model: pb.CheckModelType | None = None
        if not remote.HasField("previous_remote_check_model"):
            if error_model_type.ctype != WILDCARD:
                previous_check_model = check_model_types.get(
                    error_model_type.ctype, None
                )
        elif remote.previous_remote_check_model < len(
            error_model_type.remote_check_models
        ):
            prev_remote = error_model_type.remote_check_models[
                remote.previous_remote_check_model
            ]
            if prev_remote.expecting_ctype != WILDCARD:
                previous_check_model = check_model_types.get(
                    prev_remote.expecting_ctype, None
                )
        if (
            previous_check_model is not None
            and previous_check_model.gtype != WILDCARD
            and previous_check_model.gtype in gadget_types
        ):
            previous_gadget = gadget_types[previous_check_model.gtype]
            if remote.HasField("input"):
                if remote.input >= len(previous_gadget.inputs):
                    validity += (
                        f"(LibSpec 4.3.4) invalid input port index {remote.input} in remote "
                        + f"gadget {ri} of error model (etype={error_model_type.etype})"
                    )
            else:
                if remote.output >= len(previous_gadget.outputs):
                    validity += (
                        f"(LibSpec 4.3.4) invalid output port index {remote.output} in remote "
                        + f"gadget {ri} of error model (etype={error_model_type.etype})"
                    )

        if remote.absolute_cid != 0 and (
            remote.HasField("previous_remote_check_model")
            or remote.HasField("input")
            or remote.HasField("output")
        ):
            validity += (
                f"(LibSpec 4.3.6) remote check model {ri} of error model "
                + f"(etype={error_model_type.etype}) has absolute_cid set, "
                + "so previous_remote_check_model and input/output must not be set"
            )

    loop = next(networkx.simple_cycles(g), None)
    if loop is not None:
        validity += (
            "(LibSpec 4.3.5) remote check model references in error model "
            + f"(etype={error_model_type.etype}) contain a loop {loop}"
        )

    return validity


# pylint: disable=too-many-return-statements
def _bit_matrix_validity(matrix: util_pb.BitMatrix, rows: int, cols: int) -> Violations:
    if rows == 0 or cols == 0:
        if (
            not (matrix.rows == 0 or matrix.cols == 0)
            or len(matrix.i) != 0
            or len(matrix.j) != 0
        ):
            return Violations(["matrix should be empty but data is not"])
        return Violations()
    if matrix.rows != rows or matrix.cols != cols:
        return Violations(
            [
                f"matrix dimensions differ: expected {rows}x{cols}"
                + f", got {matrix.rows}x{matrix.cols}"
            ]
        )
    if len(matrix.i) != len(matrix.j):
        return Violations(["matrix data broken: not pair-wise i,j"])
    ones: set[tuple[int, int]] = set()
    for i, j in zip(matrix.i, matrix.j):
        if i < 0 or i >= matrix.rows or j < 0 or j >= matrix.cols:
            return Violations(["matrix data broken: index out of range"])
        if (i, j) in ones:
            return Violations(["matrix data broken: duplicate entries"])
        ones.add((i, j))
    return Violations()


def _is_placeholder_remote_gadget(gadget: pb.CheckModelType.RemoteGadget) -> bool:
    return (
        gadget.tag == "placeholder"
        and not gadget.HasField("input")
        and not gadget.HasField("output")
    )


def _is_placeholder_remote_check_model(
    remote: pb.ErrorModelType.RemoteCheckModel,
) -> bool:
    return (
        remote.tag == "placeholder"
        and not remote.HasField("input")
        and not remote.HasField("output")
    )
