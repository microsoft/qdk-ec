"""
=====================
Library Equivalence
=====================

Checking if any program calling the two libraries result in the same decoding
hypergraph. This is more strict than checking the equivalence of two programs, because
the latter only checks if the ultimate decoding hypergraphs are the same, which
might be achieved by different implementations of gadgets and their corresponding
instantiation. The equivalence of two libraries means that for arbitrary program that
instantiates gadgets, check models and error models from the two libraries result
in exactly the same decoding hypergraph. This means that one only have the freedom
to change the tagging, reordering the remote references, merging the probabilities
of errors that have exactly the same effect, splitting the measurements into
different gadgets, splitting the checks into different check models, etc.

1. (LibEq 1) the two libraries must be valid, i.e., satisfying all LibSpec. The function provides\
    an option to skip this check if the caller has already validated the libraries.
2. for every :code:`port_type_1: PortType` and :code:`port_type_2: PortType` with the same\
    :code:`ptype`

    1. (LibEq 2.1) the set of :code:`ptype` must be equal, although the ordering can be different\
        for the remaining, we will assume that :code:`port_type_1` and :code:`port_type_2` have\
        been paired up, i.e., :code:`port_type_1.ptype == port_type_2.ptype`
    2. (LibEq 2.2) the number of logical observables must be the same

3. for every pair of :code:`gadget_type_1: GadgetType` and :code:`gadget_type_2: GadgetType`

    1. (LibEq 3.1) the set of :code:`gtype` must be equal, although the ordering can be different.\
        for the remaining, we will assume that :code:`gadget_type_1` and :code:`gadget_type_2` have\
        been paired up, i.e., :code:`gadget_type_1.gtype == gadget_type_2.gtype`
    2. (LibEq 3.2) the number of measurements must be the same
    3. (LibEq 3.3) the number of input/output port instances must be the same
    4. (LibEq 3.4) for each of the input/output port instances, the :code:`ptype` must be the same
    5. (LibEq 3.5) the correction propagation matrices must be the same
    6. (LibEq 3.6) the number of readouts must be the same, and for each pair of readouts, \
        the set of measurement indices must be the same
    7. (LibEq 3.7) the readout propagation matrices must be the same
    8. (LibEq 3.8) the logical correction matrices must be the same
    9. (LibEq 3.9) the physical correction matrices must be the same

4. for every pair of :code:`check_model_type_1: CheckModelType` and \
    :code:`check_model_type_2: CheckModelType`

    1. (LibEq 4.1) the set of :code:`ctype` must be equal, although the ordering can be different.\
        for the remaining, we will assume that :code:`check_model_type_1` and \
        :code:`check_model_type_2` have been paired up, i.e., \
        :code:`check_model_type_1.ctype == check_model_type_2.ctype`
    2. (LibEq 4.2) the number of remote gadgets must be the same. We require that all the remote \
        gadgets are exactly equivalent to each other in the same order, because we need the library\
        to be equivalent for any program, including those modify the remote gadgets at runtime.
    3. for each pair of remote gadgets

        1. (LibEq 4.3.1) the :code:`previous_remote_gadget` must be the same: must be both None,\
            or both not None and have the same value
        2. (LibEq 4.3.2) the selection of :code:`input` or :code:`output` must be the same and\
            the values must also be the same
        3. (LibEq 4.3.3) the :code:`measurement_bias` must be the same
        4. (LibEq 4.3.4) the :code:`absolute_gid` must be the same
        5. Note that we don't enforce :code:`expecting_gtype` to be the same, thus two equivalent\
            libraries may not panic at the same time but if they do not panic, they are equivalent.

    4. (LibEq 4.4) the number of checks must be the same.
    5. for each pair of checks

        1. (LibEq 4.5.1) the involved set of measurements, in terms of the tuple\
            :code:`(remote_gadget, measurement_index)` must be the same. Note that (LibSpec 3.5.3)\
            ensures that no duplication of such tuple exists
        2. (LibEq 4.5.2) the value of :code:`naturally_flipped` must be the same

5. for every pair of :code:`error_model_type_1: ErrorModelType` and \
    :code:`error_model_type_2: ErrorModelType`

    1. (LibEq 5.1) the set of :code:`etype` must be equal, although the ordering can be different.\
        for the remaining, we will assume that :code:`error_model_type_1` and \
        :code:`error_model_type_2` have been paired up, i.e., \
        :code:`error_model_type_1.etype == error_model_type_2.etype`
    2. (LibEq 5.2) the number of remote check models must be the same. We require that all the \
        remote check models are exactly equivalent to each other in the same order, because we \
        need the library to be equivalent for any program, including those modify the remote \
        check models at runtime.
    3. for each pair of remote check models

        1. (LibEq 5.3.1) the :code:`previous_remote_check_model` must be the same: must be both\
            None, or both not None and have the same value
        2. (LibEq 5.3.2) the selection of :code:`input` or :code:`output` must be the same and\
            the values must also be the same
        3. (LibEq 5.3.3) the :code:`check_bias` must be the same
        4. (LibEq 5.3.4) the :code:`absolute_cid` must be the same
        5. Note that we don't enforce :code:`expecting_ctype` to be the same, thus two equivalent\
            libraries may not panic at the same time but if they do not panic, they are equivalent.

    4. (LibEq 5.4) the number of errors must be the same.
    5. for each pair of errors

        1. (LibEq 5.5.1) the involved set of checks, in terms of the tuple\
            :code:`(remote_check_model, check_index)` must be the same. Note that (LibSpec 4.6.3)\
            ensures that no duplication of such tuple exists
        2. (LibEq 5.5.2) the set of :code:`residual` must be the same. Note that (LibSpec 4.5.4)\
            ensures that no duplication exists within each :code:`residual`
        3. (LibEq 5.5.3) the set of :code:`readout_flips` must be the same. Note that \
            (LibSpec 4.5.5) ensures that no duplication exists within each :code:`naturally_flipped`
        4. (LibEq 5.5.4) the value of :code:`probability` must be the same, up to some floating\
            point precision (at most a relative error of :math:`10^{-6}`).

"""

import math
import deq.proto.deq_bin_pb2 as pb
import deq.proto.util_pb2 as util_pb
from deq.spec.violations import Violations
from deq.spec.library_validator import is_library_valid
from deq.spec.common import bitmatrix_of

# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint


# pylint: disable=too-many-return-statements
def are_libraries_equivalent(
    lib1: pb.Library, lib2: pb.Library, validate: bool = True, rel_tol: float = 1e-6
) -> Violations:
    if validate:
        validity1 = is_library_valid(lib1)
        if not validity1:
            return Violations("(LibEq 1) lib1 is not valid") + validity1
        validity2 = is_library_valid(lib2)
        if not validity2:
            return Violations("(LibEq 1) lib2 is not valid") + validity2

    equivalence = _port_equivalence(lib1, lib2)
    if not equivalence:
        return equivalence

    equivalence = _gadget_equivalence(lib1, lib2)
    if not equivalence:
        return equivalence

    equivalence = _check_model_equivalence(lib1, lib2)
    if not equivalence:
        return equivalence

    equivalence = _error_model_equivalence(lib1, lib2, rel_tol=rel_tol)
    if not equivalence:
        return equivalence

    return Violations()


def _port_equivalence(lib1: pb.Library, lib2: pb.Library) -> Violations:

    port_types_1 = {port_type.ptype: port_type for port_type in lib1.port_types}
    port_types_2 = {port_type.ptype: port_type for port_type in lib2.port_types}
    ptype_set_1 = frozenset(port_types_1.keys())
    ptype_set_2 = frozenset(port_types_2.keys())
    if ptype_set_1 != ptype_set_2:
        only_1 = set(ptype_set_1 - ptype_set_2)
        only_2 = set(ptype_set_2 - ptype_set_1)
        return Violations(
            "(LibEq 2.1) the sets of ptype are not equal,"
            + f" those only in lib1: {only_1}, those only in lib2: {only_2}"
        )

    equivalence = Violations()
    for ptype, port_type_1 in port_types_1.items():
        port_type_2 = port_types_2[ptype]

        # the number of observables must be the same
        if len(port_type_1.observables) != len(port_type_2.observables):
            equivalence += (
                f"(LibEq 2.2) port (ptype={ptype}) number of observables differ"
            )

    return equivalence


def _gadget_equivalence(lib1: pb.Library, lib2: pb.Library) -> Violations:

    gadget_types_1 = {
        gadget_type.gtype: gadget_type for gadget_type in lib1.gadget_types
    }
    gadget_types_2 = {
        gadget_type.gtype: gadget_type for gadget_type in lib2.gadget_types
    }
    gtype_set_1 = frozenset(gadget_types_1.keys())
    gtype_set_2 = frozenset(gadget_types_2.keys())
    if gtype_set_1 != gtype_set_2:
        only_1 = set(gtype_set_1 - gtype_set_2)
        only_2 = set(gtype_set_2 - gtype_set_1)
        return Violations(
            "(LibEq 3.1) the sets of gtype are not equal,"
            + f" those only in lib1: {only_1}, those only in lib2: {only_2}"
        )

    equivalence = Violations()
    for gtype, gadget_type_1 in gadget_types_1.items():
        gadget_type_2 = gadget_types_2[gtype]

        if len(gadget_type_1.measurements) != len(gadget_type_2.measurements):
            equivalence += f"(LibEq 3.2) number of measurements differ (gtype={gtype})"
            continue

        input_equivalence = _port_instances_equivalence(
            gtype,
            gadget_type_1.inputs,
            gadget_type_2.inputs,
            is_input=True,
        )
        if not input_equivalence:
            equivalence += f"gadget (gtype={gtype}) input ports nonequivalent"
            equivalence += input_equivalence
            continue

        output_equivalence = _port_instances_equivalence(
            gtype,
            gadget_type_1.outputs,
            gadget_type_2.outputs,
            is_input=False,
        )
        if not output_equivalence:
            equivalence += f"gadget (gtype={gtype}) output ports nonequivalent"
            equivalence += output_equivalence
            continue

        correction_propagation_equivalence = _bit_matrix_equivalence(
            gadget_type_1.correction_propagation, gadget_type_2.correction_propagation
        )
        if not correction_propagation_equivalence:
            equivalence += (
                f"(LibEq 3.5) correction propagation nonequivalent  (gtype={gtype})"
            )
            equivalence += correction_propagation_equivalence

        equivalence += _readouts_equivalence(
            gadget_type_1.readouts, gadget_type_2.readouts
        )

        readout_propagation_equivalence = _bit_matrix_equivalence(
            gadget_type_1.readout_propagation, gadget_type_2.readout_propagation
        )
        if not readout_propagation_equivalence:
            equivalence += (
                f"(LibEq 3.7) readout propagation nonequivalent: gadget (gtype={gtype})"
            )
            equivalence += readout_propagation_equivalence

        logical_correction_equivalence = _bit_matrix_equivalence(
            gadget_type_1.logical_correction, gadget_type_2.logical_correction
        )
        if not logical_correction_equivalence:
            equivalence += (
                "(LibEq 3.8) logical correction nonequivalent: "
                + f"gadget (gtype={gtype})"
            )
            equivalence += logical_correction_equivalence

        physical_correction_equivalence = _bit_matrix_equivalence(
            gadget_type_1.physical_correction, gadget_type_2.physical_correction
        )
        if not physical_correction_equivalence:
            equivalence += (
                "(LibEq 3.9) physical correction nonequivalent: "
                + f"gadget (gtype={gtype})"
            )
            equivalence += physical_correction_equivalence

    return equivalence


def _readouts_equivalence(
    readouts1: list[pb.GadgetType.Readout], readouts2: list[pb.GadgetType.Readout]
) -> Violations:
    if len(readouts1) != len(readouts2):
        return Violations("(LibEq 3.6) number of readouts differ")

    for ri, (readout1, readout2) in enumerate(zip(readouts1, readouts2)):
        measurement_set_1 = frozenset(readout1.measurement_indices)
        measurement_set_2 = frozenset(readout2.measurement_indices)
        if measurement_set_1 != measurement_set_2:
            only_1 = set(measurement_set_1 - measurement_set_2)
            only_2 = set(measurement_set_2 - measurement_set_1)
            return Violations(
                f"(LibEq 3.6) the sets of measurements differ in readout {ri},"
                + f" those only in lib1: {only_1}, those only in lib2: {only_2}"
            )

    return Violations()


def _port_instances_equivalence(
    gtype: int,
    ports1: list[pb.GadgetType.Port],
    ports2: list[pb.GadgetType.Port],
    is_input: bool,
) -> Violations:
    if len(ports1) != len(ports2):
        return Violations(
            "(LibEq 3.3) number of port instances differ"
            + f" ({'input' if is_input else 'output'} in gtype={gtype}"
        )

    equivalence = Violations()
    for i, (port1, port2) in enumerate(zip(ports1, ports2)):
        if port1.ptype != port2.ptype:
            equivalence += (
                f"(LibEq 3.4) {i}-th port instance types "
                + f"differ: {port1.ptype} vs {port2.ptype}"
                + f" ({'input' if is_input else 'output'} in gtype={gtype}"
            )
    return equivalence


def _check_model_equivalence(lib1: pb.Library, lib2: pb.Library) -> Violations:

    check_model_types_1 = {
        check_model_type.ctype: check_model_type
        for check_model_type in lib1.check_model_types
    }
    check_model_types_2 = {
        check_model_type.ctype: check_model_type
        for check_model_type in lib2.check_model_types
    }
    ctype_set_1 = frozenset(check_model_types_1.keys())
    ctype_set_2 = frozenset(check_model_types_2.keys())
    if ctype_set_1 != ctype_set_2:
        only_1 = set(ctype_set_1 - ctype_set_2)
        only_2 = set(ctype_set_2 - ctype_set_1)
        return Violations(
            "(LibEq 4.1) the sets of ctype are not equal,"
            + f" those only in lib1: {only_1}, those only in lib2: {only_2}"
        )

    equivalence = Violations()
    for ctype, check_model_type_1 in check_model_types_1.items():
        check_model_type_2 = check_model_types_2[ctype]

        # LibEq 4.3.*
        remote_gadget_equivalence = _remote_gadget_equivalence(
            ctype, check_model_type_1.remote_gadgets, check_model_type_2.remote_gadgets
        )
        if not remote_gadget_equivalence:
            equivalence += remote_gadget_equivalence

        # LibEq 4.4 and 4.5.*
        checks_equivalence = _checks_equivalence(
            ctype, check_model_type_1.checks, check_model_type_2.checks
        )
        if not checks_equivalence:
            equivalence += checks_equivalence

    return equivalence


def _remote_gadget_equivalence(
    ctype: int,
    remote_gadgets1: list[pb.CheckModelType.RemoteGadget],
    remote_gadgets2: list[pb.CheckModelType.RemoteGadget],
) -> Violations:

    equivalence = Violations()

    if len(remote_gadgets1) != len(remote_gadgets2):
        equivalence += Violations(
            f"(LibEq 4.2) number of remote gadgets differ (ctype={ctype})"
        )

    for ri, (remote_1, remote_2) in enumerate(zip(remote_gadgets1, remote_gadgets2)):
        has_previous_1 = remote_1.HasField("previous_remote_gadget")
        has_previous_2 = remote_2.HasField("previous_remote_gadget")
        if has_previous_1 != has_previous_2:
            equivalence += (
                "(LibEq 4.3.1) previous_remote_gadget presence differ "
                + f"in remote gadget {ri} (ctype={ctype})"
            )
        elif has_previous_1:
            if remote_1.previous_remote_gadget != remote_2.previous_remote_gadget:
                equivalence += (
                    "(LibEq 4.3.1) previous_remote_gadget values differ "
                    + f"in remote gadget {ri} (ctype={ctype})"
                )

        is_input_1 = remote_1.HasField("input")
        is_input_2 = remote_2.HasField("input")
        if is_input_1 != is_input_2:
            equivalence += (
                "(LibEq 4.3.2) input/output selection differ "
                + f"in remote gadget {ri} (ctype={ctype})"
            )
        elif is_input_1:
            if remote_1.input != remote_2.input:
                equivalence += (
                    "(LibEq 4.3.2) input values differ "
                    + f"in remote gadget {ri} (ctype={ctype})"
                )
        else:
            if remote_1.output != remote_2.output:
                equivalence += (
                    "(LibEq 4.3.2) output values differ "
                    + f"in remote gadget {ri} (ctype={ctype})"
                )

        if remote_1.measurement_bias != remote_2.measurement_bias:
            equivalence += (
                "(LibEq 4.3.3) measurement_bias values differ "
                + f"in remote gadget {ri} (ctype={ctype})"
            )

        if remote_1.absolute_gid != remote_2.absolute_gid:
            equivalence += (
                "(LibEq 4.3.4) absolute_gid value differ "
                + f"in remote gadget {ri} (ctype={ctype})"
            )

    return equivalence


def _checks_equivalence(
    ctype: int,
    checks1: list[pb.CheckModelType.Check],
    checks2: list[pb.CheckModelType.Check],
) -> Violations:

    equivalence = Violations()

    if len(checks1) != len(checks2):
        equivalence += Violations(
            f"(LibEq 4.4) number of checks differ (ctype={ctype})"
        )

    for ci, (check1, check2) in enumerate(zip(checks1, checks2)):

        def _measurement_set_of(
            check: pb.CheckModelType.Check,
        ) -> set[tuple[int | None, int]]:
            measurement_set: set[tuple[int | None, int]] = set()
            for m in check.measurements:
                remote_gadget: int | None = None
                if m.HasField("remote_gadget"):
                    remote_gadget = m.remote_gadget
                measurement_set.add((remote_gadget, m.measurement_index))
            return measurement_set

        measurement_set_1 = _measurement_set_of(check1)
        measurement_set_2 = _measurement_set_of(check2)
        if measurement_set_1 != measurement_set_2:
            only_1 = set(measurement_set_1 - measurement_set_2)
            only_2 = set(measurement_set_2 - measurement_set_1)
            equivalence += (
                f"(LibEq 4.5.1) the sets of measurements differ in check {ci} (ctype={ctype}),"
                + f" those only in lib1: {only_1}, those only in lib2: {only_2}"
            )

        if check1.naturally_flipped != check2.naturally_flipped:
            equivalence += (
                "(LibEq 4.5.2) naturally_flipped values differ in "
                + f"check {ci} (ctype={ctype})"
            )

    return equivalence


def _error_model_equivalence(
    lib1: pb.Library, lib2: pb.Library, rel_tol: float
) -> Violations:

    error_model_types_1 = {
        error_model_type.etype: error_model_type
        for error_model_type in lib1.error_model_types
    }
    error_model_types_2 = {
        error_model_type.etype: error_model_type
        for error_model_type in lib2.error_model_types
    }
    etype_set_1 = frozenset(error_model_types_1.keys())
    etype_set_2 = frozenset(error_model_types_2.keys())
    if etype_set_1 != etype_set_2:
        only_1 = set(etype_set_1 - etype_set_2)
        only_2 = set(etype_set_2 - etype_set_1)
        return Violations(
            "(LibEq 5.1) the sets of etype are not equal,"
            + f" those only in lib1: {only_1}, those only in lib2: {only_2}"
        )

    equivalence = Violations()
    for etype, error_model_type_1 in error_model_types_1.items():
        error_model_type_2 = error_model_types_2[etype]

        # LibEq 5.3.*
        remote_check_model_equivalence = _remote_check_model_equivalence(
            etype,
            error_model_type_1.remote_check_models,
            error_model_type_2.remote_check_models,
        )
        if not remote_check_model_equivalence:
            equivalence += remote_check_model_equivalence

        # LibEq 5.4 and 5.5.*
        errors_equivalence = _errors_equivalence(
            etype, error_model_type_1.errors, error_model_type_2.errors, rel_tol=rel_tol
        )
        if not errors_equivalence:
            equivalence += errors_equivalence

    return equivalence


def _remote_check_model_equivalence(
    etype: int,
    remote_check_models_1: list[pb.ErrorModelType.RemoteCheckModel],
    remote_check_models_2: list[pb.ErrorModelType.RemoteCheckModel],
) -> Violations:

    equivalence = Violations()

    if len(remote_check_models_1) != len(remote_check_models_2):
        equivalence += Violations(
            f"(LibEq 5.2) number of remote check models differ (etype={etype})"
        )

    for ri, (remote_1, remote_2) in enumerate(
        zip(remote_check_models_1, remote_check_models_2)
    ):
        has_previous_1 = remote_1.HasField("previous_remote_check_model")
        has_previous_2 = remote_2.HasField("previous_remote_check_model")
        if has_previous_1 != has_previous_2:
            equivalence += (
                "(LibEq 5.3.1) previous_remote_check_model presence differ "
                + f"in remote check model {ri} (etype={etype})"
            )
        elif has_previous_1:
            if (
                remote_1.previous_remote_check_model
                != remote_2.previous_remote_check_model
            ):
                equivalence += (
                    "(LibEq 5.3.1) previous_remote_check_model values differ "
                    + f"in remote check model {ri} (etype={etype})"
                )

        is_input_1 = remote_1.HasField("input")
        is_input_2 = remote_2.HasField("input")
        if is_input_1 != is_input_2:
            equivalence += (
                "(LibEq 5.3.2) input/output selection differ "
                + f"in remote check model {ri} (etype={etype})"
            )
        elif is_input_1:
            if remote_1.input != remote_2.input:
                equivalence += (
                    "(LibEq 5.3.2) input values differ "
                    + f"in remote check model {ri} (etype={etype})"
                )
        else:
            if remote_1.output != remote_2.output:
                equivalence += (
                    "(LibEq 5.3.2) output values differ "
                    + f"in remote check model {ri} (etype={etype})"
                )

        if remote_1.check_bias != remote_2.check_bias:
            equivalence += (
                "(LibEq 5.3.3) check_bias values differ "
                + f"in remote check model {ri} (etype={etype})"
            )

        if remote_1.absolute_cid != remote_2.absolute_cid:
            equivalence += (
                "(LibEq 5.3.4) absolute_cid value differ "
                + f"in remote check model {ri} (etype={etype})"
            )

    return equivalence


def _errors_equivalence(
    etype: int,
    errors1: list[pb.ErrorModelType.Error],
    errors2: list[pb.ErrorModelType.Error],
    rel_tol: float,
) -> Violations:

    equivalence = Violations()

    if len(errors1) != len(errors2):
        equivalence += Violations(
            f"(LibEq 5.4) number of errors differ (etype={etype}): {len(errors1)} vs {len(errors2)}"
        )

    for ei, (error1, error2) in enumerate(zip(errors1, errors2)):

        def _check_set_of(
            error: pb.ErrorModelType.Error,
        ) -> set[tuple[int | None, int]]:
            check_set: set[tuple[int | None, int]] = set()
            for c in error.checks:
                remote_check_model: int | None = None
                if c.HasField("remote_check_model"):
                    remote_check_model = c.remote_check_model
                check_set.add((remote_check_model, c.check_index))
            return check_set

        check_set_1 = _check_set_of(error1)
        check_set_2 = _check_set_of(error2)
        if check_set_1 != check_set_2:
            only_1 = set(check_set_1 - check_set_2)
            only_2 = set(check_set_2 - check_set_1)
            equivalence += (
                f"(LibEq 5.5.1) the sets of checks differ in error {ei} (etype={etype}),"
                + f" those only in lib1: {only_1}, those only in lib2: {only_2}"
            )

        residual_set_1 = set(error1.residual)
        residual_set_2 = set(error2.residual)
        if residual_set_1 != residual_set_2:
            only_1 = set(residual_set_1 - residual_set_2)
            only_2 = set(residual_set_2 - residual_set_1)
            equivalence += (
                f"(LibEq 5.5.2) the sets of residual differ in error {ei} (etype={etype}),"
                + f" those only in lib1: {only_1}, those only in lib2: {only_2}"
            )

        readout_flips_set_1 = set(error1.readout_flips)
        readout_flips_set_2 = set(error2.readout_flips)
        if readout_flips_set_1 != readout_flips_set_2:
            only_1 = set(readout_flips_set_1 - readout_flips_set_2)
            only_2 = set(readout_flips_set_2 - readout_flips_set_1)
            equivalence += (
                f"(LibEq 5.5.3) the sets of readout_flips differ in error {ei} (etype={etype}),"
                + f" those only in lib1: {only_1}, those only in lib2: {only_2}"
            )

        if not math.isclose(error1.probability, error2.probability, rel_tol=rel_tol):
            equivalence += (
                "(LibEq 5.5.4) probability values differ in "
                + f"error {ei} (etype={etype}): {error1.probability} vs {error2.probability}"
            )

    return equivalence


def _bit_matrix_equivalence(
    matrix1: util_pb.BitMatrix, matrix2: util_pb.BitMatrix
) -> Violations:
    if (matrix1.rows == 0 or matrix1.cols == 0) and (
        matrix2.rows == 0 or matrix2.cols == 0
    ):
        return Violations()  # empty matrices are always equal
    if matrix1.rows != matrix2.rows or matrix1.cols != matrix2.cols:
        return Violations("matrix dimensions differ")
    m1 = bitmatrix_of(matrix1)
    m2 = bitmatrix_of(matrix2)
    if m1 != m2:
        return Violations(f"matrix data differ: {m1} vs {m2}")
    return Violations()
