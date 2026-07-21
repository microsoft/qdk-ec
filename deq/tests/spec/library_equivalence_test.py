from deq.spec.library_equivalence import are_libraries_equivalent
from deq.spec.violations import Violations
import deq.proto.deq_bin_pb2 as pb
import deq.proto.util_pb2 as util_pb
from tests.spec.library_validator_test import default_library

# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint


def test_library_equivalence_1() -> None:
    invalid_library = pb.Library(port_types=[pb.PortType(ptype=0)])

    assert not are_libraries_equivalent(invalid_library, invalid_library)

    assert "(LibEq 1) lib1 is not valid" in are_libraries_equivalent(
        invalid_library, default_library
    )

    assert "(LibEq 1) lib2 is not valid" in are_libraries_equivalent(
        default_library, invalid_library
    )


ports_library = pb.Library(port_types=default_library.port_types)


def test_library_equivalence_2_1() -> None:

    default_library_compatibility = are_libraries_equivalent(
        ports_library, ports_library
    )
    assert "" not in default_library_compatibility
    assert ("",) not in default_library_compatibility

    assert "(LibEq 2.1) the sets of ptype are not equal" in are_libraries_equivalent(
        ports_library,
        pb.Library(
            port_types=[
                pb.PortType(ptype=100),
                pb.PortType(ptype=101),
            ]
        ),
    )


def test_library_equivalence_2_2() -> None:
    assert (
        "(LibEq 2.2) port (ptype=1) number of observables differ",
        "(LibEq 2.2) port (ptype=2) number of observables differ",
    ) in are_libraries_equivalent(
        ports_library,
        pb.Library(
            port_types=[
                pb.PortType(ptype=1),
                pb.PortType(ptype=2),
            ]
        ),
    )

    assert are_libraries_equivalent(
        ports_library,
        pb.Library(
            port_types=[
                pb.PortType(ptype=1, observables=[pb.PortType.Observable()]),
                pb.PortType(
                    ptype=2,
                    observables=[pb.PortType.Observable(), pb.PortType.Observable()],
                ),
            ]
        ),
    )


gadgets_library = pb.Library(
    port_types=default_library.port_types,
    gadget_types=default_library.gadget_types,
)


def test_library_equivalence_3_1() -> None:
    assert "(LibEq 3.1) the sets of gtype are not equal" in are_libraries_equivalent(
        gadgets_library,
        pb.Library(
            port_types=default_library.port_types,
            gadget_types=[
                default_library.gadget_types[0],
                pb.GadgetType(gtype=4),
                default_library.gadget_types[2],
            ],
        ),
    )


def test_library_equivalence_3_2() -> None:
    assert "(LibEq 3.2) number of measurements differ" in are_libraries_equivalent(
        gadgets_library,
        pb.Library(
            port_types=default_library.port_types,
            gadget_types=[
                default_library.gadget_types[0],
                pb.GadgetType(gtype=2, measurements=[]),
                default_library.gadget_types[2],
            ],
        ),
    )


def test_library_equivalence_3_3() -> None:
    assert "(LibEq 3.3) number of port instances differ" in are_libraries_equivalent(
        gadgets_library,
        pb.Library(
            port_types=default_library.port_types,
            gadget_types=[
                default_library.gadget_types[0],
                pb.GadgetType(
                    gtype=2,
                    measurements=[
                        pb.GadgetType.Measurement(),
                        pb.GadgetType.Measurement(),
                    ],
                    inputs=[],
                    physical_correction=util_pb.BitMatrix(rows=0, cols=2),
                ),
                default_library.gadget_types[2],
            ],
        ),
    )


def test_library_equivalence_3_4() -> None:
    assert "(LibEq 3.4) 0-th port instance types differ" in are_libraries_equivalent(
        gadgets_library,
        pb.Library(
            port_types=default_library.port_types,
            gadget_types=[
                default_library.gadget_types[0],
                pb.GadgetType(
                    gtype=2,
                    measurements=[
                        pb.GadgetType.Measurement(),
                        pb.GadgetType.Measurement(),
                    ],
                    inputs=default_library.gadget_types[1].inputs,
                    outputs=[pb.GadgetType.Port(ptype=1)],
                    correction_propagation=util_pb.BitMatrix(rows=1, cols=2),
                    physical_correction=util_pb.BitMatrix(rows=1, cols=2),
                ),
                default_library.gadget_types[2],
            ],
        ),
    )


def test_library_equivalence_3_5() -> None:
    assert ("(LibEq 3.5)", "matrix data differ:") in are_libraries_equivalent(
        gadgets_library,
        pb.Library(
            port_types=default_library.port_types,
            gadget_types=[
                default_library.gadget_types[0],
                pb.GadgetType(
                    gtype=2,
                    measurements=[
                        pb.GadgetType.Measurement(),
                        pb.GadgetType.Measurement(),
                    ],
                    inputs=default_library.gadget_types[1].inputs,
                    outputs=default_library.gadget_types[1].outputs,
                    correction_propagation=util_pb.BitMatrix(
                        rows=2, cols=2, i=[0], j=[1]
                    ),
                    physical_correction=util_pb.BitMatrix(rows=2, cols=2),
                ),
                default_library.gadget_types[2],
            ],
        ),
    )


gadget_type_3 = default_library.gadget_types[2]


def readouts_tester(readouts: list[pb.GadgetType.Readout]) -> Violations:
    return are_libraries_equivalent(
        gadgets_library,
        pb.Library(
            port_types=default_library.port_types,
            gadget_types=[
                default_library.gadget_types[0],
                default_library.gadget_types[1],
                pb.GadgetType(
                    gtype=3,
                    measurements=gadget_type_3.measurements,
                    inputs=gadget_type_3.inputs,
                    readouts=readouts,
                    readout_propagation=util_pb.BitMatrix(rows=len(readouts), cols=3),
                    physical_correction=util_pb.BitMatrix(rows=0, cols=3),
                ),
            ],
        ),
    )


def test_library_equivalence_3_6() -> None:
    assert "(LibEq 3.6) number of readouts differ" in readouts_tester([])
    assert "(LibEq 3.6) the sets of measurements differ in readout" in readouts_tester(
        [pb.GadgetType.Readout()]
    )
    assert "(LibEq 3.6) the sets of measurements differ in readout" in readouts_tester(
        [pb.GadgetType.Readout(measurement_indices=[1, 2])]
    )


def test_library_equivalence_3_7() -> None:
    assert "(LibEq 3.7) readout propagation nonequivalent" in are_libraries_equivalent(
        gadgets_library,
        pb.Library(
            port_types=default_library.port_types,
            gadget_types=[
                default_library.gadget_types[0],
                default_library.gadget_types[1],
                pb.GadgetType(
                    gtype=3,
                    measurements=gadget_type_3.measurements,
                    inputs=gadget_type_3.inputs,
                    readouts=gadget_type_3.readouts,
                    readout_propagation=util_pb.BitMatrix(rows=1, cols=3, i=[0], j=[0]),
                    physical_correction=util_pb.BitMatrix(rows=0, cols=3),
                ),
            ],
        ),
    )


# the default library didn't have a case that has logical correction, let's make one
library2 = pb.Library(
    port_types=default_library.port_types,
    gadget_types=[
        pb.GadgetType(
            gtype=1,
            name="initialize",
            measurements=[
                pb.GadgetType.Measurement(tag="m1"),
                pb.GadgetType.Measurement(tag="m2"),
            ],
            inputs=[pb.GadgetType.Port(ptype=1)],
            outputs=[pb.GadgetType.Port(ptype=2)],
            readouts=[pb.GadgetType.Readout(tag="r1", measurement_indices=[0, 1])],
            correction_propagation=util_pb.BitMatrix(rows=2, cols=2),
            readout_propagation=util_pb.BitMatrix(rows=1, cols=2),
            logical_correction=util_pb.BitMatrix(rows=2, cols=1),
            physical_correction=util_pb.BitMatrix(rows=2, cols=2),
        ),
    ],
)
gadget_type_2_1 = library2.gadget_types[0]


def test_library_equivalence_3_8() -> None:
    assert "(LibEq 3.8) logical correction nonequivalent" in are_libraries_equivalent(
        library2,
        pb.Library(
            port_types=library2.port_types,
            gadget_types=[
                pb.GadgetType(
                    gtype=1,
                    measurements=gadget_type_2_1.measurements,
                    inputs=gadget_type_2_1.inputs,
                    outputs=gadget_type_2_1.outputs,
                    readouts=gadget_type_2_1.readouts,
                    correction_propagation=gadget_type_2_1.correction_propagation,
                    readout_propagation=gadget_type_2_1.readout_propagation,
                    logical_correction=util_pb.BitMatrix(
                        rows=2, cols=1, i=[0], j=[0]  # different from library2
                    ),
                    physical_correction=util_pb.BitMatrix(rows=2, cols=2),
                ),
            ],
        ),
    )


check_common = {
    "port_types": default_library.port_types,
    "gadget_types": default_library.gadget_types,
}

checks_library = pb.Library(
    **check_common,
    check_model_types=default_library.check_model_types,
)


def test_library_equivalence_4_1() -> None:
    assert "(LibEq 4.1) the sets of ctype are not equal" in are_libraries_equivalent(
        checks_library,
        pb.Library(
            **check_common,
            check_model_types=[
                default_library.check_model_types[0],
                pb.CheckModelType(ctype=4),
            ],
        ),
    )


def test_library_equivalence_4_2_and_4_3_1() -> None:
    assert (
        "(LibEq 4.2) number of remote gadgets differ",
        "(LibEq 4.3.1) previous_remote_gadget presence differ",
    ) in are_libraries_equivalent(
        checks_library,
        pb.Library(
            **check_common,
            check_model_types=[
                default_library.check_model_types[0],
                pb.CheckModelType(
                    ctype=2,
                    remote_gadgets=[
                        pb.CheckModelType.RemoteGadget(
                            input=0, expecting_gtype=2, previous_remote_gadget=1
                        ),
                        pb.CheckModelType.RemoteGadget(input=0, expecting_gtype=2),
                    ],
                ),
            ],
        ),
    )


def test_library_equivalence_4_others() -> None:
    assert (
        "(LibEq 4.3.1) previous_remote_gadget values differ",
        "(LibEq 4.3.2) input/output selection differ",
        "(LibEq 4.3.2) input values differ",
        "(LibEq 4.3.2) output values differ",
        "(LibEq 4.3.3) measurement_bias values differ",
        "(LibEq 4.4) number of checks differ",
    ) in are_libraries_equivalent(
        checks_library,
        pb.Library(
            **check_common,
            check_model_types=[
                pb.CheckModelType(
                    ctype=1,
                    gtype=0,  # wildcard to confuse the validator
                    remote_gadgets=[
                        pb.CheckModelType.RemoteGadget(input=10, measurement_bias=2),
                        pb.CheckModelType.RemoteGadget(
                            output=2, previous_remote_gadget=2  # different value
                        ),
                        pb.CheckModelType.RemoteGadget(input=0),
                    ],
                ),
                default_library.check_model_types[1],
            ],
        ),
    )


def test_library_equivalence_4_5_1_and_4_5_2() -> None:
    assert (
        "(LibEq 4.5.1) the sets of measurements differ in check",
        "(LibEq 4.5.2) naturally_flipped values differ",
    ) in are_libraries_equivalent(
        checks_library,
        pb.Library(
            **check_common,
            check_model_types=[
                pb.CheckModelType(
                    ctype=1,
                    gtype=0,  # wildcard to confuse the validator
                    remote_gadgets=default_library.check_model_types[0].remote_gadgets,
                    checks=[
                        pb.CheckModelType.Check(
                            measurements=[
                                pb.CheckModelType.RemoteMeasurement(
                                    measurement_index=1
                                ),
                                # pb.CheckModelType.RemoteMeasurement(remote_gadget=0),
                                # pb.CheckModelType.RemoteMeasurement(remote_gadget=1),
                                pb.CheckModelType.RemoteMeasurement(
                                    remote_gadget=2, measurement_index=1
                                ),
                            ]
                        ),
                        pb.CheckModelType.Check(naturally_flipped=True),
                        pb.CheckModelType.Check(),
                    ],
                ),
                default_library.check_model_types[1],
            ],
        ),
    )


error_common = {
    "port_types": default_library.port_types,
    "gadget_types": default_library.gadget_types,
    "check_model_types": default_library.check_model_types,
}

errors_library = pb.Library(
    **error_common,
    error_model_types=default_library.error_model_types,
)


def test_library_equivalence_5_1() -> None:
    assert "(LibEq 5.1) the sets of etype are not equal" in are_libraries_equivalent(
        errors_library,
        pb.Library(
            **error_common,
            error_model_types=[
                default_library.error_model_types[0],
                pb.ErrorModelType(etype=4),
            ],
        ),
    )


def test_library_equivalence_5_2_and_5_3_1() -> None:
    assert (
        "(LibEq 5.2) number of remote check models differ",
        "(LibEq 5.3.1) previous_remote_check_model presence differ",
    ) in are_libraries_equivalent(
        errors_library,
        pb.Library(
            **error_common,
            error_model_types=[
                default_library.error_model_types[0],
                pb.ErrorModelType(
                    etype=2,
                    remote_check_models=[
                        pb.ErrorModelType.RemoteCheckModel(
                            input=0, expecting_ctype=2, previous_remote_check_model=1
                        ),
                        pb.ErrorModelType.RemoteCheckModel(input=0, expecting_ctype=2),
                    ],
                ),
            ],
        ),
    )


def test_library_equivalence_5_3_and_5_4() -> None:
    assert (
        "(LibEq 5.3.1) previous_remote_check_model values differ",
        "(LibEq 5.3.2) input/output selection differ",
        "(LibEq 5.3.2) input values differ",
        "(LibEq 5.3.2) output values differ",
        "(LibEq 5.3.3) check_bias values differ",
        "(LibEq 5.4) number of errors differ",
    ) in are_libraries_equivalent(
        errors_library,
        pb.Library(
            **error_common,
            error_model_types=[
                pb.ErrorModelType(
                    etype=1,
                    ctype=0,  # wildcard to confuse the validator
                    remote_check_models=[
                        pb.ErrorModelType.RemoteCheckModel(output=10, check_bias=2),
                        pb.ErrorModelType.RemoteCheckModel(
                            input=2, previous_remote_check_model=2  # different value
                        ),
                        pb.ErrorModelType.RemoteCheckModel(input=0),
                    ],
                ),
                default_library.error_model_types[1],
            ],
        ),
    )


def test_library_equivalence_5_5() -> None:
    assert (
        "(LibEq 5.5.1) the sets of checks differ in error",
        "(LibEq 5.5.2) the sets of residual differ",
        "(LibEq 5.5.3) the sets of readout_flips differ",
        "(LibEq 5.5.4) probability values differ",
    ) in are_libraries_equivalent(
        errors_library,
        pb.Library(
            **error_common,
            error_model_types=[
                pb.ErrorModelType(
                    etype=1,
                    ctype=0,  # wildcard to confuse the validator
                    remote_check_models=default_library.error_model_types[
                        0
                    ].remote_check_models,
                    errors=[
                        pb.ErrorModelType.Error(
                            probability=0.100002,  # different probability
                            checks=[
                                pb.ErrorModelType.RemoteCheck(check_index=0),
                                # pb.ErrorModelType.RemoteCheck(
                                #     remote_check_model=1, check_index=1
                                # ),
                                pb.ErrorModelType.RemoteCheck(
                                    remote_check_model=2, check_index=1
                                ),
                            ],
                        ),
                    ],
                ),
                pb.ErrorModelType(
                    etype=2,
                    ctype=2,  # bind to gtype=3
                    remote_check_models=[
                        pb.ErrorModelType.RemoteCheckModel(input=0, expecting_ctype=1),
                    ],
                    errors=[
                        pb.ErrorModelType.Error(
                            probability=0.1,
                            # readout_flips=[0],  # different
                        ),
                    ],
                ),
            ],
        ),
    )


def test_library_equivalence_4_3_4_absolute_gid() -> None:
    assert "(LibEq 4.3.4) absolute_gid value differ" in are_libraries_equivalent(
        pb.Library(
            **check_common,
            check_model_types=[
                pb.CheckModelType(
                    ctype=1,
                    remote_gadgets=[
                        pb.CheckModelType.RemoteGadget(absolute_gid=10),
                    ],
                ),
            ],
        ),
        pb.Library(
            **check_common,
            check_model_types=[
                pb.CheckModelType(
                    ctype=1,
                    remote_gadgets=[
                        pb.CheckModelType.RemoteGadget(absolute_gid=20),
                    ],
                ),
            ],
        ),
    )


def test_library_equivalence_5_3_4_absolute_cid() -> None:
    assert "(LibEq 5.3.4) absolute_cid value differ" in are_libraries_equivalent(
        pb.Library(
            **error_common,
            error_model_types=[
                pb.ErrorModelType(
                    etype=1,
                    remote_check_models=[
                        pb.ErrorModelType.RemoteCheckModel(absolute_cid=10),
                    ],
                ),
            ],
        ),
        pb.Library(
            **error_common,
            error_model_types=[
                pb.ErrorModelType(
                    etype=1,
                    remote_check_models=[
                        pb.ErrorModelType.RemoteCheckModel(absolute_cid=20),
                    ],
                ),
            ],
        ),
    )


def test_library_equivalence_3_9() -> None:
    """LibEq 3.9: two libs differing only in physical_correction are non-equivalent."""
    assert "(LibEq 3.9) physical correction nonequivalent" in are_libraries_equivalent(
        library2,
        pb.Library(
            port_types=library2.port_types,
            gadget_types=[
                pb.GadgetType(
                    gtype=1,
                    measurements=gadget_type_2_1.measurements,
                    inputs=gadget_type_2_1.inputs,
                    outputs=gadget_type_2_1.outputs,
                    readouts=gadget_type_2_1.readouts,
                    correction_propagation=gadget_type_2_1.correction_propagation,
                    readout_propagation=gadget_type_2_1.readout_propagation,
                    logical_correction=gadget_type_2_1.logical_correction,
                    physical_correction=util_pb.BitMatrix(
                        rows=2, cols=2, i=[0], j=[0]  # different from library2
                    ),
                ),
            ],
        ),
    )
