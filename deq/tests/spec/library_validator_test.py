import deq.proto.deq_bin_pb2 as pb
import deq.proto.util_pb2 as util_pb
from deq.spec.program_validator import is_valid, Violations, ExpandedProgram

# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint


default_library = pb.Library(
    port_types=[
        pb.PortType(
            ptype=1,
            name="rep-code",
            observables=[
                pb.PortType.Observable(tag="logical_z"),
            ],
        ),
        pb.PortType(
            ptype=2,
            name="surface-code",
            observables=[
                pb.PortType.Observable(tag="logical_x"),
                pb.PortType.Observable(tag="logical_z"),
            ],
        ),
    ],
    gadget_types=[
        pb.GadgetType(
            gtype=1,
            name="initialize",
            measurements=[
                pb.GadgetType.Measurement(tag="m1"),
                pb.GadgetType.Measurement(tag="m2"),
            ],
            outputs=[
                pb.GadgetType.Port(ptype=1),
            ],
            correction_propagation=util_pb.BitMatrix(rows=1, cols=1),
            physical_correction=util_pb.BitMatrix(rows=1, cols=2),
        ),
        pb.GadgetType(
            gtype=2,
            name="cnot",
            measurements=[
                pb.GadgetType.Measurement(tag="m3"),
                pb.GadgetType.Measurement(tag="m4"),
            ],
            inputs=[
                pb.GadgetType.Port(ptype=1),
            ],
            outputs=[
                pb.GadgetType.Port(ptype=2),
            ],
            # logical x propagates to logical x, also make logical x naturally flipped
            correction_propagation=util_pb.BitMatrix(
                rows=2, cols=2, i=[0, 0], j=[0, 1]
            ),
            physical_correction=util_pb.BitMatrix(rows=2, cols=2),
        ),
        pb.GadgetType(
            gtype=3,
            name="measure",
            measurements=[
                pb.GadgetType.Measurement(tag="m5"),
                pb.GadgetType.Measurement(tag="m6"),
                pb.GadgetType.Measurement(tag="m7"),
            ],
            inputs=[
                pb.GadgetType.Port(ptype=2),
            ],
            readouts=[
                pb.GadgetType.Readout(tag="r1", measurement_indices=[0, 2]),
            ],
            readout_propagation=util_pb.BitMatrix(rows=1, cols=3, i=[0, 0], j=[0, 2]),
            physical_correction=util_pb.BitMatrix(rows=0, cols=3),
        ),
    ],
    check_model_types=[
        pb.CheckModelType(
            ctype=1,
            gtype=2,
            remote_gadgets=[
                pb.CheckModelType.RemoteGadget(
                    input=0, expecting_gtype=1, measurement_bias=1
                ),
                pb.CheckModelType.RemoteGadget(
                    output=0, previous_remote_gadget=0, expecting_gtype=2
                ),
                pb.CheckModelType.RemoteGadget(output=0, expecting_gtype=3),
            ],
            checks=[
                pb.CheckModelType.Check(
                    tag="c1",
                    measurements=[  # m4, m2, m3, m6
                        pb.CheckModelType.RemoteMeasurement(measurement_index=1),
                        pb.CheckModelType.RemoteMeasurement(remote_gadget=0),
                        pb.CheckModelType.RemoteMeasurement(remote_gadget=1),
                        pb.CheckModelType.RemoteMeasurement(
                            remote_gadget=2, measurement_index=1
                        ),
                    ],
                ),
                pb.CheckModelType.Check(tag="c2"),
                pb.CheckModelType.Check(tag="c3"),
            ],
        ),
        pb.CheckModelType(
            ctype=2,
            gtype=3,
            remote_gadgets=[
                pb.CheckModelType.RemoteGadget(input=0, expecting_gtype=2),
            ],
            checks=[
                pb.CheckModelType.Check(
                    tag="c4",
                    measurements=[  # m5, m6, m7, m3
                        pb.CheckModelType.RemoteMeasurement(measurement_index=0),
                        pb.CheckModelType.RemoteMeasurement(measurement_index=1),
                        pb.CheckModelType.RemoteMeasurement(measurement_index=2),
                        pb.CheckModelType.RemoteMeasurement(remote_gadget=0),
                    ],
                ),
                pb.CheckModelType.Check(tag="c5"),
                pb.CheckModelType.Check(tag="c6"),
            ],
        ),
    ],
    error_model_types=[
        pb.ErrorModelType(
            etype=1,
            ctype=1,  # bind to gtype=2
            remote_check_models=[
                pb.ErrorModelType.RemoteCheckModel(output=0, expecting_ctype=2),
                pb.ErrorModelType.RemoteCheckModel(
                    previous_remote_check_model=0, input=0, expecting_ctype=1
                ),
                pb.ErrorModelType.RemoteCheckModel(output=0, expecting_ctype=2),
            ],
            errors=[
                pb.ErrorModelType.Error(
                    probability=0.1,
                    residual=[0],
                    checks=[  # c1, c2, c5
                        pb.ErrorModelType.RemoteCheck(check_index=0),
                        pb.ErrorModelType.RemoteCheck(
                            remote_check_model=1, check_index=1
                        ),
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
                    readout_flips=[0],
                ),
            ],
        ),
    ],
    program=[
        pb.Instruction(
            gadget=pb.Gadget(
                gtype=1,
                tag="initialize",
            )
        ),
        pb.Instruction(
            gadget=pb.Gadget(
                gtype=2,
                tag="idle",
                connectors=[
                    pb.Gadget.Connector(gid=1, port=0),
                ],
            )
        ),
        pb.Instruction(
            check_model=pb.CheckModel(
                ctype=1,
                gid=2,
            )
        ),
        pb.Instruction(
            error_model=pb.ErrorModel(
                etype=1,
                cid=1,
            )
        ),
        pb.Instruction(
            gadget=pb.Gadget(
                gtype=3,
                tag="measure",
                connectors=[
                    pb.Gadget.Connector(gid=2, port=0),
                ],
            )
        ),
        pb.Instruction(
            check_model=pb.CheckModel(
                ctype=2,
                gid=3,
            )
        ),
        pb.Instruction(
            error_model=pb.ErrorModel(
                etype=2,
                cid=2,
            )
        ),
    ],
)

default_invalid_library = pb.Library(port_types=[pb.PortType(ptype=0)])


def test_library_validity_1_1() -> None:
    default_library_validity = is_valid(default_library)
    assert default_library_validity

    assert "(LibSpec 1.1) ptype cannot be 0, it's reserved for wildcard" in is_valid(
        pb.Library(
            port_types=[
                pb.PortType(ptype=0),
            ],
        )
    )


def test_library_validity_1_2() -> None:
    assert "(LibSpec 1.2) duplicate port type" in is_valid(
        pb.Library(
            port_types=[
                pb.PortType(ptype=1),
                pb.PortType(ptype=1),
            ]
        )
    )


def test_library_validity_2_1() -> None:
    assert "(LibSpec 2.1) gtype cannot be 0, it's reserved for wildcard" in is_valid(
        pb.Library(
            gadget_types=[
                pb.GadgetType(gtype=0),
            ],
        )
    )


def test_library_validity_2_2() -> None:
    assert "(LibSpec 2.2) duplicate gadget type" in is_valid(
        pb.Library(
            port_types=[pb.PortType(ptype=1)],
            gadget_types=[
                pb.GadgetType(gtype=1),
                pb.GadgetType(gtype=1),
            ],
        )
    )


def test_library_validity_2_3_and_2_4() -> None:
    assert (
        "(LibSpec 2.3) undefined input port type",
        "(LibSpec 2.4) undefined output port type",
    ) in is_valid(
        pb.Library(
            gadget_types=[
                pb.GadgetType(gtype=1, inputs=[pb.GadgetType.Port(ptype=100)]),
                pb.GadgetType(gtype=2, outputs=[pb.GadgetType.Port(ptype=100)]),
            ],
        )
    )


def test_library_validity_2_7() -> None:
    assert ("(LibSpec 2.7)", "matrix should be empty but data is not") in is_valid(
        pb.Library(
            gadget_types=[
                pb.GadgetType(
                    gtype=1,
                    correction_propagation=util_pb.BitMatrix(rows=1, cols=1),
                    physical_correction=util_pb.BitMatrix(rows=0, cols=0),
                ),
            ],
        )
    )

    # observable correction propagation matrix size
    def correction_propagation_tester(
        matrix: util_pb.BitMatrix,
    ) -> Violations | ExpandedProgram:
        return is_valid(
            pb.Library(
                port_types=default_library.port_types,
                gadget_types=[
                    pb.GadgetType(
                        gtype=1,
                        inputs=[
                            pb.GadgetType.Port(ptype=2),
                        ],
                        outputs=[
                            pb.GadgetType.Port(ptype=1),
                        ],
                        correction_propagation=matrix,
                        physical_correction=util_pb.BitMatrix(rows=1, cols=0),
                    ),
                ],
            ),
        )

    assert (
        "(LibSpec 2.7)",
        "matrix dimensions differ",
    ) in correction_propagation_tester(util_pb.BitMatrix())
    assert (
        "(LibSpec 2.7)",
        "matrix data broken: not pair-wise i,j",
    ) in correction_propagation_tester(util_pb.BitMatrix(rows=1, cols=3, i=[0], j=[]))
    assert (
        "(LibSpec 2.7)",
        "matrix data broken: index out of range",
    ) in correction_propagation_tester(
        util_pb.BitMatrix(rows=1, cols=3, i=[100], j=[100])
    )
    assert (
        "(LibSpec 2.7)",
        "matrix data broken: duplicate entries",
    ) in correction_propagation_tester(
        util_pb.BitMatrix(rows=1, cols=3, i=[0, 0], j=[0, 0])
    )


def test_library_validity_2_8_1() -> None:

    # empty readout is invalid
    def readout_tester(
        readouts: list[pb.GadgetType.Readout],
    ) -> Violations | ExpandedProgram:
        return is_valid(
            pb.Library(
                gadget_types=[
                    pb.GadgetType(
                        gtype=1,
                        measurements=[pb.GadgetType.Measurement()],
                        readouts=readouts,
                        physical_correction=util_pb.BitMatrix(rows=0, cols=1),
                    ),
                ],
            )
        )

    assert "(LibSpec 2.8.1) duplicate measurement indices in readout" in readout_tester(
        [pb.GadgetType.Readout(measurement_indices=[0, 0])]
    )
    assert "(LibSpec 2.8.2) invalid measurement index" in readout_tester(
        [pb.GadgetType.Readout(measurement_indices=[100])]
    )


def test_library_validity_2_9() -> None:

    # need proper readout propagation
    def readout_propagation_tester(
        matrix: util_pb.BitMatrix,
    ) -> Violations | ExpandedProgram:
        return is_valid(
            pb.Library(
                port_types=default_library.port_types,
                gadget_types=[
                    pb.GadgetType(
                        gtype=1,
                        measurements=[pb.GadgetType.Measurement()],
                        inputs=[
                            pb.GadgetType.Port(ptype=2),
                        ],
                        correction_propagation=util_pb.BitMatrix(rows=0, cols=3),
                        readouts=[
                            pb.GadgetType.Readout(tag="r1", measurement_indices=[0]),
                        ],
                        readout_propagation=matrix,
                        logical_correction=util_pb.BitMatrix(rows=0, cols=1),
                        physical_correction=util_pb.BitMatrix(rows=0, cols=1),
                    ),
                ],
            )
        )

    assert readout_propagation_tester(util_pb.BitMatrix(rows=1, cols=3, i=[0], j=[0]))
    assert ("(LibSpec 2.9)", "matrix dimensions differ") in readout_propagation_tester(
        util_pb.BitMatrix()
    )
    assert (
        "(LibSpec 2.9)",
        "matrix data broken: not pair-wise i,j",
    ) in readout_propagation_tester(util_pb.BitMatrix(rows=1, cols=3, i=[0], j=[]))
    assert (
        "(LibSpec 2.9)",
        "matrix data broken: index out of range",
    ) in readout_propagation_tester(util_pb.BitMatrix(rows=1, cols=3, i=[100], j=[100]))
    assert (
        "(LibSpec 2.9)",
        "matrix data broken: duplicate entries",
    ) in readout_propagation_tester(
        util_pb.BitMatrix(rows=1, cols=3, i=[0, 0], j=[0, 0])
    )


def test_library_validity_2_10() -> None:

    def logical_correction_tester(
        matrix: util_pb.BitMatrix,
    ) -> Violations | ExpandedProgram:
        gadget_type = pb.GadgetType()
        gadget_type.CopyFrom(default_library.gadget_types[2])
        gadget_type.outputs.MergeFrom([pb.GadgetType.Port(ptype=2)])
        gadget_type.logical_correction.CopyFrom(matrix)
        gadget_type.correction_propagation.CopyFrom(util_pb.BitMatrix(rows=2, cols=3))
        gadget_type.physical_correction.CopyFrom(util_pb.BitMatrix(rows=2, cols=3))
        return is_valid(
            pb.Library(
                port_types=default_library.port_types,
                gadget_types=[gadget_type],
            )
        )

    assert logical_correction_tester(util_pb.BitMatrix(rows=2, cols=1, i=[0], j=[0]))
    assert (
        "(LibSpec 2.10)",
        "matrix dimensions differ",
    ) in logical_correction_tester(util_pb.BitMatrix(rows=3, cols=3, i=[0], j=[0]))
    assert (
        "(LibSpec 2.10)",
        "matrix data broken: not pair-wise i,j",
    ) in logical_correction_tester(util_pb.BitMatrix(rows=2, cols=1, i=[0], j=[]))
    assert (
        "(LibSpec 2.10)",
        "matrix data broken: index out of range",
    ) in logical_correction_tester(util_pb.BitMatrix(rows=2, cols=1, i=[100], j=[100]))
    assert (
        "(LibSpec 2.10)",
        "matrix data broken: duplicate entries",
    ) in logical_correction_tester(
        util_pb.BitMatrix(rows=2, cols=1, i=[0, 0], j=[0, 0])
    )


def test_library_validity_3_1() -> None:
    assert "(LibSpec 3.1) ctype cannot be 0, it's reserved for wildcard" in is_valid(
        pb.Library(
            check_model_types=[
                pb.CheckModelType(ctype=0),
            ],
        )
    )


def test_library_validity_3_2() -> None:
    assert "(LibSpec 3.2) duplicate check model type" in is_valid(
        pb.Library(
            check_model_types=[
                pb.CheckModelType(ctype=1),
                pb.CheckModelType(ctype=1),
            ]
        )
    )


def test_library_validity_3_3_5() -> None:
    assert ("(LibSpec 3.3.5)", "contain a loop") in is_valid(
        pb.Library(
            check_model_types=[
                pb.CheckModelType(
                    ctype=1,
                    remote_gadgets=[
                        pb.CheckModelType.RemoteGadget(previous_remote_gadget=2),
                        pb.CheckModelType.RemoteGadget(previous_remote_gadget=0),
                        pb.CheckModelType.RemoteGadget(previous_remote_gadget=1),
                    ],
                ),
            ],
        )
    )


def test_library_validity_3_3_3_and_3_4() -> None:
    assert (
        "(LibSpec 3.3.3) undefined gadget type",
        "(LibSpec 3.4) undefined bind gadget type",
    ) in is_valid(
        pb.Library(
            check_model_types=[
                pb.CheckModelType(
                    ctype=1,
                    remote_gadgets=[
                        pb.CheckModelType.RemoteGadget(expecting_gtype=100),
                    ],
                    gtype=100,
                ),
            ],
        )
    )


def test_library_validity_3_others() -> None:
    assert (
        "(LibSpec 3.3.3) invalid measurement bias ",
        "(LibSpec 3.3.2)",
        "must have either input or output specified",
        "(LibSpec 3.3.1)",
        "refers to an invalid previous_remote_gadget index",
        "(LibSpec 3.3.4) invalid input port index",
        "(LibSpec 3.3.4) invalid output port index",
        "(LibSpec 3.5.1) overflowed remote gadget index",
        "(LibSpec 3.5.1) overflowed remote measurement index",
        "(LibSpec 3.5.2) overflowed measurement index",
        "(LibSpec 3.5.3) duplicate measurement",
    ) in is_valid(
        pb.Library(
            port_types=default_library.port_types,
            gadget_types=default_library.gadget_types,
            check_model_types=[
                pb.CheckModelType(
                    ctype=1,
                    gtype=1,
                    remote_gadgets=[
                        pb.CheckModelType.RemoteGadget(output=0, expecting_gtype=1),
                        pb.CheckModelType.RemoteGadget(output=0),
                        pb.CheckModelType.RemoteGadget(
                            previous_remote_gadget=100,
                            expecting_gtype=1,
                            measurement_bias=100,
                        ),
                        pb.CheckModelType.RemoteGadget(
                            input=100,
                            expecting_gtype=1,
                            measurement_bias=100,
                        ),
                        pb.CheckModelType.RemoteGadget(
                            output=100,
                            expecting_gtype=1,
                            measurement_bias=100,
                        ),
                    ],
                    checks=[
                        pb.CheckModelType.Check(
                            measurements=[
                                pb.CheckModelType.RemoteMeasurement(
                                    measurement_index=2
                                ),
                                pb.CheckModelType.RemoteMeasurement(remote_gadget=100),
                                pb.CheckModelType.RemoteMeasurement(remote_gadget=1),
                                pb.CheckModelType.RemoteMeasurement(remote_gadget=2),
                                pb.CheckModelType.RemoteMeasurement(
                                    remote_gadget=2
                                ),  # duplicate
                            ]
                        ),
                    ],
                ),
            ],
        )
    )


def test_library_validity_4_1() -> None:
    assert "(LibSpec 4.1) etype cannot be 0, it's reserved for wildcard" in is_valid(
        pb.Library(
            error_model_types=[
                pb.ErrorModelType(etype=0),
            ],
        )
    )


def test_library_validity_4_2() -> None:
    assert "(LibSpec 4.2) duplicate error model type" in is_valid(
        pb.Library(
            error_model_types=[
                pb.ErrorModelType(etype=1),
                pb.ErrorModelType(etype=1),
            ]
        )
    )


def test_library_validity_4_3_5() -> None:
    assert ("(LibSpec 4.3.5)", "contain a loop") in is_valid(
        pb.Library(
            error_model_types=[
                pb.ErrorModelType(
                    etype=1,
                    remote_check_models=[
                        pb.ErrorModelType.RemoteCheckModel(
                            previous_remote_check_model=0
                        ),
                        pb.ErrorModelType.RemoteCheckModel(
                            previous_remote_check_model=1
                        ),
                    ],
                ),
            ],
        )
    )


def test_library_validity_4_3_3_and_4_4() -> None:
    assert (
        "(LibSpec 4.3.3) undefined check model type",
        "(LibSpec 4.4) undefined attachable check model type",
    ) in is_valid(
        pb.Library(
            error_model_types=[
                pb.ErrorModelType(
                    etype=1,
                    remote_check_models=[
                        pb.ErrorModelType.RemoteCheckModel(expecting_ctype=100),
                    ],
                    ctype=100,
                ),
            ],
        )
    )


def test_library_validity_4_others() -> None:
    assert (
        "(LibSpec 4.3.1)",
        "has invalid previous_remote_check_model index",
        "(LibSpec 4.3.2)",
        "must have either input or output specified",
        "(LibSpec 4.3.3) invalid check bias ",
        "(LibSpec 4.3.4)",
        "invalid input port index",
        "(LibSpec 4.3.4) invalid output port index",
        "(LibSpec 4.5.1)",
        "must be in [0, 1]",
        "(LibSpec 4.5.2) overflowed output observable index",
        "(LibSpec 4.5.3) overflowed readout index",
        "(LibSpec 4.5.4) duplicate output observable index in residual",
        "(LibSpec 4.5.5) duplicate readout index in readout_flips",
        "(LibSpec 4.6.1) overflowed remote check model index",
        "(LibSpec 4.6.1) overflowed remote check index",
        "(LibSpec 4.6.2) overflowed check index",
        "(LibSpec 4.6.3) duplicate check",
    ) in is_valid(
        pb.Library(
            port_types=default_library.port_types,
            gadget_types=default_library.gadget_types,
            check_model_types=default_library.check_model_types,
            error_model_types=[
                pb.ErrorModelType(
                    etype=1,
                    ctype=1,
                    remote_check_models=[
                        pb.ErrorModelType.RemoteCheckModel(output=0, expecting_ctype=1),
                        pb.ErrorModelType.RemoteCheckModel(output=0),
                        pb.ErrorModelType.RemoteCheckModel(
                            previous_remote_check_model=100,
                            expecting_ctype=1,
                            check_bias=100,
                        ),
                        pb.ErrorModelType.RemoteCheckModel(
                            input=100,
                            expecting_ctype=1,
                            check_bias=100,
                        ),
                        pb.ErrorModelType.RemoteCheckModel(
                            output=100,
                            expecting_ctype=1,
                            check_bias=100,
                        ),
                    ],
                    errors=[
                        pb.ErrorModelType.Error(
                            probability=1.1,
                            checks=[
                                pb.ErrorModelType.RemoteCheck(check_index=3),
                                pb.ErrorModelType.RemoteCheck(remote_check_model=100),
                                pb.ErrorModelType.RemoteCheck(remote_check_model=1),
                                pb.ErrorModelType.RemoteCheck(remote_check_model=2),
                                # duplicate
                                pb.ErrorModelType.RemoteCheck(remote_check_model=2),
                            ],
                            residual=[100],
                            readout_flips=[100],
                        ),
                        pb.ErrorModelType.Error(
                            probability=1.1,
                            residual=[0, 0],  # duplicate
                            readout_flips=[0, 0],  # duplicate
                        ),
                    ],
                ),
            ],
        )
    )


def test_library_validity_3_3_6_absolute_gid() -> None:
    assert "(LibSpec 3.3.6)" in is_valid(
        pb.Library(
            port_types=default_library.port_types,
            gadget_types=default_library.gadget_types,
            check_model_types=[
                pb.CheckModelType(
                    ctype=1,
                    gtype=1,
                    remote_gadgets=[
                        pb.CheckModelType.RemoteGadget(
                            absolute_gid=5,
                            output=0,
                        ),
                    ],
                ),
            ],
        )
    )

    assert "(LibSpec 3.3.6)" in is_valid(
        pb.Library(
            port_types=default_library.port_types,
            gadget_types=default_library.gadget_types,
            check_model_types=[
                pb.CheckModelType(
                    ctype=1,
                    gtype=1,
                    remote_gadgets=[
                        pb.CheckModelType.RemoteGadget(
                            absolute_gid=5,
                            input=0,
                        ),
                    ],
                ),
            ],
        )
    )


def test_library_validity_4_3_6_absolute_cid() -> None:
    assert "(LibSpec 4.3.6)" in is_valid(
        pb.Library(
            port_types=default_library.port_types,
            gadget_types=default_library.gadget_types,
            check_model_types=default_library.check_model_types,
            error_model_types=[
                pb.ErrorModelType(
                    etype=1,
                    ctype=1,
                    remote_check_models=[
                        pb.ErrorModelType.RemoteCheckModel(
                            absolute_cid=5,
                            output=0,
                        ),
                    ],
                ),
            ],
        )
    )

    assert "(LibSpec 4.3.6)" in is_valid(
        pb.Library(
            port_types=default_library.port_types,
            gadget_types=default_library.gadget_types,
            check_model_types=default_library.check_model_types,
            error_model_types=[
                pb.ErrorModelType(
                    etype=1,
                    ctype=1,
                    remote_check_models=[
                        pb.ErrorModelType.RemoteCheckModel(
                            absolute_cid=5,
                            input=0,
                        ),
                    ],
                ),
            ],
        )
    )
