import deq.proto.deq_bin_pb2 as pb
from deq.spec.physical_validator import is_valid_and_physical
from tests.spec.library_validator_test import default_invalid_library

# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint

default_physical_library = pb.Library(
    gadget_types=[
        pb.GadgetType(
            gtype=1,
            measurements=[
                pb.GadgetType.Measurement(tag="m1"),
                pb.GadgetType.Measurement(tag="m2"),
                pb.GadgetType.Measurement(tag="m3"),
            ],
        ),
    ],
    check_model_types=[
        pb.CheckModelType(
            ctype=1,
            gtype=1,
            checks=[
                pb.CheckModelType.Check(
                    tag="c1",
                    measurements=[
                        pb.CheckModelType.RemoteMeasurement(measurement_index=i)
                        for i in (0, 2)
                    ],
                ),
                pb.CheckModelType.Check(
                    tag="c2",
                    measurements=[
                        pb.CheckModelType.RemoteMeasurement(measurement_index=i)
                        for i in (0,)
                    ],
                ),
                pb.CheckModelType.Check(
                    tag="c3",
                    measurements=[
                        pb.CheckModelType.RemoteMeasurement(measurement_index=i)
                        for i in (2,)
                    ],
                ),
                pb.CheckModelType.Check(
                    tag="c4",
                    measurements=[
                        pb.CheckModelType.RemoteMeasurement(measurement_index=i)
                        for i in (2,)
                    ],
                ),
                pb.CheckModelType.Check(tag="c5"),  # empty check
            ],
        ),
    ],
    error_model_types=[
        pb.ErrorModelType(
            etype=1,
            ctype=1,
            errors=[
                pb.ErrorModelType.Error(
                    probability=0.1,
                    checks=[
                        pb.ErrorModelType.RemoteCheck(check_index=i) for i in (0, 2, 3)
                    ],
                ),
                pb.ErrorModelType.Error(
                    probability=0.1,
                ),
            ],
        ),
    ],
    program=[
        pb.Instruction(gadget=pb.Gadget(gtype=1)),
        pb.Instruction(check_model=pb.CheckModel(ctype=1, gid=1)),
        pb.Instruction(error_model=pb.ErrorModel(etype=1, cid=1)),
    ],
)


def test_physical_success() -> None:

    assert is_valid_and_physical(default_physical_library)

    assert (
        "(LibSpec 1.1) ptype cannot be 0, it's reserved for wildcard"
        in is_valid_and_physical(default_invalid_library)
    )


def test_physical_phy_spec_1_1() -> None:
    assert "(PhySpec 1.1) error triggers empty checks" in is_valid_and_physical(
        pb.Library(
            gadget_types=default_physical_library.gadget_types,
            check_model_types=default_physical_library.check_model_types,
            error_model_types=[
                pb.ErrorModelType(
                    etype=1,
                    ctype=1,
                    errors=[
                        pb.ErrorModelType.Error(
                            probability=0.1,
                            checks=[pb.ErrorModelType.RemoteCheck(check_index=4)],
                        ),
                    ],
                ),
            ],
            program=default_physical_library.program,
        )
    )


def test_physical_phy_spec_1_2() -> None:
    assert (
        "(PhySpec 1.2) error does not have a physical explanation"
        in is_valid_and_physical(
            pb.Library(
                gadget_types=default_physical_library.gadget_types,
                check_model_types=default_physical_library.check_model_types,
                error_model_types=[
                    pb.ErrorModelType(
                        etype=1,
                        ctype=1,
                        errors=[
                            pb.ErrorModelType.Error(
                                probability=0.1,
                                checks=[
                                    pb.ErrorModelType.RemoteCheck(check_index=i)
                                    for i in (0, 2)  # should be (0, 2, 3)
                                ],
                            ),
                        ],
                    ),
                ],
                program=default_physical_library.program,
            )
        )
    )

    assert (
        "(PhySpec 1.2) error does not have a physical explanation"
        in is_valid_and_physical(
            pb.Library(
                gadget_types=default_physical_library.gadget_types,
                check_model_types=default_physical_library.check_model_types,
                error_model_types=[
                    pb.ErrorModelType(
                        etype=1,
                        ctype=1,
                        errors=[
                            pb.ErrorModelType.Error(
                                probability=0.1,
                                checks=[
                                    pb.ErrorModelType.RemoteCheck(check_index=i)
                                    for i in (0, 1, 2, 3)
                                ],
                            ),
                        ],
                    ),
                ],
                program=default_physical_library.program,
            )
        )
    )


def test_physical_phy_spec_2_1() -> None:
    assert (
        "(PhySpec 2.1) empty check must not be naturally flipped"
        in is_valid_and_physical(
            pb.Library(
                gadget_types=default_physical_library.gadget_types,
                check_model_types=[
                    pb.CheckModelType(
                        ctype=1,
                        gtype=1,
                        checks=[
                            *default_physical_library.check_model_types[0].checks[:-1],
                            pb.CheckModelType.Check(tag="c5", naturally_flipped=True),
                        ],
                    )
                ],
                error_model_types=default_physical_library.error_model_types,
                program=default_physical_library.program,
            )
        )
    )


def test_physical_phy_spec_2_2() -> None:
    assert (
        "(PhySpec 2.2) checks involving the same set of measurements"
        in is_valid_and_physical(
            pb.Library(
                gadget_types=default_physical_library.gadget_types,
                check_model_types=[
                    pb.CheckModelType(
                        ctype=1,
                        gtype=1,
                        checks=[
                            *default_physical_library.check_model_types[0].checks[:3],
                            pb.CheckModelType.Check(
                                tag="c4",
                                measurements=[
                                    pb.CheckModelType.RemoteMeasurement(
                                        measurement_index=i
                                    )
                                    for i in (2,)
                                ],
                                naturally_flipped=True,
                            ),
                            pb.CheckModelType.Check(tag="c5"),
                        ],
                    )
                ],
                error_model_types=default_physical_library.error_model_types,
                program=default_physical_library.program,
            )
        )
    )


def test_physical_phy_spec_2_3() -> None:
    assert (
        "(PhySpec 2.3) there exists a loop of checks with odd parity of naturally_flipped"
        in is_valid_and_physical(
            pb.Library(
                gadget_types=default_physical_library.gadget_types,
                check_model_types=[
                    pb.CheckModelType(
                        ctype=1,
                        gtype=1,
                        checks=[
                            pb.CheckModelType.Check(
                                tag="c1",
                                measurements=[
                                    pb.CheckModelType.RemoteMeasurement(
                                        measurement_index=i
                                    )
                                    for i in (0, 1)
                                ],
                                naturally_flipped=True,
                            ),
                            pb.CheckModelType.Check(
                                tag="c2",
                                measurements=[
                                    pb.CheckModelType.RemoteMeasurement(
                                        measurement_index=i
                                    )
                                    for i in (0, 2)
                                ],
                                naturally_flipped=False,
                            ),
                            *(
                                [
                                    pb.CheckModelType.Check(
                                        tag="c3",
                                        measurements=[
                                            pb.CheckModelType.RemoteMeasurement(
                                                measurement_index=i
                                            )
                                            for i in (1, 2)
                                        ],
                                        naturally_flipped=False,
                                    )
                                ]
                                * 2
                            ),
                        ],
                    )
                ],
                error_model_types=default_physical_library.error_model_types,
                program=default_physical_library.program,
            )
        )
    )
