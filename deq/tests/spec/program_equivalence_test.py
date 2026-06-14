import deq.proto.deq_bin_pb2 as pb
from deq.spec.program_equivalence import are_programs_equivalent
from tests.spec.physical_validator_test import default_physical_library
from tests.spec.library_validator_test import default_invalid_library

# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint


def test_program_equivalence_success() -> None:
    assert are_programs_equivalent(default_physical_library, default_physical_library)


def test_program_equivalence_prog_eq_1() -> None:
    assert "(ProgEq 1) lib1 is not valid/physical" in are_programs_equivalent(
        default_invalid_library, default_physical_library
    )

    assert "(ProgEq 1) lib2 is not valid/physical" in are_programs_equivalent(
        default_physical_library, default_invalid_library
    )


def test_program_equivalence_prog_eq_basic() -> None:
    assert "(ProgEq 2.2) number of measurements differ" in are_programs_equivalent(
        default_physical_library, pb.Library()
    )


def test_program_equivalence_remove_empty_check_is_fine() -> None:
    assert are_programs_equivalent(
        default_physical_library,
        pb.Library(
            gadget_types=default_physical_library.gadget_types,
            check_model_types=[
                pb.CheckModelType(
                    ctype=1,
                    gtype=1,
                    checks=default_physical_library.check_model_types[0].checks[:4],
                ),
            ],
            error_model_types=default_physical_library.error_model_types,
            program=default_physical_library.program,
        ),
    )


def test_program_equivalence_remove_duplicate_check_is_fine() -> None:
    # since check[1] = {0} is a linear combination of check[0] = {0, 2} and check[2] = {2},
    # removing it does not change the decoding hypergraph
    assert are_programs_equivalent(
        default_physical_library,
        pb.Library(
            gadget_types=default_physical_library.gadget_types,
            check_model_types=[
                pb.CheckModelType(
                    ctype=1,
                    gtype=1,
                    checks=[
                        default_physical_library.check_model_types[0].checks[0],
                        default_physical_library.check_model_types[0].checks[2],
                        default_physical_library.check_model_types[0].checks[3],
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
                                pb.ErrorModelType.RemoteCheck(check_index=i)
                                for i in (0, 1, 2)
                            ],
                        ),
                        pb.ErrorModelType.Error(probability=0.1),
                    ],
                ),
            ],
            program=default_physical_library.program,
        ),
    )


def test_program_equivalence_use_equivalent_set_of_checks() -> None:
    # the error triggering checks {0,2} and {2} can be explained by triggering
    # a single measurement 2, and thus we can redesign the checks to be {2} and {0}
    assert are_programs_equivalent(
        default_physical_library,
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
                                pb.CheckModelType.RemoteMeasurement(measurement_index=0)
                            ],
                        ),
                        pb.CheckModelType.Check(
                            tag="c2",
                            measurements=[
                                pb.CheckModelType.RemoteMeasurement(measurement_index=2)
                            ],
                        ),
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
                            checks=[pb.ErrorModelType.RemoteCheck(check_index=1)],
                        ),
                        pb.ErrorModelType.Error(probability=0.1),
                    ],
                ),
            ],
            program=default_physical_library.program,
        ),
    )


def test_program_equivalence_add_new_check_fails() -> None:
    # adding a linearly independent check should fail equivalence
    assert (
        "(ProgEq 3) check is not a linear combination of other checks"
        in are_programs_equivalent(
            default_physical_library,
            pb.Library(
                gadget_types=default_physical_library.gadget_types,
                check_model_types=[
                    pb.CheckModelType(
                        ctype=1,
                        gtype=1,
                        checks=[
                            *default_physical_library.check_model_types[0].checks,
                            pb.CheckModelType.Check(
                                tag="c6",
                                measurements=[
                                    pb.CheckModelType.RemoteMeasurement(
                                        measurement_index=i
                                    )
                                    for i in (1,)
                                ],
                            ),
                        ],
                    ),
                ],
                error_model_types=default_physical_library.error_model_types,
                program=default_physical_library.program,
            ),
        )
    )


def test_program_equivalence_add_new_error_fails() -> None:
    assert (
        "(ProgEq 4.2) the gathered probabilities differ for unique error"
        in are_programs_equivalent(
            default_physical_library,
            pb.Library(
                gadget_types=default_physical_library.gadget_types,
                check_model_types=default_physical_library.check_model_types,
                error_model_types=[
                    pb.ErrorModelType(
                        etype=1,
                        ctype=1,
                        errors=[
                            pb.ErrorModelType.Error(
                                probability=0.10,
                                checks=[
                                    pb.ErrorModelType.RemoteCheck(check_index=i)
                                    for i in (0, 2, 3)
                                ],
                            ),
                            pb.ErrorModelType.Error(
                                probability=0.02,
                                checks=[
                                    pb.ErrorModelType.RemoteCheck(check_index=i)
                                    for i in (0, 2, 3)
                                ],
                            ),
                            pb.ErrorModelType.Error(probability=0.1),
                        ],
                    ),
                ],
                program=default_physical_library.program,
            ),
        )
    )
