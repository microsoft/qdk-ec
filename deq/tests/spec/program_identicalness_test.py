import deq.proto.deq_bin_pb2 as pb
import deq.proto.util_pb2 as util_pb
from deq.spec.program_validator import is_valid
from deq.spec.program_identicalness import are_programs_identical
from tests.spec.library_validator_test import default_library, default_invalid_library
from tests.spec.canonical_test import default_library_canonical

# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint


def test_program_identicalness_success() -> None:
    assert are_programs_identical(default_library, default_library)

    bijection = are_programs_identical(default_library, default_library_canonical)
    assert bijection
    assert "anything" not in bijection


def test_program_identicalness_prog_id_1() -> None:
    assert "(ProgId 1) lib1 is not valid" in are_programs_identical(
        default_invalid_library, default_library
    )

    assert "(ProgId 1) lib2 is not valid" in are_programs_identical(
        default_library, default_invalid_library
    )


library_with_observables = pb.Library(
    port_types=[pb.PortType(ptype=1, observables=[pb.PortType.Observable()])],
    gadget_types=[
        pb.GadgetType(
            gtype=1,
            measurements=[pb.GadgetType.Measurement()] * 7,
            outputs=[pb.GadgetType.Port(ptype=1)],
            readouts=[pb.GadgetType.Readout()],
            readout_propagation=util_pb.BitMatrix(rows=1, cols=1),
            correction_propagation=util_pb.BitMatrix(rows=1, cols=1),
            logical_correction=util_pb.BitMatrix(rows=1, cols=1),
            physical_correction=util_pb.BitMatrix(rows=1, cols=7),
        )
    ],
    check_model_types=default_library_canonical.check_model_types,
    error_model_types=default_library_canonical.error_model_types,
    program=default_library_canonical.program,
)


def test_program_identicalness_prog_id_2_1() -> None:
    assert is_valid(library_with_observables)

    assert "(ProgId 2.1) number of observables differ" in are_programs_identical(
        default_library, library_with_observables
    )


def test_program_identicalness_prog_id_2_2() -> None:
    assert "(ProgId 2.2) number of measurements differ" in are_programs_identical(
        default_library,
        pb.Library(
            port_types=default_library_canonical.port_types,
            gadget_types=[
                pb.GadgetType(
                    gtype=1,
                    measurements=[pb.GadgetType.Measurement()] * 100,  # differ
                    outputs=[pb.GadgetType.Port(ptype=1)],
                    readouts=[pb.GadgetType.Readout()],
                    readout_propagation=util_pb.BitMatrix(rows=1, cols=1),
                    physical_correction=util_pb.BitMatrix(rows=0, cols=100),
                )
            ],
            check_model_types=default_library_canonical.check_model_types,
            error_model_types=default_library_canonical.error_model_types,
            program=default_library_canonical.program,
        ),
    )


def test_program_identicalness_prog_id_2_3_and_2_5_and_2_6() -> None:
    # ProgId 2.6 (logical_correction differs) is no longer reachable after
    # the merge() absorption pass (canonical.py step 9) — the merged
    # ``logical_correction`` is always empty by design, so two canonical
    # forms will always agree on it.  We still assert that
    # ``correction_propagation`` (ProgId 2.3) and
    # ``readout_propagation`` (ProgId 2.5) differences are reported.
    # To trigger ProgId 2.3 we set ``cp[0, 0] = 1`` (affine flip) without
    # any compensating ``lc`` that would absorb it back to empty.
    assert (
        "(ProgId 2.3) static correction propagation differ",
        "(ProgId 2.5) static readout propagation differ",
    ) in are_programs_identical(
        library_with_observables,
        pb.Library(
            port_types=library_with_observables.port_types,
            gadget_types=[
                pb.GadgetType(
                    gtype=1,
                    measurements=[pb.GadgetType.Measurement()] * 7,
                    outputs=[pb.GadgetType.Port(ptype=1)],
                    readouts=[pb.GadgetType.Readout()],
                    readout_propagation=util_pb.BitMatrix(rows=1, cols=1, i=[0], j=[0]),
                    correction_propagation=util_pb.BitMatrix(
                        rows=1, cols=1, i=[0], j=[0]
                    ),
                    logical_correction=util_pb.BitMatrix(rows=1, cols=1),
                    physical_correction=util_pb.BitMatrix(rows=1, cols=7),
                )
            ],
            check_model_types=default_library_canonical.check_model_types,
            error_model_types=default_library_canonical.error_model_types,
            program=default_library_canonical.program,
        ),
    )


def test_program_identicalness_prog_id_2_4() -> None:
    assert "(ProgId 2.4) number of readouts differ" in are_programs_identical(
        library_with_observables,
        pb.Library(
            port_types=library_with_observables.port_types,
            gadget_types=[
                pb.GadgetType(
                    gtype=1,
                    measurements=[pb.GadgetType.Measurement()] * 7,
                    outputs=[pb.GadgetType.Port(ptype=1)],
                    readouts=[pb.GadgetType.Readout()] * 2,  # differ
                    readout_propagation=util_pb.BitMatrix(rows=2, cols=1),
                    correction_propagation=util_pb.BitMatrix(rows=1, cols=1),
                    logical_correction=util_pb.BitMatrix(rows=1, cols=2),
                    physical_correction=util_pb.BitMatrix(rows=1, cols=7),
                )
            ],
            check_model_types=default_library_canonical.check_model_types,
            error_model_types=default_library_canonical.error_model_types,
            program=default_library_canonical.program,
        ),
    )


def test_program_identicalness_prog_id_2_7() -> None:
    assert "(ProgId 2.8) number of checks differ" in are_programs_identical(
        default_library,
        pb.Library(
            port_types=default_library_canonical.port_types,
            gadget_types=default_library_canonical.gadget_types,
            check_model_types=[
                pb.CheckModelType(
                    ctype=1,
                    gtype=1,
                    checks=default_library_canonical.check_model_types[0].checks[:-1],
                )
            ],
            error_model_types=default_library_canonical.error_model_types,
            program=default_library_canonical.program,
        ),
    )


def test_program_identicalness_prog_id_3_1() -> None:
    assert "(ProgId 3.1) the set of unique checks differ" in are_programs_identical(
        default_library,
        pb.Library(
            port_types=default_library_canonical.port_types,
            gadget_types=default_library_canonical.gadget_types,
            check_model_types=[
                pb.CheckModelType(
                    ctype=1,
                    gtype=1,
                    checks=[
                        *default_library_canonical.check_model_types[0].checks[:-1],
                        pb.CheckModelType.Check(
                            measurements=[
                                pb.CheckModelType.RemoteMeasurement(measurement_index=i)
                                for i in (4, 5)
                            ]
                        ),
                    ],
                )
            ],
            error_model_types=default_library_canonical.error_model_types,
            program=default_library_canonical.program,
        ),
    )


def test_program_identicalness_prog_id_3_2() -> None:
    assert (
        "(ProgId 3.2) number of checks for unique check differ"
        in are_programs_identical(
            default_library,
            pb.Library(
                port_types=default_library_canonical.port_types,
                gadget_types=default_library_canonical.gadget_types,
                check_model_types=[
                    pb.CheckModelType(
                        ctype=1,
                        gtype=1,
                        checks=[
                            *default_library_canonical.check_model_types[0].checks[:-1],
                            pb.CheckModelType.Check(
                                measurements=[
                                    pb.CheckModelType.RemoteMeasurement(
                                        measurement_index=i
                                    )
                                    for i in (4, 5, 6, 2)
                                ]
                            ),
                        ],
                    )
                ],
                error_model_types=default_library_canonical.error_model_types,
                program=default_library_canonical.program,
            ),
        )
    )


def test_program_identicalness_prog_id_3_3() -> None:
    assert (
        "(ProgId 3.3) in lib2, not all checks for unique check share the same naturally_flipped"
        in are_programs_identical(
            default_library,
            pb.Library(
                port_types=default_library_canonical.port_types,
                gadget_types=default_library_canonical.gadget_types,
                check_model_types=[
                    pb.CheckModelType(
                        ctype=1,
                        gtype=1,
                        checks=[
                            *default_library_canonical.check_model_types[0].checks[:-1],
                            pb.CheckModelType.Check(naturally_flipped=True),
                        ],
                    )
                ],
                error_model_types=default_library_canonical.error_model_types,
                program=default_library_canonical.program,
            ),
        )
    )

    assert (
        "(ProgId 3.3) naturally_flipped differ for unique check"
        in are_programs_identical(
            default_library,
            pb.Library(
                port_types=default_library_canonical.port_types,
                gadget_types=default_library_canonical.gadget_types,
                check_model_types=[
                    pb.CheckModelType(
                        ctype=1,
                        gtype=1,
                        checks=[
                            *default_library_canonical.check_model_types[0].checks[:3],
                            pb.CheckModelType.Check(
                                measurements=[
                                    pb.CheckModelType.RemoteMeasurement(
                                        measurement_index=i
                                    )
                                    for i in (4, 5, 6, 2)
                                ],
                                naturally_flipped=True,
                            ),
                            *default_library_canonical.check_model_types[0].checks[4:],
                        ],
                    )
                ],
                error_model_types=default_library_canonical.error_model_types,
                program=default_library_canonical.program,
            ),
        )
    )


def test_program_identicalness_prog_id_4_1() -> None:
    assert "(ProgId 4.1) the set of unique errors differ" in are_programs_identical(
        default_library,
        pb.Library(
            port_types=default_library_canonical.port_types,
            gadget_types=default_library_canonical.gadget_types,
            check_model_types=default_library_canonical.check_model_types,
            error_model_types=[
                pb.ErrorModelType(
                    etype=1,
                    ctype=1,
                    errors=[
                        pb.ErrorModelType.Error(
                            probability=0.1,
                            checks=[  # c1, c2, c5
                                pb.ErrorModelType.RemoteCheck(check_index=i)
                                for i in (0, 1, 4)
                            ],
                        ),
                        pb.ErrorModelType.Error(
                            probability=0.1,
                            readout_flips=[0],
                        ),
                    ],
                )
            ],
            program=default_library_canonical.program,
        ),
    )


def test_program_identicalness_prog_id_4_2() -> None:
    assert (
        "(ProgId 4.2) the gathered probabilities differ for unique error"
        in are_programs_identical(
            default_library,
            pb.Library(
                port_types=default_library_canonical.port_types,
                gadget_types=default_library_canonical.gadget_types,
                check_model_types=default_library_canonical.check_model_types,
                error_model_types=[
                    pb.ErrorModelType(
                        etype=1,
                        ctype=1,
                        errors=[
                            pb.ErrorModelType.Error(
                                probability=0.1,
                                readout_flips=[0],
                                checks=[  # c1, c2, c5
                                    pb.ErrorModelType.RemoteCheck(check_index=i)
                                    for i in (0, 1, 4)
                                ],
                            ),
                            pb.ErrorModelType.Error(
                                probability=0.2,
                                readout_flips=[0],
                            ),
                        ],
                    )
                ],
                program=default_library_canonical.program,
            ),
        )
    )
