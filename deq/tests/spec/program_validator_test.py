import deq.proto.deq_bin_pb2 as pb
import deq.proto.util_pb2 as util_pb
from deq.spec.program_validator import (
    is_valid,
    ExpandedProgram,
    _placeholder_remote_gadget,
    _placeholder_remote_check_model,
)
from tests.spec.library_validator_test import default_library

# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint


def test_default_library_program_validity() -> None:
    program = is_valid(default_library)
    assert program
    assert "anything" not in program


common = {
    "port_types": default_library.port_types,
    "gadget_types": default_library.gadget_types,
    "check_model_types": default_library.check_model_types,
    "error_model_types": default_library.error_model_types,
}


def test_program_validity_1_1_to_1_4() -> None:
    assert (
        "(ProgSpec 1.1) undefined gadget type",
        "(ProgSpec 1.2) undefined check model type",
        "(ProgSpec 1.3) undefined error model type",
        "(ProgSpec 1.4) unsupported instruction type",
    ) in is_valid(
        pb.Library(
            **common,
            program=[
                pb.Instruction(gadget=pb.Gadget(gtype=100)),
                pb.Instruction(check_model=pb.CheckModel(ctype=100)),
                pb.Instruction(error_model=pb.ErrorModel(etype=100)),
                pb.Instruction(),
            ],
        )
    )


def test_program_validity_3_1_and_4_1() -> None:
    assert (
        "(ProgSpec 3.1) check model instance cid=1 binds to undefined gadget",
        "(ProgSpec 4.1) error model instance eid=1 attaches to undefined check model",
    ) in is_valid(
        pb.Library(
            **common,
            program=[
                pb.Instruction(gadget=pb.Gadget(gtype=1)),
                pb.Instruction(check_model=pb.CheckModel(ctype=1, gid=100)),
                pb.Instruction(error_model=pb.ErrorModel(etype=1, cid=100)),
            ],
        )
    )


def test_program_validity_2_1() -> None:
    assert ("(ProgSpec 2.1) number of connectors is wrong",) in is_valid(
        pb.Library(
            **common,
            program=[pb.Instruction(gadget=pb.Gadget(gtype=2))],
        )
    )


def test_program_validity_2_2_1() -> None:
    assert ("(ProgSpec 2.2.1)", "connects to undefined gadget") in is_valid(
        pb.Library(
            **common,
            program=[
                pb.Instruction(
                    gadget=pb.Gadget(gtype=2, connectors=[pb.Gadget.Connector(gid=100)])
                )
            ],
        )
    )


def test_program_validity_2_2_2_and_2_2_3() -> None:
    assert (
        "(ProgSpec 2.2.2)",
        "with overflowed output port",
        "(ProgSpec 2.2.3)",
        "is incompatible with the output port",
    ) in is_valid(
        pb.Library(
            **common,
            program=[
                pb.Instruction(gadget=pb.Gadget(gtype=1)),
                pb.Instruction(
                    gadget=pb.Gadget(
                        gtype=3, connectors=[pb.Gadget.Connector(gid=1, port=100)]
                    ),
                ),
                pb.Instruction(
                    gadget=pb.Gadget(gtype=3, connectors=[pb.Gadget.Connector(gid=1)]),
                ),
            ],
        )
    )


def test_program_validity_2_2_4() -> None:
    assert (
        "(ProgSpec 2.2.4)",
        "which is already connected",
    ) in is_valid(
        pb.Library(
            **common,
            program=[
                pb.Instruction(gadget=pb.Gadget(gtype=1)),
                pb.Instruction(
                    gadget=pb.Gadget(gtype=2, connectors=[pb.Gadget.Connector(gid=1)]),
                ),
                pb.Instruction(
                    gadget=pb.Gadget(gtype=2, connectors=[pb.Gadget.Connector(gid=1)]),
                ),
            ],
        )
    )


def test_program_validity_2_3_1_gadget_modifier_toggle_wrong_dimensions() -> None:
    assert (
        "(ProgSpec 2.3.1)",
        "toggle for correction_propagation has wrong dimensions",
    ) in is_valid(
        pb.Library(
            **common,
            program=[
                pb.Instruction(gadget=pb.Gadget(gtype=1)),
                pb.Instruction(
                    gadget=pb.Gadget(
                        gtype=2,
                        connectors=[pb.Gadget.Connector(gid=1)],
                        modifier=pb.GadgetModifier(
                            correction_propagation_mod=pb.BitMatrixModifier(
                                toggle=util_pb.BitMatrix(rows=10, cols=10),
                            ),
                        ),
                    )
                ),
            ],
        )
    )


def test_program_validity_2_3_2_gadget_modifier_overwrite_wrong_dimensions() -> None:
    assert (
        "(ProgSpec 2.3.2)",
        "overwrite for readout_propagation has wrong dimensions",
    ) in is_valid(
        pb.Library(
            **common,
            program=[
                pb.Instruction(gadget=pb.Gadget(gtype=1)),
                pb.Instruction(
                    gadget=pb.Gadget(
                        gtype=2,
                        connectors=[pb.Gadget.Connector(gid=1)],
                        modifier=pb.GadgetModifier(
                            readout_propagation_mod=pb.BitMatrixModifier(
                                overwrite=util_pb.BitMatrix(rows=999, cols=999),
                            ),
                        ),
                    )
                ),
            ],
        )
    )


def test_program_validity_gadget_modifier_valid() -> None:
    simple_lib = pb.Library(
        port_types=[
            pb.PortType(ptype=1, observables=[pb.PortType.Observable()]),
        ],
        gadget_types=[
            pb.GadgetType(
                gtype=1,
                measurements=[pb.GadgetType.Measurement()],
                outputs=[pb.GadgetType.Port(ptype=1)],
                correction_propagation=util_pb.BitMatrix(rows=1, cols=1),
                physical_correction=util_pb.BitMatrix(rows=1, cols=1),
            ),
            pb.GadgetType(
                gtype=2,
                measurements=[pb.GadgetType.Measurement()],
                inputs=[pb.GadgetType.Port(ptype=1)],
                outputs=[pb.GadgetType.Port(ptype=1)],
                correction_propagation=util_pb.BitMatrix(rows=1, cols=2, i=[0], j=[0]),
                physical_correction=util_pb.BitMatrix(rows=1, cols=1),
            ),
        ],
        check_model_types=[pb.CheckModelType(ctype=1, checks=[])],
        error_model_types=[pb.ErrorModelType(etype=1, errors=[])],
        program=[
            pb.Instruction(gadget=pb.Gadget(gtype=1)),
            pb.Instruction(
                gadget=pb.Gadget(
                    gtype=2,
                    connectors=[pb.Gadget.Connector(gid=1)],
                    modifier=pb.GadgetModifier(
                        correction_propagation_mod=pb.BitMatrixModifier(
                            toggle=util_pb.BitMatrix(rows=1, cols=2, i=[0], j=[1]),
                        ),
                    ),
                )
            ),
            pb.Instruction(check_model=pb.CheckModel(ctype=1, gid=2)),
            pb.Instruction(error_model=pb.ErrorModel(etype=1, cid=1)),
        ],
    )
    result = is_valid(simple_lib)
    assert isinstance(result, ExpandedProgram)
    assert 2 in result.modified_gadget_types
    modified = result.modified_gadget_types[2]
    assert modified.correction_propagation.rows == 1
    assert modified.correction_propagation.cols == 2
    assert set(
        zip(modified.correction_propagation.i, modified.correction_propagation.j)
    ) == {
        (0, 0),
        (0, 1),
    }


gadgets = [
    pb.Instruction(gadget=pb.Gadget(gtype=1)),
    pb.Instruction(
        gadget=pb.Gadget(gtype=2, connectors=[pb.Gadget.Connector(gid=1)]),
    ),
    pb.Instruction(
        gadget=pb.Gadget(gtype=3, connectors=[pb.Gadget.Connector(gid=2)]),
    ),
]


def test_program_validity_3_3_1_and_3_3_2() -> None:
    check_model_1 = default_library.check_model_types[0]
    assert (
        "(ProgSpec 3.3.1)",
        "modifier reroutes remote gadget",
        "multiple times",
        "(ProgSpec 3.3.2)",
        "remote gadget index too large",
    ) in is_valid(
        pb.Library(
            **common,
            program=[
                *gadgets,
                pb.Instruction(
                    check_model=pb.CheckModel(
                        ctype=1,
                        gid=2,
                        modifier=pb.CheckModel.CheckModelModifier(
                            reroute_remote_gadgets=[
                                pb.CheckModel.CheckModelModifier.RerouteRemoteGadget(
                                    remote_gadget_index=0,
                                    value=check_model_1.remote_gadgets[0],
                                ),
                                pb.CheckModel.CheckModelModifier.RerouteRemoteGadget(
                                    remote_gadget_index=0,
                                    value=check_model_1.remote_gadgets[0],
                                ),
                                pb.CheckModel.CheckModelModifier.RerouteRemoteGadget(
                                    remote_gadget_index=5,  # this is fine
                                    value=check_model_1.remote_gadgets[0],
                                ),
                                pb.CheckModel.CheckModelModifier.RerouteRemoteGadget(
                                    remote_gadget_index=123456789,  # too large
                                    value=check_model_1.remote_gadgets[0],
                                ),
                            ]
                        ),
                    )
                ),
            ],
        )
    )


def test_program_validity_3_4_and_3_3_5() -> None:
    assert (
        "(ProgSpec 3.4) the modified check model type is invalid",
        "(LibSpec 3.3.5)",
        "contain a loop",
    ) in is_valid(
        pb.Library(
            **common,
            program=[
                *gadgets,
                pb.Instruction(
                    check_model=pb.CheckModel(
                        ctype=1,
                        gid=2,
                        modifier=pb.CheckModel.CheckModelModifier(
                            reroute_remote_gadgets=[
                                pb.CheckModel.CheckModelModifier.RerouteRemoteGadget(
                                    remote_gadget_index=0,
                                    value=pb.CheckModelType.RemoteGadget(
                                        output=0,
                                        previous_remote_gadget=0,  # loop to itself
                                        expecting_gtype=2,
                                    ),
                                ),
                            ]
                        ),
                    )
                ),
            ],
        )
    )


def test_earlier_gadget_already_resolved() -> None:
    # construct a case where expanding an earlier remote gadget already resolves
    # a later one, so that the checker doesn't have to expand the later one again
    assert is_valid(
        pb.Library(
            **common,
            program=[
                *gadgets,
                pb.Instruction(
                    check_model=pb.CheckModel(
                        ctype=1,
                        gid=2,
                        modifier=pb.CheckModel.CheckModelModifier(
                            reroute_remote_gadgets=[
                                pb.CheckModel.CheckModelModifier.RerouteRemoteGadget(
                                    remote_gadget_index=0,
                                    value=pb.CheckModelType.RemoteGadget(
                                        input=0,
                                        previous_remote_gadget=1,  # refer to later one
                                        expecting_gtype=2,
                                    ),
                                ),
                                pb.CheckModel.CheckModelModifier.RerouteRemoteGadget(
                                    remote_gadget_index=1,
                                    value=pb.CheckModelType.RemoteGadget(
                                        output=0,
                                        expecting_gtype=3,
                                    ),
                                ),
                            ]
                        ),
                    )
                ),
            ],
        )
    )


def test_program_validity_3_5_1() -> None:
    assert (
        "(ProgSpec 3.5.1)",
        " cannot be expanded (possibly referring to a placeholder)",
        "(ProgSpec 3.5.4)",
        "expects gtype 2, but got 3",
    ) in is_valid(
        pb.Library(
            **common,
            program=[
                *gadgets,
                pb.Instruction(
                    check_model=pb.CheckModel(
                        ctype=1,
                        gid=2,
                        modifier=pb.CheckModel.CheckModelModifier(
                            reroute_remote_gadgets=[
                                pb.CheckModel.CheckModelModifier.RerouteRemoteGadget(
                                    remote_gadget_index=0,
                                    value=pb.CheckModelType.RemoteGadget(
                                        input=0,
                                        previous_remote_gadget=3,  # refer to a non-existing one
                                        expecting_gtype=2,
                                    ),
                                ),
                                pb.CheckModel.CheckModelModifier.RerouteRemoteGadget(
                                    remote_gadget_index=4,
                                    value=pb.CheckModelType.RemoteGadget(
                                        output=0, expecting_gtype=2  # should be 3
                                    ),
                                ),
                            ]
                        ),
                    )
                ),
            ],
        )
    )


def test_program_validity_3_5_2_and_3_5_2() -> None:
    assert (
        "(ProgSpec 3.5.2) invalid output port index 0 in remote gadget ri=1 of check model (cid=1)",
        "(ProgSpec 3.5.2) invalid input port index 10 in remote gadget ri=2 of check model (cid=1)",
    ) in is_valid(
        pb.Library(
            **common,
            program=[
                *gadgets,
                pb.Instruction(
                    check_model=pb.CheckModel(
                        ctype=1,
                        gid=2,
                        modifier=pb.CheckModel.CheckModelModifier(
                            reroute_remote_gadgets=[
                                pb.CheckModel.CheckModelModifier.RerouteRemoteGadget(
                                    remote_gadget_index=0,
                                    value=pb.CheckModelType.RemoteGadget(
                                        input=0,
                                        expecting_gtype=0,  # use wildcard to confuse static checker
                                    ),
                                ),
                                pb.CheckModel.CheckModelModifier.RerouteRemoteGadget(
                                    remote_gadget_index=1,
                                    value=pb.CheckModelType.RemoteGadget(
                                        output=100, previous_remote_gadget=0
                                    ),
                                ),
                                pb.CheckModel.CheckModelModifier.RerouteRemoteGadget(
                                    remote_gadget_index=2,
                                    value=pb.CheckModelType.RemoteGadget(
                                        input=10, previous_remote_gadget=0
                                    ),
                                ),
                            ]
                        ),
                    )
                ),
            ],
        )
    )


def test_program_validity_3_5_3() -> None:
    assert (
        "(ProgSpec 3.5.3) remote gadget ri=0 output port 0 of check model (cid=1) is not connected",
    ) in is_valid(
        pb.Library(
            **common,
            program=[
                pb.Instruction(gadget=pb.Gadget(gtype=1)),
                pb.Instruction(
                    gadget=pb.Gadget(gtype=2, connectors=[pb.Gadget.Connector(gid=1)]),
                ),
                pb.Instruction(
                    check_model=pb.CheckModel(
                        ctype=1,
                        gid=2,
                        modifier=pb.CheckModel.CheckModelModifier(
                            reroute_remote_gadgets=[
                                pb.CheckModel.CheckModelModifier.RerouteRemoteGadget(
                                    remote_gadget_index=0,
                                    value=pb.CheckModelType.RemoteGadget(
                                        output=0, measurement_bias=1
                                    ),
                                ),
                            ]
                        ),
                    )
                ),
            ],
        )
    )


def test_program_validity_3_5_5() -> None:
    assert "(ProgSpec 3.5.5) invalid measurement bias" in is_valid(
        pb.Library(
            **common,
            program=[
                *gadgets,
                pb.Instruction(
                    check_model=pb.CheckModel(
                        ctype=1,
                        gid=2,
                        modifier=pb.CheckModel.CheckModelModifier(
                            reroute_remote_gadgets=[
                                pb.CheckModel.CheckModelModifier.RerouteRemoteGadget(
                                    remote_gadget_index=1,
                                    value=_placeholder_remote_gadget(),
                                ),
                                pb.CheckModel.CheckModelModifier.RerouteRemoteGadget(
                                    remote_gadget_index=2,
                                    value=pb.CheckModelType.RemoteGadget(
                                        output=0, measurement_bias=100
                                    ),
                                ),
                            ]
                        ),
                    )
                ),
            ],
        )
    )


def test_program_validity_3_6_1_and_3_6_2() -> None:
    assert (
        "(ProgSpec 3.6.2) overflowed remote measurement index",
        "(ProgSpec 3.6.1) remote gadget modified to a placeholder",
    ) in is_valid(
        pb.Library(
            **common,
            program=[
                *gadgets,
                pb.Instruction(
                    check_model=pb.CheckModel(
                        ctype=1,
                        gid=2,
                        modifier=pb.CheckModel.CheckModelModifier(
                            reroute_remote_gadgets=[
                                pb.CheckModel.CheckModelModifier.RerouteRemoteGadget(
                                    remote_gadget_index=1,
                                    value=_placeholder_remote_gadget(),
                                ),
                                pb.CheckModel.CheckModelModifier.RerouteRemoteGadget(
                                    remote_gadget_index=2,
                                    value=pb.CheckModelType.RemoteGadget(
                                        output=0, measurement_bias=2
                                    ),
                                ),
                            ]
                        ),
                    )
                ),
            ],
        )
    )


def test_program_validity_3_2() -> None:
    assert (
        "(ProgSpec 3.2) check model (ctype=1) expects gadget type 2",
        "but binds to gadget instance gid=3 with gtype=3",
    ) in is_valid(
        pb.Library(
            **common,
            program=[
                *gadgets,
                pb.Instruction(
                    check_model=pb.CheckModel(
                        ctype=1,
                        gid=3,
                    )
                ),
            ],
        )
    )


check_models = [
    pb.Instruction(check_model=pb.CheckModel(ctype=1, gid=2)),
    pb.Instruction(check_model=pb.CheckModel(ctype=2, gid=3)),
]


def test_program_validity_4_2() -> None:
    assert (
        "(ProgSpec 4.2) error model (etype=1) expects check model type 1",
        "but attaches to check model instance cid=2 with ctype=2",
    ) in is_valid(
        pb.Library(
            **common,
            program=[
                *gadgets,
                *check_models,
                pb.Instruction(
                    error_model=pb.ErrorModel(
                        etype=1,
                        cid=2,
                    )
                ),
            ],
        )
    )


def test_program_validity_4_3_1() -> None:
    assert (
        "(ProgSpec 4.3.1) error model instance 1 modifier specifies 100 new probabilities"
        in is_valid(
            pb.Library(
                **common,
                program=[
                    *gadgets,
                    *check_models,
                    pb.Instruction(
                        error_model=pb.ErrorModel(
                            etype=1,
                            cid=1,
                            modifier=pb.ErrorModel.ErrorModelModifier(
                                probability_modifier=pb.ProbabilityModifier(
                                    probabilities=[0.1] * 100,
                                )
                            ),
                        )
                    ),
                ],
            )
        )
    )

    assert "(ProgSpec 4.3.1) invalid modifier probability not in [0, 1]" in is_valid(
        pb.Library(
            **common,
            program=[
                *gadgets,
                *check_models,
                pb.Instruction(
                    error_model=pb.ErrorModel(
                        etype=1,
                        cid=1,
                        modifier=pb.ErrorModel.ErrorModelModifier(
                            probability_modifier=pb.ProbabilityModifier(
                                probabilities=[1.1],
                            )
                        ),
                    )
                ),
            ],
        )
    )

    assert is_valid(
        pb.Library(
            **common,
            program=[
                *gadgets,
                *check_models,
                pb.Instruction(
                    error_model=pb.ErrorModel(
                        etype=1,
                        cid=1,
                        modifier=pb.ErrorModel.ErrorModelModifier(
                            probability_modifier=pb.ProbabilityModifier(
                                probabilities=[0.5],
                            )
                        ),
                    )
                ),
            ],
        )
    )


def test_program_validity_4_3_2_and_4_3_3() -> None:
    assert (
        "(ProgSpec 4.3.2) error model instance 1 modifier specifies unpaired",
        "(ProgSpec 4.3.3) error model instance 1 modifier has overflowed sparse index",
        "(ProgSpec 4.3.3) error model instance 1 modifier has duplicate sparse indices",
    ) in is_valid(
        pb.Library(
            **common,
            program=[
                *gadgets,
                *check_models,
                pb.Instruction(
                    error_model=pb.ErrorModel(
                        etype=1,
                        cid=1,
                        modifier=pb.ErrorModel.ErrorModelModifier(
                            probability_modifier=pb.ProbabilityModifier(
                                sparse_indices=[0, 0, 2, 4],
                                sparse_probabilities=[0.1, 0.2],
                            )
                        ),
                    )
                ),
            ],
        )
    )

    assert "(ProgSpec 4.3.3) invalid modifier probability not in [0, 1]" in is_valid(
        pb.Library(
            **common,
            program=[
                *gadgets,
                *check_models,
                pb.Instruction(
                    error_model=pb.ErrorModel(
                        etype=1,
                        cid=1,
                        modifier=pb.ErrorModel.ErrorModelModifier(
                            probability_modifier=pb.ProbabilityModifier(
                                sparse_indices=[0],
                                sparse_probabilities=[1.1],
                            )
                        ),
                    )
                ),
            ],
        )
    )


def test_program_validity_4_3_4_and_4_3_5() -> None:
    error_model_1 = default_library.error_model_types[0]
    assert (
        "(ProgSpec 4.3.4)",
        "modifier reroutes remote check model",
        "multiple times",
        "(ProgSpec 4.3.5)",
        "remote check model index too large",
    ) in is_valid(
        pb.Library(
            **common,
            program=[
                *gadgets,
                *check_models,
                pb.Instruction(
                    error_model=pb.ErrorModel(
                        etype=1,
                        cid=1,
                        modifier=pb.ErrorModel.ErrorModelModifier(
                            reroute_remote_check_models=[
                                pb.ErrorModel.ErrorModelModifier.RerouteRemoteCheckModel(
                                    remote_check_model_index=0,
                                    value=error_model_1.remote_check_models[0],
                                ),
                                pb.ErrorModel.ErrorModelModifier.RerouteRemoteCheckModel(
                                    remote_check_model_index=0,
                                    value=error_model_1.remote_check_models[0],
                                ),
                                pb.ErrorModel.ErrorModelModifier.RerouteRemoteCheckModel(
                                    remote_check_model_index=5,  # this is fine
                                    value=error_model_1.remote_check_models[0],
                                ),
                                pb.ErrorModel.ErrorModelModifier.RerouteRemoteCheckModel(
                                    remote_check_model_index=123456789,  # too large
                                    value=error_model_1.remote_check_models[0],
                                ),
                            ]
                        ),
                    )
                ),
            ],
        )
    )


def test_program_validity_4_4_and_4_3_5() -> None:
    assert (
        "(ProgSpec 4.4) the modified error model type is invalid",
        "(LibSpec 4.3.5)",
        "contain a loop",
    ) in is_valid(
        pb.Library(
            **common,
            program=[
                *gadgets,
                *check_models,
                pb.Instruction(
                    error_model=pb.ErrorModel(
                        etype=1,
                        cid=1,
                        modifier=pb.ErrorModel.ErrorModelModifier(
                            reroute_remote_check_models=[
                                pb.ErrorModel.ErrorModelModifier.RerouteRemoteCheckModel(
                                    remote_check_model_index=0,
                                    value=pb.ErrorModelType.RemoteCheckModel(
                                        output=0,
                                        previous_remote_check_model=0,  # loop to itself
                                        expecting_ctype=2,
                                    ),
                                ),
                            ]
                        ),
                    )
                ),
            ],
        )
    )


def test_earlier_check_model_already_resolved() -> None:
    # construct a case where expanding an earlier remote check model already resolves
    # a later one, so that the checker doesn't have to expand the later one again
    assert is_valid(
        pb.Library(
            **common,
            program=[
                *gadgets,
                *check_models,
                pb.Instruction(
                    error_model=pb.ErrorModel(
                        etype=1,
                        cid=1,
                        modifier=pb.ErrorModel.ErrorModelModifier(
                            reroute_remote_check_models=[
                                pb.ErrorModel.ErrorModelModifier.RerouteRemoteCheckModel(
                                    remote_check_model_index=0,
                                    value=pb.ErrorModelType.RemoteCheckModel(
                                        input=0,
                                        previous_remote_check_model=1,  # refer to later one
                                        expecting_ctype=1,
                                    ),
                                ),
                                pb.ErrorModel.ErrorModelModifier.RerouteRemoteCheckModel(
                                    remote_check_model_index=1,
                                    value=pb.ErrorModelType.RemoteCheckModel(
                                        output=0,
                                        expecting_ctype=2,
                                        check_bias=1,
                                    ),
                                ),
                            ]
                        ),
                    )
                ),
            ],
        )
    )


def test_program_validity_4_5_1() -> None:
    assert (
        "(ProgSpec 4.5.1)",
        "cannot be expanded (possibly referring to a placeholder)",
        "(ProgSpec 4.5.5)",
        "expects ctype 1, but got 2",
    ) in is_valid(
        pb.Library(
            **common,
            program=[
                *gadgets,
                *check_models,
                pb.Instruction(
                    error_model=pb.ErrorModel(
                        etype=1,
                        cid=1,
                        modifier=pb.ErrorModel.ErrorModelModifier(
                            reroute_remote_check_models=[
                                pb.ErrorModel.ErrorModelModifier.RerouteRemoteCheckModel(
                                    remote_check_model_index=0,
                                    value=pb.ErrorModelType.RemoteCheckModel(
                                        input=0,
                                        previous_remote_check_model=3,  # refer to a nonexisting one
                                        expecting_ctype=1,
                                    ),
                                ),
                                pb.ErrorModel.ErrorModelModifier.RerouteRemoteCheckModel(
                                    remote_check_model_index=4,
                                    value=pb.ErrorModelType.RemoteCheckModel(
                                        output=0, expecting_ctype=1  # should be 2
                                    ),
                                ),
                            ]
                        ),
                    )
                ),
            ],
        )
    )


def test_program_validity_4_5_2() -> None:
    assert (
        "(ProgSpec 4.5.2) invalid output port index 100 in "
        + "remote gadget ri=1 (gtype=3) of error model (eid=1)",
        "(ProgSpec 4.5.2) invalid input port index 100 in "
        + "remote gadget ri=2 (gtype=3) of error model (eid=1)",
    ) in is_valid(
        pb.Library(
            **common,
            program=[
                *gadgets,
                *check_models,
                pb.Instruction(
                    error_model=pb.ErrorModel(
                        etype=1,
                        cid=1,
                        modifier=pb.ErrorModel.ErrorModelModifier(
                            reroute_remote_check_models=[
                                pb.ErrorModel.ErrorModelModifier.RerouteRemoteCheckModel(
                                    remote_check_model_index=0,
                                    value=pb.ErrorModelType.RemoteCheckModel(
                                        output=0,
                                        expecting_ctype=0,  # use wildcard to confuse static checker
                                        check_bias=1,
                                    ),
                                ),
                                pb.ErrorModel.ErrorModelModifier.RerouteRemoteCheckModel(
                                    remote_check_model_index=1,
                                    value=pb.ErrorModelType.RemoteCheckModel(
                                        output=100, previous_remote_check_model=0
                                    ),
                                ),
                                pb.ErrorModel.ErrorModelModifier.RerouteRemoteCheckModel(
                                    remote_check_model_index=2,
                                    value=pb.ErrorModelType.RemoteCheckModel(
                                        input=100, previous_remote_check_model=0
                                    ),
                                ),
                            ]
                        ),
                    )
                ),
            ],
        )
    )


def test_program_validity_4_5_3() -> None:
    assert (
        "(ProgSpec 4.5.3) remote gadget ri=0 output port 0 of error model (eid=1) is not connected",
    ) in is_valid(
        pb.Library(
            port_types=default_library.port_types,
            gadget_types=default_library.gadget_types,
            check_model_types=[
                *default_library.check_model_types,
                pb.CheckModelType(
                    ctype=3,
                    gtype=2,
                ),
            ],
            error_model_types=[
                *default_library.error_model_types,
                pb.ErrorModelType(
                    etype=3,
                    ctype=3,  # bind to gtype=2
                ),
            ],
            program=[
                pb.Instruction(gadget=pb.Gadget(gtype=1)),
                pb.Instruction(
                    gadget=pb.Gadget(gtype=2, connectors=[pb.Gadget.Connector(gid=1)]),
                ),
                pb.Instruction(
                    check_model=pb.CheckModel(
                        ctype=3,
                        gid=2,
                    )
                ),
                pb.Instruction(
                    error_model=pb.ErrorModel(
                        etype=3,
                        cid=1,
                        modifier=pb.ErrorModel.ErrorModelModifier(
                            reroute_remote_check_models=[
                                pb.ErrorModel.ErrorModelModifier.RerouteRemoteCheckModel(
                                    remote_check_model_index=0,
                                    value=pb.ErrorModelType.RemoteCheckModel(
                                        output=0, check_bias=1
                                    ),
                                ),
                            ]
                        ),
                    )
                ),
            ],
        )
    )


def test_program_validity_4_5_4() -> None:
    assert (
        "(ProgSpec 4.5.4) remote gadget 3 is not binding to any check model",
        "(ProgSpec 4.5.4) remote gadget 1 is not binding to any check model",
    ) in is_valid(
        pb.Library(
            **common,
            program=[
                *gadgets,
                *check_models[:1],
                pb.Instruction(
                    error_model=pb.ErrorModel(
                        etype=1,
                        cid=1,
                        modifier=pb.ErrorModel.ErrorModelModifier(
                            reroute_remote_check_models=[
                                pb.ErrorModel.ErrorModelModifier.RerouteRemoteCheckModel(
                                    remote_check_model_index=2,
                                    value=pb.ErrorModelType.RemoteCheckModel(input=0),
                                ),
                            ]
                        ),
                    )
                ),
            ],
        )
    )


def test_program_validity_4_5_6() -> None:
    assert "(ProgSpec 4.5.6) invalid check bias" in is_valid(
        pb.Library(
            **common,
            program=[
                *gadgets,
                *check_models,
                pb.Instruction(
                    error_model=pb.ErrorModel(
                        etype=1,
                        cid=1,
                        modifier=pb.ErrorModel.ErrorModelModifier(
                            reroute_remote_check_models=[
                                pb.ErrorModel.ErrorModelModifier.RerouteRemoteCheckModel(
                                    remote_check_model_index=1,
                                    value=_placeholder_remote_check_model(),
                                ),
                                pb.ErrorModel.ErrorModelModifier.RerouteRemoteCheckModel(
                                    remote_check_model_index=2,
                                    value=pb.ErrorModelType.RemoteCheckModel(
                                        output=0, check_bias=100
                                    ),
                                ),
                            ]
                        ),
                    )
                ),
            ],
        )
    )


def test_program_validity_4_6_1_and_4_6_2() -> None:
    assert (
        "(ProgSpec 4.6.1) remote check model modified to a placeholder",
        "(ProgSpec 4.6.2) overflowed remote check index",
    ) in is_valid(
        pb.Library(
            **common,
            program=[
                *gadgets,
                *check_models,
                pb.Instruction(
                    error_model=pb.ErrorModel(
                        etype=1,
                        cid=1,
                        modifier=pb.ErrorModel.ErrorModelModifier(
                            reroute_remote_check_models=[
                                pb.ErrorModel.ErrorModelModifier.RerouteRemoteCheckModel(
                                    remote_check_model_index=1,
                                    value=_placeholder_remote_check_model(),
                                ),
                                pb.ErrorModel.ErrorModelModifier.RerouteRemoteCheckModel(
                                    remote_check_model_index=2,
                                    value=pb.ErrorModelType.RemoteCheckModel(
                                        output=0, check_bias=2
                                    ),
                                ),
                            ]
                        ),
                    )
                ),
            ],
        )
    )


def test_program_validity_3_6_3() -> None:
    # construct duplicate measurements that is not detectable by the library spec
    assert "(ProgSpec 3.6.3) duplicate measurement in check model" in is_valid(
        pb.Library(
            port_types=[
                pb.PortType(
                    ptype=1,
                    observables=[pb.PortType.Observable(tag="logical_z")],
                ),
            ],
            gadget_types=[
                pb.GadgetType(
                    gtype=1,
                    measurements=[
                        pb.GadgetType.Measurement(tag="m1"),
                        pb.GadgetType.Measurement(tag="m2"),
                    ],
                    outputs=[pb.GadgetType.Port(ptype=1)],
                    correction_propagation=util_pb.BitMatrix(rows=1, cols=1),
                    physical_correction=util_pb.BitMatrix(rows=1, cols=2),
                ),
                pb.GadgetType(
                    gtype=2,
                    measurements=[pb.GadgetType.Measurement(tag="m3")],
                    inputs=[pb.GadgetType.Port(ptype=1)],
                    physical_correction=util_pb.BitMatrix(rows=0, cols=1),
                ),
            ],
            check_model_types=[
                pb.CheckModelType(
                    ctype=1,
                    gtype=1,
                    remote_gadgets=[
                        pb.CheckModelType.RemoteGadget(output=0),
                        pb.CheckModelType.RemoteGadget(
                            input=0, previous_remote_gadget=0
                        ),
                    ],
                    checks=[
                        pb.CheckModelType.Check(
                            measurements=[
                                pb.CheckModelType.RemoteMeasurement(
                                    measurement_index=0
                                ),
                                pb.CheckModelType.RemoteMeasurement(
                                    remote_gadget=1, measurement_index=0
                                ),
                            ]
                        ),
                    ],
                ),
            ],
            program=[
                pb.Instruction(gadget=pb.Gadget(gtype=1)),
                pb.Instruction(
                    gadget=pb.Gadget(
                        gtype=2,
                        connectors=[pb.Gadget.Connector(gid=1, port=0)],
                    )
                ),
                pb.Instruction(
                    check_model=pb.CheckModel(
                        ctype=1,
                        gid=1,
                    )
                ),
            ],
        )
    )


def test_program_validity_4_6_3() -> None:
    # construct duplicate checks that is not detectable by the library spec
    assert "(ProgSpec 4.6.3) duplicate check in error model" in is_valid(
        pb.Library(
            port_types=[
                pb.PortType(
                    ptype=1,
                    observables=[pb.PortType.Observable(tag="logical_z")],
                ),
            ],
            gadget_types=[
                pb.GadgetType(
                    gtype=1,
                    outputs=[pb.GadgetType.Port(ptype=1)],
                    correction_propagation=util_pb.BitMatrix(rows=1, cols=1),
                    physical_correction=util_pb.BitMatrix(rows=1, cols=0),
                ),
                pb.GadgetType(
                    gtype=2,
                    inputs=[pb.GadgetType.Port(ptype=1)],
                    physical_correction=util_pb.BitMatrix(rows=0, cols=0),
                ),
            ],
            check_model_types=[
                pb.CheckModelType(
                    ctype=1,
                    gtype=1,
                    checks=[pb.CheckModelType.Check()],
                ),
                pb.CheckModelType(
                    ctype=2,
                    gtype=2,
                    checks=[pb.CheckModelType.Check()],
                ),
            ],
            error_model_types=[
                pb.ErrorModelType(
                    etype=1,
                    ctype=1,
                    remote_check_models=[
                        pb.ErrorModelType.RemoteCheckModel(output=0),
                        pb.ErrorModelType.RemoteCheckModel(
                            input=0, previous_remote_check_model=0
                        ),
                    ],
                    errors=[
                        pb.ErrorModelType.Error(
                            checks=[
                                pb.ErrorModelType.RemoteCheck(check_index=0),
                                pb.ErrorModelType.RemoteCheck(
                                    remote_check_model=1, check_index=0
                                ),
                            ]
                        ),
                    ],
                ),
            ],
            program=[
                pb.Instruction(gadget=pb.Gadget(gtype=1)),
                pb.Instruction(
                    gadget=pb.Gadget(
                        gtype=2, connectors=[pb.Gadget.Connector(gid=1, port=0)]
                    )
                ),
                pb.Instruction(check_model=pb.CheckModel(ctype=1, gid=1)),
                pb.Instruction(check_model=pb.CheckModel(ctype=2, gid=2)),
                pb.Instruction(error_model=pb.ErrorModel(etype=1, cid=1)),
            ],
        )
    )


def test_protobuf_deep_copy() -> None:
    # make sure that `CopyFrom` function does deep copy instead of shallow copy
    gadget_1 = pb.GadgetType(
        gtype=1,
        measurements=[
            pb.GadgetType.Measurement(tag="m1"),
        ],
        physical_correction=util_pb.BitMatrix(rows=0, cols=1),
    )
    gadget_2 = pb.GadgetType()
    gadget_2.CopyFrom(gadget_1)

    # make some modification
    gadget_2.gtype = 2
    gadget_2.measurements[0].tag = "m3"

    assert gadget_1.gtype == 1
    assert gadget_1.measurements[0].tag == "m1"
    assert gadget_2.gtype == 2
    assert gadget_2.measurements[0].tag == "m3"


def test_program_validity_2_4_1_wrong_matrix_dimensions() -> None:
    """Test ProgSpec 2.4.1: remote_conditional_correction matrix dimensions."""
    gadget_with_readout = pb.GadgetType(
        gtype=10,
        measurements=[pb.GadgetType.Measurement(tag="m1")],
        outputs=[pb.GadgetType.Port(ptype=1)],
        readouts=[pb.GadgetType.Readout(tag="r1")],
        correction_propagation=util_pb.BitMatrix(rows=1, cols=1),
        readout_propagation=util_pb.BitMatrix(rows=1, cols=1),
        logical_correction=util_pb.BitMatrix(rows=1, cols=1),
        physical_correction=util_pb.BitMatrix(rows=1, cols=1),
    )
    gadget_with_input = pb.GadgetType(
        gtype=11,
        inputs=[pb.GadgetType.Port(ptype=1)],
        outputs=[pb.GadgetType.Port(ptype=1)],
        correction_propagation=util_pb.BitMatrix(rows=1, cols=2),
        logical_correction=util_pb.BitMatrix(rows=1, cols=0),
        physical_correction=util_pb.BitMatrix(rows=1, cols=0),
    )
    result = is_valid(
        pb.Library(
            port_types=default_library.port_types,
            gadget_types=[gadget_with_readout, gadget_with_input],
            program=[
                pb.Instruction(gadget=pb.Gadget(gtype=10)),
                pb.Instruction(
                    gadget=pb.Gadget(
                        gtype=11,
                        connectors=[pb.Gadget.Connector(gid=1, port=0)],
                        modifier=pb.GadgetModifier(
                            remote_conditional_correction=pb.RemoteConditionalCorrection(
                                remote_readouts=[
                                    pb.RemoteConditionalCorrection.RemoteReadout(
                                        gid=1, readout_index=0
                                    )
                                ],
                                correction=util_pb.BitMatrix(rows=5, cols=5),
                            )
                        ),
                    )
                ),
            ],
        )
    )
    assert "(ProgSpec 2.4.1)" in result
    assert "wrong number of rows" in result


def test_program_validity_2_4_2_unknown_gadget() -> None:
    """Test ProgSpec 2.4.2: referencing unknown gadget."""
    gadget_with_output = pb.GadgetType(
        gtype=10,
        outputs=[pb.GadgetType.Port(ptype=1)],
        correction_propagation=util_pb.BitMatrix(rows=1, cols=1),
        logical_correction=util_pb.BitMatrix(rows=1, cols=0),
        physical_correction=util_pb.BitMatrix(rows=1, cols=0),
    )
    result = is_valid(
        pb.Library(
            port_types=default_library.port_types,
            gadget_types=[gadget_with_output],
            program=[
                pb.Instruction(
                    gadget=pb.Gadget(
                        gtype=10,
                        modifier=pb.GadgetModifier(
                            remote_conditional_correction=pb.RemoteConditionalCorrection(
                                remote_readouts=[
                                    pb.RemoteConditionalCorrection.RemoteReadout(
                                        gid=999, readout_index=0
                                    )
                                ],
                                correction=util_pb.BitMatrix(rows=1, cols=1),
                            )
                        ),
                    )
                ),
            ],
        )
    )
    assert "(ProgSpec 2.4.2)" in result
    assert "unknown gadget" in result


def test_program_validity_2_4_2_future_gadget() -> None:
    """Test ProgSpec 2.4.2: referencing future gadget (not yet instantiated)."""
    gadget_with_readout = pb.GadgetType(
        gtype=10,
        outputs=[pb.GadgetType.Port(ptype=1)],
        readouts=[pb.GadgetType.Readout(tag="r1")],
        correction_propagation=util_pb.BitMatrix(rows=1, cols=1),
        readout_propagation=util_pb.BitMatrix(rows=1, cols=1),
        logical_correction=util_pb.BitMatrix(rows=1, cols=1),
        physical_correction=util_pb.BitMatrix(rows=1, cols=0),
    )
    gadget_no_input = pb.GadgetType(
        gtype=11,
        outputs=[pb.GadgetType.Port(ptype=1)],
        correction_propagation=util_pb.BitMatrix(rows=1, cols=1),
        logical_correction=util_pb.BitMatrix(rows=1, cols=0),
        physical_correction=util_pb.BitMatrix(rows=1, cols=0),
    )
    result = is_valid(
        pb.Library(
            port_types=default_library.port_types,
            gadget_types=[gadget_with_readout, gadget_no_input],
            program=[
                pb.Instruction(
                    gadget=pb.Gadget(
                        gtype=11,
                        modifier=pb.GadgetModifier(
                            remote_conditional_correction=pb.RemoteConditionalCorrection(
                                remote_readouts=[
                                    pb.RemoteConditionalCorrection.RemoteReadout(
                                        gid=2, readout_index=0
                                    )
                                ],
                                correction=util_pb.BitMatrix(rows=1, cols=1),
                            )
                        ),
                    )
                ),
                pb.Instruction(gadget=pb.Gadget(gtype=10)),
            ],
        )
    )
    assert "(ProgSpec 2.4.2)" in result
    assert "not instantiated before" in result


def test_program_validity_2_4_3_invalid_readout_index() -> None:
    """Test ProgSpec 2.4.3: invalid readout index."""
    gadget_with_readout = pb.GadgetType(
        gtype=10,
        outputs=[pb.GadgetType.Port(ptype=1)],
        readouts=[pb.GadgetType.Readout(tag="r1")],
        correction_propagation=util_pb.BitMatrix(rows=1, cols=1),
        readout_propagation=util_pb.BitMatrix(rows=1, cols=1),
        logical_correction=util_pb.BitMatrix(rows=1, cols=1),
        physical_correction=util_pb.BitMatrix(rows=1, cols=0),
    )
    gadget_with_input = pb.GadgetType(
        gtype=11,
        inputs=[pb.GadgetType.Port(ptype=1)],
        outputs=[pb.GadgetType.Port(ptype=1)],
        correction_propagation=util_pb.BitMatrix(rows=1, cols=2),
        logical_correction=util_pb.BitMatrix(rows=1, cols=0),
        physical_correction=util_pb.BitMatrix(rows=1, cols=0),
    )
    result = is_valid(
        pb.Library(
            port_types=default_library.port_types,
            gadget_types=[gadget_with_readout, gadget_with_input],
            program=[
                pb.Instruction(gadget=pb.Gadget(gtype=10)),
                pb.Instruction(
                    gadget=pb.Gadget(
                        gtype=11,
                        connectors=[pb.Gadget.Connector(gid=1, port=0)],
                        modifier=pb.GadgetModifier(
                            remote_conditional_correction=pb.RemoteConditionalCorrection(
                                remote_readouts=[
                                    pb.RemoteConditionalCorrection.RemoteReadout(
                                        gid=1, readout_index=99
                                    )
                                ],
                                correction=util_pb.BitMatrix(rows=1, cols=1),
                            )
                        ),
                    )
                ),
            ],
        )
    )
    assert "(ProgSpec 2.4.3)" in result
    assert "invalid readout_index" in result


def test_program_validity_2_4_success_self_reference() -> None:
    """Test that a gadget can reference its own readouts."""
    gadget_with_readout = pb.GadgetType(
        gtype=10,
        outputs=[pb.GadgetType.Port(ptype=1)],
        readouts=[pb.GadgetType.Readout(tag="r1")],
        correction_propagation=util_pb.BitMatrix(rows=1, cols=1),
        readout_propagation=util_pb.BitMatrix(rows=1, cols=1),
        logical_correction=util_pb.BitMatrix(rows=1, cols=1),
        physical_correction=util_pb.BitMatrix(rows=1, cols=0),
    )
    result = is_valid(
        pb.Library(
            port_types=default_library.port_types,
            gadget_types=[gadget_with_readout],
            program=[
                pb.Instruction(
                    gadget=pb.Gadget(
                        gtype=10,
                        modifier=pb.GadgetModifier(
                            remote_conditional_correction=pb.RemoteConditionalCorrection(
                                remote_readouts=[
                                    pb.RemoteConditionalCorrection.RemoteReadout(
                                        gid=1, readout_index=0
                                    )
                                ],
                                correction=util_pb.BitMatrix(
                                    rows=1, cols=1, i=[0], j=[0]
                                ),
                            )
                        ),
                    )
                ),
            ],
        )
    )
    assert result, f"Expected valid program but got: {result}"


def test_program_validity_2_4_success_remote_reference() -> None:
    """Test successful remote conditional correction referencing a previous gadget."""
    gadget_with_readout = pb.GadgetType(
        gtype=10,
        outputs=[pb.GadgetType.Port(ptype=1)],
        readouts=[pb.GadgetType.Readout(tag="r1")],
        correction_propagation=util_pb.BitMatrix(rows=1, cols=1),
        readout_propagation=util_pb.BitMatrix(rows=1, cols=1),
        logical_correction=util_pb.BitMatrix(rows=1, cols=1),
        physical_correction=util_pb.BitMatrix(rows=1, cols=0),
    )
    gadget_with_input = pb.GadgetType(
        gtype=11,
        inputs=[pb.GadgetType.Port(ptype=1)],
        outputs=[pb.GadgetType.Port(ptype=1)],
        correction_propagation=util_pb.BitMatrix(rows=1, cols=2),
        logical_correction=util_pb.BitMatrix(rows=1, cols=0),
        physical_correction=util_pb.BitMatrix(rows=1, cols=0),
    )
    result = is_valid(
        pb.Library(
            port_types=default_library.port_types,
            gadget_types=[gadget_with_readout, gadget_with_input],
            program=[
                pb.Instruction(gadget=pb.Gadget(gtype=10)),
                pb.Instruction(
                    gadget=pb.Gadget(
                        gtype=11,
                        connectors=[pb.Gadget.Connector(gid=1, port=0)],
                        modifier=pb.GadgetModifier(
                            remote_conditional_correction=pb.RemoteConditionalCorrection(
                                remote_readouts=[
                                    pb.RemoteConditionalCorrection.RemoteReadout(
                                        gid=1, readout_index=0
                                    )
                                ],
                                correction=util_pb.BitMatrix(
                                    rows=1, cols=1, i=[0], j=[0]
                                ),
                            )
                        ),
                    )
                ),
            ],
        )
    )
    assert result, f"Expected valid program but got: {result}"
    assert isinstance(result, ExpandedProgram)
    assert 2 in result.expanded_remote_conditional_corrections


def test_program_validity_gadget_modifier_logical_and_physical_correction_valid() -> None:
    """``logical_correction_mod`` and ``physical_correction_mod`` apply cleanly
    to their base matrices when the modifier dimensions match."""
    lib = pb.Library(
        port_types=[pb.PortType(ptype=1, observables=[pb.PortType.Observable()])],
        gadget_types=[
            pb.GadgetType(
                gtype=1,
                measurements=[pb.GadgetType.Measurement()],
                outputs=[pb.GadgetType.Port(ptype=1)],
                readouts=[pb.GadgetType.Readout(measurement_indices=[0])],
                correction_propagation=util_pb.BitMatrix(rows=1, cols=1),
                readout_propagation=util_pb.BitMatrix(rows=1, cols=1),
                logical_correction=util_pb.BitMatrix(rows=1, cols=1),
                physical_correction=util_pb.BitMatrix(rows=1, cols=1),
            ),
        ],
        check_model_types=[pb.CheckModelType(ctype=1, checks=[])],
        error_model_types=[pb.ErrorModelType(etype=1, errors=[])],
        program=[
            pb.Instruction(
                gadget=pb.Gadget(
                    gtype=1,
                    modifier=pb.GadgetModifier(
                        logical_correction_mod=pb.BitMatrixModifier(
                            toggle=util_pb.BitMatrix(rows=1, cols=1, i=[0], j=[0]),
                        ),
                        physical_correction_mod=pb.BitMatrixModifier(
                            toggle=util_pb.BitMatrix(rows=1, cols=1, i=[0], j=[0]),
                        ),
                    ),
                )
            ),
            pb.Instruction(check_model=pb.CheckModel(ctype=1, gid=1)),
            pb.Instruction(error_model=pb.ErrorModel(etype=1, cid=1)),
        ],
    )
    result = is_valid(lib)
    assert isinstance(result, ExpandedProgram)
    modified = result.modified_gadget_types[1]
    assert list(zip(modified.logical_correction.i, modified.logical_correction.j)) == [(0, 0)]
    assert list(zip(modified.physical_correction.i, modified.physical_correction.j)) == [(0, 0)]


def test_program_validity_4_3_3_sparse_probabilities_valid() -> None:
    """Valid ``sparse_probabilities`` (each in [0,1]) applies without violation."""
    lib_error_type = default_library.error_model_types[0]
    valid_index = 0
    assert 0 <= valid_index < len(lib_error_type.errors)
    result = is_valid(
        pb.Library(
            **common,
            program=[
                *gadgets,
                *check_models,
                pb.Instruction(
                    error_model=pb.ErrorModel(
                        etype=1,
                        cid=1,
                        modifier=pb.ErrorModel.ErrorModelModifier(
                            probability_modifier=pb.ProbabilityModifier(
                                sparse_indices=[valid_index],
                                sparse_probabilities=[0.25],
                            )
                        ),
                    )
                ),
            ],
        )
    )
    assert isinstance(result, ExpandedProgram)
    assert result.modified_error_model_types[1].errors[valid_index].probability == 0.25
