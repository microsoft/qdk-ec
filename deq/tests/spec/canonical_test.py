import deq.proto.deq_bin_pb2 as pb
import deq.proto.util_pb2 as util_pb
from deq.spec.program_validator import is_valid
from deq.spec.library_equivalence import are_libraries_equivalent
from deq.spec.canonical import canonicalize, canonical_program
from tests.spec.library_validator_test import default_library

# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint


default_library_canonical = pb.Library(
    # since we don't have any unconnected output port, there is no observable
    port_types=[pb.PortType(ptype=1)],
    gadget_types=[
        pb.GadgetType(
            gtype=1,
            measurements=[pb.GadgetType.Measurement()] * 7,
            outputs=[pb.GadgetType.Port(ptype=1)],
            readouts=[pb.GadgetType.Readout(measurement_indices=[4, 6])],
            # although the readout is naturally flipped, the gid=2's output logical x
            # is also naturally flipped, which flips the readout back in gid=3
            readout_propagation=util_pb.BitMatrix(rows=1, cols=1),
            physical_correction=util_pb.BitMatrix(rows=0, cols=7),
        )
    ],
    check_model_types=[
        pb.CheckModelType(
            ctype=1,
            gtype=1,
            checks=[
                pb.CheckModelType.Check(
                    measurements=[
                        pb.CheckModelType.RemoteMeasurement(measurement_index=i)
                        for i in (3, 1, 2, 5)
                    ]
                ),
                pb.CheckModelType.Check(),
                pb.CheckModelType.Check(),
                pb.CheckModelType.Check(
                    measurements=[
                        pb.CheckModelType.RemoteMeasurement(measurement_index=i)
                        for i in (4, 5, 6, 2)
                    ]
                ),
                pb.CheckModelType.Check(),
                pb.CheckModelType.Check(),
            ],
        )
    ],
    error_model_types=[
        pb.ErrorModelType(
            etype=1,
            ctype=1,
            errors=[
                pb.ErrorModelType.Error(
                    probability=0.1,
                    # e1 flips gid=2's logical x, which then flips the readout
                    readout_flips=[0],
                    checks=[  # c1, c2, c5
                        pb.ErrorModelType.RemoteCheck(check_index=i) for i in (0, 1, 4)
                    ],
                ),
                pb.ErrorModelType.Error(
                    probability=0.1,
                    readout_flips=[0],
                ),
            ],
        )
    ],
    program=canonical_program(),
)


def test_canonical_default() -> None:

    canonical_form = canonicalize(default_library)

    assert is_valid(canonical_form.library)

    assert are_libraries_equivalent(canonical_form.library, default_library_canonical)

    assert canonical_form.library.program == default_library_canonical.program

    assert canonical_form.port_type == canonical_form.library.port_types[0]
    assert canonical_form.gadget_type == canonical_form.library.gadget_types[0]
    assert (
        canonical_form.check_model_type == canonical_form.library.check_model_types[0]
    )
    assert (
        canonical_form.error_model_type == canonical_form.library.error_model_types[0]
    )


def test_canonical_no_measurement() -> None:
    """test a case where we have unconnected output port"""
    library = pb.Library(
        port_types=default_library.port_types,
        gadget_types=[
            default_library.gadget_types[0],
            pb.GadgetType(
                gtype=2,
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
                readouts=[
                    pb.GadgetType.Readout(tag="r1", measurement_indices=[0, 1]),
                ],
                # logical x propagates to logical x, also make logical x naturally flipped
                correction_propagation=util_pb.BitMatrix(
                    rows=2, cols=2, i=[0, 0], j=[0, 1]
                ),
                readout_propagation=util_pb.BitMatrix(rows=1, cols=2, i=[0], j=[1]),
                logical_correction=util_pb.BitMatrix(rows=2, cols=1, i=[0], j=[0]),
                physical_correction=util_pb.BitMatrix(rows=2, cols=2),
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
                ],
                checks=[
                    pb.CheckModelType.Check(
                        measurements=[  # m4, m2
                            pb.CheckModelType.RemoteMeasurement(measurement_index=1),
                            pb.CheckModelType.RemoteMeasurement(remote_gadget=0),
                        ]
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
                        residual=[0],
                        checks=[
                            pb.ErrorModelType.RemoteCheck(check_index=0),
                        ],
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
            pb.Instruction(check_model=pb.CheckModel(ctype=1, gid=2)),
            pb.Instruction(error_model=pb.ErrorModel(etype=1, cid=1)),
        ],
    )
    assert is_valid(library)

    canonical_form = canonicalize(library)

    assert is_valid(canonical_form.library)

    assert are_libraries_equivalent(
        canonical_form.library,
        pb.Library(
            # the output port of gadget 2 is unconnected, so we have 2 observables
            port_types=[
                pb.PortType(
                    ptype=1,
                    observables=[pb.PortType.Observable(), pb.PortType.Observable()],
                )
            ],
            gadget_types=[
                pb.GadgetType(
                    gtype=1,
                    measurements=[pb.GadgetType.Measurement()] * 4,
                    outputs=[pb.GadgetType.Port(ptype=1)],
                    correction_propagation=util_pb.BitMatrix(
                        rows=2, cols=1, i=[0], j=[0]
                    ),
                    readouts=[pb.GadgetType.Readout(measurement_indices=[2, 3])],
                    readout_propagation=util_pb.BitMatrix(rows=1, cols=1, i=[0], j=[0]),
                    logical_correction=util_pb.BitMatrix(
                        rows=2, cols=1, i=[0], j=[0]
                    ),
                    physical_correction=util_pb.BitMatrix(rows=2, cols=4),
                )
            ],
            check_model_types=[
                pb.CheckModelType(
                    ctype=1,
                    gtype=1,
                    checks=[
                        pb.CheckModelType.Check(
                            measurements=[
                                pb.CheckModelType.RemoteMeasurement(measurement_index=i)
                                for i in (3, 1)
                            ]
                        ),
                    ],
                )
            ],
            error_model_types=[
                pb.ErrorModelType(
                    etype=1,
                    ctype=1,
                    errors=[
                        pb.ErrorModelType.Error(
                            probability=0.1,
                            residual=[0],
                            checks=[pb.ErrorModelType.RemoteCheck(check_index=0)],
                        ),
                    ],
                )
            ],
            program=[],
        ),
    )


def test_canonical_with_gadget_modifier_toggle() -> None:
    """Test that GadgetModifier toggle correctly modifies the propagation matrix.

    Setup:
    - gtype=1: initializer with empty correction_propagation (1x1, all zeros)
    - gtype=2: has correction_propagation=[[1,0]] (input obs propagates to output)

    The toggle modifier XORs [[0,1]] into gtype=2's matrix:
    - Original: [[1,0]] -> After toggle: [[1,1]]
    - This sets the constant column, meaning the output observable is now naturally flipped

    The canonical form should reflect this: the single output observable has a constant
    flip, shown as correction_propagation with i=[0], j=[0] (constant column set).
    Without the modifier, the canonical result would have an empty constant column.
    """
    library = pb.Library(
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
        check_model_types=[
            pb.CheckModelType(ctype=1, checks=[]),
        ],
        error_model_types=[
            pb.ErrorModelType(etype=1, errors=[]),
        ],
        program=[
            pb.Instruction(gadget=pb.Gadget(gtype=1)),
            pb.Instruction(
                gadget=pb.Gadget(
                    gtype=2,
                    connectors=[pb.Gadget.Connector(gid=1, port=0)],
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
    assert is_valid(library)

    canonical_form = canonicalize(library)
    assert is_valid(canonical_form.library)

    expected = pb.Library(
        port_types=[
            pb.PortType(ptype=1, observables=[pb.PortType.Observable()]),
        ],
        gadget_types=[
            pb.GadgetType(
                gtype=1,
                measurements=[pb.GadgetType.Measurement()] * 2,
                outputs=[pb.GadgetType.Port(ptype=1)],
                correction_propagation=util_pb.BitMatrix(rows=1, cols=1, i=[0], j=[0]),
                physical_correction=util_pb.BitMatrix(rows=1, cols=2),
            ),
        ],
        check_model_types=[pb.CheckModelType(ctype=1, gtype=1, checks=[])],
        error_model_types=[pb.ErrorModelType(etype=1, ctype=1, errors=[])],
        program=[],
    )
    assert are_libraries_equivalent(canonical_form.library, expected)


def test_canonical_with_gadget_modifier_overwrite() -> None:
    """Test that GadgetModifier overwrite completely replaces the propagation matrix.

    Setup:
    - gtype=2 originally has correction_propagation=[[1,0]] (input propagates to output)

    The overwrite modifier replaces it with an all-zero matrix [[0,0]]:
    - This removes all propagation: input no longer affects output, no constant flip

    The canonical form should have an empty correction_propagation (no bits set),
    meaning the output observable is independent of input and not naturally flipped.
    """
    library = pb.Library(
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
        check_model_types=[
            pb.CheckModelType(ctype=1, checks=[]),
        ],
        error_model_types=[
            pb.ErrorModelType(etype=1, errors=[]),
        ],
        program=[
            pb.Instruction(gadget=pb.Gadget(gtype=1)),
            pb.Instruction(
                gadget=pb.Gadget(
                    gtype=2,
                    connectors=[pb.Gadget.Connector(gid=1, port=0)],
                    modifier=pb.GadgetModifier(
                        correction_propagation_mod=pb.BitMatrixModifier(
                            overwrite=util_pb.BitMatrix(rows=1, cols=2),
                        ),
                    ),
                )
            ),
            pb.Instruction(check_model=pb.CheckModel(ctype=1, gid=2)),
            pb.Instruction(error_model=pb.ErrorModel(etype=1, cid=1)),
        ],
    )
    assert is_valid(library)

    canonical_form = canonicalize(library)
    assert is_valid(canonical_form.library)

    expected = pb.Library(
        port_types=[
            pb.PortType(ptype=1, observables=[pb.PortType.Observable()]),
        ],
        gadget_types=[
            pb.GadgetType(
                gtype=1,
                measurements=[pb.GadgetType.Measurement()] * 2,
                outputs=[pb.GadgetType.Port(ptype=1)],
                correction_propagation=util_pb.BitMatrix(rows=1, cols=1),
                physical_correction=util_pb.BitMatrix(rows=1, cols=2),
            ),
        ],
        check_model_types=[pb.CheckModelType(ctype=1, gtype=1, checks=[])],
        error_model_types=[pb.ErrorModelType(etype=1, ctype=1, errors=[])],
        program=[],
    )
    assert are_libraries_equivalent(canonical_form.library, expected)


def test_canonical_with_gadget_modifier_toggle_then_overwrite() -> None:
    """Test that toggle is applied before overwrite when both are present.

    When both toggle and overwrite are specified, toggle is applied first (XOR),
    then overwrite completely replaces the result. This means the toggle has no
    effect when overwrite is also present.

    The overwrite sets [[0,1]] (constant column only), so the output observable
    is naturally flipped regardless of input. The canonical form reflects this.
    """
    library = pb.Library(
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
                correction_propagation=util_pb.BitMatrix(rows=1, cols=2),
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
                    connectors=[pb.Gadget.Connector(gid=1, port=0)],
                    modifier=pb.GadgetModifier(
                        correction_propagation_mod=pb.BitMatrixModifier(
                            toggle=util_pb.BitMatrix(rows=1, cols=2, i=[0], j=[0]),
                            overwrite=util_pb.BitMatrix(rows=1, cols=2, i=[0], j=[1]),
                        ),
                    ),
                )
            ),
            pb.Instruction(check_model=pb.CheckModel(ctype=1, gid=2)),
            pb.Instruction(error_model=pb.ErrorModel(etype=1, cid=1)),
        ],
    )
    assert is_valid(library)

    canonical_form = canonicalize(library)
    assert is_valid(canonical_form.library)

    expected = pb.Library(
        port_types=[
            pb.PortType(ptype=1, observables=[pb.PortType.Observable()]),
        ],
        gadget_types=[
            pb.GadgetType(
                gtype=1,
                measurements=[pb.GadgetType.Measurement()] * 2,
                outputs=[pb.GadgetType.Port(ptype=1)],
                correction_propagation=util_pb.BitMatrix(rows=1, cols=1, i=[0], j=[0]),
                physical_correction=util_pb.BitMatrix(rows=1, cols=2),
            ),
        ],
        check_model_types=[pb.CheckModelType(ctype=1, gtype=1, checks=[])],
        error_model_types=[pb.ErrorModelType(etype=1, ctype=1, errors=[])],
        program=[],
    )
    assert are_libraries_equivalent(canonical_form.library, expected)


def test_canonical_remote_conditional_correction() -> None:
    """Test that remote_conditional_correction is XORed into the
    canonical logical_correction."""
    library = pb.Library(
        port_types=[
            pb.PortType(
                ptype=1,
                observables=[pb.PortType.Observable(tag="obs1")],
            ),
        ],
        gadget_types=[
            pb.GadgetType(
                gtype=1,
                measurements=[pb.GadgetType.Measurement(tag="m1")],
                outputs=[pb.GadgetType.Port(ptype=1)],
                readouts=[pb.GadgetType.Readout(tag="r1")],
                correction_propagation=util_pb.BitMatrix(rows=1, cols=1),
                readout_propagation=util_pb.BitMatrix(rows=1, cols=1),
                logical_correction=util_pb.BitMatrix(rows=1, cols=1),
                physical_correction=util_pb.BitMatrix(rows=1, cols=1),
            ),
            pb.GadgetType(
                gtype=2,
                measurements=[pb.GadgetType.Measurement(tag="m2")],
                inputs=[pb.GadgetType.Port(ptype=1)],
                outputs=[pb.GadgetType.Port(ptype=1)],
                correction_propagation=util_pb.BitMatrix(rows=1, cols=2),
                logical_correction=util_pb.BitMatrix(rows=1, cols=0),
                physical_correction=util_pb.BitMatrix(rows=1, cols=1),
            ),
        ],
        check_model_types=[
            pb.CheckModelType(ctype=1, gtype=2, checks=[]),
        ],
        error_model_types=[
            pb.ErrorModelType(etype=1, ctype=1, errors=[]),
        ],
        program=[
            pb.Instruction(gadget=pb.Gadget(gtype=1)),
            pb.Instruction(
                gadget=pb.Gadget(
                    gtype=2,
                    connectors=[pb.Gadget.Connector(gid=1, port=0)],
                    modifier=pb.GadgetModifier(
                        remote_conditional_correction=pb.RemoteConditionalCorrection(
                            remote_readouts=[
                                pb.RemoteConditionalCorrection.RemoteReadout(
                                    gid=1, readout_index=0
                                )
                            ],
                            correction=util_pb.BitMatrix(rows=1, cols=1, i=[0], j=[0]),
                        )
                    ),
                )
            ),
            pb.Instruction(check_model=pb.CheckModel(ctype=1, gid=2)),
            pb.Instruction(error_model=pb.ErrorModel(etype=1, cid=1)),
        ],
    )
    assert is_valid(library)

    canonical_form = canonicalize(library)
    assert is_valid(canonical_form.library)

    canonical_gadget_type = canonical_form.library.gadget_types[0]
    assert canonical_gadget_type.readouts, "Should have readouts in canonical form"
    cond_corr = canonical_gadget_type.logical_correction
    assert cond_corr.rows == 1, "Should have 1 output observable"
    assert cond_corr.cols == 1, "Should have 1 readout"
    assert list(cond_corr.i) == [0], "Observable 0 should be corrected"
    assert list(cond_corr.j) == [0], "Based on readout 0"


def test_canonical_remote_conditional_correction_xor() -> None:
    """Test that remote_conditional_correction XORs with existing logical_correction."""
    library = pb.Library(
        port_types=[
            pb.PortType(
                ptype=1,
                observables=[pb.PortType.Observable(tag="obs1")],
            ),
        ],
        gadget_types=[
            pb.GadgetType(
                gtype=1,
                outputs=[pb.GadgetType.Port(ptype=1)],
                readouts=[pb.GadgetType.Readout(tag="r1")],
                correction_propagation=util_pb.BitMatrix(rows=1, cols=1),
                readout_propagation=util_pb.BitMatrix(rows=1, cols=1),
                logical_correction=util_pb.BitMatrix(rows=1, cols=1, i=[0], j=[0]),
                physical_correction=util_pb.BitMatrix(rows=1, cols=0),
            ),
        ],
        check_model_types=[
            pb.CheckModelType(ctype=1, gtype=1, checks=[]),
        ],
        error_model_types=[
            pb.ErrorModelType(etype=1, ctype=1, errors=[]),
        ],
        program=[
            pb.Instruction(
                gadget=pb.Gadget(
                    gtype=1,
                    modifier=pb.GadgetModifier(
                        remote_conditional_correction=pb.RemoteConditionalCorrection(
                            remote_readouts=[
                                pb.RemoteConditionalCorrection.RemoteReadout(
                                    gid=1, readout_index=0
                                )
                            ],
                            correction=util_pb.BitMatrix(rows=1, cols=1, i=[0], j=[0]),
                        )
                    ),
                )
            ),
            pb.Instruction(check_model=pb.CheckModel(ctype=1, gid=1)),
            pb.Instruction(error_model=pb.ErrorModel(etype=1, cid=1)),
        ],
    )
    assert is_valid(library)

    canonical_form = canonicalize(library)
    assert is_valid(canonical_form.library)

    canonical_gadget_type = canonical_form.library.gadget_types[0]
    cond_corr = canonical_gadget_type.logical_correction
    assert len(cond_corr.i) == 0, "XOR of same bit should cancel out"
    assert len(cond_corr.j) == 0, "XOR of same bit should cancel out"


def test_canonical_remote_conditional_correction_multiple_gadgets() -> None:
    """Test remote_conditional_correction with multiple gadgets in a chain."""
    library = pb.Library(
        port_types=[
            pb.PortType(
                ptype=1,
                observables=[pb.PortType.Observable(tag="obs1")],
            ),
        ],
        gadget_types=[
            pb.GadgetType(
                gtype=1,
                outputs=[pb.GadgetType.Port(ptype=1)],
                readouts=[pb.GadgetType.Readout(tag="r1")],
                correction_propagation=util_pb.BitMatrix(rows=1, cols=1),
                readout_propagation=util_pb.BitMatrix(rows=1, cols=1),
                logical_correction=util_pb.BitMatrix(rows=1, cols=1),
                physical_correction=util_pb.BitMatrix(rows=1, cols=0),
            ),
            pb.GadgetType(
                gtype=2,
                inputs=[pb.GadgetType.Port(ptype=1)],
                outputs=[pb.GadgetType.Port(ptype=1)],
                readouts=[pb.GadgetType.Readout(tag="r2")],
                correction_propagation=util_pb.BitMatrix(rows=1, cols=2),
                readout_propagation=util_pb.BitMatrix(rows=1, cols=2),
                logical_correction=util_pb.BitMatrix(rows=1, cols=1),
                physical_correction=util_pb.BitMatrix(rows=1, cols=0),
            ),
            pb.GadgetType(
                gtype=3,
                inputs=[pb.GadgetType.Port(ptype=1)],
                outputs=[pb.GadgetType.Port(ptype=1)],
                correction_propagation=util_pb.BitMatrix(rows=1, cols=2),
                logical_correction=util_pb.BitMatrix(rows=1, cols=0),
                physical_correction=util_pb.BitMatrix(rows=1, cols=0),
            ),
        ],
        check_model_types=[
            pb.CheckModelType(ctype=1, gtype=3, checks=[]),
        ],
        error_model_types=[
            pb.ErrorModelType(etype=1, ctype=1, errors=[]),
        ],
        program=[
            pb.Instruction(gadget=pb.Gadget(gtype=1)),
            pb.Instruction(
                gadget=pb.Gadget(
                    gtype=2, connectors=[pb.Gadget.Connector(gid=1, port=0)]
                )
            ),
            pb.Instruction(
                gadget=pb.Gadget(
                    gtype=3,
                    connectors=[pb.Gadget.Connector(gid=2, port=0)],
                    modifier=pb.GadgetModifier(
                        remote_conditional_correction=pb.RemoteConditionalCorrection(
                            remote_readouts=[
                                pb.RemoteConditionalCorrection.RemoteReadout(
                                    gid=1, readout_index=0
                                ),
                                pb.RemoteConditionalCorrection.RemoteReadout(
                                    gid=2, readout_index=0
                                ),
                            ],
                            correction=util_pb.BitMatrix(
                                rows=1, cols=2, i=[0, 0], j=[0, 1]
                            ),
                        )
                    ),
                )
            ),
            pb.Instruction(check_model=pb.CheckModel(ctype=1, gid=3)),
            pb.Instruction(error_model=pb.ErrorModel(etype=1, cid=1)),
        ],
    )
    assert is_valid(library)

    canonical_form = canonicalize(library)
    assert is_valid(canonical_form.library)

    canonical_gadget_type = canonical_form.library.gadget_types[0]
    assert len(canonical_gadget_type.readouts) == 2, "Should have 2 readouts total"
    cond_corr = canonical_gadget_type.logical_correction
    assert cond_corr.rows == 1, "Should have 1 output observable"
    assert cond_corr.cols == 2, "Should have 2 readouts"
    correction_set = set(zip(cond_corr.i, cond_corr.j))
    assert (0, 0) in correction_set, "Observable 0 corrected by readout 0"
    assert (0, 1) in correction_set, "Observable 0 corrected by readout 1"
