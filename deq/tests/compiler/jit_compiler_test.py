# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint
from copy import deepcopy
from typing import Optional
import deq.proto.deq_jit_pb2 as jit_pb
import deq.proto.deq_bin_pb2 as pb
import deq.proto.util_pb2 as util_pb
from deq.compiler.jit_compiler import static_jit_compiler


def test_empty_jit_compile() -> None:
    jit_library = jit_pb.JitLibrary()
    library = static_jit_compiler(jit_library)
    assert library == pb.Library()


basic_jit_library = jit_pb.JitLibrary(
    port_types=[
        jit_pb.JitPortType(
            base=pb.PortType(
                ptype=1,
                observables=[pb.PortType.Observable()] * 2,
            ),
            k=1,
            stabilizers=[jit_pb.JitPortType.Stabilizer()] * 2,
        )
    ],
    gadget_types=[
        jit_pb.JitGadgetType(
            base=pb.GadgetType(
                gtype=1,
                name="prepare_z",
                measurements=[pb.GadgetType.Measurement()] * 2,
                outputs=[pb.GadgetType.Port(ptype=1)],
                # 2 output observables, 0 input observables, 0 logical readouts
                correction_propagation=util_pb.BitMatrix(rows=2, cols=1, i=[], j=[]),
                readout_propagation=util_pb.BitMatrix(rows=0, cols=1),
                logical_correction=util_pb.BitMatrix(rows=0, cols=0),
                physical_correction=util_pb.BitMatrix(rows=2, cols=2),
            ),
            finished_checks=[
                jit_pb.JitGadgetType.Check(
                    base=pb.CheckModelType.Check(),
                    measurements=[
                        jit_pb.JitGadgetType.PresentMeasurement(measurement_index=0)
                    ],
                ),
                jit_pb.JitGadgetType.Check(
                    base=pb.CheckModelType.Check(),
                    measurements=[
                        jit_pb.JitGadgetType.PresentMeasurement(measurement_index=1)
                    ],
                ),
            ],
            unfinished_checks=[
                jit_pb.JitGadgetType.Check(
                    base=pb.CheckModelType.Check(),
                    measurements=[
                        jit_pb.JitGadgetType.PresentMeasurement(measurement_index=0)
                    ],
                ),
                jit_pb.JitGadgetType.Check(
                    base=pb.CheckModelType.Check(),
                    measurements=[
                        jit_pb.JitGadgetType.PresentMeasurement(measurement_index=1)
                    ],
                ),
            ],
            # errors=[
            #     jit_pb.JitGadgetType.Error(
            #         base=pb.ErrorModelType.Error(
            #             residual=[1], readout_flips=[], probability=0.01
            #         ),
            #         finished_checks=[1],
            #         unfinished_checks=[1],
            #     ),
            # ],
        ),
        jit_pb.JitGadgetType(
            base=pb.GadgetType(
                gtype=2,
                name="measure_z",
                measurements=[pb.GadgetType.Measurement()] * 3,
                inputs=[pb.GadgetType.Port(ptype=1)],
                # 0 output observables, 2 input observables, 1 logical readouts
                correction_propagation=util_pb.BitMatrix(rows=0, cols=3),
                readout_propagation=util_pb.BitMatrix(rows=1, cols=3, i=[0], j=[0]),
                logical_correction=util_pb.BitMatrix(rows=0, cols=1),
                physical_correction=util_pb.BitMatrix(rows=0, cols=3),
                readouts=[pb.GadgetType.Readout(measurement_indices=[0, 1, 2])],
            ),
            finished_checks=[
                jit_pb.JitGadgetType.Check(
                    base=pb.CheckModelType.Check(),
                    measurements=[
                        jit_pb.JitGadgetType.PresentMeasurement(
                            input_port=0, measurement_index=0
                        ),
                        jit_pb.JitGadgetType.PresentMeasurement(measurement_index=0),
                        jit_pb.JitGadgetType.PresentMeasurement(measurement_index=1),
                    ],
                ),
                jit_pb.JitGadgetType.Check(
                    base=pb.CheckModelType.Check(),
                    measurements=[
                        jit_pb.JitGadgetType.PresentMeasurement(
                            input_port=0, measurement_index=1
                        ),
                        jit_pb.JitGadgetType.PresentMeasurement(measurement_index=1),
                        jit_pb.JitGadgetType.PresentMeasurement(measurement_index=2),
                    ],
                ),
            ],
        ),
        jit_pb.JitGadgetType(
            base=pb.GadgetType(
                gtype=3,
                name="idle",
                measurements=[pb.GadgetType.Measurement()] * 2,
                inputs=[pb.GadgetType.Port(ptype=1)],
                outputs=[pb.GadgetType.Port(ptype=1)],
                # 2 output observables, 2 input observables, 0 logical readouts
                correction_propagation=util_pb.BitMatrix(
                    rows=2, cols=3, i=[0, 0], j=[1, 1]
                ),
                readout_propagation=util_pb.BitMatrix(rows=0, cols=3),
                logical_correction=util_pb.BitMatrix(rows=2, cols=0),
                physical_correction=util_pb.BitMatrix(rows=2, cols=2),
            ),
            finished_checks=[
                jit_pb.JitGadgetType.Check(
                    base=pb.CheckModelType.Check(),
                    measurements=[
                        jit_pb.JitGadgetType.PresentMeasurement(
                            input_port=0, measurement_index=0
                        ),
                        jit_pb.JitGadgetType.PresentMeasurement(measurement_index=0),
                    ],
                ),
                jit_pb.JitGadgetType.Check(
                    base=pb.CheckModelType.Check(),
                    measurements=[
                        jit_pb.JitGadgetType.PresentMeasurement(
                            input_port=0, measurement_index=1
                        ),
                        jit_pb.JitGadgetType.PresentMeasurement(measurement_index=1),
                    ],
                ),
            ],
            unfinished_checks=[
                jit_pb.JitGadgetType.Check(
                    base=pb.CheckModelType.Check(),
                    measurements=[
                        jit_pb.JitGadgetType.PresentMeasurement(measurement_index=0)
                    ],
                ),
                jit_pb.JitGadgetType.Check(
                    base=pb.CheckModelType.Check(),
                    measurements=[
                        jit_pb.JitGadgetType.PresentMeasurement(measurement_index=1)
                    ],
                ),
            ],
        ),
    ],
)


def test_basic_jit_compile() -> None:
    jit_library = deepcopy(basic_jit_library)
    jit_library.program.append(jit_pb.JitInstruction(gadget=pb.Gadget(gtype=1)))
    jit_library.program.append(
        jit_pb.JitInstruction(
            gadget=pb.Gadget(gtype=2, connectors=[pb.Gadget.Connector(gid=1, port=0)])
        )
    )
    library = static_jit_compiler(jit_library)

    assert library.gadget_types[0] == jit_library.gadget_types[0].base
    assert library.gadget_types[1] == jit_library.gadget_types[1].base
    assert library.check_model_types[0] == pb.CheckModelType(
        ctype=1,
        gtype=1,
        checks=[
            pb.CheckModelType.Check(
                measurements=[
                    pb.CheckModelType.RemoteMeasurement(measurement_index=0),
                ]
            ),
            pb.CheckModelType.Check(
                measurements=[
                    pb.CheckModelType.RemoteMeasurement(measurement_index=1),
                ]
            ),
        ],
    )
    _assert_check_model_type(
        library.check_model_types[1],
        ctype=2,
        gtype=2,
        remote_gids={(1, 1)},
        checks=[
            [(1, 0), (None, 0), (None, 1)],
            [(1, 1), (None, 1), (None, 2)],
        ],
    )
    assert library.error_model_types[0] == pb.ErrorModelType(etype=1)
    assert library.error_model_types[1] == pb.ErrorModelType(etype=2)

    assert library.program[0] == pb.Instruction(gadget=pb.Gadget(gtype=1, gid=1))
    assert library.program[1] == pb.Instruction(
        check_model=pb.CheckModel(gid=1, ctype=1, cid=1)
    )
    assert library.program[2] == pb.Instruction(
        gadget=pb.Gadget(
            gtype=2, gid=2, connectors=[pb.Gadget.Connector(gid=1, port=0)]
        )
    )
    assert library.program[3] == pb.Instruction(
        check_model=pb.CheckModel(gid=2, ctype=2, cid=2)
    )
    # error models always come at last for static compilation, because unlike gadgets and
    # check models, error models are generated asynchronously and can only be safely await
    # when all the gadgets have been compiled. Note that this doesn't mean in runtime
    # the error models are generated at last: the runtime system can prompty wait for the
    # Future to complete and then immediately process the error model.
    assert library.program[4] == pb.Instruction(
        error_model=pb.ErrorModel(cid=1, etype=1, eid=1)
    )
    assert library.program[5] == pb.Instruction(
        error_model=pb.ErrorModel(cid=2, etype=2, eid=2)
    )


check_propagation_jit_library = jit_pb.JitLibrary(
    port_types=[
        jit_pb.JitPortType(
            base=pb.PortType(
                ptype=1,
                observables=[pb.PortType.Observable()] * 1,
            ),
            k=0,
            stabilizers=[jit_pb.JitPortType.Stabilizer()] * 1,
        )
    ],
    gadget_types=[
        jit_pb.JitGadgetType(
            base=pb.GadgetType(
                gtype=1,
                name="prepare",
                measurements=[pb.GadgetType.Measurement()] * 1,
                outputs=[pb.GadgetType.Port(ptype=1)],
                # 1 output observables, 0 input observables, 0 logical readouts
                correction_propagation=util_pb.BitMatrix(rows=1, cols=1, i=[], j=[]),
                readout_propagation=util_pb.BitMatrix(rows=0, cols=1),
                logical_correction=util_pb.BitMatrix(rows=0, cols=0),
                physical_correction=util_pb.BitMatrix(rows=1, cols=1),
            ),
            finished_checks=[
                jit_pb.JitGadgetType.Check(
                    base=pb.CheckModelType.Check(),
                    measurements=[
                        jit_pb.JitGadgetType.PresentMeasurement(measurement_index=0)
                    ],
                ),
            ],
            unfinished_checks=[
                jit_pb.JitGadgetType.Check(
                    base=pb.CheckModelType.Check(),
                    measurements=[
                        jit_pb.JitGadgetType.PresentMeasurement(measurement_index=0)
                    ],
                ),
            ],
        ),
        jit_pb.JitGadgetType(
            base=pb.GadgetType(
                gtype=2,
                name="merger",  # no measurement, just for testing
                inputs=[pb.GadgetType.Port(ptype=1), pb.GadgetType.Port(ptype=1)],
                outputs=[pb.GadgetType.Port(ptype=1)],
                # 1 output observables, 2 input observables, 0 logical readouts
                correction_propagation=util_pb.BitMatrix(rows=1, cols=3, i=[], j=[]),
                readout_propagation=util_pb.BitMatrix(rows=0, cols=3),
                logical_correction=util_pb.BitMatrix(rows=1, cols=0),
                physical_correction=util_pb.BitMatrix(rows=1, cols=0),
            ),
            unfinished_checks=[
                jit_pb.JitGadgetType.Check(
                    base=pb.CheckModelType.Check(),
                    measurements=[
                        jit_pb.JitGadgetType.PresentMeasurement(
                            input_port=0, measurement_index=0
                        ),
                        jit_pb.JitGadgetType.PresentMeasurement(
                            input_port=1, measurement_index=0
                        ),
                    ],
                ),
            ],
        ),
        jit_pb.JitGadgetType(
            base=pb.GadgetType(
                gtype=3,
                name="measure",
                measurements=[pb.GadgetType.Measurement()] * 1,
                inputs=[pb.GadgetType.Port(ptype=1)],
                # 0 output observables, 1 input observables, 1 logical readouts
                correction_propagation=util_pb.BitMatrix(rows=0, cols=2),
                readout_propagation=util_pb.BitMatrix(rows=1, cols=2),
                logical_correction=util_pb.BitMatrix(rows=0, cols=1),
                physical_correction=util_pb.BitMatrix(rows=0, cols=1),
                readouts=[pb.GadgetType.Readout(measurement_indices=[0])],
            ),
            finished_checks=[
                jit_pb.JitGadgetType.Check(
                    base=pb.CheckModelType.Check(),
                    measurements=[
                        jit_pb.JitGadgetType.PresentMeasurement(
                            input_port=0, measurement_index=0
                        ),
                        jit_pb.JitGadgetType.PresentMeasurement(measurement_index=0),
                    ],
                ),
            ],
        ),
    ],
)


def test_check_propagation_compile_two() -> None:
    jit_library = deepcopy(check_propagation_jit_library)
    jit_library.program.append(jit_pb.JitInstruction(gadget=pb.Gadget(gtype=1)))
    jit_library.program.append(jit_pb.JitInstruction(gadget=pb.Gadget(gtype=1)))
    jit_library.program.append(
        jit_pb.JitInstruction(
            gadget=pb.Gadget(
                gtype=2,
                connectors=[
                    pb.Gadget.Connector(gid=1, port=0),
                    pb.Gadget.Connector(gid=2, port=0),
                ],
            )
        )
    )
    jit_library.program.append(
        jit_pb.JitInstruction(
            gadget=pb.Gadget(gtype=3, connectors=[pb.Gadget.Connector(gid=3, port=0)])
        )
    )
    library = static_jit_compiler(jit_library)

    assert library.check_model_types[0] == pb.CheckModelType(
        ctype=1,
        gtype=1,
        checks=[
            pb.CheckModelType.Check(
                measurements=[pb.CheckModelType.RemoteMeasurement(measurement_index=0)]
            )
        ],
    )
    assert library.check_model_types[1] == pb.CheckModelType(
        ctype=2,
        gtype=1,
        checks=[
            pb.CheckModelType.Check(
                measurements=[pb.CheckModelType.RemoteMeasurement(measurement_index=0)]
            )
        ],
    )
    assert len(library.check_model_types[2].checks) == 0
    _assert_check_model_type(
        library.check_model_types[3],
        ctype=4,
        gtype=3,
        remote_gids={(1, 1), (2, 1)},
        checks=[
            [(1, 0), (2, 0), (None, 0)],
        ],
    )


def test_check_propagation_compile_three() -> None:
    jit_library = deepcopy(check_propagation_jit_library)
    jit_library.program.append(jit_pb.JitInstruction(gadget=pb.Gadget(gtype=1)))
    jit_library.program.append(jit_pb.JitInstruction(gadget=pb.Gadget(gtype=1)))
    jit_library.program.append(jit_pb.JitInstruction(gadget=pb.Gadget(gtype=1)))
    jit_library.program.append(
        jit_pb.JitInstruction(
            gadget=pb.Gadget(
                gtype=2,
                connectors=[
                    pb.Gadget.Connector(gid=2, port=0),
                    pb.Gadget.Connector(gid=3, port=0),
                ],
            )
        )
    )
    jit_library.program.append(
        jit_pb.JitInstruction(
            gadget=pb.Gadget(
                gtype=2,
                connectors=[
                    pb.Gadget.Connector(gid=4, port=0),
                    pb.Gadget.Connector(gid=1, port=0),
                ],
            )
        )
    )
    jit_library.program.append(
        jit_pb.JitInstruction(
            gadget=pb.Gadget(gtype=3, connectors=[pb.Gadget.Connector(gid=5, port=0)])
        )
    )
    library = static_jit_compiler(jit_library)

    assert len(library.check_model_types[0].checks) == 1
    assert len(library.check_model_types[1].checks) == 1
    assert len(library.check_model_types[2].checks) == 1
    assert len(library.check_model_types[3].checks) == 0
    assert len(library.check_model_types[4].checks) == 0
    assert len(library.check_model_types[5].checks) == 1
    _assert_check_model_type(
        library.check_model_types[5],
        ctype=6,
        gtype=3,
        remote_gids={(1, 1), (2, 1), (3, 1)},
        checks=[
            [(1, 0), (2, 0), (3, 0), (None, 0)],
        ],
    )


def test_check_propagation_compile_four() -> None:
    jit_library = deepcopy(check_propagation_jit_library)
    jit_library.program.append(jit_pb.JitInstruction(gadget=pb.Gadget(gtype=1)))
    jit_library.program.append(jit_pb.JitInstruction(gadget=pb.Gadget(gtype=1)))
    jit_library.program.append(jit_pb.JitInstruction(gadget=pb.Gadget(gtype=1)))
    jit_library.program.append(jit_pb.JitInstruction(gadget=pb.Gadget(gtype=1)))
    jit_library.program.append(
        jit_pb.JitInstruction(
            gadget=pb.Gadget(
                gtype=2,
                connectors=[
                    pb.Gadget.Connector(gid=1, port=0),
                    pb.Gadget.Connector(gid=2, port=0),
                ],
            )
        )
    )
    jit_library.program.append(
        jit_pb.JitInstruction(
            gadget=pb.Gadget(
                gtype=2,
                connectors=[
                    pb.Gadget.Connector(gid=3, port=0),
                    pb.Gadget.Connector(gid=4, port=0),
                ],
            )
        )
    )
    jit_library.program.append(
        jit_pb.JitInstruction(
            gadget=pb.Gadget(
                gtype=2,
                connectors=[
                    pb.Gadget.Connector(gid=5, port=0),
                    pb.Gadget.Connector(gid=6, port=0),
                ],
            )
        )
    )
    jit_library.program.append(
        jit_pb.JitInstruction(
            gadget=pb.Gadget(gtype=3, connectors=[pb.Gadget.Connector(gid=7, port=0)])
        )
    )
    library = static_jit_compiler(jit_library)

    assert len(library.check_model_types[0].checks) == 1
    assert len(library.check_model_types[1].checks) == 1
    assert len(library.check_model_types[2].checks) == 1
    assert len(library.check_model_types[3].checks) == 1
    assert len(library.check_model_types[4].checks) == 0
    assert len(library.check_model_types[5].checks) == 0
    assert len(library.check_model_types[6].checks) == 0
    assert len(library.check_model_types[7].checks) == 1
    _assert_check_model_type(
        library.check_model_types[7],
        ctype=8,
        gtype=3,
        remote_gids={(1, 1), (2, 1), (3, 1), (4, 1)},
        checks=[
            [(1, 0), (2, 0), (3, 0), (4, 0), (None, 0)],
        ],
    )


def _sort_key(m: tuple[Optional[int], int]) -> tuple[int, int]:
    return (-1 if m[0] is None else m[0], m[1])


def _resolve_measurements(
    cmt: pb.CheckModelType,
) -> list[list[tuple[Optional[int], int]]]:
    """Resolve check measurements through remote_gadget references.

    Returns a sorted list of sorted measurement lists, where each measurement is
    represented as (absolute_gid_or_None, measurement_index). This makes the
    comparison independent of the non-deterministic ordering produced by the
    async Rust JIT compiler.
    """
    resolved_checks = []
    for check in cmt.checks:
        resolved = []
        for m in check.measurements:
            if m.HasField("remote_gadget"):
                rg = cmt.remote_gadgets[m.remote_gadget]
                gid: Optional[int] = (
                    rg.absolute_gid if rg.HasField("absolute_gid") else None
                )
                resolved.append((gid, m.measurement_index))
            else:
                resolved.append((None, m.measurement_index))
        resolved_checks.append(sorted(resolved, key=_sort_key))
    return sorted(resolved_checks, key=lambda c: [_sort_key(m) for m in c])


def _assert_check_model_type(
    actual: pb.CheckModelType,
    ctype: int,
    gtype: int,
    remote_gids: set[tuple[int, int]],
    checks: list[list[tuple[Optional[int], int]]],
) -> None:
    """Assert a CheckModelType matches expected values, ignoring ordering.

    Args:
        actual: The CheckModelType to check.
        ctype: Expected ctype.
        gtype: Expected gtype.
        remote_gids: Expected set of (absolute_gid, expecting_gtype).
        checks: Expected checks as list of measurement lists, where each
            measurement is (absolute_gid_or_None, measurement_index).
    """
    assert actual.ctype == ctype
    assert actual.gtype == gtype
    actual_rgs = {(rg.absolute_gid, rg.expecting_gtype) for rg in actual.remote_gadgets}
    assert actual_rgs == remote_gids
    assert _resolve_measurements(actual) == sorted(
        (sorted(c, key=_sort_key) for c in checks),
        key=lambda c: [_sort_key(m) for m in c],
    )
