from pathlib import Path
import common
import deq.proto.deq_bin_pb2 as pb2
import deq.proto.visualizer_pb2 as vis_pb


def main():
    library = pb2.Library(
        gadget_types=[
            pb2.GadgetType(
                gtype=1,
                name="⌛",
                description="Some circuits",
                inputs=[
                    pb2.GadgetType.Port(
                        ptype=1, tag="input", relative=vis_pb.Position(t=-0.02)
                    )
                ],
                mesh=[
                    vis_pb.Mesh(
                        geometry=vis_pb.Geometry(type="box", size=[1, 1, 1]),
                        material=vis_pb.Material(type="transmission", color="#FF8F20"),
                        relative=vis_pb.Position(t=0.5),
                    ),
                    vis_pb.Mesh(
                        geometry=vis_pb.Geometry(type="html", size=[10]),
                        material=vis_pb.Material(type="Some text ⌛"),
                        relative=vis_pb.Position(t=0.5, i=0.6),
                    ),
                ],
                realization=vis_pb.Realization(
                    locations=[
                        vis_pb.Location(
                            t=0,
                            operation=vis_pb.Operation(
                                type=vis_pb.OperationType.PREPARE,
                                support=[0],
                                pauli="+X",
                            ),
                        ),
                        vis_pb.Location(
                            t=0,
                            operation=vis_pb.Operation(
                                type=vis_pb.OperationType.PREPARE,
                                support=[1],
                                pauli="+Y",
                            ),
                        ),
                        vis_pb.Location(
                            t=0,
                            operation=vis_pb.Operation(
                                type=vis_pb.OperationType.PREPARE,
                                support=[2],
                                pauli="+Z",
                            ),
                        ),
                        vis_pb.Location(
                            t=1,
                            operation=vis_pb.Operation(
                                type=vis_pb.OperationType.HADAMARD, support=[1]
                            ),
                        ),
                        vis_pb.Location(
                            t=1,
                            operation=vis_pb.Operation(  # sqrt(Y)
                                type=vis_pb.OperationType.SQRT_PAULI,
                                support=[0],
                                pauli="Y",
                            ),
                        ),
                        vis_pb.Location(
                            t=1,
                            operation=vis_pb.Operation(  # sqrt(Z)
                                type=vis_pb.OperationType.SQRT_PAULI,
                                support=[2],
                                pauli="Z",
                            ),
                        ),
                        vis_pb.Location(
                            t=2,
                            operation=vis_pb.Operation(  # sqrt(X)
                                type=vis_pb.OperationType.SQRT_PAULI,
                                support=[2],
                                pauli="X",
                            ),
                        ),
                        vis_pb.Location(
                            t=2,
                            operation=vis_pb.Operation(
                                type=vis_pb.OperationType.PREPARE,
                                support=[3],
                                pauli="+Z",
                            ),
                        ),
                        vis_pb.Location(
                            t=2,
                            operation=vis_pb.Operation(  # CNOT
                                type=vis_pb.OperationType.CONTROLLED_PAULI,
                                support=[0, 1],
                                pauli="X",
                            ),
                        ),
                        vis_pb.Location(
                            t=3,
                            operation=vis_pb.Operation(  # CZ
                                type=vis_pb.OperationType.CONTROLLED_PAULI,
                                support=[1, 3],
                                pauli="Z",
                            ),
                            noises=[
                                vis_pb.NoiseDistribution(
                                    masses=[
                                        vis_pb.NoiseMass(
                                            faults=[
                                                vis_pb.Fault(type="X", qubit=1),
                                                vis_pb.Fault(type="Y", qubit=3),
                                            ],
                                            probability=0.1,
                                        ),
                                        vis_pb.NoiseMass(
                                            faults=[
                                                vis_pb.Fault(type="Z", qubit=3),
                                            ],
                                            probability=0.2,
                                        ),
                                    ]
                                )
                            ],
                        ),
                        vis_pb.Location(
                            t=3,
                            operation=vis_pb.Operation(  # controlled Y
                                type=vis_pb.OperationType.CONTROLLED_PAULI,
                                support=[2, 0],
                                pauli="Y",
                            ),
                        ),
                        vis_pb.Location(
                            t=4,
                            operation=vis_pb.Operation(  # measure
                                type=vis_pb.OperationType.MEASURE,
                                support=[1],
                                pauli='+Y["label"]',
                            ),
                        ),
                        vis_pb.Location(
                            t=4,
                            operation=vis_pb.Operation(  # measure
                                type=vis_pb.OperationType.MEASURE,
                                support=[2, 3],
                                pauli='-XZ%+IX["label1","label2"]',
                            ),
                        ),
                        vis_pb.Location(
                            t=5,
                            operation=vis_pb.Operation(
                                type=vis_pb.OperationType.DISCARD,
                                support=[0],
                            ),
                        ),
                        vis_pb.Location(
                            t=5,
                            operation=vis_pb.Operation(
                                type=vis_pb.OperationType.PAULI,
                                support=[1],
                                pauli="X",
                            ),
                        ),
                        vis_pb.Location(
                            t=5,
                            operation=vis_pb.Operation(
                                type=vis_pb.OperationType.PAULI,
                                support=[2],
                                pauli="Y",
                            ),
                        ),
                        vis_pb.Location(
                            t=5,
                            operation=vis_pb.Operation(
                                type=vis_pb.OperationType.PAULI,
                                support=[3],
                                pauli="Z",
                            ),
                        ),
                        vis_pb.Location(
                            t=6,
                            operation=vis_pb.Operation(
                                type=vis_pb.OperationType.CONDITIONAL_PAULI,
                                support=[3],
                                pauli="-Y",
                            ),
                        ),
                        vis_pb.Location(
                            t=6,
                            operation=vis_pb.Operation(
                                type=vis_pb.OperationType.PREPARE,
                                support=[0],
                                pauli="+X",
                            ),
                        ),
                        vis_pb.Location(
                            t=7,
                            operation=vis_pb.Operation(
                                type=vis_pb.OperationType.DISCARD,
                                support=[0],
                            ),
                        ),
                        vis_pb.Location(
                            t=7,
                            operation=vis_pb.Operation(
                                type=vis_pb.OperationType.SHUFFLE,
                                support=[1, 2, 2, 3, 3, 1],
                            ),
                        ),
                    ],
                    positions=[
                        vis_pb.Position2D(i=-0.5, j=-0.5),
                        vis_pb.Position2D(i=0.5, j=-0.5),
                        vis_pb.Position2D(i=0, j=0.5),
                        vis_pb.Position2D(i=0, j=1.5),
                    ],
                ),
            ),
        ],
        port_types=[
            pb2.PortType(
                ptype=1,
                name="▩",
                description="some port",
                mesh=[
                    vis_pb.Mesh(
                        geometry=vis_pb.Geometry(type="box", size=[0.02, 0.8, 0.8]),
                        material=vis_pb.Material(type="standard", color="#00FFFF"),
                        relative=vis_pb.Position(t=0.01),
                    )
                ],
                positions=[
                    vis_pb.Position(i=0, j=0),
                    vis_pb.Position(i=0, j=1),
                    vis_pb.Position(i=0, j=2),
                ],
            )
        ],
        program=[
            pb2.Instruction(gadget=pb2.Gadget(gtype=1, position=vis_pb.Position(t=0))),
        ],
        visual_config=vis_pb.VisualConfig(qubit_radius=0.2),
    )

    common.generate_example(library, Path(__file__).stem)


if __name__ == "__main__":
    main()
