import json
import os
from collections.abc import Mapping
import hjson
from errata.circuits.clifford import (
    CliffordOperation,
    InverseOperation,
    ControlledPauli,
    ControlledX,
    ControlledY,
    ControlledZ,
    Operand,
    PauliOperation,
    ConditionalPauli,
    Hadamard,
    SqrtZ,
    SqrtX,
    SqrtY,
    Prepare,
    Measurement,
    Shuffle,
    Discard,
)
from errata.simulation.noise import LocationStochasticNoiseModel, Probability, FaultSet
from errata.simulation.fault_evaluation import Location
from errata.statistics.distribution import (
    DiscreteDistribution,
    ProductDistribution,
    ExplicitDiscreteDistribution,
)


this_dir = os.path.abspath(os.path.dirname(__file__))
import deq.proto.coordinator_pb2 as pb2
import deq.proto.visualizer_pb2 as vis_pb
from google.protobuf.json_format import MessageToJson, MessageToDict


def visualize_operation(
    operation: CliffordOperation, support_indices: Mapping[Operand, int]
) -> vis_pb.Operation:
    # iteratively remove the inverse wrapper
    inverted = False
    while isinstance(operation, InverseOperation):
        inverted = not inverted
        operation = operation._operation
    # check each class
    if isinstance(operation, ControlledPauli):
        if isinstance(operation, ControlledX):
            pauli = "X"
        elif isinstance(operation, ControlledY):
            pauli = "Y"
        elif isinstance(operation, ControlledZ):
            pauli = "Z"
        else:
            raise ValueError("Invalid controlled operation", operation)
        return vis_pb.Operation(
            type=vis_pb.CONTROLLED_PAULI,
            support=[
                support_indices[operation.control],
                support_indices[operation.target],
            ],
            pauli=pauli,
            inverted=inverted,
        )
    if isinstance(operation, PauliOperation):
        return vis_pb.Operation(
            type=vis_pb.PAULI,
            support=[support_indices[operation._qubit]],
            pauli=operation._name,
            inverted=inverted,
        )
    if isinstance(operation, ConditionalPauli):
        support = []
        pauli = "+" if operation.negate else "-"
        for operand in operation.characters:
            support.append(support_indices[operand])
            pauli += operation.characters[operand]
        if not operation.conditions:
            # there are conditions, add them as JSON string
            conditions = [str(condition) for condition in operation.conditions]
            pauli += json.dumps(conditions)
        return vis_pb.Operation(
            type=vis_pb.CONDITIONAL_PAULI,
            support=support,
            pauli=pauli,
            inverted=inverted,
        )
    if isinstance(operation, Hadamard):
        return vis_pb.Operation(
            type=vis_pb.HADAMARD,
            support=[support_indices[operation._qubit]],
            inverted=inverted,
        )
    if (
        isinstance(operation, SqrtZ)
        or isinstance(operation, SqrtX)
        or isinstance(operation, SqrtY)
    ):
        pauli = (
            "X"
            if isinstance(operation, SqrtX)
            else "Y" if isinstance(operation, SqrtY) else "Z"
        )
        return vis_pb.Operation(
            type=vis_pb.SQRT_PAULI,
            support=[support_indices[operation._qubit]],
            pauli=pauli,
            inverted=inverted,
        )
    if isinstance(operation, Prepare):
        pauli = ("+" if operation._pauli_sign else "-") + operation._pauli_label
        return vis_pb.Operation(
            type=vis_pb.PREPARE,
            support=[support_indices[operation._qubit]],
            pauli=pauli,
        )
    if isinstance(operation, Measurement):
        support = []
        observable = "-" if operation.observable.phase == -1 else "+"
        correction = "-" if operation.correction.phase == -1 else "+"
        for operand in operation.support:
            support.append(support_indices[operand])
            observable += operation.observable[operand]
            correction += operation.correction[operand]
        pauli = observable + "%" + correction
        if operation.labels:
            # there are labels, add them as JSON string
            labels = [str(label) for label in operation.labels]
            pauli += json.dumps(labels)
        return vis_pb.Operation(
            type=vis_pb.MEASURE,
            support=support,
            pauli=pauli,
            inverted=operation.is_invertible and inverted,
        )
    if isinstance(operation, Shuffle):
        support = []
        for source, target in zip(operation._keys, operation._values):
            support.append(support_indices[source])
            support.append(support_indices[target])
        return vis_pb.Operation(
            type=vis_pb.SHUFFLE,
            support=support,
            inverted=inverted,
        )
    if isinstance(operation, Discard):
        return vis_pb.Operation(
            type=vis_pb.DISCARD,
            support=[support_indices[operation._qubit]],
        )

    raise ValueError("Invalid operation", operation, type(operation))
