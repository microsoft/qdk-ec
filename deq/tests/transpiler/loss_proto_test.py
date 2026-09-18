"""Tests for single-gadget loss protobuf metadata."""

import deq.proto.deq_bin_pb2 as bin_pb
import deq.proto.deq_jit_pb2 as jit_pb


def test_loss_model_round_trip() -> None:
    gadget = jit_pb.JitGadgetType(
        base=bin_pb.GadgetType(
            measurements=[bin_pb.GadgetType.Measurement() for _ in range(2)],
            loss_model=bin_pb.GadgetType.LossModel(
                losses=[
                    bin_pb.GadgetType.LossModel.Loss(
                        probability=0.1,
                        continuation_errors=[0],
                        source_errors=[1],
                        child_losses=[1],
                    ),
                    bin_pb.GadgetType.LossModel.Loss(
                        probability=0.2,
                        continuation_errors=[2],
                        source_errors=[0],
                        child_output_qubits=[1],
                        loss_measurements=[1],
                    ),
                ],
                input_losses=[
                    bin_pb.GadgetType.LossModel.InputLoss(),
                    bin_pb.GadgetType.LossModel.InputLoss(
                        continuation_errors=[2],
                        child_losses=[0],
                        child_output_qubits=[0],
                        loss_measurements=[0],
                    ),
                ],
            ),
        ),
        errors=[
            jit_pb.JitGadgetType.Error(
                base=bin_pb.ErrorModelType.Error(residual=[0], probability=0.0),
                finished_checks=[0],
            ),
            jit_pb.JitGadgetType.Error(
                base=bin_pb.ErrorModelType.Error(residual=[1], probability=0.0),
            ),
            jit_pb.JitGadgetType.Error(
                base=bin_pb.ErrorModelType.Error(readout_flips=[0], probability=0.0),
            ),
        ],
    )

    decoded = jit_pb.JitGadgetType.FromString(gadget.SerializeToString())

    assert decoded == gadget
    loss_model = decoded.base.loss_model
    assert loss_model.losses[0].child_losses == [1]
    assert loss_model.losses[0].continuation_errors == [0]
    assert loss_model.losses[1].child_output_qubits == [1]
    assert loss_model.input_losses[1].continuation_errors == [2]
    assert loss_model.input_losses[1].child_losses == [0]
    assert loss_model.input_losses[1].child_output_qubits == [0]
    assert loss_model.input_losses[1].loss_measurements == [0]


def test_port_type_records_physical_qubit_count() -> None:
    port = jit_pb.JitPortType(base=bin_pb.PortType(ptype=1), k=1, n=7)

    decoded = jit_pb.JitPortType.FromString(port.SerializeToString())

    assert decoded == port
    assert decoded.k == 1
    assert decoded.n == 7
