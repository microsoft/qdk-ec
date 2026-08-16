"""Bidirectional codec between ``LOSS(...)`` syntax and runtime loss metadata.

This pass does **not** analyze ``LOSS_ERROR`` instructions. It only packs the
explicit ``LOSS`` statements a user (or the annotator) writes into
``GadgetType.LossModel``, mirroring how ``ERROR`` statements are packed by
:func:`deq.transpiler.jit_library_builder._build_errors`. Deriving a loss
model from ``LOSS_ERROR`` circuit analysis is a separate transpiler pass.

Index conventions (all local to the gadget):

- ``SE<k>`` / ``CE<k>`` index ``JitGadgetType.errors``;
- ``L<k>`` indexes the source ``losses`` list (a ``LOSS(p) ...`` statement's
  position among source losses, in body order);
- ``OUT<i>.L<j>`` is output port ``i``, physical qubit ``j``; it is flattened
  to a position in ``[0, sum of output-port n)`` using each code's ``n``;
- ``LOSS(IN<i>.L<j>)`` places the input continuation at the flat position
  ``offset(input_port_i) + j`` of the ``input_losses`` array, which always has
  exactly ``sum of input-port n`` entries.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import deq.proto.deq_bin_pb2 as bin_pb
from deq.circuit.model import (
    CodeDefinition,
    GadgetDefinition,
    Instruction,
    InputPort,
    LossStatement,
    OutputPort,
)
from deq.transpiler.jit_transpiler import flatten_body


@dataclass(frozen=True)
class PhysicalPortLayout:
    """Flatten and unflatten physical-qubit positions across ordered ports."""

    ports: Sequence[InputPort | OutputPort]
    codes: dict[str, CodeDefinition]

    def __post_init__(self) -> None:
        offsets: list[int] = []
        running = 0
        for port in self.ports:
            offsets.append(running)
            running += self.codes[port.code_name].n
        object.__setattr__(self, "offsets", tuple(offsets))
        object.__setattr__(self, "size", running)

    def flatten(self, port: int, qubit: int, *, label: str) -> int:
        if not 0 <= port < len(self.ports):
            raise ValueError(
                f"{label} references port {port}, but there are "
                f"{len(self.ports)} ports"
            )
        count = self.codes[self.ports[port].code_name].n
        if not 0 <= qubit < count:
            raise ValueError(
                f"{label} references physical qubit {qubit} on port {port}, "
                f"which has {count}"
            )
        return self.offsets[port] + qubit

    def unflatten(self, slot: int, *, label: str) -> tuple[int, int]:
        if not 0 <= slot < self.size:
            raise ValueError(
                f"{label} flat position {slot} is outside [0, {self.size})"
            )
        for port, offset in enumerate(self.offsets):
            count = self.codes[self.ports[port].code_name].n
            if slot < offset + count:
                return port, slot - offset
        raise AssertionError("validated slot must resolve to a port")

    def qubit_slots(self) -> dict[int, int]:
        return {
            qubit: self.flatten(port, position, label="port qubit")
            for port, item in enumerate(self.ports)
            for position, qubit in enumerate(item.qubit_indices)
        }

    def qubit_coordinates(self) -> dict[int, tuple[int, int]]:
        return {
            qubit: (port, position)
            for port, item in enumerate(self.ports)
            for position, qubit in enumerate(item.qubit_indices)
        }


def transpile_declared_loss_model(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
    *,
    num_errors: int,
    num_measurements: int,
) -> bin_pb.GadgetType.LossModel | None:
    """Build a ``LossModel`` from a gadget's explicit ``LOSS`` statements.

    Returns ``None`` when the gadget declares no ``LOSS`` statements, so the
    ``loss_model`` field is left unset for loss-free gadgets.
    """
    flat = flatten_body(list(gadget.body))
    loss_statements = [s for s in flat if isinstance(s, LossStatement)]
    if not loss_statements:
        return None

    if any(isinstance(s, Instruction) and s.name.upper() == "LOSS_ERROR" for s in flat):
        raise ValueError(
            f"GADGET {gadget.name!r} mixes explicit LOSS statements with a "
            f"LOSS_ERROR instruction; comment out LOSS_ERROR (or drop the LOSS "
            f"statements) so the loss model has a single source of truth"
        )

    input_layout = PhysicalPortLayout(gadget.input_ports, codes)
    output_layout = PhysicalPortLayout(gadget.output_ports, codes)

    source_statements = [s for s in loss_statements if not s.is_input]
    input_statements = [s for s in loss_statements if s.is_input]
    num_source = len(source_statements)

    def _flat_output(statement: LossStatement, port: int, qubit: int) -> int:
        return output_layout.flatten(
            port,
            qubit,
            label=f"GADGET {gadget.name!r}: {statement} output",
        )

    def _validate_refs(
        statement: LossStatement,
        *,
        source_index: int | None = None,
    ) -> None:
        def _validate_unique(references: Sequence[str], label: str) -> None:
            seen: set[str] = set()
            for reference in references:
                if reference in seen:
                    raise ValueError(
                        f"GADGET {gadget.name!r}: {statement} contains duplicate "
                        f"{label} reference {reference}"
                    )
                seen.add(reference)

        _validate_unique(
            [f"SE{index}" for index in statement.source_errors],
            "source-error",
        )
        _validate_unique(
            [f"CE{index}" for index in statement.continuation_errors],
            "continuation-error",
        )
        _validate_unique(
            [f"L{index}" for index in statement.child_losses],
            "child-loss",
        )
        _validate_unique(
            [f"OUT{port}.L{qubit}" for port, qubit in statement.output_qubits],
            "output-qubit",
        )
        _validate_unique(
            [f"M{index}" for index in statement.measurement_indices],
            "measurement",
        )
        for index in (
            *statement.source_errors,
            *statement.continuation_errors,
        ):
            if not 0 <= index < num_errors:
                raise ValueError(
                    f"GADGET {gadget.name!r}: {statement} references error index "
                    f"{index}, but the gadget has {num_errors} errors"
                )
        for index in statement.child_losses:
            if not 0 <= index < num_source:
                raise ValueError(
                    f"GADGET {gadget.name!r}: {statement} references child loss "
                    f"L{index}, but the gadget has {num_source} source losses"
                )
            if source_index is not None and index <= source_index:
                raise ValueError(
                    f"GADGET {gadget.name!r}: source loss L{source_index} "
                    f"references child L{index}; source children must have "
                    "greater indices"
                )
        for index in statement.measurement_indices:
            if not 0 <= index < num_measurements:
                raise ValueError(
                    f"GADGET {gadget.name!r}: {statement} references measurement "
                    f"M{index}, but the gadget has {num_measurements} measurements"
                )

    losses_pb: list[bin_pb.GadgetType.LossModel.Loss] = []
    for source_index, statement in enumerate(source_statements):
        _validate_refs(statement, source_index=source_index)
        losses_pb.append(
            bin_pb.GadgetType.LossModel.Loss(
                probability=statement.probability,
                continuation_errors=sorted(statement.continuation_errors),
                source_errors=sorted(statement.source_errors),
                child_losses=sorted(statement.child_losses),
                child_output_qubits=sorted(
                    _flat_output(statement, port, qubit)
                    for port, qubit in statement.output_qubits
                ),
                loss_measurements=sorted(statement.measurement_indices),
            )
        )

    input_losses_pb = [
        bin_pb.GadgetType.LossModel.InputLoss() for _ in range(input_layout.size)
    ]
    occupied_slots: set[int] = set()
    for statement in input_statements:
        _validate_refs(statement)
        port = statement.input_port
        qubit = statement.input_qubit
        assert port is not None and qubit is not None
        slot = input_layout.flatten(
            port,
            qubit,
            label=f"GADGET {gadget.name!r}: {statement} input",
        )
        if slot in occupied_slots:
            raise ValueError(
                f"GADGET {gadget.name!r}: duplicate input loss for "
                f"IN{port}.L{qubit}"
            )
        occupied_slots.add(slot)
        input_losses_pb[slot] = bin_pb.GadgetType.LossModel.InputLoss(
            continuation_errors=sorted(statement.continuation_errors),
            child_losses=sorted(statement.child_losses),
            child_output_qubits=sorted(
                _flat_output(statement, port, qubit)
                for port, qubit in statement.output_qubits
            ),
            loss_measurements=sorted(statement.measurement_indices),
        )

    return bin_pb.GadgetType.LossModel(
        losses=losses_pb,
        input_losses=input_losses_pb,
    )


def loss_model_to_statements(
    loss_model: bin_pb.GadgetType.LossModel,
    *,
    input_ports: Sequence[InputPort],
    output_ports: Sequence[OutputPort],
    codes: dict[str, CodeDefinition],
    gadget_name: str,
) -> tuple[list[LossStatement], list[LossStatement]]:
    """Decode runtime loss metadata into source and input ``LOSS`` statements."""
    input_layout = PhysicalPortLayout(input_ports, codes)
    output_layout = PhysicalPortLayout(output_ports, codes)

    def output_qubits(slots) -> list[tuple[int, int]]:
        return [
            output_layout.unflatten(slot, label=f"GADGET {gadget_name!r} output loss")
            for slot in slots
        ]

    source_statements = [
        LossStatement(
            probability=loss.probability,
            source_errors=list(loss.source_errors),
            continuation_errors=list(loss.continuation_errors),
            child_losses=list(loss.child_losses),
            output_qubits=output_qubits(loss.child_output_qubits),
            measurement_indices=list(loss.loss_measurements),
        )
        for loss in loss_model.losses
    ]

    input_statements: list[LossStatement] = []
    for slot, loss in enumerate(loss_model.input_losses):
        if loss.SerializeToString() == b"":
            continue
        port, qubit = input_layout.unflatten(
            slot, label=f"GADGET {gadget_name!r} input loss"
        )
        input_statements.append(
            LossStatement(
                input_port=port,
                input_qubit=qubit,
                continuation_errors=list(loss.continuation_errors),
                child_losses=list(loss.child_losses),
                output_qubits=output_qubits(loss.child_output_qubits),
                measurement_indices=list(loss.loss_measurements),
            )
        )
    return source_statements, input_statements
