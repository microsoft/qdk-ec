"""Semantic validation of definition bodies, independent of the parser backend.

These checks operate on ``model.py`` objects, so they live in one place that any
producer of those objects can call. They enforce deq's structural rules that go
beyond grammar: INPUT/OUTPUT port ordering, the requirement that CONDITIONAL /
PROPAGATE follow all OUTPUT ports, PRESELECT placement, and the ban on ports
inside REPEAT blocks.
"""

from __future__ import annotations

import warnings
from typing import Any

from deq.circuit.model import (
    ConditionalStatement,
    InputPort,
    Instruction,
    MeasurementRecordTarget,
    OutputPort,
    PhysicalMeasurementTarget,
    PreselectStatement,
    PropagateStatement,
    QubitTarget,
    RepeatBlock,
)
from deq.transpiler.stim_constants import (
    ANNOTATION_INSTRUCTIONS,
    instruction_num_measurements,
)


def _at(item: Any) -> str:
    """Render a ``" (line N)"`` suffix for a statement that carries a location."""
    line = getattr(item, "source_line", None)
    return f" (line {line})" if line is not None else ""


def validate_repeat_body(body: list[Any]) -> None:
    """Raise SyntaxError if INPUT/OUTPUT ports appear inside a REPEAT block."""
    for item in body:
        match item:
            case InputPort():
                kind = "INPUT"
            case OutputPort():
                kind = "OUTPUT"
            case _:
                continue
        raise SyntaxError(f"{kind} port cannot appear inside a REPEAT block{_at(item)}")


def validate_conditional_after_output(body: list[Any], gadget_name: str) -> None:
    """Raise SyntaxError if any CONDITIONAL appears before the last OUTPUT."""
    last_output = -1
    first_conditional = len(body)
    for i, item in enumerate(body):
        match item:
            case OutputPort():
                last_output = i
            case ConditionalStatement() if i < first_conditional:
                first_conditional = i
    if first_conditional < last_output:
        raise SyntaxError(
            f"CONDITIONAL must appear after all OUTPUT statements in "
            f"GADGET {gadget_name!r}{_at(body[first_conditional])}; the logical "
            f"correction is applied at the end of the gadget, not mid-circuit"
        )


def validate_propagate_after_output(body: list[Any], gadget_name: str) -> None:
    """Raise SyntaxError if any PROPAGATE appears before the last OUTPUT."""
    last_output = -1
    first_propagate = len(body)
    for i, item in enumerate(body):
        match item:
            case OutputPort():
                last_output = i
            case PropagateStatement() if i < first_propagate:
                first_propagate = i
    if first_propagate < last_output:
        raise SyntaxError(
            f"PROPAGATE must appear after all OUTPUT statements in "
            f"GADGET {gadget_name!r}{_at(body[first_propagate])}; it pins one "
            f"row of the output correction propagation, which is meaningful "
            f"only after the OUTPUT layout is fixed"
        )


def _walk_preselect_aware(body: list[Any]) -> Any:
    """Yield (kind, item) tuples for preselect validation.

    ``kind`` is one of: 'instruction', 'preselect',
    'repeat_enter', 'repeat_exit'.
    """
    for item in body:
        match item:
            case RepeatBlock():
                yield ("repeat_enter", item)
                for sub in item.body:
                    yield from _walk_preselect_aware([sub])
                yield ("repeat_exit", item)
            case Instruction():
                yield ("instruction", item)
            case PreselectStatement():
                yield ("preselect", item)


def validate_preselect(body: list[Any], gadget_name: str) -> None:
    """Validate PRESELECT placement and data-qubit isolation.

    Rules enforced:
    1. PRESELECT cannot appear inside a REPEAT block.
    2. rec[-k] must reference an existing measurement.
    3. No instruction before the last PRESELECT may touch any qubit
       declared in an INPUT port — this ensures the retry region
       (gadget start → last PRESELECT) is isolated from data qubits
       and safe to re-execute.
    """
    # Collect INPUT qubit indices.
    input_qubits: set[int] = set()
    for item in body:
        if isinstance(item, InputPort):
            input_qubits.update(item.qubit_indices)

    # First pass: validate placement and collect the position of the
    # last PRESELECT in the flat walk order.
    cum_measurements = 0
    repeat_depth = 0
    has_preselect = False
    for kind, item in _walk_preselect_aware(body):
        if kind == "repeat_enter":
            repeat_depth += 1
        elif kind == "repeat_exit":
            repeat_depth -= 1
        elif kind == "instruction":
            cum_measurements += instruction_num_measurements(str(item))
        elif kind == "preselect":
            has_preselect = True
            if repeat_depth > 0:
                raise SyntaxError(
                    f"PRESELECT cannot appear inside a REPEAT block in "
                    f"GADGET {gadget_name!r}{_at(item)}; unroll the REPEAT or "
                    f"move the PRESELECT outside"
                )
            for cond in item.conditions:
                match cond:
                    case MeasurementRecordTarget(offset):
                        if offset < 1 or offset > cum_measurements:
                            raise SyntaxError(
                                f"PRESELECT rec[-{offset}] in GADGET {gadget_name!r}{_at(item)} "
                                f"refers to a measurement that has not occurred yet "
                                f"(only {cum_measurements} measurement(s) so far)"
                            )
                    case PhysicalMeasurementTarget(index):
                        if index < 0 or index >= cum_measurements:
                            raise SyntaxError(
                                f"PRESELECT M{index} in GADGET {gadget_name!r}{_at(item)} "
                                f"refers to a measurement that has not occurred yet "
                                f"(only {cum_measurements} measurement(s) so far)"
                            )
                    case _:
                        # InputVirtualTarget / OutputVirtualTarget — virtual
                        # stabilizer measurements are not internal physical
                        # measurements and cannot be preselected on.
                        raise SyntaxError(
                            f"PRESELECT in GADGET {gadget_name!r}{_at(item)} requires an "
                            f"internal physical measurement reference "
                            f"(rec[-k] or M<i>); virtual stabilizer measurements "
                            f"(IN<p>.S<s> / OUT<p>.S<s>) are not allowed"
                        )

    if not has_preselect or not input_qubits:
        return

    # Second pass: warn (not error) if any instruction before the last
    # PRESELECT touches an INPUT qubit.  This means the preselect retry
    # simulator cannot be used — only the resample (static) simulator
    # is safe.  We emit a warning instead of an error because the
    # resample mode handles it correctly.
    seen_last_preselect = False
    for item in reversed(body):
        if isinstance(item, PreselectStatement):
            if not seen_last_preselect:
                seen_last_preselect = True
            continue
        if not seen_last_preselect:
            continue
        if isinstance(item, Instruction):
            touched = {t.index for t in item.targets if isinstance(t, QubitTarget)}
            overlap = touched & input_qubits
            if overlap:
                warnings.warn(
                    f"Instruction '{item.name}' in GADGET {gadget_name!r} "
                    f"touches INPUT qubit(s) {sorted(overlap)} before the "
                    f"last PRESELECT. The preselect retry simulator "
                    f"(--simulator preselect) cannot safely retry this "
                    f"circuit; use --simulator static (resample mode) "
                    f"instead.",
                    stacklevel=2,
                )
                return


def validate_port_ordering(body: list[Any], gadget_name: str) -> None:
    """Raise SyntaxError if INPUT/OUTPUT ports violate the required ordering.

    The measurement layout convention requires:
      1. All INPUT ports appear before any circuit instruction.
      2. All OUTPUT ports appear after all circuit instructions.

    Gate instructions, noise instructions, and REPEAT blocks must all
    appear between INPUT and OUTPUT.  Only CHECK, READOUT, ERROR,
    CONDITIONAL, VIRTUAL, and PROPAGATE statements may appear after OUTPUT.
    """
    seen_instruction = False
    seen_output = False
    for item in body:
        match item:
            case Instruction():
                name = item.name.upper()
                if name in ANNOTATION_INSTRUCTIONS:
                    continue
                seen_instruction = True
                if seen_output:
                    raise SyntaxError(
                        f"instruction '{item.name}' appears after an OUTPUT "
                        f"port in GADGET {gadget_name!r}{_at(item)}; all OUTPUT "
                        f"ports must come after all circuit and noise instructions"
                    )
            case InputPort():
                if seen_instruction:
                    raise SyntaxError(
                        f"INPUT port appears after a circuit instruction in "
                        f"GADGET {gadget_name!r}{_at(item)}; all INPUT ports must "
                        f"come before any circuit instruction"
                    )
                if seen_output:
                    raise SyntaxError(
                        f"INPUT port appears after an OUTPUT port in "
                        f"GADGET {gadget_name!r}{_at(item)}; all INPUT ports must "
                        f"come before all OUTPUT ports"
                    )
            case OutputPort():
                seen_output = True
            case RepeatBlock():
                seen_instruction = True
                if seen_output:
                    raise SyntaxError(
                        f"REPEAT block appears after an OUTPUT port in "
                        f"GADGET {gadget_name!r}{_at(item)}; all OUTPUT ports must "
                        f"come after all circuit instructions"
                    )


def validate_gadget_body(body: list[Any], gadget_name: str) -> None:
    """Run every GADGET-body validator.

    The order is significant: each validator raises on its first violation, so
    it decides which message a body with several problems reports.
    """
    validate_port_ordering(body, gadget_name)
    validate_conditional_after_output(body, gadget_name)
    validate_propagate_after_output(body, gadget_name)
    validate_preselect(body, gadget_name)
