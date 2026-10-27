"""Translate supported official Stim instructions into declared instruction set calls."""

from collections.abc import Mapping, Sequence
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any

from .actions import Observe
from .instructions import InstructionCall

if TYPE_CHECKING:
    from . import Instruction, InstructionSet

_MAX_CALLS = 1_000_000
_ANNOTATIONS = {
    "DETECTOR",
    "OBSERVABLE_INCLUDE",
    "TICK",
    "QUBIT_COORDS",
    "SHIFT_COORDS",
}


def parse(source: str, instruction_set: "InstructionSet") -> list[InstructionCall]:
    try:
        stim = import_module("stim")
    except ModuleNotFoundError as error:
        if error.name != "stim":
            raise
        raise ValueError(
            "Stim parsing requires the optional dependency; install 'qodec[parsers]'"
        ) from error

    declarations = instruction_set.instructions
    widths = {block.name: block.encodes for block in instruction_set.blocks}

    def translate(native: Any) -> list[InstructionCall]:
        calls: list[InstructionCall] = []
        for operation in native:
            if isinstance(operation, stim.CircuitRepeatBlock):
                _append_repeated(
                    calls, translate(operation.body_copy()), operation.repeat_count
                )
                continue
            if operation.name in _ANNOTATIONS:
                continue
            if operation.name == "MPAD":
                _append_padding(calls, operation, instruction_set.name, declarations)
                continue
            mnemonic, data = _resolve_gate(
                operation, stim, instruction_set.name, declarations
            )
            _validate_declaration(operation.name, data, declarations[mnemonic], widths)
            _append_gate_calls(calls, operation, mnemonic)
        return calls

    try:
        return translate(stim.Circuit(source))
    except RecursionError as error:
        raise ValueError(
            "Stim repeat nesting exceeds the supported recursion depth"
        ) from error


def _resolve_gate(
    operation: Any,
    stim: ModuleType,
    instruction_set_name: str,
    declarations: Mapping[str, "Instruction"],
) -> tuple[str, Any]:
    name = operation.name
    if operation.gate_args_copy():
        raise ValueError(f"Stim gate arguments are not supported: {operation}")
    data = stim.gate_data(name)
    if not (data.is_single_qubit_gate or data.is_two_qubit_gate):
        raise ValueError(f"Stim instruction {name!r} is not supported by this adapter")
    if name in declarations:
        return name, data
    aliases = [alias for alias in data.aliases if alias in declarations]
    if not aliases:
        raise ValueError(
            f"call to unknown instruction {name!r} in instruction set {instruction_set_name!r}"
        )
    if len(aliases) != 1:
        raise ValueError(
            f"Stim gate {name!r} has ambiguous aliases in instruction set {instruction_set_name!r}: {aliases}"
        )
    return aliases[0], data


def _validate_declaration(
    name: str,
    data: Any,
    declaration: "Instruction",
    widths: Mapping[str, int],
) -> None:
    arity = 2 if data.is_two_qubit_gate else 1
    boundaries = [*declaration.inputs, *declaration.outputs]
    if max(len(declaration.inputs), len(declaration.outputs)) != arity or any(
        operand.is_variadic or widths.get(operand.block) != 1 for operand in boundaries
    ):
        raise ValueError(
            f"Stim gate {name!r} requires {arity} single-qubit block operands"
        )
    outcomes = sum(
        len(step.observables)
        for step in declaration.action
        if isinstance(step, Observe)
    )
    if declaration.flags or outcomes != int(data.produces_measurements):
        raise ValueError(
            f"Stim gate {name!r}: target instruction changes the measurement-record shape"
        )


def _append_repeated(
    calls: list[InstructionCall], body: Sequence[InstructionCall], repeat_count: int
) -> None:
    if len(calls) + len(body) * repeat_count > _MAX_CALLS:
        raise ValueError(f"Stim expansion exceeds {_MAX_CALLS} instruction calls")
    if body:
        calls.extend(
            InstructionCall(call.mnemonic, operands=call.operands)
            for _ in range(repeat_count)
            for call in body
        )


def _append_gate_calls(
    calls: list[InstructionCall], operation: Any, mnemonic: str
) -> None:
    for group in operation.target_groups():
        if any(
            not target.is_qubit_target or target.is_inverted_result_target
            for target in group
        ):
            raise ValueError(
                f"Stim record controls, inverted targets, and Pauli targets are not supported: {operation}"
            )
        if len(calls) >= _MAX_CALLS:
            raise ValueError(f"Stim expansion exceeds {_MAX_CALLS} instruction calls")
        calls.append(
            InstructionCall(mnemonic, operands=[target.value for target in group])
        )


def _append_padding(
    calls: list[InstructionCall],
    operation: Any,
    instruction_set_name: str,
    declarations: Mapping[str, "Instruction"],
) -> None:
    if any(operation.gate_args_copy()):
        raise ValueError(f"Stim MPAD noise is not supported: {operation}")
    bits = [target.value for target in operation.targets_copy()]
    for bit in sorted(set(bits)):
        mnemonic = f"MPAD{bit}"
        declaration = declarations.get(mnemonic)
        if declaration is None:
            raise ValueError(
                f"Stim MPAD {bit} requires instruction {mnemonic!r} in instruction set {instruction_set_name!r}"
            )
        steps = declaration.action
        observables = (
            steps[0].observables
            if len(steps) == 1 and isinstance(steps[0], Observe)
            else []
        )
        identities = {"", "+", "I", "+I"} if bit == 0 else {"-", "-I"}
        if (
            declaration.inputs
            or declaration.outputs
            or declaration.parameters
            or declaration.flags
            or len(observables) != 1
            or observables[0].strip() not in identities
        ):
            sign = "+I" if bit == 0 else "-I"
            raise ValueError(
                f"Stim MPAD {bit}: {mnemonic!r} must have no operands, parameters, or flags and a single observe of {sign}"
            )
    if len(calls) + len(bits) > _MAX_CALLS:
        raise ValueError(f"Stim expansion exceeds {_MAX_CALLS} instruction calls")
    calls.extend(InstructionCall(f"MPAD{bit}") for bit in bits)
