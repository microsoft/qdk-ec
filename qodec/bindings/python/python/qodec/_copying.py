"""Copy Python object graphs without serializing or losing shared children."""

from copy import copy, deepcopy
from typing import Any

from ._collections import _View, _plain

_FIELDS = {
    "Qodec": ("layers", "name", "description", "schema_version", "metadata"),
    "Layer": ("instruction_set", "gadgets", "codes"),
    "Code": ("name", "stabilizers", "x", "z", "description", "metadata"),
    "InstructionSet": ("name", "description", "blocks", "instructions", "metadata"),
    "Instruction": ("mnemonic", "description", "inputs", "outputs", "flags", "parameters", "action", "metadata"),
    "Circuit": ("instruction_set", "source", "format"),
    "Encoding": ("code", "support", "block_types"),
    "Gadget": ("implements", "circuit", "inputs", "outputs", "checks", "readouts", "frames", "parameter_bindings", "metadata"),
    "Block": ("name", "encodes"),
    "BlockOperand": ("block", "is_variadic"),
    "Parameter": ("name", "kind"),
    "Condition": ("predicates", "invert"),
    "Stabilize": ("operators", "condition"),
    "Clifford": ("generators", "condition"),
    "Pauli": ("operator", "condition"),
    "Observe": ("observables",),
    "Rotate": ("pauli", "angle", "condition"),
    "InstructionCall": ("mnemonic", "operands", "arguments", "select"),
    "Reference": ("value",),
}

_CHILDREN = {
    "Qodec": ("layers",),
    "Layer": ("instruction_set", "gadgets", "codes"),
    "Code": (),
    "InstructionSet": ("instructions",),
    "Instruction": (),
    "Circuit": ("instruction_set",),
    "Encoding": ("code",),
    "Gadget": ("implements", "circuit", "inputs", "outputs"),
    "Block": (),
    "BlockOperand": (),
    "Parameter": (),
    "Condition": (),
    "Stabilize": (),
    "Clifford": (),
    "Pauli": (),
    "Observe": (),
    "Rotate": (),
    "InstructionCall": ("operands", "arguments"),
    "Readout": (),
    "Outcome": (),
    "Flag": (),
    "Reference": (),
}


def _deepcopy(owner: Any, memo: dict[int, Any]) -> Any:
    if id(owner) in memo:
        return memo[id(owner)]
    children = _CHILDREN[type(owner).__name__]
    result = copy(owner)
    memo[id(owner)] = result
    for field in children:
        value = getattr(owner, field)
        setattr(result, field, deepcopy(value._read() if isinstance(value, _View) else value, memo))
    return result


def _replace(owner: Any, changes: dict[str, Any]) -> Any:
    name = type(owner).__name__
    fields = _FIELDS[name]
    unknown = changes.keys() - set(fields)
    if unknown:
        raise TypeError(f"{name} has no replaceable field {sorted(unknown)[0]!r}")
    if name == "Qodec":
        values = owner._replace_fields() | changes
    elif name == "Reference":
        values = {"value": owner} | changes
    else:
        values = {
            field: _plain(getattr(owner, field)) for field in fields if field not in changes
        } | changes
    result = type(owner)(**values)
    if name == "Qodec":
        owner._copy_history_to(result)
    return result