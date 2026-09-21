"""QuEra PPVM adapter for DEQ's extended Stim circuits.

The pinned native API is loaded without PPVM's optional Bloqade frontend,
whose dependency stack does not support every Python version DEQ supports.
Angles in DEQ are multiples of pi; PPVM's rotation API uses radians.
"""

from functools import lru_cache
from dataclasses import replace
from hashlib import blake2b
from importlib.machinery import PathFinder
from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import module_from_spec
import math
import secrets
import sys

import stim

from deq.circuit.model import Instruction, MeasurementRecordTarget, QubitTarget
from deq.circuit.parser import parse
from deq.transpiler.jit_transpiler import flatten_body, max_qubit_index
from deq.transpiler.stim_constants import (
    ANNOTATION_INSTRUCTIONS,
    NON_CLIFFORD_AXES,
    instruction_num_measurements,
)

_PPVM_INSTALL_HELP = (
    "Install the tested QuEra PPVM native package into DEQ's Python environment "
    "(requires Git and a Rust toolchain):\n"
    f'  "{sys.executable}" -m pip install --no-deps --force-reinstall '
    '"ppvm @ git+https://github.com/QuEraComputing/ppvm.git@'
    '731e59fc98489f206767822f8dc8798ba23a5551#subdirectory=ppvm-python"\n'
    "PyPI's ppvm is unrelated. --no-deps avoids the optional Bloqade frontend dependencies."
)


@lru_cache(maxsize=1)
def _load_core():
    """Load QuEra's native extension without importing the Bloqade frontend."""
    try:
        package = distribution("ppvm")
    except PackageNotFoundError as error:
        raise ImportError(
            "QuEra PPVM is not installed.\n" + _PPVM_INSTALL_HELP
        ) from error
    spec = PathFinder.find_spec("ppvm._core", [str(package.locate_file("ppvm"))])
    if spec is None or spec.loader is None:
        raise ImportError("QuEra PPVM native extension not found.\n" + _PPVM_INSTALL_HELP)
    core = module_from_spec(spec)
    spec.loader.exec_module(core)
    return core


class Sampler:
    """Sample extended Stim text using independent seeded PPVM trajectories."""

    def __init__(self, circuit_text: str, config: dict):
        core = _load_core()
        if config.get("loss_config") is not None:
            raise ValueError("PPVM does not implement QDK loss_config overrides")
        self._seed = int(config.get("seed", secrets.randbits(64)))
        self._shot_index = int(config.get("skip_shots", 0))
        self._min_abs_coeff = float(config.get("min_abs_coeff", 0.0))
        if not 0 <= self._seed < 2**64 or self._shot_index < 0:
            raise ValueError("seed must fit u64 and skip_shots must be nonnegative")
        if not math.isfinite(self._min_abs_coeff) or self._min_abs_coeff < 0:
            raise ValueError("min_abs_coeff must be finite and nonnegative")
        try:
            gadget = parse("GADGET Sampling {\n" + circuit_text + "\n}").definitions[0]
        except SyntaxError as error:
            raise ValueError(
                "PPVM sampler requires a flat or REPEAT extended-Stim circuit; "
                "SELECT/REQUIRE preselection is not supported"
            ) from error
        body = flatten_body(gadget.body)
        self._num_qubits = max(1, max_qubit_index(body) + 1)
        if self._num_qubits > 2048:
            raise ValueError("PPVM Python bindings support at most 2048 qubits")
        self._tableau_class = getattr(core, f"GeneralizedTableau{(self._num_qubits + 63) // 64}")
        self._operations = []
        self._inverted_measurements = set()
        pending = []
        measurement_count = 0

        def flush():
            if pending:
                self._operations.append(("stim", core.StimProgram.parse("\n".join(pending))))
                pending.clear()

        for statement in body:
            if not isinstance(statement, Instruction):
                continue
            name = statement.name.upper()
            if name in ANNOTATION_INSTRUCTIONS:
                continue
            if any(getattr(target, "inverted", False) for target in statement.targets):
                instruction = stim.CircuitInstruction(str(statement))
                for index, group in enumerate(instruction.target_groups()):
                    if sum(target.is_inverted_result_target for target in group) % 2:
                        self._inverted_measurements.add(measurement_count + index)
                statement = replace(statement, targets=[
                    replace(target, inverted=False) if getattr(target, "inverted", False) else target
                    for target in statement.targets
                ])
            if name in NON_CLIFFORD_AXES:
                flush()
                if name in {"R_X", "R_Y", "R_Z"}:
                    angle = statement.arguments[0]
                elif name in {"T", "TX", "TY"}:
                    angle = 0.25
                elif name in {"T_DAG", "TX_DAG", "TY_DAG"}:
                    angle = -0.25
                else:
                    raise ValueError(f"PPVM does not support non-Clifford gate {name}")
                self._operations.append((
                    "rotation", "r" + NON_CLIFFORD_AXES[name].lower(),
                    [target.index for target in statement.targets], math.pi * angle,
                ))
            elif any(isinstance(target, MeasurementRecordTarget) for target in statement.targets):
                flush()
                if name not in {"CX", "CNOT", "ZCX", "CY", "ZCY", "CZ", "ZCZ"}:
                    raise ValueError(f"PPVM does not support record controls on {name}")
                axis = {"CNOT": "X", "ZCX": "X", "ZCY": "Y", "ZCZ": "Z"}.get(name, name[-1])
                if len(statement.targets) % 2:
                    raise ValueError(f"{name} requires target pairs")
                for first, second in zip(statement.targets[::2], statement.targets[1::2]):
                    if axis == "Z" and isinstance(second, MeasurementRecordTarget):
                        first, second = second, first
                    if isinstance(first, MeasurementRecordTarget) and isinstance(second, QubitTarget):
                        if not 1 <= first.offset <= measurement_count:
                            raise ValueError("record control refers before the measurement record")
                        self._operations.append(("conditional", axis.lower(), first.offset, second.index))
                    elif isinstance(first, QubitTarget) and isinstance(second, QubitTarget):
                        self._operations.append(("stim", core.StimProgram.parse(f"{name} {first} {second}")))
                    else:
                        raise ValueError(f"unsupported {name} record-control targets")
            else:
                pending.append(str(Instruction(name=name, arguments=statement.arguments, targets=statement.targets)))
            measurement_count += instruction_num_measurements(str(statement))
        flush()
        self._num_measurements = measurement_count
        if int(config.get("num_measurements", measurement_count)) != measurement_count:
            raise ValueError("num_measurements does not match the circuit")

    def sample(self) -> str:
        seed = int.from_bytes(blake2b(
            f"{self._seed}:{self._shot_index}".encode("ascii"), digest_size=8
        ).digest(), "little")
        self._shot_index += 1
        tableau = self._tableau_class(self._num_qubits, self._min_abs_coeff, seed)
        for operation in self._operations:
            if operation[0] == "stim":
                tableau.run(operation[1])
            elif operation[0] == "rotation":
                getattr(tableau, operation[1])(operation[2], operation[3])
            elif self._record(tableau)[-operation[2]] == 1:
                getattr(tableau, operation[1])([operation[3]])
        record = self._record(tableau)
        if len(record) != self._num_measurements:
            raise RuntimeError("PPVM returned an unexpected measurement count")
        return "".join("01-"[outcome] for outcome in record)

    def _record(self, tableau):
        return [
            outcome ^ (index in self._inverted_measurements) if outcome != 2 else 2
            for index, outcome in enumerate(tableau.current_measurement_record())
        ]
