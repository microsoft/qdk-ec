"""Write the evaluated DEQ brickwall circuit and its lattice-surgery expansion.

The brickwall file is the source compiled by ``run.py`` at the requested width:
logical ``CNOT`` sub-program calls on one surface-code patch per data qubit.
The expanded file inlines every ``CNOT`` into its lattice-surgery gadgets with
the JIT compiler's own expansion. Both files place data qubit ``i`` on wire
``2i`` and the ancilla of ``CNOT 2i 2i+2`` on wire ``2i+1``, and both must
compile to the same JIT program as the consecutively numbered evaluated source.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jit-basics"))

from deqagram import (
    CodeDefinition,
    ComposeDefinition,
    ConditionalCorrection,
    Decorator,
    GadgetApplication,
    GadgetDefinition,
    Instruction,
    KeywordArg,
    ProgramDefinition,
    QubitTarget,
    VirtualCorrection,
)
from deq.cli.jit import (
    _expand_program_calls,
    _flatten_repeats,
    _program_call_types,
    _remap_stmt,
    compile_program_for_jit,
)
from deq.transpiler.jit_library_builder import build_jit_library, build_jit_program
from deq.transpiler.loss.model_none import NoLossModel
from deq.transpiler.program_timing import Clock, Schedule

from deq.circuit.parser import parse
from lattice_surgery import circuit_source, decimal_seconds, seconds_to_ns
from run import BASIS, Settings

HERE = Path(__file__).resolve().parent
PROGRAM = "Brickwall"
SUBROUTINE = "\nPROGRAM CNOT {"
WIRE_STRIDE = 2


def statement_wires(statement: object) -> list[int]:
    if isinstance(statement, GadgetApplication):
        return [*(statement.in_indices or []), *(statement.out_indices or [])]
    if isinstance(statement, (VirtualCorrection, ConditionalCorrection)):
        return [statement.wire]
    if isinstance(statement, Instruction):
        return [
            target.index
            for target in statement.targets
            if isinstance(target, QubitTarget)
        ]
    return []


def expand(source: str, data_wires: set[int]) -> list[object]:
    parsed = parse(source)
    programs = {
        definition.name: definition
        for definition in parsed.definitions
        if isinstance(definition, ProgramDefinition)
    }
    gadget_names = {
        definition.name
        for definition in parsed.definitions
        if isinstance(definition, (GadgetDefinition, ComposeDefinition))
    }
    body = list(programs[PROGRAM].body)
    used = {
        wire
        for statement in _flatten_repeats(body)
        for wire in statement_wires(statement)
    }
    if used != data_wires:
        raise ValueError("the brickwall must use exactly its data wires")
    library = build_jit_program(parsed)
    gadget_types = {item.base.name: item.base for item in library.gadget_types}
    port_types = {item.base.name: item.base.ptype for item in library.port_types}
    schedule = Schedule()
    statements = _expand_program_calls(
        body,
        programs,
        gadget_names,
        [max(used) + 1],
        gadget_types=gadget_types,
        port_types=port_types,
        call_types=_program_call_types(programs, port_types) | gadget_types,
        clock=Clock(schedule),
    )
    neighbors: dict[int, set[int]] = {}
    for statement in statements:
        wires = statement_wires(statement)
        for wire in wires:
            if wire not in data_wires:
                neighbors.setdefault(wire, set()).update(
                    other for other in wires if other in data_wires
                )
    placement = {wire: wire for wire in data_wires}
    for ancilla, coupled in neighbors.items():
        left, right = min(coupled), max(coupled)
        if len(coupled) != 2 or right - left != WIRE_STRIDE:
            raise ValueError("each ancilla must couple two neighboring data patches")
        placement[ancilla] = left + 1
    result = []
    for statement in statements:
        timing = schedule.statements.get(id(statement))
        statement = _remap_stmt(statement, placement)
        if isinstance(statement, GadgetApplication):
            inputs = statement.in_indices or []
            outputs = statement.out_indices or []
            wires = inputs if len(inputs) >= len(outputs) else outputs
            if inputs != wires[: len(inputs)] or outputs != wires[: len(outputs)]:
                raise ValueError(
                    f"expanded {statement.gadget_name} does not have positional port bindings"
                )
            statement = Instruction(
                name=statement.gadget_name,
                targets=[QubitTarget(wire) for wire in wires],
                decorators=statement.decorators,
            )
        if isinstance(statement, Instruction) and timing is not None:
            statement.decorators = [
                decorator for decorator in statement.decorators
                if decorator.name not in ("START", "DELAY")
            ] + [
                Decorator(
                    name="START",
                    arguments=[
                        KeywordArg(key="at", value=Decimal(decimal_seconds(timing.start))),
                        KeywordArg(key="duration", value=Decimal(decimal_seconds(timing.duration))),
                    ],
                )
            ]
        result.append(statement)
    return result


def compiled_program(source: str) -> bytes:
    parsed = parse(source)
    library = build_jit_library(parsed, jobs=1, loss_model=NoLossModel())
    programs = {
        definition.name: definition
        for definition in parsed.definitions
        if isinstance(definition, ProgramDefinition)
    }
    codes = {
        definition.name: definition
        for definition in parsed.definitions
        if isinstance(definition, CodeDefinition)
    }
    compiled, assertions = compile_program_for_jit(
        library, programs[PROGRAM], dict(programs), codes=codes
    )
    return repr(assertions).encode() + b"".join(
        instruction.SerializeToString() for instruction, _ in compiled
    )


def render_statement(statement: object) -> str:
    lines = []
    for decorator in getattr(statement, "decorators", ()):
        if decorator.name == "START":
            arguments = ",".join(
                f"{argument.key}={decimal_seconds(seconds_to_ns(argument.value))}"
                for argument in decorator.arguments
            )
            lines.append(f"@START({arguments})")
        else:
            lines.append(str(decorator))
    lines.append(str(statement))
    return "".join(f"    {line}\n" for line in lines)


def generate(width: int, output: Path, settings: Settings) -> tuple[Path, Path]:
    parameters = {
        "distance": settings.distance,
        "probability": settings.probability,
        "width": width,
        "layers": settings.layers,
        "basis": BASIS,
        "seed": settings.seed,
        "block_seconds": settings.interval_ms / 1000,
    }
    evaluated = circuit_source(**parameters)
    source = circuit_source(**parameters, wire_stride=WIRE_STRIDE)
    if source.count(SUBROUTINE) != 1:
        raise ValueError("expected one CNOT sub-program in the surgery library")
    library = source[: source.index(SUBROUTINE)]
    statements = expand(source, set(range(0, WIRE_STRIDE * width, WIRE_STRIDE)))
    cnots = sum(
        isinstance(statement, Instruction) and statement.name == "MZZ"
        for statement in statements
    )
    settings_line = ", ".join(f"{key}={value}" for key, value in parameters.items())
    layout = (
        f"# Data qubit i is wire {WIRE_STRIDE}i (wires 0-{WIRE_STRIDE * (width - 1)}); "
        "the lattice-surgery ancilla\n"
        "# for CNOT 2i 2i+2 is wire 2i+1, between the patches it couples.\n"
    )
    original = (
        f"# CNOT brickwall evaluated by run.py ({settings_line}).\n"
        + layout
        + "# run.py numbers data qubits consecutively; this numbering matches\n"
        f"# brickwall-w{width}-lattice-surgery.deq and compiles to the same JIT program.\n"
        "# Generated by generate_circuits.py; do not edit.\n\n" + source
    )
    expanded = (
        f"# Lattice-surgery expansion of brickwall-w{width}.deq ({settings_line}).\n"
        f"# {cnots} logical CNOTs become PrepareXNonFT, MZZ, MXX, and MeasureZ.\n"
        + layout
        + "# Generated by generate_circuits.py; do not edit.\n\n"
        + library
        + f"\nPROGRAM {PROGRAM} {{\n"
        + "".join(render_statement(statement) for statement in statements)
        + "}\n"
    )
    reference = compiled_program(evaluated)
    if compiled_program(original) != reference:
        raise ValueError("the renumbered brickwall compiles differently")
    if compiled_program(expanded) != reference:
        raise ValueError("the expanded circuit compiles differently")
    output.mkdir(parents=True, exist_ok=True)
    original_path = output / f"brickwall-w{width}.deq"
    expanded_path = output / f"brickwall-w{width}-lattice-surgery.deq"
    original_path.write_text(original)
    expanded_path.write_text(expanded)
    return original_path, expanded_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--output", type=Path, default=HERE)
    arguments = parser.parse_args()
    for path in generate(arguments.width, arguments.output, Settings(threads=1)):
        print(path)


if __name__ == "__main__":
    main()
