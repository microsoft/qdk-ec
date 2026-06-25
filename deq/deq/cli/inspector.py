# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint
import os

import arguably
import deq.proto.deq_bin_pb2 as pb
import deq.proto.deq_jit_pb2 as jit_pb
from deq.spec.canonical import canonicalize


@arguably.command
def inspect(
    file: str,
    *,
    gadget: int | None = None,
    jit: bool = False,
    bin_: bool = False,
) -> None:
    """Inspect a .deq.jit or .deq.bin file and print its contents.

    Auto-detects the file type from the extension. Use ``--jit`` or
    ``--bin`` to override when the extension is ambiguous.

    Args:
        file: path to the .deq.jit or .deq.bin file.
        gadget: [-g] if set, only print detailed info for the gadget with this gtype (JIT only).
        jit: force interpretation as a .deq.jit file.
        bin_: force interpretation as a .deq.bin file.
    """
    if jit and bin_:
        raise ValueError("Cannot specify both --jit and --bin")
    if jit:
        is_jit = True
    elif bin_:
        is_jit = False
    elif file.endswith(".deq.jit"):
        is_jit = True
    elif file.endswith(".deq.bin"):
        is_jit = False
    else:
        raise ValueError(
            f"Cannot determine file type for {file!r}. "
            f"Use --jit or --bin to specify."
        )
    if is_jit:
        inspect__deq_jit(file, gadget=gadget)
    else:
        if gadget is not None:
            raise ValueError("--gadget is only supported for .deq.jit files")
        inspect__deq_bin(file)


def inspect__deq_bin(file: str) -> None:
    with open(file, "rb") as f:
        deq_bin = pb.Library.FromString(f.read())

    print("==== Library ====")
    print("    len(port_types):", len(deq_bin.port_types))
    print("    len(gadget_types):", len(deq_bin.gadget_types))
    print("    len(check_model_types):", len(deq_bin.check_model_types))
    print("    len(error_model_types):", len(deq_bin.error_model_types))

    gadget_types = {
        gadget_type.gtype: gadget_type for gadget_type in deq_bin.gadget_types
    }
    check_model_types = {
        check_model_type.ctype: check_model_type
        for check_model_type in deq_bin.check_model_types
    }
    error_model_types = {
        error_model_type.etype: error_model_type
        for error_model_type in deq_bin.error_model_types
    }

    print("==== Program ====")
    print("    len(instructions):", len(deq_bin.program))

    def count_instruction(field: str) -> int:
        count = 0
        for instruction in deq_bin.program:
            if instruction.HasField(field):
                count += 1
        return count

    print("        len(gadgets):", count_instruction("gadget"))
    print("        len(check_models):", count_instruction("check_model"))
    print("        len(error_models):", count_instruction("error_model"))

    print("==== Realization ====")
    num_physical_gates = 0
    num_physical_measurements = 0
    num_readouts = 0
    for instruction in deq_bin.program:
        if not instruction.HasField("gadget"):
            continue
        gadget = instruction.gadget
        gadget_type = gadget_types[gadget.gtype]
        realization = gadget_type.realization
        num_physical_gates += len(realization.locations)
        num_physical_measurements += len(gadget_type.measurements)
        num_readouts += len(gadget_type.readouts)
    print("number of physical gates:", num_physical_gates)
    print("number of measurements:", num_physical_measurements)
    print("number of readouts:", num_readouts)

    print("==== Check Error Model ====")
    num_checks = 0
    check_measurements: dict[int, int] = {}  # how many measurements per check
    for instruction in deq_bin.program:
        if not instruction.HasField("check_model"):
            continue
        check_model_type = check_model_types[instruction.check_model.ctype]
        num_checks += len(check_model_type.checks)
        for check in check_model_type.checks:
            if len(check.measurements) not in check_measurements:
                check_measurements[len(check.measurements)] = 0
            check_measurements[len(check.measurements)] += 1
    print("number of checks:", num_checks)
    print(
        "check measurement weights distribution:",
        str_weight_distribution(check_measurements),
    )
    num_errors = 0
    for instruction in deq_bin.program:
        if not instruction.HasField("error_model"):
            continue
        error_model_type = error_model_types[instruction.error_model.etype]
        num_errors += len(error_model_type.errors)
    print("number of errors: ", num_errors)

    print("==== Monolithic Check Matrix ====")
    canonical = canonicalize(deq_bin)
    errors: set[frozenset[int]] = set()
    for error in canonical.error_model_type.errors:
        syndrome = frozenset([check.check_index for check in error.checks])
        errors.add(syndrome)
    print("number of rows (checks):", len(canonical.check_model_type.checks))
    print("number of columns (distinct errors):", len(errors))
    column_weights: dict[int, int] = {}
    for error in errors:
        degree = len(error)
        if degree not in column_weights:
            column_weights[degree] = 0
        column_weights[degree] += 1
    print("column weight distribution:", str_weight_distribution(column_weights))
    check_weights = [0 for _ in range(len(canonical.check_model_type.checks))]
    for error in errors:
        for check_index in error:
            check_weights[check_index] += 1
    row_weights: dict[int, int] = {}
    for weight in check_weights:
        if weight not in row_weights:
            row_weights[weight] = 0
        row_weights[weight] += 1
    print("row weight distribution:", str_weight_distribution(row_weights))


def inspect__deq_jit(jit_file: str, *, gadget: int | None = None) -> None:
    """
    Inspect a .deq.jit file and print its contents.

    Args:
        jit_file: path to the .deq.jit file.
        gadget: [-g] if set, only print detailed info for the gadget with this gtype.
    """

    with open(jit_file, "rb") as f:
        jit_library = jit_pb.JitLibrary.FromString(f.read())

    if gadget is None:
        print("JIT Library contents:")
        print(f"  Port types: {len(jit_library.port_types)}")
        print(f"  Gadget types: {len(jit_library.gadget_types)}")
        print()

        for port in jit_library.port_types:
            print(f"Port type {port.base.ptype}:")
            print(f"  Observables: {len(port.base.observables)}")
            print(f"  Stabilizers: {len(port.stabilizers)}")
        print()

        for gtype in jit_library.gadget_types:
            print_jit_gadget_summary(gtype)
        return

    matches = [g for g in jit_library.gadget_types if g.base.gtype == gadget]
    if not matches:
        raise ValueError(f"No gadget type with gtype={gadget} found in {jit_file}")
    if len(matches) > 1:
        raise ValueError(f"Multiple gadget types with gtype={gadget} found")
    gtype = matches[0]
    print_jit_gadget_summary(gtype)

    finished_check_weights: dict[int, int] = {}
    for check in gtype.finished_checks:
        weight = len(check.measurements)
        finished_check_weights[weight] = finished_check_weights.get(weight, 0) + 1
    print(
        "  Finished check weight distribution:",
        str_weight_distribution(finished_check_weights),
    )

    unfinished_check_weights: dict[int, int] = {}
    for check in gtype.unfinished_checks:
        weight = len(check.measurements)
        unfinished_check_weights[weight] = unfinished_check_weights.get(weight, 0) + 1
    print(
        "  Unfinished check weight distribution:",
        str_weight_distribution(unfinished_check_weights),
    )

    error_weights: dict[int, int] = {}
    for error in gtype.errors:
        weight = len(error.finished_checks) + len(error.unfinished_checks)
        error_weights[weight] = error_weights.get(weight, 0) + 1
    print(
        "  Error weight distribution (checks triggered):",
        str_weight_distribution(error_weights),
    )


def print_jit_gadget_summary(gtype: jit_pb.JitGadgetType) -> None:
    print(f"Gadget type {gtype.base.gtype} ({gtype.base.name or 'unnamed'}):")
    print(f"  Measurements: {len(gtype.base.measurements)}")
    print(f"  Readouts: {len(gtype.base.readouts)}")
    print(f"  Inputs: {len(gtype.base.inputs)}, Outputs: {len(gtype.base.outputs)}")
    print(f"  Finished checks: {len(gtype.finished_checks)}")
    print(f"  Unfinished checks: {len(gtype.unfinished_checks)}")
    print(f"  Errors: {len(gtype.errors)}")


def str_weight_distribution(distribution: dict[int, int]) -> str:
    items = sorted(distribution.items(), key=lambda x: x[0])
    return "{" + ", ".join(f"{weight}:{count}" for weight, count in items) + "}"
