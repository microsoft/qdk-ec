# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint
import arguably
import deq.proto.deq_bin_pb2 as pb
from deq.spec.canonical import CanonicalForm, canonicalize
from deq.spec.common import MeasurementIndex, CheckIndex, ReadoutIndex
from deq.cli.util import parse_bits, bits_to_hex, bits_to_str


def _build_gadget_groups(
    library: pb.Library,
    canonical: CanonicalForm,
) -> list[tuple[str, list[int], list[int], list[int]]]:
    """Build per-gadget groups of (name, measurement_globals, check_globals, readout_globals).

    Returns groups in program order. Each group corresponds to one gadget
    instance in the original (non-canonical) program.
    """
    gtype_map = {gt.gtype: gt for gt in library.gadget_types}
    ctype_map = {ct.ctype: ct for ct in library.check_model_types}

    # Map gid -> gadget info
    gid_order: list[int] = []
    gid_to_gtype: dict[int, int] = {}
    for instr in library.program:
        if instr.HasField("gadget"):
            gid = instr.gadget.gid
            gid_order.append(gid)
            gid_to_gtype[gid] = instr.gadget.gtype

    # Map cid -> (gid, ctype)
    cid_to_info: dict[int, tuple[int, int]] = {}
    for instr in library.program:
        if instr.HasField("check_model"):
            cid_to_info[instr.check_model.cid] = (
                instr.check_model.gid,
                instr.check_model.ctype,
            )

    groups: list[tuple[str, list[int], list[int], list[int]]] = []
    for gid in gid_order:
        gt = gtype_map[gid_to_gtype[gid]]
        name = gt.name if gt.name else f"G{gt.gtype}"

        # Measurements for this gadget
        meas_globals: list[int] = []
        for local_idx in range(len(gt.measurements)):
            local = MeasurementIndex(gid=gid, measurement_index=local_idx)
            if local in canonical.measurement_map.atob:
                meas_globals.append(
                    canonical.measurement_map.atob[local].measurement_index
                )

        # Checks for this gadget (find cid(s) attached to this gid)
        check_globals: list[int] = []
        for cid, (attached_gid, ctype) in cid_to_info.items():
            if attached_gid != gid:
                continue
            cmt = ctype_map[ctype]
            for local_idx in range(len(cmt.checks)):
                local = CheckIndex(cid=cid, check_index=local_idx)
                if local in canonical.check_map.atob:
                    check_globals.append(canonical.check_map.atob[local].check_index)

        # Readouts for this gadget
        readout_globals: list[int] = []
        for local_idx in range(len(gt.readouts)):
            local = ReadoutIndex(gid=gid, readout_index=local_idx)
            if local in canonical.readout_map.atob:
                readout_globals.append(canonical.readout_map.atob[local].readout_index)

        groups.append((name, meas_globals, check_globals, readout_globals))
    return groups


def interpret_measurements(
    library: pb.Library,
    measurements: str,
    *,
    verbose: bool = False,
) -> tuple[str, str]:
    """Interpret raw measurement bits into syndromes and readouts.

    Canonicalizes the library, computes all check parities and readout
    values, prints grouped output, and returns ``(syndrome_bits, readout_bits)``
    strings.  Does not invoke the decoder.

    *measurements* requires a ``0b`` or ``0x`` prefix.
    """
    canonical = canonicalize(library)
    gadget_type = canonical.gadget_type
    check_model_type = canonical.check_model_type
    num_measurements = len(gadget_type.measurements)
    bits = parse_bits(measurements, num_measurements)

    # Evaluate all checks
    syndrome: list[int] = []
    for check in check_model_type.checks:
        parity = 0
        for rm in check.measurements:
            parity ^= bits[rm.measurement_index]
        if check.naturally_flipped:
            parity ^= 1
        syndrome.append(parity)

    # Evaluate readouts. The canonical readout_propagation's last column
    # is the affine (constant) column — a 1 there means the readout is
    # deterministically flipped (e.g. from a VIRTUAL Pauli correction).
    rp = gadget_type.readout_propagation
    affine_col = rp.cols - 1 if rp.cols > 0 else -1
    readout_affine: list[bool] = [False] * len(gadget_type.readouts)
    for r, c in zip(rp.i, rp.j):
        if c == affine_col:
            readout_affine[r] = not readout_affine[r]

    readout_values: list[int] = []
    for idx, readout in enumerate(gadget_type.readouts):
        parity = 0
        for mi in readout.measurement_indices:
            parity ^= bits[mi]
        if readout_affine[idx]:
            parity ^= 1
        readout_values.append(parity)

    groups = _build_gadget_groups(library, canonical)

    # Build reverse mapping: global measurement index → gadget name
    meas_to_gadget: dict[int, str] = {}
    for name, meas_globals, _, _ in groups:
        for mi in meas_globals:
            meas_to_gadget[mi] = name

    # Print grouped output
    print(f"Raw measurements ({num_measurements} total, hex={measurements}):")
    for name, meas_globals, _, _ in groups:
        if not meas_globals:
            print(f"  {name}: (none)")
        else:
            meas_bits = [bits[g] for g in meas_globals]
            print(f"  {name}: {bits_to_str(meas_bits)}")

    print(f"\nChecks (syndrome hex={bits_to_hex(syndrome)}):")
    for name, _, check_globals, _ in groups:
        if not check_globals:
            continue
        check_bits = [syndrome[g] for g in check_globals]
        line = f"  {name}: {bits_to_str(check_bits)}"
        if verbose:
            details: list[str] = []
            for gi in check_globals:
                check = check_model_type.checks[gi]
                parts = [
                    f"m{rm.measurement_index}={bits[rm.measurement_index]}"
                    for rm in check.measurements
                ]
                details.append(f"c{gi}={syndrome[gi]}({' ⊕ '.join(parts)})")
            line += "  " + "  ".join(details)
        print(line)

    if readout_values:
        print("\nReadouts:")
        for name, _, _, readout_globals in groups:
            if not readout_globals:
                continue
            for gi in readout_globals:
                readout = gadget_type.readouts[gi]
                tag = readout.tag if readout.tag else f"r{gi}"
                mi_list = list(readout.measurement_indices)
                mi_str = " ⊕ ".join(f"m{mi}" for mi in mi_list)
                line = f"  {name}: {tag} = {readout_values[gi]}  ({mi_str})"
                if verbose:
                    parts = []
                    for mi in mi_list:
                        g = meas_to_gadget.get(mi, "?")
                        parts.append(f"m{mi}[{g}]={bits[mi]}")
                    line = (
                        f"  {name}: {tag} = {readout_values[gi]}  ({' ⊕ '.join(parts)})"
                    )
                print(line)

    syndrome_bits = bits_to_str(syndrome)
    readout_bits = bits_to_str(readout_values)
    print(f"\nSyndrome: {syndrome_bits}")
    print(f"Readout:  {readout_bits}")
    return syndrome_bits, readout_bits


@arguably.command
def interpret(
    file: str,
    *,
    measurements: str,
    #: name of the PROGRAM block (required for .deq files)
    program: str | None = None,
    verbose: bool = False,
) -> tuple[str, str]:
    """Interpret raw measurement bits into syndromes and readouts.

    Accepts a ``.deq.bin`` file (canonical or not), a ``.deq.jit``
    file (with ``--program``), or a ``.deq`` source file (with
    ``--program``).  For ``.deq.jit`` and ``.deq`` inputs the file
    is compiled to a ``.deq.bin`` internally first.

    The measurements argument requires a ``0b`` (binary) or ``0x`` (hex) prefix.

    Hex uses BitVector convention:
    bit 0 = 0x80, bit 1 = 0x40, etc. (MSB-first within each byte).

    Example: for 5 measurements, ``0b00000`` or ``0x00`` = all zero,
    ``0b10100`` or ``0xa0`` = bits 0 and 2 set.
    """
    library = _load_library(file, program=program)
    return interpret_measurements(library, measurements, verbose=verbose)


def _load_library(
    file: str,
    *,
    program: str | None = None,
) -> pb.Library:
    """Load or compile a ``Library`` from any supported file format."""
    if file.endswith(".deq.bin"):
        with open(file, "rb") as f:
            return pb.Library.FromString(f.read())

    if file.endswith(".deq.jit"):
        from deq.compiler.jit_compiler import static_jit_compiler
        import deq.proto.deq_jit_pb2 as jit_pb

        with open(file, "rb") as f:
            jit_library = jit_pb.JitLibrary.FromString(f.read())
        return static_jit_compiler(jit_library)

    if file.endswith(".deq"):
        if program is None:
            raise ValueError(
                ".deq files require --program to specify which "
                "PROGRAM block to compile"
            )
        import os
        import tempfile
        from deq.circuit.parser import render_and_parse_file
        from deq.transpiler.jit_library_builder import build_jit_library
        from deq.cli.jit import jit_compile_program_to_file
        from deq.compiler.jit_compiler import static_jit_compiler
        import deq.proto.deq_jit_pb2 as jit_pb

        qfile = render_and_parse_file(file)
        jit_library = build_jit_library(qfile)
        with tempfile.TemporaryDirectory() as tmpdir:
            jit_path = os.path.join(tmpdir, "temp.deq.jit")
            jit_compile_program_to_file(jit_library, qfile, jit_path, program=program)
            with open(jit_path, "rb") as f:
                jit_library = jit_pb.JitLibrary.FromString(f.read())
        return static_jit_compiler(jit_library)

    raise ValueError(
        f"Unsupported file type: {file!r}. " f"Expected .deq.bin, .deq.jit, or .deq"
    )
