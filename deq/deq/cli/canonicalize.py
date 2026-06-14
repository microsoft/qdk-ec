# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint
import arguably
import deq.proto.deq_bin_pb2 as pb
from deq.spec.canonical import canonicalize as _canonicalize_lib


@arguably.command
def canonicalize(
    file: str,
    *,
    out: str | None = None,
) -> None:
    """Canonicalize a .deq.bin file into a single-type flattened form.

    The canonical form resolves all remote references and produces a library with
    a single gadget type (all measurements concatenated), a single check model
    (all checks with absolute measurement indices), and a single error model.
    """
    with open(file, "rb") as f:
        library = pb.Library.FromString(f.read())

    canonical = _canonicalize_lib(library)

    if out is None:
        base = file
        if base.endswith(".deq.bin"):
            base = base[: -len(".deq.bin")]
        out = f"{base}.canonical.deq.bin"

    with open(out, "wb") as f:
        f.write(canonical.library.SerializeToString())

    with open(f"{out}.txt", "w", encoding="utf-8") as f:
        f.write(str(canonical.library))

    num_measurements = len(canonical.gadget_type.measurements)
    num_checks = len(canonical.check_model_type.checks)
    num_errors = len(canonical.error_model_type.errors)
    num_readouts = len(canonical.gadget_type.readouts)
    print(f"Canonicalized: {out}")
    print(f"  Measurements: {num_measurements}")
    print(f"  Checks: {num_checks}")
    print(f"  Errors: {num_errors}")
    print(f"  Readouts: {num_readouts}")
