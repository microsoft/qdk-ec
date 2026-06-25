# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint
import arguably
import deq.proto.deq_bin_pb2 as pb
from deq.spec.physical_validator import is_valid_and_physical
from deq.spec.program_equivalence import are_programs_equivalent
from deq.spec.violations import Violations


@arguably.command
def spec__are_programs_equivalent(file1: str, file2: str) -> None:
    with open(file1, "rb") as f:
        deq_bin_1 = pb.Library.FromString(f.read())
    with open(file2, "rb") as f:
        deq_bin_2 = pb.Library.FromString(f.read())
    result = are_programs_equivalent(deq_bin_1, deq_bin_2)
    if not result:
        print(result)
    else:
        print("programs are equivalent")


@arguably.command
def spec__is_valid_and_physical(file: str) -> None:
    with open(file, "rb") as f:
        deq_bin = pb.Library.FromString(f.read())
    result = is_valid_and_physical(deq_bin)
    if isinstance(result, Violations):
        print(result.violations)
    else:
        print("program is valid and physical")


# TODO: add others
