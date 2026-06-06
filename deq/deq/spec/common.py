from typing import TypeVar, Generic
from abc import ABC
from collections.abc import Hashable
from dataclasses import dataclass, field
from binar import BitMatrix, BitVector, solve
import deq.proto.util_pb2 as util_pb
import deq.proto.deq_bin_pb2 as bin_pb

# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint


class NoViolations(ABC):
    """
    A mixin class indicating no violations, helpful when returning :code:`Violations | YourClass`
    """

    def __contains__(self, _sub_message: str | tuple[str, ...]) -> bool:
        return False

    def __bool__(self) -> bool:
        return True


Index = TypeVar("Index", bound=Hashable)


@dataclass
class Bijection(Generic[Index], NoViolations):
    atob: dict[Index, Index] = field(default_factory=dict)
    btoa: dict[Index, Index] = field(default_factory=dict)

    def add(self, a: Index, b: Index, unique: bool = True) -> None:
        assert not unique or (a not in self.atob and b not in self.btoa)
        self.atob[a] = b
        self.btoa[b] = a

    def __len__(self) -> int:
        assert len(self.atob) == len(self.btoa)
        return len(self.atob)

    def chain(self, other: "Bijection[Index]") -> "Bijection[Index]":
        result: Bijection[Index] = Bijection()
        for a, b in self.atob.items():
            result.add(a, other.atob[b])
        return result

    def reverse(self) -> "Bijection[Index]":
        return Bijection(atob=self.btoa.copy(), btoa=self.atob.copy())


@dataclass(frozen=True, kw_only=True, eq=True)
class LocalObservableIndex:
    port: int  # port index of the gadget
    observable_index: int  # local observable index of the port

    def to_global(self, gid: int) -> "ObservableIndex":
        return ObservableIndex(
            gid=gid, port=self.port, observable_index=self.observable_index
        )


@dataclass(frozen=True, kw_only=True, eq=True)
class ObservableIndex:
    gid: int  # gadget id
    port: int  # port index of the gadget
    observable_index: int  # local observable index of the port


@dataclass(frozen=True, kw_only=True, eq=True)
class MeasurementIndex:
    gid: int  # gadget id
    measurement_index: int  # local measurement index of the gadget


@dataclass(frozen=True, kw_only=True, eq=True)
class PortIndex:
    gid: int  # gadget id
    port_index: int  # local port index of the gadget


@dataclass(frozen=True, kw_only=True, eq=True)
class InputPortIndex(PortIndex): ...


@dataclass(frozen=True, kw_only=True, eq=True)
class OutputPortIndex(PortIndex): ...


@dataclass(frozen=True, kw_only=True, eq=True)
class ReadoutIndex:
    gid: int  # gadget id
    readout_index: int  # local readout index of the gadget


@dataclass(frozen=True, kw_only=True, eq=True)
class CheckIndex:
    cid: int  # check model id
    check_index: int  # local check index of the check model


@dataclass(frozen=True, kw_only=True, eq=True)
class ErrorIndex:
    eid: int  # error model id
    error_index: int  # local error index of the error model


@dataclass
class LinearMap(Generic[Index]):
    atob: dict[Index, set[Index]] = field(default_factory=dict)
    btoa: dict[Index, set[Index]] = field(default_factory=dict)


def reverse_involved_of(
    linear_map: dict[Index, set[Index]],
) -> dict[Index, set[Index]]:
    involved: dict[Index, set[Index]] = {}
    for a, bs in linear_map.items():
        for b in bs:
            if b not in involved:
                involved[b] = set()
            involved[b].add(a)
    return involved


def bitmatrix_of(matrix: util_pb.BitMatrix) -> BitMatrix:
    m = BitMatrix.zeros(rows=matrix.rows, columns=matrix.cols)
    for i, j in zip(matrix.i, matrix.j):
        m[(i, j)] = True
    return m


def bitmatrix_to_proto(matrix: BitMatrix) -> util_pb.BitMatrix:
    ones_i: list[int] = []
    ones_j: list[int] = []
    for i in range(matrix.row_count):
        for j in range(matrix.column_count):
            if matrix[(i, j)]:
                ones_i.append(i)
                ones_j.append(j)
    return util_pb.BitMatrix(
        rows=matrix.row_count, cols=matrix.column_count, i=ones_i, j=ones_j
    )


def apply_bitmatrix_modifier(
    original: util_pb.BitMatrix, modifier: "bin_pb.BitMatrixModifier | None"
) -> util_pb.BitMatrix:
    if modifier is None:
        return original
    result = bitmatrix_of(original)
    if modifier.HasField("toggle"):
        toggle = bitmatrix_of(modifier.toggle)
        for i in range(toggle.row_count):
            for j in range(toggle.column_count):
                if toggle[(i, j)]:
                    result[(i, j)] = not result[(i, j)]
    if modifier.HasField("overwrite"):
        result = bitmatrix_of(modifier.overwrite)
    return bitmatrix_to_proto(result)


def linear_solve(matrix: BitMatrix, target: BitVector) -> BitVector | None:
    if matrix.column_count > 0:
        return solve(matrix, target)
    else:
        return None if target.weight > 0 else BitVector.zeros(0)
