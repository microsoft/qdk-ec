"""
=====================
Program Equivalence
=====================

Being equivalent decoding hypergraphs means that there exists linear transformation from\
    the vertices (checks) in one hypergraph to the other, such that the hyperedges connecting\
    them have the same effect (residual, readout flips and probability). This is a more\
    relaxed condition than being the identical decoding hypergraph but they are still describing \
    the same decoding problem. Note that a decoder might have different accuracy on two equivalent \
    decoding hypergraphs because decoders are not necessarily optimal.

Prerequisite: both programs must satisfy the LibSpec, ProgSpec and PhySpec. We require PhySpec \
    so that we can solve the check linear equations without worrying that errors may not be \
    physical.

Invariants: identical programs are always equivalent, i.e., ProgId :math:`\\Rightarrow` ProgEq.

1. (ProgEq 1) The two programs must satisfy the LibSpec, ProgSpec and PhySpec. This implies that\
    that they both have canonical forms.
2. The basic properties of the two canonical forms must be the same (this is similar\
    to ProgId 2.1 to 2.6, except for ProgId 2.7 which requires the same number of checks):

    1. (ProgEq 2.1) The canonical output ports must have the same number of observables.
    2. (ProgEq 2.2) The canonical gadgets must have the same number of measurements.\
        This ensures that equivalent programs accepts the same sequence of measurements,\
        although measurements may be loaded into different gadgets, e.g., one program\
        having one giant gadget while the other has several smaller ones.
    3. (ProgEq 2.3) The canonical gadgets must have the same static output observables,\
        as defined by the :code:`correction_propagation` field (a single-column matrix)
    4. (ProgEq 2.4) The canonical gadgets must have the same number of readouts
    5. (ProgEq 2.5) The canonical gadgets must have the same static readout values,\
        as defined by the :code:`readout_propagation` field (a single-column matrix)
    6. (ProgEq 2.6) The canonical gadgets must have the same conditional correction,\
        as defined by the :code:`logical_correction` field. After the merge() absorption\
        pass (canonical.py step 9), the merged :code:`logical_correction` is always\
        empty by design — local and remote conditional corrections are absorbed into\
        :code:`correction_propagation` and :code:`physical_correction` (and into\
        per-error :code:`residual`).  This check therefore reduces to verifying that\
        both canonical forms have empty :code:`logical_correction`, which is trivially\
        satisfied.
    7. Note that we do NOT require the same number of checks in the two canonical forms,\
        because checks can be linearly combined to form new checks, and thus the number\
        of checks can be different while the overall effect is the same. We will elaborate more\
        details on the identicalness of the checks in (ProgEq 3.*)
    8. Similar to ProgId 2.8, we do NOT require the same number of errors in the canonical forms

3. (ProgEq 3) There must exist a linear mapping from the checks in canonical1 to\
    canonical2 and vice versa, such that:

    * For each check, the set of measurements it involves must equal to\
        the symmetric difference of the measurements involved by the mapped checks.
    * For each check, the value of :code:`naturally_flipped` must equal to\
        the parity of the :code:`naturally_flipped` values of the mapped checks.
    * Note that the linear mapping is not necessarily unique. We only attempt to find\
        one such mapping, because PhySpec 2.3 guarantees that all mappings are equivalent\
        in terms of the value of :code:`naturally_flipped`.

4. Once we have the linear map between the checks, we can then map the checks involved \
    in each error to the checks in the other library and see if they are equivalent.\
    Please refer to ProgEq 4.* for the comparison of the errors. The requirements are:

    1. (ProgEq 4.1) the two sets of unique errors must be the same.
    2. (ProgEq 4.2) for each unique error, the gathered probability must be the same,\
        up to some small floating-point precision (at most a relative error of :math:`10^{-6}`).

"""

# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint

from dataclasses import dataclass
from binar import BitMatrix, BitVector
from deq.spec.violations import Violations
from deq.spec.physical_validator import is_valid_and_physical
from deq.spec.common import (
    LinearMap,
    CheckIndex,
    NoViolations,
    reverse_involved_of,
    linear_solve,
)
from deq.spec.canonical import canonicalize, CanonicalForm
from deq.spec.program_identicalness import (
    _validate_canonical_compatibility,
    _compare_mirrored_hyperedges,
)
import deq.proto.deq_bin_pb2 as pb


@dataclass
class EquivalenceMap(NoViolations):
    """The linear mapping between two equivalent programs."""

    check_map: LinearMap[CheckIndex]


def are_programs_equivalent(
    lib1: pb.Library, lib2: pb.Library, validate: bool = True, rel_tol: float = 1e-6
) -> Violations | EquivalenceMap:
    """Check if two programs result in equivalent decoding hypergraphs."""

    if validate:
        validity1 = is_valid_and_physical(lib1)
        if isinstance(validity1, Violations):
            return Violations("(ProgEq 1) lib1 is not valid/physical") + validity1

        validity2 = is_valid_and_physical(lib2)
        if isinstance(validity2, Violations):
            return Violations("(ProgEq 1) lib2 is not valid/physical") + validity2

    canonical1 = canonicalize(lib1)
    canonical2 = canonicalize(lib2)

    # first check basic properties of the canonical forms
    basic_violations = _validate_canonical_compatibility(
        canonical1, canonical2, prefix="ProgEq 2."
    )
    if not basic_violations:
        return basic_violations

    check_map = _canonical_check_linear_map(canonical1, canonical2)
    if isinstance(check_map, Violations):
        return check_map

    # check if all the hyperedges look alike
    hyperedge_violations = _validate_hyperedges_by_linear_map(
        canonical1, canonical2, check_map, rel_tol
    )
    if not hyperedge_violations:
        return hyperedge_violations

    return EquivalenceMap(check_map=check_map)


def _canonical_check_linear_map(
    canonical1: CanonicalForm, canonical2: CanonicalForm
) -> Violations | LinearMap[CheckIndex]:
    """
    Find the linear mapping between the checks in two canonical forms.

    in the check matrix, each row corresponds to a measurement and each column to a check
    we will solve the problem Ax=b where A is the check matrix of canonical1,
    x is the linear map from a check2 in canonical2, and b is the measurements involved
    in check2. rustpaulimer's EchelonForm can help to solve such linear equations efficiently.
    """

    check_matrix_1 = BitMatrix.zeros(
        rows=len(canonical1.gadget_type.measurements),
        columns=len(canonical1.check_model_type.checks),
    )
    check_matrix_2 = BitMatrix.zeros(
        rows=len(canonical2.gadget_type.measurements),
        columns=len(canonical2.check_model_type.checks),
    )

    # populate the check matrices
    for check_matrix, canonical in [
        (check_matrix_1, canonical1),
        (check_matrix_2, canonical2),
    ]:
        for check_index, check in enumerate(canonical.check_model_type.checks):
            for measurement in check.measurements:
                check_matrix[(measurement.measurement_index, check_index)] = True

    # check if all the checks can be formed by linear combination
    check_map: LinearMap[CheckIndex] = LinearMap()
    for this_id, this_matrix, peer_id, peer_matrix, mapper in (
        (1, check_matrix_1, 2, check_matrix_2, check_map.atob),
        (2, check_matrix_2, 1, check_matrix_1, check_map.btoa),
    ):
        for check_index in range(this_matrix.column_count):
            b = BitVector.zeros(this_matrix.row_count)
            for r in range(this_matrix.row_count):
                b[r] = this_matrix[(r, check_index)]
            solution = linear_solve(peer_matrix, b)
            if solution is None:
                return Violations(
                    "(ProgEq 3) check is not a linear combination of other checks: "
                    + f"check {check_index} in canonical{this_id} "
                    + f"cannot be formed by linear combination of checks in canonical{peer_id}"
                )
            mapper[CheckIndex(cid=1, check_index=check_index)] = {
                CheckIndex(cid=1, check_index=ci)
                for ci in range(peer_matrix.column_count)
                if solution[ci]
            }

    return check_map


def _validate_hyperedges_by_linear_map(
    canonical1: CanonicalForm,
    canonical2: CanonicalForm,
    check_map: LinearMap[CheckIndex],
    rel_tol: float,
) -> Violations:

    # first mirror all the errors in canonical1 to using mapped checks in canonical2
    mirrored_errors_1: list[pb.ErrorModelType.Error] = []
    reverse_involved = reverse_involved_of(check_map.btoa)
    for error in canonical1.library.error_model_types[0].errors:
        checks_in_1 = {
            CheckIndex(cid=1, check_index=check.check_index) for check in error.checks
        }
        mapped_checks_in_2: set[CheckIndex] = set()
        for checks in checks_in_1:
            mapped_checks_in_2 ^= reverse_involved.get(checks, set())
        mirrored_errors_1.append(
            pb.ErrorModelType.Error(
                probability=error.probability,
                residual=error.residual,
                readout_flips=error.readout_flips,
                checks=[
                    pb.ErrorModelType.RemoteCheck(check_index=check.check_index)
                    for check in mapped_checks_in_2
                ],
            )
        )

    return _compare_mirrored_hyperedges(
        mirrored_errors_1, canonical2, rel_tol, prefix="ProgEq 4."
    )
