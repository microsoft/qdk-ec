"""
=====================
Program Identicalness
=====================

Being identical decoding hypergraph means that there exists a bijection mapping between\
    the vertices (checks) in the two hypergraphs, and the hyperedges connecting them have the same\
    effect (residual, readout flips and probability). Note that we permit duplicate checks but\
    we do not make attempts to match these duplicate checks: we simply match them by the order they\
    appear because in a realistic check error model, these duplicate checks should connect to \
    exactly the same set of hyperedges and there is no need to distinguish them. We do not consider\
    ill-defined decoding hypergraphs, although they may pass the equivalence test.

Prerequisite: both programs must satisfy the LibSpec and ProgSpec.

Invariants: the canonical form of a program is always identical to the program itself.

1. (ProgId 1) The two programs must have canonical forms, which implies that they both \
    satisfy LibSpec and ProgSpec.
2. The basic properties of the two canonical forms must be the same:

    1. (ProgId 2.1) The canonical output ports must have the same number of observables.
    2. (ProgId 2.2) The canonical gadgets must have the same number of measurements.\
        This ensures that identical programs accepts the same sequence of measurements,\
        although measurements may be loaded into different gadgets, e.g., one program\
        having one giant gadget while the other has several smaller ones.
    3. (ProgId 2.3) The canonical gadgets must have the same static output observables,\
        as defined by the :code:`correction_propagation` field (a single-column matrix)
    4. (ProgId 2.4) The canonical gadgets must have the same number of readouts
    5. (ProgId 2.5) The canonical gadgets must have the same static readout values,\
        as defined by the :code:`readout_propagation` field (a single-column matrix)
    6. (ProgId 2.6) The canonical gadgets must have the same conditional correction,\
        as defined by the :code:`logical_correction` field. After the merge() absorption\
        pass (canonical.py step 9), the merged :code:`logical_correction` is always\
        empty by design — local and remote conditional corrections are absorbed into\
        :code:`correction_propagation` and :code:`physical_correction` (and into\
        per-error :code:`residual`).  This check therefore reduces to verifying that\
        both canonical forms have empty :code:`logical_correction`, which is trivially\
        satisfied; the real content of the conditional correction is compared via\
        ProgId 2.3 (cp) and ProgId 2.7 (pc).
    7. (ProgId 2.7) The canonical gadgets must have the same physical conditional correction,\
        as defined by the :code:`physical_correction` field
    8. (ProgId 2.8) The canonical check models must have the same number of checks.\
        We will elaborate more details on the identicalness of the checks in (ProgId 3.*)
    9. Note that we do NOT require the number of errors to be the same, because errors\
        are simply overlaid on each other according to the exclusive probability rule\
        :math:`p' = p_1 (1 - p_2) + (1 - p_1) p_2` and thus the number of errors\
        can be different while the overall effect is the same. We will elaborate more\
        on this in (ProgId 4.*)

3. For the checks in the canonical forms, we try to pair them up based on their measurements:

    1. (ProgId 3.1) Representing each check as a set over the measurements and thus \
        the checks as :code:`set[set[MeasurementIndex]]`, the two sets must be equal. \
        We call this "unique check".
    2. (ProgId 3.2) the number of checks corresponding to each unique check\
        :code:`set[MeasurementIndex]` must be the same
    3. (ProgId 3.3) For each unique check, its corresponding checks must share the same\
        value of :code:`naturally_flipped`. This is similar to (PhySpec 2.2) but since\
        ProgId does not require PhySpec to be satisfied, we name it explicitly.
    4. By satisfying the above three conditions, we can construct a bijection mapping\
        between the checks in the two canonical forms. Since there is also bijection mapping\
        between the checks in each canonical form and the original program, we can then\
        construct a bijection mapping between the checks in the two original programs.\
        It is returned as a field in the :code:`IdenticalBijection`.

4. For the errors in the canonical forms, we gather the errors based on their effect\
    :code:`(checks, residual, readout_flips): ErrorKey` (where the type is \
    :code:`type ErrorKey = tuple[set[int], set[int], set[int]]`). \
    We call each error key a "unique error". The errors corresponding to the same unique error\
    have exactly the same effect and thus their probabilities can be merged using\
    :math:`p' = p_1 (1 - p_2) + (1 - p_1) p_2`.\
    The indices are the global indices in the canonical forms. Note that we first mirror\
    the errors in the first program to the second program using the check mapping constructed\
    in (ProgId 3.*), so that their check indices are comparable.

    1. (ProgId 4.1) the two sets of unique errors must be the same.
    2. (ProgId 4.2) for each unique error, the gathered probability must be the same,\
        up to some small floating-point precision (at most a relative error of :math:`10^{-6}`).

"""

# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint

import math
from dataclasses import dataclass, field
import deq.proto.deq_bin_pb2 as pb
from deq.spec.violations import Violations
from deq.spec.canonical import canonicalize, CanonicalForm
from deq.spec.program_validator import is_valid
from deq.spec.common import NoViolations, Bijection, CheckIndex, bitmatrix_of


def exclusive_probability_of(probability_a: float, probability_b: float) -> float:
    """Given the probabilities of two independent events A and B, returns the
    probability that A occurs or B occurs, but not both.
    """
    return probability_a + probability_b - 2 * probability_a * probability_b


@dataclass
class IdenticalBijection(NoViolations):
    """The bijection mapping between two identical programs."""

    check_map: Bijection[CheckIndex]


def are_programs_identical(
    lib1: pb.Library, lib2: pb.Library, validate: bool = True, rel_tol: float = 1e-6
) -> Violations | IdenticalBijection:
    """Check if two programs result in identical decoding hypergraphs."""

    if validate:
        validity1 = is_valid(lib1)
        if isinstance(validity1, Violations):
            return Violations("(ProgId 1) lib1 is not valid") + validity1
        validity2 = is_valid(lib2)
        if isinstance(validity2, Violations):
            return Violations("(ProgId 1) lib2 is not valid") + validity2

    canonical1 = canonicalize(lib1)
    canonical2 = canonicalize(lib2)

    # first check basic properties of the canonical forms
    basic_violations = _validate_basic(canonical1, canonical2)
    if not basic_violations:
        return basic_violations

    # try to construct a bijection between the checks in the two canonical forms
    canonical_check_map = _canonical_check_map(canonical1, canonical2)
    if isinstance(canonical_check_map, Violations):
        return canonical_check_map

    # check if all the hyperedges look alike
    hyperedge_violations = _validate_hyperedges_by_bijection(
        canonical1, canonical2, canonical_check_map, rel_tol
    )
    if not hyperedge_violations:
        return hyperedge_violations

    check_map = canonical1.check_map.chain(canonical_check_map).chain(
        canonical2.check_map.reverse()
    )
    return IdenticalBijection(check_map=check_map)


def _validate_basic(canonical1: CanonicalForm, canonical2: CanonicalForm) -> Violations:

    violations = _validate_canonical_compatibility(canonical1, canonical2)

    num_checks_1 = len(canonical1.library.check_model_types[0].checks)
    num_checks_2 = len(canonical2.library.check_model_types[0].checks)
    if num_checks_1 != num_checks_2:
        violations += (
            f"(ProgId 2.8) number of checks differ: {num_checks_1} != {num_checks_2}"
        )

    return violations


def _validate_canonical_compatibility(
    canonical1: CanonicalForm, canonical2: CanonicalForm, prefix: str = "ProgId 2."
) -> Violations:
    violations = Violations()

    num_observables_1 = len(canonical1.library.port_types[0].observables)
    num_observables_2 = len(canonical2.library.port_types[0].observables)
    if num_observables_1 != num_observables_2:
        # ProgId 2.1, ProgEq 2.1
        violations += (
            f"({prefix}1) number of observables differ: "
            + f"{num_observables_1} != {num_observables_2}"
        )

    num_measurements_1 = len(canonical1.library.gadget_types[0].measurements)
    num_measurements_2 = len(canonical2.library.gadget_types[0].measurements)
    if num_measurements_1 != num_measurements_2:
        # ProgId 2.2, ProgEq 2.2
        violations += (
            f"({prefix}2) number of measurements differ: "
            + f"{num_measurements_1} != {num_measurements_2}"
        )

    correction_propagation_1 = bitmatrix_of(
        canonical1.library.gadget_types[0].correction_propagation
    )
    correction_propagation_2 = bitmatrix_of(
        canonical2.library.gadget_types[0].correction_propagation
    )
    if correction_propagation_1 != correction_propagation_2:
        # ProgId 2.3, ProgEq 2.3
        violations += (
            f"({prefix}3) static correction propagation differ: "
            + f"{correction_propagation_1.T} != {correction_propagation_2.T}"
        )

    num_readouts_1 = len(canonical1.library.gadget_types[0].readouts)
    num_readouts_2 = len(canonical2.library.gadget_types[0].readouts)
    if num_readouts_1 != num_readouts_2:
        # ProgId 2.4, ProgEq 2.4
        violations += (
            f"({prefix}4) number of readouts differ: "
            + f"{num_readouts_1} != {num_readouts_2}"
        )

    readout_propagation_1 = bitmatrix_of(
        canonical1.library.gadget_types[0].readout_propagation
    )
    readout_propagation_2 = bitmatrix_of(
        canonical2.library.gadget_types[0].readout_propagation
    )
    if readout_propagation_1 != readout_propagation_2:
        # ProgId 2.5, ProgEq 2.5
        violations += (
            f"({prefix}5) static readout propagation differ: "
            + f"{readout_propagation_1.T} != {readout_propagation_2.T}"
        )

    logical_correction_1 = bitmatrix_of(
        canonical1.library.gadget_types[0].logical_correction
    )
    logical_correction_2 = bitmatrix_of(
        canonical2.library.gadget_types[0].logical_correction
    )
    if logical_correction_1 != logical_correction_2:
        # ProgId 2.6, ProgEq 2.6
        violations += (
            f"({prefix}6) logical correction differ: "
            + f"{logical_correction_1} != {logical_correction_2}"
        )

    physical_correction_1 = bitmatrix_of(
        canonical1.library.gadget_types[0].physical_correction
    )
    physical_correction_2 = bitmatrix_of(
        canonical2.library.gadget_types[0].physical_correction
    )
    if physical_correction_1 != physical_correction_2:
        # ProgId 2.7, ProgEq 2.7
        violations += (
            f"({prefix}7) physical correction differ: "
            + f"{physical_correction_1} != {physical_correction_2}"
        )

    return violations


def _canonical_check_map(
    canonical1: CanonicalForm, canonical2: CanonicalForm
) -> Violations | Bijection[CheckIndex]:

    # map: frozenset[measurement_index] -> list[check_index]
    check_group1: dict[frozenset[int], list[int]] = {}
    check_group2: dict[frozenset[int], list[int]] = {}
    for check_group, canonical in (
        (check_group1, canonical1),
        (check_group2, canonical2),
    ):
        for check_index, check in enumerate(
            canonical.library.check_model_types[0].checks
        ):
            measurements = frozenset(
                measurement.measurement_index for measurement in check.measurements
            )
            if measurements not in check_group:
                check_group[measurements] = []
            check_group[measurements].append(check_index)
    check_group1_keys = set(check_group1.keys())
    check_group2_keys = set(check_group2.keys())
    if check_group1_keys != check_group2_keys:
        only_1 = check_group1_keys - check_group2_keys
        only_2 = check_group2_keys - check_group1_keys
        return Violations(
            "(ProgId 3.1) the set of unique checks differ"
            + f": those only in lib1: {only_1}, those only in lib2: {only_2}"
        )

    keys = list(check_group1_keys)
    for key in keys:
        if len(check_group1[key]) != len(check_group2[key]):
            return Violations(
                f"(ProgId 3.2) number of checks for unique check differ: unique check={set(key)}"
                + f"{len(check_group1[key])} != {len(check_group2[key])}"
            )

    # we then simply build the bijection based on the order of the checks appearing
    canonical_check_map: Bijection[CheckIndex] = Bijection()
    for key, checks1 in check_group1.items():
        checks2 = check_group2[key]
        for name, checks, canonical in (
            ("lib1", checks1, canonical1),
            ("lib2", checks2, canonical2),
        ):
            naturally_flipped = (
                canonical.library.check_model_types[0]
                .checks[checks[0]]
                .naturally_flipped
            )
            for check_index in checks:
                if (
                    canonical.library.check_model_types[0]
                    .checks[check_index]
                    .naturally_flipped
                    != naturally_flipped
                ):
                    return Violations(
                        f"(ProgId 3.3) in {name}, not all checks for unique check share the "
                        + f"same naturally_flipped value ({set(key)})"
                    )
        naturally_flipped1 = (
            canonical1.library.check_model_types[0].checks[checks1[0]].naturally_flipped
        )
        naturally_flipped2 = (
            canonical2.library.check_model_types[0].checks[checks2[0]].naturally_flipped
        )
        if naturally_flipped1 != naturally_flipped2:
            return Violations(
                f"(ProgId 3.3) naturally_flipped differ for unique check {set(key)}: "
                + f"{naturally_flipped1} != {naturally_flipped2}"
            )
        for check_index_1, check_index_2 in zip(checks1, checks2):
            canonical_check_map.add(
                CheckIndex(cid=1, check_index=check_index_1),
                CheckIndex(cid=1, check_index=check_index_2),
            )

    return canonical_check_map


@dataclass(frozen=True, kw_only=True, eq=True)
class ErrorKey:
    checks: frozenset[int]
    residual: frozenset[int]
    readout_flips: frozenset[int]

    @staticmethod
    def from_error(
        error: pb.ErrorModelType.Error,
    ) -> "ErrorKey":
        return ErrorKey(
            checks=frozenset(check.check_index for check in error.checks),
            residual=frozenset(error.residual),
            readout_flips=frozenset(error.readout_flips),
        )


@dataclass
class GatheredError:
    probability: float = 0.0
    error_lists: list[int] = field(default_factory=list)

    def __iadd__(self, element: tuple[int, pb.ErrorModelType.Error]) -> "GatheredError":
        error_index, error = element
        p = error.probability
        self.probability = exclusive_probability_of(self.probability, p)
        self.error_lists.append(error_index)
        return self


def _validate_hyperedges_by_bijection(
    canonical1: CanonicalForm,
    canonical2: CanonicalForm,
    check_map: Bijection[CheckIndex],
    rel_tol: float,
) -> Violations:

    # first mirror all the errors in canonical1 to using mapped checks in canonical2
    mirrored_errors_1: list[pb.ErrorModelType.Error] = [
        pb.ErrorModelType.Error(
            probability=error.probability,
            residual=error.residual,
            readout_flips=error.readout_flips,
            checks=[
                pb.ErrorModelType.RemoteCheck(
                    check_index=check_map.atob[
                        CheckIndex(cid=1, check_index=check.check_index)
                    ].check_index
                )
                for check in error.checks
            ],
        )
        for error in canonical1.library.error_model_types[0].errors
    ]

    return _compare_mirrored_hyperedges(mirrored_errors_1, canonical2, rel_tol)


def _compare_mirrored_hyperedges(
    mirrored_errors_1: list[pb.ErrorModelType.Error],
    canonical2: CanonicalForm,
    rel_tol: float,
    prefix: str = "ProgId 4.",
) -> Violations:
    gathered_errors_1: dict[ErrorKey, GatheredError] = {}
    gathered_errors_2: dict[ErrorKey, GatheredError] = {}
    for errors, gathered_errors in (
        (mirrored_errors_1, gathered_errors_1),
        (list(canonical2.library.error_model_types[0].errors), gathered_errors_2),
    ):
        for error_index, error in enumerate(errors):
            key = ErrorKey.from_error(error)
            if key not in gathered_errors:
                gathered_errors[key] = GatheredError()
            gathered_errors[key] += (error_index, error)

    gathered_errors_1_keys = set(gathered_errors_1.keys())
    gathered_errors_2_keys = set(gathered_errors_2.keys())
    if gathered_errors_1_keys != gathered_errors_2_keys:
        only_1 = gathered_errors_1_keys - gathered_errors_2_keys
        only_2 = gathered_errors_2_keys - gathered_errors_1_keys
        # ProgId 4.1, ProgEq 4.1
        return Violations(
            f"({prefix}1) the set of unique errors differ"
            + f": those only in lib1: {only_1}, those only in lib2: {only_2}"
        )

    for key, gathered_error_1 in gathered_errors_1.items():
        gathered_error_2 = gathered_errors_2[key]
        p1 = gathered_error_1.probability
        p2 = gathered_error_2.probability
        if not math.isclose(p1, p2, rel_tol=rel_tol):
            # ProgId 4.2, ProgEq 4.2
            return Violations(
                f"({prefix}2) the gathered probabilities differ for unique error {key}: "
                + f"{p1} != {p2}"
            )

    return Violations()
