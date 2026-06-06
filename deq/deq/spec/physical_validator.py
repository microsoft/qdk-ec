"""
======================
Physical Specification
======================

Satisfying LibSpec and ProgSpec results in a valid decoding problem, but do not necessarily \
    corresponds to a physical one. A physical decoding problem should also make sure that all \
    errors have a possible physical explanation, i.e., each error should flip a set of\
    measurements, which then flips the set of checks that it produces

1. for every :code:`error: Error` among the error models in :code:`program: ExpandedProgram`:

    1. (PhySpec 1.1) it must not trigger empty checks, i.e., those checks that do not connect\
        to any measurement. While a check may be designed to be empty, it's never \
        flipped by any physically realistic error
    2. (PhySpec 1.2) let the checks it flips be :math:`\\mathcal{C}`, let the involved \
        measurements be :math:`\\mathcal{M} = \\cup_{c \\in \\mathcal{C}}\\ c`, and let all the \
        checks that involve these measurements be \
        :math:`\\mathcal{C}_M = \\{c | \\exists m \\in \\mathcal{M}, m \\in c \\}`, clearly we\
        have :math:`\\mathcal{C} \\subseteq \\mathcal{C}_M` given that there is no empty check.\
        We now require that there exists\
        at least one physical explanation of the error: that is, the error flips a subset of\
        measurements :math:`\\mathcal{M}_e \\subseteq \\mathcal{M}`, such that every check in\
        :math:`c \\in \\mathcal{C}_M` is flipped if and only if :math:`c \\in \\mathcal{C}`.\
        This is a :math:`\\mathbb{F}_2` linear equation involving :math:`|\\mathcal{M}|` variables\
        and :math:`|\\mathcal{C}_M|` constraints (among which :math:`|\\mathcal{C}|` are odd\
        parity constraints and the rest are even parity constraints).

2. for every :code:`check: Check` among the check models in :code:`program: ExpandedProgram`:

    1. (PhySpec 2.1) for those :code:`check: Check` that involve no measurement, \
        it must have a value of :code:`naturally_flipped` equal to :code:`False`. This is because\
        an empty check cannot be connected by any error as required by (PhySpec 1.1), \
        and thus for the problem to have a feasible solution, it must not be naturally flipped.
    2. (PhySpec 2.2) it must share the same value of :code:`naturally_flipped` with all the other\
        checks that involve the same set of measurements. This is to ensure that the check\
        has a consistent physical meaning.
    3. (PhySpec 2.3) there exists no loops of checks (a loop is defined by a set of checks such\
        that the symmetric difference of their involved measurements is empty) with odd parity
        of :code:`naturally_flipped`. This is because such loops would create contradictory\
        constraints on the involved measurements, making the decoding problem non-physical.\
        Note that PhySpec 2.1 and PhySpec 2.2 are special cases of this rule: they\
        forbid loops of size 1 with odd parity. However, we would like to list them explicitly\
        because they are easier to check and provide clearer error messages.

"""

import copy
from binar import BitMatrix, BitVector
import deq.proto.deq_bin_pb2 as pb
from deq.spec.violations import Violations
from deq.spec.program_validator import ExpandedProgram, is_valid
from deq.spec.common import MeasurementIndex, CheckIndex, linear_solve

# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint


def is_valid_and_physical(lib: pb.Library) -> "Violations | ExpandedProgram":
    program = is_valid(lib)
    if isinstance(program, Violations):
        return program

    physical_violations = is_physical(program)
    if not physical_violations:
        return physical_violations

    return program


def is_physical(program: ExpandedProgram) -> Violations:
    violations = Violations()
    violations += _validate_errors_physical(program)
    violations += _validate_checks_physical(program)
    return violations


def _validate_errors_physical(program: ExpandedProgram) -> Violations:
    violations = Violations()

    # the global measurement matrix is a bit matrix that solves a linear equation to find
    # a physical explanation for each error.
    # the $|M|$ columns correspond to the measurements, and the $|C|$ rows correspond to the checks.
    # For each error that triggers a set of checks, we then solve a linear equation to find a
    # set of measurements that triggers the same set of checks (considering the
    # naturally_flipped bits).

    global_measurement_matrix = BitMatrix.zeros(
        rows=len(program.global_checks), columns=len(program.global_measurements)
    )
    global_bit_flips = BitVector.zeros(length=len(program.global_checks))
    for row_index, global_check in enumerate(program.global_checks):
        cid = global_check.cid
        ci = global_check.check_index
        measurements = program.expanded_checks[cid][ci]

        naturally_flipped = program.check_model_types[cid].checks[ci].naturally_flipped

        for measurement in measurements:
            column_index = program.global_measurement_to_idx[measurement]
            global_measurement_matrix[(row_index, column_index)] = True
        global_bit_flips[row_index] = naturally_flipped

    for eid in sorted(program.expanded_errors.keys()):
        for ei, checks in enumerate(program.expanded_errors[eid]):

            if not checks:
                continue  # nothing to check

            empty_checks = [
                c for c in checks if not program.expanded_checks[c.cid][c.check_index]
            ]
            if empty_checks:
                violations += (
                    f"(PhySpec 1.1) error triggers empty checks: eid={eid} error_index={ei}"
                    + f" empty_checks={empty_checks}"
                )
                break

            rhs = copy.deepcopy(global_bit_flips)
            for check in checks:
                rhs[program.global_check_to_idx[check]] ^= True

            explanation = linear_solve(global_measurement_matrix, rhs)
            if explanation is None:
                return Violations(
                    "(PhySpec 1.2) error does not have a physical explanation: "
                    + f"eid={eid}, error_index={ei}, checks={checks}"
                )

    return violations


def _validate_checks_physical(program: ExpandedProgram) -> Violations:
    violations = Violations()

    # the global check matrix is a bit matrix that solves a linear equation to find loops
    # among the checks.
    # The $|C|$ columns correspond to the checks, and the first $|M|$ rows correspond to the
    # measurements that each check involves. The last row corresponds to the parity of
    # their naturally_flipped bit. The goal is to find a solution with odd parity in the last row
    # while even parity in all other rows, such that the selected checks sums to empty measurement
    # set with odd parity of naturally_flipped.
    global_check_matrix = BitMatrix.zeros(
        rows=len(program.global_measurements) + 1, columns=len(program.global_checks)
    )

    unique_checks: dict[frozenset[MeasurementIndex], tuple[CheckIndex, bool]] = {}
    for column_index, global_check in enumerate(program.global_checks):
        cid = global_check.cid
        ci = global_check.check_index
        measurements = program.expanded_checks[cid][ci]

        naturally_flipped = program.check_model_types[cid].checks[ci].naturally_flipped

        for measurement in measurements:
            row_index = program.global_measurement_to_idx[measurement]
            global_check_matrix[(row_index, column_index)] = True
        global_check_matrix[(len(program.global_measurements), column_index)] = (
            naturally_flipped
        )

        if not measurements:
            if naturally_flipped:
                violations += (
                    "(PhySpec 2.1) empty check must not be naturally flipped: "
                    + f"CheckIndex(cid={cid}, check_index={ci})"
                )
            continue

        key = frozenset(measurements)
        if key not in unique_checks:
            unique_checks[key] = (
                CheckIndex(cid=cid, check_index=ci),
                naturally_flipped,
            )
        else:
            previous_check, previous_naturally_flipped = unique_checks[key]
            if previous_naturally_flipped != naturally_flipped:
                violations += (
                    "(PhySpec 2.2) checks involving the same set of measurements "
                    + "have different naturally_flipped: "
                    + f"check1={previous_check}, "
                    + f"check2=CheckIndex(cid={cid}, check_index={ci}), "
                    + f"measurements={set(measurements)}"
                )

    if not violations:
        return violations

    constraint = BitVector.zeros(length=global_check_matrix.row_count)
    constraint[len(program.global_measurements)] = True
    solution = linear_solve(global_check_matrix, constraint)

    if solution is not None:
        violating_checks = [
            program.global_checks[i]
            for i in range(global_check_matrix.column_count)
            if solution[i]
        ]
        violations += (
            "(PhySpec 2.3) there exists a loop of checks with odd parity of "
            + f"naturally_flipped: violating_checks={violating_checks}"
        )

    return violations
