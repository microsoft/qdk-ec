"""Validate ``CodeDefinition`` algebraic constraints.

Checks that stabilizers and logical operators satisfy the required
commutation relations of a valid stabilizer code:

- All stabilizer and logical operator qubit indices are within ``[0, n)``.
- All stabilizers commute pairwise.
- All stabilizers commute with every logical operator.
- Within each logical qubit, the X and Z operators anticommute.
- Logical operators of *different* logical qubits commute.
"""


from deq.circuit.model import CodeDefinition, PauliProduct


def _pauli_products_commute(a: PauliProduct, b: PauliProduct) -> bool:
    """Return ``True`` if two Pauli products commute.

    Two multi-qubit Pauli operators commute iff the number of qubits
    where they have *different* non-identity Pauli types is even.
    """
    map_a = {t.index: t.pauli for t in a.terms if t.pauli != "I"}
    map_b = {t.index: t.pauli for t in b.terms if t.pauli != "I"}
    anticommuting_count = 0
    for q in map_a.keys() & map_b.keys():
        if map_a[q] != map_b[q]:
            anticommuting_count += 1
    return anticommuting_count % 2 == 0


def _code_location(code: CodeDefinition) -> str:
    """Return a ``" at file:line"`` suffix when source info is known, else ``""``."""
    if code.source_file is not None and code.source_line is not None:
        return f" at {code.source_file}:{code.source_line}"
    if code.source_line is not None:
        return f" at line {code.source_line}"
    return ""


def _check_qubit_indices_in_range(code: CodeDefinition) -> None:
    """Raise ``ValueError`` if any qubit index in stabilizers or logicals is >= n.

    This catches a common authoring mistake where ``[[n,k,d]]`` does not
    match the qubit indices actually used in the ``STABILIZER`` /
    ``LOGICAL`` declarations.  Without this check the offending index
    later triggers a deep ``IndexError`` from the JIT transpiler with no
    indication of which CODE or operator is at fault.
    """
    n = code.n
    operators: list[tuple[str, PauliProduct]] = []
    for idx, stab in enumerate(code.stabilizers):
        operators.append((f"STABILIZER #{idx} ({stab})", stab))
    for idx, logical in enumerate(code.logicals):
        operators.append(
            (f"LOGICAL X{idx} ({logical.x_operator})", logical.x_operator)
        )
        operators.append(
            (f"LOGICAL Z{idx} ({logical.z_operator})", logical.z_operator)
        )

    max_used = -1
    for _, op in operators:
        for term in op.terms:
            if term.pauli == "I":
                continue
            if term.index > max_used:
                max_used = term.index

    if max_used < n:
        return

    for label, op in operators:
        for term in op.terms:
            if term.pauli == "I":
                continue
            if term.index >= n:
                required_n = max_used + 1
                d_str = f",{code.d}" if code.d is not None else ""
                raise ValueError(
                    f"CODE {code.name!r}{_code_location(code)} declares "
                    f"[[{n},{code.k}{d_str}]] (n={n} physical qubit"
                    f"{'s' if n != 1 else ''}), but {label} uses qubit "
                    f"index {term.index}, which is out of range [0, {n}).\n"
                    f"  Hint: either increase n in the [[n,k,d]] header "
                    f"(need n >= {required_n} for the qubit indices used "
                    f"in this CODE) or remove the offending qubit index."
                )


def validate_code(code: CodeDefinition) -> None:
    """Validate the algebraic consistency of a ``CodeDefinition``.

    Raises ``ValueError`` with a descriptive message on the first
    violation found.
    """
    # 0. Qubit indices in stabilizers and logicals are within [0, n).
    # This must run before commutation checks; otherwise an out-of-range
    # index later surfaces as a bare ``IndexError`` from the JIT
    # transpiler with no hint at the offending CODE.
    _check_qubit_indices_in_range(code)

    # 1. Stabilizers commute pairwise.
    for i, si in enumerate(code.stabilizers):
        for j in range(i + 1, len(code.stabilizers)):
            sj = code.stabilizers[j]
            if not _pauli_products_commute(si, sj):
                raise ValueError(
                    f"CODE {code.name}: stabilizers {si} and {sj} "
                    f"do not commute"
                )

    all_logical_ops: list[tuple[str, PauliProduct]] = []
    for idx, logical in enumerate(code.logicals):
        all_logical_ops.append(
            (f"logical[{idx}].X ({logical.x_operator})", logical.x_operator)
        )
        all_logical_ops.append(
            (f"logical[{idx}].Z ({logical.z_operator})", logical.z_operator)
        )

    # 2. Stabilizers commute with all logical operators.
    for stab in code.stabilizers:
        for label, op in all_logical_ops:
            if not _pauli_products_commute(stab, op):
                raise ValueError(
                    f"CODE {code.name}: stabilizer {stab} does not "
                    f"commute with {label}"
                )

    # 3. Within each logical qubit, X and Z anticommute.
    for idx, logical in enumerate(code.logicals):
        if _pauli_products_commute(logical.x_operator, logical.z_operator):
            raise ValueError(
                f"CODE {code.name}: logical qubit {idx} operators "
                f"{logical.x_operator} (X) and {logical.z_operator} (Z) "
                f"commute, but they must anticommute"
            )

    # 4. Logical operators of different qubits commute.
    for i, li in enumerate(code.logicals):
        for j in range(i + 1, len(code.logicals)):
            lj = code.logicals[j]
            for label_a, op_a in [
                (f"X{i}", li.x_operator),
                (f"Z{i}", li.z_operator),
            ]:
                for label_b, op_b in [
                    (f"X{j}", lj.x_operator),
                    (f"Z{j}", lj.z_operator),
                ]:
                    if not _pauli_products_commute(op_a, op_b):
                        raise ValueError(
                            f"CODE {code.name}: logical operators "
                            f"{label_a} ({op_a}) and {label_b} ({op_b}) "
                            f"of different logical qubits do not commute"
                        )
