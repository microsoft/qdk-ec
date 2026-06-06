"""Validate ``CodeDefinition`` algebraic constraints.

Checks that stabilizers and logical operators satisfy the required
commutation relations of a valid stabilizer code:

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


def validate_code(code: CodeDefinition) -> None:
    """Validate the algebraic consistency of a ``CodeDefinition``.

    Raises ``ValueError`` with a descriptive message on the first
    violation found.
    """
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
