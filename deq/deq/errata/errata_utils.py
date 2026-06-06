"""Shared errata utility functions for the DEQ transpiler and CLI.

Functions for remapping qubit indices in errata codes and operations,
extending code support, and converting between errata and stim formats.
Used by ``cli/logical_action.py``.
"""

from errata.codes import SubsystemCode
from errata.codes.separable import SeparableCode
from errata.pauli import Pauli, PauliGroup
from deq.circuit.model import CodeDefinition, InputPort, OutputPort, PauliProduct


def pauli_product_to_errata(product: PauliProduct) -> Pauli:
    """Convert a DEQ ``PauliProduct`` to an errata ``Pauli``.

    Each term in the product has a Pauli character (``X``, ``Y``, or
    ``Z``) and a qubit index.  Identity terms (``I``) are skipped.
    Duplicate qubit indices are rejected.
    """
    chars: dict[int, str] = {}
    for term in product.terms:
        if term.pauli == "I":
            continue
        if term.index in chars:
            raise ValueError(
                f"Duplicate qubit index {term.index} in Pauli product "
                f"('{chars[term.index]}{term.index}' and "
                f"'{term.pauli}{term.index}')"
            )
        chars[term.index] = term.pauli
    return Pauli(chars)


def remap_code(
    code_def: CodeDefinition,
    qubit_maps: list[dict[int, int]],
) -> SubsystemCode:
    """Build a SubsystemCode with qubit IDs remapped to circuit space.

    Redundant generators are reduced to independent ones.  Qubits in the
    port's index set that are not touched by any stabilizer or logical
    are exposed as gauge degrees of freedom so that ``action_of``
    quotients out their contribution.
    """
    all_generators: list[Pauli] = []
    all_logicals: list[Pauli] = []
    all_support: set[int] = set()

    for qmap in qubit_maps:
        for stab_pp in code_def.stabilizers:
            p = pauli_product_to_errata(stab_pp)
            all_generators.append(p.shuffled(by=qmap))
        for logical in code_def.logicals:
            px = pauli_product_to_errata(logical.x_operator)
            pz = pauli_product_to_errata(logical.z_operator)
            all_logicals.append(px.shuffled(by=qmap))
            all_logicals.append(pz.shuffled(by=qmap))
        all_support.update(qmap.values())

    independent: list[Pauli] = []
    for gen in all_generators:
        candidate = independent + [gen]
        if PauliGroup(candidate, all_commute=True).binary_rank == len(candidate):
            independent.append(gen)

    used_support = PauliGroup(list(independent) + list(all_logicals)).support
    extra = all_support - used_support
    gauge_basis = (
        list(SubsystemCode.standard_basis(over=sorted(extra))) if extra else None
    )
    return SubsystemCode(
        stabilizers=independent,
        logical_basis=all_logicals,
        gauge_basis=gauge_basis,
    )


def build_port_code(
    ports: list[InputPort | OutputPort],
    all_code_defs: dict[str, CodeDefinition],
) -> SubsystemCode | None:
    """Build a SubsystemCode for a list of ports, supporting multiple codes.

    Each port is looked up by ``code_name`` in *all_code_defs* and remapped
    to its physical qubit indices.  When ports use different codes (or the
    same code on disjoint qubits), the per-port codes are combined via
    ``SeparableCode`` into a single tensor-product code.

    Returns ``None`` if *ports* is empty.

    Raises :class:`ValueError` if a port's code name is not found.
    """
    if not ports:
        return None

    per_port_codes: list[SubsystemCode] = []
    for port in ports:
        if port.code_name not in all_code_defs:
            raise ValueError(f"code '{port.code_name}' not found in definitions")
        code_def = all_code_defs[port.code_name]
        qmap = dict(enumerate(port.qubit_indices))
        per_port_codes.append(remap_code(code_def, [qmap]))

    if len(per_port_codes) == 1:
        return per_port_codes[0]
    return SeparableCode(*per_port_codes)


def extend_support(
    code: SubsystemCode,
    circuit_qubits: set[int],
) -> SubsystemCode:
    """Return a copy of *code* whose support includes all *circuit_qubits*.

    Extra qubits beyond the original support are added as gauge degrees
    of freedom so that ``action_of`` quotients out their contribution.
    """
    extra = circuit_qubits - code.support
    if not extra:
        return code
    extra_gauge = list(SubsystemCode.standard_basis(over=sorted(extra)))
    return SubsystemCode(
        stabilizers=list(code.stabilizers),
        logical_basis=list(code.logical_basis),
        gauge_basis=list(code.gauge_basis) + extra_gauge,
    )


def make_trivial_code(circuit_qubits: set[int]) -> SubsystemCode:
    """Create a trivial code (no stabilizers, no logicals) over the given qubits.

    All physical qubits are gauge qubits, so any operation on them is
    quotiented out by ``action_of``.
    """
    return SubsystemCode(
        stabilizers=[],
        logical_basis=[],
        gauge_basis=list(SubsystemCode.standard_basis(over=sorted(circuit_qubits))),
    )


def make_wrt(
    code_in: SubsystemCode | None,
    code_out: SubsystemCode | None,
    circuit_qubits: set[int],
) -> tuple[SubsystemCode, SubsystemCode]:
    """Build the ``with_respect_to`` argument for ``action_of``."""
    if code_in is not None and code_out is not None:
        return (code_in, code_out)
    elif code_in is not None:
        return (code_in, make_trivial_code(circuit_qubits))
    elif code_out is not None:
        return (make_trivial_code(circuit_qubits), code_out)
    else:
        raise ValueError("At least one of code_in or code_out must be provided")
