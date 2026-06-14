"""Inject biased (Z-biased on 2-qubit gates) noise into a plain DEQ file."""


from collections.abc import Sequence

from .common import fmt_prob
from .injector import NoiseEmitter, inject_noise


def _emit_z_bias_correlated(
    targets: str, p_total: float, indent: str, ending: str
) -> list[str]:
    """Emit CORRELATED_ERROR / ELSE_CORRELATED_ERROR for Z-biased 2-qubit noise."""
    qubits = targets.split()
    r = p_total / 3.0
    p1 = r
    p2 = r / (1.0 - r) if r < 1.0 else r
    p3 = r / (1.0 - 2.0 * r) if 2.0 * r < 1.0 else r
    lines: list[str] = []
    for i in range(0, len(qubits), 2):
        q1, q2 = qubits[i], qubits[i + 1]
        lines.append(f"{indent}CORRELATED_ERROR({fmt_prob(p1)}) Z{q1}{ending}")
        lines.append(f"{indent}ELSE_CORRELATED_ERROR({fmt_prob(p2)}) Z{q2}{ending}")
        lines.append(
            f"{indent}ELSE_CORRELATED_ERROR({fmt_prob(p3)}) Z{q1} Z{q2}{ending}"
        )
    return lines


class _BiasedEmitter(NoiseEmitter):
    def __init__(
        self,
        p1q: float,
        p2q_sym: float,
        p2q_zbias: float,
        excluded: frozenset[str],
    ) -> None:
        self._p1q = p1q
        self._p2q_sym = p2q_sym
        self._p2q_zbias = p2q_zbias
        self._excluded = excluded

    def model_name(self) -> str:
        return "biased"

    def skip_gate(self, name_upper: str) -> bool:
        return name_upper in self._excluded

    def emit_before_measure(self, name: str) -> float | None:
        return self._p1q

    def emit_after_gate(
        self, targets: str, gate_kind: str, indent: str, ending: str
    ) -> list[str]:
        if gate_kind == "2q":
            lines = [
                f"{indent}DEPOLARIZE2({fmt_prob(self._p2q_sym)}) {targets}{ending}"
            ]
            lines.extend(
                _emit_z_bias_correlated(targets, self._p2q_zbias, indent, ending)
            )
            return lines
        return [f"{indent}DEPOLARIZE1({fmt_prob(self._p1q)}) {targets}{ending}"]

    def emit_after_reset(
        self, targets: str, basis: str, indent: str, ending: str
    ) -> list[str]:
        chan = "X_ERROR" if basis == "Z" else "Z_ERROR"
        return [f"{indent}{chan}({fmt_prob(self._p1q)}) {targets}{ending}"]


def inject_biased(
    text: str,
    p: float,
    p1q: float | None = None,
    eta: float = 10.0,
    exclude: Sequence[str] = (),
) -> str:
    """Inject biased noise into a plain ``.deq`` file.

    The biased noise model applies asymmetric noise rates to 1-qubit and
    2-qubit gates, with a strong Z-bias on entangling gates.  It corresponds
    to ``make_noise_model_biased("exclusive", [], p1q, p, eta)`` in bedec.

    Parameters
    ----------
    text:
        The full text of a ``.deq`` file.  Must **not** contain
        Mako template syntax.
    p:
        Physical error rate for 2-qubit gates.
    p1q:
        Physical error rate for 1-qubit gates, measurements, and resets.
        Defaults to ``p / 4`` when not provided.
    eta:
        Bias parameter controlling Z-error weight on 2-qubit gates.
        Default is ``10.0``.
    exclude:
        Gate names to skip noise injection on (case-insensitive).

    Returns
    -------
    str
        The file text with biased noise instructions inserted.

    Raises
    ------
    ValueError
        If the file contains Mako template syntax or pre-existing
        noise instructions inside a GADGET block.
    """
    if p1q is None:
        p1q = p / 4.0
    p2q_sym = p / (eta + 1.0)
    p2q_zbias = eta * p / (eta + 1.0)
    excluded = frozenset(g.upper() for g in exclude)
    return inject_noise(text, _BiasedEmitter(p1q, p2q_sym, p2q_zbias, excluded))
