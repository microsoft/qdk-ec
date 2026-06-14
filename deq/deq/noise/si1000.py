"""Inject SI1000 noise into a plain DEQ file, preserving comments and structure.

The SI1000 (Standard Independent) noise model applies uniform-rate noise
to every gate operation:

- After each 1-qubit gate: ``DEPOLARIZE1(p)``
- After each 2-qubit gate: ``DEPOLARIZE2(p)``
- Each measurement: flip probability ``p`` embedded in the instruction,
  e.g. ``M(p) 0 1``
- After each Z-basis reset: ``X_ERROR(p)``
- After each X-basis reset: ``Z_ERROR(p)``

Idle noise is deliberately omitted — users who want realistic idle
depolarization should insert explicit ``I`` gates in their circuit.
"""


from .common import fmt_prob
from .injector import NoiseEmitter, inject_noise


class _SI1000Emitter(NoiseEmitter):
    def __init__(self, p: float) -> None:
        self._p = p

    def model_name(self) -> str:
        return "SI1000"

    def emit_before_measure(self, name: str) -> float | None:
        return self._p

    def emit_after_gate(
        self, targets: str, gate_kind: str, indent: str, ending: str
    ) -> list[str]:
        chan = "DEPOLARIZE2" if gate_kind == "2q" else "DEPOLARIZE1"
        return [f"{indent}{chan}({fmt_prob(self._p)}) {targets}{ending}"]

    def emit_after_reset(
        self, targets: str, basis: str, indent: str, ending: str
    ) -> list[str]:
        chan = "X_ERROR" if basis == "Z" else "Z_ERROR"
        return [f"{indent}{chan}({fmt_prob(self._p)}) {targets}{ending}"]


def inject_si1000(text: str, p: float) -> str:
    """Inject SI1000 noise into a plain ``.deq`` file.

    Parameters
    ----------
    text:
        The full text of a ``.deq`` file.  Must **not** contain
        Mako template syntax (``<% %>``, ``${…}``, ``% for …``).
    p:
        Physical error rate applied uniformly to every noise channel.

    Returns
    -------
    str
        The file text with SI1000 noise instructions inserted,
        preserving all comments, blank lines, and indentation.

    Raises
    ------
    ValueError
        If the file contains Mako template syntax or pre-existing
        noise instructions inside a GADGET block.
    """
    return inject_noise(text, _SI1000Emitter(p))
