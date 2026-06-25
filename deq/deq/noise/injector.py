"""Shared noise injection loop for ``.deq`` files.

Subclass :class:`NoiseEmitter` and override the ``emit_*`` methods to
define a custom noise model.  Call :func:`inject_noise` with the emitter
to apply it to a ``.deq`` source string.
"""

from abc import ABC, abstractmethod

from .common import (
    ALL_KNOWN_GADGET_INSTRUCTIONS,
    BLOCK_OPEN_RE,
    INSTRUCTION_RE,
    count_braces,
    extract_targets,
    fmt_prob,
    split_two_qubit_gate_targets,
    strip_comment,
    strip_parens,
)
from deq.transpiler.stim_constants import (
    NOISE_INSTRUCTIONS,
    ONE_QUBIT_GATES,
    PAIR_MEASURE_GATES,
    TWO_QUBIT_GATES,
    X_BASIS_MEASURE,
    X_BASIS_RESET,
    Y_BASIS_MEASURE,
    Y_BASIS_RESET,
    Z_BASIS_MEASURE,
    Z_BASIS_RESET,
)


class NoiseEmitter(ABC):
    """Override ``emit_*`` methods to define a noise model."""

    @abstractmethod
    def emit_before_measure(self, name: str) -> float | None:
        """Return measurement flip probability, or ``None`` for no noise.

        The probability is embedded directly in the measurement
        instruction, e.g. ``M(0.001) 0 1``.

        *name* is the uppercased instruction name, e.g. ``"M"``,
        ``"MX"``, ``"MRZ"``, ``"MXX"``, ``"MPP"``.
        """

    @abstractmethod
    def emit_after_gate(
        self, targets: str, gate_kind: str, indent: str, ending: str
    ) -> list[str]:
        """Return noise lines to insert *after* a gate instruction.

        *gate_kind* is ``"1q"`` or ``"2q"``.
        """

    @abstractmethod
    def emit_after_reset(
        self, targets: str, basis: str, indent: str, ending: str
    ) -> list[str]:
        """Return noise lines to insert *after* a reset instruction.

        *basis* is ``"Z"``, ``"X"``, or ``"Y"``.
        """

    def model_name(self) -> str:
        """Human-readable name for error messages."""
        return type(self).__name__

    def skip_gate(self, name_upper: str) -> bool:
        """Return True to skip noise injection for this gate entirely."""
        return False


def inject_noise(text: str, emitter: NoiseEmitter) -> str:
    """Walk a ``.deq`` source and insert noise via *emitter* callbacks."""
    lines = text.splitlines(keepends=True)
    result: list[str] = []
    block_stack: list[bool] = []

    for line in lines:
        opens, closes = count_braces(line)
        block_kind_match = BLOCK_OPEN_RE.match(strip_comment(line)) if opens else None

        if opens:
            if block_kind_match:
                kind = block_kind_match.group("kind")
                if kind == "GADGET":
                    is_gadget = True
                elif kind == "REPEAT":
                    is_gadget = block_stack[-1] if block_stack else False
                else:
                    is_gadget = False
            else:
                is_gadget = block_stack[-1] if block_stack else False
            for _ in range(opens):
                block_stack.append(is_gadget)

        in_gadget = bool(block_stack) and block_stack[-1]

        for _ in range(closes):
            if block_stack:
                block_stack.pop()

        if opens or closes:
            result.append(line)
            continue

        if not in_gadget:
            result.append(line)
            continue

        if line.lstrip().startswith("@"):
            result.append(line)
            continue

        m = INSTRUCTION_RE.match(line.rstrip("\n\r"))
        if not m:
            result.append(line)
            continue

        name_upper = m.group("name").upper()
        indent = m.group("indent")
        after = m.group("after")

        if name_upper not in ALL_KNOWN_GADGET_INSTRUCTIONS:
            raise ValueError(
                f"Unknown instruction '{m.group('name')}' found inside a "
                f"GADGET block.  {emitter.model_name()} noise injection does "
                f"not know how to handle this instruction.  If this is a "
                f"valid Stim gate, please update the gate classification in "
                f"deq/transpiler/stim_constants.py."
            )

        targets = extract_targets(after)
        ending = line[len(line.rstrip("\n\r")) :]
        if not ending:
            ending = "\n"

        if emitter.skip_gate(name_upper):
            result.append(line)
            continue

        # --- Measurement noise (modify the instruction probability) ---
        _ALL_MEASURE = (
            Z_BASIS_MEASURE | X_BASIS_MEASURE | Y_BASIS_MEASURE | PAIR_MEASURE_GATES
        )
        measure_prob: float | None = None
        if name_upper in _ALL_MEASURE or name_upper == "MPP":
            measure_prob = emitter.emit_before_measure(name_upper)

        if measure_prob is not None:
            after_clean = strip_parens(after) if "(" in after else after
            line = f"{indent}{m.group('name')}({fmt_prob(measure_prob)}){after_clean}{ending}"

        # --- The (possibly modified) instruction ---
        result.append(line)

        # --- Gate noise (after the instruction) ---
        if name_upper in TWO_QUBIT_GATES:
            qubit_pairs, single_qubits = split_two_qubit_gate_targets(after)
            if qubit_pairs:
                result.extend(emitter.emit_after_gate(qubit_pairs, "2q", indent, ending))
            if single_qubits:
                result.extend(emitter.emit_after_gate(single_qubits, "1q", indent, ending))
        elif targets and name_upper in ONE_QUBIT_GATES:
            result.extend(emitter.emit_after_gate(targets, "1q", indent, ending))

        # --- Reset noise (after the instruction) ---
        if targets and name_upper in Z_BASIS_RESET:
            result.extend(emitter.emit_after_reset(targets, "Z", indent, ending))
        elif targets and name_upper in X_BASIS_RESET:
            result.extend(emitter.emit_after_reset(targets, "X", indent, ending))
        elif targets and name_upper in Y_BASIS_RESET:
            result.extend(emitter.emit_after_reset(targets, "Y", indent, ending))

    return "".join(result)
