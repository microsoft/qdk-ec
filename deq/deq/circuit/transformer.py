"""Lark Transformer that builds the semantic model from a parse tree."""

import re
from typing import Any, Literal

from lark import Token, Transformer, Tree

from deq.circuit.model import (
    AssertStatement,
    CheckStatement,
    CheckTarget,
    CodeDefinition,
    CombinerTarget,
    ComposeDefinition,
    ConditionalCorrection,
    ConditionalStatement,
    DestabilizerTarget,
    Decorator,
    DecoratorArg,
    Definition,
    ErrorStatement,
    ErrorTarget,
    GadgetApplication,
    GadgetDefinition,
    ImportStatement,
    InputPort,
    InputVirtualTarget,
    Instruction,
    KeywordArg,
    LogicalOperator,
    LogicalPauliTarget,
    MeasurementRecordTarget,
    MeasurementRefTarget,
    OutputPort,
    OutputVirtualTarget,
    PauliProduct,
    PauliTarget,
    PauliTerm,
    PhysicalMeasurementTarget,
    PreselectStatement,
    ProgramDefinition,
    PropagateStatement,
    PropagateTerm,
    DeqFile,
    QubitTarget,
    ReadoutStatement,
    ReadoutTarget,
    ReadoutTargetItem,
    RepeatBlock,
    SweepBitTarget,
    Target,
    VirtualCorrection,
    VirtualLogicalStatement,
)
from deq.transpiler.stim_constants import (
    ANNOTATION_INSTRUCTIONS,
    instruction_num_measurements,
)

_REC_RE = re.compile(r"rec\[-(\d+)\]")
_SWEEP_RE = re.compile(r"sweep\[(\d+)\]")
_PAULI_RE = re.compile(r"([IXYZ])(\d+)")
_INPUT_DESTAB_RE = re.compile(r"IN(\d+)\.DS(\d+)")
_PHYS_MEAS_RE = re.compile(r"M(\d+)")
_INPUT_VIRTUAL_RE = re.compile(r"IN(\d+)\.S(\d+)")
_OUTPUT_VIRTUAL_RE = re.compile(r"OUT(\d+)\.S(\d+)")

# Token types whose lexeme refers to a measurement (relative or absolute).
_MEAS_REF_TOKEN_TYPES = frozenset(
    {
        "MEASUREMENT_RECORD_TARGET",
        "PHYS_MEAS_TARGET",
        "INPUT_VIRTUAL_TARGET",
        "OUTPUT_VIRTUAL_TARGET",
    }
)


def _token_to_measurement_ref(token: Token) -> MeasurementRefTarget:
    """Convert any of the four measurement-reference tokens to its AST node."""
    text = str(token)
    if token.type == "MEASUREMENT_RECORD_TARGET":
        m = _REC_RE.match(text)
        if not m:
            raise SyntaxError(f"invalid measurement record target: {text!r}")
        return MeasurementRecordTarget(offset=int(m.group(1)))
    if token.type == "PHYS_MEAS_TARGET":
        m = _PHYS_MEAS_RE.match(text)
        if not m:
            raise SyntaxError(f"invalid physical measurement target: {text!r}")
        return PhysicalMeasurementTarget(index=int(m.group(1)))
    if token.type == "INPUT_VIRTUAL_TARGET":
        m = _INPUT_VIRTUAL_RE.match(text)
        if not m:
            raise SyntaxError(f"invalid input virtual target: {text!r}")
        return InputVirtualTarget(
            port_index=int(m.group(1)), stabilizer_index=int(m.group(2))
        )
    if token.type == "OUTPUT_VIRTUAL_TARGET":
        m = _OUTPUT_VIRTUAL_RE.match(text)
        if not m:
            raise SyntaxError(f"invalid output virtual target: {text!r}")
        return OutputVirtualTarget(
            port_index=int(m.group(1)), stabilizer_index=int(m.group(2))
        )
    raise SyntaxError(f"unexpected measurement-reference token: {token!r}")


# Body-level semantic validators live in `body_validation` so they can be
# shared with the deqagram parser shim; alias them under the private names
# this module's transformer methods call.
from deq.circuit.body_validation import (  # noqa: E402
    validate_conditional_after_output as _validate_conditional_after_output,
    validate_port_ordering as _validate_port_ordering,
    validate_preselect as _validate_preselect,
    validate_propagate_after_output as _validate_propagate_after_output,
    validate_repeat_body as _validate_repeat_body,
)


class DeqTransformer(Transformer):
    """Bottom-up transformer: rule methods receive already-transformed children."""

    # ── Top-level ────────────────────────────────────────────────────

    def start(self, items: list[Any]) -> DeqFile:
        definitions: list[Definition] = []
        imports: list[ImportStatement] = []
        for item in items:
            if isinstance(item, ImportStatement):
                imports.append(item)
            elif isinstance(item, tuple):
                # (decorators, definition) from decorated_definition
                decorators, defn = item
                defn.decorators = decorators
                definitions.append(defn)
        return DeqFile(definitions=definitions, imports=imports)

    def import_statement(self, items: list[Any]) -> ImportStatement:
        return ImportStatement(path=items[0])

    def decorated_definition(
        self, items: list[Any]
    ) -> tuple[list[Decorator], Definition]:
        decorators: list[Decorator] = []
        defn: Definition | None = None
        for item in items:
            if isinstance(item, Decorator):
                decorators.append(item)
            else:
                defn = item
        if defn is None:
            raise SyntaxError("decorated_definition contains no definition")
        return (decorators, defn)

    # ── CODE ─────────────────────────────────────────────────────────

    def code_definition(self, items: list[Any]) -> CodeDefinition:
        name_token = items[0]
        name = str(name_token)
        line = name_token.line if hasattr(name_token, "line") else None
        n, k, d = items[1]  # code_params returns (n, k, d)
        logicals: list[LogicalOperator] = []
        stabilizers: list[PauliProduct] = []
        for item in items[2:]:
            if isinstance(item, LogicalOperator):
                logicals.append(item)
            elif isinstance(item, list):
                # stabilizer_declaration returns list of PauliProducts
                stabilizers.extend(item)
            elif isinstance(item, PauliProduct):
                stabilizers.append(item)
        if len(logicals) != k:
            raise SyntaxError(
                f"CODE {name!r} declares [[{n},{k}]] but has "
                f"{len(logicals)} LOGICAL declaration(s); expected {k}"
            )
        return CodeDefinition(
            name=name,
            n=n,
            k=k,
            d=d,
            logicals=logicals,
            stabilizers=stabilizers,
            source_line=line,
        )

    def code_params(self, items: list[Token]) -> tuple[int, int, int | None]:
        n = int(items[0])
        k = int(items[1])
        d = int(items[2]) if len(items) > 2 and items[2] is not None else None
        if n < 1:
            raise SyntaxError(f"CODE parameter n must be >= 1, got {n}")
        if k < 0:
            raise SyntaxError(f"CODE parameter k must be >= 0, got {k}")
        if k > n:
            raise SyntaxError(f"CODE parameter k ({k}) must be <= n ({n})")
        if d is not None and d < 1:
            raise SyntaxError(f"CODE parameter d must be >= 1, got {d}")
        return (n, k, d)

    def logical_declaration(self, items: list[PauliProduct]) -> LogicalOperator:
        return LogicalOperator(x_operator=items[0], z_operator=items[1])

    def stabilizer_declaration(self, items: list[PauliProduct]) -> list[PauliProduct]:
        # Return list of PauliProducts; code_definition collects them
        return items

    def pauli_product(self, items: list[Token]) -> PauliProduct:
        terms = []
        for token in items:
            text = str(token)
            if text == "_":
                # Identity placeholder — empty pauli product.
                continue
            m = _PAULI_RE.match(text)
            if not m:
                raise SyntaxError(f"invalid Pauli term: {token!r}")
            terms.append(PauliTerm(pauli=m.group(1), index=int(m.group(2))))
        return PauliProduct(terms=tuple(terms))

    # ── GADGET ───────────────────────────────────────────────────────

    def gadget_definition(self, items: list[Any]) -> GadgetDefinition:
        name_token = items[0]
        name = str(name_token)
        line = name_token.line if hasattr(name_token, "line") else None
        body = _collect_body_with_decorators(items[1:])
        _validate_port_ordering(body, name)
        _validate_conditional_after_output(body, name)
        _validate_propagate_after_output(body, name)
        _validate_preselect(body, name)
        return GadgetDefinition(name=name, body=body, source_line=line)

    def repeat_block_gadget(self, items: list[Any]) -> RepeatBlock:
        count = int(items[0])
        if count < 1:
            raise SyntaxError(f"REPEAT count must be >= 1, got {count}")
        body = _collect_body_with_decorators(items[1:])
        _validate_repeat_body(body)
        return RepeatBlock(count=count, body=body)

    # ── COMPOSE ──────────────────────────────────────────────────────

    def compose_definition(self, items: list[Any]) -> ComposeDefinition:
        name_token = items[0]
        name = str(name_token)
        line = name_token.line if hasattr(name_token, "line") else None
        body = _collect_body_with_decorators(items[1:])
        return ComposeDefinition(name=name, body=body, source_line=line)

    def repeat_block_compose(self, items: list[Any]) -> RepeatBlock:
        count = int(items[0])
        if count < 1:
            raise SyntaxError(f"REPEAT count must be >= 1, got {count}")
        body = _collect_body_with_decorators(items[1:])
        _validate_repeat_body(body)
        return RepeatBlock(count=count, body=body)

    # ── PROGRAM ──────────────────────────────────────────────────────

    def program_definition(self, items: list[Any]) -> ProgramDefinition:
        name_token = items[0]
        name = str(name_token)
        line = name_token.line if hasattr(name_token, "line") else None
        body = _collect_body_with_decorators(items[1:])
        return ProgramDefinition(name=name, body=body, source_line=line)

    def repeat_block_program(self, items: list[Any]) -> RepeatBlock:
        count = int(items[0])
        if count < 1:
            raise SyntaxError(f"REPEAT count must be >= 1, got {count}")
        body = _collect_body_with_decorators(items[1:])
        _validate_repeat_body(body)
        return RepeatBlock(count=count, body=body)

    # ── Shared: ports ────────────────────────────────────────────────

    def input_port(self, items: list[Token]) -> InputPort:
        # items[0] is INPUT_KW, items[1] is code name, rest are indices
        code_name = str(items[1])
        indices = [int(t) for t in items[2:]]
        return InputPort(code_name=code_name, qubit_indices=indices)

    def output_port(self, items: list[Token]) -> OutputPort:
        # items[0] is OUTPUT_KW, items[1] is code name, rest are indices
        code_name = str(items[1])
        indices = [int(t) for t in items[2:]]
        return OutputPort(code_name=code_name, qubit_indices=indices)

    # ── Gadget application ───────────────────────────────────────────

    def gadget_application(self, items: list[Any]) -> GadgetApplication:
        gadget_name = str(items[0])
        in_indices: list[int] | None = None
        out_indices: list[int] | None = None
        for item in items[1:]:
            if isinstance(item, tuple):
                kind, indices = item
                if kind == "in":
                    in_indices = indices
                else:
                    out_indices = indices
            # "()" token: explicit no-port call, leave both as None
        line = items[0].line if hasattr(items[0], "line") else None
        return GadgetApplication(
            gadget_name=gadget_name,
            in_indices=in_indices,
            out_indices=out_indices,
            source_line=line,
        )

    def port_binding_in(self, items: list[Token]) -> tuple[str, list[int]]:
        return ("in", [int(t) for t in items])

    def port_binding_out(self, items: list[Token]) -> tuple[str, list[int]]:
        return ("out", [int(t) for t in items])

    # ── GADGET-only statements ───────────────────────────────────────

    def readout_statement(self, items: list[Any]) -> ReadoutStatement:
        flip = any(isinstance(t, Token) and t.type == "FLIP_KW" for t in items)
        targets: list[ReadoutTargetItem] = []
        for t in items:
            if t is None:
                continue
            if isinstance(t, Token) and t.type == "FLIP_KW":
                continue
            if isinstance(t, Token) and t.type == "INPUT_DESTAB_TARGET":
                m = _INPUT_DESTAB_RE.match(str(t))
                if not m:
                    raise SyntaxError(
                        f"invalid INPUT destabilizer target on READOUT: {t!r}"
                    )
                targets.append(
                    DestabilizerTarget(
                        port_index=int(m.group(1)),
                        stab_index=int(m.group(2)),
                    )
                )
                continue
            targets.append(t)
        return ReadoutStatement(targets=targets, flip=flip)

    def check_statement(self, items: list[Any]) -> CheckStatement:
        flip = any(isinstance(t, Token) and t.type == "FLIP_KW" for t in items)
        targets: list[Target] = [
            t
            for t in items
            if t is not None and not (isinstance(t, Token) and t.type == "FLIP_KW")
        ]
        return CheckStatement(targets=targets, flip=flip)

    def error_statement(self, items: list[Any]) -> ErrorStatement:
        probability = float(items[0])
        if not (0.0 <= probability <= 1.0):
            raise SyntaxError(f"ERROR probability must be in [0, 1], got {probability}")
        targets: list[ErrorTarget] = list(items[1:])
        return ErrorStatement(probability=probability, targets=targets)

    def conditional_statement(self, items: list[Any]) -> ConditionalStatement:
        condition = items[0]
        if isinstance(condition, Token) and condition.type in _MEAS_REF_TOKEN_TYPES:
            condition = _token_to_measurement_ref(condition)
        targets = [i for i in items[1:] if isinstance(i, LogicalPauliTarget)]
        return ConditionalStatement(condition=condition, targets=targets)

    def preselect_statement(self, items: list[Any]) -> PreselectStatement:
        non_token_items = [
            it
            for it in items
            if not (isinstance(it, Token) and it.type == "PRESELECT_KW")
        ]
        condition_token = non_token_items[0]
        if not (
            isinstance(condition_token, Token)
            and condition_token.type in _MEAS_REF_TOKEN_TYPES
        ):
            raise SyntaxError(
                "PRESELECT requires a rec[-k]/M<i>/IN<p>.S<s>/OUT<p>.S<s> target; "
                f"got {condition_token!r}"
            )
        condition = _token_to_measurement_ref(condition_token)
        expected_value = int(non_token_items[1])
        if expected_value not in (0, 1):
            raise SyntaxError(
                f"PRESELECT expected value must be 0 or 1; got {expected_value}"
            )
        return PreselectStatement(condition=condition, expected_value=expected_value)

    def virtual_logical_statement(self, items: list[Any]) -> VirtualLogicalStatement:
        targets = [item for item in items if isinstance(item, LogicalPauliTarget)]
        return VirtualLogicalStatement(targets=targets)

    def propagate_statement(self, items: list[Any]) -> PropagateStatement:
        if not items or not isinstance(items[0], LogicalPauliTarget):
            raise SyntaxError("PROPAGATE requires a logical target on the left of FROM")
        target = items[0]
        flip = False
        terms: list[PropagateTerm] = []
        for item in items[1:]:
            if item is None:
                continue
            if isinstance(item, LogicalPauliTarget):
                terms.append(item)
            elif isinstance(item, ReadoutTarget):
                terms.append(item)
            elif isinstance(item, Token) and item.type == "INPUT_DESTAB_TARGET":
                m = _INPUT_DESTAB_RE.match(str(item))
                if not m:
                    raise SyntaxError(f"invalid INPUT destabilizer target: {item!r}")
                terms.append(
                    DestabilizerTarget(
                        port_index=int(m.group(1)),
                        stab_index=int(m.group(2)),
                    )
                )
            elif isinstance(item, Token) and item.type in _MEAS_REF_TOKEN_TYPES:
                terms.append(_token_to_measurement_ref(item))
            elif isinstance(item, Token) and item.type == "FLIP_KW":
                flip ^= True
            elif isinstance(item, Token) and item.type == "FROM_KW":
                continue
            else:
                raise SyntaxError(f"unexpected PROPAGATE term: {item!r}")
        return PropagateStatement(target=target, terms=terms, flip=flip)

    def check_target(self, items: list[Token]) -> CheckTarget:
        return CheckTarget(index=int(str(items[0])[1:]))

    def readout_target(self, items: list[Token]) -> ReadoutTarget:
        return ReadoutTarget(index=int(str(items[0])[1:]))

    def logical_pauli_target(self, items: list[Token]) -> LogicalPauliTarget:
        text = str(items[0])
        # Bare ``L<P><i>`` (port-global).
        if text.startswith("L"):
            return LogicalPauliTarget(pauli=text[1], index=int(text[2:]))
        # Port-qualified ``IN<p>.L<P><i>`` / ``OUT<p>.L<P><i>``.
        prefix, suffix = text.split(".", 1)
        if prefix.startswith("IN"):
            port_kind: Literal["IN", "OUT"] = "IN"
            port_index = int(prefix[2:])
        else:
            port_kind = "OUT"
            port_index = int(prefix[3:])
        return LogicalPauliTarget(
            pauli=suffix[1],
            index=int(suffix[2:]),
            port_kind=port_kind,
            port_index=port_index,
        )

    def error_pauli_target(self, items: list[Token]) -> PauliTarget:
        m = _PAULI_RE.match(str(items[0]))
        if not m:
            raise SyntaxError(f"invalid Pauli target: {items[0]!r}")
        return PauliTarget(pauli=m.group(1), index=int(m.group(2)), inverted=False)

    # ── PROGRAM-only statements ──────────────────────────────────────

    def assert_statement(self, items: list[Any]) -> AssertStatement:
        return AssertStatement(target=items[0], expected_value=int(items[1]))

    def virtual_correction(self, items: list[Any]) -> VirtualCorrection:
        paulis: list[tuple[str, int]] = []
        for item in items:
            tok = str(item)
            if tok == "VIRTUAL" or tok == "*":
                continue
            m = _PAULI_RE.match(tok)
            if m:
                paulis.append((m.group(1), int(m.group(2))))
                continue
            # Last item is the wire integer
            if tok.isdigit():
                wire = int(tok)
        if not paulis:
            raise SyntaxError("VIRTUAL requires at least one Pauli operator")
        return VirtualCorrection(paulis=paulis, wire=wire)

    def conditional_correction(self, items: list[Any]) -> ConditionalCorrection:
        # Grammar: MEASUREMENT_RECORD_TARGET PAULI_OPERATOR (COMBINER PAULI_OPERATOR)* INT
        rec_token = items[0]
        m = _REC_RE.match(str(rec_token))
        if not m:
            raise SyntaxError(
                f"CONDITIONAL requires a rec[-k] readout reference; got {rec_token!r}"
            )
        readout_offset = int(m.group(1))
        if readout_offset < 1:
            raise SyntaxError(
                f"CONDITIONAL rec[-k] requires k >= 1; got rec[-{readout_offset}]"
            )
        # Last item is the wire integer; intermediate items are Pauli operators
        # (and COMBINER tokens which are filtered out).
        wire_token = items[-1]
        if not str(wire_token).isdigit():
            raise SyntaxError(
                f"CONDITIONAL requires a non-negative wire integer at the end; "
                f"got {wire_token!r}"
            )
        wire = int(wire_token)
        paulis: list[tuple[str, int]] = []
        for item in items[1:-1]:
            tok = str(item)
            if tok == "*":
                continue
            pm = _PAULI_RE.match(tok)
            if pm:
                paulis.append((pm.group(1), int(pm.group(2))))
        if not paulis:
            raise SyntaxError(
                "CONDITIONAL requires at least one Pauli operator between "
                "the readout reference and the wire"
            )
        return ConditionalCorrection(
            readout_offset=readout_offset, paulis=paulis, wire=wire
        )

    # ── Stim instructions ────────────────────────────────────────────

    def instruction(self, items: list[Any]) -> Instruction:
        name = str(items[0])
        tag: str | None = None
        arguments: list[float] = []
        targets: list[Target] = []
        for item in items[1:]:
            if item is None:
                continue
            if isinstance(item, str):
                tag = item
            elif isinstance(item, list):
                # Could be arguments (list[float]) or shouldn't happen
                arguments = item
            elif isinstance(
                item,
                (
                    QubitTarget,
                    MeasurementRecordTarget,
                    PhysicalMeasurementTarget,
                    InputVirtualTarget,
                    OutputVirtualTarget,
                    SweepBitTarget,
                    PauliTarget,
                    CombinerTarget,
                ),
            ):
                targets.append(item)
        return Instruction(name=name, tag=tag, arguments=arguments, targets=targets)

    def tag(self, items: list[Any]) -> str | None:
        if not items or items[0] is None:
            return ""
        return _decode_tag(str(items[0]))

    def parenthesized_arguments(self, items: list[Token]) -> list[float]:
        return [float(t) for t in items]

    def target(self, items: list[Any]) -> Target:
        item = items[0]
        if isinstance(item, Token):
            text = str(item)
            if item.type in _MEAS_REF_TOKEN_TYPES:
                return _token_to_measurement_ref(item)
            if item.type == "SWEEP_BIT_TARGET":
                m = _SWEEP_RE.match(text)
                if not m:
                    raise SyntaxError(f"invalid sweep bit target: {text!r}")
                return SweepBitTarget(index=int(m.group(1)))
            if item.type == "COMBINER":
                return CombinerTarget()
        # Already transformed by qubit_target/pauli_target
        return item

    def qubit_target(self, items: list[Token]) -> QubitTarget:
        return QubitTarget(index=int(items[0]), inverted=False)

    def inverted_qubit_target(self, items: list[Token]) -> QubitTarget:
        return QubitTarget(index=int(items[1]), inverted=True)

    def pauli_target(self, items: list[Token]) -> PauliTarget:
        m = _PAULI_RE.match(str(items[0]))
        if not m:
            raise SyntaxError(f"invalid Pauli target: {items[0]!r}")
        return PauliTarget(pauli=m.group(1), index=int(m.group(2)), inverted=False)

    def inverted_pauli_target(self, items: list[Token]) -> PauliTarget:
        m = _PAULI_RE.match(str(items[1]))
        if not m:
            raise SyntaxError(f"invalid Pauli target: {items[1]!r}")
        return PauliTarget(pauli=m.group(1), index=int(m.group(2)), inverted=True)

    # ── Decorators ───────────────────────────────────────────────────

    def decorator(self, items: list[Any]) -> Decorator:
        name = str(items[0]).lstrip("@")
        arguments: tuple[DecoratorArg, ...] = ()
        if len(items) > 1 and items[1] is not None:
            arguments = tuple(items[1])
        return Decorator(name=name, arguments=arguments)

    def decorator_arguments(self, items: list[Any]) -> list[DecoratorArg]:
        return [item for item in items if item is not None]

    def keyword_argument(self, items: list[Any]) -> KeywordArg:
        key = str(items[0])
        value = items[1]
        return KeywordArg(key=key, value=value)

    def decorator_value(self, items: list[Any]) -> str | int | float:
        item = items[0]
        if isinstance(item, Token):
            text = str(item)
            if item.type == "INT":
                return int(text)
            # NUMBER token
            if "." in text or "e" in text.lower():
                return float(text)
            return int(text)
        # Already decoded string from string_literal
        return item

    def string_literal(self, items: list[Token]) -> str:
        raw = str(items[0])
        # Strip surrounding quotes
        return _decode_escaped_string(raw[1:-1])


def _collect_body_with_decorators(items: list[Any]) -> list[Any]:
    """Attach pending decorators to the following statement."""
    result: list[Any] = []
    pending: list[Decorator] = []
    for item in items:
        if isinstance(item, Decorator):
            pending.append(item)
            continue
        # Handle stabilizer_declaration returning a list of PauliProducts
        if isinstance(item, list):
            for sub_item in item:
                result.append(sub_item)
            continue
        if hasattr(item, "decorators"):
            item.decorators = pending
            pending = []
        result.append(item)
    if pending:
        import warnings

        names = [f"@{d.name}" for d in pending]
        warnings.warn(
            f"Decorators {', '.join(names)} at end of block have no target "
            f"and will be ignored",
            stacklevel=2,
        )
    return result


def _decode_escaped_string(raw: str) -> str:
    """Decode escape sequences in a string literal."""
    result: list[str] = []
    i = 0
    while i < len(raw):
        if raw[i] == "\\" and i + 1 < len(raw):
            c = raw[i + 1]
            if c == "n":
                result.append("\n")
            elif c == "r":
                result.append("\r")
            elif c == "t":
                result.append("\t")
            elif c == "\\":
                result.append("\\")
            elif c == '"':
                result.append('"')
            else:
                result.append(c)
            i += 2
        else:
            result.append(raw[i])
            i += 1
    return "".join(result)


def _decode_tag(raw: str) -> str:
    return (
        raw.replace("\\C", "]")
        .replace("\\r", "\r")
        .replace("\\n", "\n")
        .replace("\\B", "\\")
    )
