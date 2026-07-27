"""Map deqagram's parsed AST onto deq's ``model.py`` dataclasses.

deqagram (a Rust ``pest`` parser exposed via PyO3) parses ``.deq`` source into
its own typed AST. This shim converts that AST into the ``model.py`` model deq's
transpiler consumes, so deqagram is the parser behind
:func:`deq.circuit.parser.parse` without ``model.py`` or any consumer changing.

Body-level decorators are folded onto the statement that follows them via
deqagram's ``parse_attached`` (which runs deqagram's own attachment pass),
mirroring ``model.py`` where each statement carries its own ``decorators`` list.

This is an incremental port: ``CODE`` definitions are supported; the other
definition kinds raise :class:`NotImplementedError` for now.
"""

from __future__ import annotations

import warnings

import deqagram

from deq.circuit import body_validation, model


def _warn_dangling(dangling: list[deqagram.Decorator]) -> None:
    """Warn about decorators with no statement to attach to.

    deqagram's attachment pass surfaces these separately rather than discarding
    them, so a dangling decorator is a warning here and is then ignored.
    """
    if dangling:
        names = ", ".join(f"@{d.name}" for d in dangling)
        warnings.warn(
            f"Decorators {names} at end of block have no target "
            f"and will be ignored",
            stacklevel=2,
        )


# deqagram's Pauli enum is not hashable, but its int value is stable
# (I=0, X=1, Y=2, Z=3), matching this order.
_PAULI_LETTERS = ("I", "X", "Y", "Z")


def _pauli_letter(pauli: object) -> str:
    return _PAULI_LETTERS[int(pauli)]


def _decorator_arg(arg: object) -> model.DecoratorArg:
    """Convert a deqagram ``DecoratorArg`` to a ``model.DecoratorArg``."""
    match arg:
        case deqagram.DecoratorArg.Keyword(key, value):
            return model.KeywordArg(key=key, value=_decorator_value(value))
        case deqagram.DecoratorArg.Value(value):
            return _decorator_value(value)
        case _:
            raise TypeError(f"unexpected decorator argument: {arg!r}")


def _decorator_value(value: object) -> str | int | float:
    """Convert a deqagram ``DecoratorValue`` to its Python scalar."""
    match value:
        case deqagram.DecoratorValue.String(v):
            return v
        case deqagram.DecoratorValue.Int(v):
            return v
        case deqagram.DecoratorValue.Float(v):
            return v
        case _:
            raise TypeError(f"unexpected decorator value: {value!r}")


def _decorator(decorator: deqagram.Decorator) -> model.Decorator:
    return model.Decorator(
        name=decorator.name,
        arguments=tuple(_decorator_arg(a) for a in decorator.arguments),
    )


def _pauli_product(product: object) -> model.PauliProduct:
    """Convert a deqagram ``PauliProduct`` to a ``model.PauliProduct``.

    The identity product ``_`` carries no factors, so it maps to an empty term
    tuple rather than to a term holding an explicit identity Pauli.
    """
    match product:
        case deqagram.PauliProduct.Identity():
            return model.PauliProduct(terms=())
        case deqagram.PauliProduct.Terms(terms):
            return model.PauliProduct(
                terms=tuple(
                    model.PauliTerm(pauli=_pauli_letter(t.pauli), index=t.index)
                    for t in terms
                )
            )
        case _:
            raise TypeError(f"unexpected Pauli product: {product!r}")


# ── Targets ──────────────────────────────────────────────────────────


def _port_kind(kind: object) -> str:
    """Map a deqagram ``PortKind`` (In/Out) to model's ``"IN"``/``"OUT"``."""
    return "IN" if kind == deqagram.PortKind.In else "OUT"


def _logical_pauli(target: deqagram.LogicalPauliTarget) -> model.LogicalPauliTarget:
    port = target.port
    return model.LogicalPauliTarget(
        pauli=_pauli_letter(target.pauli),
        index=target.index,
        port_kind=_port_kind(port.kind) if port is not None else None,
        port_index=port.index if port is not None else None,
    )


def _measurement_ref(ref: object) -> model.MeasurementRefTarget:
    match ref:
        case deqagram.MeasurementRef.Record(offset):
            return model.MeasurementRecordTarget(offset=offset)
        case deqagram.MeasurementRef.Physical(index):
            return model.PhysicalMeasurementTarget(index=index)
        case deqagram.MeasurementRef.InputVirtual(port, stabilizer):
            return model.InputVirtualTarget(
                port_index=port, stabilizer_index=stabilizer
            )
        case deqagram.MeasurementRef.OutputVirtual(port, stabilizer):
            return model.OutputVirtualTarget(
                port_index=port, stabilizer_index=stabilizer
            )
        case _:
            raise TypeError(f"unexpected measurement reference: {ref!r}")


def _target(target: object) -> model.Target:
    match target:
        case deqagram.Target.Qubit(inverted, index):
            return model.QubitTarget(index=index, inverted=inverted)
        case deqagram.Target.Pauli(inverted, pauli, index):
            return model.PauliTarget(
                pauli=_pauli_letter(pauli), index=index, inverted=inverted
            )
        case deqagram.Target.MeasurementRecord(offset):
            return model.MeasurementRecordTarget(offset=offset)
        case deqagram.Target.PhysicalMeasurement(index):
            return model.PhysicalMeasurementTarget(index=index)
        case deqagram.Target.InputVirtual(port, stabilizer):
            return model.InputVirtualTarget(
                port_index=port, stabilizer_index=stabilizer
            )
        case deqagram.Target.OutputVirtual(port, stabilizer):
            return model.OutputVirtualTarget(
                port_index=port, stabilizer_index=stabilizer
            )
        case deqagram.Target.SweepBit(index):
            return model.SweepBitTarget(index=index)
        case deqagram.Target.Combiner():
            return model.CombinerTarget()
        case _:
            raise TypeError(f"unexpected target: {target!r}")


def _readout_item(item: object) -> model.ReadoutTargetItem:
    match item:
        case deqagram.ReadoutTargetItem.Target(target):
            return _target(target)
        case deqagram.ReadoutTargetItem.Logical(logical):
            return _logical_pauli(logical)
        case deqagram.ReadoutTargetItem.Destabilizer(port, stabilizer):
            return model.DestabilizerTarget(port_index=port, stab_index=stabilizer)
        case _:
            raise TypeError(f"unexpected readout item: {item!r}")


def _error_target(target: object) -> model.ErrorTarget:
    match target:
        case deqagram.ErrorTarget.Check(index):
            return model.CheckTarget(index=index)
        case deqagram.ErrorTarget.Readout(index):
            return model.ReadoutTarget(index=index)
        case deqagram.ErrorTarget.Logical(logical):
            return _logical_pauli(logical)
        case _:
            raise TypeError(f"unexpected error target: {target!r}")


def _propagate_term(term: object) -> model.PropagateTerm:
    match term:
        case deqagram.PropagateTerm.Logical(logical):
            return _logical_pauli(logical)
        case deqagram.PropagateTerm.Destabilizer(port, stabilizer):
            return model.DestabilizerTarget(port_index=port, stab_index=stabilizer)
        case deqagram.PropagateTerm.MeasurementRecord(offset):
            return model.MeasurementRecordTarget(offset=offset)
        case deqagram.PropagateTerm.PhysicalMeasurement(index):
            return model.PhysicalMeasurementTarget(index=index)
        case deqagram.PropagateTerm.Readout(index):
            return model.ReadoutTarget(index=index)
        case _:
            raise TypeError(f"unexpected propagate term: {term!r}")


def _condition(condition: object) -> model.ReadoutTarget | model.MeasurementRefTarget:
    match condition:
        case deqagram.Condition.Readout(index):
            return model.ReadoutTarget(index=index)
        case deqagram.Condition.Measurement(measurement):
            return _measurement_ref(measurement)
        case _:
            raise TypeError(f"unexpected condition: {condition!r}")


def _pauli_pairs(paulis: object) -> list[tuple[str, int]]:
    """Convert a list of deqagram ``PauliTerm`` to ``(letter, index)`` tuples."""
    return [(_pauli_letter(t.pauli), t.index) for t in paulis]


def _tag_line(statement: object, source_line: int | None) -> object:
    """Record ``source_line`` on a statement that carries that field.

    Statement dataclasses that support source locations (Instruction, ports,
    REPEAT blocks, CONDITIONAL/PROPAGATE/PRESELECT, gadget applications) get the
    line from their deqagram span, so semantic diagnostics can point at them.
    Statements without the field (CHECK/READOUT/ERROR/VIRTUAL) are left as-is.
    """
    fields = getattr(type(statement), "__dataclass_fields__", {})
    if "source_line" in fields and getattr(statement, "source_line", None) is None:
        statement.source_line = source_line
    return statement


# ── Leaf statements (shared across body kinds) ───────────────────────


def _instruction(
    instruction: deqagram.Instruction, decorators: list[model.Decorator]
) -> model.Instruction:
    return model.Instruction(
        name=instruction.name,
        tag=instruction.tag,
        arguments=list(instruction.arguments),
        targets=[_target(t) for t in instruction.targets],
        decorators=decorators,
    )


def _input_port(
    port: deqagram.PortDeclaration, decorators: list[model.Decorator]
) -> model.InputPort:
    return model.InputPort(
        code_name=port.code_name,
        qubit_indices=list(port.qubit_indices),
        decorators=decorators,
    )


def _output_port(
    port: deqagram.PortDeclaration, decorators: list[model.Decorator]
) -> model.OutputPort:
    return model.OutputPort(
        code_name=port.code_name,
        qubit_indices=list(port.qubit_indices),
        decorators=decorators,
    )


def _gadget_application(
    application: deqagram.GadgetApplication,
    decorators: list[model.Decorator],
    source_line: int | None,
) -> model.GadgetApplication:
    return model.GadgetApplication(
        gadget_name=application.gadget_name,
        in_indices=(
            list(application.in_indices) if application.in_indices is not None else None
        ),
        out_indices=(
            list(application.out_indices)
            if application.out_indices is not None
            else None
        ),
        decorators=decorators,
        source_line=source_line,
    )


def _conditional_correction(
    correction: deqagram.ConditionalCorrection,
) -> model.ConditionalCorrection:
    return model.ConditionalCorrection(
        readout_offset=correction.readout_offset,
        paulis=_pauli_pairs(correction.paulis),
        wire=correction.wire,
    )


def _repeat_block(
    count: int, body: list[object], decorators: list[model.Decorator]
) -> model.RepeatBlock:
    """Build a RepeatBlock, applying deq's REPEAT-count and port-in-body rules.

    deqagram's parser accepts these permissively; deq rejects a count below 1
    and INPUT/OUTPUT ports inside a REPEAT, so enforce both here for parity.
    """
    if count < 1:
        raise SyntaxError(f"REPEAT count must be >= 1, got {count}")
    body_validation.validate_repeat_body(body)
    return model.RepeatBlock(count=count, body=body, decorators=decorators)


# ── Body-statement dispatch (per definition kind) ────────────────────


def _gadget_statement(
    decorated: deqagram.DecoratedGadget, source: str | None
) -> model.GadgetStatement:
    statement = _gadget_statement_impl(decorated, source)
    return _tag_line(statement, _source_line(decorated.span, source))


def _gadget_statement_impl(
    decorated: deqagram.DecoratedGadget, source: str | None
) -> model.GadgetStatement:
    decorators = [_decorator(d) for d in decorated.decorators]
    match decorated.statement:
        case deqagram.AttachedGadget.Repeat(count, body):
            return _repeat_block(
                count, [_gadget_statement(s, source) for s in body], decorators
            )
        case deqagram.AttachedGadget.Statement(
            deqagram.GadgetStatement.Instruction(instruction)
        ):
            return _instruction(instruction, decorators)
        case deqagram.AttachedGadget.Statement(
            deqagram.GadgetStatement.InputPort(port)
        ):
            return _input_port(port, decorators)
        case deqagram.AttachedGadget.Statement(
            deqagram.GadgetStatement.OutputPort(port)
        ):
            return _output_port(port, decorators)
        case deqagram.AttachedGadget.Statement(
            deqagram.GadgetStatement.Readout(readout)
        ):
            return model.ReadoutStatement(
                targets=[_readout_item(t) for t in readout.targets],
                flip=readout.flip,
                decorators=decorators,
            )
        case deqagram.AttachedGadget.Statement(deqagram.GadgetStatement.Check(check)):
            return model.CheckStatement(
                targets=[_target(t) for t in check.targets],
                flip=check.flip,
                decorators=decorators,
            )
        case deqagram.AttachedGadget.Statement(deqagram.GadgetStatement.Error(error)):
            if not 0.0 <= error.probability <= 1.0:
                raise SyntaxError(
                    f"ERROR probability must be in [0, 1], got {error.probability}"
                )
            return model.ErrorStatement(
                probability=error.probability,
                targets=[_error_target(t) for t in error.targets],
                decorators=decorators,
            )
        case deqagram.AttachedGadget.Statement(
            deqagram.GadgetStatement.Conditional(conditional)
        ):
            return model.ConditionalStatement(
                condition=_condition(conditional.condition),
                targets=[_logical_pauli(t) for t in conditional.targets],
                decorators=decorators,
            )
        case deqagram.AttachedGadget.Statement(
            deqagram.GadgetStatement.VirtualLogical(virtual)
        ):
            return model.VirtualLogicalStatement(
                targets=[_logical_pauli(t) for t in virtual.targets],
                decorators=decorators,
            )
        case deqagram.AttachedGadget.Statement(
            deqagram.GadgetStatement.Propagate(propagate)
        ):
            return model.PropagateStatement(
                target=_logical_pauli(propagate.target),
                terms=[_propagate_term(t) for t in propagate.terms],
                flip=propagate.flip,
                decorators=decorators,
            )
        case deqagram.AttachedGadget.Statement(
            deqagram.GadgetStatement.Preselect(preselect)
        ):
            if preselect.expected_value not in (0, 1):
                raise SyntaxError(
                    f"PRESELECT expected parity must be 0 or 1; "
                    f"got {preselect.expected_value}"
                )
            return model.PreselectStatement(
                conditions=[_measurement_ref(c) for c in preselect.conditions],
                expected_value=preselect.expected_value,
                decorators=decorators,
            )
        case _:
            raise TypeError(f"unexpected gadget statement: {decorated.statement!r}")


def _compose_statement(
    decorated: deqagram.DecoratedCompose, source: str | None
) -> model.ComposeStatement:
    statement = _compose_statement_impl(decorated, source)
    return _tag_line(statement, _source_line(decorated.span, source))


def _compose_statement_impl(
    decorated: deqagram.DecoratedCompose, source: str | None
) -> model.ComposeStatement:
    decorators = [_decorator(d) for d in decorated.decorators]
    match decorated.statement:
        case deqagram.AttachedCompose.Repeat(count, body):
            return _repeat_block(
                count, [_compose_statement(s, source) for s in body], decorators
            )
        case deqagram.AttachedCompose.Statement(
            deqagram.ComposeStatement.Instruction(instruction)
        ):
            return _instruction(instruction, decorators)
        case deqagram.AttachedCompose.Statement(
            deqagram.ComposeStatement.InputPort(port)
        ):
            return _input_port(port, decorators)
        case deqagram.AttachedCompose.Statement(
            deqagram.ComposeStatement.OutputPort(port)
        ):
            return _output_port(port, decorators)
        case deqagram.AttachedCompose.Statement(
            deqagram.ComposeStatement.GadgetApplication(application)
        ):
            return _gadget_application(
                application, decorators, _source_line(decorated.span, source)
            )
        case deqagram.AttachedCompose.Statement(
            deqagram.ComposeStatement.ConditionalCorrection(correction)
        ):
            return _conditional_correction(correction)
        case _:
            raise TypeError(f"unexpected compose statement: {decorated.statement!r}")


def _program_statement(
    decorated: deqagram.DecoratedProgram, source: str | None
) -> model.ProgramStatement:
    statement = _program_statement_impl(decorated, source)
    return _tag_line(statement, _source_line(decorated.span, source))


def _program_statement_impl(
    decorated: deqagram.DecoratedProgram, source: str | None
) -> model.ProgramStatement:
    decorators = [_decorator(d) for d in decorated.decorators]
    match decorated.statement:
        case deqagram.AttachedProgram.Repeat(count, body):
            return _repeat_block(
                count, [_program_statement(s, source) for s in body], decorators
            )
        case deqagram.AttachedProgram.Statement(
            deqagram.ProgramStatement.Instruction(instruction)
        ):
            return _instruction(instruction, decorators)
        case deqagram.AttachedProgram.Statement(
            deqagram.ProgramStatement.InputPort(port)
        ):
            return _input_port(port, decorators)
        case deqagram.AttachedProgram.Statement(
            deqagram.ProgramStatement.OutputPort(port)
        ):
            return _output_port(port, decorators)
        case deqagram.AttachedProgram.Statement(
            deqagram.ProgramStatement.GadgetApplication(application)
        ):
            return _gadget_application(
                application, decorators, _source_line(decorated.span, source)
            )
        case deqagram.AttachedProgram.Statement(
            deqagram.ProgramStatement.Assert(assertion)
        ):
            return model.AssertStatement(
                target=_target(assertion.target),
                expected_value=assertion.expected_value,
                decorators=decorators,
            )
        case deqagram.AttachedProgram.Statement(
            deqagram.ProgramStatement.VirtualCorrection(correction)
        ):
            return model.VirtualCorrection(
                paulis=_pauli_pairs(correction.paulis), wire=correction.wire
            )
        case deqagram.AttachedProgram.Statement(
            deqagram.ProgramStatement.ConditionalCorrection(correction)
        ):
            return _conditional_correction(correction)
        case _:
            raise TypeError(f"unexpected program statement: {decorated.statement!r}")


def _code_definition(
    code: deqagram.CodeDefinition,
    *,
    source_line: int | None = None,
) -> model.CodeDefinition:
    # deqagram parses the [[n,k,d]] header permissively; deq rejects
    # out-of-range parameters at parse time (deqagram's `k` is unsigned, so a
    # k >= 0 check would be vacuous and is omitted).
    # The LOGICAL count is checked last so a malformed header is reported first.
    if code.n < 1:
        raise SyntaxError(f"CODE parameter n must be >= 1, got {code.n}")
    if code.k > code.n:
        raise SyntaxError(f"CODE parameter k ({code.k}) must be <= n ({code.n})")
    if code.d is not None and code.d < 1:
        raise SyntaxError(f"CODE parameter d must be >= 1, got {code.d}")
    if len(code.logicals) != code.k:
        raise SyntaxError(
            f"CODE {code.name!r} declares [[{code.n},{code.k}]] but has "
            f"{len(code.logicals)} LOGICAL declaration(s); expected {code.k}"
        )
    return model.CodeDefinition(
        name=code.name,
        n=code.n,
        k=code.k,
        d=code.d,
        logicals=[
            model.LogicalOperator(
                x_operator=_pauli_product(logical.x_operator),
                z_operator=_pauli_product(logical.z_operator),
            )
            for logical in code.logicals
        ],
        stabilizers=[_pauli_product(s) for s in code.stabilizers],
        decorators=[_decorator(d) for d in code.decorators],
        source_line=source_line,
    )


def _source_line(span: deqagram.Span, source: str | None) -> int | None:
    """Resolve a span's 1-based source line, if the source text is available."""
    if source is None:
        return None
    location = span.line_col(source)
    return location[0] if location is not None else None


def _definition(definition: object, source: str | None) -> model.Definition:
    match definition:
        case deqagram.AttachedDefinition.Code() as code_def:
            return _code_definition(
                code_def.code,
                source_line=_source_line(code_def.span, source),
            )
        case deqagram.AttachedDefinition.Gadget() as gadget_def:
            _warn_dangling(gadget_def.dangling)
            body = [_gadget_statement(s, source) for s in gadget_def.body]
            body_validation.validate_gadget_body(body, gadget_def.name)
            return model.GadgetDefinition(
                name=gadget_def.name,
                body=body,
                decorators=[_decorator(d) for d in gadget_def.decorators],
                source_line=_source_line(gadget_def.span, source),
            )
        case deqagram.AttachedDefinition.Compose() as compose_def:
            _warn_dangling(compose_def.dangling)
            return model.ComposeDefinition(
                name=compose_def.name,
                body=[_compose_statement(s, source) for s in compose_def.body],
                decorators=[_decorator(d) for d in compose_def.decorators],
                source_line=_source_line(compose_def.span, source),
            )
        case deqagram.AttachedDefinition.Program() as program_def:
            _warn_dangling(program_def.dangling)
            return model.ProgramDefinition(
                name=program_def.name,
                body=[_program_statement(s, source) for s in program_def.body],
                decorators=[_decorator(d) for d in program_def.decorators],
                source_line=_source_line(program_def.span, source),
            )
        case _:
            raise TypeError(f"unexpected definition: {definition!r}")


def to_model(
    file: deqagram.AttachedDeqFile,
    *,
    source: str | None = None,
    source_file: str | None = None,
) -> model.DeqFile:
    """Convert a deqagram ``AttachedDeqFile`` to a ``model.DeqFile``.

    When ``source`` (the original text) is given, definition ``source_line``
    fields are populated from deqagram's spans, so deq's diagnostics can point at
    the offending line. ``source_file`` is recorded on the returned file.
    """
    return model.DeqFile(
        definitions=[_definition(d, source) for d in file.definitions],
        imports=[model.ImportStatement(path=path) for path in file.imports],
        source_file=source_file,
    )


def parse(text: str, *, source_file: str | None = None) -> model.DeqFile:
    """Parse ``.deq`` ``text`` via deqagram and return a ``model.DeqFile``."""
    return to_model(deqagram.parse_attached(text), source=text, source_file=source_file)
