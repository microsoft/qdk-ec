"""Map deqagram's parsed AST onto deq's ``model.py`` dataclasses.

deqagram (a Rust ``pest`` parser exposed via PyO3) parses ``.deq`` source into
its own typed AST. This shim converts that AST into the ``model.py`` model deq's
transpiler consumes, so deqagram can replace the lark grammar + transformer
behind :func:`deq.circuit.parser.parse` without changing ``model.py`` or any
consumer.

Body-level decorators are folded onto the statement that follows them via
deqagram's ``parse_attached`` (which runs deqagram's own attachment pass),
mirroring ``model.py`` where each statement carries its own ``decorators`` list.

This is an incremental port: ``CODE`` definitions are supported; the other
definition kinds raise :class:`NotImplementedError` for now.
"""

from __future__ import annotations

import warnings

import deqagram

from deq.circuit import model


def _warn_dangling(dangling: list[deqagram.Decorator]) -> None:
    """Warn about decorators with no statement to attach to.

    deqagram's attachment pass surfaces these separately; deq's transformer
    warns (and ignores) them, so replicate that behavior for parity.
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
    if isinstance(arg, deqagram.DecoratorArg.Keyword):
        return model.KeywordArg(key=arg.key, value=_decorator_value(arg.value))
    if isinstance(arg, deqagram.DecoratorArg.Value):
        return _decorator_value(arg.value)
    raise TypeError(f"unexpected decorator argument: {arg!r}")


def _decorator_value(value: object) -> str | int | float:
    """Convert a deqagram ``DecoratorValue`` to its Python scalar."""
    if isinstance(value, deqagram.DecoratorValue.String):
        return value.value
    if isinstance(value, deqagram.DecoratorValue.Int):
        return value.value
    if isinstance(value, deqagram.DecoratorValue.Float):
        return value.value
    raise TypeError(f"unexpected decorator value: {value!r}")


def _decorator(decorator: deqagram.Decorator) -> model.Decorator:
    return model.Decorator(
        name=decorator.name,
        arguments=tuple(_decorator_arg(a) for a in decorator.arguments),
    )


def _pauli_product(product: object) -> model.PauliProduct:
    """Convert a deqagram ``PauliProduct`` to a ``model.PauliProduct``.

    The identity product ``_`` maps to an empty term tuple, matching deq's
    transformer.
    """
    if isinstance(product, deqagram.PauliProduct.Identity):
        return model.PauliProduct(terms=())
    if isinstance(product, deqagram.PauliProduct.Terms):
        return model.PauliProduct(
            terms=tuple(
                model.PauliTerm(pauli=_pauli_letter(t.pauli), index=t.index)
                for t in product.terms
            )
        )
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
    if isinstance(ref, deqagram.MeasurementRef.Record):
        return model.MeasurementRecordTarget(offset=ref.offset)
    if isinstance(ref, deqagram.MeasurementRef.Physical):
        return model.PhysicalMeasurementTarget(index=ref.index)
    if isinstance(ref, deqagram.MeasurementRef.InputVirtual):
        return model.InputVirtualTarget(
            port_index=ref.port, stabilizer_index=ref.stabilizer
        )
    if isinstance(ref, deqagram.MeasurementRef.OutputVirtual):
        return model.OutputVirtualTarget(
            port_index=ref.port, stabilizer_index=ref.stabilizer
        )
    raise TypeError(f"unexpected measurement reference: {ref!r}")


def _target(target: object) -> model.Target:
    if isinstance(target, deqagram.Target.Qubit):
        return model.QubitTarget(index=target.index, inverted=target.inverted)
    if isinstance(target, deqagram.Target.Pauli):
        return model.PauliTarget(
            pauli=_pauli_letter(target.pauli),
            index=target.index,
            inverted=target.inverted,
        )
    if isinstance(target, deqagram.Target.MeasurementRecord):
        return model.MeasurementRecordTarget(offset=target.offset)
    if isinstance(target, deqagram.Target.PhysicalMeasurement):
        return model.PhysicalMeasurementTarget(index=target.index)
    if isinstance(target, deqagram.Target.InputVirtual):
        return model.InputVirtualTarget(
            port_index=target.port, stabilizer_index=target.stabilizer
        )
    if isinstance(target, deqagram.Target.OutputVirtual):
        return model.OutputVirtualTarget(
            port_index=target.port, stabilizer_index=target.stabilizer
        )
    if isinstance(target, deqagram.Target.SweepBit):
        return model.SweepBitTarget(index=target.index)
    if isinstance(target, deqagram.Target.Combiner):
        return model.CombinerTarget()
    raise TypeError(f"unexpected target: {target!r}")


def _readout_item(item: object) -> model.ReadoutTargetItem:
    if isinstance(item, deqagram.ReadoutTargetItem.Target):
        return _target(item.target)
    if isinstance(item, deqagram.ReadoutTargetItem.Logical):
        return _logical_pauli(item.logical)
    if isinstance(item, deqagram.ReadoutTargetItem.Destabilizer):
        return model.DestabilizerTarget(
            port_index=item.port, stab_index=item.stabilizer
        )
    raise TypeError(f"unexpected readout item: {item!r}")


def _error_target(target: object) -> model.ErrorTarget:
    if isinstance(target, deqagram.ErrorTarget.Check):
        return model.CheckTarget(index=target.index)
    if isinstance(target, deqagram.ErrorTarget.Readout):
        return model.ReadoutTarget(index=target.index)
    if isinstance(target, deqagram.ErrorTarget.Logical):
        return _logical_pauli(target.logical)
    if isinstance(target, deqagram.ErrorTarget.Pauli):
        return model.PauliTarget(pauli=_pauli_letter(target.pauli), index=target.index)
    raise TypeError(f"unexpected error target: {target!r}")


def _propagate_term(term: object) -> model.PropagateTerm:
    if isinstance(term, deqagram.PropagateTerm.Logical):
        return _logical_pauli(term.logical)
    if isinstance(term, deqagram.PropagateTerm.Destabilizer):
        return model.DestabilizerTarget(
            port_index=term.port, stab_index=term.stabilizer
        )
    if isinstance(term, deqagram.PropagateTerm.MeasurementRecord):
        return model.MeasurementRecordTarget(offset=term.offset)
    if isinstance(term, deqagram.PropagateTerm.PhysicalMeasurement):
        return model.PhysicalMeasurementTarget(index=term.index)
    if isinstance(term, deqagram.PropagateTerm.Readout):
        return model.ReadoutTarget(index=term.index)
    raise TypeError(f"unexpected propagate term: {term!r}")


def _condition(condition: object) -> model.ReadoutTarget | model.MeasurementRefTarget:
    if isinstance(condition, deqagram.Condition.Readout):
        return model.ReadoutTarget(index=condition.index)
    if isinstance(condition, deqagram.Condition.Measurement):
        return _measurement_ref(condition.measurement)
    raise TypeError(f"unexpected condition: {condition!r}")


def _pauli_pairs(paulis: object) -> list[tuple[str, int]]:
    """Convert a list of deqagram ``PauliTerm`` to ``(letter, index)`` tuples."""
    return [(_pauli_letter(t.pauli), t.index) for t in paulis]


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


# ── Body-statement dispatch (per definition kind) ────────────────────


def _gadget_statement(
    decorated: deqagram.DecoratedGadget, source: str | None
) -> model.GadgetStatement:
    decorators = [_decorator(d) for d in decorated.decorators]
    statement = decorated.statement
    if isinstance(statement, deqagram.AttachedGadget.Repeat):
        return model.RepeatBlock(
            count=statement.count,
            body=[_gadget_statement(s, source) for s in statement.body],
            decorators=decorators,
        )
    leaf = statement.statement
    if isinstance(leaf, deqagram.GadgetStatement.Instruction):
        return _instruction(leaf.instruction, decorators)
    if isinstance(leaf, deqagram.GadgetStatement.InputPort):
        return _input_port(leaf.port, decorators)
    if isinstance(leaf, deqagram.GadgetStatement.OutputPort):
        return _output_port(leaf.port, decorators)
    if isinstance(leaf, deqagram.GadgetStatement.Readout):
        return model.ReadoutStatement(
            targets=[_readout_item(t) for t in leaf.readout.targets],
            flip=leaf.readout.flip,
            decorators=decorators,
        )
    if isinstance(leaf, deqagram.GadgetStatement.Check):
        return model.CheckStatement(
            targets=[_target(t) for t in leaf.check.targets],
            flip=leaf.check.flip,
            decorators=decorators,
        )
    if isinstance(leaf, deqagram.GadgetStatement.Error):
        return model.ErrorStatement(
            probability=leaf.error.probability,
            targets=[_error_target(t) for t in leaf.error.targets],
            decorators=decorators,
        )
    if isinstance(leaf, deqagram.GadgetStatement.Conditional):
        return model.ConditionalStatement(
            condition=_condition(leaf.conditional.condition),
            targets=[_logical_pauli(t) for t in leaf.conditional.targets],
            decorators=decorators,
        )
    if isinstance(leaf, deqagram.GadgetStatement.VirtualLogical):
        return model.VirtualLogicalStatement(
            targets=[_logical_pauli(t) for t in leaf.statement.targets],
            decorators=decorators,
        )
    if isinstance(leaf, deqagram.GadgetStatement.Propagate):
        return model.PropagateStatement(
            target=_logical_pauli(leaf.propagate.target),
            terms=[_propagate_term(t) for t in leaf.propagate.terms],
            flip=leaf.propagate.flip,
            decorators=decorators,
        )
    if isinstance(leaf, deqagram.GadgetStatement.Preselect):
        return model.PreselectStatement(
            condition=_measurement_ref(leaf.preselect.condition),
            expected_value=leaf.preselect.expected_value,
            decorators=decorators,
        )
    raise TypeError(f"unexpected gadget statement: {leaf!r}")


def _compose_statement(
    decorated: deqagram.DecoratedCompose, source: str | None
) -> model.ComposeStatement:
    decorators = [_decorator(d) for d in decorated.decorators]
    statement = decorated.statement
    if isinstance(statement, deqagram.AttachedCompose.Repeat):
        return model.RepeatBlock(
            count=statement.count,
            body=[_compose_statement(s, source) for s in statement.body],
            decorators=decorators,
        )
    leaf = statement.statement
    if isinstance(leaf, deqagram.ComposeStatement.Instruction):
        return _instruction(leaf.instruction, decorators)
    if isinstance(leaf, deqagram.ComposeStatement.InputPort):
        return _input_port(leaf.port, decorators)
    if isinstance(leaf, deqagram.ComposeStatement.OutputPort):
        return _output_port(leaf.port, decorators)
    if isinstance(leaf, deqagram.ComposeStatement.GadgetApplication):
        return _gadget_application(
            leaf.application, decorators, _source_line(decorated.span, source)
        )
    if isinstance(leaf, deqagram.ComposeStatement.ConditionalCorrection):
        return _conditional_correction(leaf.correction)
    raise TypeError(f"unexpected compose statement: {leaf!r}")


def _program_statement(
    decorated: deqagram.DecoratedProgram, source: str | None
) -> model.ProgramStatement:
    decorators = [_decorator(d) for d in decorated.decorators]
    statement = decorated.statement
    if isinstance(statement, deqagram.AttachedProgram.Repeat):
        return model.RepeatBlock(
            count=statement.count,
            body=[_program_statement(s, source) for s in statement.body],
            decorators=decorators,
        )
    leaf = statement.statement
    if isinstance(leaf, deqagram.ProgramStatement.Instruction):
        return _instruction(leaf.instruction, decorators)
    if isinstance(leaf, deqagram.ProgramStatement.InputPort):
        return _input_port(leaf.port, decorators)
    if isinstance(leaf, deqagram.ProgramStatement.OutputPort):
        return _output_port(leaf.port, decorators)
    if isinstance(leaf, deqagram.ProgramStatement.GadgetApplication):
        return _gadget_application(
            leaf.application, decorators, _source_line(decorated.span, source)
        )
    if isinstance(leaf, deqagram.ProgramStatement.Assert):
        return model.AssertStatement(
            target=_target(leaf.assertion.target),
            expected_value=leaf.assertion.expected_value,
            decorators=decorators,
        )
    if isinstance(leaf, deqagram.ProgramStatement.VirtualCorrection):
        return model.VirtualCorrection(
            paulis=_pauli_pairs(leaf.correction.paulis),
            wire=leaf.correction.wire,
        )
    if isinstance(leaf, deqagram.ProgramStatement.ConditionalCorrection):
        return _conditional_correction(leaf.correction)
    raise TypeError(f"unexpected program statement: {leaf!r}")


def _code_definition(
    code: deqagram.CodeDefinition,
    *,
    source_line: int | None = None,
) -> model.CodeDefinition:
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
    if isinstance(definition, deqagram.AttachedDefinition.Code):
        return _code_definition(
            definition.code,
            source_line=_source_line(definition.span, source),
        )
    if isinstance(definition, deqagram.AttachedDefinition.Gadget):
        _warn_dangling(definition.dangling)
        return model.GadgetDefinition(
            name=definition.name,
            body=[_gadget_statement(s, source) for s in definition.body],
            decorators=[_decorator(d) for d in definition.decorators],
            source_line=_source_line(definition.span, source),
        )
    if isinstance(definition, deqagram.AttachedDefinition.Compose):
        _warn_dangling(definition.dangling)
        return model.ComposeDefinition(
            name=definition.name,
            body=[_compose_statement(s, source) for s in definition.body],
            decorators=[_decorator(d) for d in definition.decorators],
            source_line=_source_line(definition.span, source),
        )
    if isinstance(definition, deqagram.AttachedDefinition.Program):
        _warn_dangling(definition.dangling)
        return model.ProgramDefinition(
            name=definition.name,
            body=[_program_statement(s, source) for s in definition.body],
            decorators=[_decorator(d) for d in definition.decorators],
            source_line=_source_line(definition.span, source),
        )
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
