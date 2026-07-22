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

import deqagram

from deq.circuit import model

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
    raise NotImplementedError(
        f"deqagram shim does not yet handle {type(definition).__qualname__}"
    )


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
