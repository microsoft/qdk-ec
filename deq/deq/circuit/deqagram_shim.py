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


def _code_definition(code: deqagram.CodeDefinition) -> model.CodeDefinition:
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
    )


def _definition(definition: object) -> model.Definition:
    if isinstance(definition, deqagram.AttachedDefinition.Code):
        return _code_definition(definition.code)
    raise NotImplementedError(
        f"deqagram shim does not yet handle {type(definition).__qualname__}"
    )


def to_model(file: deqagram.AttachedDeqFile) -> model.DeqFile:
    """Convert a deqagram ``AttachedDeqFile`` to a ``model.DeqFile``."""
    return model.DeqFile(
        definitions=[_definition(d) for d in file.definitions],
        imports=[model.ImportStatement(path=path) for path in file.imports],
    )


def parse(text: str) -> model.DeqFile:
    """Parse ``.deq`` ``text`` via deqagram and return a ``model.DeqFile``."""
    return to_model(deqagram.parse_attached(text))
