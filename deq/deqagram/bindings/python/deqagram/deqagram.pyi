"""Type stubs for the deqagram PyO3 extension module (`deqagram.deqagram`).

Mirrors deqagram's own `.deq` AST. Enums are pyo3 complex enums: a base class
with nested, constructible variant classes. Structs are read-only and not
constructible from Python. Source spans are not exposed.
"""

from typing import Optional, final

__all__ = [
    'Pauli',
    'PortKind',
    'Port',
    'LogicalPauliTarget',
    'PauliTerm',
    'PauliProduct',
    'Target',
    'MeasurementRef',
    'ReadoutTargetItem',
    'ErrorTarget',
    'PropagateTerm',
    'Condition',
    'DecoratorValue',
    'DecoratorArg',
    'Decorator',
    'Instruction',
    'PortDeclaration',
    'GadgetApplication',
    'ReadoutStatement',
    'CheckStatement',
    'ErrorStatement',
    'ConditionalStatement',
    'VirtualLogicalStatement',
    'PropagateStatement',
    'PreselectStatement',
    'AssertStatement',
    'VirtualCorrection',
    'ConditionalCorrection',
    'GadgetStatement',
    'ComposeStatement',
    'ProgramStatement',
    'LogicalOperator',
    'CodeDefinition',
    'GadgetDefinition',
    'ComposeDefinition',
    'ProgramDefinition',
    'Definition',
    'DeqFile',
    'DecoratedGadget',
    'AttachedGadget',
    'DecoratedCompose',
    'AttachedCompose',
    'DecoratedProgram',
    'AttachedProgram',
    'AttachedDefinition',
    'AttachedDeqFile',
    'Span',
    'parse',
    'parse_attached',
]

@final
class Span:
    start: int
    end: int
    def line_col(self, source: str) -> Optional[tuple[int, int]]: ...
    def __repr__(self) -> str: ...

@final
class Pauli:
    I: Pauli
    X: Pauli
    Y: Pauli
    Z: Pauli

@final
class PortKind:
    In: PortKind
    Out: PortKind

@final
class Port:
    kind: PortKind
    index: int

@final
class LogicalPauliTarget:
    pauli: Pauli
    index: int
    port: Optional[Port]

@final
class PauliTerm:
    pauli: Pauli
    index: int

class PauliProduct:
    @final
    class Identity(PauliProduct):
        __match_args__ = ()
        def __new__(cls) -> PauliProduct.Identity: ...
    @final
    class Terms(PauliProduct):
        terms: list[PauliTerm]
        __match_args__ = ('terms',)
        def __new__(cls, terms: list[PauliTerm]) -> PauliProduct.Terms: ...

class Target:
    @final
    class Qubit(Target):
        inverted: bool
        index: int
        __match_args__ = ('inverted', 'index')
        def __new__(cls, inverted: bool, index: int) -> Target.Qubit: ...
    @final
    class Pauli(Target):
        inverted: bool
        pauli: Pauli
        index: int
        __match_args__ = ('inverted', 'pauli', 'index')
        def __new__(cls, inverted: bool, pauli: Pauli, index: int) -> Target.Pauli: ...
    @final
    class MeasurementRecord(Target):
        offset: int
        __match_args__ = ('offset',)
        def __new__(cls, offset: int) -> Target.MeasurementRecord: ...
    @final
    class PhysicalMeasurement(Target):
        index: int
        __match_args__ = ('index',)
        def __new__(cls, index: int) -> Target.PhysicalMeasurement: ...
    @final
    class InputVirtual(Target):
        port: int
        stabilizer: int
        __match_args__ = ('port', 'stabilizer')
        def __new__(cls, port: int, stabilizer: int) -> Target.InputVirtual: ...
    @final
    class OutputVirtual(Target):
        port: int
        stabilizer: int
        __match_args__ = ('port', 'stabilizer')
        def __new__(cls, port: int, stabilizer: int) -> Target.OutputVirtual: ...
    @final
    class SweepBit(Target):
        index: int
        __match_args__ = ('index',)
        def __new__(cls, index: int) -> Target.SweepBit: ...
    @final
    class Combiner(Target):
        __match_args__ = ()
        def __new__(cls) -> Target.Combiner: ...

class MeasurementRef:
    @final
    class Record(MeasurementRef):
        offset: int
        __match_args__ = ('offset',)
        def __new__(cls, offset: int) -> MeasurementRef.Record: ...
    @final
    class Physical(MeasurementRef):
        index: int
        __match_args__ = ('index',)
        def __new__(cls, index: int) -> MeasurementRef.Physical: ...
    @final
    class InputVirtual(MeasurementRef):
        port: int
        stabilizer: int
        __match_args__ = ('port', 'stabilizer')
        def __new__(cls, port: int, stabilizer: int) -> MeasurementRef.InputVirtual: ...
    @final
    class OutputVirtual(MeasurementRef):
        port: int
        stabilizer: int
        __match_args__ = ('port', 'stabilizer')
        def __new__(cls, port: int, stabilizer: int) -> MeasurementRef.OutputVirtual: ...

class ReadoutTargetItem:
    @final
    class Target(ReadoutTargetItem):
        target: Target
        __match_args__ = ('target',)
        def __new__(cls, target: Target) -> ReadoutTargetItem.Target: ...
    @final
    class Logical(ReadoutTargetItem):
        logical: LogicalPauliTarget
        __match_args__ = ('logical',)
        def __new__(cls, logical: LogicalPauliTarget) -> ReadoutTargetItem.Logical: ...
    @final
    class Destabilizer(ReadoutTargetItem):
        port: int
        stabilizer: int
        __match_args__ = ('port', 'stabilizer')
        def __new__(cls, port: int, stabilizer: int) -> ReadoutTargetItem.Destabilizer: ...

class ErrorTarget:
    @final
    class Check(ErrorTarget):
        index: int
        __match_args__ = ('index',)
        def __new__(cls, index: int) -> ErrorTarget.Check: ...
    @final
    class Readout(ErrorTarget):
        index: int
        __match_args__ = ('index',)
        def __new__(cls, index: int) -> ErrorTarget.Readout: ...
    @final
    class Logical(ErrorTarget):
        logical: LogicalPauliTarget
        __match_args__ = ('logical',)
        def __new__(cls, logical: LogicalPauliTarget) -> ErrorTarget.Logical: ...

class PropagateTerm:
    @final
    class Logical(PropagateTerm):
        logical: LogicalPauliTarget
        __match_args__ = ('logical',)
        def __new__(cls, logical: LogicalPauliTarget) -> PropagateTerm.Logical: ...
    @final
    class Destabilizer(PropagateTerm):
        port: int
        stabilizer: int
        __match_args__ = ('port', 'stabilizer')
        def __new__(cls, port: int, stabilizer: int) -> PropagateTerm.Destabilizer: ...
    @final
    class MeasurementRecord(PropagateTerm):
        offset: int
        __match_args__ = ('offset',)
        def __new__(cls, offset: int) -> PropagateTerm.MeasurementRecord: ...
    @final
    class PhysicalMeasurement(PropagateTerm):
        index: int
        __match_args__ = ('index',)
        def __new__(cls, index: int) -> PropagateTerm.PhysicalMeasurement: ...
    @final
    class Readout(PropagateTerm):
        index: int
        __match_args__ = ('index',)
        def __new__(cls, index: int) -> PropagateTerm.Readout: ...

class Condition:
    @final
    class Readout(Condition):
        index: int
        __match_args__ = ('index',)
        def __new__(cls, index: int) -> Condition.Readout: ...
    @final
    class Measurement(Condition):
        measurement: MeasurementRef
        __match_args__ = ('measurement',)
        def __new__(cls, measurement: MeasurementRef) -> Condition.Measurement: ...

class DecoratorValue:
    @final
    class String(DecoratorValue):
        value: str
        __match_args__ = ('value',)
        def __new__(cls, value: str) -> DecoratorValue.String: ...
    @final
    class Int(DecoratorValue):
        value: int
        __match_args__ = ('value',)
        def __new__(cls, value: int) -> DecoratorValue.Int: ...
    @final
    class Float(DecoratorValue):
        value: float
        __match_args__ = ('value',)
        def __new__(cls, value: float) -> DecoratorValue.Float: ...

class DecoratorArg:
    @final
    class Value(DecoratorArg):
        value: DecoratorValue
        __match_args__ = ('value',)
        def __new__(cls, value: DecoratorValue) -> DecoratorArg.Value: ...
    @final
    class Keyword(DecoratorArg):
        key: str
        value: DecoratorValue
        __match_args__ = ('key', 'value')
        def __new__(cls, key: str, value: DecoratorValue) -> DecoratorArg.Keyword: ...

@final
class Decorator:
    name: str
    arguments: list[DecoratorArg]

@final
class Instruction:
    name: str
    tag: Optional[str]
    arguments: list[float]
    targets: list[Target]

@final
class PortDeclaration:
    code_name: str
    qubit_indices: list[int]

@final
class GadgetApplication:
    gadget_name: str
    in_indices: Optional[list[int]]
    out_indices: Optional[list[int]]

@final
class ReadoutStatement:
    targets: list[ReadoutTargetItem]
    flip: bool

@final
class CheckStatement:
    targets: list[Target]
    flip: bool

@final
class ErrorStatement:
    probability: float
    targets: list[ErrorTarget]

@final
class ConditionalStatement:
    condition: Condition
    targets: list[LogicalPauliTarget]

@final
class VirtualLogicalStatement:
    targets: list[LogicalPauliTarget]

@final
class PropagateStatement:
    target: LogicalPauliTarget
    terms: list[PropagateTerm]
    flip: bool

@final
class PreselectStatement:
    conditions: list[MeasurementRef]
    expected_value: int

@final
class AssertStatement:
    target: Target
    expected_value: int

@final
class VirtualCorrection:
    paulis: list[PauliTerm]
    wire: int

@final
class ConditionalCorrection:
    readout_offset: int
    paulis: list[PauliTerm]
    wire: int

class GadgetStatement:
    @final
    class Instruction(GadgetStatement):
        instruction: Instruction
        __match_args__ = ('instruction',)
        def __new__(cls, instruction: Instruction) -> GadgetStatement.Instruction: ...
    @final
    class Repeat(GadgetStatement):
        count: int
        body: list[GadgetStatement]
        __match_args__ = ('count', 'body')
        def __new__(cls, count: int, body: list[GadgetStatement]) -> GadgetStatement.Repeat: ...
    @final
    class InputPort(GadgetStatement):
        port: PortDeclaration
        __match_args__ = ('port',)
        def __new__(cls, port: PortDeclaration) -> GadgetStatement.InputPort: ...
    @final
    class OutputPort(GadgetStatement):
        port: PortDeclaration
        __match_args__ = ('port',)
        def __new__(cls, port: PortDeclaration) -> GadgetStatement.OutputPort: ...
    @final
    class Readout(GadgetStatement):
        readout: ReadoutStatement
        __match_args__ = ('readout',)
        def __new__(cls, readout: ReadoutStatement) -> GadgetStatement.Readout: ...
    @final
    class Check(GadgetStatement):
        check: CheckStatement
        __match_args__ = ('check',)
        def __new__(cls, check: CheckStatement) -> GadgetStatement.Check: ...
    @final
    class Error(GadgetStatement):
        error: ErrorStatement
        __match_args__ = ('error',)
        def __new__(cls, error: ErrorStatement) -> GadgetStatement.Error: ...
    @final
    class Conditional(GadgetStatement):
        conditional: ConditionalStatement
        __match_args__ = ('conditional',)
        def __new__(cls, conditional: ConditionalStatement) -> GadgetStatement.Conditional: ...
    @final
    class VirtualLogical(GadgetStatement):
        statement: VirtualLogicalStatement
        __match_args__ = ('statement',)
        def __new__(cls, statement: VirtualLogicalStatement) -> GadgetStatement.VirtualLogical: ...
    @final
    class Propagate(GadgetStatement):
        propagate: PropagateStatement
        __match_args__ = ('propagate',)
        def __new__(cls, propagate: PropagateStatement) -> GadgetStatement.Propagate: ...
    @final
    class Preselect(GadgetStatement):
        preselect: PreselectStatement
        __match_args__ = ('preselect',)
        def __new__(cls, preselect: PreselectStatement) -> GadgetStatement.Preselect: ...
    @final
    class Decorator(GadgetStatement):
        decorator: Decorator
        __match_args__ = ('decorator',)
        def __new__(cls, decorator: Decorator) -> GadgetStatement.Decorator: ...

class ComposeStatement:
    @final
    class Instruction(ComposeStatement):
        instruction: Instruction
        __match_args__ = ('instruction',)
        def __new__(cls, instruction: Instruction) -> ComposeStatement.Instruction: ...
    @final
    class Repeat(ComposeStatement):
        count: int
        body: list[ComposeStatement]
        __match_args__ = ('count', 'body')
        def __new__(cls, count: int, body: list[ComposeStatement]) -> ComposeStatement.Repeat: ...
    @final
    class InputPort(ComposeStatement):
        port: PortDeclaration
        __match_args__ = ('port',)
        def __new__(cls, port: PortDeclaration) -> ComposeStatement.InputPort: ...
    @final
    class OutputPort(ComposeStatement):
        port: PortDeclaration
        __match_args__ = ('port',)
        def __new__(cls, port: PortDeclaration) -> ComposeStatement.OutputPort: ...
    @final
    class GadgetApplication(ComposeStatement):
        application: GadgetApplication
        __match_args__ = ('application',)
        def __new__(cls, application: GadgetApplication) -> ComposeStatement.GadgetApplication: ...
    @final
    class ConditionalCorrection(ComposeStatement):
        correction: ConditionalCorrection
        __match_args__ = ('correction',)
        def __new__(cls, correction: ConditionalCorrection) -> ComposeStatement.ConditionalCorrection: ...
    @final
    class Decorator(ComposeStatement):
        decorator: Decorator
        __match_args__ = ('decorator',)
        def __new__(cls, decorator: Decorator) -> ComposeStatement.Decorator: ...

class ProgramStatement:
    @final
    class Instruction(ProgramStatement):
        instruction: Instruction
        __match_args__ = ('instruction',)
        def __new__(cls, instruction: Instruction) -> ProgramStatement.Instruction: ...
    @final
    class Repeat(ProgramStatement):
        count: int
        body: list[ProgramStatement]
        __match_args__ = ('count', 'body')
        def __new__(cls, count: int, body: list[ProgramStatement]) -> ProgramStatement.Repeat: ...
    @final
    class InputPort(ProgramStatement):
        port: PortDeclaration
        __match_args__ = ('port',)
        def __new__(cls, port: PortDeclaration) -> ProgramStatement.InputPort: ...
    @final
    class OutputPort(ProgramStatement):
        port: PortDeclaration
        __match_args__ = ('port',)
        def __new__(cls, port: PortDeclaration) -> ProgramStatement.OutputPort: ...
    @final
    class GadgetApplication(ProgramStatement):
        application: GadgetApplication
        __match_args__ = ('application',)
        def __new__(cls, application: GadgetApplication) -> ProgramStatement.GadgetApplication: ...
    @final
    class Assert(ProgramStatement):
        assertion: AssertStatement
        __match_args__ = ('assertion',)
        def __new__(cls, assertion: AssertStatement) -> ProgramStatement.Assert: ...
    @final
    class VirtualCorrection(ProgramStatement):
        correction: VirtualCorrection
        __match_args__ = ('correction',)
        def __new__(cls, correction: VirtualCorrection) -> ProgramStatement.VirtualCorrection: ...
    @final
    class ConditionalCorrection(ProgramStatement):
        correction: ConditionalCorrection
        __match_args__ = ('correction',)
        def __new__(cls, correction: ConditionalCorrection) -> ProgramStatement.ConditionalCorrection: ...
    @final
    class Decorator(ProgramStatement):
        decorator: Decorator
        __match_args__ = ('decorator',)
        def __new__(cls, decorator: Decorator) -> ProgramStatement.Decorator: ...

@final
class LogicalOperator:
    x_operator: PauliProduct
    z_operator: PauliProduct

@final
class CodeDefinition:
    name: str
    n: int
    k: int
    d: Optional[int]
    logicals: list[LogicalOperator]
    stabilizers: list[PauliProduct]
    decorators: list[Decorator]

@final
class GadgetDefinition:
    name: str
    body: list[GadgetStatement]
    decorators: list[Decorator]

@final
class ComposeDefinition:
    name: str
    body: list[ComposeStatement]
    decorators: list[Decorator]

@final
class ProgramDefinition:
    name: str
    body: list[ProgramStatement]
    decorators: list[Decorator]

class Definition:
    @final
    class Code(Definition):
        code: CodeDefinition
        __match_args__ = ('code',)
        def __new__(cls, code: CodeDefinition) -> Definition.Code: ...
    @final
    class Gadget(Definition):
        gadget: GadgetDefinition
        __match_args__ = ('gadget',)
        def __new__(cls, gadget: GadgetDefinition) -> Definition.Gadget: ...
    @final
    class Compose(Definition):
        compose: ComposeDefinition
        __match_args__ = ('compose',)
        def __new__(cls, compose: ComposeDefinition) -> Definition.Compose: ...
    @final
    class Program(Definition):
        program: ProgramDefinition
        __match_args__ = ('program',)
        def __new__(cls, program: ProgramDefinition) -> Definition.Program: ...

@final
class DeqFile:
    imports: list[str]
    definitions: list[Definition]

# ── Attached (decorator-folded) views ────────────────────────────────

@final
class DecoratedGadget:
    decorators: list[Decorator]
    statement: AttachedGadget
    span: Span

class AttachedGadget:
    @final
    class Statement(AttachedGadget):
        statement: GadgetStatement
        __match_args__ = ('statement',)
        def __new__(cls, statement: GadgetStatement) -> AttachedGadget.Statement: ...
    @final
    class Repeat(AttachedGadget):
        count: int
        body: list[DecoratedGadget]
        __match_args__ = ('count', 'body')
        def __new__(cls, count: int, body: list[DecoratedGadget]) -> AttachedGadget.Repeat: ...

@final
class DecoratedCompose:
    decorators: list[Decorator]
    statement: AttachedCompose
    span: Span

class AttachedCompose:
    @final
    class Statement(AttachedCompose):
        statement: ComposeStatement
        __match_args__ = ('statement',)
        def __new__(cls, statement: ComposeStatement) -> AttachedCompose.Statement: ...
    @final
    class Repeat(AttachedCompose):
        count: int
        body: list[DecoratedCompose]
        __match_args__ = ('count', 'body')
        def __new__(cls, count: int, body: list[DecoratedCompose]) -> AttachedCompose.Repeat: ...

@final
class DecoratedProgram:
    decorators: list[Decorator]
    statement: AttachedProgram
    span: Span

class AttachedProgram:
    @final
    class Statement(AttachedProgram):
        statement: ProgramStatement
        __match_args__ = ('statement',)
        def __new__(cls, statement: ProgramStatement) -> AttachedProgram.Statement: ...
    @final
    class Repeat(AttachedProgram):
        count: int
        body: list[DecoratedProgram]
        __match_args__ = ('count', 'body')
        def __new__(cls, count: int, body: list[DecoratedProgram]) -> AttachedProgram.Repeat: ...

class AttachedDefinition:
    @final
    class Code(AttachedDefinition):
        code: CodeDefinition
        span: Span
        __match_args__ = ('code', 'span')
        def __new__(cls, code: CodeDefinition, span: Span) -> AttachedDefinition.Code: ...
    @final
    class Gadget(AttachedDefinition):
        name: str
        decorators: list[Decorator]
        body: list[DecoratedGadget]
        dangling: list[Decorator]
        span: Span
        __match_args__ = ('name', 'decorators', 'body', 'dangling', 'span')
        def __new__(
            cls,
            name: str,
            decorators: list[Decorator],
            body: list[DecoratedGadget],
            dangling: list[Decorator],
            span: Span,
        ) -> AttachedDefinition.Gadget: ...
    @final
    class Compose(AttachedDefinition):
        name: str
        decorators: list[Decorator]
        body: list[DecoratedCompose]
        dangling: list[Decorator]
        span: Span
        __match_args__ = ('name', 'decorators', 'body', 'dangling', 'span')
        def __new__(
            cls,
            name: str,
            decorators: list[Decorator],
            body: list[DecoratedCompose],
            dangling: list[Decorator],
            span: Span,
        ) -> AttachedDefinition.Compose: ...
    @final
    class Program(AttachedDefinition):
        name: str
        decorators: list[Decorator]
        body: list[DecoratedProgram]
        dangling: list[Decorator]
        span: Span
        __match_args__ = ('name', 'decorators', 'body', 'dangling', 'span')
        def __new__(
            cls,
            name: str,
            decorators: list[Decorator],
            body: list[DecoratedProgram],
            dangling: list[Decorator],
            span: Span,
        ) -> AttachedDefinition.Program: ...

@final
class AttachedDeqFile:
    imports: list[str]
    definitions: list[AttachedDefinition]


def parse(source: str) -> DeqFile:
    """Parse a ``.deq`` source into a :class:`DeqFile`. Raises ``ValueError``."""
    ...

def parse_attached(source: str) -> AttachedDeqFile:
    """Parse a ``.deq`` source, folding body-level decorators onto the following
    statement. Raises ``ValueError``."""
    ...

