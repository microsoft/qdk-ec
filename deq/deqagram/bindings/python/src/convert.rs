//! Builds deqagram's Python dataclasses (`deqagram._nodes`) from its Rust AST.
//!
//! Body decorators are folded onto the statement that follows them (see
//! [`deqagram::decorators`]). Decorators the model has nowhere to store, either
//! dangling at the end of a block or placed on a `CONDITIONAL`/`VIRTUAL`
//! correction, raise a `UserWarning` and are ignored.

use std::ffi::CString;

use ::deqagram::ast::{self, ComposeStatement, GadgetStatement, Pauli, ProgramStatement};
use ::deqagram::decorators::{Attached, Decorated, attach_compose_body, attach_gadget_body, attach_program_body};
use ::deqagram::{Span, Spanned};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyUserWarning;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

type Model<'py> = Bound<'py, PyModule>;
type Object<'py> = Bound<'py, PyAny>;

/// Calls `deqagram._nodes.<class>(field=value, ...)`.
macro_rules! new {
    ($model:expr, $class:expr, { $($field:ident: $value:expr),* $(,)? }) => {{
        let kwargs = PyDict::new($model.py());
        $(kwargs.set_item(stringify!($field), $value)?;)*
        $model.getattr($class)?.call((), Some(&kwargs))
    }};
}

/// Converts `file`, parsed from `source`, into a `deqagram.DeqFile`.
pub fn deq_file<'py>(py: Python<'py>, file: ast::DeqFile, source: &str) -> PyResult<Object<'py>> {
    let model = PyModule::import(py, "deqagram._nodes")?;
    let lines = Lines::new(source);
    let imports = each(&file.imports, |path| new!(model, "ImportStatement", { path: path }))?;
    let definitions = file
        .definitions
        .into_iter()
        .map(|spanned| definition(&model, &lines, spanned))
        .collect::<PyResult<Vec<_>>>()?;
    new!(model, "DeqFile", { definitions: definitions, imports: imports })
}

/// Byte offsets of each line's start, so a span's 1-based line is a binary search.
struct Lines(Vec<usize>);

impl Lines {
    fn new(source: &str) -> Self {
        Self(
            std::iter::once(0)
                .chain(source.match_indices('\n').map(|(index, _)| index + 1))
                .collect(),
        )
    }

    fn line(&self, span: Span) -> usize {
        self.0.partition_point(|&start| start <= span.start)
    }
}

fn each<'py, T>(items: &[T], convert: impl FnMut(&T) -> PyResult<Object<'py>>) -> PyResult<Vec<Object<'py>>> {
    items.iter().map(convert).collect()
}

fn warn_ignored<'a>(
    py: Python<'_>,
    decorators: impl IntoIterator<Item = &'a ast::Decorator>,
    reason: &str,
) -> PyResult<()> {
    let names: Vec<String> = decorators.into_iter().map(|d| format!("@{}", d.name)).collect();
    if names.is_empty() {
        return Ok(());
    }
    let message = CString::new(format!("Decorators {} {reason} and will be ignored", names.join(", ")))?;
    PyErr::warn(py, &py.get_type::<PyUserWarning>(), &message, 1)
}

// ── Definitions ──────────────────────────────────────────────────────

fn definition<'py>(model: &Model<'py>, lines: &Lines, definition: Spanned<ast::Definition>) -> PyResult<Object<'py>> {
    let line = lines.line(definition.span);
    match definition.node {
        ast::Definition::Code(code) => new!(model, "CodeDefinition", {
            name: code.name,
            n: code.n,
            k: code.k,
            d: code.d,
            logicals: each(&code.logicals, |logical| new!(model, "LogicalOperator", {
                x_operator: pauli_product(model, &logical.x_operator)?,
                z_operator: pauli_product(model, &logical.z_operator)?,
            }))?,
            stabilizers: each(&code.stabilizers, |product| pauli_product(model, product))?,
            decorators: decorators(model, &code.decorators)?,
            source_line: line,
        }),
        ast::Definition::Gadget(gadget) => {
            let (body, dangling) = attach_gadget_body(gadget.body);
            warn_ignored(
                model.py(),
                dangling.iter().map(|d| &d.node),
                "at end of block have no target",
            )?;
            new!(model, "GadgetDefinition", {
                name: gadget.name,
                body: statements(model, lines, &body, gadget_statement)?,
                decorators: decorators(model, &gadget.decorators)?,
                source_line: line,
            })
        }
        ast::Definition::Compose(compose) => {
            let (body, dangling) = attach_compose_body(compose.body);
            warn_ignored(
                model.py(),
                dangling.iter().map(|d| &d.node),
                "at end of block have no target",
            )?;
            new!(model, "ComposeDefinition", {
                name: compose.name,
                body: statements(model, lines, &body, compose_statement)?,
                decorators: decorators(model, &compose.decorators)?,
                source_line: line,
            })
        }
        ast::Definition::Program(program) => {
            let (body, dangling) = attach_program_body(program.body);
            warn_ignored(
                model.py(),
                dangling.iter().map(|d| &d.node),
                "at end of block have no target",
            )?;
            new!(model, "ProgramDefinition", {
                name: program.name,
                body: statements(model, lines, &body, program_statement)?,
                decorators: decorators(model, &program.decorators)?,
                source_line: line,
            })
        }
    }
}

// ── Statements ───────────────────────────────────────────────────────

/// Converts one leaf statement, given its attached decorators and 1-based line.
type Leaf<S> = for<'py> fn(&Model<'py>, &S, &[ast::Decorator], usize) -> PyResult<Object<'py>>;

fn statements<'py, S>(
    model: &Model<'py>,
    lines: &Lines,
    body: &[Decorated<S>],
    leaf: Leaf<S>,
) -> PyResult<Vec<Object<'py>>> {
    each(body, |decorated| {
        let line = lines.line(decorated.statement.span);
        match &decorated.statement.node {
            Attached::Repeat { count, body } => new!(model, "RepeatBlock", {
                count: *count,
                body: statements(model, lines, body, leaf)?,
                decorators: decorators(model, &decorated.decorators)?,
                source_line: line,
            }),
            Attached::Statement(statement) => leaf(model, statement, &decorated.decorators, line),
        }
    })
}

fn gadget_statement<'py>(
    model: &Model<'py>,
    statement: &GadgetStatement,
    attached: &[ast::Decorator],
    line: usize,
) -> PyResult<Object<'py>> {
    match statement {
        GadgetStatement::Instruction(instruction) => self::instruction(model, instruction, attached, line),
        GadgetStatement::InputPort(port) => self::port(model, "InputPort", port, attached, line),
        GadgetStatement::OutputPort(port) => self::port(model, "OutputPort", port, attached, line),
        GadgetStatement::Readout(readout) => new!(model, "ReadoutStatement", {
            targets: each(&readout.targets, |item| readout_item(model, item))?,
            flip: readout.flip,
            decorators: decorators(model, attached)?,
        }),
        GadgetStatement::Check(check) => new!(model, "CheckStatement", {
            targets: each(&check.targets, |target| self::target(model, target))?,
            flip: check.flip,
            decorators: decorators(model, attached)?,
        }),
        GadgetStatement::Error(error) => new!(model, "ErrorStatement", {
            probability: error.probability,
            targets: each(&error.targets, |target| error_target(model, target))?,
            decorators: decorators(model, attached)?,
        }),
        GadgetStatement::Conditional(conditional) => new!(model, "ConditionalStatement", {
            condition: condition(model, conditional.condition)?,
            targets: each(&conditional.targets, |target| logical(model, target))?,
            decorators: decorators(model, attached)?,
            source_line: line,
        }),
        GadgetStatement::VirtualLogical(virtual_logical) => new!(model, "VirtualLogicalStatement", {
            targets: each(&virtual_logical.targets, |target| logical(model, target))?,
            decorators: decorators(model, attached)?,
        }),
        GadgetStatement::Propagate(propagate) => new!(model, "PropagateStatement", {
            target: logical(model, &propagate.target)?,
            terms: each(&propagate.terms, |term| propagate_term(model, term))?,
            flip: propagate.flip,
            decorators: decorators(model, attached)?,
            source_line: line,
        }),
        GadgetStatement::Preselect(preselect) => new!(model, "PreselectStatement", {
            conditions: each(&preselect.conditions, |reference| measurement_ref(model, *reference))?,
            expected_value: preselect.expected_value,
            decorators: decorators(model, attached)?,
            source_line: line,
        }),
        GadgetStatement::Loss(loss) => new!(model, "LossStatement", {
            probability: loss.probability,
            input_port: loss.input_port,
            input_qubit: loss.input_qubit,
            source_errors: loss.source_errors.as_slice(),
            continuation_errors: loss.continuation_errors.as_slice(),
            child_losses: loss.child_losses.as_slice(),
            output_qubits: loss.output_qubits.as_slice(),
            measurement_indices: loss.measurement_indices.as_slice(),
            decorators: decorators(model, attached)?,
        }),
        GadgetStatement::Repeat { .. } | GadgetStatement::Decorator(_) => {
            unreachable!("decorator attachment folds REPEAT blocks and decorators")
        }
    }
}

fn compose_statement<'py>(
    model: &Model<'py>,
    statement: &ComposeStatement,
    attached: &[ast::Decorator],
    line: usize,
) -> PyResult<Object<'py>> {
    match statement {
        ComposeStatement::Instruction(instruction) => self::instruction(model, instruction, attached, line),
        ComposeStatement::InputPort(port) => self::port(model, "InputPort", port, attached, line),
        ComposeStatement::OutputPort(port) => self::port(model, "OutputPort", port, attached, line),
        ComposeStatement::GadgetApplication(application) => gadget_application(model, application, attached, line),
        ComposeStatement::ConditionalCorrection(correction) => {
            warn_ignored(model.py(), attached, "on a CONDITIONAL correction are not supported")?;
            conditional_correction(model, correction)
        }
        ComposeStatement::Repeat { .. } | ComposeStatement::Decorator(_) => {
            unreachable!("decorator attachment folds REPEAT blocks and decorators")
        }
    }
}

fn program_statement<'py>(
    model: &Model<'py>,
    statement: &ProgramStatement,
    attached: &[ast::Decorator],
    line: usize,
) -> PyResult<Object<'py>> {
    match statement {
        ProgramStatement::Instruction(instruction) => self::instruction(model, instruction, attached, line),
        ProgramStatement::InputPort(port) => self::port(model, "InputPort", port, attached, line),
        ProgramStatement::OutputPort(port) => self::port(model, "OutputPort", port, attached, line),
        ProgramStatement::GadgetApplication(application) => gadget_application(model, application, attached, line),
        ProgramStatement::Assert(assertion) => new!(model, "AssertStatement", {
            target: target(model, &assertion.target)?,
            expected_value: assertion.expected_value,
            decorators: decorators(model, attached)?,
        }),
        ProgramStatement::VirtualCorrection(correction) => {
            warn_ignored(model.py(), attached, "on a VIRTUAL correction are not supported")?;
            new!(model, "VirtualCorrection", {
                paulis: pauli_pairs(&correction.paulis),
                wire: correction.wire,
            })
        }
        ProgramStatement::ConditionalCorrection(correction) => {
            warn_ignored(model.py(), attached, "on a CONDITIONAL correction are not supported")?;
            conditional_correction(model, correction)
        }
        ProgramStatement::Repeat { .. } | ProgramStatement::Decorator(_) => {
            unreachable!("decorator attachment folds REPEAT blocks and decorators")
        }
    }
}

fn instruction<'py>(
    model: &Model<'py>,
    instruction: &ast::Instruction,
    attached: &[ast::Decorator],
    line: usize,
) -> PyResult<Object<'py>> {
    new!(model, "Instruction", {
        name: &instruction.name,
        tag: instruction.tag.as_deref(),
        arguments: instruction.arguments.as_slice(),
        targets: each(&instruction.targets, |target| self::target(model, target))?,
        decorators: decorators(model, attached)?,
        source_line: line,
    })
}

fn port<'py>(
    model: &Model<'py>,
    class: &str,
    port: &ast::PortDeclaration,
    attached: &[ast::Decorator],
    line: usize,
) -> PyResult<Object<'py>> {
    new!(model, class, {
        code_name: port.code_name.as_str(),
        qubit_indices: port.qubit_indices.as_slice(),
        decorators: decorators(model, attached)?,
        source_line: line,
    })
}

fn gadget_application<'py>(
    model: &Model<'py>,
    application: &ast::GadgetApplication,
    attached: &[ast::Decorator],
    line: usize,
) -> PyResult<Object<'py>> {
    new!(model, "GadgetApplication", {
        gadget_name: application.gadget_name.as_str(),
        in_indices: application.in_indices.as_deref(),
        out_indices: application.out_indices.as_deref(),
        decorators: decorators(model, attached)?,
        source_line: line,
    })
}

fn conditional_correction<'py>(model: &Model<'py>, correction: &ast::ConditionalCorrection) -> PyResult<Object<'py>> {
    new!(model, "ConditionalCorrection", {
        readout_offset: correction.readout_offset,
        paulis: pauli_pairs(&correction.paulis),
        wire: correction.wire,
    })
}

// ── Decorators ───────────────────────────────────────────────────────

fn decorators<'py>(model: &Model<'py>, decorators: &[ast::Decorator]) -> PyResult<Vec<Object<'py>>> {
    each(decorators, |decorator| {
        let arguments = each(&decorator.arguments, |argument| match argument {
            ast::DecoratorArg::Value(value) => decorator_value(model.py(), value),
            ast::DecoratorArg::Keyword { key, value } => new!(model, "KeywordArg", {
                key: key,
                value: decorator_value(model.py(), value)?,
            }),
        })?;
        new!(model, "Decorator", {
            name: &decorator.name,
            arguments: PyTuple::new(model.py(), arguments)?,
        })
    })
}

fn decorator_value<'py>(py: Python<'py>, value: &ast::DecoratorValue) -> PyResult<Object<'py>> {
    match value {
        ast::DecoratorValue::String(string) => string.into_bound_py_any(py),
        ast::DecoratorValue::Int(integer) => integer.into_bound_py_any(py),
        ast::DecoratorValue::Float(float) => float.into_bound_py_any(py),
    }
}

// ── Targets ──────────────────────────────────────────────────────────

fn letter(pauli: Pauli) -> String {
    pauli.to_string()
}

fn pauli_pairs(paulis: &[(Pauli, u64)]) -> Vec<(String, u64)> {
    paulis.iter().map(|&(pauli, index)| (letter(pauli), index)).collect()
}

/// The identity product `_` has no factors, so it maps to an empty term tuple.
fn pauli_product<'py>(model: &Model<'py>, product: &ast::PauliProduct) -> PyResult<Object<'py>> {
    let terms = match product {
        ast::PauliProduct::Identity => Vec::new(),
        ast::PauliProduct::Terms(terms) => each(terms, |term| {
            new!(model, "PauliTerm", {
                pauli: letter(term.pauli),
                index: term.index,
            })
        })?,
    };
    new!(model, "PauliProduct", { terms: PyTuple::new(model.py(), terms)? })
}

fn target<'py>(model: &Model<'py>, target: &ast::Target) -> PyResult<Object<'py>> {
    match *target {
        ast::Target::Qubit { inverted, index } => new!(model, "QubitTarget", { index: index, inverted: inverted }),
        ast::Target::Loss { index } => new!(model, "LossTarget", { index: index }),
        ast::Target::Pauli { inverted, pauli, index } => new!(model, "PauliTarget", {
            pauli: letter(pauli),
            index: index,
            inverted: inverted,
        }),
        ast::Target::MeasurementRecord { offset } => new!(model, "MeasurementRecordTarget", { offset: offset }),
        ast::Target::PhysicalMeasurement { index } => new!(model, "PhysicalMeasurementTarget", { index: index }),
        ast::Target::InputVirtual { port, stabilizer } => new!(model, "InputVirtualTarget", {
            port_index: port,
            stabilizer_index: stabilizer,
        }),
        ast::Target::OutputVirtual { port, stabilizer } => new!(model, "OutputVirtualTarget", {
            port_index: port,
            stabilizer_index: stabilizer,
        }),
        ast::Target::SweepBit { index } => new!(model, "SweepBitTarget", { index: index }),
        ast::Target::Combiner => new!(model, "CombinerTarget", {}),
    }
}

fn logical<'py>(model: &Model<'py>, target: &ast::LogicalPauliTarget) -> PyResult<Object<'py>> {
    new!(model, "LogicalPauliTarget", {
        pauli: letter(target.pauli),
        index: target.index,
        port_kind: target.port.map(|port| port.kind.to_string()),
        port_index: target.port.map(|port| port.index),
    })
}

fn measurement_ref<'py>(model: &Model<'py>, reference: ast::MeasurementRef) -> PyResult<Object<'py>> {
    match reference {
        ast::MeasurementRef::Record { offset } => new!(model, "MeasurementRecordTarget", { offset: offset }),
        ast::MeasurementRef::Physical { index } => new!(model, "PhysicalMeasurementTarget", { index: index }),
        ast::MeasurementRef::InputVirtual { port, stabilizer } => new!(model, "InputVirtualTarget", {
            port_index: port,
            stabilizer_index: stabilizer,
        }),
        ast::MeasurementRef::OutputVirtual { port, stabilizer } => new!(model, "OutputVirtualTarget", {
            port_index: port,
            stabilizer_index: stabilizer,
        }),
    }
}

fn destabilizer<'py>(model: &Model<'py>, port: u64, stabilizer: u64) -> PyResult<Object<'py>> {
    new!(model, "DestabilizerTarget", { port_index: port, stab_index: stabilizer })
}

fn readout_item<'py>(model: &Model<'py>, item: &ast::ReadoutTargetItem) -> PyResult<Object<'py>> {
    match item {
        ast::ReadoutTargetItem::Target(target) => self::target(model, target),
        ast::ReadoutTargetItem::Logical(target) => logical(model, target),
        ast::ReadoutTargetItem::Destabilizer { port, stabilizer } => destabilizer(model, *port, *stabilizer),
    }
}

fn error_target<'py>(model: &Model<'py>, target: &ast::ErrorTarget) -> PyResult<Object<'py>> {
    match target {
        ast::ErrorTarget::Check(index) => new!(model, "CheckTarget", { index: *index }),
        ast::ErrorTarget::Readout(index) => new!(model, "ReadoutTarget", { index: *index }),
        ast::ErrorTarget::Logical(target) => logical(model, target),
    }
}

fn propagate_term<'py>(model: &Model<'py>, term: &ast::PropagateTerm) -> PyResult<Object<'py>> {
    match *term {
        ast::PropagateTerm::Logical(target) => logical(model, &target),
        ast::PropagateTerm::Destabilizer { port, stabilizer } => destabilizer(model, port, stabilizer),
        ast::PropagateTerm::MeasurementRecord { offset } => new!(model, "MeasurementRecordTarget", { offset: offset }),
        ast::PropagateTerm::PhysicalMeasurement { index } => new!(model, "PhysicalMeasurementTarget", { index: index }),
        ast::PropagateTerm::Readout { index } => new!(model, "ReadoutTarget", { index: index }),
    }
}

fn condition<'py>(model: &Model<'py>, condition: ast::Condition) -> PyResult<Object<'py>> {
    match condition {
        ast::Condition::Readout(index) => new!(model, "ReadoutTarget", { index: index }),
        ast::Condition::Measurement(reference) => measurement_ref(model, reference),
    }
}
