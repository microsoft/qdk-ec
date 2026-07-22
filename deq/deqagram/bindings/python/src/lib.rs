//! Python bindings for [`deqagram`], the `.deq` parser.
//!
//! Exposes deqagram's **own** AST shapes to Python — deliberately not deq's
//! `model.py`. A deq-side shim maps these to `model.py`; keeping this layer
//! deq-agnostic lets any Python consumer use the parser.
//!
//! Each AST type has a `Py*` mirror (in the `targets`, `decorators`,
//! `statements`, and `definitions` modules) that converts from it via `From`.
//! Enums map to pyo3 complex enums, with AST tuple variants rewritten as named
//! struct variants for a cleaner Python API. Source spans are not exposed.
//! [`parse`] returns the whole [`DeqFile`](definitions::DeqFile) tree.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod attached;
mod decorators;
mod definitions;
mod span;
mod statements;
mod targets;

/// Parses a `.deq` source into a [`DeqFile`](definitions::DeqFile).
///
/// Raises `ValueError` on a parse error.
#[pyfunction]
fn parse(source: &str) -> PyResult<definitions::DeqFile> {
    let file: ::deqagram::ast::DeqFile = source.parse().map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok((&file).into())
}

/// Parses a `.deq` source, folding standalone body-level decorators onto the
/// statement that follows them (see [`attached`]).
///
/// Raises `ValueError` on a parse error.
#[pyfunction]
fn parse_attached(source: &str) -> PyResult<attached::AttachedDeqFile> {
    let file: ::deqagram::ast::DeqFile = source.parse().map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok(file.into())
}

/// Registers every `Py*` class on the module.
macro_rules! add_classes {
    ($m:expr, $($t:ty),+ $(,)?) => {
        $($m.add_class::<$t>()?;)+
    };
}

/// The `deqagram` extension module (imported as `deqagram.deqagram`).
#[pymodule]
#[allow(clippy::too_many_lines)] // Flat registration of every pyclass + `__all__`.
fn deqagram(m: &Bound<'_, PyModule>) -> PyResult<()> {
    use attached::{
        AttachedCompose, AttachedDefinition, AttachedDeqFile, AttachedGadget, AttachedProgram, DecoratedCompose,
        DecoratedGadget, DecoratedProgram,
    };
    use decorators::{Decorator, DecoratorArg, DecoratorValue};
    use definitions::{
        CodeDefinition, ComposeDefinition, Definition, DeqFile, GadgetDefinition, LogicalOperator, ProgramDefinition,
    };
    use statements::{
        AssertStatement, CheckStatement, ComposeStatement, ConditionalCorrection, ConditionalStatement, ErrorStatement,
        GadgetApplication, GadgetStatement, Instruction, PortDeclaration, PreselectStatement, ProgramStatement,
        PropagateStatement, ReadoutStatement, VirtualCorrection, VirtualLogicalStatement,
    };
    use targets::{
        Condition, ErrorTarget, LogicalPauliTarget, MeasurementRef, Pauli, PauliProduct, PauliTerm, Port, PortKind,
        PropagateTerm, ReadoutTargetItem, Target,
    };

    add_classes!(
        m, // targets
        Pauli,
        PortKind,
        Port,
        LogicalPauliTarget,
        PauliTerm,
        PauliProduct,
        Target,
        MeasurementRef,
        ReadoutTargetItem,
        ErrorTarget,
        PropagateTerm,
        Condition,
        // decorators
        DecoratorValue,
        DecoratorArg,
        Decorator,
        // statements
        Instruction,
        PortDeclaration,
        GadgetApplication,
        ReadoutStatement,
        CheckStatement,
        ErrorStatement,
        ConditionalStatement,
        VirtualLogicalStatement,
        PropagateStatement,
        PreselectStatement,
        AssertStatement,
        VirtualCorrection,
        ConditionalCorrection,
        GadgetStatement,
        ComposeStatement,
        ProgramStatement,
        // definitions
        LogicalOperator,
        CodeDefinition,
        GadgetDefinition,
        ComposeDefinition,
        ProgramDefinition,
        Definition,
        DeqFile,
        // attached (decorator-folded) views
        DecoratedGadget,
        AttachedGadget,
        DecoratedCompose,
        AttachedCompose,
        DecoratedProgram,
        AttachedProgram,
        AttachedDefinition,
        AttachedDeqFile,
        span::Span,
    );

    m.add_function(wrap_pyfunction!(parse, m)?)?;
    m.add_function(wrap_pyfunction!(parse_attached, m)?)?;

    m.add(
        "__all__",
        [
            "Pauli",
            "PortKind",
            "Port",
            "LogicalPauliTarget",
            "PauliTerm",
            "PauliProduct",
            "Target",
            "MeasurementRef",
            "ReadoutTargetItem",
            "ErrorTarget",
            "PropagateTerm",
            "Condition",
            "DecoratorValue",
            "DecoratorArg",
            "Decorator",
            "Instruction",
            "PortDeclaration",
            "GadgetApplication",
            "ReadoutStatement",
            "CheckStatement",
            "ErrorStatement",
            "ConditionalStatement",
            "VirtualLogicalStatement",
            "PropagateStatement",
            "PreselectStatement",
            "AssertStatement",
            "VirtualCorrection",
            "ConditionalCorrection",
            "GadgetStatement",
            "ComposeStatement",
            "ProgramStatement",
            "LogicalOperator",
            "CodeDefinition",
            "GadgetDefinition",
            "ComposeDefinition",
            "ProgramDefinition",
            "Definition",
            "DeqFile",
            "DecoratedGadget",
            "AttachedGadget",
            "DecoratedCompose",
            "AttachedCompose",
            "DecoratedProgram",
            "AttachedProgram",
            "AttachedDefinition",
            "AttachedDeqFile",
            "Span",
            "parse",
            "parse_attached",
        ],
    )?;
    Ok(())
}
