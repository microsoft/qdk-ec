//! Python wrappers for definition AST types and the top-level `DeqFile`.

use ::deqagram::ast;
use pyo3::prelude::*;

use crate::decorators::Decorator;
use crate::statements::{ComposeStatement, GadgetStatement, ProgramStatement};
use crate::targets::PauliProduct;

/// A pair of X and Z logical operators for one logical qubit.
#[pyclass(name = "LogicalOperator", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub struct LogicalOperator {
    pub x_operator: PauliProduct,
    pub z_operator: PauliProduct,
}

impl From<&ast::LogicalOperator> for LogicalOperator {
    fn from(l: &ast::LogicalOperator) -> Self {
        Self {
            x_operator: (&l.x_operator.node).into(),
            z_operator: (&l.z_operator.node).into(),
        }
    }
}

/// A `CODE Name [[n,k,d]] { ... }` definition.
#[pyclass(name = "CodeDefinition", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub struct CodeDefinition {
    pub name: String,
    pub n: u64,
    pub k: u64,
    pub d: Option<u64>,
    pub logicals: Vec<LogicalOperator>,
    pub stabilizers: Vec<PauliProduct>,
    pub decorators: Vec<Decorator>,
}

impl From<&ast::CodeDefinition> for CodeDefinition {
    fn from(c: &ast::CodeDefinition) -> Self {
        Self {
            name: c.name.clone(),
            n: c.n,
            k: c.k,
            d: c.d,
            logicals: c.logicals.iter().map(Into::into).collect(),
            stabilizers: c.stabilizers.iter().map(|s| (&s.node).into()).collect(),
            decorators: c.decorators.iter().map(Into::into).collect(),
        }
    }
}

/// A `GADGET Name { ... }` definition.
#[pyclass(name = "GadgetDefinition", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub struct GadgetDefinition {
    pub name: String,
    pub body: Vec<GadgetStatement>,
    pub decorators: Vec<Decorator>,
}

impl From<&ast::GadgetDefinition> for GadgetDefinition {
    fn from(g: &ast::GadgetDefinition) -> Self {
        Self {
            name: g.name.clone(),
            body: g.body.iter().map(|s| (&s.node).into()).collect(),
            decorators: g.decorators.iter().map(Into::into).collect(),
        }
    }
}

/// A `COMPOSE Name { ... }` definition.
#[pyclass(name = "ComposeDefinition", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub struct ComposeDefinition {
    pub name: String,
    pub body: Vec<ComposeStatement>,
    pub decorators: Vec<Decorator>,
}

impl From<&ast::ComposeDefinition> for ComposeDefinition {
    fn from(c: &ast::ComposeDefinition) -> Self {
        Self {
            name: c.name.clone(),
            body: c.body.iter().map(|s| (&s.node).into()).collect(),
            decorators: c.decorators.iter().map(Into::into).collect(),
        }
    }
}

/// A `PROGRAM Name { ... }` definition.
#[pyclass(name = "ProgramDefinition", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub struct ProgramDefinition {
    pub name: String,
    pub body: Vec<ProgramStatement>,
    pub decorators: Vec<Decorator>,
}

impl From<&ast::ProgramDefinition> for ProgramDefinition {
    fn from(p: &ast::ProgramDefinition) -> Self {
        Self {
            name: p.name.clone(),
            body: p.body.iter().map(|s| (&s.node).into()).collect(),
            decorators: p.decorators.iter().map(Into::into).collect(),
        }
    }
}

/// A top-level definition.
#[pyclass(name = "Definition", eq)]
#[derive(Clone, PartialEq)]
pub enum Definition {
    Code { code: CodeDefinition },
    Gadget { gadget: GadgetDefinition },
    Compose { compose: ComposeDefinition },
    Program { program: ProgramDefinition },
}

impl From<&ast::Definition> for Definition {
    fn from(d: &ast::Definition) -> Self {
        match d {
            ast::Definition::Code(c) => Self::Code { code: c.into() },
            ast::Definition::Gadget(g) => Self::Gadget { gadget: g.into() },
            ast::Definition::Compose(c) => Self::Compose { compose: c.into() },
            ast::Definition::Program(p) => Self::Program { program: p.into() },
        }
    }
}

/// A complete `.deq` file: imports followed by definitions.
#[pyclass(name = "DeqFile", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub struct DeqFile {
    pub imports: Vec<String>,
    pub definitions: Vec<Definition>,
}

impl From<&ast::DeqFile> for DeqFile {
    fn from(f: &ast::DeqFile) -> Self {
        Self {
            imports: f.imports.clone(),
            definitions: f.definitions.iter().map(|d| (&d.node).into()).collect(),
        }
    }
}
