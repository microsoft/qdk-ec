//! Decorator-attached views of the per-kind bodies (feature of `parse_attached`).
//!
//! [`parse_attached`](crate::parse_attached) runs deqagram's Rust decorator
//! attachment pass and exposes the result: standalone body-level decorators are
//! folded onto the statement that follows them, and any decorator with no
//! following statement is surfaced separately as *dangling*. This mirrors deq's
//! `model.py`, where every statement carries its own `decorators` list.
//!
//! Only the recursive `REPEAT` spine is re-typed (`Attached*`); leaf statements
//! reuse the same `Py*` statement mirrors as the flat [`parse`](crate::parse).

use ::deqagram::ast;
use ::deqagram::decorators::{Attached, Decorated, attach_compose_body, attach_gadget_body, attach_program_body};
use pyo3::prelude::*;

use crate::decorators::Decorator;
use crate::definitions::CodeDefinition;
use crate::span::Span;
use crate::statements::{ComposeStatement, GadgetStatement, ProgramStatement};

/// Generates the `Decorated*`/`Attached*` wrapper pair for one body kind and the
/// recursive conversion from the Rust `Decorated<S>` tree.
macro_rules! attached_kind {
    (
        $decorated:ident, $attached:ident, $leaf:ty, $ast:ty, $convert:ident
    ) => {
        /// A statement together with the decorators attached to it, and the
        /// source span of the statement.
        #[pyclass(frozen, get_all, eq)]
        #[derive(Clone, PartialEq)]
        pub struct $decorated {
            pub decorators: Vec<Decorator>,
            pub statement: $attached,
            pub span: Span,
        }

        /// A statement in an attached body: a leaf statement, or a `REPEAT`
        /// block whose body is itself attached.
        #[pyclass(eq)]
        #[derive(Clone, PartialEq)]
        pub enum $attached {
            Statement { statement: $leaf },
            Repeat { count: u64, body: Vec<$decorated> },
        }

        fn $convert(decorated: Decorated<$ast>) -> $decorated {
            let span = decorated.statement.span.into();
            let statement = match decorated.statement.node {
                Attached::Statement(leaf) => $attached::Statement {
                    statement: (&leaf).into(),
                },
                Attached::Repeat { count, body } => $attached::Repeat {
                    count,
                    body: body.into_iter().map($convert).collect(),
                },
            };
            $decorated {
                decorators: decorated.decorators.iter().map(Into::into).collect(),
                statement,
                span,
            }
        }
    };
}

attached_kind!(
    DecoratedGadget,
    AttachedGadget,
    GadgetStatement,
    ast::GadgetStatement,
    convert_gadget
);
attached_kind!(
    DecoratedCompose,
    AttachedCompose,
    ComposeStatement,
    ast::ComposeStatement,
    convert_compose
);
attached_kind!(
    DecoratedProgram,
    AttachedProgram,
    ProgramStatement,
    ast::ProgramStatement,
    convert_program
);

/// Converts a list of dangling decorators, dropping their spans.
fn convert_dangling(dangling: &[::deqagram::Spanned<ast::Decorator>]) -> Vec<Decorator> {
    dangling.iter().map(|d| (&d.node).into()).collect()
}

/// A top-level definition with its body's decorators attached.
///
/// `CODE` has no decorator-bearing body, so its variant reuses the flat
/// [`CodeDefinition`](crate::definitions::CodeDefinition) unchanged.
#[pyclass(eq)]
#[derive(Clone, PartialEq)]
pub enum AttachedDefinition {
    Code {
        code: CodeDefinition,
        span: Span,
    },
    Gadget {
        name: String,
        decorators: Vec<Decorator>,
        body: Vec<DecoratedGadget>,
        dangling: Vec<Decorator>,
        span: Span,
    },
    Compose {
        name: String,
        decorators: Vec<Decorator>,
        body: Vec<DecoratedCompose>,
        dangling: Vec<Decorator>,
        span: Span,
    },
    Program {
        name: String,
        decorators: Vec<Decorator>,
        body: Vec<DecoratedProgram>,
        dangling: Vec<Decorator>,
        span: Span,
    },
}

impl From<::deqagram::Spanned<ast::Definition>> for AttachedDefinition {
    fn from(definition: ::deqagram::Spanned<ast::Definition>) -> Self {
        let span = definition.span.into();
        match definition.node {
            ast::Definition::Code(code) => Self::Code {
                code: (&code).into(),
                span,
            },
            ast::Definition::Gadget(g) => {
                let decorators = g.decorators.iter().map(Into::into).collect();
                let (body, dangling) = attach_gadget_body(g.body);
                Self::Gadget {
                    name: g.name,
                    decorators,
                    body: body.into_iter().map(convert_gadget).collect(),
                    dangling: convert_dangling(&dangling),
                    span,
                }
            }
            ast::Definition::Compose(c) => {
                let decorators = c.decorators.iter().map(Into::into).collect();
                let (body, dangling) = attach_compose_body(c.body);
                Self::Compose {
                    name: c.name,
                    decorators,
                    body: body.into_iter().map(convert_compose).collect(),
                    dangling: convert_dangling(&dangling),
                    span,
                }
            }
            ast::Definition::Program(p) => {
                let decorators = p.decorators.iter().map(Into::into).collect();
                let (body, dangling) = attach_program_body(p.body);
                Self::Program {
                    name: p.name,
                    decorators,
                    body: body.into_iter().map(convert_program).collect(),
                    dangling: convert_dangling(&dangling),
                    span,
                }
            }
        }
    }
}

/// A complete `.deq` file with every body's decorators attached.
#[pyclass(frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub struct AttachedDeqFile {
    pub imports: Vec<String>,
    pub definitions: Vec<AttachedDefinition>,
}

impl From<ast::DeqFile> for AttachedDeqFile {
    fn from(file: ast::DeqFile) -> Self {
        Self {
            imports: file.imports,
            definitions: file.definitions.into_iter().map(Into::into).collect(),
        }
    }
}
