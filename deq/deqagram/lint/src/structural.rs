//! Structural checks: constraints deqagram parses but does not enforce, plus
//! dangling-decorator detection. None of these need the QEC algebra.

use deqagram::Span;
use deqagram::Spanned;
use deqagram::ast::{CodeDefinition, ComposeStatement, GadgetStatement, ProgramStatement};
use deqagram::decorators::{attach_compose_body, attach_gadget_body, attach_program_body};

use crate::{Diagnostic, Rule};

/// Checks the `[[n,k,d]]` parameters of a `CODE` definition.
pub(crate) fn check_code_params(code: &CodeDefinition, span: Span, out: &mut Vec<Diagnostic>) {
    if code.n == 0 {
        out.push(Diagnostic::new(
            Rule::CodeParamN,
            span,
            format!("CODE {}: parameter n must be >= 1", code.name),
        ));
    }
    if code.k > code.n {
        out.push(Diagnostic::new(
            Rule::CodeParamKGreaterThanN,
            span,
            format!("CODE {}: parameter k ({}) must be <= n ({})", code.name, code.k, code.n),
        ));
    }
    if code.d == Some(0) {
        out.push(Diagnostic::new(
            Rule::CodeParamD,
            span,
            format!("CODE {}: parameter d must be >= 1", code.name),
        ));
    }
}

/// Reports a `REPEAT count` of zero. `count` is `u64`, so `== 0` is the only
/// out-of-range value.
fn check_repeat_count(count: u64, span: Span, out: &mut Vec<Diagnostic>) {
    if count == 0 {
        out.push(Diagnostic::new(Rule::RepeatCount, span, "REPEAT count must be >= 1"));
    }
}

/// Reports an `ERROR` probability outside `[0, 1]`.
fn check_probability(probability: f64, span: Span, out: &mut Vec<Diagnostic>) {
    if !(0.0..=1.0).contains(&probability) {
        out.push(Diagnostic::new(
            Rule::ErrorProbability,
            span,
            format!("ERROR probability must be in [0, 1], got {probability}"),
        ));
    }
}

/// Recurses a `GADGET` body for `REPEAT` counts and `ERROR` probabilities.
fn walk_gadget(body: &[Spanned<GadgetStatement>], out: &mut Vec<Diagnostic>) {
    for stmt in body {
        match &stmt.node {
            GadgetStatement::Repeat { count, body } => {
                check_repeat_count(*count, stmt.span, out);
                walk_gadget(body, out);
            }
            GadgetStatement::Error(error) => check_probability(error.probability, stmt.span, out),
            _ => {}
        }
    }
}

/// Recurses a `COMPOSE` body for `REPEAT` counts.
fn walk_compose(body: &[Spanned<ComposeStatement>], out: &mut Vec<Diagnostic>) {
    for stmt in body {
        if let ComposeStatement::Repeat { count, body } = &stmt.node {
            check_repeat_count(*count, stmt.span, out);
            walk_compose(body, out);
        }
    }
}

/// Recurses a `PROGRAM` body for `REPEAT` counts.
fn walk_program(body: &[Spanned<ProgramStatement>], out: &mut Vec<Diagnostic>) {
    for stmt in body {
        if let ProgramStatement::Repeat { count, body } = &stmt.node {
            check_repeat_count(*count, stmt.span, out);
            walk_program(body, out);
        }
    }
}

/// Reports each dangling decorator (a decorator with no statement to attach to)
/// found anywhere in `dangling`.
fn report_dangling(dangling: Vec<Spanned<deqagram::ast::Decorator>>, out: &mut Vec<Diagnostic>) {
    for decorator in dangling {
        out.push(Diagnostic::new(
            Rule::DanglingDecorator,
            decorator.span,
            format!("@{} has no statement to attach to", decorator.node.name),
        ));
    }
}

/// Runs all structural checks on a `GADGET` body.
pub(crate) fn check_gadget_body(body: Vec<Spanned<GadgetStatement>>, out: &mut Vec<Diagnostic>) {
    walk_gadget(&body, out);
    let (_, dangling) = attach_gadget_body(body);
    report_dangling(dangling, out);
}

/// Runs all structural checks on a `COMPOSE` body.
pub(crate) fn check_compose_body(body: Vec<Spanned<ComposeStatement>>, out: &mut Vec<Diagnostic>) {
    walk_compose(&body, out);
    let (_, dangling) = attach_compose_body(body);
    report_dangling(dangling, out);
}

/// Runs all structural checks on a `PROGRAM` body.
pub(crate) fn check_program_body(body: Vec<Spanned<ProgramStatement>>, out: &mut Vec<Diagnostic>) {
    walk_program(&body, out);
    let (_, dangling) = attach_program_body(body);
    report_dangling(dangling, out);
}
