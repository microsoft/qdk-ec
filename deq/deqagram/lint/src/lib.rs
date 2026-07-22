//! A semantic linter for `.deq` files.
//!
//! [`deqagram`] parses `.deq` source and rejects malformed *syntax*; this crate
//! validates *meaning* on the parsed [`DeqFile`]. It reports two families of
//! problems:
//!
//! - **Structural** constraints that `deqagram` parses but does not enforce
//!   (e.g. `CODE` parameter bounds, `REPEAT` counts, `ERROR` probabilities,
//!   dangling decorators).
//! - **QEC** well-formedness of each `CODE` block: that the stabilizers commute,
//!   are independent, have rank consistent with the declared `[[n,k,d]]`, and
//!   that the logical operators form a valid canonical set. These checks are
//!   backed by [`paulimer`]'s Pauli algebra.
//!
//! The declared distance `d` is only bounds-checked (`d >= 1`); its true value
//! is the minimum weight of a nontrivial logical operator, which this linter
//! does not compute. Subsystem/gauge codes are not modeled: their stabilizer
//! rank is below `n - k`, which is reported as a warning rather than an error.
//!
//! The entry point is [`lint`], which returns a [`Diagnostic`] for every problem
//! found. Diagnostics carry byte-offset [`Span`]s; render them to line/column
//! with [`Span::line_col`](deqagram::Span::line_col) against the original source.

use std::fmt;

use deqagram::Span;
use deqagram::ast::{Definition, DeqFile};

mod qec;
mod structural;

/// How serious a [`Diagnostic`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// A definite defect: the file is not a valid `.deq` program.
    Error,
    /// A likely mistake or redundancy that does not by itself invalidate the
    /// file (mirrors what deq warns about rather than rejects).
    Warning,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Error => "error",
            Self::Warning => "warning",
        })
    }
}

/// The specific rule a [`Diagnostic`] reports. Each has a stable kebab-case code
/// (via [`Rule::code`]) suitable for filtering or suppression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    // Structural.
    CodeParamN,
    CodeParamKGreaterThanN,
    CodeParamD,
    RepeatCount,
    ErrorProbability,
    DanglingDecorator,
    // QEC.
    QubitIndexOutOfRange,
    StabilizersNotCommuting,
    StabilizerGroupContainsMinusIdentity,
    StabilizerRankTooHigh,
    StabilizerRankTooLow,
    RedundantStabilizer,
    LogicalCountMismatch,
    LogicalStabilizerAnticommute,
    LogicalCanonicalForm,
    TrivialLogical,
}

impl Rule {
    /// The stable machine-readable code for this rule.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::CodeParamN => "code-param-n",
            Self::CodeParamKGreaterThanN => "code-param-k-gt-n",
            Self::CodeParamD => "code-param-d",
            Self::RepeatCount => "repeat-count",
            Self::ErrorProbability => "error-probability",
            Self::DanglingDecorator => "dangling-decorator",
            Self::QubitIndexOutOfRange => "qubit-index-out-of-range",
            Self::StabilizersNotCommuting => "stabilizers-not-commuting",
            Self::StabilizerGroupContainsMinusIdentity => "stabilizer-group-minus-identity",
            Self::StabilizerRankTooHigh => "stabilizer-rank-too-high",
            Self::StabilizerRankTooLow => "stabilizer-rank-too-low",
            Self::RedundantStabilizer => "redundant-stabilizer",
            Self::LogicalCountMismatch => "logical-count-mismatch",
            Self::LogicalStabilizerAnticommute => "logical-stabilizer-anticommute",
            Self::LogicalCanonicalForm => "logical-canonical-form",
            Self::TrivialLogical => "trivial-logical",
        }
    }

    /// The severity this rule reports at.
    #[must_use]
    pub const fn severity(self) -> Severity {
        match self {
            Self::RedundantStabilizer | Self::DanglingDecorator | Self::StabilizerRankTooLow => Severity::Warning,
            _ => Severity::Error,
        }
    }
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

/// A single problem found by the linter.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// The rule that produced this diagnostic.
    pub rule: Rule,
    /// The severity, taken from [`Rule::severity`].
    pub severity: Severity,
    /// The source span the problem is anchored to (a specific operator when
    /// known, otherwise the enclosing definition).
    pub span: Span,
    /// A human-readable description of the problem.
    pub message: String,
}

impl Diagnostic {
    pub(crate) fn new(rule: Rule, span: Span, message: impl Into<String>) -> Self {
        Self {
            rule,
            severity: rule.severity(),
            span,
            message: message.into(),
        }
    }
}

/// Lints a parsed `.deq` file, returning every problem found in source order.
///
/// The returned diagnostics reference byte offsets into the source that produced
/// `file`; use [`Span::line_col`](deqagram::Span::line_col) to render them.
#[must_use]
pub fn lint(file: &DeqFile) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    for definition in &file.definitions {
        let span = definition.span;
        match &definition.node {
            Definition::Code(code) => {
                structural::check_code_params(code, span, &mut diagnostics);
                qec::check_code(code, span, &mut diagnostics);
            }
            Definition::Gadget(g) => {
                structural::check_gadget_body(g.body.clone(), &mut diagnostics);
            }
            Definition::Compose(c) => {
                structural::check_compose_body(c.body.clone(), &mut diagnostics);
            }
            Definition::Program(p) => {
                structural::check_program_body(p.body.clone(), &mut diagnostics);
            }
        }
    }
    diagnostics
}
