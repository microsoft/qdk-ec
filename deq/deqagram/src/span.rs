//! Source spans for AST nodes.

use std::fmt;
use std::ops::Deref;

/// A byte-offset range into the original `.deq` source (`start..end`).
///
/// Offsets are owned (unlike [`pest::Span`], which borrows the input), so a
/// span can travel with the AST after the source string is dropped. They are
/// the interchange format expected by diagnostic renderers such as
/// `codespan-reporting`, `ariadne`, and `miette`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    /// Resolves this span's start to a 1-based `(line, column)` against the
    /// original `source`. Returns `None` if `start` is not a valid offset in
    /// `source` (e.g. the span came from a different string).
    #[must_use]
    pub fn line_col(self, source: &str) -> Option<(usize, usize)> {
        pest::Position::new(source, self.start).map(|p| p.line_col())
    }
}

impl From<pest::Span<'_>> for Span {
    fn from(span: pest::Span<'_>) -> Self {
        Self {
            start: span.start(),
            end: span.end(),
        }
    }
}

/// An AST node paired with the source [`Span`] it was parsed from.
///
/// Equality considers only the wrapped `node`, never the span, so the roundtrip
/// invariant (`parse -> Display -> re-parse` yields an equal AST) still holds
/// even though `Display` does not reproduce byte offsets. [`Deref`] and a
/// forwarding [`Display`](std::fmt::Display) make a `Spanned<T>` usable almost
/// anywhere a `T` is.
#[derive(Debug, Clone, Copy)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    #[must_use]
    pub const fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }
}

impl<T> Deref for Spanned<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.node
    }
}

impl<T: PartialEq> PartialEq for Spanned<T> {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
    }
}

impl<T: Eq> Eq for Spanned<T> {}

/// Compares a `Spanned<T>` directly against a bare `T`, ignoring the span.
/// Keeps consumers and tests concise (`spanned == bare_node`).
impl<T: PartialEq> PartialEq<T> for Spanned<T> {
    fn eq(&self, other: &T) -> bool {
        &self.node == other
    }
}

impl<T: fmt::Display> fmt::Display for Spanned<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.node, f)
    }
}
