//! Source spans exposed to Python.

use pyo3::prelude::*;

/// A byte-offset range `start..end` into the original `.deq` source.
///
/// [`line_col`](Span::line_col) resolves the start to a 1-based `(line, column)`
/// against the source string the span came from — deqagram omits the source
/// text from the AST, so the caller supplies it.
#[pyclass(name = "Span", frozen, get_all, eq)]
#[derive(Clone, Copy, PartialEq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

#[pymethods]
impl Span {
    /// Resolves the span's start to a 1-based `(line, column)` against `source`.
    ///
    /// Returns `None` if `start` is not a valid offset in `source` (e.g. the
    /// span came from a different string).
    fn line_col(&self, source: &str) -> Option<(usize, usize)> {
        ::deqagram::Span {
            start: self.start,
            end: self.end,
        }
        .line_col(source)
    }

    fn __repr__(&self) -> String {
        format!("Span(start={}, end={})", self.start, self.end)
    }
}

impl From<::deqagram::Span> for Span {
    fn from(span: ::deqagram::Span) -> Self {
        Self {
            start: span.start,
            end: span.end,
        }
    }
}
