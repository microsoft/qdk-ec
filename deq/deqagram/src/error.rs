//! Error type for parsing `.deq` sources.

use std::fmt;

use pest::Span;
use pest::error::{Error, ErrorVariant};

use super::Rule;

/// An error produced while parsing a `.deq` source.
///
/// Wraps pest's parser error — syntax errors, with line/column and a
/// caret-underlined span — and also carries *semantic* errors detected after a
/// structurally valid parse: currently out-of-range or non-finite numeric
/// literals (e.g. an integer larger than [`u64::MAX`], or a float that overflows
/// to infinity). Both kinds render with the same span-annotated formatting.
#[derive(Debug, Clone)]
pub struct ParseError(Error<Rule>);

impl ParseError {
    /// Builds a semantic error carrying `message`, anchored at `span`.
    pub(crate) fn at_span(span: Span<'_>, message: impl Into<String>) -> Self {
        Self(Error::new_from_span(
            ErrorVariant::CustomError {
                message: message.into(),
            },
            span,
        ))
    }

    /// Returns the underlying pest error.
    ///
    /// This intentionally exposes pest's `Error<Rule>` as an escape hatch for
    /// callers that want pest's own rendering or programmatic error fields. It
    /// is part of the same deliberate low-level surface as
    /// [`DeqParser`](crate::DeqParser) and [`Rule`](crate::Rule) (see the crate
    /// docs); it ties this accessor to pest's major version.
    #[must_use]
    pub fn into_inner(self) -> Error<Rule> {
        self.0
    }
}

impl From<Error<Rule>> for ParseError {
    fn from(error: Error<Rule>) -> Self {
        // Friendly rule labels (`error.renamed_rules(..)`) would be applied here.
        Self(error)
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for ParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}
