//! A [pest](https://pest.rs/)-based parser and typed AST for the deq (`.deq`)
//! quantum error correction file format.
//!
//! Most users need only the typed, roundtrip-fidelity API in [`ast`]:
//! [`ast::DeqFile`] implements [`FromStr`](std::str::FromStr) with
//! [`ParseError`], so `input.parse::<DeqFile>()` is the intended entry point.
//!
//! ```
//! use deqagram::ast::DeqFile;
//!
//! let src = "CODE Rep [[3,1,1]] {\n    STABILIZER Z0*Z1 Z1*Z2\n}\n";
//! let file: DeqFile = src.parse().unwrap();
//! assert_eq!(file.definitions.len(), 1);
//!
//! // Round-trips: displaying then re-parsing yields an equal AST.
//! assert_eq!(file, file.to_string().parse().unwrap());
//! ```
//!
//! [`DeqParser`] and the generated [`Rule`] enum are **also** public, as a
//! deliberate lower-level escape hatch for consumers that want to walk pest's
//! raw concrete parse tree (`Pairs`/`Token`s) directly rather than the typed
//! AST. This mirrors pest's own design and is not incidental:
//!
//! * pest's parsing entry point is
//!   `Parser::parse(Rule, &str) -> Result<Pairs, Error<Rule>>`, so `Rule` is the
//!   handle every pest consumer uses to name a start rule and to inspect pairs
//!   (see the pest book's *Parser API* chapter,
//!   <https://pest.rs/book/parser_api.html>).
//! * `#[derive(Parser)]` generates the `Rule` enum with the **same visibility**
//!   as the parser struct (see the `pest_derive` docs,
//!   <https://docs.rs/pest_derive>). Exposing `DeqParser` therefore necessarily
//!   exposes `Rule`; they are coupled by pest's API, so a public parser can only
//!   be offered together with a public `Rule`.
//!
//! Two semver consequences follow, and are accepted for the low-level access:
//! `Rule` mirrors the grammar's rule names, so renaming or removing a grammar
//! rule is a breaking change; and [`ParseError`]'s pest-typed conversions
//! (`From<pest::error::Error<Rule>>` and [`ParseError::into_inner`]) tie the
//! public API to pest's major version. Callers who only use the typed AST are
//! insulated from both.

pub mod ast;
mod common;
pub mod decorators;
pub(crate) mod error;
pub mod imports;
pub(crate) mod span;

pub use error::ParseError;
use pest_derive::Parser;
pub use span::{Span, Spanned};

/// The pest parser for the `.deq` grammar.
///
/// A low-level handle over pest's raw parse tree; see the [crate
/// documentation](crate) for when to use this versus the typed [`ast::DeqFile`]
/// API, and why it (and the generated [`Rule`] enum) are part of the public
/// surface.
#[derive(Parser)]
#[grammar = "deq.pest"]
pub struct DeqParser;

#[cfg(test)]
mod parser_tests;
