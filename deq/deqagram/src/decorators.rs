//! Decorator attachment: an opt-in transform that folds standalone body
//! decorators onto the statement that follows them, producing a fully-attached
//! tree.
//!
//! In `.deq`, a decorator precedes the statement it modifies:
//!
//! ```text
//! @SIMULATE_ONLY
//! M 0 1
//! ```
//!
//! The parser keeps decorators as standalone body items (the `Decorator`
//! variant of each body-statement enum), so the parsed AST is purely syntactic:
//! a formatter or linter sees every decorator as its own spanned node, and the
//! `parse -> Display -> re-parse` roundtrip is exact. This module produces the
//! *attached* view that semantic consumers want — each statement paired with the
//! decorators that preceded it, recursively through `REPEAT` blocks.
//!
//! # Design trade-offs
//!
//! The attached tree is a **separate, derived structure**, not a mutation of the
//! parsed AST. Decorators are deliberately *not* stored as a field on the parsed
//! statement nodes, and the transform does not run during parsing. Two reasons:
//!
//! * **Roundtrip fidelity stays exact.** Displaying and re-parsing the parsed
//!   AST must yield the same AST. Attaching decorators to nodes would make
//!   `Display` reconstruct their position, and a *dangling* decorator (one with
//!   no following statement, e.g. just before `}`) has nowhere to be rendered
//!   from — it would be lost on the round trip. Keeping attachment out of the
//!   parsed AST keeps that invariant total, not conditional.
//! * **The syntactic form remains available.** A linter wants the lossless
//!   parsed shape — every decorator as an ordered, spanned node — and wants
//!   dangling decorators *reported*, not silently dropped. This transform
//!   preserves that by leaving the parsed AST untouched and returning dangling
//!   decorators to the caller (see below).
//!
//! The attached tree ([`Attached<S>`]) re-types only the recursive spine — the
//! `REPEAT` block — and reuses the parsed statement enums unchanged for every
//! other (leaf) statement. This avoids duplicating the ~30 statement variants
//! into a parallel hierarchy, and means a new leaf statement added to the
//! grammar is carried through automatically with no change here. The one cost of
//! that reuse: [`Attached::Statement`] nominally holds the whole parsed enum, so
//! its type does not *statically* forbid the `Repeat`/`Decorator` variants even
//! though this module never constructs one there (`Repeat` becomes
//! [`Attached::Repeat`]; `Decorator` is consumed). Treat `Attached::Statement`
//! as "a leaf statement" by construction.
//!
//! # Dangling decorators
//!
//! A decorator with no statement to attach to — trailing at the end of a body,
//! or the only thing in a block — is **not** an error and is **not** dropped. It
//! is returned to the caller alongside the attached tree. Dangling decorators
//! found inside nested `REPEAT` blocks are bubbled up into the single returned
//! list; each retains its own [`Span`](crate::Span), so its source location is
//! preserved regardless of nesting depth. The caller decides the policy (for
//! example: warn and ignore, or emit a diagnostic).

use crate::Spanned;
use crate::ast::{ComposeStatement, Decorator, GadgetStatement, ProgramStatement};

/// A statement in an attached tree: either a leaf statement or a `REPEAT` block
/// whose body has itself been attached.
///
/// `S` is one of the parsed body-statement enums ([`GadgetStatement`],
/// [`ComposeStatement`], [`ProgramStatement`]). Leaf statements reuse that enum
/// directly; only the recursive `REPEAT` spine is re-typed here.
#[derive(Debug, Clone, PartialEq)]
pub enum Attached<S> {
    /// A leaf statement (any parsed variant other than `REPEAT`/decorator).
    Statement(S),
    /// A `REPEAT count { ... }` block whose body is attached.
    Repeat { count: u64, body: Vec<Decorated<S>> },
}

/// A statement together with the decorators attached to it, in source order.
#[derive(Debug, Clone, PartialEq)]
pub struct Decorated<S> {
    pub decorators: Vec<Decorator>,
    pub statement: Spanned<Attached<S>>,
}

/// The three shapes a body item can take once decorators are folded.
enum Class<S> {
    Decorator(Decorator),
    Repeat { count: u64, body: Vec<Spanned<S>> },
    Leaf(S),
}

/// A body-statement enum whose decorator and repeat variants can be split out.
///
/// Implemented for the three `.deq` body-statement enums so the attachment
/// logic can be written once.
trait BodyStatement: Sized {
    fn classify(self) -> Class<Self>;
}

macro_rules! impl_body_statement {
    ($($ty:ty),+ $(,)?) => {
        $(impl BodyStatement for $ty {
            fn classify(self) -> Class<Self> {
                match self {
                    Self::Decorator(d) => Class::Decorator(d),
                    Self::Repeat { count, body } => Class::Repeat { count, body },
                    other => Class::Leaf(other),
                }
            }
        })+
    };
}

impl_body_statement!(GadgetStatement, ComposeStatement, ProgramStatement);

/// Folds standalone decorators onto the following statement, recursively through
/// nested `REPEAT` blocks.
///
/// Returns the attached tree and every dangling decorator found at any depth
/// (see the [module docs](self)).
fn attach<S: BodyStatement>(body: Vec<Spanned<S>>) -> (Vec<Decorated<S>>, Vec<Spanned<Decorator>>) {
    let mut pending: Vec<Spanned<Decorator>> = Vec::new();
    let mut out: Vec<Decorated<S>> = Vec::new();
    let mut dangling: Vec<Spanned<Decorator>> = Vec::new();

    for Spanned { node, span } in body {
        match node.classify() {
            Class::Decorator(decorator) => pending.push(Spanned::new(decorator, span)),
            Class::Repeat { count, body } => {
                let (inner, inner_dangling) = attach(body);
                dangling.extend(inner_dangling);
                out.push(Decorated {
                    decorators: pending.drain(..).map(|d| d.node).collect(),
                    statement: Spanned::new(Attached::Repeat { count, body: inner }, span),
                });
            }
            Class::Leaf(node) => out.push(Decorated {
                decorators: pending.drain(..).map(|d| d.node).collect(),
                statement: Spanned::new(Attached::Statement(node), span),
            }),
        }
    }
    // Decorators still pending at the end of this body have no target.
    dangling.extend(pending);
    (out, dangling)
}

/// Attaches decorators within a `GADGET` body. See the [module docs](self).
///
/// ```
/// use deqagram::ast::{Definition, DeqFile};
/// use deqagram::decorators::attach_gadget_body;
///
/// let file: DeqFile = "GADGET G {\n    @SIMULATE_ONLY\n    M 0\n}\n".parse().unwrap();
/// let Definition::Gadget(g) = file.definitions.into_iter().next().unwrap().node else {
///     unreachable!()
/// };
///
/// let (attached, dangling) = attach_gadget_body(g.body);
/// assert!(dangling.is_empty());
/// assert_eq!(attached[0].decorators[0].name, "SIMULATE_ONLY");
/// ```
#[must_use]
pub fn attach_gadget_body(
    body: Vec<Spanned<GadgetStatement>>,
) -> (Vec<Decorated<GadgetStatement>>, Vec<Spanned<Decorator>>) {
    attach(body)
}

/// Attaches decorators within a `COMPOSE` body. See the [module docs](self).
#[must_use]
pub fn attach_compose_body(
    body: Vec<Spanned<ComposeStatement>>,
) -> (Vec<Decorated<ComposeStatement>>, Vec<Spanned<Decorator>>) {
    attach(body)
}

/// Attaches decorators within a `PROGRAM` body. See the [module docs](self).
#[must_use]
pub fn attach_program_body(
    body: Vec<Spanned<ProgramStatement>>,
) -> (Vec<Decorated<ProgramStatement>>, Vec<Spanned<Decorator>>) {
    attach(body)
}

#[cfg(test)]
mod tests;
