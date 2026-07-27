//! Typed AST for the DEQ file format, mirroring deq's own `model.py`.
//!
//! Every construct parses from a string via [`FromStr`] and serializes back via
//! [`Display`](std::fmt::Display). Roundtrip fidelity holds at the AST level: parsing a file,
//! displaying it, and re-parsing yields the same AST. Because the format is
//! whitespace- and newline-insensitive, `Display` reformats canonically rather
//! than reproducing the original bytes; comments are not retained.
//!
//! `CHECK`/`DETECTOR` and `READOUT`/`OBSERVABLE_INCLUDE` are spelling aliases
//! rather than distinct constructs, so each pair is conflated into a single node
//! kind ([`CheckStatement`] / [`ReadoutStatement`]); `Display` emits the
//! canonical `CHECK` / `READOUT` spelling.

use std::fmt;
use std::str::FromStr;

use pest::Parser;
use pest::iterators::Pair;

use super::{DeqParser, ParseError, Rule, Span, Spanned};
use crate::common::{decode_string, decode_tag, encode_string, encode_tag, write_separated};

// ── Pauli letters ────────────────────────────────────────────────────

/// A Pauli letter. `I` only appears in `PAULI_TERM` positions (`CODE`
/// operators); operator positions (`LX0`, `X1`, ...) are always `X`/`Y`/`Z`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

impl Pauli {
    fn from_char(c: char) -> Self {
        match c {
            'I' => Self::I,
            'X' => Self::X,
            'Y' => Self::Y,
            'Z' => Self::Z,
            _ => unreachable!("grammar guarantees a Pauli letter, got {c:?}"),
        }
    }
}

impl fmt::Display for Pauli {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::I => "I",
            Self::X => "X",
            Self::Y => "Y",
            Self::Z => "Z",
        })
    }
}

// ── Stim-level targets ───────────────────────────────────────────────

/// A target of an embedded Stim instruction, `CHECK`, or `ASSERT_EQ`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    Qubit { inverted: bool, index: u64 },
    Pauli { inverted: bool, pauli: Pauli, index: u64 },
    MeasurementRecord { offset: u64 },
    PhysicalMeasurement { index: u64 },
    InputVirtual { port: u64, stabilizer: u64 },
    OutputVirtual { port: u64, stabilizer: u64 },
    SweepBit { index: u64 },
    Combiner,
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qubit { inverted, index } => {
                write!(f, "{}{index}", if *inverted { "!" } else { "" })
            }
            Self::Pauli { inverted, pauli, index } => write!(f, "{}{pauli}{index}", if *inverted { "!" } else { "" }),
            Self::MeasurementRecord { offset } => write!(f, "rec[-{offset}]"),
            Self::PhysicalMeasurement { index } => write!(f, "M{index}"),
            Self::InputVirtual { port, stabilizer } => write!(f, "IN{port}.S{stabilizer}"),
            Self::OutputVirtual { port, stabilizer } => write!(f, "OUT{port}.S{stabilizer}"),
            Self::SweepBit { index } => write!(f, "sweep[{index}]"),
            Self::Combiner => f.write_str("*"),
        }
    }
}

/// A logical Pauli operator: `LX0`, or port-scoped `IN0.LX0` / `OUT1.LZ2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogicalPauliTarget {
    pub pauli: Pauli,
    pub index: u64,
    pub port: Option<Port>,
}

/// The `IN`/`OUT` side and index of a port-scoped target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Port {
    pub kind: PortKind,
    pub index: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortKind {
    In,
    Out,
}

impl fmt::Display for PortKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::In => "IN",
            Self::Out => "OUT",
        })
    }
}

impl fmt::Display for LogicalPauliTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(port) = self.port {
            write!(f, "{}{}.L{}{}", port.kind, port.index, self.pauli, self.index)
        } else {
            write!(f, "L{}{}", self.pauli, self.index)
        }
    }
}

/// A measurement reference usable in `CONDITIONAL` / `PRESELECT` conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementRef {
    Record { offset: u64 },
    Physical { index: u64 },
    InputVirtual { port: u64, stabilizer: u64 },
    OutputVirtual { port: u64, stabilizer: u64 },
}

impl fmt::Display for MeasurementRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Record { offset } => write!(f, "rec[-{offset}]"),
            Self::Physical { index } => write!(f, "M{index}"),
            Self::InputVirtual { port, stabilizer } => {
                write!(f, "IN{port}.S{stabilizer}")
            }
            Self::OutputVirtual { port, stabilizer } => {
                write!(f, "OUT{port}.S{stabilizer}")
            }
        }
    }
}

// ── Statement-specific target sets ───────────────────────────────────

/// An item of a `READOUT` statement: a Stim target, a logical Pauli, or an
/// INPUT-port destabilizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadoutTargetItem {
    Target(Target),
    Logical(LogicalPauliTarget),
    Destabilizer { port: u64, stabilizer: u64 },
}

impl fmt::Display for ReadoutTargetItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Target(t) => write!(f, "{t}"),
            Self::Logical(l) => write!(f, "{l}"),
            Self::Destabilizer { port, stabilizer } => {
                write!(f, "IN{port}.DS{stabilizer}")
            }
        }
    }
}

/// A target of an `ERROR` statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorTarget {
    Check(u64),
    Readout(u64),
    Logical(LogicalPauliTarget),
}

impl fmt::Display for ErrorTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Check(i) => write!(f, "C{i}"),
            Self::Readout(i) => write!(f, "R{i}"),
            Self::Logical(l) => write!(f, "{l}"),
        }
    }
}

/// A term after `FROM` in a `PROPAGATE` statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PropagateTerm {
    Logical(LogicalPauliTarget),
    Destabilizer { port: u64, stabilizer: u64 },
    MeasurementRecord { offset: u64 },
    PhysicalMeasurement { index: u64 },
    Readout { index: u64 },
}

impl fmt::Display for PropagateTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Logical(l) => write!(f, "{l}"),
            Self::Destabilizer { port, stabilizer } => {
                write!(f, "IN{port}.DS{stabilizer}")
            }
            Self::MeasurementRecord { offset } => write!(f, "rec[-{offset}]"),
            Self::PhysicalMeasurement { index } => write!(f, "M{index}"),
            Self::Readout { index } => write!(f, "R{index}"),
        }
    }
}

// ── Decorators ───────────────────────────────────────────────────────

/// A decorator value: a string literal (stored without surrounding quotes),
/// an integer, or a float.
#[derive(Debug, Clone, PartialEq)]
pub enum DecoratorValue {
    String(String),
    Int(i64),
    Float(f64),
}

impl fmt::Display for DecoratorValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String(s) => write!(f, "\"{}\"", encode_string(s)),
            Self::Int(i) => write!(f, "{i}"),
            // Force a decimal point so an integral float (e.g. `1.0`) does not
            // re-parse as a `DecoratorValue::Int`, which would break roundtrip.
            Self::Float(x) => {
                let s = x.to_string();
                if s.bytes().all(|b| b.is_ascii_digit() || b == b'-') {
                    write!(f, "{s}.0")
                } else {
                    f.write_str(&s)
                }
            }
        }
    }
}

/// A decorator argument: either a positional value or a `key=value` pair.
#[derive(Debug, Clone, PartialEq)]
pub enum DecoratorArg {
    Value(DecoratorValue),
    Keyword { key: String, value: DecoratorValue },
}

impl fmt::Display for DecoratorArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value(v) => write!(f, "{v}"),
            Self::Keyword { key, value } => write!(f, "{key}={value}"),
        }
    }
}

/// A decorator like `@GTYPE(1)` or `@CHECKS("syndrome")`. `name` excludes `@`.
#[derive(Debug, Clone, PartialEq)]
pub struct Decorator {
    pub name: String,
    pub arguments: Vec<DecoratorArg>,
}

impl fmt::Display for Decorator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "@{}", self.name)?;
        if !self.arguments.is_empty() {
            write!(f, "(")?;
            write_separated(f, &self.arguments, ", ")?;
            write!(f, ")")?;
        }
        Ok(())
    }
}

// ── Embedded Stim instruction ────────────────────────────────────────

/// An embedded Stim instruction, e.g. `X_ERROR(0.03) 0 1 2` or `MPP X1 * Z2`.
#[derive(Debug, Clone, PartialEq)]
pub struct Instruction {
    pub name: String,
    pub tag: Option<String>,
    pub arguments: Vec<f64>,
    pub targets: Vec<Spanned<Target>>,
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.name, encode_tag(self.tag.as_deref()))?;
        if !self.arguments.is_empty() {
            write!(f, "(")?;
            write_separated(f, &self.arguments, ", ")?;
            write!(f, ")")?;
        }
        for target in &self.targets {
            write!(f, " {target}")?;
        }
        Ok(())
    }
}

// ── Pauli products (CODE) ────────────────────────────────────────────

/// A single Pauli term on a qubit, e.g. `Z0` or `X3`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PauliTerm {
    pub pauli: Pauli,
    pub index: u64,
}

impl fmt::Display for PauliTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.pauli, self.index)
    }
}

/// A product of Pauli terms (`Z0*Z1*Z2`) or the identity (`_`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PauliProduct {
    Identity,
    Terms(Vec<PauliTerm>),
}

impl fmt::Display for PauliProduct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identity => f.write_str("_"),
            Self::Terms(terms) => write_separated(f, terms, "*"),
        }
    }
}

// ── Ports and gadget application ─────────────────────────────────────

/// An `INPUT`/`OUTPUT CodeName qubit_indices...` port declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortDeclaration {
    pub code_name: Spanned<String>,
    pub qubit_indices: Vec<u64>,
}

/// A gadget application with explicit port bindings, e.g. `Idle IN(0 1) OUT(0 1)`
/// or the empty `Name ()`. The shortcut form (`Name integer+`) parses as an
/// [`Instruction`] instead, matching deq.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GadgetApplication {
    pub gadget_name: Spanned<String>,
    pub in_indices: Option<Vec<u64>>,
    pub out_indices: Option<Vec<u64>>,
}

impl fmt::Display for GadgetApplication {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.gadget_name)?;
        if let Some(indices) = &self.in_indices {
            write!(f, " IN(")?;
            write_separated(f, indices, " ")?;
            write!(f, ")")?;
        }
        if let Some(indices) = &self.out_indices {
            write!(f, " OUT(")?;
            write_separated(f, indices, " ")?;
            write!(f, ")")?;
        }
        if self.in_indices.is_none() && self.out_indices.is_none() {
            write!(f, " ()")?;
        }
        Ok(())
    }
}

// ── GADGET-only statements ───────────────────────────────────────────

/// A `READOUT`/`OBSERVABLE_INCLUDE targets... [FLIP]` statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadoutStatement {
    pub targets: Vec<Spanned<ReadoutTargetItem>>,
    pub flip: bool,
}

impl fmt::Display for ReadoutStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "READOUT ")?;
        write_separated(f, &self.targets, " ")?;
        if self.flip {
            write!(f, " FLIP")?;
        }
        Ok(())
    }
}

/// A `CHECK`/`DETECTOR targets... [FLIP]` statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckStatement {
    pub targets: Vec<Spanned<Target>>,
    pub flip: bool,
}

impl fmt::Display for CheckStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CHECK ")?;
        write_separated(f, &self.targets, " ")?;
        if self.flip {
            write!(f, " FLIP")?;
        }
        Ok(())
    }
}

/// An `ERROR(p) targets...` statement.
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorStatement {
    pub probability: f64,
    pub targets: Vec<Spanned<ErrorTarget>>,
}

impl fmt::Display for ErrorStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ERROR({}) ", self.probability)?;
        write_separated(f, &self.targets, " ")
    }
}

/// The condition of a `CONDITIONAL` statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Condition {
    Readout(u64),
    Measurement(MeasurementRef),
}

impl fmt::Display for Condition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Readout(i) => write!(f, "R{i}"),
            Self::Measurement(m) => write!(f, "{m}"),
        }
    }
}

/// A `CONDITIONAL condition L<P><i>...` statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConditionalStatement {
    pub condition: Condition,
    pub targets: Vec<Spanned<LogicalPauliTarget>>,
}

impl fmt::Display for ConditionalStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CONDITIONAL {} ", self.condition)?;
        write_separated(f, &self.targets, " ")
    }
}

/// A `VIRTUAL L<P><i>...` logical-correction statement (inside a GADGET).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualLogicalStatement {
    pub targets: Vec<Spanned<LogicalPauliTarget>>,
}

impl fmt::Display for VirtualLogicalStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VIRTUAL ")?;
        write_separated(f, &self.targets, " ")
    }
}

/// A `PROPAGATE L<P><i> FROM terms... [FLIP]` statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropagateStatement {
    pub target: Spanned<LogicalPauliTarget>,
    pub terms: Vec<Spanned<PropagateTerm>>,
    pub flip: bool,
}

impl fmt::Display for PropagateStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PROPAGATE {} FROM", self.target)?;
        for term in &self.terms {
            write!(f, " {term}")?;
        }
        if self.flip {
            write!(f, " FLIP")?;
        }
        Ok(())
    }
}

/// A `PRESELECT <target>+ [<bit>]` statement. Each target is a concrete
/// physical measurement (`rec[-k]` or `M<i>`); the trailing parity bit is
/// optional and defaults to 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreselectStatement {
    pub conditions: Vec<MeasurementRef>,
    pub expected_value: u64,
}

impl fmt::Display for PreselectStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PRESELECT")?;
        for condition in &self.conditions {
            write!(f, " {condition}")?;
        }
        write!(f, " {}", self.expected_value)
    }
}

// ── PROGRAM-only statements ──────────────────────────────────────────

/// An `ASSERT_EQ target value` statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssertStatement {
    pub target: Spanned<Target>,
    pub expected_value: u64,
}

impl fmt::Display for AssertStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ASSERT_EQ {} {}", self.target, self.expected_value)
    }
}

/// A `VIRTUAL X0*Y1 wire` Pauli-correction statement (inside a PROGRAM).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualCorrection {
    pub paulis: Vec<(Pauli, u64)>,
    pub wire: u64,
}

impl fmt::Display for VirtualCorrection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VIRTUAL ")?;
        for (i, (pauli, index)) in self.paulis.iter().enumerate() {
            if i > 0 {
                write!(f, "*")?;
            }
            write!(f, "{pauli}{index}")?;
        }
        write!(f, " {}", self.wire)
    }
}

/// A `CONDITIONAL rec[-k] X0*Y1 wire` conditional Pauli-correction statement.
///
/// Valid inside a COMPOSE or PROGRAM. Applies the logical Pauli product to
/// `wire` conditioned on the `readout_offset`-th most recent logical readout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConditionalCorrection {
    pub readout_offset: u64,
    pub paulis: Vec<(Pauli, u64)>,
    pub wire: u64,
}

impl fmt::Display for ConditionalCorrection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CONDITIONAL rec[-{}] ", self.readout_offset)?;
        for (i, (pauli, index)) in self.paulis.iter().enumerate() {
            if i > 0 {
                write!(f, "*")?;
            }
            write!(f, "{pauli}{index}")?;
        }
        write!(f, " {}", self.wire)
    }
}

// ── Body statements (per definition kind) ────────────────────────────

/// A statement inside a `GADGET` body.
#[derive(Debug, Clone, PartialEq)]
pub enum GadgetStatement {
    Instruction(Instruction),
    Repeat { count: u64, body: Vec<Spanned<Self>> },
    InputPort(PortDeclaration),
    OutputPort(PortDeclaration),
    Readout(ReadoutStatement),
    Check(CheckStatement),
    Error(ErrorStatement),
    Conditional(ConditionalStatement),
    VirtualLogical(VirtualLogicalStatement),
    Propagate(PropagateStatement),
    Preselect(PreselectStatement),
    Decorator(Decorator),
}

/// A statement inside a `COMPOSE` body.
#[derive(Debug, Clone, PartialEq)]
pub enum ComposeStatement {
    Instruction(Instruction),
    Repeat { count: u64, body: Vec<Spanned<Self>> },
    InputPort(PortDeclaration),
    OutputPort(PortDeclaration),
    GadgetApplication(GadgetApplication),
    ConditionalCorrection(ConditionalCorrection),
    Decorator(Decorator),
}

/// A statement inside a `PROGRAM` body.
#[derive(Debug, Clone, PartialEq)]
pub enum ProgramStatement {
    Instruction(Instruction),
    Repeat { count: u64, body: Vec<Spanned<Self>> },
    InputPort(PortDeclaration),
    OutputPort(PortDeclaration),
    GadgetApplication(GadgetApplication),
    Assert(AssertStatement),
    VirtualCorrection(VirtualCorrection),
    ConditionalCorrection(ConditionalCorrection),
    Decorator(Decorator),
}

// ── Definitions ──────────────────────────────────────────────────────

/// A pair of X and Z logical operators for one logical qubit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalOperator {
    pub x_operator: Spanned<PauliProduct>,
    pub z_operator: Spanned<PauliProduct>,
}

/// A `CODE Name [[n,k,d]] { ... }` definition.
#[derive(Debug, Clone, PartialEq)]
pub struct CodeDefinition {
    pub name: String,
    pub n: u64,
    pub k: u64,
    pub d: Option<u64>,
    pub logicals: Vec<LogicalOperator>,
    pub stabilizers: Vec<Spanned<PauliProduct>>,
    pub decorators: Vec<Decorator>,
}

/// A `GADGET Name { ... }` definition.
#[derive(Debug, Clone, PartialEq)]
pub struct GadgetDefinition {
    pub name: String,
    pub body: Vec<Spanned<GadgetStatement>>,
    pub decorators: Vec<Decorator>,
}

/// A `COMPOSE Name { ... }` definition.
#[derive(Debug, Clone, PartialEq)]
pub struct ComposeDefinition {
    pub name: String,
    pub body: Vec<Spanned<ComposeStatement>>,
    pub decorators: Vec<Decorator>,
}

/// A `PROGRAM Name { ... }` definition.
#[derive(Debug, Clone, PartialEq)]
pub struct ProgramDefinition {
    pub name: String,
    pub body: Vec<Spanned<ProgramStatement>>,
    pub decorators: Vec<Decorator>,
}

/// A top-level definition.
#[derive(Debug, Clone, PartialEq)]
pub enum Definition {
    Code(CodeDefinition),
    Gadget(GadgetDefinition),
    Compose(ComposeDefinition),
    Program(ProgramDefinition),
}

/// A complete `.deq` file: imports followed by definitions.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DeqFile {
    pub imports: Vec<String>,
    pub definitions: Vec<Spanned<Definition>>,
}

// ── Parsing helpers ──────────────────────────────────────────────────

/// Captures a pair's source span as an owned [`Span`].
fn span_of(pair: &Pair<Rule>) -> Span {
    pair.as_span().into()
}

/// Convenience methods over pest [`Pair`]s, cutting boilerplate that recurs
/// across the parser.
trait PairExt<'i> {
    /// The single inner pair. Grammar rules that wrap exactly one child (most of
    /// them) guarantee this; like the `unreachable!` arms, it trusts the grammar.
    fn only(self) -> Pair<'i, Rule>;

    /// Parses `self` with `f`, pairing the result with `self`'s source span.
    fn spanned<T>(self, f: impl FnOnce(Pair<'i, Rule>) -> Result<T, ParseError>) -> Result<Spanned<T>, ParseError>;
}

impl<'i> PairExt<'i> for Pair<'i, Rule> {
    fn only(self) -> Pair<'i, Rule> {
        self.into_inner().next().unwrap()
    }

    fn spanned<T>(self, f: impl FnOnce(Pair<'i, Rule>) -> Result<T, ParseError>) -> Result<Spanned<T>, ParseError> {
        let span = span_of(&self);
        Ok(Spanned::new(f(self)?, span))
    }
}

/// Parses the whole text of `pair` as a `u64`, reporting an out-of-range error
/// anchored at the pair's span.
fn int_u64(pair: &Pair<Rule>) -> Result<u64, ParseError> {
    sub_u64(pair, pair.as_str())
}

/// Parses `text` — a numeric substring of `pair` — as a `u64`. Errors point at
/// the whole `pair` span.
fn sub_u64(pair: &Pair<Rule>, text: &str) -> Result<u64, ParseError> {
    text.parse()
        .map_err(|_| ParseError::at_span(pair.as_span(), format!("integer out of range (0..={})", u64::MAX)))
}

/// Parses `text` — a numeric substring of `pair` — as an `i64`.
fn sub_i64(pair: &Pair<Rule>, text: &str) -> Result<i64, ParseError> {
    text.parse().map_err(|_| {
        ParseError::at_span(
            pair.as_span(),
            format!("integer out of range ({}..={})", i64::MIN, i64::MAX),
        )
    })
}

/// Parses `text` — a numeric substring of `pair` — as a finite `f64`. A literal
/// that overflows to infinity (or parses to NaN) is rejected, since it would not
/// round-trip through `Display`.
fn sub_f64(pair: &Pair<Rule>, text: &str) -> Result<f64, ParseError> {
    let value: f64 = text
        .parse()
        .map_err(|_| ParseError::at_span(pair.as_span(), "invalid floating-point number"))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ParseError::at_span(
            pair.as_span(),
            "floating-point number must be finite",
        ))
    }
}

/// Splits a `<letter><digits>` terminal into its letter and index.
fn letter_index(pair: &Pair<Rule>) -> Result<(char, u64), ParseError> {
    let s = pair.as_str();
    let letter = s.chars().next().unwrap();
    let index = sub_u64(pair, &s[letter.len_utf8()..])?;
    Ok((letter, index))
}

/// Parses the integer between two known delimiters, e.g. `rec[-3]` -> 3.
fn bracketed_int(pair: &Pair<Rule>, prefix: &str, suffix: &str) -> Result<u64, ParseError> {
    let digits = pair
        .as_str()
        .strip_prefix(prefix)
        .and_then(|s| s.strip_suffix(suffix))
        .unwrap();
    sub_u64(pair, digits)
}

/// Parses a `<PORT><p>.<KIND><s>` terminal (e.g. `IN0.S2`, `IN3.DS1`) into its
/// port index and trailing index, given the port and kind prefixes.
fn port_indexed(pair: &Pair<Rule>, port_prefix: &str, kind_prefix: &str) -> Result<(u64, u64), ParseError> {
    let (left, right) = pair.as_str().split_once('.').unwrap();
    let port = sub_u64(pair, left.strip_prefix(port_prefix).unwrap())?;
    let index = sub_u64(pair, right.strip_prefix(kind_prefix).unwrap())?;
    Ok((port, index))
}

fn parse_logical(pair: Pair<Rule>) -> Result<LogicalPauliTarget, ParseError> {
    let terminal = pair.only();
    let s = terminal.as_str();
    match terminal.as_rule() {
        Rule::LOGICAL_PAULI_TARGET => {
            // "L<P><i>"
            let pauli = Pauli::from_char(s.chars().nth(1).unwrap());
            let index = sub_u64(&terminal, &s[2..])?;
            Ok(LogicalPauliTarget {
                pauli,
                index,
                port: None,
            })
        }
        Rule::INPUT_LOGICAL_TARGET | Rule::OUTPUT_LOGICAL_TARGET => {
            // "IN<p>.L<P><i>" / "OUT<p>.L<P><i>"
            let (left, right) = s.split_once('.').unwrap();
            let (kind, port_prefix) = if terminal.as_rule() == Rule::INPUT_LOGICAL_TARGET {
                (PortKind::In, "IN")
            } else {
                (PortKind::Out, "OUT")
            };
            let port_index = sub_u64(&terminal, left.strip_prefix(port_prefix).unwrap())?;
            let pauli = Pauli::from_char(right.chars().nth(1).unwrap());
            let index = sub_u64(&terminal, &right[2..])?;
            Ok(LogicalPauliTarget {
                pauli,
                index,
                port: Some(Port {
                    kind,
                    index: port_index,
                }),
            })
        }
        rule => unreachable!("unexpected logical target rule {rule:?}"),
    }
}

fn parse_measurement_ref(pair: &Pair<Rule>) -> Result<MeasurementRef, ParseError> {
    let s = pair.as_str();
    Ok(match pair.as_rule() {
        Rule::MEASUREMENT_RECORD_TARGET => MeasurementRef::Record {
            offset: bracketed_int(pair, "rec[-", "]")?,
        },
        Rule::PHYS_MEAS_TARGET => MeasurementRef::Physical {
            index: sub_u64(pair, s.strip_prefix('M').unwrap())?,
        },
        Rule::INPUT_VIRTUAL_TARGET => {
            let (port, stabilizer) = port_indexed(pair, "IN", "S")?;
            MeasurementRef::InputVirtual { port, stabilizer }
        }
        Rule::OUTPUT_VIRTUAL_TARGET => {
            let (port, stabilizer) = port_indexed(pair, "OUT", "S")?;
            MeasurementRef::OutputVirtual { port, stabilizer }
        }
        rule => unreachable!("unexpected measurement-ref rule {rule:?}"),
    })
}

fn parse_target(pair: Pair<Rule>) -> Result<Target, ParseError> {
    let inner = pair.only();
    let s = inner.as_str();
    Ok(match inner.as_rule() {
        Rule::MEASUREMENT_RECORD_TARGET => Target::MeasurementRecord {
            offset: bracketed_int(&inner, "rec[-", "]")?,
        },
        Rule::SWEEP_BIT_TARGET => Target::SweepBit {
            index: bracketed_int(&inner, "sweep[", "]")?,
        },
        Rule::INPUT_VIRTUAL_TARGET => {
            let (port, stabilizer) = port_indexed(&inner, "IN", "S")?;
            Target::InputVirtual { port, stabilizer }
        }
        Rule::OUTPUT_VIRTUAL_TARGET => {
            let (port, stabilizer) = port_indexed(&inner, "OUT", "S")?;
            Target::OutputVirtual { port, stabilizer }
        }
        Rule::PHYS_MEAS_TARGET => Target::PhysicalMeasurement {
            index: sub_u64(&inner, s.strip_prefix('M').unwrap())?,
        },
        Rule::pauli_target => {
            let inverted = s.starts_with('!');
            let operator = inner.only();
            let (letter, index) = letter_index(&operator)?;
            Target::Pauli {
                inverted,
                pauli: Pauli::from_char(letter),
                index,
            }
        }
        Rule::qubit_target => {
            let inverted = s.starts_with('!');
            let int = inner.only();
            let index = int_u64(&int)?;
            Target::Qubit { inverted, index }
        }
        Rule::combiner => Target::Combiner,
        rule => unreachable!("unexpected target rule {rule:?}"),
    })
}

fn parse_pauli_product(pair: Pair<Rule>) -> Result<PauliProduct, ParseError> {
    let mut terms = Vec::new();
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::PAULI_IDENTITY => return Ok(PauliProduct::Identity),
            Rule::PAULI_TERM => {
                let (letter, index) = letter_index(&inner)?;
                terms.push(PauliTerm {
                    pauli: Pauli::from_char(letter),
                    index,
                });
            }
            rule => unreachable!("unexpected pauli-product rule {rule:?}"),
        }
    }
    Ok(PauliProduct::Terms(terms))
}

fn parse_decorator_value(pair: Pair<Rule>) -> Result<DecoratorValue, ParseError> {
    let inner = pair.only();
    Ok(match inner.as_rule() {
        Rule::string_literal => DecoratorValue::String(string_literal_content(&inner)),
        Rule::NUMBER => {
            let s = inner.as_str();
            if s.contains('.') || s.contains('e') || s.contains('E') {
                DecoratorValue::Float(sub_f64(&inner, s)?)
            } else {
                DecoratorValue::Int(sub_i64(&inner, s)?)
            }
        }
        rule => unreachable!("unexpected decorator-value rule {rule:?}"),
    })
}

fn parse_decorator(pair: Pair<Rule>) -> Result<Decorator, ParseError> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().strip_prefix('@').unwrap().to_string();
    let mut arguments = Vec::new();
    if let Some(args) = inner.next() {
        for arg in args.into_inner() {
            match arg.as_rule() {
                Rule::keyword_argument => {
                    let mut kv = arg.into_inner();
                    let key = kv.next().unwrap().as_str().to_string();
                    let value = parse_decorator_value(kv.next().unwrap())?;
                    arguments.push(DecoratorArg::Keyword { key, value });
                }
                Rule::decorator_value => {
                    arguments.push(DecoratorArg::Value(parse_decorator_value(arg)?));
                }
                rule => unreachable!("unexpected decorator-arg rule {rule:?}"),
            }
        }
    }
    Ok(Decorator { name, arguments })
}

/// Returns a string literal's content, without the surrounding quotes.
fn string_literal_content(pair: &Pair<Rule>) -> String {
    let escaped = pair.as_str();
    decode_string(&escaped[1..escaped.len() - 1])
}

fn parse_instruction(pair: Pair<Rule>) -> Result<Instruction, ParseError> {
    let mut name = String::new();
    let mut tag = None;
    let mut arguments = Vec::new();
    let mut targets = Vec::new();
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::IDENT => name = inner.as_str().to_string(),
            Rule::tag => tag = Some(decode_tag(inner.into_inner())),
            Rule::parenthesized_arguments => {
                arguments = inner
                    .into_inner()
                    .map(|n| sub_f64(&n, n.as_str()))
                    .collect::<Result<Vec<_>, _>>()?;
            }
            Rule::target => targets.push(inner.spanned(parse_target)?),
            rule => unreachable!("unexpected instruction rule {rule:?}"),
        }
    }
    Ok(Instruction {
        name,
        tag,
        arguments,
        targets,
    })
}

fn parse_port(pair: Pair<Rule>) -> Result<PortDeclaration, ParseError> {
    let mut inner = pair.into_inner();
    let name_pair = inner.next().unwrap();
    let code_name = Spanned::new(name_pair.as_str().to_string(), span_of(&name_pair));
    let qubit_indices = inner.map(|p| int_u64(&p)).collect::<Result<Vec<_>, _>>()?;
    Ok(PortDeclaration {
        code_name,
        qubit_indices,
    })
}

fn parse_gadget_application(pair: Pair<Rule>) -> Result<GadgetApplication, ParseError> {
    let mut inner = pair.into_inner();
    let name_pair = inner.next().unwrap();
    let gadget_name = Spanned::new(name_pair.as_str().to_string(), span_of(&name_pair));
    let mut in_indices = None;
    let mut out_indices = None;
    for binding in inner {
        let rule = binding.as_rule();
        let indices = binding
            .into_inner()
            .map(|p| int_u64(&p))
            .collect::<Result<Vec<_>, _>>()?;
        match rule {
            Rule::port_binding_in => in_indices = Some(indices),
            Rule::port_binding_out => out_indices = Some(indices),
            rule => unreachable!("unexpected gadget-application rule {rule:?}"),
        }
    }
    Ok(GadgetApplication {
        gadget_name,
        in_indices,
        out_indices,
    })
}

fn parse_readout_statement(pair: Pair<Rule>) -> Result<ReadoutStatement, ParseError> {
    let mut targets = Vec::new();
    let mut flip = false;
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::target => {
                let span = span_of(&inner);
                targets.push(Spanned::new(ReadoutTargetItem::Target(parse_target(inner)?), span));
            }
            Rule::logical_pauli_target => {
                let span = span_of(&inner);
                targets.push(Spanned::new(ReadoutTargetItem::Logical(parse_logical(inner)?), span));
            }
            Rule::destabilizer_target => {
                let span = span_of(&inner);
                let term = inner.only();
                let (port, stabilizer) = port_indexed(&term, "IN", "DS")?;
                targets.push(Spanned::new(ReadoutTargetItem::Destabilizer { port, stabilizer }, span));
            }
            Rule::flip_flag => flip = true,
            rule => unreachable!("unexpected readout rule {rule:?}"),
        }
    }
    Ok(ReadoutStatement { targets, flip })
}

fn parse_check_statement(pair: Pair<Rule>) -> Result<CheckStatement, ParseError> {
    let mut targets = Vec::new();
    let mut flip = false;
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::target => targets.push(inner.spanned(parse_target)?),
            Rule::flip_flag => flip = true,
            rule => unreachable!("unexpected check rule {rule:?}"),
        }
    }
    Ok(CheckStatement { targets, flip })
}

fn parse_error_statement(pair: Pair<Rule>) -> Result<ErrorStatement, ParseError> {
    let mut inner = pair.into_inner();
    let probability_pair = inner.next().unwrap();
    let probability = sub_f64(&probability_pair, probability_pair.as_str())?;
    let targets = inner
        .map(|t| -> Result<Spanned<ErrorTarget>, ParseError> {
            let span = span_of(&t);
            let node = match t.as_rule() {
                Rule::check_target => ErrorTarget::Check(sub_u64(&t, t.as_str().strip_prefix('C').unwrap())?),
                Rule::readout_target => ErrorTarget::Readout(sub_u64(&t, t.as_str().strip_prefix('R').unwrap())?),
                Rule::logical_pauli_target => ErrorTarget::Logical(parse_logical(t)?),
                rule => unreachable!("unexpected error-target rule {rule:?}"),
            };
            Ok(Spanned::new(node, span))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ErrorStatement { probability, targets })
}

fn parse_conditional_statement(pair: Pair<Rule>) -> Result<ConditionalStatement, ParseError> {
    let mut inner = pair.into_inner();
    let first = inner.next().unwrap();
    let condition = match first.as_rule() {
        Rule::readout_target => Condition::Readout(sub_u64(&first, first.as_str().strip_prefix('R').unwrap())?),
        _ => Condition::Measurement(parse_measurement_ref(&first)?),
    };
    let targets = inner
        .map(|p| p.spanned(parse_logical))
        .collect::<Result<Vec<_>, ParseError>>()?;
    Ok(ConditionalStatement { condition, targets })
}

fn parse_propagate_statement(pair: Pair<Rule>) -> Result<PropagateStatement, ParseError> {
    let mut inner = pair.into_inner();
    let target_pair = inner.next().unwrap();
    let target = target_pair.spanned(parse_logical)?;
    let mut terms = Vec::new();
    let mut flip = false;
    for item in inner {
        let span = span_of(&item);
        let node = match item.as_rule() {
            Rule::logical_pauli_target => PropagateTerm::Logical(parse_logical(item)?),
            Rule::destabilizer_target => {
                let term = item.only();
                let (port, stabilizer) = port_indexed(&term, "IN", "DS")?;
                PropagateTerm::Destabilizer { port, stabilizer }
            }
            Rule::MEASUREMENT_RECORD_TARGET => PropagateTerm::MeasurementRecord {
                offset: bracketed_int(&item, "rec[-", "]")?,
            },
            Rule::PHYS_MEAS_TARGET => PropagateTerm::PhysicalMeasurement {
                index: sub_u64(&item, item.as_str().strip_prefix('M').unwrap())?,
            },
            Rule::readout_target => PropagateTerm::Readout {
                index: sub_u64(&item, item.as_str().strip_prefix('R').unwrap())?,
            },
            Rule::flip_flag => {
                flip = true;
                continue;
            }
            rule => unreachable!("unexpected propagate rule {rule:?}"),
        };
        terms.push(Spanned::new(node, span));
    }
    Ok(PropagateStatement { target, terms, flip })
}

fn parse_virtual_correction(pair: Pair<Rule>) -> Result<VirtualCorrection, ParseError> {
    let mut paulis = Vec::new();
    let mut wire = 0;
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::PAULI_OPERATOR => {
                let (letter, index) = letter_index(&inner)?;
                paulis.push((Pauli::from_char(letter), index));
            }
            Rule::INT => wire = int_u64(&inner)?,
            rule => unreachable!("unexpected virtual-correction rule {rule:?}"),
        }
    }
    Ok(VirtualCorrection { paulis, wire })
}

fn parse_conditional_correction(pair: Pair<Rule>) -> Result<ConditionalCorrection, ParseError> {
    let mut readout_offset = 0;
    let mut paulis = Vec::new();
    let mut wire = 0;
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::MEASUREMENT_RECORD_TARGET => {
                readout_offset = bracketed_int(&inner, "rec[-", "]")?;
            }
            Rule::PAULI_OPERATOR => {
                let (letter, index) = letter_index(&inner)?;
                paulis.push((Pauli::from_char(letter), index));
            }
            Rule::INT => wire = int_u64(&inner)?,
            rule => unreachable!("unexpected conditional-correction rule {rule:?}"),
        }
    }
    Ok(ConditionalCorrection {
        readout_offset,
        paulis,
        wire,
    })
}

// The three `parse_*_statement` functions below (and the `write_*_body` trio in
// the Display section) share a spine — `Repeat`/`InputPort`/`OutputPort`/
// `Instruction`/`Decorator` are handled identically — but are deliberately kept
// as separate copies rather than folded behind a macro/trait like
// `decorators::BodyStatement`. The set of statement kinds per body context is
// stable, so these dispatch tables change ~never, and three boring functions
// read clearer than one clever abstraction. Keep the shared arms in sync by hand.
fn parse_gadget_statement(pair: Pair<Rule>) -> Result<Spanned<GadgetStatement>, ParseError> {
    let span = span_of(&pair);
    let node = match pair.as_rule() {
        Rule::repeat_block_gadget => {
            let mut inner = pair.into_inner();
            let count = int_u64(&inner.next().unwrap())?;
            let body = inner.map(parse_gadget_statement).collect::<Result<Vec<_>, _>>()?;
            GadgetStatement::Repeat { count, body }
        }
        Rule::input_port => GadgetStatement::InputPort(parse_port(pair)?),
        Rule::output_port => GadgetStatement::OutputPort(parse_port(pair)?),
        Rule::readout_statement => GadgetStatement::Readout(parse_readout_statement(pair)?),
        Rule::check_statement => GadgetStatement::Check(parse_check_statement(pair)?),
        Rule::error_statement => GadgetStatement::Error(parse_error_statement(pair)?),
        Rule::conditional_statement => GadgetStatement::Conditional(parse_conditional_statement(pair)?),
        Rule::preselect_statement => {
            let mut conditions = Vec::new();
            let mut expected_value = 0;
            for item in pair.into_inner() {
                if item.as_rule() == Rule::INT {
                    expected_value = int_u64(&item)?;
                } else {
                    conditions.push(parse_measurement_ref(&item)?);
                }
            }
            GadgetStatement::Preselect(PreselectStatement {
                conditions,
                expected_value,
            })
        }
        Rule::virtual_logical_statement => {
            let targets = pair
                .into_inner()
                .map(|p| p.spanned(parse_logical))
                .collect::<Result<Vec<_>, ParseError>>()?;
            GadgetStatement::VirtualLogical(VirtualLogicalStatement { targets })
        }
        Rule::propagate_statement => GadgetStatement::Propagate(parse_propagate_statement(pair)?),
        Rule::decorator => GadgetStatement::Decorator(parse_decorator(pair)?),
        Rule::instruction => GadgetStatement::Instruction(parse_instruction(pair)?),
        rule => unreachable!("unexpected gadget statement rule {rule:?}"),
    };
    Ok(Spanned::new(node, span))
}

fn parse_compose_statement(pair: Pair<Rule>) -> Result<Spanned<ComposeStatement>, ParseError> {
    let span = span_of(&pair);
    let node = match pair.as_rule() {
        Rule::repeat_block_compose => {
            let mut inner = pair.into_inner();
            let count = int_u64(&inner.next().unwrap())?;
            let body = inner.map(parse_compose_statement).collect::<Result<Vec<_>, _>>()?;
            ComposeStatement::Repeat { count, body }
        }
        Rule::input_port => ComposeStatement::InputPort(parse_port(pair)?),
        Rule::output_port => ComposeStatement::OutputPort(parse_port(pair)?),
        Rule::gadget_application => ComposeStatement::GadgetApplication(parse_gadget_application(pair)?),
        Rule::conditional_correction => ComposeStatement::ConditionalCorrection(parse_conditional_correction(pair)?),
        Rule::decorator => ComposeStatement::Decorator(parse_decorator(pair)?),
        Rule::instruction => ComposeStatement::Instruction(parse_instruction(pair)?),
        rule => unreachable!("unexpected compose statement rule {rule:?}"),
    };
    Ok(Spanned::new(node, span))
}

fn parse_program_statement(pair: Pair<Rule>) -> Result<Spanned<ProgramStatement>, ParseError> {
    let span = span_of(&pair);
    let node = match pair.as_rule() {
        Rule::repeat_block_program => {
            let mut inner = pair.into_inner();
            let count = int_u64(&inner.next().unwrap())?;
            let body = inner.map(parse_program_statement).collect::<Result<Vec<_>, _>>()?;
            ProgramStatement::Repeat { count, body }
        }
        Rule::input_port => ProgramStatement::InputPort(parse_port(pair)?),
        Rule::output_port => ProgramStatement::OutputPort(parse_port(pair)?),
        Rule::gadget_application => ProgramStatement::GadgetApplication(parse_gadget_application(pair)?),
        Rule::assert_statement => {
            let mut inner = pair.into_inner();
            let target_pair = inner.next().unwrap();
            let target = target_pair.spanned(parse_target)?;
            let expected_value = int_u64(&inner.next().unwrap())?;
            ProgramStatement::Assert(AssertStatement { target, expected_value })
        }
        Rule::virtual_correction => ProgramStatement::VirtualCorrection(parse_virtual_correction(pair)?),
        Rule::conditional_correction => ProgramStatement::ConditionalCorrection(parse_conditional_correction(pair)?),
        Rule::decorator => ProgramStatement::Decorator(parse_decorator(pair)?),
        Rule::instruction => ProgramStatement::Instruction(parse_instruction(pair)?),
        rule => unreachable!("unexpected program statement rule {rule:?}"),
    };
    Ok(Spanned::new(node, span))
}

fn parse_code_definition(pair: Pair<Rule>, decorators: Vec<Decorator>) -> Result<CodeDefinition, ParseError> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    let mut params = inner.next().unwrap().into_inner();
    let n = int_u64(&params.next().unwrap())?;
    let k = int_u64(&params.next().unwrap())?;
    let d = params.next().map(|p| int_u64(&p)).transpose()?;

    let mut logicals = Vec::new();
    let mut stabilizers = Vec::new();
    for item in inner {
        match item.as_rule() {
            Rule::logical_declaration => {
                let mut products = item.into_inner();
                let x_operator = products.next().unwrap().spanned(parse_pauli_product)?;
                let z_operator = products.next().unwrap().spanned(parse_pauli_product)?;
                logicals.push(LogicalOperator { x_operator, z_operator });
            }
            Rule::stabilizer_declaration => {
                for product in item.into_inner() {
                    stabilizers.push(product.spanned(parse_pauli_product)?);
                }
            }
            rule => unreachable!("unexpected code-body rule {rule:?}"),
        }
    }
    Ok(CodeDefinition {
        name,
        n,
        k,
        d,
        logicals,
        stabilizers,
        decorators,
    })
}

fn parse_definition(pair: Pair<Rule>, decorators: Vec<Decorator>) -> Result<Definition, ParseError> {
    Ok(match pair.as_rule() {
        Rule::code_definition => Definition::Code(parse_code_definition(pair, decorators)?),
        Rule::gadget_definition => {
            let mut inner = pair.into_inner();
            let name = inner.next().unwrap().as_str().to_string();
            let body = inner.map(parse_gadget_statement).collect::<Result<Vec<_>, _>>()?;
            Definition::Gadget(GadgetDefinition { name, body, decorators })
        }
        Rule::compose_definition => {
            let mut inner = pair.into_inner();
            let name = inner.next().unwrap().as_str().to_string();
            let body = inner.map(parse_compose_statement).collect::<Result<Vec<_>, _>>()?;
            Definition::Compose(ComposeDefinition { name, body, decorators })
        }
        Rule::program_definition => {
            let mut inner = pair.into_inner();
            let name = inner.next().unwrap().as_str().to_string();
            let body = inner.map(parse_program_statement).collect::<Result<Vec<_>, _>>()?;
            Definition::Program(ProgramDefinition { name, body, decorators })
        }
        rule => unreachable!("unexpected definition rule {rule:?}"),
    })
}

// ── FromStr ──────────────────────────────────────────────────────────

impl FromStr for DeqFile {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let file = DeqParser::parse(Rule::deq_file, s)?.next().unwrap();
        let mut imports = Vec::new();
        let mut definitions = Vec::new();
        for item in file.into_inner() {
            match item.as_rule() {
                Rule::import_statement => {
                    let literal = item.only();
                    imports.push(string_literal_content(&literal));
                }
                Rule::decorated_definition => {
                    let mut decorators = Vec::new();
                    let mut definition = None;
                    for inner in item.into_inner() {
                        if inner.as_rule() == Rule::decorator {
                            decorators.push(parse_decorator(inner)?);
                        } else {
                            definition = Some(inner);
                        }
                    }
                    // Span the definition itself (from its `CODE`/`GADGET`/… keyword),
                    // not any leading decorators — the decorators are captured
                    // separately, and a definition-anchored span gives cleaner
                    // diagnostics.
                    let definition = definition.unwrap();
                    let span = span_of(&definition);
                    definitions.push(Spanned::new(parse_definition(definition, decorators)?, span));
                }
                Rule::EOI => {}
                rule => unreachable!("unexpected top-level rule {rule:?}"),
            }
        }
        Ok(Self { imports, definitions })
    }
}

// ── Display ──────────────────────────────────────────────────────────

const INDENT: &str = "    ";

fn write_decorators(f: &mut fmt::Formatter<'_>, decorators: &[Decorator]) -> fmt::Result {
    for decorator in decorators {
        writeln!(f, "{decorator}")?;
    }
    Ok(())
}

fn write_indent(f: &mut fmt::Formatter<'_>, level: usize) -> fmt::Result {
    for _ in 0..level {
        f.write_str(INDENT)?;
    }
    Ok(())
}

fn write_port(f: &mut fmt::Formatter<'_>, level: usize, keyword: &str, port: &PortDeclaration) -> fmt::Result {
    write_indent(f, level)?;
    write!(f, "{keyword} {}", port.code_name)?;
    for index in &port.qubit_indices {
        write!(f, " {index}")?;
    }
    writeln!(f)
}

fn write_line<T: fmt::Display>(f: &mut fmt::Formatter<'_>, level: usize, value: &T) -> fmt::Result {
    write_indent(f, level)?;
    writeln!(f, "{value}")
}

// Deliberate copy of the `parse_*_statement` shape; see the note there.
fn write_gadget_body(f: &mut fmt::Formatter<'_>, body: &[Spanned<GadgetStatement>], level: usize) -> fmt::Result {
    for statement in body {
        match &statement.node {
            GadgetStatement::Repeat { count, body } => {
                write_indent(f, level)?;
                writeln!(f, "REPEAT {count} {{")?;
                write_gadget_body(f, body, level + 1)?;
                write_indent(f, level)?;
                writeln!(f, "}}")?;
            }
            GadgetStatement::InputPort(port) => write_port(f, level, "INPUT", port)?,
            GadgetStatement::OutputPort(port) => write_port(f, level, "OUTPUT", port)?,
            GadgetStatement::Instruction(v) => write_line(f, level, v)?,
            GadgetStatement::Readout(v) => write_line(f, level, v)?,
            GadgetStatement::Check(v) => write_line(f, level, v)?,
            GadgetStatement::Error(v) => write_line(f, level, v)?,
            GadgetStatement::Conditional(v) => write_line(f, level, v)?,
            GadgetStatement::VirtualLogical(v) => write_line(f, level, v)?,
            GadgetStatement::Propagate(v) => write_line(f, level, v)?,
            GadgetStatement::Preselect(v) => write_line(f, level, v)?,
            GadgetStatement::Decorator(v) => write_line(f, level, v)?,
        }
    }
    Ok(())
}

fn write_compose_body(f: &mut fmt::Formatter<'_>, body: &[Spanned<ComposeStatement>], level: usize) -> fmt::Result {
    for statement in body {
        match &statement.node {
            ComposeStatement::Repeat { count, body } => {
                write_indent(f, level)?;
                writeln!(f, "REPEAT {count} {{")?;
                write_compose_body(f, body, level + 1)?;
                write_indent(f, level)?;
                writeln!(f, "}}")?;
            }
            ComposeStatement::InputPort(port) => write_port(f, level, "INPUT", port)?,
            ComposeStatement::OutputPort(port) => write_port(f, level, "OUTPUT", port)?,
            ComposeStatement::GadgetApplication(v) => write_line(f, level, v)?,
            ComposeStatement::ConditionalCorrection(v) => write_line(f, level, v)?,
            ComposeStatement::Instruction(v) => write_line(f, level, v)?,
            ComposeStatement::Decorator(v) => write_line(f, level, v)?,
        }
    }
    Ok(())
}

fn write_program_body(f: &mut fmt::Formatter<'_>, body: &[Spanned<ProgramStatement>], level: usize) -> fmt::Result {
    for statement in body {
        match &statement.node {
            ProgramStatement::Repeat { count, body } => {
                write_indent(f, level)?;
                writeln!(f, "REPEAT {count} {{")?;
                write_program_body(f, body, level + 1)?;
                write_indent(f, level)?;
                writeln!(f, "}}")?;
            }
            ProgramStatement::InputPort(port) => write_port(f, level, "INPUT", port)?,
            ProgramStatement::OutputPort(port) => write_port(f, level, "OUTPUT", port)?,
            ProgramStatement::GadgetApplication(v) => write_line(f, level, v)?,
            ProgramStatement::Assert(v) => write_line(f, level, v)?,
            ProgramStatement::VirtualCorrection(v) => write_line(f, level, v)?,
            ProgramStatement::ConditionalCorrection(v) => write_line(f, level, v)?,
            ProgramStatement::Instruction(v) => write_line(f, level, v)?,
            ProgramStatement::Decorator(v) => write_line(f, level, v)?,
        }
    }
    Ok(())
}

impl fmt::Display for CodeDefinition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_decorators(f, &self.decorators)?;
        match self.d {
            Some(d) => writeln!(f, "CODE {} [[{},{},{}]] {{", self.name, self.n, self.k, d)?,
            None => writeln!(f, "CODE {} [[{},{}]] {{", self.name, self.n, self.k)?,
        }
        for logical in &self.logicals {
            writeln!(f, "{INDENT}LOGICAL {} {}", logical.x_operator, logical.z_operator)?;
        }
        for stabilizer in &self.stabilizers {
            writeln!(f, "{INDENT}STABILIZER {stabilizer}")?;
        }
        writeln!(f, "}}")
    }
}

impl fmt::Display for GadgetDefinition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_decorators(f, &self.decorators)?;
        writeln!(f, "GADGET {} {{", self.name)?;
        write_gadget_body(f, &self.body, 1)?;
        writeln!(f, "}}")
    }
}

impl fmt::Display for ComposeDefinition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_decorators(f, &self.decorators)?;
        writeln!(f, "COMPOSE {} {{", self.name)?;
        write_compose_body(f, &self.body, 1)?;
        writeln!(f, "}}")
    }
}

impl fmt::Display for ProgramDefinition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_decorators(f, &self.decorators)?;
        writeln!(f, "PROGRAM {} {{", self.name)?;
        write_program_body(f, &self.body, 1)?;
        writeln!(f, "}}")
    }
}

impl fmt::Display for Definition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Code(d) => write!(f, "{d}"),
            Self::Gadget(d) => write!(f, "{d}"),
            Self::Compose(d) => write!(f, "{d}"),
            Self::Program(d) => write!(f, "{d}"),
        }
    }
}

impl fmt::Display for DeqFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for import in &self.imports {
            writeln!(f, "IMPORT \"{}\"", encode_string(import))?;
        }
        for (i, definition) in self.definitions.iter().enumerate() {
            if i > 0 || !self.imports.is_empty() {
                writeln!(f)?;
            }
            write!(f, "{definition}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
