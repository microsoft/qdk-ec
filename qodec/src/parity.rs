//! Parity equations and their property-path references.
//!
//! A parity equation is an XOR of referenced bits and integer literals 0 or 1.
//!
//! ## Reference grammar
//!
//! Every reference is a [JsonPath](https://datatracker.ietf.org/doc/rfc9535/)-style
//! property path evaluated against the gadget document root. Positional
//! indices use bracket-selector form (`[<index>]`):
//!
//! - `circuit.readouts[<index>]` - a circuit readout by its zero-based position
//!   in measurement order, including flags returned by called gadgets.
//! - `readouts[<index>]` - a readout declared by the gadget.
//! - `in[<entry>].stabilizers[<index>]` — a stabilizer-generator
//!   sign of one of the gadget's input encodings.
//! - `in[<entry>].x[<index>]` — a logical-X sign of one of the
//!   gadget's input encodings.
//! - `in[<entry>].z[<index>]` — a logical-Z sign of one of the
//!   gadget's input encodings.
//! - `out[<entry>].{stabilizers,x,z}[<index>]` — output-encoding
//!   counterparts of the above.
//!
//! ## Roles
//!
//! - **Check** (`checks:`, a list): a deterministic syndrome bit. Its
//!   references XOR to zero on a noiseless +1 codeword. Indexed positionally.
//! - **Readout** (`readouts:`, a list): a terminal bit the gadget
//!   exposes, in declaration order — the implemented instruction's `observe`
//!   outcomes first (the *observables*), then its `flags:` flags. Inside the gadget,
//!   position `i` is `readouts[i]`; a caller receives it at its call's readout
//!   offset plus `i` in `circuit.readouts`. An entry's role
//!   (observable vs flag) is fixed by that position against the implemented
//!   instruction. An observable represents an instruction's measurement result;
//!   a flag reports a parity that is zero under noiseless execution. A flag does
//!   not prescribe how the caller uses it.
//!
//! ## Serialization
//!
//! [`ParityEquation`] is an alias for `Vec<ParityTerm>`. A [`ReadoutSpec`] is either that same
//! bare array (anonymous) or a single-key `{name: array}` map (named). The
//! collections contain one parity equation per entry in declaration order. Resolving a
//! gadget turns each spec into a [`Readout`], which adds the position and
//! whether the entry is a flag. References are strings on the wire and parsed
//! values in memory; constants remain integers. Deserialization parses once; serialization preserves each
//! expression's original spelling, including slices and unions.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A parity-check entry: a flat array of references and literal bits that XOR to
/// zero on noiseless +1-codeword execution.
///
/// References retain their authored spelling, including selectors such as
/// `circuit.readouts[0:4]`. Use [`Reference::expand`] or [`Reference::indices`]
/// to consume their selected positions without reparsing.
pub type ParityEquation = Vec<ParityTerm>;

/// One XOR term: a property-path reference or the literal bit 0 or 1.
/// Serialization preserves references as strings and constants as integers.
#[derive(Debug, Clone, PartialEq, Eq, derive_more::From)]
pub enum ParityTerm {
    /// An available bit or encoding-sign reference.
    #[from]
    Reference(Reference),
    /// An integer literal on disk; `false` stores 0 and `true` stores 1.
    Bit(bool),
}

impl fmt::Display for ParityTerm {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reference(reference) => reference.fmt(formatter),
            Self::Bit(value) => u8::from(*value).fmt(formatter),
        }
    }
}

impl Serialize for ParityTerm {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Reference(reference) => reference.serialize(serializer),
            Self::Bit(value) => serializer.serialize_u8(u8::from(*value)),
        }
    }
}

impl<'de> Deserialize<'de> for ParityTerm {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct TermVisitor;
        impl serde::de::Visitor<'_> for TermVisitor {
            type Value = ParityTerm;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a property-path reference or integer bit 0 or 1")
            }

            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Reference::parse(value).map(ParityTerm::Reference).map_err(E::custom)
            }

            fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<Self::Value, E> {
                match value {
                    0 | 1 => Ok(ParityTerm::Bit(value == 1)),
                    _ => Err(E::invalid_value(serde::de::Unexpected::Unsigned(value), &self)),
                }
            }

            fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<Self::Value, E> {
                match value {
                    0 | 1 => Ok(ParityTerm::Bit(value == 1)),
                    _ => Err(E::invalid_value(serde::de::Unexpected::Signed(value), &self)),
                }
            }
        }
        deserializer.deserialize_any(TermVisitor)
    }
}

/// A declared readout: a parity equation with an optional name.
///
/// A gadget's `readouts:` list holds, in order, the implemented instruction's
/// `observe` outcomes followed by its flags. Position determines each entry's
/// role; a name is a readability alias, not an identifier for references.
///
/// On the wire an entry is either a bare parity array (anonymous) or a
/// single-key map `{name: [...]}` (named):
///
/// ```yaml
/// readouts:
///   - ["circuit.readouts[0]", "in[0].z[0]"]
///   - reject: ["circuit.readouts[1]"]
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReadoutSpec {
    /// Optional readability alias. Identity is positional; the name, when
    /// present, mirrors the implemented instruction's `flags:` name (for a
    /// flag) or simply labels an observable.
    pub name: Option<String>,
    /// The parity equation: a flat array of references and literal bits whose
    /// XOR gives the bit.
    pub equation: ParityEquation,
}

impl ReadoutSpec {
    /// Construct an anonymous readout from parsed reference expressions.
    #[must_use]
    pub fn new<T: Into<ParityTerm>>(terms: impl IntoIterator<Item = T>) -> Self {
        Self {
            name: None,
            equation: terms.into_iter().map(Into::into).collect(),
        }
    }

    /// Construct a named readout from parsed reference expressions.
    #[must_use]
    pub fn named<N: Into<String>, T: Into<ParityTerm>>(name: N, terms: impl IntoIterator<Item = T>) -> Self {
        Self {
            name: Some(name.into()),
            equation: terms.into_iter().map(Into::into).collect(),
        }
    }

    /// Iterate the reference and literal terms of this readout's equation.
    pub fn terms(&self) -> impl Iterator<Item = &ParityTerm> {
        self.equation.iter()
    }
}

impl Serialize for ReadoutSpec {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match &self.name {
            None => self.equation.serialize(serializer),
            Some(name) => {
                use serde::ser::SerializeMap;
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry(name, &self.equation)?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for ReadoutSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ReadoutVisitor;

        impl<'de> serde::de::Visitor<'de> for ReadoutVisitor {
            type Value = ReadoutSpec;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a parity array or a single-key `{name: array}` map")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<ReadoutSpec, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut equation = Vec::new();
                while let Some(item) = seq.next_element::<ParityTerm>()? {
                    equation.push(item);
                }
                Ok(ReadoutSpec { name: None, equation })
            }

            fn visit_map<A>(self, mut map: A) -> Result<ReadoutSpec, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let Some((name, equation)) = map.next_entry::<String, ParityEquation>()? else {
                    return Err(serde::de::Error::custom("a named readout must have exactly one key"));
                };
                if map.next_key::<String>()?.is_some() {
                    return Err(serde::de::Error::custom("a named readout must have exactly one key"));
                }
                Ok(ReadoutSpec {
                    name: Some(name),
                    equation,
                })
            }
        }

        deserializer.deserialize_any(ReadoutVisitor)
    }
}

/// The authored readouts list: one [`ReadoutSpec`] per gadget readout, in
/// declaration order. The implemented instruction's `observe` outcomes come
/// first (each referenced as `readouts[i]`), then its `flags:` flags.
pub type ReadoutsList = Vec<ReadoutSpec>;

/// One of a gadget's readouts, resolved against the instruction it implements.
///
/// Adds a [`position`](Self::position) and [`is_flag`](Self::is_flag) to the
/// authored [`ReadoutSpec`]. Observe outcomes precede flags in declaration order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Readout {
    /// Index in the gadget's `readouts:` list. Identity is positional, so this
    /// is what a `readouts[<i>]` reference addresses.
    pub position: usize,
    /// Optional name copied from the authored entry.
    pub name: Option<String>,
    /// Whether this realizes one of the instruction's declared `flags:` rather
    /// than an `observe:` outcome.
    pub is_flag: bool,
    /// The parity equation: a flat array of references and literal bits whose XOR
    /// gives the bit.
    pub equation: ParityEquation,
}

impl Readout {
    /// Iterate the reference and literal terms of this readout's equation.
    pub fn terms(&self) -> impl Iterator<Item = &ParityTerm> {
        self.equation.iter()
    }

    /// Assign positions and roles to an authored list.
    ///
    /// Pass [`Instruction::observe_count`](crate::Instruction::observe_count)
    /// as `observe_count`: entries before it are observables; the rest are flags.
    /// This method does not validate the number or content of entries.
    #[must_use]
    pub fn resolve_list(specs: &[ReadoutSpec], observe_count: usize) -> Vec<Self> {
        specs
            .iter()
            .enumerate()
            .map(|(position, spec)| Self {
                position,
                name: spec.name.clone(),
                is_flag: position >= observe_count,
                equation: spec.equation.clone(),
            })
            .collect()
    }

    /// The authored form, dropping what resolution derived.
    #[must_use]
    pub fn to_spec(&self) -> ReadoutSpec {
        ReadoutSpec {
            name: self.name.clone(),
            equation: self.equation.clone(),
        }
    }
}

// ── Reference: structured property-path reference ───────────────────────────────

/// Which of a gadget's two boundaries an encoding reference addresses.
///
/// A gadget has an input boundary and an output boundary; a pure-preparation
/// gadget has no input boundary at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GadgetBoundary {
    /// Input encoding — the `in[<entry>].*` path family.
    In,
    /// Output encoding — the `out[<entry>].*` path family.
    Out,
}

impl GadgetBoundary {
    /// The property-path token (`in` or `out`).
    #[must_use]
    pub fn as_path_token(self) -> &'static str {
        match self {
            Self::In => "in",
            Self::Out => "out",
        }
    }
}

/// The kind of property an encoding reference addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EncodingPropertyKind {
    /// `stabilizers[<index>]` - a stabilizer-generator sign.
    Stabilizer,
    /// `x[<index>]` - a logical-X operator sign.
    LogicalX,
    /// `z[<index>]` - a logical-Z operator sign.
    LogicalZ,
}

impl EncodingPropertyKind {
    /// The property-path token (`stabilizers`, `x`, or `z`).
    #[must_use]
    pub fn as_path_token(self) -> &'static str {
        match self {
            Self::Stabilizer => "stabilizers",
            Self::LogicalX => "x",
            Self::LogicalZ => "z",
        }
    }
}

/// The most positions one slice selector may select.
///
/// A slice is stored compactly, but every consumer that expands it allocates one
/// reference per position. Without a limit, `circuit.readouts[0:18446744073709551615]`
/// parses and then exhausts memory in the C view, the Python `expand()`, and
/// [`Reference::parse_many`]. The limit is far above any addressable gadget.
pub const MAX_SELECTED_POSITIONS: usize = 1 << 20;

/// One authored property-path expression and its parsed target and selector.
///
/// Construction validates syntax, not bounds against a gadget. Display and
/// serialization preserve the original text, including selector spelling.
/// Equality, ordering, and hashing distinguish differently spelled expressions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Reference {
    path: String,
    target: ReferenceTarget,
    selector: ReferenceSelector,
}

/// The collection addressed by a [`Reference`]'s final index selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ReferenceTarget {
    /// Circuit measurement-record bits, in measurement order.
    CircuitReadout,
    /// Declared gadget readouts, with observe outcomes before flags.
    Readout,
    /// Signs of a code property on one input or output encoding.
    EncodingProperty {
        /// Whether this reference addresses the input or the output boundary.
        boundary: GadgetBoundary,
        /// 0-based position of the encoding in the `in:` / `out:` list.
        entry: usize,
        /// Which property of the encoding is referenced.
        property: EncodingPropertyKind,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum ReferenceSelector {
    Index(usize),
    Union(Vec<usize>),
    Slice { start: usize, stop: usize, step: usize },
}

/// Parse errors for reference strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferenceParseError {
    /// The string did not match any known reference shape.
    Unrecognized(String),
    /// A selector denotes no indices.
    EmptySelection(String),
    /// A slice selects more positions than [`MAX_SELECTED_POSITIONS`].
    SelectionTooLarge {
        /// The full reference string that contained the slice.
        atom: String,
        /// How many positions the slice would select.
        selected: usize,
    },
    /// A numeric index was present but failed to parse.
    BadIndex {
        /// The full reference string that contained the bad index.
        atom: String,
        /// The substring that failed to parse as a `usize`.
        index_token: String,
    },
}

impl fmt::Display for ReferenceParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unrecognized(atom) => write!(
                f,
                "unrecognized reference '{atom}': expected `circuit.readouts[<i>]`, \
                 `readouts[<i>]`, or `(in|out)[<entry>].(stabilizers|x|z)[<i>]`"
            ),
            Self::EmptySelection(atom) => write!(f, "reference '{atom}' selects no indices"),
            Self::SelectionTooLarge { atom, selected } => write!(
                f,
                "reference '{atom}' selects {selected} positions, more than the limit of {MAX_SELECTED_POSITIONS}"
            ),
            Self::BadIndex { atom, index_token } => write!(
                f,
                "reference '{atom}' has malformed index '{index_token}': expected a non-negative integer"
            ),
        }
    }
}

impl std::error::Error for ReferenceParseError {}

impl Reference {
    /// Parse one expression relative to the gadget root, retaining its spelling.
    ///
    /// The final selector accepts an index, a union, or an exclusive-stop slice
    /// with an optional positive step. An encoding's boundary entry is always
    /// a single index. Slice parsing does not allocate one value per index.
    ///
    /// ```
    /// use qodec::Reference;
    ///
    /// let readout = Reference::parse("circuit.readouts[0]")?;
    /// assert_eq!(readout.to_string(), "circuit.readouts[0]");
    ///
    /// let sign = Reference::parse("in[0].stabilizers[1]")?;
    /// assert_eq!(sign.to_string(), "in[0].stabilizers[1]");
    ///
    /// let selected = Reference::parse("in[0].x[00:3:2]")?;
    /// assert_eq!(selected.indices().collect::<Vec<_>>(), vec![0, 2]);
    /// assert_eq!(selected.to_string(), "in[0].x[00:3:2]");
    /// # Ok::<(), qodec::ReferenceParseError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceParseError`] for invalid syntax, an empty selection, or a
    /// slice selecting more than [`MAX_SELECTED_POSITIONS`] positions.
    /// Does not check whether the selected indices exist in a gadget.
    pub fn parse(atom: &str) -> Result<Self, ReferenceParseError> {
        let open = atom
            .rfind('[')
            .ok_or_else(|| ReferenceParseError::Unrecognized(atom.to_owned()))?;
        if !atom.ends_with(']') {
            return Err(ReferenceParseError::Unrecognized(atom.to_owned()));
        }
        let head = &atom[..open];
        let parts: Vec<&str> = head.split('.').collect();
        let target = match parts.as_slice() {
            ["circuit", "readouts"] => ReferenceTarget::CircuitReadout,
            ["readouts"] => ReferenceTarget::Readout,
            [head, kind_token] => {
                let (boundary, entry_token) =
                    boundary_entry(head).ok_or_else(|| ReferenceParseError::Unrecognized(atom.to_owned()))?;
                let entry = entry_token
                    .parse::<usize>()
                    .map_err(|_| ReferenceParseError::BadIndex {
                        atom: atom.to_owned(),
                        index_token: entry_token.to_owned(),
                    })?;
                let property = match *kind_token {
                    "stabilizers" => EncodingPropertyKind::Stabilizer,
                    "x" => EncodingPropertyKind::LogicalX,
                    "z" => EncodingPropertyKind::LogicalZ,
                    _ => return Err(ReferenceParseError::Unrecognized(atom.to_owned())),
                };
                ReferenceTarget::EncodingProperty {
                    boundary,
                    entry,
                    property,
                }
            }
            _ => return Err(ReferenceParseError::Unrecognized(atom.to_owned())),
        };
        let selector = ReferenceSelector::parse(&atom[open + 1..atom.len() - 1], atom)?;
        Ok(Self {
            path: atom.to_owned(),
            target,
            selector,
        })
    }

    /// Parse and expand an expression into single-index references.
    /// For an existing value, use [`Self::expand`] without reparsing.
    ///
    /// # Errors
    ///
    /// As [`Self::parse`].
    pub fn parse_many(atom: &str) -> Result<Vec<Self>, ReferenceParseError> {
        Ok(Self::parse(atom)?.expand().collect())
    }

    /// The original path text, including selector spelling.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The collection whose positions this expression selects.
    #[must_use]
    pub fn target(&self) -> ReferenceTarget {
        self.target
    }

    /// Selected zero-based positions, in selector order, including duplicates.
    /// Positions index the measurement record, readouts, or encoding property
    /// identified by [`Self::target`]. Does not expand slices into storage.
    pub fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        let (index, union, range, step): (Option<usize>, &[usize], _, _) = match &self.selector {
            ReferenceSelector::Index(index) => (Some(*index), &[], 0..0, 1),
            ReferenceSelector::Union(indices) => (None, indices, 0..0, 1),
            ReferenceSelector::Slice { start, stop, step } => (None, &[], *start..*stop, *step),
        };
        index
            .into_iter()
            .chain(union.iter().copied())
            .chain(range.step_by(step))
    }

    /// One canonical single-index reference per selected position, without parsing.
    /// Expansion preserves order and duplicates, and returns the canonical spelling
    /// of the whole path: both the boundary entry and the selector lose any
    /// authored leading zeros or spacing.
    pub fn expand(&self) -> impl Iterator<Item = Self> + '_ {
        self.indices().map(|index| {
            let path = match self.target {
                ReferenceTarget::CircuitReadout => format!("circuit.readouts[{index}]"),
                ReferenceTarget::Readout => format!("readouts[{index}]"),
                ReferenceTarget::EncodingProperty {
                    boundary,
                    entry,
                    property,
                } => {
                    format!(
                        "{}[{entry}].{}[{index}]",
                        boundary.as_path_token(),
                        property.as_path_token()
                    )
                }
            };
            Self {
                path,
                target: self.target,
                selector: ReferenceSelector::Index(index),
            }
        })
    }
}

impl ReferenceSelector {
    fn parse(token: &str, atom: &str) -> Result<Self, ReferenceParseError> {
        if token.is_empty() {
            return Err(ReferenceParseError::BadIndex {
                atom: atom.to_owned(),
                index_token: String::new(),
            });
        }
        let bad_index = |part: &str| ReferenceParseError::BadIndex {
            atom: atom.to_owned(),
            index_token: part.to_owned(),
        };
        if token.contains(',') && !token.contains(':') {
            let mut out = Vec::new();
            for part in token.split(',') {
                let part = part.trim();
                let idx = part.parse::<usize>().map_err(|_| bad_index(part))?;
                out.push(idx);
            }
            return Ok(Self::Union(out));
        }
        if token.contains(':') {
            let parts: Vec<&str> = token.split(':').collect();
            let (first_token, limit_token, stride_token) = match parts.as_slice() {
                [first, limit] => (first.trim(), limit.trim(), "1"),
                [first, limit, stride] => (first.trim(), limit.trim(), stride.trim()),
                _ => return Err(bad_index(token)),
            };
            let first = first_token.parse::<usize>().map_err(|_| bad_index(first_token))?;
            let limit = limit_token.parse::<usize>().map_err(|_| bad_index(limit_token))?;
            let stride = stride_token.parse::<usize>().map_err(|_| bad_index(stride_token))?;
            if stride == 0 {
                return Err(bad_index(stride_token));
            }
            if first >= limit {
                return Err(ReferenceParseError::EmptySelection(atom.to_owned()));
            }
            let selected = (limit - first).div_ceil(stride);
            if selected > MAX_SELECTED_POSITIONS {
                return Err(ReferenceParseError::SelectionTooLarge {
                    atom: atom.to_owned(),
                    selected,
                });
            }
            return Ok(Self::Slice {
                start: first,
                stop: limit,
                step: stride,
            });
        }
        let index = token.parse::<usize>().map_err(|_| bad_index(token))?;
        Ok(Self::Index(index))
    }
}

impl Serialize for Reference {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.path)
    }
}

impl<'de> Deserialize<'de> for Reference {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        Self::parse(&text).map_err(serde::de::Error::custom)
    }
}

/// Split an `in[<entry>]` / `out[<entry>]` head into its boundary and entry token.
/// The `[<entry>]` selector is mandatory — there is no bare-head `in` / `out` sugar.
fn boundary_entry(token: &str) -> Option<(GadgetBoundary, &str)> {
    let (boundary, rest) = match token.strip_prefix("in") {
        Some(rest) => (GadgetBoundary::In, rest),
        None => (GadgetBoundary::Out, token.strip_prefix("out")?),
    };
    Some((boundary, rest.strip_prefix('[')?.strip_suffix(']')?))
}

impl fmt::Display for Reference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.path)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EncodingPropertyKind, GadgetBoundary, MAX_SELECTED_POSITIONS, ParityEquation, ReadoutSpec, ReadoutsList,
        Reference, ReferenceParseError, ReferenceTarget,
    };

    #[test]
    fn reference_retains_authored_selector_spelling() {
        for expression in ["circuit.readouts[00:03]", "out[01].z[3, 1,3]", "readouts[02]"] {
            let reference = Reference::parse(expression).expect("one authored expression");
            assert_eq!(reference.to_string(), expression);
            assert_eq!(reference.path(), expression);
            let yaml = serde_yaml::to_string(&reference).unwrap();
            assert_eq!(serde_yaml::from_str::<String>(&yaml).unwrap(), expression);
            assert_eq!(serde_yaml::from_str::<Reference>(&yaml).unwrap(), reference);
        }
    }

    #[test]
    fn reference_expansion_preserves_order_and_duplicates() {
        let reference = Reference::parse("out[01].z[3, 1,3]").unwrap();
        assert_eq!(reference.indices().collect::<Vec<_>>(), [3, 1, 3]);
        let expanded: Vec<_> = reference.expand().collect();
        assert_eq!(
            expanded.iter().map(Reference::path).collect::<Vec<_>>(),
            ["out[1].z[3]", "out[1].z[1]", "out[1].z[3]"]
        );
        assert!(expanded.iter().all(|atom| atom.target() == reference.target()));
        assert_ne!(reference, Reference::parse("out[1].z[3,1,3]").unwrap());
    }

    #[test]
    fn reference_slice_storage_is_independent_of_its_length() {
        let reference = Reference::parse(&format!("circuit.readouts[0:{MAX_SELECTED_POSITIONS}]")).unwrap();
        assert!(matches!(reference.selector, super::ReferenceSelector::Slice { .. }));
        assert_eq!(reference.indices().take(3).collect::<Vec<_>>(), [0, 1, 2]);
        assert_eq!(reference.indices().count(), MAX_SELECTED_POSITIONS);
    }

    #[test]
    fn a_slice_selecting_more_than_the_limit_is_rejected() {
        let too_many = format!("circuit.readouts[0:{}]", MAX_SELECTED_POSITIONS + 1);
        assert_eq!(
            Reference::parse(&too_many),
            Err(ReferenceParseError::SelectionTooLarge {
                atom: too_many.clone(),
                selected: MAX_SELECTED_POSITIONS + 1,
            })
        );
        assert!(
            Reference::parse(&format!("circuit.readouts[0:{}]", usize::MAX)).is_err(),
            "an unbounded slice must not parse",
        );
        // A stride keeps a wide span within the limit.
        assert!(Reference::parse(&format!("circuit.readouts[0:{}:2]", MAX_SELECTED_POSITIONS + 1)).is_ok());
    }

    #[test]
    fn reference_rejects_empty_selectors() {
        for head in ["circuit.readouts", "readouts", "in[0].z"] {
            for selector in ["0:0", "4:2", "2:2:3"] {
                let path = format!("{head}[{selector}]");
                assert!(matches!(
                    Reference::parse(&path),
                    Err(ReferenceParseError::EmptySelection(_))
                ));
                assert!(serde_yaml::from_str::<Reference>(&format!("'{path}'")).is_err());
            }
        }
    }

    #[test]
    fn parse_source_readout() {
        for index in [0, 42] {
            let reference = Reference::parse(&format!("circuit.readouts[{index}]")).unwrap();
            assert_eq!(reference.target(), ReferenceTarget::CircuitReadout);
            assert_eq!(reference.indices().collect::<Vec<_>>(), [index]);
        }
    }

    #[test]
    fn reject_separate_flag_record_reference() {
        assert!(matches!(
            Reference::parse("circuit.flags[0]"),
            Err(ReferenceParseError::Unrecognized(_))
        ));
        assert!(matches!(
            Reference::parse_many("circuit.flags[0:2]"),
            Err(ReferenceParseError::Unrecognized(_))
        ));
        // The dotted (named) form is likewise not a reference.
        assert!(matches!(
            Reference::parse("circuit.flags.reject"),
            Err(ReferenceParseError::Unrecognized(_))
        ));
    }

    #[test]
    fn reject_source_readout_field_suffix() {
        assert!(matches!(
            Reference::parse("circuit.readouts[0].leak"),
            Err(ReferenceParseError::Unrecognized(_))
        ));
        assert!(matches!(
            Reference::parse("circuit.readouts.m_L.lost"),
            Err(ReferenceParseError::Unrecognized(_))
        ));
    }

    #[test]
    fn reject_named_readout_reference() {
        assert!(matches!(
            Reference::parse("circuit.readouts.m_L"),
            Err(ReferenceParseError::Unrecognized(_))
        ));
    }

    #[test]
    fn reject_source_readout_bad_token() {
        // A bare integer-shaped suffix without brackets is not a valid
        // bracket-selector, so it must be rejected.
        assert!(matches!(
            Reference::parse("circuit.readouts[1bad]"),
            Err(ReferenceParseError::BadIndex { .. })
        ));
        assert!(matches!(
            Reference::parse("circuit.readouts.1bad"),
            Err(ReferenceParseError::Unrecognized(_))
        ));
        assert!(matches!(
            Reference::parse("circuit.readouts.0"),
            Err(ReferenceParseError::Unrecognized(_))
        ));
    }

    #[test]
    fn round_trip_named_variants() {
        let cases = ["circuit.readouts[0]", "circuit.readouts[42]"];
        for case in cases {
            let parsed = Reference::parse(case).unwrap();
            assert_eq!(parsed.to_string(), case);
        }
    }

    #[test]
    fn parse_input_stabilizer() {
        let atom = Reference::parse("in[0].stabilizers[3]").unwrap();
        assert_eq!(atom.indices().collect::<Vec<_>>(), [3]);
        assert_eq!(
            atom.target(),
            ReferenceTarget::EncodingProperty {
                boundary: GadgetBoundary::In,
                entry: 0,
                property: EncodingPropertyKind::Stabilizer,
            }
        );
    }

    #[test]
    fn parse_output_logical_x() {
        let atom = Reference::parse("out[1].x[0]").unwrap();
        assert_eq!(atom.indices().collect::<Vec<_>>(), [0]);
        assert_eq!(
            atom.target(),
            ReferenceTarget::EncodingProperty {
                boundary: GadgetBoundary::Out,
                entry: 1,
                property: EncodingPropertyKind::LogicalX,
            }
        );
    }

    #[test]
    fn parse_input_logical_z() {
        let atom = Reference::parse("in[2].z[1]").unwrap();
        assert_eq!(atom.indices().collect::<Vec<_>>(), [1]);
        assert_eq!(
            atom.target(),
            ReferenceTarget::EncodingProperty {
                boundary: GadgetBoundary::In,
                entry: 2,
                property: EncodingPropertyKind::LogicalZ,
            }
        );
    }

    #[test]
    fn entry_index_is_mandatory() {
        // The `[<entry>]` selector is required — the bare-head sugar
        // (`in.stabilizers[i]`) is removed and now fails to parse, on both the
        // single-reference and the expanding-selector paths.
        assert!(matches!(
            Reference::parse("in.stabilizers[0]"),
            Err(ReferenceParseError::Unrecognized(_))
        ));
        assert!(matches!(
            Reference::parse("out.z[1]"),
            Err(ReferenceParseError::Unrecognized(_))
        ));
        assert!(matches!(
            Reference::parse_many("in.stabilizers[0:2]"),
            Err(ReferenceParseError::Unrecognized(_))
        ));
        // The explicit form still parses.
        assert_eq!(
            Reference::parse("in[0].stabilizers[0]").unwrap().target(),
            ReferenceTarget::EncodingProperty {
                boundary: GadgetBoundary::In,
                entry: 0,
                property: EncodingPropertyKind::Stabilizer,
            }
        );
    }

    #[test]
    fn reject_unknown_prefix() {
        assert!(matches!(
            Reference::parse("foo.bar"),
            Err(ReferenceParseError::Unrecognized(_))
        ));
    }

    #[test]
    fn reject_unknown_kind() {
        assert!(matches!(
            Reference::parse("in[0].bogus[0]"),
            Err(ReferenceParseError::Unrecognized(_))
        ));
    }

    #[test]
    fn reject_bad_index() {
        // Negative numbers, empty brackets, and trailing garbage in the
        // bracket selector all fail as bad indices.
        assert!(matches!(
            Reference::parse("in[0].stabilizers[-1]"),
            Err(ReferenceParseError::BadIndex { .. })
        ));
        assert!(matches!(
            Reference::parse("in[0].stabilizers[]"),
            Err(ReferenceParseError::BadIndex { .. })
        ));
    }

    #[test]
    fn reject_bad_entry() {
        assert!(matches!(
            Reference::parse("in[].stabilizers[0]"),
            Err(ReferenceParseError::BadIndex { .. })
        ));
        assert!(matches!(
            Reference::parse("in[x].stabilizers[0]"),
            Err(ReferenceParseError::BadIndex { .. })
        ));
    }

    #[test]
    fn reject_named_operand_for_encoding_property() {
        // The old `{in,out}.<operand>.kind[<i>]` named form must no
        // longer parse — it should fall through to Unrecognized.
        assert!(matches!(
            Reference::parse("in.block.stabilizers[0]"),
            Err(ReferenceParseError::Unrecognized(_))
        ));
        assert!(matches!(
            Reference::parse("out.target.x[0]"),
            Err(ReferenceParseError::Unrecognized(_))
        ));
    }

    #[test]
    fn round_trip_via_display() {
        for atom_str in [
            "circuit.readouts[0]",
            "circuit.readouts[123]",
            "in[0].stabilizers[0]",
            "in[0].x[0]",
            "in[1].z[7]",
            "out[0].x[0]",
            "out[2].z[0]",
        ] {
            let parsed = Reference::parse(atom_str).expect(atom_str);
            assert_eq!(parsed.to_string(), atom_str);
        }
    }

    #[test]
    fn parse_many_single_index() {
        let atoms = Reference::parse_many("circuit.readouts[3]").unwrap();
        assert_eq!(atoms.len(), 1);
        assert_eq!(atoms[0].target(), ReferenceTarget::CircuitReadout);
        assert_eq!(atoms[0].indices().collect::<Vec<_>>(), [3]);
    }

    #[test]
    fn parse_many_slice_expands_to_range() {
        let atoms = Reference::parse_many("circuit.readouts[0:4]").unwrap();
        assert_eq!(
            atoms.iter().map(Reference::path).collect::<Vec<_>>(),
            [
                "circuit.readouts[0]",
                "circuit.readouts[1]",
                "circuit.readouts[2]",
                "circuit.readouts[3]"
            ]
        );
    }

    #[test]
    fn parse_many_strided_slice() {
        let atoms = Reference::parse_many("circuit.readouts[0:6:2]").unwrap();
        assert_eq!(
            atoms.iter().map(Reference::path).collect::<Vec<_>>(),
            ["circuit.readouts[0]", "circuit.readouts[2]", "circuit.readouts[4]"]
        );
    }

    #[test]
    fn parse_many_union_preserves_order() {
        let atoms = Reference::parse_many("circuit.readouts[0,2,5]").unwrap();
        assert_eq!(
            atoms.iter().map(Reference::path).collect::<Vec<_>>(),
            ["circuit.readouts[0]", "circuit.readouts[2]", "circuit.readouts[5]"]
        );
    }

    #[test]
    fn parse_many_slice_on_encoding_property() {
        let atoms = Reference::parse_many("in[0].stabilizers[0:2]").unwrap();
        assert_eq!(atoms.len(), 2);
        for (index, atom) in atoms.iter().enumerate() {
            assert_eq!(
                atom.target(),
                ReferenceTarget::EncodingProperty {
                    boundary: GadgetBoundary::In,
                    entry: 0,
                    property: EncodingPropertyKind::Stabilizer,
                }
            );
            assert_eq!(atom.indices().collect::<Vec<_>>(), [index]);
        }
    }

    #[test]
    fn parse_many_positional_atom_unchanged() {
        let atoms = Reference::parse_many("circuit.readouts[3]").unwrap();
        assert_eq!(atoms, vec![Reference::parse("circuit.readouts[3]").unwrap()]);
    }

    #[test]
    fn parse_accepts_slice_and_union_expressions() {
        assert_eq!(
            Reference::parse("circuit.readouts[0:4]")
                .unwrap()
                .indices()
                .collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
        assert_eq!(
            Reference::parse("in[0].stabilizers[0,1]")
                .unwrap()
                .indices()
                .collect::<Vec<_>>(),
            [0, 1]
        );
    }

    #[test]
    fn atom_parses_readout_index() {
        for index in [0, 42] {
            let reference = Reference::parse(&format!("readouts[{index}]")).unwrap();
            assert_eq!(reference.target(), ReferenceTarget::Readout);
            assert_eq!(reference.indices().collect::<Vec<_>>(), [index]);
        }
    }

    #[test]
    fn atom_readout_ref_distinct_from_body_readout() {
        // `readouts[i]` is a gadget readout reference; the physical
        // `circuit.readouts[i]` bit is a different atom.
        assert_eq!(
            Reference::parse("readouts[1]").unwrap().target(),
            ReferenceTarget::Readout
        );
        assert_eq!(
            Reference::parse("circuit.readouts[1]").unwrap().target(),
            ReferenceTarget::CircuitReadout
        );
    }

    #[test]
    fn atom_parses_encoding_sign() {
        let flip = Reference::parse("out[0].z[0]").unwrap();
        assert!(matches!(flip.target(), ReferenceTarget::EncodingProperty { .. }));
    }

    #[test]
    fn atom_rejects_empty_readout() {
        assert!(matches!(
            Reference::parse("readouts."),
            Err(ReferenceParseError::Unrecognized(_))
        ));
    }

    #[test]
    fn reference_atom_round_trip_via_display() {
        for atom_str in [
            "readouts[0]",
            "readouts[5]",
            "out[0].x[0]",
            "out[1].z[1]",
            "circuit.readouts[3]",
        ] {
            let parsed = Reference::parse(atom_str).expect(atom_str);
            assert_eq!(parsed.to_string(), atom_str);
        }
    }

    // ── Flag serde round-trips (checks-shaped list of parity equations) ─────

    #[test]
    fn flag_list_of_equations_deserializes() {
        let yaml = "\
- [\"circuit.readouts[0]\", \"circuit.readouts[2]\", \"circuit.readouts[4]\"]
- [\"circuit.readouts[1]\", \"circuit.readouts[3]\", \"circuit.readouts[5]\"]
";
        let flag: Vec<ParityEquation> = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(
            flag.iter()
                .map(|equation| equation.iter().map(ToString::to_string).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            vec![
                vec!["circuit.readouts[0]", "circuit.readouts[2]", "circuit.readouts[4]"],
                vec!["circuit.readouts[1]", "circuit.readouts[3]", "circuit.readouts[5]"],
            ]
        );
    }

    #[test]
    fn anonymous_readout_serializes_as_bare_array() {
        let readout =
            ReadoutSpec::new(["circuit.readouts[0]", "in[0].z[0]"].map(|path| Reference::parse(path).unwrap()));
        let yaml = serde_yaml::to_string(&readout).unwrap();
        assert_eq!(yaml, "- circuit.readouts[0]\n- in[0].z[0]\n");
        let back: ReadoutSpec = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(back, readout);
        assert_eq!(back.name, None);
    }

    #[test]
    fn named_readout_serializes_as_single_key_map() {
        let readout = ReadoutSpec::named(
            "reject",
            ["circuit.readouts[1]", "circuit.readouts[2]"].map(|path| Reference::parse(path).unwrap()),
        );
        let yaml = serde_yaml::to_string(&readout).unwrap();
        assert_eq!(yaml, "reject:\n- circuit.readouts[1]\n- circuit.readouts[2]\n");
        let back: ReadoutSpec = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(back, readout);
        assert_eq!(back.name.as_deref(), Some("reject"));
    }

    #[test]
    fn readouts_list_round_trips_mixed_named_and_anonymous() {
        let yaml = "\
- [\"circuit.readouts[0]\", \"in[0].z[0]\"]
- reject: [\"circuit.readouts[1]\"]
";
        let readouts: ReadoutsList = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(readouts.len(), 2);
        assert_eq!(
            readouts[0],
            ReadoutSpec::new(["circuit.readouts[0]", "in[0].z[0]"].map(|path| Reference::parse(path).unwrap()))
        );
        assert_eq!(
            readouts[1],
            ReadoutSpec::named("reject", [Reference::parse("circuit.readouts[1]").unwrap()])
        );
        let reemitted = serde_yaml::to_string(&readouts).unwrap();
        let again: ReadoutsList = serde_yaml::from_str(&reemitted).unwrap();
        assert_eq!(readouts, again);
    }
}

#[cfg(test)]
mod resolved_readout_tests {
    use super::{Readout, ReadoutSpec, Reference};

    fn specs() -> Vec<ReadoutSpec> {
        vec![
            ReadoutSpec::new([Reference::parse("circuit.readouts[0]").unwrap()]),
            ReadoutSpec::new([Reference::parse("circuit.readouts[1]").unwrap()]),
            ReadoutSpec::named("reject", [Reference::parse("circuit.readouts[2]").unwrap()]),
        ]
    }

    /// The observe count is the split: entries before it are observables, the
    /// rest realize the instruction's declared flags.
    #[test]
    fn resolving_marks_everything_past_the_observe_count_a_flag() {
        let resolved = Readout::resolve_list(&specs(), 2);
        assert_eq!(
            resolved.iter().map(|r| r.is_flag).collect::<Vec<_>>(),
            vec![false, false, true]
        );
        assert_eq!(resolved.iter().map(|r| r.position).collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    /// A name is a readability alias, not what makes an entry a flag.
    #[test]
    fn a_named_entry_before_the_split_is_still_an_observable() {
        let resolved = Readout::resolve_list(
            &[ReadoutSpec::named(
                "alias",
                [Reference::parse("circuit.readouts[0]").unwrap()],
            )],
            1,
        );
        assert_eq!(resolved[0].name.as_deref(), Some("alias"));
        assert!(!resolved[0].is_flag);
    }

    #[test]
    fn an_instruction_with_no_observes_makes_every_entry_a_flag() {
        let resolved = Readout::resolve_list(&specs(), 0);
        assert!(resolved.iter().all(|r| r.is_flag));
    }

    /// Resolution adds only derived facts, so the authored form comes back.
    #[test]
    fn to_spec_round_trips_the_authored_form() {
        let authored = specs();
        let back: Vec<ReadoutSpec> = Readout::resolve_list(&authored, 2)
            .iter()
            .map(Readout::to_spec)
            .collect();
        assert_eq!(back, authored);
    }
}
