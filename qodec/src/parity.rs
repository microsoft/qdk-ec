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

use crate::node::path::{ModelPath, Segment, indices};
use serde::{Deserialize, Serialize};
use std::fmt;

/// A parity equation: a flat array of references and literal bits whose XOR gives a bit.
/// Checks require that bit to be zero on noiseless +1-codeword execution;
/// readout and frame equations define their output bits.
///
/// References retain their authored spelling, including selectors such as
/// `circuit.readouts[0:4]`. Use [`Reference::expand`] to expand the final selector
/// or [`Reference::segments`] to inspect structure without reparsing.
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

impl ParityTerm {
    /// Check that a reference uses gadget-local parity syntax; literal bits are valid.
    /// This does not check bounds, source interpretation, or protocol correctness.
    ///
    /// # Errors
    /// Returns [`ReferenceParseError::NotParity`] for a general model address.
    pub fn validate(&self) -> Result<(), ReferenceParseError> {
        match self {
            Self::Reference(reference) => reference.require_parity(),
            Self::Bit(_) => Ok(()),
        }
    }
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
            Self::Reference(reference) => {
                reference.require_parity().map_err(serde::ser::Error::custom)?;
                reference.serialize(serializer)
            }
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
                Reference::parse_parity(value)
                    .map(ParityTerm::Reference)
                    .map_err(E::custom)
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

/// The most positions one slice selector may select.
///
/// A slice is stored compactly, but every consumer that expands it allocates one
/// reference per position. Without a limit, `circuit.readouts[0:18446744073709551615]`
/// parses and then exhausts memory in the C view, the Python `expand()`, and
/// [`Reference::parse_many`]. The limit is far above any addressable gadget.
pub const MAX_SELECTED_POSITIONS: usize = 1 << 20;

/// One structural step in a model reference.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ReferenceSegment {
    /// A dotted model field name.
    Field(String),
    /// A literal JSON-quoted mapping key.
    Key(String),
    /// A zero-based sequence position.
    Index(usize),
    /// An exclusive-stop selection of sequence positions with a positive step.
    Slice {
        /// First selected position.
        start: usize,
        /// Exclusive upper bound.
        stop: usize,
        /// Positive stride between selected positions.
        step: usize,
    },
    /// Sequence positions in authored order, including duplicates.
    Union(Vec<usize>),
}

/// An authored model path and its parsed segments.
///
/// Construction checks model-path syntax. Model lookup checks target existence. Display and
/// serialization preserve the original text, including selector spelling.
/// Equality, ordering, and hashing distinguish differently spelled expressions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Reference {
    path: String,
    pub(crate) parsed: ModelPath,
}

/// Parse errors for reference strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferenceParseError {
    /// The string did not match any known reference shape.
    Unrecognized(String),
    /// A valid model address was used where a gadget parity reference is required.
    NotParity(String),
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
                "unrecognized reference '{atom}': expected a model path with dotted fields, \
                  JSON-quoted mapping keys, and nonnegative indices, slices, or unions"
            ),
            Self::NotParity(atom) => write!(f, "model address '{atom}' is not a parity reference"),
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
    /// Parse one model address, retaining its spelling.
    ///
    /// Brackets accept a JSON-quoted mapping key, index, union, or exclusive-stop
    /// slice with an optional positive step. Slices are stored compactly.
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
    /// assert_eq!(selected.segments().last(), Some(&qodec::ReferenceSegment::Slice { start: 0, stop: 3, step: 2 }));
    /// assert_eq!(selected.to_string(), "in[0].x[00:3:2]");
    /// # Ok::<(), qodec::ReferenceParseError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceParseError`] for invalid syntax, an empty selection, or a
    /// slice selecting more than [`MAX_SELECTED_POSITIONS`] positions.
    /// Model lookup checks whether the selected targets exist.
    pub fn parse(atom: &str) -> Result<Self, ReferenceParseError> {
        Ok(Self {
            path: atom.to_owned(),
            parsed: ModelPath::parse_reference(atom)?,
        })
    }

    fn from_parsed(parsed: ModelPath) -> Self {
        Self {
            path: parsed.to_string(),
            parsed,
        }
    }

    pub(crate) fn require_parity(&self) -> Result<(), ReferenceParseError> {
        if self.has_final_selector() {
            let allowed = match &self.parsed.0[..self.parsed.0.len() - 1] {
                [Segment::Field(circuit), Segment::Field(readouts)]
                    if circuit == "circuit" && readouts == "readouts" =>
                {
                    true
                }
                [Segment::Field(readouts)] if readouts == "readouts" => true,
                [Segment::Field(boundary), Segment::Index(_), Segment::Field(property)] => {
                    matches!(boundary.as_str(), "in" | "out") && matches!(property.as_str(), "stabilizers" | "x" | "z")
                }
                _ => false,
            };
            if allowed {
                return Ok(());
            }
        }
        Err(ReferenceParseError::NotParity(self.path.clone()))
    }

    pub(crate) fn parse_parity(atom: &str) -> Result<Self, ReferenceParseError> {
        let reference = Self::parse(atom)?;
        reference.require_parity()?;
        Ok(reference)
    }

    /// Parse an expression and expand its final index selector.
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

    /// Parsed model-path steps. Authored spelling remains available through [`Self::path`].
    #[must_use]
    pub fn segments(&self) -> &[ReferenceSegment] {
        &self.parsed.0
    }

    /// Selected zero-based positions, in selector order, including duplicates.
    /// These are positions in the final index selector, not earlier selectors
    /// in the path. Returns no positions for a path ending in a field or mapping
    /// key. Does not expand slices into storage.
    fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.parsed.0.last().into_iter().flat_map(indices)
    }

    fn has_final_selector(&self) -> bool {
        matches!(
            self.parsed.0.last(),
            Some(Segment::Index(_) | Segment::Slice { .. } | Segment::Union(_))
        )
    }

    /// Expand the final index selector into canonical references without parsing.
    /// Expansion preserves order and duplicates, and returns the canonical spelling
    /// of the whole path: indices, selector spacing, and JSON key escapes are normalized.
    pub fn expand(&self) -> impl Iterator<Item = Self> + '_ {
        self.indices()
            .map(|index| {
                let mut parsed = ModelPath(self.parsed.0[..self.parsed.0.len() - 1].to_vec());
                parsed.0.push(Segment::Index(index));
                Self::from_parsed(parsed)
            })
            .chain((!self.has_final_selector()).then(|| Self::from_parsed(self.parsed.clone())))
    }
}

impl TryFrom<&str> for Reference {
    type Error = ReferenceParseError;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::parse(value)
    }
}

impl TryFrom<&String> for Reference {
    type Error = ReferenceParseError;
    fn try_from(value: &String) -> Result<Self, Self::Error> {
        Self::parse(value)
    }
}

impl From<&Reference> for Reference {
    fn from(value: &Reference) -> Self {
        value.clone()
    }
}

impl TryFrom<String> for Reference {
    type Error = ReferenceParseError;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(&value)
    }
}

pub(crate) mod frames {
    use super::{ParityEquation, Reference};
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;

    pub(crate) fn serialize<S: serde::Serializer>(
        values: &BTreeMap<Reference, ParityEquation>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        for reference in values.keys() {
            reference.require_parity().map_err(serde::ser::Error::custom)?;
        }
        values.serialize(serializer)
    }

    pub(crate) fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<BTreeMap<Reference, ParityEquation>, D::Error> {
        let values = BTreeMap::<String, ParityEquation>::deserialize(deserializer)?;
        values
            .into_iter()
            .map(|(path, equation)| {
                Reference::parse_parity(&path)
                    .map(|reference| (reference, equation))
                    .map_err(serde::de::Error::custom)
            })
            .collect()
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

impl fmt::Display for Reference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.path)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_SELECTED_POSITIONS, ParityEquation, ReadoutSpec, ReadoutsList, Reference, ReferenceParseError, Segment,
    };

    #[test]
    fn malformed_paths_and_non_parity_addresses_have_distinct_messages() {
        let error = Reference::parse("metadata.").unwrap_err();
        assert_eq!(
            error.to_string(),
            "unrecognized reference 'metadata.': expected a model path with dotted fields, JSON-quoted mapping keys, and nonnegative indices, slices, or unions"
        );
        let error = Reference::parse_parity("metadata").unwrap_err();
        assert_eq!(error, ReferenceParseError::NotParity("metadata".into()));
        assert_eq!(error.to_string(), "model address 'metadata' is not a parity reference");
    }

    #[test]
    fn model_addresses_are_not_parity_terms() {
        for path in [
            "",
            "metadata[\"description\"]",
            "layers[0].gadgets[\"M\"].in[0].z[0]",
            "in[0].code.z[0]",
        ] {
            let reference = Reference::parse(path).unwrap();
            assert!(reference.require_parity().is_err());
            let encoded = serde_json::to_string(&reference).unwrap();
            assert_eq!(serde_json::from_str::<Reference>(&encoded).unwrap(), reference);
            assert!(serde_json::from_str::<super::ParityTerm>(&encoded).is_err());
            assert!(serde_json::to_string(&super::ParityTerm::Reference(reference)).is_err());
        }
        let reference = Reference::parse("metadata[\"items\"][2,0,2]").unwrap();
        assert_eq!(
            reference.expand().map(|item| item.to_string()).collect::<Vec<_>>(),
            [
                "metadata[\"items\"][2]",
                "metadata[\"items\"][0]",
                "metadata[\"items\"][2]",
            ]
        );
        assert_eq!(Reference::parse("name").unwrap().expand().count(), 1);
    }

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
        assert!(
            expanded
                .iter()
                .all(|atom| atom.segments()[..3] == reference.segments()[..3])
        );
        assert_ne!(reference, Reference::parse("out[1].z[3,1,3]").unwrap());
    }

    #[test]
    fn reference_slice_storage_is_independent_of_its_length() {
        let reference = Reference::parse(&format!("circuit.readouts[0:{MAX_SELECTED_POSITIONS}]")).unwrap();
        assert!(matches!(
            reference.segments().last(),
            Some(super::Segment::Slice { .. })
        ));
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
            assert_eq!(
                reference.segments(),
                &[
                    Segment::Field("circuit".into()),
                    Segment::Field("readouts".into()),
                    Segment::Index(index)
                ]
            );
            assert!(reference.require_parity().is_ok());
            assert_eq!(reference.indices().collect::<Vec<_>>(), [index]);
        }
    }

    #[test]
    fn reject_separate_flag_record_reference() {
        assert!(matches!(
            Reference::parse_parity("circuit.flags[0]"),
            Err(ReferenceParseError::NotParity(_))
        ));
        assert!(matches!(
            Reference::parse_parity("circuit.flags[0:2]"),
            Err(ReferenceParseError::NotParity(_))
        ));
        assert!(matches!(
            Reference::parse_parity("circuit.flags.reject"),
            Err(ReferenceParseError::NotParity(_))
        ));
    }

    #[test]
    fn reject_source_readout_field_suffix() {
        assert!(matches!(
            Reference::parse_parity("circuit.readouts[0].leak"),
            Err(ReferenceParseError::NotParity(_))
        ));
        assert!(matches!(
            Reference::parse_parity("circuit.readouts.m_L.lost"),
            Err(ReferenceParseError::NotParity(_))
        ));
    }

    #[test]
    fn reject_named_readout_reference() {
        assert!(matches!(
            Reference::parse_parity("circuit.readouts.m_L"),
            Err(ReferenceParseError::NotParity(_))
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
            atom.segments(),
            &[
                Segment::Field("in".into()),
                Segment::Index(0),
                Segment::Field("stabilizers".into()),
                Segment::Index(3)
            ]
        );
        assert!(atom.require_parity().is_ok());
    }

    #[test]
    fn parse_output_logical_x() {
        let atom = Reference::parse("out[1].x[0]").unwrap();
        assert_eq!(atom.indices().collect::<Vec<_>>(), [0]);
        assert_eq!(
            atom.segments(),
            &[
                Segment::Field("out".into()),
                Segment::Index(1),
                Segment::Field("x".into()),
                Segment::Index(0)
            ]
        );
        assert!(atom.require_parity().is_ok());
    }

    #[test]
    fn parse_input_logical_z() {
        let atom = Reference::parse("in[2].z[1]").unwrap();
        assert_eq!(atom.indices().collect::<Vec<_>>(), [1]);
        assert_eq!(
            atom.segments(),
            &[
                Segment::Field("in".into()),
                Segment::Index(2),
                Segment::Field("z".into()),
                Segment::Index(1)
            ]
        );
        assert!(atom.require_parity().is_ok());
    }

    #[test]
    fn entry_index_is_mandatory() {
        assert!(matches!(
            Reference::parse_parity("in.stabilizers[0]"),
            Err(ReferenceParseError::NotParity(_))
        ));
        assert!(matches!(
            Reference::parse_parity("out.z[1]"),
            Err(ReferenceParseError::NotParity(_))
        ));
        assert!(matches!(
            Reference::parse_parity("in.stabilizers[0:2]"),
            Err(ReferenceParseError::NotParity(_))
        ));
        assert!(Reference::parse_parity("in[0].stabilizers[0]").is_ok());
    }

    #[test]
    fn reject_unknown_prefix() {
        assert!(matches!(
            Reference::parse_parity("foo.bar"),
            Err(ReferenceParseError::NotParity(_))
        ));
    }

    #[test]
    fn reject_unknown_kind() {
        assert!(matches!(
            Reference::parse_parity("in[0].bogus[0]"),
            Err(ReferenceParseError::NotParity(_))
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
        assert!(matches!(
            Reference::parse_parity("in.block.stabilizers[0]"),
            Err(ReferenceParseError::NotParity(_))
        ));
        assert!(matches!(
            Reference::parse_parity("out.target.x[0]"),
            Err(ReferenceParseError::NotParity(_))
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
        assert!(atoms[0].require_parity().is_ok());
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
                atom.segments(),
                &[
                    Segment::Field("in".into()),
                    Segment::Index(0),
                    Segment::Field("stabilizers".into()),
                    Segment::Index(index)
                ]
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
            assert_eq!(
                reference.segments(),
                &[Segment::Field("readouts".into()), Segment::Index(index)]
            );
            assert!(reference.require_parity().is_ok());
            assert_eq!(reference.indices().collect::<Vec<_>>(), [index]);
        }
    }

    #[test]
    fn atom_readout_ref_distinct_from_body_readout() {
        // `readouts[i]` is a gadget readout reference; the physical
        // `circuit.readouts[i]` bit is a different atom.
        assert_eq!(
            Reference::parse("readouts[1]").unwrap().segments(),
            &[Segment::Field("readouts".into()), Segment::Index(1)]
        );
        assert_eq!(
            Reference::parse("circuit.readouts[1]").unwrap().segments(),
            &[
                Segment::Field("circuit".into()),
                Segment::Field("readouts".into()),
                Segment::Index(1)
            ]
        );
    }

    #[test]
    fn atom_parses_encoding_sign() {
        let flip = Reference::parse("out[0].z[0]").unwrap();
        assert!(flip.require_parity().is_ok());
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
