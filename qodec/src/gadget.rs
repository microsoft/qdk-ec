//! The gadget format.
//!
//! A [`GadgetSpec`] describes the instruction it implements, its circuit,
//! boundary encodings, parameter bindings, and parity equations.
//! References such as `in[0].stabilizers[0]` and `circuit.readouts[0]`
//! are paths relative to the gadget.

use crate::BlockName;
use crate::Sourced;
use crate::{ParityEquation, ReadoutsList};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A reference to a logical instruction by its declaring instruction set file and
/// its mnemonic within that instruction set, written on disk as the single string
/// `<instruction_set>#<mnemonic>` (an RFC 3986 plain-name fragment): the `#` crosses
/// from the instruction set file to the mnemonic declared within it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Implements {
    pub instruction_set: String,
    pub mnemonic: String,
}

impl Serialize for Implements {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        format!("{}#{}", self.instruction_set, self.mnemonic).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Implements {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;
        let raw = String::deserialize(deserializer)?;
        // A mnemonic is an identifier and cannot contain `#`, so the last one
        // separates it from an instruction set path that may itself contain one.
        let (instruction_set, mnemonic) = raw.rsplit_once('#').ok_or_else(|| {
            D::Error::custom(format!(
                "`implements` must be an `<instruction_set>#<mnemonic>` reference, got `{raw}`"
            ))
        })?;
        if instruction_set.is_empty() || mnemonic.is_empty() {
            return Err(D::Error::custom(format!(
                "`implements` must be a non-empty `<instruction_set>#<mnemonic>` reference, got `{raw}`"
            )));
        }
        Ok(Self {
            instruction_set: instruction_set.to_owned(),
            mnemonic: mnemonic.to_owned(),
        })
    }
}

/// One side (input or output) of a gadget's encoding map: the circuit
/// support — a list of typed block ids drawn from the realization's operand
/// space — that forms the codeword.
///
/// Stored on the gadget as an ordered, positional list aligned with the
/// implemented instruction's `in:` / `out:` operand lists (`in: [EncodingSpec, ...]`,
/// likewise for `out`). On disk each entry is a **single-key map**
/// `{<block-type>: [<support…>]}` — e.g. `c4c6_block: [0, 1, 2]`. The key
/// names the block type the instruction declares at this position; because
/// position already binds an entry to its operand, the key is a readable,
/// checkable label (the loader verifies it agrees with the operand) rather
/// than load-bearing data. Two same-type operands therefore become two
/// positional entries with the same key, distinguished by order.
///
/// The code an entry encodes into is *not* stated here: it is bound once on
/// the gadget's source layer (`LayerSpec::codes`), keyed by that same block
/// type. The property-path DSL resolves a reference
/// `in[<entry>].stabilizers[<i>]` by indexing `inputs` at `<entry>`, taking
/// the layer-bound [`crate::Code`] for the instruction's block type at
/// that position, and indexing into the code's `stabilizers` (or `x`/`z`)
/// at position `<i>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodingSpec {
    /// The block type the positionally-aligned instruction operand declares.
    /// On disk this is the entry's single map key; the loader checks it
    /// agrees with the operand at this position.
    pub block_type: String,
    /// Ordered list of circuit-operand ids the code's internal blocks land
    /// on. `support[i]` names the circuit operand occupied by the
    /// code's i-th internal block on the side this encoding lives on (input or
    /// output, determined by the enclosing `in` / `out` key). The
    /// block-type for each id is looked up in [`CircuitSpec::inputs`]
    /// or [`CircuitSpec::outputs`] at resolution time.
    pub support: Vec<BlockName>,
}

impl Serialize for EncodingSpec {
    /// Emit the entry as a single-key map `{block_type: [support…]}`.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(1))?;
        map.serialize_entry(&self.block_type, &self.support)?;
        map.end()
    }
}

impl<'de> Deserialize<'de> for EncodingSpec {
    /// Read the entry from a single-key map `{block_type: [support…]}`.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;
        let map: BTreeMap<String, Vec<BlockName>> = BTreeMap::deserialize(deserializer)?;
        let mut entries = map.into_iter();
        let (block_type, support) = entries.next().ok_or_else(|| {
            D::Error::custom(
                "an `in`/`out` encoding entry must be a single-key map `{block_type: [support…]}`, got an empty map",
            )
        })?;
        if entries.next().is_some() {
            return Err(D::Error::custom(
                "an `in`/`out` encoding entry must be a single-key map `{block_type: [support…]}`; \
                 list one entry per operand instead of several keys in one map",
            ));
        }
        Ok(Self { block_type, support })
    }
}

/// The realization circuit: a circuit source plus optional metadata
/// describing the operands it expects.
///
/// On-disk shapes accepted by [`CircuitSpec`]'s [`Deserialize`]:
/// - `circuit: ./foo.stim` is shorthand for `circuit: {source: ./foo.stim}`.
/// - `circuit: [{M: [0]}]` is shorthand for `circuit: {source: [{M: [0]}]}`.
/// - The object form has `source` and optional `instruction_set`, `format`, `in`, and `out`.
///
/// When only `source` is set, serialization uses the shorthand form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CircuitSpec {
    /// The instruction set the circuit's source calls into. Block-type names used in
    /// [`CircuitSpec::inputs`] / [`CircuitSpec::outputs`] are interpreted in this
    /// instruction set's `blocks:` table. Optional: when absent, the layer that lists
    /// the owning gadget supplies the target instruction set (the layer below's). When
    /// present, the loader checks it agrees with that layer.
    pub instruction_set: Option<String>,
    /// The circuit source: either a sibling-file path or an inline
    /// sequence of instructions.
    pub source: Sourced<String>,
    /// Source-format tag for inline text, independent of parser availability.
    ///
    /// Its presence is authoritative for how a string `source` is read:
    /// when set, the string is the verbatim inline circuit; when absent, a
    /// string `source` is a sibling-file path (the format is inferred
    /// from the file extension) and a sequence `source` is inline-YAML.
    pub format: Option<String>,
    /// Optional typed input operands: a map from operand name (as used
    /// inside the circuit, e.g. a stim qubit-id label or a QASM register
    /// name) to the block-type name in [`CircuitSpec::instruction_set`].
    pub inputs: BTreeMap<String, String>,
    /// Optional typed output operands. Symmetric to `inputs`.
    pub outputs: BTreeMap<String, String>,
}

impl Default for CircuitSpec {
    fn default() -> Self {
        Self {
            instruction_set: None,
            source: Sourced::inline(String::new()),
            format: None,
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
        }
    }
}

impl CircuitSpec {
    /// Returns `true` iff only `source` is populated (no `instruction_set`, `format`,
    /// or typed operands), so the circuit can serialize in the bare-string /
    /// inline-sequence shorthand form. With the circuit `instruction_set` now optional
    /// (the layer can supply it), this also holds for on-disk gadgets whose
    /// circuit is nothing but its source.
    fn is_shorthand_compatible(&self) -> bool {
        self.instruction_set.is_none() && self.format.is_none() && self.inputs.is_empty() && self.outputs.is_empty()
    }
}

impl Serialize for CircuitSpec {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;

        /// The `source` field follows the A1 rule: a file path serializes as
        /// the bare path string, inline content with an explicit `format` as a
        /// raw string, and inline content without one as a YAML sequence.
        struct SourceField<'a> {
            source: &'a Sourced<String>,
            format: Option<&'a String>,
        }
        impl Serialize for SourceField<'_> {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                match self.source {
                    Sourced::File { path, .. } => path.serialize(serializer),
                    Sourced::Inline(text) if self.format.is_some() => text.serialize(serializer),
                    Sourced::Inline(_) => source_serde::serialize(self.source, serializer),
                }
            }
        }

        if self.is_shorthand_compatible() {
            return source_serde::serialize(&self.source, serializer);
        }

        let mut map = serializer.serialize_map(None)?;
        if let Some(instruction_set) = &self.instruction_set {
            map.serialize_entry("instruction_set", instruction_set)?;
        }
        map.serialize_entry(
            "source",
            &SourceField {
                source: &self.source,
                format: self.format.as_ref(),
            },
        )?;
        if let Some(format) = &self.format {
            map.serialize_entry("format", format)?;
        }
        if !self.inputs.is_empty() {
            map.serialize_entry("in", &self.inputs)?;
        }
        if !self.outputs.is_empty() {
            map.serialize_entry("out", &self.outputs)?;
        }
        map.end()
    }
}

/// Read the mapping form's `source` field, applying the rule stated on
/// [`CircuitSpec::format`]: a sequence is inline YAML, a string with an explicit
/// format is inline content, and a bare string is a sibling-file path.
fn source_from_value(source: serde_yaml::Value, format: Option<&str>) -> Result<Sourced<String>, String> {
    match source {
        serde_yaml::Value::Sequence(sequence) => serde_yaml::to_string(&sequence)
            .map(Sourced::inline)
            .map_err(|error| error.to_string()),
        serde_yaml::Value::String(text) if format.is_some() => Ok(Sourced::inline(text)),
        serde_yaml::Value::String(text) => Ok(Sourced::file(text)),
        _ => Err("circuit source must be a file path (string) or inline calls (sequence)".to_owned()),
    }
}

impl<'de> Deserialize<'de> for CircuitSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;

        #[derive(Deserialize)]
        struct Shim(#[serde(with = "source_serde")] Sourced<String>);

        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct CircuitMap {
            #[serde(default)]
            instruction_set: Option<String>,
            source: serde_yaml::Value,
            #[serde(default)]
            format: Option<String>,
            #[serde(rename = "in", default)]
            inputs: BTreeMap<String, String>,
            #[serde(rename = "out", default)]
            outputs: BTreeMap<String, String>,
        }

        let value = serde_yaml::Value::deserialize(deserializer)?;
        match &value {
            serde_yaml::Value::String(_) | serde_yaml::Value::Sequence(_) => {
                let shim: Shim = serde_yaml::from_value(value).map_err(D::Error::custom)?;
                Ok(Self {
                    source: shim.0,
                    ..Self::default()
                })
            }
            serde_yaml::Value::Mapping(_) => {
                let parsed: CircuitMap = serde_yaml::from_value(value).map_err(D::Error::custom)?;
                let source = source_from_value(parsed.source, parsed.format.as_deref()).map_err(D::Error::custom)?;
                Ok(Self {
                    instruction_set: parsed.instruction_set,
                    source,
                    format: parsed.format,
                    inputs: parsed.inputs,
                    outputs: parsed.outputs,
                })
            }
            _ => Err(D::Error::custom(
                "circuit source must be a string (file path), a sequence (inline calls), or a mapping",
            )),
        }
    }
}

/// A gadget document: the implemented instruction, its circuit, boundary
/// encodings, parameter bindings, and check/readout equations.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GadgetSpec {
    /// The instruction set-level instruction this gadget realizes, written
    /// `<instruction_set>#<mnemonic>`. Optional: when omitted, the layer that lists
    /// this gadget supplies the source instruction set (its own) and the mnemonic (the
    /// map key it is listed under). When present, the loader checks both
    /// agree with the layer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub implements: Option<Implements>,

    /// The realization circuit: source text plus typed operand declarations.
    /// The circuit's
    /// `instruction_set:` field names the instruction set the source calls into; encoding-map
    /// `support:` entries reference that instruction set's block types via
    /// `circuit.in.<id>` / `circuit.out.<id>`.
    pub circuit: CircuitSpec,

    /// Input encodings, positionally aligned with the implemented
    /// instruction's `in:` operand list. Empty for pure-preparation gadgets
    /// that have no input boundary.
    #[serde(rename = "in", default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<EncodingSpec>,

    /// Output encodings, positionally aligned with the implemented
    /// instruction's `out:` operand list. Empty for pure-measurement gadgets
    /// that destroy their input.
    #[serde(rename = "out", default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<EncodingSpec>,

    /// Deterministic syndrome bits. Each entry is a property-path parity
    /// equation that XORs to zero on noiseless +1-codeword execution.
    #[serde(default = "default_empty_checks", skip_serializing_if = "checks_empty")]
    pub checks: Sourced<Vec<ParityEquation>>,

    /// Terminal readouts the gadget exposes, as one positional list in
    /// declaration order: the implemented instruction's `observe` outcomes
    /// first (the observables), then its `flags:` flags (each a single
    /// parity). Entry `i` is referenced one layer up as `readouts[i]`; its
    /// role (observable vs flag) is fixed by position against the implemented
    /// instruction.
    #[serde(default = "default_empty_readouts", skip_serializing_if = "readouts_empty")]
    pub readouts: Sourced<ReadoutsList>,

    /// Additional output logical-sign corrections, keyed by `out[entry].x[index]`
    /// or `out[entry].z[index]`. Keys are parsed as references, so a misspelled
    /// target fails to load; which targets are legal is an audit question.
    /// Values XOR circuit readouts, literal bits,
    /// or readout aliases resolving to those terms. Encoding signs are not
    /// delta inputs. Audit checks this rule; loading preserves drafts.
    /// Omitted entries apply no additional correction.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty", with = "crate::parity::frames")]
    pub frames: BTreeMap<crate::Reference, ParityEquation>,

    /// Maps instruction parameter names to circuit-source parameter names.
    ///
    /// The supplied value for each instruction parameter is forwarded to its
    /// source parameter. Include only forwarded parameters. The instruction
    /// declares their types; this map handles both literal and bit arguments.
    ///
    /// Values use `circuit.source.<name>` on disk. In memory, the prefix is
    /// stripped and only the source parameter name is stored.
    #[serde(
        default,
        with = "parameter_bindings_serde",
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    pub parameter_bindings: BTreeMap<String, String>,
    /// Free-form, qodec-opaque annotations (see [`crate::Metadata`]).
    #[serde(default, skip_serializing_if = "crate::Metadata::is_empty")]
    pub metadata: crate::Metadata,
}

// ── Defaults / emptiness predicates for #[serde(skip_serializing_if)] ─────

fn default_empty_checks() -> Sourced<Vec<ParityEquation>> {
    Sourced::inline(Vec::new())
}

fn default_empty_readouts() -> Sourced<ReadoutsList> {
    Sourced::inline(ReadoutsList::new())
}

fn checks_empty(checks: &Sourced<Vec<ParityEquation>>) -> bool {
    checks.is_inline_empty(Vec::is_empty)
}

fn readouts_empty(readouts: &Sourced<ReadoutsList>) -> bool {
    readouts.is_inline_empty(ReadoutsList::is_empty)
}

/// Values are `circuit.source.<name>` on disk and the bare source parameter name
/// in memory, so the prefix is added on the way out and required on the way in.
mod parameter_bindings_serde {
    use serde::de::Error as _;
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;

    pub fn serialize<S>(map: &BTreeMap<String, String>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let prefixed: BTreeMap<&str, String> = map
            .iter()
            .map(|(implements, source_name)| (implements.as_str(), format!("circuit.source.{source_name}")))
            .collect();
        prefixed.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<BTreeMap<String, String>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw: BTreeMap<String, String> = BTreeMap::deserialize(deserializer)?;
        raw.into_iter()
            .map(|(implements, reference)| {
                let source_name = reference.strip_prefix("circuit.source.").ok_or_else(|| {
                    D::Error::custom(format!(
                        "parameter_bindings: value '{reference}' for implements parameter \
                         '{implements}' must be a `circuit.source.<name>` reference"
                    ))
                })?;
                Ok((implements, source_name.to_owned()))
            })
            .collect()
    }
}

// ── source: Sourced<String> serde adapter ────────────────────────────────────

mod source_serde {
    use crate::Sourced;
    use serde::de::Error as _;
    use serde::{Deserialize, Serialize};
    use serde_yaml::Sequence;

    pub fn serialize<S>(source: &Sourced<String>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match source {
            Sourced::Inline(text) => serde_yaml::from_str::<Sequence>(text)
                .map_err(|error| serde::ser::Error::custom(format!("inline source must be a YAML sequence: {error}")))?
                .serialize(serializer),
            Sourced::File { path, .. } => path.serialize(serializer),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Sourced<String>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Path(String),
            Inline(Sequence),
        }
        match Repr::deserialize(deserializer)? {
            Repr::Path(path) => Ok(Sourced::file(path)),
            Repr::Inline(sequence) => serde_yaml::to_string(&sequence)
                .map(Sourced::inline)
                .map_err(|error| D::Error::custom(format!("inline source sequence should serialize: {error}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn circuit_instruction_set_key_round_trips() {
        let circuit: super::CircuitSpec =
            serde_yaml::from_str("instruction_set: physical.isa.yaml\nsource: ./body.stim\n").unwrap();
        assert_eq!(circuit.instruction_set.as_deref(), Some("physical.isa.yaml"));
        let serialized = serde_yaml::to_value(&circuit).unwrap();
        let keys: std::collections::BTreeSet<_> = serialized
            .as_mapping()
            .unwrap()
            .keys()
            .map(|key| key.as_str().unwrap())
            .collect();
        assert_eq!(keys, std::collections::BTreeSet::from(["instruction_set", "source"]));
        assert_eq!(
            serde_yaml::from_value::<super::CircuitSpec>(serialized).unwrap(),
            circuit
        );
    }

    #[test]
    fn circuit_rejects_unknown_isa_key() {
        for source in [
            "isa: old.yaml\nsource: ./body.stim\n",
            "instruction_set: physical.yaml\nisa: old.yaml\nsource: ./body.stim\n",
        ] {
            let error = serde_yaml::from_str::<super::CircuitSpec>(source).unwrap_err();
            assert!(error.to_string().contains("unknown field `isa`"), "{error}");
        }
    }

    #[test]
    fn implements_separates_on_the_last_hash() {
        // An instruction set path may contain `#`; a mnemonic is an identifier and cannot.
        let parsed: super::Implements = serde_yaml::from_str("we#ird.isa.yaml#cx").expect("parses");
        assert_eq!(parsed.instruction_set, "we#ird.isa.yaml");
        assert_eq!(parsed.mnemonic, "cx");
    }

    use super::GadgetSpec;

    #[test]
    fn parity_literal_bits_round_trip_in_every_equation_role() {
        let source = "circuit: []\nchecks: [[1, 0, 'circuit.readouts[0]']]\nreadouts: [{reject: [1]}]\nframes: {'out[0].z[0]': [1, 'circuit.readouts[0]']}\n";
        let parsed: GadgetSpec = serde_yaml::from_str(source).expect("literal bits should parse");
        let expected: serde_yaml::Value = serde_yaml::from_str(source).unwrap();
        assert_eq!(serde_yaml::to_value(&parsed).unwrap(), expected);
        round_trip(source);
        for invalid in ["2", "-1", "1.0", "true", "false", "'1'", "null"] {
            let source = format!("circuit: []\nchecks: [[{invalid}]]\n");
            assert!(
                serde_yaml::from_str::<GadgetSpec>(&source).is_err(),
                "accepted {invalid}"
            );
        }
    }

    #[test]
    fn sparse_frames_round_trip_without_filling_omitted_entries() {
        let source = "circuit: []\nframes:\n  'out[0].z[0]': ['circuit.readouts[6]', 'circuit.readouts[8]']\n  'out[0].x[0]': []\n";
        let parsed: GadgetSpec = serde_yaml::from_str(source).expect("frames should parse");
        let encoded = serde_yaml::to_value(&parsed).unwrap();
        let original: serde_yaml::Value = serde_yaml::from_str(source).unwrap();
        assert_eq!(encoded["frames"], original["frames"]);
        assert_eq!(serde_yaml::from_value::<GadgetSpec>(encoded).unwrap(), parsed);
        let empty: GadgetSpec = serde_yaml::from_str("circuit: []\nframes: {}\n").unwrap();
        assert!(serde_yaml::to_value(empty).unwrap().get("frames").is_none());
    }

    fn round_trip(yaml: &str) {
        let parsed: GadgetSpec = serde_yaml::from_str(yaml).expect("gadget should parse");
        let reserialized = serde_yaml::to_string(&parsed).expect("gadget should serialize");
        let reparsed: GadgetSpec = serde_yaml::from_str(&reserialized).expect("reserialized gadget should re-parse");
        assert_eq!(parsed, reparsed);
    }

    #[test]
    fn minimal_v3_5_gadget() {
        round_trip(
            r#"
implements: ./test.isa.yaml#idle
circuit:
  instruction_set: ./stim.isa.yaml
  source: ./idle.stim
  in: {"0": qubit, "1": qubit, "2": qubit, "3": qubit}
  out: {"0": qubit, "1": qubit, "2": qubit, "3": qubit}
in:
- block: [0, 1, 2, 3]
out:
- block: [0, 1, 2, 3]
checks:
- ["circuit.readouts[0]", "in[0].stabilizers[0]"]
- ["circuit.readouts[1]", "out[0].stabilizers[0]"]
"#,
        );
    }

    #[test]
    fn gadget_with_readouts() {
        round_trip(
            r#"
implements: ./test.isa.yaml#measure_z
circuit:
  instruction_set: ./stim.isa.yaml
  source: ./measure_z.stim
  in: {"0": qubit, "1": qubit, "2": qubit, "3": qubit}
in:
- block: [0, 1, 2, 3]
readouts:
- ["in[0].z[0]", "circuit.readouts[0]", "circuit.readouts[1]"]
"#,
        );
    }

    #[test]
    fn gadget_with_typed_block_support() {
        round_trip(
            "\
implements: ./test.isa.yaml#controlled_x_all
circuit:
  instruction_set: ./c4c6.isa.yaml
  source:
  - controlled_x_all: [0, 3]
  - controlled_x_all: [1, 4]
  - controlled_x_all: [2, 5]
  in: {\"0\": block, \"1\": block, \"2\": block, \"3\": block, \"4\": block, \"5\": block}
  out: {\"0\": block, \"1\": block, \"2\": block, \"3\": block, \"4\": block, \"5\": block}
in:
- block: [0, 1, 2]
- block: [3, 4, 5]
out:
- block: [0, 1, 2]
- block: [3, 4, 5]
",
        );
    }

    #[test]
    fn parameterized_gadget_binds_body_source_immediate() {
        let yaml = r#"
implements: ./test.isa.yaml#rotate_z
circuit:
  instruction_set: ./stim+rz.isa.yaml
  source:
  - rotate_z: [0, theta: theta]
  in: {"0": qubit}
  out: {"0": qubit}
in:
- block: [0, 1, 2]
out:
- block: [0, 1, 2]
parameter_bindings:
  theta: circuit.source.theta
"#;
        let parsed: GadgetSpec = serde_yaml::from_str(yaml).expect("gadget should parse");
        assert_eq!(
            parsed.parameter_bindings.get("theta").map(String::as_str),
            Some("theta")
        );
        let reserialized = serde_yaml::to_string(&parsed).expect("gadget should serialize");
        assert!(reserialized.contains("theta: circuit.source.theta"));
        let reparsed: GadgetSpec = serde_yaml::from_str(&reserialized).expect("reserialized gadget should re-parse");
        assert_eq!(parsed, reparsed);
    }

    #[test]
    fn inline_stim_body_is_content_not_path() {
        use crate::Sourced;

        let yaml = "\
implements: ./test.isa.yaml#idle
circuit:
  instruction_set: ./stim.isa.yaml
  format: stim
  source: |
    R 3 4
    CX 0 3 1 3
    M 3 4
in:
- block: [0, 1, 2]
";
        let parsed: GadgetSpec = serde_yaml::from_str(yaml).expect("gadget should parse");
        let circuit = &parsed.circuit;
        assert_eq!(circuit.format.as_deref(), Some("stim"));
        match &circuit.source {
            Sourced::Inline(text) => {
                assert!(text.contains("CX 0 3 1 3"), "inline content preserved: {text:?}");
            }
            Sourced::File { .. } => panic!("inline stim source must be inline content, not a path"),
        }
        round_trip(yaml);
    }

    #[test]
    fn string_source_without_format_is_a_path() {
        use crate::Sourced;

        let yaml = "\
implements: ./test.isa.yaml#idle
circuit:
  instruction_set: ./stim.isa.yaml
  source: ./idle.stim
in:
- block: [0, 1, 2]
";
        let parsed: GadgetSpec = serde_yaml::from_str(yaml).expect("gadget should parse");
        match &parsed.circuit.source {
            Sourced::File { path, .. } => assert_eq!(path, "./idle.stim"),
            Sourced::Inline(_) => panic!("string source without format must be a file path"),
        }
    }

    #[test]
    fn unknown_format_preserves_source_text() {
        let yaml = "\
implements: ./test.isa.yaml#idle
circuit:
  instruction_set: ./stim.isa.yaml
  format: bogus
  source: |
    R 0
in:
- block: [0, 1, 2]
";
        let gadget = serde_yaml::from_str::<GadgetSpec>(yaml).unwrap();
        let saved = serde_yaml::to_string(&gadget).unwrap();
        assert_eq!(serde_yaml::from_str::<GadgetSpec>(&saved).unwrap(), gadget);
    }
}
