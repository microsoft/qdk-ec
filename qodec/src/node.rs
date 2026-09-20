//! Exact navigation through the resolved model.

pub(crate) mod path;
mod value;
use crate as model;
use value::Value;
pub(crate) mod source;

use std::collections::BTreeMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use crate::{
    ActionStep, Block, BlockOperand, Circuit, Code, Condition, Encoding, Gadget, Instruction, InstructionSet, Layer,
    Parameter, ParameterKind, Qodec, Readout, Reference,
};
pub use path::PathError;
use path::{ModelPath, Segment};

/// A point in the actual file read by the loader, not an internal bundle key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceLocation {
    pub(crate) path: PathBuf,
    pub(crate) line: usize,
}

impl SourceLocation {
    /// The actual source file, resolved when loading.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// One-based line in the loaded file revision.
    #[must_use]
    pub fn line(&self) -> usize {
        self.line
    }
}

/// One occurrence or selection, relative to its owning qodec or standalone gadget.
///
/// Equality and hashing use owner identity and the canonical path, not contents.
/// Source information does not participate in equality. Nodes borrow the model;
/// mutation requires releasing that borrow. Obtain nodes with [`Qodec::resolve`]
/// or [`Gadget::resolve`].
#[derive(Clone)]
pub struct Node<'model> {
    owner: Owner<'model>,
    segments: ModelPath,
    path: String,
    value: Value<'model>,
    selected: Option<Vec<Self>>,
}

#[derive(Clone, Copy)]
enum Owner<'model> {
    Qodec(&'model Qodec),
    Gadget(&'model Gadget),
}

impl Owner<'_> {
    fn identity(self) -> (u8, *const ()) {
        match self {
            Self::Qodec(value) => (0, std::ptr::from_ref(value).cast()),
            Self::Gadget(value) => (1, std::ptr::from_ref(value).cast()),
        }
    }
}

impl Qodec {
    /// Resolve an exact model path. The empty string selects this qodec.
    ///
    /// Fields use dots, mapping keys JSON-quoted brackets, and sequence indexes
    /// nonnegative brackets. Circuit parsing and analysis are never implicit.
    ///
    /// # Errors
    /// Returns [`PathError`] for invalid syntax or a missing target.
    pub fn resolve<Relative: TryInto<Reference>>(&self, path: Relative) -> Result<Node<'_>, PathError>
    where
        Relative::Error: fmt::Display,
    {
        Node {
            owner: Owner::Qodec(self),
            segments: ModelPath::default(),
            path: String::new(),
            value: Value::Qodec(self),
            selected: None,
        }
        .resolve(path)
    }
}

impl Gadget {
    /// Resolve a reference relative to this standalone gadget, without parsing circuits.
    /// Nodes use this gadget's identity and have no loaded source locations.
    ///
    /// # Errors
    /// Returns [`PathError`] for invalid syntax or any missing selected target.
    pub fn resolve<Relative: TryInto<Reference>>(&self, path: Relative) -> Result<Node<'_>, PathError>
    where
        Relative::Error: fmt::Display,
    {
        Node {
            owner: Owner::Gadget(self),
            segments: ModelPath::default(),
            path: String::new(),
            value: Value::Gadget(self),
            selected: None,
        }
        .resolve(path)
    }
}

macro_rules! accessor {
	($name:ident, $variant:ident, $result:ty) => {
		#[doc = concat!("Return the selected ", stringify!($variant), ", or `None` for another type. No conversion is performed.")]
		#[must_use]
		pub fn $name(&self) -> Option<$result> {
			if let Value::$variant(value) = &self.value { Some(*value) } else { None }
		}
	};
}

impl<'model> Node<'model> {
    /// Canonical path relative to the owner. The root has an empty path.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Source of this exact field, if retained and still reliable.
    #[must_use]
    pub fn source_location(&self) -> Option<&SourceLocation> {
        match self.owner {
            Owner::Qodec(owner) => owner.locations.get(&self.path),
            Owner::Gadget(_) => None,
        }
    }

    /// Whether this occurrence holds an absent optional value or JSON null.
    #[must_use]
    pub fn is_none(&self) -> bool {
        self.selected.is_none() && matches!(self.value, Value::None)
    }

    /// Resolve a path relative to this occurrence, returning a root-relative node.
    ///
    /// # Errors
    /// Returns [`PathError`] for invalid syntax or a missing target.
    pub fn resolve<Relative: TryInto<Reference>>(&self, path: Relative) -> Result<Self, PathError>
    where
        Relative::Error: fmt::Display,
    {
        let relative = path.try_into().map_err(|error| PathError::Syntax(error.to_string()))?;
        let mut node = self.clone();
        for segment in relative.parsed.0 {
            node = node.select(segment)?;
        }
        Ok(node)
    }

    fn select(&self, segment: Segment) -> Result<Self, PathError> {
        let missing = || PathError::Missing(self.segments.child(segment.clone()).to_string());
        if matches!(segment, Segment::Slice { .. } | Segment::Union(_)) {
            let selected = path::indices(&segment)
                .map(|index| self.select(Segment::Index(index)))
                .collect::<Result<Vec<_>, _>>()?;
            let mut node = self.child(segment, Value::None);
            node.selected = Some(selected);
            return Ok(node);
        }
        if let Some(selected) = &self.selected {
            return match segment {
                Segment::Index(index) => selected.get(index).cloned().ok_or_else(missing),
                _ => Err(missing()),
            };
        }
        let value = self.value.child(&segment).ok_or_else(missing)?;
        Ok(self.child(segment, value))
    }

    accessor!(as_qodec, Qodec, &'model Qodec);
    accessor!(as_layer, Layer, &'model Layer);
    accessor!(as_instruction_set, InstructionSet, &'model InstructionSet);
    accessor!(as_instruction, Instruction, &'model Instruction);
    accessor!(as_code, Code, &'model Code);
    accessor!(as_gadget, Gadget, &'model Gadget);
    accessor!(as_circuit, Circuit, &'model Circuit);
    accessor!(as_encoding, Encoding, &'model Encoding);
    accessor!(as_block, Block, &'model Block);
    accessor!(as_block_operand, BlockOperand, &'model BlockOperand);
    accessor!(as_parameter, Parameter, &'model Parameter);
    accessor!(as_parameter_kind, ParameterKind, ParameterKind);
    accessor!(as_action, Action, &'model ActionStep);
    accessor!(as_condition, Condition, &'model Condition);
    accessor!(as_readout, Readout, &'model Readout);
    accessor!(as_reference, Reference, &'model Reference);
    accessor!(as_str, Str, &'model str);
    accessor!(as_int, Int, i128);
    accessor!(as_float, Float, f64);
    accessor!(as_bool, Bool, bool);

    /// All entries of the selected sequence, in order, or `None` for another type.
    #[must_use]
    pub fn as_sequence(&self) -> Option<Vec<Self>> {
        if let Some(selected) = &self.selected {
            return Some(selected.clone());
        }
        let Value::Sequence(values) = &self.value else {
            return None;
        };
        Some(
            values
                .iter()
                .enumerate()
                .map(|(index, value)| self.child(Segment::Index(index), value))
                .collect(),
        )
    }

    /// All entries of the selected mapping keyed by their literal strings.
    ///
    /// The mapping is total for the selected collection; returns `None` for another type.
    #[must_use]
    pub fn as_mapping(&self) -> Option<BTreeMap<&'model str, Self>> {
        let Value::Mapping(values) = &self.value else {
            return None;
        };
        Some(
            values
                .entries()
                .into_iter()
                .map(|(key, value)| (key, self.child(Segment::Key(key.to_owned()), value)))
                .collect(),
        )
    }

    fn child(&self, segment: Segment, value: Value<'model>) -> Self {
        let segments = self.segments.child(segment);
        Self {
            owner: self.owner,
            path: segments.to_string(),
            segments,
            value,
            selected: None,
        }
    }
}

impl PartialEq for Node<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.owner.identity() == other.owner.identity() && self.segments == other.segments
    }
}
impl Eq for Node<'_> {}
impl Hash for Node<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.owner.identity().hash(state);
        self.segments.hash(state);
    }
}
impl fmt::Display for Node<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.path)
    }
}
impl fmt::Debug for Node<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Node")
            .field("path", &self.path)
            .field(
                "type",
                &if self.selected.is_some() {
                    "list"
                } else {
                    self.value.kind()
                },
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use crate::{PathError, Qodec, Reference};

    #[test]
    fn references_resolve_from_protocol_nodes_and_standalone_gadgets() {
        let model = Qodec::load("examples/repetition3/repetition3.qodec.yaml").unwrap();
        let node = model.resolve("layers[0].gadgets[\"measure_z\"]").unwrap();
        let reference = Reference::parse("in[0].stabilizers[1]").unwrap();
        let resolved = node.resolve(&reference).unwrap();
        assert_eq!(
            resolved,
            model
                .resolve(Reference::parse(&format!("{}.{}", node.path(), reference)).unwrap())
                .unwrap()
        );
        let gadget = node.as_gadget().unwrap();
        let standalone = gadget.resolve(&reference).unwrap();
        assert_eq!(standalone.as_str(), resolved.as_str());
        assert_eq!(standalone.path(), reference.path());
        assert_ne!(standalone, resolved);
        assert!(standalone.source_location().is_none());
        assert!(gadget.resolve("circuit.readouts[0]").is_err());
    }

    #[test]
    fn selections_preserve_member_paths_order_duplicates_and_fail_atomically() {
        let model = Qodec::load("examples/repetition3/repetition3.qodec.yaml").unwrap();
        let node = model
            .resolve("layers[0].gadgets[\"measure_z\"].in[0].stabilizers")
            .unwrap();
        let selected = node.resolve("[1, 0,1]").unwrap();
        assert!(!selected.is_none());
        assert!(selected.as_str().is_none());
        let members = selected.as_sequence().unwrap();
        assert_eq!(
            members,
            vec![
                node.resolve("[1]").unwrap(),
                node.resolve("[0]").unwrap(),
                node.resolve("[1]").unwrap()
            ]
        );
        assert_eq!(selected.resolve("[0]").unwrap(), members[0]);
        assert_eq!(
            node.resolve("[0:2]").unwrap().as_sequence().unwrap(),
            node.as_sequence().unwrap()
        );
        assert!(node.resolve("[0,999]").is_err());
        assert!(node.resolve("[0:3]").is_err());
        assert!(selected.resolve("name").is_err());
        assert!(node.resolve("[0:1]").unwrap().as_sequence().is_some());
    }

    #[test]
    fn encoding_paths_use_parity_spelling() {
        let model = Qodec::load("examples/repetition3/repetition3.qodec.yaml").unwrap();
        let gadget = model.resolve("layers[0].gadgets[\"measure_z\"]").unwrap();
        assert_eq!(
            gadget.resolve("in[0].stabilizers[1]").unwrap().as_str(),
            gadget.resolve("in[0].code.stabilizers[1]").unwrap().as_str()
        );
        assert!(gadget.resolve("in[0].code").unwrap().as_code().is_some());
        assert!(gadget.resolve("implements.in[0]").unwrap().as_block_operand().is_some());
        assert!(gadget.resolve("out").unwrap().as_sequence().is_some());
        for old_path in ["inputs", "outputs", "implements.inputs", "implements.outputs"] {
            assert!(matches!(gadget.resolve(old_path), Err(PathError::Missing(_))));
        }
    }

    #[test]
    fn navigates_stored_model_without_parsing_circuits() {
        let model = Qodec::load("examples/repetition3/repetition3.qodec.yaml").unwrap();
        let root = model.resolve("").unwrap();
        assert!(std::ptr::eq(root.as_qodec().unwrap(), &raw const model));
        let layers = root.resolve("layers").unwrap().as_sequence().unwrap();
        assert_eq!(layers.len(), model.layers().len());
        let gadgets = layers[0].resolve("gadgets").unwrap().as_mapping().unwrap();
        for (name, node) in gadgets {
            assert_eq!(node.as_gadget().unwrap().implements.mnemonic, name);
            assert!(node.resolve("circuit.source").unwrap().as_str().is_some());
            assert!(matches!(node.resolve("circuit.calls"), Err(PathError::Missing(_))));
            for readout in node.resolve("readouts").unwrap().as_sequence().unwrap() {
                for term in readout.resolve("equation").unwrap().as_sequence().unwrap() {
                    assert!(term.as_reference().is_some());
                    assert_eq!(root.resolve(term.path()).unwrap(), term);
                }
            }
        }
        assert_eq!(
            model.resolve("layers[00]").unwrap(),
            model.resolve("layers[0]").unwrap()
        );
        assert!(model.resolve("layers[0]").unwrap().as_gadget().is_none());
    }

    #[test]
    fn scalar_types_and_occurrence_identity_are_exact() {
        let mut model = Qodec::new(None, None, vec![]);
        model.metadata_mut().insert(
            "data".into(),
            serde_json::json!([true, 1, 1.5, null, "text", {"a.b": 3}]),
        );
        let node = model.resolve(r#"metadata["data"]"#).unwrap();
        let values = node.as_sequence().unwrap();
        assert_eq!(values[0].as_bool(), Some(true));
        assert_eq!(values[0].as_int(), None);
        assert_eq!(values[1].as_int(), Some(1));
        assert_eq!(values[2].as_float(), Some(1.5));
        assert!(values[3].is_none());
        assert_eq!(values[4].as_str(), Some("text"));
        assert_eq!(values[5].resolve(r#"["a.b"]"#).unwrap().as_int(), Some(3));
        let other = Qodec::new(None, None, vec![]);
        assert_ne!(model.resolve("").unwrap(), other.resolve("").unwrap());
    }
}
