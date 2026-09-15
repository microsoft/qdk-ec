//! Exact navigation through the resolved model.

mod path;
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

/// One occurrence in a resolved model, addressed relative to its owning qodec.
///
/// Equality and hashing use owner identity and the canonical path, not contents.
/// Source information does not participate in equality. Nodes borrow the model;
/// mutation requires releasing that borrow. Obtain nodes with [`Qodec::resolve`].
#[derive(Clone)]
pub struct Node<'model> {
    owner: &'model Qodec,
    segments: ModelPath,
    path: String,
    value: Value<'model>,
}

impl Qodec {
    /// Resolve an exact model path. The empty string selects this qodec.
    ///
    /// Fields use dots, mapping keys JSON-quoted brackets, and sequence indexes
    /// nonnegative brackets. Circuit parsing and analysis are never implicit.
    ///
    /// # Errors
    /// Returns [`PathError`] for invalid syntax or a missing target.
    pub fn resolve(&self, path: &str) -> Result<Node<'_>, PathError> {
        Node {
            owner: self,
            segments: ModelPath::default(),
            path: String::new(),
            value: Value::Qodec(self),
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
    /// Canonical path relative to the owning qodec. The root has an empty path.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Source of this exact field, if retained and still reliable.
    #[must_use]
    pub fn source_location(&self) -> Option<&SourceLocation> {
        self.owner.locations.get(&self.path)
    }

    /// Whether this occurrence holds an absent optional value or JSON null.
    #[must_use]
    pub fn is_none(&self) -> bool {
        matches!(self.value, Value::None)
    }

    /// Resolve a path relative to this occurrence, returning a root-relative node.
    ///
    /// # Errors
    /// Returns [`PathError`] for invalid syntax or a missing target.
    pub fn resolve(&self, path: &str) -> Result<Self, PathError> {
        let relative = ModelPath::parse(path)?;
        let mut node = self.clone();
        for segment in relative.0 {
            node.segments = node.segments.child(segment.clone());
            node.path = node.segments.to_string();
            node.value = node
                .value
                .child(&segment)
                .ok_or_else(|| PathError::Missing(node.path.clone()))?;
        }
        Ok(node)
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
        }
    }
}

impl PartialEq for Node<'_> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.owner, other.owner) && self.segments == other.segments
    }
}
impl Eq for Node<'_> {}
impl Hash for Node<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::ptr::from_ref(self.owner).hash(state);
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
            .field("type", &self.value.kind())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use crate::{PathError, Qodec};

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
