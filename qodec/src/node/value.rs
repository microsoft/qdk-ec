//! Borrowed model values shared by native navigation adapters.

use super::model::{
    Action, ActionStep, Block, BlockOperand, Circuit, Code, Condition, Encoding, Gadget, Instruction, InstructionSet,
    Layer, Parameter, ParameterKind, Qodec, Readout, Reference, Scalar,
};
use super::model::{Metadata, Observable, ParityEquation, ParityTerm, PauliString};
use super::path::Segment;
use std::collections::BTreeMap;

#[cfg(test)]
thread_local! {
    static JSON_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[derive(Clone, Copy)]
pub(super) enum Value<'model> {
    Qodec(&'model Qodec),
    Layer(&'model Layer),
    InstructionSet(&'model InstructionSet),
    Instruction(&'model Instruction),
    Code(&'model Code),
    Gadget(&'model Gadget),
    Circuit(&'model Circuit),
    Encoding(&'model Encoding),
    Block(&'model Block),
    BlockOperand(&'model BlockOperand),
    Parameter(&'model Parameter),
    ParameterKind(ParameterKind),
    Action(&'model ActionStep),
    Condition(&'model Condition),
    Readout(&'model Readout),
    ReadoutSpec {
        spec: &'model super::model::ReadoutSpec,
        position: usize,
        observe_count: usize,
    },
    Reference(&'model Reference),
    Sequence(Sequence<'model>),
    Mapping(Mapping<'model>),
    Str(&'model str),
    Int(i128),
    Float(f64),
    Bool(bool),
    None,
}

macro_rules! sequence_views {
    ($($variant:ident($item:ty) => $wrap:expr),* $(,)?) => {
        #[derive(Clone, Copy)]
        pub(super) enum Sequence<'model> {
            $($variant(&'model [$item]),)*
            #[allow(dead_code, reason = "This shared view is constructed by the Python navigation adapter.")]
            ReadoutSpecs(&'model [super::model::ReadoutSpec], usize),
        }

        impl<'model> Sequence<'model> {
            pub(super) fn len(self) -> usize {
                match self {
                    $(Self::$variant(values) => values.len(),)*
                    Self::ReadoutSpecs(values, _) => values.len(),
                }
            }

            pub(super) fn get(self, index: usize) -> Option<Value<'model>> {
                match self {
                    $(Self::$variant(values) => values.get(index).map($wrap),)*
                    Self::ReadoutSpecs(values, observe_count) => values.get(index)
                        .map(|spec| Value::ReadoutSpec { spec, position: index, observe_count }),
                }
            }

            pub(super) fn iter(self) -> impl Iterator<Item = Value<'model>> {
                (0..self.len()).map(move |index| self.get(index).expect("index within borrowed slice"))
            }
        }
    };
}

sequence_views! {
    Layers(Layer) => Value::Layer,
    Blocks(Block) => Value::Block,
    Operands(BlockOperand) => Value::BlockOperand,
    Parameters(Parameter) => Value::Parameter,
    Actions(ActionStep) => Value::Action,
    Encodings(Encoding) => Value::Encoding,
    Checks(ParityEquation) => |terms| Value::Sequence(Sequence::Terms(terms)),
    Terms(ParityTerm) => Value::parity_term,
    Readouts(Readout) => Value::Readout,
    Strings(String) => |value| Value::Str(value),
    Paulis(PauliString) => |value| Value::Str(&value.0),
    Observables(Observable) => |value| Value::Str(&value.pauli.0),
    Json(serde_json::Value) => Value::json,
}

#[derive(Clone, Copy)]
pub(super) enum Mapping<'model> {
    Gadgets(&'model BTreeMap<String, Gadget>),
    Instructions(&'model [Instruction]),
    Metadata(&'model Metadata),
    Frames(&'model BTreeMap<Reference, ParityEquation>),
    ParameterBindings(&'model BTreeMap<String, String>),
    Generators(&'model BTreeMap<PauliString, PauliString>),
    InstructionSets(&'model [Layer]),
    Codes(&'model [Layer]),
    LayerCodes(&'model BTreeMap<String, std::sync::Arc<Code>>),
}

impl<'model> Mapping<'model> {
    fn codes(layers: &'model [Layer]) -> impl Iterator<Item = &'model Code> {
        layers.iter().flat_map(|layer| {
            layer
                .codes
                .iter()
                .filter(|(block, _)| {
                    layer
                        .instruction_set
                        .blocks
                        .iter()
                        .any(|declared| &declared.name == *block)
                })
                .map(|(_, code)| code.as_ref())
                .chain(
                    layer
                        .gadgets
                        .values()
                        .flat_map(|gadget| gadget.inputs.iter().chain(&gadget.outputs))
                        .map(|encoding| encoding.code.as_ref()),
                )
        })
    }

    pub(super) fn get(self, key: &str) -> Option<Value<'model>> {
        match self {
            Self::Gadgets(values) => values.get(key).map(Value::Gadget),
            Self::Instructions(values) => values
                .iter()
                .find(|value| value.mnemonic == key)
                .map(Value::Instruction),
            Self::Metadata(values) => values.get(key).map(Value::json),
            Self::Frames(values) => values
                .iter()
                .find(|(name, _)| name.path() == key)
                .map(|(_, terms)| Value::Sequence(Sequence::Terms(terms))),
            Self::ParameterBindings(values) => values.get(key).map(|value| Value::Str(value)),
            Self::Generators(values) => values
                .iter()
                .find(|(name, _)| name.0 == key)
                .map(|(_, value)| Value::Str(&value.0)),
            Self::InstructionSets(layers) => layers
                .iter()
                .rev()
                .find(|layer| layer.instruction_set.name == key)
                .map(|layer| Value::InstructionSet(&layer.instruction_set)),
            Self::Codes(layers) => Self::codes(layers).find(|code| code.name == key).map(Value::Code),
            Self::LayerCodes(values) => values.get(key).map(|code| Value::Code(code)),
        }
    }

    pub(super) fn entries(self) -> BTreeMap<&'model str, Value<'model>> {
        match self {
            Self::Gadgets(values) => values
                .iter()
                .map(|(key, value)| (key.as_str(), Value::Gadget(value)))
                .collect(),
            Self::Instructions(values) => values
                .iter()
                .map(|value| (value.mnemonic.as_str(), Value::Instruction(value)))
                .collect(),
            Self::Metadata(values) => values
                .iter()
                .map(|(key, value)| (key.as_str(), Value::json(value)))
                .collect(),
            Self::Frames(values) => values
                .iter()
                .map(|(key, terms)| (key.path(), Value::Sequence(Sequence::Terms(terms))))
                .collect(),
            Self::ParameterBindings(values) => values
                .iter()
                .map(|(key, value)| (key.as_str(), Value::Str(value)))
                .collect(),
            Self::Generators(values) => values
                .iter()
                .map(|(key, value)| (key.0.as_str(), Value::Str(&value.0)))
                .collect(),
            Self::InstructionSets(layers) => layers
                .iter()
                .map(|layer| {
                    (
                        layer.instruction_set.name.as_str(),
                        Value::InstructionSet(&layer.instruction_set),
                    )
                })
                .collect(),
            Self::Codes(layers) => {
                let mut values = BTreeMap::new();
                for code in Self::codes(layers) {
                    values.entry(code.name.as_str()).or_insert(Value::Code(code));
                }
                values
            }
            Self::LayerCodes(values) => values
                .iter()
                .map(|(block, code)| (block.as_str(), Value::Code(code)))
                .collect(),
        }
    }
}

impl<'model> Value<'model> {
    fn parity_term(term: &'model super::model::ParityTerm) -> Self {
        match term {
            super::model::ParityTerm::Reference(reference) => Self::Reference(reference),
            super::model::ParityTerm::Bit(value) => Self::Int(i128::from(*value)),
        }
    }

    fn strings(values: &'model [String]) -> Self {
        Self::Sequence(Sequence::Strings(values))
    }
    fn paulis(values: &'model [super::model::PauliString]) -> Self {
        Self::Sequence(Sequence::Paulis(values))
    }
    fn optional(value: Option<&'model str>) -> Self {
        value.map_or(Self::None, Self::Str)
    }
    fn metadata(values: &'model super::model::Metadata) -> Self {
        Self::Mapping(Mapping::Metadata(values))
    }
    fn json(value: &'model serde_json::Value) -> Self {
        #[cfg(test)]
        JSON_VISITS.with(|visits| visits.set(visits.get() + 1));
        match value {
            serde_json::Value::Null => Self::None,
            serde_json::Value::Bool(value) => Self::Bool(*value),
            serde_json::Value::String(value) => Self::Str(value),
            serde_json::Value::Array(values) => Self::Sequence(Sequence::Json(values)),
            serde_json::Value::Object(values) => Self::metadata(values),
            serde_json::Value::Number(value) => value
                .as_i64()
                .map(|number| Self::Int(number.into()))
                .or_else(|| value.as_u64().map(|number| Self::Int(number.into())))
                .unwrap_or_else(|| Self::Float(value.as_f64().unwrap_or_default())),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn field(&self, field: &str) -> Option<Self> {
        Some(match (self, field) {
            (Self::Qodec(value), "layers") => Self::Sequence(Sequence::Layers(value.layers())),
            (Self::Qodec(value), "name") => Self::Str(value.name().unwrap_or_default()),
            (Self::Qodec(value), "description") => Self::Str(value.description().unwrap_or_default()),
            (Self::Qodec(value), "schema_version") => value
                .schema_version()
                .map_or(Self::None, |version| Self::Int(version.into())),
            (Self::Qodec(value), "manifest_filename") => Self::Str(value.manifest_filename()),
            (Self::Qodec(value), "metadata") => Self::metadata(value.metadata()),
            (Self::Qodec(value), "instruction_sets") => Self::Mapping(Mapping::InstructionSets(value.layers())),
            (Self::Qodec(value), "codes") => Self::Mapping(Mapping::Codes(value.layers())),
            (Self::Layer(value), "instruction_set") => Self::InstructionSet(&value.instruction_set),
            (Self::Layer(value), "codes") => Self::Mapping(Mapping::LayerCodes(&value.codes)),
            (Self::Layer(value), "gadgets") => Self::Mapping(Mapping::Gadgets(&value.gadgets)),
            (Self::InstructionSet(value), "name") => Self::Str(&value.name),
            (Self::InstructionSet(value), "description") => Self::Str(&value.description),
            (Self::InstructionSet(value), "metadata") => Self::metadata(&value.metadata),
            (Self::InstructionSet(value), "blocks") => Self::Sequence(Sequence::Blocks(&value.blocks)),
            (Self::InstructionSet(value), "instructions") => Self::Mapping(Mapping::Instructions(&value.instructions)),
            (Self::Instruction(value), "mnemonic") => Self::Str(&value.mnemonic),
            (Self::Instruction(value), "description") => Self::Str(&value.description),
            (Self::Instruction(value), "in") => Self::Sequence(Sequence::Operands(&value.inputs)),
            (Self::Instruction(value), "out") => Self::Sequence(Sequence::Operands(&value.outputs)),
            (Self::Instruction(value), "parameters") => Self::Sequence(Sequence::Parameters(&value.parameters)),
            (Self::Instruction(value), "flags") => Self::strings(&value.flags),
            (Self::Instruction(value), "action") => Self::Sequence(Sequence::Actions(&value.action)),
            (Self::Instruction(value), "metadata") => Self::metadata(&value.metadata),
            (Self::Code(value), "name") => Self::Str(&value.name),
            (Self::Code(value), "description") => Self::Str(&value.description),
            (Self::Code(value), "stabilizers") => Self::paulis(&value.stabilizers),
            (Self::Code(value), "x") => Self::paulis(&value.x),
            (Self::Code(value), "z") => Self::paulis(&value.z),
            (Self::Code(value), "metadata") => Self::metadata(&value.metadata),
            (Self::Gadget(value), "implements") => Self::Instruction(&value.implements),
            (Self::Gadget(value), "circuit") => Self::Circuit(&value.circuit),
            (Self::Gadget(value), "in") => Self::Sequence(Sequence::Encodings(&value.inputs)),
            (Self::Gadget(value), "out") => Self::Sequence(Sequence::Encodings(&value.outputs)),
            (Self::Gadget(value), "checks") => Self::Sequence(Sequence::Checks(&value.checks)),
            (Self::Gadget(value), "readouts") => Self::Sequence(Sequence::Readouts(&value.readouts)),
            (Self::Gadget(value), "frames") => Self::Mapping(Mapping::Frames(&value.frames)),
            (Self::Gadget(value), "metadata") => Self::metadata(&value.metadata),
            (Self::Gadget(value), "parameter_bindings") => {
                Self::Mapping(Mapping::ParameterBindings(&value.parameter_bindings))
            }
            (Self::Circuit(value), "instruction_set") => Self::InstructionSet(&value.instruction_set),
            (Self::Circuit(value), "source") => Self::Str(&value.source),
            (Self::Circuit(value), "format") => Self::optional(value.format.as_deref()),
            (Self::Encoding(value), "code") => Self::Code(&value.code),
            (Self::Encoding(value), "stabilizers") => Self::paulis(&value.code.stabilizers),
            (Self::Encoding(value), "x") => Self::paulis(&value.code.x),
            (Self::Encoding(value), "z") => Self::paulis(&value.code.z),
            (Self::Encoding(value), "support") => Self::strings(&value.support),
            (Self::Encoding(value), "block_types") => Self::strings(&value.block_types),
            (Self::Block(value), "name") => Self::Str(&value.name),
            (Self::Block(value), "encodes") => Self::Int(value.encodes as i128),
            (Self::BlockOperand(value), "block") => Self::Str(&value.block),
            (Self::BlockOperand(value), "is_variadic") => Self::Bool(value.is_variadic),
            (Self::Parameter(value), "name") => Self::Str(&value.name),
            (Self::Parameter(value), "kind") => Self::ParameterKind(value.kind),
            (Self::ParameterKind(value), "value") => Self::Str(match value {
                ParameterKind::Bit => "bit",
                ParameterKind::Number => "number",
                ParameterKind::Integer => "integer",
                ParameterKind::Boolean => "boolean",
                ParameterKind::String => "string",
                ParameterKind::Pauli => "pauli",
            }),
            (Self::Action(value), "condition") => value.condition.as_ref().map_or(Self::None, Self::Condition),
            (Self::Action(value), field) => match (&value.action, field) {
                (Action::Stabilize(values), "operators") => Self::paulis(values),
                (Action::Observe(values), "observables") => Self::Sequence(Sequence::Observables(values)),
                (Action::Clifford(values), "generators") => Self::Mapping(Mapping::Generators(values)),
                (Action::Pauli(value), "operator") | (Action::Rotate { pauli: value, .. }, "pauli") => {
                    Self::Str(&value.0)
                }
                (Action::Rotate { angle, .. }, "angle") => match angle {
                    Scalar::Literal(value) => Self::Float(*value),
                    Scalar::Parameter(value) => Self::Str(value),
                },
                _ => return None,
            },
            (Self::Condition(value), "predicates") => Self::strings(&value.predicates),
            (Self::Condition(value), "invert") => Self::Bool(value.invert),
            (Self::Readout(value), "position") => Self::Int(value.position as i128),
            (Self::Readout(value), "name") => Self::optional(value.name.as_deref()),
            (Self::Readout(value), "is_flag") => Self::Bool(value.is_flag),
            (Self::Readout(value), "equation") => Self::Sequence(Sequence::Terms(&value.equation)),
            (Self::ReadoutSpec { position, .. }, "position") => Self::Int(*position as i128),
            (Self::ReadoutSpec { spec, .. }, "name") => Self::optional(spec.name.as_deref()),
            (
                Self::ReadoutSpec {
                    position,
                    observe_count,
                    ..
                },
                "is_flag",
            ) => Self::Bool(position >= observe_count),
            (Self::ReadoutSpec { spec, .. }, "equation") => Self::Sequence(Sequence::Terms(&spec.equation)),
            (Self::Reference(value), "path") => Self::Str(value.path()),
            _ => return None,
        })
    }

    pub(super) fn child(&self, segment: &Segment) -> Option<Self> {
        match (self, segment) {
            (_, Segment::Field(field)) => self.field(field),
            (Self::Sequence(values), Segment::Index(index)) => values.get(*index),
            (Self::Mapping(values), Segment::Key(key)) => values.get(key),
            _ => None,
        }
    }

    pub(super) fn kind(&self) -> &'static str {
        match self {
            Self::Qodec(_) => "Qodec",
            Self::Layer(_) => "Layer",
            Self::InstructionSet(_) => "InstructionSet",
            Self::Instruction(_) => "Instruction",
            Self::Code(_) => "Code",
            Self::Gadget(_) => "Gadget",
            Self::Circuit(_) => "Circuit",
            Self::Encoding(_) => "Encoding",
            Self::Block(_) => "Block",
            Self::BlockOperand(_) => "BlockOperand",
            Self::Parameter(_) => "Parameter",
            Self::ParameterKind(_) => "Parameter.Kind",
            Self::Action(_) => "Action",
            Self::Condition(_) => "Condition",
            Self::Readout(_) | Self::ReadoutSpec { .. } => "Readout",
            Self::Reference(_) => "Reference",
            Self::Sequence(_) => "sequence",
            Self::Mapping(_) => "mapping",
            Self::Str(_) => "str",
            Self::Int(_) => "int",
            Self::Float(_) => "float",
            Self::Bool(_) => "bool",
            Self::None => "None",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{JSON_VISITS, Mapping, Qodec, Sequence, Value};

    #[test]
    fn lookup_does_not_visit_unselected_metadata() {
        let mut model = Qodec::new(None, None, Vec::new());
        model
            .metadata_mut()
            .insert("selected".to_owned(), serde_json::json!(42));
        model
            .metadata_mut()
            .insert("unrelated".to_owned(), serde_json::json!(vec![vec![0; 1000]; 100]));
        JSON_VISITS.with(|visits| visits.set(0));
        let metadata = model.resolve("metadata").unwrap();
        assert_eq!(metadata.resolve("[\"selected\"]").unwrap().as_int(), Some(42));
        assert!(metadata.resolve("[\"missing\"]").is_err());
        assert_eq!(JSON_VISITS.with(std::cell::Cell::get), 1);
        assert_eq!(metadata.as_mapping().unwrap().len(), 2);
        assert_eq!(JSON_VISITS.with(std::cell::Cell::get), 3);
    }

    #[test]
    fn collection_views_borrow_the_original_storage() {
        let values = vec![serde_json::json!([1, 2]), serde_json::json!({"nested": 3})];
        let view = Sequence::Json(&values);
        assert_eq!(view.len(), 2);
        assert!(view.get(2).is_none());
        assert!(
            matches!(view.get(0), Some(Value::Sequence(Sequence::Json(child))) if std::ptr::eq(child, values[0].as_array().unwrap().as_slice()))
        );
        assert!(
            matches!(view.get(1), Some(Value::Mapping(Mapping::Metadata(child))) if std::ptr::eq(child, values[1].as_object().unwrap()))
        );
    }
}
