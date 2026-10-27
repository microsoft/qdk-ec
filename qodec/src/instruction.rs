//! Instruction definitions and the formal action language.
//!
//! An [`Instruction`] declares an operation in an instruction set: its mnemonic, block
//! operands, classical parameters, flags, and action. The action specifies the
//! operation that a gadget must implement.
//!
//! An [`InstructionCall`](crate::InstructionCall) applies a declared instruction.

use crate::PauliString;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// An operation on logical qubits in an instruction set.
///
/// Each side's operands contribute consecutive logical qubit indexes starting
/// at zero, with each block's size given by [`Block::encodes`](crate::Block::encodes).
/// Actions address those indexes directly (`X_0`, `Z_3`). A qubit present on
/// both sides remains live; an output-only qubit is allocated, and an
/// input-only qubit is consumed.
///
/// An unconditional [`Action::Stabilize`] can introduce temporary qubits at
/// indexes outside both boundaries. Only its named indexes are introduced;
/// later steps can use them. They are local to an invocation. At the end of
/// the action, all qubits absent from `outputs` are traced out without emitting
/// outcomes. These semantic qubits do not prescribe physical gadget ancillas.
///
/// Classical outputs are **readouts**, ordered as outcomes followed by flags:
/// - **Outcomes** are bits produced by `Observe` actions, in action and
///   observable order. Action guards refer to them as `outcomes[i]`.
/// - **Flags** are named bits reported alongside outcomes. The instruction set declares
///   their names; a gadget supplies their parity equations. They do not
///   prescribe how a caller must handle the result.
///
/// A gadget's `readouts` list supplies one parity equation per readout. These
/// equations use gadget references, distinct from action guards.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Instruction {
    /// Unique name within its instruction set; how a call and a gadget refer to it.
    pub mnemonic: String,
    /// Free-form prose for readers.
    pub description: String,
    /// Input block operands, in order. Their qubits form a flat index space the
    /// action addresses as `X_0`, `Z_3`.
    #[serde(rename = "in", default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<BlockOperand>,
    /// Output block operands, in order, sharing the input flat index space.
    #[serde(rename = "out", default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<BlockOperand>,
    /// Classical parameters a call supplies arguments for, in declaration order.
    #[serde(default, with = "parameters_serde", skip_serializing_if = "Vec::is_empty")]
    pub parameters: Vec<Parameter>,
    /// What the instruction does, as an ordered list of steps. Empty means undeclared.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub action: Vec<ActionStep>,
    /// Names of the ideal-zero side channels the instruction exposes, after its
    /// `observe` outcomes, in the order a gadget's `readouts` continues.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub flags: Vec<String>,
    /// Annotations for external tools; see [`crate::Metadata`].
    #[serde(default, skip_serializing_if = "crate::Metadata::is_empty")]
    pub metadata: crate::Metadata,
}

impl Instruction {
    /// The total number of outcomes declared by all `Observe` actions.
    ///
    /// In a gadget's `readouts` list, outcomes precede [`flags`](Self::flags).
    /// This method does not validate the action.
    #[must_use]
    pub fn observe_count(&self) -> usize {
        self.action
            .iter()
            .map(|step| match &step.action {
                Action::Observe(observables) => observables.len(),
                _ => 0,
            })
            .sum()
    }
}

/// A positional block operand in an instruction's `in:` / `out:` list.
///
/// Serialized as a block type name (`c422`) or a one-element list (`[c422]`)
/// for a variadic operand. A variadic operand accepts multiple blocks of the
/// same type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockOperand {
    /// Block type name declared by the instruction set.
    pub block: String,
    /// Whether this operand accepts a variable number of blocks.
    pub is_variadic: bool,
}

/// A classical parameter declaration.
///
/// A `bit` parameter is a runtime input usable in action guards. The other
/// [`ParameterKind`] values declare compile-time literal arguments.
///
/// Serialized as a single-key map `{name: type}`, e.g. `{theta: number}`
/// or `{enabled: bit}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parameter {
    /// The name a call supplies an argument for.
    pub name: String,
    /// The value type the call must supply.
    pub kind: ParameterKind,
}

impl Serialize for Parameter {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(1))?;
        map.serialize_entry(&self.name, &self.kind)?;
        map.end()
    }
}

/// (De)serialization for an instruction's `parameters:` list.
///
/// The on-disk form is a plain map `{theta: number, enabled: bit}` of
/// parameter names to types. Parameter order is not semantically meaningful
/// (parameters are referenced by name), but author order is preserved in both
/// directions.
mod parameters_serde {
    use super::{Parameter, ParameterKind};
    use serde::de::{MapAccess, Visitor};
    use serde::ser::SerializeMap;
    use serde::{Deserializer, Serializer};
    use std::fmt;

    pub fn serialize<S>(parameters: &[Parameter], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(parameters.len()))?;
        for parameter in parameters {
            map.serialize_entry(&parameter.name, &parameter.kind)?;
        }
        map.end()
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<Parameter>, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ParametersVisitor;

        impl<'de> Visitor<'de> for ParametersVisitor {
            type Value = Vec<Parameter>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a map of parameter names to types")
            }

            fn visit_map<A>(self, mut access: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut parameters = Vec::new();
                while let Some((name, kind)) = access.next_entry::<String, ParameterKind>()? {
                    parameters.push(Parameter { name, kind });
                }
                Ok(parameters)
            }
        }

        deserializer.deserialize_map(ParametersVisitor)
    }
}

/// Types allowed for a classical parameter.
///
/// `Bit` is a runtime input usable in action guards; the other variants are
/// compile-time literal types. Serialized names are lowercase.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterKind {
    /// A runtime classical bit, supplied by a call as a measurement reference.
    Bit,
    /// A real literal.
    Number,
    /// An integer literal.
    Integer,
    /// A boolean literal.
    Boolean,
    /// A string literal.
    String,
    /// A Pauli operator written as text.
    Pauli,
}

/// An action with an optional XOR condition.
///
/// A step's YAML map can include `if:` or `unless:`:
///
/// ```yaml
/// - observe: Z_0
/// - pauli: X_0
///   if: ["outcomes[0]"]
/// ```
///
/// `if:` applies when the predicate parity is 1; `unless:` applies when it is
/// 0 and sets [`Condition::invert`]. Deserialization rejects a step containing
/// both. A conditional [`Action::Observe`] can be preserved as a draft, but
/// operations requiring a statically sized readout record reject it.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionStep {
    /// What this step does.
    pub action: Action,
    /// When present, the step applies only if the condition holds.
    pub condition: Option<Condition>,
}

impl Serialize for ActionStep {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        struct Repr<'a> {
            #[serde(flatten)]
            action: &'a Action,
            #[serde(rename = "if", skip_serializing_if = "Option::is_none")]
            if_predicates: Option<&'a Vec<String>>,
            #[serde(skip_serializing_if = "Option::is_none")]
            unless: Option<&'a Vec<String>>,
        }
        let (if_predicates, unless) = match &self.condition {
            Some(Condition {
                predicates,
                invert: false,
            }) => (Some(predicates), None),
            Some(Condition {
                predicates,
                invert: true,
            }) => (None, Some(predicates)),
            None => (None, None),
        };
        Repr {
            action: &self.action,
            if_predicates,
            unless,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ActionStep {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Repr {
            #[serde(flatten)]
            action: Action,
            #[serde(rename = "if", default)]
            if_predicates: Option<Vec<String>>,
            #[serde(default)]
            unless: Option<Vec<String>>,
        }
        let repr = Repr::deserialize(deserializer)?;
        let condition = match (repr.if_predicates, repr.unless) {
            (Some(predicates), None) => Some(Condition {
                predicates,
                invert: false,
            }),
            (None, Some(predicates)) => Some(Condition {
                predicates,
                invert: true,
            }),
            (None, None) => None,
            (Some(_), Some(_)) => {
                return Err(serde::de::Error::custom(
                    "action step cannot have both `if` and `unless`",
                ));
            }
        };
        Ok(Self {
            action: repr.action,
            condition,
        })
    }
}

/// The action performed by a step.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// Force the state into the +1 eigenspace of each Pauli operator.
    ///
    /// Accepts either a sequence of Pauli operators (e.g. `stabilize:
    /// [Z_0]`) or a single operator as shorthand (e.g. `stabilize:
    /// Z_0`).
    ///
    /// Without a condition, introduces temporary qubits at referenced indexes
    /// outside the instruction's input/output range. Subsequent steps can use
    /// those indexes; unmentioned gaps are not introduced. Temporary qubits
    /// are traced out when the instruction ends. Conditional stabilization
    /// may reset existing qubits but cannot introduce temporary ones.
    ///
    /// Operators act sequentially. No outcome is emitted. A partial eigenspace
    /// constraint does not select a state or recovery within that eigenspace.
    Stabilize(#[serde(deserialize_with = "deserialize_paulis")] Vec<PauliString>),
    /// Apply a Clifford unitary in stabilizer tableau format.
    ///
    /// Each entry is one generator mapping: the key is the input Pauli
    /// generator (e.g. `"X_0"`) and the value is the output
    /// Pauli image (e.g. `"X_0 X_1"`). Keys are unique
    /// by construction; insertion order is not significant.
    Clifford(BTreeMap<PauliString, PauliString>),
    /// Apply a Pauli operator as a unitary.
    Pauli(PauliString),
    /// Measure Pauli observables. Produces one outcome per entry.
    ///
    /// Accepts either a sequence of observables (e.g. `observe: [Z_0]`)
    /// or a single observable as shorthand (e.g. `observe: Z_0`).
    Observe(
        #[serde(
            serialize_with = "serialize_observables",
            deserialize_with = "deserialize_observables"
        )]
        Vec<Observable>,
    ),
    /// Apply `exp(-i * angle/2 * pauli)`, with `angle` in radians.
    Rotate {
        /// The rotation axis.
        pauli: PauliString,
        /// The angle in radians, as a literal or a parameter name.
        angle: Scalar,
    },
}

/// Deserialize the payload of a `stabilize:` action, accepting either a single
/// Pauli operator as shorthand (`stabilize: Z_0`) or a sequence of them
/// (`stabilize: [Z_0]`).
fn deserialize_paulis<'de, D>(deserializer: D) -> Result<Vec<PauliString>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        Many(Vec<PauliString>),
        One(PauliString),
    }

    match OneOrMany::deserialize(deserializer)? {
        OneOrMany::Many(paulis) => Ok(paulis),
        OneOrMany::One(pauli) => Ok(vec![pauli]),
    }
}

/// Deserialize the payload of an `observe:` action, accepting either a single
/// observable as shorthand (`observe: Z_0`) or a sequence of them
/// (`observe: [Z_0]`). Each entry is a bare Pauli string; observables are
/// positional (no aliases).
fn deserialize_observables<'de, D>(deserializer: D) -> Result<Vec<Observable>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        Many(Vec<Observable>),
        One(Observable),
    }

    match OneOrMany::deserialize(deserializer)? {
        OneOrMany::Many(observables) => Ok(observables),
        OneOrMany::One(observable) => Ok(vec![observable]),
    }
}

/// Serialize the payload of an `observe:` action, emitting the single-observable
/// shorthand (`observe: Z_0`) when exactly one observable is present and
/// the sequence form otherwise.
fn serialize_observables<S>(observables: &[Observable], serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    if let [single] = observables {
        single.serialize(serializer)
    } else {
        observables.serialize(serializer)
    }
}

/// An action condition computed as the XOR parity of bit references.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Condition {
    /// Bit parameter names or `outcomes[i]` references. Outcome indexes are
    /// zero-based across all `Observe` actions in the instruction.
    pub predicates: Vec<String>,
    /// `false` tests for parity 1 (`if:`); `true` tests for parity 0 (`unless:`).
    pub invert: bool,
}

/// A Pauli observable whose measurement produces one outcome bit.
///
/// Action guards use `outcomes[i]`, with zero-based indexes assigned across
/// all `Observe` steps in action and observable order. Serialized as a bare
/// Pauli string (`"Z_0"`); deserialization rejects the empty string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observable {
    /// The operator measured, as a Pauli string.
    pub pauli: PauliString,
}

impl Serialize for Observable {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.pauli.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Observable {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;

        let pauli = String::deserialize(deserializer)?;
        if pauli.is_empty() {
            return Err(D::Error::custom(
                "observable: pauli is required \
                 (for a Pauliless flag, declare it in the instruction's `flags:` field instead)",
            ));
        }
        Ok(Self {
            pauli: PauliString(pauli),
        })
    }
}

/// A numeric literal or a reference to a `number` or `integer` parameter by name.
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
#[derive(Debug, Clone, PartialEq)]
pub enum Scalar {
    /// A numeric literal.
    Literal(f64),
    /// The name of a declared `number` or `integer` parameter, serialized as a string.
    Parameter(String),
}

impl Serialize for BlockOperand {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if self.is_variadic {
            [&self.block].serialize(serializer)
        } else {
            self.block.serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for BlockOperand {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            /// Bare block name: `c422`.
            Bare(String),
            /// Variadic entry: a one-element list `[c422]`.
            Variadic(Vec<String>),
            /// Unsupported pinned indices: `{c422: [2, 3]}`.
            Pinned(BTreeMap<String, Vec<usize>>),
        }

        match Repr::deserialize(deserializer)? {
            Repr::Bare(block) => Ok(Self {
                block,
                is_variadic: false,
            }),
            Repr::Variadic(blocks) => {
                let mut blocks = blocks.into_iter();
                let block = blocks.next().ok_or_else(|| {
                    serde::de::Error::custom("variadic block operand must list exactly one block type, e.g. `[qubit]`")
                })?;
                if blocks.next().is_some() {
                    return Err(serde::de::Error::custom(
                        "variadic block operand must list exactly one block type, e.g. `[qubit]`",
                    ));
                }
                Ok(Self {
                    block,
                    is_variadic: true,
                })
            }
            Repr::Pinned(entries) => {
                let block = entries.keys().next().map_or("block", String::as_str);
                Err(serde::de::Error::custom(format!(
                    "block operand `{{{block}: [...]}}` cannot pin flat indices; \
                     write the bare block type `{block}` and let it take the next contiguous range"
                )))
            }
        }
    }
}

#[cfg(test)]
mod block_operand_tests {
    use super::BlockOperand;

    #[test]
    fn bare_string_is_not_variadic() {
        let entry: BlockOperand = serde_yaml::from_str("qubit").unwrap();
        assert_eq!(entry.block, "qubit");
        assert!(!entry.is_variadic);
    }

    #[test]
    fn one_element_list_declares_variadic() {
        let entry: BlockOperand = serde_yaml::from_str("[qubit]").unwrap();
        assert_eq!(entry.block, "qubit");
        assert!(entry.is_variadic);
    }

    #[test]
    fn map_form_is_rejected() {
        let result: Result<BlockOperand, _> = serde_yaml::from_str("{c422: [2, 3]}");
        let message = result.expect_err("block operands cannot pin flat indices").to_string();
        assert!(message.contains("c422"), "{message}");
        assert!(message.contains("cannot pin flat indices"), "{message}");
        assert!(message.contains("write the bare block type `c422`"), "{message}");
    }

    #[test]
    fn bare_round_trips_to_string() {
        let entry = BlockOperand {
            block: "qubit".to_owned(),
            is_variadic: false,
        };
        let serialized = serde_yaml::to_string(&entry).unwrap();
        assert_eq!(serialized.trim(), "qubit");
        let reparsed: BlockOperand = serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(entry, reparsed);
    }

    #[test]
    fn variadic_round_trips_to_list_form() {
        let entry: BlockOperand = serde_yaml::from_str("[qubit]").unwrap();
        let serialized = serde_yaml::to_string(&entry).unwrap();
        let reparsed: BlockOperand = serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(entry, reparsed);
        assert!(reparsed.is_variadic);
    }

    #[test]
    fn empty_list_value_is_rejected() {
        let result: Result<BlockOperand, _> = serde_yaml::from_str("[]");
        assert!(result.is_err());
    }

    #[test]
    fn multiple_list_values_are_rejected() {
        let result: Result<BlockOperand, _> = serde_yaml::from_str("[a, b]");
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod operands_serde_tests {
    use super::Instruction;

    fn parse(yaml: &str) -> Instruction {
        serde_yaml::from_str(yaml).expect("instruction should parse")
    }

    #[test]
    fn list_form_in_out_parses() {
        let instruction = parse(
            "\
mnemonic: cx
description: ''
in: [c6, c6]
out: [c6, c6]
",
        );
        assert_eq!(instruction.inputs.len(), 2);
        assert_eq!(instruction.inputs[0].block, "c6");
        assert_eq!(instruction.inputs[1].block, "c6");
        assert_eq!(instruction.outputs.len(), 2);
    }

    #[test]
    fn map_form_is_rejected() {
        let result: Result<Instruction, _> = serde_yaml::from_str(
            "\
mnemonic: cx
description: ''
in: {control: c6, target: c6}
",
        );
        assert!(result.is_err(), "instruction operands must be a list, not a named map");
    }

    #[test]
    fn variadic_list_form_parses() {
        let instruction = parse(
            "\
mnemonic: mpp
description: ''
in: [[carbon]]
",
        );
        assert_eq!(instruction.inputs.len(), 1);
        assert!(instruction.inputs[0].is_variadic);
        assert_eq!(instruction.inputs[0].block, "carbon");
    }

    #[test]
    fn explicit_indices_are_rejected() {
        let yaml = "\
mnemonic: switch
description: ''
in: [a, b]
out: [{b: [2, 3]}, c]
";
        let result: Result<Instruction, _> = serde_yaml::from_str(yaml);
        assert!(result.is_err(), "block operands cannot pin flat indices");
    }

    #[test]
    fn coded_operands_emit_list_form() {
        let instruction = parse(
            "\
mnemonic: prepare
description: ''
out: [c6]
",
        );
        let serialized = serde_yaml::to_string(&instruction).unwrap();
        assert!(
            serialized.contains("out:\n- c6"),
            "fresh-allocation out should emit list form, got:\n{serialized}"
        );
    }

    #[test]
    fn round_trips_through_list_form() {
        let instruction = parse(
            "\
mnemonic: idle
description: ''
in: [c6]
out: [c6]
",
        );
        let serialized = serde_yaml::to_string(&instruction).unwrap();
        let reparsed: Instruction = serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(instruction, reparsed);
    }
}

#[cfg(test)]
mod observe_serde_tests {
    use super::{Action, ActionStep};

    fn parse(yaml: &str) -> ActionStep {
        serde_yaml::from_str(yaml).expect("action step should parse")
    }

    #[test]
    fn single_observable_shorthand_parses() {
        let step = parse("observe: Z_0");
        match step.action {
            Action::Observe(observables) => {
                assert_eq!(observables.len(), 1);
                assert_eq!(observables[0].pauli.0, "Z_0");
            }
            other => panic!("expected Observe, got {other:?}"),
        }
    }

    #[test]
    fn shorthand_and_sequence_forms_are_equivalent() {
        let shorthand = parse("observe: Z_0");
        let sequence = parse("observe: [Z_0]");
        assert_eq!(shorthand.action, sequence.action);
    }

    #[test]
    fn aliased_observable_is_rejected() {
        // Observables are positional; the `{alias: pauli}` map form is no
        // longer accepted.
        let result: Result<ActionStep, _> = serde_yaml::from_str("observe: {logical: Z_0}");
        assert!(result.is_err(), "aliased observable map form must be rejected");
    }

    #[test]
    fn mixed_string_and_alias_sequence_is_rejected() {
        let result: Result<ActionStep, _> = serde_yaml::from_str("observe: [Z_0, {foo: X_0}]");
        assert!(result.is_err(), "aliased observable map form must be rejected");
    }

    #[test]
    fn two_key_map_form_is_rejected() {
        let result: Result<ActionStep, _> = serde_yaml::from_str("observe: {name: logical, pauli: Z_0}");
        assert!(result.is_err(), "the old two-key {{name, pauli}} form must be rejected");
    }

    #[test]
    fn single_observable_serializes_as_shorthand() {
        let step = parse("observe: Z_0");
        let yaml = serde_yaml::to_string(&step).expect("serialize");
        assert_eq!(yaml.trim(), "observe: Z_0");
    }

    #[test]
    fn aliased_observable_serialization_round_trips_as_bare_pauli() {
        let step = parse("observe: [Z_0, X_0]");
        let yaml = serde_yaml::to_string(&step).expect("serialize");
        assert_eq!(yaml.trim(), "observe:\n- Z_0\n- X_0");
    }

    #[test]
    fn sequence_form_still_parses() {
        let step = parse("observe: [Z_0, X_1]");
        match step.action {
            Action::Observe(observables) => assert_eq!(observables.len(), 2),
            other => panic!("expected Observe, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod stabilize_serde_tests {
    use super::{Action, ActionStep};

    fn parse(yaml: &str) -> ActionStep {
        serde_yaml::from_str(yaml).expect("action step should parse")
    }

    #[test]
    fn single_pauli_shorthand_parses() {
        let step = parse("stabilize: Z_0");
        match step.action {
            Action::Stabilize(paulis) => {
                assert_eq!(paulis.len(), 1);
                assert_eq!(paulis[0].0, "Z_0");
            }
            other => panic!("expected Stabilize, got {other:?}"),
        }
    }

    #[test]
    fn shorthand_and_sequence_forms_are_equivalent() {
        let shorthand = parse("stabilize: Z_0");
        let sequence = parse("stabilize: [Z_0]");
        assert_eq!(shorthand.action, sequence.action);
    }

    #[test]
    fn sequence_form_still_parses() {
        let step = parse("stabilize: [Z_0, X_1]");
        match step.action {
            Action::Stabilize(paulis) => assert_eq!(paulis.len(), 2),
            other => panic!("expected Stabilize, got {other:?}"),
        }
    }
}
