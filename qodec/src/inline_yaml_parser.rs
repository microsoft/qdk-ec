//! Inline-YAML source parser: converts a YAML sequence of single-key
//! instruction-call maps into [`InstructionCall`]s.
//!
//! This is the format used by a gadget's `circuit.source` when the circuit
//! is written directly in the gadget YAML (rather than referencing an
//! external file or embedding stim text). Each item is one instruction
//! call with optional `operands`, `arguments`, and `select` fields:
//!
//! ```yaml
//! - rotate_z:
//!     operands: [0]
//!     arguments: {theta: pi}
//! - prepare: {operands: [0], select: [{reject: 0}]}
//! - cx: [0, 1]
//! - rotate_z: [0, theta: pi]
//! - tick: []
//! ```
//!
//! An operand is a scalar block label (an integer or a name).
//! The list shorthand starts with block operands, followed by named argument
//! values. No argument name is reserved; selection uses the object form.
use crate::{Argument, InstructionCall, Operand, SelectPattern};
use std::collections::BTreeMap;

/// Parse a YAML-sequence source string into a sequence of [`InstructionCall`]s.
///
/// Each sequence item is a single-entry mapping `{mnemonic: call}`. A call
/// object has optional `operands` (a list), `arguments` (a name-to-value map),
/// and `select` (a list of flag patterns). Omitted fields are empty.
/// The list shorthand contains block operands followed by named arguments;
/// `null` is also accepted for an empty call.
///
/// # Errors
///
/// Returns a description on parse failure (not a YAML sequence, item not a
/// single-key mapping, an operand after a named argument, an unsupported value
/// shape, or a malformed `select:` block).
///
/// # Panics
///
/// Panics if a single-entry mapping yields no entry, which `serde_yaml`'s
/// length check has already ruled out.
pub fn parse_inline_yaml(source: &str) -> Result<Vec<InstructionCall>, String> {
    let value: serde_yaml::Value =
        serde_yaml::from_str(source).map_err(|error| format!("inline source must be valid YAML: {error}"))?;
    let sequence = value
        .as_sequence()
        .ok_or_else(|| "inline source must be a YAML sequence".to_owned())?;

    let mut calls = Vec::with_capacity(sequence.len());
    for (index, item) in sequence.iter().enumerate() {
        let mapping = item
            .as_mapping()
            .ok_or_else(|| format!("inline source item {index}: expected a single-key mapping"))?;
        if mapping.len() != 1 {
            return Err(format!(
                "inline source item {index}: expected exactly one key (the mnemonic), got {}",
                mapping.len()
            ));
        }
        let (key, args_value) = mapping.iter().next().expect("len == 1");
        let mnemonic = key
            .as_str()
            .ok_or_else(|| format!("inline source item {index}: mnemonic must be a string"))?
            .to_owned();

        let ParsedCall {
            operands,
            arguments,
            select,
        } = parse_call_args(&mnemonic, args_value)?;
        calls.push(InstructionCall {
            mnemonic,
            operands,
            arguments,
            select,
        });
    }
    Ok(calls)
}

/// Parsed block operands, arguments, and selection patterns.
#[derive(Default)]
struct ParsedCall {
    operands: Vec<Operand>,
    arguments: BTreeMap<String, Argument>,
    select: Vec<SelectPattern>,
}

fn parse_call_args(mnemonic: &str, value: &serde_yaml::Value) -> Result<ParsedCall, String> {
    if value.is_null() {
        return Ok(ParsedCall::default());
    }
    if let Some(fields) = value.as_mapping() {
        return parse_call_object(mnemonic, fields);
    }
    let sequence = value.as_sequence().ok_or_else(|| {
        format!(
            "inline source for {mnemonic:?}: call must be an object or a list of operands and `key: value` arguments"
        )
    })?;
    parse_call_shorthand(mnemonic, sequence)
}

fn parse_call_object(mnemonic: &str, fields: &serde_yaml::Mapping) -> Result<ParsedCall, String> {
    let mut call = ParsedCall::default();
    for (key, field) in fields {
        match key.as_str() {
            Some("operands") => {
                let operands = field
                    .as_sequence()
                    .ok_or_else(|| format!("inline source for {mnemonic:?}: `operands:` must be a YAML sequence"))?;
                call.operands = operands
                    .iter()
                    .map(|operand| parse_operand_value(mnemonic, operand))
                    .collect::<Result<_, _>>()?;
            }
            Some("arguments") => {
                let arguments = field
                    .as_mapping()
                    .ok_or_else(|| format!("inline source for {mnemonic:?}: `arguments:` must be a YAML mapping"))?;
                parse_arguments(mnemonic, arguments, &mut call.arguments)?;
            }
            Some("select") => call.select = parse_select(mnemonic, field)?,
            Some(name) => return Err(format!("inline source for {mnemonic:?}: unknown call field {name:?}")),
            None => {
                return Err(format!(
                    "inline source for {mnemonic:?}: call field name must be a string"
                ));
            }
        }
    }
    Ok(call)
}

fn parse_call_shorthand(mnemonic: &str, sequence: &[serde_yaml::Value]) -> Result<ParsedCall, String> {
    let mut call = ParsedCall::default();
    let mut has_named_arguments = false;
    for element in sequence {
        if let Some(map) = element.as_mapping() {
            has_named_arguments = true;
            parse_arguments(mnemonic, map, &mut call.arguments)?;
        } else {
            if has_named_arguments {
                return Err(format!(
                    "inline source for {mnemonic:?}: a positional operand cannot follow a named argument"
                ));
            }
            call.operands.push(parse_operand_value(mnemonic, element)?);
        }
    }
    Ok(call)
}

fn parse_arguments(
    mnemonic: &str,
    mapping: &serde_yaml::Mapping,
    arguments: &mut BTreeMap<String, Argument>,
) -> Result<(), String> {
    for (key, value) in mapping {
        let name = key
            .as_str()
            .ok_or_else(|| format!("inline source for {mnemonic:?}: argument name must be a string"))?;
        if arguments
            .insert(name.to_owned(), parse_argument_value(mnemonic, name, value)?)
            .is_some()
        {
            return Err(format!("inline source for {mnemonic:?}: duplicate argument {name:?}"));
        }
    }
    Ok(())
}

/// Parse one positional operand: a scalar block id (an integer index or a block name).
fn parse_operand_value(mnemonic: &str, value: &serde_yaml::Value) -> Result<Operand, String> {
    if let Some(i) = value.as_u64() {
        let index = usize::try_from(i)
            .map_err(|_| format!("inline source for {mnemonic:?}: operand index {i} out of range"))?;
        return Ok(Operand::Index(index));
    }
    if let Some(s) = value.as_str() {
        return Ok(Operand::Name(s.to_owned()));
    }
    Err(format!(
        "inline source for {mnemonic:?}: an operand must be an integer index or a block name"
    ))
}

fn parse_select(mnemonic: &str, value: &serde_yaml::Value) -> Result<Vec<SelectPattern>, String> {
    let sequence = value
        .as_sequence()
        .ok_or_else(|| format!("inline source for {mnemonic:?}: `select:` must be a YAML sequence"))?;
    let mut patterns = Vec::with_capacity(sequence.len());
    for (i, item) in sequence.iter().enumerate() {
        let mapping = item.as_mapping().ok_or_else(|| {
            format!("inline source for {mnemonic:?}: `select:` entry {i} must be a `{{name: 0|1, ...}}` mapping")
        })?;
        let mut pattern = SelectPattern::new();
        for (key, val) in mapping {
            let name = key
                .as_str()
                .ok_or_else(|| {
                    format!("inline source for {mnemonic:?}: `select:` entry {i}: atom name must be a string")
                })?
                .to_owned();
            let bit = val.as_u64().and_then(|number| u8::try_from(number).ok()).ok_or_else(|| {
                format!("inline source for {mnemonic:?}: `select:` entry {i}: atom {name:?} bit must be a non-negative integer")
            })?;
            if bit > 1 {
                return Err(format!(
                    "inline source for {mnemonic:?}: `select:` entry {i}: atom {name:?} bit must be 0 or 1 (got {bit})",
                ));
            }
            if pattern.insert(name.clone(), bit).is_some() {
                return Err(format!(
                    "inline source for {mnemonic:?}: `select:` entry {i}: duplicate atom {name:?}",
                ));
            }
        }
        patterns.push(pattern);
    }
    Ok(patterns)
}

fn parse_argument_value(mnemonic: &str, parameter: &str, value: &serde_yaml::Value) -> Result<Argument, String> {
    if let Some(boolean) = value.as_bool() {
        return Ok(Argument::Boolean(boolean));
    }
    // A parameter argument is an integer whatever its sign; only an operand names a qubit.
    if let Some(i) = value.as_i64() {
        return Ok(Argument::Integer(i));
    }
    if let Some(i) = value.as_u64() {
        return Err(format!(
            "inline source for {mnemonic:?} parameter {parameter:?}: integer {i} is out of range"
        ));
    }
    if let Some(f) = value.as_f64() {
        return Ok(Argument::Number(f));
    }
    if let Some(text) = value.as_str() {
        return Argument::parse_text(text);
    }
    if let Some(seq) = value.as_sequence() {
        if seq.iter().all(|element| element.as_u64().is_some()) {
            let qubits: Vec<usize> = seq
                .iter()
                .map(|element| {
                    let index = element.as_u64().expect("guarded by all(as_u64)");
                    usize::try_from(index).map_err(|_| {
                        format!(
                            "inline source for {mnemonic:?} parameter {parameter:?}: qubit index {index} out of range"
                        )
                    })
                })
                .collect::<Result<_, _>>()?;
            return Ok(Argument::QubitList(qubits));
        }
        if seq.iter().all(|element| element.as_str().is_some()) {
            let strings: Vec<String> = seq
                .iter()
                .map(|element| element.as_str().expect("guarded by all(as_str)").to_owned())
                .collect();
            return Ok(Argument::StringList(strings));
        }
    }
    Err(format!(
        "inline source for {mnemonic:?} parameter {parameter:?}: unsupported value shape"
    ))
}

/// Emit a sequence of [`InstructionCall`]s as an inline-YAML source string.
///
/// Produces the object form accepted by [`parse_inline_yaml`], omitting empty
/// fields. The output is suitable for storing as a gadget's inline source.
///
/// # Errors
///
/// Returns a description on YAML serialization failure.
#[cfg(test)]
fn emit_inline_yaml(calls: &[InstructionCall]) -> Result<String, String> {
    let mut sequence = serde_yaml::Sequence::with_capacity(calls.len());
    for call in calls {
        let mut fields = serde_yaml::Mapping::new();
        if !call.operands.is_empty() {
            fields.insert(
                serde_yaml::Value::String("operands".to_owned()),
                serde_yaml::Value::Sequence(call.operands.iter().map(operand_to_yaml).collect()),
            );
        }
        if !call.arguments.is_empty() {
            let arguments = call
                .arguments
                .iter()
                .map(|(name, argument)| (serde_yaml::Value::String(name.clone()), argument_to_yaml(argument)))
                .collect();
            fields.insert(
                serde_yaml::Value::String("arguments".to_owned()),
                serde_yaml::Value::Mapping(arguments),
            );
        }
        if !call.select.is_empty() {
            fields.insert(
                serde_yaml::Value::String("select".to_owned()),
                select_to_yaml(&call.select),
            );
        }
        let mut entry = serde_yaml::Mapping::with_capacity(1);
        entry.insert(
            serde_yaml::Value::String(call.mnemonic.clone()),
            serde_yaml::Value::Mapping(fields),
        );
        sequence.push(serde_yaml::Value::Mapping(entry));
    }
    serde_yaml::to_string(&serde_yaml::Value::Sequence(sequence))
        .map_err(|error| format!("inline-YAML emitter: serialization failed: {error}"))
}

#[cfg(test)]
fn operand_to_yaml(operand: &Operand) -> serde_yaml::Value {
    match operand {
        Operand::Index(index) => serde_yaml::Value::Number((*index as u64).into()),
        Operand::Name(name) => serde_yaml::Value::String(name.clone()),
    }
}

#[cfg(test)]
fn argument_to_yaml(argument: &Argument) -> serde_yaml::Value {
    match argument {
        Argument::Qubit(index) => serde_yaml::Value::Number((*index as u64).into()),
        Argument::QubitList(indices) => serde_yaml::Value::Sequence(
            indices
                .iter()
                .map(|index| serde_yaml::Value::Number((*index as u64).into()))
                .collect(),
        ),
        Argument::Integer(value) => serde_yaml::Value::Number((*value).into()),
        Argument::Number(value) => serde_yaml::Value::Number((*value).into()),
        Argument::Boolean(value) => serde_yaml::Value::Bool(*value),
        Argument::Text(text) => serde_yaml::Value::String(text.clone()),
        Argument::StringList(items) => serde_yaml::Value::Sequence(
            items
                .iter()
                .map(|text| serde_yaml::Value::String(text.clone()))
                .collect(),
        ),
        Argument::Readout(index) => serde_yaml::Value::String(format!("circuit.readouts[{index}]")),
    }
}

#[cfg(test)]
fn select_to_yaml(select: &[SelectPattern]) -> serde_yaml::Value {
    let sequence = select
        .iter()
        .map(|pattern| {
            let mut map = serde_yaml::Mapping::with_capacity(pattern.len());
            for (atom, bit) in pattern {
                map.insert(
                    serde_yaml::Value::String(atom.clone()),
                    serde_yaml::Value::Number(u64::from(*bit).into()),
                );
            }
            serde_yaml::Value::Mapping(map)
        })
        .collect();
    serde_yaml::Value::Sequence(sequence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_operands_and_arguments() {
        let src = "- cx: [0, 1]\n- rotate_z: [2, theta: pi]\n";
        let calls = parse_inline_yaml(src).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].mnemonic, "cx");
        assert_eq!(calls[0].operands, vec![Operand::Index(0), Operand::Index(1)]);
        assert!(calls[0].arguments.is_empty());
        assert_eq!(calls[1].mnemonic, "rotate_z");
        assert_eq!(calls[1].operands, vec![Operand::Index(2)]);
        assert_eq!(calls[1].arguments["theta"], Argument::Text("pi".to_owned()));
    }

    #[test]
    fn braced_argument_map_is_equivalent() {
        let bare = parse_inline_yaml("- rotate_z: [0, theta: pi]\n").unwrap();
        let braced = parse_inline_yaml("- rotate_z: [0, {theta: pi}]\n").unwrap();
        assert_eq!(bare, braced);
    }

    #[test]
    fn multiple_argument_pairs() {
        let calls = parse_inline_yaml("- gate: [0, alpha: a, beta: b]\n").unwrap();
        let call = &calls[0];
        assert_eq!(call.operands, vec![Operand::Index(0)]);
        assert_eq!(call.arguments["alpha"], Argument::Text("a".to_owned()));
        assert_eq!(call.arguments["beta"], Argument::Text("b".to_owned()));
    }

    #[test]
    fn rejects_operand_after_argument() {
        let err = parse_inline_yaml("- gate: [0, theta: pi, 1]\n").unwrap_err();
        assert!(err.contains("operand cannot follow"));
    }

    #[test]
    fn parses_and_round_trips_readout_operand() {
        let src = "- measure_z: [1]\n- correct_z: [0, c: \"circuit.readouts[0]\"]\n";
        let calls = parse_inline_yaml(src).unwrap();
        let call = &calls[1];
        assert_eq!(call.operands, vec![Operand::Index(0)]);
        assert_eq!(call.arguments["c"], Argument::Readout(0));
        let emitted = emit_inline_yaml(&calls).unwrap();
        let reparsed = parse_inline_yaml(&emitted).unwrap();
        assert_eq!(calls, reparsed);
    }

    #[test]
    fn rejects_the_bare_readouts_spelling() {
        let src = "- correct_z: [0, c: \"readouts[0]\"]\n";
        let error = parse_inline_yaml(src).unwrap_err();
        assert!(error.contains("use `circuit.readouts[0]`"), "{error}");
    }

    #[test]
    fn rejects_malformed_readout_operand() {
        let src = "- correct_z: [0, c: \"circuit.readouts[abc]\"]\n";
        let err = parse_inline_yaml(src).unwrap_err();
        assert!(err.contains("readout reference"));
    }

    #[test]
    fn empty_operands() {
        let calls = parse_inline_yaml("- tick: []\n- barrier:\n").unwrap();
        assert_eq!(calls.len(), 2);
        for call in &calls {
            assert!(call.operands.is_empty());
            assert!(call.arguments.is_empty());
        }
    }

    #[test]
    fn rejects_non_sequence() {
        assert!(parse_inline_yaml("foo: bar\n").is_err());
    }

    #[test]
    fn emit_then_parse_round_trips() {
        let original = "- cx: [0, 1]\n- rotate_z: [2, theta: pi]\n- tick: []\n";
        let parsed = parse_inline_yaml(original).unwrap();
        let emitted = emit_inline_yaml(&parsed).unwrap();
        let reparsed = parse_inline_yaml(&emitted).unwrap();
        assert_eq!(parsed, reparsed);
    }

    #[test]
    fn select_flow_unbraced_single_atom() {
        let src = "- foo: {select: [a: 0]}\n";
        let calls = parse_inline_yaml(src).unwrap();
        assert_eq!(calls, parse_inline_yaml("- foo: {select: [{a: 0}]}\n").unwrap());
        let call = &calls[0];
        assert_eq!(call.select.len(), 1);
        assert_eq!(call.select[0]["a"], 0);
    }

    #[test]
    fn select_flow_unbraced_two_patterns() {
        let src = "- foo: {select: [a: 0, b: 1]}\n";
        let calls = parse_inline_yaml(src).unwrap();
        let call = &calls[0];
        assert_eq!(call.select.len(), 2);
        assert_eq!(call.select[0]["a"], 0);
        assert_eq!(call.select[1]["b"], 1);
    }

    #[test]
    fn parses_operand_with_select() {
        let src = "- prepare_x_all: {operands: [3], select: [{ancilla1_reject: 0}]}\n";
        let calls = parse_inline_yaml(src).unwrap();
        assert_eq!(calls.len(), 1);
        let call = &calls[0];
        assert_eq!(call.operands, vec![Operand::Index(3)]);
        assert_eq!(call.select.len(), 1);
        assert_eq!(call.select[0]["ancilla1_reject"], 0);
    }

    #[test]
    fn select_requires_a_list() {
        for source in ["- gate: {select: {reject: 0}}", "- gate: {select: {}}"] {
            assert!(error(source).contains("`select:` must be a YAML sequence"));
        }
    }

    #[test]
    fn select_round_trip() {
        let src = "- prepare_x_all: {operands: [3], select: [{a_rej: 0}, {a_rej: 1}]}\n";
        let parsed = parse_inline_yaml(src).unwrap();
        let emitted = emit_inline_yaml(&parsed).unwrap();
        let reparsed = parse_inline_yaml(&emitted).unwrap();
        assert_eq!(parsed, reparsed);
    }

    #[test]
    fn select_multi_atom_pattern() {
        let src = "- foo: {select: [{a: 0, b: 1}]}\n";
        let calls = parse_inline_yaml(src).unwrap();
        let call = &calls[0];
        assert_eq!(call.select.len(), 1);
        assert_eq!(call.select[0]["a"], 0);
        assert_eq!(call.select[0]["b"], 1);
    }

    #[test]
    fn rejects_select_bit_out_of_range() {
        let src = "- foo: {select: [{a: 2}]}\n";
        let err = parse_inline_yaml(src).unwrap_err();
        assert!(err.contains("0 or 1"));
    }

    #[test]
    fn empty_select_is_unconstrained() {
        let src = "- foo: {select: []}\n";
        let calls = parse_inline_yaml(src).unwrap();
        let call = &calls[0];
        assert!(call.select.is_empty());
    }

    #[test]
    fn calls_reject_predicates() {
        assert!(error("- cx: {predicates: [enabled]}").contains("unknown call field \"predicates\""));
    }

    fn error(source: &str) -> String {
        parse_inline_yaml(source).expect_err("should not parse")
    }

    fn only_call(source: &str) -> InstructionCall {
        let calls = parse_inline_yaml(source).expect("should parse");
        assert_eq!(calls.len(), 1);
        calls.into_iter().next().unwrap()
    }

    #[test]
    fn a_block_operand_may_be_a_name() {
        assert_eq!(
            only_call("- gate: [c4]\n").operands,
            vec![Operand::Name("c4".to_owned())]
        );
    }

    #[test]
    fn a_block_operand_must_be_an_index_or_a_name() {
        assert!(error("- gate: [true]\n").contains("must be an integer index or a block name"));
    }

    #[test]
    fn a_call_must_be_an_object_or_list() {
        assert!(error("- gate: 3\n").contains("must be an object or a list"));
    }

    #[test]
    fn argument_names_must_be_strings() {
        assert!(error("- gate: [{1: 2}]\n").contains("argument name must be a string"));
    }

    #[test]
    fn duplicate_arguments_are_rejected() {
        assert!(error("- gate: [0, {a: 1}, {a: 2}]\n").contains("duplicate argument"));
    }

    #[test]
    fn sequence_arguments_become_qubit_or_string_lists() {
        assert_eq!(
            only_call("- gate: [0, targets: [1, 2]]\n").arguments["targets"],
            Argument::QubitList(vec![1, 2])
        );
        assert_eq!(
            only_call("- gate: [0, names: [a, b]]\n").arguments["names"],
            Argument::StringList(vec!["a".to_owned(), "b".to_owned()])
        );
    }

    #[test]
    fn a_mixed_sequence_operand_is_rejected() {
        assert!(error("- gate: [0, mixed: [1, a]]\n").contains("unsupported value shape"));
    }

    #[test]
    fn boolean_arguments_preserve_their_type_in_both_call_forms() {
        let shorthand = "- gate: [0, enabled: true, disabled: false, count: 1, zero: 0, text: 'true']";
        let object =
            "- gate: {operands: [0], arguments: {enabled: true, disabled: false, count: 1, zero: 0, text: 'true'}}";
        let calls = parse_inline_yaml(shorthand).unwrap();
        assert_eq!(calls, parse_inline_yaml(object).unwrap());
        let call = only_call(shorthand);
        assert_eq!(call.arguments["enabled"], Argument::Boolean(true));
        assert_eq!(call.arguments["disabled"], Argument::Boolean(false));
        // A parameter argument is an integer whatever its sign.
        assert_eq!(call.arguments["count"], Argument::Integer(1));
        assert_eq!(call.arguments["zero"], Argument::Integer(0));
        assert_eq!(call.arguments["text"], Argument::Text("true".to_owned()));
        assert_eq!(parse_inline_yaml(&emit_inline_yaml(&calls).unwrap()).unwrap(), calls);
        assert!(error("- gate: [true]").contains("an operand must be"));
        assert!(error("- gate: {select: [{reject: true}]}").contains("non-negative integer"));
    }

    #[test]
    fn select_patterns_are_parsed() {
        let call = only_call("- gate: {operands: [0], select: [{f: 0, g: 1}]}\n");
        assert_eq!(call.select.len(), 1);
        assert_eq!(call.select[0]["f"], 0);
        assert_eq!(call.select[0]["g"], 1);
    }

    #[test]
    fn malformed_select_blocks_are_rejected() {
        assert!(error("- gate: {select: 1}\n").contains("must be a YAML sequence"));
        assert!(error("- gate: {select: [1]}\n").contains("must be a `{name: 0|1, ...}` mapping"));
        assert!(error("- gate: {select: [{f: -1}]}\n").contains("must be a non-negative integer"));
        assert!(error("- gate: {select: [{1: 0}]}\n").contains("atom name must be a string"));
        assert!(error("- gate: {select: [], select: [{g: 1}]}\n").contains("duplicate"));
    }

    #[test]
    fn list_shorthand_matches_the_object_form() {
        for (shorthand, object) in [
            ("- tick: []", "- tick: {}"),
            ("- tick:", "- tick: {operands: [], arguments: {}, select: []}"),
            ("- cx: [0, block]", "- cx: {operands: [0, block]}"),
            (
                "- rotate_z: [0, theta: pi]",
                "- rotate_z: {operands: [0], arguments: {theta: pi}}",
            ),
            (
                "- gate: [0, positive: 1, negative: -1, angle: 1.5, indices: [1, 2], names: [a, b]]",
                "- gate: {operands: [0], arguments: {positive: 1, negative: -1, angle: 1.5, indices: [1, 2], names: [a, b]}}",
            ),
            (
                "- correct_z: [0, c: 'circuit.readouts[0]']",
                "- correct_z: {arguments: {c: 'circuit.readouts[0]'}, operands: [0]}",
            ),
            (
                "- select: [select: 1, operands: 2, arguments: 3]",
                "- select: {arguments: {select: 1, operands: 2, arguments: 3}}",
            ),
        ] {
            let calls = parse_inline_yaml(object).unwrap();
            assert_eq!(parse_inline_yaml(shorthand).unwrap(), calls, "{shorthand}");
            assert_eq!(parse_inline_yaml(&emit_inline_yaml(&calls).unwrap()).unwrap(), calls);
        }
    }

    #[test]
    fn select_can_name_an_instruction_parameter_and_flag() {
        let source = "- select: {operands: [3], arguments: {select: 1}, select: [{select: 0}]}";
        let call = only_call(source);
        assert_eq!(call.mnemonic, "select");
        assert_eq!(call.operands, vec![Operand::Index(3)]);
        assert_eq!(call.arguments["select"], Argument::Integer(1));
        assert_eq!(call.select, vec![BTreeMap::from([("select".to_owned(), 0)])]);
        let emitted = emit_inline_yaml(std::slice::from_ref(&call)).unwrap();
        assert_eq!(only_call(&emitted), call);
        let value: serde_yaml::Value = serde_yaml::from_str(&emitted).unwrap();
        assert!(value[0]["select"].is_mapping());
        assert_eq!(value[0]["select"]["arguments"]["select"].as_u64(), Some(1));
    }

    #[test]
    fn selection_is_not_metadata_in_list_shorthand() {
        assert!(error("- gate: [0, select: [{reject: 0}]]").contains("unsupported value shape"));
        let call = only_call("- gate: [select: []]");
        assert_eq!(call.arguments["select"], Argument::QubitList(vec![]));
        assert!(call.select.is_empty());
    }

    #[test]
    fn malformed_call_objects_are_rejected() {
        for (source, expected) in [
            ("- gate: {operand: [0]}", "unknown call field"),
            ("- gate: {1: []}", "call field name must be a string"),
            ("- gate: {operands: null}", "`operands:` must be a YAML sequence"),
            (
                "- gate: {operands: [true]}",
                "an operand must be an integer index or a block name",
            ),
            ("- gate: {arguments: []}", "`arguments:` must be a YAML mapping"),
            ("- gate: {arguments: {1: 2}}", "argument name must be a string"),
            ("- gate: {arguments: {theta: {value: 1}}}", "unsupported value shape"),
            ("- gate: {arguments: {theta: 1, theta: 2}}", "duplicate"),
            ("- gate: {operands: [], operands: [0]}", "duplicate"),
            ("- gate: {select: null}", "`select:` must be a YAML sequence"),
        ] {
            assert!(error(source).contains(expected), "{source}");
        }
    }

    #[test]
    fn emitter_uses_objects_and_omits_empty_fields() {
        let source = parse_inline_yaml("- cx: [0, 1]\n- rotate_z: [0, theta: pi]\n- tick: []").unwrap();
        let emitted: serde_yaml::Value = serde_yaml::from_str(&emit_inline_yaml(&source).unwrap()).unwrap();
        let expected: serde_yaml::Value = serde_yaml::from_str(
            "- cx: {operands: [0, 1]}\n- rotate_z: {operands: [0], arguments: {theta: pi}}\n- tick: {}",
        )
        .unwrap();
        assert_eq!(emitted, expected);
    }
}
