use std::fmt::Debug;

use qodec::{ActionStep, Code, InstructionSet};
use serde::{Serialize, de::DeserializeOwned};

fn assert_yaml_round_trip<T>(yaml: &str)
where
    T: Debug + PartialEq + Serialize + DeserializeOwned,
{
    let decoded: T = serde_yaml::from_str(yaml).expect("fixture should deserialize");
    let encoded = serde_yaml::to_string(&decoded).expect("value should serialize");
    let decoded_again: T = serde_yaml::from_str(&encoded).expect("serialized value should deserialize");
    assert_eq!(decoded, decoded_again);
}

#[test]
fn code_yaml_round_trip() {
    assert_yaml_round_trip::<Code>(include_str!("../examples/c4c6/c4.code.yaml"));
}

#[test]
fn instruction_set_yaml_round_trip() {
    assert_yaml_round_trip::<InstructionSet>(PARAMETERIZED_INSTRUCTION_SET);
}

#[test]
fn action_unknown_fields_are_rejected() {
    for source in [
        r#"{"stabilize": ["Z_0"], "typo": true}"#,
        r#"{"clifford": {"X_0": "Z_0"}, "typo": true}"#,
        r#"{"pauli": "X_0", "typo": true}"#,
        r#"{"observe": "Z_0", "typo": true}"#,
        r#"{"rotate": {"pauli": "Z_0", "angle": 1.0}, "typo": true}"#,
    ] {
        let yaml_error = serde_yaml::from_str::<ActionStep>(source).expect_err("unknown action field");
        assert!(yaml_error.to_string().contains("typo"), "{source}: {yaml_error}");
        let json_error = serde_json::from_str::<ActionStep>(source).expect_err("unknown action field");
        assert!(json_error.to_string().contains("typo"), "{source}: {json_error}");
    }
}

#[test]
fn action_rotate_unknown_fields_are_rejected() {
    let source = r#"{"rotate": {"pauli": "Z_0", "angle": 1.0, "typo": true}, "if": []}"#;
    let yaml_error = serde_yaml::from_str::<ActionStep>(source).expect_err("unknown rotation field");
    assert!(yaml_error.to_string().contains("typo"), "{yaml_error}");
    let json_error = serde_json::from_str::<ActionStep>(source).expect_err("unknown rotation field");
    assert!(json_error.to_string().contains("typo"), "{json_error}");
}

#[test]
fn action_empty_drafts_and_conditions_round_trip() {
    for action in [
        "stabilize: []",
        "clifford: {}",
        "pauli: ''",
        "observe: []",
        "rotate: {pauli: '', angle: ''}",
    ] {
        for condition in [
            "",
            ", if: []",
            ", unless: []",
            ", if: [enabled]",
            ", unless: ['outcomes[0]']",
        ] {
            assert_yaml_round_trip::<ActionStep>(&format!("{{{action}{condition}}}"));
        }
    }
}

#[test]
fn action_if_and_unless_together_are_rejected() {
    let error = serde_yaml::from_str::<ActionStep>("{pauli: X_0, if: [], unless: []}")
        .expect_err("both action guards must be rejected");
    assert!(
        error.to_string().contains("cannot have both `if` and `unless`"),
        "{error}"
    );
}

const PARAMETERIZED_INSTRUCTION_SET: &str = r#"
name: C4
description: Logical instruction set for the [[4,2,2]] error-detecting code

blocks: {c4: 2}

instructions:
  - mnemonic: prepare_zz
    description: "Prepare a C4 block in |00>"
    out: [c4]
    action:
      - stabilize: ["Z_0", "Z_1"]

  - mnemonic: transversal_cx
    description: Transversal CNOT between two C4 blocks
    in: [c4, c4]
    out: [c4, c4]
    parameters: {enabled: bit, theta: number}
    action:
      - clifford:
          X_0: X_0 X_2
          Z_2: Z_0 Z_2
      - rotate:
          pauli: "Z_0"
          angle: theta
        if: [enabled]
"#;

/// `metadata` is an arbitrary, qodec-opaque mapping that survives a
/// serde round trip on every definition that carries it. Here it is
/// exercised at both the instruction-set level and on an individual
/// instruction, with nested scalars, sequences, and maps.
fn instruction_set_with_nested_metadata() -> InstructionSet {
    serde_yaml::from_str(
        r#"
name: C4
description: ""
blocks: {c4: 2}
metadata:
  backend: ion-trap-A
  references: ["arXiv:2207.06431"]
instructions:
  - mnemonic: prepare_zz
    description: ""
    out: [c4]
    action:
      - stabilize: ["Z_0", "Z_1"]
    metadata:
      duration_ns: 800
      vendor: {ibm: {native: true}}
"#,
    )
    .expect("instruction_set with metadata should deserialize")
}

#[test]
fn metadata_round_trips() {
    let instruction_set = instruction_set_with_nested_metadata();
    assert!(
        !instruction_set.metadata.is_empty(),
        "instruction_set metadata should be populated"
    );
    assert!(
        !instruction_set.instructions[0].metadata.is_empty(),
        "instruction metadata should be populated"
    );

    let encoded = serde_yaml::to_string(&instruction_set).expect("value should serialize");
    let decoded_again: InstructionSet = serde_yaml::from_str(&encoded).expect("serialized value should deserialize");
    assert_eq!(instruction_set, decoded_again, "metadata must survive the round trip");
    assert!(
        encoded.contains("backend") && encoded.contains("duration_ns"),
        "serialized form should carry the metadata keys: {encoded}"
    );
}

/// `Metadata` is a JSON object, which is what the schemas have always declared
/// (`"type": "object"`). Two consequences worth pinning: keys come back sorted
/// rather than in author order, and a non-string key is coerced to a string
/// rather than preserved.
fn instruction_set_with_numeric_metadata_key() -> InstructionSet {
    serde_yaml::from_str(
        r#"
name: C4
description: ""
blocks: {c4: 2}
metadata:
  zeta: 1
  alpha: {nested: [1, 2, true]}
  7: numeric-key
instructions: []
"#,
    )
    .expect("metadata should deserialize")
}

#[test]
fn metadata_is_a_string_keyed_json_object() {
    let instruction_set = instruction_set_with_numeric_metadata_key();
    assert_eq!(
        instruction_set.metadata.keys().collect::<Vec<_>>(),
        ["7", "alpha", "zeta"],
        "keys are sorted, and a numeric key becomes a string"
    );
    assert_eq!(
        instruction_set.metadata["alpha"]["nested"],
        serde_json::json!([1, 2, true]),
        "nested sequences and scalars survive"
    );

    let encoded = serde_yaml::to_string(&instruction_set).expect("value should serialize");
    let again: InstructionSet = serde_yaml::from_str(&encoded).expect("re-read");
    assert_eq!(instruction_set, again, "a second pass is stable");
}

/// Empty `metadata` is skipped on serialize, so a definition without
/// metadata emits no `metadata:` key (and still round-trips).
#[test]
fn empty_metadata_is_omitted() {
    let code: Code =
        serde_yaml::from_str(include_str!("../examples/c4c6/c4.code.yaml")).expect("code fixture should deserialize");
    assert!(code.metadata.is_empty(), "fixture has no metadata");
    let encoded = serde_yaml::to_string(&code).expect("code should serialize");
    assert!(
        !encoded.contains("metadata"),
        "empty metadata must not be serialized: {encoded}"
    );
}

/// `metadata` participates in structural equality like any other field:
/// two definitions that differ only in `metadata` are not equal.
#[test]
fn metadata_participates_in_equality() {
    let base = r#"
name: rep
stabilizers: ["Z_0 Z_1"]
x: ["X_0 X_1"]
z: ["Z_0"]
"#;
    let with_metadata = r#"
name: rep
stabilizers: ["Z_0 Z_1"]
x: ["X_0 X_1"]
z: ["Z_0"]
metadata: {provenance: "doi:10.0/x"}
"#;
    let plain: Code = serde_yaml::from_str(base).expect("base code should deserialize");
    let annotated: Code = serde_yaml::from_str(with_metadata).expect("annotated code should deserialize");
    assert_ne!(plain, annotated, "metadata difference must make codes unequal");
}

/// A non-mapping `metadata:` value is rejected (the object-only rule).
#[test]
fn non_mapping_metadata_is_rejected() {
    let bad = r#"
name: rep
stabilizers: ["Z_0 Z_1"]
x: ["X_0 X_1"]
z: ["Z_0"]
metadata: 5
"#;
    let result: Result<Code, _> = serde_yaml::from_str(bad);
    assert!(result.is_err(), "a scalar metadata value must be rejected");
}

/// `blocks:` and instruction `parameters:` use a plain-map form
/// (`{c4: 2}`, `{theta: number}`).
fn instruction_set_with_map_parameters() -> InstructionSet {
    serde_yaml::from_str(
        r#"
name: C4
description: ""
blocks: {c4: 2, c6: 3}
instructions:
  - mnemonic: rotate
    description: ""
    in: [c4]
    out: [c4]
    parameters: {theta: number, phi: number}
    action:
      - rotate:
          pauli: "Z_0"
          angle: theta
"#,
    )
    .expect("map form should deserialize")
}

#[test]
fn block_and_parameters_use_map_form() {
    let map_form = instruction_set_with_map_parameters();
    assert_eq!(map_form.blocks.len(), 2);

    let sequence_form: Result<InstructionSet, _> = serde_yaml::from_str(
        r#"
name: C4
description: ""
blocks:
  - {c4: 2}
  - {c6: 3}
instructions: []
"#,
    );
    assert!(
        sequence_form.is_err(),
        "blocks must be a map, not a sequence of single-key maps"
    );
}
