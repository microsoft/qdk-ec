//! Validate the explicitly listed example manifests and the artifacts they
//! reference. Reference fields select schemas; filenames do not select types.

#[path = "../common/mod.rs"]
mod common;

use super::{Document, ingest_artifacts, load_manifest, parse_bundle_str};
use jsonschema::Validator;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn load_schema(filename: &str) -> Validator {
    let path = repo_root().join("schemas").join(filename);
    let raw =
        std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("could not read {}: {error}", path.display()));
    let schema: serde_json::Value = serde_json::from_str(&raw)
        .unwrap_or_else(|error| panic!("schema {} is not valid JSON: {error}", path.display()));
    Validator::new(&schema)
        .unwrap_or_else(|error| panic!("schema {} is not a valid JSON schema: {error}", path.display()))
}

#[test]
fn empty_action_payloads_and_predicates_match_schema() {
    let validator = load_schema("instruction_set_schema.json");
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("instruction_set.yaml");
    for action in [
        serde_json::json!({"observe": []}),
        serde_json::json!({"stabilize": []}),
        serde_json::json!({"stabilize": ""}),
        serde_json::json!({"stabilize": [""]}),
        serde_json::json!({"clifford": {}}),
        serde_json::json!({"clifford": {"X_0": ""}}),
        serde_json::json!({"clifford": {"": ""}}),
        serde_json::json!({"pauli": ""}),
        serde_json::json!({"rotate": {"pauli": "", "angle": ""}}),
    ] {
        for guard in [None, Some("if"), Some("unless")] {
            let mut step = action.clone();
            if let Some(guard) = guard {
                step[guard] = serde_json::json!([]);
            }
            let value = serde_json::json!({
                "name": "draft", "blocks": {},
                "instructions": [{"mnemonic": "idle", "description": "", "action": [step]}]
            });
            std::fs::write(&path, serde_yaml::to_string(&value).unwrap()).unwrap();
            let loaded = crate::InstructionSet::load(&path);
            assert!(loaded.is_ok(), "{value}: {loaded:?}");
            assert!(validator.is_valid(&value), "schema rejected {value}");
        }
    }
}

#[test]
fn code_schema_and_loader_agree_on_pauli_tokens() {
    let validator = load_schema("code_definition_schema.json");
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("code.yaml");
    let cases = [
        ("X_0 Z_1", true),
        ("I X Y Z", true),
        ("I_7", true),
        ("X_+00 Y_02", true),
        ("\tX_0\nZ_1\r", true),
        ("X\u{a0}Y\u{2003}Z", true),
        ("X\u{85}Y", true),
        ("X\u{feff}Y", false),
        ("", true),
        (" \t\n", true),
        ("-Z_0", false),
        ("+Z_0", false),
        ("-", false),
        ("+", false),
        ("X_0\n-Z_1", false),
        ("Q_0", false),
        ("x_0", false),
        ("target.X_0", false),
        ("X_", false),
        ("X_-1", false),
        ("X_ 0", false),
        ("X_0Y_1", false),
        ("X_0_1", false),
        ("X_0.0", false),
    ];
    let mut schema_mismatches = Vec::new();
    for field in ["stabilizers", "x", "z"] {
        for (operator, accepted) in cases {
            let mut value = serde_json::json!({"name": "test", "stabilizers": [], "x": [], "z": []});
            value[field] = serde_json::json!([operator]);
            std::fs::write(&path, serde_yaml::to_string(&value).unwrap()).unwrap();
            let loaded = crate::Code::load(&path);
            assert_eq!(loaded.is_ok(), accepted, "{field}: {operator:?}: {loaded:?}");
            if validator.is_valid(&value) != accepted {
                schema_mismatches.push((field, operator));
            }
        }
    }
    assert!(schema_mismatches.is_empty(), "{schema_mismatches:?}");
}

#[test]
fn code_index_limits_are_checked_only_by_the_loader() {
    let validator = load_schema("code_definition_schema.json");
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("code.yaml");
    for (index, accepted) in [
        ((usize::MAX - 1).to_string(), true),
        (usize::MAX.to_string(), false),
        (format!("{}0", usize::MAX), false),
    ] {
        let value = serde_json::json!({"name": "draft", "stabilizers": [format!("X_{index}")], "x": [], "z": []});
        assert!(validator.is_valid(&value), "schema rejected lexical token {index}");
        std::fs::write(&path, serde_yaml::to_string(&value).unwrap()).unwrap();
        let loaded = crate::Code::load(&path);
        assert_eq!(loaded.is_ok(), accepted, "{index}: {loaded:?}");
    }
}

#[test]
fn code_schema_and_loader_preserve_algebraic_drafts() {
    let value = serde_json::json!({"name": "draft", "stabilizers": ["X", "Z"], "x": ["X"], "z": []});
    let validator = load_schema("code_definition_schema.json");
    assert!(validator.is_valid(&value));
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("code.yaml");
    std::fs::write(&path, serde_yaml::to_string(&value).unwrap()).unwrap();
    crate::Code::load(path).expect("noncommuting stabilizers and unequal X/Z lists remain loadable");
}

fn yaml_to_json(value: &serde_yaml::Value, label: &str) -> serde_json::Value {
    serde_json::to_value(value).unwrap_or_else(|error| panic!("{label} could not be transcoded to JSON: {error}"))
}

struct ValidationUnit {
    label: String,
    role: &'static str,
    json: serde_json::Value,
}

fn validation_units(path: &Path) -> Vec<ValidationUnit> {
    let mut referenced = BTreeMap::new();
    let mut visit = |relative: &Path, kind: &'static str, document: &Document| {
        let role = match kind {
            "instruction set" => "instruction_set",
            "check list" => "checks",
            "readout list" => "readouts",
            "circuit source" => "source",
            other => other,
        };
        let value = match document {
            Document::Parsed(value) => value.clone(),
            Document::Text(text) if role == "source" => serde_yaml::Value::String(text.clone()),
            Document::Text(text) => serde_yaml::from_str(text).expect("document YAML"),
        };
        referenced.insert((role, relative.to_owned()), value);
    };
    let (manifest, filename, documents, origins) = load_manifest(path, &mut visit).expect("load manifest");
    ingest_artifacts(
        path.parent().unwrap(),
        &manifest,
        &filename,
        documents,
        origins,
        &mut visit,
    )
    .expect("load referenced artifacts");
    referenced
        .into_iter()
        .map(|((role, relative), value)| {
            let label = format!("{}#{} ({role})", path.display(), relative.display());
            let json = yaml_to_json(&value, &label);
            if matches!(role, "checks" | "readouts") {
                assert!(json.is_array(), "{label}: a referenced parity list must be an array");
            }
            let json = match role {
                "checks" => serde_json::json!({"circuit": [], "checks": json}),
                "readouts" => serde_json::json!({"circuit": [], "readouts": json}),
                "source" => serde_json::json!({"circuit": {"source": json}}),
                _ => json,
            };
            ValidationUnit { label, role, json }
        })
        .collect()
}

const VALID_CALL_SOURCES: &[&str] = &[
    "- tick: {}",
    "- tick: []",
    "- tick: null",
    "- tick: {operands: [], arguments: {}, select: []}",
    "- cx: {operands: [0, block]}",
    "- cx: [0, block]",
    "- rotate_z: {operands: [0], arguments: {theta: 1.5708}}",
    "- rotate_z: [0, theta: 1.5708]",
    "- gate: {arguments: {indices: [0, 1], names: [a, b], empty: [], count: -1}}",
    "- gate: [0, {indices: [0, 1], names: [a, b], empty: [], count: -1}]",
    "- gate: {arguments: {enabled: true, disabled: false, text: 'true', names: ['true', 'false']}}",
    "- gate: [0, {enabled: true, disabled: false, text: 'false', names: ['true', 'false']}]",
    "- correct_z: {arguments: {c: 'circuit.readouts[0]'}}",
    "- prepare: {operands: [3], select: [reject: 0, leak: 1]}",
    "- select: {arguments: {select: 1, operands: 2, arguments: 3}, select: [{select: 0}]}",
    "- select: [select: 1, operands: 2, arguments: 3]",
    "- select: {arguments: {select: true, operands: false, arguments: true}, select: [{select: 0}]}",
    "- select: [select: true, operands: false, arguments: true]",
    "- prepare: {select: [{}]}",
];

fn gadget_source_forms(source: &str) -> [serde_json::Value; 2] {
    let calls: serde_yaml::Value = serde_yaml::from_str(source).unwrap();
    let calls = yaml_to_json(&calls, source);
    [
        serde_json::json!({"circuit": calls}),
        serde_json::json!({"circuit": {"source": calls}}),
    ]
}

#[test]
fn inline_call_forms_match_the_schema() {
    let validator = load_schema("gadget_schema.json");
    for source in VALID_CALL_SOURCES {
        for gadget in gadget_source_forms(source) {
            let errors: Vec<_> = validator.iter_errors(&gadget).collect();
            assert!(errors.is_empty(), "{source}: {errors:?}");
        }
    }
}

const MALFORMED_CALL_SOURCES: &[&str] = &[
    "- {}",
    "- {first: [], second: []}",
    "- gate: 3",
    "- gate: {operand: [0]}",
    "- gate: {operands: null}",
    "- gate: {operands: [true]}",
    "- gate: {operands: [false]}",
    "- gate: [true]",
    "- gate: [0, false]",
    "- gate: {operands: [-1]}",
    "- gate: {arguments: []}",
    "- gate: {arguments: {value: [true, false]}}",
    "- gate: [0, value: [true, false]]",
    "- gate: {arguments: {value: [0, true]}}",
    "- gate: [0, value: [false, 1]]",
    "- gate: {arguments: {value: null}}",
    "- gate: {arguments: {value: {nested: 1}}}",
    "- gate: {arguments: {value: [1, a]}}",
    "- gate: [0, value: [1, a]]",
    "- gate: [0, select: [{reject: 0}]]",
    "- gate: {select: null}",
    "- gate: {select: [1]}",
    "- gate: {select: [{reject: true}]}",
    "- gate: {select: [{reject: false}]}",
    "- gate: {select: [{reject: 2}]}",
    "- gate: {select: {reject: 0}}",
    "- gate: {select: {}}",
];

fn yaml_circuit(source: &str) -> crate::Circuit {
    crate::Circuit {
        instruction_set: std::sync::Arc::new(serde_yaml::from_str("name: test\nblocks: {}\ninstructions: []").unwrap()),
        source: source.to_owned(),
        format: Some("yaml".to_owned()),
    }
}

#[test]
fn malformed_inline_calls_are_stored_but_fail_interpretation() {
    let validator = load_schema("gadget_schema.json");
    for source in MALFORMED_CALL_SOURCES {
        for gadget in gadget_source_forms(source) {
            assert!(validator.is_valid(&gadget), "could not store {source}");
        }
        let circuit = yaml_circuit(source);
        assert!(circuit.calls().is_err(), "interpreted malformed call {source}");
    }
}

#[test]
fn parity_lists_can_be_inline_or_referenced() {
    let validator = load_schema("gadget_schema.json");
    for field in ["checks", "readouts"] {
        for value in [
            serde_json::json!([]),
            serde_json::json!([["circuit.readouts[0]"]]),
            serde_json::json!([["readouts[00:03]", "out[01].z[3, 1,3]", "circuit.readouts[00:03:2]"]]),
            serde_json::json!("../parities"),
        ] {
            let mut gadget = serde_json::json!({"circuit": []});
            gadget[field] = value;
            let errors: Vec<_> = validator.iter_errors(&gadget).collect();
            assert!(errors.is_empty(), "{gadget}: {errors:?}");
        }
        for value in [
            serde_json::json!(null),
            serde_json::json!(true),
            serde_json::json!({}),
            serde_json::json!([["in.z[0]"]]),
            serde_json::json!([["in[0:2].z[0]"]]),
            serde_json::json!([["circuit.flags[0]"]]),
        ] {
            let mut gadget = serde_json::json!({"circuit": []});
            gadget[field] = value;
            assert!(!validator.is_valid(&gadget), "accepted {gadget}");
        }
    }
}

#[test]
fn reference_schema_and_parser_agree_on_selector_spelling() {
    let validator = load_schema("gadget_schema.json");
    let mut schema_mismatches = Vec::new();
    for (expression, accepted) in [
        ("readouts[00]", true),
        ("readouts[+01]", true),
        ("circuit.readouts[ 00 : +03 ]", true),
        ("circuit.readouts[\t00 :\n+03 : +02\r]", true),
        ("readouts[\u{a0}0:3\u{2003}]", true),
        ("readouts[\u{85}0:3\u{85}]", true),
        ("readouts[\u{feff}0:3\u{feff}]", false),
        ("readouts[ +00, 02 ,+01 ]", true),
        ("in[+01].x[+00]", true),
        ("out[01].z[ 00 : 03 : 02 ]", true),
        ("in[0].stabilizers[0, 1]", true),
        ("readouts[ 0]", false),
        ("readouts[0 ]", false),
        ("in[ 0].x[0]", false),
        ("out[0 ].z[0]", false),
        ("readouts[-1]", false),
        ("readouts[0:-3]", false),
        ("readouts[0:3:-1]", false),
        ("readouts[+]", false),
        ("readouts[]", false),
        ("readouts[:3]", false),
        ("readouts[0:]", false),
        ("readouts[0:3:]", false),
        ("readouts[0:3:1:2]", false),
        ("readouts[0,]", false),
        ("readouts[0,1:2]", false),
        ("readouts[0]\n", false),
    ] {
        let parsed = crate::Reference::parse(expression);
        assert_eq!(parsed.is_ok(), accepted, "{expression:?}: {parsed:?}");
        let gadget = serde_json::json!({"circuit": [], "checks": [[expression]]});
        if validator.is_valid(&gadget) != accepted {
            schema_mismatches.push(expression);
        }
    }
    assert!(schema_mismatches.is_empty(), "{schema_mismatches:?}");
}

#[test]
fn reference_numeric_limits_are_checked_only_by_the_parser() {
    let validator = load_schema("gadget_schema.json");
    let limit = crate::parity::MAX_SELECTED_POSITIONS;
    for (selector, accepted) in [
        ("0:0".to_owned(), false),
        ("3:2".to_owned(), false),
        ("0:3:0".to_owned(), false),
        (format!("0:{limit}"), true),
        (format!("0:{}", limit + 1), false),
        (format!("0:{}:2", limit + 1), true),
        (usize::MAX.to_string(), true),
        (format!("0:{}", usize::MAX), false),
        (format!("{}0", usize::MAX), false),
        (format!("0:1:{}0", usize::MAX), false),
    ] {
        let expression = format!("circuit.readouts[{selector}]");
        let gadget = serde_json::json!({"circuit": [], "checks": [[expression]]});
        assert!(
            validator.is_valid(&gadget),
            "schema rejected lexical selector {selector}"
        );
        let parsed = crate::Reference::parse(&expression);
        assert_eq!(parsed.is_ok(), accepted, "{expression}: {parsed:?}");
    }
}

#[test]
fn parity_bits_match_the_schema_without_accepting_booleans() {
    let validator = load_schema("gadget_schema.json");
    for (term, accepted) in [
        (serde_json::json!(0), true),
        (serde_json::json!(1), true),
        (serde_json::json!(true), false),
        (serde_json::json!(false), false),
        (serde_json::json!(2), false),
        (serde_json::json!(-1), false),
        (serde_json::json!("0"), false),
    ] {
        for field in ["checks", "readouts"] {
            let mut gadget = serde_json::json!({"circuit": []});
            gadget[field] = serde_json::json!([[term]]);
            assert_eq!(validator.is_valid(&gadget), accepted, "{gadget}");
            let parsed = serde_json::from_value::<crate::GadgetSpec>(gadget.clone());
            assert_eq!(parsed.is_ok(), accepted, "{gadget}: {parsed:?}");
        }
    }
}

#[test]
fn implements_schema_and_parser_split_at_the_last_hash() {
    let validator = load_schema("gadget_schema.json");
    let mut schema_mismatches = Vec::new();
    for (reference, accepted) in [
        ("isa#idle", true),
        ("dir/isa#revision.yaml#idle", true),
        ("#isa##idle", true),
        ("isa\nrevision#idle", true),
        ("isa", false),
        ("#idle", false),
        ("isa#", false),
        ("isa#revision#", false),
        ("", false),
    ] {
        let gadget = serde_json::json!({"circuit": [], "implements": reference});
        let parsed = serde_json::from_value::<crate::GadgetSpec>(gadget.clone());
        assert_eq!(parsed.is_ok(), accepted, "{reference:?}: {parsed:?}");
        if validator.is_valid(&gadget) != accepted {
            schema_mismatches.push(reference);
        }
    }
    assert!(schema_mismatches.is_empty(), "{schema_mismatches:?}");
}

fn artifact_validators() -> BTreeMap<&'static str, Validator> {
    [
        ("manifest", "manifest_schema.json"),
        ("instruction_set", "instruction_set_schema.json"),
        ("code", "code_definition_schema.json"),
        ("gadget", "gadget_schema.json"),
    ]
    .into_iter()
    .map(|(role, schema)| (role, load_schema(schema)))
    .collect()
}

fn validate_referenced_artifacts(
    manifests: &[PathBuf],
    validators: &BTreeMap<&str, Validator>,
) -> (Vec<String>, BTreeMap<&'static str, usize>) {
    let mut failures: Vec<String> = Vec::new();
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();

    for path in manifests {
        for unit in validation_units(path) {
            *counts.entry(unit.role).or_insert(0) += 1;
            let schema_role = match unit.role {
                "checks" | "readouts" | "source" => "gadget",
                role => role,
            };
            let validator = validators.get(schema_role).expect("validator exists");
            let errors: Vec<_> = validator.iter_errors(&unit.json).collect();
            if !errors.is_empty() {
                let detail = errors
                    .iter()
                    .map(|error| format!("    at /{}: {}", error.instance_path, error))
                    .collect::<Vec<_>>()
                    .join("\n");
                failures.push(format!("{}:\n{detail}", unit.label));
            }
        }
    }
    (failures, counts)
}

#[test]
fn hash_filenames_load_and_match_referenced_schemas() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("bundle.yaml");
    let documents = [
        serde_json::json!({"manifest": {"layers": [
            {"instruction_set": "isa#revision", "gadgets": {"idle": "gadget"}},
            {"instruction_set": "isa#revision"}
        ]}}),
        serde_json::json!({"isa#revision": {"name": "draft", "blocks": {}, "instructions": [
            {"mnemonic": "idle", "description": ""}
        ]}}),
        serde_json::json!({"gadget": {"implements": "isa#revision#idle", "circuit": []}}),
    ];
    let bundle = documents
        .iter()
        .map(|document| serde_yaml::to_string(document).unwrap())
        .collect::<Vec<_>>()
        .join("\n---\n");
    std::fs::write(&path, bundle).unwrap();
    crate::Qodec::load(&path).expect("the last hash separates the referenced file from the mnemonic");
    let (failures, counts) = validate_referenced_artifacts(&[path], &artifact_validators());
    assert!(failures.is_empty(), "{failures:?}");
    assert_eq!(counts.values().sum::<usize>(), 3);
}

#[test]
fn examples_match_schemas() {
    let validators = artifact_validators();
    let manifests = common::example_manifests();
    assert!(
        manifests.len() >= 10,
        "the example corpus is walked, not listed; {} manifests found",
        manifests.len()
    );
    let (failures, counts) = validate_referenced_artifacts(&manifests, &validators);
    assert_eq!(counts.get("manifest"), Some(&manifests.len()));
    for role in validators.keys() {
        assert!(
            counts.get(role).is_some_and(|count| *count > 0),
            "no {role} artifacts validated"
        );
    }

    assert!(
        failures.is_empty(),
        "{} example file(s) failed schema validation:\n{}",
        failures.len(),
        failures.join("\n\n")
    );
    println!("Schema-validated referenced artifacts by role: {counts:?}");
}

const ROLE_REFERENCE_BUNDLE: &str = concat!(
    "./draft/../catalog/entry:\n",
    "  layers:\n",
    "    - instruction_set: &shared ../definitions/../.hidden/isa.code.yaml\n",
    "      codes: {q: ../code.gadget.yaml}\n",
    "      gadgets: {idle: ../gates/gadget.code.yaml}\n",
    "    - instruction_set: *shared\n",
    r#"
---
./.hidden/isa.code.yaml:
  name: instruction_set
  description: test
  blocks: {q: 1}
  instructions: []
---
code.gadget.yaml:
  name: code
  description: test
  stabilizers: []
  x: [X_0]
  z: [Z_0]
---
gates/./gadget.code.yaml:
  circuit: {source: ../target/source}
  checks: ../metadata/./checks.isa.yaml
  readouts: ../metadata/readouts
---
./metadata/checks.isa.yaml: [["circuit.readouts[0]"]]
---
metadata/readouts: [{result: ["circuit.readouts[0]"]}]
---
target/./source: [{R: [0]}]
---
unused.qodec.yaml: {not: a manifest}
---
unused.gadget.yaml: {not: a gadget}
"#,
);

fn write_role_reference_fixture(root: &Path, bundled: bool) -> PathBuf {
    let path = root.join(if bundled { "protocol.txt" } else { "catalog/entry" });
    if bundled {
        std::fs::write(&path, ROLE_REFERENCE_BUNDLE).expect("write bundle");
    } else {
        for (key, body) in parse_bundle_str(ROLE_REFERENCE_BUNDLE, &path).unwrap().unwrap() {
            let destination = root.join(key);
            std::fs::create_dir_all(destination.parent().expect("document directory")).expect("mkdir");
            std::fs::write(destination, serde_yaml::to_string(&body).expect("serialize document"))
                .expect("write document");
        }
        std::fs::write(root.join("bad.isa.yaml"), "[invalid YAML\n").expect("write ignored file");
    }
    path
}

fn assert_normalized_artifact_roles(units: &[ValidationUnit], path: &Path, bundled: bool) {
    assert_eq!(
        units.iter().map(|unit| unit.role).collect::<Vec<_>>(),
        [
            "checks",
            "code",
            "gadget",
            "instruction_set",
            "manifest",
            "readouts",
            "source"
        ]
    );
    for (unit, relative) in units.iter().zip([
        "metadata/checks.isa.yaml",
        "code.gadget.yaml",
        "gates/gadget.code.yaml",
        ".hidden/isa.code.yaml",
        if bundled { "catalog/entry" } else { "entry" },
        "metadata/readouts",
        "target/source",
    ]) {
        let relative = if !bundled && unit.role != "manifest" {
            format!("../{relative}")
        } else {
            relative.to_owned()
        };
        assert_eq!(unit.label, format!("{}#{relative} ({})", path.display(), unit.role));
    }
}

#[test]
fn validation_follows_roles_and_normalized_paths_in_both_layouts() {
    for bundled in [false, true] {
        let directory = tempfile::tempdir().expect("temp dir");
        let path = write_role_reference_fixture(directory.path(), bundled);
        let units = validation_units(&path);
        assert_normalized_artifact_roles(&units, &path, bundled);
        assert_eq!(
            units[4].json["layers"][0]["instruction_set"],
            "../definitions/../.hidden/isa.code.yaml"
        );
        assert_eq!(units[2].json["circuit"]["source"], "../target/source");
        assert_eq!(units[2].json["checks"], "../metadata/./checks.isa.yaml");
        assert_eq!(units[0].json["checks"], serde_json::json!([["circuit.readouts[0]"]]));
        assert_eq!(
            units[5].json["readouts"],
            serde_json::json!([{"result": ["circuit.readouts[0]"]}])
        );
        let source = &units[6].json["circuit"]["source"];
        if bundled {
            assert_eq!(source, &serde_json::json!([{"R": [0]}]));
        } else {
            assert!(source.is_string(), "file-backed source is verbatim text");
        }
    }
}

#[test]
fn schemas_reject_unknown_fields() {
    for (schema, mut value) in [
        (
            "manifest_schema.json",
            serde_json::json!({"layers": [{"instruction_set": "top"}, {"instruction_set": "physical"}]}),
        ),
        (
            "instruction_set_schema.json",
            serde_json::json!({"name": "instruction_set", "description": "test", "blocks": {"q": 1}, "instructions": []}),
        ),
        (
            "code_definition_schema.json",
            serde_json::json!({
                "name": "code", "description": "test", "stabilizers": [], "x": ["X_0"], "z": ["Z_0"]
            }),
        ),
        ("gadget_schema.json", serde_json::json!({"circuit": []})),
    ] {
        let validator = load_schema(schema);
        assert!(validator.is_valid(&value), "{schema}: baseline must be valid");
        value["unknown_field"] = serde_json::json!(true);
        assert!(!validator.is_valid(&value), "{schema}: unknown field must be rejected");
    }
}

#[test]
fn malformed_action_shapes_fail_schema_and_deserialization() {
    let validator = load_schema("instruction_set_schema.json");
    for source in [
        "{}",
        "{observe: ''}",
        "{observe: ['']}",
        "{observe: {result: Z_0}}",
        "{observe: true}",
        "{stabilize: true}",
        "{clifford: {X_0: true}}",
        "{pauli: true}",
        "{rotate: {pauli: X_0, angle: true}}",
        "{pauli: X_0, if: [], unless: []}",
        "{pauli: X_0, if: [true]}",
        "{pauli: X_0, unless: [false]}",
    ] {
        let step: serde_yaml::Value = serde_yaml::from_str(source).unwrap();
        let value = serde_json::json!({"name": "draft", "blocks": {}, "instructions": [
            {"mnemonic": "idle", "description": "", "action": [yaml_to_json(&step, source)]}
        ]});
        assert!(!validator.is_valid(&value), "schema accepted {source}");
        assert!(
            serde_yaml::from_value::<crate::ActionStep>(step).is_err(),
            "Rust accepted {source}"
        );
    }
}

const UNKNOWN_ACTION_FIELDS: &[&str] = &[
    "{unknown: []}",
    "{observe: [], unknown: []}",
    "{stabilize: [], unknown: []}",
    "{clifford: {}, unknown: []}",
    "{pauli: X_0, unknown: []}",
    "{rotate: {pauli: X_0, angle: 0}, unknown: []}",
    "{rotate: {pauli: X_0, angle: 0, unknown: 0}}",
];

#[test]
fn action_unknown_fields_are_rejected_by_the_schema() {
    let validator = load_schema("instruction_set_schema.json");
    for source in UNKNOWN_ACTION_FIELDS {
        let step: serde_yaml::Value = serde_yaml::from_str(source).unwrap();
        let value = serde_json::json!({"name": "draft", "blocks": {}, "instructions": [
            {"mnemonic": "idle", "description": "", "action": [yaml_to_json(&step, source)]}
        ]});
        assert!(!validator.is_valid(&value), "schema accepted {source}");
    }
}

#[test]
fn action_unknown_fields_are_rejected_by_rust() {
    for source in UNKNOWN_ACTION_FIELDS {
        assert!(
            serde_yaml::from_str::<crate::ActionStep>(source).is_err(),
            "Rust accepted {source}"
        );
    }
}

#[test]
fn gadget_schema_accepts_named_readouts_and_single_term_parities() {
    let validator = load_schema("gadget_schema.json");
    let mut value = serde_json::json!({
        "circuit": [],
        "checks": [["in[0].stabilizers[0]"]],
        "readouts": [{"result": ["circuit.readouts[0]"]}]
    });
    assert!(validator.is_valid(&value));
    value["checks"] = serde_json::json!([["in.target.stabilizers[0]"]]);
    assert!(
        !validator.is_valid(&value),
        "named encoding references must be rejected"
    );
    value["checks"] = serde_json::json!(["in[0].stabilizers[0]"]);
    assert!(
        !validator.is_valid(&value),
        "a single term still needs its parity array"
    );
}
