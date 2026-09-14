use std::fs;
use std::path::Path;

use qodec::{Gadget, LoadError, Qodec, ReferenceTarget};

fn load_from_text_and_file(text: &str) -> [Result<Qodec, LoadError>; 2] {
    let directory = tempfile::tempdir().expect("create bundle fixture");
    let path = directory.path().join("bundle");
    fs::write(&path, text).expect("write bundle fixture");
    [Qodec::from_bundle_str(text), Qodec::load(path)]
}

fn idle(protocol: &Qodec) -> &Gadget {
    &protocol.layers()[0].gadgets["idle"]
}

fn protocol_with_source(format: &str, source: &str) -> Qodec {
    let mut protocol = Qodec::from_bundle_str(bundle()).expect("load fixture");
    let circuit = &mut protocol.layers_mut()[0].gadgets.get_mut("idle").unwrap().circuit;
    circuit.source.clear();
    circuit.source.push_str(source);
    circuit.format = Some(format.to_owned());
    protocol
}

fn rebuild_from_layers(protocol: &Qodec) -> Qodec {
    Qodec::new(None, None, protocol.layers().to_vec())
}

fn round_trip_bundle_text(protocol: &Qodec) -> Qodec {
    let text = protocol.to_bundle_string().expect("serialize bundle text");
    Qodec::from_bundle_str(&text).expect("reload bundle text")
}

fn save_layout(protocol: &Qodec, single_file: bool) -> tempfile::TempDir {
    let directory = tempfile::tempdir().expect("create save destination");
    if single_file {
        protocol.save_bundle(directory.path()).expect("save bundle");
    } else {
        protocol.save(directory.path()).expect("save separate artifacts");
    }
    directory
}

fn reload_saved_protocol(protocol: &Qodec, root: &Path) -> Qodec {
    Qodec::load(root.join(protocol.manifest_filename())).expect("reload saved manifest")
}

fn round_trip_all_save_forms(protocol: &Qodec) -> [Qodec; 3] {
    let separate = save_layout(protocol, false);
    let bundle = save_layout(protocol, true);
    [
        round_trip_bundle_text(protocol),
        reload_saved_protocol(protocol, separate.path()),
        reload_saved_protocol(protocol, bundle.path()),
    ]
}

fn inconsistent_model_cases() -> [(String, &'static str, &'static str); 2] {
    [
        (
            bundle()
                .replace("    in: [q]", "    in: [q, q]")
                .replace("  circuit: ../source", "  in: [{q: [0]}]\n  circuit: ../source"),
            "input encodings for 2 operands",
            "operations/run.isa.yaml",
        ),
        (
            bundle().replace(
                "  - instruction_set: definitions/physical",
                "  - instruction_set: definitions/physical\n    gadgets: {I: operations/run.isa.yaml}",
            ),
            "bottom layer must have no gadgets",
            "entry",
        ),
    ]
}

#[test]
fn loading_checks_resolved_model_consistency() {
    for (text, expected, source) in inconsistent_model_cases() {
        let error = Qodec::from_bundle_str(&text).expect_err("reject inconsistent layer model");
        let (path, error) = match error {
            LoadError::InvalidGadget { gadget, error } => (gadget, error),
            LoadError::InvalidQodec { manifest, error } => (manifest, error),
            other => panic!("expected consistency error, got {other}"),
        };
        assert_eq!(path, Path::new(source));
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn code_validation_reports_its_file_whether_or_not_a_gadget_uses_it() {
    let invalid: qodec::Code = serde_yaml::from_str("name: qubit\nstabilizers: []\nx: [X_0]\nz: [Q_0]\n").unwrap();
    let expected = invalid.validate().expect_err("malformed Pauli token");
    for used in [false, true] {
        let text = if used {
            bundle().to_owned()
        } else {
            bundle().replace("    gadgets: {idle: operations/run.isa.yaml}\n", "")
        };
        Qodec::from_bundle_str(&text).expect("valid fixture before changing the code");
        let text = text.replace("  z: [Z_0]", "  z: [Q_0]");
        for result in load_from_text_and_file(&text) {
            let LoadError::InvalidCode { code, error } = result.expect_err("check every referenced code") else {
                panic!("expected code validation error");
            };
            assert_eq!(code.to_str(), Some("code"));
            assert_eq!(error, expected);
        }
    }
}

#[test]
fn model_validation_preserves_instruction_set_file_context() {
    let text = bundle().replace(
        "  - mnemonic: I\n",
        "  - mnemonic: I\n    parameters: {value: number, value: bit}\n",
    );
    for result in load_from_text_and_file(&text) {
        let error = result
            .expect_err("duplicate names cannot resolve unambiguously")
            .to_string();
        assert!(error.contains("definitions/physical"), "{error}");
        assert!(error.contains("duplicate"), "{error}");
    }
}

#[test]
fn undeclared_parameter_bindings_are_preserved_for_audit() {
    let text = bundle().replace(
        "  circuit: ../source",
        "  parameter_bindings: {theta: circuit.source.theta}\n  circuit: ../source",
    );
    for result in load_from_text_and_file(&text) {
        let mut protocol = result.unwrap();
        protocol.layers_mut()[0].gadgets.get_mut("idle").unwrap().circuit.format = Some("stim".to_owned());
        let saved = Qodec::from_bundle_str(&protocol.to_bundle_string().unwrap()).unwrap();
        assert_eq!(saved.layers()[0].gadgets["idle"].parameter_bindings["theta"], "theta");
    }
}

#[test]
fn resolution_preserves_unknown_implements_error() {
    let text = bundle().replace(
        "gadgets: {idle: operations/run.isa.yaml}",
        "gadgets: {missing: operations/run.isa.yaml}",
    );
    let LoadError::GadgetUnknownImplements {
        gadget,
        implements,
        instruction_set,
    } = Qodec::from_bundle_str(&text).expect_err("resolve the layer's gadget membership")
    else {
        panic!("expected unknown implements error");
    };
    assert_eq!(gadget, Path::new("operations/run.isa.yaml"));
    assert_eq!(implements, "missing");
    assert_eq!(instruction_set, "logical");
}

#[test]
fn loaded_reference_expressions_retain_their_spelling_when_saved() {
    for inline in [false, true] {
        let loaded = Qodec::from_bundle_str(&bundle_with_selector_equations(inline)).expect("load selectors");
        assert_authored_selector_fields(idle(&loaded));
        let constructed = rebuild_from_layers(&loaded);
        for protocol in [&loaded, &constructed] {
            for saved in round_trip_all_save_forms(protocol) {
                assert_eq!(idle(&saved).checks, idle(&loaded).checks);
                assert_eq!(idle(&saved).readouts, idle(&loaded).readouts);
            }
        }
    }
}

fn bundle_with_selector_equations(inline: bool) -> String {
    let checks = "[['circuit.readouts[00:03]', 'out[00].z[00, 0,00]']]";
    let readouts = "[{reject: ['readouts[00]']}]";
    let text = bundle()
        .replace("circuit: ../source", "circuit: {format: stim, source: 'I 0'}")
        .replace(
            "  - mnemonic: I\n",
            "  - mnemonic: I\n    flags: [first, second, third]\n",
        )
        .replacen("    action: []", "    action: []\n    flags: [reject]", 1);
    if inline {
        text.replace("checks: ../checks", &format!("checks: {checks}"))
            .replace("readouts: ../readouts", &format!("readouts: {readouts}"))
    } else {
        text.replace("checks: []", &format!("checks: {checks}"))
            .replace("readouts: []", &format!("readouts: {readouts}"))
    }
}

fn assert_authored_selector_fields(gadget: &Gadget) {
    let [
        qodec::ParityTerm::Reference(record),
        qodec::ParityTerm::Reference(output),
    ] = gadget.checks[0].as_slice()
    else {
        panic!("expected reference terms");
    };
    assert_eq!(record.path(), "circuit.readouts[00:03]");
    assert_eq!(record.target(), ReferenceTarget::CircuitReadout);
    assert_eq!(record.indices().collect::<Vec<_>>(), [0, 1, 2]);
    assert_eq!(output.path(), "out[00].z[00, 0,00]");
    assert_eq!(output.indices().collect::<Vec<_>>(), [0, 0, 0]);
    assert_eq!(gadget.readouts[0].equation[0].to_string(), "readouts[00]");
}

#[test]
fn loading_rejects_invalid_reference_expressions() {
    for expression in ["circuit.readouts[2:2]", "out[0].x[3:1]", "readouts[0:3:0]", "checks[0]"] {
        for field in ["checks", "readouts"] {
            for inline in [false, true] {
                let text = if inline {
                    bundle().replace(&format!("{field}: ../{field}"), &format!("{field}: [['{expression}']]"))
                } else {
                    bundle().replace(&format!("{field}: []"), &format!("{field}: [['{expression}']]"))
                };
                for result in load_from_text_and_file(&text) {
                    let error = result.expect_err("invalid reference must be rejected").to_string();
                    assert!(error.contains(expression), "{field}: {expression} missing from {error}");
                }
            }
        }
    }
}

#[test]
fn uninterpretable_stim_source_round_trips_without_losing_text() {
    for source in ["NOT_A_GATE 0\n", "X 0\n", "CX 0\n"] {
        let protocol = protocol_with_source("stim", source);
        let constructed = rebuild_from_layers(&protocol);
        for original in [&protocol, &constructed] {
            original
                .validate()
                .expect("source interpretation is not required for persistence");
            let reloaded = round_trip_bundle_text(original);
            let circuit = &idle(&reloaded).circuit;
            assert_eq!(circuit.source, source);
            assert!(circuit.calls().is_err());
        }
    }
}

#[test]
fn loading_preserves_invalid_calls_until_inspection() {
    for (source, expected) in [
        ("'- missing: [0]'", "unknown instruction"),
        ("'- missing: {operands: [0]}'", "unknown instruction"),
        ("'- missing: {}'", "unknown instruction"),
        ("'- missing: [enabled: true]'", "unknown instruction"),
        ("'X 0'", "No source parser registered for '.stim'"),
        ("'CX 0'", "No source parser registered for '.stim'"),
    ] {
        let text = bundle().replace("source: |\n  I 0\n", &format!("source: {source}\n"));
        for result in load_from_text_and_file(&text) {
            let protocol = result.unwrap();
            let error = protocol.layers()[0].gadgets["idle"].circuit.calls().unwrap_err();
            assert!(error.contains(expected), "{source}: {error}");
        }
    }
}

#[test]
fn synthesized_yaml_format_round_trips_as_a_call_list() {
    for source in ["[]", "# no calls\n[]", "---\n[]"] {
        let protocol = rebuild_from_layers(&protocol_with_source("yaml", source));
        let text = protocol.to_bundle_string().expect("serialize YAML circuit");
        assert!(!text.contains("format: yaml"));
        for reloaded in round_trip_all_save_forms(&protocol) {
            let circuit = &idle(&reloaded).circuit;
            assert_eq!(circuit.effective_format(), "yaml");
            assert!(circuit.calls().expect("parse saved YAML").is_empty());
        }
    }
}

#[test]
fn synthesized_openqasm_preserves_opaque_source_in_every_save_form() {
    for source in ["OPENQASM 3.0;\nqubit[3] data;\n", "opaque text not parsed by qodec\n"] {
        let protocol = rebuild_from_layers(&protocol_with_source("openqasm", source));
        protocol.validate().expect("opaque source is allowed");
        for reloaded in round_trip_all_save_forms(&protocol) {
            let circuit = &idle(&reloaded).circuit;
            assert_eq!(circuit.source, source);
            assert_eq!(circuit.format.as_deref(), Some("openqasm"));
            assert!(circuit.calls().unwrap_err().contains("openqasm"));
        }
    }
}

#[test]
fn stim_source_survives_without_a_registered_parser() {
    let source = "REPEAT 3 {\nM 0\n}\nMPAD 0 1\n";
    let protocol = protocol_with_source("stim", source);
    for reloaded in round_trip_all_save_forms(&protocol) {
        let circuit = &idle(&reloaded).circuit;
        assert_eq!(circuit.source, source);
        assert_eq!(circuit.calls().unwrap_err(), "No source parser registered for '.stim'");
    }
}

#[test]
fn synthesis_preserves_unknown_formats_and_invalid_sources() {
    for (format, source, expected) in [
        ("custom", "I 0", "custom"),
        ("yaml", "[not valid yaml", "yaml"),
        ("stim", "NOT_A_GATE 0", "no source parser registered for '.stim'"),
        ("stim", "X 0", "no source parser registered for '.stim'"),
    ] {
        let protocol = rebuild_from_layers(&protocol_with_source(format, source));
        let saved = round_trip_bundle_text(&protocol);
        let circuit = &idle(&saved).circuit;
        assert_eq!(circuit.source, source);
        assert_eq!(circuit.effective_format(), format);
        let error = circuit.calls().unwrap_err();
        assert!(error.to_lowercase().contains(expected), "{error}");
    }
}

fn invalid_encoding_cases() -> [(Vec<&'static str>, Vec<&'static str>, &'static str); 3] {
    [
        (
            vec!["0", "0"],
            vec!["qubit", "other"],
            "conflicting block types 'qubit' and 'other'",
        ),
        (
            vec!["0", "1"],
            vec!["qubit"],
            "different support and block-type lengths",
        ),
        (vec!["0"], vec!["missing"], "undeclared circuit block type 'missing'"),
    ]
}

fn protocol_with_two_physical_block_types() -> Qodec {
    let loaded = Qodec::from_bundle_str(bundle()).unwrap();
    let mut layers = loaded.layers().to_vec();
    let target = std::sync::Arc::make_mut(&mut layers[1].instruction_set);
    target.blocks.push(qodec::Block {
        name: "other".to_owned(),
        encodes: 1,
    });
    layers[0].gadgets.get_mut("idle").unwrap().circuit.instruction_set = layers[1].instruction_set.clone();
    let protocol = Qodec::new(None, None, layers);
    protocol.validate().expect("valid baseline");
    protocol.to_bundle_string().expect("serializable baseline");
    protocol
}

fn assert_all_save_forms_reject(protocol: &Qodec, expected: &str) {
    let directory = tempfile::tempdir().unwrap();
    for result in [
        protocol.to_bundle_string().map(|_| ()),
        protocol.save(directory.path().join("separate")).map(|_| ()),
        protocol.save_bundle(directory.path().join("bundle")).map(|_| ()),
    ] {
        let error = result.expect_err("invalid model must not serialize");
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn encoding_support_validation_and_serialization_agree() {
    for (support, block_types, expected) in invalid_encoding_cases() {
        let mut protocol = protocol_with_two_physical_block_types();
        let encoding = &mut protocol.layers_mut()[0].gadgets.get_mut("idle").unwrap().inputs[0];
        encoding.support = support.into_iter().map(str::to_owned).collect();
        encoding.block_types = block_types.into_iter().map(str::to_owned).collect();
        let error = protocol.validate().expect_err("invalid encoding");
        assert!(error.contains(expected), "{error}");
        assert_all_save_forms_reject(&protocol, expected);
    }
}

fn write_extensionless_artifact_fixture(root: &Path) {
    fs::write(
        root.join("protocol"),
        "layers:\n- instruction_set: logical\n  codes: {q: encoding}\n  gadgets: {idle: implementation}\n- instruction_set: physical\n",
    )
    .expect("manifest");
    fs::write(
        root.join("logical"),
        "name: logical\nblocks: {q: 1}\ninstructions:\n- mnemonic: idle\n  description: Identity\n  in: [q]\n  out: [q]\n  action: []\n",
    )
    .expect("logical instruction set");
    fs::write(
        root.join("physical"),
        "name: physical\nblocks: {qubit: 1}\ninstructions:\n- mnemonic: I\n  description: Identity\n  in: [qubit]\n  out: [qubit]\n  action: []\n",
    )
    .expect("physical instruction set");
    fs::write(
        root.join("encoding"),
        "name: qubit\nstabilizers: []\nx: [X_0]\nz: [Z_0]\n",
    )
    .expect("code");
    fs::write(
        root.join("implementation"),
        "circuit: {format: stim, source: 'I 0'}\nchecks: parity-data\nreadouts: output-data\n",
    )
    .expect("gadget");
    fs::write(root.join("parity-data"), "[]\n").expect("checks");
    fs::write(root.join("output-data"), "[]\n").expect("readouts");
    fs::write(root.join("unreferenced.gadget.yaml"), "[invalid yaml").expect("unreferenced file");
}

#[test]
fn referenced_artifacts_are_typed_by_context() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let root = directory.path();
    write_extensionless_artifact_fixture(root);
    let protocol = Qodec::load(root.join("protocol")).expect("load by reference context");
    assert_eq!(protocol.layers().len(), 2);
    let idle = &protocol.layers()[0].gadgets["idle"];
    assert_eq!(idle.inputs[0].code.name, "qubit");
    assert!(idle.checks.is_empty());
    assert!(idle.readouts.is_empty());

    let saved = root.join("saved");
    protocol.save(&saved).expect("save arbitrary filenames");
    assert!(saved.join("implementation").is_file());
    assert!(!saved.join("unreferenced.gadget.yaml").exists());
    assert_eq!(Qodec::load(saved.join("protocol")).expect("reload"), protocol);
}

fn bundle() -> &'static str {
    "entry:\n  layers:\n  - instruction_set: definitions/logical.code.yaml\n    codes: {q: code}\n    gadgets: {idle: operations/run.isa.yaml}\n  - instruction_set: definitions/physical\n---\noperations/run.isa.yaml:\n  circuit: ../source\n  checks: ../checks\n  readouts: ../readouts\n---\nsource: |\n  I 0\n---\nchecks: []\n---\nreadouts: []\n---\ncode:\n  name: qubit\n  stabilizers: []\n  x: [X_0]\n  z: [Z_0]\n---\ndefinitions/logical.code.yaml:\n  name: logical\n  blocks: {q: 1}\n  instructions:\n  - mnemonic: idle\n    description: Identity\n    in: [q]\n    out: [q]\n    action: []\n---\ndefinitions/physical:\n  name: physical\n  blocks: {qubit: 1}\n  instructions:\n  - mnemonic: I\n    description: Identity\n    in: [qubit]\n    out: [qubit]\n    action: []\n---\nqodec.yaml: this is an unused entry, not a second manifest\n"
}

#[test]
fn bundle_roles_come_from_references_not_suffixes() {
    let protocol = Qodec::from_bundle_str(bundle()).expect("extensionless manifest and mislabeled artifacts");
    assert_eq!(protocol.manifest_filename(), "entry");
    assert_eq!(protocol.layers()[0].instruction_set.name, "logical");
    let idle = &protocol.layers()[0].gadgets["idle"];
    assert_eq!(idle.circuit.source, "I 0\n");
    assert_eq!(idle.inputs[0].code.name, "qubit");
    assert_eq!(idle.circuit.effective_format(), "stim");

    for single_file in [false, true] {
        let destination = save_layout(&protocol, single_file);
        let reloaded = reload_saved_protocol(&protocol, destination.path());
        assert_eq!(reloaded.layers()[0].gadgets["idle"].circuit.source, "I 0\n");
    }
}

fn write_manifest_only_bundle(root: &Path) {
    fs::write(
        root.join("outer"),
        "manifest:\n  layers:\n  - instruction_set: first\n  - instruction_set: second\n",
    )
    .expect("bundle");
    for name in ["first", "second"] {
        fs::write(
            root.join(name),
            format!("name: {name}\nblocks: {{q: 1}}\ninstructions: []\n"),
        )
        .expect("instruction set");
    }
}

#[test]
fn one_document_bundle_needs_no_filename_extension() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let root = directory.path();
    write_manifest_only_bundle(root);
    let protocol = Qodec::load(root.join("outer")).expect("manifest-only bundle with referenced files");
    assert_eq!(protocol.layers().len(), 2);
    assert_eq!(protocol.manifest_filename(), "manifest");
    let text = protocol.to_bundle_string().expect("serialize");
    assert_eq!(Qodec::from_bundle_str(&text).expect("reload"), protocol);
}

#[test]
fn bundle_manifest_must_be_first() {
    let text = format!("unrelated: {{}}\n---\n{}", bundle());
    let error = Qodec::from_bundle_str(&text).expect_err("first entry is not a manifest");
    assert!(matches!(error, LoadError::MalformedBundle { .. }));
    assert!(error.to_string().contains("first bundle document must be the manifest"));
}

#[test]
fn equivalent_bundle_keys_are_rejected() {
    let text = format!("{}---\noperations/../source: I 1\n", bundle());
    let error = Qodec::from_bundle_str(&text).expect_err("duplicate normalized key");
    assert!(matches!(error, LoadError::MalformedBundle { .. }));
    assert!(error.to_string().contains("duplicate document key"));
}

#[test]
fn conflicting_reference_roles_are_rejected() {
    for (original, replacement, roles) in [
        (
            "circuit: ../source",
            "circuit: ../entry",
            ["manifest", "circuit source"],
        ),
        (
            "readouts: ../readouts",
            "readouts: ../checks",
            ["check list", "readout list"],
        ),
    ] {
        let text = bundle().replacen(original, replacement, 1);
        let error = Qodec::from_bundle_str(&text).expect_err("conflicting artifact roles");
        assert_conflicting_roles(&error, &roles);
    }
}

fn assert_conflicting_roles(error: &LoadError, roles: &[&str]) {
    assert!(
        matches!(error, LoadError::ConflictingArtifactKind { .. }),
        "unexpected error: {error:?}"
    );
    let rendered = error.to_string();
    for role in roles {
        assert!(rendered.contains(role), "{rendered}");
    }
}

#[test]
fn source_language_is_separate_from_artifact_type() {
    let text = bundle()
        .replace("../source", "../program.qasm")
        .replace("\nsource: |\n  I 0", "\nprogram.qasm: |\n  OPENQASM 3.0;");
    let protocol = Qodec::from_bundle_str(&text).expect("load circuit text without a parser");
    let circuit = &protocol.layers()[0].gadgets["idle"].circuit;
    assert_eq!(circuit.format.as_deref(), Some("openqasm"));
    assert_eq!(circuit.effective_format(), "openqasm");
    assert!(circuit.calls().expect_err("parser not available").contains("openqasm"));
}

fn nested_manifest_bundle() -> String {
    bundle()
        .replacen("entry:", "nested/entry:", 1)
        .replacen(
            "instruction_set: definitions/logical.code.yaml",
            "instruction_set: ../definitions/logical.code.yaml",
            1,
        )
        .replacen("q: code", "q: ../code", 1)
        .replacen("idle: operations/run.isa.yaml", "idle: ../operations/run.isa.yaml", 1)
        .replacen(
            "instruction_set: definitions/physical",
            "instruction_set: ../definitions/physical",
            1,
        )
}

#[test]
fn nested_bundle_sidecar_cannot_overwrite_the_manifest() {
    let text = nested_manifest_bundle()
        .replace("circuit: ../source", "circuit: ../entry")
        .replace("\nsource: |", "\nentry: |");
    let protocol = Qodec::from_bundle_str(&text).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let error = protocol.save_bundle(directory.path()).expect_err("colliding sidecar");
    assert!(error.to_string().contains("conflicting output artifact path"));
    assert!(!directory.path().join("nested/entry").exists());
}

#[test]
fn loaded_tagged_yaml_preserves_source_text() {
    let source = "# retained\n- I: [0]\n";
    let circuit = serde_json::json!({"format": "yaml", "source": source});
    let text = bundle().replace("circuit: ../source", &format!("circuit: {circuit}"));
    let protocol = Qodec::from_bundle_str(&text).unwrap();
    for reloaded in round_trip_all_save_forms(&protocol) {
        assert_eq!(idle(&reloaded).circuit.source, source);
    }
}

fn assert_nested_protocol_contents(protocol: &Qodec) {
    assert_eq!(protocol.layers()[0].instruction_set.name, "logical");
    assert_eq!(protocol.layers()[1].instruction_set.name, "physical");
    let gadget = idle(protocol);
    assert_eq!(gadget.circuit.source, "I 0\n");
    assert_eq!(gadget.inputs[0].code.name, "qubit");
    assert!(gadget.checks.is_empty());
    assert!(gadget.readouts.is_empty());
}

fn assert_unused_manifest_entry_was_not_saved(root: &Path) {
    assert!(!root.join("qodec.yaml").exists());
    let manifest = fs::read_to_string(root.join("nested/entry")).expect("saved manifest");
    assert!(!manifest.contains("this is an unused entry"));
}

fn assert_unknown_source_nested_layout(protocol: &Qodec, single_file: bool) {
    let directory = save_layout(protocol, single_file);
    let destination = directory.path();
    let manifest_path = destination.join("nested/entry");
    assert!(manifest_path.is_file());
    let source_path = if single_file {
        assert!(!destination.join("source").exists());
        destination.join("nested/source")
    } else {
        assert!(destination.join("definitions/logical.code.yaml").is_file());
        assert!(destination.join("operations/run.isa.yaml").is_file());
        destination.join("source")
    };
    assert_eq!(fs::read_to_string(source_path).expect("source sidecar"), "I 0\n");
    assert_unused_manifest_entry_was_not_saved(destination);
    let reloaded = Qodec::load(&manifest_path).expect("reload nested manifest");
    assert_nested_protocol_contents(&reloaded);
    if single_file {
        assert_eq!(Path::new(reloaded.manifest_filename()), Path::new("nested/entry"));
    }
}

#[test]
fn bundle_manifest_paths_are_relative_to_its_key() {
    let text = nested_manifest_bundle();
    let protocol = Qodec::from_bundle_str(&text).expect("manifest-relative bundle references");
    assert_eq!(Path::new(protocol.manifest_filename()), Path::new("nested/entry"));
    assert_eq!(protocol.layers()[0].gadgets["idle"].circuit.source, "I 0\n");

    for single_file in [false, true] {
        assert_unknown_source_nested_layout(&protocol, single_file);
    }

    let error = protocol
        .to_bundle_string()
        .expect_err("unknown-format source still needs a sidecar");
    assert!(error.to_string().contains("as a separate file"));
    assert_eq!(Path::new(protocol.manifest_filename()), Path::new("nested/entry"));
    assert_eq!(protocol.layers()[0].gadgets["idle"].circuit.source, "I 0\n");
}

#[test]
fn yaml_file_sources_defer_call_interpretation() {
    let text = bundle()
        .replace("../source", "../source.yaml")
        .replace("\nsource: |\n  I 0", "\nsource.yaml: |\n  - missing: [typo: 0]");
    let protocol = Qodec::from_bundle_str(&text).unwrap();
    assert!(
        protocol.layers()[0].gadgets["idle"]
            .circuit
            .calls()
            .unwrap_err()
            .contains("unknown instruction")
    );
}

fn bundle_with_yaml_source(extension: &str, source: &str) -> String {
    bundle()
        .replace("circuit: ../source", &format!("circuit: ../calls.{extension}"))
        .replace(
            "source: |\n  I 0\n",
            &format!("calls.{extension}: {}\n", serde_json::to_string(source).unwrap()),
        )
}

#[test]
fn yaml_file_sources_round_trip_without_bundle_sidecars() {
    for extension in ["yaml", "yml"] {
        for source in [
            "# retained source\n- I: [0]\n",
            "- missing: [typo: 0]\n",
            "[invalid YAML",
        ] {
            let text = bundle_with_yaml_source(extension, source);
            for result in load_from_text_and_file(&text) {
                let protocol = result.expect("source stays uninterpreted during loading");
                let calls = idle(&protocol).circuit.calls();
                for restored in round_trip_all_save_forms(&protocol) {
                    assert_eq!(restored, protocol, "{extension}: {source}");
                    assert_eq!(idle(&restored).circuit.source, source);
                    assert_eq!(idle(&restored).circuit.effective_format(), "yaml");
                    assert_eq!(idle(&restored).circuit.calls(), calls);
                }
                let output = save_layout(&protocol, true);
                assert_eq!(
                    fs::read_dir(output.path()).unwrap().count(),
                    1,
                    "no YAML source sidecar"
                );
                assert!(output.path().join(protocol.manifest_filename()).is_file());
            }
        }
    }
}

#[test]
fn absolute_reference_cannot_reinterpret_the_manifest() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let manifest = directory.path().join("entry");
    let text = bundle().replace("circuit: ../source", &format!("circuit: {}", manifest.display()));
    fs::write(&manifest, text).expect("write bundle");
    let error = Qodec::load(&manifest).expect_err("manifest cannot also be a source");
    assert_conflicting_roles(&error, &["manifest", "circuit source"]);
}

fn nested_bundle_with_stim_source() -> String {
    nested_manifest_bundle()
        .replacen("circuit: ../source", "circuit: ../source.stim", 1)
        .replacen("\nsource: |", "\nsource.stim: |", 1)
}

fn assert_stim_nested_protocol(protocol: &Qodec, manifest_filename: &str) {
    assert_eq!(Path::new(protocol.manifest_filename()), Path::new(manifest_filename));
    assert_nested_protocol_contents(protocol);
    assert_eq!(idle(protocol).circuit.effective_format(), "stim");
    assert_eq!(idle(protocol).outputs[0].code.name, "qubit");
}

fn assert_stim_nested_layout(protocol: &Qodec, single_file: bool) {
    let directory = save_layout(protocol, single_file);
    let destination = directory.path();
    let reloaded = reload_saved_protocol(protocol, destination);
    let expected_filename = if single_file {
        assert!(!destination.join("nested/nested/idle.stim").exists());
        "nested/entry"
    } else {
        assert_eq!(
            fs::read_to_string(destination.join("source.stim")).expect("source sidecar"),
            "I 0\n"
        );
        assert!(destination.join("definitions/logical.code.yaml").is_file());
        "entry"
    };
    assert_stim_nested_protocol(&reloaded, expected_filename);
    assert_unused_manifest_entry_was_not_saved(destination);
}

#[test]
fn nested_bundle_manifest_with_known_source_roundtrips() {
    let protocol = Qodec::from_bundle_str(&nested_bundle_with_stim_source()).expect("load nested manifest");
    assert_stim_nested_protocol(&protocol, "nested/entry");
    for single_file in [false, true] {
        assert_stim_nested_layout(&protocol, single_file);
    }
    let saved = protocol.to_bundle_string().expect("known-format source can be inlined");
    assert!(!saved.contains("this is an unused entry"));
    assert_stim_nested_protocol(&Qodec::from_bundle_str(&saved).unwrap(), "nested/entry");
    assert_stim_nested_protocol(&protocol, "nested/entry");
}

fn write_external_artifact_fixture(root: &Path) {
    fs::create_dir_all(root.join("project/.internal")).expect("nested folder");
    fs::create_dir(root.join("shared")).expect("shared folder");
    fs::write(root.join("project/start"), "layers:\n- instruction_set: ../shared/logical\n  codes: {q: ../shared/code}\n  gadgets: {idle: .internal/gadget}\n- instruction_set: ../shared/physical\n").expect("manifest");
    fs::write(root.join("shared/logical"), "name: logical\nblocks: {q: 1}\ninstructions:\n- mnemonic: idle\n  description: Identity\n  in: [q]\n  out: [q]\n  action: []\n").expect("instruction set");
    fs::write(
        root.join("shared/physical"),
        "name: physical\nblocks: {q: 1}\ninstructions:\n- mnemonic: I\n  description: Identity\n  in: [q]\n  out: [q]\n  action: []\n",
    )
    .expect("instruction set");
    fs::write(
        root.join("shared/code"),
        "name: qubit\nstabilizers: []\nx: [X_0]\nz: [Z_0]\n",
    )
    .expect("code");
    fs::write(root.join("project/.internal/gadget"), "implements: ../../shared/logical#idle\ncircuit:\n  instruction_set: ../../shared/physical\n  source: ../../shared/source\nchecks: ../../shared/parity\n").expect("gadget");
    fs::write(root.join("shared/source"), "I 0\n").expect("source");
    fs::write(root.join("shared/parity"), "[]\n").expect("checks");
}

#[test]
fn external_artifacts_and_nested_references_are_loaded() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let root = directory.path();
    write_external_artifact_fixture(root);
    let protocol = Qodec::load(root.join("project/start")).expect("external references");
    assert_eq!(protocol.layers()[0].gadgets["idle"].circuit.source, "I 0\n");

    fs::remove_file(root.join("shared/parity")).expect("remove referenced file");
    let error = Qodec::load(root.join("project/start")).expect_err("missing referenced checks");
    let LoadError::MissingArtifact { referenced_from, path } = error else {
        panic!("expected missing artifact, got {error:?}");
    };
    assert_eq!(referenced_from, Path::new(".internal/gadget"));
    assert_eq!(path, Path::new("../shared/parity"));
}

const COLLIDING_MANIFEST_NAMES: [&str; 8] = [
    "logical.isa.yaml",
    "qubit.code.yaml",
    "idle.gadget.yaml",
    "idle.stim",
    "nested/logical.isa.yaml",
    "nested/qubit.code.yaml",
    "nested/idle.gadget.yaml",
    "nested/idle.stim",
];

type Documents = std::collections::BTreeMap<String, serde_yaml::Value>;

fn collision_documents(inline_empty: bool) -> Documents {
    use serde::Deserialize;
    use serde_yaml::Value;

    let mut artifacts: Documents = serde_yaml::Deserializer::from_str(bundle())
        .flat_map(|document| Documents::deserialize(document).expect("bundle document"))
        .collect();
    let mut gadget = artifacts.remove("operations/run.isa.yaml").expect("gadget");
    artifacts.remove("source");
    artifacts.insert("sources/second.stim".to_owned(), Value::from("I 0\nI 0\n"));
    gadget["circuit"] = Value::from("../sources/second.stim");
    artifacts.insert("operations/second".to_owned(), gadget.clone());
    gadget["circuit"] = if inline_empty {
        Value::Sequence(Vec::new())
    } else {
        artifacts.insert("sources/original.stim".to_owned(), Value::from("I 0\n"));
        Value::from("../sources/original.stim")
    };
    artifacts.insert("operations/run.isa.yaml".to_owned(), gadget);
    let instructions = artifacts.get_mut("definitions/logical.code.yaml").unwrap()["instructions"]
        .as_sequence_mut()
        .unwrap();
    let mut second = instructions[0].clone();
    second["mnemonic"] = Value::from("idle.1");
    instructions.push(second);
    artifacts.get_mut("entry").unwrap()["layers"][0]["gadgets"]["idle.1"] = Value::from("operations/second");
    artifacts
}

fn make_manifest_references_parent_relative(manifest: &mut serde_yaml::Value) {
    use serde_yaml::Value;
    for layer in manifest["layers"].as_sequence_mut().unwrap() {
        layer["instruction_set"] = Value::from(format!("../{}", layer["instruction_set"].as_str().unwrap()));
        for field in ["codes", "gadgets"] {
            if let Some(paths) = layer[field].as_mapping_mut() {
                for path in paths.values_mut() {
                    *path = Value::from(format!("../{}", path.as_str().unwrap()));
                }
            }
        }
    }
}

fn collision_bundle(artifacts: &Documents, manifest_filename: &str) -> String {
    let mut manifest = artifacts["entry"].clone();
    if Path::new(manifest_filename).starts_with("nested") {
        make_manifest_references_parent_relative(&mut manifest);
    }
    let mut text = serde_yaml::to_string(&std::collections::BTreeMap::from([(manifest_filename, &manifest)]))
        .expect("serialize manifest envelope");
    for (path, artifact) in artifacts.iter().filter(|(path, _)| path.as_str() != "entry") {
        text.push_str("---\n");
        text.push_str(&serde_yaml::to_string(&std::collections::BTreeMap::from([(path, artifact)])).unwrap());
    }
    text
}

fn load_collision_fixture(artifacts: &Documents, filename: &str, inline_empty: bool) -> Qodec {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("input");
    fs::write(&input, collision_bundle(artifacts, filename)).unwrap();
    let protocol = Qodec::load(input).expect("load colliding manifest filename");
    assert_eq!(Path::new(protocol.manifest_filename()), Path::new(filename));
    assert_eq!(protocol.layers().len(), 2);
    assert_eq!(protocol.layers()[0].gadgets.len(), 2);
    assert_eq!(
        idle(&protocol).circuit.source.trim(),
        if inline_empty { "[]" } else { "I 0" }
    );
    assert_eq!(protocol.layers()[0].gadgets["idle.1"].circuit.source, "I 0\nI 0\n");
    protocol
}

fn assert_gadget_semantics(actual: &Gadget, expected: &Gadget) {
    assert_eq!(actual.implements, expected.implements);
    assert_eq!(actual.circuit.instruction_set, expected.circuit.instruction_set);
    assert_eq!(actual.circuit.source.trim(), expected.circuit.source.trim());
    assert_eq!(actual.circuit.effective_format(), expected.circuit.effective_format());
    assert_eq!(actual.checks, expected.checks);
    assert_eq!(actual.readouts, expected.readouts);
    for (actual, expected) in [(&actual.inputs, &expected.inputs), (&actual.outputs, &expected.outputs)] {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.code, expected.code);
            assert_eq!(actual.support, expected.support);
            assert_eq!(actual.block_types, expected.block_types);
        }
    }
}

fn assert_collision_semantics(actual: &Qodec, expected: &Qodec) {
    assert_eq!(actual.layers().len(), expected.layers().len());
    for (actual, expected) in actual.layers().iter().zip(expected.layers()) {
        assert_eq!(actual.instruction_set, expected.instruction_set);
        assert_eq!(
            actual.gadgets.keys().collect::<Vec<_>>(),
            expected.gadgets.keys().collect::<Vec<_>>()
        );
        for (mnemonic, expected) in &expected.gadgets {
            assert_gadget_semantics(&actual.gadgets[mnemonic], expected);
        }
    }
}

fn assert_original_collision_files(root: &Path, artifacts: &Documents, inline_empty: bool) {
    for path in artifacts
        .keys()
        .filter(|path| !["entry", "qodec.yaml"].contains(&path.as_str()))
    {
        assert!(root.join(path).is_file(), "original artifact missing: {path}");
    }
    assert_eq!(
        fs::read_to_string(root.join("sources/second.stim")).unwrap(),
        "I 0\nI 0\n"
    );
    if !inline_empty {
        assert_eq!(fs::read_to_string(root.join("sources/original.stim")).unwrap(), "I 0\n");
    }
}

fn assert_collision_save_forms(protocol: &Qodec, artifacts: &Documents, inline_empty: bool) {
    let saved = protocol.to_bundle_string().expect("serialize collision case");
    assert_eq!(saved, protocol.to_bundle_string().expect("deterministic filenames"));
    let reloaded = Qodec::from_bundle_str(&saved).unwrap();
    assert_eq!(reloaded.manifest_filename(), protocol.manifest_filename());
    assert_collision_semantics(&reloaded, protocol);
    for single_file in [false, true] {
        let destination = save_layout(protocol, single_file);
        let reloaded = reload_saved_protocol(protocol, destination.path());
        assert_collision_semantics(&reloaded, protocol);
        if single_file {
            assert_eq!(reloaded.manifest_filename(), protocol.manifest_filename());
            assert_eq!(
                fs::read_to_string(destination.path().join(protocol.manifest_filename())).unwrap(),
                saved
            );
        } else {
            assert_original_collision_files(destination.path(), artifacts, inline_empty);
        }
    }
}

#[test]
fn synthesized_artifacts_avoid_manifest_filename_collisions() {
    for inline_empty in [false, true] {
        let artifacts = collision_documents(inline_empty);
        for filename in COLLIDING_MANIFEST_NAMES {
            let protocol = load_collision_fixture(&artifacts, filename, inline_empty);
            assert_collision_save_forms(&protocol, &artifacts, inline_empty);
            let synthesized = rebuild_from_layers(&protocol);
            let destination = save_layout(&synthesized, false);
            let reloaded = reload_saved_protocol(&synthesized, destination.path());
            assert_collision_semantics(&reloaded, &protocol);
        }
    }
}
