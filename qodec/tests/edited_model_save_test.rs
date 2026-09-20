use std::fs;
use std::path::Path;
use std::sync::Arc;

use qodec::{Qodec, Reference};

fn protocol_with_shared_artifacts() -> Qodec {
    Qodec::from_bundle_str(concat!(
        "entry:\n  schema_version: 1\n",
        r#"  name: state
  layers:
  - instruction_set: definitions/logical
    codes: {q: definitions/code, spare: definitions/unused}
    gadgets: {a: operations/a, b: operations/b}
  - instruction_set: definitions/physical
---
definitions/logical:
  name: logical
  blocks: {q: 1, spare: 1}
  instructions:
  - {mnemonic: a, description: first, in: [q], out: [q], flags: [reject], parameters: {theta: number}, action: []}
  - {mnemonic: b, description: second, in: [q], out: [q], flags: [reject], action: []}
---
definitions/physical:
  name: physical
  blocks: {qubit: 1}
  instructions:
  - {mnemonic: I, description: identity, in: [qubit], out: [qubit], action: []}
---
definitions/code:
  name: qubit
  stabilizers: []
  x: [X_0]
  z: [Z_0]
---
definitions/unused:
  name: retained
  stabilizers: []
  x: [X_0]
  z: [Z_0]
---
operations/a:
  circuit: ../source.stim
  checks: ../checks
  readouts: ../readouts
  parameter_bindings: {theta: circuit.source.theta}
---
operations/b:
  circuit: ../source.stim
  checks: ../checks
  readouts: ../readouts
---
source.stim: "I 0\n"
---
checks: [['out[0].z[0]']]
---
readouts: [{reject: ['out[0].z[0]']}]
"#,
    ))
    .unwrap()
}

fn read_document(root: &Path, relative: &str) -> serde_yaml::Value {
    let text = fs::read_to_string(root.join(relative)).expect("read saved document");
    serde_yaml::from_str(&text).expect("parse saved document")
}

fn save_to_temporary_directory(protocol: &Qodec) -> tempfile::TempDir {
    let directory = tempfile::tempdir().expect("create save destination");
    protocol.save(directory.path()).expect("save current protocol");
    directory
}

fn assert_all_save_forms_preserve_current_values(protocol: &Qodec) {
    let directory = tempfile::tempdir().expect("create save destinations");
    protocol
        .save(directory.path().join("separate"))
        .expect("save separate artifacts");
    protocol
        .save_bundle(directory.path().join("bundle"))
        .expect("save a bundle");
    let text = protocol.to_bundle_string().expect("serialize bundle text");
    for (layout, reloaded) in [
        (
            "separate",
            Qodec::load(directory.path().join("separate").join(protocol.manifest_filename())),
        ),
        (
            "bundle",
            Qodec::load(directory.path().join("bundle").join(protocol.manifest_filename())),
        ),
        ("text", Qodec::from_bundle_str(&text)),
    ] {
        let reloaded = reloaded.unwrap_or_else(|error| panic!("reload {layout}: {error}"));
        assert_eq!(protocol, &reloaded, "{layout}: current model changed");
        for (original_layer, saved_layer) in protocol.layers().iter().zip(reloaded.layers()) {
            for (mnemonic, original) in &original_layer.gadgets {
                assert_eq!(
                    original.circuit.source, saved_layer.gadgets[mnemonic].circuit.source,
                    "{layout}: {mnemonic} circuit text changed"
                );
            }
        }
    }
}

fn shared_section_paths(root: &Path) -> Vec<(serde_yaml::Value, serde_yaml::Value)> {
    let first = read_document(root, "operations/a");
    let second = read_document(root, "operations/b");
    ["circuit", "checks", "readouts"]
        .iter()
        .map(|section| (first[section].clone(), second[section].clone()))
        .collect()
}

fn protocol_with_external_instruction_set(root: &Path, absolute: bool) -> Qodec {
    fs::create_dir_all(root.join("input")).unwrap();
    let source = root.join("shared.yaml");
    fs::write(&source, "name: physical\nblocks: {qubit: 1}\ninstructions: []\n").unwrap();
    let reference = if absolute { source } else { "../shared.yaml".into() };
    let manifest = serde_json::json!({"layers": [{"instruction_set": reference}]});
    fs::write(root.join("input/entry"), manifest.to_string()).unwrap();
    Qodec::load(root.join("input/entry")).unwrap()
}

#[test]
fn external_instruction_sets_are_reused_or_copied_without_overwriting() {
    for absolute in [false, true] {
        for edited in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let root = directory.path();
            let mut protocol = protocol_with_external_instruction_set(root, absolute);
            let original = fs::read_to_string(root.join("shared.yaml")).unwrap();
            if edited {
                Arc::make_mut(&mut protocol.layers_mut()[0].instruction_set).description = "edited".to_owned();
            }
            let manifest = protocol.save(root.join("export")).unwrap();
            assert_eq!(fs::read_to_string(root.join("shared.yaml")).unwrap(), original);
            assert_eq!(Qodec::load(&manifest).unwrap(), protocol);
            let reference = read_document(root, "export/entry")["layers"][0]["instruction_set"]
                .as_str()
                .unwrap()
                .to_owned();
            let target = fs::canonicalize(manifest.parent().unwrap().join(reference)).unwrap();
            let export = fs::canonicalize(root.join("export")).unwrap();
            assert_eq!(target.starts_with(export), edited);
        }
    }
}

#[test]
fn changed_external_files_fail_before_creating_the_destination() {
    let directory = tempfile::tempdir().unwrap();
    let protocol = protocol_with_external_instruction_set(directory.path(), true);
    fs::write(directory.path().join("shared.yaml"), "name: changed\n").unwrap();
    let output = directory.path().join("export");
    let error = protocol.save(&output).unwrap_err();
    assert!(error.to_string().contains("changed since loading"), "{error}");
    assert!(!output.exists());
}

#[test]
fn external_files_are_not_overwritten_by_explicit_output_collisions() {
    for single_file in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let mut protocol = protocol_with_external_instruction_set(root, true);
        let original = fs::read_to_string(root.join("shared.yaml")).unwrap();
        protocol.set_manifest_filename("../shared.yaml".to_owned());
        let output = root.join("export");
        let result = if single_file {
            protocol.save_bundle(&output)
        } else {
            protocol.save(&output)
        };
        assert!(result.unwrap_err().to_string().contains("overwrite external artifact"));
        assert_eq!(fs::read_to_string(root.join("shared.yaml")).unwrap(), original);
        assert!(!output.exists());
    }
}

#[test]
fn missing_external_files_do_not_block_saving_local_edits() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let mut protocol = protocol_with_external_instruction_set(root, false);
    fs::remove_file(root.join("shared.yaml")).unwrap();
    let output = root.join("export");
    assert!(
        protocol
            .save(&output)
            .unwrap_err()
            .to_string()
            .contains("cannot reuse external artifact")
    );
    assert!(!output.exists());
    Arc::make_mut(&mut protocol.layers_mut()[0].instruction_set).description = "edited".to_owned();
    let manifest = protocol.save(&output).unwrap();
    assert_eq!(Qodec::load(manifest).unwrap(), protocol);
}

fn protocol_with_external_gadgets(root: &Path) -> Qodec {
    protocol_with_shared_artifacts().save(root.join("shared")).unwrap();
    fs::create_dir_all(root.join("input")).unwrap();
    let manifest = serde_json::json!({"layers": [
        {"instruction_set": "../shared/definitions/logical",
         "codes": {"q": "../shared/definitions/code", "spare": "../shared/definitions/unused"},
         "gadgets": {"a": "../shared/operations/a", "b": "../shared/operations/b"}},
        {"instruction_set": "../shared/definitions/physical"},
    ]});
    fs::write(root.join("input/entry"), manifest.to_string()).unwrap();
    Qodec::load(root.join("input/entry")).unwrap()
}

#[test]
fn external_gadgets_and_their_dependencies_remain_linked() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let protocol = protocol_with_external_gadgets(root);
    let manifest = protocol.save(root.join("export")).unwrap();
    assert_eq!(Qodec::load(&manifest).unwrap(), protocol);
    assert_eq!(fs::read_dir(root.join("export")).unwrap().count(), 1);
    let saved = read_document(root, "export/entry");
    let reference = saved["layers"][0]["gadgets"]["a"].as_str().unwrap();
    assert_eq!(Path::new(reference), root.join("shared/operations/a"));
}

#[test]
fn unchanged_external_gadgets_keep_redundant_circuit_type_declarations() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let _ = protocol_with_external_gadgets(root);
    let path = root.join("shared/operations/a");
    let mut document = read_document(root, "shared/operations/a");
    document["circuit"] =
        serde_yaml::from_str("source: ../source.stim\nin: {'0': qubit}\nout: {'0': qubit}\n").unwrap();
    fs::write(&path, serde_yaml::to_string(&document).unwrap()).unwrap();
    let protocol = Qodec::load(root.join("input/entry")).unwrap();
    let manifest = protocol.save(root.join("export")).unwrap();
    assert_eq!(Qodec::load(manifest).unwrap(), protocol);
    let saved = read_document(root, "export/entry");
    assert_eq!(Path::new(saved["layers"][0]["gadgets"]["a"].as_str().unwrap()), path);
}

#[test]
fn edited_external_sidecars_copy_the_referring_gadget() {
    for section in ["source", "checks", "readouts"] {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let mut protocol = protocol_with_external_gadgets(root);
        let original = fs::read_to_string(root.join("shared/operations/a")).unwrap();
        let gadget = protocol.layers_mut()[0].gadgets.get_mut("a").unwrap();
        match section {
            "source" => gadget.circuit.source.push_str("I 1\n"),
            "checks" => gadget.checks.clear(),
            _ => gadget.readouts[0].equation.clear(),
        }
        let manifest = protocol.save(root.join("export")).unwrap();
        assert_eq!(Qodec::load(&manifest).unwrap(), protocol);
        assert_eq!(fs::read_to_string(root.join("shared/operations/a")).unwrap(), original);
        let saved = read_document(root, "export/entry");
        assert!(!Path::new(saved["layers"][0]["gadgets"]["a"].as_str().unwrap()).is_absolute());
        assert_eq!(
            Path::new(saved["layers"][0]["gadgets"]["b"].as_str().unwrap()),
            root.join("shared/operations/b")
        );
    }
}

#[test]
fn bundles_embed_external_artifacts_without_reading_them_again() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let protocol = protocol_with_external_gadgets(root);
    fs::remove_dir_all(root.join("shared")).unwrap();
    let text = protocol.to_bundle_string().unwrap();
    assert_eq!(Qodec::from_bundle_str(&text).unwrap(), protocol);
    let manifest = protocol.save_bundle(root.join("export")).unwrap();
    assert_eq!(Qodec::load(manifest).unwrap(), protocol);
    assert_eq!(fs::read_dir(root.join("export")).unwrap().count(), 1);
}

#[test]
fn external_code_edits_survive_gadget_removal_without_changing_the_source() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let mut protocol = protocol_with_external_gadgets(root);
    let original = fs::read_to_string(root.join("shared/definitions/code")).unwrap();
    edit_shared_code_and_target_instruction_set(&mut protocol);
    protocol.layers_mut()[0].gadgets.clear();
    let manifest = protocol.save(root.join("export")).unwrap();
    assert_eq!(Qodec::load(manifest).unwrap(), protocol);
    assert_eq!(
        fs::read_to_string(root.join("shared/definitions/code")).unwrap(),
        original
    );
    assert_eq!(
        read_document(root, "export/qubit.code.yaml")["description"],
        "edited code"
    );
}

#[test]
fn bundle_sidecars_copy_external_sources_without_interpreting_them() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let _ = protocol_with_external_gadgets(root);
    rename_source_to_unknown_extension(&root.join("shared"));
    let protocol = Qodec::load(root.join("input/entry")).unwrap();
    let linked = protocol.save(root.join("linked")).unwrap();
    assert_eq!(Qodec::load(linked).unwrap(), protocol);
    assert_eq!(fs::read_dir(root.join("linked")).unwrap().count(), 1);
    fs::remove_dir_all(root.join("shared")).unwrap();
    let bundled = protocol.save_bundle(root.join("bundle")).unwrap();
    assert_eq!(Qodec::load(bundled).unwrap(), protocol);
    assert!(
        protocol
            .to_bundle_string()
            .unwrap_err()
            .to_string()
            .contains("separate file")
    );
}

#[test]
fn absolute_references_inside_the_input_directory_are_copied() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let _ = protocol_with_external_instruction_set(root, true);
    fs::rename(root.join("shared.yaml"), root.join("input/shared.yaml")).unwrap();
    let manifest = serde_json::json!({"layers": [{"instruction_set": root.join("input/shared.yaml")}]});
    fs::write(root.join("input/entry"), manifest.to_string()).unwrap();
    let protocol = Qodec::load(root.join("input/entry")).unwrap();
    let saved = protocol.save(root.join("export")).unwrap();
    fs::remove_dir_all(root.join("input")).unwrap();
    assert_eq!(Qodec::load(saved).unwrap(), protocol);
}

#[test]
fn generated_artifact_paths_are_unique_after_normalization() {
    let layers = ["same", "nested/../same", "same.1"]
        .into_iter()
        .map(|name| qodec::Layer {
            instruction_set: Arc::new(qodec::InstructionSet {
                name: name.to_owned(),
                description: String::new(),
                blocks: Vec::new(),
                instructions: Vec::new(),
                metadata: qodec::Metadata::default(),
            }),
            codes: std::collections::BTreeMap::default(),
            gadgets: std::collections::BTreeMap::default(),
        })
        .collect();
    let mut protocol = Qodec::new(None, None, layers);
    assert_all_save_forms_preserve_current_values(&protocol);
    protocol.set_manifest_filename("nested/../same.isa.yaml".to_owned());
    assert_all_save_forms_preserve_current_values(&protocol);
}

#[test]
fn path_like_names_produce_distinct_local_artifacts() {
    let directory = tempfile::tempdir().unwrap();
    let destination = directory.path().join("output");
    let layers = [
        "same".to_owned(),
        destination.join("same").to_string_lossy().into_owned(),
    ]
    .into_iter()
    .map(|name| qodec::Layer {
        instruction_set: Arc::new(qodec::InstructionSet {
            name,
            description: String::new(),
            blocks: Vec::new(),
            instructions: Vec::new(),
            metadata: qodec::Metadata::default(),
        }),
        codes: std::collections::BTreeMap::default(),
        gadgets: std::collections::BTreeMap::default(),
    })
    .collect();
    let protocol = Qodec::new(None, None, layers);
    let manifest = protocol.save(&destination).unwrap();
    assert_eq!(Qodec::load(manifest).unwrap(), protocol);
    assert_eq!(fs::read_dir(&destination).unwrap().count(), 3);
    let bundled = protocol.save_bundle(directory.path().join("bundle")).unwrap();
    assert_eq!(Qodec::load(bundled).unwrap(), protocol);
}

#[test]
fn saving_a_large_nondefault_encoding_does_not_expand_its_support() {
    let documents = [
        serde_json::json!({"entry": {"layers": [
            {"instruction_set": "logical", "codes": {"a": "first", "b": "second"}, "gadgets": {"prepare": "gadget"}},
            {"instruction_set": "physical"},
        ]}}),
        serde_json::json!({"logical": {"name": "logical", "blocks": {"a": 1, "b": 1}, "instructions": [
            {"mnemonic": "prepare", "description": "", "out": ["a", "b"]},
        ]}}),
        serde_json::json!({"physical": {"name": "physical", "blocks": {"qubit": 1}, "instructions": []}}),
        serde_json::json!({"first": {"name": "first", "stabilizers": ["Z_0"], "x": [], "z": []}}),
        serde_json::json!({"second": {"name": "second", "stabilizers": ["Z_0"], "x": [], "z": []}}),
        serde_json::json!({"gadget": {"circuit": {"format": "custom", "source": "text"}}}),
    ];
    let source = documents
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n---\n");
    let mut protocol = Qodec::from_bundle_str(&source).unwrap();
    let gadget = protocol.layers_mut()[0].gadgets.get_mut("prepare").unwrap();
    Arc::make_mut(&mut gadget.outputs[1].code).stabilizers = vec![format!("Z_{}", usize::MAX - 1).into()];
    let code = gadget.outputs[1].code.clone();
    protocol.layers_mut()[0].codes.insert("b".to_owned(), code);
    assert_all_save_forms_preserve_current_values(&protocol);
}

fn edit_first_gadget_sections(protocol: &mut Qodec) {
    let gadget = protocol.layers_mut()[0].gadgets.get_mut("a").expect("first gadget");
    gadget.circuit.source.push_str("# edited circuit\nI 0\n");
    gadget.checks = vec![vec![Reference::parse("out[0].z[0,0]").unwrap().into()]];
    gadget.readouts[0].equation = vec![
        Reference::parse("in[0].z[0]").unwrap().into(),
        qodec::ParityTerm::Bit(true),
    ];
    gadget.frames.insert(
        Reference::parse("out[0].z[0]").unwrap(),
        vec![Reference::parse("in[0].z[0]").unwrap().into()],
    );
    gadget
        .frames
        .insert(Reference::parse("out[0].x[0]").unwrap(), Vec::new());
    gadget
        .parameter_bindings
        .insert("theta".to_owned(), "changed".to_owned());
    gadget.metadata.insert("edited".to_owned(), true.into());
}

#[test]
fn shared_sections_split_when_only_one_gadget_changes() {
    let mut protocol = protocol_with_shared_artifacts();
    let baseline = protocol_with_shared_artifacts();
    assert_eq!(protocol, baseline);
    let before = save_to_temporary_directory(&protocol);
    assert_sections_are_shared(before.path());

    edit_first_gadget_sections(&mut protocol);
    assert_ne!(protocol, baseline);
    protocol.validate().expect("edited protocol remains preservable");
    assert_all_save_forms_preserve_current_values(&protocol);

    let after = save_to_temporary_directory(&protocol);
    assert_sections_are_separate(after.path());
    assert_eq!(
        read_document(after.path(), "definitions/unused"),
        read_document(before.path(), "definitions/unused")
    );
}

fn assert_sections_are_shared(root: &Path) {
    for (first, second) in shared_section_paths(root) {
        assert_eq!(first, second, "unchanged sections should share one file");
    }
}

fn assert_sections_are_separate(root: &Path) {
    for (first, second) in shared_section_paths(root) {
        assert_ne!(first, second, "divergent sections need separate files");
    }
}

fn edit_shared_code_and_target_instruction_set(protocol: &mut Qodec) {
    let mut code = protocol.layers()[0].gadgets["a"].inputs[0].code.as_ref().clone();
    code.description.clear();
    code.description.push_str("edited code");
    let code = Arc::new(code);
    protocol.layers_mut()[0].codes.insert("q".to_owned(), code.clone());
    for gadget in protocol.layers_mut()[0].gadgets.values_mut() {
        for encoding in gadget.inputs.iter_mut().chain(&mut gadget.outputs) {
            encoding.code = code.clone();
        }
    }
    let physical = Arc::make_mut(&mut protocol.layers_mut()[1].instruction_set);
    physical.description.clear();
    physical.description.push_str("edited instruction set");
    let physical = protocol.layers()[1].instruction_set.clone();
    for gadget in protocol.layers_mut()[0].gadgets.values_mut() {
        gadget.circuit.instruction_set = physical.clone();
    }
}

#[test]
fn code_and_instruction_set_edits_are_saved() {
    let mut protocol = protocol_with_shared_artifacts();
    edit_shared_code_and_target_instruction_set(&mut protocol);
    protocol.layers_mut()[0].gadgets.get_mut("a").unwrap().inputs[0].support[0] = "2".to_owned();
    protocol.validate().expect("updated bindings are consistent");
    assert_all_save_forms_preserve_current_values(&protocol);

    let directory = save_to_temporary_directory(&protocol);
    assert_eq!(
        read_document(directory.path(), "definitions/code")["description"],
        "edited code"
    );
    assert_eq!(
        read_document(directory.path(), "definitions/physical")["description"],
        "edited instruction set"
    );
}

#[test]
fn code_edits_survive_removing_the_last_gadget() {
    let mut protocol = protocol_with_shared_artifacts();
    edit_shared_code_and_target_instruction_set(&mut protocol);
    let before = save_to_temporary_directory(&protocol);
    protocol.layers_mut()[0].gadgets.clear();
    assert_all_save_forms_preserve_current_values(&protocol);
    let after = save_to_temporary_directory(&protocol);
    assert_eq!(
        read_document(after.path(), "definitions/code"),
        read_document(before.path(), "definitions/code")
    );
    assert_eq!(protocol.codes()["qubit"].description, "edited code");
    assert_eq!(
        protocol.resolve("layers[0].codes[\"q\"].description").unwrap().as_str(),
        Some("edited code")
    );
    assert_eq!(
        protocol.resolve("codes[\"qubit\"].description").unwrap().as_str(),
        Some("edited code")
    );
}

#[test]
fn slices_preserve_retained_layer_code_bindings() {
    let protocol = Qodec::load("examples/distillation-15/distillation-15.qodec.yaml").unwrap();
    for (start, stop) in [(0, 2), (0, 1), (1, 2)] {
        let sliced = protocol.slice(start, stop).unwrap();
        for (original, retained) in protocol.layers()[start..stop].iter().zip(sliced.layers()) {
            assert_eq!(original.codes, retained.codes);
        }
        assert_all_save_forms_preserve_current_values(&sliced);
    }
}

#[test]
fn slices_retain_codes_inferred_from_removed_gadgets() {
    let mut protocol = protocol_with_shared_artifacts();
    protocol.layers_mut()[0].codes.clear();
    let expected = protocol.codes()["qubit"].clone();
    let sliced = protocol.slice(0, 1).unwrap();
    assert!(sliced.layers()[0].gadgets.is_empty());
    assert_eq!(sliced.layers()[0].codes["q"], expected);
    assert_all_save_forms_preserve_current_values(&sliced);
}

fn replace_first_instruction_and_gadget(protocol: &mut Qodec) {
    let mut gadget = protocol.layers_mut()[0].gadgets.remove("a").unwrap();
    let logical = Arc::make_mut(&mut protocol.layers_mut()[0].instruction_set);
    logical.instructions.retain(|instruction| instruction.mnemonic != "a");
    gadget.implements.mnemonic.clear();
    gadget.implements.mnemonic.push_str("new");
    logical.instructions.push(gadget.implements.clone());
    protocol.layers_mut()[0].gadgets.insert("new".to_owned(), gadget);
}

#[test]
fn model_additions_and_removals_control_the_output() {
    let mut protocol = protocol_with_shared_artifacts();
    replace_first_instruction_and_gadget(&mut protocol);
    assert_all_save_forms_preserve_current_values(&protocol);

    let directory = save_to_temporary_directory(&protocol);
    assert!(!directory.path().join("operations/a").exists());
    let manifest = read_document(directory.path(), "entry");
    assert!(manifest["layers"][0]["gadgets"]["a"].is_null());
    assert!(manifest["layers"][0]["gadgets"]["new"].is_string());

    protocol.layers_mut()[0].gadgets.clear();
    assert_all_save_forms_preserve_current_values(&protocol);
}

fn rebuild_at_different_manifest_path(original: &Qodec) -> Qodec {
    let mut constructed = Qodec::new(
        original.name().map(str::to_owned),
        original.description().map(str::to_owned),
        original.layers().to_vec(),
    );
    constructed.set_schema_version(original.schema_version());
    constructed.set_manifest_filename("elsewhere/manifest".to_owned());
    constructed
}

#[test]
fn equality_uses_current_values_not_paths_or_loading_history() {
    let original = protocol_with_shared_artifacts();
    let mut constructed = rebuild_at_different_manifest_path(&original);
    assert_eq!(original, constructed);
    append_source_comment(&mut constructed);
    assert_ne!(
        original, constructed,
        "source text affects equality even when calls are unchanged"
    );
    assert_all_save_forms_preserve_current_values(&constructed);
    constructed.layers_mut()[0].gadgets.get_mut("a").unwrap().circuit.source =
        original.layers()[0].gadgets["a"].circuit.source.clone();
    assert_eq!(original, constructed);
    constructed.metadata_mut().insert("note".to_owned(), "edited".into());
    assert_ne!(original, constructed);
}

fn append_source_comment(protocol: &mut Qodec) {
    protocol.layers_mut()[0]
        .gadgets
        .get_mut("a")
        .unwrap()
        .circuit
        .source
        .push_str("# source-only edit\n");
}

#[test]
fn invalid_current_models_fail_before_writing() {
    let mut protocol = protocol_with_shared_artifacts();
    let gadget = protocol.layers()[0].gadgets["a"].clone();
    protocol
        .layers_mut()
        .last_mut()
        .unwrap()
        .gadgets
        .insert("a".to_owned(), gadget);
    let directory = tempfile::tempdir().unwrap();
    assert!(protocol.save(directory.path().join("separate")).is_err());
    assert!(protocol.save_bundle(directory.path().join("bundle")).is_err());
    assert!(protocol.to_bundle_string().is_err());
    assert!(!directory.path().join("separate").exists());
    assert!(!directory.path().join("bundle").exists());
}

#[test]
fn unsupported_schema_versions_fail_before_writing() {
    let mut protocol = protocol_with_shared_artifacts();
    let directory = tempfile::tempdir().unwrap();
    for version in [0, qodec::CURRENT_SCHEMA_VERSION + 1, u32::MAX] {
        protocol.set_schema_version(Some(version));
        let expected = format!(
            "schema_version must be {} or omitted (got {version})",
            qodec::CURRENT_SCHEMA_VERSION
        );
        assert_eq!(protocol.validate().unwrap_err(), expected);
        assert_eq!(protocol.to_bundle_string().unwrap_err().to_string(), expected);
        assert_eq!(
            protocol
                .save(directory.path().join("separate"))
                .unwrap_err()
                .to_string(),
            expected
        );
        assert_eq!(
            protocol
                .save_bundle(directory.path().join("bundle"))
                .unwrap_err()
                .to_string(),
            expected
        );
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 0);
    }
    for version in [None, Some(qodec::CURRENT_SCHEMA_VERSION)] {
        protocol.set_schema_version(version);
        assert_all_save_forms_preserve_current_values(&protocol);
    }
}

#[test]
fn moving_the_manifest_preserves_relative_artifact_locations() {
    let mut protocol = protocol_with_shared_artifacts();
    protocol.set_manifest_filename("nested/deeper/manifest".to_owned());
    assert_all_save_forms_preserve_current_values(&protocol);
    let directory = save_to_temporary_directory(&protocol);
    assert!(directory.path().join("definitions/logical").exists());
    assert!(directory.path().join("operations/a").exists());
    assert_eq!(
        Path::new(
            read_document(directory.path(), "nested/deeper/manifest")["layers"][0]["instruction_set"]
                .as_str()
                .unwrap()
        ),
        Path::new("../../definitions/logical")
    );
}

#[test]
fn moving_the_manifest_above_the_output_relocates_relative_artifacts() {
    for manifest in ["../entry", "../../nested/entry"] {
        let mut protocol = protocol_with_shared_artifacts();
        protocol.set_manifest_filename(manifest.to_owned());
        let bundled = protocol.to_bundle_string().unwrap();
        assert_eq!(Qodec::from_bundle_str(&bundled).unwrap(), protocol);
        assert!(bundled.contains("name: retained"));
        for single_file in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let output = directory.path().join("root/stage/output");
            if single_file {
                protocol.save_bundle(&output).unwrap();
            } else {
                protocol.save(&output).unwrap();
            }
            assert_eq!(Qodec::load(output.join(manifest)).unwrap(), protocol);
        }
    }
}

fn rename_source_to_unknown_extension(root: &Path) {
    fs::rename(root.join("source.stim"), root.join("source.data")).unwrap();
    for name in ["operations/a", "operations/b"] {
        let path = root.join(name);
        let text = fs::read_to_string(&path).unwrap().replace("source.stim", "source.data");
        fs::write(path, text).unwrap();
    }
}

#[test]
fn unknown_source_extensions_do_not_hide_current_edits() {
    let directory = save_to_temporary_directory(&protocol_with_shared_artifacts());
    rename_source_to_unknown_extension(directory.path());
    let mut protocol = Qodec::load(directory.path().join("entry")).unwrap();
    protocol.layers_mut()[0].gadgets.get_mut("a").unwrap().circuit.source = "I 0\nI 0\n".to_owned();
    assert_file_save_forms_preserve_both_sources(&protocol);
    assert!(
        protocol
            .to_bundle_string()
            .unwrap_err()
            .to_string()
            .contains("separate file")
    );
}

fn assert_file_save_forms_preserve_both_sources(protocol: &Qodec) {
    let destination = tempfile::tempdir().unwrap();
    protocol.save(destination.path().join("separate")).unwrap();
    protocol.save_bundle(destination.path().join("bundle")).unwrap();
    for shape in ["separate", "bundle"] {
        let reloaded = Qodec::load(destination.path().join(shape).join("entry")).unwrap();
        assert_eq!(reloaded.layers()[0].gadgets["a"].circuit.source, "I 0\nI 0\n");
        assert_eq!(reloaded.layers()[0].gadgets["b"].circuit.source, "I 0\n");
    }
}

#[test]
fn adding_removing_and_reordering_layers_is_persisted() {
    let mut protocol = protocol_with_shared_artifacts();
    let mut unused_layer = protocol.layers()[1].clone();
    Arc::make_mut(&mut unused_layer.instruction_set).name = "new bottom".to_owned();
    protocol.layers_mut().push(unused_layer);
    assert_all_save_forms_preserve_current_values(&protocol);
    protocol.layers_mut()[0].gadgets.clear();
    protocol.layers_mut().swap(0, 1);
    assert_all_save_forms_preserve_current_values(&protocol);
    protocol.layers_mut().remove(1);
    assert_all_save_forms_preserve_current_values(&protocol);
}

fn duplicate_layers_with_a_distinct_unused_code(root: &Path) {
    let mut manifest = read_document(root, "entry");
    let layers = manifest["layers"].as_sequence_mut().unwrap();
    layers.extend(layers.clone());
    layers[2]["codes"]["spare"] = "definitions/other-unused".into();
    let mut other_code = read_document(root, "definitions/unused");
    other_code["name"] = "other-retained".into();
    fs::write(
        root.join("definitions/other-unused"),
        serde_yaml::to_string(&other_code).unwrap(),
    )
    .unwrap();
    fs::write(root.join("entry"), serde_yaml::to_string(&manifest).unwrap()).unwrap();
}

#[test]
fn unchanged_shared_gadget_documents_stay_shared() {
    let directory = save_to_temporary_directory(&protocol_with_shared_artifacts());
    duplicate_layers_with_a_distinct_unused_code(directory.path());
    let mut protocol = Qodec::load(directory.path().join("entry")).unwrap();

    let saved = save_to_temporary_directory(&protocol);
    let manifest = read_document(saved.path(), "entry");
    assert_eq!(manifest["layers"][0]["gadgets"], manifest["layers"][2]["gadgets"]);
    assert_eq!(
        Path::new(manifest["layers"][2]["codes"]["spare"].as_str().unwrap()),
        Path::new("definitions/other-unused")
    );
    protocol.layers_mut()[2]
        .gadgets
        .get_mut("a")
        .unwrap()
        .metadata
        .insert("changed".to_owned(), true.into());
    assert_all_save_forms_preserve_current_values(&protocol);
}
