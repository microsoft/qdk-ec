use qodec::{LoadError, Qodec};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

fn write_yaml(path: &Path, value: &serde_yaml::Value) {
    write(path, &serde_yaml::to_string(value).unwrap());
}

fn fixture() -> (tempfile::TempDir, Qodec) {
    let root = tempfile::tempdir().unwrap();
    let documents: BTreeMap<String, serde_yaml::Value> =
        serde_yaml::Deserializer::from_str(include_str!("../examples/repetition3/repetition3.qodec.yaml"))
            .flat_map(|doc| BTreeMap::<String, serde_yaml::Value>::deserialize(doc).unwrap())
            .collect();
    write(
        &root.path().join("model.yaml"),
        "layers:\n  - logical/layer.yaml\n  - instruction_set: physical.yaml\n",
    );
    write(
        &root.path().join("logical/layer.yaml"),
        "instruction_set: instructions/logical.yaml\ncodes: {repetition3: code.yaml}\ngadgets:\n  prepare_z: gadgets/prepare.yaml\n  idle: gadgets/idle.yaml\n  measure_z: gadgets/measure.yaml\n  rotate_z: gadgets/rotate.yaml\n",
    );
    let mut isa = documents["repetition3.isa.yaml"].clone();
    let instructions = isa["instructions"].as_sequence_mut().unwrap();
    let first = std::mem::replace(&mut instructions[0], serde_yaml::Value::String("prepare.yaml".into()));
    write_yaml(&root.path().join("logical/instructions/logical.yaml"), &isa);
    write_yaml(&root.path().join("logical/instructions/prepare.yaml"), &first);
    write_yaml(
        &root.path().join("logical/code.yaml"),
        &documents["repetition3.code.yaml"],
    );
    write_yaml(&root.path().join("physical.yaml"), &documents["stim+rz.isa.yaml"]);
    for (source, dest) in [
        ("prepare_z.gadget.yaml", "prepare"),
        ("idle.gadget.yaml", "idle"),
        ("measure_z.gadget.yaml", "measure"),
        ("rotate_z.gadget.yaml", "rotate"),
    ] {
        write_yaml(
            &root.path().join("logical/gadgets").join(format!("{dest}.yaml")),
            &documents[source],
        );
    }
    let model = Qodec::load(root.path().join("model.yaml")).unwrap();
    (root, model)
}

#[test]
fn nested_layer_references_compose_and_save_in_both_formats() {
    let (source, model) = fixture();
    assert_eq!(model.layers().len(), 2);
    assert_eq!(model.layers()[0].instruction_set.instructions[0].mnemonic, "prepare_z");
    assert_eq!(model.layers()[0].gadgets.len(), 4);
    let node = model.resolve("layers[0]").unwrap();
    let location = node.source_location().unwrap();
    assert_eq!(location.path(), source.path().join("logical/layer.yaml"));
    let node = model
        .resolve("layers[0].instruction_set.instructions[\"prepare_z\"].description")
        .unwrap();
    let location = node.source_location().unwrap();
    assert_eq!(location.path(), source.path().join("logical/instructions/prepare.yaml"));

    let output = tempfile::tempdir().unwrap();
    let manifest = model.save(output.path()).unwrap();
    let saved: serde_yaml::Value = serde_yaml::from_str(&std::fs::read_to_string(&manifest).unwrap()).unwrap();
    assert_eq!(
        Path::new(saved["layers"][0].as_str().unwrap()),
        Path::new("logical").join("layer.yaml")
    );
    assert!(saved["layers"][1].is_mapping());
    assert_eq!(Qodec::load(manifest).unwrap(), model);
    let saved_layer: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(output.path().join("logical/layer.yaml")).unwrap()).unwrap();
    assert_eq!(
        Path::new(saved_layer["instruction_set"].as_str().unwrap()),
        Path::new("instructions").join("logical.yaml")
    );
    assert!(output.path().join("logical/instructions/prepare.yaml").is_file());
    assert!(output.path().join("logical/gadgets/prepare.yaml").is_file());

    let bundle = model.to_bundle_string().unwrap();
    let first = serde_yaml::Deserializer::from_str(&bundle).next().unwrap();
    let envelope = BTreeMap::<String, serde_yaml::Value>::deserialize(first).unwrap();
    assert_eq!(
        Path::new(envelope["model.yaml"]["layers"][0].as_str().unwrap()),
        Path::new("logical").join("layer.yaml")
    );
    assert_eq!(Qodec::from_bundle_str(&bundle).unwrap(), model);
    let bundle_output = tempfile::tempdir().unwrap();
    assert_eq!(
        Qodec::load(model.save_bundle(bundle_output.path()).unwrap()).unwrap(),
        model
    );
}

#[test]
fn layer_file_errors_include_manifest_index_and_reference() {
    let directory = tempfile::tempdir().unwrap();
    let manifest = directory.path().join("entry");
    write(&manifest, "layers: [nested/layer.yaml]\n");
    let error = Qodec::load(&manifest).unwrap_err();
    let LoadError::LayerFile {
        manifest: referring,
        index,
        path,
        source,
    } = error
    else {
        panic!("expected layer file error");
    };
    assert_eq!(referring, Path::new("entry"));
    assert_eq!(index, 0);
    assert_eq!(path, Path::new("nested").join("layer.yaml"));
    assert!(matches!(*source, LoadError::MissingArtifact { .. }));
    write(&directory.path().join("nested/layer.yaml"), "instruction_set: [bad]\n");
    assert!(
        matches!(Qodec::load(&manifest), Err(LoadError::LayerFile { source, .. }) if matches!(*source, LoadError::Yaml { .. }))
    );
    write(
        &directory.path().join("nested/layer.yaml"),
        "instruction_set: absent.yaml\n",
    );
    assert!(
        matches!(Qodec::load(&manifest), Err(LoadError::MissingArtifact { referenced_from, path }) if referenced_from == Path::new("nested").join("layer.yaml") && path == Path::new("nested").join("absent.yaml"))
    );
    write(
        &directory.path().join("nested/layer.yaml"),
        "instruction_set: layer.yaml\n",
    );
    assert!(matches!(
        Qodec::load(&manifest),
        Err(LoadError::ConflictingArtifactKind {
            kind: "instruction set",
            previous: "layer",
            ..
        })
    ));
}

#[test]
fn malformed_inline_layer_reports_the_nested_field_location() {
    let directory = tempfile::tempdir().unwrap();
    let manifest = directory.path().join("entry");
    write(
        &manifest,
        "layers:\n  - instruction_set: isa.yaml\n    codes:\n      qubit: [bad]\n",
    );
    let LoadError::Yaml { source, .. } = Qodec::load(&manifest).unwrap_err() else {
        panic!("expected manifest YAML error");
    };
    assert_eq!(source.location(), Some((4, 14)));
    assert!(source.to_string().contains("layers[0].codes.qubit"), "{source}");
}

#[test]
fn nested_bundle_keys_determine_each_reference_base() {
    let bundle = "\
model/entry:
  layers:
    - logical/layer
    - instruction_set: physical
---
model/logical/layer:
  instruction_set: isa
---
model/logical/isa:
  name: logical
  blocks: {}
  instructions: [definitions/reset]
---
model/logical/definitions/reset:
  mnemonic: reset
  description: Reset.
---
model/physical:
  name: physical
  blocks: {}
  instructions: []
";
    let model = Qodec::from_bundle_str(bundle).unwrap();
    assert_eq!(model.layers()[0].instruction_set.instructions[0].mnemonic, "reset");
    let saved = model.to_bundle_string().unwrap();
    assert_eq!(Qodec::from_bundle_str(&saved).unwrap(), model);
}

#[test]
fn edited_layer_preserves_factoring_and_current_values() {
    let (_source, mut model) = fixture();
    std::sync::Arc::make_mut(&mut model.layers_mut()[0].instruction_set).description = "Edited.".into();
    let output = tempfile::tempdir().unwrap();
    let manifest = model.save(output.path()).unwrap();
    let isa = std::fs::read_to_string(output.path().join("logical/instructions/logical.yaml")).unwrap();
    assert!(isa.contains("Edited."));
    assert_eq!(Qodec::load(manifest).unwrap(), model);
}

#[test]
fn external_layers_are_reused_only_when_unchanged_and_never_overwritten() {
    let source = tempfile::tempdir().unwrap();
    let manifest = source.path().join("project/entry");
    let layer = source.path().join("shared/layer");
    write(&manifest, "layers: [../shared/layer]\n");
    write(&layer, "instruction_set: isa\n");
    write(
        &source.path().join("shared/isa"),
        "name: shared\nblocks: {}\ninstructions: []\n",
    );
    let mut model = Qodec::load(&manifest).unwrap();
    let original = std::fs::read_to_string(&layer).unwrap();
    let output = tempfile::tempdir().unwrap();
    let saved = model.save(output.path()).unwrap();
    let value: serde_yaml::Value = serde_yaml::from_str(&std::fs::read_to_string(&saved).unwrap()).unwrap();
    assert!(Path::new(value["layers"][0].as_str().unwrap()).is_absolute());
    assert_eq!(Qodec::load(saved).unwrap(), model);
    let bundle = model.to_bundle_string().unwrap();
    assert_eq!(Qodec::from_bundle_str(&bundle).unwrap(), model);

    std::sync::Arc::make_mut(&mut model.layers_mut()[0].instruction_set).description = "Changed.".into();
    let edited = tempfile::tempdir().unwrap();
    let saved = model.save(edited.path()).unwrap();
    let value: serde_yaml::Value = serde_yaml::from_str(&std::fs::read_to_string(&saved).unwrap()).unwrap();
    assert!(!Path::new(value["layers"][0].as_str().unwrap()).is_absolute());
    assert_eq!(std::fs::read_to_string(&layer).unwrap(), original);
    assert_eq!(Qodec::load(saved).unwrap(), model);

    let unchanged = Qodec::load(&manifest).unwrap();
    write(&layer, "instruction_set: isa\n# changed after loading\n");
    let destination = tempfile::tempdir().unwrap();
    assert!(
        unchanged
            .save(destination.path())
            .unwrap_err()
            .to_string()
            .contains("changed since loading")
    );
}
