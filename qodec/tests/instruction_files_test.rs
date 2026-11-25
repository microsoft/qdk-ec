use qodec::{InstructionSet, LoadError, Qodec};
use serde::Deserialize;
use std::path::Path;

const RESET: &str = "mnemonic: reset\ndescription: Prepare zero.\nout: [qubit]\naction: [stabilize: Z_0]\n";

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

fn factored_fixture() -> (tempfile::TempDir, Qodec) {
    let directory = tempfile::tempdir().unwrap();
    write(
        &directory.path().join("entry.yaml"),
        "layers: [{instruction_set: isa/logical.yaml}]\n",
    );
    write(
        &directory.path().join("isa").join("logical.yaml"),
        "name: logical\nblocks: {qubit: 1}\ninstructions:\n\
         - definitions/reset.instruction.yaml\n- mnemonic: inline\n  description: Inline declaration.\n",
    );
    write(
        &directory
            .path()
            .join("isa")
            .join("definitions")
            .join("reset.instruction.yaml"),
        RESET,
    );
    let model = Qodec::load(directory.path().join("entry.yaml")).unwrap();
    (directory, model)
}

#[test]
fn standalone_loading_resolves_file_entries_and_preserves_inline_entries() {
    let directory = tempfile::tempdir().unwrap();
    let isa = directory.path().join("isa").join("logical.yaml");
    let instruction = directory
        .path()
        .join("isa")
        .join("definitions")
        .join("reset.instruction.yaml");
    write(&instruction, RESET);
    write(
        &isa,
        "name: logical\nblocks: {qubit: 1}\ninstructions:\n\
         - definitions/reset.instruction.yaml\n- mnemonic: inline\n  description: Inline declaration.\n",
    );

    let loaded = InstructionSet::load(&isa).unwrap();
    assert!(
        serde_yaml::from_str::<InstructionSet>(&std::fs::read_to_string(&isa).unwrap()).is_err(),
        "direct serde deserialization remains an inline-only value operation"
    );
    assert_eq!(
        loaded
            .instructions
            .iter()
            .map(|instruction| instruction.mnemonic.as_str())
            .collect::<Vec<_>>(),
        ["reset", "inline"]
    );
    let saved = directory.path().join("saved").join("standalone.yaml");
    loaded.save(&saved).unwrap();
    std::fs::remove_file(instruction).unwrap();
    assert_eq!(InstructionSet::load(saved).unwrap(), loaded);
}

#[test]
fn standalone_instruction_file_errors_include_the_isa_entry_and_path() {
    let directory = tempfile::tempdir().unwrap();
    let isa = directory.path().join("logical.yaml");
    let instruction = directory.path().join("missing");
    write(&isa, "name: logical\nblocks: {qubit: 1}\ninstructions: [missing]\n");

    let error = InstructionSet::load(&isa).unwrap_err();
    let LoadError::InstructionFile {
        instruction_set,
        index,
        path,
        source,
    } = error
    else {
        panic!("expected referenced instruction error");
    };
    assert_eq!(instruction_set, isa);
    assert_eq!(index, 0);
    assert_eq!(path, instruction);
    assert!(matches!(*source, LoadError::Io(_)));

    for text in [
        "another.yaml",
        "- mnemonic: list\n  description: Not a single instruction.\n",
        "name: another_isa\nblocks: {}\ninstructions: []\n",
        "mnemonic: malformed\nunknown: 1\n",
    ] {
        write(&instruction, text);
        let error = InstructionSet::load(&isa).unwrap_err();
        let LoadError::InstructionFile { source, .. } = error else {
            panic!("expected referenced instruction parse error");
        };
        assert!(matches!(*source, LoadError::Yaml { .. }), "{text}");
    }

    write(&instruction, RESET);
    write(
        &isa,
        "name: logical\nblocks: {qubit: 1}\ninstructions: [missing, missing]\n",
    );
    assert!(matches!(
        InstructionSet::load(&isa),
        Err(LoadError::InvalidInstructionSet { .. })
    ));
}

#[test]
fn qodec_instruction_file_errors_retain_reference_context() {
    let directory = tempfile::tempdir().unwrap();
    write(
        &directory.path().join("entry.yaml"),
        "layers: [{instruction_set: isa/logical.yaml}]\n",
    );
    write(
        &directory.path().join("isa").join("logical.yaml"),
        "name: logical\nblocks: {qubit: 1}\ninstructions: [missing]\n",
    );
    let error = Qodec::load(directory.path().join("entry.yaml")).unwrap_err();
    let LoadError::InstructionFile {
        instruction_set,
        index,
        path,
        source,
    } = error
    else {
        panic!("expected referenced instruction error");
    };
    assert_eq!(instruction_set, Path::new("isa").join("logical.yaml"));
    assert_eq!(index, 0);
    assert_eq!(path, Path::new("isa").join("missing"));
    assert!(matches!(*source, LoadError::MissingArtifact { .. }));
}

#[test]
fn qodec_directory_and_bundle_loads_resolve_instruction_paths_from_the_isa() {
    let (_source, model) = factored_fixture();
    assert_eq!(model.layers()[0].instruction_set.instructions[0].mnemonic, "reset");
    assert_eq!(model.layers()[0].instruction_set.instructions[1].mnemonic, "inline");

    let bundle = model.to_bundle_string().unwrap();
    let loaded = Qodec::from_bundle_str(&bundle).unwrap();
    assert_eq!(loaded, model);
    let destination = tempfile::tempdir().unwrap();
    assert_eq!(
        Qodec::load(loaded.save_bundle(destination.path()).unwrap()).unwrap(),
        model
    );
}

#[test]
fn qodec_saves_preserve_instruction_file_choices_and_current_edits() {
    let (_source, model) = factored_fixture();
    let directory = tempfile::tempdir().unwrap();
    let manifest = model.save(directory.path()).unwrap();
    let isa: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(directory.path().join("isa").join("logical.yaml")).unwrap())
            .unwrap();
    assert!(isa["instructions"][0].is_string());
    assert!(isa["instructions"][1].is_mapping());
    assert!(
        directory
            .path()
            .join("isa")
            .join("definitions")
            .join("reset.instruction.yaml")
            .is_file()
    );
    assert_eq!(Qodec::load(manifest).unwrap(), model);

    let (_source, mut edited) = factored_fixture();
    let isa = std::sync::Arc::make_mut(&mut edited.layers_mut()[0].instruction_set);
    isa.instructions[0].description = "Edited reset.".to_owned();
    isa.instructions.swap(0, 1);
    let destination = tempfile::tempdir().unwrap();
    let manifest = edited.save(destination.path()).unwrap();
    let fragment = std::fs::read_to_string(
        destination
            .path()
            .join("isa")
            .join("definitions")
            .join("reset.instruction.yaml"),
    )
    .unwrap();
    assert!(fragment.contains("Edited reset."));
    let isa: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(destination.path().join("isa").join("logical.yaml")).unwrap())
            .unwrap();
    assert!(isa["instructions"][0].is_mapping());
    assert!(isa["instructions"][1].is_string());
    assert_eq!(Qodec::load(manifest).unwrap(), edited);
}

#[test]
fn renamed_instructions_are_saved_inline_without_reusing_the_old_file() {
    let (source, mut model) = factored_fixture();
    std::sync::Arc::make_mut(&mut model.layers_mut()[0].instruction_set).instructions[0].mnemonic =
        "prepare_zero".to_owned();

    let destination = tempfile::tempdir().unwrap();
    let manifest = model.save(destination.path()).unwrap();
    let isa: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(destination.path().join("isa").join("logical.yaml")).unwrap())
            .unwrap();
    assert!(isa["instructions"][0].is_mapping());
    assert_eq!(isa["instructions"][0]["mnemonic"], "prepare_zero");
    let fragment = Path::new("isa").join("definitions").join("reset.instruction.yaml");
    assert!(!destination.path().join(&fragment).exists());
    assert_eq!(std::fs::read_to_string(source.path().join(fragment)).unwrap(), RESET);
    assert_eq!(Qodec::load(manifest).unwrap(), model);

    let bundle = model.to_bundle_string().unwrap();
    assert!(!bundle.contains("reset.instruction.yaml"));
    assert_eq!(Qodec::from_bundle_str(&bundle).unwrap(), model);
}

#[test]
fn instruction_source_locations_point_to_the_fragment_document() {
    let (source, model) = factored_fixture();
    let node = model
        .resolve("layers[0].instruction_set.instructions[\"reset\"].description")
        .unwrap();
    let location = node.source_location().unwrap();
    assert_eq!(
        location.path(),
        source
            .path()
            .join("isa")
            .join("definitions")
            .join("reset.instruction.yaml")
    );
    assert_eq!(location.line(), 2);
}

#[test]
fn unchanged_external_instruction_files_are_reused_and_checked_before_saving() {
    let source = tempfile::tempdir().unwrap();
    let manifest = source.path().join("project").join("entry");
    let instruction_set = source.path().join("shared").join("logical");
    let instruction = source.path().join("shared").join("reset");
    write(&manifest, "layers: [{instruction_set: ../shared/logical}]\n");
    write(
        &instruction_set,
        "name: logical\nblocks: {qubit: 1}\ninstructions: [reset]\n",
    );
    write(&instruction, RESET);
    let model = Qodec::load(manifest).unwrap();

    let destination = tempfile::tempdir().unwrap();
    let saved = model.save(destination.path()).unwrap();
    assert_eq!(Qodec::load(saved).unwrap(), model);
    assert_eq!(std::fs::read_dir(destination.path()).unwrap().count(), 1);

    write(&instruction, &RESET.replace("Prepare zero.", "Changed externally."));
    let not_written = destination.path().join("not-written");
    assert!(
        model
            .save(&not_written)
            .unwrap_err()
            .to_string()
            .contains("changed since loading")
    );
    assert!(!not_written.exists());
}

#[test]
fn unchanged_shared_instruction_files_remain_shared_in_directories_and_bundles() {
    let source = tempfile::tempdir().unwrap();
    write(
        &source.path().join("entry"),
        "layers: [{instruction_set: first}, {instruction_set: second}]\n",
    );
    for name in ["first", "second"] {
        write(
            &source.path().join(name),
            &format!("name: {name}\nblocks: {{qubit: 1}}\ninstructions: [reset]\n"),
        );
    }
    write(&source.path().join("reset"), RESET);
    let model = Qodec::load(source.path().join("entry")).unwrap();

    let destination = tempfile::tempdir().unwrap();
    let manifest = model.save(destination.path()).unwrap();
    let references = ["first", "second"].map(|name| {
        let isa: serde_yaml::Value =
            serde_yaml::from_str(&std::fs::read_to_string(destination.path().join(name)).unwrap()).unwrap();
        isa["instructions"][0].as_str().unwrap().to_owned()
    });
    assert_eq!(references[0], references[1]);
    assert_eq!(std::fs::read_dir(destination.path()).unwrap().count(), 4);
    assert_eq!(
        serde_yaml::from_str::<qodec::Instruction>(
            &std::fs::read_to_string(destination.path().join(&references[0])).unwrap()
        )
        .unwrap(),
        model.layers()[0].instruction_set.instructions[0]
    );
    assert_eq!(Qodec::load(manifest).unwrap(), model);

    let bundle = model.to_bundle_string().unwrap();
    let documents: std::collections::BTreeMap<String, serde_yaml::Value> = serde_yaml::Deserializer::from_str(&bundle)
        .flat_map(|document| std::collections::BTreeMap::<String, serde_yaml::Value>::deserialize(document).unwrap())
        .collect();
    let references = ["first", "second"].map(|name| documents[name]["instructions"][0].as_str().unwrap().to_owned());
    assert_eq!(references[0], references[1]);
    assert_eq!(documents.len(), 4);
    assert_eq!(documents[&references[0]]["mnemonic"], "reset");
    assert_eq!(
        documents
            .values()
            .filter(|document| document["mnemonic"] == "reset")
            .count(),
        1
    );
    assert_eq!(Qodec::from_bundle_str(&bundle).unwrap(), model);
}

#[test]
fn shared_instruction_files_split_when_definitions_diverge() {
    let source = tempfile::tempdir().unwrap();
    write(
        &source.path().join("entry"),
        "layers: [{instruction_set: first}, {instruction_set: second}]\n",
    );
    for (file, name) in [("first", "first"), ("second", "second")] {
        write(
            &source.path().join(file),
            &format!("name: {name}\nblocks: {{qubit: 1}}\ninstructions: [reset]\n"),
        );
    }
    write(&source.path().join("reset"), RESET);
    let mut model = Qodec::load(source.path().join("entry")).unwrap();
    std::sync::Arc::make_mut(&mut model.layers_mut()[0].instruction_set).instructions[0].description =
        "First layer only.".to_owned();

    let destination = tempfile::tempdir().unwrap();
    assert_eq!(Qodec::load(model.save(destination.path()).unwrap()).unwrap(), model);
    let paths: Vec<_> = ["first", "second"]
        .map(|file| {
            let document: serde_yaml::Value =
                serde_yaml::from_str(&std::fs::read_to_string(destination.path().join(file)).unwrap()).unwrap();
            document["instructions"][0].as_str().unwrap().to_owned()
        })
        .to_vec();
    assert_ne!(paths[0], paths[1]);
    assert_eq!(std::fs::read_to_string(source.path().join("reset")).unwrap(), RESET);
}

#[test]
fn bundles_resolve_instruction_files_relative_to_isa_keys() {
    let source = "entry:\n  layers: [{instruction_set: definitions/logical}]\n\
        ---\ndefinitions/logical:\n  name: logical\n  blocks: {qubit: 1}\n  instructions: [../operations/reset]\n\
        ---\noperations/reset:\n  mnemonic: reset\n  description: Prepare zero.\n\
        ---\nunused:\n  not: an_instruction\n";
    let model = Qodec::from_bundle_str(source).unwrap();
    assert_eq!(model.layers()[0].instruction_set.instructions[0].mnemonic, "reset");
    let bundle: Vec<_> = serde_yaml::Deserializer::from_str(&model.to_bundle_string().unwrap())
        .map(|document| {
            let value = serde_yaml::Value::deserialize(document).unwrap();
            value.as_mapping().unwrap().values().next().unwrap().clone()
        })
        .collect();
    let instruction_set = bundle.iter().find(|document| document["name"] == "logical").unwrap();
    assert!(instruction_set["instructions"][0].is_string());
    assert!(bundle.iter().any(|document| document["mnemonic"] == "reset"));
    assert_eq!(
        Qodec::from_bundle_str(&model.to_bundle_string().unwrap()).unwrap(),
        model
    );
}
