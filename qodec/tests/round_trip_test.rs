//! Every shipped example must have content and survive both save layouts.

mod common;

use std::path::{Path, PathBuf};

use qodec::Qodec;

#[test]
fn every_example_survives_both_layouts_and_reconstruction() {
    let manifests = common::example_manifests();
    assert!(!manifests.is_empty(), "no example manifests listed");

    for manifest in manifests {
        let original = Qodec::load(&manifest).unwrap_or_else(|error| panic!("load {}: {error}", manifest.display()));
        assert_example_has_layers_gadgets_and_codes(&original, &manifest);
        assert_directory_round_trip(&original, &manifest);
        assert_single_file_round_trip(&original, &manifest);

        let constructed = rebuild_from_current_layers(&original);
        assert_eq!(
            constructed,
            original,
            "{}: reconstructed model differs",
            manifest.display()
        );
        assert_directory_round_trip(&constructed, &manifest);
    }
}

fn assert_example_has_layers_gadgets_and_codes(protocol: &Qodec, manifest: &Path) {
    assert!(!protocol.layers().is_empty(), "{}: no layers", manifest.display());
    assert!(!protocol.codes().is_empty(), "{}: no codes", manifest.display());
    for (index, layer) in protocol.layers().iter().enumerate() {
        if index + 1 < protocol.layers().len() {
            assert!(
                !layer.gadgets.is_empty(),
                "{}: layer {index} has no gadgets",
                manifest.display()
            );
        }
    }
}

fn assert_directory_round_trip(protocol: &Qodec, manifest: &Path) {
    let destination = tempfile::tempdir().expect("create directory destination");
    protocol
        .save(destination.path())
        .unwrap_or_else(|error| panic!("{}: directory save: {error}", manifest.display()));
    let reloaded = Qodec::load(destination.path().join(protocol.manifest_filename()))
        .unwrap_or_else(|error| panic!("{}: directory reload: {error}", manifest.display()));
    assert_eq!(
        protocol,
        &reloaded,
        "{}: directory round trip changed the model",
        manifest.display()
    );
}

fn assert_single_file_round_trip(protocol: &Qodec, manifest: &Path) {
    let destination = tempfile::tempdir().expect("create bundle destination");
    protocol
        .save_bundle(destination.path())
        .unwrap_or_else(|error| panic!("{}: bundle save: {error}", manifest.display()));
    let entries = std::fs::read_dir(destination.path())
        .expect("read bundle destination")
        .map(|entry| entry.expect("read saved entry").path())
        .collect::<Vec<_>>();
    assert_eq!(
        entries,
        [destination.path().join(protocol.manifest_filename())],
        "{}: bundle must be the only saved file",
        manifest.display()
    );
    let reloaded =
        Qodec::load(&entries[0]).unwrap_or_else(|error| panic!("{}: bundle reload: {error}", manifest.display()));
    assert_eq!(
        protocol,
        &reloaded,
        "{}: bundle round trip changed the model",
        manifest.display()
    );
}

fn rebuild_from_current_layers(original: &Qodec) -> Qodec {
    let mut rebuilt = Qodec::new(
        original.name().map(str::to_owned),
        original.description().map(str::to_owned),
        original.layers().to_vec(),
    );
    rebuilt.set_schema_version(original.schema_version());
    rebuilt.metadata_mut().clone_from(original.metadata());
    rebuilt
}

fn example(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("examples").join(name)
}

#[test]
fn an_artifact_loaded_from_above_its_manifest_is_saved_inside_the_destination() {
    let manifest = example("steane/steane.qodec.yaml");
    let text = std::fs::read_to_string(&manifest).expect("read steane manifest");
    assert!(
        text.contains("../stim.isa.yaml"),
        "this test needs an example whose artifacts sit above its own manifest",
    );

    let root = tempfile::tempdir().expect("create destination root");
    let destination = root.path().join("output");
    let original = Qodec::load(&manifest).expect("load steane");
    original.save(&destination).expect("save steane");

    let outside: Vec<_> = std::fs::read_dir(root.path())
        .expect("read destination root")
        .map(|entry| entry.expect("read entry").file_name())
        .filter(|name| name != "output")
        .collect();
    assert!(outside.is_empty(), "save wrote outside the destination: {outside:?}");

    let reloaded = Qodec::load(destination.join(original.manifest_filename())).expect("reload steane");
    assert_eq!(
        original, reloaded,
        "relocating the shared instruction set changed the model"
    );
}
