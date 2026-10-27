//! External readouts support: round-trip and unreferenced-file isolation.
//!
//! A saved C4/C6 example supplies the referenced artifacts without copying
//! unrelated catalog files. Readout edits operate on parsed YAML values.

use std::path::Path;

use qodec::Qodec;

fn save_c4c6_example() -> (tempfile::TempDir, Qodec) {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/c4c6/qodec.yaml");
    let protocol = Qodec::load(manifest).expect("load C4/C6");
    let destination = tempfile::tempdir().expect("create fixture directory");
    protocol
        .save(destination.path())
        .expect("save referenced example artifacts");
    (destination, protocol)
}

fn externalize_measurement_readouts(root: &Path) {
    let gadget_path = root.join("c4/measure_z_all.gadget.yaml");
    let text = std::fs::read_to_string(&gadget_path).expect("read measurement gadget");
    let mut gadget: serde_yaml::Value = serde_yaml::from_str(&text).expect("parse measurement gadget");
    let inline_readouts = gadget
        .get("readouts")
        .cloned()
        .expect("original gadget has inline readouts");
    assert!(
        inline_readouts.as_sequence().is_some_and(|s| !s.is_empty()),
        "fixture invariant: readouts non-empty"
    );

    gadget["readouts"] = "./measure_z_all.readouts.yaml".into();
    std::fs::write(
        &gadget_path,
        serde_yaml::to_string(&gadget).expect("serialize external reference"),
    )
    .expect("write measurement gadget");
    std::fs::write(
        root.join("c4/measure_z_all.readouts.yaml"),
        serde_yaml::to_string(&inline_readouts).expect("serialize readouts"),
    )
    .expect("write external readouts yaml");
}

#[test]
fn external_readouts_file_round_trips_and_resolves_identically() {
    let (directory, original) = save_c4c6_example();
    externalize_measurement_readouts(directory.path());
    let reloaded = Qodec::load(directory.path().join(original.manifest_filename())).expect("resolve external readouts");
    assert_eq!(original, reloaded, "externalizing readouts must preserve the model");
}

#[test]
fn unreferenced_readouts_file_is_ignored() {
    let (directory, original) = save_c4c6_example();
    let unreferenced = directory.path().join("c4/never_referenced.readouts.yaml");
    std::fs::write(&unreferenced, "[invalid YAML\n").expect("write unreferenced readouts");

    let reloaded = Qodec::load(directory.path().join(original.manifest_filename()))
        .expect("unreferenced malformed readouts are ignored");
    assert_eq!(original, reloaded);
}
