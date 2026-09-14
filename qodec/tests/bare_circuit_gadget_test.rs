//! Circuit-only gadgets use minimal YAML wrappers such as `circuit: ./idle.stim`.
//! Every manifest gadget reference names a YAML gadget document, regardless of
//! its file extension. Loading preserves the circuit source and infers the
//! implemented instruction and target instruction set from the layers. Saving retains both
//! the wrapper and its circuit file. Fixtures strip checks from five C4 gadgets
//! to test loadable drafts without removing checks from the shipped example.

use std::path::{Path, PathBuf};

use qodec::Qodec;

/// C4 gadgets reduced to circuit-only wrappers in the temporary fixtures.
const CIRCUIT_ONLY: &[&str] = &["controlled_x_all", "hadamard_all", "idle", "mul_u", "mul_u_sq"];

fn c4c6_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("examples").join("c4c6")
}

fn load_c4c6() -> Qodec {
    Qodec::load(c4c6_dir().join("qodec.yaml")).expect("load C4/C6")
}

fn save_c4c6() -> (tempfile::TempDir, Qodec) {
    let mut protocol = load_c4c6();
    for mnemonic in CIRCUIT_ONLY {
        protocol.layers_mut()[1]
            .gadgets
            .get_mut(*mnemonic)
            .unwrap()
            .checks
            .clear();
    }
    let directory = tempfile::tempdir().expect("create wrapper fixture");
    protocol
        .save(directory.path())
        .expect("save C4/C6 wrappers and source files");
    for mnemonic in CIRCUIT_ONLY {
        std::fs::write(
            directory.path().join(format!("c4/{mnemonic}.gadget.yaml")),
            format!("circuit: ./{mnemonic}.stim\n"),
        )
        .expect("write minimal wrapper fixture");
    }
    (directory, protocol)
}

fn set_idle_gadget_reference(manifest_path: &Path, reference: &str) {
    let text = std::fs::read_to_string(manifest_path).expect("read saved manifest");
    let mut manifest: serde_yaml::Value = serde_yaml::from_str(&text).expect("parse saved manifest");
    manifest["layers"][1]["gadgets"]["idle"] = reference.into();
    std::fs::write(
        manifest_path,
        serde_yaml::to_string(&manifest).expect("serialize manifest"),
    )
    .expect("update idle gadget reference");
}

#[test]
fn minimal_yaml_wrappers_load_and_resolve() {
    let (directory, original) = save_c4c6();
    let qodec = Qodec::load(directory.path().join(original.manifest_filename())).expect("load circuit-only wrappers");
    for mnemonic in CIRCUIT_ONLY {
        assert_wrapper_contains_only_source_path(directory.path(), mnemonic);
        assert_resolved_wrapper_matches_sidecar(&qodec, mnemonic);
    }
}

fn assert_wrapper_contains_only_source_path(root: &Path, mnemonic: &str) {
    let wrapper = std::fs::read_to_string(root.join(format!("c4/{mnemonic}.gadget.yaml"))).expect("read YAML wrapper");
    let wrapper: serde_yaml::Value = serde_yaml::from_str(&wrapper).expect("parse YAML wrapper");
    assert_eq!(wrapper.as_mapping().expect("wrapper is a mapping").len(), 1);
    assert_eq!(wrapper["circuit"], format!("./{mnemonic}.stim"));
}

fn assert_resolved_wrapper_matches_sidecar(qodec: &Qodec, mnemonic: &str) {
    let source = std::fs::read_to_string(c4c6_dir().join(format!("c4/{mnemonic}.stim"))).expect("read circuit source");
    let resolved = qodec.layers()[1]
        .gadgets
        .get(mnemonic)
        .unwrap_or_else(|| panic!("{mnemonic} should resolve"));
    assert_eq!(resolved.implements.mnemonic, mnemonic);
    assert_eq!(
        resolved.circuit.instruction_set.name,
        qodec.layers()[2].instruction_set.name
    );
    assert_eq!(resolved.circuit.source, source);
}

#[test]
fn minimal_yaml_wrapper_save_preserves_wrapper_and_circuit() {
    let (directory, qodec) = save_c4c6();
    let destination = directory.path();

    let manifest_path = destination.join(qodec.manifest_filename());
    let manifest = std::fs::read_to_string(&manifest_path).expect("read saved manifest");
    let manifest: serde_yaml::Value = serde_yaml::from_str(&manifest).expect("parse saved manifest");
    let reloaded = Qodec::load(&manifest_path).expect("the saved bundle should reload");

    for mnemonic in CIRCUIT_ONLY {
        assert_saved_wrapper_files(destination, &manifest, mnemonic);
        assert_saved_wrapper_values(&qodec, &reloaded, mnemonic);
    }
    assert!(destination.join("c4/prepare_x_all.gadget.yaml").exists());
}

fn assert_saved_wrapper_files(destination: &Path, manifest: &serde_yaml::Value, mnemonic: &str) {
    let wrapper_path = Path::new("c4").join(format!("{mnemonic}.gadget.yaml"));
    assert!(
        destination.join(&wrapper_path).is_file(),
        "the YAML wrapper should be written for {mnemonic}",
    );
    let reference = manifest["layers"][1]["gadgets"][mnemonic]
        .as_str()
        .expect("gadget path");
    assert_eq!(Path::new(reference), wrapper_path);

    let circuit_path = format!("c4/{mnemonic}.stim");
    assert!(
        destination.join(&circuit_path).is_file(),
        "the `.stim` source should be written for {mnemonic}",
    );
    assert_eq!(
        std::fs::read(destination.join(&circuit_path)).expect("read saved circuit"),
        std::fs::read(c4c6_dir().join(&circuit_path)).expect("read original circuit"),
    );
}

fn assert_saved_wrapper_values(qodec: &Qodec, reloaded: &Qodec, mnemonic: &str) {
    let original = &qodec.layers()[1].gadgets[mnemonic];
    let saved = &reloaded.layers()[1].gadgets[mnemonic];
    assert_eq!(saved.implements.mnemonic, original.implements.mnemonic);
    assert_eq!(
        saved.circuit.instruction_set.name,
        original.circuit.instruction_set.name
    );
    assert_eq!(saved.circuit.source, original.circuit.source);
}

#[test]
fn gadget_references_do_not_infer_type_from_extension() {
    let (directory, qodec) = save_c4c6();
    let destination = directory.path();
    let manifest_path = destination.join(qodec.manifest_filename());

    for wrapper_path in ["c4/idle-wrapper", "c4/idle-wrapper.stim"] {
        std::fs::copy(destination.join("c4/idle.gadget.yaml"), destination.join(wrapper_path))
            .expect("copy YAML wrapper");
        set_idle_gadget_reference(&manifest_path, wrapper_path);

        let loaded = Qodec::load(&manifest_path).expect("gadget reference should load YAML regardless of extension");
        assert_eq!(
            loaded.layers()[1].gadgets["idle"].circuit.source,
            qodec.layers()[1].gadgets["idle"].circuit.source,
        );
    }

    set_idle_gadget_reference(&manifest_path, "c4/idle.stim");
    let error = Qodec::load(&manifest_path).expect_err("a gadget reference must name YAML, not a bare circuit");
    let circuit_path = Path::new("c4").join("idle.stim");
    assert!(
        error.to_string().contains(&circuit_path.display().to_string()),
        "{error}"
    );
}
