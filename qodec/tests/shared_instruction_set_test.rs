//! Shared instruction set file support.
//!
//! An example references one canonical `../stim.isa.yaml` instead of
//! inlining the standard gate set in its bundle. These tests copy the real
//! `steane` example (whose bottom layer is `instruction_set: ../stim.isa.yaml`) plus the
//! shared file into a temp layout and verify that `Qodec::load` resolves the
//! external instruction set — and that omitting the shared file is a clean load error
//! rather than a silent success.

use std::path::{Path, PathBuf};

use qodec::Qodec;

fn examples() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("examples")
}

/// Copy `examples/steane/steane.qodec.yaml` into `<root>/steane/`, mirroring
/// the examples layout. The bundle's `instruction_set: ../stim.isa.yaml` resolves to
/// `<root>/stim.isa.yaml`.
fn place_example(root: &Path) -> PathBuf {
    let example = root.join("steane");
    std::fs::create_dir_all(&example).expect("mkdir example");
    std::fs::copy(
        examples().join("steane/steane.qodec.yaml"),
        example.join("steane.qodec.yaml"),
    )
    .expect("copy bundle");
    example
}

#[test]
fn example_resolves_shared_instruction_set_file() {
    let directory = tempfile::tempdir().expect("create shared instruction set fixture");
    let example = place_example(directory.path());
    std::fs::copy(examples().join("stim.isa.yaml"), directory.path().join("stim.isa.yaml"))
        .expect("copy shared instruction set");

    let qodec =
        Qodec::load(example.join("steane.qodec.yaml")).expect("load example whose physical layer is a shared file");
    let names: Vec<&str> = qodec
        .layers()
        .iter()
        .map(|layer| layer.instruction_set.name.as_str())
        .collect();
    assert_eq!(
        names.last().copied(),
        Some("stim"),
        "the shared bottom-layer file must resolve into the stack, got {names:?}",
    );
    assert!(
        qodec.layers()[0].instruction_set.name != "stim",
        "the logical layer must keep its own instruction set",
    );
}

#[test]
fn missing_shared_instruction_set_file_is_a_load_error() {
    let directory = tempfile::tempdir().expect("create fixture without the shared instruction set");
    let example = place_example(directory.path());
    let error =
        Qodec::load(example.join("steane.qodec.yaml")).expect_err("missing shared instruction set must fail to load");
    assert!(error.to_string().contains("stim.isa.yaml"), "{error}");
}
