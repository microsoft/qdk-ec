//! Generated qodecs must survive both on-disk representations.
//!
//! The fixed-example round trips in `round_trip_test.rs` prove the shipped
//! examples survive; they cannot say anything about shapes no example happens
//! to have. These generate the qodec instead, so the property under test is
//! "directory and bundle are interchangeable", not "these twelve files are".

use std::fmt::Write as _;
use std::path::Path;

use proptest::prelude::*;
use qodec::Qodec;

/// Identifiers the loader treats as ordinary names — no numeric or reserved
/// forms, which have their own targeted tests.
fn identifier() -> impl Strategy<Value = String> {
    "[a-z][a-z0-9_]{0,7}"
}

/// A two-layer declaration with generated names and operand counts.
fn qodec_source() -> impl Strategy<Value = (String, String, Vec<(String, usize)>)> {
    (
        identifier(),
        identifier(),
        // Mnemonics must be distinct: the loader rejects a duplicate, so a
        // generator that emits one is testing its own bug, not the model.
        proptest::collection::btree_set(identifier(), 1..4)
            .prop_flat_map(|mnemonics| {
                let arities = proptest::collection::vec(1usize..4, mnemonics.len());
                (Just(mnemonics), arities)
            })
            .prop_map(|(mnemonics, arities)| mnemonics.into_iter().zip(arities).collect()),
    )
}

const PHYSICAL: &str = "name: phys\ndescription: physical layer\nblocks: {qubit: 1}\ninstructions:\n  - mnemonic: R\n    description: reset\n    out: [qubit]\n    action: [stabilize: \"Z_0\"]\n";

fn write_directory(root: &Path, name: &str, block: &str, instructions: &[(String, usize)]) {
    std::fs::create_dir_all(root).expect("create fixture root");

    let mut instruction_set = format!("name: top\ndescription: generated\nblocks: {{{block}: 1}}\ninstructions:\n");
    for (mnemonic, arity) in instructions {
        let operands = std::iter::repeat_n(block, *arity).collect::<Vec<_>>().join(", ");
        write!(
            instruction_set,
            "  - mnemonic: {mnemonic}\n    description: generated\n    in: [{operands}]\n    out: [{operands}]\n    action: []\n"
        )
        .expect("write to a String cannot fail");
    }

    for (relative, body) in [
        (
            "q.qodec.yaml",
            &format!(
                "name: {name}\ndescription: generated\nlayers:\n  - instruction_set: top.isa.yaml\n  - instruction_set: phys.isa.yaml\n"
            ),
        ),
        ("top.isa.yaml", &instruction_set),
        ("phys.isa.yaml", &PHYSICAL.to_owned()),
    ] {
        std::fs::write(root.join(relative), body).expect("write fixture file");
    }
}

fn load_generated_protocol(name: &str, block: &str, instructions: &[(String, usize)]) -> Qodec {
    let source = tempfile::tempdir().expect("create fixture directory");
    write_directory(source.path(), name, block, instructions);
    Qodec::load(source.path().join("q.qodec.yaml")).expect("load generated protocol")
}

fn assert_directory_round_trip(protocol: &Qodec, destination: &Path) {
    protocol.save(destination).expect("save separate artifacts");
    let reloaded = Qodec::load(destination.join(protocol.manifest_filename())).expect("reload separate artifacts");
    assert_eq!(protocol, &reloaded, "directory round trip changed the model");
}

fn assert_bundle_round_trip(protocol: &Qodec, destination: &Path) {
    protocol.save_bundle(destination).expect("save bundle");
    let reloaded = Qodec::load(destination.join(protocol.manifest_filename())).expect("reload bundle");
    assert_eq!(protocol, &reloaded, "bundle round trip changed the model");
}

fn assert_bundle_text_round_trip(protocol: &Qodec) {
    let text = protocol.to_bundle_string().expect("serialize bundle text");
    let reloaded = Qodec::from_bundle_str(&text).expect("reload bundle text");
    assert_eq!(protocol, &reloaded, "bundle text round trip changed the model");
}

fn rebuild_from_layers(protocol: &Qodec) -> Qodec {
    Qodec::new(
        protocol.name().map(str::to_owned),
        protocol.description().map(str::to_owned),
        protocol.layers().to_vec(),
    )
}

fn saved_manifest_text(directory: &Path, protocol: &Qodec) -> String {
    std::fs::read_to_string(directory.join(protocol.manifest_filename())).expect("read saved bundle")
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn generated_protocols_survive_all_representations((name, block, instructions) in qodec_source()) {
        let original = load_generated_protocol(&name, &block, &instructions);
        let destination = tempfile::tempdir().expect("create round-trip destinations");

        assert_directory_round_trip(&original, &destination.path().join("directory"));
        assert_bundle_round_trip(&original, &destination.path().join("bundle"));
        assert_bundle_text_round_trip(&original);

        let rebuilt = rebuild_from_layers(&original);
        prop_assert_eq!(&rebuilt, &original, "reconstruction changed the model");
        assert_directory_round_trip(&rebuilt, &destination.path().join("rebuilt"));
    }

    /// A bundle re-saved as a bundle must be byte-stable: the second pass has
    /// nothing left to normalize, so any difference is the writer disagreeing
    /// with itself.
    #[test]
    fn saving_a_bundle_twice_is_stable((name, block, instructions) in qodec_source()) {
        let original = load_generated_protocol(&name, &block, &instructions);
        let first = tempfile::tempdir().expect("create first bundle destination");
        let second = tempfile::tempdir().expect("create second bundle destination");

        original.save_bundle(first.path()).expect("first save");
        let reloaded = Qodec::load(first.path().join(original.manifest_filename())).expect("reload first bundle");
        reloaded.save_bundle(second.path()).expect("second save");

        prop_assert_eq!(saved_manifest_text(first.path(), &original), saved_manifest_text(second.path(), &reloaded),
            "saving the same bundle twice changed its bytes");
    }
}
