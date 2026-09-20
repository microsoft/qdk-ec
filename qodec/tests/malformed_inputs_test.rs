//! Malformed inputs must produce errors, not panics.
//!
//! Several hot paths carry `expect` on invariants a *previous* pass is
//! supposed to have established — the resolver assumes every layer instruction set was
//! ingested, the lowering builder assumes every instruction set name resolves. Those hold
//! only as long as the checks that establish them run first, which is exactly
//! the kind of coupling that rots. Each case below reaches one of them.

use std::path::Path;

struct Fixture(tempfile::TempDir);

impl Fixture {
    fn new(name: &str, files: &[(&str, &str)]) -> Self {
        let root = tempfile::Builder::new()
            .prefix(&format!("qodec-malformed-{name}-"))
            .tempdir()
            .expect("create fixture root");
        for (relative, body) in files {
            let path = root.path().join(relative);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).expect("create fixture subdirectory");
            }
            std::fs::write(path, body).expect("write fixture file");
        }
        Self(root)
    }

    fn path(&self) -> &Path {
        self.0.path()
    }
}

const STIM_ISA: &str = "name: stim\ndescription: physical\nblocks: {qubit: 1}\ninstructions:\n  - mnemonic: R\n    description: reset\n    out: [qubit]\n    action: [stabilize: \"Z_0\"]\n";

fn manifest(layers: &str) -> String {
    format!("name: probe\ndescription: probe\nlayers:\n{layers}")
}

#[test]
fn a_layer_naming_a_missing_instruction_set_is_an_error() {
    let fixture = Fixture::new(
        "missing_isa",
        &[
            (
                "q.qodec.yaml",
                &manifest("  - instruction_set: absent.isa.yaml\n  - instruction_set: stim.isa.yaml\n"),
            ),
            ("stim.isa.yaml", STIM_ISA),
        ],
    );
    let error =
        qodec::Qodec::load(fixture.path().join("q.qodec.yaml")).expect_err("missing instruction set must not load");
    assert!(error.to_string().contains("absent.isa.yaml"), "{error}");
}

#[test]
fn two_layers_naming_the_same_isa_file_still_resolve() {
    let fixture = Fixture::new(
        "shared_isa",
        &[
            (
                "q.qodec.yaml",
                &manifest("  - instruction_set: stim.isa.yaml\n  - instruction_set: stim.isa.yaml\n"),
            ),
            ("stim.isa.yaml", STIM_ISA),
        ],
    );
    let protocol = qodec::Qodec::load(fixture.path().join("q.qodec.yaml"))
        .expect("two layers may reference the same instruction-set file");
    assert_eq!(protocol.layers().len(), 2);
    assert_eq!(protocol.layers()[0].instruction_set.name, "stim");
    assert_eq!(
        protocol.layers()[0].instruction_set,
        protocol.layers()[1].instruction_set
    );
}

#[test]
fn a_truncated_artifact_is_an_error() {
    let fixture = Fixture::new(
        "truncated",
        &[
            (
                "q.qodec.yaml",
                &manifest("  - instruction_set: broken.isa.yaml\n  - instruction_set: stim.isa.yaml\n"),
            ),
            ("stim.isa.yaml", STIM_ISA),
            (
                "broken.isa.yaml",
                "name: broken\ndescription: d\nblocks: {q: 1}\ninstructions:\n  - mnemonic:",
            ),
        ],
    );
    let error = qodec::Qodec::load(fixture.path().join("q.qodec.yaml")).expect_err("truncated artifact must not load");
    assert!(error.to_string().contains("broken.isa.yaml"), "{error}");
}

#[test]
fn an_empty_manifest_is_an_error() {
    let fixture = Fixture::new("empty_manifest", &[("q.qodec.yaml", "")]);
    let _ = qodec::Qodec::load(fixture.path().join("q.qodec.yaml")).expect_err("an empty manifest must not load");
}

#[test]
fn malformed_referenced_artifacts_report_their_normalized_path() {
    for (role, layer) in [
        ("instruction_set", "  - instruction_set: ./unused/../artifact\n"),
        (
            "code",
            "  - instruction_set: stim.isa.yaml\n    codes: {qubit: ./unused/../artifact}\n",
        ),
        (
            "gadget",
            "  - instruction_set: stim.isa.yaml\n    gadgets: {R: ./unused/../artifact}\n",
        ),
    ] {
        let fixture = malformed_artifact_fixture(role, layer);
        let error = qodec::Qodec::load(fixture.path().join("q.qodec.yaml"))
            .expect_err("malformed referenced artifact must not load");
        assert!(
            matches!(&error, qodec::LoadError::Yaml { path, .. } if path == Path::new("artifact")),
            "{role}: {error}"
        );
    }
}

fn malformed_artifact_fixture(role: &str, layer: &str) -> Fixture {
    Fixture::new(
        &format!("malformed_referenced_{role}"),
        &[
            (
                "q.qodec.yaml",
                &manifest(&format!("{layer}  - instruction_set: stim.isa.yaml\n")),
            ),
            ("stim.isa.yaml", STIM_ISA),
            ("artifact", "[invalid YAML\n"),
        ],
    )
}

#[test]
fn a_manifest_that_is_a_yaml_sequence_is_an_error() {
    let fixture = Fixture::new("sequence_manifest", &[("q.qodec.yaml", "- not\n- a\n- mapping\n")]);
    let _ = qodec::Qodec::load(fixture.path().join("q.qodec.yaml")).expect_err("a sequence manifest must not load");
}

fn gadget_with_missing_instruction_fixture() -> Fixture {
    Fixture::new(
        "absent_objective",
        &[
            (
                "q.qodec.yaml",
                &manifest(
                    "  - instruction_set: top.isa.yaml\n    gadgets:\n      nope: g.gadget.yaml\n  - instruction_set: stim.isa.yaml\n",
                ),
            ),
            ("stim.isa.yaml", STIM_ISA),
            (
                "top.isa.yaml",
                "name: top\ndescription: d\nblocks: {b: 1}\ninstructions:\n  - mnemonic: real\n    description: d\n    out: [b]\n    action: []\n",
            ),
            (
                "g.gadget.yaml",
                "implements: top.isa.yaml#nope\ncircuit:\n  source: |-\n    R 0\n",
            ),
        ],
    )
}

#[test]
fn a_gadget_naming_an_absent_instruction_is_an_error() {
    let fixture = gadget_with_missing_instruction_fixture();
    let _ = qodec::Qodec::load(fixture.path().join("q.qodec.yaml"))
        .expect_err("a gadget with no such objective must not load");
}

#[test]
fn a_missing_manifest_is_an_error() {
    let fixture = Fixture::new("no_manifest", &[("stim.isa.yaml", STIM_ISA)]);
    let _ = qodec::Qodec::load(fixture.path().join("q.qodec.yaml")).expect_err("a missing manifest must not load");
}

const TOP_ISA: &str = "name: top\ndescription: logical\nblocks: {b: 1}\ninstructions:\n  - mnemonic: real\n    description: d\n    out: [b]\n    action: []\n";

fn agreeing_gadget() -> serde_json::Value {
    serde_json::json!({
        "implements": "top.isa.yaml#real",
        "circuit": {"instruction_set": "stim.isa.yaml", "format": "yaml", "source": "- R: [0]"},
    })
}

fn qodec_with_gadget(case: &str, gadget: &serde_json::Value) -> Fixture {
    Fixture::new(
        case,
        &[
            (
                "q.qodec.yaml",
                &manifest(
                    "  - instruction_set: top.isa.yaml\n    codes: {b: c.code.yaml}\n    gadgets: {real: g.gadget.yaml}\n  - instruction_set: stim.isa.yaml\n",
                ),
            ),
            ("stim.isa.yaml", STIM_ISA),
            ("top.isa.yaml", TOP_ISA),
            ("c.code.yaml", "name: c\nstabilizers: []\nx: [X_0]\nz: [Z_0]\n"),
            ("g.gadget.yaml", &gadget.to_string()),
        ],
    )
}

#[test]
fn a_gadget_agreeing_with_its_layer_loads() {
    let fixture = qodec_with_gadget("agreeing_gadget", &agreeing_gadget());
    let protocol = qodec::Qodec::load(fixture.path().join("q.qodec.yaml")).expect("the gadget agrees with its layer");
    assert_eq!(protocol.layers()[0].gadgets["real"].implements.mnemonic, "real");
}

#[test]
fn unrepresentable_default_support_returns_a_located_load_error() {
    let fixture = qodec_with_gadget("oversized_support", &agreeing_gadget());
    let code = serde_json::json!({
        "name": "large", "stabilizers": [format!("Z_{}", usize::MAX - 1)], "x": [], "z": [],
    });
    std::fs::write(fixture.path().join("c.code.yaml"), code.to_string()).unwrap();
    let error = qodec::Qodec::load(fixture.path().join("q.qodec.yaml")).unwrap_err();
    let qodec::LoadError::InvalidGadget { gadget, error } = error else {
        panic!("expected a located gadget error, got {error}");
    };
    assert_eq!(gadget, Path::new("g.gadget.yaml"));
    assert!(
        error.starts_with("`out` encoding entry 0: cannot allocate default support"),
        "{error}"
    );
}

#[test]
fn a_gadget_disagreeing_with_its_layer_names_both_sides() {
    for (case, field_path, replacement, expected) in [
        (
            "implements_mnemonic",
            "/implements",
            "top.isa.yaml#other",
            "gadget states `implements` mnemonic 'other', but the layer lists it under 'real'",
        ),
        (
            "implements_isa",
            "/implements",
            "stim.isa.yaml#real",
            "gadget states `implements` instruction set 'stim.isa.yaml' (resolving to stim.isa.yaml), but its layer's instruction set is top.isa.yaml",
        ),
        (
            "circuit_isa",
            "/circuit/instruction_set",
            "top.isa.yaml",
            "gadget states circuit `instruction_set` 'top.isa.yaml' (resolving to top.isa.yaml), but the layer below's instruction set is stim.isa.yaml",
        ),
    ] {
        let mut gadget = agreeing_gadget();
        *gadget.pointer_mut(field_path).unwrap() = replacement.into();
        let fixture = qodec_with_gadget(case, &gadget);
        let error = qodec::Qodec::load(fixture.path().join("q.qodec.yaml"))
            .expect_err("a gadget must not contradict its layer");
        assert!(error.to_string().contains(expected), "{case}: {error}");
    }
}
