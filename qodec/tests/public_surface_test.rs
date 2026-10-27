//! Pins the crate's public surface.
//!
//! The list below is the whole of what `qodec::*` offers. It is asserted by
//! set equality in both directions, so adding a name fails this test just as
//! removing one does. That is the point: growing the surface should be a
//! decision someone made, not something that happened.
//!
//! When a change here is intended, update the list and say why in the commit.

/// Every name re-exported from the crate root.
const SURFACE: &[&str] = &[
    // artifacts and their parts
    "Block",
    "BlockOperand",
    "Code",
    "Instruction",
    "InstructionSet",
    "Parameter",
    "ParameterKind",
    "PauliString",
    "Qodec",
    // instruction semantics
    "Action",
    "ActionStep",
    "Condition",
    "Observable",
    "Scalar",
    // resolved model
    "Circuit",
    "Encoding",
    "Gadget",
    "Layer",
    "Node",
    "SourceLocation",
    // decoding surface
    "ParityEquation",
    "ParityTerm",
    "Readout",
    "ReadoutSpec",
    "Reference",
    "ReferenceSegment",
    // lowering IR
    "Argument",
    "CircuitReadout",
    "InstructionCall",
    "Operand",
    "SelectPattern",
    // errors
    "LoadError",
    "ParseError",
    "PathError",
    "ReferenceParseError",
    "ResolveError",
    "SliceError",
    // odds and ends
    "CURRENT_SCHEMA_VERSION",
    "MAX_SELECTED_POSITIONS",
    "Metadata",
    "register",
];

/// Names the crate root exports, read from `src/lib.rs`.
///
/// Covers re-exports and items declared at the root. Scanning only `pub use`
/// missed `pub type Metadata`, and would have missed any new `pub struct`,
/// `pub fn` or `pub mod` added beside it.
fn exported_names() -> std::collections::BTreeSet<String> {
    let source = include_str!("../src/lib.rs");
    let mut names = std::collections::BTreeSet::new();
    for line in source.lines() {
        let Some(item) = line.strip_prefix("pub ") else {
            continue;
        };
        if let Some(list) = item.strip_prefix("use ") {
            names.extend(reexported_names(list, source));
        } else if let Some(name) = declared_name(item) {
            names.insert(name);
        }
    }
    names
}

/// The name a root-level `pub <keyword> <name>` item introduces.
fn declared_name(item: &str) -> Option<String> {
    let (keyword, rest) = item.split_once(' ')?;
    if !matches!(
        keyword,
        "type" | "struct" | "enum" | "trait" | "union" | "const" | "static" | "fn" | "mod"
    ) {
        return None;
    }
    let name: String = rest
        .trim_start_matches("mut ")
        .chars()
        .take_while(|character| character.is_alphanumeric() || *character == '_')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// The names one `pub use` brings in, following a multi-line brace list.
fn reexported_names(list: &str, source: &str) -> Vec<String> {
    let list = if list.contains(';') {
        list.split(';').next().unwrap_or(list).to_owned()
    } else {
        // A wrapped list: take everything up to the terminating semicolon.
        let start = source.find(list).expect("the line came from this source");
        let tail = &source[start..];
        tail[..tail.find(';').expect("a `pub use` ends in a semicolon")].to_owned()
    };
    let list = list.rsplit("::").next().unwrap_or(&list).to_owned();
    list.trim_matches(['{', '}'].as_slice())
        .split(',')
        .map(|name| name.trim().trim_matches(['{', '}'].as_slice()).trim().to_owned())
        .filter(|name| !name.is_empty())
        .collect()
}

#[test]
fn text_arguments_distinguish_record_indices_from_literal_text() {
    use qodec::Argument;
    for text in [
        "",
        "label",
        "007",
        "true",
        "readouts",
        "readouts_label[0]",
        "circuit.readouts",
        "in[0].z[0]",
    ] {
        assert_eq!(Argument::parse_text(text).unwrap(), Argument::Text(text.to_owned()));
    }
    for (text, index) in [
        ("circuit.readouts[0]", 0),
        ("circuit.readouts[003]", 3),
        ("circuit.readouts[+3]", 3),
        ("circuit.readouts[0:1]", 0),
        ("circuit.readouts[03:04]", 3),
        ("circuit.readouts[3:5:2]", 3),
    ] {
        assert_eq!(Argument::parse_text(text).unwrap(), Argument::Readout(index));
    }
    assert_eq!(
        Argument::parse_text(&format!("circuit.readouts[{}]", usize::MAX)).unwrap(),
        Argument::Readout(usize::MAX)
    );
}

#[test]
fn text_arguments_require_one_valid_record_position() {
    for text in [
        "circuit.readouts[0:2]",
        "circuit.readouts[0:1048576]",
        "circuit.readouts[0:0]",
        "circuit.readouts[0:2:0]",
        "circuit.readouts[0,1]",
        "circuit.readouts[0,0]",
        "circuit.readouts[0:1][0]",
        "circuit.readouts[0].name",
        "circuit.readouts[]",
        "circuit.readouts[-1]",
        "circuit.readouts[0",
        "circuit.readouts[0]suffix",
        "circuit.readouts[ 0]",
        "readouts[0]",
    ] {
        let error = qodec::Argument::parse_text(text).expect_err("not one circuit record index");
        assert!(error.contains(text), "{error}");
        assert!(error.contains("readout"), "{error}");
    }
    let overflowing = format!("circuit.readouts[{}0]", usize::MAX);
    assert!(
        qodec::Argument::parse_text(&overflowing)
            .unwrap_err()
            .contains("non-negative integer")
    );
}

#[test]
fn public_surface_is_exactly_the_pinned_list() {
    let pinned: std::collections::BTreeSet<String> = SURFACE.iter().map(|s| (*s).to_owned()).collect();
    let actual = exported_names();

    let added: Vec<_> = actual.difference(&pinned).collect();
    let removed: Vec<_> = pinned.difference(&actual).collect();

    assert!(
        added.is_empty() && removed.is_empty(),
        "public surface drifted.\n  added (not in the pinned list): {added:?}\n  removed (pinned but gone): {removed:?}\n\
         Update SURFACE in this file if the change is intended."
    );
}

#[test]
fn parity_inherent_methods_are_exactly_the_pinned_lists() {
    let expected: &[(&str, &[&str])] = &[
        ("ReadoutSpec", &["new", "named"]),
        ("Readout", &["resolve_list", "to_spec"]),
        ("Reference", &["parse", "path", "segments", "expand"]),
    ];
    let source = include_str!("../src/parity.rs");
    for (owner, methods) in expected {
        let declaration = format!("impl {owner} {{");
        let actual: std::collections::BTreeSet<_> = source
            .split(&declaration)
            .skip(1)
            .flat_map(|block| block.lines().take_while(|line| *line != "}"))
            .filter_map(|line| line.trim_start().strip_prefix("pub ").and_then(declared_name))
            .collect();
        let pinned: std::collections::BTreeSet<_> = methods.iter().map(|method| (*method).to_owned()).collect();
        assert_eq!(actual, pinned, "{owner} public methods drifted");
    }
}

#[test]
fn pinned_list_has_no_duplicates() {
    let unique: std::collections::BTreeSet<_> = SURFACE.iter().collect();
    assert_eq!(unique.len(), SURFACE.len(), "the pinned list repeats a name");
}

#[test]
fn surface_stays_small() {
    // A ceiling, not a target. Raising it should be deliberate.
    // Raised to 43 for MAX_SELECTED_POSITIONS, which callers need to bound a
    // selector before parsing one.
    assert!(
        SURFACE.len() <= 43,
        "public surface is {} names, over the 43 the crate has agreed to",
        SURFACE.len()
    );
}
