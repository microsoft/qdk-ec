//! Property-based round-trip tests for the on-disk representations.
//!
//! The fixed-example tests elsewhere check specific artifacts; these check the
//! *property* that serialization and deserialization are inverse across a
//! generated domain.

use proptest::prelude::*;
use qodec::Code;
use qodec::PauliString;
use qodec::{Reference, ReferenceSegment};

fn pauli_string(qubits: usize) -> impl Strategy<Value = String> {
    proptest::collection::vec(prop_oneof![Just('X'), Just('Y'), Just('Z')], 1..=qubits).prop_map(|axes| {
        axes.into_iter()
            .enumerate()
            .map(|(index, axis)| format!("{axis}_{index}"))
            .collect::<Vec<_>>()
            .join(" ")
    })
}

proptest! {
    #[test]
    fn codes_round_trip_through_yaml(
        name in "[a-z][a-z0-9_]{0,7}",
        stabilizers in proptest::collection::vec(pauli_string(6), 1..4),
        x in proptest::collection::vec(pauli_string(6), 1..3),
        z in proptest::collection::vec(pauli_string(6), 1..3),
    ) {
        // `x` and `z` must agree in length: one entry per logical qubit.
        let logical = x.len().min(z.len());
        let code = Code {
            name,
            description: String::new(),
            stabilizers: stabilizers.into_iter().map(PauliString).collect(),
            x: x.into_iter().take(logical).map(PauliString).collect(),
            z: z.into_iter().take(logical).map(PauliString).collect(),
            metadata: qodec::Metadata::new(),
        };

        let yaml = serde_yaml::to_string(&code).expect("serializable");
        let back: Code = serde_yaml::from_str(&yaml).expect("deserializable");
        prop_assert_eq!(&back, &code);
        prop_assert_eq!(back.logical_count(), logical);
    }
}

/// Any reference the grammar can represent.
fn any_reference() -> impl Strategy<Value = Reference> {
    prop_oneof![
        (0usize..64).prop_map(|index| Reference::parse(&format!("circuit.readouts[{index}]")).unwrap()),
        (0usize..64).prop_map(|index| Reference::parse(&format!("readouts[{index}]")).unwrap()),
        (
            prop_oneof![Just("in"), Just("out")],
            0usize..8,
            prop_oneof![Just("stabilizers"), Just("x"), Just("z"),],
            0usize..32,
        )
            .prop_map(|(boundary, entry, property, index)| {
                Reference::parse(&format!("{boundary}[{entry}].{property}[{index}]")).unwrap()
            }),
    ]
}

/// The three selector-bearing heads and their parsed targets.
fn selector_head() -> impl Strategy<Value = (String, Vec<ReferenceSegment>)> {
    prop_oneof![
        Just((
            "circuit.readouts".to_owned(),
            vec![
                ReferenceSegment::Field("circuit".into()),
                ReferenceSegment::Field("readouts".into())
            ]
        )),
        Just(("readouts".to_owned(), vec![ReferenceSegment::Field("readouts".into())])),
        Just((
            "in[0].stabilizers".to_owned(),
            vec![
                ReferenceSegment::Field("in".into()),
                ReferenceSegment::Index(0),
                ReferenceSegment::Field("stabilizers".into())
            ]
        )),
    ]
}

proptest! {
    /// Rendering and parsing are inverse across the whole grammar. The fixed
    /// cases in `parity.rs` pin specific spellings; this pins the relationship.
    #[test]
    fn references_round_trip_through_their_rendering(reference in any_reference()) {
        let rendered = reference.to_string();
        let parsed = Reference::parse(&rendered)
            .unwrap_or_else(|error| panic!("{rendered} failed to parse: {error}"));
        prop_assert_eq!(parsed, reference);
    }

    /// A selector-free atom parses to exactly one reference, and to the same
    /// one either way in.
    #[test]
    fn parse_many_agrees_with_parse_on_single_atoms(reference in any_reference()) {
        let rendered = reference.to_string();
        let many = Reference::parse_many(&rendered).expect("parses");
        prop_assert_eq!(many, vec![reference]);
    }

    /// `head[first:limit:stride]` expands to exactly the indices the slice
    /// denotes, in order.
    #[test]
    fn slice_selectors_expand_to_their_range(
        (head, target) in selector_head(),
        first in 0usize..12,
        span in 1usize..12,
        stride in 1usize..4,
    ) {
        let limit = first + span;
        let atom = format!("{head}[{first:02}:{limit:02}:{stride}]");
        let reference = Reference::parse(&atom).expect("a slice selector parses");
        let expected: Vec<usize> = (first..limit).step_by(stride).collect();
        prop_assert_eq!(&reference.segments()[..target.len()], target.as_slice());
        prop_assert_eq!(reference.expand().map(|value| value.to_string()).collect::<Vec<_>>(), expected.iter().map(|index| format!("{head}[{index}]")).collect::<Vec<_>>());
        prop_assert_eq!(reference.path(), atom.as_str());
        let yaml = serde_yaml::to_string(&reference).unwrap();
        prop_assert_eq!(serde_yaml::from_str::<Reference>(&yaml).unwrap(), reference);
    }

    /// A union selector expands to its members, in the order written.
    #[test]
    fn union_selectors_expand_to_their_members(
        (head, target) in selector_head(),
        members in proptest::collection::vec(0usize..32, 1..6),
    ) {
        let atom = format!("{head}[{}]", members.iter().map(usize::to_string).collect::<Vec<_>>().join(", "));
        let reference = Reference::parse(&atom).expect("a union selector parses");
        prop_assert_eq!(&reference.segments()[..target.len()], target.as_slice());
        let expanded: Vec<_> = reference.expand().collect();
        let expected: Vec<_> = members.iter().map(|index| format!("{head}[{index}]")).collect();
        prop_assert!(expanded.iter().all(|term| &term.segments()[..target.len()] == target.as_slice()));
        prop_assert_eq!(expanded.iter().map(ToString::to_string).collect::<Vec<_>>(), expected);
        prop_assert_eq!(Reference::parse_many(&atom).unwrap(), expanded);
        let yaml = serde_yaml::to_string(&reference).unwrap();
        prop_assert_eq!(serde_yaml::from_str::<Reference>(&yaml).unwrap(), reference);
    }

    #[test]
    fn empty_slices_are_rejected((head, _) in selector_head(), stop in 0usize..12, excess in 0usize..12) {
        let atom = format!("{head}[{}:{stop}]", stop + excess);
        prop_assert!(Reference::parse(&atom).is_err(), "{atom} must be rejected");
    }

    /// A zero stride has no meaning and must be rejected rather than looping.
    #[test]
    fn zero_stride_selectors_are_rejected((head, _) in selector_head(), first in 0usize..8, span in 1usize..8) {
        let atom = format!("{head}[{first}:{}:0]", first + span);
        prop_assert!(Reference::parse_many(&atom).is_err(), "{atom} must be rejected");
    }
}
