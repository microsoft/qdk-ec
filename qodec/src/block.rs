//! Block names — the elements of an encoding's support.
//!
//! A [`BlockName`] names one element a gadget's encoding lands on. At the physical
//! floor that is a bare qubit index; in a concatenated code it may itself be an
//! encoded block at an inner level. The type exists so both spellings share one
//! representation and one serde form.

use serde::{Deserialize, Serialize};

/// A block name identifying an element in an encoding's support.
///
/// In concatenated codes, each block may itself be an encoded block at an
/// inner level. Physical qubits are the trivial base case.
///
/// Accepts both integers and strings from YAML; integers are stored as their
/// string representation.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, derive_more::Display, derive_more::From, derive_more::AsRef,
)]
#[from(String, &str)]
#[as_ref(str)]
pub struct BlockName(pub String);

impl BlockName {
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

impl Serialize for BlockName {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Only the canonical spelling of an integer may be emitted as one:
        // `"007"` would otherwise come back as `"7"`.
        match self.0.parse::<u64>() {
            Ok(index) if index.to_string() == self.0 => index.serialize(serializer),
            _ => self.0.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for BlockName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Index(u64),
            Name(String),
        }

        match Repr::deserialize(deserializer)? {
            Repr::Index(index) => Ok(Self(index.to_string())),
            Repr::Name(name) => Ok(Self(name)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BlockName;

    fn yaml(block: &BlockName) -> String {
        serde_yaml::to_string(block).expect("serializable").trim().to_owned()
    }

    fn parse(text: &str) -> BlockName {
        serde_yaml::from_str(text).expect("deserializable")
    }

    #[test]
    fn an_integer_deserializes_to_its_string_form() {
        assert_eq!(parse("3\n"), BlockName::new("3"));
        assert_eq!(parse("q0\n"), BlockName::new("q0"));
    }

    #[test]
    fn numeric_names_serialize_unquoted_and_others_as_strings() {
        assert_eq!(yaml(&BlockName::new("3")), "3");
        assert_eq!(yaml(&BlockName::new("c4")), "c4");
        assert_eq!(
            yaml(&BlockName::new("-1")),
            "'-1'",
            "negatives are not u64, so they stay strings"
        );
    }

    #[test]
    fn display_is_the_bare_name() {
        assert_eq!(BlockName::new("c4").to_string(), "c4");
    }

    #[test]
    fn common_names_round_trip() {
        for name in ["0", "12", "q0", "c4", "-1", "", "007", "0x10", "1_000", "+7"] {
            let block = BlockName::new(name);
            assert_eq!(parse(&format!("{}\n", yaml(&block))), block, "round trip of {name:?}");
        }
    }

    /// A zero-padded name is not the canonical spelling of its integer, so it
    /// stays a quoted string rather than being flattened to one.
    #[test]
    fn leading_zero_names_round_trip() {
        let padded = BlockName::new("007");
        assert_eq!(yaml(&padded), "'007'");
        assert_eq!(parse(&format!("{}\n", yaml(&padded))), padded);
        // The canonical form is still emitted bare.
        assert_eq!(yaml(&BlockName::new("7")), "7");
        assert_eq!(parse("7\n"), BlockName::new("7"));
    }

    proptest::proptest! {
        /// Non-numeric names, canonical numerals, and zero-padded ones, which
        /// are not the canonical spelling of their integer and so must survive
        /// as strings.
        #[test]
        fn names_round_trip(name in {
            use proptest::strategy::Strategy;
            proptest::prop_oneof![
                "[a-z][a-z0-9_]{0,7}",
                (0u32..10_000).prop_map(|index| index.to_string()),
                (1usize..4, 0u32..1000).prop_map(|(pad, index)| format!("{:0width$}", index, width = pad + 4)),
            ]
        }) {
            let block = BlockName::new(name);
            proptest::prop_assert_eq!(parse(&format!("{}\n", yaml(&block))), block);
        }
    }
}
