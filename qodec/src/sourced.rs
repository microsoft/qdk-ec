//! Inline-or-file content.
//!
//! Several gadget sections may be written inline or factored into a sibling
//! file and referenced by path — small lists usually inline, large generated
//! ones usually external. [`Sourced`] represents both without the rest of the
//! model caring which was authored.

use serde::{Deserialize, Serialize};

/// A value that may be authored inline or referenced from a file.
///
/// Serializes inline values as themselves and file references as their path.
/// Deserializes strings as file paths and anything else as inline values.
///
/// A bare string is a path, and deserializing never reads the file; anything
/// else is an inline value:
///
/// ```text
/// checks.yaml        ->  Sourced::File { path: "checks.yaml" }
/// [a, b]             ->  Sourced::Inline(["a", "b"])
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sourced<T> {
    Inline(T),
    File { path: String },
}

impl<T: Serialize> Serialize for Sourced<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Inline(value) => value.serialize(serializer),
            Self::File { path, .. } => serializer.serialize_str(path),
        }
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Sourced<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        match serde_yaml::Value::deserialize(deserializer)? {
            serde_yaml::Value::String(path) => Ok(Self::file(path)),
            value => T::deserialize(value)
                .map(Self::inline)
                .map_err(serde::de::Error::custom),
        }
    }
}

impl<T> Sourced<T> {
    #[must_use]
    pub fn inline(value: T) -> Self {
        Self::Inline(value)
    }

    #[must_use]
    pub fn file(path: String) -> Self {
        Self::File { path }
    }

    #[must_use]
    pub fn path(&self) -> Option<&str> {
        match self {
            Self::Inline(_) => None,
            Self::File { path, .. } => Some(path),
        }
    }

    /// Whether this is an inline value (no file path) that `is_empty`
    /// considers empty.
    ///
    /// Centralizes the `#[serde(skip_serializing_if)]` predicate shared by
    /// the gadget's optional [`Sourced`] sections (`checks`,
    /// `readouts`, `flags`): a section is omitted on serialization only
    /// when it is inline *and* empty. A file-backed section is always
    /// emitted (as its path), even if its resolved value is empty.
    #[must_use]
    pub fn is_inline_empty(&self, is_empty: impl FnOnce(&T) -> bool) -> bool {
        match self {
            Self::Inline(value) => is_empty(value),
            Self::File { .. } => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Sourced;

    fn yaml(value: &Sourced<Vec<String>>) -> String {
        serde_yaml::to_string(value).expect("serializable")
    }

    fn from_yaml(text: &str) -> Sourced<Vec<String>> {
        serde_yaml::from_str(text).expect("deserializable")
    }

    #[test]
    fn inline_serializes_as_the_bare_value() {
        assert_eq!(yaml(&Sourced::inline(vec!["a".to_owned()])), "- a\n");
    }

    #[test]
    fn file_serializes_as_its_path() {
        assert_eq!(yaml(&Sourced::file("checks.yaml".to_owned())), "checks.yaml\n");
    }

    #[test]
    fn a_string_deserializes_to_an_unresolved_file_reference() {
        let sourced: Sourced<Vec<String>> = from_yaml("checks.yaml\n");
        assert_eq!(sourced.path(), Some("checks.yaml"));
    }

    #[test]
    fn a_non_string_deserializes_to_an_inline_value() {
        let sourced = from_yaml("- a\n- b\n");
        assert_eq!(sourced.path(), None);
        assert_eq!(sourced, Sourced::inline(vec!["a".to_owned(), "b".to_owned()]));
    }

    /// The `skip_serializing_if` predicate: an empty *inline* section is
    /// omitted, but a file-backed one is always emitted as its path.
    #[test]
    fn is_inline_empty_only_holds_for_inline_values() {
        let empty = Sourced::inline(Vec::<String>::new());
        let full = Sourced::inline(vec!["a".to_owned()]);
        let file = Sourced::<Vec<String>>::file("checks.yaml".to_owned());

        assert!(empty.is_inline_empty(Vec::is_empty));
        assert!(!full.is_inline_empty(Vec::is_empty));
        assert!(
            !file.is_inline_empty(Vec::is_empty),
            "a file-backed section is emitted as its path even if empty once read"
        );
    }

    #[test]
    fn inline_and_file_round_trip_through_yaml() {
        for original in [
            Sourced::inline(vec!["a".to_owned()]),
            Sourced::file("checks.yaml".to_owned()),
        ] {
            assert_eq!(from_yaml(&yaml(&original)), original);
        }
    }

    proptest::proptest! {
        #[test]
        fn round_trips_either_way(
            path in "[a-z][a-z0-9_]{0,7}\\.yaml",
            inline in proptest::collection::vec("[a-z]{1,5}", 0..4),
        ) {
            for original in [Sourced::file(path.clone()), Sourced::inline(inline.clone())] {
                proptest::prop_assert_eq!(from_yaml(&yaml(&original)), original);
            }
        }
    }
}
