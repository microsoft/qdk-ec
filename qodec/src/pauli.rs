//! Text representation of Pauli operators.

use serde::{Deserialize, Serialize};

/// Text describing a Pauli operator, e.g. `"X_0 Z_1 Y_2"`.
///
/// Construction, conversion, and deserialization preserve the text without
/// validating it. Equality and hashing compare the text, not the operators.
/// Serialized as a string.
#[derive(Serialize, Deserialize)]
#[serde(transparent)]
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, derive_more::Display, derive_more::From, derive_more::AsRef,
)]
#[from(String, &str)]
#[as_ref(str)]
pub struct PauliString(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Basis {
    I,
    X,
    Y,
    Z,
}

/// One `basis[_index]` token, including explicitly addressed identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PauliToken {
    pub index: usize,
    pub basis: Basis,
}

impl PauliToken {
    pub(crate) fn parse(token: &str) -> Result<Self, String> {
        if token.starts_with(['+', '-']) {
            return Err("code Paulis must not have a sign".to_owned());
        }
        if token.contains('.') {
            return Err(format!(
                "invalid Pauli token '{token}': operand prefixes are not part of the grammar; \
                 address code qubits directly, as in `X_0`"
            ));
        }
        let (basis_str, index_str) = match token.split_once('_') {
            Some((basis, index)) => (basis, Some(index)),
            None => (token, None),
        };
        let basis = match basis_str {
            "I" => Basis::I,
            "X" => Basis::X,
            "Y" => Basis::Y,
            "Z" => Basis::Z,
            other => return Err(format!("invalid Pauli token '{token}': unknown basis '{other}'")),
        };
        let index = match index_str {
            Some(string) => string
                .parse::<usize>()
                .map_err(|_| format!("invalid qubit index in Pauli token '{token}'"))?,
            None => 0,
        };
        if index == usize::MAX {
            return Err(format!(
                "qubit index in Pauli token '{token}' exceeds the supported code dimension"
            ));
        }
        Ok(Self { index, basis })
    }
}

#[cfg(test)]
mod token_tests {
    use super::{Basis, PauliString, PauliToken};
    use proptest::prelude::*;

    #[test]
    fn tokens_carry_declared_identity_indices() {
        for (text, index, basis) in [
            ("I_7", 7, Basis::I),
            ("X", 0, Basis::X),
            ("Y_02", 2, Basis::Y),
            ("Z", 0, Basis::Z),
        ] {
            assert_eq!(PauliToken::parse(text).unwrap(), PauliToken { index, basis });
        }
    }

    #[test]
    fn the_retired_operand_prefix_form_is_rejected() {
        for text in ["target.X", "target_1.Y_02", "source.Z_0"] {
            let error = PauliToken::parse(text).expect_err("operand prefixes were removed from the grammar");
            assert!(
                error.contains("operand prefixes are not part of the grammar"),
                "{error}"
            );
        }
    }

    #[test]
    fn code_validation_preserves_token_errors() {
        for (text, expected) in [
            ("Q_7", "invalid Pauli token 'Q_7': unknown basis 'Q'"),
            ("X_bad", "invalid qubit index in Pauli token 'X_bad'"),
            ("X_-1", "invalid qubit index in Pauli token 'X_-1'"),
            ("Z_", "invalid qubit index in Pauli token 'Z_'"),
            (
                "outer.inner.X_9",
                "invalid Pauli token 'outer.inner.X_9': operand prefixes are not part of the grammar; \
                 address code qubits directly, as in `X_0`",
            ),
        ] {
            assert_eq!(PauliToken::parse(text).unwrap_err(), expected);
            let code = crate::Code {
                name: "tokens".to_owned(),
                description: String::new(),
                stabilizers: vec![PauliString::from(format!("X_0 {text}"))],
                x: Vec::new(),
                z: Vec::new(),
                metadata: crate::Metadata::default(),
            };
            assert_eq!(code.validate().unwrap_err(), format!("stabilizer 0: {expected}"));
        }
        let overflowing = format!("X_{}0", usize::MAX);
        assert_eq!(
            PauliToken::parse(&overflowing).unwrap_err(),
            format!("invalid qubit index in Pauli token '{overflowing}'")
        );
    }

    proptest! {
        #[test]
        fn counting_and_validation_share_token_interpretation(
            first in 0usize..128,
            second in 0usize..128,
            identity in 0usize..128,
        ) {
            let expression = format!("X_{first:02}\tZ_{second}\nI_{identity}");
            let code = crate::Code {
                name: "tokens".to_owned(),
                description: String::new(),
                stabilizers: vec![PauliString::from(expression)],
                x: Vec::new(),
                z: Vec::new(),
                metadata: crate::Metadata::default(),
            };
            prop_assert_eq!(code.physical_qubit_count(), first.max(second).max(identity) + 1);
            prop_assert!(code.validate().is_ok());
        }
    }
}

#[cfg(test)]
mod newtype_tests {
    use super::PauliString;
    use crate::BlockName;

    /// `BlockName` and `PauliString` are both string newtypes and must offer the
    /// same surface; before `derive_more` only `BlockName` had one.
    #[test]
    fn both_string_newtypes_convert_and_display_alike() {
        assert_eq!(BlockName::from("q").to_string(), "q");
        assert_eq!(BlockName::from("q".to_owned()), BlockName::new("q"));
        assert_eq!(AsRef::<str>::as_ref(&BlockName::from("q")), "q");

        assert_eq!(PauliString::from("X_0 Z_1").to_string(), "X_0 Z_1");
        assert_eq!(PauliString::from("X_0".to_owned()), PauliString::from("X_0"));
        assert_eq!(AsRef::<str>::as_ref(&PauliString::from("X_0")), "X_0");
    }
}
