//! Quantum error-correcting code definitions.
//!
//! A [`Code`] supplies stabilizer and logical operators for gadget boundary
//! encodings between adjacent instruction set layers.

use crate::PauliString;
use crate::pauli::PauliToken;
use serde::{Deserialize, Serialize};

/// A quantum error-correcting code: stabilizer generators, logical X and Z
/// operator lists (one entry per logical qubit), and optional metadata.
/// Code operators are unsigned: neither an explicit `+` nor `-` is accepted.
///
/// [`Self::logical_count`] counts logical X operators; [`Self::physical_qubit_count`]
/// infers the physical size from Pauli indexes. Deserialization does not check
/// code validity; use [`Self::validate`] or [`Self::load`] to check it.
///
/// # Examples
///
/// ```
/// use qodec::Code;
///
/// // The three-qubit bit-flip repetition code.
/// let code: Code = serde_yaml::from_str(
///     "name: repetition3\nstabilizers: [Z_0 Z_1, Z_1 Z_2]\nx: [X_0 X_1 X_2]\nz: [Z_0]\n",
/// )?;
///
/// assert_eq!(code.logical_count(), 1);
/// assert_eq!(code.physical_qubit_count(), 3);
/// # Ok::<(), serde_yaml::Error>(())
/// ```
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Code {
    /// Unique name within a qodec; how layers and gadgets refer to this code.
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    /// Free-form prose for readers; empty when the author supplied none.
    pub description: String,
    /// Stabilizer generators, indexed from zero in gadget parity references
    /// such as `in[entry].stabilizers[i]` and `out[entry].stabilizers[i]`.
    pub stabilizers: Vec<PauliString>,
    /// Logical X operators, indexed by logical qubit. For example,
    /// `in[entry].x[i]` refers to logical qubit `i` of input encoding `entry`;
    /// both indexes are zero-based.
    pub x: Vec<PauliString>,
    /// Logical Z operators, in the same qubit order and with the same length
    /// as [`Self::x`]. Referenced as `in[entry].z[i]` or `out[entry].z[i]`.
    pub z: Vec<PauliString>,
    /// Annotations for external tools; see [`crate::Metadata`].
    #[serde(default, skip_serializing_if = "crate::Metadata::is_empty")]
    pub metadata: crate::Metadata,
}

impl Code {
    /// Read and validate a standalone code definition from a YAML file.
    ///
    /// # Errors
    ///
    /// Returns [`crate::LoadError::Io`] if reading fails,
    /// [`crate::LoadError::Yaml`] if parsing fails, or
    /// [`crate::LoadError::InvalidCode`] if [`Self::validate`] fails.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, crate::LoadError> {
        let path = path.as_ref();
        let code: Self = crate::qodec::loader::read_yaml(path)?;
        code.validate().map_err(|error| crate::LoadError::InvalidCode {
            code: path.to_path_buf(),
            error,
        })?;
        Ok(code)
    }

    /// Validate this code and write it as YAML, creating parent directories as needed.
    ///
    /// # Errors
    ///
    /// Returns [`std::io::Error`] if validation, serialization, directory
    /// creation, or writing fails.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        crate::qodec::loader::save_artifact(self, path.as_ref(), Self::validate)
    }

    /// Number of declared logical X operators.
    ///
    /// Equals `x.len()`. A valid code has equally many logical Z operators;
    /// this accessor does not require the draft's lists to match.
    #[must_use]
    pub fn logical_count(&self) -> usize {
        self.x.len()
    }

    /// One more than the highest qubit index in the stabilizers and logical operators.
    ///
    /// Uses the Pauli token grammar, including identity tokens such as `I_7`.
    /// A missing `_index` suffix means index 0. Malformed tokens are ignored;
    /// returns zero when no valid token is present. This does not check code
    /// validity; use [`Self::validate`] to reject malformed operators.
    /// An index must be less than `usize::MAX` so its dimension is representable.
    #[must_use]
    pub fn physical_qubit_count(&self) -> usize {
        self.stabilizers
            .iter()
            .chain(self.x.iter())
            .chain(self.z.iter())
            .flat_map(|pauli| pauli.0.split_whitespace())
            .filter_map(|token| PauliToken::parse(token).ok())
            .filter_map(|token| token.index.checked_add(1))
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paulis(strings: &[&str]) -> Vec<PauliString> {
        strings.iter().map(|text| PauliString((*text).to_owned())).collect()
    }

    #[test]
    fn physical_qubit_count_repetition3() {
        let code = Code {
            name: "repetition3".to_owned(),
            description: String::new(),
            stabilizers: paulis(&["Z_0 Z_1", "Z_1 Z_2"]),
            x: paulis(&["X_0 X_1 X_2"]),
            z: paulis(&["Z_0"]),
            metadata: crate::Metadata::default(),
        };
        assert_eq!(code.physical_qubit_count(), 3);
    }

    #[test]
    fn physical_qubit_count_c832_from_high_stabilizer() {
        let code = Code {
            name: "c832".to_owned(),
            description: String::new(),
            stabilizers: paulis(&["X_0 X_1 X_2 X_3 X_4 X_5 X_6 X_7"]),
            x: paulis(&["X_0 X_1 X_2 X_3"]),
            z: paulis(&["Z_0 Z_4"]),
            metadata: crate::Metadata::default(),
        };
        assert_eq!(code.physical_qubit_count(), 8);
    }

    #[test]
    fn physical_qubit_count_handles_the_integer_boundary() {
        let mut code = Code {
            name: "boundary".to_owned(),
            description: String::new(),
            stabilizers: paulis(&[&format!("Z_{}", usize::MAX - 1)]),
            x: Vec::new(),
            z: Vec::new(),
            metadata: crate::Metadata::default(),
        };
        assert!(code.validate().is_ok());
        assert_eq!(code.physical_qubit_count(), usize::MAX);
        code.stabilizers = paulis(&[&format!("Z_{}", usize::MAX)]);
        assert!(
            code.validate()
                .unwrap_err()
                .contains("exceeds the supported code dimension")
        );
        assert_eq!(code.physical_qubit_count(), 0);
    }

    #[test]
    fn physical_qubit_count_empty_is_zero() {
        let code = Code {
            name: "empty".to_owned(),
            description: String::new(),
            stabilizers: Vec::new(),
            x: Vec::new(),
            z: Vec::new(),
            metadata: crate::Metadata::default(),
        };
        assert_eq!(code.physical_qubit_count(), 0);
    }

    #[test]
    fn physical_qubit_count_uses_pauli_token_syntax() {
        for (text, expected) in [
            ("I_7", 8),
            ("\tX_1\n Y_3  I_9\r\n", 10),
            ("X_2 X_2", 3),
            ("Q_7", 0),
            ("invalid", 0),
            ("target.X", 0),
            ("outer.inner.X_9", 0),
            ("X_bad", 0),
            ("X_2 Q_9 Z_bad", 3),
        ] {
            let code = Code {
                name: "tokens".to_owned(),
                description: String::new(),
                stabilizers: paulis(&[text]),
                x: Vec::new(),
                z: Vec::new(),
                metadata: crate::Metadata::default(),
            };
            assert_eq!(code.physical_qubit_count(), expected, "{text}");
        }
    }
}

#[cfg(test)]
mod standalone_io_tests {
    use super::Code;

    fn example_code() -> Code {
        serde_yaml::from_str(include_str!("../examples/c4c6/c4.code.yaml")).expect("the example parses")
    }

    #[test]
    fn a_code_round_trips_through_its_own_file() {
        let dir = std::env::temp_dir().join(format!("qodec-code-io-{}", std::process::id()));
        let path = dir.join("nested/out.code.yaml");
        let original = example_code();
        original.save(&path).expect("saves, creating the parent");
        assert_eq!(Code::load(&path).expect("loads"), original);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn saving_a_structurally_invalid_code_does_not_write_it() {
        let mut code = example_code();
        code.x = vec![crate::PauliString("Q_0".to_owned())];
        let path = std::env::temp_dir().join(format!("qodec-bad-{}.code.yaml", std::process::id()));
        let error = code.save(&path).expect_err("malformed Pauli token");
        assert!(error.to_string().contains("unknown basis 'Q'"), "got: {error}");
        assert!(!path.exists(), "nothing should be written");
    }
}
