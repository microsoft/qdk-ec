//! Errors from loading qodecs and standalone artifacts.

use std::path::PathBuf;

/// A YAML parse failure with a message and optional source location.
#[derive(Debug, derive_more::Display, derive_more::Error)]
pub struct ParseError(serde_yaml::Error);

impl ParseError {
    /// The reported `(line, column)`, both one-based, or `None` if unavailable.
    #[must_use]
    pub fn location(&self) -> Option<(usize, usize)> {
        self.0.location().map(|at| (at.line(), at.column()))
    }
}

impl ParseError {
    pub(crate) fn new(error: serde_yaml::Error) -> Self {
        Self(error)
    }
}

/// Errors from loading a qodec, code definition, or instruction set.
///
/// [`Self::Io`] and [`Self::Yaml`] expose their underlying error through
/// [`std::error::Error::source`]. Other variants have no error source.
#[derive(Debug, derive_more::Display, derive_more::Error, derive_more::From)]
pub enum LoadError {
    /// Reading a file or directory failed.
    #[from]
    #[display("I/O error: {_0}")]
    Io(std::io::Error),
    /// An artifact could not be parsed; `path` identifies the artifact.
    #[display("YAML error in {}: {source}", path.display())]
    Yaml {
        /// The artifact that failed to parse.
        path: PathBuf,
        /// The underlying parse failure, with its source location.
        source: ParseError,
    },
    /// The manifest's declared schema version differs from the supported version.
    #[display(
        "manifest {} declares schema_version {declared}; this loader supports schema_version {supported}",
        manifest.display()
    )]
    UnsupportedSchemaVersion {
        /// The manifest that declared the version.
        manifest: PathBuf,
        /// The version the manifest declares.
        declared: u32,
        /// The only version this loader accepts.
        supported: u32,
    },
    /// `referenced_from` names an artifact at `path` that does not exist.
    #[display("{} references missing artifact {}", referenced_from.display(), path.display())]
    MissingArtifact {
        /// The document holding the reference.
        referenced_from: PathBuf,
        /// The referenced path that does not exist.
        path: PathBuf,
    },
    /// A gadget's implemented instruction is not declared in its source instruction set.
    #[display(
        "gadget {} implements '{implements}', which is not declared in source instruction set '{instruction_set}'",
        gadget.display()
    )]
    GadgetUnknownImplements {
        /// The gadget document.
        gadget: PathBuf,
        /// The mnemonic it claims to implement.
        implements: String,
        /// The source instruction set that does not declare it.
        instruction_set: String,
    },
    /// An instruction set failed validation; `error` describes the failure.
    #[display("invalid instruction set {}: {error}", instruction_set.display())]
    InvalidInstructionSet {
        /// The instruction-set document.
        instruction_set: PathBuf,
        /// What failed validation.
        error: String,
    },
    /// A gadget failed validation; `error` describes the failure.
    #[display("invalid gadget {}: {error}", gadget.display())]
    InvalidGadget {
        /// The gadget document.
        gadget: PathBuf,
        /// What failed validation.
        error: String,
    },
    /// A code definition failed validation; `error` describes the failure.
    #[display("invalid code {}: {error}", code.display())]
    InvalidCode {
        /// The code document.
        code: PathBuf,
        /// What failed validation.
        error: String,
    },
    /// The resolved layer model is inconsistent; `error` identifies the component.
    #[display("invalid qodec {}: {error}", manifest.display())]
    InvalidQodec {
        /// The manifest of the qodec that failed validation.
        manifest: PathBuf,
        /// Which component is inconsistent.
        error: String,
    },
    /// Two instruction set artifacts declare the same name.
    #[display(
        "instruction set name '{name}' is declared by two files {} and {}; names must be unique within a qodec",
        first.display(),
        second.display()
    )]
    DuplicateInstructionSetName {
        /// The name declared twice.
        name: String,
        /// The first document declaring it.
        first: PathBuf,
        /// The second document declaring it.
        second: PathBuf,
    },
    /// Two code artifacts declare the same name.
    #[display(
        "code name '{name}' is declared by two files {} and {}; code names must be unique within a qodec",
        first.display(),
        second.display()
    )]
    DuplicateCodeName {
        /// The name declared twice.
        name: String,
        /// The first document declaring it.
        first: PathBuf,
        /// The second document declaring it.
        second: PathBuf,
    },
    /// A single-file bundle has an invalid document structure.
    #[display("single-file qodec bundle {} is malformed: {reason}", manifest.display())]
    MalformedBundle {
        /// The bundle that could not be split into documents.
        manifest: PathBuf,
        /// What is wrong with its structure.
        reason: String,
    },
    /// A manifest path names a directory rather than a manifest file.
    #[display("expected a manifest file path, got directory {}", path.display())]
    NotAManifestFile {
        /// The directory that was passed where a manifest file was expected.
        path: PathBuf,
    },
    /// One path is referenced as two different artifact kinds.
    #[display(
        "{} references {} as {kind}, but that path already identifies {previous}",
        referenced_from.display(),
        path.display()
    )]
    ConflictingArtifactKind {
        /// The document holding the second reference.
        referenced_from: PathBuf,
        /// The path referenced as two kinds.
        path: PathBuf,
        /// The kind this reference asks for.
        kind: &'static str,
        /// The kind an earlier reference already established.
        previous: &'static str,
    },
}

impl LoadError {
    pub(super) fn from_validation(
        issue: crate::validation::ModelIssue,
        qodec: &super::Qodec,
        manifest: &crate::Manifest,
    ) -> Self {
        use crate::validation::ModelIssue;

        let message = issue.to_string();
        let located =
            match issue {
                ModelIssue::InstructionSet { layer, error, .. } => {
                    manifest.layers.get(layer).map(|layer| Self::InvalidInstructionSet {
                        instruction_set: PathBuf::from(&layer.instruction_set),
                        error,
                    })
                }
                ModelIssue::Code { name, error } => qodec
                    .code_artifacts
                    .iter()
                    .find(|(_, code)| code.name == name)
                    .map(|(path, _)| Self::InvalidCode {
                        code: path.clone(),
                        error,
                    }),
                ModelIssue::Gadget { layer, mnemonic, error } => manifest
                    .layers
                    .get(layer)
                    .and_then(|layer| layer.gadgets.get(&mnemonic))
                    .map(|path| Self::InvalidGadget {
                        gadget: PathBuf::from(path),
                        error,
                    }),
                ModelIssue::Model(_) => None,
            };
        located.unwrap_or_else(|| Self::InvalidQodec {
            manifest: PathBuf::from(qodec.manifest_filename()),
            error: message,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{LoadError, ParseError};
    use std::path::PathBuf;

    fn parse_error() -> ParseError {
        ParseError::new(serde_yaml::from_str::<serde_yaml::Value>("{[}").expect_err("malformed YAML"))
    }

    /// Every variant, paired with the operands its message must name, **in the
    /// order they are rendered**. A discriminant check (`matches!`) cannot catch
    /// a message that drops a field or renders two of them the wrong way round;
    /// an ordered check catches both, where a plain `contains` misses the swap.
    #[allow(clippy::too_many_lines)] // One table entry per variant; splitting it hides the exhaustiveness.
    fn every_variant() -> Vec<(LoadError, Vec<String>)> {
        let owned = |items: &[&str]| items.iter().map(|item| (*item).to_owned()).collect();
        vec![
            (
                LoadError::Io(std::io::Error::other("device not ready")),
                owned(&["device not ready"]),
            ),
            (
                LoadError::Yaml {
                    path: PathBuf::from("broken.isa.yaml"),
                    source: parse_error(),
                },
                owned(&["broken.isa.yaml"]),
            ),
            (
                LoadError::UnsupportedSchemaVersion {
                    manifest: PathBuf::from("old.qodec.yaml"),
                    declared: 2,
                    supported: 4,
                },
                owned(&["old.qodec.yaml", "2", "4"]),
            ),
            (
                LoadError::MissingArtifact {
                    referenced_from: PathBuf::from("stack.qodec.yaml"),
                    path: PathBuf::from("absent.isa.yaml"),
                },
                owned(&["stack.qodec.yaml", "absent.isa.yaml"]),
            ),
            (
                LoadError::GadgetUnknownImplements {
                    gadget: PathBuf::from("prep.gadget.yaml"),
                    implements: "prepare_z".to_owned(),
                    instruction_set: "repetition3".to_owned(),
                },
                owned(&["prep.gadget.yaml", "prepare_z", "repetition3"]),
            ),
            (
                LoadError::InvalidInstructionSet {
                    instruction_set: PathBuf::from("bad.isa.yaml"),
                    error: "duplicate mnemonic".to_owned(),
                },
                owned(&["bad.isa.yaml", "duplicate mnemonic"]),
            ),
            (
                LoadError::InvalidGadget {
                    gadget: PathBuf::from("bad.gadget.yaml"),
                    error: "unknown operand".to_owned(),
                },
                owned(&["bad.gadget.yaml", "unknown operand"]),
            ),
            (
                LoadError::InvalidCode {
                    code: PathBuf::from("bad.code.yaml"),
                    error: "malformed Pauli token".to_owned(),
                },
                owned(&["bad.code.yaml", "malformed Pauli token"]),
            ),
            (
                LoadError::InvalidQodec {
                    manifest: PathBuf::from("bad.qodec.yaml"),
                    error: "inconsistent layer".to_owned(),
                },
                owned(&["bad.qodec.yaml", "inconsistent layer"]),
            ),
            (
                LoadError::DuplicateInstructionSetName {
                    name: "stim".to_owned(),
                    first: PathBuf::from("one.isa.yaml"),
                    second: PathBuf::from("two.isa.yaml"),
                },
                owned(&["stim", "one.isa.yaml", "two.isa.yaml"]),
            ),
            (
                LoadError::DuplicateCodeName {
                    name: "c422".to_owned(),
                    first: PathBuf::from("one.code.yaml"),
                    second: PathBuf::from("two.code.yaml"),
                },
                owned(&["c422", "one.code.yaml", "two.code.yaml"]),
            ),
            (
                LoadError::MalformedBundle {
                    manifest: PathBuf::from("bundle.qodec.yaml"),
                    reason: "second document is not a mapping".to_owned(),
                },
                owned(&["bundle.qodec.yaml", "second document is not a mapping"]),
            ),
            (
                LoadError::NotAManifestFile {
                    path: PathBuf::from("examples/steane"),
                },
                owned(&["examples/steane"]),
            ),
            (
                LoadError::ConflictingArtifactKind {
                    referenced_from: PathBuf::from("stack.qodec.yaml"),
                    path: PathBuf::from("shared.yaml"),
                    kind: "a code",
                    previous: "an instruction set",
                },
                owned(&["stack.qodec.yaml", "shared.yaml", "a code", "an instruction set"]),
            ),
        ]
    }

    #[test]
    fn every_message_names_its_own_operands_in_order() {
        for (error, expected) in every_variant() {
            let rendered = error.to_string();
            let mut searched_from = 0;
            for operand in &expected {
                let found = rendered[searched_from..].find(operand.as_str()).unwrap_or_else(|| {
                    panic!("{error:?} rendered as {rendered:?}, which omits {operand:?} (or renders it out of order)")
                });
                searched_from += found + operand.len();
            }
        }
    }

    #[test]
    fn every_message_is_a_lowercase_phrase_using_model_terms() {
        // Messages are composed into larger sentences by callers, so they start
        // lowercase and carry no trailing period. `body`/`slot`/`objective` name
        // fields outside the on-disk format.
        for (error, _) in every_variant() {
            let rendered = error.to_string();
            assert!(!rendered.ends_with('.'), "{rendered:?} ends with a period");
            let head = rendered.split_whitespace().next().unwrap_or_default();
            let sentence_cased = head.starts_with(char::is_uppercase) && head.contains(char::is_lowercase);
            assert!(!sentence_cased, "{rendered:?} starts with a capitalized word");
            for unsupported in ["body", "slot", "objective", "assume"] {
                assert!(
                    !rendered.contains(unsupported),
                    "{rendered:?} uses unsupported vocabulary {unsupported:?}"
                );
            }
        }
    }

    fn declared_variant_count(source: &str) -> usize {
        source
            .split_once("pub enum LoadError {")
            .map(|(_, body)| {
                body.lines()
                    .take_while(|line| *line != "}")
                    .filter(|line| {
                        let trimmed = line.trim_start();
                        line.len() - trimmed.len() == 4 && trimmed.starts_with(char::is_uppercase)
                    })
                    .count()
            })
            .expect("LoadError is declared in this file")
    }

    #[test]
    fn the_table_covers_every_variant() {
        let lines = include_str!("error.rs").lines().collect::<Vec<_>>();
        for newline in ["\n", "\r\n"] {
            assert_eq!(
                every_variant().len(),
                declared_variant_count(&lines.join(newline)),
                "the message table is missing a variant with {newline:?} line endings"
            );
        }
    }

    #[test]
    fn source_is_exposed_only_for_wrapping_variants() {
        use std::error::Error as _;
        for (error, _) in every_variant() {
            let wraps = matches!(error, LoadError::Io(_) | LoadError::Yaml { .. });
            assert_eq!(error.source().is_some(), wraps, "{error:?} disagrees about its source");
        }
    }

    #[test]
    fn parse_errors_preserve_the_underlying_message_and_source() {
        use std::error::Error as _;
        let error = parse_error();
        assert_eq!(error.to_string(), error.0.to_string());
        assert!(std::ptr::eq(
            error.source().unwrap().downcast_ref::<serde_yaml::Error>().unwrap(),
            &raw const error.0,
        ));
    }

    #[test]
    fn io_conversion_preserves_the_underlying_error() {
        use std::error::Error as _;
        let error = LoadError::from(std::io::Error::new(std::io::ErrorKind::PermissionDenied, "blocked"));
        assert!(matches!(error, LoadError::Io(_)));
        assert_eq!(error.to_string(), "I/O error: blocked");
        let source = error.source().unwrap().downcast_ref::<std::io::Error>().unwrap();
        assert_eq!(source.kind(), std::io::ErrorKind::PermissionDenied);
        assert_eq!(source.to_string(), "blocked");
    }
}
