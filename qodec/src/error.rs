//! Errors from slicing, resolving, and saving qodecs.

/// An invalid layer range passed to [`crate::Qodec::slice`].
///
/// `start` and `stop` are zero-based layer indexes.
#[derive(Debug)]
pub enum SliceError {
    /// `start > stop`.
    InvertedRange {
        /// The requested first layer.
        start: usize,
        /// The requested stop, which is below `start`.
        stop: usize,
    },
    /// `stop` is outside the stack of `layer_count` layers.
    StopOutOfRange {
        /// The requested stop, which is above `layer_count`.
        stop: usize,
        /// How many layers the qodec has.
        layer_count: usize,
    },
}

impl std::fmt::Display for SliceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvertedRange { start, stop } => {
                write!(f, "slice: start index {start} must be <= stop index {stop}")
            }
            Self::StopOutOfRange { stop, layer_count } => write!(
                f,
                "slice: stop index {stop} out of range (qodec has {layer_count} layers)"
            ),
        }
    }
}

impl std::error::Error for SliceError {}

/// A gadget could not be resolved by [`crate::InstructionSet::resolve`].
#[derive(Debug)]
pub enum ResolveError {
    /// A gadget's `implements` mnemonic is not declared by its source instruction set.
    InstructionNotFound {
        /// The mnemonic that was looked up.
        mnemonic: String,
        /// The name of the instruction set searched.
        instruction_set: String,
    },
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InstructionNotFound {
                mnemonic,
                instruction_set,
            } => {
                write!(
                    f,
                    "instruction '{mnemonic}' not found in source instruction set '{instruction_set}'"
                )
            }
        }
    }
}

impl std::error::Error for ResolveError {}

/// Failure synthesizing raw, on-disk artifacts from a resolved qodec on the
/// [`crate::Qodec::save`] path.
#[derive(Debug)]
pub(crate) enum SynthesisError {
    /// A referenced code is missing from the synthesized code map.
    CodeMissing { name: String },
    /// An encoding's support types are inconsistent or cannot be inferred.
    InvalidEncoding { entry: usize, error: String },
}

impl std::fmt::Display for SynthesisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CodeMissing { name } => {
                write!(f, "synthesis: code '{name}' missing from synthesized code map")
            }
            Self::InvalidEncoding { entry, error } => write!(f, "synthesis: encoding entry {entry}: {error}"),
        }
    }
}

impl std::error::Error for SynthesisError {}

#[cfg(test)]
mod tests {
    use super::{ResolveError, SliceError, SynthesisError};

    #[test]
    fn slice_errors_report_their_indices() {
        assert_eq!(
            SliceError::InvertedRange { start: 3, stop: 1 }.to_string(),
            "slice: start index 3 must be <= stop index 1"
        );
        assert_eq!(
            SliceError::StopOutOfRange {
                stop: 9,
                layer_count: 2
            }
            .to_string(),
            "slice: stop index 9 out of range (qodec has 2 layers)"
        );
    }

    #[test]
    fn resolve_error_reports_mnemonic_and_instruction_set() {
        assert_eq!(
            ResolveError::InstructionNotFound {
                mnemonic: "measure_z".to_owned(),
                instruction_set: "repetition3".to_owned(),
            }
            .to_string(),
            "instruction 'measure_z' not found in source instruction set 'repetition3'"
        );
    }

    #[test]
    fn synthesis_errors_report_their_subjects() {
        assert_eq!(
            SynthesisError::CodeMissing { name: "c6".to_owned() }.to_string(),
            "synthesis: code 'c6' missing from synthesized code map"
        );
        assert_eq!(
            SynthesisError::InvalidEncoding {
                entry: 1,
                error: "conflicting block types".to_owned(),
            }
            .to_string(),
            "synthesis: encoding entry 1: conflicting block types"
        );
    }

    #[test]
    fn errors_are_std_errors() {
        fn assert_error<E: std::error::Error>(_: &E) {}
        assert_error(&SliceError::InvertedRange { start: 1, stop: 0 });
        assert_error(&ResolveError::InstructionNotFound {
            mnemonic: String::new(),
            instruction_set: String::new(),
        });
        assert_error(&SynthesisError::CodeMissing { name: String::new() });
    }
}
