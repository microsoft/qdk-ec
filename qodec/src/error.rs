//! Errors from slicing, resolving, and saving qodecs.

/// An invalid layer range passed to [`crate::Qodec::slice`].
///
/// `start` and `stop` are zero-based layer indexes.
#[derive(Debug, derive_more::Display, derive_more::Error)]
pub enum SliceError {
    /// `start > stop`.
    #[display("slice: start index {start} must be <= stop index {stop}")]
    InvertedRange {
        /// The requested first layer.
        start: usize,
        /// The requested stop, which is below `start`.
        stop: usize,
    },
    /// `stop` is outside the stack of `layer_count` layers.
    #[display("slice: stop index {stop} out of range (qodec has {layer_count} layers)")]
    StopOutOfRange {
        /// The requested stop, which is above `layer_count`.
        stop: usize,
        /// How many layers the qodec has.
        layer_count: usize,
    },
}

/// A gadget could not be resolved by [`crate::InstructionSet::resolve`].
#[derive(Debug, derive_more::Display, derive_more::Error)]
pub enum ResolveError {
    /// A gadget's `implements` mnemonic is not declared by its source instruction set.
    #[display("instruction '{mnemonic}' not found in source instruction set '{instruction_set}'")]
    InstructionNotFound {
        /// The mnemonic that was looked up.
        mnemonic: String,
        /// The name of the instruction set searched.
        instruction_set: String,
    },
}

/// Failure synthesizing raw, on-disk artifacts from a resolved qodec on the
/// [`crate::Qodec::save`] path.
#[derive(Debug, derive_more::Display, derive_more::Error)]
pub(crate) enum SynthesisError {
    /// A referenced code is missing from the synthesized code map.
    #[display("synthesis: code '{name}' missing from synthesized code map")]
    CodeMissing { name: String },
    /// An encoding's support types are inconsistent or cannot be inferred.
    #[display("synthesis: encoding entry {entry}: {error}")]
    InvalidEncoding { entry: usize, error: String },
}

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
