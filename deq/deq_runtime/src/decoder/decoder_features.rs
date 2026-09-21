//! Optional capabilities supported or required by a decoder request.

use crate::decoder::blackbox_decoder;
use std::fmt;

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct DecoderFeatures: u32 {
        const REWEIGHTS = 1 << 0;
        const LOSS = 1 << 1;
    }
}

impl DecoderFeatures {
    #[must_use]
    pub const fn required(has_reweights: bool, has_loss: bool) -> Self {
        let mut features = Self::empty();
        if has_reweights {
            features = features.union(Self::REWEIGHTS);
        }
        if has_loss {
            features = features.union(Self::LOSS);
        }
        features
    }

    #[cfg(any(feature = "python", test))]
    pub(crate) fn from_protocol_name(name: &str) -> Option<Self> {
        match name {
            "reweights" => Some(Self::REWEIGHTS),
            "loss" => Some(Self::LOSS),
            _ => None,
        }
    }

    pub(crate) fn require_supported_by(self, supported: Self) -> Result<(), Self> {
        let unsupported = self.difference(supported);
        if unsupported.is_empty() { Ok(()) } else { Err(unsupported) }
    }

    pub(crate) fn to_proto(self) -> blackbox_decoder::DecoderCapabilities {
        let mut features = Vec::with_capacity(2);
        if self.contains(Self::REWEIGHTS) {
            features.push(blackbox_decoder::DecoderFeature::Reweights as i32);
        }
        if self.contains(Self::LOSS) {
            features.push(blackbox_decoder::DecoderFeature::Loss as i32);
        }
        blackbox_decoder::DecoderCapabilities { features }
    }
}

impl fmt::Display for DecoderFeatures {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut separator = "";
        if self.contains(Self::REWEIGHTS) {
            write!(formatter, "reweights")?;
            separator = ", ";
        }
        if self.contains(Self::LOSS) {
            write!(formatter, "{separator}loss")?;
        }
        if self.is_empty() {
            write!(formatter, "none")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_features_compose_independently() {
        assert_eq!(DecoderFeatures::required(false, false), DecoderFeatures::empty());
        assert_eq!(DecoderFeatures::required(true, false), DecoderFeatures::REWEIGHTS);
        assert_eq!(DecoderFeatures::required(false, true), DecoderFeatures::LOSS);
        assert_eq!(
            DecoderFeatures::required(true, true),
            DecoderFeatures::REWEIGHTS | DecoderFeatures::LOSS
        );
        assert!(
            DecoderFeatures::LOSS
                .require_supported_by(DecoderFeatures::REWEIGHTS | DecoderFeatures::LOSS)
                .is_ok()
        );
        assert_eq!(
            DecoderFeatures::required(true, true).require_supported_by(DecoderFeatures::LOSS),
            Err(DecoderFeatures::REWEIGHTS)
        );
    }

    #[test]
    fn names_and_proto_use_the_protocol_vocabulary() {
        assert_eq!(
            DecoderFeatures::from_protocol_name("reweights"),
            Some(DecoderFeatures::REWEIGHTS)
        );
        assert_eq!(DecoderFeatures::from_protocol_name("loss"), Some(DecoderFeatures::LOSS));
        assert_eq!(DecoderFeatures::from_protocol_name("unknown"), None);
        assert_eq!(
            (DecoderFeatures::REWEIGHTS | DecoderFeatures::LOSS).to_proto().features,
            vec![
                blackbox_decoder::DecoderFeature::Reweights as i32,
                blackbox_decoder::DecoderFeature::Loss as i32,
            ]
        );
    }
}
