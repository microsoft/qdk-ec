//! Union-find graph decoder backed by Fusion Blossom.

use crate::decoder::blackbox_decoder::{DecodingHypergraph, ParityFactor};
use crate::decoder::mwpm_decoder::{MwpmDecoderConfig, MwpmDecoderInstance, default_max_half_weight};
use crate::decoder::thread_pooling::{
    DecodeError, DecodeRequest, DecoderInstance, ThreadPoolingConfig, ThreadPoolingDecoder,
};
use serde::{Deserialize, Serialize};
#[cfg(feature = "cli")]
use structdoc::StructDoc;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct UfDecoderConfig {
    #[serde(flatten)]
    pub thread_pooling_config: ThreadPoolingConfig,
    /// largest half-weight used when scaling probability log-odds
    #[serde(default = "default_max_half_weight")]
    pub max_half_weight: u32,
}

impl UfDecoderConfig {
    fn into_mwpm_config(self) -> MwpmDecoderConfig {
        MwpmDecoderConfig {
            thread_pooling_config: self.thread_pooling_config,
            max_half_weight: self.max_half_weight,
            max_tree_size: Some(0),
        }
    }
}

pub struct UfDecoderInstance {
    inner: MwpmDecoderInstance,
}

impl DecoderInstance for UfDecoderInstance {
    fn validate_hypergraph(hypergraph: &DecodingHypergraph, config: &serde_json::Value) -> Result<(), String> {
        let _: UfDecoderConfig = serde_json::from_value(config.clone()).map_err(|error| error.to_string())?;
        MwpmDecoderInstance::validate_hypergraph(hypergraph, config)
    }

    fn new(hypergraph: &DecodingHypergraph, config: &serde_json::Value) -> Self {
        let config: UfDecoderConfig = serde_json::from_value(config.clone()).unwrap();
        Self {
            inner: MwpmDecoderInstance::new_with_config(hypergraph, &config.into_mwpm_config()),
        }
    }

    fn decode(&mut self, request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
        self.inner.decode(request)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

pub type UfDecoder = ThreadPoolingDecoder<UfDecoderInstance>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_mwpm_tuning_except_for_fixed_tree_limit() {
        let config: UfDecoderConfig = serde_json::from_value(serde_json::json!({
            "parallel": 3,
            "max_half_weight": 700,
        }))
        .unwrap();
        let config = config.into_mwpm_config();

        assert_eq!(config.thread_pooling_config.parallel, 3);
        assert_eq!(config.max_half_weight, 700);
        assert_eq!(config.max_tree_size, Some(0));
    }
}
