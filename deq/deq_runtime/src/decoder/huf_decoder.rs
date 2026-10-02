//! Hypergraph union-find decoder backed by the public `mwpf` crate.

use crate::decoder::blackbox_decoder::{DecodingHypergraph, ParityFactor};
use crate::decoder::mwpf_decoder::{MwpfDecoderConfig, MwpfDecoderInstance, default_timeout};
use crate::decoder::thread_pooling::{
    DecodeError, DecodeRequest, DecoderInstance, ThreadPoolingConfig, ThreadPoolingDecoder,
};
use serde::{Deserialize, Serialize};
#[cfg(feature = "cli")]
use structdoc::StructDoc;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct HufDecoderConfig {
    #[serde(flatten)]
    pub thread_pooling_config: ThreadPoolingConfig,
    /// timeout in seconds for each decoding problem
    #[serde(default = "default_timeout")]
    pub timeout: f64,
    /// defer solving the primal problem until the end of decoding
    #[serde(default)]
    pub only_solve_primal_once: bool,
}

impl HufDecoderConfig {
    fn into_mwpf_config(self) -> MwpfDecoderConfig {
        MwpfDecoderConfig {
            thread_pooling_config: self.thread_pooling_config,
            timeout: self.timeout,
            cluster_node_limit: 0,
            only_solve_primal_once: self.only_solve_primal_once,
        }
    }
}

pub struct HufDecoderInstance {
    inner: MwpfDecoderInstance,
}

impl DecoderInstance for HufDecoderInstance {
    fn validate_hypergraph(hypergraph: &DecodingHypergraph, config: &serde_json::Value) -> Result<(), String> {
        let _: HufDecoderConfig = serde_json::from_value(config.clone()).map_err(|error| error.to_string())?;
        MwpfDecoderInstance::validate_hypergraph(hypergraph, config)
    }

    fn new(hypergraph: &DecodingHypergraph, config: &serde_json::Value) -> Self {
        let config: HufDecoderConfig = serde_json::from_value(config.clone()).unwrap();
        Self {
            inner: MwpfDecoderInstance::new_with_config(hypergraph, &config.into_mwpf_config()),
        }
    }

    fn decode(&mut self, request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
        self.inner.decode(request)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

pub type HufDecoder = ThreadPoolingDecoder<HufDecoderInstance>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_mwpf_tuning_except_for_fixed_cluster_limit() {
        let config: HufDecoderConfig = serde_json::from_value(serde_json::json!({
            "parallel": 3,
            "timeout": 2.5,
            "only_solve_primal_once": true,
        }))
        .unwrap();
        let config = config.into_mwpf_config();

        assert_eq!(config.thread_pooling_config.parallel, 3);
        assert_eq!(config.timeout, 2.5);
        assert_eq!(config.cluster_node_limit, 0);
        assert!(config.only_solve_primal_once);
    }
}
