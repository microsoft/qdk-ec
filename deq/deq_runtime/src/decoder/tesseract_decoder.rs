//! Tesseract decoder
//!
//! Google's Tesseract beam-search QEC decoder, bridged from C++ via cxx.
//!

use crate::decoder::blackbox_decoder::{self, ParityFactor};
use crate::decoder::tesseract_ffi::{TesseractCxxConfig, TesseractCxxDecoder};
use crate::decoder::thread_pooling::{
    DecodeError, DecodeRequest, DecoderFeatures, DecoderInstance, ThreadPoolingConfig, ThreadPoolingDecoder,
};
use crate::misc::bit_vector::to_sparse_indices;
use blackbox_decoder::DecodingHypergraph;
use serde::{Deserialize, Serialize};
#[cfg(feature = "cli")]
use structdoc::StructDoc;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct TesseractDecoderConfig {
    /// we want to recognize all the thread pooling config fields
    #[serde(flatten)]
    pub thread_pooling_config: ThreadPoolingConfig,
    /// beam cutoff (max number of active detectors in a search state)
    #[serde(default = "default_det_beam")]
    pub det_beam: i32,
    /// enable beam climbing heuristic
    #[serde(default)]
    pub beam_climbing: bool,
    /// prevent revisiting syndrome patterns
    #[serde(default = "default_true")]
    pub no_revisit_dets: bool,
    /// merge indistinguishable error mechanisms
    #[serde(default = "default_true")]
    pub merge_errors: bool,
    /// priority queue size limit
    #[serde(default = "default_pqlimit")]
    pub pqlimit: u64,
    /// penalty added per detector visited
    #[serde(default)]
    pub det_penalty: f64,
}

fn default_det_beam() -> i32 {
    5
}

fn default_true() -> bool {
    true
}

fn default_pqlimit() -> u64 {
    200_000
}

pub struct TesseractDecoderInstance {
    decoder: TesseractCxxDecoder,
    /// The probabilities the loaded decoder was built from, in Tesseract error
    /// order. Kept so [`DecoderInstance::decode`] can restore them
    /// after a shot-scoped reweighting.
    base_probabilities: Vec<f64>,
}

impl DecoderInstance for TesseractDecoderInstance {
    fn supported_features(_config: &serde_json::Value) -> DecoderFeatures {
        DecoderFeatures::REWEIGHTS
    }

    fn new(hypergraph: &DecodingHypergraph, config: &serde_json::Value) -> Self {
        let config: TesseractDecoderConfig = serde_json::from_value(config.clone()).unwrap();
        // Every hyperedge is loaded, including the zero-probability ones. Such an
        // edge is a *declared* impossibility rather than an absent one -- the
        // producer emits it deliberately and its index is part of the decoding
        // interface, referenced by `LossInfo` and by per-shot prior overrides --
        // so dropping it would both break that indexing and make the edge
        // impossible to raise later. Tesseract carries it at a cost far beyond
        // any real explanation until something raises it.
        let (edge_vertices, edge_offsets, edge_probabilities) = flatten_hypergraph(hypergraph);
        let tess_config = TesseractCxxConfig {
            det_beam: config.det_beam,
            beam_climbing: config.beam_climbing,
            no_revisit_dets: config.no_revisit_dets,
            merge_errors: config.merge_errors,
            pqlimit: config.pqlimit,
            det_penalty: config.det_penalty,
        };
        Self {
            decoder: TesseractCxxDecoder::new(
                hypergraph.vertex_num,
                &edge_vertices,
                &edge_offsets,
                &edge_probabilities,
                &tess_config,
            ),
            base_probabilities: edge_probabilities,
        }
    }

    fn decode(&mut self, request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
        if !request.reweights.is_empty() {
            let mut probabilities = self.base_probabilities.clone();
            for &(edge, probability) in request.reweights {
                let position = usize::try_from(edge).map_err(|_| {
                    DecodeError::InvalidInput(format!("reweighted edge {edge} is outside the loaded hypergraph"))
                })?;
                let target = probabilities.get_mut(position).ok_or_else(|| {
                    DecodeError::InvalidInput(format!("reweighted edge {edge} is outside the loaded hypergraph"))
                })?;
                *target = probability;
            }
            self.decoder.update_error_costs(&probabilities);
        }
        let detections: Vec<u64> = to_sparse_indices(request.syndrome);
        let error_indices = self.decoder.decode(&detections);
        if !request.reweights.is_empty() {
            self.decoder.update_error_costs(&self.base_probabilities);
        }
        Ok(ParityFactor { subgraph: error_indices })
    }

    fn reset(&mut self) {
        // Tesseract clears its internal buffers at the start of each decode call.
    }
}

/// Flatten a decoding hypergraph into CSR arrays for the C++ bridge.
fn flatten_hypergraph(hypergraph: &DecodingHypergraph) -> (Vec<u64>, Vec<u64>, Vec<f64>) {
    let total_vertices: usize = hypergraph.hyperedges.iter().map(|edge| edge.vertices.len()).sum();

    let mut edge_vertices = Vec::with_capacity(total_vertices);
    let mut edge_offsets = Vec::with_capacity(hypergraph.hyperedges.len() + 1);
    let mut edge_probabilities = Vec::with_capacity(hypergraph.hyperedges.len());

    edge_offsets.push(0u64);
    for edge in &hypergraph.hyperedges {
        edge_vertices.extend_from_slice(&edge.vertices);
        edge_offsets.push(edge_vertices.len() as u64);
        edge_probabilities.push(edge.probability);
    }

    (edge_vertices, edge_offsets, edge_probabilities)
}

pub type TesseractDecoder = ThreadPoolingDecoder<TesseractDecoderInstance>;
