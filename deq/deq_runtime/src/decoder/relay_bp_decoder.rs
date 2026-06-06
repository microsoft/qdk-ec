//! Relay-BP decoder backed by the public [`relay-bp`](https://crates.io/crates/relay-bp) crate.
//!

use crate::decoder::blackbox_decoder::{self, ParityFactor};
use crate::decoder::thread_pooling::{DecoderInstance, ThreadPoolingConfig, ThreadPoolingDecoder};
use crate::misc::bit_vector::to_sparse_indices;
use crate::util::BitVector;
use blackbox_decoder::DecodingHypergraph;
use core::panic;
use ndarray::{Array1, Array2};
use num_traits::{Bounded, Float, FromPrimitive, Signed};
use relay_bp::bp::min_sum::MinSumDecoderConfig;
use relay_bp::bp::relay::{RelayDecoder, RelayDecoderConfig, StoppingCriterion};
use relay_bp::decoder::{Bit, Decoder, SparseBitMatrix};
use serde::{Deserialize, Serialize};
use sprs::TriMat;
use std::ops::{AddAssign, DivAssign, MulAssign};
use std::sync::Arc;
#[cfg(feature = "cli")]
use structdoc::StructDoc;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct RelayBPDecoderConfig {
    /// we want to recognize all the thread pooling config fields
    #[serde(flatten)]
    pub thread_pooling_config: ThreadPoolingConfig,
    /// BP decoder parameters
    #[serde(flatten)]
    pub min_sum_decoder_config: SerdeMinSumBPDecoderConfig,
    /// relay-BP decoder parameters
    #[serde(flatten)]
    pub relay_decoder_config: SerdeRelayDecoderConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct SerdeMinSumBPDecoderConfig {
    #[serde(default = "max_iter")]
    pub max_iter: usize,
    pub alpha: Option<f64>,
    #[serde(default = "alpha_iteration_scaling_factor")]
    pub alpha_iteration_scaling_factor: f64,
    #[serde(default = "gamma0")]
    pub gamma0: Option<f64>,
}

pub fn max_iter() -> usize {
    200
}

pub fn alpha_iteration_scaling_factor() -> f64 {
    1.0
}

pub fn gamma0() -> Option<f64> {
    Some(0.65)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct SerdeRelayDecoderConfig {
    #[serde(default = "pre_iter")]
    pub pre_iter: usize,
    #[serde(default = "num_sets")]
    pub num_sets: usize,
    #[serde(default = "set_max_iter")]
    pub set_max_iter: usize,
    #[serde(default = "gamma_dist_interval_lower")]
    pub gamma_dist_interval_lower: f64,
    #[serde(default = "gamma_dist_interval_upper")]
    pub gamma_dist_interval_upper: f64,
    pub explicit_gammas: Option<Vec<Vec<f64>>>,
    #[serde(default = "stopping_criterion")]
    pub stopping_criterion: String,
    #[serde(default)]
    pub seed: u64,
}

pub fn pre_iter() -> usize {
    80
}

pub fn num_sets() -> usize {
    100
}

pub fn set_max_iter() -> usize {
    60
}

pub fn gamma_dist_interval_lower() -> f64 {
    -0.24
}

pub fn gamma_dist_interval_upper() -> f64 {
    0.66
}

pub fn stopping_criterion() -> String {
    "NConv(stop_after=5)".to_string()
}

pub fn stopping_criterion_of(stopping_criterion: &str) -> StoppingCriterion {
    if stopping_criterion == "PreIter" {
        StoppingCriterion::PreIter
    } else if stopping_criterion == "All" {
        StoppingCriterion::All
    } else if stopping_criterion.starts_with("NConv(stop_after=") && stopping_criterion.ends_with(")") {
        let stop_after = &stopping_criterion[17..stopping_criterion.len() - 1];
        let stop_after: usize = stop_after.parse().unwrap();
        StoppingCriterion::NConv { stop_after }
    } else {
        panic!("unknown stopping criterion: {}", stopping_criterion);
    }
}

pub trait RelayBPDecoderDataType:
    Float
    + Default
    + std::fmt::Debug
    + std::marker::Send
    + std::marker::Sync
    + std::fmt::Display
    + Signed
    + Bounded
    + FromPrimitive
    + AddAssign
    + MulAssign
    + DivAssign
    + 'static
{
}

impl RelayBPDecoderDataType for f64 {}
impl RelayBPDecoderDataType for f32 {}

pub struct RelayBPDecoderInstance<N: RelayBPDecoderDataType + 'static = f64> {
    solver: RelayDecoder<N>,
}

impl<N: RelayBPDecoderDataType + 'static> DecoderInstance for RelayBPDecoderInstance<N> {
    fn new(hypergraph: &DecodingHypergraph, config: &serde_json::Value) -> Self {
        let config: RelayBPDecoderConfig = serde_json::from_value(config.clone()).unwrap();
        // build check matrix
        let mut check_matrix = TriMat::new((hypergraph.vertex_num as usize, hypergraph.hyperedges.len()));
        let mut error_priors = Vec::with_capacity(hypergraph.hyperedges.len());
        for (j, hyperedge) in hypergraph.hyperedges.iter().enumerate() {
            for &i in hyperedge.vertices.iter() {
                check_matrix.add_triplet(i as usize, j, 1);
            }
            error_priors.push(hyperedge.probability);
        }
        let check_matrix: SparseBitMatrix = check_matrix.to_csr();
        let min_sum_decoder_config = MinSumDecoderConfig {
            error_priors: error_priors.into(),
            max_iter: config.min_sum_decoder_config.max_iter,
            alpha: config.min_sum_decoder_config.alpha,
            alpha_iteration_scaling_factor: config.min_sum_decoder_config.alpha_iteration_scaling_factor,
            gamma0: config.min_sum_decoder_config.gamma0,
            // Fixed-point options are not exposed via this serde config; use
            // floating-point defaults that match the public crate's `Default`.
            data_scale_value: None,
            max_data_value: None,
            int_bits: None,
            frac_bits: None,
        };
        let relay_decoder_config = RelayDecoderConfig {
            pre_iter: config.relay_decoder_config.pre_iter,
            num_sets: config.relay_decoder_config.num_sets,
            set_max_iter: config.relay_decoder_config.set_max_iter,
            gamma_dist_interval: (
                config.relay_decoder_config.gamma_dist_interval_lower,
                config.relay_decoder_config.gamma_dist_interval_upper,
            ),
            explicit_gammas: config.relay_decoder_config.explicit_gammas.map(|v| {
                if v.is_empty() {
                    return Array2::zeros((0, 0));
                }
                let rows = v.len();
                let columns = v[0].len();
                let mut gammas = Array2::zeros((rows, columns));
                for (i, row) in v.iter().enumerate() {
                    assert!(row.len() == columns);
                    for (j, v) in row.iter().enumerate() {
                        gammas[[i, j]] = *v;
                    }
                }
                gammas
            }),
            stopping_criterion: stopping_criterion_of(&config.relay_decoder_config.stopping_criterion),
            // Disable the public crate's file-based logging side effect.
            logging: false,
            seed: config.relay_decoder_config.seed,
        };
        let solver = RelayDecoder::new(
            Arc::new(check_matrix.into_csc()),
            Arc::new(min_sum_decoder_config),
            Arc::new(relay_decoder_config),
        );
        Self { solver }
    }

    fn decode(&mut self, syndrome: &BitVector) -> ParityFactor {
        let mut detectors = Array1::<Bit>::zeros(syndrome.size as usize);
        for index in to_sparse_indices(syndrome) {
            detectors[index as usize] = 1;
        }
        let decoding = self.solver.decode(detectors.view());
        ParityFactor {
            subgraph: decoding
                .iter()
                .enumerate()
                .filter_map(|(i, &bit)| if bit == 1 { Some(i as u64) } else { None })
                .collect(),
        }
    }

    fn reset(&mut self) {}
}

pub type RelayBPDecoder<N = f64> = ThreadPoolingDecoder<RelayBPDecoderInstance<N>>;
