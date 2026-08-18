//! Reweight-handling pass for decoder problems.
//!
//! This module owns the loaded-decoder context, hyperedge deduplication, and
//! translation of shot-scoped probability updates onto decoder hypergraphs.
//! It is the probability-reweight counterpart to `loss_handler`.
//!
//! Reweights begin in the original hypergraph's edge numbering. Before a graph
//! is loaded, same-syndrome edges may be collapsed. Window decoders preserve
//! vertex numbering but ignore history-boundary vertices with no incident edge.
//! [`DecodeProjection`] retains the
//! original graph and edge mappings needed to translate every later shot;
//! [`LoadedDecoder`] retains correction representatives and, only when needed
//! locally, the decoder-facing graph.

use crate::bin;
use crate::decoder::DynDecoder;
use crate::decoder::blackbox_decoder;
use crate::decoder::decoder_features::DecoderFeatures;
use crate::misc::index::ErrorIndex;
use crate::misc::util::exclusive_probability_of;
use crate::util::BitVector;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tonic::Status;

/// Build and load the stable decoder graph for a persistent cache entry.
///
/// Deduplication and original-edge projection are shared by coordinators.
/// Window decoding may additionally ignore edge-isolated history-boundary
/// syndrome vertices without renumbering the graph.
pub(crate) async fn load_projected_decoder(
    decoder: &DynDecoder,
    base_hypergraph: blackbox_decoder::DecodingHypergraph,
    base_errors: Arc<Vec<ErrorIndex>>,
    representative_priors: &[f64],
    deduplicate: bool,
    retain_decoding_hypergraph: bool,
    ignore_isolated_vertices: bool,
) -> Result<LoadedDecoder, Status> {
    let (projection, prepared) = prepare_decoder(base_hypergraph, base_errors, representative_priors, deduplicate);
    let PreparedDecoderInput {
        hypergraph,
        representatives,
    } = prepared;
    let ignored_syndrome_vertices = if ignore_isolated_vertices {
        Arc::new(edge_isolated_vertices(&hypergraph))
    } else {
        Arc::new(vec![])
    };
    let decoding_hypergraph = retain_decoding_hypergraph.then(|| Arc::new(hypergraph.clone()));
    let hid = decoder.load_hypergraph(hypergraph).await?.hid;
    Ok(LoadedDecoder {
        hid,
        errors: Arc::new(representatives),
        decoding_hypergraph,
        ignored_syndrome_vertices,
        projection: Arc::new(projection),
    })
}

fn prepare_decoder(
    base_hypergraph: blackbox_decoder::DecodingHypergraph,
    base_errors: Arc<Vec<ErrorIndex>>,
    representative_priors: &[f64],
    deduplicate: bool,
) -> (DecodeProjection, PreparedDecoderInput) {
    debug_assert_eq!(base_hypergraph.hyperedges.len(), base_errors.len());
    debug_assert_eq!(base_hypergraph.hyperedges.len(), representative_priors.len());
    let (prepared, edge_projection) = if deduplicate {
        deduplicate_by_syndrome(&base_hypergraph, &base_errors, representative_priors)
    } else {
        (
            PreparedDecoderInput {
                hypergraph: base_hypergraph.clone(),
                representatives: base_errors.as_ref().clone(),
            },
            EdgeProjection::Identity,
        )
    };
    (
        DecodeProjection {
            base_hypergraph,
            base_errors,
            edge_projection,
        },
        prepared,
    )
}

fn edge_isolated_vertices(hypergraph: &blackbox_decoder::DecodingHypergraph) -> Vec<u64> {
    let mut incident = vec![false; hypergraph.vertex_num as usize];
    for hyperedge in &hypergraph.hyperedges {
        for &vertex in &hyperedge.vertices {
            incident[vertex as usize] = true;
        }
    }
    incident
        .into_iter()
        .enumerate()
        .filter_map(|(vertex, incident)| (!incident).then_some(vertex as u64))
        .collect()
}

/// Clear syndrome bits for history-boundary vertices that no decoder edge can
/// affect, while preserving the graph's stable vertex numbering.
pub(crate) fn ignore_edge_isolated_history_vertices(
    hypergraph: &blackbox_decoder::DecodingHypergraph,
    syndrome: &mut BitVector,
) {
    for vertex in edge_isolated_vertices(hypergraph) {
        crate::misc::bit_vector::set_bit(syndrome, vertex, false);
    }
}

/// Convert gadget-local `(error model, generator)` assignments into updates in
/// the original decoding hypergraph's edge numbering.
///
/// Several modifiers may target the same edge; the last assignment wins. A
/// later call to [`DecodeProjection::translate_reweights`] translates these
/// original indices into the edge numbering used by a persistent decoder.
pub(crate) fn probability_reweights<'a>(
    error_reference: &[ErrorIndex],
    modifiers: impl IntoIterator<Item = (usize, &'a bin::ProbabilityModifier)>,
) -> Vec<(u64, f64)> {
    let modifiers: Vec<_> = modifiers.into_iter().collect();
    if modifiers.is_empty() {
        return vec![];
    }
    let mut edge_of = hashbrown::HashMap::with_capacity(error_reference.len());
    for (edge, error) in error_reference.iter().enumerate() {
        edge_of.insert((error.eid, error.error_index), u64::try_from(edge).unwrap());
    }
    let mut overrides = hashbrown::HashMap::new();
    for (local_eid, modifier) in modifiers {
        for (error_index, &probability) in modifier.probabilities.iter().enumerate() {
            if let Some(&edge) = edge_of.get(&(local_eid, error_index)) {
                overrides.insert(edge, probability);
            }
        }
        for (&error_index, &probability) in modifier.sparse_indices.iter().zip(modifier.sparse_probabilities.iter()) {
            if let Some(&edge) = edge_of.get(&(local_eid, error_index as usize)) {
                overrides.insert(edge, probability);
            }
        }
    }
    let mut reweights: Vec<_> = overrides.into_iter().collect();
    reweights.sort_unstable_by_key(|&(edge, _)| edge);
    reweights
}

/// Materialize edge probability updates directly into a hypergraph.
pub(crate) fn apply_reweights(hypergraph: &mut blackbox_decoder::DecodingHypergraph, reweights: &[(u64, f64)]) {
    for &(edge, probability) in reweights {
        hypergraph.hyperedges[edge as usize].probability = probability;
    }
}

/// Select how shot-scoped edge updates reach a persistent decoder.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(structdoc::StructDoc))]
#[serde(rename_all = "snake_case")]
pub enum DecoderReweighting {
    /// Use loaded reweights when the decoder advertises support; otherwise
    /// materialize a temporary hypergraph for that shot.
    #[default]
    Auto,
    /// Require the decoder to accept reweights alongside a loaded graph.
    Enabled,
    /// Always materialize a temporary hypergraph when a shot changes weights.
    Disabled,
}

impl DecoderReweighting {
    pub(crate) fn use_loaded(self, persistent_decoder: bool, features: DecoderFeatures) -> Result<bool, String> {
        if !persistent_decoder {
            return match self {
                Self::Enabled => Err(
                    "decoder_reweighting is enabled but persistent_decoder is disabled; loaded reweights cannot be used"
                        .to_string(),
                ),
                Self::Auto | Self::Disabled => Ok(false),
            };
        }
        match self {
            Self::Auto => Ok(features.contains(DecoderFeatures::REWEIGHTS)),
            Self::Enabled if features.contains(DecoderFeatures::REWEIGHTS) => Ok(true),
            Self::Enabled => Err("decoder_reweighting is enabled but the decoder does not support reweights".to_string()),
            Self::Disabled => Ok(false),
        }
    }
}

/// A decoder loaded with one stable hypergraph and its shot-projection context.
#[derive(Debug, Clone)]
pub struct LoadedDecoder {
    /// Backend handle returned when the stable deduplicated graph was loaded.
    pub hid: u64,
    /// One correction representative per decoder edge, used to interpret the
    /// subgraph returned by the backend. These are post-deduplication errors;
    /// [`DecodeProjection::base_errors`] retains the original edge list.
    pub errors: Arc<Vec<ErrorIndex>>,
    /// Decoder-facing graph after edge deduplication. It is retained only when
    /// a shot may need a materialized fallback graph or when parity-factor
    /// assertions need the graph locally.
    pub decoding_hypergraph: Option<Arc<blackbox_decoder::DecodingHypergraph>>,
    /// History-boundary vertices with no incident decoder edge. Window
    /// coordinators clear these syndrome bits instead of renumbering vertices;
    /// monolithic coordinators leave this list empty.
    pub ignored_syndrome_vertices: Arc<Vec<u64>>,
    /// Stable original-edge context used on cache hits to translate each shot's
    /// probability and loss updates into the loaded decoder's edge numbering.
    /// Shared because cached [`LoadedDecoder`] values are cloned per shot.
    pub projection: Arc<DecodeProjection>,
}

impl LoadedDecoder {
    /// Apply the stable window-boundary syndrome projection for this graph.
    pub(crate) fn project_syndrome(&self, mut syndrome: BitVector) -> BitVector {
        for &vertex in self.ignored_syndrome_vertices.iter() {
            crate::misc::bit_vector::set_bit(&mut syndrome, vertex, false);
        }
        syndrome
    }
}

/// Decode one projected shot, either by updating the already-loaded graph or
/// by materializing a one-shot graph when the backend lacks loaded reweights.
/// Structured loss accompanies the request in either path.
pub(crate) async fn decode_projected(
    decoder: &DynDecoder,
    loaded: &LoadedDecoder,
    syndrome: BitVector,
    reweights: Vec<blackbox_decoder::EdgeReweight>,
    loss: Option<blackbox_decoder::LossInfo>,
    use_loaded_reweights: bool,
) -> Result<blackbox_decoder::ParityFactor, Status> {
    if reweights.is_empty() || use_loaded_reweights {
        return decoder
            .decode_loaded(blackbox_decoder::LoadedDecodingProblem {
                hid: loaded.hid,
                syndrome: Some(syndrome),
                reweights,
                loss,
            })
            .await;
    }

    // The backend cannot modify its loaded graph. Recreate that exact
    // decoder-facing graph, apply this shot's updates, and use one-shot decode.
    let mut hypergraph = (**loaded
        .decoding_hypergraph
        .as_ref()
        .ok_or_else(|| Status::internal(format!("hid={} has no materializable hypergraph", loaded.hid)))?)
    .clone();
    for reweight in reweights {
        let hyperedge = hypergraph.hyperedges.get_mut(reweight.edge as usize).ok_or_else(|| {
            Status::invalid_argument(format!(
                "reweighted edge {} is outside loaded hypergraph hid={}",
                reweight.edge, loaded.hid
            ))
        })?;
        hyperedge.probability = reweight.probability;
    }
    decoder
        .decode(blackbox_decoder::DecodingProblem {
            hypergraph: Some(hypergraph),
            syndrome: Some(syndrome),
            loss,
        })
        .await
}

/// Stable context for projecting shot-scoped updates onto a loaded hypergraph.
///
/// The base graph and error reference use original edge numbering. The private
/// edge projection translates that space into decoder edge numbering. The
/// decoder-facing graph and correction representatives are consumed separately
/// through [`PreparedDecoderInput`].
#[derive(Debug)]
pub struct DecodeProjection {
    /// Graph before same-syndrome edge deduplication and before any shot-scoped
    /// probability or loss update.
    pub base_hypergraph: blackbox_decoder::DecodingHypergraph,
    /// Error-model generator represented by each edge of [`Self::base_hypergraph`].
    pub base_errors: Arc<Vec<ErrorIndex>>,
    /// Translation between original edge indices and decoder edge indices.
    edge_projection: EdgeProjection,
}

/// Transient decoder-space values produced while building a projection.
///
/// The hypergraph moves into the decoder backend and the representatives move
/// into [`LoadedDecoder::errors`]; neither remains duplicated in
/// [`DecodeProjection`].
#[derive(Debug)]
pub(crate) struct PreparedDecoderInput {
    pub(crate) hypergraph: blackbox_decoder::DecodingHypergraph,
    pub(crate) representatives: Vec<ErrorIndex>,
}

/// Bidirectional relationship between original and decoder edge numbering.
/// Identity projections allocate no mapping arrays.
#[derive(Debug)]
enum EdgeProjection {
    Identity,
    Merged {
        decoder_edge_of_original: Vec<usize>,
        original_edges_of_decoder: Vec<Vec<usize>>,
    },
}

/// Collapse same-syndrome hyperedges while preserving the highest-prior
/// correction representative for each group. The supplied slices must be
/// edge-aligned in original numbering.
fn deduplicate_by_syndrome(
    hypergraph: &blackbox_decoder::DecodingHypergraph,
    errors: &[ErrorIndex],
    priors: &[f64],
) -> (PreparedDecoderInput, EdgeProjection) {
    let mut seen: hashbrown::HashMap<Vec<u64>, (usize, f64)> = hashbrown::HashMap::with_capacity(errors.len());
    let mut hyperedges: Vec<blackbox_decoder::Hyperedge> = Vec::with_capacity(errors.len());
    let mut representatives = Vec::with_capacity(errors.len());
    let mut decoder_edge_of_original = Vec::with_capacity(errors.len());
    let mut original_edges_of_decoder: Vec<Vec<usize>> = Vec::with_capacity(errors.len());
    for (position, ((hyperedge, error), &prior)) in
        hypergraph.hyperedges.iter().zip(errors.iter()).zip(priors.iter()).enumerate()
    {
        let mut syndrome = hyperedge.vertices.clone();
        syndrome.sort_unstable();
        debug_assert!({
            let degree = syndrome.len();
            syndrome.dedup();
            syndrome.len() == degree
        });
        if let Some((index, best_prior)) = seen.get_mut(&syndrome) {
            let combined = hyperedges[*index].probability;
            hyperedges[*index].probability = exclusive_probability_of(combined, hyperedge.probability);
            if prior > *best_prior {
                *best_prior = prior;
                representatives[*index] = error.clone();
            }
            original_edges_of_decoder[*index].push(position);
            decoder_edge_of_original.push(*index);
        } else {
            let index = representatives.len();
            hyperedges.push(blackbox_decoder::Hyperedge {
                probability: hyperedge.probability,
                vertices: syndrome.clone(),
            });
            representatives.push(error.clone());
            original_edges_of_decoder.push(vec![position]);
            decoder_edge_of_original.push(index);
            seen.insert(syndrome, (index, prior));
        }
    }
    (
        PreparedDecoderInput {
            hypergraph: blackbox_decoder::DecodingHypergraph {
                vertex_num: hypergraph.vertex_num,
                hyperedges,
            },
            representatives,
        },
        EdgeProjection::Merged {
            decoder_edge_of_original,
            original_edges_of_decoder,
        },
    )
}

/// Deduplicate a one-shot graph whose edge mapping will not be cached.
pub(crate) fn deduplicate_decoder_input(
    hypergraph: &blackbox_decoder::DecodingHypergraph,
    errors: &[ErrorIndex],
    priors: &[f64],
) -> PreparedDecoderInput {
    deduplicate_by_syndrome(hypergraph, errors, priors).0
}

impl DecodeProjection {
    #[cfg(test)]
    pub(crate) fn identity(
        base_hypergraph: blackbox_decoder::DecodingHypergraph,
        base_errors: Arc<Vec<ErrorIndex>>,
    ) -> Self {
        Self {
            base_hypergraph,
            base_errors,
            edge_projection: EdgeProjection::Identity,
        }
    }

    /// Translate original-edge assignments into decoder-edge assignments.
    pub(crate) fn translate_reweights(&self, reweights: &[(u64, f64)]) -> Vec<(u64, f64)> {
        self.edge_projection.translate_reweights(&self.base_hypergraph, reweights)
    }
}

impl EdgeProjection {
    fn translate_reweights(
        &self,
        base_hypergraph: &blackbox_decoder::DecodingHypergraph,
        reweights: &[(u64, f64)],
    ) -> Vec<(u64, f64)> {
        match self {
            Self::Identity => {
                let mut overrides = hashbrown::HashMap::with_capacity(reweights.len());
                for &(edge, probability) in reweights {
                    overrides.insert(edge, probability);
                }
                let mut translated: Vec<_> = overrides.into_iter().collect();
                translated.sort_unstable_by_key(|&(edge, _)| edge);
                translated
            }
            Self::Merged {
                decoder_edge_of_original,
                original_edges_of_decoder,
            } => {
                let mut overrides = hashbrown::HashMap::with_capacity(reweights.len());
                let mut affected = Vec::with_capacity(reweights.len());
                for &(edge, probability) in reweights {
                    let original = usize::try_from(edge).unwrap();
                    overrides.insert(original, probability);
                    affected.push(decoder_edge_of_original[original]);
                }
                affected.sort_unstable();
                affected.dedup();
                affected
                    .into_iter()
                    .map(|decoder_edge| {
                        let combined =
                            original_edges_of_decoder[decoder_edge]
                                .iter()
                                .fold(0.0, |accumulated, &original_edge| {
                                    let probability = overrides
                                        .get(&original_edge)
                                        .copied()
                                        .unwrap_or(base_hypergraph.hyperedges[original_edge].probability);
                                    exclusive_probability_of(accumulated, probability)
                                });
                        (u64::try_from(decoder_edge).unwrap(), combined)
                    })
                    .collect()
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/reweight_handler_test.rs"]
mod tests;
