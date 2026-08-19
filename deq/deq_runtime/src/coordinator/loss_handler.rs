//! Loss-handling strategy: what a coordinator does with heralded atom losses.
//!
//! The coordinator config exposes two fields — a [`LossStrategy`] and an opaque
//! `loss_config` JSON blob — instead of one flag per mechanism. The blob is
//! parsed and validated *once*, here, into a [`LossHandler`]; a coordinator then
//! only asks the handler what to do. Options that belong to another strategy are
//! rejected at construction rather than silently ignored, which is the failure
//! mode the flat `loss_envelope_matching` / `loss_weight_fraction` /
//! `loss_mle_decoding` triple invited.

use super::reweight_handler::{DecodeProjection, ProjectedErrors, apply_reweights};
use crate::decoder::blackbox_decoder;
use crate::jit::loss_compiler::CrossGadgetLossSite;
use crate::misc::index::ErrorIndex;
use crate::misc::util::{exclusive_probability_of, probability_of_weight, weight_of};
use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
#[cfg(feature = "cli")]
use structdoc::StructDoc;

/// Default envelope-matching exponent.
///
/// Swept on a `d = 3` mid-swap memory (`eta = 0.7`) over `f` in `[0.05, 1.0]`.
/// At low loss rates the logical error rate is flat in `f` — a 10x range moves it
/// by less than noise — because most edges are activated by a single site. At
/// `p_loss` of `1-2%`, where several sites share an edge, the curve is a clear
/// `5-6 sigma` bowl with a broad optimum at `0.5-0.7`: below it the loss edges
/// get cheap enough for the decoder to chain them, above it (up to the
/// undiscounted `f = 1`) the envelope stops helping. `0.5` is best or tied-best
/// at every rate measured, and coincides with the reference's space-like value.
const DEFAULT_WEIGHT_FRACTION: f64 = 0.5;

/// One possible loss site projected onto a decode window, before its generator
/// indices are mapped onto decoder hyperedges.
///
/// `local_eid` identifies the error model whose generator list the site
/// addresses. It is `None` for structural pass-through sites, which remain in
/// the graph to preserve probability, herald, and child relationships.
pub(crate) struct RawLossSite {
    pub(crate) local_eid: Option<usize>,
    pub(crate) probability: f64,
    pub(crate) source_generators: Vec<usize>,
    pub(crate) continuation_generators: Vec<usize>,
    pub(crate) children: Vec<usize>,
    pub(crate) heralds: Vec<usize>,
}

impl RawLossSite {
    pub(crate) fn from_compiled(site: CrossGadgetLossSite, local_eid: Option<usize>) -> Self {
        Self {
            local_eid,
            probability: site.probability,
            source_generators: site.source_generators,
            continuation_generators: site.continuation_generators,
            children: site.children,
            heralds: site.heralds,
        }
    }
}

/// Map window-local loss generators onto decoder hyperedge indices.
///
/// Generators absent from `error_reference` have no decoder-visible effect and
/// are omitted. Child and herald indices already address the flat shot-local
/// loss payload and are preserved.
pub(crate) fn build_loss_info(loss_sites: &[RawLossSite], error_reference: &[ErrorIndex]) -> blackbox_decoder::LossInfo {
    let mut edge_of: hashbrown::HashMap<(usize, usize), u64> = hashbrown::HashMap::with_capacity(error_reference.len());
    for (index, error) in error_reference.iter().enumerate() {
        let index = u64::try_from(index).unwrap();
        edge_of.insert((error.eid, error.error_index), index);
    }
    let map_edges = |local_eid: Option<usize>, generators: &[usize]| -> Vec<u64> {
        let Some(local_eid) = local_eid else {
            return vec![];
        };
        generators
            .iter()
            .filter_map(|&generator| edge_of.get(&(local_eid, generator)).copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    };
    let sites = loss_sites
        .iter()
        .map(|site| blackbox_decoder::LossSite {
            source_edges: map_edges(site.local_eid, &site.source_generators),
            continuation_edges: map_edges(site.local_eid, &site.continuation_generators),
            probability: site.probability,
            children: site.children.iter().map(|&child| u64::try_from(child).unwrap()).collect(),
            heralds: site.heralds.iter().map(|&herald| u64::try_from(herald).unwrap()).collect(),
        })
        .collect();
    blackbox_decoder::LossInfo { sites }
}

/// Replace each lost measurement outcome with an independent random bit before
/// syndrome construction.
pub fn apply_loss_random_imputation<R: rand::Rng>(
    outcomes: &mut crate::util::BitVector,
    loss_mask: &crate::util::BitVector,
    rng: &mut R,
) {
    use crate::misc::bit_vector;
    use rand::RngExt;
    assert_eq!(
        outcomes.size, loss_mask.size,
        "loss_mask size {} does not match outcomes size {}",
        loss_mask.size, outcomes.size,
    );
    for index in 0..outcomes.size {
        if bit_vector::get_bit(loss_mask, index) {
            bit_vector::set_bit(outcomes, index, rng.random::<bool>());
        }
    }
}

/// Whether `gtype` carries the static model needed by the loss pipeline.
pub(crate) fn has_loss_model(
    gadget_types: &hashbrown::HashMap<u64, std::sync::Arc<crate::bin::GadgetType>>,
    gtype: u64,
) -> bool {
    gadget_types.get(&gtype).is_some_and(|gadget| gadget.loss_model.is_some())
}

/// How a coordinator handles the atom losses heralded by ``Outcomes.loss_mask``.
///
/// Independent of ``loss_random_imputation``, which decides what the *syndrome*
/// does with a lost measurement bit and applies under every strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(rename_all = "snake_case")]
pub enum LossStrategy {
    /// Discard the loss information. The lost measurements still take part in
    /// the syndrome (imputed when ``loss_random_imputation`` is on), but the
    /// decoder is never told a loss happened. Takes no options.
    Ignore,
    /// Raise each observed loss's Pauli-envelope generator edges from their
    /// (typically zero) prior to a weight comparable to ordinary edges, softly
    /// enforcing the per-atom exclusivity, then hand the decoder an ordinary
    /// problem. The default: it needs no decoder support, and costs nothing on a
    /// program whose gadget types declare no loss model. Options:
    /// ``weight_fraction`` and ``scale``.
    #[default]
    Reweight,
    /// Pass the assembled loss sites to the decoder as a structured
    /// [`LossInfo`](crate::decoder::blackbox_decoder::LossInfo) and let it do
    /// the work — the loss-aware decode path, used by decoders such as the
    /// ``mle_loss_decoder``. Takes no options.
    Handoff,
}

impl LossStrategy {
    fn name(self) -> &'static str {
        match self {
            Self::Ignore => "ignore",
            Self::Reweight => "reweight",
            Self::Handoff => "handoff",
        }
    }
}

/// Options accepted by [`LossStrategy::Ignore`] and [`LossStrategy::Handoff`].
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NoOptions {}

/// How observed losses reweight their Pauli-envelope edges.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvelopeReweightPolicy {
    /// Exponent applied to each observed loss's Pauli-envelope edges,
    /// equivalently the fraction of the edge's own scale weight it is lowered
    /// to. Near ``1`` the loss edges barely help; near ``0`` they become free
    /// and the decoder chains them into logical errors.
    #[serde(default = "default_weight_fraction")]
    pub weight_fraction: f64,
    /// Which scale weight the fraction is taken of.
    #[serde(default)]
    pub scale: ReweightScale,
}

/// Which scale weight [`EnvelopeReweightPolicy`] lowers an activated edge to.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(rename_all = "snake_case")]
pub enum ReweightScale {
    /// The edge's own prior combined with the probability a loss activates it.
    #[default]
    Local,
    /// The mean weight of the graph's regular edges -- the reference
    /// construction, kept for comparison.
    GlobalMean,
    /// The mean weight of the regular edges adjacent to the activated edge: the
    /// reference's semantics without a graph-wide statistic.
    NeighbourhoodMean,
}

impl Default for EnvelopeReweightPolicy {
    fn default() -> Self {
        Self {
            weight_fraction: DEFAULT_WEIGHT_FRACTION,
            scale: ReweightScale::Local,
        }
    }
}

impl EnvelopeReweightPolicy {
    #[must_use]
    pub fn new(weight_fraction: f64) -> Self {
        Self {
            weight_fraction,
            scale: ReweightScale::Local,
        }
    }

    /// Apply the configured fraction to a supplied scale weight.
    #[must_use]
    pub fn scaled_probability(self, scale_weight: f64) -> f64 {
        probability_of_weight(self.weight_fraction * scale_weight.max(0.0))
    }

    /// Reweight against the edge's own prior combined with its activation.
    #[must_use]
    pub fn locally_reweighted_probability(self, edge_probability: f64, activation_probability: f64) -> f64 {
        let scale = exclusive_probability_of(edge_probability, activation_probability);
        probability_of_weight(self.weight_fraction * weight_of(scale).max(0.0))
    }
}

fn loss_reweights(
    loss: &blackbox_decoder::LossInfo,
    hypergraph: &blackbox_decoder::DecodingHypergraph,
    policy: EnvelopeReweightPolicy,
) -> Vec<(u64, f64)> {
    let accumulated = accumulated_site_probabilities(&loss.sites);
    let prior_of = |edge: u64| -> f64 {
        hypergraph
            .hyperedges
            .get(edge as usize)
            .map_or(0.0, |hyperedge| hyperedge.probability)
    };

    let mut order = Vec::new();
    let mut activation_of: HashMap<u64, f64> = HashMap::new();
    let mut activate = |edge: u64, site_probability: f64, order: &mut Vec<u64>| match activation_of.entry(edge) {
        hashbrown::hash_map::Entry::Occupied(mut slot) => {
            let combined = exclusive_probability_of(*slot.get(), site_probability);
            slot.insert(combined);
        }
        hashbrown::hash_map::Entry::Vacant(slot) => {
            slot.insert(site_probability);
            order.push(edge);
        }
    };
    for (index, site) in loss.sites.iter().enumerate() {
        for &edge in &site.source_edges {
            activate(edge, site.probability, &mut order);
        }
        for &edge in &site.continuation_edges {
            activate(edge, accumulated[index], &mut order);
        }
    }

    let neighbourhood = if policy.scale == ReweightScale::NeighbourhoodMean {
        Some(NeighbourhoodScale::new(hypergraph))
    } else {
        None
    };
    let global_mean = if policy.scale == ReweightScale::GlobalMean {
        regular_mean_weight(hypergraph)
    } else {
        0.0
    };

    order
        .into_iter()
        .map(|edge| {
            let activation = activation_of[&edge];
            let probability = match policy.scale {
                ReweightScale::Local => policy.locally_reweighted_probability(prior_of(edge), activation),
                ReweightScale::GlobalMean => policy.scaled_probability(global_mean),
                ReweightScale::NeighbourhoodMean => {
                    match neighbourhood.as_ref().and_then(|scale| scale.mean_weight(edge, hypergraph)) {
                        Some(mean) => policy.scaled_probability(mean),
                        None => policy.locally_reweighted_probability(prior_of(edge), activation),
                    }
                }
            };
            (edge, probability)
        })
        .collect()
}

/// Adjacency index for the optional neighbourhood-mean reweighting policy.
///
/// Each syndrome vertex maps to the decoder edges incident on it. To find a
/// scale for one activated edge, [`Self::mean_weight`] gathers every edge that
/// shares at least one of its syndrome vertices. An edge can be found through
/// several vertices, so the collected indices are deduplicated before their
/// weights are averaged.
struct NeighbourhoodScale {
    /// Decoder-edge indices keyed by incident syndrome vertex.
    edges_of_vertex: HashMap<u64, Vec<u32>>,
}

impl NeighbourhoodScale {
    fn new(hypergraph: &blackbox_decoder::DecodingHypergraph) -> Self {
        let mut edges_of_vertex: HashMap<u64, Vec<u32>> = HashMap::new();
        for (index, hyperedge) in hypergraph.hyperedges.iter().enumerate() {
            if hyperedge.probability == 0.0 {
                continue;
            }
            let index = u32::try_from(index).expect("hyperedge index must fit in u32");
            for &vertex in &hyperedge.vertices {
                edges_of_vertex.entry(vertex).or_default().push(index);
            }
        }
        Self { edges_of_vertex }
    }

    fn mean_weight(&self, edge: u64, hypergraph: &blackbox_decoder::DecodingHypergraph) -> Option<f64> {
        let hyperedge = hypergraph.hyperedges.get(edge as usize)?;
        let mut seen = HashSet::new();
        let mut total = 0.0;
        for vertex in &hyperedge.vertices {
            for &neighbour in self.edges_of_vertex.get(vertex).into_iter().flatten() {
                if seen.insert(neighbour) {
                    total += weight_of(hypergraph.hyperedges[neighbour as usize].probability);
                }
            }
        }
        (!seen.is_empty()).then(|| total / seen.len() as f64)
    }
}

fn regular_mean_weight(hypergraph: &blackbox_decoder::DecodingHypergraph) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for hyperedge in &hypergraph.hyperedges {
        if hyperedge.probability == 0.0 {
            continue;
        }
        total += weight_of(hyperedge.probability);
        count += 1;
    }
    if count == 0 { 0.0 } else { total / count as f64 }
}

fn accumulated_site_probabilities(sites: &[blackbox_decoder::LossSite]) -> Vec<f64> {
    let mut parents: Vec<Vec<usize>> = vec![Vec::new(); sites.len()];
    for (index, site) in sites.iter().enumerate() {
        for &child in &site.children {
            if let Some(slot) = parents.get_mut(child as usize) {
                slot.push(index);
            }
        }
    }
    let mut accumulated = vec![None; sites.len()];
    let mut visiting = vec![false; sites.len()];
    for index in 0..sites.len() {
        accumulate_site(index, sites, &parents, &mut accumulated, &mut visiting);
    }
    accumulated.into_iter().map(|value| value.unwrap_or(0.0)).collect()
}

fn accumulate_site(
    index: usize,
    sites: &[blackbox_decoder::LossSite],
    parents: &[Vec<usize>],
    accumulated: &mut [Option<f64>],
    visiting: &mut [bool],
) -> f64 {
    if let Some(value) = accumulated[index] {
        return value;
    }
    if visiting[index] {
        return 0.0;
    }
    visiting[index] = true;
    let mut total = sites[index].probability;
    for &parent in &parents[index] {
        total = exclusive_probability_of(total, accumulate_site(parent, sites, parents, accumulated, visiting));
    }
    visiting[index] = false;
    accumulated[index] = Some(total);
    total
}

fn default_weight_fraction() -> f64 {
    DEFAULT_WEIGHT_FRACTION
}

/// A validated loss strategy together with its options.
///
/// Build with [`LossHandler::new`] at coordinator construction; every later
/// decision is a total function on this value, so a decode never fails on
/// configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LossHandler {
    Ignore,
    Reweight(EnvelopeReweightPolicy),
    Handoff,
}

pub(crate) struct ProjectedShot {
    pub(crate) reweights: Vec<blackbox_decoder::EdgeReweight>,
    pub(crate) loss: Option<blackbox_decoder::LossInfo>,
    pub(crate) errors: ProjectedErrors,
}

/// Matches [`LossStrategy::default`]: reweight at [`DEFAULT_WEIGHT_FRACTION`].
impl Default for LossHandler {
    fn default() -> Self {
        Self::Reweight(EnvelopeReweightPolicy::default())
    }
}

impl LossHandler {
    /// Parse and validate `config` against `strategy`.
    ///
    /// Returns a human-readable error when the blob carries an option the
    /// chosen strategy does not accept, or a value outside its domain.
    pub fn new(strategy: LossStrategy, config: serde_json::Value) -> Result<Self, String> {
        let config = if config.is_null() {
            serde_json::Value::Object(serde_json::Map::new())
        } else {
            config
        };
        let name = strategy.name();
        match strategy {
            LossStrategy::Ignore | LossStrategy::Handoff => {
                serde_json::from_value::<NoOptions>(config)
                    .map_err(|e| format!("loss_config is not valid for loss_strategy \"{name}\", which takes none: {e}"))?;
                Ok(if strategy == LossStrategy::Ignore {
                    Self::Ignore
                } else {
                    Self::Handoff
                })
            }
            LossStrategy::Reweight => {
                let policy: EnvelopeReweightPolicy = serde_json::from_value(config)
                    .map_err(|e| format!("loss_config is not valid for loss_strategy \"{name}\": {e}"))?;
                if !(policy.weight_fraction > 0.0 && policy.weight_fraction <= 1.0) {
                    return Err(format!(
                        "loss_config.weight_fraction must lie in (0, 1], got {}",
                        policy.weight_fraction
                    ));
                }
                Ok(Self::Reweight(policy))
            }
        }
    }

    /// Whether observed losses have to be recorded per gadget and assembled into
    /// loss sites. False only for [`LossStrategy::Ignore`], which lets the
    /// coordinator skip the whole loss pipeline.
    pub fn tracks_losses(self) -> bool {
        !matches!(self, Self::Ignore)
    }

    /// Whether the assembled sites go to the decoder as a structured `LossInfo`
    /// rather than being flattened into reweighted hyperedges.
    pub fn hands_off_to_decoder(self) -> bool {
        matches!(self, Self::Handoff)
    }

    /// The configured policy when this handler reweights loss envelopes.
    pub fn reweight_policy(self) -> Option<EnvelopeReweightPolicy> {
        match self {
            Self::Reweight(policy) => Some(policy),
            Self::Ignore | Self::Handoff => None,
        }
    }

    /// Apply this strategy to loss sites projected onto a freshly built graph.
    pub(crate) fn apply_sites(
        self,
        mut hypergraph: blackbox_decoder::DecodingHypergraph,
        loss_sites: &[RawLossSite],
        error_reference: &[ErrorIndex],
    ) -> (blackbox_decoder::DecodingHypergraph, Option<blackbox_decoder::LossInfo>) {
        if !self.tracks_losses() || loss_sites.is_empty() {
            return (hypergraph, None);
        }
        let loss = build_loss_info(loss_sites, error_reference);
        if self.hands_off_to_decoder() {
            return (hypergraph, Some(loss));
        }
        let policy = self.reweight_policy().expect("non-ignore, non-handoff handler must reweight");
        for (edge, probability) in loss_reweights(&loss, &hypergraph, policy) {
            if let Some(hyperedge) = hypergraph.hyperedges.get_mut(edge as usize) {
                hyperedge.probability = probability;
            }
        }
        (hypergraph, None)
    }

    /// Project one shot's probability updates and loss onto a loaded graph.
    pub(crate) fn project_shot(
        self,
        projection: &DecodeProjection,
        probability_reweights: &[(u64, f64)],
        loss_sites: &[RawLossSite],
    ) -> ProjectedShot {
        if probability_reweights.is_empty() && loss_sites.is_empty() {
            let (_, errors) = projection.project_reweights(&[]);
            return ProjectedShot {
                reweights: vec![],
                loss: None,
                errors,
            };
        }
        let loss =
            (self.tracks_losses() && !loss_sites.is_empty()).then(|| build_loss_info(loss_sites, &projection.base_errors));
        if self.hands_off_to_decoder() {
            let (reweights, errors) = projection.project_reweights(probability_reweights);
            return ProjectedShot {
                reweights: reweights
                    .into_iter()
                    .map(|(edge, probability)| blackbox_decoder::EdgeReweight { edge, probability })
                    .collect(),
                loss,
                errors,
            };
        }
        let mut combined = probability_reweights.to_vec();
        if let Some(loss) = loss {
            let live_hypergraph;
            let hypergraph = if probability_reweights.is_empty() {
                &projection.base_hypergraph
            } else {
                live_hypergraph = {
                    let mut hypergraph = projection.base_hypergraph.clone();
                    apply_reweights(&mut hypergraph, probability_reweights);
                    hypergraph
                };
                &live_hypergraph
            };
            let loss_reweights = loss_reweights(&loss, hypergraph, self.reweight_policy().unwrap());
            let mut position_of = hashbrown::HashMap::with_capacity(combined.len() + loss_reweights.len());
            for (position, &(edge, _)) in combined.iter().enumerate() {
                position_of.insert(edge, position);
            }
            for (edge, probability) in loss_reweights {
                if let Some(&position) = position_of.get(&edge) {
                    // This is the final loss transform of the user-updated
                    // prior, not a competing assignment. Combining both would
                    // apply the user update twice.
                    combined[position].1 = probability;
                } else {
                    position_of.insert(edge, combined.len());
                    combined.push((edge, probability));
                }
            }
        }
        let (reweights, errors) = projection.project_reweights(&combined);
        ProjectedShot {
            reweights: reweights
                .into_iter()
                .map(|(edge, probability)| blackbox_decoder::EdgeReweight { edge, probability })
                .collect(),
            loss: None,
            errors,
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/loss_handler_test.rs"]
mod tests;
