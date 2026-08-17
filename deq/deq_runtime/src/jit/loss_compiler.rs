//! Runtime compiler for loss sites.
//!
//! A gadget type's [`LossModel`] describes where losses can originate or enter,
//! how they propagate, which measurements herald them, and which Pauli-envelope
//! generators they activate. That type-level description is not enough to select
//! sites for a shot: cross-gadget propagation depends on the connectivity of the
//! gadget instances in the current decode window, and feasibility depends on the
//! observed loss masks. [`build_cross_gadget_loss_sites`] combines all three into
//! the loss sites consistent with that runtime context.
//!
//! This mirrors JIT compilation: the static loss models are instantiated over a
//! runtime gadget graph, linked across connected ports, evaluated against the
//! shot's observations, and emitted as a window-local site graph. The result is
//! strategy-neutral. A coordinator may project it onto hyperedges and reweight
//! them, or hand the same structured sites to a loss-aware decoder.
//!
//! # Pauli envelope as a linear span
//!
//! A single atom loss has a *Pauli envelope* `E` — a set of Pauli error
//! configurations that bounds the loss's effect on detectors and observables.
//! The envelope is *linear*: it is the GF(2) span of one generator per relevant
//! space-time location (the loss site, after each Hadamard on the lost atom, and
//! just before its measurement). The forward loss transpiler has already
//! projected those generators into each gadget's shared error list and recorded
//! them on the loss model as `source_errors` (active only if the loss starts
//! there) and `continuation_errors` (active whenever a loss passes through).
//!
//! # Runtime compilation
//!
//! One physical atom lost early and measured (replenished) later spans several
//! gadgets: its generators live in the gadget where each Pauli acts, but its
//! herald — the loss-resolving measurement that reveals it — may only appear in a
//! downstream gadget. The compiler first instantiates every gadget's fresh and
//! input loss nodes, then links `child_losses` within each instance and connects
//! `child_output_qubits` to downstream `input_losses` using the resolved instance
//! connectivity. Finally, it folds the loss-mask evidence backward through that
//! graph. A site is *possible* when some herald in its forward reach was observed
//! as a loss and none was observed as a non-loss, so a downstream herald keeps
//! the upstream generators.
//!
//! Connections leaving the decode window are deliberately unresolved: the chain
//! ends at that boundary and can be compiled again by a later window that contains
//! its continuation.
//!
//! # Downstream strategies
//!
//! Runtime compilation does not decide how losses are decoded. The coordinator's
//! reweight strategy maps each compiled site's generators to hyperedges and uses
//! [`EnvelopeReweightPolicy`] to adjust their weights. The handoff strategy
//! preserves the compiled site graph, including parent-child relationships, for a
//! loss-aware decoder. The policy types later in this module support the first
//! strategy but are not part of site compilation.

use crate::bin::gadget_type::LossModel;
use crate::misc::bit_vector::get_bit;
use crate::misc::util::{exclusive_probability_of, probability_of_weight, weight_of};
use crate::util::BitVector;
use hashbrown::HashMap;

/// One instantiated gadget's input to the runtime loss-site compiler.
///
/// `loss_model` is the static type-level description. `observed` is this
/// instance's shot loss mask, indexed by local measurement index (empty when the
/// gadget recorded no loss this shot); an index at or beyond its size counts as
/// not lost. `output_links` is the resolved runtime connectivity: it maps each
/// output flat slot on which a loss can leave the instance to the `(downstream
/// gadget index, input flat slot)` where it continues. A slot with no entry
/// leaves the decode region or connects to an instance with no loss model, so the
/// compiled chain ends there.
pub struct GadgetLoss<'a> {
    pub loss_model: &'a LossModel,
    pub observed: &'a BitVector,
    pub output_links: &'a HashMap<usize, (usize, usize)>,
}

/// One possible loss site compiled for the current decode region and shot.
///
/// Generators index the error list of the gadget identified by `gadget_index`
/// in the [`build_cross_gadget_loss_sites`] input slice, so a downstream strategy
/// can map them onto that gadget's hyperedges. `children` are positions in the
/// returned site list (forward parent -> child links, possibly crossing gadgets).
/// `probability` is the declared `LOSS_ERROR` probability of the loss starting
/// here, or `0` for a continuation entered from a parent in another gadget.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CrossGadgetLossSite {
    pub gadget_index: usize,
    pub probability: f64,
    pub source_generators: Vec<usize>,
    pub continuation_generators: Vec<usize>,
    pub children: Vec<usize>,
}

struct CrossGadgetNode {
    gadget_index: usize,
    probability: f64,
    source: Vec<usize>,
    continuation: Vec<usize>,
    heralds: Vec<usize>,
    children: Vec<usize>,
}

/// Intermediate representation of the instantiated loss graph. The per-gadget
/// lookups wire its forward links: `fresh_ids[gadget][local_index]` and
/// `input_ids[gadget][input_slot]` give the node ID of a fresh or entering loss.
struct LossNodeGraph {
    nodes: Vec<CrossGadgetNode>,
    fresh_ids: Vec<Vec<usize>>,
    input_ids: Vec<Vec<usize>>,
}

/// Compile the possible loss sites for a connected set of gadget instances.
///
/// Each input supplies a static gadget loss model, that instance's observed loss
/// mask, and its resolved output links within the decode region. Compilation
/// instantiates the models, follows each `child_output_qubits` entry into the
/// connected instance's `input_losses`, and filters the resulting graph using the
/// observed heralds. A node is possible when some herald in its forward reach was
/// observed as a loss and none was observed as a non-loss, so a downstream herald
/// keeps the upstream generators.
///
/// Only sites consistent with the observations are returned. They remain
/// ungrouped and retain their child links so either coordinator-side reweighting
/// or decoder handoff can consume the same result.
///
/// A single gadget with no output links degenerates to a within-gadget analysis:
/// each possible loss site with its `child_losses` links preserved.
///
/// # Panics
///
/// Panics if a protobuf loss-model index does not fit in `usize`.
#[must_use]
pub fn build_cross_gadget_loss_sites(gadgets: &[GadgetLoss]) -> Vec<CrossGadgetLossSite> {
    let LossNodeGraph {
        mut nodes,
        fresh_ids,
        input_ids,
    } = build_loss_nodes(gadgets);

    // Link the instantiated graph: within-gadget `child_losses` and cross-gadget
    // `child_output_qubits` that reach a connected gadget's input.
    for (gadget_index, gadget) in gadgets.iter().enumerate() {
        for (local_index, loss) in gadget.loss_model.losses.iter().enumerate() {
            let node = fresh_ids[gadget_index][local_index];
            resolve_children(
                node,
                gadget_index,
                proto_indices(&loss.child_losses),
                proto_indices(&loss.child_output_qubits),
                gadget.output_links,
                &fresh_ids,
                &input_ids,
                &mut nodes,
            );
        }
        for (&node, input_loss) in input_ids[gadget_index].iter().zip(&gadget.loss_model.input_losses) {
            resolve_children(
                node,
                gadget_index,
                proto_indices(&input_loss.child_losses),
                proto_indices(&input_loss.child_output_qubits),
                gadget.output_links,
                &fresh_ids,
                &input_ids,
                &mut nodes,
            );
        }
    }

    emit_possible_sites(&nodes, gadgets)
}

/// Instantiate a node for every fresh loss and input loss, together with the
/// per-gadget lookups that link the runtime graph.
fn build_loss_nodes(gadgets: &[GadgetLoss]) -> LossNodeGraph {
    let mut nodes: Vec<CrossGadgetNode> = Vec::new();
    let mut fresh_ids: Vec<Vec<usize>> = Vec::with_capacity(gadgets.len());
    let mut input_ids: Vec<Vec<usize>> = Vec::with_capacity(gadgets.len());
    for (gadget_index, gadget) in gadgets.iter().enumerate() {
        let mut fresh = Vec::with_capacity(gadget.loss_model.losses.len());
        for loss in &gadget.loss_model.losses {
            fresh.push(nodes.len());
            nodes.push(CrossGadgetNode {
                gadget_index,
                probability: loss.probability,
                source: proto_indices(&loss.source_errors).collect(),
                continuation: proto_indices(&loss.continuation_errors).collect(),
                heralds: proto_indices(&loss.loss_measurements).collect(),
                children: Vec::new(),
            });
        }
        let mut inputs = Vec::with_capacity(gadget.loss_model.input_losses.len());
        for input_loss in &gadget.loss_model.input_losses {
            inputs.push(nodes.len());
            nodes.push(CrossGadgetNode {
                gadget_index,
                probability: 0.0,
                source: Vec::new(),
                continuation: proto_indices(&input_loss.continuation_errors).collect(),
                heralds: proto_indices(&input_loss.loss_measurements).collect(),
                children: Vec::new(),
            });
        }
        fresh_ids.push(fresh);
        input_ids.push(inputs);
    }
    LossNodeGraph {
        nodes,
        fresh_ids,
        input_ids,
    }
}

#[allow(clippy::too_many_arguments)]
fn resolve_children(
    node: usize,
    gadget_index: usize,
    child_losses: impl IntoIterator<Item = usize>,
    child_output_qubits: impl IntoIterator<Item = usize>,
    output_links: &HashMap<usize, (usize, usize)>,
    fresh_ids: &[Vec<usize>],
    input_ids: &[Vec<usize>],
    nodes: &mut [CrossGadgetNode],
) {
    for child in child_losses {
        if let Some(&child_node) = fresh_ids[gadget_index].get(child) {
            nodes[node].children.push(child_node);
        }
    }
    for slot in child_output_qubits {
        let Some(&(downstream, input_slot)) = output_links.get(&slot) else {
            continue;
        };
        nodes[node].children.push(input_ids[downstream][input_slot]);
    }
}

/// Evaluate the instantiated graph against the shot's loss masks and emit sites
/// supported by an observed herald in their forward reach and contradicted by
/// none, with `children` remapped to positions in the returned list.
fn emit_possible_sites(nodes: &[CrossGadgetNode], gadgets: &[GadgetLoss]) -> Vec<CrossGadgetLossSite> {
    let direct: Vec<(bool, bool)> = nodes
        .iter()
        .map(|node| {
            let observed = gadgets[node.gadget_index].observed;
            let observed_size = usize::try_from(observed.size).expect("loss-mask size must fit in usize");
            let mut supported = false;
            let mut contradicted = false;
            for herald in &node.heralds {
                if *herald < observed_size
                    && get_bit(observed, u64::try_from(*herald).expect("herald index originated as u64"))
                {
                    supported = true;
                } else {
                    contradicted = true;
                }
            }
            (supported, contradicted)
        })
        .collect();
    let mut memo: Vec<Option<(bool, bool)>> = vec![None; nodes.len()];
    let possible: Vec<bool> = (0..nodes.len())
        .map(|node| {
            let (supported, contradicted) = fold_evidence(node, nodes, &direct, &mut memo);
            supported && !contradicted
        })
        .collect();

    let mut emitted_position: Vec<Option<usize>> = vec![None; nodes.len()];
    let kept: Vec<usize> = (0..nodes.len()).filter(|&node| possible[node]).collect();
    for (position, &node) in kept.iter().enumerate() {
        emitted_position[node] = Some(position);
    }
    kept.iter()
        .map(|&node| CrossGadgetLossSite {
            gadget_index: nodes[node].gadget_index,
            probability: nodes[node].probability,
            source_generators: nodes[node].source.clone(),
            continuation_generators: nodes[node].continuation.clone(),
            children: nodes[node]
                .children
                .iter()
                .filter_map(|&child| emitted_position[child])
                .collect(),
        })
        .collect()
}

fn fold_evidence(
    node: usize,
    nodes: &[CrossGadgetNode],
    direct: &[(bool, bool)],
    memo: &mut [Option<(bool, bool)>],
) -> (bool, bool) {
    if let Some(value) = memo[node] {
        return value;
    }
    // Record the direct evidence provisionally before recursing so a cycle (which
    // a well-formed forward loss graph never contains) cannot loop forever.
    memo[node] = Some(direct[node]);
    let (mut supported, mut contradicted) = direct[node];
    for &child in &nodes[node].children {
        let (child_supported, child_contradicted) = fold_evidence(child, nodes, direct, memo);
        supported |= child_supported;
        contradicted |= child_contradicted;
    }
    memo[node] = Some((supported, contradicted));
    (supported, contradicted)
}

fn proto_indices(indices: &[u64]) -> impl Iterator<Item = usize> + '_ {
    indices
        .iter()
        .map(|&index| usize::try_from(index).expect("loss-model index must fit in usize"))
}

/// The envelope-matching reweighting policy.
///
/// An edge a loss envelope can flip is lowered to `weight_fraction` of the weight
/// of `p_e (+) p_activation`, where `p_e` is its own prior and `p_activation` the
/// total probability that some observed loss activates it:
/// `w <- weight_fraction * w_scale`, with `w = -ln(p/(1-p))`. Taking the fraction
/// in weight space rather than exponentiating the probability is what keeps every
/// reweighted edge at non-negative weight, and applying it once per edge — over
/// the accumulated activation probability — rather than once per activating site
/// is what keeps the fraction a guarantee about the edge; see
/// [`Self::locally_reweighted_probability`].
///
/// Combining with `p_e` first also means a high-prior edge that merely happens to
/// lie in an envelope is not dragged down to the loss scale, and leaves a
/// loss-only edge — prior `0`, i.e. infinite weight until a loss activates it —
/// at `weight_fraction` of the site's own weight.
///
/// `weight_fraction` is the single knob that trades off the two failure modes of
/// loss reweighting:
///
/// * near `1.0` the loss edges keep ~full weight, so a heralded loss is barely
///   cheaper to explain than an ordinary error and the envelope hardly helps;
/// * near `0.0` the loss edges become free, so the decoder can chain several
///   edges of *one* loss into a zero-cost logical path — no exclusivity — which
///   was measured to be worse than plain random imputation beyond `d = 3`.
///
/// The soft per-atom exclusivity lives in between, and is measurably a bowl: on a
/// `d = 3` mid-swap memory at `p_loss` of `1-2%` the logical error rate falls
/// from `f = 0.05` to a broad optimum at `0.5-0.7` and rises again toward `1.0`,
/// a `5-6 sigma` effect. At low loss rates it is flat in `f`, since most edges are
/// then activated by a single site. `0.5` — the default, and the original paper's
/// space-like value — is best or tied-best at every rate measured; the original paper
/// additionally lowers *time-like* edges (a measurement error on the same ancilla
/// across rounds, which cannot advance a logical operator spatially) to `0.25`,
/// since making those cheaper is "safe". This single-fraction policy cannot
/// express that split — hyperedges carry no spatial or temporal geometry — so it
/// applies one fraction throughout, and the optimum for a given code and noise is
/// worth sweeping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnvelopeReweightPolicy {
    /// Fraction of the scale weight an activated edge is lowered to. Must lie in
    /// `(0, 1]`; `0.5` is the reference's space-like value and this crate's
    /// default.
    pub weight_fraction: f64,
    /// Where the scale weight comes from. [`ReweightScale::Local`] is the default
    /// and the rule described above; [`ReweightScale::GlobalMean`] reproduces the
    /// reference construction for comparison.
    pub scale: ReweightScale,
}

/// Which scale weight [`EnvelopeReweightPolicy`] lowers an activated edge to.
///
/// Kept selectable so the two constructions can be measured against each other
/// on the same graphs, decoders and shots.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReweightScale {
    /// `w_scale = w(p_e (+) p_activation)`: the edge's own prior combined with
    /// the total probability that an observed loss activates it.
    #[default]
    Local,
    /// `w_scale = mean weight of the graph's regular edges`, the reference
    /// construction. Every activated edge is assigned the same weight, whatever
    /// its own prior and however likely the loss was to reach it.
    GlobalMean,
    /// `w_scale = mean weight of the regular edges sharing a vertex with this
    /// one`. Keeps the reference's semantics -- a heralded loss costs a fixed
    /// fraction of an *ordinary error* -- while reading that scale from the
    /// edge's own neighbourhood, so no graph-wide statistic is needed and a
    /// heterogeneous graph is tracked locally. Falls back to [`Self::Local`] for
    /// an edge with no regular neighbour, which is what makes it total on a
    /// graph that has no regular edges at all.
    NeighbourhoodMean,
}

impl Default for EnvelopeReweightPolicy {
    fn default() -> Self {
        Self {
            weight_fraction: 0.5,
            scale: ReweightScale::Local,
        }
    }
}

impl EnvelopeReweightPolicy {
    /// A policy with the given fraction.
    #[must_use]
    pub fn new(weight_fraction: f64) -> Self {
        Self {
            weight_fraction,
            scale: ReweightScale::Local,
        }
    }

    /// The reference's assignment for an activated edge, given a scale weight
    /// read off the graph's regular edges (globally or in a neighbourhood):
    /// `w <- weight_fraction * scale_weight`.
    #[must_use]
    pub fn scaled_probability(self, scale_weight: f64) -> f64 {
        probability_of_weight(self.weight_fraction * scale_weight.max(0.0))
    }

    /// An activated edge's locally reweighted probability: the scale
    /// `p_scale = p_e (+) p_activation` lowered to `weight_fraction` of its
    /// *weight*, `w <- weight_fraction * w(p_scale)`. This uses the edge's own
    /// prior and is selected by [`ReweightScale::Local`], or as the fallback for
    /// [`ReweightScale::NeighbourhoodMean`] when no regular neighbour exists.
    ///
    /// `edge_probability` is the edge's own prior and `activation_probability`
    /// the total probability that some observed loss activates it — the union
    /// over every site that lists the edge, accumulated by the caller *before*
    /// this is applied. Applying the fraction once per edge rather than once per
    /// site is what keeps the guarantee at edge granularity; see
    /// [`loss_reweights`](crate::decoder::blackbox_util::loss_reweights).
    #[must_use]
    pub fn locally_reweighted_probability(self, edge_probability: f64, activation_probability: f64) -> f64 {
        let scale = exclusive_probability_of(edge_probability, activation_probability);
        probability_of_weight(self.weight_fraction * weight_of(scale).max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bin::gadget_type::LossModel;
    use crate::bin::gadget_type::loss_model::Loss;

    fn loss(loss_measurements: &[u64], child_losses: &[u64], source_errors: &[u64], continuation_errors: &[u64]) -> Loss {
        Loss {
            probability: 0.1,
            loss_measurements: loss_measurements.to_vec(),
            child_losses: child_losses.to_vec(),
            source_errors: source_errors.to_vec(),
            continuation_errors: continuation_errors.to_vec(),
            ..Loss::default()
        }
    }

    fn observed(indices: &[u64]) -> BitVector {
        let size = indices.iter().copied().max().map_or(0, |max| max + 1);
        crate::misc::bit_vector::from_sparse_indices(size, indices)
    }

    #[test]
    fn reweight_policy_lowers_edge_weight_to_the_fraction() {
        // The rule is an exact fraction in MWPM weight space,
        // `w_target = fraction * w_scale`, which is where the "costs a fixed
        // fraction of an ordinary edge" reading comes from. A loss-only edge
        // (prior weight infinite) is raised to a usable value.
        let site = 0.001f64;
        let half = EnvelopeReweightPolicy::default().locally_reweighted_probability(0.0, site);
        let quarter = EnvelopeReweightPolicy::new(0.25).locally_reweighted_probability(0.0, site);
        assert!((weight_of(half) - 0.5 * weight_of(site)).abs() < 1e-12);
        assert!((weight_of(quarter) - 0.25 * weight_of(site)).abs() < 1e-12);
        assert!(half > 0.0 && quarter > half, "smaller fraction -> cheaper edge");
        assert!(
            weight_of(half) < weight_of(site),
            "activated edge is cheaper than the loss itself"
        );
    }

    /// A probability above `1/2` is a negative weight, which would tell the
    /// decoder the mechanism is cheaper than free. Exponentiating the
    /// probability crossed that line at `p_scale > (1/2)^(1/fraction)` — only
    /// `6.25%` at `fraction = 0.25` — so the fraction is taken in weight space
    /// instead, which cannot.
    #[test]
    fn reweight_policy_never_reaches_negative_weight() {
        for &fraction in &[0.05, 0.25, 0.5, 0.9, 1.0] {
            let policy = EnvelopeReweightPolicy::new(fraction);
            for &prior in &[0.0, 1e-9, 0.01, 0.1, 0.3, 0.5] {
                for &site in &[0.0, 1e-9, 0.01, 0.1, 0.3, 0.5] {
                    let contribution = policy.locally_reweighted_probability(prior, site);
                    assert!(
                        contribution <= 0.5 + 1e-12,
                        "fraction={fraction} prior={prior} site={site} gave {contribution} > 1/2"
                    );
                    assert!(weight_of(contribution) >= -1e-12, "negative weight at fraction={fraction}");
                }
            }
        }
    }

    #[test]
    fn reweight_policy_never_lowers_a_high_prior_edge() {
        // A high-prior edge that merely lies in an envelope keeps its own scale
        // rather than being pinned to the loss rate.
        let policy = EnvelopeReweightPolicy::default();
        let prior = 0.2;
        assert!(policy.locally_reweighted_probability(prior, 1e-6) > prior);
    }

    fn loss_out(
        loss_measurements: &[u64],
        child_losses: &[u64],
        source_errors: &[u64],
        continuation_errors: &[u64],
        child_output_qubits: &[u64],
    ) -> Loss {
        Loss {
            child_output_qubits: child_output_qubits.to_vec(),
            ..loss(loss_measurements, child_losses, source_errors, continuation_errors)
        }
    }

    fn input_loss(
        loss_measurements: &[u64],
        child_losses: &[u64],
        continuation_errors: &[u64],
        child_output_qubits: &[u64],
    ) -> crate::bin::gadget_type::loss_model::InputLoss {
        crate::bin::gadget_type::loss_model::InputLoss {
            loss_measurements: loss_measurements.to_vec(),
            child_losses: child_losses.to_vec(),
            continuation_errors: continuation_errors.to_vec(),
            child_output_qubits: child_output_qubits.to_vec(),
        }
    }

    fn output_links(pairs: &[(usize, (usize, usize))]) -> HashMap<usize, (usize, usize)> {
        pairs.iter().copied().collect()
    }

    #[test]
    fn single_gadget_chain_keeps_possible_sites_with_children() {
        // One gadget, no output links: a heralded parent -> child chain. Both
        // sites are possible; the within-gadget forward link is preserved
        // (remapped to a position in the returned list).
        let model = LossModel {
            losses: vec![loss(&[0], &[1], &[0], &[1]), loss(&[1], &[], &[2], &[3])],
            ..LossModel::default()
        };
        let obs = observed(&[0, 1]);
        let links = output_links(&[]);
        let sites = build_cross_gadget_loss_sites(&[GadgetLoss {
            loss_model: &model,
            observed: &obs,
            output_links: &links,
        }]);
        assert_eq!(sites.len(), 2);
        assert_eq!(sites[0].source_generators, vec![0]);
        assert_eq!(sites[0].continuation_generators, vec![1]);
        assert_eq!(sites[0].children, vec![1]);
        assert_eq!(sites[1].source_generators, vec![2]);
        assert_eq!(sites[1].children, Vec::<usize>::new());
    }

    #[test]
    fn single_gadget_contradicted_parent_drops_the_parent() {
        // Same chain, but only herald 1 is observed: the parent (herald 0 not a
        // loss) is contradicted and dropped; only the child survives.
        let model = LossModel {
            losses: vec![loss(&[0], &[1], &[0], &[1]), loss(&[1], &[], &[2], &[3])],
            ..LossModel::default()
        };
        let obs = observed(&[1]);
        let links = output_links(&[]);
        let sites = build_cross_gadget_loss_sites(&[GadgetLoss {
            loss_model: &model,
            observed: &obs,
            output_links: &links,
        }]);
        assert_eq!(sites.len(), 1);
        assert_eq!(sites[0].source_generators, vec![2]);
        assert_eq!(sites[0].children, Vec::<usize>::new());
    }

    #[test]
    fn cross_gadget_downstream_herald_supports_upstream_generators() {
        // The loss starts in gadget 0 (generators 0, 1) with no local herald and
        // leaves on output slot 0; it enters gadget 1 where it is finally heralded
        // by measurement 9. Observing 9 must keep BOTH the upstream (gadget 0) and
        // the downstream (gadget 1) generators.
        let g0 = LossModel {
            losses: vec![loss_out(&[], &[], &[0], &[1], &[0])],
            ..LossModel::default()
        };
        let g1 = LossModel {
            input_losses: vec![input_loss(&[9], &[], &[2], &[])],
            ..LossModel::default()
        };
        let obs0 = observed(&[]);
        let obs1 = observed(&[9]);
        let links0 = output_links(&[(0, (1, 0))]);
        let links1 = output_links(&[]);
        let sites = build_cross_gadget_loss_sites(&[
            GadgetLoss {
                loss_model: &g0,
                observed: &obs0,
                output_links: &links0,
            },
            GadgetLoss {
                loss_model: &g1,
                observed: &obs1,
                output_links: &links1,
            },
        ]);
        assert_eq!(sites.len(), 2);
        let upstream = sites.iter().find(|s| s.gadget_index == 0).unwrap();
        assert_eq!(upstream.source_generators, vec![0]);
        assert_eq!(upstream.continuation_generators, vec![1]);
        let downstream = sites.iter().find(|s| s.gadget_index == 1).unwrap();
        assert_eq!(downstream.continuation_generators, vec![2]);
        assert!(downstream.source_generators.is_empty());
    }

    #[test]
    fn cross_gadget_unobserved_herald_yields_no_sites() {
        // Same chain, but the downstream herald is not observed: the whole chain
        // is unsupported, so the runtime compiler emits no sites.
        let g0 = LossModel {
            losses: vec![loss_out(&[], &[], &[0], &[1], &[0])],
            ..LossModel::default()
        };
        let g1 = LossModel {
            input_losses: vec![input_loss(&[9], &[], &[2], &[])],
            ..LossModel::default()
        };
        let obs = observed(&[]);
        let links0 = output_links(&[(0, (1, 0))]);
        let links1 = output_links(&[]);
        let sites = build_cross_gadget_loss_sites(&[
            GadgetLoss {
                loss_model: &g0,
                observed: &obs,
                output_links: &links0,
            },
            GadgetLoss {
                loss_model: &g1,
                observed: &obs,
                output_links: &links1,
            },
        ]);
        assert!(sites.is_empty());
    }

    #[test]
    fn cross_gadget_empty_input_loss_is_filtered() {
        let g0 = LossModel {
            losses: vec![loss_out(&[0], &[], &[0], &[], &[0])],
            ..LossModel::default()
        };
        let g1 = LossModel {
            input_losses: vec![input_loss(&[], &[], &[], &[])],
            ..LossModel::default()
        };
        let obs0 = observed(&[0]);
        let obs1 = observed(&[]);
        let links0 = output_links(&[(0, (1, 0))]);
        let links1 = output_links(&[]);
        let sites = build_cross_gadget_loss_sites(&[
            GadgetLoss {
                loss_model: &g0,
                observed: &obs0,
                output_links: &links0,
            },
            GadgetLoss {
                loss_model: &g1,
                observed: &obs1,
                output_links: &links1,
            },
        ]);
        assert_eq!(sites.len(), 1);
        assert_eq!(sites[0].source_generators, vec![0]);
        assert!(sites[0].children.is_empty());
    }

    #[test]
    fn cross_gadget_contradicted_downstream_prunes_upstream() {
        // The downstream node carries a second herald (7) that is observed as a
        // non-loss, contradicting the chain even though herald 9 is observed. The
        // whole chain is then impossible.
        let g0 = LossModel {
            losses: vec![loss_out(&[], &[], &[0], &[1], &[0])],
            ..LossModel::default()
        };
        let g1 = LossModel {
            input_losses: vec![input_loss(&[7, 9], &[], &[2], &[])],
            ..LossModel::default()
        };
        let obs0 = observed(&[]);
        let obs1 = observed(&[9]);
        let links0 = output_links(&[(0, (1, 0))]);
        let links1 = output_links(&[]);
        let sites = build_cross_gadget_loss_sites(&[
            GadgetLoss {
                loss_model: &g0,
                observed: &obs0,
                output_links: &links0,
            },
            GadgetLoss {
                loss_model: &g1,
                observed: &obs1,
                output_links: &links1,
            },
        ]);
        assert!(sites.is_empty());
    }

    #[test]
    fn cross_gadget_dangling_output_link_ends_the_chain() {
        // The loss leaves gadget 0 on a slot with no downstream link (the port
        // exits the decode region). With no herald anywhere, it is unsupported.
        let g0 = LossModel {
            losses: vec![loss_out(&[], &[], &[0], &[1], &[0])],
            ..LossModel::default()
        };
        let obs = observed(&[]);
        let links = output_links(&[]);
        let sites = build_cross_gadget_loss_sites(&[GadgetLoss {
            loss_model: &g0,
            observed: &obs,
            output_links: &links,
        }]);
        assert!(sites.is_empty());
    }
}
