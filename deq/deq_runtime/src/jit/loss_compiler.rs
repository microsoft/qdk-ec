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
//! Runtime compilation does not decide how losses are decoded. The coordinator
//! either reweights the compiled sites or hands their structure to a loss-aware
//! decoder.

use crate::bin::gadget_type::LossModel;
use crate::misc::bit_vector::get_bit;
use crate::util::BitVector;
use hashbrown::HashMap;
use std::collections::BTreeSet;

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
/// `heralds` are direct window-local herald IDs; equal values identify the same
/// gadget-instance measurement across sites.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CrossGadgetLossSite {
    pub gadget_index: usize,
    pub probability: f64,
    pub source_generators: Vec<usize>,
    pub continuation_generators: Vec<usize>,
    pub children: Vec<usize>,
    pub heralds: Vec<usize>,
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

fn port_offsets(
    ports: &[crate::bin::gadget_type::Port],
    port_types: &HashMap<u64, std::sync::Arc<crate::bin::PortType>>,
) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(ports.len());
    let mut running = 0usize;
    for port in ports {
        offsets.push(running);
        running += port_types
            .get(&port.ptype)
            .map_or(0, |port_type| usize::try_from(port_type.n).unwrap());
    }
    offsets
}

/// Resolve each loss-bearing output slot to its downstream gadget and input
/// slot within the supplied loss-bearing slice. A connector whose upstream
/// instance is absent from the slice is an unresolved boundary link.
pub(crate) fn build_cross_gadget_output_links(
    gadget_instances: &[&crate::bin::Gadget],
    index_of_gid: &HashMap<u64, usize>,
    gadget_types: &HashMap<u64, std::sync::Arc<crate::bin::GadgetType>>,
    port_types: &HashMap<u64, std::sync::Arc<crate::bin::PortType>>,
) -> Vec<HashMap<usize, (usize, usize)>> {
    let mut output_links = vec![HashMap::new(); gadget_instances.len()];
    for (downstream_index, downstream) in gadget_instances.iter().enumerate() {
        let downstream_type = gadget_types.get(&downstream.gtype).unwrap();
        let input_offsets = port_offsets(&downstream_type.inputs, port_types);
        for (input_port, connector) in downstream.connectors.iter().enumerate() {
            let Some(&upstream_index) = index_of_gid.get(&connector.gid) else {
                continue;
            };
            let upstream_type = gadget_types.get(&gadget_instances[upstream_index].gtype).unwrap();
            let output_port = connector.port as usize;
            debug_assert!(output_port < upstream_type.outputs.len());
            let output_offsets = port_offsets(&upstream_type.outputs, port_types);
            let upstream_type = port_types.get(&upstream_type.outputs[output_port].ptype).unwrap();
            let output_offset = output_offsets[output_port];
            let input_offset = input_offsets[input_port];
            for position in 0..(upstream_type.n as usize) {
                output_links[upstream_index].insert(output_offset + position, (downstream_index, input_offset + position));
            }
        }
    }
    output_links
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
            let observed_size = usize::try_from(observed.size).unwrap();
            let mut supported = false;
            let mut contradicted = false;
            for herald in &node.heralds {
                if *herald < observed_size && get_bit(observed, u64::try_from(*herald).unwrap()) {
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
    let mut herald_id_of: HashMap<(usize, usize), usize> = HashMap::new();
    for &node in &kept {
        for &herald in &nodes[node].heralds {
            let next_id = herald_id_of.len();
            herald_id_of.entry((nodes[node].gadget_index, herald)).or_insert(next_id);
        }
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
            heralds: nodes[node]
                .heralds
                .iter()
                .map(|&herald| herald_id_of[&(nodes[node].gadget_index, herald)])
                .collect::<BTreeSet<_>>()
                .into_iter()
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
    indices.iter().map(|&index| usize::try_from(index).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bin::gadget_type::LossModel;
    use crate::bin::gadget_type::loss_model::Loss;
    use std::sync::Arc;

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
    fn output_links_ignore_an_upstream_gadget_outside_the_loss_slice() {
        let downstream = crate::bin::Gadget {
            gid: 2,
            gtype: 2,
            connectors: vec![crate::bin::gadget::Connector { gid: 1, port: 0 }],
            ..Default::default()
        };
        let gadget_types = HashMap::from([(
            2,
            Arc::new(crate::bin::GadgetType {
                gtype: 2,
                inputs: vec![crate::bin::gadget_type::Port {
                    ptype: 1,
                    ..Default::default()
                }],
                ..Default::default()
            }),
        )]);
        let port_types = HashMap::from([(
            1,
            Arc::new(crate::bin::PortType {
                ptype: 1,
                n: 1,
                ..Default::default()
            }),
        )]);
        let index_of_gid = HashMap::from([(2, 0)]);

        let links = build_cross_gadget_output_links(&[&downstream], &index_of_gid, &gadget_types, &port_types);

        assert_eq!(links, vec![HashMap::new()]);
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
        assert_eq!(sites[0].heralds, vec![0]);
        assert_eq!(sites[1].source_generators, vec![2]);
        assert_eq!(sites[1].children, Vec::<usize>::new());
        assert_eq!(sites[1].heralds, vec![1]);
    }

    #[test]
    fn herald_ids_preserve_equality_without_cross_gadget_collisions() {
        let g0 = LossModel {
            losses: vec![loss(&[0], &[], &[0], &[]), loss(&[0], &[], &[1], &[])],
            ..LossModel::default()
        };
        let g1 = LossModel {
            losses: vec![loss(&[0], &[], &[2], &[])],
            ..LossModel::default()
        };
        let obs0 = observed(&[0]);
        let obs1 = observed(&[0]);
        let links0 = output_links(&[]);
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

        assert_eq!(sites.len(), 3);
        assert_eq!(sites[0].heralds, sites[1].heralds);
        assert_ne!(sites[0].heralds, sites[2].heralds);
    }

    #[test]
    fn herald_ids_are_deduplicated_and_ordered() {
        let model = LossModel {
            losses: vec![loss(&[2], &[], &[0], &[]), loss(&[1, 2, 1], &[], &[1], &[])],
            ..LossModel::default()
        };
        let obs = observed(&[1, 2]);
        let links = output_links(&[]);

        let sites = build_cross_gadget_loss_sites(&[GadgetLoss {
            loss_model: &model,
            observed: &obs,
            output_links: &links,
        }]);

        assert_eq!(sites[1].heralds, vec![0, 1]);
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
