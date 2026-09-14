//! Forced-gap queries share a graph across shots and memoize scores within a shot.

use crate::coordinator::reweight_handler::apply_reweights;
use crate::decoder::DynDecoder;
use crate::decoder::blackbox_decoder::{
    DecodingHypergraph, DecodingProblem, EdgeReweight, LoadedDecodingProblem, ParityFactor,
};
use crate::decoder::blackbox_util::is_parity_factor;
use crate::misc::bit_vector::{extend_num_bits, get_bit, set_bit};
use crate::misc::util::{probability_of_weight, weight_of};
use crate::util::BitVector;
use binar::{BitMatrix, BitVec, EchelonForm};
use futures_util::future::try_join_all;
use hashbrown::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};
use tokio::sync::OnceCell;
use tonic::Status;

pub(crate) fn forced_hypergraph(
    hypergraph: &DecodingHypergraph,
    logical_actions: &[Vec<u64>],
    target: usize,
) -> DecodingHypergraph {
    debug_assert_eq!(hypergraph.hyperedges.len(), logical_actions.len());
    let mut forced = hypergraph.clone();
    let forced_vertex = forced.vertex_num;
    forced.vertex_num += 1;
    let target = u64::try_from(target).unwrap();
    for (hyperedge, action) in forced.hyperedges.iter_mut().zip(logical_actions) {
        if action.contains(&target) {
            hyperedge.vertices.push(forced_vertex);
        }
    }
    forced
}

pub(crate) async fn load_forced_hypergraphs(
    decoder: &DynDecoder,
    hypergraph: &DecodingHypergraph,
    logical_actions: &[Vec<u64>],
    target_count: usize,
) -> Result<Vec<Option<u64>>, Status> {
    try_join_all((0..target_count).map(|target| load_forced_hypergraph(decoder, hypergraph, logical_actions, target))).await
}

pub(crate) async fn load_forced_hypergraph(
    decoder: &DynDecoder,
    hypergraph: &DecodingHypergraph,
    logical_actions: &[Vec<u64>],
    target: usize,
) -> Result<Option<u64>, Status> {
    if !logical_actions
        .iter()
        .any(|action| action.contains(&u64::try_from(target).unwrap()))
    {
        return Ok(None);
    }
    decoder
        .load_hypergraph(forced_hypergraph(hypergraph, logical_actions, target))
        .await
        .map(|response| Some(response.hid))
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn forced_gap_probabilities(
    decoder: &DynDecoder,
    hypergraph: &DecodingHypergraph,
    logical_actions: &[Vec<u64>],
    forced_hids: Option<&[Option<u64>]>,
    syndrome: &BitVector,
    baseline: &ParityFactor,
    reweights: &[EdgeReweight],
    loss: Option<&LossInfo>,
    use_loaded_reweights: bool,
    target_count: usize,
) -> Result<Vec<f64>, Status> {
    try_join_all((0..target_count).map(|target| {
        let hid = forced_hids.and_then(|hids| hids.get(target)).copied().flatten();
        forced_gap_probability(
            decoder,
            hypergraph,
            logical_actions,
            hid,
            syndrome,
            baseline,
            reweights,
            loss,
            use_loaded_reweights,
            target,
        )
    }))
    .await
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn forced_gap_probability(
    decoder: &DynDecoder,
    hypergraph: &DecodingHypergraph,
    logical_actions: &[Vec<u64>],
    forced_hid: Option<u64>,
    syndrome: &BitVector,
    baseline: &ParityFactor,
    reweights: &[EdgeReweight],
    loss: Option<&LossInfo>,
    use_loaded_reweights: bool,
    target_index: usize,
) -> Result<f64, Status> {
    debug_assert_eq!(hypergraph.hyperedges.len(), logical_actions.len());
    debug_assert_eq!(syndrome.size, hypergraph.vertex_num);
    let target = u64::try_from(target_index).unwrap();
    if !logical_actions.iter().any(|action| action.contains(&target)) {
        return Ok(0.0);
    }

    let baseline_bit = logical_bit(logical_actions, baseline, target);
    let forced_bit = !baseline_bit;
    let mut forced_syndrome = syndrome.clone();
    let forced_vertex = forced_syndrome.size;
    extend_num_bits(&mut forced_syndrome, 1);
    set_bit(&mut forced_syndrome, forced_vertex, forced_bit);
    let result = if let Some(hid) = forced_hid
        && (reweights.is_empty() || use_loaded_reweights)
    {
        decoder
            .decode_loaded(LoadedDecodingProblem {
                hid,
                syndrome: Some(forced_syndrome),
                reweights: reweights.to_vec(),
                loss: loss.cloned(),
            })
            .await
    } else {
        let mut forced_graph = forced_hypergraph(hypergraph, logical_actions, target_index);
        apply_reweights(
            &mut forced_graph,
            reweights.iter().map(|reweight| (reweight.edge, reweight.probability)),
        );
        decoder
            .decode(DecodingProblem {
                hypergraph: Some(forced_graph),
                syndrome: Some(forced_syndrome),
                loss: loss.cloned(),
            })
            .await
    };

    match result {
        Ok(candidate)
            if is_parity_factor(hypergraph, &candidate, syndrome)
                && logical_bit(logical_actions, &candidate, target) == forced_bit =>
        {
            Ok(candidate_probability(hypergraph, baseline, &candidate, reweights))
        }
        Ok(_) => Ok(0.0),
        Err(error) => Err(error),
    }
}

fn logical_bit(logical_actions: &[Vec<u64>], candidate: &ParityFactor, target: u64) -> bool {
    candidate
        .subgraph
        .iter()
        .filter(|&&edge| logical_actions[usize::try_from(edge).unwrap()].contains(&target))
        .count()
        % 2
        == 1
}

fn candidate_cost(probabilities: &[f64], candidate: &ParityFactor) -> f64 {
    candidate
        .subgraph
        .iter()
        .map(|&edge| weight_of(probabilities[usize::try_from(edge).unwrap()]))
        .sum()
}

fn candidate_probability(
    hypergraph: &DecodingHypergraph,
    baseline: &ParityFactor,
    candidate: &ParityFactor,
    reweights: &[EdgeReweight],
) -> f64 {
    let mut probabilities: Vec<_> = hypergraph.hyperedges.iter().map(|edge| edge.probability).collect();
    for reweight in reweights {
        probabilities[usize::try_from(reweight.edge).unwrap()] = reweight.probability;
    }

    let agreeing_cost = candidate_cost(&probabilities, baseline);
    let differing_cost = candidate_cost(&probabilities, candidate);
    let gap = differing_cost - agreeing_cost;
    if gap.is_nan() { 0.0 } else { probability_of_weight(gap) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::blackbox_decoder::Hyperedge;
    use crate::decoder::{DynDecoder, MockDecoder};
    use std::sync::Arc;

    fn test_hypergraph() -> DecodingHypergraph {
        DecodingHypergraph {
            vertex_num: 1,
            hyperedges: vec![
                Hyperedge {
                    vertices: vec![0],
                    probability: 0.1,
                },
                Hyperedge {
                    vertices: vec![0],
                    probability: 0.01,
                },
            ],
        }
    }

    #[test]
    fn forced_hypergraph_appends_action_row() {
        let forced = forced_hypergraph(&test_hypergraph(), &[vec![0], vec![]], 0);

        assert_eq!(forced.vertex_num, 2);
        assert_eq!(forced.hyperedges[0].vertices, vec![0, 1]);
        assert_eq!(forced.hyperedges[1].vertices, vec![0]);
    }

    #[test]
    fn candidate_probability_is_max_log_odds() {
        let hypergraph = test_hypergraph();
        let baseline = ParityFactor { subgraph: vec![] };
        let alternative = ParityFactor { subgraph: vec![0] };

        let probability = candidate_probability(&hypergraph, &baseline, &alternative, &[]);

        assert!((probability - 0.1).abs() < 1e-12);
    }

    #[test]
    fn cheaper_opposite_class_scores_above_half() {
        let hypergraph = DecodingHypergraph {
            vertex_num: 1,
            hyperedges: vec![
                Hyperedge {
                    vertices: vec![0],
                    probability: 0.01,
                },
                Hyperedge {
                    vertices: vec![0],
                    probability: 0.2,
                },
            ],
        };
        let baseline = ParityFactor { subgraph: vec![0] };
        let alternative = ParityFactor { subgraph: vec![1] };

        let probability = candidate_probability(&hypergraph, &baseline, &alternative, &[]);

        assert!(probability.is_finite());
        assert!(probability > 0.5);
    }

    #[test]
    fn candidate_probability_uses_requested_candidate() {
        let mut hypergraph = test_hypergraph();
        hypergraph.hyperedges.push(Hyperedge {
            vertices: vec![0],
            probability: 0.2,
        });
        let baseline = ParityFactor { subgraph: vec![] };
        let first = candidate_probability(&hypergraph, &baseline, &ParityFactor { subgraph: vec![0] }, &[]);
        let second = candidate_probability(&hypergraph, &baseline, &ParityFactor { subgraph: vec![2] }, &[]);

        assert!((first - 0.1).abs() < 1e-12);
        assert!((second - 0.2).abs() < 1e-12);
    }

    #[tokio::test]
    async fn persistent_forced_graph_decodes_action_only_edge() {
        let mock = Arc::new(MockDecoder::new());
        let decoder = DynDecoder::Mock(Arc::clone(&mock));
        let hypergraph = DecodingHypergraph {
            vertex_num: 0,
            hyperedges: vec![Hyperedge {
                vertices: vec![],
                probability: 0.1,
            }],
        };
        let logical_actions = vec![vec![0]];
        let hids = load_forced_hypergraphs(&decoder, &hypergraph, &logical_actions, 1)
            .await
            .unwrap();
        mock.set_response(vec![0b1000_0000], vec![0]).await;

        let probabilities = forced_gap_probabilities(
            &decoder,
            &hypergraph,
            &logical_actions,
            Some(&hids),
            &BitVector { size: 0, data: vec![] },
            &ParityFactor { subgraph: vec![] },
            &[],
            None,
            true,
            1,
        )
        .await
        .unwrap();

        assert!((probabilities[0] - 0.1).abs() < 1e-12);
        let state = mock.state.read().await;
        assert_eq!(state.decode_loaded_calls.len(), 1);
        assert_eq!(state.decode_loaded_calls[0].hid, hids[0].unwrap());
    }

    #[tokio::test]
    async fn unavailable_logical_target_returns_zero_without_decoding() {
        let mock = Arc::new(MockDecoder::new());
        let decoder = DynDecoder::Mock(Arc::clone(&mock));
        let hypergraph = DecodingHypergraph {
            vertex_num: 0,
            hyperedges: vec![Hyperedge {
                vertices: vec![],
                probability: 0.1,
            }],
        };
        let logical_actions = vec![vec![0]];
        let hids = load_forced_hypergraphs(&decoder, &hypergraph, &logical_actions, 2)
            .await
            .unwrap();
        assert!(hids[0].is_some());
        assert!(hids[1].is_none());
        mock.set_response(vec![0b1000_0000], vec![0]).await;

        let probabilities = forced_gap_probabilities(
            &decoder,
            &hypergraph,
            &logical_actions,
            Some(&hids),
            &BitVector::default(),
            &ParityFactor::default(),
            &[],
            None,
            true,
            2,
        )
        .await
        .unwrap();

        assert!((probabilities[0] - 0.1).abs() < 1e-12);
        assert!(probabilities[1].abs() < f64::EPSILON);
        let state = mock.state.read().await;
        assert_eq!(state.loaded_hypergraphs.len(), 1);
        assert_eq!(state.decode_loaded_calls.len(), 1);
    }

    #[tokio::test]
    async fn persistent_forced_graph_propagates_stale_handle() {
        let decoder = DynDecoder::Mock(Arc::new(MockDecoder::new()));
        let error = forced_gap_probabilities(
            &decoder,
            &test_hypergraph(),
            &[vec![0], vec![]],
            Some(&[Some(u64::MAX)]),
            &BitVector { size: 1, data: vec![0] },
            &ParityFactor { subgraph: vec![] },
            &[],
            None,
            true,
            1,
        )
        .await
        .unwrap_err();

        assert_eq!(error.code(), tonic::Code::NotFound);
    }
}
