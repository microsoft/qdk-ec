use crate::coordinator::reweight_handler::apply_reweights;
use crate::decoder::DynDecoder;
use crate::decoder::blackbox_decoder::{
    DecodingHypergraph, DecodingProblem, EdgeReweight, LoadedDecodingProblem, LossInfo, ParityFactor,
};
use crate::decoder::blackbox_util::is_parity_factor;
use crate::misc::bit_vector::{extend_num_bits, set_bit};
use crate::misc::util::{probability_of_weight, weight_of};
use crate::util::BitVector;
use futures_util::future::join_all;
use hashbrown::HashMap;
use tonic::Status;

pub(crate) fn forced_hypergraph(
    hypergraph: &DecodingHypergraph,
    logical_actions: &[Vec<u64>],
    readout: usize,
) -> DecodingHypergraph {
    debug_assert_eq!(hypergraph.hyperedges.len(), logical_actions.len());
    let mut forced = hypergraph.clone();
    let forced_vertex = forced.vertex_num;
    forced.vertex_num += 1;
    let readout = u64::try_from(readout).unwrap();
    for (hyperedge, action) in forced.hyperedges.iter_mut().zip(logical_actions) {
        if action.contains(&readout) {
            hyperedge.vertices.push(forced_vertex);
        }
    }
    forced
}

pub(crate) async fn load_forced_hypergraphs(
    decoder: &DynDecoder,
    hypergraph: &DecodingHypergraph,
    logical_actions: &[Vec<u64>],
    readout_count: usize,
) -> Result<Vec<Option<u64>>, Status> {
    let loads = (0..readout_count).map(|readout| {
        let decoder = decoder.clone();
        let hypergraph = logical_actions
            .iter()
            .any(|action| action.contains(&u64::try_from(readout).unwrap()))
            .then(|| forced_hypergraph(hypergraph, logical_actions, readout));
        async move {
            let Some(hypergraph) = hypergraph else {
                return Ok(None);
            };
            decoder.load_hypergraph(hypergraph).await.map(|response| Some(response.hid))
        }
    });
    join_all(loads).await.into_iter().collect()
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
    readout_count: usize,
) -> Result<Vec<f64>, Status> {
    debug_assert_eq!(hypergraph.hyperedges.len(), logical_actions.len());
    let baseline_class = logical_class(logical_actions, baseline, readout_count);
    let decode_futures = (0..readout_count).map(|readout| {
        let decoder = decoder.clone();
        let has_opposite_class = logical_actions
            .iter()
            .any(|action| action.contains(&u64::try_from(readout).unwrap()));
        let mut forced_syndrome = syndrome.clone();
        let forced_bit = !baseline_class[readout];
        let forced_vertex = forced_syndrome.size;
        extend_num_bits(&mut forced_syndrome, 1);
        set_bit(&mut forced_syndrome, forced_vertex, forced_bit);
        let reweights = reweights.to_vec();
        let loss = loss.cloned();
        let hid = forced_hids.and_then(|hids| hids.get(readout)).copied().flatten();

        async move {
            if !has_opposite_class {
                return Ok(None);
            }
            let result = if let Some(hid) = hid
                && (reweights.is_empty() || use_loaded_reweights)
            {
                decoder
                    .decode_loaded(LoadedDecodingProblem {
                        hid,
                        syndrome: Some(forced_syndrome),
                        reweights,
                        loss,
                    })
                    .await
            } else {
                let mut forced_graph = forced_hypergraph(hypergraph, logical_actions, readout);
                apply_reweights(
                    &mut forced_graph,
                    reweights.iter().map(|reweight| (reweight.edge, reweight.probability)),
                );
                decoder
                    .decode(DecodingProblem {
                        hypergraph: Some(forced_graph),
                        syndrome: Some(forced_syndrome),
                        loss,
                    })
                    .await
            };

            match result {
                Ok(candidate)
                    if is_parity_factor(hypergraph, &candidate, syndrome)
                        && logical_class(logical_actions, &candidate, readout_count)[readout] == forced_bit =>
                {
                    Ok(Some(candidate))
                }
                Ok(_) => Ok(None),
                Err(error) => Err(error),
            }
        }
    });
    let candidates: Result<Vec<_>, _> = join_all(decode_futures).await.into_iter().collect();
    Ok(candidate_probabilities(
        hypergraph,
        logical_actions,
        baseline,
        candidates?.iter().flatten(),
        reweights,
        readout_count,
    ))
}

fn logical_class(logical_actions: &[Vec<u64>], candidate: &ParityFactor, readout_count: usize) -> Vec<bool> {
    let mut class = vec![false; readout_count];
    for &edge in &candidate.subgraph {
        for &readout in &logical_actions[usize::try_from(edge).unwrap()] {
            class[usize::try_from(readout).unwrap()] ^= true;
        }
    }
    class
}

fn candidate_cost(probabilities: &[f64], candidate: &ParityFactor) -> f64 {
    candidate
        .subgraph
        .iter()
        .map(|&edge| weight_of(probabilities[usize::try_from(edge).unwrap()]))
        .sum()
}

fn candidate_probabilities<'a>(
    hypergraph: &DecodingHypergraph,
    logical_actions: &[Vec<u64>],
    baseline: &'a ParityFactor,
    candidates: impl IntoIterator<Item = &'a ParityFactor>,
    reweights: &[EdgeReweight],
    readout_count: usize,
) -> Vec<f64> {
    let mut probabilities: Vec<_> = hypergraph.hyperedges.iter().map(|edge| edge.probability).collect();
    for reweight in reweights {
        probabilities[usize::try_from(reweight.edge).unwrap()] = reweight.probability;
    }

    let baseline_class = logical_class(logical_actions, baseline, readout_count);
    let mut class_costs = HashMap::new();
    for candidate in std::iter::once(baseline).chain(candidates) {
        let class = logical_class(logical_actions, candidate, readout_count);
        let cost = candidate_cost(&probabilities, candidate);
        if !cost.is_nan() {
            class_costs
                .entry(class)
                .and_modify(|best: &mut f64| *best = best.min(cost))
                .or_insert(cost);
        }
    }

    (0..readout_count)
        .map(|readout| {
            let mut agreeing_cost = f64::INFINITY;
            let mut differing_cost = f64::INFINITY;
            for (class, &cost) in &class_costs {
                if class[readout] == baseline_class[readout] {
                    agreeing_cost = agreeing_cost.min(cost);
                } else {
                    differing_cost = differing_cost.min(cost);
                }
            }
            if differing_cost.is_infinite() {
                0.0
            } else {
                // A negative gap is valid when the primary decoder selected a
                // more expensive class; it intentionally maps above 0.5.
                probability_of_weight(differing_cost - agreeing_cost)
            }
        })
        .collect()
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

        let probabilities = candidate_probabilities(&hypergraph, &[vec![0], vec![]], &baseline, [&alternative], &[], 1);

        assert!((probabilities[0] - 0.1).abs() < 1e-12);
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

        let probabilities = candidate_probabilities(&hypergraph, &[vec![], vec![0]], &baseline, [&alternative], &[], 1);

        assert!(probabilities[0].is_finite());
        assert!(probabilities[0] > 0.5);
    }

    #[test]
    fn candidate_probabilities_are_indexed_per_readout() {
        let mut hypergraph = test_hypergraph();
        hypergraph.hyperedges.push(Hyperedge {
            vertices: vec![0],
            probability: 0.2,
        });
        let baseline = ParityFactor { subgraph: vec![] };
        let alternatives = [ParityFactor { subgraph: vec![0] }, ParityFactor { subgraph: vec![2] }];

        let probabilities =
            candidate_probabilities(&hypergraph, &[vec![0], vec![], vec![1]], &baseline, &alternatives, &[], 2);

        assert!((probabilities[0] - 0.1).abs() < 1e-12);
        assert!((probabilities[1] - 0.2).abs() < 1e-12);
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
        assert_eq!(probabilities[1], 0.0);
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
