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

#[derive(Debug)]
pub struct ForcedGapGraph {
    /// Stable scoring hypergraph before adding a forced-target parity constraint.
    hypergraph: Arc<DecodingHypergraph>,
    /// Logical target indices flipped by each hyperedge, in hypergraph edge order.
    logical_flips: Arc<Vec<Vec<u64>>>,
    /// Total number of readout and boundary targets, including duplicates and syndrome-fixed targets.
    target_count: usize,
    /// Lazily loaded forced-graph handles per target; `None` disables persistent backend loading.
    handles: Option<Vec<OnceCell<u64>>>,
    /// Target representatives computed once per graph; `None` entries mark syndrome-fixed targets.
    target_representatives: OnceLock<Vec<Option<usize>>>,
}

impl ForcedGapGraph {
    pub(crate) fn new(
        hypergraph: Arc<DecodingHypergraph>,
        logical_flips: Arc<Vec<Vec<u64>>>,
        target_count: usize,
        persistent: bool,
    ) -> Self {
        Self {
            hypergraph,
            logical_flips,
            target_count,
            handles: persistent.then(|| (0..target_count).map(|_| OnceCell::new()).collect()),
            target_representatives: OnceLock::new(),
        }
    }

    fn representative_target(&self, target: usize) -> Option<usize> {
        self.target_representatives.get_or_init(|| {
            let mut checks =
                BitMatrix::zeros(usize::try_from(self.hypergraph.vertex_num).unwrap(), self.logical_flips.len());
            for (edge, hyperedge) in self.hypergraph.hyperedges.iter().enumerate() {
                for &vertex in &hyperedge.vertices {
                    checks.set((usize::try_from(vertex).unwrap(), edge), true);
                }
            }
            let checks = EchelonForm::new(checks);
            let mut representatives = HashMap::new();
            (0..self.target_count)
                .map(|target| {
                    let target_index = u64::try_from(target).unwrap();
                    let edges: Vec<_> = self
                        .logical_flips
                        .iter()
                        .enumerate()
                        .filter_map(|(edge, flips)| flips.contains(&target_index).then_some(edge))
                        .collect();
                    *representatives.entry(edges).or_insert_with(|| {
                        let flips: BitVec = self.logical_flips.iter().map(|flips| flips.contains(&target_index)).collect();
                        checks.transpose_solve(&flips.as_view()).is_none().then_some(target)
                    })
                })
                .collect()
        })[target]
    }

    pub(crate) fn problem(
        self: &Arc<Self>,
        decoder: DynDecoder,
        syndrome: BitVector,
        decoder_seed: Option<u64>,
        baseline: ParityFactor,
        reweights: Vec<EdgeReweight>,
        use_loaded_reweights: bool,
    ) -> ForcedGapProblem {
        let baseline_is_valid = is_parity_factor(&self.hypergraph, &baseline, &syndrome);
        ForcedGapProblem {
            graph: Arc::clone(self),
            decoder,
            syndrome,
            decoder_seed,
            baseline,
            baseline_is_valid,
            reweights,
            use_loaded_reweights,
            probabilities: (0..self.target_count).map(|_| OnceCell::new()).collect(),
        }
    }
}

pub(crate) struct ForcedGapProblem {
    /// Shared scoring hypergraph, target representatives, and persistent decoder handles.
    graph: Arc<ForcedGapGraph>,
    /// Decoder service used to load and solve forced-target graphs.
    decoder: DynDecoder,
    /// Syndrome constraints shared by the baseline and every forced correction.
    syndrome: BitVector,
    /// Decoder seed forwarded to every forced solve for this problem.
    decoder_seed: Option<u64>,
    /// Primary correction defining the reference cost and target values to oppose.
    baseline: ParityFactor,
    /// Cached check that the baseline satisfies the scoring graph's syndrome.
    baseline_is_valid: bool,
    /// Shot-specific probability overrides indexed by scoring-graph edge.
    reweights: Vec<EdgeReweight>,
    /// Whether reweights may be sent to loaded decoder handles instead of materializing a graph.
    use_loaded_reweights: bool,
    /// Per-shot scores or errors computed lazily and cached at representative-target indices.
    probabilities: Vec<OnceCell<Result<f64, Status>>>,
}

impl ForcedGapProblem {
    pub(crate) async fn probability(&self, target: usize) -> Result<f64, Status> {
        if !self.baseline_is_valid {
            return Err(Status::internal("forced-gap baseline does not satisfy the syndrome"));
        }
        let Some(target) = self.graph.representative_target(target) else {
            return Ok(0.0);
        };
        self.probabilities[target].get_or_init(|| self.solve(target)).await.clone()
    }

    pub(crate) async fn probabilities(&self) -> Result<Vec<f64>, Status> {
        try_join_all((0..self.probabilities.len()).map(|target| self.probability(target))).await
    }

    async fn solve(&self, target: usize) -> Result<f64, Status> {
        let graph = &self.graph;
        let handle = if let Some(handles) = &graph.handles
            && (self.reweights.is_empty() || self.use_loaded_reweights)
        {
            Some(
                *handles[target]
                    .get_or_try_init(|| async {
                        self.decoder
                            .load_hypergraph(forced_hypergraph(&graph.hypergraph, &graph.logical_flips, target))
                            .await
                            .map(|response| response.hid)
                    })
                    .await?,
            )
        } else {
            None
        };
        forced_gap_probability(self, handle, target).await
    }
}

fn forced_hypergraph(hypergraph: &DecodingHypergraph, logical_flips: &[Vec<u64>], target: usize) -> DecodingHypergraph {
    debug_assert_eq!(hypergraph.hyperedges.len(), logical_flips.len());
    let mut forced = hypergraph.clone();
    let forced_vertex = forced.vertex_num;
    forced.vertex_num += 1;
    let target = u64::try_from(target).unwrap();
    for (hyperedge, flips) in forced.hyperedges.iter_mut().zip(logical_flips) {
        if flips.contains(&target) {
            hyperedge.vertices.push(forced_vertex);
        }
    }
    forced
}

async fn forced_gap_probability(
    problem: &ForcedGapProblem,
    forced_hid: Option<u64>,
    target_index: usize,
) -> Result<f64, Status> {
    let ForcedGapProblem {
        graph,
        decoder,
        syndrome,
        decoder_seed,
        baseline,
        reweights,
        use_loaded_reweights,
        ..
    } = problem;
    let hypergraph = &graph.hypergraph;
    let logical_flips = graph.logical_flips.as_slice();
    debug_assert_eq!(hypergraph.hyperedges.len(), logical_flips.len());
    debug_assert_eq!(syndrome.size, hypergraph.vertex_num);
    let target = u64::try_from(target_index).unwrap();

    let baseline_bit = logical_bit(logical_flips, baseline, target);
    let forced_bit = !baseline_bit;
    let mut forced_syndrome = syndrome.clone();
    let forced_vertex = forced_syndrome.size;
    extend_num_bits(&mut forced_syndrome, 1);
    set_bit(&mut forced_syndrome, forced_vertex, forced_bit);
    let result = if let Some(hid) = forced_hid
        && (reweights.is_empty() || *use_loaded_reweights)
    {
        decoder
            .decode_loaded(LoadedDecodingProblem {
                hid,
                syndrome: Some(forced_syndrome.clone()),
                reweights: reweights.clone(),
                loss: None,
                decoder_seed: *decoder_seed,
            })
            .await
    } else {
        let mut forced_graph = forced_hypergraph(hypergraph, logical_flips, target_index);
        apply_reweights(
            &mut forced_graph,
            reweights.iter().map(|reweight| (reweight.edge, reweight.probability)),
        );
        decoder
            .decode(DecodingProblem {
                hypergraph: Some(forced_graph),
                syndrome: Some(forced_syndrome.clone()),
                loss: None,
                decoder_seed: *decoder_seed,
            })
            .await
    };

    match result {
        Ok(candidate)
            if is_parity_factor(hypergraph, &candidate, syndrome)
                && logical_bit(logical_flips, &candidate, target) == forced_bit =>
        {
            candidate_probability(hypergraph, baseline, &candidate, reweights)
        }
        Ok(_) => {
            let mut forced_graph = forced_hypergraph(hypergraph, logical_flips, target_index);
            apply_reweights(
                &mut forced_graph,
                reweights.iter().map(|reweight| (reweight.edge, reweight.probability)),
            );
            if has_matching_parity_factor(&forced_graph, &forced_syndrome) {
                Err(Status::internal(format!(
                    "decoder did not satisfy the reachable forced-gap constraint for target {target_index}"
                )))
            } else {
                Ok(0.0)
            }
        }
        Err(error) => {
            let mut forced_graph = forced_hypergraph(hypergraph, logical_flips, target_index);
            apply_reweights(
                &mut forced_graph,
                reweights.iter().map(|reweight| (reweight.edge, reweight.probability)),
            );
            let reachable = has_matching_parity_factor(&forced_graph, &forced_syndrome);
            Err(Status::new(
                error.code(),
                format!(
                    "forced-gap target {target_index} failed (reachable={reachable}, vertices={}, edges={}): {}",
                    forced_graph.vertex_num,
                    forced_graph.hyperedges.len(),
                    error.message()
                ),
            ))
        }
    }
}

fn has_matching_parity_factor(hypergraph: &DecodingHypergraph, syndrome: &BitVector) -> bool {
    let mut target: Vec<_> = (0..syndrome.size).map(|index| get_bit(syndrome, index)).collect();
    let mut matrix = BitMatrix::zeros(target.len(), hypergraph.hyperedges.len());
    for (edge_index, edge) in hypergraph.hyperedges.iter().enumerate() {
        for &vertex in &edge.vertices {
            let vertex = usize::try_from(vertex).unwrap();
            if edge.probability >= 1.0 {
                target[vertex] ^= true;
            } else if edge.probability > 0.0 {
                matrix.set((vertex, edge_index), true);
            }
        }
    }
    EchelonForm::new(matrix).solve(&BitVec::from_iter(target).as_view()).is_some()
}

fn logical_bit(logical_flips: &[Vec<u64>], candidate: &ParityFactor, target: u64) -> bool {
    candidate
        .subgraph
        .iter()
        .filter(|&&edge| logical_flips[usize::try_from(edge).unwrap()].contains(&target))
        .count()
        % 2
        == 1
}

fn candidate_probability(
    hypergraph: &DecodingHypergraph,
    baseline: &ParityFactor,
    candidate: &ParityFactor,
    reweights: &[EdgeReweight],
) -> Result<f64, Status> {
    let mut probabilities: Vec<_> = hypergraph.hyperedges.iter().map(|edge| edge.probability).collect();
    for reweight in reweights {
        probabilities[usize::try_from(reweight.edge).unwrap()] = reweight.probability;
    }

    let baseline_edges: HashSet<_> = baseline.subgraph.iter().copied().collect();
    let candidate_edges: HashSet<_> = candidate.subgraph.iter().copied().collect();
    let is_possible = |edges: &HashSet<u64>| {
        probabilities.iter().enumerate().all(|(edge, &probability)| {
            let selected = edges.contains(&u64::try_from(edge).unwrap());
            (probability > 0.0 || !selected) && (probability < 1.0 || selected)
        })
    };
    match (is_possible(&baseline_edges), is_possible(&candidate_edges)) {
        (false, false) => return Err(Status::internal("forced-gap likelihood comparison is undefined")),
        (false, true) => return Ok(1.0),
        (true, false) => return Ok(0.0),
        (true, true) => {}
    }
    let mut multiplicities = HashMap::<u64, f64>::new();
    for &edge in baseline_edges.symmetric_difference(&candidate_edges) {
        let probability = probabilities[usize::try_from(edge).unwrap()];
        *multiplicities.entry(probability.to_bits()).or_default() +=
            if candidate_edges.contains(&edge) { 1.0 } else { -1.0 };
    }
    let mut terms: Vec<_> = multiplicities.into_iter().filter(|&(_, count)| count != 0.0).collect();
    terms.sort_unstable_by_key(|&(probability, _)| probability);
    let gap: f64 = terms
        .into_iter()
        .map(|(probability, count)| count * weight_of(f64::from_bits(probability)))
        .sum();
    if gap.is_nan() {
        Err(Status::internal("forced-gap likelihood comparison is undefined"))
    } else {
        Ok(probability_of_weight(gap))
    }
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

    fn syndrome_free_hypergraph() -> DecodingHypergraph {
        DecodingHypergraph {
            vertex_num: 0,
            hyperedges: vec![Hyperedge {
                vertices: vec![],
                probability: 0.1,
            }],
        }
    }

    fn test_problem(mock: &Arc<MockDecoder>, hypergraph: DecodingHypergraph, persistent: bool) -> ForcedGapProblem {
        let syndrome = crate::misc::bit_vector::from_sparse_indices(hypergraph.vertex_num, &[]);
        let logical_flips = (0..hypergraph.hyperedges.len())
            .map(|edge| if edge == 0 { vec![0] } else { vec![] })
            .collect();
        Arc::new(ForcedGapGraph::new(
            Arc::new(hypergraph),
            Arc::new(logical_flips),
            2,
            persistent,
        ))
        .problem(
            DynDecoder::Mock(Arc::clone(mock)),
            syndrome,
            None,
            ParityFactor::default(),
            vec![],
            true,
        )
    }

    #[test]
    fn forced_hypergraph_appends_logical_target_row() {
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

        let probability = candidate_probability(&hypergraph, &baseline, &alternative, &[]).unwrap();

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

        let probability = candidate_probability(&hypergraph, &baseline, &alternative, &[]).unwrap();

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
        let first = candidate_probability(&hypergraph, &baseline, &ParityFactor { subgraph: vec![0] }, &[]).unwrap();
        let second = candidate_probability(&hypergraph, &baseline, &ParityFactor { subgraph: vec![2] }, &[]).unwrap();

        assert!((first - 0.1).abs() < 1e-12);
        assert!((second - 0.2).abs() < 1e-12);
    }

    #[test]
    fn shared_edges_do_not_change_the_likelihood_gap() {
        for common_probability in [1e-200, 1.0] {
            let mut hypergraph = test_hypergraph();
            let without_common = candidate_probability(
                &hypergraph,
                &ParityFactor::default(),
                &ParityFactor { subgraph: vec![1] },
                &[],
            )
            .unwrap();
            hypergraph.hyperedges[0].probability = common_probability;
            let with_common = candidate_probability(
                &hypergraph,
                &ParityFactor { subgraph: vec![0] },
                &ParityFactor { subgraph: vec![1, 0] },
                &[],
            )
            .unwrap();
            assert_eq!(with_common.to_bits(), without_common.to_bits());
        }
    }

    #[test]
    fn equal_priors_cancel_independently_of_edge_identity_and_order() {
        let hyperedge = Hyperedge {
            vertices: vec![],
            probability: 0.000_300_631_896_544_833_8,
        };
        let hypergraph = DecodingHypergraph {
            vertex_num: 0,
            hyperedges: vec![hyperedge; 17],
        };
        let expected = candidate_probability(
            &hypergraph,
            &ParityFactor::default(),
            &ParityFactor { subgraph: vec![0] },
            &[],
        )
        .unwrap();
        for baseline_count in 0..8 {
            let baseline = ParityFactor {
                subgraph: (0..baseline_count).collect(),
            };
            let candidate = ParityFactor {
                subgraph: (baseline_count..=2 * baseline_count).rev().collect(),
            };
            let actual = candidate_probability(&hypergraph, &baseline, &candidate, &[]).unwrap();
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn undefined_likelihood_gap_is_not_zero_risk() {
        let mut hypergraph = test_hypergraph();
        for edge in &mut hypergraph.hyperedges {
            edge.probability = 0.0;
        }
        let error = candidate_probability(
            &hypergraph,
            &ParityFactor { subgraph: vec![0] },
            &ParityFactor { subgraph: vec![1] },
            &[],
        )
        .unwrap_err();
        assert_eq!(error.code(), tonic::Code::Internal);
    }

    #[test]
    fn shared_impossible_edges_do_not_hide_undefined_likelihoods() {
        let mut hypergraph = test_hypergraph();
        hypergraph.hyperedges[0].probability = 0.0;
        let error = candidate_probability(
            &hypergraph,
            &ParityFactor { subgraph: vec![0] },
            &ParityFactor { subgraph: vec![0, 1] },
            &[],
        )
        .unwrap_err();
        assert_eq!(error.code(), tonic::Code::Internal);
    }

    #[test]
    fn deterministic_priors_select_the_only_possible_candidate() {
        let mut hypergraph = test_hypergraph();
        let selected = ParityFactor { subgraph: vec![0] };
        for prior in [0.0, 1.0] {
            hypergraph.hyperedges[0].probability = prior;
            let probability = candidate_probability(&hypergraph, &ParityFactor::default(), &selected, &[]).unwrap();
            assert!((probability - prior).abs() < f64::EPSILON);
            let probability = candidate_probability(&hypergraph, &selected, &ParityFactor::default(), &[]).unwrap();
            assert!((probability - (1.0 - prior)).abs() < f64::EPSILON);
        }
    }

    #[tokio::test]
    async fn persistent_forced_graph_decodes_syndrome_free_edge() {
        let mock = Arc::new(MockDecoder::new());
        let problem = test_problem(&mock, syndrome_free_hypergraph(), true);
        mock.set_response(vec![0b1000_0000], vec![0]).await;

        let probability = problem.probability(0).await.unwrap();

        assert!((probability - 0.1).abs() < 1e-12);
        let state = mock.state.read().await;
        assert_eq!(state.decode_loaded_calls.len(), 1);
        assert_eq!(state.loaded_hypergraphs.len(), 1);
        assert_eq!(
            state.decode_loaded_calls[0].hid,
            *problem.graph.handles.as_ref().unwrap()[0].get().unwrap()
        );
    }

    #[tokio::test]
    async fn unavailable_logical_target_returns_zero_without_decoding() {
        let mock = Arc::new(MockDecoder::new());
        let problem = test_problem(&mock, syndrome_free_hypergraph(), true);
        mock.set_response(vec![0b1000_0000], vec![0]).await;

        let probabilities = problem.probabilities().await.unwrap();

        assert!((probabilities[0] - 0.1).abs() < 1e-12);
        assert!(probabilities[1].abs() < f64::EPSILON);
        let state = mock.state.read().await;
        assert_eq!(state.loaded_hypergraphs.len(), 1);
        assert_eq!(state.decode_loaded_calls.len(), 1);
    }

    #[tokio::test]
    async fn identical_constraints_share_one_forced_solve() {
        for persistent in [false, true] {
            let mock = Arc::new(MockDecoder::new());
            mock.set_response(vec![0b1000_0000], vec![0]).await;
            let graph = Arc::new(ForcedGapGraph::new(
                Arc::new(syndrome_free_hypergraph()),
                Arc::new(vec![vec![0, 1]]),
                2,
                persistent,
            ));
            for _shot in 0..2 {
                let problem = graph.problem(
                    DynDecoder::Mock(Arc::clone(&mock)),
                    BitVector::default(),
                    None,
                    ParityFactor::default(),
                    vec![],
                    true,
                );
                let scores = problem.probabilities().await.unwrap();
                assert_eq!(scores.len(), 2);
                assert!(scores.iter().all(|score| (score - 0.1).abs() < 1e-12));
            }
            let state = mock.state.read().await;
            assert_eq!(state.decode_calls.len() + state.decode_loaded_calls.len(), 2);
            assert_eq!(state.loaded_hypergraphs.len(), usize::from(persistent));
        }
    }

    #[test]
    fn distinct_constraints_keep_separate_representatives() {
        let graph = ForcedGapGraph::new(
            Arc::new(DecodingHypergraph {
                vertex_num: 0,
                hyperedges: vec![
                    Hyperedge {
                        vertices: vec![],
                        probability: 0.1,
                    },
                    Hyperedge {
                        vertices: vec![],
                        probability: 0.2,
                    },
                ],
            }),
            Arc::new(vec![vec![0, 1], vec![2]]),
            4,
            true,
        );
        assert_eq!(
            (0..4).map(|target| graph.representative_target(target)).collect::<Vec<_>>(),
            vec![Some(0), Some(0), Some(2), None],
        );
    }

    #[tokio::test]
    async fn persistent_forced_graph_propagates_stale_handle() {
        let problem = test_problem(&Arc::new(MockDecoder::new()), test_hypergraph(), true);
        problem.graph.handles.as_ref().unwrap()[0].set(u64::MAX).unwrap();
        let error = problem.probability(0).await.unwrap_err();

        assert_eq!(error.code(), tonic::Code::NotFound);
        assert!(error.message().contains("target 0 failed (reachable=true"));
    }

    #[tokio::test]
    async fn failed_forced_search_does_not_report_zero_risk() {
        let problem = test_problem(&Arc::new(MockDecoder::new()), syndrome_free_hypergraph(), false);
        let error = problem.probability(0).await.unwrap_err();

        assert_eq!(error.code(), tonic::Code::Internal);
        assert!(error.message().contains("reachable forced-gap constraint for target 0"));
    }

    #[tokio::test]
    async fn impossible_shot_reweighted_alternative_is_not_a_decoder_failure() {
        let mock = Arc::new(MockDecoder::new());
        let mut problem = test_problem(&mock, syndrome_free_hypergraph(), false);
        problem.reweights.push(EdgeReweight {
            edge: 0,
            probability: 0.0,
        });
        assert_eq!(problem.probability(0).await.unwrap(), 0.0);
    }

    #[tokio::test]
    async fn syndrome_determined_target_has_no_opposite_class() {
        let hypergraph = DecodingHypergraph {
            vertex_num: 1,
            hyperedges: vec![Hyperedge {
                vertices: vec![0],
                probability: 0.1,
            }],
        };
        let mock = Arc::new(MockDecoder::new());
        let problem = test_problem(&mock, hypergraph, true);
        let probability = problem.probability(0).await.unwrap();

        assert!(probability.abs() < f64::EPSILON);
        let state = mock.state.read().await;
        assert!(state.loaded_hypergraphs.is_empty());
        assert!(state.decode_calls.is_empty());
        assert!(state.decode_loaded_calls.is_empty());
    }

    #[tokio::test]
    async fn zero_prior_alternatives_remain_available_to_later_shot_reweights() {
        let mock = Arc::new(MockDecoder::new());
        let mut hypergraph = syndrome_free_hypergraph();
        hypergraph.hyperedges[0].probability = 0.0;
        let mut problem = test_problem(&mock, hypergraph, true);
        problem.reweights.push(EdgeReweight {
            edge: 0,
            probability: 0.1,
        });
        mock.set_response(vec![0b1000_0000], vec![0]).await;

        assert!((problem.probability(0).await.unwrap() - 0.1).abs() < 1e-12);
        assert_eq!(mock.state.read().await.decode_loaded_calls.len(), 1);
    }

    #[tokio::test]
    async fn failed_score_is_memoized_within_the_shot() {
        let mock = Arc::new(MockDecoder::new());
        let problem = test_problem(&mock, test_hypergraph(), true);
        for _ in 0..2 {
            assert_eq!(problem.probability(0).await.unwrap_err().code(), tonic::Code::Internal);
        }
        assert_eq!(mock.state.read().await.decode_loaded_calls.len(), 1);
    }

    #[cfg(feature = "tesseract")]
    #[tokio::test]
    async fn reachable_alternative_can_require_a_wider_detector_beam() {
        use crate::decoder::DecoderType;
        use serde_json::json;

        for persistent in [false, true] {
            for beam in [1, 2] {
                let graph = Arc::new(ForcedGapGraph::new(
                    Arc::new(DecodingHypergraph {
                        vertex_num: 3,
                        hyperedges: vec![
                            Hyperedge {
                                vertices: vec![0, 1, 2],
                                probability: 0.1,
                            },
                            Hyperedge {
                                vertices: vec![0, 1, 2],
                                probability: 0.1,
                            },
                        ],
                    }),
                    Arc::new(vec![vec![0], vec![]]),
                    1,
                    persistent,
                ));
                let decoder = DecoderType::BlackBoxTesseract.create(json!({
                    "parallel": 1, "det_beam": beam, "pqlimit": 2000,
                    "det_penalty": 30, "beam_climbing": false
                }));
                let problem = graph.problem(
                    decoder,
                    crate::misc::bit_vector::from_sparse_indices(3, &[]),
                    None,
                    ParityFactor::default(),
                    vec![],
                    true,
                );
                if beam == 1 {
                    let error = problem.probability(0).await.unwrap_err();
                    assert!(error.message().contains("reachable=true"));
                    assert!(error.message().contains("det_beam=1"));
                } else {
                    let score = problem.probability(0).await.unwrap();
                    assert!((score - 1.0 / 82.0).abs() < 1e-12);
                }
            }
        }
    }

    #[cfg(feature = "tesseract")]
    #[tokio::test]
    async fn bounded_search_recovery_produces_a_valid_forced_gap_score() {
        use crate::decoder::DecoderType;
        use serde_json::json;

        let mut hyperedges = vec![
            Hyperedge {
                vertices: vec![0],
                probability: 0.1,
            },
            Hyperedge {
                vertices: vec![0],
                probability: 0.1,
            },
        ];
        for vertices in [vec![1, 2, 3], vec![1, 2, 4], vec![1, 3, 4], vec![2, 3, 4]] {
            hyperedges.push(Hyperedge {
                vertices,
                probability: 0.2,
            });
        }
        let graph = Arc::new(ForcedGapGraph::new(
            Arc::new(DecodingHypergraph {
                vertex_num: 5,
                hyperedges,
            }),
            Arc::new(vec![vec![0], vec![], vec![0], vec![0], vec![0], vec![0]]),
            1,
            true,
        ));
        let decoder = DecoderType::BlackBoxTesseract.create(json!({"parallel": 1, "det_beam": 5, "pqlimit": 3}));
        let problem = graph.problem(
            decoder,
            crate::misc::bit_vector::from_sparse_indices(5, &[]),
            None,
            ParityFactor::default(),
            vec![],
            true,
        );
        let probability = problem.probability(0).await.unwrap();
        assert!((probability - 1.0 / 82.0).abs() < 1e-12);
        assert_eq!(probability, problem.probability(0).await.unwrap());
    }
}
