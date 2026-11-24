//! Forced-gap snapshots, causal history, and logical confidence propagation.

use super::{ErrorModel, WindowCoordinator};
use crate::coordinator::forced_gap_handler::{ForcedGapGraph, ForcedGapProblem};
use crate::coordinator::reweight_handler::ProjectedErrors;
use crate::decoder::DynDecoder;
use crate::decoder::blackbox_decoder::{DecodingHypergraph, EdgeReweight, Hyperedge, ParityFactor};
use crate::decoder::blackbox_util::is_parity_factor;
use crate::misc::bit_vector::flip_bit;
use crate::misc::index::ErrorIndex;
use crate::misc::pauli_frame_symbolic_propagator::{CorrectionBasis, PauliFrameSymbolicPropagator};
use crate::misc::relative_program::{RelativeMapping, RelativeProgram};
use crate::util::BitVector;
use futures_util::future::{join_all, try_join_all};
use hashbrown::{HashMap, HashSet};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tonic::Status;

/// A commit-edge graph retaining every window check row and its edge mapping.
#[derive(Debug)]
pub struct CommitRegionDecoder {
    pub(super) graph: Arc<ForcedGapGraph>,
    local_edge_of: Vec<Option<u64>>,
}

impl CommitRegionDecoder {
    pub(super) fn new(
        hypergraph: &DecodingHypergraph,
        logical_flips: &[Vec<u64>],
        retained_edges: &[bool],
        target_count: usize,
        persistent: bool,
    ) -> Self {
        assert_eq!(retained_edges.len(), hypergraph.hyperedges.len());
        assert_eq!(logical_flips.len(), hypergraph.hyperedges.len());
        let mut hyperedges = vec![];
        let mut retained_logical_flips = vec![];
        let local_edge_of = retained_edges
            .iter()
            .enumerate()
            .map(|(edge, &retained)| {
                retained.then(|| {
                    let local_edge = u64::try_from(hyperedges.len()).unwrap();
                    hyperedges.push(hypergraph.hyperedges[edge].clone());
                    retained_logical_flips.push(logical_flips[edge].clone());
                    local_edge
                })
            })
            .collect();
        Self {
            graph: Arc::new(ForcedGapGraph::new(
                Arc::new(DecodingHypergraph {
                    vertex_num: hypergraph.vertex_num,
                    hyperedges,
                }),
                Arc::new(retained_logical_flips),
                target_count,
                persistent,
            )),
            local_edge_of,
        }
    }

    pub(super) fn project(
        &self,
        hypergraph: &DecodingHypergraph,
        mut syndrome: BitVector,
        baseline: &ParityFactor,
        reweights: Vec<EdgeReweight>,
    ) -> Result<(BitVector, ParityFactor, Vec<EdgeReweight>), Status> {
        if !is_parity_factor(hypergraph, baseline, &syndrome) {
            return Err(Status::internal("forced-gap baseline does not satisfy the syndrome"));
        }
        let subgraph = baseline
            .subgraph
            .iter()
            .filter_map(|&edge| {
                let edge = usize::try_from(edge).unwrap();
                if let Some(local_edge) = self.local_edge_of[edge] {
                    Some(local_edge)
                } else {
                    for &vertex in &hypergraph.hyperedges[edge].vertices {
                        flip_bit(&mut syndrome, vertex);
                    }
                    None
                }
            })
            .collect();
        let reweights = reweights
            .into_iter()
            .filter_map(|reweight| {
                match usize::try_from(reweight.edge)
                    .ok()
                    .and_then(|edge| self.local_edge_of.get(edge))
                {
                    Some(Some(edge)) => Some(Ok(EdgeReweight {
                        edge: *edge,
                        probability: reweight.probability,
                    })),
                    Some(None) => None,
                    None => Some(Err(Status::invalid_argument("reweighted edge is outside the window graph"))),
                }
            })
            .collect::<Result<_, _>>()?;
        Ok((syndrome, ParityFactor { subgraph }, reweights))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn problem(
        &self,
        decoder: DynDecoder,
        hypergraph: &DecodingHypergraph,
        syndrome: BitVector,
        decoder_seed: Option<u64>,
        baseline: &ParityFactor,
        reweights: Vec<EdgeReweight>,
        use_loaded_reweights: bool,
    ) -> Result<ForcedGapProblem, Status> {
        let (syndrome, baseline, reweights) = self.project(hypergraph, syndrome, baseline, reweights)?;
        Ok(self
            .graph
            .problem(decoder, syndrome, decoder_seed, baseline, reweights, use_loaded_reweights))
    }
}

#[derive(Clone)]
pub(super) struct RecordedGapEdge {
    pub(super) gid: u64,
    pub(super) checks: Vec<(u64, u64)>,
    pub(super) probability: f64,
    pub(super) residual: Vec<u64>,
    pub(super) readouts: Vec<u64>,
    pub(super) selected: bool,
}

pub(super) struct CausalGapSnapshot {
    pub(super) hypergraph: DecodingHypergraph,
    pub(super) syndrome: BitVector,
    pub(super) baseline: ParityFactor,
    pub(super) edges: Vec<RecordedGapEdge>,
}

impl CausalGapSnapshot {
    pub(super) fn restore_history(
        &mut self,
        history: impl IntoIterator<Item = RecordedGapEdge>,
        vertices: &HashMap<(u64, u64), u64>,
    ) {
        for edge in history {
            let detectors: Vec<_> = edge.checks.iter().filter_map(|check| vertices.get(check).copied()).collect();
            if edge.selected {
                self.baseline.subgraph.push(u64::try_from(self.edges.len()).unwrap());
                for &vertex in &detectors {
                    flip_bit(&mut self.syndrome, vertex);
                }
            }
            self.hypergraph.hyperedges.push(Hyperedge {
                observable_flips: None,
                vertices: detectors,
                probability: edge.probability,
            });
            self.edges.push(edge);
        }
    }
}

pub(super) struct CausalGapProblem {
    problems: Vec<Arc<ForcedGapProblem>>,
}

pub(super) fn original_scoring_baseline(
    original: &[ErrorIndex],
    projected: &ProjectedErrors,
    baseline: &ParityFactor,
) -> ParityFactor {
    let indices: HashMap<_, _> = original
        .iter()
        .enumerate()
        .map(|(index, error)| ((error.eid, error.error_index), u64::try_from(index).unwrap()))
        .collect();
    ParityFactor {
        subgraph: baseline
            .subgraph
            .iter()
            .map(|&index| {
                let error = &projected[usize::try_from(index).unwrap()];
                indices[&(error.eid, error.error_index)]
            })
            .collect(),
    }
}

impl CausalGapProblem {
    pub(super) async fn probability(&self, target: usize) -> Result<f64, Status> {
        self.problems[target].probability(target).await
    }
}

#[derive(Clone)]
pub(super) enum ForcedGapScore {
    Ready(Result<f64, Status>),
    Deferred { problem: Arc<CausalGapProblem>, target: usize },
}

impl ForcedGapScore {
    pub(super) async fn probability(&self) -> Result<f64, Status> {
        match self {
            Self::Ready(probability) => probability.clone(),
            Self::Deferred { problem, target } => problem.probability(*target).await,
        }
    }
}

pub(super) struct ForcedGapState {
    pub(super) symbolic: PauliFrameSymbolicPropagator,
    pub(super) scores: HashMap<CorrectionBasis, ForcedGapScore>,
    pub(super) history: HashMap<u64, Vec<RecordedGapEdge>>,
}

impl ForcedGapState {
    pub(super) fn new() -> Self {
        Self {
            symbolic: PauliFrameSymbolicPropagator::new(),
            scores: HashMap::new(),
            history: HashMap::new(),
        }
    }

    pub(super) fn reset(&mut self) {
        self.symbolic.reset();
        self.scores.clear();
        self.history.clear();
    }

    pub(super) fn register(&mut self, targets: &[CorrectionBasis], scores: Vec<ForcedGapScore>) {
        assert_eq!(targets.len(), scores.len());
        for (&target, score) in targets.iter().zip(scores) {
            assert!(
                self.scores.insert(target, score).is_none(),
                "forced-gap target registered more than once for {target:?}"
            );
        }
    }

    fn readout_scores(&self, gid: u64) -> Vec<Vec<ForcedGapScore>> {
        self.symbolic
            .readout_components(gid)
            .into_iter()
            .map(|components| {
                components
                    .into_iter()
                    .filter_map(|target| self.scores.get(&target).cloned())
                    .collect()
            })
            .collect()
    }

    pub(super) fn output_scores(&self, gid: u64) -> Vec<Vec<ForcedGapScore>> {
        let components = self.symbolic.output_components(gid);
        if (0..components.len()).any(|index| !self.scores.contains_key(&CorrectionBasis::Residual { gid, index })) {
            return vec![];
        }
        components
            .into_iter()
            .map(|components| {
                components
                    .into_iter()
                    .filter_map(|target| self.scores.get(&target).cloned())
                    .collect()
            })
            .collect()
    }
}

impl WindowCoordinator {
    pub(super) async fn wait_for_forced_gap_scores(&self, gid: u64, token: CancellationToken) -> Result<Vec<f64>, Status> {
        let state = self.forced_gap_state.as_ref().unwrap();
        let readouts = state.read().await.readout_scores(gid);
        Self::resolve_forced_gap_scores(readouts, token).await
    }

    pub(super) async fn resolve_forced_gap_scores(
        readouts: Vec<Vec<ForcedGapScore>>,
        token: CancellationToken,
    ) -> Result<Vec<f64>, Status> {
        let solve = try_join_all(readouts.into_iter().map(|scores| async move {
            let probabilities = try_join_all(scores.iter().map(ForcedGapScore::probability)).await?;
            Ok::<_, Status>(probabilities.into_iter().fold(0.0, f64::max))
        }));
        tokio::select! {
            results = solve => results,
            () = token.cancelled() => {
                Err(Status::cancelled("forced-gap scoring cancelled by reset"))
            }
        }
    }

    pub(super) async fn register_forced_gap_scores(
        &self,
        logical_targets: &[CorrectionBasis],
        problem: &Result<Arc<CausalGapProblem>, Status>,
    ) {
        let scores = (0..logical_targets.len())
            .map(|target| match problem {
                Ok(problem) => ForcedGapScore::Deferred {
                    problem: Arc::clone(problem),
                    target,
                },
                Err(error) => ForcedGapScore::Ready(Err(error.clone())),
            })
            .collect();
        self.forced_gap_state
            .as_ref()
            .unwrap()
            .write()
            .await
            .register(logical_targets, scores);
    }

    pub(super) async fn complete_eager_scores(&self, logical_targets: &[CorrectionBasis], problem: &CausalGapProblem) {
        let probabilities = join_all((0..logical_targets.len()).map(|target| problem.probability(target))).await;
        let mut state = self.forced_gap_state.as_ref().unwrap().write().await;
        for (&target, probability) in logical_targets.iter().zip(probabilities) {
            *state.scores.get_mut(&target).unwrap() = ForcedGapScore::Ready(probability);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn causal_gap_problem(
        &self,
        hypergraph: DecodingHypergraph,
        syndrome: BitVector,
        decoder_seed: Option<u64>,
        baseline: &ParityFactor,
        errors: &ProjectedErrors,
        committing_cids: &HashSet<u64>,
        logical_targets: &[CorrectionBasis],
        window_gids: &HashSet<u64>,
        relative_program: &RelativeProgram,
        mapping: &RelativeMapping,
        error_models: &HashMap<u64, ErrorModel>,
    ) -> Result<CausalGapProblem, Status> {
        let mut detector_keys = vec![(0, 0); usize::try_from(hypergraph.vertex_num).unwrap()];
        for gadget in &relative_program.local_gadgets {
            if let Some(check_model) = &gadget.check_model {
                let local_cid = usize::try_from(check_model.cid).unwrap();
                let start = mapping.start_indices[local_cid];
                for check in 0..check_model.count_checks {
                    detector_keys[start + check] = (mapping.global_cid_of[local_cid], u64::try_from(check).unwrap());
                }
            }
        }
        let vertices = detector_keys
            .iter()
            .copied()
            .enumerate()
            .map(|(index, key)| (key, u64::try_from(index).unwrap()))
            .collect();
        let selected: HashSet<_> = baseline.subgraph.iter().copied().collect();
        let mut edges = Vec::with_capacity(errors.len());
        let mut committed = HashMap::<u64, Vec<RecordedGapEdge>>::new();
        {
            let error_types = self.error_model_types.read().await;
            let check_models = self.check_models.read().await;
            for (index, hyperedge) in hypergraph.hyperedges.iter().enumerate() {
                let reference = &errors[index];
                let model = &error_models[&mapping.global_eid_of[reference.eid]];
                let error = &error_types[&model.instance.etype].errors[reference.error_index];
                let gid = check_models[&model.instance.cid].instance.gid;
                let edge = RecordedGapEdge {
                    gid,
                    checks: hyperedge
                        .vertices
                        .iter()
                        .map(|&vertex| detector_keys[usize::try_from(vertex).unwrap()])
                        .collect(),
                    probability: hyperedge.probability,
                    residual: error.residual.clone(),
                    readouts: error.readout_flips.clone(),
                    selected: selected.contains(&u64::try_from(index).unwrap()),
                };
                if committing_cids.contains(&model.instance.cid) {
                    committed.entry(gid).or_default().push(edge.clone());
                }
                edges.push(edge);
            }
        }
        let mut snapshot = CausalGapSnapshot {
            hypergraph,
            syndrome,
            baseline: baseline.clone(),
            edges,
        };
        let mut state = self.forced_gap_state.as_ref().unwrap().write().await;
        snapshot.restore_history(
            mapping
                .global_gid_of
                .iter()
                .filter_map(|gid| state.history.get(gid))
                .flatten()
                .cloned(),
            &vertices,
        );
        let logical_cache = state
            .symbolic
            .logical_flip_cache(window_gids.iter().copied(), logical_targets);
        let logical_flips: Vec<_> = snapshot
            .edges
            .iter()
            .map(|edge| logical_cache.propagated_logical_flips(edge.gid, &edge.residual, &edge.readouts))
            .collect();
        let mut by_gadget = HashMap::new();
        let mut problems = Vec::with_capacity(logical_targets.len());
        for target in logical_targets {
            let (CorrectionBasis::Readout { gid, .. } | CorrectionBasis::Residual { gid, .. }) = *target;
            let problem = by_gadget.entry(gid).or_insert_with(|| {
                let ancestors = state.symbolic.causal_gadgets(gid, window_gids);
                let retained: Vec<_> = snapshot.edges.iter().map(|edge| ancestors.contains(&edge.gid)).collect();
                CommitRegionDecoder::new(&snapshot.hypergraph, &logical_flips, &retained, logical_targets.len(), false)
                    .problem(
                        self.gap_decoder().clone(),
                        &snapshot.hypergraph,
                        snapshot.syndrome.clone(),
                        decoder_seed,
                        &snapshot.baseline,
                        vec![],
                        false,
                    )
                    .map(Arc::new)
            });
            problems.push(problem.clone()?);
        }
        state.history.extend(committed);
        Ok(CausalGapProblem { problems })
    }
}
