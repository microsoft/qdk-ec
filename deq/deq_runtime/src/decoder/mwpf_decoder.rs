//! Minimum-weight parity factor decoder backed by the public `mwpf` crate.

use crate::decoder::blackbox_decoder::{DecodingHypergraph, ParityFactor};
use crate::decoder::thread_pooling::{
    DecodeError, DecodeRequest, DecoderInstance, ThreadPoolingConfig, ThreadPoolingDecoder,
};
use crate::misc::bit_vector::to_sparse_indices;
use hashbrown::HashMap;
use mwpf::mwpf_solver::{SolverSerialJointSingleHair, SolverTrait};
use mwpf::ordered_float::OrderedFloat;
use mwpf::util::{HyperEdge, SolverInitializer, SyndromePattern};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::sync::Arc;
#[cfg(feature = "cli")]
use structdoc::StructDoc;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct MwpfDecoderConfig {
    #[serde(flatten)]
    pub thread_pooling_config: ThreadPoolingConfig,
    /// timeout in seconds for each decoding problem
    #[serde(default = "default_timeout")]
    pub timeout: f64,
    /// maximum dual variables in a cluster before falling back to union-find
    #[serde(default = "default_cluster_node_limit")]
    pub cluster_node_limit: usize,
    /// defer solving the primal problem until the end of decoding
    #[serde(default)]
    pub only_solve_primal_once: bool,
}

pub(crate) fn default_timeout() -> f64 {
    f64::MAX
}

fn default_cluster_node_limit() -> usize {
    usize::try_from((2u64 << 53) - 1).unwrap_or(usize::MAX)
}

impl MwpfDecoderConfig {
    fn solver_config(&self) -> serde_json::Value {
        serde_json::json!({
            "timeout": self.timeout,
            "cluster_node_limit": self.cluster_node_limit,
            "only_solve_primal_once": self.only_solve_primal_once,
        })
    }
}

pub struct MwpfDecoderInstance {
    solver: Option<SolverSerialJointSingleHair>,
    solver_edge_to_hyperedge: Vec<u64>,
    vertex_to_solver: Vec<Option<usize>>,
}

impl MwpfDecoderInstance {
    pub(crate) fn new_with_config(hypergraph: &DecodingHypergraph, config: &MwpfDecoderConfig) -> Self {
        let (initializer, solver_edge_to_hyperedge, vertex_to_solver) = build_initializer(hypergraph);
        let solver = (!initializer.weighted_edges.is_empty())
            .then(|| SolverSerialJointSingleHair::new(&Arc::new(initializer), config.solver_config()));
        Self {
            solver,
            solver_edge_to_hyperedge,
            vertex_to_solver,
        }
    }
}

impl DecoderInstance for MwpfDecoderInstance {
    fn validate_hypergraph(hypergraph: &DecodingHypergraph, _config: &serde_json::Value) -> Result<(), String> {
        for (edge_index, hyperedge) in hypergraph.hyperedges.iter().enumerate() {
            if hyperedge.probability > 0.5 {
                return Err(format!(
                    "MWPF hyperedge {edge_index} probability must not exceed 0.5, got {}",
                    hyperedge.probability
                ));
            }
        }
        Ok(())
    }

    fn new(hypergraph: &DecodingHypergraph, config: &serde_json::Value) -> Self {
        let config: MwpfDecoderConfig = serde_json::from_value(config.clone()).unwrap();
        Self::new_with_config(hypergraph, &config)
    }

    fn decode(&mut self, request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
        let defect_vertices: Vec<usize> = to_sparse_indices(request.syndrome)
            .into_iter()
            .map(|vertex| {
                let vertex = usize::try_from(vertex)
                    .map_err(|_| DecodeError::InvalidInput(format!("syndrome vertex {vertex} does not fit in usize")))?;
                self.vertex_to_solver[vertex].ok_or_else(|| {
                    DecodeError::Backend(format!(
                        "syndrome contains isolated vertex {vertex}, which has no positive-probability incident edge"
                    ))
                })
            })
            .collect::<Result<_, _>>()?;
        if defect_vertices.is_empty() {
            return Ok(ParityFactor { subgraph: vec![] });
        }
        let solver = self.solver.as_mut().ok_or_else(|| {
            DecodeError::Backend("MWPF found no positive-probability edges for the nonzero syndrome".to_string())
        })?;
        solver.solve(SyndromePattern::new_vertices(defect_vertices));
        let mut subgraph = Vec::new();
        for solver_edge in solver.subgraph() {
            let hyperedge = self.solver_edge_to_hyperedge.get(solver_edge).ok_or_else(|| {
                DecodeError::Backend(format!(
                    "MWPF returned edge {solver_edge}, but the solver hypergraph has {} edges",
                    self.solver_edge_to_hyperedge.len()
                ))
            })?;
            subgraph.push(*hyperedge);
        }
        subgraph.sort_unstable();
        Ok(ParityFactor { subgraph })
    }

    fn reset(&mut self) {
        if let Some(solver) = &mut self.solver {
            solver.clear();
        }
    }
}

fn build_initializer(hypergraph: &DecodingHypergraph) -> (SolverInitializer, Vec<u64>, Vec<Option<usize>>) {
    let original_vertex_num = usize::try_from(hypergraph.vertex_num).expect("detector count does not fit in usize");
    let active_vertices: BTreeSet<usize> = hypergraph
        .hyperedges
        .iter()
        .filter(|hyperedge| hyperedge.probability > 0.0 && !hyperedge.vertices.is_empty())
        .flat_map(|hyperedge| {
            hyperedge
                .vertices
                .iter()
                .map(|&vertex| usize::try_from(vertex).expect("detector index does not fit in usize"))
        })
        .collect();
    let mut vertex_to_solver = vec![None; original_vertex_num];
    for (solver_vertex, original_vertex) in active_vertices.iter().copied().enumerate() {
        vertex_to_solver[original_vertex] = Some(solver_vertex);
    }

    let mut weighted_edges: Vec<HyperEdge> = Vec::with_capacity(hypergraph.hyperedges.len());
    let mut solver_edge_to_hyperedge = Vec::with_capacity(hypergraph.hyperedges.len());
    let mut edge_positions = HashMap::<Vec<usize>, usize>::new();
    for (hyperedge_index, hyperedge) in hypergraph.hyperedges.iter().enumerate() {
        if hyperedge.probability == 0.0 || hyperedge.vertices.is_empty() {
            continue;
        }
        let mut vertices: Vec<usize> = hyperedge
            .vertices
            .iter()
            .map(|&vertex| {
                let vertex = usize::try_from(vertex).expect("detector index does not fit in usize");
                vertex_to_solver[vertex].expect("active detector is missing from the solver mapping")
            })
            .collect();
        vertices.sort_unstable();
        let log_odds = (-hyperedge.probability).ln_1p() - hyperedge.probability.ln();
        let weight = OrderedFloat::new(log_odds);
        let original_index = u64::try_from(hyperedge_index).expect("hyperedge index does not fit in u64");
        // MWPF rejects parallel edges. For nonnegative weights, keeping the
        // cheapest representative preserves the minimum-weight parity objective.
        if let Some(&position) = edge_positions.get(&vertices) {
            if weight < weighted_edges[position].weight {
                weighted_edges[position].weight = weight;
                solver_edge_to_hyperedge[position] = original_index;
            }
        } else {
            edge_positions.insert(vertices.clone(), weighted_edges.len());
            weighted_edges.push(HyperEdge::new(vertices, weight));
            solver_edge_to_hyperedge.push(original_index);
        }
    }
    (
        SolverInitializer::new(active_vertices.len(), weighted_edges),
        solver_edge_to_hyperedge,
        vertex_to_solver,
    )
}

pub type MwpfDecoder = ThreadPoolingDecoder<MwpfDecoderInstance>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::blackbox_decoder::Hyperedge;
    use crate::misc::bit_vector::from_sparse_indices;
    use serde_json::json;

    fn decode(hypergraph: &DecodingHypergraph, defect_vertices: &[u64]) -> Result<ParityFactor, DecodeError> {
        let mut decoder = MwpfDecoderInstance::new(hypergraph, &json!({}));
        let syndrome = from_sparse_indices(hypergraph.vertex_num, defect_vertices);
        decoder.decode(DecodeRequest {
            syndrome: &syndrome,
            decoder_seed: None,
            reweights: &[],
            loss: None,
        })
    }

    #[test]
    fn decodes_degree_three_hyperedge() {
        let hypergraph = DecodingHypergraph {
            vertex_num: 3,
            hyperedges: vec![Hyperedge {
                observable_flips: None,
                vertices: vec![0, 1, 2],
                probability: 0.1,
            }],
        };

        assert_eq!(decode(&hypergraph, &[0, 1, 2]).unwrap().subgraph, vec![0]);
    }

    #[test]
    fn preserves_original_indices_when_zero_probability_edges_are_omitted() {
        let hypergraph = DecodingHypergraph {
            vertex_num: 3,
            hyperedges: vec![
                Hyperedge {
                    observable_flips: None,
                    vertices: vec![0, 1, 2],
                    probability: 0.0,
                },
                Hyperedge {
                    observable_flips: None,
                    vertices: vec![0, 1, 2],
                    probability: 0.1,
                },
            ],
        };

        assert_eq!(decode(&hypergraph, &[0, 1, 2]).unwrap().subgraph, vec![1]);
    }

    #[test]
    fn half_probability_hyperedges_are_supported() {
        let hypergraph = DecodingHypergraph {
            vertex_num: 3,
            hyperedges: vec![Hyperedge {
                observable_flips: None,
                vertices: vec![0, 1, 2],
                probability: 0.5,
            }],
        };

        assert_eq!(decode(&hypergraph, &[0, 1, 2]).unwrap().subgraph, vec![0]);
    }

    #[test]
    fn parallel_edges_keep_cheapest_original_index() {
        let hypergraph = DecodingHypergraph {
            vertex_num: 3,
            hyperedges: vec![
                Hyperedge {
                    observable_flips: None,
                    vertices: vec![0, 1, 2],
                    probability: 0.01,
                },
                Hyperedge {
                    observable_flips: None,
                    vertices: vec![0, 1, 2],
                    probability: 0.2,
                },
                Hyperedge {
                    observable_flips: None,
                    vertices: vec![0, 1, 2],
                    probability: 0.1,
                },
            ],
        };
        let (initializer, mapping, _) = build_initializer(&hypergraph);
        assert_eq!(initializer.weighted_edges.len(), 1);
        assert_eq!(mapping, vec![1]);
        assert_eq!(decode(&hypergraph, &[0, 1, 2]).unwrap().subgraph, vec![1]);
    }

    #[test]
    fn parallel_edges_are_canonicalized_and_ties_keep_first_index() {
        let hypergraph = DecodingHypergraph {
            vertex_num: 4,
            hyperedges: vec![
                Hyperedge {
                    observable_flips: None,
                    vertices: vec![3, 0, 2],
                    probability: 0.1,
                },
                Hyperedge {
                    observable_flips: None,
                    vertices: vec![2, 3, 0],
                    probability: 0.1,
                },
            ],
        };
        let (initializer, mapping, _) = build_initializer(&hypergraph);
        assert_eq!(initializer.weighted_edges.len(), 1);
        assert_eq!(initializer.weighted_edges[0].vertices, vec![0, 1, 2]);
        assert_eq!(mapping, vec![0]);
        assert_eq!(decode(&hypergraph, &[0, 2, 3]).unwrap().subgraph, vec![0]);
    }
}
