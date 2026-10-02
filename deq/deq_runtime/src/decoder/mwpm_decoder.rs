//! Minimum-weight perfect matching decoder backed by `fusion-blossom`.
//!
//! Degree-one errors are connected to virtual boundary vertices, degree-two
//! errors become ordinary graph edges, and all other hyperedges are excluded
//! from the matching graph.

use crate::decoder::blackbox_decoder::{self, DecodingHypergraph, ParityFactor};
use crate::decoder::thread_pooling::{
    DecodeError, DecodeRequest, DecoderInstance, ThreadPoolingConfig, ThreadPoolingDecoder,
};
use crate::misc::bit_vector::to_sparse_indices;
use crate::misc::union_find::ExampleUnionFind;
use fusion_blossom::mwpm_solver::{PrimalDualSolver, SolverSerial};
use fusion_blossom::util::{SolverInitializer, SyndromePattern, Weight};
use hashbrown::{HashMap, HashSet};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
#[cfg(feature = "cli")]
use structdoc::StructDoc;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct MwpmDecoderConfig {
    #[serde(flatten)]
    pub thread_pooling_config: ThreadPoolingConfig,
    /// largest half-weight used when scaling probability log-odds
    #[serde(default = "default_max_half_weight")]
    pub max_half_weight: u32,
}

fn default_max_half_weight() -> u32 {
    500
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum SolverEdgeEndpoints {
    Boundary(usize),
    Pair(usize, usize),
}

struct SolverEdge {
    endpoints: SolverEdgeEndpoints,
    hyperedge: u64,
    log_odds: f64,
}

struct GraphComponents {
    component_by_vertex: Vec<usize>,
    components_with_virtual_vertices: HashSet<usize>,
}

pub struct MwpmDecoderInstance {
    solver: Option<SolverSerial>,
    solver_edge_to_hyperedge: Vec<u64>,
    components: GraphComponents,
}

impl DecoderInstance for MwpmDecoderInstance {
    fn validate_hypergraph(hypergraph: &DecodingHypergraph, _config: &serde_json::Value) -> Result<(), String> {
        for (edge_index, hyperedge) in hypergraph.hyperedges.iter().enumerate() {
            if hyperedge.vertices.len() <= 2 && hyperedge.probability > 0.5 {
                return Err(format!(
                    "MWPM hyperedge {edge_index} probability must not exceed 0.5, got {}",
                    hyperedge.probability
                ));
            }
        }
        Ok(())
    }

    fn new(hypergraph: &DecodingHypergraph, config: &serde_json::Value) -> Self {
        let config: MwpmDecoderConfig = serde_json::from_value(config.clone()).unwrap();
        assert!(config.max_half_weight > 0, "max_half_weight must be positive");

        let (initializer, solver_edge_to_hyperedge) = build_initializer(hypergraph, config.max_half_weight);
        let components = GraphComponents::new(&initializer, hypergraph.vertex_num);
        let solver = (!initializer.weighted_edges.is_empty()).then(|| SolverSerial::new(&initializer));
        Self {
            solver,
            solver_edge_to_hyperedge,
            components,
        }
    }

    fn decode(&mut self, request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
        let defect_vertices: Vec<usize> = to_sparse_indices(request.syndrome)
            .into_iter()
            .map(|vertex| {
                usize::try_from(vertex)
                    .map_err(|_| DecodeError::InvalidInput(format!("syndrome vertex {vertex} does not fit in usize")))
            })
            .collect::<Result<_, _>>()?;
        self.components.validate_defects(&defect_vertices)?;

        if defect_vertices.is_empty() {
            return Ok(ParityFactor { subgraph: vec![] });
        }
        let solver = self.solver.as_mut().ok_or_else(|| {
            DecodeError::Backend("MWPM found no usable degree-one or degree-two edges for the nonzero syndrome".to_string())
        })?;
        solver.solve(&SyndromePattern::new_vertices(defect_vertices));
        let solver_edges = solver.subgraph();
        let mut subgraph = Vec::with_capacity(solver_edges.len());
        for solver_edge in solver_edges {
            let hyperedge = self.solver_edge_to_hyperedge.get(solver_edge).ok_or_else(|| {
                DecodeError::Backend(format!(
                    "fusion-blossom returned edge {solver_edge}, but the matching graph has {} edges",
                    self.solver_edge_to_hyperedge.len()
                ))
            })?;
            subgraph.push(*hyperedge);
        }
        subgraph.sort_unstable();
        Ok(blackbox_decoder::ParityFactor { subgraph })
    }

    fn reset(&mut self) {
        if let Some(solver) = &mut self.solver {
            solver.clear();
        }
    }
}

impl GraphComponents {
    fn new(initializer: &SolverInitializer, detector_num: u64) -> Self {
        let mut union_find = ExampleUnionFind::new(initializer.vertex_num);
        for &(left, right, _) in &initializer.weighted_edges {
            union_find.union(left, right);
        }

        let mut components_with_virtual_vertices = HashSet::with_capacity(initializer.virtual_vertices.len());
        for &virtual_vertex in &initializer.virtual_vertices {
            let component = union_find.find(virtual_vertex);
            components_with_virtual_vertices.insert(component);
        }
        let detector_num = usize::try_from(detector_num).expect("detector count does not fit in usize");
        Self {
            component_by_vertex: (0..detector_num).map(|vertex| union_find.find(vertex)).collect(),
            components_with_virtual_vertices,
        }
    }

    fn validate_defects(&self, defect_vertices: &[usize]) -> Result<(), DecodeError> {
        let mut odd_components = HashSet::with_capacity(defect_vertices.len());
        for &vertex in defect_vertices {
            let component = self.component_by_vertex[vertex];
            if !odd_components.insert(component) {
                odd_components.remove(&component);
            }
        }
        if odd_components
            .iter()
            .any(|component| !self.components_with_virtual_vertices.contains(component))
        {
            return Err(DecodeError::Backend(
                "syndrome cannot be matched after excluding non-graph hyperedges".to_string(),
            ));
        }
        Ok(())
    }
}

fn build_initializer(hypergraph: &DecodingHypergraph, max_half_weight: u32) -> (SolverInitializer, Vec<u64>) {
    let detector_num = usize::try_from(hypergraph.vertex_num).expect("detector count does not fit in usize");
    let mut solver_edges = Vec::<SolverEdge>::new();
    let mut edge_positions = HashMap::<SolverEdgeEndpoints, usize>::new();

    for (hyperedge_index, hyperedge) in hypergraph.hyperedges.iter().enumerate() {
        if hyperedge.probability == 0.0 || !(1..=2).contains(&hyperedge.vertices.len()) {
            continue;
        }
        let log_odds = (-hyperedge.probability).ln_1p() - hyperedge.probability.ln();
        let hyperedge_index = u64::try_from(hyperedge_index).expect("hyperedge index does not fit in u64");
        let endpoints = if hyperedge.vertices.len() == 1 {
            SolverEdgeEndpoints::Boundary(
                usize::try_from(hyperedge.vertices[0]).expect("detector index does not fit in usize"),
            )
        } else {
            let left = usize::try_from(hyperedge.vertices[0]).expect("detector index does not fit in usize");
            let right = usize::try_from(hyperedge.vertices[1]).expect("detector index does not fit in usize");
            if left < right {
                SolverEdgeEndpoints::Pair(left, right)
            } else {
                SolverEdgeEndpoints::Pair(right, left)
            }
        };
        if let Some(&solver_edge_index) = edge_positions.get(&endpoints) {
            if log_odds < solver_edges[solver_edge_index].log_odds {
                solver_edges[solver_edge_index].hyperedge = hyperedge_index;
                solver_edges[solver_edge_index].log_odds = log_odds;
            }
        } else {
            edge_positions.insert(endpoints, solver_edges.len());
            solver_edges.push(SolverEdge {
                endpoints,
                hyperedge: hyperedge_index,
                log_odds,
            });
        }
    }

    let boundary_edge_count = solver_edges
        .iter()
        .filter(|edge| matches!(edge.endpoints, SolverEdgeEndpoints::Boundary(_)))
        .count();
    let vertex_num = detector_num
        .checked_add(boundary_edge_count)
        .expect("matching graph vertex count overflowed usize");
    let vertex_num_as_weight =
        Weight::try_from(vertex_num.max(1)).expect("matching graph vertex count does not fit in Weight");
    let maximum_safe_half_weight = Weight::MAX / vertex_num_as_weight / 2;
    assert!(
        maximum_safe_half_weight > 0,
        "matching graph is too large for fusion-blossom weights"
    );
    let configured_max_half_weight = Weight::try_from(max_half_weight).unwrap_or(Weight::MAX);
    let effective_max_half_weight = configured_max_half_weight.min(maximum_safe_half_weight);
    let maximum_log_odds = solver_edges.iter().map(|edge| edge.log_odds).fold(0.0_f64, f64::max);
    let mut next_virtual_vertex = detector_num;
    let mut virtual_vertices = Vec::with_capacity(boundary_edge_count);
    let weighted_edges = solver_edges
        .iter()
        .map(|edge| {
            let vertices = match edge.endpoints {
                SolverEdgeEndpoints::Boundary(detector) => {
                    let virtual_vertex = next_virtual_vertex;
                    next_virtual_vertex += 1;
                    virtual_vertices.push(virtual_vertex);
                    (detector, virtual_vertex)
                }
                SolverEdgeEndpoints::Pair(left, right) => (left, right),
            };
            let half_weight = if edge.log_odds == 0.0 {
                0
            } else {
                (effective_max_half_weight.to_f64().expect("Weight does not fit in f64") * edge.log_odds / maximum_log_odds)
                    .round()
                    .to_isize()
                    .expect("scaled MWPM weight does not fit in isize")
                    .max(1)
            };
            let weight = half_weight.checked_mul(2).expect("scaled MWPM weight overflowed");
            (vertices.0, vertices.1, weight)
        })
        .collect();
    debug_assert_eq!(next_virtual_vertex, vertex_num);
    let solver_edge_to_hyperedge = solver_edges.iter().map(|edge| edge.hyperedge).collect();

    (
        SolverInitializer {
            vertex_num,
            weighted_edges,
            virtual_vertices,
        },
        solver_edge_to_hyperedge,
    )
}

pub type MwpmDecoder = ThreadPoolingDecoder<MwpmDecoderInstance>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::blackbox_decoder::Hyperedge;
    use crate::misc::bit_vector::from_sparse_indices;
    use serde_json::json;

    fn decode(hypergraph: &DecodingHypergraph, defect_vertices: &[u64]) -> Result<ParityFactor, DecodeError> {
        let mut decoder = MwpmDecoderInstance::new(hypergraph, &json!({}));
        let syndrome = from_sparse_indices(hypergraph.vertex_num, defect_vertices);
        decoder.decode(DecodeRequest {
            syndrome: &syndrome,
            reweights: &[],
            loss: None,
        })
    }

    #[test]
    fn ignores_hyperedges_and_preserves_original_edge_indices() {
        let hypergraph = DecodingHypergraph {
            vertex_num: 3,
            hyperedges: vec![
                Hyperedge {
                    vertices: vec![0, 1, 2],
                    probability: 1.0,
                },
                Hyperedge {
                    vertices: vec![0, 1],
                    probability: 0.1,
                },
            ],
        };

        assert_eq!(decode(&hypergraph, &[0, 1]).unwrap().subgraph, vec![1]);
        assert!(matches!(
            decode(&hypergraph, &[0, 1, 2]),
            Err(DecodeError::Backend(message))
                if message.contains("excluding non-graph hyperedges")
        ));
    }

    #[test]
    fn decodes_degree_one_edges_through_virtual_boundaries() {
        let hypergraph = DecodingHypergraph {
            vertex_num: 2,
            hyperedges: vec![
                Hyperedge {
                    vertices: vec![0],
                    probability: 0.1,
                },
                Hyperedge {
                    vertices: vec![1],
                    probability: 0.2,
                },
            ],
        };

        assert_eq!(decode(&hypergraph, &[1]).unwrap().subgraph, vec![1]);
    }

    #[test]
    fn parallel_edges_use_the_most_likely_original_edge() {
        let hypergraph = DecodingHypergraph {
            vertex_num: 2,
            hyperedges: vec![
                Hyperedge {
                    vertices: vec![0, 1],
                    probability: 0.1,
                },
                Hyperedge {
                    vertices: vec![1, 0],
                    probability: 0.2,
                },
            ],
        };

        assert_eq!(decode(&hypergraph, &[0, 1]).unwrap().subgraph, vec![1]);
    }

    #[test]
    fn degree_one_edges_are_deduplicated_before_adding_virtual_vertices() {
        let hypergraph = DecodingHypergraph {
            vertex_num: 1,
            hyperedges: vec![
                Hyperedge {
                    vertices: vec![0],
                    probability: 0.1,
                },
                Hyperedge {
                    vertices: vec![0],
                    probability: 0.2,
                },
            ],
        };

        let (initializer, solver_edge_to_hyperedge) = build_initializer(&hypergraph, default_max_half_weight());
        assert_eq!(initializer.vertex_num, 2);
        assert_eq!(initializer.virtual_vertices, vec![1]);
        assert_eq!(initializer.weighted_edges.len(), 1);
        assert_eq!(solver_edge_to_hyperedge, vec![1]);
    }

    #[test]
    fn configured_weight_is_capped_at_fusion_blossom_safe_bound() {
        let hypergraph = DecodingHypergraph {
            vertex_num: 2,
            hyperedges: vec![Hyperedge {
                vertices: vec![0, 1],
                probability: 0.1,
            }],
        };

        let (initializer, _) = build_initializer(&hypergraph, u32::MAX);
        let maximum_safe_weight = Weight::MAX / Weight::try_from(initializer.vertex_num).unwrap();
        assert!(initializer.weighted_edges[0].2 <= maximum_safe_weight);
    }

    #[test]
    fn half_probability_edges_are_supported() {
        let hypergraph = DecodingHypergraph {
            vertex_num: 1,
            hyperedges: vec![Hyperedge {
                vertices: vec![0],
                probability: 0.5,
            }],
        };

        assert_eq!(decode(&hypergraph, &[0]).unwrap().subgraph, vec![0]);
    }
}
