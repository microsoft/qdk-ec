//! Weighted hypergraph union-find decoder.
//!
//! Each invalid cluster grows its untight incident edges at unit rate. Tight
//! edges merge clusters; exact GF(2) column-space membership decides validity.
//! No history of dual subgraphs is needed because this engine never shrinks.
//!
//! `binar` performs the repeated span-membership checks during cluster growth.
//! After growth finishes, MWPF's matrix solver is built once per cluster to
//! choose a weight-local-minimum correction; `binar` returns an arbitrary
//! feasible solution, while using MWPF matrices during growth would make
//! frequent cluster merges more expensive and complicated.

use super::blackbox_decoder::{DecodingHypergraph, ParityFactor};
use super::thread_pooling::{DecodeError, DecodeRequest, DecoderInstance, ThreadPoolingConfig, ThreadPoolingDecoder};
use crate::misc::bit_vector::to_sparse_indices;
use binar::{BitMatrix, BitVec, Bitwise, BitwiseMut, BitwisePairMut, EchelonForm};
use hashbrown::{HashMap, HashSet};
use mwpf::matrix::{BasicMatrix, Echelon, MatrixBasic, MatrixEchelon};
use mwpf::ordered_float::OrderedFloat;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::time::Instant;
#[cfg(feature = "cli")]
use structdoc::StructDoc;

#[derive(Clone)]
struct Edge {
    vertices: Vec<usize>,
    weight: f64,
    original: u64,
}

struct CompactHuf {
    vertex_num: usize,
    edges: Vec<Edge>,
    adjacency: Vec<Vec<usize>>,
    timeout: f64,
}

#[derive(Default)]
struct TightEdgeSpan(Vec<Vec<usize>>);

impl TightEdgeSpan {
    fn insert(&mut self, vertices: Vec<usize>) {
        self.0.push(vertices);
    }

    fn contains(&self, rhs: &BitVec) -> bool {
        if self.0.is_empty() {
            return rhs.is_zero();
        }
        let columns: Vec<_> = self
            .0
            .iter()
            .map(|vertices| {
                let mut column = BitVec::zeros(rhs.len());
                for &vertex in vertices {
                    column.assign_index(vertex, true);
                }
                column
            })
            .collect();
        let transposed = BitMatrix::from_row_iter(columns.iter().map(|column| column.as_view()), rhs.len());
        EchelonForm::new(transposed).transpose_solve(&rhs.as_view()).is_some()
    }
}

struct Cluster {
    vertices: Vec<usize>,
    edges: Vec<usize>,
    boundary: HashSet<usize>,
    tight_edge_span: TightEdgeSpan,
    syndrome: BitVec,
    invalid: bool,
}

#[derive(Clone, Copy)]
struct Event {
    time: f64,
    edge: usize,
    version: u64,
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Event {}
impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        self.time
            .total_cmp(&other.time)
            .then(self.edge.cmp(&other.edge))
            .then(self.version.cmp(&other.version))
    }
}

struct Growth {
    slack: f64,
    updated: f64,
    speed: usize,
    version: u64,
    tight: bool,
}

fn root(parent: &mut [usize], mut vertex: usize) -> usize {
    while parent[vertex] != vertex {
        parent[vertex] = parent[parent[vertex]];
        vertex = parent[vertex];
    }
    vertex
}

fn speed_as_f64(speed: usize) -> f64 {
    speed.to_f64().expect("hyperedge degree does not fit in f64")
}

impl CompactHuf {
    fn new(graph: &DecodingHypergraph, timeout: f64) -> Self {
        let vertex_num = usize::try_from(graph.vertex_num).expect("vertex count fits usize");
        let mut edges: Vec<Edge> = vec![];
        let mut positions = HashMap::<Vec<usize>, usize>::new();
        for (index, edge) in graph.hyperedges.iter().enumerate() {
            if edge.probability == 0.0 || edge.vertices.is_empty() {
                continue;
            }
            let mut vertices: Vec<_> = edge
                .vertices
                .iter()
                .map(|&v| usize::try_from(v).expect("vertex fits usize"))
                .collect();
            vertices.sort_unstable();
            let weight = (-edge.probability).ln_1p() - edge.probability.ln();
            if let Some(&position) = positions.get(&vertices) {
                if weight < edges[position].weight {
                    edges[position].weight = weight;
                    edges[position].original = index as u64;
                }
            } else {
                positions.insert(vertices.clone(), edges.len());
                edges.push(Edge {
                    vertices,
                    weight,
                    original: index as u64,
                });
            }
        }
        let mut adjacency = vec![vec![]; vertex_num];
        for (index, edge) in edges.iter().enumerate() {
            for &vertex in &edge.vertices {
                adjacency[vertex].push(index);
            }
        }
        Self {
            vertex_num,
            edges,
            adjacency,
            timeout,
        }
    }

    fn decode(&self, defects: &[usize]) -> Result<Vec<u64>, DecodeError> {
        if defects.is_empty() {
            return Ok(vec![]);
        }
        let started = Instant::now();
        let mut syndrome = vec![false; self.vertex_num];
        for &v in defects {
            if v >= self.vertex_num || syndrome[v] {
                return Err(DecodeError::InvalidInput("invalid or duplicate syndrome vertex".into()));
            }
            syndrome[v] = true;
        }
        let mut parent: Vec<_> = (0..self.vertex_num).collect();
        let mut clusters: Vec<_> = (0..self.vertex_num)
            .map(|v| {
                let mut rhs = BitVec::zeros(self.vertex_num);
                if syndrome[v] {
                    rhs.assign_index(v, true);
                }
                Cluster {
                    vertices: vec![v],
                    edges: vec![],
                    boundary: self.adjacency[v].iter().copied().collect(),
                    tight_edge_span: TightEdgeSpan::default(),
                    syndrome: rhs,
                    invalid: syndrome[v],
                }
            })
            .collect();
        let mut growth: Vec<_> = self
            .edges
            .iter()
            .map(|edge| Growth {
                slack: edge.weight,
                updated: 0.0,
                speed: edge.vertices.iter().filter(|&&v| syndrome[v]).count(),
                version: 0,
                tight: false,
            })
            .collect();
        let mut queue = BinaryHeap::new();
        for (edge, state) in growth.iter().enumerate() {
            if state.speed > 0 {
                queue.push(Reverse(Event {
                    time: state.slack / speed_as_f64(state.speed),
                    edge,
                    version: 0,
                }));
            }
        }
        let mut invalid = defects.len();
        while invalid > 0 {
            if started.elapsed().as_secs_f64() > self.timeout {
                return Err(DecodeError::Backend("compact HUF exceeded configured timeout".into()));
            }
            let event = loop {
                let Some(Reverse(event)) = queue.pop() else {
                    return Err(DecodeError::Backend(
                        "syndrome is not in the span of positive-probability hyperedges".into(),
                    ));
                };
                let state = &growth[event.edge];
                if !state.tight && state.version == event.version {
                    break event;
                }
            };
            let time = event.time;
            growth[event.edge].tight = true;
            let mut roots: Vec<_> = self.edges[event.edge]
                .vertices
                .iter()
                .map(|&v| root(&mut parent, v))
                .collect();
            roots.sort_unstable();
            roots.dedup();
            let owner = *roots
                .iter()
                .max_by_key(|&&r| (clusters[r].vertices.len(), Reverse(r)))
                .unwrap();
            let previous_invalid = clusters[owner].invalid;
            let single_root = roots.len() == 1;
            for &r in &roots {
                invalid -= usize::from(clusters[r].invalid);
            }
            for &other in &roots {
                if other == owner {
                    continue;
                }
                parent[other] = owner;
                let vertices = std::mem::take(&mut clusters[other].vertices);
                let edges = std::mem::take(&mut clusters[other].edges);
                let boundary = std::mem::take(&mut clusters[other].boundary);
                let tight_edge_span = std::mem::take(&mut clusters[other].tight_edge_span);
                let rhs = std::mem::replace(&mut clusters[other].syndrome, BitVec::zeros(self.vertex_num));
                clusters[other].invalid = false;
                let cluster = &mut clusters[owner];
                cluster.vertices.extend(vertices);
                cluster.edges.extend(edges);
                cluster.boundary.extend(boundary);
                cluster.syndrome.bitxor_assign(&rhs);
                cluster.tight_edge_span.0.extend(tight_edge_span.0);
            }
            let cluster = &mut clusters[owner];
            cluster.edges.push(event.edge);
            cluster.boundary.remove(&event.edge);
            cluster.tight_edge_span.insert(self.edges[event.edge].vertices.clone());
            cluster.invalid = !cluster.tight_edge_span.contains(&cluster.syndrome);
            invalid += usize::from(cluster.invalid);
            if single_root && previous_invalid == cluster.invalid {
                continue;
            }
            let mut affected: Vec<_> = cluster.boundary.iter().copied().collect();
            affected.sort_unstable();
            for edge in affected {
                let state = &mut growth[edge];
                if state.tight {
                    continue;
                }
                let mut owners: Vec<_> = self.edges[edge].vertices.iter().map(|&v| root(&mut parent, v)).collect();
                owners.sort_unstable();
                owners.dedup();
                let speed = owners.iter().filter(|&&r| clusters[r].invalid).count();
                if speed == state.speed {
                    continue;
                }
                state.slack = (state.slack - (time - state.updated) * speed_as_f64(state.speed)).max(0.0);
                state.updated = time;
                state.speed = speed;
                state.version += 1;
                if state.speed > 0 {
                    queue.push(Reverse(Event {
                        time: time + state.slack / speed_as_f64(state.speed),
                        edge,
                        version: state.version,
                    }));
                }
            }
            // Bound stale-event storage rather than retaining the full growth history.
            if queue.len() > self.edges.len().saturating_mul(4).max(16) {
                queue.retain(|Reverse(event)| !growth[event.edge].tight && event.version == growth[event.edge].version);
            }
        }
        let mut selected = vec![];
        for cluster in clusters {
            if cluster.edges.is_empty() {
                continue;
            }
            if started.elapsed().as_secs_f64() > self.timeout {
                return Err(DecodeError::Backend("compact HUF exceeded configured timeout".into()));
            }
            let mut matrix = Echelon::<BasicMatrix>::new();
            let mut edges = cluster.edges;
            edges.sort_unstable_by(|&a, &b| {
                self.edges[a]
                    .weight
                    .total_cmp(&self.edges[b].weight)
                    .then(self.edges[a].original.cmp(&self.edges[b].original))
            });
            let set: HashSet<_> = edges.iter().copied().collect();
            for &edge in &edges {
                matrix.add_variable(edge);
            }
            for vertex in cluster.vertices {
                let incident: Vec<_> = self.adjacency[vertex].iter().copied().filter(|e| set.contains(e)).collect();
                matrix.add_constraint(vertex, &incident, syndrome[vertex]);
            }
            let solution = matrix
                .get_solution_local_minimum(|e| OrderedFloat::new(self.edges[e].weight))
                .ok_or_else(|| DecodeError::Backend("compact HUF final cluster lost parity feasibility".into()))?;
            selected.extend(solution);
            if started.elapsed().as_secs_f64() > self.timeout {
                return Err(DecodeError::Backend("compact HUF exceeded configured timeout".into()));
            }
        }
        selected.sort_unstable();
        let mut actual = vec![false; self.vertex_num];
        for &index in &selected {
            let edge = &self.edges[index];
            for &vertex in &edge.vertices {
                actual[vertex] ^= true;
            }
        }
        if actual != syndrome {
            return Err(DecodeError::Backend(
                "compact HUF correction does not reproduce the syndrome".into(),
            ));
        }
        let mut result: Vec<_> = selected.into_iter().map(|index| self.edges[index].original).collect();
        result.sort_unstable();
        Ok(result)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct HufDecoderConfig {
    #[serde(flatten)]
    pub thread_pooling_config: ThreadPoolingConfig,
    /// timeout in seconds for each decoding problem
    #[serde(default = "default_timeout")]
    pub timeout: f64,
}

fn default_timeout() -> f64 {
    f64::MAX
}

pub struct HufDecoderInstance {
    inner: CompactHuf,
}

impl DecoderInstance for HufDecoderInstance {
    fn validate_hypergraph(hypergraph: &DecodingHypergraph, config: &serde_json::Value) -> Result<(), String> {
        let parsed: HufDecoderConfig = serde_json::from_value(config.clone()).map_err(|error| error.to_string())?;
        if !parsed.timeout.is_finite() || parsed.timeout <= 0.0 {
            return Err("timeout must be positive and finite".into());
        }
        for (edge_index, edge) in hypergraph.hyperedges.iter().enumerate() {
            if edge.probability > 0.5 {
                return Err(format!(
                    "HUF hyperedge {edge_index} probability must not exceed 0.5, got {}",
                    edge.probability
                ));
            }
        }
        Ok(())
    }

    fn new(hypergraph: &DecodingHypergraph, config: &serde_json::Value) -> Self {
        let config: HufDecoderConfig = serde_json::from_value(config.clone()).unwrap();
        Self {
            inner: CompactHuf::new(hypergraph, config.timeout),
        }
    }

    fn decode(&mut self, request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
        let defects: Result<Vec<_>, _> = to_sparse_indices(request.syndrome)
            .into_iter()
            .map(|vertex| usize::try_from(vertex).map_err(|_| DecodeError::InvalidInput("syndrome vertex overflow".into())))
            .collect();
        Ok(ParityFactor {
            subgraph: self.inner.decode(&defects?)?,
        })
    }

    fn reset(&mut self) {}
}

pub type HufDecoder = ThreadPoolingDecoder<HufDecoderInstance>;

#[cfg(test)]
mod tests {
    use super::super::blackbox_decoder::Hyperedge;
    use super::super::mwpf_decoder::MwpfDecoderInstance;
    use super::*;
    use crate::misc::bit_vector::from_sparse_indices;

    fn graph(vertex_num: u64, edges: &[(&[u64], f64)]) -> DecodingHypergraph {
        DecodingHypergraph {
            vertex_num,
            hyperedges: edges
                .iter()
                .map(|(vs, p)| Hyperedge {
                    vertices: vs.to_vec(),
                    probability: *p,
                })
                .collect(),
        }
    }

    fn verify(graph: &DecodingHypergraph, defects: &[usize]) -> Vec<u64> {
        let solver = CompactHuf::new(graph, 30.0);
        let result = solver.decode(defects).unwrap();
        verify_correction(graph, defects, &result);
        assert_eq!(solver.decode(defects).unwrap(), result);
        result
    }

    fn verify_correction(graph: &DecodingHypergraph, defects: &[usize], result: &[u64]) {
        let mut actual = vec![false; graph.vertex_num as usize];
        for &edge in result {
            for &vertex in &graph.hyperedges[edge as usize].vertices {
                actual[vertex as usize] ^= true;
            }
        }
        assert_eq!(
            actual
                .iter()
                .enumerate()
                .filter_map(|(i, &set)| set.then_some(i))
                .collect::<Vec<_>>(),
            defects
        );
    }

    #[test]
    fn compact_hyperedges_parallel_edges_and_zero_weights() {
        let input = graph(3, &[(&[0, 1, 2], 0.0), (&[2, 0, 1], 0.01), (&[0, 1, 2], 0.5)]);
        assert_eq!(verify(&input, &[0, 1, 2]), vec![2]);
        assert!(verify(&input, &[]).is_empty());
    }

    #[test]
    fn compact_reports_infeasible_and_disconnected_syndromes() {
        let input = graph(4, &[(&[0, 1], 0.1), (&[2, 3], 0.1)]);
        verify(&input, &[0, 1, 2, 3]);
        assert!(CompactHuf::new(&input, 10.0).decode(&[0]).is_err());
        assert!(CompactHuf::new(&input, 10.0).decode(&[4]).is_err());
        assert!(CompactHuf::new(&input, 10.0).decode(&[0, 0]).is_err());
    }

    #[test]
    fn compact_all_small_syndromes_match_exact_span() {
        let mut rng = 81337u64;
        for n in 1..=7 {
            for _ in 0..20 {
                let mut edges = vec![];
                for _ in 0..12 {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    let mask = rng as usize & ((1 << n) - 1);
                    edges.push(Hyperedge {
                        vertices: (0..n).filter(|i| mask & (1 << i) != 0).map(|i| i as u64).collect(),
                        probability: [0.0, 0.01, 0.1, 0.5][(rng >> 12) as usize % 4],
                    });
                }
                let input = DecodingHypergraph {
                    vertex_num: n as u64,
                    hyperedges: edges,
                };
                let mut reachable: HashSet<usize> = HashSet::from([0usize]);
                for edge in &input.hyperedges {
                    if edge.probability == 0.0 {
                        continue;
                    }
                    let mask = edge.vertices.iter().fold(0usize, |m, &v| m ^ (1 << v));
                    let extra: Vec<_> = reachable.iter().map(|&v| v ^ mask).collect();
                    reachable.extend(extra);
                }
                let solver = CompactHuf::new(&input, 10.0);
                for syndrome in 0..(1 << n) {
                    let defects: Vec<_> = (0..n).filter(|i| syndrome & (1 << i) != 0).collect();
                    assert_eq!(solver.decode(&defects).is_ok(), reachable.contains(&syndrome));
                    if reachable.contains(&syndrome) {
                        verify(&input, &defects);
                    }
                }
            }
        }
    }

    #[test]
    fn compact_handles_bitset_word_boundaries() {
        let input = graph(130, &[(&[0, 63, 64, 129], 0.1), (&[1, 65, 128], 0.2)]);
        verify(&input, &[0, 1, 63, 64, 65, 128, 129]);
    }

    #[test]
    fn compact_weighted_growth_can_prefer_a_two_edge_path() {
        let input = graph(3, &[(&[0, 2], 0.001), (&[0, 1], 0.2), (&[1, 2], 0.2)]);
        assert_eq!(verify(&input, &[0, 2]), vec![1, 2]);
    }

    #[test]
    fn compact_sums_uniform_growth_from_each_invalid_cluster() {
        let probability = |weight: f64| 1.0 / (weight.exp() + 1.0);
        let input = graph(
            2,
            &[
                (&[0, 1], probability(2.0)),
                (&[0], probability(1.5)),
                (&[1], probability(1.5)),
            ],
        );
        assert_eq!(verify(&input, &[0, 1]), vec![0]);
    }

    #[test]
    fn compact_matches_mwpf_union_find_feasibility_on_random_reachable_syndromes() {
        let mut rng = 0x517c_c1b7_2722_0a95u64;
        for vertex_num in 1..=7 {
            for _ in 0..12 {
                let mut hyperedges = vec![];
                for edge_index in 0..12 {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    let mask = (rng as usize & ((1 << vertex_num) - 1)).max(1);
                    hyperedges.push(Hyperedge {
                        vertices: (0..vertex_num)
                            .filter(|vertex| mask & (1 << vertex) != 0)
                            .map(|vertex| vertex as u64)
                            .collect(),
                        probability: [0.01, 0.05, 0.1, 0.2, 0.5][(rng >> 12) as usize % 5],
                    });
                    if edge_index % 5 == 0 {
                        hyperedges.push(hyperedges.last().unwrap().clone());
                    }
                }
                let input = DecodingHypergraph {
                    vertex_num: vertex_num as u64,
                    hyperedges,
                };
                let compact = CompactHuf::new(&input, 30.0);
                let mut mwpf =
                    MwpfDecoderInstance::new(&input, &serde_json::json!({"cluster_node_limit": 0, "timeout": 30.0}));
                for _ in 0..8 {
                    let mut syndrome = vec![false; vertex_num];
                    for hyperedge in &input.hyperedges {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        if rng & 1 == 0 {
                            for &vertex in &hyperedge.vertices {
                                syndrome[vertex as usize] ^= true;
                            }
                        }
                    }
                    let defects: Vec<_> = syndrome
                        .iter()
                        .enumerate()
                        .filter_map(|(vertex, &is_defect)| is_defect.then_some(vertex))
                        .collect();
                    let compact_result = compact.decode(&defects).unwrap();
                    verify_correction(&input, &defects, &compact_result);

                    let sparse_defects: Vec<_> = defects.iter().map(|&vertex| vertex as u64).collect();
                    let syndrome = from_sparse_indices(input.vertex_num, &sparse_defects);
                    let mwpf_result = mwpf
                        .decode(DecodeRequest {
                            syndrome: &syndrome,
                            decoder_seed: None,
                            reweights: &[],
                            loss: None,
                        })
                        .unwrap()
                        .subgraph;
                    verify_correction(&input, &defects, &mwpf_result);
                    mwpf.reset();
                }
            }
        }
    }

    #[test]
    fn compact_large_solvable_graphs_remain_deterministic() {
        let mut rng = 78221u64;
        for n in [65, 130, 257] {
            let mut edges = vec![];
            let mut syndrome = vec![false; n];
            for index in 0..(n * 5) {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let mut vertices = vec![rng as usize % n];
                for offset in 1..=(rng >> 9) as usize % 5 {
                    vertices.push((rng as usize + offset * 17) % n);
                }
                vertices.sort_unstable();
                vertices.dedup();
                if index % 19 == 0 {
                    for &vertex in &vertices {
                        syndrome[vertex] ^= true;
                    }
                }
                edges.push(Hyperedge {
                    vertices: vertices.iter().map(|&v| v as u64).collect(),
                    probability: [0.01, 0.1, 0.2][index % 3],
                });
            }
            let input = DecodingHypergraph {
                vertex_num: n as u64,
                hyperedges: edges,
            };
            let defects: Vec<_> = (0..n).filter(|&v| syndrome[v]).collect();
            verify(&input, &defects);
        }
    }

    #[test]
    fn compact_many_inactive_vertices_do_not_require_dense_cluster_syndromes() {
        let input = graph(100_000, &[(&[0, 99_999], 0.1), (&[5, 7], 0.2)]);
        assert_eq!(verify(&input, &[0, 99_999]), vec![0]);
    }

    #[test]
    fn decoder_handles_parallel_edges_and_reset() {
        let hypergraph = graph(3, &[(&[0, 1, 2], 0.01), (&[2, 0, 1], 0.1)]);
        let mut decoder = HufDecoderInstance::new(&hypergraph, &serde_json::json!({}));
        let syndrome = from_sparse_indices(3, &[0, 1, 2]);
        for _ in 0..2 {
            let result = decoder
                .decode(DecodeRequest {
                    syndrome: &syndrome,
                    decoder_seed: None,
                    reweights: &[],
                    loss: None,
                })
                .unwrap();
            assert_eq!(result.subgraph, vec![1]);
            decoder.reset();
        }
    }

    #[test]
    fn decoder_validates_config_and_repeated_decoding() {
        let hypergraph = graph(3, &[(&[0, 1, 2], 0.1)]);
        let config = serde_json::json!({"parallel": 1, "timeout": 30.0});
        assert!(HufDecoderInstance::validate_hypergraph(&hypergraph, &config).is_ok());
        for invalid in [
            serde_json::json!({"engine": "unknown"}),
            serde_json::json!({"engin": "compact"}),
            serde_json::json!({"timeout": 0}),
            serde_json::json!({"only_solve_primal_once": true}),
        ] {
            assert!(HufDecoderInstance::validate_hypergraph(&hypergraph, &invalid).is_err());
        }
        let mut decoder = HufDecoderInstance::new(&hypergraph, &config);
        for defects in [vec![0, 1, 2], vec![], vec![0, 1, 2]] {
            let syndrome = from_sparse_indices(3, &defects);
            let response = decoder
                .decode(DecodeRequest {
                    syndrome: &syndrome,
                    decoder_seed: None,
                    reweights: &[],
                    loss: None,
                })
                .unwrap();
            assert_eq!(response.subgraph, if defects.is_empty() { vec![] } else { vec![0] });
            decoder.reset();
        }
    }
}
