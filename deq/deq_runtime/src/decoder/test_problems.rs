//! Standard decoding test problems
//!
//! A small curated set of decoding problems used by the standard decoder test
//! harness. The library carries **no policy** about which problems each decoder
//! should solve correctly — that is up to each caller (integration tests, CLI)
//! to specify.

use crate::decoder::blackbox_decoder::{DecodingHypergraph, Hyperedge};
use crate::misc::bit_vector::from_sparse_indices;
use crate::util::BitVector;

/// One standard test problem: a hypergraph plus a list of syndrome inputs.
#[derive(Clone, Debug)]
pub struct StandardTestProblem {
    pub name: &'static str,
    pub hypergraph: DecodingHypergraph,
    pub cases: Vec<TestCase>,
}

/// A single syndrome input for a standard problem.
#[derive(Clone, Debug)]
pub struct TestCase {
    pub name: &'static str,
    pub syndrome: BitVector,
}

/// Format the identifier of a (problem, case) pair as `"problem/case"`.
pub fn case_id(problem: &str, case: &str) -> String {
    format!("{problem}/{case}")
}

fn edge(vertices: &[u64], probability: f64) -> Hyperedge {
    Hyperedge {
        vertices: vertices.to_vec(),
        probability,
    }
}

fn case(name: &'static str, vertex_num: u64, defect_indices: &[u64]) -> TestCase {
    TestCase {
        name,
        syndrome: from_sparse_indices(vertex_num, defect_indices),
    }
}

/// Return the standard list of decoding problems exercised by the harness.
///
/// The set is intentionally small and well-defined so every decoder can be
/// reasoned about case-by-case.
pub fn standard_test_problems() -> Vec<StandardTestProblem> {
    vec![
        StandardTestProblem {
            name: "empty_graph",
            hypergraph: DecodingHypergraph {
                vertex_num: 0,
                hyperedges: vec![],
            },
            cases: vec![case("zero", 0, &[])],
        },
        StandardTestProblem {
            name: "no_edges",
            hypergraph: DecodingHypergraph {
                vertex_num: 3,
                hyperedges: vec![],
            },
            cases: vec![case("zero", 3, &[])],
        },
        StandardTestProblem {
            name: "single_vertex_boundary",
            hypergraph: DecodingHypergraph {
                vertex_num: 1,
                hyperedges: vec![edge(&[0], 0.1)],
            },
            cases: vec![case("zero", 1, &[]), case("vertex_0", 1, &[0])],
        },
        StandardTestProblem {
            name: "simple_edge",
            hypergraph: DecodingHypergraph {
                vertex_num: 2,
                hyperedges: vec![edge(&[0, 1], 0.1)],
            },
            cases: vec![case("zero", 2, &[]), case("both", 2, &[0, 1])],
        },
        StandardTestProblem {
            name: "single_hyperedge_3",
            hypergraph: DecodingHypergraph {
                vertex_num: 3,
                hyperedges: vec![edge(&[0, 1, 2], 0.1)],
            },
            cases: vec![case("zero", 3, &[]), case("all", 3, &[0, 1, 2])],
        },
        StandardTestProblem {
            name: "line_chain_3",
            hypergraph: DecodingHypergraph {
                vertex_num: 3,
                hyperedges: vec![edge(&[0, 1], 0.1), edge(&[1, 2], 0.1)],
            },
            cases: vec![
                case("zero", 3, &[]),
                case("left", 3, &[0, 1]),
                case("right", 3, &[1, 2]),
                case("ends", 3, &[0, 2]),
            ],
        },
        StandardTestProblem {
            name: "triangle",
            hypergraph: DecodingHypergraph {
                vertex_num: 3,
                hyperedges: vec![edge(&[0, 1], 0.1), edge(&[1, 2], 0.1), edge(&[0, 2], 0.1)],
            },
            cases: vec![
                case("zero", 3, &[]),
                case("v01", 3, &[0, 1]),
                case("v12", 3, &[1, 2]),
                case("v02", 3, &[0, 2]),
            ],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn problems_are_well_formed() {
        for problem in standard_test_problems() {
            assert!(!problem.cases.is_empty(), "problem {} has no cases", problem.name);
            for case in &problem.cases {
                assert_eq!(
                    case.syndrome.size, problem.hypergraph.vertex_num,
                    "case {}/{} syndrome size mismatch",
                    problem.name, case.name
                );
            }
        }
    }

    #[test]
    fn case_id_formats_as_problem_slash_case() {
        assert_eq!(case_id("simple_edge", "both"), "simple_edge/both");
    }
}
