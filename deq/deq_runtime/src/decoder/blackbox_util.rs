use crate::decoder::blackbox_decoder::{DecodingHypergraph, ParityFactor};
use crate::misc::bit_vector::to_sparse_indices;
use crate::util::BitVector;
use hashbrown::HashSet;

/// Hypergraph indices of the hyperedges a core decoder should be built from:
/// those with a usable prior probability.
///
/// The decoding hypergraph may carry mechanisms with probability zero so their
/// stable edge indices remain available for shot-scoped updates. Core solvers
/// should omit those infinite-weight edges until a reweight makes them usable.
pub fn active_edge_indices(hypergraph: &DecodingHypergraph) -> Vec<u64> {
    hypergraph
        .hyperedges
        .iter()
        .enumerate()
        .filter(|(_, hyperedge)| hyperedge.probability > 0.0)
        .map(|(index, _)| index as u64)
        .collect()
}

pub fn is_parity_factor(
    decoding_hypergraph: &DecodingHypergraph,
    parity_factor: &ParityFactor,
    syndrome: &BitVector,
) -> bool {
    let mut selected_edges = HashSet::with_capacity(parity_factor.subgraph.len());
    let mut flips = HashSet::<u64>::new();
    for &edge_index in &parity_factor.subgraph {
        if !selected_edges.insert(edge_index) {
            return false;
        }
        let Ok(edge_index) = usize::try_from(edge_index) else {
            return false;
        };
        let Some(edge) = decoding_hypergraph.hyperedges.get(edge_index) else {
            return false;
        };
        for &vertex in &edge.vertices {
            if !flips.insert(vertex) {
                flips.remove(&vertex);
            }
        }
    }

    let syndrome = to_sparse_indices(syndrome);
    let mut flips: Vec<u64> = flips.into_iter().collect();
    flips.sort_unstable();
    syndrome == flips
}

pub fn assert_parity_factor(decoding_hypergraph: &DecodingHypergraph, parity_factor: &ParityFactor, syndrome: &BitVector) {
    if !is_parity_factor(decoding_hypergraph, parity_factor, syndrome) {
        panic!(
            "the provided parity factor does not match the syndrome: parity factor {parity_factor:?}, syndrome {syndrome:?}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_edge_graph() -> DecodingHypergraph {
        DecodingHypergraph {
            vertex_num: 1,
            hyperedges: vec![crate::decoder::blackbox_decoder::Hyperedge {
                vertices: vec![0],
                probability: 0.1,
            }],
        }
    }

    #[test]
    fn parity_factor_rejects_invalid_and_duplicate_edges() {
        let hypergraph = single_edge_graph();
        let syndrome = crate::misc::bit_vector::from_sparse_indices(1, &[0]);

        assert!(is_parity_factor(&hypergraph, &ParityFactor { subgraph: vec![0] }, &syndrome));
        assert!(!is_parity_factor(&hypergraph, &ParityFactor { subgraph: vec![1] }, &syndrome));
        assert!(!is_parity_factor(
            &hypergraph,
            &ParityFactor { subgraph: vec![0, 0] },
            &syndrome
        ));
    }
}
