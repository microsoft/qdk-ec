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
    let mut flips = HashSet::<u64>::new();
    for &edge_index in &parity_factor.subgraph {
        let edge = &decoding_hypergraph.hyperedges[edge_index as usize];
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
