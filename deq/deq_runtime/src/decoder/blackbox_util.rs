use crate::decoder::blackbox_decoder::{DecodingHypergraph, ParityFactor};
use crate::misc::bit_vector::to_sparse_indices;
use crate::util::BitVector;
use hashbrown::HashSet;

pub fn is_parity_factor(
    decoding_hypergraph: &DecodingHypergraph,
    parity_factor: &ParityFactor,
    syndrome: &BitVector,
) -> bool {
    // calculate the error syndromes
    let mut flips = HashSet::<u64>::new();
    for &edge_idx in &parity_factor.subgraph {
        let edge = &decoding_hypergraph.hyperedges[edge_idx as usize];
        for &vertex in &edge.vertices {
            if !flips.insert(vertex) {
                flips.remove(&vertex);
            }
        }
    }

    // compare with the syndrome
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
