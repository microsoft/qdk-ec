//! Validation of black-box decoder protocol messages.

use crate::decoder::blackbox_decoder::{DecodingHypergraph, LossInfo, ParityFactor};
use crate::util::BitVector;

pub(crate) fn validate_parity_factor(parity_factor: ParityFactor, edge_count: usize) -> Result<ParityFactor, String> {
    if let Some(&edge) = parity_factor
        .subgraph
        .iter()
        .find(|&&edge| usize::try_from(edge).map_or(true, |edge| edge >= edge_count))
    {
        return Err(format!(
            "decoder returned edge {edge}, but the hypergraph has {edge_count} edges"
        ));
    }
    Ok(parity_factor)
}

pub(crate) fn validate_hypergraph(hypergraph: &DecodingHypergraph) -> Result<(), String> {
    for (edge, hyperedge) in hypergraph.hyperedges.iter().enumerate() {
        if !hyperedge.probability.is_finite() || !(0.0..=1.0).contains(&hyperedge.probability) {
            return Err(format!(
                "hyperedge {edge} probability must lie in [0, 1], got {}",
                hyperedge.probability
            ));
        }
        let mut vertices = hashbrown::HashSet::with_capacity(hyperedge.vertices.len());
        for &vertex in &hyperedge.vertices {
            if vertex >= hypergraph.vertex_num {
                return Err(format!(
                    "hyperedge {edge} contains vertex {vertex}, outside [0, {})",
                    hypergraph.vertex_num
                ));
            }
            if !vertices.insert(vertex) {
                return Err(format!("hyperedge {edge} contains vertex {vertex} more than once"));
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_syndrome(syndrome: &BitVector, vertex_num: u64) -> Result<(), String> {
    crate::misc::bit_vector::validate_data_len(syndrome, "syndrome")?;
    if syndrome.size != vertex_num {
        return Err(format!(
            "syndrome size {} does not match hypergraph vertex count {vertex_num}",
            syndrome.size
        ));
    }
    Ok(())
}

pub(crate) fn validate_reweights(reweights: &[(u64, f64)], edge_count: usize) -> Result<(), String> {
    let mut seen = hashbrown::HashSet::with_capacity(reweights.len());
    for &(edge, probability) in reweights {
        if usize::try_from(edge).map_or(true, |edge| edge >= edge_count) {
            return Err(format!(
                "reweighted edge {edge} is outside loaded hypergraph with {edge_count} edges"
            ));
        }
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(format!(
                "reweighted edge {edge} probability must lie in [0, 1], got {probability}"
            ));
        }
        if !seen.insert(edge) {
            return Err(format!("reweighted edge {edge} is assigned more than once"));
        }
    }
    Ok(())
}

pub(crate) fn validate_loss(loss: Option<&LossInfo>, edge_count: usize) -> Result<(), String> {
    let Some(loss) = loss else {
        return Ok(());
    };
    let mut indegree = vec![0usize; loss.sites.len()];
    let mut children = vec![Vec::new(); loss.sites.len()];
    for (site_index, site) in loss.sites.iter().enumerate() {
        if !site.probability.is_finite() || !(0.0..=1.0).contains(&site.probability) {
            return Err(format!(
                "loss site {site_index} probability must lie in [0, 1], got {}",
                site.probability
            ));
        }
        for (field, edges) in [
            ("source_edges", &site.source_edges),
            ("continuation_edges", &site.continuation_edges),
        ] {
            let mut seen = hashbrown::HashSet::with_capacity(edges.len());
            for &edge in edges {
                if usize::try_from(edge).map_or(true, |edge| edge >= edge_count) {
                    return Err(format!(
                        "loss site {site_index} {field} contains edge {edge}, outside [0, {edge_count})"
                    ));
                }
                if !seen.insert(edge) {
                    return Err(format!("loss site {site_index} {field} contains edge {edge} more than once"));
                }
            }
        }
        let mut seen_children = hashbrown::HashSet::with_capacity(site.children.len());
        for &child in &site.children {
            let Ok(child_index) = usize::try_from(child) else {
                return Err(format!(
                    "loss site {site_index} contains child {child}, outside [0, {})",
                    loss.sites.len()
                ));
            };
            let Some(degree) = indegree.get_mut(child_index) else {
                return Err(format!(
                    "loss site {site_index} contains child {child}, outside [0, {})",
                    loss.sites.len()
                ));
            };
            if !seen_children.insert(child_index) {
                return Err(format!("loss site {site_index} contains child {child} more than once"));
            }
            *degree += 1;
            children[site_index].push(child_index);
        }
        let mut seen_heralds = hashbrown::HashSet::with_capacity(site.heralds.len());
        for &herald in &site.heralds {
            if !seen_heralds.insert(herald) {
                return Err(format!("loss site {site_index} contains herald {herald} more than once"));
            }
        }
    }
    let mut frontier: Vec<usize> = indegree
        .iter()
        .enumerate()
        .filter_map(|(site, &degree)| (degree == 0).then_some(site))
        .collect();
    let mut visited = 0usize;
    while let Some(site_index) = frontier.pop() {
        visited += 1;
        for &child_index in &children[site_index] {
            indegree[child_index] -= 1;
            if indegree[child_index] == 0 {
                frontier.push(child_index);
            }
        }
    }
    if visited != loss.sites.len() {
        return Err("loss site children graph contains a cycle".to_string());
    }
    Ok(())
}
