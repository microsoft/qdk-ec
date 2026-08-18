//! Validation of runtime protocol and coordinator payloads.

use crate::bin::{GadgetType, ProbabilityModifier};
use crate::coordinator::loss_handler::LossHandler;
use crate::decoder::decoder_features::DecoderFeatures;
use crate::util::BitVector;
use tonic::Status;

const REMOTE_REROUTE_INDEX_LIMIT: u64 = 65_536;

/// Validate that a bit vector has exactly the bytes required by its declared
/// number of bits.
pub fn validate_data_len(bit_vector: &BitVector, name: &str) -> Result<(), String> {
    let required = crate::misc::bit_vector::bit_vector_len(bit_vector.size);
    if bit_vector.data.len() != required {
        Err(format!(
            "{name} data length ({}) does not match required length ({required}) for {} bits",
            bit_vector.data.len(),
            bit_vector.size
        ))
    } else {
        Ok(())
    }
}

pub fn validate_outcomes(outcomes: &BitVector, loss_mask: Option<&BitVector>, expected_size: u64) -> Result<(), String> {
    validate_data_len(outcomes, "outcomes")?;
    if outcomes.size != expected_size {
        return Err(format!(
            "outcomes size {} does not match gadget measurement count {expected_size}",
            outcomes.size
        ));
    }
    if let Some(loss_mask) = loss_mask {
        validate_data_len(loss_mask, "loss_mask")?;
        if outcomes.size != loss_mask.size {
            return Err(format!(
                "loss_mask size {} does not match outcomes size {}",
                loss_mask.size, outcomes.size
            ));
        }
    }
    Ok(())
}

/// Validate the shape, indices, and values of a probability modifier for an
/// error model with `error_count` errors.
pub fn validate_probability_modifier(modifier: &ProbabilityModifier, error_count: usize) -> Result<(), String> {
    if !modifier.probabilities.is_empty() && modifier.probabilities.len() != error_count {
        return Err(format!(
            "dense probability modifier has length {}, expected 0 or {error_count}",
            modifier.probabilities.len()
        ));
    }
    if modifier.sparse_indices.len() != modifier.sparse_probabilities.len() {
        return Err(format!(
            "sparse probability modifier has {} indices but {} probabilities",
            modifier.sparse_indices.len(),
            modifier.sparse_probabilities.len()
        ));
    }
    for &index in &modifier.sparse_indices {
        if index >= error_count as u64 {
            return Err(format!(
                "sparse probability modifier index {index} is outside [0, {error_count})"
            ));
        }
    }
    for &probability in modifier.probabilities.iter().chain(modifier.sparse_probabilities.iter()) {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(format!("probability modifier value must lie in [0, 1], got {probability}"));
        }
    }
    Ok(())
}

fn remote_reroute_position(index: u64, kind: &str) -> Result<usize, String> {
    if index >= REMOTE_REROUTE_INDEX_LIMIT {
        return Err(format!(
            "{kind} reroute index {index} must be less than {REMOTE_REROUTE_INDEX_LIMIT}"
        ));
    }
    usize::try_from(index).map_err(|_| format!("{kind} reroute index does not fit usize"))
}

fn validate_remote_reroutes(indices: impl IntoIterator<Item = u64>, kind: &str) -> Result<(), String> {
    for index in indices {
        remote_reroute_position(index, kind)?;
    }
    Ok(())
}

pub(crate) fn validate_check_model_reroutes(
    modifier: Option<&crate::bin::check_model::CheckModelModifier>,
) -> Result<(), String> {
    validate_remote_reroutes(
        modifier
            .into_iter()
            .flat_map(|modifier| &modifier.reroute_remote_gadgets)
            .map(|reroute| reroute.remote_gadget_index),
        "remote gadget",
    )
}

pub(crate) fn validate_error_model_reroutes(
    modifier: Option<&crate::bin::error_model::ErrorModelModifier>,
) -> Result<(), String> {
    validate_remote_reroutes(
        modifier
            .into_iter()
            .flat_map(|modifier| &modifier.reroute_remote_check_models)
            .map(|reroute| reroute.remote_check_model_index),
        "remote check model",
    )
}

fn apply_remote_reroutes<T>(
    base: &[T],
    reroutes: impl IntoIterator<Item = (u64, Option<T>)>,
    kind: &str,
) -> Result<Vec<Option<T>>, String>
where
    T: Clone,
{
    let mut modified: Vec<_> = base.iter().cloned().map(Some).collect();
    for (index, value) in reroutes {
        let index = remote_reroute_position(index, kind)?;
        if index >= modified.len() {
            modified.resize_with(index + 1, || None);
        }
        modified[index] = value;
    }
    Ok(modified)
}

pub(crate) fn apply_check_model_reroutes(
    base: &[crate::bin::check_model_type::RemoteGadget],
    modifier: Option<&crate::bin::check_model::CheckModelModifier>,
) -> Result<Vec<Option<crate::bin::check_model_type::RemoteGadget>>, String> {
    apply_remote_reroutes(
        base,
        modifier
            .into_iter()
            .flat_map(|modifier| &modifier.reroute_remote_gadgets)
            .map(|reroute| (reroute.remote_gadget_index, reroute.value.clone())),
        "remote gadget",
    )
}

pub(crate) fn apply_error_model_reroutes(
    base: &[crate::bin::error_model_type::RemoteCheckModel],
    modifier: Option<&crate::bin::error_model::ErrorModelModifier>,
) -> Result<Vec<Option<crate::bin::error_model_type::RemoteCheckModel>>, String> {
    apply_remote_reroutes(
        base,
        modifier
            .into_iter()
            .flat_map(|modifier| &modifier.reroute_remote_check_models)
            .map(|reroute| (reroute.remote_check_model_index, reroute.value.clone())),
        "remote check model",
    )
}

impl LossHandler {
    /// Validate that the decoder capabilities support this loss strategy for
    /// the supplied gadget library.
    pub fn validate_capability(self, gadget_types: &[GadgetType], features: DecoderFeatures) -> Result<(), Status> {
        if self.hands_off_to_decoder()
            && gadget_types.iter().any(|gadget_type| gadget_type.loss_model.is_some())
            && !features.contains(DecoderFeatures::LOSS)
        {
            return Err(Status::failed_precondition(
                "loss_strategy handoff requires a decoder with structured loss support for this library",
            ));
        }
        Ok(())
    }
}

use crate::decoder::blackbox_decoder::{DecodingHypergraph, LossInfo, ParityFactor};

/// Validate that every edge returned by a decoder belongs to its hypergraph.
pub fn validate_parity_factor(parity_factor: ParityFactor, edge_count: usize) -> Result<ParityFactor, String> {
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

/// Validate hyperedge probabilities and vertex references.
pub fn validate_hypergraph(hypergraph: &DecodingHypergraph) -> Result<(), String> {
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

/// Validate a syndrome's storage and size against a hypergraph vertex count.
pub fn validate_syndrome(syndrome: &BitVector, vertex_num: u64) -> Result<(), String> {
    validate_data_len(syndrome, "syndrome")?;
    if syndrome.size != vertex_num {
        return Err(format!(
            "syndrome size {} does not match hypergraph vertex count {vertex_num}",
            syndrome.size
        ));
    }
    Ok(())
}

/// Validate loaded-graph reweight indices, probabilities, and uniqueness.
pub fn validate_reweights(reweights: &[(u64, f64)], edge_count: usize) -> Result<(), String> {
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

/// Validate structured loss sites, references, uniqueness, and acyclicity.
pub fn validate_loss(loss: Option<&LossInfo>, edge_count: usize) -> Result<(), String> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probability_modifier_validation_rejects_malformed_inputs() {
        let dense_wrong_length = ProbabilityModifier {
            probabilities: vec![0.1],
            ..Default::default()
        };
        assert!(validate_probability_modifier(&dense_wrong_length, 2).is_err());

        let sparse_length_mismatch = ProbabilityModifier {
            sparse_indices: vec![0, 1],
            sparse_probabilities: vec![0.2],
            ..Default::default()
        };
        assert!(validate_probability_modifier(&sparse_length_mismatch, 2).is_err());

        let sparse_out_of_range = ProbabilityModifier {
            sparse_indices: vec![2],
            sparse_probabilities: vec![0.2],
            ..Default::default()
        };
        assert!(validate_probability_modifier(&sparse_out_of_range, 2).is_err());

        let invalid_probability = ProbabilityModifier {
            probabilities: vec![0.1, f64::NAN],
            ..Default::default()
        };
        assert!(validate_probability_modifier(&invalid_probability, 2).is_err());
    }

    #[test]
    fn remote_reroutes_extend_within_the_spec_bound() {
        let modifier = crate::bin::check_model::CheckModelModifier {
            reroute_remote_gadgets: vec![crate::bin::check_model::check_model_modifier::RerouteRemoteGadget {
                remote_gadget_index: 5,
                value: Some(crate::bin::check_model_type::RemoteGadget::default()),
            }],
        };

        let modified = apply_check_model_reroutes(&[], Some(&modifier)).unwrap();

        assert_eq!(modified.len(), 6);
        assert!(modified[5].is_some());
    }

    #[test]
    fn remote_reroutes_reject_indices_at_the_spec_bound() {
        let check_modifier = crate::bin::check_model::CheckModelModifier {
            reroute_remote_gadgets: vec![crate::bin::check_model::check_model_modifier::RerouteRemoteGadget {
                remote_gadget_index: REMOTE_REROUTE_INDEX_LIMIT,
                value: None,
            }],
        };
        let error_modifier = crate::bin::error_model::ErrorModelModifier {
            reroute_remote_check_models: vec![crate::bin::error_model::error_model_modifier::RerouteRemoteCheckModel {
                remote_check_model_index: REMOTE_REROUTE_INDEX_LIMIT,
                value: None,
            }],
            ..Default::default()
        };

        assert!(apply_check_model_reroutes(&[], Some(&check_modifier)).is_err());
        assert!(apply_error_model_reroutes(&[], Some(&error_modifier)).is_err());
    }

    #[test]
    fn handoff_requires_loss_support_only_for_loss_bearing_libraries() {
        assert!(
            LossHandler::Handoff
                .validate_capability(&[GadgetType::default()], DecoderFeatures::empty())
                .is_ok()
        );
        let loss_bearing = GadgetType {
            loss_model: Some(crate::bin::gadget_type::LossModel::default()),
            ..Default::default()
        };

        let error = LossHandler::Handoff
            .validate_capability(std::slice::from_ref(&loss_bearing), DecoderFeatures::empty())
            .unwrap_err();

        assert_eq!(error.code(), tonic::Code::FailedPrecondition);
        assert!(
            LossHandler::Handoff
                .validate_capability(&[loss_bearing], DecoderFeatures::LOSS)
                .is_ok()
        );
    }
}
