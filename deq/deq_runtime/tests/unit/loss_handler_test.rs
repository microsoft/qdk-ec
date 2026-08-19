//! Unit tests for the coordinator loss-handling pass.

use super::*;
use crate::decoder::blackbox_decoder::{DecodingHypergraph, Hyperedge, LossInfo, LossSite};
use crate::misc::bit_vector;
use crate::simulator::DeterministicRng;
use crate::util::BitVector;
use rand::SeedableRng;
use serde_json::json;
use std::sync::Arc;

#[test]
fn missing_config_is_accepted_and_defaults() {
    let handler = LossHandler::new(LossStrategy::Reweight, serde_json::Value::Null).unwrap();
    assert_eq!(
        handler,
        LossHandler::Reweight(EnvelopeReweightPolicy {
            weight_fraction: DEFAULT_WEIGHT_FRACTION,
            scale: ReweightScale::Local,
        })
    );
    assert!(LossHandler::new(LossStrategy::Ignore, serde_json::Value::Null).is_ok());
}

#[test]
fn reweight_reads_its_weight_fraction() {
    let handler = LossHandler::new(LossStrategy::Reweight, json!({ "weight_fraction": 0.25 })).unwrap();
    assert!((handler.reweight_policy().unwrap().weight_fraction - 0.25).abs() < 1e-12);
}

#[test]
fn reweight_policy_lowers_edge_weight_to_the_fraction() {
    let site = 0.001;
    let half = EnvelopeReweightPolicy::default().locally_reweighted_probability(0.0, site);
    let quarter = EnvelopeReweightPolicy::new(0.25).locally_reweighted_probability(0.0, site);
    assert!((weight_of(half) - 0.5 * weight_of(site)).abs() < 1e-12);
    assert!((weight_of(quarter) - 0.25 * weight_of(site)).abs() < 1e-12);
    assert!(half > 0.0 && quarter > half);
}

#[test]
fn reweight_policy_never_reaches_negative_weight() {
    for &fraction in &[0.05, 0.25, 0.5, 0.9, 1.0] {
        let policy = EnvelopeReweightPolicy::new(fraction);
        for &prior in &[0.0, 1e-9, 0.01, 0.1, 0.3, 0.5] {
            for &site in &[0.0, 1e-9, 0.01, 0.1, 0.3, 0.5] {
                let probability = policy.locally_reweighted_probability(prior, site);
                assert!(probability <= 0.5 + 1e-12);
                assert!(weight_of(probability) >= -1e-12);
            }
        }
    }
}

#[test]
fn reweight_policy_never_lowers_a_high_prior_edge() {
    let policy = EnvelopeReweightPolicy::default();
    let prior = 0.2;
    assert!(policy.locally_reweighted_probability(prior, 1e-6) > prior);
}

/// The whole point of the hierarchy: an option belonging to another strategy
/// is a construction-time error, not a silently ignored field.
#[test]
fn weight_fraction_is_rejected_by_strategies_that_ignore_it() {
    for strategy in [LossStrategy::Ignore, LossStrategy::Handoff] {
        let error = LossHandler::new(strategy, json!({ "weight_fraction": 0.25 })).unwrap_err();
        assert!(error.contains("takes none"), "unexpected message: {error}");
    }
}

#[test]
fn unknown_option_is_rejected() {
    let error = LossHandler::new(LossStrategy::Reweight, json!({ "weight_fractoin": 0.25 })).unwrap_err();
    assert!(error.contains("weight_fractoin"), "unexpected message: {error}");
}

#[test]
fn weight_fraction_outside_the_unit_interval_is_rejected() {
    for bad in [0.0, -0.5, 1.5] {
        let error = LossHandler::new(LossStrategy::Reweight, json!({ "weight_fraction": bad })).unwrap_err();
        assert!(error.contains("(0, 1]"), "unexpected message: {error}");
    }
}

#[test]
fn only_ignore_skips_the_loss_pipeline() {
    let reweight = LossHandler::Reweight(EnvelopeReweightPolicy {
        weight_fraction: 0.5,
        scale: ReweightScale::Local,
    });
    assert!(!LossHandler::Ignore.tracks_losses());
    assert!(reweight.tracks_losses());
    assert!(LossHandler::Handoff.tracks_losses());
    assert!(LossHandler::Handoff.hands_off_to_decoder());
    assert!(!reweight.hands_off_to_decoder());
}

/// The reference construction stays reachable from config, so the two rules
/// can be measured against each other on identical shots.
#[test]
fn scale_selects_the_reference_construction() {
    let handler = LossHandler::new(
        LossStrategy::Reweight,
        json!({ "weight_fraction": 0.5, "scale": "global_mean" }),
    )
    .unwrap();
    assert_eq!(handler.reweight_policy().unwrap().scale, ReweightScale::GlobalMean);
    let default = LossHandler::new(LossStrategy::Reweight, serde_json::Value::Null).unwrap();
    assert_eq!(default.reweight_policy().unwrap().scale, ReweightScale::Local);
    let error = LossHandler::new(LossStrategy::Reweight, json!({ "scale": "globl_mean" })).unwrap_err();
    assert!(error.contains("globl_mean"), "unexpected message: {error}");
}

#[test]
fn strategy_deserializes_from_snake_case() {
    let strategy: LossStrategy = serde_json::from_value(json!("handoff")).unwrap();
    assert_eq!(strategy, LossStrategy::Handoff);
    assert_eq!(LossStrategy::default(), LossStrategy::Reweight);
}

#[test]
fn build_loss_info_maps_generators_through_the_site_local_eid() {
    let error_reference = vec![
        ErrorIndex { eid: 10, error_index: 0 },
        ErrorIndex { eid: 10, error_index: 1 },
        ErrorIndex { eid: 20, error_index: 0 },
    ];
    let loss_sites = vec![
        RawLossSite {
            local_eid: Some(10),
            probability: 0.1,
            source_generators: vec![0],
            continuation_generators: vec![1],
            children: vec![1],
            heralds: vec![7],
        },
        RawLossSite {
            local_eid: Some(20),
            probability: 0.0,
            source_generators: vec![],
            continuation_generators: vec![0],
            children: vec![],
            heralds: vec![8],
        },
    ];

    let loss_info = build_loss_info(&loss_sites, &error_reference);

    assert_eq!(loss_info.sites[0].source_edges, vec![0]);
    assert_eq!(loss_info.sites[0].continuation_edges, vec![1]);
    assert_eq!(loss_info.sites[0].children, vec![1]);
    assert_eq!(loss_info.sites[0].heralds, vec![7]);
    assert_eq!(loss_info.sites[1].continuation_edges, vec![2]);
}

#[test]
fn build_loss_info_preserves_structural_sites_without_error_models() {
    let loss_sites = vec![
        RawLossSite {
            local_eid: None,
            probability: 0.1,
            source_generators: vec![0],
            continuation_generators: vec![],
            children: vec![1],
            heralds: vec![],
        },
        RawLossSite {
            local_eid: Some(20),
            probability: 0.0,
            source_generators: vec![],
            continuation_generators: vec![0],
            children: vec![],
            heralds: vec![0],
        },
    ];
    let loss_info = build_loss_info(&loss_sites, &[ErrorIndex { eid: 20, error_index: 0 }]);

    assert!(loss_info.sites[0].source_edges.is_empty());
    assert_eq!(loss_info.sites[0].children, vec![1]);
    assert_eq!(loss_info.sites[1].continuation_edges, vec![0]);
}

fn single_edge_projection(prior: f64) -> DecodeProjection {
    let hypergraph = blackbox_decoder::DecodingHypergraph {
        vertex_num: 1,
        hyperedges: vec![blackbox_decoder::Hyperedge {
            vertices: vec![0],
            probability: prior,
        }],
    };
    let errors = Arc::new(vec![ErrorIndex { eid: 0, error_index: 0 }]);
    DecodeProjection::identity(hypergraph, errors)
}

fn single_edge_loss(probability: f64) -> Vec<RawLossSite> {
    vec![RawLossSite {
        local_eid: Some(0),
        probability,
        source_generators: vec![0],
        continuation_generators: vec![],
        children: vec![],
        heralds: vec![0],
    }]
}

#[test]
fn handoff_preserves_user_reweights_and_structured_loss_together() {
    let projection = single_edge_projection(0.1);
    let projected = LossHandler::Handoff.project_shot(&projection, &[(0, 0.2)], &single_edge_loss(0.3));

    assert_eq!(projected.reweights[0].edge, 0);
    assert!((projected.reweights[0].probability - 0.2).abs() < 1e-12);
    assert_eq!(projected.loss.unwrap().sites[0].source_edges, vec![0]);
}

#[test]
fn loss_reweighting_uses_the_user_updated_prior() {
    let projection = single_edge_projection(0.1);
    let policy = EnvelopeReweightPolicy::new(0.5);
    let projected = LossHandler::Reweight(policy).project_shot(&projection, &[(0, 0.2)], &single_edge_loss(0.3));

    assert!(projected.loss.is_none());
    assert!((projected.reweights[0].probability - policy.locally_reweighted_probability(0.2, 0.3)).abs() < 1e-12);
}

#[test]
fn ignore_discards_loss_but_preserves_user_reweights() {
    let projection = single_edge_projection(0.1);
    let projected = LossHandler::Ignore.project_shot(&projection, &[(0, 0.2)], &single_edge_loss(0.3));

    assert!(projected.loss.is_none());
    assert!((projected.reweights[0].probability - 0.2).abs() < 1e-12);
}

#[test]
fn apply_loss_random_imputation_leaves_non_loss_bits_untouched() {
    let mut outcomes = BitVector {
        size: 4,
        data: vec![0b1010_0000],
    };
    let loss_mask = BitVector { size: 4, data: vec![0] };
    let before = outcomes.clone();

    apply_loss_random_imputation(&mut outcomes, &loss_mask, &mut DeterministicRng::seed_from_u64(42));

    assert_eq!(outcomes, before);
}

#[test]
fn apply_loss_random_imputation_only_replaces_marked_bits() {
    let mut rng = DeterministicRng::seed_from_u64(1);
    let loss_mask = BitVector {
        size: 4,
        data: vec![0b0101_0000],
    };
    let mut bit1_zero_count = 0usize;
    let mut bit3_zero_count = 0usize;
    let trials = 1000usize;
    for _ in 0..trials {
        let mut outcomes = BitVector {
            size: 4,
            data: vec![0b1010_0000],
        };
        apply_loss_random_imputation(&mut outcomes, &loss_mask, &mut rng);
        assert!(bit_vector::get_bit(&outcomes, 0));
        assert!(bit_vector::get_bit(&outcomes, 2));
        bit1_zero_count += usize::from(!bit_vector::get_bit(&outcomes, 1));
        bit3_zero_count += usize::from(!bit_vector::get_bit(&outcomes, 3));
    }
    assert!((trials / 4..3 * trials / 4).contains(&bit1_zero_count));
    assert!((trials / 4..3 * trials / 4).contains(&bit3_zero_count));
}

#[test]
fn apply_loss_random_imputation_is_deterministic_with_same_seed() {
    let loss_mask = BitVector {
        size: 8,
        data: vec![0xff],
    };
    let mut first = BitVector { size: 8, data: vec![0] };
    let mut second = first.clone();

    apply_loss_random_imputation(&mut first, &loss_mask, &mut DeterministicRng::seed_from_u64(123));
    apply_loss_random_imputation(&mut second, &loss_mask, &mut DeterministicRng::seed_from_u64(123));

    assert_eq!(first, second);
}

#[test]
#[should_panic(expected = "does not match outcomes size")]
fn apply_loss_random_imputation_panics_on_size_mismatch() {
    apply_loss_random_imputation(
        &mut BitVector { size: 4, data: vec![0] },
        &BitVector {
            size: 5,
            data: vec![0b0000_1000],
        },
        &mut DeterministicRng::seed_from_u64(0),
    );
}

fn reweight_hypergraph(probabilities: &[f64]) -> DecodingHypergraph {
    DecodingHypergraph {
        vertex_num: 2,
        hyperedges: probabilities
            .iter()
            .map(|&probability| Hyperedge {
                vertices: vec![0, 1],
                probability,
            })
            .collect(),
    }
}

fn reweight_site(probability: f64, source: Vec<u64>, continuation: Vec<u64>, children: Vec<u64>) -> LossSite {
    LossSite {
        source_edges: source,
        continuation_edges: continuation,
        children,
        probability,
        heralds: vec![],
    }
}

fn local_loss(sites: Vec<LossSite>, weight_fraction: f64) -> (LossInfo, EnvelopeReweightPolicy) {
    (
        LossInfo { sites },
        EnvelopeReweightPolicy {
            weight_fraction,
            scale: ReweightScale::Local,
        },
    )
}

fn global_mean_loss(sites: Vec<LossSite>, weight_fraction: f64) -> (LossInfo, EnvelopeReweightPolicy) {
    (
        LossInfo { sites },
        EnvelopeReweightPolicy {
            weight_fraction,
            scale: ReweightScale::GlobalMean,
        },
    )
}

fn neighbourhood_loss(sites: Vec<LossSite>, weight_fraction: f64) -> (LossInfo, EnvelopeReweightPolicy) {
    (
        LossInfo { sites },
        EnvelopeReweightPolicy {
            weight_fraction,
            scale: ReweightScale::NeighbourhoodMean,
        },
    )
}

fn reweights(built: (LossInfo, EnvelopeReweightPolicy), graph: &DecodingHypergraph) -> Vec<(u64, f64)> {
    let (loss, policy) = built;
    loss_reweights(&loss, graph, policy)
}

#[test]
fn continuation_site_inherits_its_ancestors_probability() {
    let sites = vec![
        reweight_site(0.01, vec![0], vec![], vec![1]),
        reweight_site(0.0, vec![], vec![1], vec![]),
    ];
    let accumulated = accumulated_site_probabilities(&sites);
    assert!((accumulated[0] - 0.01).abs() < 1e-12);
    assert!((accumulated[1] - 0.01).abs() < 1e-12);
}

#[test]
fn accumulated_probability_merges_multiple_ancestors() {
    let sites = vec![
        reweight_site(0.01, vec![], vec![], vec![2]),
        reweight_site(0.02, vec![], vec![], vec![2]),
        reweight_site(0.0, vec![], vec![0], vec![]),
    ];
    let accumulated = accumulated_site_probabilities(&sites);
    assert!((accumulated[2] - exclusive_probability_of(0.01, 0.02)).abs() < 1e-12);
}

#[test]
fn local_rule_lowers_a_loss_only_edge_to_the_fraction_of_its_weight() {
    let graph = reweight_hypergraph(&[0.0]);
    let (loss, policy) = local_loss(vec![reweight_site(0.01, vec![0], vec![], vec![])], 0.5);
    let (edge, probability) = loss_reweights(&loss, &graph, policy)[0];
    assert_eq!(edge, 0);
    assert!((weight_of(probability) - 0.5 * weight_of(0.01)).abs() < 1e-12);
}

#[test]
fn local_rule_never_produces_a_negative_weight() {
    for &fraction in &[0.05, 0.25, 0.5, 1.0] {
        for &prior in &[0.0, 0.01, 0.2, 0.5] {
            let graph = reweight_hypergraph(&[prior, prior]);
            let (loss, policy) = local_loss(
                vec![
                    reweight_site(0.4, vec![0, 1], vec![], vec![1]),
                    reweight_site(0.4, vec![0, 1], vec![], vec![]),
                ],
                fraction,
            );
            for (edge, probability) in loss_reweights(&loss, &graph, policy) {
                assert!(
                    probability <= 0.5 + 1e-12,
                    "fraction={fraction} prior={prior} edge={edge} gave {probability} > 1/2"
                );
            }
        }
    }
}

#[test]
fn local_rule_never_lowers_an_edge() {
    let prior = 0.3;
    let graph = reweight_hypergraph(&[prior]);
    let (loss, policy) = local_loss(vec![reweight_site(0.01, vec![0], vec![], vec![])], 0.5);
    let (_, probability) = loss_reweights(&loss, &graph, policy)[0];
    assert!(probability >= prior);
}

#[test]
fn local_rule_makes_propagated_continuation_edges_usable() {
    let graph = reweight_hypergraph(&[0.0, 0.0]);
    let (loss, policy) = local_loss(
        vec![
            reweight_site(0.01, vec![0], vec![], vec![1]),
            reweight_site(0.0, vec![], vec![1], vec![]),
        ],
        0.5,
    );
    let reweights = loss_reweights(&loss, &graph, policy);
    let continuation = reweights.iter().find(|(edge, _)| *edge == 1).expect("edge 1 reweighted");
    assert!(continuation.1 > 0.0);
    assert!((weight_of(continuation.1) - 0.5 * weight_of(0.01)).abs() < 1e-12);
}

#[test]
fn local_rule_applies_the_fraction_once_per_edge_not_once_per_site() {
    let fraction = 0.25;
    let each = 0.01;
    for count in [1usize, 2, 4, 10] {
        let graph = reweight_hypergraph(&[0.0]);
        let sites = (0..count).map(|_| reweight_site(each, vec![0], vec![], vec![])).collect();
        let (_, probability) = reweights(local_loss(sites, fraction), &graph)[0];
        let total = (0..count).fold(0.0, |accumulated, _| exclusive_probability_of(accumulated, each));
        assert!((weight_of(probability) - fraction * weight_of(total)).abs() < 1e-12);
    }
}

#[test]
fn global_mean_rule_assigns_the_graph_average_to_every_activated_edge() {
    let graph = reweight_hypergraph(&[0.001, 0.05, 0.0]);
    let mean = (weight_of(0.001) + weight_of(0.05)) / 2.0;
    let (loss, policy) = global_mean_loss(
        vec![
            reweight_site(0.01, vec![0], vec![], vec![]),
            reweight_site(0.2, vec![2], vec![], vec![]),
        ],
        0.5,
    );
    for (_, probability) in loss_reweights(&loss, &graph, policy) {
        assert!((weight_of(probability) - 0.5 * mean).abs() < 1e-12);
    }
}

#[test]
fn the_two_scales_disagree_on_a_heterogeneous_graph() {
    let graph = reweight_hypergraph(&[0.001, 0.05, 0.0]);
    let sites = vec![reweight_site(0.01, vec![2], vec![], vec![])];
    let local = reweights(local_loss(sites.clone(), 0.5), &graph)[0].1;
    let global = reweights(global_mean_loss(sites, 0.5), &graph)[0].1;
    assert!((local - global).abs() > 1e-6);
}

#[test]
fn neighbourhood_scale_matches_the_global_mean_on_a_homogeneous_graph() {
    let graph = reweight_hypergraph(&[0.01, 0.01, 0.01, 0.0]);
    let sites = vec![reweight_site(0.02, vec![3], vec![], vec![])];
    let global = reweights(global_mean_loss(sites.clone(), 0.5), &graph)[0].1;
    let local_mean = reweights(neighbourhood_loss(sites, 0.5), &graph)[0].1;
    assert!((global - local_mean).abs() < 1e-12);
}

#[test]
fn neighbourhood_scale_is_defined_without_regular_edges() {
    let graph = reweight_hypergraph(&[0.0, 0.0]);
    let sites = vec![reweight_site(0.01, vec![0], vec![], vec![])];
    let fallback = reweights(neighbourhood_loss(sites.clone(), 0.5), &graph)[0].1;
    let local = reweights(local_loss(sites, 0.5), &graph)[0].1;
    assert!((fallback - local).abs() < 1e-12);
    assert!(fallback > 0.0 && fallback <= 0.5);
}
