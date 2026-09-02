//! Unit tests for the coordinator reweight-handling pass.

use super::*;

#[test]
fn sparse_probability_values_override_dense_values() {
    let errors = vec![ErrorIndex { eid: 4, error_index: 0 }, ErrorIndex { eid: 4, error_index: 1 }];
    let modifier = bin::ProbabilityModifier {
        probabilities: vec![0.1, 0.2],
        sparse_indices: vec![0],
        sparse_probabilities: vec![0.3],
    };

    let expected = vec![(0, 0.3), (1, 0.2)];
    assert_eq!(probability_reweights(&errors, [(4, &modifier)]), expected);

    let lookup = ErrorEdgeLookup::new(&errors);
    assert_eq!(lookup.project([(4, &modifier)]), expected);
}

#[test]
fn cached_lookup_projects_only_sparse_modifier_entries() {
    let errors: Vec<_> = (0..1_000).map(|error_index| ErrorIndex { eid: 4, error_index }).collect();
    let lookup = ErrorEdgeLookup::new(&errors);
    let modifier = bin::ProbabilityModifier {
        probabilities: vec![],
        sparse_indices: vec![731],
        sparse_probabilities: vec![0.42],
    };

    assert_eq!(lookup.project([(4, &modifier)]), vec![(731, 0.42)]);
}

#[test]
fn cached_lookup_is_sparse_in_eid_and_preserves_error_index_gaps() {
    let errors = vec![
        ErrorIndex {
            eid: 1_000_000_000,
            error_index: 0,
        },
        ErrorIndex {
            eid: 1_000_000_000,
            error_index: 2,
        },
        ErrorIndex { eid: 7, error_index: 3 },
    ];
    let lookup = ErrorEdgeLookup::new(&errors);
    let modifier = bin::ProbabilityModifier {
        probabilities: vec![],
        sparse_indices: vec![1, 2],
        sparse_probabilities: vec![0.13, 0.42],
    };

    assert_eq!(lookup.edges_by_eid.len(), 2);
    assert_eq!(lookup.edges_by_eid[&1_000_000_000].len(), 3);
    assert_eq!(lookup.project([(1_000_000_000, &modifier)]), vec![(1, 0.42)]);
}

#[test]
fn decoder_reweighting_policy_resolves_only_transport() {
    assert!(!DecoderReweighting::Auto.use_loaded(true, DecoderFeatures::empty()).unwrap());
    assert!(DecoderReweighting::Auto.use_loaded(true, DecoderFeatures::REWEIGHTS).unwrap());
    assert!(
        !DecoderReweighting::Disabled
            .use_loaded(true, DecoderFeatures::REWEIGHTS)
            .unwrap()
    );
    assert!(
        DecoderReweighting::Enabled
            .use_loaded(true, DecoderFeatures::empty())
            .is_err()
    );
    assert!(
        DecoderReweighting::Enabled
            .use_loaded(true, DecoderFeatures::REWEIGHTS)
            .unwrap()
    );
    assert!(
        !DecoderReweighting::Auto
            .use_loaded(false, DecoderFeatures::REWEIGHTS)
            .unwrap()
    );
    assert!(
        DecoderReweighting::Enabled
            .use_loaded(false, DecoderFeatures::REWEIGHTS)
            .is_err()
    );
}

async fn loaded_decoder_for_test(mock: &Arc<crate::decoder::MockDecoder>) -> (DynDecoder, LoadedDecoder) {
    let hypergraph = blackbox_decoder::DecodingHypergraph {
        vertex_num: 1,
        hyperedges: vec![blackbox_decoder::Hyperedge {
            vertices: vec![0],
            probability: 0.1,
        }],
    };
    let decoder = DynDecoder::Mock(mock.clone());
    let errors = Arc::new(vec![ErrorIndex { eid: 0, error_index: 0 }]);
    let loaded = load_projected_decoder(&decoder, hypergraph, errors, false, true, false)
        .await
        .unwrap();
    (decoder, loaded)
}

fn test_loss_info() -> blackbox_decoder::LossInfo {
    blackbox_decoder::LossInfo {
        sites: vec![blackbox_decoder::LossSite {
            source_edges: vec![0],
            probability: 0.2,
            heralds: vec![5],
            ..Default::default()
        }],
    }
}

#[tokio::test]
async fn projected_decode_sends_reweights_and_loss_together_when_supported() {
    let mock = Arc::new(crate::decoder::MockDecoder::new());
    let (client, loaded) = loaded_decoder_for_test(&mock).await;
    let loss = test_loss_info();
    decode_projected(
        &client,
        &loaded,
        BitVector {
            size: 1,
            data: vec![0b1000_0000],
        },
        vec![blackbox_decoder::EdgeReweight {
            edge: 0,
            probability: 0.3,
        }],
        Some(loss.clone()),
        true,
    )
    .await
    .unwrap();

    let state = mock.state.read().await;
    assert_eq!(state.decode_loaded_calls[0].reweights[0].probability, 0.3);
    assert_eq!(state.decode_loaded_calls[0].loss, Some(loss));
    assert!(state.decode_calls.is_empty());
}

#[tokio::test]
async fn projected_decode_materializes_reweights_without_dropping_loss() {
    let mock = Arc::new(crate::decoder::MockDecoder::with_features(DecoderFeatures::LOSS));
    let (client, loaded) = loaded_decoder_for_test(&mock).await;
    let loss = test_loss_info();
    decode_projected(
        &client,
        &loaded,
        BitVector {
            size: 1,
            data: vec![0b1000_0000],
        },
        vec![blackbox_decoder::EdgeReweight {
            edge: 0,
            probability: 0.3,
        }],
        Some(loss.clone()),
        false,
    )
    .await
    .unwrap();

    let state = mock.state.read().await;
    assert!((state.decode_calls[0].hypergraph.hyperedges[0].probability - 0.3).abs() < 1e-12);
    assert_eq!(state.decode_calls[0].loss, Some(loss));
    assert!(state.decode_loaded_calls.is_empty());
}

#[tokio::test]
async fn loaded_projection_zeros_isolated_vertices_without_renumbering() {
    let mock = Arc::new(crate::decoder::MockDecoder::new());
    let decoder = DynDecoder::Mock(Arc::clone(&mock));
    let hypergraph = blackbox_decoder::DecodingHypergraph {
        vertex_num: 2,
        hyperedges: vec![blackbox_decoder::Hyperedge {
            vertices: vec![0],
            probability: 0.1,
        }],
    };
    let errors = Arc::new(vec![ErrorIndex { eid: 0, error_index: 0 }]);

    let loaded = load_projected_decoder(&decoder, hypergraph, errors, false, true, true)
        .await
        .unwrap();

    assert_eq!(loaded.decoding_hypergraph.as_ref().unwrap().vertex_num, 2);
    assert_eq!(loaded.ignored_syndrome_vertices.as_slice(), &[1]);
    let projected = loaded.project_syndrome(BitVector {
        size: 2,
        data: vec![0b1100_0000],
    });
    assert_eq!(projected.size, 2);
    assert_eq!(projected.data, vec![0b1000_0000]);
    assert_eq!(mock.state.read().await.loaded_hypergraphs[&loaded.hid].vertex_num, 2);
}

#[test]
fn deduplication_keeps_the_highest_probability_correction() {
    let hypergraph = blackbox_decoder::DecodingHypergraph {
        vertex_num: 3,
        hyperedges: vec![
            blackbox_decoder::Hyperedge {
                probability: 0.31,
                vertices: vec![1, 0],
            },
            blackbox_decoder::Hyperedge {
                probability: 0.35,
                vertices: vec![0, 1],
            },
            blackbox_decoder::Hyperedge {
                probability: 0.02,
                vertices: vec![2],
            },
        ],
    };
    let errors = vec![
        ErrorIndex { eid: 0, error_index: 7 },
        ErrorIndex { eid: 0, error_index: 99 },
        ErrorIndex { eid: 0, error_index: 5 },
    ];

    let (deduplicated, _) = deduplicate_by_syndrome(&hypergraph, &errors);

    assert_eq!(deduplicated.hypergraph.hyperedges.len(), 2);
    assert_eq!(deduplicated.hypergraph.hyperedges[0].vertices, vec![0, 1]);
    assert_eq!(deduplicated.representatives[0], ErrorIndex { eid: 0, error_index: 99 });
    let combined = 0.31 + 0.35 - 2.0 * 0.31 * 0.35;
    assert!((deduplicated.hypergraph.hyperedges[0].probability - combined).abs() < 1e-12);
}

#[test]
fn deduplication_is_the_identity_when_every_syndrome_is_distinct() {
    let hypergraph = blackbox_decoder::DecodingHypergraph {
        vertex_num: 2,
        hyperedges: vec![
            blackbox_decoder::Hyperedge {
                probability: 0.1,
                vertices: vec![0],
            },
            blackbox_decoder::Hyperedge {
                probability: 0.2,
                vertices: vec![1],
            },
        ],
    };
    let errors = vec![ErrorIndex { eid: 0, error_index: 0 }, ErrorIndex { eid: 0, error_index: 1 }];
    let (deduplicated, _) = deduplicate_by_syndrome(&hypergraph, &errors);
    assert_eq!(deduplicated.hypergraph.hyperedges.len(), 2);
    assert_eq!(deduplicated.representatives.as_ref(), &errors);
}

#[test]
fn identity_grouping_matches_deduplicating_a_collision_free_graph() {
    let hypergraph = blackbox_decoder::DecodingHypergraph {
        vertex_num: 3,
        hyperedges: vec![
            blackbox_decoder::Hyperedge {
                probability: 0.1,
                vertices: vec![0],
            },
            blackbox_decoder::Hyperedge {
                probability: 0.2,
                vertices: vec![1],
            },
            blackbox_decoder::Hyperedge {
                probability: 0.0,
                vertices: vec![2],
            },
        ],
    };
    let errors = vec![
        ErrorIndex { eid: 0, error_index: 0 },
        ErrorIndex { eid: 0, error_index: 1 },
        ErrorIndex { eid: 1, error_index: 0 },
    ];
    let (identity_projection, identity) = prepare_decoder(hypergraph.clone(), Arc::new(errors.clone()), false);
    let (collapsed_projection, collapsed) = prepare_decoder(hypergraph, Arc::new(errors.clone()), true);
    assert_eq!(identity.hypergraph, collapsed.hypergraph);
    assert_eq!(identity.representatives, collapsed.representatives);
    let reweights = [(0, 0.15), (2, 0.3)];
    let (identity_reweights, identity_errors) = identity_projection.project_reweights(&reweights);
    let (collapsed_reweights, collapsed_errors) = collapsed_projection.project_reweights(&reweights);
    assert_eq!(identity_reweights, collapsed_reweights);
    assert_eq!(identity_errors.len(), collapsed_errors.len());
    for index in 0..identity_errors.len() {
        assert_eq!(identity_errors[index], collapsed_errors[index]);
    }
}

#[test]
fn shot_reweight_changes_the_merged_correction_representative() {
    let hypergraph = blackbox_decoder::DecodingHypergraph {
        vertex_num: 2,
        hyperedges: vec![
            blackbox_decoder::Hyperedge {
                probability: 0.3,
                vertices: vec![0, 1],
            },
            blackbox_decoder::Hyperedge {
                probability: 0.1,
                vertices: vec![1, 0],
            },
        ],
    };
    let errors = Arc::new(vec![
        ErrorIndex { eid: 0, error_index: 7 },
        ErrorIndex { eid: 0, error_index: 99 },
    ]);
    let (projection, prepared) = prepare_decoder(hypergraph, errors, true);

    assert_eq!(prepared.representatives[0], ErrorIndex { eid: 0, error_index: 7 });

    let (_, unchanged_errors) = projection.project_reweights(&[(0, 0.2)]);
    assert!(Arc::ptr_eq(&unchanged_errors.baseline, &projection.decoder_errors));
    assert!(unchanged_errors.replacements.is_empty());

    let (reweights, projected_errors) = projection.project_reweights(&[(1, 0.4)]);

    assert_eq!(projected_errors[0], ErrorIndex { eid: 0, error_index: 99 });
    assert_eq!(projected_errors.replacements.len(), 1);
    assert_eq!(reweights.len(), 1);
    assert!((reweights[0].1 - exclusive_probability_of(0.3, 0.4)).abs() < 1e-12);
}

#[test]
fn shot_reweight_re_elects_only_affected_merged_representatives() {
    let hypergraph = blackbox_decoder::DecodingHypergraph {
        vertex_num: 3,
        hyperedges: vec![
            blackbox_decoder::Hyperedge {
                probability: 0.3,
                vertices: vec![0, 1],
            },
            blackbox_decoder::Hyperedge {
                probability: 0.1,
                vertices: vec![1, 0],
            },
            blackbox_decoder::Hyperedge {
                probability: 0.25,
                vertices: vec![1, 2],
            },
            blackbox_decoder::Hyperedge {
                probability: 0.2,
                vertices: vec![2, 1],
            },
        ],
    };
    let errors = Arc::new(vec![
        ErrorIndex { eid: 0, error_index: 7 },
        ErrorIndex { eid: 0, error_index: 99 },
        ErrorIndex { eid: 1, error_index: 12 },
        ErrorIndex { eid: 1, error_index: 13 },
    ]);
    let (projection, prepared) = prepare_decoder(hypergraph, errors, true);

    assert_eq!(prepared.representatives[0], ErrorIndex { eid: 0, error_index: 7 });
    assert_eq!(prepared.representatives[1], ErrorIndex { eid: 1, error_index: 12 });

    let (reweights, projected_errors) = projection.project_reweights(&[(0, 0.05)]);

    assert_eq!(projected_errors[0], ErrorIndex { eid: 0, error_index: 99 });
    assert_eq!(projected_errors[1], ErrorIndex { eid: 1, error_index: 12 });
    assert_eq!(reweights.len(), 1);
    assert!((reweights[0].1 - exclusive_probability_of(0.05, 0.1)).abs() < 1e-12);

    let (reweights, projected_errors) = projection.project_reweights(&[(0, 0.05), (1, 0.15)]);

    assert_eq!(projected_errors[0], ErrorIndex { eid: 0, error_index: 99 });
    assert_eq!(projected_errors[1], ErrorIndex { eid: 1, error_index: 12 });
    assert_eq!(reweights.len(), 1);
    assert!((reweights[0].1 - exclusive_probability_of(0.05, 0.15)).abs() < 1e-12);
}

#[test]
fn translated_reweights_match_deduplicating_an_already_reweighted_graph() {
    let priors = [3.7e-4, 3.7e-4, 0.0, 0.02];
    let vertices = [vec![0, 1], vec![1, 0], vec![0, 1], vec![2]];
    let errors: Vec<ErrorIndex> = (0..4).map(|error_index| ErrorIndex { eid: 0, error_index }).collect();
    let base = blackbox_decoder::DecodingHypergraph {
        vertex_num: 3,
        hyperedges: priors
            .iter()
            .zip(vertices.iter())
            .map(|(&probability, vertex_set)| blackbox_decoder::Hyperedge {
                probability,
                vertices: vertex_set.clone(),
            })
            .collect(),
    };
    let reweights = vec![(2u64, 0.31)];
    let (projection, _) = prepare_decoder(base.clone(), Arc::new(errors.clone()), true);
    let (translated, _) = projection.project_reweights(&reweights);
    let mut reweighted = base.clone();
    apply_reweights(&mut reweighted, &reweights);
    let (expected, _) = deduplicate_by_syndrome(&reweighted, &errors);

    assert_eq!(translated.len(), 1);
    let (edge, probability) = translated[0];
    assert!((probability - expected.hypergraph.hyperedges[edge as usize].probability).abs() < 1e-12);
    assert!(translated.iter().all(|&(index, _)| index != 1));
}
