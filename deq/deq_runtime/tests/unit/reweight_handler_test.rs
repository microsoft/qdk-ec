//! Unit tests for the coordinator reweight-handling pass.

use super::*;

#[tokio::test]
async fn correction_weights_use_merged_priors_and_shot_overrides() {
    let mock = Arc::new(crate::decoder::MockDecoder::new());
    let decoder = DynDecoder::Mock(mock);
    let hypergraph = blackbox_decoder::DecodingHypergraph {
        vertex_num: 2,
        hyperedges: vec![
            blackbox_decoder::Hyperedge {
                vertices: vec![0],
                probability: 0.1,
            },
            blackbox_decoder::Hyperedge {
                vertices: vec![0],
                probability: 0.2,
            },
            blackbox_decoder::Hyperedge {
                vertices: vec![1],
                probability: 0.3,
            },
        ],
    };
    let errors = Arc::new((0..3).map(|error_index| ErrorIndex { eid: 0, error_index }).collect());
    let (projection, prepared) = prepare_decoder(hypergraph, errors, vec![vec![]; 3], true, |_| 0);
    let correction = blackbox_decoder::ParityFactor { subgraph: vec![1, 0] };
    let expected = vec![weight_of(0.3), weight_of(0.26)];
    for (actual, expected) in correction_weights(&prepared.hypergraph, &correction).iter().zip(&expected) {
        assert!((actual - expected).abs() < 1e-12);
    }
    let loaded = load_projected_decoder(&decoder, projection, prepared, false, false)
        .await
        .unwrap();
    assert!(loaded.decoding_hypergraph.is_none());
    for (actual, expected) in loaded.correction_weights(&correction, &[]).iter().zip(&expected) {
        assert!((actual - expected).abs() < 1e-12);
    }
    let reweights = vec![
        blackbox_decoder::EdgeReweight {
            edge: 1,
            probability: 0.4,
        },
        blackbox_decoder::EdgeReweight {
            edge: 0,
            probability: 0.5,
        },
        blackbox_decoder::EdgeReweight {
            edge: 1,
            probability: 0.8,
        },
    ];
    assert_eq!(loaded.correction_weights(&correction, &reweights), vec![weight_of(0.8), 0.0]);
    assert!(
        loaded
            .correction_weights(&blackbox_decoder::ParityFactor::default(), &reweights)
            .is_empty()
    );
}

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

#[test]
fn hard_decoding_zeroes_syndrome_free_logical_edges() {
    let hypergraph = blackbox_decoder::DecodingHypergraph {
        vertex_num: 1,
        hyperedges: vec![
            blackbox_decoder::Hyperedge {
                vertices: vec![],
                probability: 0.1,
            },
            blackbox_decoder::Hyperedge {
                vertices: vec![0],
                probability: 0.2,
            },
        ],
    };

    let hard = hard_decoding_hypergraph(hypergraph, &[vec![0], vec![0]]);

    assert_eq!(hard.hyperedges[0].probability, 0.0);
    assert_eq!(hard.hyperedges[1].probability, 0.2);
}

#[tokio::test]
async fn loaded_decoder_preserves_syndrome_free_logical_priors_for_scoring() {
    let mock = Arc::new(crate::decoder::MockDecoder::new());
    let decoder = DynDecoder::Mock(Arc::clone(&mock));
    let (projection, prepared) = prepare_decoder(
        blackbox_decoder::DecodingHypergraph {
            vertex_num: 0,
            hyperedges: vec![blackbox_decoder::Hyperedge {
                vertices: vec![],
                probability: 0.1,
            }],
        },
        Arc::new(vec![ErrorIndex { eid: 0, error_index: 0 }]),
        vec![vec![0]],
        false,
        |_| 0,
    );
    let loaded = load_projected_decoder(&decoder, projection, prepared, true, false)
        .await
        .unwrap();

    assert_eq!(loaded.decoding_hypergraph.as_ref().unwrap().hyperedges[0].probability, 0.1);
    assert_eq!(
        mock.state.read().await.loaded_hypergraphs[&loaded.hid].hyperedges[0].probability,
        0.0
    );

    decode_projected(
        &decoder,
        &loaded,
        BitVector::default(),
        None,
        vec![blackbox_decoder::EdgeReweight {
            edge: 0,
            probability: 0.4,
        }],
        None,
        true,
    )
    .await
    .unwrap();
    assert!(mock.state.read().await.decode_loaded_calls[0].reweights.is_empty());
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
    let (projection, prepared) = prepare_decoder(hypergraph, errors, vec![vec![]], false, |_| 0);
    let loaded = load_projected_decoder(&decoder, projection, prepared, true, false)
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
        Some(42),
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
    assert_eq!(state.decode_loaded_calls[0].decoder_seed, Some(42));
    assert!(state.decode_calls.is_empty());
}

#[tokio::test]
async fn projected_decode_materializes_reweights_without_dropping_loss() {
    let mock = Arc::new(crate::decoder::MockDecoder::with_features(
        DecoderFeatures::LOSS | DecoderFeatures::SEED,
    ));
    let (client, loaded) = loaded_decoder_for_test(&mock).await;
    let loss = test_loss_info();
    decode_projected(
        &client,
        &loaded,
        BitVector {
            size: 1,
            data: vec![0b1000_0000],
        },
        Some(42),
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
    assert_eq!(state.decode_calls[0].decoder_seed, Some(42));
    assert!(state.decode_loaded_calls.is_empty());
}

#[tokio::test]
async fn materialized_reweights_reject_invalid_edges_without_panicking() {
    let mock = Arc::new(crate::decoder::MockDecoder::new());
    let (decoder, loaded) = loaded_decoder_for_test(&mock).await;
    for edge in [1, u64::MAX] {
        let error = decode_projected(
            &decoder,
            &loaded,
            BitVector { size: 1, data: vec![0] },
            None,
            vec![blackbox_decoder::EdgeReweight { edge, probability: 0.3 }],
            None,
            false,
        )
        .await
        .unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
    }
    assert!(mock.state.read().await.decode_calls.is_empty());
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

    let (projection, prepared) = prepare_decoder(hypergraph, errors, vec![vec![]], false, |_| 0);
    let loaded = load_projected_decoder(&decoder, projection, prepared, true, true)
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

    let (deduplicated, _) = deduplicate_by_syndrome(&hypergraph, &errors, &vec![vec![]; 3], |_| 0);

    assert_eq!(deduplicated.hypergraph.hyperedges.len(), 2);
    assert_eq!(deduplicated.hypergraph.hyperedges[0].vertices, vec![0, 1]);
    assert_eq!(deduplicated.representatives[0], ErrorIndex { eid: 0, error_index: 99 });
    let combined = 0.31 + 0.35 - 2.0 * 0.31 * 0.35;
    assert!((deduplicated.hypergraph.hyperedges[0].probability - combined).abs() < 1e-12);
}

#[test]
fn deduplication_keeps_equal_syndromes_with_different_logical_flips() {
    let hypergraph = blackbox_decoder::DecodingHypergraph {
        vertex_num: 1,
        hyperedges: vec![
            blackbox_decoder::Hyperedge {
                probability: 0.1,
                vertices: vec![0],
            },
            blackbox_decoder::Hyperedge {
                probability: 0.2,
                vertices: vec![0],
            },
        ],
    };
    let errors = vec![ErrorIndex { eid: 0, error_index: 0 }, ErrorIndex { eid: 0, error_index: 1 }];

    let logical_flips = vec![vec![], vec![0]];
    let (deduplicated, _) = deduplicate_by_syndrome(&hypergraph, &errors, &logical_flips, |_| 0);

    assert_eq!(deduplicated.hypergraph.hyperedges.len(), 2);
    assert_eq!(deduplicated.representatives.as_ref(), &errors);
    assert_eq!(deduplicated.logical_flips.as_ref(), &logical_flips);
}

#[test]
fn merge_classes_preserve_edge_order_and_reweights() {
    let hypergraph = blackbox_decoder::DecodingHypergraph {
        vertex_num: 4,
        hyperedges: [0.49, 0.1, 0.3, 0.2, 0.4, 0.05]
            .into_iter()
            .enumerate()
            .map(|(edge, probability)| blackbox_decoder::Hyperedge {
                vertices: if edge < 4 { vec![0, 2] } else { vec![1] },
                probability,
            })
            .collect(),
    };
    let errors: Arc<Vec<_>> = Arc::new((0..6).map(|eid| ErrorIndex { eid, error_index: 0 }).collect());
    let (projection, prepared) = prepare_decoder(hypergraph.clone(), Arc::clone(&errors), vec![vec![]; 6], true, |error| {
        error.eid % 2
    });
    assert_eq!(projection.base_hypergraph, hypergraph);
    assert_eq!(projection.base_errors, errors);
    assert_eq!(prepared.hypergraph.vertex_num, 4);
    assert_eq!(
        prepared
            .hypergraph
            .hyperedges
            .iter()
            .map(|edge| edge.vertices.clone())
            .collect::<Vec<_>>(),
        vec![vec![0, 2], vec![0, 2], vec![1], vec![1]],
    );
    assert_eq!(
        prepared.representatives.iter().map(|error| error.eid).collect::<Vec<_>>(),
        vec![0, 3, 4, 5]
    );
    let (reweights, errors) = projection.project_reweights(&[(2, 0.6), (1, 0.45), (3, 0.1)]);
    assert_eq!(reweights.iter().map(|&(edge, _)| edge).collect::<Vec<_>>(), vec![0, 1]);
    assert!((reweights[0].1 - 0.502).abs() < 1e-12);
    assert!((reweights[1].1 - 0.46).abs() < 1e-12);
    assert_eq!(
        (0..errors.len()).map(|edge| errors[edge].eid).collect::<Vec<_>>(),
        vec![2, 1, 4, 5]
    );
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
    let (deduplicated, _) = deduplicate_by_syndrome(&hypergraph, &errors, &vec![vec![]; 2], |_| 0);
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
    let logical_flips = vec![vec![]; 3];
    let (identity_projection, identity) = prepare_decoder(
        hypergraph.clone(),
        Arc::new(errors.clone()),
        logical_flips.clone(),
        false,
        |_| panic!("disabled merging must not evaluate merge classes"),
    );
    let (collapsed_projection, collapsed) =
        prepare_decoder(hypergraph, Arc::new(errors.clone()), logical_flips, true, |_| 0);
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
    let (projection, prepared) = prepare_decoder(hypergraph, errors, vec![vec![]; 2], true, |_| 0);

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
    let (projection, prepared) = prepare_decoder(hypergraph, errors, vec![vec![]; 4], true, |_| 0);

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
    let logical_flips = vec![vec![]; errors.len()];
    let (projection, _) = prepare_decoder(base.clone(), Arc::new(errors.clone()), logical_flips.clone(), true, |_| 0);
    let (translated, _) = projection.project_reweights(&reweights);
    let mut reweighted = base.clone();
    apply_reweights(&mut reweighted, reweights.iter().copied());
    let (expected, _) = deduplicate_by_syndrome(&reweighted, &errors, &logical_flips, |_| 0);

    assert_eq!(translated.len(), 1);
    let (edge, probability) = translated[0];
    assert!((probability - expected.hypergraph.hyperedges[edge as usize].probability).abs() < 1e-12);
    assert!(translated.iter().all(|&(index, _)| index != 1));
}
