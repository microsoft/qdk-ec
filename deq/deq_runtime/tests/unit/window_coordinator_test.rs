//! Unit tests for the `WindowCoordinator`'s cache-key helpers.
//!
//! `build_modifier_fingerprints` and `committing_local_cids_sorted` are
//! the two pieces of state that the `WindowCoordinator` folds into the
//! `DecoderCacheKey` beyond the `RelativeProgram`.  These tests pin
//! down their behaviour so that:
//!
//!   - per-eid modifier changes (probability / `check_bias`) and
//!     per-etype structural changes change the fingerprint vector;
//!   - the commit-region vector is filtered to local cids and
//!     canonicalised by sorting (so equal sets map to equal vectors).
use super::*;
use crate::bin::error_model::ErrorModelModifier;
use crate::bin::error_model_type::{Error, RemoteCheckModel, remote_check_model};
use crate::coordinator::ErrorModelFingerprint;

fn scoring_hypergraph() -> DecodingHypergraph {
    DecodingHypergraph {
        vertex_num: 1,
        hyperedges: [0.49, 0.1, 0.01]
            .into_iter()
            .map(|probability| Hyperedge {
                vertices: vec![0],
                probability,
            })
            .collect(),
    }
}

#[test]
fn window_merging_preserves_commit_ownership_without_logical_flips() {
    let hypergraph = DecodingHypergraph {
        vertex_num: 1,
        hyperedges: [0.1, 0.49, 0.2]
            .into_iter()
            .map(|probability| Hyperedge {
                vertices: vec![0],
                probability,
            })
            .collect(),
    };
    let errors = Arc::new((0..3).map(|eid| ErrorIndex { eid, error_index: 0 }).collect());
    let committing_eids: HashSet<_> = [0, 2].into_iter().collect();
    let (projection, prepared) = prepare_decoder(hypergraph, errors, vec![vec![]; 3], true, |error| {
        usize::from(committing_eids.contains(&error.eid))
    });
    assert_eq!(prepared.hypergraph.hyperedges.len(), 2);
    assert_eq!(prepared.representatives[0].eid, 2);
    assert_eq!(prepared.representatives[1].eid, 1);
    assert!((prepared.hypergraph.hyperedges[0].probability - 0.26).abs() < 1e-12);
    let (_, reweighted_errors) = projection.project_reweights(&[(0, 0.45)]);
    assert_eq!(reweighted_errors[0].eid, 0);
    assert_eq!(reweighted_errors[1].eid, 1);
}

#[tokio::test]
async fn commit_region_freezes_buffer_and_remaps_reweights() {
    for (persistent, use_loaded_reweights) in [(false, false), (true, false), (true, true)] {
        let mock = Arc::new(crate::decoder::MockDecoder::new());
        mock.set_response(vec![0b0100_0000], vec![0, 1]).await;
        let hypergraph = scoring_hypergraph();
        let scorer = CommitRegionDecoder::new(&hypergraph, &[vec![], vec![0], vec![]], &[false, true, true], 1, persistent);
        let syndrome = BitVector {
            size: 1,
            data: vec![0x80],
        };
        let baseline = ParityFactor { subgraph: vec![0] };
        let reweights = vec![
            EdgeReweight {
                edge: 0,
                probability: 0.9,
            },
            EdgeReweight {
                edge: 2,
                probability: 0.2,
            },
        ];
        let (projected_syndrome, projected_baseline, projected_reweights) = scorer
            .project(&hypergraph, syndrome.clone(), &baseline, reweights.clone())
            .unwrap();
        assert_eq!(projected_syndrome, BitVector { size: 1, data: vec![0] });
        assert!(projected_baseline.subgraph.is_empty());
        assert_eq!(
            projected_reweights,
            vec![EdgeReweight {
                edge: 1,
                probability: 0.2
            }]
        );

        let problem = scorer
            .problem(
                DynDecoder::Mock(Arc::clone(&mock)),
                &hypergraph,
                syndrome,
                &baseline,
                reweights,
                use_loaded_reweights,
            )
            .unwrap();
        assert!((problem.probability(0).await.unwrap() - 1.0 / 37.0).abs() < 1e-12);
        let state = mock.state.read().await;
        let forced = if persistent && use_loaded_reweights {
            &state.loaded_hypergraphs[&state.decode_loaded_calls[0].hid]
        } else {
            assert!(state.loaded_hypergraphs.is_empty());
            &state.decode_calls[0].hypergraph
        };
        assert_eq!(forced.hyperedges.len(), 2);
        assert_eq!(forced.hyperedges[0].vertices, vec![0, 1]);
        assert_eq!(forced.hyperedges[1].vertices, vec![0]);
    }
}

#[test]
fn commit_region_alternatives_preserve_the_original_window_syndrome() {
    let hypergraph = DecodingHypergraph {
        vertex_num: 3,
        hyperedges: [vec![0], vec![0, 1], vec![1], vec![1, 2], vec![2], vec![0, 2]]
            .into_iter()
            .map(|vertices| Hyperedge {
                vertices,
                probability: 0.1,
            })
            .collect(),
    };
    let scorer = CommitRegionDecoder::new(
        &hypergraph,
        &[vec![], vec![0], vec![], vec![], vec![], vec![]],
        &[false, true, false, true, false, true],
        1,
        false,
    );
    let restricted = DecodingHypergraph {
        vertex_num: hypergraph.vertex_num,
        hyperedges: [1, 3, 5].map(|edge| hypergraph.hyperedges[edge].clone()).to_vec(),
    };
    for selection in 0..64 {
        let baseline = ParityFactor {
            subgraph: (0..6).filter(|edge| selection & (1 << edge) != 0).collect(),
        };
        let mut syndrome = BitVector { size: 3, data: vec![0] };
        for &edge in &baseline.subgraph {
            for &vertex in &hypergraph.hyperedges[usize::try_from(edge).unwrap()].vertices {
                flip_bit(&mut syndrome, vertex);
            }
        }
        let (projected_syndrome, projected_baseline, _) =
            scorer.project(&hypergraph, syndrome.clone(), &baseline, vec![]).unwrap();
        assert!(is_parity_factor(&restricted, &projected_baseline, &projected_syndrome));
        for alternative in 0..8 {
            let candidate = ParityFactor {
                subgraph: (0..3).filter(|edge| alternative & (1 << edge) != 0).collect(),
            };
            if is_parity_factor(&restricted, &candidate, &projected_syndrome) {
                let combined = ParityFactor {
                    subgraph: baseline
                        .subgraph
                        .iter()
                        .copied()
                        .filter(|edge| edge % 2 == 0)
                        .chain(candidate.subgraph.iter().map(|edge| 2 * edge + 1))
                        .collect(),
                };
                assert!(is_parity_factor(&hypergraph, &combined, &syndrome));
            }
        }
    }
}

#[tokio::test]
async fn commit_region_cannot_escape_through_buffer() {
    let mock = Arc::new(crate::decoder::MockDecoder::new());
    let hypergraph = scoring_hypergraph();
    let scorer = CommitRegionDecoder::new(&hypergraph, &[vec![], vec![0], vec![]], &[false, true, false], 1, true);
    let problem = scorer
        .problem(
            DynDecoder::Mock(Arc::clone(&mock)),
            &hypergraph,
            BitVector { size: 1, data: vec![0] },
            &ParityFactor::default(),
            vec![],
            true,
        )
        .unwrap();
    assert!(problem.probability(0).await.unwrap().abs() < f64::EPSILON);
    let state = mock.state.read().await;
    assert!(state.loaded_hypergraphs.is_empty());
    assert!(state.decode_loaded_calls.is_empty());
}

#[test]
fn commit_region_rejects_invalid_baselines_and_reweights() {
    let hypergraph = scoring_hypergraph();
    let scorer = CommitRegionDecoder::new(&hypergraph, &[vec![], vec![0], vec![]], &[false, true, true], 1, false);
    let syndrome = BitVector {
        size: 1,
        data: vec![0x80],
    };
    for subgraph in [vec![], vec![3], vec![u64::MAX], vec![0, 0]] {
        let error = scorer
            .project(&hypergraph, syndrome.clone(), &ParityFactor { subgraph }, vec![])
            .unwrap_err();
        assert_eq!(error.code(), tonic::Code::Internal);
    }
    for edge in [3, u64::MAX] {
        let error = scorer
            .project(
                &hypergraph,
                syndrome.clone(),
                &ParityFactor { subgraph: vec![0] },
                vec![EdgeReweight { edge, probability: 0.1 }],
            )
            .unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
    }
}

#[tokio::test]
async fn commit_projection_errors_reach_registered_scores() {
    let coordinator = WindowCoordinator::new(
        serde_json::json!({ "forced_gap": true }),
        DynDecoder::Mock(Arc::new(crate::decoder::MockDecoder::new())),
    );
    let target = CorrectionBasis::Readout { gid: 1, index: 0 };
    let error = Status::internal("invalid commit-region baseline");
    coordinator.register_forced_gap_scores(&[target], &Err(error.clone())).await;
    let score = coordinator.forced_gap_state.as_ref().unwrap().read().await.scores[&target].clone();
    let actual = score.probability().await.unwrap_err();
    assert_eq!(actual.code(), error.code());
    assert_eq!(actual.message(), error.message());
}

#[tokio::test]
async fn identical_commit_constraints_share_one_forced_solve() {
    for persistent in [false, true] {
        let mock = Arc::new(crate::decoder::MockDecoder::new());
        mock.set_response(vec![0b1000_0000], vec![0]).await;
        let hypergraph = DecodingHypergraph {
            vertex_num: 0,
            hyperedges: [0.2, 0.1]
                .into_iter()
                .map(|probability| Hyperedge {
                    vertices: vec![],
                    probability,
                })
                .collect(),
        };
        let scorer = CommitRegionDecoder::new(&hypergraph, &[vec![0], vec![0, 1]], &[false, true], 2, persistent);
        for _shot in 0..2 {
            let probabilities = scorer
                .problem(
                    DynDecoder::Mock(Arc::clone(&mock)),
                    &hypergraph,
                    BitVector::default(),
                    &ParityFactor { subgraph: vec![0] },
                    vec![],
                    true,
                )
                .unwrap()
                .probabilities()
                .await
                .unwrap();
            assert_eq!(probabilities.len(), 2);
            assert!(probabilities.iter().all(|probability| (probability - 0.1).abs() < 1e-12));
        }
        let state = mock.state.read().await;
        assert_eq!(state.decode_calls.len() + state.decode_loaded_calls.len(), 2);
        assert_eq!(state.loaded_hypergraphs.len(), usize::from(persistent));
    }
}

#[cfg(feature = "tesseract")]
#[tokio::test]
async fn tesseract_finds_the_commit_only_alternative_instead_of_the_cheaper_buffer_path() {
    let decoder = crate::decoder::DecoderType::BlackBoxTesseract.create(serde_json::json!({ "parallel": 1 }));
    let hypergraph = Arc::new(scoring_hypergraph());
    let logical_flips = Arc::new(vec![vec![], vec![0], vec![]]);
    for persistent in [false, true] {
        let unrestricted = Arc::new(ForcedGapGraph::new(
            Arc::clone(&hypergraph),
            Arc::clone(&logical_flips),
            1,
            persistent,
        ));
        let restricted = CommitRegionDecoder::new(&hypergraph, &logical_flips, &[false, true, true], 1, persistent);
        let syndrome = BitVector {
            size: 1,
            data: vec![0x80],
        };
        let baseline = ParityFactor { subgraph: vec![0] };
        let whole_window = unrestricted
            .problem(decoder.clone(), syndrome.clone(), baseline.clone(), vec![], false)
            .probability(0)
            .await
            .unwrap();
        let commit_only = restricted
            .problem(decoder.clone(), &hypergraph, syndrome, &baseline, vec![], false)
            .unwrap()
            .probability(0)
            .await
            .unwrap();
        assert!((whole_window - 17.0 / 164.0).abs() < 1e-12);
        assert!((commit_only - 1.0 / 892.0).abs() < 1e-12);
    }
}

// ─── helpers ─────────────────────────────────────────────────────────

fn mapping_with_eids(global_eid_of: Vec<u64>) -> RelativeMapping {
    RelativeMapping {
        global_eid_of,
        ..Default::default()
    }
}

fn mapping_with_local_cids(local_cid_of: &[(u64, usize)]) -> RelativeMapping {
    let mut map = RelativeMapping::default();
    for &(gcid, lcid) in local_cid_of {
        map.local_cid_of.insert(gcid, lcid);
    }
    map
}

fn pm_dense(probabilities: Vec<f64>) -> bin::ProbabilityModifier {
    bin::ProbabilityModifier {
        probabilities,
        sparse_indices: vec![],
        sparse_probabilities: vec![],
    }
}

fn make_error_model_instance(eid: u64, etype: u64, modifier: Option<bin::ProbabilityModifier>) -> bin::ErrorModel {
    bin::ErrorModel {
        eid,
        etype,
        cid: 1,
        modifier: modifier.map(|p| ErrorModelModifier {
            probability_modifier: Some(p),
            reroute_remote_check_models: vec![],
        }),
        ..Default::default()
    }
}

fn make_error_model(instance: bin::ErrorModel, remote_check_models: Vec<Option<RemoteCheckModel>>) -> ErrorModel {
    ErrorModel {
        instance,
        modified_remote_check_models: Arc::new(remote_check_models),
    }
}

fn make_emt(etype: u64, errors: Vec<Error>) -> bin::ErrorModelType {
    bin::ErrorModelType {
        etype,
        ctype: 1,
        errors,
        remote_check_models: vec![],
        ..Default::default()
    }
}

fn make_error(probability: f64) -> Error {
    Error {
        checks: vec![bin::error_model_type::RemoteCheck {
            remote_check_model: None,
            check_index: 0,
        }],
        probability,
        ..Default::default()
    }
}

fn make_remote_check(check_bias: u64) -> RemoteCheckModel {
    RemoteCheckModel {
        previous_remote_check_model: None,
        port: Some(remote_check_model::Port::Output(0)),
        expecting_ctype: 0,
        check_bias,
        absolute_cid: None,
        ..Default::default()
    }
}

fn history_gadget(gid: u64, state: GadgetState, next_gid: Option<u64>) -> Gadget {
    Gadget {
        instance: bin::Gadget {
            gid,
            ..Default::default()
        },
        outcomes: watch::channel(None).0,
        probability_modifiers: vec![],
        loss_mask: None,
        binding_cid: Some(gid),
        outputs: vec![watch::channel(next_gid.map(|gid| bin::gadget::Connector { gid, port: 0 })).0],
        pauli_frame: watch::channel(None).0,
        correction_count: 0,
        correction_weight: 0.0,
        is_free_hop: false,
        state: watch::channel(state).0,
    }
}

#[test]
fn retained_remote_check_resolves_through_a_gadget_outside_the_window() {
    let gadgets = HashMap::from([
        (1, history_gadget(1, GadgetState::Uncommitted, Some(2))),
        (2, history_gadget(2, GadgetState::Uncommitted, Some(3))),
        (3, history_gadget(3, GadgetState::Committed, None)),
    ]);
    let mut terminal = make_remote_check(0);
    terminal.previous_remote_check_model = Some(0);
    let error_model = make_error_model(
        make_error_model_instance(1, 1, None),
        vec![Some(make_remote_check(0)), Some(terminal)],
    );
    assert_eq!(
        WindowCoordinator::expand_remote_check_models_in_window(1, &error_model, &gadgets, &HashSet::from([1, 3])),
        vec![None, Some(3)],
    );
}

fn history_check_model(cid: u64, attaching_eid_vec: Vec<u64>) -> CheckModel {
    CheckModel {
        instance: bin::CheckModel {
            cid,
            gid: cid,
            ..Default::default()
        },
        attaching_eid_vec,
        modified_remote_gadgets: Arc::new(vec![]),
        expanded_remote_gadgets: Some(vec![]),
        syndrome: watch::channel(None).0,
        syndrome_count: 0,
        referring_eids: vec![],
    }
}

#[test]
fn history_retention_preserves_committed_checks_without_waiting_for_future_gadgets() {
    for target_state in [
        GadgetState::Committed,
        GadgetState::Uncommitted,
        GadgetState::Decoding { leader_gid: 3 },
    ] {
        let target_is_committed = target_state == GadgetState::Committed;
        let gadgets = HashMap::from([
            (1, history_gadget(1, GadgetState::Decoding { leader_gid: 1 }, Some(2))),
            (2, history_gadget(2, GadgetState::Uncommitted, Some(3))),
            (3, history_gadget(3, target_state, None)),
        ]);
        let mut terminal = make_remote_check(0);
        terminal.previous_remote_check_model = Some(0);
        let mut future = make_remote_check(0);
        future.previous_remote_check_model = Some(1);
        let error_models = HashMap::from([(
            1,
            make_error_model(
                make_error_model_instance(1, 1, None),
                vec![Some(make_remote_check(0)), Some(terminal), Some(future)],
            ),
        )]);
        let check_models = HashMap::from([(1, history_check_model(1, vec![1])), (3, history_check_model(3, vec![]))]);
        let original = HashSet::from([1]);
        let retained = WindowCoordinator::retain_committed_check_history(&original, &gadgets, &check_models, &error_models);
        assert_eq!(
            retained,
            if target_is_committed {
                HashSet::from([1, 3])
            } else {
                HashSet::from([1])
            }
        );
        assert_eq!(original, HashSet::from([1]));
    }
}

#[test]
fn history_retention_uses_absolute_reroutes_and_ignores_disabled_or_missing_targets() {
    let gadgets = HashMap::from([
        (1, history_gadget(1, GadgetState::Uncommitted, None)),
        (3, history_gadget(3, GadgetState::Committed, None)),
    ]);
    let check_models = HashMap::from([(1, history_check_model(1, vec![1])), (3, history_check_model(3, vec![]))]);
    let error_models = HashMap::from([(
        1,
        make_error_model(
            make_error_model_instance(1, 1, None),
            vec![
                None,
                Some(RemoteCheckModel {
                    absolute_cid: Some(3),
                    ..Default::default()
                }),
                Some(RemoteCheckModel {
                    absolute_cid: Some(99),
                    ..Default::default()
                }),
            ],
        ),
    )]);
    assert_eq!(
        WindowCoordinator::retain_committed_check_history(&HashSet::from([1]), &gadgets, &check_models, &error_models),
        HashSet::from([1, 3]),
    );
    gadgets[&1].state.send_replace(GadgetState::Committed);
    assert_eq!(
        WindowCoordinator::retain_committed_check_history(&HashSet::from([1]), &gadgets, &check_models, &error_models),
        HashSet::from([1]),
    );
}

// ─── build_modifier_fingerprints ─────────────────────────────────────

#[test]
fn build_modifier_fingerprints_picks_up_probability_modifier() {
    let mapping = mapping_with_eids(vec![1]);
    let mut emts: HashMap<u64, Arc<bin::ErrorModelType>> = HashMap::new();
    emts.insert(1, Arc::new(make_emt(1, vec![make_error(0.1)])));

    let mut models_a: HashMap<u64, ErrorModel> = HashMap::new();
    models_a.insert(
        1,
        make_error_model(make_error_model_instance(1, 1, Some(pm_dense(vec![0.1]))), vec![]),
    );

    let mut models_b: HashMap<u64, ErrorModel> = HashMap::new();
    models_b.insert(
        1,
        make_error_model(make_error_model_instance(1, 1, Some(pm_dense(vec![0.2]))), vec![]),
    );

    let fps_a = build_modifier_fingerprints(&mapping, &models_a, &emts);
    let fps_b = build_modifier_fingerprints(&mapping, &models_b, &emts);
    assert_ne!(fps_a, fps_b);
}

/// `check_bias` lives in `modified_remote_check_models` (not the
/// instance modifier), so this is a separate code path inside
/// `ErrorModelFingerprint::new`.  Two windows that resolve the same
/// `eid` to different remote-check biases must map to different
/// fingerprints.
#[test]
fn build_modifier_fingerprints_picks_up_check_bias() {
    let mapping = mapping_with_eids(vec![1]);
    let mut emts: HashMap<u64, Arc<bin::ErrorModelType>> = HashMap::new();
    emts.insert(1, Arc::new(make_emt(1, vec![make_error(0.1)])));

    let mut models_bias0: HashMap<u64, ErrorModel> = HashMap::new();
    models_bias0.insert(
        1,
        make_error_model(make_error_model_instance(1, 1, None), vec![Some(make_remote_check(0))]),
    );
    let mut models_bias5: HashMap<u64, ErrorModel> = HashMap::new();
    models_bias5.insert(
        1,
        make_error_model(make_error_model_instance(1, 1, None), vec![Some(make_remote_check(5))]),
    );

    let fps0 = build_modifier_fingerprints(&mapping, &models_bias0, &emts);
    let fps5 = build_modifier_fingerprints(&mapping, &models_bias5, &emts);
    assert_ne!(fps0, fps5);
}

#[test]
fn build_modifier_fingerprints_picks_up_etype_structure() {
    let mapping = mapping_with_eids(vec![1]);
    let mut models: HashMap<u64, ErrorModel> = HashMap::new();
    models.insert(1, make_error_model(make_error_model_instance(1, 1, None), vec![]));

    let mut emts_v1: HashMap<u64, Arc<bin::ErrorModelType>> = HashMap::new();
    emts_v1.insert(1, Arc::new(make_emt(1, vec![make_error(0.1)])));

    let mut emts_v2: HashMap<u64, Arc<bin::ErrorModelType>> = HashMap::new();
    emts_v2.insert(1, Arc::new(make_emt(1, vec![make_error(0.2)])));

    let fps_v1 = build_modifier_fingerprints(&mapping, &models, &emts_v1);
    let fps_v2 = build_modifier_fingerprints(&mapping, &models, &emts_v2);
    assert_ne!(fps_v1, fps_v2);
}

// ─── committing_local_cids_sorted ────────────────────────────────────

/// Global cids that fall outside this window's `local_cid_of` mapping
/// are dropped — they can't influence which hyperedges are kept for
/// *this* window.
#[test]
fn committing_local_cids_sorted_filters_out_global_cids_not_in_window() {
    let mapping = mapping_with_local_cids(&[(10, 0), (20, 1)]);
    let committing: HashSet<u64> = [10, 20, 99].into_iter().collect();
    let out = committing_local_cids_sorted(&committing, &mapping);
    assert_eq!(out, vec![0, 1]);
}

/// Two `HashSet`s with the same contents but different internal
/// iteration order must produce equal vectors, so the resulting
/// `committing_local_cids` field of `DecoderCacheKey` is a canonical
/// form (equal sets ⇒ equal keys).
#[test]
fn committing_local_cids_sorted_is_canonical_across_set_orders() {
    let mapping = mapping_with_local_cids(&[(10, 5), (20, 1), (30, 3), (40, 7)]);
    let s1: HashSet<u64> = [10, 20, 30, 40].into_iter().collect();
    let s2: HashSet<u64> = [40, 30, 20, 10].into_iter().collect();
    let v1 = committing_local_cids_sorted(&s1, &mapping);
    let v2 = committing_local_cids_sorted(&s2, &mapping);
    assert_eq!(v1, v2);
    assert_eq!(v1, vec![1, 3, 5, 7]);
}

/// Different commit-region subsets must produce different sorted
/// vectors, so the resulting `DecoderCacheKey`s differ — the
/// behavioural promise of the window cache-key fix.
#[test]
fn committing_local_cids_sorted_distinguishes_different_subsets() {
    let mapping = mapping_with_local_cids(&[(10, 0), (20, 1), (30, 2)]);
    let s_all: HashSet<u64> = [10, 20, 30].into_iter().collect();
    let s_partial: HashSet<u64> = [10, 20].into_iter().collect();
    let v_all = committing_local_cids_sorted(&s_all, &mapping);
    let v_partial = committing_local_cids_sorted(&s_partial, &mapping);
    assert_ne!(v_all, v_partial);
}

#[test]
fn logical_flip_signature_distinguishes_boundary_components() {
    let mut mapping = RelativeMapping::default();
    mapping.local_gid_of.insert(10, 0);
    let first = logical_flip_signature(&[CorrectionBasis::Residual { gid: 10, index: 1 }], &mapping);
    let second = logical_flip_signature(&[CorrectionBasis::Residual { gid: 10, index: 3 }], &mapping);
    assert_ne!(first, second);
}

/// Sanity: feeding both helpers into a `DecoderCacheKey` with the
/// same `RelativeProgram` but different commit regions yields
/// inequal keys — the cross-module wiring works as advertised.
#[test]
fn cache_key_built_from_helpers_distinguishes_commit_regions() {
    let r = RelativeProgram {
        local_gadgets: vec![],
        count_checks: 0,
    };
    let mapping = mapping_with_local_cids(&[(10, 0), (20, 1), (30, 2)]);
    let fps: Vec<ErrorModelFingerprint> = vec![];

    let k_all = DecoderCacheKey {
        relative_program: r.clone(),
        error_model_fingerprints: fps.clone(),
        committing_local_cids: committing_local_cids_sorted(&[10, 20, 30].into_iter().collect(), &mapping),
        logical_flip_signature: vec![],
    };
    let k_partial = DecoderCacheKey {
        relative_program: r,
        error_model_fingerprints: fps,
        committing_local_cids: committing_local_cids_sorted(&[10, 20].into_iter().collect(), &mapping),
        logical_flip_signature: vec![],
    };
    assert_ne!(k_all, k_partial);
}
