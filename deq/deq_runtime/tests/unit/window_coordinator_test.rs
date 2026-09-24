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
use crate::decoder::MockDecoder;

fn syndrome_free_hypergraph() -> DecodingHypergraph {
    DecodingHypergraph {
        vertex_num: 0,
        hyperedges: vec![Hyperedge {
            vertices: vec![],
            probability: 0.1,
        }],
    }
}

#[tokio::test]
async fn impossible_alternative_skips_backend_with_deterministic_priors() {
    for persistent in [false, true] {
        for use_loaded_reweights in [false, true] {
            for reweighted in [false, true] {
                for probability in [0.0, 1.0] {
                    let mock = Arc::new(MockDecoder::new());
                    mock.state.write().await.decode_error = Some(Status::internal("infeasible: search exhausted"));
                    let mut hypergraph = syndrome_free_hypergraph();
                    let reweights = if reweighted {
                        vec![EdgeReweight { edge: 0, probability }]
                    } else {
                        hypergraph.hyperedges[0].probability = probability;
                        vec![]
                    };
                    let baseline = if probability == 1.0 {
                        ParityFactor { subgraph: vec![0] }
                    } else {
                        ParityFactor::default()
                    };
                    let graph = Arc::new(ForcedGapGraph::new(
                        Arc::new(hypergraph),
                        Arc::new(vec![vec![0]]),
                        1,
                        persistent,
                    ));
                    let problem = graph.problem(
                        DynDecoder::Mock(Arc::clone(&mock)),
                        BitVector::default(),
                        baseline,
                        reweights,
                        use_loaded_reweights,
                    );
                    assert_eq!(problem.probability(0).await.unwrap(), 0.0);
                    let state = mock.state.read().await;
                    assert!(state.decode_calls.is_empty());
                    assert!(state.decode_loaded_calls.is_empty());
                    assert!(state.loaded_hypergraphs.is_empty());
                }
            }
        }
    }
}

#[tokio::test]
async fn deterministic_priors_preserve_reachable_backend_errors() {
    for persistent in [false, true] {
        for probability in [0.0, 1.0] {
            let mock = Arc::new(MockDecoder::new());
            mock.state.write().await.decode_error = Some(Status::resource_exhausted("search budget exhausted"));
            let mut hypergraph = syndrome_free_hypergraph();
            hypergraph.hyperedges[0].probability = probability;
            hypergraph.hyperedges.push(Hyperedge {
                vertices: vec![],
                probability: 0.1,
            });
            let baseline = if probability == 1.0 {
                ParityFactor { subgraph: vec![0] }
            } else {
                ParityFactor::default()
            };
            let graph = Arc::new(ForcedGapGraph::new(
                Arc::new(hypergraph),
                Arc::new(vec![vec![0], vec![0]]),
                1,
                persistent,
            ));
            let problem = graph.problem(DynDecoder::Mock(mock), BitVector::default(), baseline, vec![], true);
            let error = problem.probability(0).await.unwrap_err();
            assert_eq!(error.code(), tonic::Code::ResourceExhausted);
            assert!(error.message().contains("reachable=true"));
            assert!(error.message().contains("search budget exhausted"));
        }
    }
}

#[tokio::test]
async fn zero_probability_baseline_is_not_reported_as_zero_risk() {
    for probability in [0.0, 1.0] {
        for reweighted in [false, true] {
            for flips in [vec![], vec![0]] {
                let mock = Arc::new(MockDecoder::new());
                let mut hypergraph = syndrome_free_hypergraph();
                let reweights = if reweighted {
                    vec![EdgeReweight { edge: 0, probability }]
                } else {
                    hypergraph.hyperedges[0].probability = probability;
                    vec![]
                };
                let baseline = if probability == 0.0 {
                    ParityFactor { subgraph: vec![0] }
                } else {
                    ParityFactor::default()
                };
                let graph = Arc::new(ForcedGapGraph::new(Arc::new(hypergraph), Arc::new(vec![flips]), 1, false));
                let problem = graph.problem(
                    DynDecoder::Mock(Arc::clone(&mock)),
                    BitVector::default(),
                    baseline,
                    reweights,
                    true,
                );
                let error = problem.probability(0).await.unwrap_err();
                assert!(error.message().contains("baseline has zero probability"));
                assert!(mock.state.read().await.decode_calls.is_empty());
            }
        }
    }
}

#[tokio::test]
async fn invalid_baseline_is_not_reported_as_zero_risk() {
    let hypergraph = DecodingHypergraph {
        vertex_num: 1,
        hyperedges: vec![Hyperedge {
            vertices: vec![0],
            probability: 0.0,
        }],
    };
    let mock = Arc::new(MockDecoder::new());
    let graph = Arc::new(ForcedGapGraph::new(Arc::new(hypergraph), Arc::new(vec![vec![0]]), 1, false));
    let problem = graph.problem(
        DynDecoder::Mock(Arc::clone(&mock)),
        crate::misc::bit_vector::from_sparse_indices(1, &[0]),
        ParityFactor::default(),
        vec![],
        true,
    );
    let error = problem.probability(0).await.unwrap_err();

    assert!(error.message().contains("baseline does not satisfy the syndrome"));
    assert!(mock.state.read().await.decode_calls.is_empty());
}

#[cfg(feature = "tesseract")]
#[tokio::test]
async fn tesseract_deterministic_alternatives_follow_shot_reweights() {
    for persistent in [false, true] {
        for use_loaded_reweights in [false, true] {
            for base_probability in [0.0, 0.1, 1.0] {
                let mut hypergraph = syndrome_free_hypergraph();
                hypergraph.hyperedges[0].probability = base_probability;
                let hypergraph = Arc::new(hypergraph);
                let graph = Arc::new(ForcedGapGraph::new(
                    Arc::clone(&hypergraph),
                    Arc::new(vec![vec![0]]),
                    1,
                    persistent,
                ));
                let decoder = crate::decoder::DecoderType::BlackBoxTesseract.create(serde_json::json!({
                    "parallel": 1, "det_beam": 0, "pqlimit": 200000, "det_penalty": 0,
                    "beam_climbing": false,
                }));
                for probability in [base_probability, 0.0, 0.001, 0.0, 1.0, 0.001] {
                    let baseline = if probability == 1.0 {
                        ParityFactor { subgraph: vec![0] }
                    } else {
                        ParityFactor::default()
                    };
                    let reweights = if probability == base_probability {
                        vec![]
                    } else {
                        vec![EdgeReweight { edge: 0, probability }]
                    };
                    let problem = graph.problem(
                        decoder.clone(),
                        BitVector::default(),
                        baseline,
                        reweights,
                        use_loaded_reweights,
                    );
                    let expected = if probability == 1.0 { 0.0 } else { probability };
                    let actual = problem.probability(0).await.unwrap();
                    if expected == 0.0 {
                        assert_eq!(actual, 0.0);
                    } else {
                        assert!((actual - expected).abs() < 1e-12);
                    }
                    assert_eq!(actual, problem.probability(0).await.unwrap());
                }
                assert_eq!(hypergraph.hyperedges.len(), 1);
                assert_eq!(hypergraph.hyperedges[0].probability, base_probability);
            }
        }
    }
}

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
fn causal_scoring_unmerges_reweighted_representatives_without_changing_hard_correction() {
    let graph = DecodingHypergraph {
        vertex_num: 1,
        hyperedges: [0.1, 0.2]
            .into_iter()
            .map(|probability| Hyperedge {
                vertices: vec![0],
                probability,
            })
            .collect(),
    };
    let original = Arc::new(vec![
        ErrorIndex { eid: 0, error_index: 0 },
        ErrorIndex { eid: 1, error_index: 0 },
    ]);
    let (projection, _) = prepare_decoder(graph, Arc::clone(&original), vec![vec![], vec![]], true, |_| 0);
    let (_, projected) = projection.project_reweights(&[(0, 0.3)]);
    let hard = ParityFactor { subgraph: vec![0] };
    assert_eq!(original_scoring_baseline(&original, &projected, &hard).subgraph, vec![0]);
    let (_, projected) = projection.project_reweights(&[]);
    assert_eq!(original_scoring_baseline(&original, &projected, &hard).subgraph, vec![1]);
    assert_eq!(hard.subgraph, vec![0]);
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
async fn history_commitment_wait_is_cancelled_without_reserving_gadgets() {
    for parallelism in [WindowParallelism::Sliding, WindowParallelism::Serial] {
        let mut coordinator = WindowCoordinator::new(
            serde_json::json!({}),
            DynDecoder::Mock(Arc::new(crate::decoder::MockDecoder::new())),
        );
        for gid in [1, 2, 3] {
            coordinator.gadgets.write().await.insert(
                gid,
                Gadget {
                    instance: bin::Gadget {
                        gid,
                        connectors: if gid == 2 && parallelism == WindowParallelism::Sliding {
                            vec![bin::gadget::Connector { gid: 1, port: 0 }]
                        } else {
                            vec![]
                        },
                        ..Default::default()
                    },
                    outcomes: watch::channel(None).0,
                    probability_modifiers: vec![],
                    loss_mask: None,
                    binding_cid: None,
                    outputs: vec![],
                    pauli_frame: watch::channel(None).0,
                    correction_count: 0,
                    correction_weight: 0.0,
                    is_free_hop: false,
                    state: watch::channel(GadgetState::default()).0,
                },
            );
        }
        coordinator.config.window_parallelism = WindowParallelism::FullyParallel;
        assert_eq!(
            serde_json::to_value(coordinator.config.window_parallelism).unwrap(),
            "fully_parallel"
        );
        assert!(serde_json::from_value::<WindowParallelism>(serde_json::json!("all")).is_err());
        tokio::time::timeout(std::time::Duration::from_secs(1), coordinator.wait_for_history_commitment(2))
            .await
            .expect("fully parallel mode must not wait for uncommitted history")
            .unwrap();
        coordinator.config.window_parallelism = parallelism;
        coordinator.wait_for_history_commitment(1).await.unwrap();
        {
            let waiting = coordinator.wait_for_history_commitment(2);
            tokio::pin!(waiting);
            assert!(futures_util::poll!(&mut waiting).is_pending());
            coordinator.gadgets.write().await[&1]
                .state
                .send_modify(|state| state.committed = true);
            tokio::time::timeout(std::time::Duration::from_secs(1), waiting)
                .await
                .expect("uncommitted higher GIDs must not block")
                .unwrap();
        }
        coordinator.gadgets.write().await[&1]
            .state
            .send_modify(|state| state.committed = false);
        coordinator.gadgets.write().await.get_mut(&1).unwrap().is_free_hop = true;
        tokio::time::timeout(std::time::Duration::from_secs(1), coordinator.wait_for_history_commitment(2))
            .await
            .expect("free-hop predecessors must be left for adjacent windows to commit")
            .unwrap();
        coordinator.gadgets.write().await.get_mut(&1).unwrap().is_free_hop = false;
        let (result, ()) = tokio::join!(coordinator.wait_for_history_commitment(2), async {
            tokio::task::yield_now().await;
            coordinator.cancel_pending().await;
        });
        assert_eq!(result.unwrap_err().code(), tonic::Code::Cancelled);
        assert!(
            coordinator
                .gadgets
                .read()
                .await
                .values()
                .all(|gadget| gadget.state.borrow().reserved_by.is_none())
        );
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
async fn causal_history_restores_priors_and_correction_without_reopening_future() {
    let decoder = crate::decoder::DecoderType::BlackBoxTesseract.create(serde_json::json!({ "parallel": 1 }));
    let recorded = |gid, probability, selected| RecordedGapEdge {
        gid,
        checks: vec![(10, 0)],
        probability,
        residual: vec![],
        readouts: vec![],
        selected,
    };
    let mut snapshot = CausalGapSnapshot {
        hypergraph: DecodingHypergraph {
            vertex_num: 1,
            hyperedges: vec![
                Hyperedge {
                    vertices: vec![0],
                    probability: 0.1,
                },
                Hyperedge {
                    vertices: vec![0],
                    probability: 0.49,
                },
            ],
        },
        syndrome: BitVector { size: 1, data: vec![0] },
        baseline: ParityFactor::default(),
        edges: vec![recorded(2, 0.1, false), recorded(3, 0.49, false)],
    };
    snapshot.restore_history([recorded(1, 0.2, true)], &HashMap::from([((10, 0), 0)]));
    assert_eq!(snapshot.hypergraph.hyperedges[2].probability, 0.2);
    assert_eq!(snapshot.baseline.subgraph, vec![2]);
    assert!(is_parity_factor(&snapshot.hypergraph, &snapshot.baseline, &snapshot.syndrome));
    let scorer = CommitRegionDecoder::new(
        &snapshot.hypergraph,
        &[vec![0], vec![], vec![]],
        &[true, false, true],
        1,
        false,
    );
    let probability = scorer
        .problem(
            decoder,
            &snapshot.hypergraph,
            snapshot.syndrome,
            &snapshot.baseline,
            vec![],
            false,
        )
        .unwrap()
        .probability(0)
        .await
        .unwrap();
    assert!((probability - 4.0 / 13.0).abs() < 1e-12);
}

#[test]
fn causal_history_retains_shot_priors_and_clears_on_reset() {
    let mut state = ForcedGapState::new();
    let mut edge = RecordedGapEdge {
        gid: 7,
        checks: vec![(7, 0)],
        probability: 0.17,
        residual: vec![0],
        readouts: vec![],
        selected: true,
    };
    state.history.insert(7, vec![edge.clone()]);
    edge.probability = 0.49;
    assert_eq!(state.history[&7][0].probability, 0.17);
    assert_eq!(edge.probability, 0.49);
    state.reset();
    assert!(state.history.is_empty());
}

#[test]
fn causal_history_projects_old_boundary_checks_without_dropping_the_edge() {
    let mut snapshot = CausalGapSnapshot {
        hypergraph: DecodingHypergraph {
            vertex_num: 1,
            hyperedges: vec![],
        },
        syndrome: BitVector { size: 1, data: vec![0] },
        baseline: ParityFactor::default(),
        edges: vec![],
    };
    snapshot.restore_history(
        [RecordedGapEdge {
            gid: 7,
            checks: vec![(6, 0), (7, 0)],
            probability: 0.17,
            residual: vec![0],
            readouts: vec![],
            selected: true,
        }],
        &HashMap::from([((7, 0), 0)]),
    );
    assert_eq!(snapshot.hypergraph.hyperedges.len(), 1);
    assert_eq!(snapshot.hypergraph.hyperedges[0].vertices, vec![0]);
    assert_eq!(snapshot.hypergraph.hyperedges[0].probability, 0.17);
    assert_eq!(snapshot.baseline.subgraph, vec![0]);
    assert!(is_parity_factor(&snapshot.hypergraph, &snapshot.baseline, &snapshot.syndrome));
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

#[test]
fn gadget_state_release_preserves_commitment_and_checks_owner() {
    assert_eq!(
        GadgetState::default(),
        GadgetState {
            committed: false,
            reserved_by: None
        }
    );
    for committed in [false, true] {
        for reserved_by in [None, Some(7)] {
            let state = GadgetState { committed, reserved_by };
            let released = GadgetState {
                committed,
                reserved_by: None,
            };
            assert_eq!(state.release_buffer(7), reserved_by.map(|_| released));
            assert_eq!(state.release_buffer(8), None);
        }
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
fn remote_check_resolution_includes_only_reserved_endpoints() {
    let mut gadgets = HashMap::from([
        (1, history_gadget(1, GadgetState::default(), Some(2))),
        (2, history_gadget(2, GadgetState::default(), Some(3))),
        (
            3,
            history_gadget(
                3,
                GadgetState {
                    committed: true,
                    reserved_by: None,
                },
                None,
            ),
        ),
    ]);
    for gid in [2, 3] {
        gadgets
            .get_mut(&gid)
            .unwrap()
            .instance
            .connectors
            .push(bin::gadget::Connector { gid: gid - 1, port: 0 });
    }
    assert_eq!(
        WindowCoordinator::terminal_boundary_gids(&gadgets, &HashSet::from([1])),
        HashSet::from([1])
    );
    assert_eq!(
        WindowCoordinator::terminal_boundary_gids(&gadgets, &HashSet::from([1, 2])),
        HashSet::from([2])
    );
    assert_eq!(
        WindowCoordinator::terminal_boundary_gids(&gadgets, &HashSet::from([1, 3])),
        HashSet::from([3])
    );
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
    assert_eq!(
        WindowCoordinator::expand_remote_check_models_in_window(1, &error_model, &gadgets, &HashSet::from([1, 2, 3])),
        vec![Some(2), Some(3)],
    );
}

#[test]
fn terminal_boundaries_match_forward_reachability() {
    let connections = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)];
    for connection_mask in 0_u64..(1 << connections.len()) {
        let mut gadgets: HashMap<_, _> = (1..=4)
            .map(|gid| (gid, history_gadget(gid, GadgetState::default(), None)))
            .collect();
        for (index, &(source, target)) in connections.iter().enumerate() {
            if connection_mask & (1 << index) == 0 {
                continue;
            }
            let output_port = u64::try_from(gadgets[&source].outputs.len()).unwrap();
            let input_port = u64::try_from(gadgets[&target].instance.connectors.len()).unwrap();
            gadgets.get_mut(&source).unwrap().outputs.push(
                watch::channel(Some(bin::gadget::Connector {
                    gid: target,
                    port: input_port,
                }))
                .0,
            );
            gadgets
                .get_mut(&target)
                .unwrap()
                .instance
                .connectors
                .push(bin::gadget::Connector {
                    gid: source,
                    port: output_port,
                });
        }
        for window_mask in 0_u64..16 {
            let window: HashSet<_> = (1..=4).filter(|&gid| window_mask & (1 << (gid - 1)) != 0).collect();
            let expected = window
                .iter()
                .copied()
                .filter(|&gid| {
                    let mut pending = vec![gid];
                    let mut visited: HashSet<u64> = HashSet::from([gid]);
                    while let Some(current) = pending.pop() {
                        for output in &gadgets[&current].outputs {
                            let Some(peer) = *output.borrow() else { continue };
                            if window.contains(&peer.gid) {
                                return false;
                            }
                            if visited.insert(peer.gid) {
                                pending.push(peer.gid);
                            }
                        }
                    }
                    true
                })
                .collect::<HashSet<_>>();
            assert_eq!(
                WindowCoordinator::terminal_boundary_gids(&gadgets, &window),
                expected,
                "connections={connection_mask}, window={window_mask}"
            );
        }
    }
}

#[test]
fn remote_check_resolution_handles_inputs_and_absolute_reroutes() {
    let mut middle = history_gadget(2, GadgetState::default(), Some(3));
    middle.instance.connectors.push(bin::gadget::Connector { gid: 1, port: 0 });
    let mut terminal = history_gadget(
        3,
        GadgetState {
            committed: true,
            reserved_by: None,
        },
        None,
    );
    terminal.instance.connectors.push(bin::gadget::Connector { gid: 2, port: 0 });
    let gadgets = HashMap::from([
        (1, history_gadget(1, GadgetState::default(), Some(2))),
        (2, middle),
        (3, terminal),
    ]);
    let first = RemoteCheckModel {
        port: Some(remote_check_model::Port::Input(0)),
        ..Default::default()
    };
    let second = RemoteCheckModel {
        previous_remote_check_model: Some(0),
        ..first.clone()
    };
    let error_model = make_error_model(
        make_error_model_instance(1, 1, None),
        vec![
            Some(first),
            Some(second),
            None,
            Some(RemoteCheckModel {
                absolute_cid: Some(99),
                ..Default::default()
            }),
            Some(make_remote_check(0)),
            Some(RemoteCheckModel {
                previous_remote_check_model: Some(2),
                ..make_remote_check(0)
            }),
        ],
    );
    assert_eq!(
        WindowCoordinator::expand_remote_check_models_in_window(3, &error_model, &gadgets, &HashSet::from([1, 3])),
        vec![None, Some(1), None, Some(99), None, None],
    );
    assert_eq!(
        WindowCoordinator::expand_remote_check_models_in_window(3, &error_model, &gadgets, &HashSet::from([1, 2, 3])),
        vec![Some(2), Some(1), None, Some(99), None, None],
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
        error_model_ready: watch::channel(Some(())).0,
        modified_remote_gadgets: Arc::new(vec![]),
        expanded_remote_gadgets: Some(vec![]),
        syndrome: watch::channel(None).0,
        syndrome_count: 0,
        referring_eids: vec![],
    }
}

fn empty_readout_gadget_type() -> Arc<bin::GadgetType> {
    Arc::new(bin::GadgetType {
        readouts: vec![bin::gadget_type::Readout::default()],
        correction_propagation: Some(crate::util::BitMatrix {
            rows: 0,
            cols: 1,
            ..Default::default()
        }),
        readout_propagation: Some(crate::util::BitMatrix {
            rows: 1,
            cols: 1,
            ..Default::default()
        }),
        logical_correction: Some(crate::util::BitMatrix {
            rows: 0,
            cols: 1,
            ..Default::default()
        }),
        ..Default::default()
    })
}

async fn add_shared_context_gadgets(coordinator: &WindowCoordinator, gadget_type: &bin::GadgetType, disconnected: bool) {
    for gid in [1, 2, 3] {
        let is_history = gid == 3;
        let mut gadget = history_gadget(
            gid,
            GadgetState {
                committed: is_history,
                reserved_by: None,
            },
            match gid {
                1 => Some(3),
                3 => Some(2),
                _ => None,
            },
        );
        if gid == 2 {
            gadget.outputs.clear();
        }
        if gid != 1 {
            gadget.instance.connectors.push(bin::gadget::Connector {
                gid: if is_history { 1 } else { 3 },
                port: 0,
            });
        }
        if disconnected {
            gadget.outputs.clear();
            gadget.instance.connectors.clear();
        }
        if is_history {
            gadget.outcomes.send_replace(Some(BitVector::default()));
        }
        coordinator.gadgets.write().await.insert(gid, gadget);
        let mut check_model = history_check_model(gid, if is_history { vec![] } else { vec![gid] });
        check_model.instance.ctype = gid;
        check_model.syndrome.send_replace(Some(if is_history {
            BitVector {
                size: 1,
                data: vec![0x80],
            }
        } else {
            BitVector::default()
        }));
        if is_history {
            check_model.referring_eids = vec![1, 2];
        }
        coordinator.check_models.write().await.insert(gid, check_model);
        coordinator.check_model_types.write().await.insert(
            gid,
            Arc::new(bin::CheckModelType {
                ctype: gid,
                checks: if is_history {
                    vec![bin::check_model_type::Check::default()]
                } else {
                    vec![]
                },
                ..Default::default()
            }),
        );
        coordinator
            .pauli_frame_tracker
            .lock()
            .await
            .add_gadget(gid, gadget_type, None, &HashMap::new(), &[]);
        if !is_history {
            let mut instance = make_error_model_instance(gid, 1, None);
            instance.cid = gid;
            let remote = RemoteCheckModel {
                absolute_cid: Some(3),
                ..Default::default()
            };
            coordinator
                .error_models
                .write()
                .await
                .insert(gid, make_error_model(instance, vec![Some(remote)]));
        }
    }
}

async fn shared_context_coordinator(
    persistent_decoder: bool,
    disconnected: bool,
) -> (Arc<WindowCoordinator>, Arc<crate::decoder::MockDecoder>) {
    let mock = Arc::new(crate::decoder::MockDecoder::new());
    mock.set_response(vec![0x80], vec![0]).await;
    let coordinator = Arc::new(WindowCoordinator::new(
        serde_json::json!({"buffer_radius": 1, "lookahead_radius": 0, "persistent_decoder": persistent_decoder, "merge_hyperedges": false, "loss_strategy": "ignore"}),
        DynDecoder::Mock(Arc::clone(&mock)),
    ));
    let gadget_type = empty_readout_gadget_type();
    coordinator.gadget_types.write().await.insert(0, Arc::clone(&gadget_type));
    add_shared_context_gadgets(&coordinator, &gadget_type, disconnected).await;
    let error = Error {
        probability: 0.1,
        checks: vec![bin::error_model_type::RemoteCheck {
            remote_check_model: Some(0),
            check_index: 0,
        }],
        readout_flips: vec![0],
        ..Default::default()
    };
    coordinator
        .error_model_types
        .write()
        .await
        .insert(1, Arc::new(make_emt(1, vec![error])));
    (coordinator, mock)
}

#[tokio::test]
async fn committed_buffer_syndrome_is_not_consumed_by_two_windows() {
    use crate::coordinator::coordinator_server::Coordinator;

    for (persistent_decoder, disconnected) in [(false, false), (true, false), (false, true), (true, true)] {
        let (coordinator, mock) = shared_context_coordinator(persistent_decoder, disconnected).await;
        let decode = |gid| {
            let coordinator = Arc::clone(&coordinator);
            tokio::spawn(async move {
                Coordinator::decode(
                    coordinator.as_ref(),
                    Request::new(coordinator::Outcomes {
                        gid,
                        outcomes: Some(BitVector::default()),
                        ..Default::default()
                    }),
                )
                .await
                .unwrap()
                .into_inner()
            })
        };
        let blocker = mock.block_next_decode();
        let first = decode(1);
        tokio::time::timeout(std::time::Duration::from_secs(2), blocker.wait_until_started())
            .await
            .unwrap();
        let mut second = decode(2);
        let early_second = tokio::time::timeout(std::time::Duration::from_millis(100), &mut second).await;
        let waited_for_context = early_second.is_err();
        let (waiting_state, context_state) = {
            let gadgets = coordinator.gadgets.read().await;
            (gadgets[&2].state.borrow().clone(), gadgets[&3].state.borrow().clone())
        };
        blocker.release();
        let first = tokio::time::timeout(std::time::Duration::from_secs(2), first)
            .await
            .unwrap()
            .unwrap();
        let second = match early_second {
            Ok(result) => result.unwrap(),
            Err(_) => tokio::time::timeout(std::time::Duration::from_secs(2), second)
                .await
                .unwrap()
                .unwrap(),
        };
        assert!(waited_for_context);
        assert_eq!(waiting_state, GadgetState::default());
        assert_eq!(
            context_state,
            GadgetState {
                committed: true,
                reserved_by: Some(1)
            }
        );
        assert_eq!(
            first.correction_count + second.correction_count,
            1,
            "a committed buffer's syndrome must be consumed only once"
        );
        assert!(!get_bit(
            coordinator.check_models.read().await[&3].syndrome.borrow().as_ref().unwrap(),
            0
        ));
        assert_eq!(
            *coordinator.gadgets.read().await[&3].state.borrow(),
            GadgetState {
                committed: true,
                reserved_by: None
            }
        );
    }
}

async fn bounded_window_coordinator(
    persistent_decoder: bool,
    forced_gap: bool,
    remote_state: GadgetState,
) -> (Arc<WindowCoordinator>, Arc<crate::decoder::MockDecoder>) {
    let mock = Arc::new(crate::decoder::MockDecoder::new());
    mock.set_response(vec![0x40], vec![0, 1]).await;
    let coordinator = Arc::new(WindowCoordinator::new(
        serde_json::json!({"buffer_radius": 1, "lookahead_radius": 0, "persistent_decoder": persistent_decoder, "forced_gap": forced_gap, "merge_hyperedges": false, "loss_strategy": "ignore"}),
        DynDecoder::Mock(Arc::clone(&mock)),
    ));
    let gadget_type = empty_readout_gadget_type();
    coordinator.gadget_types.write().await.insert(0, Arc::clone(&gadget_type));
    for gid in [1, 3] {
        let is_remote = gid == 3;
        let state = if is_remote {
            remote_state.clone()
        } else {
            GadgetState {
                committed: false,
                reserved_by: Some(gid),
            }
        };
        let mut gadget = history_gadget(gid, state, None);
        gadget.outputs.clear();
        gadget.outcomes.send_replace(Some(BitVector::default()));
        coordinator.gadgets.write().await.insert(gid, gadget);
        let mut check_model = history_check_model(gid, if is_remote { vec![] } else { vec![gid] });
        check_model.instance.ctype = gid;
        check_model.syndrome.send_replace(Some(BitVector {
            size: 1,
            data: vec![if is_remote { 0x80 } else { 0 }],
        }));
        if is_remote {
            check_model.referring_eids = vec![1];
        }
        coordinator.check_models.write().await.insert(gid, check_model);
        coordinator.check_model_types.write().await.insert(
            gid,
            Arc::new(bin::CheckModelType {
                ctype: gid,
                checks: vec![bin::check_model_type::Check::default()],
                ..Default::default()
            }),
        );
        let mut tracker = coordinator.pauli_frame_tracker.lock().await;
        tracker.add_gadget(gid, &gadget_type, None, &HashMap::new(), &[]);
        tracker.load_raw(gid, &[false], &BitVector::default());
        if let Some(state) = &coordinator.forced_gap_state {
            state.write().await.symbolic.add_gadget(gid, &tracker.gadgets[&gid]);
        }
    }
    let remote = RemoteCheckModel {
        absolute_cid: Some(3),
        ..Default::default()
    };
    coordinator
        .error_models
        .write()
        .await
        .insert(1, make_error_model(make_error_model_instance(1, 1, None), vec![Some(remote)]));
    let mut flipping_error = make_error(0.1);
    flipping_error.readout_flips = vec![0];
    let mut distant_error = make_error(0.4);
    distant_error.checks.push(bin::error_model_type::RemoteCheck {
        remote_check_model: Some(0),
        check_index: 0,
    });
    coordinator.error_model_types.write().await.insert(
        1,
        Arc::new(make_emt(1, vec![flipping_error, make_error(0.02), distant_error])),
    );
    (coordinator, mock)
}

#[tokio::test]
async fn missing_full_model_waits_without_a_terminal_fallback() {
    use crate::coordinator::coordinator_server::Coordinator;

    let (coordinator, mock) = bounded_window_coordinator(false, false, GadgetState::default()).await;
    coordinator.error_models.write().await.remove(&1);
    Arc::make_mut(coordinator.error_model_types.write().await.get_mut(&1).unwrap()).remote_check_models =
        vec![RemoteCheckModel {
            absolute_cid: Some(3),
            ..Default::default()
        }];
    {
        let mut checks = coordinator.check_models.write().await;
        let check = checks.get_mut(&1).unwrap();
        check.attaching_eid_vec.clear();
        check.error_model_ready.send_replace(None);
    }
    let mut decode = tokio::spawn({
        let coordinator = Arc::clone(&coordinator);
        async move {
            let region = HashSet::from([1]);
            coordinator.decode_and_commit(1, &region, &region, &region).await
        }
    });
    let pending = tokio::time::timeout(std::time::Duration::from_millis(50), &mut decode).await;
    let waited = pending.is_err();
    assert_eq!(mock.state.read().await.decode_calls.is_empty(), waited);
    Coordinator::execute(
        coordinator.as_ref(),
        Request::new(bin::Instruction {
            create: Some(bin::instruction::Create::ErrorModel(make_error_model_instance(1, 1, None))),
        }),
    )
    .await
    .unwrap();
    if waited {
        tokio::time::timeout(std::time::Duration::from_secs(2), decode)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }
    assert!(waited, "absence of a terminal fallback must not imply a ready error model");
}

#[tokio::test]
async fn dropping_external_commit_edges_preserves_future_syndromes_and_old_corrections() {
    for persistent in [false, true] {
        let (coordinator, mock) = bounded_window_coordinator(persistent, false, GadgetState::default()).await;
        coordinator.check_models.read().await[&1]
            .syndrome
            .send_replace(Some(BitVector {
                size: 1,
                data: vec![0x80],
            }));
        mock.set_response(vec![0x80], vec![0]).await;
        let first = HashSet::from([1]);
        coordinator.decode_and_commit(1, &first, &first, &first).await.unwrap();
        assert!(get_bit(
            coordinator.check_models.read().await[&3].syndrome.borrow().as_ref().unwrap(),
            0
        ));
        let first_frame = coordinator.gadgets.read().await[&1].pauli_frame.borrow().clone();
        coordinator
            .error_model_types
            .write()
            .await
            .insert(3, Arc::new(make_emt(3, vec![make_error(0.02)])));
        let mut model = make_error_model_instance(3, 3, None);
        model.cid = 3;
        coordinator
            .error_models
            .write()
            .await
            .insert(3, make_error_model(model, vec![]));
        coordinator.check_models.write().await.get_mut(&3).unwrap().attaching_eid_vec = vec![3];
        coordinator.gadgets.read().await[&3]
            .state
            .send_modify(|state| state.reserved_by = Some(3));
        mock.set_response(vec![0x80], vec![0]).await;
        let second = HashSet::from([3]);
        coordinator.decode_and_commit(3, &second, &second, &second).await.unwrap();
        assert_eq!(
            coordinator.gadgets.read().await[&3].correction_count,
            1,
            "the later window must decode its own unchanged syndrome"
        );
        assert_eq!(coordinator.gadgets.read().await[&1].pauli_frame.borrow().clone(), first_frame);
        assert_eq!(coordinator.gadgets.read().await[&1].correction_count, 1);
    }
}

#[tokio::test]
async fn ready_remote_checks_preserve_commit_error_hypotheses() {
    use crate::coordinator::coordinator_server::Coordinator;

    for persistent_decoder in [false, true] {
        for (remote_state, remote_ready) in [
            (GadgetState::default(), true),
            (
                GadgetState {
                    committed: true,
                    reserved_by: None,
                },
                true,
            ),
            (GadgetState::default(), false),
        ] {
            let (coordinator, mock) = bounded_window_coordinator(persistent_decoder, false, remote_state.clone()).await;
            let gadget_type = Arc::clone(&coordinator.gadget_types.read().await[&0]);
            {
                let mut tracker = coordinator.pauli_frame_tracker.lock().await;
                tracker.gadgets.remove(&1);
                tracker.add_gadget(1, &gadget_type, None, &HashMap::new(), &[]);
            }
            coordinator.gadgets.read().await[&1]
                .state
                .send_replace(GadgetState::default());
            coordinator.check_models.read().await[&1]
                .syndrome
                .send_replace(Some(BitVector {
                    size: 1,
                    data: vec![0x80],
                }));
            if !remote_ready {
                coordinator.check_models.read().await[&3].syndrome.send_replace(None);
                coordinator.gadgets.read().await[&3].outcomes.send_replace(None);
            }
            mock.set_response(vec![0x80], vec![0]).await;
            mock.set_response(vec![0xc0], vec![2]).await;
            let readouts = tokio::time::timeout(
                std::time::Duration::from_secs(2),
                Coordinator::decode(
                    coordinator.as_ref(),
                    Request::new(coordinator::Outcomes {
                        gid: 1,
                        outcomes: Some(BitVector::default()),
                        ..Default::default()
                    }),
                ),
            )
            .await
            .unwrap()
            .unwrap()
            .into_inner();
            assert_eq!(readouts.correction_count, 1);
            assert_eq!(
                readouts.readouts,
                Some(BitVector {
                    size: 1,
                    data: vec![if remote_ready { 0 } else { 0x80 }]
                }),
                "a ready remote check must not cause the physical commit-error hypothesis to be discarded"
            );
            let gadgets = coordinator.gadgets.read().await;
            assert_eq!(*gadgets[&3].state.borrow(), remote_state);
            assert_eq!(gadgets[&3].correction_count, 0);
            let checks = coordinator.check_models.read().await;
            assert!(!get_bit(checks[&1].syndrome.borrow().as_ref().unwrap(), 0));
            assert_eq!(
                *checks[&3].syndrome.borrow(),
                remote_ready.then_some(BitVector { size: 1, data: vec![0] })
            );
        }
    }
}

#[tokio::test]
async fn decoding_and_scoring_do_not_expand_the_selected_window() {
    for persistent_decoder in [false, true] {
        for forced_gap in [false, true] {
            for (remote_state, remote_ready) in [
                (GadgetState::default(), false),
                (GadgetState::default(), true),
                (
                    GadgetState {
                        committed: true,
                        reserved_by: None,
                    },
                    true,
                ),
                (
                    GadgetState {
                        committed: false,
                        reserved_by: Some(3),
                    },
                    true,
                ),
            ] {
                let (coordinator, mock) =
                    bounded_window_coordinator(persistent_decoder, forced_gap, remote_state.clone()).await;
                let remote_syndrome = remote_ready.then_some(BitVector {
                    size: 1,
                    data: vec![0x80],
                });
                coordinator.check_models.read().await[&3]
                    .syndrome
                    .send_replace(remote_syndrome.clone());
                let region = HashSet::from([1]);
                let readouts = tokio::time::timeout(std::time::Duration::from_secs(2), async {
                    coordinator.decode_and_commit(1, &region, &region, &region).await.unwrap();
                    coordinator.wait_for_pauli_frame(1).await.unwrap().into_inner()
                })
                .await
                .expect("out-of-window checks must not add a dependency");
                assert_eq!(readouts.correction_count, 0);
                assert_eq!(readouts.readouts, Some(BitVector { size: 1, data: vec![0] }));
                if forced_gap {
                    let odds = 0.1 * 0.02 / (0.9 * 0.98);
                    assert!((readouts.probabilities[0] - odds / (1.0 + odds)).abs() < 1e-12);
                } else {
                    assert!(readouts.probabilities.is_empty());
                }
                let state = mock.state.read().await;
                let graphs: Vec<_> = state
                    .decode_calls
                    .iter()
                    .map(|call| &call.hypergraph)
                    .chain(state.loaded_hypergraphs.values())
                    .collect();
                assert_eq!(graphs.len(), if forced_gap { 2 } else { 1 });
                assert!(graphs.iter().all(|graph| graph.hyperedges.len() == 2));
                let (hard_graph, hard_syndrome) = if persistent_decoder {
                    let call = &state.decode_loaded_calls[0];
                    assert!(call.reweights.is_empty());
                    (&state.loaded_hypergraphs[&call.hid], &call.syndrome)
                } else {
                    let call = &state.decode_calls[0];
                    (&call.hypergraph, &call.syndrome)
                };
                assert_eq!(
                    *hard_graph,
                    DecodingHypergraph {
                        vertex_num: 1,
                        hyperedges: [0.1, 0.02]
                            .into_iter()
                            .map(|probability| Hyperedge {
                                vertices: vec![0],
                                probability,
                            })
                            .collect(),
                    },
                );
                assert_eq!(*hard_syndrome, BitVector { size: 1, data: vec![0] });
                let checks = coordinator.check_models.read().await;
                assert_eq!(*checks[&3].syndrome.borrow(), remote_syndrome);
                let gadgets = coordinator.gadgets.read().await;
                assert_eq!(*gadgets[&3].state.borrow(), remote_state);
                assert_eq!(gadgets[&3].correction_count, 0);
            }
        }
    }
}

#[cfg(feature = "tesseract")]
#[tokio::test]
async fn forced_gap_excludes_external_commit_edges() {
    for persistent in [false, true] {
        let (mut coordinator, _) = bounded_window_coordinator(persistent, true, GadgetState::default()).await;
        Arc::get_mut(&mut coordinator).unwrap().decoder =
            crate::decoder::DecoderType::BlackBoxTesseract.create(serde_json::json!({ "parallel": 1 }));
        let region = HashSet::from([1]);
        coordinator.decode_and_commit(1, &region, &region, &region).await.unwrap();
        let readouts = coordinator.wait_for_pauli_frame(1).await.unwrap().into_inner();
        assert_eq!(readouts.correction_count, 0);
        assert!((readouts.probabilities[0] - 1.0 / 442.0).abs() < 1e-12);
    }
}

#[tokio::test]
async fn reset_cancels_context_wait_without_a_partial_reservation() {
    use crate::coordinator::coordinator_server::Coordinator;

    for committed in [false, true] {
        let remote_state = GadgetState {
            committed,
            reserved_by: Some(3),
        };
        let (coordinator, mock) = bounded_window_coordinator(false, false, remote_state.clone()).await;
        let gadget_type = Arc::clone(&coordinator.gadget_types.read().await[&0]);
        {
            let mut tracker = coordinator.pauli_frame_tracker.lock().await;
            tracker.gadgets.remove(&1);
            tracker.add_gadget(1, &gadget_type, None, &HashMap::new(), &[]);
        }
        coordinator.gadgets.read().await[&1]
            .state
            .send_replace(GadgetState::default());
        let mut decoding = tokio::spawn({
            let coordinator = Arc::clone(&coordinator);
            async move {
                Coordinator::decode(
                    coordinator.as_ref(),
                    Request::new(coordinator::Outcomes {
                        gid: 1,
                        outcomes: Some(BitVector::default()),
                        ..Default::default()
                    }),
                )
                .await
            }
        });
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(100), &mut decoding)
                .await
                .is_err()
        );
        {
            let gadgets = coordinator.gadgets.read().await;
            assert_eq!(*gadgets[&1].state.borrow(), GadgetState::default());
            assert_eq!(*gadgets[&3].state.borrow(), remote_state);
        }
        assert!(mock.state.read().await.decode_calls.is_empty());
        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            Coordinator::reset(coordinator.as_ref(), Request::new(coordinator::ResetRequest::default())),
        )
        .await
        .unwrap()
        .unwrap();
        let error = decoding.await.unwrap().unwrap_err();
        assert_eq!(error.code(), tonic::Code::Cancelled);
        assert!(coordinator.gadgets.read().await.is_empty());
        assert!(coordinator.check_models.read().await.is_empty());
    }
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
