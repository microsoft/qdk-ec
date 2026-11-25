//! Window exploration and missing-state regressions.
use super::*;

#[tokio::test]
async fn failed_reset_does_not_publish_an_uncancelled_epoch() {
    let directory = tempfile::tempdir().unwrap();
    let coordinator = WindowCoordinator::new(
        serde_json::json!({"trace_filepath": directory.path()}),
        DynDecoder::Mock(Arc::new(MockDecoder::new())),
    );
    let error = coordinator
        .reset(Request::new(coordinator::ResetRequest::default()))
        .await
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::Internal);
    assert!(coordinator.cancellation.read().await.is_cancelled());
    assert!(coordinator.wait_token(99).await.is_cancelled());
}

#[tokio::test]
async fn abandoned_raw_publication_cancels_the_whole_shot_before_retry() {
    use futures_util::FutureExt as _;

    let coordinator = Arc::new(WindowCoordinator::new(
        serde_json::json!({"buffer_radius": 0}),
        DynDecoder::Mock(Arc::new(MockDecoder::new())),
    ));
    let empty_matrix = crate::util::BitMatrix {
        cols: 1,
        ..Default::default()
    };
    coordinator
        .load_library(Request::new(bin::Library {
            gadget_types: vec![bin::GadgetType {
                gtype: 1,
                measurements: vec![bin::gadget_type::Measurement::default()],
                correction_propagation: Some(empty_matrix.clone()),
                readout_propagation: Some(empty_matrix.clone()),
                physical_correction: Some(empty_matrix),
                ..Default::default()
            }],
            ..Default::default()
        }))
        .await
        .unwrap();
    coordinator
        .execute(Request::new(bin::Instruction {
            create: Some(bin::instruction::Create::Gadget(bin::Gadget {
                gid: 1,
                gtype: 1,
                ..Default::default()
            })),
        }))
        .await
        .unwrap();
    let (started, ready) = tokio::sync::oneshot::channel();
    let (release, blocked) = std::sync::mpsc::sync_channel(0);
    let frame_task = tokio::spawn({
        let coordinator = Arc::clone(&coordinator);
        async move {
            coordinator
                .frame_executor
                .run(&CancellationToken::new(), move |_| {
                    started.send(()).unwrap();
                    blocked.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
                })
                .await
        }
    });
    ready.await.unwrap();
    let mut publication = Box::pin(Coordinator::decode(
        coordinator.as_ref(),
        Request::new(coordinator::Outcomes {
            gid: 1,
            outcomes: Some(BitVector { size: 1, data: vec![0] }),
            ..Default::default()
        }),
    ));
    assert!(publication.as_mut().now_or_never().is_none());
    drop(publication);
    let cancelled = coordinator.cancellation.read().await.is_cancelled();
    let other_lane_cancelled = coordinator.wait_token(999).await.is_cancelled();
    release.send(()).unwrap();
    frame_task.await.unwrap().unwrap();
    coordinator
        .reset(Request::new(coordinator::ResetRequest::default()))
        .await
        .unwrap();
    assert!(
        cancelled && other_lane_cancelled,
        "unpublished accepted work must cancel the shot root"
    );
    assert!(!coordinator.cancellation.read().await.is_cancelled());
}

#[tokio::test]
async fn cancellation_after_publication_aborts_waiting_readouts() {
    let coordinator = Arc::new(WindowCoordinator::new(
        serde_json::json!({"buffer_radius": 1}),
        DynDecoder::Mock(Arc::new(MockDecoder::new())),
    ));
    let propagation = crate::util::BitMatrix {
        cols: 1,
        ..Default::default()
    };
    coordinator
        .load_library(Request::new(bin::Library {
            gadget_types: vec![bin::GadgetType {
                gtype: 1,
                correction_propagation: Some(propagation.clone()),
                readout_propagation: Some(propagation),
                physical_correction: Some(crate::util::BitMatrix::default()),
                ..Default::default()
            }],
            ..Default::default()
        }))
        .await
        .unwrap();
    coordinator
        .execute(Request::new(bin::Instruction {
            create: Some(bin::instruction::Create::Gadget(bin::Gadget {
                gid: 1,
                gtype: 1,
                ..Default::default()
            })),
        }))
        .await
        .unwrap();
    let mut published = coordinator.gadgets.read().await[&1].outcomes.subscribe();
    let request = tokio::spawn({
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
    tokio::time::timeout(std::time::Duration::from_secs(5), published.wait_for(Option::is_some))
        .await
        .unwrap()
        .unwrap();
    request.abort();
    assert!(request.await.unwrap_err().is_cancelled());
    assert!(coordinator.cancellation.read().await.is_cancelled());
    coordinator
        .reset(Request::new(coordinator::ResetRequest::default()))
        .await
        .unwrap();
}

#[tokio::test]
async fn radius_zero_does_not_include_adjacent_free_hops() {
    let coordinator = WindowCoordinator::new(
        serde_json::json!({"buffer_radius": 0, "lookahead_radius": 0}),
        DynDecoder::Mock(Arc::new(crate::decoder::MockDecoder::new())),
    );
    for gid in [1, 2] {
        coordinator.gadgets.write().await.insert(
            gid,
            Gadget {
                instance: bin::Gadget {
                    gid,
                    connectors: if gid == 2 {
                        vec![bin::gadget::Connector { gid: 1, port: 0 }]
                    } else {
                        vec![]
                    },
                    ..Default::default()
                },
                outcomes: watch::channel(None).0,
                outcome_publication: Mutex::default(),
                outcome_metadata: std::sync::OnceLock::new(),
                raw_frame: None,
                binding_cid: None,
                outputs: if gid == 1 {
                    vec![watch::channel(Some(bin::gadget::Connector { gid: 2, port: 0 })).0]
                } else {
                    vec![]
                },
                pauli_frame: watch::channel(None).0,
                is_free_hop: true,
                state: watch::channel(GadgetState {
                    committed: false,
                    reserved_by: None,
                })
                .0,
                correction_statistics: Default::default(),
            },
        );
    }
    for gid in [1, 2] {
        let mut pins = StatePins::new(&coordinator.pinned_gadgets);
        let explored = coordinator.explore_mandatory_zone(gid, &mut pins).await.unwrap();
        assert_eq!(explored.gadgets, HashSet::from([gid]));
        assert!(explored.frontier.is_empty());
    }
}

#[tokio::test]
async fn missing_window_state_is_internal_not_cancelled() {
    for (entity, id) in [
        ("gadget", 11),
        ("check model", 22),
        ("check model type", 33),
        ("expanded remote gadgets for check model", 22),
        ("error model", 44),
        ("referring error model", 55),
        ("cancellation", 0),
    ] {
        let coordinator = WindowCoordinator::new(
            serde_json::json!({}),
            DynDecoder::Mock(Arc::new(crate::decoder::MockDecoder::new())),
        );
        coordinator.gadgets.write().await.insert(
            11,
            Gadget {
                instance: bin::Gadget {
                    gid: 11,
                    ..Default::default()
                },
                outcomes: watch::channel(Some(BitVector::default())).0,
                outcome_publication: Mutex::default(),
                outcome_metadata: std::sync::OnceLock::new(),
                raw_frame: None,
                binding_cid: Some(22),
                outputs: vec![],
                pauli_frame: watch::channel(None).0,
                is_free_hop: false,
                state: watch::channel(GadgetState {
                    committed: false,
                    reserved_by: Some(11),
                })
                .0,
                correction_statistics: Default::default(),
            },
        );
        coordinator.check_models.write().await.insert(
            22,
            CheckModel {
                instance: bin::CheckModel {
                    cid: 22,
                    ctype: 33,
                    gid: 11,
                    ..Default::default()
                },
                attaching_eid_vec: vec![44],
                error_model_ready: watch::channel(Some(())).0,
                modified_remote_gadgets: Arc::new(vec![]),
                expanded_remote_gadgets: Some(vec![]),
                syndrome: watch::channel(Some(BitVector::default())).0,
                referring_eids: vec![],
                syndrome_count: std::sync::atomic::AtomicU64::new(0),
            },
        );
        coordinator
            .check_model_types
            .write()
            .await
            .insert(33, Arc::new(bin::CheckModelType::default()));
        match entity {
            "gadget" => coordinator.gadgets.write().await.clear(),
            "check model" => coordinator.check_models.write().await.clear(),
            "check model type" => coordinator.check_model_types.write().await.clear(),
            "expanded remote gadgets for check model" => {
                coordinator
                    .check_models
                    .write()
                    .await
                    .get_mut(&22)
                    .unwrap()
                    .expanded_remote_gadgets = None;
            }
            "referring error model" => {
                let mut models = coordinator.check_models.write().await;
                let model = models.get_mut(&22).unwrap();
                model.attaching_eid_vec.clear();
                model.referring_eids.push(55);
            }
            "cancellation" => coordinator.cancel_pending().await,
            _ => {}
        }
        let error = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            coordinator.decode_and_commit(11, &HashSet::from([11]), &HashSet::from([22]), &HashSet::from([11])),
        )
        .await
        .expect("ready window must not block")
        .unwrap_err();
        if entity == "cancellation" {
            assert_eq!(error.code(), tonic::Code::Cancelled);
        } else {
            assert_eq!(error.code(), tonic::Code::Internal, "{entity}: {error}");
            let expected_entity = if entity == "referring error model" {
                "error model"
            } else {
                entity
            };
            assert_eq!(
                error.message(),
                format!("window centered at gid=11: missing {expected_entity} for id={id}")
            );
            assert!(!coordinator.cancellation.read().await.is_cancelled());
        }
    }
}
