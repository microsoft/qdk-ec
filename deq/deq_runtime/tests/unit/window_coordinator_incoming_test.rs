//! Window exploration and missing-state regressions.
use super::*;

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
                probability_modifiers: vec![],
                loss_mask: None,
                decoder_seed: None,
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
                correction_count: 0,
                correction_weight: 0.0,
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
                probability_modifiers: vec![],
                loss_mask: None,
                decoder_seed: None,
                binding_cid: Some(22),
                outputs: vec![],
                pauli_frame: watch::channel(None).0,
                is_free_hop: false,
                state: watch::channel(GadgetState {
                    committed: false,
                    reserved_by: Some(11),
                })
                .0,
                correction_count: 0,
                correction_weight: 0.0,
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
                syndrome_count: 0,
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
