use super::*;
use futures_util::FutureExt;

fn coordinator() -> WindowCoordinator {
    WindowCoordinator::new(serde_json::json!({}), DynDecoder::Mock(Arc::new(MockDecoder::new())))
}

async fn load_empty_gadget(coordinator: &WindowCoordinator) {
    use crate::misc::bit_matrix::zeros;

    coordinator
        .load_library(Request::new(bin::Library {
            gadget_types: vec![bin::GadgetType {
                gtype: 1,
                correction_propagation: Some(zeros(0, 1)),
                readout_propagation: Some(zeros(0, 1)),
                logical_correction: Some(zeros(0, 0)),
                physical_correction: Some(zeros(0, 0)),
                ..Default::default()
            }],
            check_model_types: vec![bin::CheckModelType {
                ctype: 1,
                gtype: 1,
                ..Default::default()
            }],
            error_model_types: vec![bin::ErrorModelType {
                etype: 1,
                ctype: 1,
                ..Default::default()
            }],
            ..Default::default()
        }))
        .await
        .unwrap();
}

#[cfg(feature = "mwpm")]
#[tokio::test]
async fn persistent_windows_survive_shot_reset_but_not_library_or_decoder_reset() {
    for reset_library in [false, true] {
        let coordinator = WindowCoordinator::new(
            serde_json::json!({"buffer_radius": 0, "persistent_decoder": true}),
            DynDecoder::BlackBoxMwpm(Arc::new(crate::decoder::MwpmDecoder::new(serde_json::json!({"parallel": 2})))),
        );
        load_empty_gadget(&coordinator).await;
        let mut first_handle = None;
        for _ in 0..2 {
            register_empty_gadget(&coordinator).await;
            coordinator
                .execute(Request::new(bin::Instruction {
                    create: Some(bin::instruction::Create::ErrorModel(bin::ErrorModel {
                        eid: 1,
                        cid: 1,
                        etype: 1,
                        ..Default::default()
                    })),
                }))
                .await
                .unwrap();
            coordinator
                .decode(Request::new(coordinator::Outcomes {
                    gid: 1,
                    outcomes: Some(BitVector::default()),
                    ..Default::default()
                }))
                .await
                .unwrap();
            let handle = {
                let cached = coordinator.persistent_prepared_windows.lock().unwrap();
                assert_eq!(cached.len(), 1);
                *cached.values().next().unwrap().loaded_hid.get().unwrap()
            };
            assert!(coordinator.loaded_decoders.read().await.is_empty());
            if let Some(first_handle) = first_handle {
                assert_eq!(handle, first_handle);
            } else {
                first_handle = Some(handle);
            }
            coordinator
                .reset(Request::new(coordinator::ResetRequest::default()))
                .await
                .unwrap();
        }
        coordinator
            .reset(Request::new(coordinator::ResetRequest {
                reset_library,
                reset_decoder_service: !reset_library,
                ..Default::default()
            }))
            .await
            .unwrap();
        assert!(coordinator.persistent_prepared_windows.lock().unwrap().is_empty());
        assert!(coordinator.loaded_decoders.read().await.is_empty());
        let error = coordinator
            .decoder
            .decode_loaded(blackbox_decoder::LoadedDecodingProblem {
                hid: first_handle.unwrap(),
                syndrome: Some(BitVector::default()),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert_eq!(error.code(), tonic::Code::NotFound);
    }
}

#[tokio::test]
async fn duplicate_outcomes_do_not_overwrite_pending_measurements_or_seed() {
    let coordinator = WindowCoordinator::new(
        serde_json::json!({"buffer_radius": 0}),
        DynDecoder::Mock(Arc::new(MockDecoder::new())),
    );
    load_empty_gadget(&coordinator).await;
    register_empty_gadget(&coordinator).await;
    let outcomes = coordinator::Outcomes {
        gid: 1,
        outcomes: Some(BitVector::default()),
        decoder_seed: Some(13),
        ..Default::default()
    };
    let mut first = Box::pin(coordinator.decode(Request::new(outcomes.clone())));
    assert!(first.as_mut().now_or_never().is_none());
    let second = coordinator.decode(Request::new(coordinator::Outcomes {
        decoder_seed: Some(42),
        ..outcomes
    }));
    let error = second
        .now_or_never()
        .expect("duplicates must be rejected before dependency waits")
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::AlreadyExists);
    assert_eq!(coordinator.gadgets.read().await[&1].decoder_seed, Some(13));
    coordinator.cancel_pending().await;
    assert_eq!(first.await.unwrap_err().code(), tonic::Code::Cancelled);
}

#[tokio::test]
async fn dropping_a_reserved_decode_cancels_the_shot_instead_of_stranding_followers() {
    let mock = Arc::new(MockDecoder::new());
    let coordinator = Arc::new(WindowCoordinator::new(
        serde_json::json!({"buffer_radius": 0}),
        DynDecoder::Mock(Arc::clone(&mock)),
    ));
    load_empty_gadget(&coordinator).await;
    register_empty_gadget(&coordinator).await;
    coordinator
        .execute(Request::new(bin::Instruction {
            create: Some(bin::instruction::Create::ErrorModel(bin::ErrorModel {
                eid: 1,
                cid: 1,
                etype: 1,
                ..Default::default()
            })),
        }))
        .await
        .unwrap();
    let blocker = mock.block_next_decode();
    let decoding = tokio::spawn({
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
    tokio::time::timeout(std::time::Duration::from_secs(5), blocker.wait_until_started())
        .await
        .unwrap();
    assert_eq!(coordinator.gadgets.read().await[&1].state.borrow().reserved_by, Some(1));
    decoding.abort();
    assert!(decoding.await.unwrap_err().is_cancelled());
    assert!(coordinator.cancellation.read().await.is_cancelled());
    blocker.release();
    coordinator
        .reset(Request::new(coordinator::ResetRequest::default()))
        .await
        .unwrap();
    assert!(coordinator.pinned_gadgets.lock().unwrap().is_empty());
    assert!(!coordinator.cancellation.read().await.is_cancelled());
}

async fn register_empty_gadget(coordinator: &WindowCoordinator) {
    for create in [
        bin::instruction::Create::Gadget(bin::Gadget {
            gid: 1,
            gtype: 1,
            ..Default::default()
        }),
        bin::instruction::Create::CheckModel(bin::CheckModel {
            cid: 1,
            gid: 1,
            ctype: 1,
            ..Default::default()
        }),
    ] {
        coordinator
            .execute(Request::new(bin::Instruction { create: Some(create) }))
            .await
            .unwrap();
    }
}

#[cfg(feature = "mwpm")]
#[tokio::test]
#[allow(
    clippy::too_many_lines,
    reason = "Keep the cache-path matrix and live-window lifetime assertions together."
)]
async fn evicted_types_release_only_unused_window_graphs_and_decoder_instances() {
    for (persistent, modified) in [(false, false), (true, false), (true, true)] {
        let coordinator = WindowCoordinator::new(
            serde_json::json!({"buffer_radius": 0, "persistent_decoder": persistent}),
            DynDecoder::BlackBoxMwpm(Arc::new(crate::decoder::MwpmDecoder::new(serde_json::json!({"parallel": 1})))),
        );
        load_empty_gadget(&coordinator).await;
        coordinator
            .load_library(Request::new(bin::Library {
                check_model_types: vec![bin::CheckModelType {
                    ctype: 2,
                    gtype: 1,
                    ..Default::default()
                }],
                error_model_types: vec![bin::ErrorModelType {
                    etype: 2,
                    ctype: 2,
                    ..Default::default()
                }],
                ..Default::default()
            }))
            .await
            .unwrap();
        let mut pins = StatePins::new(&coordinator.pinned_gadgets);
        pins.retain([1]);
        for id in 1..=2 {
            for create in [
                bin::instruction::Create::Gadget(bin::Gadget {
                    gid: id,
                    gtype: 1,
                    ..Default::default()
                }),
                bin::instruction::Create::CheckModel(bin::CheckModel {
                    cid: id,
                    gid: id,
                    ctype: id,
                    ..Default::default()
                }),
                bin::instruction::Create::ErrorModel(bin::ErrorModel {
                    eid: id,
                    cid: id,
                    etype: id,
                    modifier: modified.then(|| ErrorModelModifier {
                        probability_modifier: Some(bin::ProbabilityModifier::default()),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
            ] {
                coordinator
                    .execute(Request::new(bin::Instruction { create: Some(create) }))
                    .await
                    .unwrap();
            }
            coordinator
                .decode(Request::new(coordinator::Outcomes {
                    gid: id,
                    outcomes: Some(BitVector::default()),
                    ..Default::default()
                }))
                .await
                .unwrap();
        }
        let handles: Vec<_> = if persistent && !modified {
            coordinator
                .persistent_prepared_windows
                .lock()
                .unwrap()
                .iter()
                .map(|(key, window)| {
                    (
                        key.relative_program.local_gadgets[0].check_model.as_ref().unwrap().ctype,
                        *window.loaded_hid.get().unwrap(),
                    )
                })
                .collect()
        } else {
            coordinator
                .loaded_decoders
                .read()
                .await
                .iter()
                .map(|(key, loaded)| {
                    (
                        key.relative_program.local_gadgets[0].check_model.as_ref().unwrap().ctype,
                        loaded.hid,
                    )
                })
                .collect()
        };
        assert_eq!(handles.len(), if persistent { 2 } else { 0 });
        coordinator
            .unload(Request::new(coordinator::UnloadLibrary {
                ctype: vec![1],
                etype: vec![1],
                ..Default::default()
            }))
            .await
            .unwrap();
        assert!(coordinator.check_model_types.read().await.contains_key(&1));
        for (_, hid) in &handles {
            coordinator
                .decoder
                .decode_loaded(blackbox_decoder::LoadedDecodingProblem {
                    hid: *hid,
                    syndrome: Some(BitVector::default()),
                    ..Default::default()
                })
                .await
                .unwrap();
        }
        drop(pins);
        coordinator.collect_garbage().await.unwrap();
        assert!(!coordinator.check_model_types.read().await.contains_key(&1));
        assert!(coordinator.check_model_types.read().await.contains_key(&2));
        assert_eq!(
            coordinator.loaded_decoders.read().await.len(),
            usize::from(persistent && modified)
        );
        assert_eq!(
            coordinator.persistent_prepared_windows.lock().unwrap().len(),
            usize::from(persistent && !modified)
        );
        assert_eq!(coordinator.prepared_windows.lock().unwrap().len(), usize::from(!persistent));
        for (ctype, hid) in handles {
            let result = coordinator
                .decoder
                .decode_loaded(blackbox_decoder::LoadedDecodingProblem {
                    hid,
                    syndrome: Some(BitVector::default()),
                    ..Default::default()
                })
                .await;
            if ctype == 1 {
                assert_eq!(result.unwrap_err().code(), tonic::Code::NotFound);
            } else {
                result.unwrap();
            }
        }
    }
}

#[tokio::test]
async fn concurrent_persistent_misses_and_cancelled_eviction_do_not_orphan_handles() {
    let mock = Arc::new(MockDecoder::new());
    let coordinator = WindowCoordinator::new(
        serde_json::json!({"buffer_radius": 0, "window_parallelism": "fully_parallel"}),
        DynDecoder::Mock(Arc::clone(&mock)),
    );
    load_empty_gadget(&coordinator).await;
    for id in 1..=2 {
        for create in [
            bin::instruction::Create::Gadget(bin::Gadget {
                gid: id,
                gtype: 1,
                ..Default::default()
            }),
            bin::instruction::Create::CheckModel(bin::CheckModel {
                cid: id,
                gid: id,
                ctype: 1,
                ..Default::default()
            }),
            bin::instruction::Create::ErrorModel(bin::ErrorModel {
                eid: id,
                cid: id,
                etype: 1,
                modifier: Some(ErrorModelModifier {
                    probability_modifier: Some(bin::ProbabilityModifier::default()),
                    ..Default::default()
                }),
                ..Default::default()
            }),
        ] {
            coordinator
                .execute(Request::new(bin::Instruction { create: Some(create) }))
                .await
                .unwrap();
        }
    }
    let loading = mock.state.write().await;
    let decode = |gid| {
        coordinator.decode(Request::new(coordinator::Outcomes {
            gid,
            outcomes: Some(BitVector::default()),
            ..Default::default()
        }))
    };
    let mut first = Box::pin(decode(1));
    let mut second = Box::pin(decode(2));
    assert!(first.as_mut().now_or_never().is_none());
    assert!(second.as_mut().now_or_never().is_none());
    drop(loading);
    let (first, second) = tokio::join!(first, second);
    first.unwrap();
    second.unwrap();
    assert_eq!(mock.state.read().await.loaded_hypergraphs.len(), 1);
    assert_eq!(coordinator.loaded_decoders.read().await.len(), 1);

    let unloading = mock.state.write().await;
    let request = coordinator::UnloadLibrary {
        etype: vec![1],
        ..Default::default()
    };
    let mut eviction = Box::pin(coordinator.unload(Request::new(request.clone())));
    assert!(eviction.as_mut().now_or_never().is_none());
    drop(eviction);
    assert!(coordinator.error_model_types.read().await.contains_key(&1));
    assert_eq!(coordinator.loaded_decoders.read().await.len(), 1);
    drop(unloading);
    coordinator.unload(Request::new(request)).await.unwrap();
    assert!(coordinator.loaded_decoders.read().await.is_empty());
    assert!(mock.state.read().await.loaded_hypergraphs.is_empty());
}

#[tokio::test]
async fn metadata_admission_does_not_wait_for_collection() {
    let coordinator = coordinator();
    load_empty_gadget(&coordinator).await;
    let collection = coordinator.collection_completed.lock().await;
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        register_empty_gadget(&coordinator).await;
        coordinator
            .execute(Request::new(bin::Instruction {
                create: Some(bin::instruction::Create::ErrorModel(bin::ErrorModel {
                    eid: 1,
                    cid: 1,
                    etype: 1,
                    ..Default::default()
                })),
            }))
            .await
            .unwrap();
    })
    .await
    .expect("publishing unfinished metadata must not wait for GC");
    assert_eq!(coordinator.collection_requests.load(Ordering::Acquire), 0);
    drop(collection);
    coordinator
        .reset(Request::new(coordinator::ResetRequest::default()))
        .await
        .unwrap();
    assert!(coordinator.gadgets.read().await.is_empty());
}

#[tokio::test]
async fn late_error_models_request_collection_after_readout_completion() {
    let coordinator = coordinator();
    load_empty_gadget(&coordinator).await;
    register_empty_gadget(&coordinator).await;
    coordinator.completed_readouts.lock().await.insert(1);
    coordinator
        .execute(Request::new(bin::Instruction {
            create: Some(bin::instruction::Create::ErrorModel(bin::ErrorModel {
                eid: 1,
                cid: 1,
                etype: 1,
                ..Default::default()
            })),
        }))
        .await
        .unwrap();
    assert_eq!(coordinator.collection_requests.load(Ordering::Acquire), 1);
    coordinator
        .reset(Request::new(coordinator::ResetRequest::default()))
        .await
        .unwrap();
}

#[tokio::test]
async fn prepared_windows_reuse_graphs_without_loading_persistent_decoders() {
    let coordinator = WindowCoordinator::new(
        serde_json::json!({"buffer_radius": 0, "persistent_decoder": false}),
        DynDecoder::Mock(Arc::new(MockDecoder::new())),
    );
    load_empty_gadget(&coordinator).await;
    let mut original = None;
    for gid in 1..=3 {
        for create in [
            bin::instruction::Create::Gadget(bin::Gadget {
                gid,
                gtype: 1,
                ..Default::default()
            }),
            bin::instruction::Create::CheckModel(bin::CheckModel {
                gid,
                cid: gid,
                ctype: 1,
                ..Default::default()
            }),
            bin::instruction::Create::ErrorModel(bin::ErrorModel {
                eid: gid,
                cid: gid,
                etype: 1,
                modifier: (gid == 3).then(|| bin::error_model::ErrorModelModifier {
                    probability_modifier: Some(bin::ProbabilityModifier::default()),
                    ..Default::default()
                }),
                ..Default::default()
            }),
        ] {
            coordinator
                .execute(Request::new(bin::Instruction { create: Some(create) }))
                .await
                .unwrap();
        }
        let preparation = if gid == 1 {
            None
        } else {
            Some(coordinator.prepared_window_preparation.lock().await)
        };
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            coordinator.decode(Request::new(coordinator::Outcomes {
                gid,
                outcomes: Some(BitVector::default()),
                ..Default::default()
            })),
        )
        .await
        .unwrap()
        .unwrap();
        drop(preparation);
        assert!(coordinator.loaded_decoders.read().await.is_empty());
        let cache = coordinator.prepared_windows.lock().unwrap();
        assert_eq!(cache.len(), 1);
        let cached = Arc::clone(cache.iter().next().unwrap().1);
        if let Some(original) = &original {
            assert!(Arc::ptr_eq(original, &cached));
        } else {
            original = Some(cached);
        }
    }
    coordinator
        .reset(Request::new(coordinator::ResetRequest::default()))
        .await
        .unwrap();
    assert!(coordinator.prepared_windows.lock().unwrap().is_empty());
}

#[tokio::test]
async fn graph_collection_does_not_lock_unmodified_type_libraries() {
    let coordinator = coordinator();
    coordinator.completed_readouts.lock().await.insert(1);
    let _check_types = coordinator.check_model_types.read().await;
    let _error_types = coordinator.error_model_types.read().await;
    coordinator
        .collect_garbage()
        .now_or_never()
        .expect("graph retirement must not require exclusive type-library access")
        .unwrap();
}

#[tokio::test]
async fn memory_stats_finish_before_a_queued_type_retirement() {
    let coordinator = coordinator();
    load_empty_gadget(&coordinator).await;
    {
        let mut pending = coordinator.pending_type_unloads.lock().await;
        pending.ctype.push(1);
        pending.etype.push(1);
    }
    let gadgets = coordinator.gadgets.write().await;
    let mut statistics = Box::pin(coordinator.get_memory_stats(Request::new(())));
    assert!(statistics.as_mut().now_or_never().is_none());
    assert!(coordinator.check_model_types.try_write().is_err());
    assert!(coordinator.error_model_types.try_write().is_err());
    let mut retirement = Box::pin(coordinator.collect_unused_types());
    assert!(retirement.as_mut().now_or_never().is_none());
    drop(gadgets);

    let (statistics, ()) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        tokio::join!(statistics, retirement)
    })
    .await
    .expect("queued type writers must not prevent the existing statistics reader from finishing");
    let statistics = statistics.unwrap().into_inner();
    assert_eq!(statistics.check_model_types, 1);
    assert_eq!(statistics.error_model_types, 1);
    assert!(coordinator.check_model_types.read().await.is_empty());
    assert!(coordinator.error_model_types.read().await.is_empty());
}

#[tokio::test]
async fn memory_stats_and_collection_release_graph_locks_before_retiring_types() {
    let coordinator = WindowCoordinator::new(
        serde_json::json!({"buffer_radius": 0}),
        DynDecoder::Mock(Arc::new(MockDecoder::new())),
    );
    load_empty_gadget(&coordinator).await;
    register_empty_gadget(&coordinator).await;
    coordinator
        .execute(Request::new(bin::Instruction {
            create: Some(bin::instruction::Create::ErrorModel(bin::ErrorModel {
                eid: 1,
                cid: 1,
                etype: 1,
                ..Default::default()
            })),
        }))
        .await
        .unwrap();
    let mut pins = StatePins::new(&coordinator.pinned_gadgets);
    pins.retain([1]);
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        coordinator.decode(Request::new(coordinator::Outcomes {
            gid: 1,
            outcomes: Some(BitVector::default()),
            ..Default::default()
        })),
    )
    .await
    .unwrap()
    .unwrap();
    drop(pins);
    {
        let mut pending = coordinator.pending_type_unloads.lock().await;
        pending.ctype.push(1);
        pending.etype.push(1);
    }
    let tracker = coordinator.pauli_frame_tracker.lock().await;
    let mut collection = Box::pin(coordinator.collect_garbage());
    assert!(collection.as_mut().now_or_never().is_none());
    assert!(coordinator.gadgets.try_read().is_err());
    assert!(coordinator.check_models.try_read().is_err());
    assert!(coordinator.error_models.try_read().is_err());
    let mut statistics = Box::pin(coordinator.get_memory_stats(Request::new(())));
    assert!(statistics.as_mut().now_or_never().is_none());
    assert!(coordinator.check_model_types.try_write().is_err());
    drop(tracker);

    let (collection, statistics) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        tokio::join!(collection, statistics)
    })
    .await
    .expect("collection must release graph locks before waiting for statistics to release type locks");
    collection.unwrap();
    let statistics = statistics.unwrap().into_inner();
    assert_eq!(
        (statistics.gadgets, statistics.check_models, statistics.error_models),
        (0, 0, 0)
    );
    assert_eq!(statistics.frame_gadgets, 0);
    assert_eq!(statistics.collected_id_ranges, 3);
    assert!(coordinator.check_model_types.read().await.is_empty());
    assert!(coordinator.error_model_types.read().await.is_empty());
}

#[tokio::test]
async fn memory_stats_can_finish_while_a_reset_is_waiting() {
    for reset_library in [false, true] {
        let coordinator = coordinator();
        load_empty_gadget(&coordinator).await;
        let tracker = coordinator.pauli_frame_tracker.lock().await;
        let mut statistics = Box::pin(coordinator.get_memory_stats(Request::new(())));
        assert!(statistics.as_mut().now_or_never().is_none());
        assert!(coordinator.gadgets.try_write().is_err());
        let mut reset = Box::pin(coordinator.reset(Request::new(coordinator::ResetRequest {
            reset_library,
            ..Default::default()
        })));
        assert!(reset.as_mut().now_or_never().is_none());
        drop(tracker);

        let (statistics, reset) =
            tokio::time::timeout(std::time::Duration::from_secs(5), async { tokio::join!(statistics, reset) })
                .await
                .expect("statistics must finish without waiting for a reset that needs its locks");
        let statistics = statistics.unwrap().into_inner();
        assert_eq!(statistics.check_model_types, 1);
        assert_eq!(statistics.error_model_types, 1);
        reset.unwrap();
        assert_eq!(coordinator.check_model_types.read().await.is_empty(), reset_library);
        assert_eq!(coordinator.error_model_types.read().await.is_empty(), reset_library);
    }
}

#[tokio::test]
async fn type_collection_rechecks_models_after_graph_retirement() {
    let coordinator = coordinator();
    coordinator
        .error_model_types
        .write()
        .await
        .insert(9, Arc::new(bin::ErrorModelType::default()));
    coordinator.pending_type_unloads.lock().await.etype.push(9);
    coordinator.collect_gadget_state().await.unwrap();
    coordinator.error_models.write().await.insert(
        1,
        ErrorModel {
            instance: bin::ErrorModel {
                etype: 9,
                ..Default::default()
            },
            modified_remote_check_models: Arc::new(vec![]),
        },
    );
    coordinator.collect_unused_types().await;
    assert!(coordinator.error_model_types.read().await.contains_key(&9));
    assert_eq!(coordinator.pending_type_unloads.lock().await.etype, vec![9]);
    coordinator.error_models.write().await.clear();
    coordinator.collect_unused_types().await;
    assert!(coordinator.error_model_types.read().await.is_empty());
    assert!(coordinator.pending_type_unloads.lock().await.etype.is_empty());
}

#[tokio::test]
async fn collection_before_readout_completion_skips_graph_retirement() {
    let coordinator = coordinator();
    let _tracker = coordinator.pauli_frame_tracker.lock().await;
    coordinator
        .collect_garbage()
        .now_or_never()
        .expect("unfinished readouts root all gadgets; retirement must not acquire the tracker")
        .unwrap();
}

#[tokio::test]
async fn collection_before_readout_completion_still_unloads_unused_types() {
    let coordinator = coordinator();
    coordinator
        .check_model_types
        .write()
        .await
        .insert(1, Arc::new(bin::CheckModelType::default()));
    coordinator.pending_type_unloads.lock().await.ctype.push(1);
    coordinator.collect_garbage().await.unwrap();
    assert!(coordinator.check_model_types.read().await.is_empty());
    assert!(coordinator.pending_type_unloads.lock().await.ctype.is_empty());
}

#[tokio::test]
async fn completed_readouts_still_trigger_graph_retirement() {
    let coordinator = coordinator();
    coordinator.completed_readouts.lock().await.insert(1);
    let tracker = coordinator.pauli_frame_tracker.lock().await;
    let mut collection = Box::pin(coordinator.collect_garbage());
    assert!(collection.as_mut().now_or_never().is_none());
    drop(tracker);
    collection.await.unwrap();
}

#[tokio::test]
async fn readout_collection_scales_batches_with_a_minimum_and_drains_quiescent_state() {
    for (live, expected_interval) in [(1, 32), (32, 32), (256, 32), (263, 32), (264, 33), (1024, 128)] {
        let coordinator = coordinator();
        for gid in 1..=live {
            coordinator.gadgets.write().await.insert(
                gid,
                Gadget {
                    instance: bin::Gadget {
                        gid,
                        ..Default::default()
                    },
                    outcomes: watch::channel(None).0,
                    probability_modifiers: vec![],
                    binding_cid: None,
                    outputs: vec![],
                    pauli_frame: watch::channel(None).0,
                    correction_count: 0,
                    correction_weight: 0.0,
                    is_free_hop: false,
                    state: watch::channel(GadgetState::default()).0,
                    loss_mask: None,
                    decoder_seed: None,
                },
            );
        }
        for _ in 1..expected_interval {
            coordinator.collect_after_readout().await.unwrap();
            assert_eq!(*coordinator.collection_completed.lock().await, 0, "live={live}");
        }
        coordinator.collect_after_readout().await.unwrap();
        assert_eq!(
            *coordinator.collection_completed.lock().await,
            expected_interval,
            "live={live}"
        );
        coordinator.gadgets.write().await.clear();
        coordinator.collect_after_readout().await.unwrap();
        assert_eq!(*coordinator.collection_completed.lock().await, expected_interval + 1);
    }
}

#[tokio::test]
async fn concurrent_requests_share_a_completed_collection_pass() {
    let coordinator = coordinator();
    let collection = coordinator.collection_completed.lock().await;
    let mut requests: Vec<_> = (0..8).map(|_| Box::pin(coordinator.collect_garbage())).collect();
    for request in &mut requests {
        assert!(request.as_mut().now_or_never().is_none());
    }
    drop(collection);
    requests.remove(0).await.unwrap();
    let _gadgets = coordinator.gadgets.write().await;
    for request in requests {
        request
            .now_or_never()
            .expect("requests covered by the completed pass must not acquire graph locks again")
            .unwrap();
    }
}

#[tokio::test]
async fn a_request_during_collection_requires_a_later_pass() {
    let coordinator = coordinator();
    let gadgets = coordinator.gadgets.write().await;
    let mut first = Box::pin(coordinator.collect_garbage());
    assert!(first.as_mut().now_or_never().is_none());
    let mut second = Box::pin(coordinator.collect_garbage());
    assert!(second.as_mut().now_or_never().is_none());
    drop(gadgets);
    first.await.unwrap();
    let gadgets = coordinator.gadgets.write().await;
    assert!(second.as_mut().now_or_never().is_none());
    drop(gadgets);
    second.await.unwrap();
}

#[tokio::test]
async fn cancelled_collection_does_not_acknowledge_pending_work() {
    let coordinator = coordinator();
    let gadgets = coordinator.gadgets.write().await;
    let mut collection = Box::pin(coordinator.collect_garbage());
    assert!(collection.as_mut().now_or_never().is_none());
    drop(collection);
    assert_eq!(*coordinator.collection_completed.lock().await, 0);
    drop(gadgets);
    coordinator.collect_garbage().await.unwrap();
    assert_eq!(*coordinator.collection_completed.lock().await, 2);
}

#[tokio::test]
async fn collection_request_counter_can_wrap() {
    let coordinator = coordinator();
    coordinator.collection_requests.store(u64::MAX, Ordering::Release);
    *coordinator.collection_completed.lock().await = u64::MAX;
    coordinator.collect_garbage().await.unwrap();
    assert_eq!(*coordinator.collection_completed.lock().await, 0);
    coordinator.collect_garbage().await.unwrap();
    assert_eq!(*coordinator.collection_completed.lock().await, 1);
}
