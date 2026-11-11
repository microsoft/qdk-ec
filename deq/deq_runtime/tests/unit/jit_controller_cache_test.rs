use super::*;
use crate::coordinator::{DynCoordinator, MockCoordinator};
use futures_util::FutureExt;

#[test]
fn cache_limit_ratio_defaults_and_validation() {
    let default: JitControllerConfig = serde_json::from_value(serde_json::json!({})).unwrap();
    assert_eq!(default.cache_limit_ratio, 10);
    assert_eq!(default.cache_limit(2), 20);
    assert_eq!(default.cache_limit(0), 0);
    assert_eq!(default.cache_limit(usize::MAX), usize::MAX);
    let explicit: JitControllerConfig = serde_json::from_value(serde_json::json!({"cache_limit_ratio": 5})).unwrap();
    assert_eq!(explicit.cache_limit(2), 10);
    assert_eq!(serde_json::to_value(explicit).unwrap()["cache_limit_ratio"], 5);
    assert_eq!(
        JitController::new_from_library(jit::JitLibrary::default())
            .config
            .cache_limit_ratio,
        10
    );
    let disabled: JitControllerConfig = serde_json::from_value(serde_json::json!({"cache_limit_ratio": 0})).unwrap();
    assert_eq!(disabled.cache_limit(usize::MAX), 0);
    assert_eq!(serde_json::to_value(disabled).unwrap()["cache_limit_ratio"], 0);
    for invalid in [serde_json::json!(-1), serde_json::json!(1.5), serde_json::json!("3")] {
        assert!(serde_json::from_value::<JitControllerConfig>(serde_json::json!({"cache_limit_ratio": invalid})).is_err());
    }
    let error = serde_json::from_value::<JitControllerConfig>(serde_json::json!({"cache_enabled": false})).unwrap_err();
    assert!(error.to_string().contains("unknown field `cache_enabled`"));
    let error = serde_json::from_value::<JitControllerConfig>(serde_json::json!({"error_template_cache_capacity": 128}))
        .unwrap_err();
    assert!(error.to_string().contains("unknown field `error_template_cache_capacity`"));
    #[cfg(feature = "cli")]
    {
        let help = crate::controller::ControllerType::config_help();
        assert!(help.contains("cache_limit_ratio"));
        assert!(!help.contains("cache_enabled"));
        assert!(!help.contains("error_template_cache_capacity"));
    }
}

fn collecting_controller() -> Arc<JitController> {
    let controller = JitController::new(serde_json::json!({}));
    controller.automatic_collection.store(true, Ordering::Release);
    controller
}

async fn register_cache_budget_type(controller: &JitController) {
    controller
        .compiler
        .load_library(jit::JitLibrary {
            gadget_types: vec![jit::JitGadgetType {
                base: Some(bin::GadgetType {
                    gtype: 1,
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        })
        .await;
}

#[tokio::test]
async fn zero_ratio_allocates_fresh_types_and_retires_uncached_declarations() {
    for automatic_collection in [false, true] {
        let controller = JitController::new(serde_json::json!({"cache_limit_ratio": 0}));
        let coordinator = CoordinatorClient::Local(DynCoordinator::Window(Arc::new(
            crate::coordinator::WindowCoordinator::new(
                serde_json::json!({"forced_gap": !automatic_collection}),
                crate::decoder::DynDecoder::Mock(Arc::new(crate::decoder::MockDecoder::new())),
            ),
        )));
        controller.start(coordinator.clone()).await;
        controller
            .load_library(jit::JitLibrary {
                gadget_types: vec![jit::JitGadgetType {
                    base: Some(bin::GadgetType {
                        gtype: 1,
                        ..Default::default()
                    }),
                    ..Default::default()
                }],
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(controller.automatic_collection.load(Ordering::Acquire), automatic_collection);
        let mut check = bin::CheckModelType {
            gtype: 1,
            ..Default::default()
        };
        let error = Arc::new(bin::ErrorModelType::default());
        let mut ctypes = HashSet::new();
        let mut etypes = HashSet::new();
        for _ in 0..3 {
            assert!(ctypes.insert(controller.get_or_load_ctype(&mut check, &coordinator).await.unwrap()));
            assert!(etypes.insert(controller.get_or_load_etype(&error, &coordinator).await.unwrap()));
        }
        {
            let cache = controller.type_cache.read().await;
            assert!(cache.check_model_types.is_empty());
            assert!(cache.error_model_types.is_empty());
            assert!(cache.check_model_recency.lock().unwrap().is_empty());
            assert!(cache.error_model_recency.lock().unwrap().is_empty());
            assert!(cache.error_model_identities.lock().unwrap().is_empty());
        }
        controller.collect_unused_generated_types().await.unwrap();
        let stats = controller.memory_stats().await.unwrap();
        assert_eq!(stats.check_model_types, if automatic_collection { 0 } else { 3 });
        assert_eq!(stats.error_model_types, if automatic_collection { 1 } else { 4 });
        controller
            .reset(crate::coordinator::ResetRequest {
                reset_library: true,
                ..Default::default()
            })
            .await
            .unwrap();
        let stats = controller.memory_stats().await.unwrap();
        assert_eq!(stats.check_model_types, 0);
        assert_eq!(stats.error_model_types, 0);
    }
}

mod background_failures {
    use super::*;

    fn instruction(gid: u64) -> jit::JitInstruction {
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gid,
                gtype: 1,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    async fn controller() -> (Arc<JitController>, Arc<MockCoordinator>) {
        let controller = JitController::new_from_library(jit::JitLibrary {
            gadget_types: vec![jit::JitGadgetType {
                base: Some(bin::GadgetType {
                    gtype: 1,
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        });
        let mock = MockCoordinator::new();
        controller.start(CoordinatorClient::from_mock(Arc::clone(&mock))).await;
        (controller, mock)
    }

    fn outcomes(gid: u64) -> crate::coordinator::Outcomes {
        crate::coordinator::Outcomes {
            gid,
            outcomes: Some(crate::util::BitVector::default()),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn background_load_attachment_and_collection_failures_reach_callers_until_successful_reset() {
        for operation in ["load", "attach", "collect"] {
            let (controller, mock) = controller().await;
            controller.try_execute(instruction(1)).await.unwrap();
            let failure = tonic::Status::data_loss(format!("{operation} rejected"));
            {
                let mut state = mock.state.write().await;
                match operation {
                    "load" => state.load_error = Some(failure),
                    "attach" => state.error_model_error = Some(failure),
                    _ => {
                        controller.automatic_collection.store(true, Ordering::Relaxed);
                        controller.generated_types.lock().await.ctype.push(99);
                        state.unload_error = Some(failure);
                    }
                }
            }
            tokio::time::timeout(std::time::Duration::from_secs(5), controller.task_counter.wait_for_zero())
                .await
                .unwrap();
            let error = controller.decode_single(outcomes(1)).await.unwrap_err();
            assert_eq!(error.code(), tonic::Code::DataLoss);
            assert!(error.message().contains("gid=1"));
            assert!(error.message().contains(&format!("{operation} rejected")));
            assert!(
                matches!(controller.batch_execute(vec![instruction(2)]).await.unwrap_err(), BatchExecuteError::Execution(status) if status.message() == error.message())
            );
            assert_eq!(
                controller.try_execute(instruction(2)).await.unwrap_err().message(),
                error.message()
            );
            assert_eq!(controller.batch_decode(vec![]).await.unwrap_err().message(), error.message());
            assert_eq!(
                controller
                    .load_library(jit::JitLibrary::default())
                    .await
                    .unwrap_err()
                    .message(),
                error.message()
            );
            controller.memory_stats().await.unwrap();
            mock.state.write().await.reset_error = Some(tonic::Status::unavailable("reset failed"));
            assert_eq!(
                controller
                    .reset(crate::coordinator::ResetRequest::default())
                    .await
                    .unwrap_err()
                    .code(),
                tonic::Code::Unavailable
            );
            if operation == "collect" {
                assert_eq!(controller.generated_types.lock().await.ctype, vec![99]);
            }
            assert_eq!(
                controller.decode_single(outcomes(1)).await.unwrap_err().message(),
                error.message()
            );
            {
                let mut state = mock.state.write().await;
                state.reset_error = None;
                state.load_error = None;
                state.error_model_error = None;
                state.unload_error = None;
            }
            controller.reset(crate::coordinator::ResetRequest::default()).await.unwrap();
            assert!(controller.background_error.lock().unwrap().is_none());
            controller.try_execute(instruction(1)).await.unwrap();
            controller.task_counter.wait_for_zero().await;
            controller.decode_single(outcomes(1)).await.unwrap();
        }
    }

    #[tokio::test]
    async fn pending_operations_return_the_first_failure_with_details_and_metadata() {
        let (controller, _) = controller().await;
        controller.undecoded_gids.write().await.extend([1, 2]);
        let token = controller.cancellation.read().await.clone();
        let coordinator = controller.coordinator.write().await;
        let publication = controller.collection.write().await;
        let mut single = Box::pin(controller.decode_single(outcomes(1)));
        let mut batch = Box::pin(controller.batch_decode(vec![outcomes(2)]));
        let mut execute = Box::pin(controller.try_execute(instruction(3)));
        let mut batch_execute = Box::pin(controller.batch_execute(vec![instruction(4)]));
        let mut loading = Box::pin(controller.load_library(jit::JitLibrary {
            gadget_types: vec![jit::JitGadgetType {
                base: Some(bin::GadgetType {
                    gtype: 2,
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }));
        assert!(single.as_mut().now_or_never().is_none());
        assert!(batch.as_mut().now_or_never().is_none());
        assert!(execute.as_mut().now_or_never().is_none());
        assert!(batch_execute.as_mut().now_or_never().is_none());
        assert!(loading.as_mut().now_or_never().is_none());
        let mut error =
            tonic::Status::with_details(tonic::Code::ResourceExhausted, "type budget exhausted", vec![1, 2].into());
        error.metadata_mut().insert("error-kind", "test".parse().unwrap());
        controller.record_background_error(&token, &error, 9, "failed to attach error model");
        controller.record_background_error(&token, &tonic::Status::internal("secondary failure"), 10, "collection failed");
        let errors = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            let BatchExecuteError::Execution(execution_error) = batch_execute.await.unwrap_err() else {
                panic!("expected execution error")
            };
            vec![
                single.await.unwrap_err(),
                batch.await.unwrap_err(),
                execute.await.unwrap_err(),
                execution_error,
                loading.await.unwrap_err(),
            ]
        })
        .await
        .unwrap();
        for error in errors {
            assert_eq!(error.code(), tonic::Code::ResourceExhausted);
            assert_eq!(
                error.message(),
                "failed to attach error model for gid=9: type budget exhausted"
            );
            assert_eq!(error.details(), [1, 2]);
            assert_eq!(error.metadata().get("error-kind").unwrap(), "test");
            #[cfg(feature = "cli")]
            {
                let rpc_error = tonic::Status::from(BatchExecuteError::Execution(error.clone()));
                assert_eq!(rpc_error.code(), error.code());
                assert_eq!(rpc_error.details(), error.details());
                assert_eq!(rpc_error.metadata().get("error-kind"), error.metadata().get("error-kind"));
            }
        }
        drop(publication);
        drop(coordinator);
    }

    #[tokio::test]
    async fn shutdown_cancellation_and_old_tasks_do_not_create_a_new_shot_failure() {
        let (controller, _) = controller().await;
        let token = controller.cancellation.read().await.clone();
        controller.cancel_pending().await;
        controller.record_background_error(&token, &tonic::Status::internal("late failure"), 1, "attachment failed");
        let error = controller.decode_single(outcomes(1)).await.unwrap_err();
        assert_eq!(error.code(), tonic::Code::Cancelled);
        assert!(error.message().contains("shutdown or reset"));
        assert!(controller.background_error.lock().unwrap().is_none());
        controller.reset(crate::coordinator::ResetRequest::default()).await.unwrap();
        controller.record_background_error(&token, &tonic::Status::internal("old task"), 1, "attachment failed");
        assert!(controller.background_error.lock().unwrap().is_none());
        controller.try_execute(instruction(1)).await.unwrap();
        controller.task_counter.wait_for_zero().await;
    }

    #[tokio::test]
    async fn failed_type_loads_do_not_populate_cache_entries() {
        let (controller, mock) = controller().await;
        let coordinator = CoordinatorClient::from_mock(Arc::clone(&mock));
        mock.state.write().await.load_error = Some(tonic::Status::unavailable("load failed"));
        assert_eq!(
            controller
                .get_or_load_ctype(&mut bin::CheckModelType::default(), &coordinator)
                .await
                .unwrap_err()
                .code(),
            tonic::Code::Unavailable
        );
        assert_eq!(
            controller
                .get_or_load_etype(&Arc::new(bin::ErrorModelType::default()), &coordinator)
                .await
                .unwrap_err()
                .code(),
            tonic::Code::Unavailable
        );
        let cache = controller.type_cache.read().await;
        assert!(cache.check_model_types.is_empty());
        assert!(cache.error_model_types.is_empty());
        assert!(cache.error_model_identities.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn cancelled_type_unload_preserves_pending_ids_for_retry() {
        let (controller, mock) = controller().await;
        controller.automatic_collection.store(true, Ordering::Relaxed);
        controller.generated_types.lock().await.ctype.push(99);
        let blocked = mock.state.write().await;
        let mut collection = Box::pin(controller.collect_unused_generated_types());
        assert!(collection.as_mut().now_or_never().is_none());
        drop(collection);
        assert_eq!(controller.generated_types.lock().await.ctype, vec![99]);
        drop(blocked);
        controller.collect_unused_generated_types().await.unwrap();
        assert!(controller.generated_types.lock().await.ctype.is_empty());
    }
}

#[tokio::test]
async fn queued_collections_share_exclusive_access_without_blocking_new_readers() {
    let controller = collecting_controller();
    let graph = controller.compiler.gadgets.write().await;
    let mut requests: Vec<_> = (0..1024).map(|_| Box::pin(controller.collect_garbage())).collect();
    for request in &mut requests {
        assert!(request.as_mut().now_or_never().is_none());
    }
    drop(graph);
    requests.remove(0).await.unwrap();
    let _new_reader = controller.collection.read().await;
    let _gadgets = controller.compiler.gadgets.write().await;
    let results = tokio::time::timeout(std::time::Duration::from_secs(5), futures_util::future::join_all(requests))
        .await
        .expect("covered collection requests must finish while new readers hold the collection gate");
    for result in results {
        result.unwrap();
    }
    assert_eq!(*controller.collection_completed.lock().await, 1024);
}

#[tokio::test]
async fn a_collection_request_during_the_pass_is_not_lost() {
    let controller = collecting_controller();
    controller.generated_types.lock().await.ctype.push(1);
    *controller.coordinator.write().await = Some(CoordinatorClient::Local(DynCoordinator::Mock(MockCoordinator::new())));
    let publication = controller.collection.read().await;
    let mut first = Box::pin(controller.collect_garbage());
    assert!(first.as_mut().now_or_never().is_none());
    let mut second = Box::pin(controller.collect_garbage());
    assert!(second.as_mut().now_or_never().is_none());
    drop(publication);
    first.await.unwrap();
    let gadgets = controller.compiler.gadgets.write().await;
    assert!(second.as_mut().now_or_never().is_none());
    drop(gadgets);
    second.await.unwrap();
    assert_eq!(*controller.collection_completed.lock().await, 2);
}

#[tokio::test]
async fn cancelled_collection_keeps_its_generation_pending() {
    let controller = collecting_controller();
    let gadgets = controller.compiler.gadgets.write().await;
    let mut collection = Box::pin(controller.collect_garbage());
    assert!(collection.as_mut().now_or_never().is_none());
    drop(collection);
    assert_eq!(*controller.collection_completed.lock().await, 0);
    drop(gadgets);
    controller.collect_garbage().await.unwrap();
    assert_eq!(*controller.collection_completed.lock().await, 2);
}

#[tokio::test]
async fn cancelled_collector_does_not_leave_a_writer_blocking_admission() {
    let controller = collecting_controller();
    let reader = controller.compiler.gadgets.read().await;
    let mut collection = Box::pin(controller.collect_garbage());
    assert!(collection.as_mut().now_or_never().is_none());
    assert!(controller.compiler.gadgets.try_read().is_err());
    drop(collection);
    assert!(controller.compiler.gadgets.try_read().is_ok());
    drop(reader);
    controller.collect_garbage().await.unwrap();
    assert_eq!(*controller.collection_completed.lock().await, 2);
}

#[tokio::test]
async fn failed_collection_does_not_acknowledge_pending_work() {
    let controller = collecting_controller();
    controller.generated_types.lock().await.ctype.push(1);
    let error = controller.collect_garbage().await.unwrap_err();
    assert_eq!(error.code(), tonic::Code::FailedPrecondition);
    assert_eq!(*controller.collection_completed.lock().await, 0);
    *controller.coordinator.write().await = Some(CoordinatorClient::Local(DynCoordinator::Mock(MockCoordinator::new())));
    controller.collect_garbage().await.unwrap();
    assert_eq!(*controller.collection_completed.lock().await, 2);
    assert!(controller.generated_types.lock().await.ctype.is_empty());
}

#[tokio::test]
async fn graph_collection_does_not_block_on_type_publication_guards() {
    let controller = collecting_controller();
    let _publication = controller.collection.read().await;
    controller
        .collect_garbage()
        .now_or_never()
        .expect("compiler graph collection must not take the type-publication barrier")
        .unwrap();
}

#[tokio::test]
async fn type_unloading_still_waits_for_in_flight_publications() {
    let controller = collecting_controller();
    controller.generated_types.lock().await.ctype.push(1);
    *controller.coordinator.write().await = Some(CoordinatorClient::Local(DynCoordinator::Mock(MockCoordinator::new())));
    let publication = controller.collection.read().await;
    let mut collection = Box::pin(controller.collect_garbage());
    assert!(collection.as_mut().now_or_never().is_none());
    assert_eq!(controller.generated_types.lock().await.ctype, vec![1]);
    drop(publication);
    collection.await.unwrap();
    assert!(controller.generated_types.lock().await.ctype.is_empty());
}

#[tokio::test]
async fn rejected_type_unloads_remain_pending_for_retry() {
    use crate::coordinator::WindowCoordinator;
    use crate::decoder::{DynDecoder, MockDecoder};

    let controller = collecting_controller();
    controller.generated_types.lock().await.ctype.push(1);
    *controller.coordinator.write().await = Some(CoordinatorClient::Local(DynCoordinator::Window(Arc::new(
        WindowCoordinator::new(
            serde_json::json!({"forced_gap": true}),
            DynDecoder::Mock(Arc::new(MockDecoder::new())),
        ),
    ))));
    assert!(controller.collect_garbage().await.is_err());
    assert_eq!(controller.generated_types.lock().await.ctype, vec![1]);
    assert_eq!(*controller.collection_completed.lock().await, 0);
}

#[tokio::test]
async fn collection_request_generations_can_wrap() {
    let controller = collecting_controller();
    controller.collection_requests.store(u64::MAX, Ordering::Release);
    *controller.collection_completed.lock().await = u64::MAX;
    controller.collect_garbage().await.unwrap();
    assert_eq!(*controller.collection_completed.lock().await, 0);
    controller.collect_garbage().await.unwrap();
    assert_eq!(*controller.collection_completed.lock().await, 1);
}

#[tokio::test]
async fn cache_hits_do_not_require_exclusive_access() {
    let controller = JitController::new(serde_json::json!({}));
    register_cache_budget_type(&controller).await;
    let coordinator = CoordinatorClient::Local(DynCoordinator::Mock(MockCoordinator::new()));
    let mut check_model = bin::CheckModelType::default();
    let error_model = Arc::new(bin::ErrorModelType::default());
    {
        let mut cache = controller.type_cache.write().await;
        cache.insert_check_model_type(&check_model, 10);
        cache.insert_error_model_type(&error_model, 20);
        cache.remember_error_model_identity(&error_model, 20, 1);
    }
    let _reader = controller.type_cache.read().await;
    assert_eq!(
        controller
            .get_or_load_ctype(&mut check_model, &coordinator)
            .now_or_never()
            .map(Result::unwrap),
        Some(10),
    );
    assert_eq!(
        controller
            .get_or_load_etype(&error_model, &coordinator)
            .now_or_never()
            .map(Result::unwrap),
        Some(20),
    );
}

#[tokio::test]
async fn concurrent_cache_misses_load_one_type() {
    let controller = JitController::new(serde_json::json!({}));
    register_cache_budget_type(&controller).await;
    let mock = MockCoordinator::new();
    let coordinator = CoordinatorClient::Local(DynCoordinator::Mock(Arc::clone(&mock)));
    let state = mock.state.write().await;
    let first_model = Arc::new(bin::ErrorModelType::default());
    let second_model = Arc::new(bin::ErrorModelType::default());
    let mut first = Box::pin(controller.get_or_load_etype(&first_model, &coordinator));
    let mut second = Box::pin(controller.get_or_load_etype(&second_model, &coordinator));
    assert!(first.as_mut().now_or_never().is_none());
    assert!(second.as_mut().now_or_never().is_none());
    drop(state);
    let (first, second) = tokio::join!(first, second);
    assert_eq!(first.unwrap(), second.unwrap());
    assert_eq!(controller.type_cache.read().await.error_model_types.len(), 1);
}

#[tokio::test]
async fn streaming_cache_budget_retires_evicted_types() {
    for (ratio, identity_hit) in [(None, false), (Some(1), false), (Some(1), true), (Some(5), true)] {
        let config = ratio.map_or_else(
            || serde_json::json!({}),
            |ratio| serde_json::json!({"cache_limit_ratio": ratio}),
        );
        let controller = JitController::new(config);
        let coordinator = CoordinatorClient::Local(DynCoordinator::Mock(MockCoordinator::new()));
        controller.start(coordinator.clone()).await;
        controller
            .load_library(jit::JitLibrary {
                gadget_types: (1..=2)
                    .map(|gtype| jit::JitGadgetType {
                        base: Some(bin::GadgetType {
                            gtype,
                            ..Default::default()
                        }),
                        ..Default::default()
                    })
                    .collect(),
                ..Default::default()
            })
            .await
            .unwrap();
        controller.automatic_collection.store(true, Ordering::Release);
        let limit = 2 * ratio.unwrap_or(DEFAULT_CACHE_LIMIT_RATIO);
        let mut models: Vec<(bin::CheckModelType, Arc<bin::ErrorModelType>, u64, u64)> = Vec::new();
        for index in 0..=limit {
            if index == limit {
                let (check, error, ctype, etype) = &mut models[0];
                assert_eq!(controller.get_or_load_ctype(check, &coordinator).await.unwrap(), *ctype);
                let error = if identity_hit {
                    Arc::clone(error)
                } else {
                    Arc::new(error.as_ref().clone())
                };
                assert_eq!(controller.get_or_load_etype(&error, &coordinator).await.unwrap(), *etype);
            }
            let mut check_model = bin::CheckModelType {
                gtype: 1,
                checks: vec![check_model_type::Check::default(); index],
                ..Default::default()
            };
            let error_model = Arc::new(bin::ErrorModelType {
                errors: vec![
                    error_model_type::Error {
                        probability: 0.1,
                        ..Default::default()
                    };
                    index
                ],
                ..Default::default()
            });
            let ctype = controller.get_or_load_ctype(&mut check_model, &coordinator).await.unwrap();
            let etype = controller.get_or_load_etype(&error_model, &coordinator).await.unwrap();
            assert_eq!(
                controller.get_or_load_ctype(&mut check_model, &coordinator).await.unwrap(),
                ctype
            );
            assert_eq!(controller.get_or_load_etype(&error_model, &coordinator).await.unwrap(), etype);
            models.push((check_model, error_model, ctype, etype));
            let cache = controller.type_cache.read().await;
            assert_eq!(cache.check_model_types.len(), (index + 1).min(limit));
            assert_eq!(cache.error_model_types.len(), (index + 1).min(limit));
            let evicted = controller.generated_types.lock().await;
            assert_eq!(evicted.ctype.len(), usize::from(index == limit));
            assert_eq!(evicted.etype.len(), usize::from(index == limit));
        }
        let cache = controller.type_cache.read().await;
        let evicted = controller.generated_types.lock().await;
        assert_eq!(evicted.ctype, vec![models[1].2]);
        assert_eq!(evicted.etype, vec![models[1].3]);
        for (index, (check, error, ctype, etype)) in models.iter().enumerate() {
            let retained = index != 1;
            assert_eq!(cache.get_check_model_type(check), retained.then_some(*ctype));
            assert_eq!(cache.get_error_model_type(error), retained.then_some(*etype));
            let alias_retained = retained && (index != 0 || identity_hit);
            assert_eq!(cache.get_shared_error_model_type(error), alias_retained.then_some(*etype));
        }
    }
}

#[tokio::test]
async fn cache_budget_is_independent_of_persistence_and_collection() {
    use crate::coordinator::WindowCoordinator;
    use crate::decoder::{DynDecoder, MockDecoder};

    for (persistent, forced_gap) in [(false, false), (true, false), (false, true), (true, true)] {
        let controller = JitController::new(serde_json::json!({"cache_limit_ratio": 1}));
        register_cache_budget_type(&controller).await;
        let coordinator = CoordinatorClient::Local(DynCoordinator::Window(Arc::new(WindowCoordinator::new(
            serde_json::json!({"persistent_decoder": persistent, "forced_gap": forced_gap}),
            DynDecoder::Mock(Arc::new(MockDecoder::new())),
        ))));
        controller.start(coordinator.clone()).await;
        for index in 0..2 {
            let mut check_model = bin::CheckModelType {
                checks: vec![check_model_type::Check::default(); index],
                ..Default::default()
            };
            let error_model = Arc::new(bin::ErrorModelType {
                errors: vec![
                    error_model_type::Error {
                        probability: 0.1,
                        ..Default::default()
                    };
                    index
                ],
                ..Default::default()
            });
            controller.get_or_load_ctype(&mut check_model, &coordinator).await.unwrap();
            controller.get_or_load_etype(&error_model, &coordinator).await.unwrap();
        }
        {
            let cache = controller.type_cache.read().await;
            assert_eq!(cache.check_model_types.len(), 1);
            assert_eq!(cache.error_model_types.len(), 1);
        }
        controller.collect_unused_generated_types().await.unwrap();
        let state = controller.memory_stats().await.unwrap();
        let retained_types = if forced_gap { 2 } else { 1 };
        assert_eq!(state.check_model_types, retained_types);
        assert_eq!(state.error_model_types, retained_types);
        controller.reset(crate::coordinator::ResetRequest::default()).await.unwrap();
        let cache = controller.type_cache.read().await;
        assert_eq!(cache.check_model_types.len(), 1);
        assert_eq!(cache.error_model_types.len(), 1);
    }
}

#[tokio::test]
async fn shared_error_identity_is_invalidated_by_eviction_and_clear() {
    let controller = JitController::new(serde_json::json!({"cache_limit_ratio": 1}));
    controller.automatic_collection.store(true, Ordering::Release);
    register_cache_budget_type(&controller).await;
    let coordinator = CoordinatorClient::Local(DynCoordinator::Mock(MockCoordinator::new()));
    let first = Arc::new(bin::ErrorModelType::default());
    let second = Arc::new(bin::ErrorModelType {
        ctype: 1,
        ..Default::default()
    });
    let first_id = controller.get_or_load_etype(&first, &coordinator).await.unwrap();
    controller.get_or_load_etype(&second, &coordinator).await.unwrap();
    assert!(
        controller
            .type_cache
            .read()
            .await
            .get_shared_error_model_type(&first)
            .is_none()
    );
    let new_id = controller.get_or_load_etype(&first, &coordinator).await.unwrap();
    assert_ne!(new_id, first_id);
    controller.clear_cache().await;
    assert!(
        controller
            .type_cache
            .read()
            .await
            .error_model_identities
            .lock()
            .unwrap()
            .is_empty()
    );
    assert_ne!(controller.get_or_load_etype(&first, &coordinator).await.unwrap(), new_id);
}

#[test]
fn shared_error_identity_cannot_alias_mutated_types_or_retain_their_payloads() {
    let mut cache = TypeCache::new();
    let mut original = Arc::new(bin::ErrorModelType::default());
    cache.insert_error_model_type(&original, 1);
    cache.remember_error_model_identity(&original, 1, 1);
    let allocation = Arc::downgrade(&original);
    assert_eq!(cache.get_shared_error_model_type(&original), Some(1));
    Arc::make_mut(&mut original).ctype = 2;
    assert_eq!(cache.get_shared_error_model_type(&original), None);
    assert_eq!(cache.get_error_model_type(&original), None);
    assert!(allocation.upgrade().is_none());
    cache.insert_error_model_type(&original, 2);
    cache.remember_error_model_identity(&original, 2, 1);
    let allocation = Arc::downgrade(&original);
    drop(original);
    assert!(allocation.upgrade().is_none());
    cache.clear();
    assert!(cache.error_model_identities.get_mut().unwrap().is_empty());
}

#[tokio::test]
async fn identity_lru_uses_the_type_budget_as_the_library_grows() {
    for ratio in [1, 5, DEFAULT_CACHE_LIMIT_RATIO] {
        let controller = JitController::new(serde_json::json!({"cache_limit_ratio": ratio}));
        let coordinator = CoordinatorClient::Local(DynCoordinator::Mock(MockCoordinator::new()));
        controller.start(coordinator.clone()).await;
        for gadget_types in 1..=2 {
            if gadget_types > 0 {
                controller
                    .load_library(jit::JitLibrary {
                        gadget_types: vec![jit::JitGadgetType {
                            base: Some(bin::GadgetType {
                                gtype: gadget_types,
                                ..Default::default()
                            }),
                            ..Default::default()
                        }],
                        ..Default::default()
                    })
                    .await
                    .unwrap();
            }
            let limit = cache_capacity(usize::try_from(gadget_types).unwrap(), ratio);
            let models: Vec<_> = (0..=limit).map(|_| Arc::new(bin::ErrorModelType::default())).collect();
            let etype = controller.get_or_load_etype(&models[0], &coordinator).await.unwrap();
            for model in &models[..limit] {
                assert_eq!(controller.get_or_load_etype(model, &coordinator).await.unwrap(), etype);
            }
            {
                let cache = controller.type_cache.read().await;
                let identities = cache.error_model_identities.lock().unwrap();
                assert_eq!(identities.cap().get(), limit);
                assert_eq!(identities.len(), limit);
            }
            // Refreshing an existing alias at capacity must preserve all other entries.
            controller
                .type_cache
                .write()
                .await
                .remember_error_model_identity(&models[0], etype, limit);
            {
                let cache = controller.type_cache.read().await;
                assert_eq!(cache.error_model_identities.lock().unwrap().len(), limit);
                for model in &models[..limit] {
                    assert_eq!(cache.get_shared_error_model_type(model), Some(etype));
                }
            }
            assert_eq!(controller.get_or_load_etype(&models[0], &coordinator).await.unwrap(), etype);
            assert_eq!(
                controller.get_or_load_etype(&models[limit], &coordinator).await.unwrap(),
                etype
            );
            {
                let cache = controller.type_cache.read().await;
                let evicted = usize::from(limit > 1);
                for (index, model) in models.iter().enumerate() {
                    assert_eq!(cache.get_shared_error_model_type(model), (index != evicted).then_some(etype));
                }
                assert_eq!(cache.error_model_identities.lock().unwrap().len(), limit);
                assert_eq!(cache.error_model_types.len(), 1);
            }
            let next_id = controller.next_etype.load(Ordering::Relaxed);
            for model in &models {
                assert_eq!(controller.get_or_load_etype(model, &coordinator).await.unwrap(), etype);
            }
            assert_eq!(
                controller.next_etype.load(Ordering::Relaxed),
                next_id,
                "alias misses must reuse the structural type"
            );
            assert!(
                controller.generated_types.lock().await.etype.is_empty(),
                "alias eviction must not retire a type"
            );
        }
    }
}

#[test]
fn replacing_a_structural_type_invalidates_shared_identity() {
    let mut cache = TypeCache::new();
    let original = Arc::new(bin::ErrorModelType::default());
    cache.insert_error_model_type(&original, 1);
    let alias = Arc::new(original.as_ref().clone());
    let other = Arc::new(bin::ErrorModelType {
        ctype: 9,
        ..Default::default()
    });
    cache.insert_error_model_type(&other, 9);
    cache.remember_error_model_identity(&original, 1, 3);
    cache.remember_error_model_identity(&alias, 1, 3);
    cache.remember_error_model_identity(&other, 9, 3);
    cache.insert_error_model_type(&original, 2);
    assert_eq!(cache.get_shared_error_model_type(&original), None);
    assert_eq!(cache.get_shared_error_model_type(&alias), None);
    assert_eq!(cache.get_shared_error_model_type(&other), Some(9));
    assert_eq!(cache.get_error_model_type(&original), Some(2));
    assert_eq!(cache.error_model_recency.get_mut().unwrap().len(), 2);
    cache.remember_error_model_identity(&original, 2, 3);
    cache.remember_error_model_identity(&alias, 2, 3);
    cache.get_shared_error_model_type(&other);
    assert_eq!(cache.pop_error_model_type(), Some(2));
    assert!(cache.get_shared_error_model_type(&original).is_none());
    assert!(cache.get_shared_error_model_type(&alias).is_none());
    assert_eq!(cache.get_shared_error_model_type(&other), Some(9));
    assert_eq!(cache.error_model_types.len(), 1);
    assert_eq!(cache.error_model_identities.get_mut().unwrap().len(), 1);
}

#[test]
fn replacing_an_id_keeps_structural_and_lru_indices_consistent() {
    let mut cache = TypeCache::new();
    let check = bin::CheckModelType::default();
    let other_check = bin::CheckModelType {
        gtype: 1,
        ..Default::default()
    };
    cache.insert_check_model_type(&check, 1);
    cache.insert_check_model_type(&check, 2);
    cache.insert_check_model_type(&other_check, 2);
    assert_eq!(cache.get_check_model_type(&check), None);
    assert_eq!(cache.get_check_model_type(&other_check), Some(2));
    assert_eq!(cache.pop_check_model_type(), Some(2));
    assert!(cache.check_model_types.is_empty());
    assert_eq!(cache.pop_check_model_type(), None);

    let error = Arc::new(bin::ErrorModelType::default());
    let other_error = bin::ErrorModelType {
        ctype: 1,
        ..Default::default()
    };
    cache.insert_error_model_type(&error, 1);
    cache.remember_error_model_identity(&error, 1, 1);
    cache.insert_error_model_type(&other_error, 1);
    assert_eq!(cache.get_error_model_type(&error), None);
    assert_eq!(cache.get_shared_error_model_type(&error), None);
    assert_eq!(cache.get_error_model_type(&other_error), Some(1));
    cache.clear();
    assert_eq!(cache.pop_error_model_type(), None);
}

#[tokio::test]
async fn cancelled_batch_caller_keeps_spawned_compilations_protected() {
    let controller = JitController::new_from_library(jit::JitLibrary {
        gadget_types: vec![jit::JitGadgetType {
            base: Some(bin::GadgetType {
                gtype: 1,
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    });
    let mock = MockCoordinator::new();
    controller
        .start(CoordinatorClient::Local(DynCoordinator::Mock(Arc::clone(&mock))))
        .await;
    let state = mock.state.write().await;
    let mut batch = Box::pin(controller.batch_execute(vec![jit::JitInstruction {
        gadget: Some(bin::Gadget {
            gid: 1,
            gtype: 1,
            ..Default::default()
        }),
        ..Default::default()
    }]));
    assert!(batch.as_mut().now_or_never().is_none());
    tokio::task::yield_now().await;
    drop(batch);
    assert!(
        controller.collection.try_write().is_err(),
        "collection must not unload a type while a detached compilation is publishing it",
    );
    drop(state);
    tokio::time::timeout(std::time::Duration::from_secs(5), controller.task_counter.wait_for_zero())
        .await
        .unwrap();
    assert!(controller.collection.try_write().is_ok());
}
