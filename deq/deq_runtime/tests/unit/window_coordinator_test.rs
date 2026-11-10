//! Unit tests for `WindowCoordinator` behavior and cache-key helpers.
//!
//! The cache-key tests cover `build_modifier_fingerprints` and
//! `committing_local_cids_sorted`, the two pieces of state that the
//! `WindowCoordinator` folds into the `DecoderCacheKey` beyond the
//! `RelativeProgram`. They pin down their behaviour so that:
//!
//!   - per-eid modifier changes (probability / `check_bias`) and
//!     per-etype structural changes change the fingerprint vector;
//!   - the commit-region vector is filtered to local cids and
//!     canonicalised by sorting (so equal sets map to equal vectors).
use super::collection::StatePins;
use super::decoding::{committing_local_cids_sorted, logical_flip_signature};
use super::error_metadata::{referenced_error_remotes, terminal_check_shape, terminal_shape_fits};
use super::scoring::{CausalGapSnapshot, RecordedGapEdge, original_scoring_baseline};
use super::*;
use crate::bin::error_model::ErrorModelModifier;
use crate::bin::error_model_type::{Error, RemoteCheckModel, remote_check_model};
use crate::coordinator::ErrorModelFingerprint;
use crate::coordinator::build_modifier_fingerprints;
use crate::coordinator::coordinator_server::Coordinator;
use crate::coordinator::forced_gap_handler::ForcedGapGraph;
use crate::coordinator::reweight_handler::prepare_decoder;
use crate::decoder::MockDecoder;
use crate::decoder::blackbox_decoder::{DecodingHypergraph, EdgeReweight, Hyperedge, ParityFactor};
use crate::decoder::blackbox_util::is_parity_factor;
#[cfg(feature = "mwpm")]
use crate::decoder::{DecoderType, blackbox_decoder};
use crate::misc::bit_vector::{self, flip_bit, get_bit};
use crate::misc::index::{ErrorIndex, WILDCARD};
use crate::misc::pauli_frame_symbolic_propagator::CorrectionBasis;
#[cfg(feature = "mwpm")]
use crate::misc::relative_program;
use crate::misc::relative_program::{RelativeMapping, RelativeProgram};
use binar::BitVec;
use prost::Message;
use std::sync::atomic::Ordering;
use tonic::{Request, Response, Status};

#[tokio::test]
async fn shot_resets_retain_history_only_when_tracing_is_enabled() {
    let output = tempfile::NamedTempFile::new().unwrap();
    for tracing in [false, true] {
        let config = if tracing {
            serde_json::json!({"trace_filepath": output.path()})
        } else {
            serde_json::json!({})
        };
        let coordinator = WindowCoordinator::new(config, DynDecoder::Mock(Arc::new(MockDecoder::new())));
        for _ in 0..16 {
            coordinator
                .reset(Request::new(coordinator::ResetRequest::default()))
                .await
                .unwrap();
        }
        assert_eq!(coordinator.trace.lock().await.shots.len(), if tracing { 16 } else { 0 });
        if tracing {
            let trace = trace::WindowCoordinatorTrace::decode(std::fs::read(output.path()).unwrap().as_slice()).unwrap();
            assert_eq!(trace.shots.len(), 16);
        }
    }
}

#[tokio::test]
async fn a_panicking_queued_decode_cancels_dependent_waiters() {
    let window = Arc::new(WindowCoordinator::new(
        serde_json::json!({}),
        DynDecoder::BlackBoxRelayBP(Arc::new(crate::decoder::RelayBPDecoder::new(
            serde_json::json!({"parallel": 1}),
        ))),
    ));
    window.gadget_types.write().await.insert(
        0,
        Arc::new(bin::GadgetType {
            measurements: vec![bin::gadget_type::Measurement::default()],
            readouts: vec![bin::gadget_type::Readout {
                measurement_indices: vec![8],
                ..Default::default()
            }],
            ..Default::default()
        }),
    );
    window
        .gadgets
        .write()
        .await
        .insert(1, history_gadget(1, GadgetState::default(), None));
    let token = window.cancellation.read().await.clone();
    let error = window
        .queue_decode(
            Request::new(coordinator::Outcomes {
                gid: 1,
                outcomes: Some(BitVector { size: 1, data: vec![0] }),
                ..Default::default()
            }),
            tokio::runtime::Handle::current(),
        )
        .unwrap()
        .await
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::Internal);
    assert!(token.is_cancelled(), "a contained panic must not strand other windows");
    window
        .reset(Request::new(coordinator::ResetRequest::default()))
        .await
        .unwrap();
    assert!(!window.cancellation.read().await.is_cancelled());
}

#[tokio::test]
async fn invalid_connectors_do_not_publish_partial_output_connections() {
    let coordinator = WindowCoordinator::new(serde_json::json!({}), DynDecoder::Mock(Arc::new(MockDecoder::new())));
    coordinator.gadget_types.write().await.insert(
        1,
        Arc::new(bin::GadgetType {
            gtype: 1,
            inputs: vec![bin::gadget_type::Port::default(); 2],
            ..Default::default()
        }),
    );
    for gid in 1..=2 {
        coordinator
            .gadgets
            .write()
            .await
            .insert(gid, history_gadget(gid, GadgetState::default(), None));
    }
    for connectors in [
        vec![bin::gadget::Connector { gid: 1, port: 0 }],
        vec![bin::gadget::Connector { gid: 1, port: 0 }; 2],
        vec![
            bin::gadget::Connector { gid: 1, port: 0 },
            bin::gadget::Connector { gid: 2, port: u64::MAX },
        ],
    ] {
        let error = coordinator
            .execute(Request::new(bin::Instruction {
                create: Some(bin::instruction::Create::Gadget(bin::Gadget {
                    gid: 3,
                    gtype: 1,
                    connectors,
                    ..Default::default()
                })),
            }))
            .await
            .unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
        let gadgets = coordinator.gadgets.read().await;
        assert!(!gadgets.contains_key(&3));
        assert!(gadgets[&1].outputs[0].borrow().is_none());
        assert!(gadgets[&2].outputs[0].borrow().is_none());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn local_decode_entry_points_queue_only_when_a_caller_slot_is_available() {
    let runtime = tokio::runtime::Handle::current();
    let (finished, result) = tokio::sync::oneshot::channel();
    let thread = std::thread::spawn(move || {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .use_current_thread()
                .build()
                .unwrap(),
        );
        let window = Arc::new(WindowCoordinator::new(
            serde_json::json!({}),
            DynDecoder::BlackBoxRelayBP(Arc::new(crate::decoder::RelayBPDecoder::with_thread_pool(
                serde_json::json!({}),
                Some(Arc::clone(&pool)),
            ))),
        ));
        let client = coordinator::CoordinatorClient::Local(coordinator::DynCoordinator::Window(Arc::clone(&window)));
        crate::misc::pool_executor::block_on(&pool, &runtime, async {
            let request = || coordinator::Outcomes {
                gid: 42,
                ..Default::default()
            };
            for use_client in [false, true] {
                let queued = async {
                    if use_client {
                        client.decode(request()).await
                    } else {
                        window.decode(Request::new(request())).await.map(Response::into_inner)
                    }
                };
                tokio::pin!(queued);
                assert!(futures_util::poll!(&mut queued).is_pending());
                assert!(!crate::misc::pool_executor::can_submit_inline(&pool));
                let overflow = client.decode(request());
                tokio::pin!(overflow);
                let std::task::Poll::Ready(result) = futures_util::poll!(&mut overflow) else {
                    panic!("overflow must use the ordinary service path rather than queue another task");
                };
                assert_eq!(result.unwrap_err().code(), tonic::Code::NotFound);
                assert_eq!(queued.await.unwrap_err().code(), tonic::Code::NotFound);
            }
        })
        .unwrap();
        finished.send(()).unwrap();
    });
    tokio::time::timeout(std::time::Duration::from_secs(2), result)
        .await
        .unwrap()
        .unwrap();
    thread.join().unwrap();
}

#[tokio::test]
async fn reset_waits_for_admitted_work_that_has_not_been_polled() {
    for cancel in [false, true] {
        let decoder = Arc::new(crate::decoder::RelayBPDecoder::new(serde_json::json!({"parallel": 1})));
        let (started, ready) = tokio::sync::oneshot::channel();
        let (release, waiting) = std::sync::mpsc::channel();
        decoder.thread_pool.spawn(move || {
            started.send(()).unwrap();
            waiting.recv().unwrap();
        });
        ready.await.unwrap();
        let window = Arc::new(WindowCoordinator::new(
            serde_json::json!({}),
            DynDecoder::BlackBoxRelayBP(decoder),
        ));
        let mut decoding = Box::pin(
            window
                .queue_decode(
                    Request::new(coordinator::Outcomes::default()),
                    tokio::runtime::Handle::current(),
                )
                .unwrap(),
        );
        assert!(futures_util::poll!(&mut decoding).is_pending());
        let reset = window.reset(Request::new(coordinator::ResetRequest::default()));
        tokio::pin!(reset);
        assert!(futures_util::poll!(&mut reset).is_pending());
        if cancel {
            drop(decoding);
            assert!(futures_util::poll!(&mut reset).is_pending());
            release.send(()).unwrap();
        } else {
            release.send(()).unwrap();
            let error = decoding.await.unwrap_err();
            assert_eq!(error.code(), tonic::Code::Unavailable);
        }
        tokio::time::timeout(std::time::Duration::from_secs(2), reset)
            .await
            .unwrap()
            .unwrap();
        assert!(window.gadgets.read().await.is_empty());
    }
}

#[cfg(feature = "mwpm")]
#[tokio::test]
async fn full_window_graph_preserves_boundaries_and_decoder_correction_indices() {
    for backend in [DecoderType::BlackBoxMwpm, DecoderType::BlackBoxUf] {
        let coordinator = WindowCoordinator::new(
            serde_json::json!({"persistent_decoder": false}),
            backend.create(serde_json::json!({"parallel": 1})),
        );
        let mut errors: Vec<_> = [vec![0, 1, 2], vec![0], vec![1]]
            .into_iter()
            .map(|indices| Error {
                checks: indices
                    .into_iter()
                    .map(|check_index| bin::error_model_type::RemoteCheck {
                        check_index,
                        remote_check_model: None,
                    })
                    .collect(),
                probability: 0.1,
                ..Default::default()
            })
            .collect();
        errors.push(Error {
            checks: [(None, 0), (Some(0), 0), (Some(0), 1)]
                .into_iter()
                .map(|(remote_check_model, check_index)| bin::error_model_type::RemoteCheck {
                    remote_check_model,
                    check_index,
                })
                .collect(),
            probability: 0.2,
            ..Default::default()
        });
        let model_type = Arc::new(bin::ErrorModelType {
            etype: 1,
            errors,
            ..Default::default()
        });
        coordinator.error_model_types.write().await.insert(1, model_type);
        let expanded = relative_program::ExpandedGadget {
            gid: 1,
            gtype: 1,
            inputs: vec![],
            outputs: vec![],
            check_model: Some(relative_program::ExpandedCheckModel {
                cid: 1,
                ctype: 1,
                remote_gadgets: vec![],
                count_checks: 4,
            }),
            error_models: vec![relative_program::ExpandedErrorModel {
                eid: 1,
                etype: 1,
                remote_check_models: vec![None],
            }],
        };
        let (program, mapping) = RelativeProgram::new(&[expanded]);
        let models = HashMap::from([(
            1,
            ErrorModel {
                instance: bin::ErrorModel {
                    eid: 1,
                    cid: 1,
                    etype: 1,
                    ..Default::default()
                },
                modified_remote_check_models: Arc::new(vec![Some(RemoteCheckModel {
                    absolute_cid: Some(99),
                    ..Default::default()
                })]),
            },
        )]);
        let committed = HashSet::new();
        let prepared = coordinator
            .prepare_window(&committed, &committed, &program, &mapping, &models, false)
            .await;
        assert_eq!(prepared.hypergraph.hyperedges.len(), 4);
        assert_eq!(
            prepared.errors.iter().map(|error| error.error_index).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(prepared.ignored_vertices, vec![3]);
        assert_eq!(prepared.hypergraph.vertex_num, 4);
        assert_eq!(prepared.hypergraph.hyperedges[0].vertices, vec![0, 1, 2]);
        assert_eq!(
            prepared.hypergraph.hyperedges[3].vertices,
            vec![0],
            "retain projected boundary"
        );
        let correction = coordinator
            .decoder
            .decode_shared(
                Arc::clone(&prepared.hypergraph),
                bit_vector::from_sparse_indices(4, &[1]),
                None,
                None,
            )
            .await
            .unwrap();
        assert_eq!(correction.subgraph, vec![2], "decoder must return original graph indices");
    }
}

#[test]
fn cached_error_remotes_are_unique_and_bound_to_the_type_instance() {
    let mut model_type = Arc::new(bin::ErrorModelType {
        etype: 1,
        remote_check_models: vec![RemoteCheckModel::default(); 4],
        errors: vec![Error {
            checks: [Some(2), None, Some(0), Some(2)]
                .into_iter()
                .map(|remote_check_model| bin::error_model_type::RemoteCheck {
                    remote_check_model,
                    check_index: 0,
                })
                .collect(),
            ..Default::default()
        }],
        ..Default::default()
    });
    let cached = CachedErrorRemotes::new(&model_type);
    assert_eq!(cached.indices_for_type(&model_type).unwrap().as_ref(), &[0, 2]);
    assert_eq!(
        cached.digest_for_type(&model_type),
        Some(crate::coordinator::decoder_cache_key::etype_digest(&model_type))
    );
    assert_eq!(cached.terminal_for_type(&model_type, 3), Some(false));
    let replacement = Arc::new(bin::ErrorModelType {
        etype: 1,
        errors: vec![Error {
            checks: vec![bin::error_model_type::RemoteCheck {
                remote_check_model: Some(1),
                check_index: 0,
            }],
            ..Default::default()
        }],
        ..Default::default()
    });
    assert!(cached.indices_for_type(&replacement).is_none());
    assert!(cached.digest_for_type(&replacement).is_none());
    assert!(cached.terminal_for_type(&replacement, 3).is_none());
    assert_eq!(referenced_error_remotes(&replacement).as_ref(), &[1]);
    let allocation = Arc::downgrade(&model_type);
    Arc::make_mut(&mut model_type).errors[0].probability = 0.02;
    assert!(cached.indices_for_type(&model_type).is_none());
    assert!(cached.digest_for_type(&model_type).is_none());
    assert!(cached.terminal_for_type(&model_type, 3).is_none());
    drop(model_type);
    assert!(
        allocation.upgrade().is_none(),
        "cached metadata must not retain the type payload"
    );
}

#[test]
fn terminal_type_metadata_preserves_local_and_reserved_remote_bounds() {
    let mut model_type = bin::ErrorModelType {
        remote_check_models: vec![bin::error_model_type::RemoteCheckModel {
            absolute_cid: Some(crate::misc::index::FUTURE_CHECK_CID),
            ..Default::default()
        }],
        errors: vec![Error {
            checks: vec![
                bin::error_model_type::RemoteCheck {
                    remote_check_model: None,
                    check_index: 2,
                },
                bin::error_model_type::RemoteCheck {
                    remote_check_model: Some(0),
                    check_index: 0,
                },
            ],
            ..Default::default()
        }],
        ..Default::default()
    };
    assert!(!terminal_shape_fits(terminal_check_shape(&model_type), 2));
    assert!(terminal_shape_fits(terminal_check_shape(&model_type), 3));
    model_type.errors[0].checks[1].check_index = 1;
    assert!(!terminal_shape_fits(terminal_check_shape(&model_type), 3));
    model_type.errors[0].checks[1].check_index = 0;
    model_type.errors[0].checks[1].remote_check_model = Some(1);
    assert!(!terminal_shape_fits(terminal_check_shape(&model_type), 3));
    model_type.errors[0].checks[1].remote_check_model = Some(0);
    model_type.remote_check_models[0].absolute_cid = Some(7);
    assert!(!terminal_shape_fits(terminal_check_shape(&model_type), 3));
}

#[tokio::test]
async fn prepared_window_keys_normalize_ids_but_preserve_bias_and_type_identity() {
    let coordinator = WindowCoordinator::new(serde_json::json!({}), DynDecoder::Mock(Arc::new(MockDecoder::new())));
    let mut model_type = bin::ErrorModelType {
        etype: 9,
        errors: vec![Error {
            probability: 0.1,
            ..Default::default()
        }],
        ..Default::default()
    };
    coordinator
        .error_model_types
        .write()
        .await
        .insert(9, Arc::new(model_type.clone()));
    let program = RelativeProgram {
        local_gadgets: vec![],
        count_checks: 1,
    };
    let mapping = RelativeMapping {
        global_eid_of: vec![20],
        local_cid_of: [(10, 0)].into_iter().collect(),
        ..Default::default()
    };
    let models = HashMap::from([(
        20,
        ErrorModel {
            instance: bin::ErrorModel {
                eid: 20,
                cid: 10,
                etype: 9,
                ..Default::default()
            },
            modified_remote_check_models: Arc::new(vec![Some(bin::error_model_type::RemoteCheckModel {
                absolute_cid: Some(30),
                check_bias: 2,
                ..Default::default()
            })]),
        },
    )]);
    let committed = HashSet::from([10]);
    let key = coordinator.prepared_window_key(&committed, &program, &mapping, &models).await;
    let mut changed = models.clone();
    Arc::make_mut(&mut changed.get_mut(&20).unwrap().modified_remote_check_models)[0]
        .as_mut()
        .unwrap()
        .absolute_cid = Some(40);
    assert_eq!(
        key,
        coordinator
            .prepared_window_key(&committed, &program, &mapping, &changed)
            .await
    );
    Arc::make_mut(&mut changed.get_mut(&20).unwrap().modified_remote_check_models)[0]
        .as_mut()
        .unwrap()
        .check_bias = 3;
    assert_ne!(
        key,
        coordinator
            .prepared_window_key(&committed, &program, &mapping, &changed)
            .await
    );
    model_type.errors[0].probability = 0.2;
    coordinator.error_model_types.write().await.insert(9, Arc::new(model_type));
    assert_ne!(
        key,
        coordinator.prepared_window_key(&committed, &program, &mapping, &models).await
    );
}

#[tokio::test]
async fn referenced_remote_cache_follows_type_unload_and_library_reset() {
    let coordinator = WindowCoordinator::new(serde_json::json!({}), DynDecoder::Mock(Arc::new(MockDecoder::new())));
    for etype in [9, 10] {
        coordinator
            .load_library(Request::new(bin::Library {
                error_model_types: vec![bin::ErrorModelType {
                    etype,
                    ..Default::default()
                }],
                ..Default::default()
            }))
            .await
            .unwrap();
    }
    assert_eq!(coordinator.referenced_error_remotes.read().await.len(), 2);
    coordinator
        .unload(Request::new(coordinator::UnloadLibrary {
            etype: vec![9],
            ..Default::default()
        }))
        .await
        .unwrap();
    assert!(!coordinator.referenced_error_remotes.read().await.contains_key(&9));
    coordinator
        .reset(Request::new(coordinator::ResetRequest {
            reset_library: true,
            ..Default::default()
        }))
        .await
        .unwrap();
    assert!(coordinator.referenced_error_remotes.read().await.is_empty());
}


async fn absolute_cid_fixture() -> (WindowCoordinator, Arc<MockDecoder>) {
    let decoder = Arc::new(MockDecoder::new());
    let coordinator = WindowCoordinator::new(serde_json::json!({}), DynDecoder::Mock(decoder.clone()));
    for gtype in [1, 2] {
        coordinator.gadget_types.write().await.insert(
            gtype,
            Arc::new(bin::GadgetType {
                gtype,
                measurements: if gtype == 2 {
                    vec![bin::gadget_type::Measurement::default()]
                } else {
                    vec![]
                },
                correction_propagation: Some(crate::util::BitMatrix {
                    cols: 1,
                    ..Default::default()
                }),
                readout_propagation: Some(crate::util::BitMatrix {
                    cols: 1,
                    ..Default::default()
                }),
                physical_correction: Some(crate::util::BitMatrix {
                    cols: u64::from(gtype == 2),
                    ..Default::default()
                }),
                ..Default::default()
            }),
        );
    }
    for ctype in [1, 2] {
        coordinator.check_model_types.write().await.insert(
            ctype,
            Arc::new(bin::CheckModelType {
                ctype,
                gtype: ctype,
                checks: if ctype == 2 {
                    vec![bin::check_model_type::Check {
                        measurements: vec![bin::check_model_type::RemoteMeasurement {
                            measurement_index: 0,
                            ..Default::default()
                        }],
                        ..Default::default()
                    }]
                } else {
                    vec![]
                },
                ..Default::default()
            }),
        );
    }
    coordinator.error_model_types.write().await.insert(
        9,
        Arc::new(bin::ErrorModelType {
            etype: 9,
            ctype: WILDCARD,
            remote_check_models: [21, 22, 21, 7, crate::misc::index::FUTURE_CHECK_CID]
                .into_iter()
                .map(|cid| RemoteCheckModel {
                    absolute_cid: Some(cid),
                    ..Default::default()
                })
                .collect(),
            errors: vec![Error {
                probability: 0.25,
                checks: vec![bin::error_model_type::RemoteCheck {
                    remote_check_model: Some(1),
                    check_index: 0,
                }],
                ..Default::default()
            }],
            ..Default::default()
        }),
    );
    absolute_cid_check(&coordinator, 7, 70).await;
    (coordinator, decoder)
}

async fn absolute_cid_check(coordinator: &WindowCoordinator, cid: u64, gid: u64) {
    let kind = if cid == 22 { 2 } else { 1 };
    coordinator
        .execute(Request::new(bin::Instruction {
            create: Some(bin::instruction::Create::Gadget(bin::Gadget {
                gid,
                gtype: kind,
                ..Default::default()
            })),
        }))
        .await
        .unwrap();
    coordinator
        .execute(Request::new(bin::Instruction {
            create: Some(bin::instruction::Create::CheckModel(bin::CheckModel {
                cid,
                gid,
                ctype: kind,
                error_model_count: Some(u64::from(cid == 7)),
                ..Default::default()
            })),
        }))
        .await
        .unwrap();
    coordinator.gadgets.read().await[&gid]
        .outcomes
        .send_replace(Some(if kind == 2 {
            bit_vector::from_sparse_indices(1, &[0])
        } else {
            BitVector::default()
        }));
    let mut syndrome = coordinator.check_models.read().await[&cid].syndrome.subscribe();
    let ready_syndrome = tokio::time::timeout(std::time::Duration::from_secs(5), syndrome.wait_for(Option::is_some))
        .await
        .unwrap()
        .unwrap();
    if kind == 2 {
        assert!(get_bit(ready_syndrome.as_ref().unwrap(), 0));
    }
}

async fn absolute_cid_error(coordinator: &WindowCoordinator) {
    coordinator
        .execute(Request::new(bin::Instruction {
            create: Some(bin::instruction::Create::ErrorModel(bin::ErrorModel {
                eid: 77,
                cid: 7,
                etype: 9,
                ..Default::default()
            })),
        }))
        .await
        .unwrap();
}

async fn complete_absolute_cid_source(coordinator: &WindowCoordinator) {
    let gadgets = coordinator.gadgets.read().await;
    let mut tracker = coordinator.pauli_frame_tracker.lock().await;
    tracker.load_raw(70, &[], &BitVector::default());
    for (gid, frame) in tracker.load_correction(70, BitVec::zeros(0), BitVec::zeros(0)) {
        gadgets[&gid].pauli_frame.send_replace(Some(frame));
    }
    tracker.finalize(70);
    gadgets[&70].state.send_modify(|state| state.committed = true);
    drop(tracker);
    drop(gadgets);
    coordinator.completed_readouts.lock().await.insert(70);
}

#[tokio::test]
async fn committed_absolute_cid_referrals_are_collected_before_the_target_arrives() {
    let (coordinator, _) = absolute_cid_fixture().await;
    absolute_cid_error(&coordinator).await;
    assert_eq!(coordinator.pending_referring_by_cid.lock().await.len(), 2);
    complete_absolute_cid_source(&coordinator).await;
    coordinator.collect_garbage().await.unwrap();
    assert!(coordinator.pending_referring_by_cid.lock().await.is_empty());
    assert!(coordinator.error_models.read().await.is_empty());
    for cid in [21, 22] {
        absolute_cid_check(&coordinator, cid, cid * 10).await;
        assert!(coordinator.check_models.read().await[&cid].referring_eids.is_empty());
    }
}

#[tokio::test]
async fn absolute_cid_referrals_existing_future_and_reset() {
    for future_order in [None, Some([21, 22]), Some([22, 21])] {
        let (coordinator, _) = absolute_cid_fixture().await;
        if future_order.is_none() {
            for cid in [21, 22] {
                absolute_cid_check(&coordinator, cid, cid * 10).await;
            }
        }
        absolute_cid_error(&coordinator).await;
        if let Some(order) = future_order {
            for cid in order {
                absolute_cid_check(&coordinator, cid, cid * 10).await;
            }
        }
        for cid in [21, 22] {
            assert_eq!(coordinator.check_models.read().await[&cid].referring_eids, vec![77]);
        }
        assert!(coordinator.check_models.read().await[&7].referring_eids.is_empty());
        assert!(coordinator.pending_referring_by_gid.lock().await.is_empty());
        assert!(coordinator.pending_referring_by_cid.lock().await.is_empty());
        assert!(coordinator.pending_referring_by_port.lock().await.is_empty());
    }
    let (coordinator, _) = absolute_cid_fixture().await;
    absolute_cid_error(&coordinator).await;
    coordinator
        .reset(Request::new(coordinator::ResetRequest::default()))
        .await
        .unwrap();
    assert!(coordinator.pending_referring_by_cid.lock().await.is_empty());
    absolute_cid_check(&coordinator, 22, 220).await;
    assert!(coordinator.check_models.read().await[&22].referring_eids.is_empty());
}

#[tokio::test]
async fn absolute_cid_referrals_preserve_uncommitted_boundary_error() {
    for owner_committed in [false, true] {
        let (coordinator, decoder) = absolute_cid_fixture().await;
        absolute_cid_error(&coordinator).await;
        absolute_cid_check(&coordinator, 22, 220).await;
        absolute_cid_check(&coordinator, 3, 30).await;
        for (gid, committed) in [(70, owner_committed), (220, true)] {
            coordinator.gadgets.read().await[&gid].state.send_replace(GadgetState {
                committed,
                reserved_by: None,
            });
        }
        decoder.state.write().await.decode_error = Some(Status::internal("mock captured model"));
        let error = coordinator
            .decode_and_commit(30, &HashSet::from([30]), &HashSet::from([3]), &HashSet::from([30, 220]))
            .await
            .expect_err("the mock records the actual request instead of solving it");
        assert!(error.message().contains("mock captured model"));
        let state = decoder.state.read().await;
        let request = state.decode_loaded_calls.last().unwrap();
        let graph = &state.loaded_hypergraphs[&request.hid];
        assert_eq!(graph.vertex_num, 1);
        assert_eq!(graph.hyperedges.len(), usize::from(!owner_committed));
        assert_eq!(
            request.syndrome,
            bit_vector::from_sparse_indices(1, if owner_committed { &[] } else { &[0] })
        );
        if !owner_committed {
            assert_eq!(graph.hyperedges[0].vertices, vec![0]);
            assert_eq!(graph.hyperedges[0].probability, 0.25);
        }
    }
}

#[tokio::test]
async fn absolute_cid_referrals_merge_with_port_referrals() {
    for target_ready in [false, true] {
        for absolute_first in [false, true] {
            let (coordinator, _) = absolute_cid_fixture().await;
            absolute_cid_check(&coordinator, 21, 210).await;
            coordinator.gadgets.write().await.get_mut(&70).unwrap().outputs =
                vec![watch::channel(Some(bin::gadget::Connector { gid: 210, port: 0 })).0];
            if !target_ready {
                coordinator.check_models.write().await.remove(&21);
                coordinator.gadgets.write().await.get_mut(&210).unwrap().binding_cid = None;
                coordinator.gadgets.read().await[&210].outcomes.send_replace(None);
            }
            let mut remotes = vec![
                RemoteCheckModel {
                    absolute_cid: Some(21),
                    ..Default::default()
                },
                RemoteCheckModel {
                    port: Some(remote_check_model::Port::Output(0)),
                    ..Default::default()
                },
            ];
            if !absolute_first {
                remotes.reverse();
            }
            Arc::make_mut(coordinator.error_model_types.write().await.get_mut(&9).unwrap()).remote_check_models = remotes;
            absolute_cid_error(&coordinator).await;
            if !target_ready {
                coordinator
                    .execute(Request::new(bin::Instruction {
                        create: Some(bin::instruction::Create::CheckModel(bin::CheckModel {
                            cid: 21,
                            gid: 210,
                            ctype: 1,
                            error_model_count: Some(0),
                            ..Default::default()
                        })),
                    }))
                    .await
                    .unwrap();
                coordinator.gadgets.read().await[&210]
                    .outcomes
                    .send_replace(Some(BitVector::default()));
            }
            assert_eq!(coordinator.check_models.read().await[&21].referring_eids, vec![77]);
            assert!(coordinator.pending_referring_by_gid.lock().await.is_empty());
            assert!(coordinator.pending_referring_by_cid.lock().await.is_empty());
            assert!(coordinator.pending_referring_by_port.lock().await.is_empty());
        }
    }
}

#[tokio::test]
async fn absolute_cid_referrals_follow_modified_remotes() {
    use crate::bin::error_model::error_model_modifier::RerouteRemoteCheckModel;

    for target_ready in [false, true] {
        let (coordinator, _) = absolute_cid_fixture().await;
        if target_ready {
            absolute_cid_check(&coordinator, 22, 220).await;
        }
        coordinator
            .execute(Request::new(bin::Instruction {
                create: Some(bin::instruction::Create::ErrorModel(bin::ErrorModel {
                    eid: 77,
                    cid: 7,
                    etype: 9,
                    modifier: Some(ErrorModelModifier {
                        reroute_remote_check_models: vec![
                            RerouteRemoteCheckModel {
                                remote_check_model_index: 0,
                                value: Some(RemoteCheckModel {
                                    absolute_cid: Some(22),
                                    ..Default::default()
                                }),
                            },
                            RerouteRemoteCheckModel {
                                remote_check_model_index: 2,
                                value: None,
                            },
                        ],
                        ..Default::default()
                    }),
                    ..Default::default()
                })),
            }))
            .await
            .unwrap();
        if !target_ready {
            absolute_cid_check(&coordinator, 22, 220).await;
        }
        absolute_cid_check(&coordinator, 21, 210).await;
        assert_eq!(coordinator.check_models.read().await[&22].referring_eids, vec![77]);
        assert!(coordinator.check_models.read().await[&21].referring_eids.is_empty());
        assert!(coordinator.pending_referring_by_cid.lock().await.is_empty());
    }
}

fn syndrome_free_hypergraph() -> DecodingHypergraph {
    DecodingHypergraph {
        vertex_num: 0,
        hyperedges: vec![Hyperedge {
            observable_flips: None,
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
                        None,
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
                observable_flips: None,
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
            let problem = graph.problem(DynDecoder::Mock(mock), BitVector::default(), None, baseline, vec![], true);
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
                    None,
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
            observable_flips: None,
            vertices: vec![0],
            probability: 0.0,
        }],
    };
    let mock = Arc::new(MockDecoder::new());
    let graph = Arc::new(ForcedGapGraph::new(Arc::new(hypergraph), Arc::new(vec![vec![0]]), 1, false));
    let problem = graph.problem(
        DynDecoder::Mock(Arc::clone(&mock)),
        crate::misc::bit_vector::from_sparse_indices(1, &[0]),
        None,
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
                        None,
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
                observable_flips: None,
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
                observable_flips: None,
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
                observable_flips: None,
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
                None,
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
                observable_flips: None,
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
            None,
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
async fn sliding_history_subscribes_only_at_the_blocked_frontier() {
    let coordinator = WindowCoordinator::new(serde_json::json!({}), DynDecoder::Mock(Arc::new(MockDecoder::new())));
    {
        let mut gadgets = coordinator.gadgets.write().await;
        for gid in 1..=1024 {
            let mut gadget = history_gadget(gid, GadgetState::default(), None);
            if gid > 1 {
                gadget
                    .instance
                    .connectors
                    .push(bin::gadget::Connector { gid: gid - 1, port: 0 });
            }
            gadgets.insert(gid, gadget);
        }
    }
    {
        let waiting = coordinator.wait_for_history_commitment(1024);
        tokio::pin!(waiting);
        assert!(futures_util::poll!(&mut waiting).is_pending());
        {
            let gadgets = coordinator.gadgets.try_write().expect("waiting must release the graph lock");
            assert_eq!(gadgets.values().map(|gadget| gadget.state.receiver_count()).sum::<usize>(), 1);
            assert_eq!(gadgets[&1023].state.receiver_count(), 1);
            gadgets[&1023].state.send_modify(|state| state.committed = true);
        }
        assert!(futures_util::poll!(&mut waiting).is_pending());
        {
            let gadgets = coordinator.gadgets.write().await;
            assert_eq!(gadgets[&1022].state.receiver_count(), 1);
            assert_eq!(gadgets.values().map(|gadget| gadget.state.receiver_count()).sum::<usize>(), 1);
            for gid in 1..=1022 {
                gadgets[&gid].state.send_modify(|state| state.committed = true);
                gadgets[&gid].pauli_frame.send_replace(Some(BitVector::default()));
            }
        }
        tokio::time::timeout(std::time::Duration::from_secs(1), waiting)
            .await
            .unwrap()
            .unwrap();
    }
    assert!(coordinator.pinned_gadgets.lock().unwrap().is_empty());
}

#[tokio::test]
async fn sliding_history_yields_the_graph_lock_during_long_free_hop_traversals() {
    let coordinator = WindowCoordinator::new(serde_json::json!({}), DynDecoder::Mock(Arc::new(MockDecoder::new())));
    {
        let mut gadgets = coordinator.gadgets.write().await;
        for gid in 1..=1024 {
            let mut gadget = history_gadget(gid, GadgetState::default(), None);
            gadget.is_free_hop = true;
            if gid > 1 {
                gadget
                    .instance
                    .connectors
                    .push(bin::gadget::Connector { gid: gid - 1, port: 0 });
            }
            gadgets.insert(gid, gadget);
        }
    }
    {
        let waiting = coordinator.wait_for_history_commitment(1024);
        tokio::pin!(waiting);
        assert!(futures_util::poll!(&mut waiting).is_pending());
        assert!(coordinator.gadgets.try_write().is_ok());
        assert!(!coordinator.pinned_gadgets.lock().unwrap().is_empty());
        coordinator.cancel_pending().await;
        assert_eq!(waiting.await.unwrap_err().code(), tonic::Code::Cancelled);
    }
    assert!(coordinator.pinned_gadgets.lock().unwrap().is_empty());
}

#[tokio::test]
async fn sliding_history_matches_transitive_dependencies_for_every_four_node_dag() {
    let coordinator = WindowCoordinator::new(serde_json::json!({}), DynDecoder::Mock(Arc::new(MockDecoder::new())));
    let possible_edges = [(1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (3, 4)];
    for edge_mask in 0_u8..64 {
        for free_mask in 0_u8..16 {
            for committed_mask in 0_u8..16 {
                let expected_ready = {
                    let mut gadgets = coordinator.gadgets.write().await;
                    gadgets.clear();
                    for gid in 1..=4 {
                        let mut gadget = history_gadget(
                            gid,
                            GadgetState {
                                committed: committed_mask & (1 << (gid - 1)) != 0,
                                reserved_by: None,
                            },
                            None,
                        );
                        gadget.is_free_hop = free_mask & (1 << (gid - 1)) != 0;
                        gadgets.insert(gid, gadget);
                    }
                    for (index, (source, target)) in possible_edges.into_iter().enumerate() {
                        if edge_mask & (1 << index) == 0 {
                            continue;
                        }
                        let gadget = gadgets.get_mut(&target).unwrap();
                        if index % 2 == 0 {
                            gadget
                                .instance
                                .connectors
                                .push(bin::gadget::Connector { gid: source, port: 0 });
                        } else {
                            gadget
                                .instance
                                .modifier
                                .get_or_insert_default()
                                .remote_conditional_correction
                                .get_or_insert_default()
                                .remote_readouts
                                .push(bin::remote_conditional_correction::RemoteReadout {
                                    gid: source,
                                    readout_index: 0,
                                });
                        }
                    }
                    let mut ancestors = HashSet::new();
                    let mut pending = vec![4];
                    while let Some(current) = pending.pop() {
                        for (index, (source, target)) in possible_edges.into_iter().enumerate() {
                            if target == current && edge_mask & (1 << index) != 0 && ancestors.insert(source) {
                                pending.push(source);
                            }
                        }
                    }
                    ancestors
                        .iter()
                        .all(|ancestor| gadgets[ancestor].is_free_hop || gadgets[ancestor].state.borrow().committed)
                };
                {
                    let waiting = tokio::task::unconstrained(coordinator.wait_for_history_commitment(4));
                    tokio::pin!(waiting);
                    let result = futures_util::poll!(&mut waiting);
                    assert_eq!(
                        result.is_ready(),
                        expected_ready,
                        "edges={edge_mask} free={free_mask} committed={committed_mask}"
                    );
                    if let std::task::Poll::Ready(result) = result {
                        result.unwrap();
                    }
                }
                assert!(coordinator.pinned_gadgets.lock().unwrap().is_empty());
            }
        }
    }
}

#[tokio::test]
async fn sliding_history_pins_remote_frontiers_across_collection() {
    let (coordinator, _) = absolute_cid_fixture().await;
    absolute_cid_error(&coordinator).await;
    coordinator.gadgets.write().await.get_mut(&70).unwrap().is_free_hop = false;
    let mut dependent = history_gadget(71, GadgetState::default(), None);
    dependent.instance.modifier = Some(bin::GadgetModifier {
        remote_conditional_correction: Some(bin::RemoteConditionalCorrection {
            remote_readouts: vec![bin::remote_conditional_correction::RemoteReadout {
                gid: 70,
                readout_index: 0,
            }],
            ..Default::default()
        }),
        ..Default::default()
    });
    coordinator.gadgets.write().await.insert(71, dependent);
    {
        let waiting = coordinator.wait_for_history_commitment(71);
        tokio::pin!(waiting);
        assert!(futures_util::poll!(&mut waiting).is_pending());
        complete_absolute_cid_source(&coordinator).await;
        coordinator.collect_garbage().await.unwrap();
        assert!(coordinator.gadgets.read().await.contains_key(&70));
        waiting.await.unwrap();
    }
    assert!(coordinator.pinned_gadgets.lock().unwrap().is_empty());
    coordinator.collect_garbage().await.unwrap();
    assert!(!coordinator.gadgets.read().await.contains_key(&70));
    coordinator.wait_for_history_commitment(71).await.unwrap();
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
                    decoder_seed: None,
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
                    observable_flips: None,
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
                    None,
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
                    observable_flips: None,
                    vertices: vec![0],
                    probability: 0.1,
                },
                Hyperedge {
                    observable_flips: None,
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
            None,
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
            .problem(decoder.clone(), syndrome.clone(), None, baseline.clone(), vec![], false)
            .probability(0)
            .await
            .unwrap();
        let commit_only = restricted
            .problem(decoder.clone(), &hypergraph, syndrome, None, &baseline, vec![], false)
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
        decoder_seed: None,
        binding_cid: Some(gid),
        outputs: vec![watch::channel(next_gid.map(|gid| bin::gadget::Connector { gid, port: 0 })).0],
        pauli_frame: watch::channel(None).0,
        correction_count: 0,
        correction_weight: 0.0,
        is_free_hop: false,
        state: watch::channel(state).0,
    }
}

fn collection_graph(edges: &[(u64, u64)], free: &[u64], roots: &[u64]) -> HashMap<u64, Gadget> {
    let mut gadgets = HashMap::new();
    for &(source, target) in edges {
        for gid in [source, target] {
            gadgets.entry(gid).or_insert_with(|| {
                let mut gadget = history_gadget(
                    gid,
                    GadgetState {
                        committed: !roots.contains(&gid),
                        reserved_by: None,
                    },
                    None,
                );
                gadget.outputs.clear();
                gadget.binding_cid = None;
                gadget.is_free_hop = free.contains(&gid);
                gadget
            });
        }
        let output = gadgets[&source].outputs.len() as u64;
        let input = gadgets[&target].instance.connectors.len() as u64;
        gadgets.get_mut(&source).unwrap().outputs.push(
            watch::channel(Some(bin::gadget::Connector {
                gid: target,
                port: input,
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
                port: output,
            });
    }
    gadgets
}

#[test]
fn committed_context_inside_a_window_is_not_an_artificial_boundary() {
    let coordinator = WindowCoordinator::new(
        serde_json::json!({"buffer_radius": 2, "lookahead_radius": 0}),
        DynDecoder::Mock(Arc::new(MockDecoder::new())),
    );
    let mut gadgets = collection_graph(&[(1, 2), (2, 3), (3, 4), (4, 5)], &[], &[2, 3, 4, 5]);
    for (&gid, gadget) in &mut gadgets {
        gadget.binding_cid = Some(gid);
    }
    let mut explored = ExploredWindow {
        center_gid: 2,
        gadgets: HashSet::from([1, 2, 3, 4]),
        center_distance: HashMap::from([(1, 1), (2, 0), (3, 1), (4, 2)]),
        phase: HashMap::new(),
        frontier: std::collections::VecDeque::new(),
        commit_region: HashSet::new(),
        committing_cids: HashSet::new(),
        decoder_window: HashSet::new(),
    };
    coordinator.select_commit_region(&mut explored, &gadgets);
    assert_eq!(explored.commit_region, HashSet::from([2]));
    assert_eq!(explored.committing_cids, HashSet::from([2]));
    coordinator.shrink_window(&mut explored, &gadgets);
    assert_eq!(explored.decoder_window, HashSet::from([1, 2, 3, 4]));
    assert!(gadgets[&1].state.borrow().committed);
}

#[tokio::test]
async fn lookahead_waits_for_free_hop_publication_and_syndrome_without_blocking() {
    use futures_util::FutureExt;

    let coordinator = WindowCoordinator::new(
        serde_json::json!({"buffer_radius": 1, "lookahead_radius": 1}),
        DynDecoder::Mock(Arc::new(MockDecoder::new())),
    );
    *coordinator.gadgets.write().await = collection_graph(&[(1, 2), (2, 3)], &[3], &[1, 2, 3]);
    for (outcomes_ready, syndrome_ready) in [(false, false), (true, false), (true, true)] {
        if outcomes_ready {
            coordinator.gadgets.write().await.get_mut(&3).unwrap().binding_cid = Some(3);
            coordinator.gadgets.read().await[&3]
                .outcomes
                .send_replace(Some(BitVector::default()));
            coordinator.check_models.write().await.insert(
                3,
                CheckModel {
                    instance: bin::CheckModel {
                        cid: 3,
                        gid: 3,
                        ..Default::default()
                    },
                    attaching_eid_vec: vec![],
                    error_model_ready: watch::channel(Some(())).0,
                    modified_remote_gadgets: Arc::new(vec![]),
                    expanded_remote_gadgets: Some(vec![]),
                    syndrome: watch::channel(syndrome_ready.then(BitVector::default)).0,
                    syndrome_count: 0,
                    referring_eids: vec![],
                },
            );
        }
        {
            let mut pins = StatePins::new(&coordinator.pinned_gadgets);
            let mut explored = coordinator.explore_mandatory_zone(1, &mut pins).await.unwrap();
            assert_eq!(explored.gadgets, HashSet::from([1, 2]));
            coordinator
                .explore_lookahead_zone(&mut explored, &mut pins)
                .now_or_never()
                .expect("lookahead must not wait for missing outcomes or syndrome")
                .unwrap();
            assert_eq!(explored.gadgets.contains(&3), syndrome_ready);
            assert_eq!(coordinator.pinned_gadgets.lock().unwrap().contains_key(&3), syndrome_ready);
        }
        assert!(coordinator.pinned_gadgets.lock().unwrap().is_empty());
    }
}

#[test]
fn collection_must_preserve_paths_that_leave_and_reenter_a_window() {
    let coordinator = WindowCoordinator::new(
        serde_json::json!({"buffer_radius": 1, "lookahead_radius": 0}),
        DynDecoder::Mock(Arc::new(MockDecoder::new())),
    );
    let edges: Vec<_> = (1..=12).map(|gid| (gid, gid + 1)).chain([(0, 1), (0, 13)]).collect();
    let mut gadgets = collection_graph(&edges, &[], &[1, 13]);
    let completed: HashSet<_> = gadgets.keys().copied().filter(|gid| ![1, 13].contains(gid)).collect();
    let window = HashSet::from([0, 1, 13]);
    let before = WindowCoordinator::terminal_boundary_gids(&gadgets, &window);
    assert!(
        !before.contains(&1),
        "the long branch connects the source back into the window"
    );
    let mut retained = coordinator.live_window_gadgets(&gadgets, &completed, &HashMap::new());
    WindowCoordinator::retain_pending_dependencies(&mut retained, &gadgets, &HashMap::new(), &HashMap::new()).unwrap();
    WindowCoordinator::retain_connecting_paths(&mut retained, &gadgets);
    gadgets.retain(|gid, _| retained.contains(gid));
    assert_eq!(
        WindowCoordinator::terminal_boundary_gids(&gadgets, &window),
        before,
        "collection must not change whether a full error model is required",
    );
}

#[tokio::test]
async fn cancelled_multi_step_remote_expansion_must_not_unwrap_a_missing_predecessor() {
    use futures_util::FutureExt;
    let mut gadgets = collection_graph(&[(1, 2)], &[], &[2]);
    gadgets.get_mut(&2).unwrap().outputs.push(watch::channel(None).0);
    let gadgets = RwLock::new(gadgets);
    let remotes = vec![
        Some(bin::check_model_type::RemoteGadget {
            port: Some(bin::check_model_type::remote_gadget::Port::Output(0)),
            ..Default::default()
        }),
        Some(bin::check_model_type::RemoteGadget {
            previous_remote_gadget: Some(0),
            port: Some(bin::check_model_type::remote_gadget::Port::Output(0)),
            ..Default::default()
        }),
    ];
    let token = CancellationToken::new();
    let check = bin::CheckModel {
        gid: 2,
        ..Default::default()
    };
    let mut waiting = Box::pin(WindowCoordinator::expand_remote_gadgets(
        &check,
        &remotes,
        &gadgets,
        token.clone(),
    ));
    assert!(waiting.as_mut().now_or_never().is_none());
    token.cancel();
    let result = tokio::time::timeout(std::time::Duration::from_secs(2), waiting)
        .await
        .unwrap();
    assert_eq!(result.unwrap_err().code(), tonic::Code::Cancelled);
}

#[tokio::test]
async fn invalid_remote_output_ports_are_reported_without_panicking() {
    let gadgets = RwLock::new(collection_graph(&[(1, 2)], &[], &[2]));
    let check = bin::CheckModel {
        gid: 1,
        ..Default::default()
    };
    for port in [99, u64::MAX] {
        let remotes = vec![Some(bin::check_model_type::RemoteGadget {
            port: Some(bin::check_model_type::remote_gadget::Port::Output(port)),
            ..Default::default()
        })];
        let error = WindowCoordinator::expand_remote_gadgets(&check, &remotes, &gadgets, CancellationToken::new())
            .await
            .unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
    }
}

#[test]
fn collection_path_closure_matches_all_five_vertex_dags() {
    let connections: Vec<_> = (1_u64..=5)
        .flat_map(|source| ((source + 1)..=5).map(move |target| (source, target)))
        .collect();
    for mask in 0_u16..1024 {
        let edges: Vec<_> = connections
            .iter()
            .enumerate()
            .filter_map(|(index, &edge)| (mask & (1 << index) != 0).then_some(edge))
            .collect();
        let mut gadgets = collection_graph(&edges, &[], &[]);
        for gid in 1..=5 {
            gadgets.entry(gid).or_insert_with(|| {
                let mut gadget = history_gadget(
                    gid,
                    GadgetState {
                        committed: true,
                        reserved_by: None,
                    },
                    None,
                );
                gadget.outputs.clear();
                gadget
            });
        }
        let mut reaches = [[false; 5]; 5];
        for &(source, target) in &edges {
            reaches[usize::try_from(source - 1).unwrap()][usize::try_from(target - 1).unwrap()] = true;
        }
        for via in 0..5 {
            for source in 0..5 {
                for target in 0..5 {
                    reaches[source][target] |= reaches[source][via] && reaches[via][target];
                }
            }
        }
        for root_mask in 1_u8..32 {
            let roots: HashSet<u64> = (1..=5).filter(|gid| root_mask & (1 << (gid - 1)) != 0).collect();
            let before = WindowCoordinator::terminal_boundary_gids(&gadgets, &roots);
            let mut retained = roots.clone();
            WindowCoordinator::retain_connecting_paths(&mut retained, &gadgets);
            for gid in 1..=5 {
                let index = usize::try_from(gid - 1).unwrap();
                let on_path = roots
                    .iter()
                    .any(|source| reaches[usize::try_from(*source - 1).unwrap()][index])
                    && roots
                        .iter()
                        .any(|target| reaches[index][usize::try_from(*target - 1).unwrap()]);
                assert_eq!(
                    retained.contains(&gid),
                    roots.contains(&gid) || on_path,
                    "edges={mask} roots={root_mask} gid={gid}"
                );
            }
            let filtered: HashMap<_, _> = gadgets
                .iter()
                .filter(|(gid, _)| retained.contains(*gid))
                .map(|(&gid, original)| {
                    let mut copy = history_gadget(gid, GadgetState::default(), None);
                    copy.instance = original.instance.clone();
                    copy.outputs.clone_from(&original.outputs);
                    (gid, copy)
                })
                .collect();
            assert_eq!(WindowCoordinator::terminal_boundary_gids(&filtered, &roots), before);
        }
    }
}

#[tokio::test]
async fn collection_keeps_error_types_owned_by_a_suspended_decoding_snapshot() {
    let (coordinator, _) = absolute_cid_fixture().await;
    absolute_cid_error(&coordinator).await;
    let mut pins = StatePins::new(&coordinator.pinned_gadgets);
    pins.retain([70]);
    let selected = coordinator.error_models.read().await[&77].clone();
    complete_absolute_cid_source(&coordinator).await;
    coordinator
        .pending_type_unloads
        .lock()
        .await
        .etype
        .push(selected.instance.etype);
    coordinator.collect_garbage().await.unwrap();
    assert!(
        coordinator
            .error_model_types
            .read()
            .await
            .contains_key(&selected.instance.etype),
        "an active snapshot still reads its error-model type after a competing window commits the owner",
    );
    drop(pins);
    coordinator.collect_garbage().await.unwrap();
    assert!(
        !coordinator
            .error_model_types
            .read()
            .await
            .contains_key(&selected.instance.etype)
    );
}

#[tokio::test]
async fn collection_preserves_overlapping_in_flight_pins_and_releases_cancelled_owners() {
    use crate::coordinator::coordinator_server::Coordinator;
    use futures_util::FutureExt;

    let coordinator = Arc::new(WindowCoordinator::new(
        serde_json::json!({"buffer_radius": 0, "persistent_decoder": false}),
        DynDecoder::Mock(Arc::new(MockDecoder::new())),
    ));
    coordinator.gadget_types.write().await.insert(0, empty_readout_gadget_type());
    coordinator
        .execute(Request::new(bin::Instruction {
            create: Some(bin::instruction::Create::Gadget(bin::Gadget {
                gid: 17,
                ..Default::default()
            })),
        }))
        .await
        .unwrap();
    let mut first = StatePins::new(&coordinator.pinned_gadgets);
    let mut second = StatePins::new(&coordinator.pinned_gadgets);
    {
        let gadgets = coordinator.gadgets.read().await;
        assert!(gadgets.contains_key(&17));
        first.retain([17, 17]);
        second.retain([17]);
    }
    let mut paused = Box::pin(async move {
        let _first = first;
        std::future::pending::<()>().await;
    });
    assert!(paused.as_mut().now_or_never().is_none());
    coordinator
        .decode(Request::new(coordinator::Outcomes {
            gid: 17,
            outcomes: Some(BitVector::default()),
            ..Default::default()
        }))
        .await
        .unwrap();
    assert!(coordinator.gadgets.read().await.contains_key(&17));
    assert_eq!(coordinator.pinned_gadgets.lock().unwrap()[&17], 2);
    drop(paused);
    coordinator.collect_garbage().await.unwrap();
    assert!(coordinator.gadgets.read().await.contains_key(&17));
    drop(second);
    coordinator.collect_garbage().await.unwrap();
    assert!(coordinator.gadgets.read().await.is_empty());
    assert!(coordinator.pauli_frame_tracker.lock().await.gadgets.is_empty());
    assert!(coordinator.pinned_gadgets.lock().unwrap().is_empty());
}

#[test]
fn collection_distances_are_origin_relative_not_symmetric() {
    let edges = [(1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 4), (4, 7), (7, 8)];
    for (buffer, lookahead) in [(0, 0), (1, 0), (1, 1), (3, 2)] {
        let coordinator = WindowCoordinator::new(
            serde_json::json!({"buffer_radius": buffer, "lookahead_radius": lookahead}),
            DynDecoder::Mock(Arc::new(MockDecoder::new())),
        );
        for free_mask in 0_u16..256 {
            let free: Vec<_> = (1..=8).filter(|gid| free_mask & (1 << (gid - 1)) != 0).collect();
            for root in 1..=8 {
                let gadgets = collection_graph(&edges, &free, &[root]);
                let completed = gadgets.keys().copied().filter(|gid| *gid != root).collect();
                let retained = coordinator.live_window_gadgets(&gadgets, &completed, &HashMap::new());
                let mut distance: HashMap<u64, usize> = HashMap::from([(root, 0_usize)]);
                for _ in 0..8 {
                    for &(source, target) in &edges {
                        for (from, to) in [(source, target), (target, source)] {
                            if let Some(&value) = distance.get(&from) {
                                let candidate = value + usize::from(!free.contains(&to));
                                if distance.get(&to).is_none_or(|old| candidate < *old) {
                                    distance.insert(to, candidate);
                                }
                            }
                        }
                    }
                }
                for (&gid, &hops) in &distance {
                    if hops <= buffer + lookahead {
                        assert!(retained.contains(&gid), "root={root} gid={gid} free_mask={free_mask}");
                        for &(source, target) in &edges {
                            if source == gid || target == gid {
                                let peer = if source == gid { target } else { source };
                                assert!(
                                    retained.contains(&peer),
                                    "unprotected boundary lookup from root={root} gid={gid} to={peer}"
                                );
                            }
                        }
                    }
                }
            }
        }
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

#[tokio::test]
async fn ready_syndromes_publish_once_and_do_not_overwrite_applied_corrections() {
    let coordinator = WindowCoordinator::new(serde_json::json!({}), DynDecoder::Mock(Arc::new(MockDecoder::new())));
    coordinator.check_model_types.write().await.insert(
        1,
        Arc::new(bin::CheckModelType {
            ctype: 1,
            checks: vec![bin::check_model_type::Check {
                naturally_flipped: true,
                measurements: vec![
                    bin::check_model_type::RemoteMeasurement {
                        measurement_index: 0,
                        ..Default::default()
                    },
                    bin::check_model_type::RemoteMeasurement {
                        remote_gadget: Some(0),
                        measurement_index: 0,
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        }),
    );
    for gid in 1..=2 {
        coordinator
            .gadgets
            .write()
            .await
            .insert(gid, history_gadget(gid, GadgetState::default(), None));
    }
    let mut check = history_check_model(1, vec![]);
    check.instance.ctype = 1;
    check.expanded_remote_gadgets = Some(vec![Some(2)]);
    check.modified_remote_gadgets = Arc::new(vec![Some(bin::check_model_type::RemoteGadget {
        absolute_gid: Some(2),
        measurement_bias: 1,
        ..Default::default()
    })]);
    coordinator.check_models.write().await.insert(1, check);
    for ready in 0..=2 {
        if ready == 1 {
            coordinator.gadgets.read().await[&1].outcomes.send_replace(Some(BitVector {
                size: 1,
                data: vec![0x80],
            }));
        } else if ready == 2 {
            coordinator.gadgets.read().await[&2].outcomes.send_replace(Some(BitVector {
                size: 2,
                data: vec![0x40],
            }));
        }
        assert_eq!(
            WindowCoordinator::try_publish_syndrome(
                1,
                &coordinator.gadgets,
                &coordinator.check_models,
                &coordinator.check_model_types,
                None,
            )
            .await
            .unwrap(),
            (ready == 2).then_some(1),
        );
    }
    {
        let checks = coordinator.check_models.read().await;
        assert_eq!(checks[&1].syndrome_count, 1);
        assert!(get_bit(checks[&1].syndrome.borrow().as_ref().unwrap(), 0));
        checks[&1].syndrome.send_replace(Some(BitVector { size: 1, data: vec![0] }));
    }
    assert!(
        WindowCoordinator::try_publish_syndrome(
            1,
            &coordinator.gadgets,
            &coordinator.check_models,
            &coordinator.check_model_types,
            None,
        )
        .await
        .unwrap()
        .is_none()
    );
    let checks = coordinator.check_models.read().await;
    assert_eq!(checks[&1].syndrome_count, 1);
    assert!(!get_bit(checks[&1].syndrome.borrow().as_ref().unwrap(), 0));
    let published = checks[&1].syndrome.clone();
    drop(checks);
    coordinator.check_models.write().await.remove(&1);
    assert!(
        WindowCoordinator::try_publish_syndrome(
            1,
            &coordinator.gadgets,
            &coordinator.check_models,
            &coordinator.check_model_types,
            Some(&published),
        )
        .await
        .unwrap()
        .is_none()
    );
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
            coordinator.gadgets.read().await[&1].outcomes.send_replace(None);
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
            let committed_syndrome = coordinator.check_models.read().await[&1].syndrome.subscribe();
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
            assert!(!get_bit(committed_syndrome.borrow().as_ref().unwrap(), 0));
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
                                observable_flips: None,
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
        coordinator.gadgets.read().await[&1].outcomes.send_replace(None);
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
