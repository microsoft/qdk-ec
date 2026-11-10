//! Automatic collection must preserve the live frontier without resets.

#[path = "../common/test_library.rs"]
mod test_library;

use crate as runtime;
use crate::bin;
use crate::controller::jit_controller::JitController;
use crate::coordinator::{CoordinatorClient, DynCoordinator, Outcomes, WindowCoordinator};
use crate::decoder::{DynDecoder, RelayBPDecoder};
use crate::jit;
use crate::util::BitVector;
use futures_util::FutureExt;
use std::sync::Arc;
use std::time::Duration;
use test_library::test_jit_library;
use tokio_util::sync::CancellationToken;

fn instruction(gid: u64, gtype: u64) -> jit::JitInstruction {
    jit::JitInstruction {
        gadget: Some(bin::Gadget {
            gid,
            gtype,
            connectors: if gid == 1 {
                vec![]
            } else {
                vec![bin::gadget::Connector { gid: gid - 1, port: 0 }]
            },
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[tokio::test]
async fn cancelled_compilation_does_not_consume_predecessor_checks() {
    let coordinator = Arc::new(WindowCoordinator::new(
        serde_json::json!({"buffer_radius": 0, "persistent_decoder": false}),
        DynDecoder::BlackBoxRelayBP(Arc::new(RelayBPDecoder::new(serde_json::json!({"parallel": 1})))),
    ));
    let controller = JitController::new_from_library(test_jit_library());
    controller
        .start(CoordinatorClient::Local(DynCoordinator::Window(coordinator)))
        .await;
    controller.try_execute(instruction(1, 1)).await.unwrap();
    let gadgets = controller.compiler.gadgets.read().await;
    let mut compilation = Box::pin(controller.compiler.compile(instruction(2, 4), CancellationToken::new()));
    assert!(compilation.as_mut().now_or_never().is_none());
    assert!(!gadgets.contains_key(&2));
    assert!(gadgets[&1].outputs[0].borrow().is_none());
    drop(compilation);
    drop(gadgets);
    controller.try_execute(instruction(2, 4)).await.unwrap();
    let gadgets = controller.compiler.gadgets.read().await;
    assert_eq!(gadgets[&1].outputs[0].borrow().as_ref().unwrap().gid, 2);
    drop(gadgets);
    controller.try_execute(instruction(3, 2)).await.unwrap();
    let outcomes = [(1, 2), (2, 0), (3, 3)]
        .into_iter()
        .map(|(gid, size)| Outcomes {
            gid,
            outcomes: Some(BitVector {
                size,
                data: if size == 0 { vec![] } else { vec![0] },
            }),
            ..Default::default()
        })
        .collect();
    let readouts = tokio::time::timeout(Duration::from_secs(5), controller.batch_decode(outcomes))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(readouts.iter().map(|readout| readout.gid).collect::<Vec<_>>(), vec![1, 2, 3]);
    controller.reset(Default::default()).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reset_releases_unfinished_execution_state_but_retains_cached_types() {
    let coordinator = Arc::new(WindowCoordinator::new(
        serde_json::json!({"buffer_radius": 0, "persistent_decoder": false}),
        DynDecoder::BlackBoxRelayBP(Arc::new(RelayBPDecoder::new(serde_json::json!({"parallel": 1})))),
    ));
    let library = test_jit_library();
    let terminal_types = library.gadget_types.len() as u64;
    let controller = JitController::new_from_library(library);
    controller
        .start(CoordinatorClient::Local(DynCoordinator::Window(coordinator)))
        .await;
    for attempt in 0..10 {
        controller.try_execute(instruction(1, 1)).await.unwrap();
        if attempt % 2 == 0 {
            controller
                .decode_single(Outcomes {
                    gid: 1,
                    outcomes: Some(BitVector { size: 2, data: vec![0] }),
                    ..Default::default()
                })
                .await
                .unwrap();
        }
        controller.reset(Default::default()).await.unwrap();
        let stats = controller.memory_stats().await.unwrap();
        assert_eq!(stats.gadgets, 0);
        assert_eq!(stats.jit_gadgets, 0);
        assert_eq!(stats.check_model_types, 1);
        // The cancelled background task may already have cached its error type.
        assert!((terminal_types..=terminal_types + 1).contains(&stats.error_model_types));
    }
    controller.clear_cache().await;
    controller.reset(Default::default()).await.unwrap();
    let stats = controller.memory_stats().await.unwrap();
    assert_eq!(stats.check_model_types, 0);
    assert_eq!(stats.error_model_types, terminal_types);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unresolved_future_checks_keep_their_old_measurements_live() {
    tokio::time::timeout(Duration::from_secs(10), async {
        let coordinator = Arc::new(WindowCoordinator::new(
            serde_json::json!({"buffer_radius": 0, "persistent_decoder": false}),
            DynDecoder::BlackBoxRelayBP(Arc::new(RelayBPDecoder::new(serde_json::json!({"parallel": 1})))),
        ));
        let controller = JitController::new_from_library(test_jit_library());
        controller
            .start(CoordinatorClient::Local(DynCoordinator::Window(coordinator)))
            .await;
        for gid in 1..=40 {
            controller
                .try_execute(instruction(gid, if gid == 1 { 1 } else { 4 }))
                .await
                .unwrap();
            controller
                .decode_single(Outcomes {
                    gid,
                    outcomes: Some(BitVector {
                        size: if gid == 1 { 2 } else { 0 },
                        data: if gid == 1 { vec![0] } else { vec![] },
                    }),
                    ..Default::default()
                })
                .await
                .unwrap();
            let state = controller.memory_stats().await.unwrap();
            assert_eq!(state.jit_gadgets, gid);
        }
        controller.try_execute(instruction(41, 2)).await.unwrap();
        controller
            .decode_single(Outcomes {
                gid: 41,
                outcomes: Some(BitVector { size: 3, data: vec![0] }),
                ..Default::default()
            })
            .await
            .unwrap();
        loop {
            if controller.memory_stats().await.unwrap().jit_gadgets == 0 {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("late check completion must remain possible during collection");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn alternating_gadgets_reuse_a_bounded_type_working_set() {
    let mut library = test_jit_library();
    let mut alternate = library
        .gadget_types
        .iter()
        .find(|gadget| gadget.base.as_ref().unwrap().gtype == 5)
        .unwrap()
        .clone();
    alternate.base.as_mut().unwrap().gtype = 6;
    alternate.base.as_mut().unwrap().name = "alternate_idle".to_string();
    alternate.errors[0].base.as_mut().unwrap().probability = 0.07;
    library.gadget_types.push(alternate);
    let type_limit = u64::try_from(library.gadget_types.len()).unwrap();
    let coordinator = Arc::new(WindowCoordinator::new(
        serde_json::json!({"buffer_radius": 0, "persistent_decoder": false}),
        DynDecoder::BlackBoxRelayBP(Arc::new(RelayBPDecoder::new(serde_json::json!({"parallel": 1})))),
    ));
    let controller = JitController::new_from_library(library);
    controller
        .start(CoordinatorClient::Local(DynCoordinator::Window(Arc::clone(&coordinator))))
        .await;
    let mut check_types = std::collections::HashMap::new();
    let mut error_types = std::collections::HashMap::new();
    tokio::time::timeout(Duration::from_secs(10), async {
        for gid in 1..=80 {
            let gtype = if gid == 1 { 1 } else { 5 + gid % 2 };
            controller.try_execute(instruction(gid, gtype)).await.unwrap();
            controller
                .decode_single(Outcomes {
                    gid,
                    outcomes: Some(BitVector { size: 2, data: vec![0] }),
                    ..Default::default()
                })
                .await
                .unwrap();
            if gid > 6 {
                let ctype = coordinator.check_models.read().await[&gid].instance.ctype;
                if let Some(previous) = check_types.insert(gtype, ctype) {
                    assert_eq!(ctype, previous, "alternating check types must remain cached");
                }
                let mut ready = coordinator.check_models.read().await[&(gid - 1)]
                    .error_model_ready
                    .subscribe();
                ready.wait_for(Option::is_some).await.unwrap();
                let etype = coordinator.error_models.read().await[&(gid - 1)].instance.etype;
                if let Some(previous) = error_types.insert(5 + (gid - 1) % 2, etype) {
                    assert_eq!(etype, previous, "alternating error types must remain cached");
                }
            }
            let state = controller.memory_stats().await.unwrap();
            assert!(state.check_model_types <= type_limit);
            assert!(state.error_model_types <= 2 * type_limit);
        }
        let before_reset = controller.memory_stats().await.unwrap();
        controller.reset(crate::coordinator::ResetRequest::default()).await.unwrap();
        let state = controller.memory_stats().await.unwrap();
        assert_eq!(state.check_model_types, before_reset.check_model_types);
        assert!(state.error_model_types <= 2 * type_limit);
        // Reset can finish a pending error-type publication, but must preserve cached IDs.
        let retained_error_types = coordinator.error_model_types.read().await;
        assert!(error_types.values().all(|id| retained_error_types.contains_key(id)));
    })
    .await
    .expect("type reuse must not prevent stream progress or reset");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn continuous_jit_history_is_bounded_and_releases_all_terminal_state() {
    tokio::time::timeout(Duration::from_secs(30), async {
        for (radius, persistent) in [(0, false), (1, false), (3, false), (1, true)] {
            let coordinator = Arc::new(WindowCoordinator::new(
                serde_json::json!({"buffer_radius": radius, "lookahead_radius": 0, "persistent_decoder": persistent}),
                DynDecoder::BlackBoxRelayBP(Arc::new(RelayBPDecoder::new(serde_json::json!({"parallel": 1})))),
            ));
            let controller = JitController::new_from_library(test_jit_library());
            controller
                .start(CoordinatorClient::Local(DynCoordinator::Window(coordinator)))
                .await;
            let mut pending = tokio::task::JoinSet::new();
            let mut completed = 0;
            let limit = 2 * radius + 8;
            // Up to 31 readout completions can accumulate before the next collection pass.
            let retained_limit = limit + 31;
            for gid in 1..=500 {
                let gtype = if gid == 1 {
                    1
                } else if gid == 500 {
                    2
                } else {
                    5
                };
                controller.try_execute(instruction(gid, gtype)).await.unwrap();
                let controller_clone = Arc::clone(&controller);
                pending.spawn(async move {
                    controller_clone
                        .decode_single(Outcomes {
                            gid,
                            outcomes: Some(BitVector {
                                size: if gid == 500 { 3 } else { 2 },
                                data: vec![0],
                            }),
                            ..Default::default()
                        })
                        .await
                });
                tokio::task::yield_now().await;
                while let Some(result) = pending.try_join_next() {
                    assert!(result.unwrap().unwrap().correction_count == 0);
                    completed += 1;
                }
                while completed + radius < gid {
                    pending.join_next().await.unwrap().unwrap().unwrap();
                    completed += 1;
                }
                let state = controller.memory_stats().await.unwrap();
                assert!(
                    state.gadgets <= retained_limit,
                    "unbounded coordinator history at gid={gid}: {state:?}"
                );
                assert!(
                    state.jit_gadgets <= limit,
                    "unbounded compiler history at gid={gid}: {state:?}"
                );
                assert!(state.check_model_types <= limit);
                if (20..500).contains(&gid) {
                    assert_eq!(
                        state.check_model_types, 2,
                        "preparation and repeated syndrome gadgets should each retain one cached type"
                    );
                    assert!(state.error_model_types <= 7, "streaming type cache must not grow: {state:?}");
                }
                assert!(
                    state.error_model_types <= 2 * limit + 5,
                    "unbounded error types at gid={gid}: {state:?}"
                );
                assert!(state.error_models <= retained_limit);
            }
            while let Some(result) = pending.join_next().await {
                result.unwrap().unwrap();
                completed += 1;
            }
            assert_eq!(completed, 500);
            let state = controller.memory_stats().await.unwrap();
            assert_eq!(state.gadgets, 0);
            assert_eq!(state.jit_gadgets, 0);
            assert_eq!(state.frame_gadgets, 0);
            assert_eq!(state.check_models, 0);
            assert_eq!(state.error_models, 0);
            assert!(state.collected_id_ranges <= 3);
            assert_eq!(
                controller.try_execute(instruction(1, 1)).await.unwrap_err().code(),
                tonic::Code::InvalidArgument,
            );
            assert!(controller.batch_execute(vec![instruction(1, 1)]).await.is_err());
        }
    })
    .await
    .expect("history collection must not deadlock the live frontier");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn asymmetric_streamed_branches_match_the_uncollected_reference() {
    use crate::decoder::NaiveDecoder;
    use crate::simulator::DeterministicRng;
    use rand::{RngExt, SeedableRng};

    for policy in ["sliding", "fully_parallel", "serial"] {
        for seed in [7, 19, 43] {
            let mut reference = None;
            for collect in [false, true] {
                let trace = tempfile::NamedTempFile::new().unwrap();
                let mut config = serde_json::json!({
                    "buffer_radius": 2, "lookahead_radius": 1, "persistent_decoder": false,
                    "window_parallelism": policy,
                });
                if !collect {
                    config["trace_filepath"] = serde_json::json!(trace.path());
                }
                let coordinator = Arc::new(WindowCoordinator::new(
                    config,
                    DynDecoder::BlackBoxNaive(Arc::new(NaiveDecoder::new(serde_json::json!({})))),
                ));
                let controller = JitController::new_from_library(test_jit_library());
                controller
                    .start(CoordinatorClient::Local(DynCoordinator::Window(coordinator)))
                    .await;
                let results = tokio::time::timeout(Duration::from_secs(20), async {
                    let mut rng = DeterministicRng::seed_from_u64(seed);
                    let mut program = vec![instruction(1, 1), instruction(2, 1)];
                    program[1].gadget.as_mut().unwrap().connectors.clear();
                    let mut tails = [(1, 0), (2, 0)];
                    let mut append = |gtype: u64, wires: &[usize]| {
                        let gid = program.len() as u64 + 1;
                        let connectors = wires
                            .iter()
                            .map(|&wire| bin::gadget::Connector {
                                gid: tails[wire].0,
                                port: tails[wire].1,
                            })
                            .collect();
                        program.push(jit::JitInstruction {
                            gadget: Some(bin::Gadget {
                                gid,
                                gtype,
                                connectors,
                                ..Default::default()
                            }),
                            ..Default::default()
                        });
                        for (port, &wire) in wires.iter().enumerate() {
                            tails[wire] = (gid, port as u64);
                        }
                    };
                    for round in 0..30 {
                        for wire in 0..2 {
                            for _ in 0..rng.random_range(0..8) {
                                append(4, &[wire]);
                            }
                        }
                        if round % 3 == 0 {
                            append(3, &[0, 1]);
                        }
                        append(5, &[0]);
                        if round % 4 == 0 {
                            append(5, &[1]);
                        }
                    }
                    append(2, &[0]);
                    append(2, &[1]);
                    let mut pending = tokio::task::JoinSet::new();
                    let mut results = Vec::new();
                    for instruction in program {
                        let gadget = instruction.gadget.as_ref().unwrap();
                        let gid = gadget.gid;
                        let count = match gadget.gtype {
                            1 | 5 => 2,
                            2 => 3,
                            _ => 0,
                        };
                        let data = if count == 0 {
                            vec![]
                        } else {
                            vec![if gid % 13 == 0 { 0x40 } else { 0 }]
                        };
                        controller.try_execute(instruction).await.unwrap();
                        let controller = Arc::clone(&controller);
                        pending.spawn(async move {
                            controller
                                .decode_single(Outcomes {
                                    gid,
                                    outcomes: Some(BitVector { size: count, data }),
                                    ..Default::default()
                                })
                                .await
                        });
                        tokio::task::yield_now().await;
                        while let Some(result) = pending.try_join_next() {
                            let readout = result.unwrap().unwrap();
                            results.push((readout.gid, readout.readouts, readout.syndrome_count));
                        }
                    }
                    while let Some(result) = pending.join_next().await {
                        let readout = result.unwrap().unwrap();
                        results.push((readout.gid, readout.readouts, readout.syndrome_count));
                    }
                    results.sort_by_key(|result| result.0);
                    results
                })
                .await
                .expect("asymmetric branches must keep making progress");
                if let Some(reference) = &reference {
                    assert_eq!(&results, reference, "policy={policy}, seed={seed}");
                    let state = controller.memory_stats().await.unwrap();
                    assert_eq!(state.gadgets, 0);
                    assert_eq!(state.frame_gadgets, 0);
                } else {
                    reference = Some(results);
                }
            }
        }
    }
}
