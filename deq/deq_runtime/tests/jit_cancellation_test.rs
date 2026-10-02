//! Reset/cancellation tests that exercise the generic `JitController` only.

use deq_runtime::bin;
use deq_runtime::controller::jit_controller::JitController;
use deq_runtime::coordinator::{CoordinatorClient, MockCoordinator, Outcomes, ResetRequest};
use deq_runtime::jit;
use deq_runtime::util::BitVector;
use std::sync::Arc;
use std::time::Duration;

mod common;
use common::test_library::test_jit_library;

fn reset_flags() -> ResetRequest {
    ResetRequest {
        reset_library: true,
        ..Default::default()
    }
}

async fn setup_jit(library: jit::JitLibrary) -> (Arc<JitController>, Arc<MockCoordinator>) {
    let mock = MockCoordinator::new();
    let client = CoordinatorClient::from_mock(mock.clone());
    let controller = JitController::new_from_library(library, false);
    controller.start(client).await;
    (controller, mock)
}

fn chain_instructions() -> Vec<jit::JitInstruction> {
    vec![
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gid: 1,
                gtype: 1,
                ..Default::default()
            }),
            ..Default::default()
        },
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gid: 2,
                gtype: 2,
                connectors: vec![bin::gadget::Connector { gid: 1, port: 0 }],
                ..Default::default()
            }),
            ..Default::default()
        },
    ]
}

fn chain_outcomes() -> Vec<Outcomes> {
    [(1, 2), (2, 3)]
        .into_iter()
        .map(|(gid, size)| Outcomes {
            gid,
            outcomes: Some(BitVector { size, data: vec![0] }),
            ..Default::default()
        })
        .collect()
}

async fn assert_empty_shot(controller: &JitController, mock: &MockCoordinator) {
    assert!(controller.compiler.gadgets.read().await.is_empty());
    let state = mock.state.read().await;
    assert!(state.instructions.is_empty(), "old batch instructions must not survive reset");
    assert!(state.gadgets.is_empty(), "old batch gadgets must not survive reset");
    assert!(state.check_models.is_empty(), "old batch checks must not survive reset");
    assert!(state.error_models.is_empty(), "old batch errors must not survive reset");
}

#[tokio::test]
async fn test_jit_reset_gid_sequence() {
    let (controller, _mock) = setup_jit(test_jit_library()).await;

    for _round in 0..10 {
        // Execute prepare_z (gtype 1) then measure_z (gtype 2), expect gids 1,2
        let gid1 = controller
            .execute(jit::JitInstruction {
                gadget: Some(bin::Gadget {
                    gid: 0,
                    gtype: 1,
                    ..Default::default()
                }),
                ..Default::default()
            })
            .await;
        let gid2 = controller
            .execute(jit::JitInstruction {
                gadget: Some(bin::Gadget {
                    gid: 0,
                    gtype: 2,
                    connectors: vec![bin::gadget::Connector { gid: gid1, port: 0 }],
                    ..Default::default()
                }),
                ..Default::default()
            })
            .await;

        assert_eq!(gid1, 1, "round {_round}: first gid should be 1");
        assert_eq!(gid2, 2, "round {_round}: second gid should be 2");

        controller.reset(reset_flags()).await.unwrap();
    }
}

#[tokio::test]
async fn test_reset_during_batch_execute() {
    for abort_caller in [false, true] {
        let (controller, mock) = setup_jit(test_jit_library()).await;
        let execute_blocker = mock.block_next_execute();
        let batch_controller = Arc::clone(&controller);
        let mut exec_handle = tokio::spawn(async move { batch_controller.batch_execute(chain_instructions()).await });

        execute_blocker.wait_until_started().await;
        if abort_caller {
            exec_handle.abort();
            assert!((&mut exec_handle).await.unwrap_err().is_cancelled());
        }
        tokio::time::timeout(Duration::from_secs(30), controller.reset(reset_flags()))
            .await
            .expect("reset should not hang during batch_execute")
            .expect("reset should not error");

        execute_blocker.release();
        if !abort_caller {
            tokio::time::timeout(Duration::from_secs(30), exec_handle)
                .await
                .expect("batch_execute should finish after reset")
                .expect("batch_execute should not panic")
                .unwrap();
        }
        assert_empty_shot(&controller, &mock).await;
        assert_eq!(Arc::strong_count(&controller), 1, "reset must drain detached batch children");
    }
}

#[tokio::test]
async fn reset_drains_queued_execution_after_batch_caller_is_dropped() {
    for reset_library in [false, true] {
        let (controller, mock) = setup_jit(test_jit_library()).await;
        let mut batch = Box::pin(controller.batch_execute(chain_instructions()));
        assert!(futures_util::poll!(&mut batch).is_pending());
        drop(batch);

        tokio::time::timeout(
            Duration::from_secs(30),
            controller.reset(ResetRequest {
                reset_library,
                ..Default::default()
            }),
        )
        .await
        .unwrap()
        .unwrap();
        assert_empty_shot(&controller, &mock).await;
        assert_eq!(Arc::strong_count(&controller), 1, "reset must drain unpolled execution tasks");

        assert_eq!(controller.batch_execute(chain_instructions()).await.unwrap(), vec![1, 2]);
        let readouts = controller.batch_decode(chain_outcomes()).await.unwrap();
        assert_eq!(readouts.iter().map(|readout| readout.gid).collect::<Vec<_>>(), vec![1, 2]);
    }
}

#[tokio::test]
async fn reset_drains_queued_decodes_before_gadget_ids_are_reused() {
    for reset_library in [false, true] {
        let (controller, mock) = setup_jit(test_jit_library()).await;
        controller.batch_execute(chain_instructions()).await.unwrap();
        tokio::time::timeout(Duration::from_secs(30), mock.wait_for_error_models(2))
            .await
            .unwrap();
        let mut batch = Box::pin(controller.batch_decode(chain_outcomes()));
        assert!(futures_util::poll!(&mut batch).is_pending());
        drop(batch);

        tokio::time::timeout(
            Duration::from_secs(30),
            controller.reset(ResetRequest {
                reset_library,
                ..Default::default()
            }),
        )
        .await
        .unwrap()
        .unwrap();
        assert_empty_shot(&controller, &mock).await;
        assert_eq!(Arc::strong_count(&controller), 1, "reset must drain unpolled decode tasks");

        controller.batch_execute(chain_instructions()).await.unwrap();
        let readouts = controller.batch_decode(chain_outcomes()).await.unwrap();
        assert_eq!(readouts.iter().map(|readout| readout.gid).collect::<Vec<_>>(), vec![1, 2]);
    }
}

#[tokio::test]
async fn reset_drains_single_execute_before_admitting_a_new_batch() {
    let (controller, mock) = setup_jit(test_jit_library()).await;
    let execute_blocker = mock.block_next_execute();
    let mut executing = Box::pin(controller.execute(chain_instructions().remove(0)));
    assert!(futures_util::poll!(&mut executing).is_pending());
    execute_blocker.wait_until_started().await;

    let mut resetting = Box::pin(controller.reset(reset_flags()));
    assert!(futures_util::poll!(&mut resetting).is_pending());
    let mut next_batch = Box::pin(controller.batch_execute(chain_instructions()));
    assert!(futures_util::poll!(&mut next_batch).is_pending());

    execute_blocker.release();
    let (old_gid, reset_result, new_gids) = tokio::time::timeout(Duration::from_secs(30), async {
        tokio::join!(executing, resetting, next_batch)
    })
    .await
    .expect("reset and new admissions must not deadlock");
    assert_eq!(old_gid, 1);
    reset_result.unwrap();
    assert_eq!(new_gids.unwrap(), vec![1, 2]);
    let readouts = controller.batch_decode(chain_outcomes()).await.unwrap();
    assert_eq!(readouts.iter().map(|readout| readout.gid).collect::<Vec<_>>(), vec![1, 2]);
    assert_eq!(controller.compiler.gadgets.read().await.len(), 2);
    assert_eq!(mock.state.read().await.gadgets.len(), 2);
}
