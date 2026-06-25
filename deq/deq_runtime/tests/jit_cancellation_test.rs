//! Reset/cancellation tests that exercise the generic `JitController` only.

use deq_runtime::bin;
use deq_runtime::controller::jit_controller::JitController;
use deq_runtime::coordinator::{CoordinatorClient, MockCoordinator, ResetRequest};
use deq_runtime::jit;
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
    let (controller, _mock) = setup_jit(test_jit_library()).await;

    // Start a batch execute with dependencies
    let ctrl2 = Arc::clone(&controller);
    let exec_handle = tokio::spawn(async move {
        let instructions = vec![
            jit::JitInstruction {
                gadget: Some(bin::Gadget {
                    gid: 1,
                    gtype: 1,
                    connectors: vec![],
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
        ];
        let _ = ctrl2.batch_execute(instructions).await;
    });

    tokio::time::sleep(Duration::from_millis(10)).await;
    tokio::time::timeout(Duration::from_secs(5), controller.reset(reset_flags()))
        .await
        .expect("reset should not hang during batch_execute")
        .expect("reset should not error");

    tokio::time::timeout(Duration::from_secs(2), exec_handle)
        .await
        .expect("batch_execute should finish after reset")
        .expect("batch_execute should not panic");
}
