use super::*;
use crate::controller::static_controller::static_controller_server::StaticController as _;
use crate::coordinator::{DynCoordinator, MockCoordinator, WindowCoordinator};
use crate::decoder::{DynDecoder, MockDecoder};
use crate::misc::bit_matrix::zeros;
use prost::Message;

fn controller() -> Arc<StaticController> {
    let library = bin::Library {
        gadget_types: vec![bin::GadgetType {
            gtype: 1,
            measurements: vec![bin::gadget_type::Measurement::default()],
            correction_propagation: Some(zeros(0, 1)),
            readout_propagation: Some(zeros(0, 1)),
            logical_correction: Some(zeros(0, 0)),
            physical_correction: Some(zeros(0, 1)),
            ..Default::default()
        }],
        program: (1..=2)
            .map(|gid| bin::Instruction {
                create: Some(bin::instruction::Create::Gadget(bin::Gadget {
                    gid,
                    gtype: 1,
                    ..Default::default()
                })),
            })
            .collect(),
        ..Default::default()
    };
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), library.encode_to_vec()).unwrap();
    Arc::new(StaticController::new(serde_json::json!({"filepath": file.path()})))
}

fn batch() -> Request<coordinator::Outcomes> {
    Request::new(coordinator::Outcomes {
        outcomes: Some(BitVector { size: 1, data: vec![0] }),
        ..Default::default()
    })
}

#[tokio::test]
async fn malformed_batches_leave_input_and_seed_state_unchanged() {
    let controller = controller();
    controller
        .start(CoordinatorClient::Local(DynCoordinator::Mock(MockCoordinator::new())))
        .await;
    for (outcomes, loss_mask) in [
        (BitVector { size: 1, data: vec![] }, None),
        (
            BitVector { size: 1, data: vec![0] },
            Some(BitVector { size: 2, data: vec![0] }),
        ),
        (BitVector { size: 3, data: vec![0] }, None),
    ] {
        let error = controller
            .decode_gadgets(Request::new(coordinator::Outcomes {
                outcomes: Some(outcomes),
                loss_mask,
                decoder_seed: Some(42),
                ..Default::default()
            }))
            .await
            .unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
        let state = controller.state.lock().await;
        assert!(state.outcomes.is_empty());
        assert!(state.loss_mask.is_none());
        assert!(state.decoder_seed.is_none());
        assert_eq!(state.next_index, 0);
    }
    controller.decode_gadgets(batch()).await.unwrap();
    assert_eq!(
        controller.decode_gadgets(batch()).await.unwrap().into_inner().readouts.len(),
        2
    );
}

#[tokio::test]
async fn reset_drains_unpolled_decodes_before_reinstantiating_the_program() {
    struct Dropped(Option<tokio::sync::oneshot::Sender<()>>);
    impl Drop for Dropped {
        fn drop(&mut self) {
            if let Some(sender) = self.0.take() {
                let _ = sender.send(());
            }
        }
    }
    let mock = MockCoordinator::new();
    let controller = controller();
    controller
        .start(CoordinatorClient::Local(DynCoordinator::Mock(Arc::clone(&mock))))
        .await;
    let (sender, mut dropped) = tokio::sync::oneshot::channel();
    let guard = Dropped(Some(sender));
    controller.state.lock().await.pending_decodes.spawn(async move {
        let _guard = guard;
        std::future::pending::<Result<(usize, coordinator::Readouts), Status>>().await
    });
    let blocker = mock.block_next_execute();
    let resetting = tokio::spawn({
        let controller = Arc::clone(&controller);
        async move { controller.reset(Request::new(())).await }
    });
    tokio::time::timeout(std::time::Duration::from_secs(2), blocker.wait_until_started())
        .await
        .unwrap();
    assert!(controller.state.lock().await.pending_decodes.is_empty());
    dropped.try_recv().unwrap();
    blocker.release();
    tokio::time::timeout(std::time::Duration::from_secs(2), resetting)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn immediately_ready_decodes_do_not_need_a_background_task() {
    let controller = controller();
    controller
        .start(CoordinatorClient::Local(DynCoordinator::Mock(MockCoordinator::new())))
        .await;
    assert!(
        controller
            .decode_gadgets(batch())
            .await
            .unwrap()
            .into_inner()
            .readouts
            .is_empty()
    );
    {
        let state = controller.state.lock().await;
        assert!(state.pending_decodes.is_empty());
        assert!(state.pending_readouts[0].is_some());
    }
    let readouts = controller.decode_gadgets(batch()).await.unwrap().into_inner().readouts;
    assert_eq!(readouts.iter().map(|readout| readout.gid).collect::<Vec<_>>(), vec![1, 2]);
    controller.reset(Request::new(())).await.unwrap();
    assert_eq!(controller.state.lock().await.next_index, 0);
}

#[tokio::test]
async fn a_pending_first_poll_does_not_block_partial_input_delivery() {
    let window = Arc::new(WindowCoordinator::new(
        serde_json::json!({"buffer_radius": 0}),
        DynDecoder::Mock(Arc::new(MockDecoder::new())),
    ));
    let controller = controller();
    controller
        .start(CoordinatorClient::Local(DynCoordinator::Window(Arc::clone(&window))))
        .await;
    let gadgets = window.gadgets.write().await;
    {
        let partial = controller.decode_gadgets(batch());
        tokio::pin!(partial);
        let std::task::Poll::Ready(result) = futures_util::poll!(&mut partial) else {
            panic!("partial input delivery must not await an unavailable coordinator");
        };
        assert!(result.unwrap().into_inner().readouts.is_empty());
    }
    assert_eq!(controller.state.lock().await.pending_decodes.len(), 1);
    drop(gadgets);
    let readouts = tokio::time::timeout(std::time::Duration::from_secs(2), controller.decode_gadgets(batch()))
        .await
        .unwrap()
        .unwrap()
        .into_inner()
        .readouts;
    assert_eq!(readouts.iter().map(|readout| readout.gid).collect::<Vec<_>>(), vec![1, 2]);
    controller.reset(Request::new(())).await.unwrap();
}
