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

#[test]
fn zero_task_batch_size_is_rejected_before_loading_the_library() {
    let error = serde_json::from_value::<StaticControllerConfig>(
        serde_json::json!({"filepath": "not-read", "decode_task_batch_size": 0}),
    )
    .unwrap_err();
    assert!(error.to_string().contains("nonzero"));
}

#[tokio::test]
async fn explicit_boundaries_do_not_release_zero_measurement_gadgets_early() {
    let mut library = controller().library.clone();
    library.gadget_types[0].measurements.clear();
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), library.encode_to_vec()).unwrap();
    let controller = Arc::new(StaticController::new(serde_json::json!({"filepath": file.path()})));
    controller
        .start(CoordinatorClient::Local(DynCoordinator::Mock(MockCoordinator::new())))
        .await;
    let empty = || {
        Request::new(coordinator::Outcomes {
            outcomes: Some(BitVector::default()),
            ..Default::default()
        })
    };
    assert!(
        controller
            .decode_gadgets_through(empty(), Some(1))
            .await
            .unwrap()
            .into_inner()
            .readouts
            .is_empty()
    );
    assert_eq!(controller.state.lock().await.next_index, 1);
    let error = controller.decode_gadgets_through(empty(), Some(0)).await.unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    let readouts = controller
        .decode_gadgets_through(empty(), Some(2))
        .await
        .unwrap()
        .into_inner()
        .readouts;
    assert_eq!(readouts.len(), 2);
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
        assert_eq!(state.outcomes.size, 0);
        assert!(state.outcomes.data.is_empty());
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
        std::future::pending::<Result<Vec<(usize, coordinator::Readouts)>, Status>>().await
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

#[tokio::test]
async fn pending_gadgets_share_tasks_without_waiting_for_more_input() {
    for batch_size in [1, 16, DECODE_TASK_BATCH_SIZE] {
        check_pending_gadget_batches(batch_size).await;
    }
}

async fn check_pending_gadget_batches(batch_size: usize) {
    let mut library = controller().library.clone();
    let total = DECODE_TASK_BATCH_SIZE * 2 + 6;
    library.program = (1..=total)
        .map(|gid| bin::Instruction {
            create: Some(bin::instruction::Create::Gadget(bin::Gadget {
                gid: u64::try_from(gid).unwrap(),
                gtype: 1,
                ..Default::default()
            })),
        })
        .collect();
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), library.encode_to_vec()).unwrap();
    let controller = Arc::new(StaticController::new(
        serde_json::json!({"filepath": file.path(), "decode_task_batch_size": batch_size}),
    ));
    let window = Arc::new(WindowCoordinator::new(
        serde_json::json!({"buffer_radius": 0}),
        DynDecoder::Mock(Arc::new(MockDecoder::new())),
    ));
    controller
        .start(CoordinatorClient::Local(DynCoordinator::Window(Arc::clone(&window))))
        .await;
    let input = |size| {
        Request::new(coordinator::Outcomes {
            outcomes: Some(crate::misc::bit_vector::from_sparse_indices(size, &[])),
            ..Default::default()
        })
    };
    let gadgets = window.gadgets.write().await;
    let partial = u64::try_from(total - 5).unwrap();
    assert!(
        controller
            .decode_gadgets(input(partial))
            .await
            .unwrap()
            .into_inner()
            .readouts
            .is_empty()
    );
    assert_eq!(
        controller.state.lock().await.pending_decodes.len(),
        (total - 5).div_ceil(batch_size)
    );
    drop(gadgets);
    let completed = tokio::time::timeout(std::time::Duration::from_secs(2), controller.decode_gadgets(input(5)))
        .await
        .unwrap()
        .unwrap()
        .into_inner();
    assert_eq!(
        completed.readouts.iter().map(|readout| readout.gid).collect::<Vec<_>>(),
        (1..=u64::try_from(total).unwrap()).collect::<Vec<_>>()
    );
    controller.reset(Request::new(())).await.unwrap();
    assert!(controller.state.lock().await.pending_decodes.is_empty());
}

#[tokio::test]
async fn packed_batches_preserve_nonbyte_aligned_gadgets_and_loss_flags() {
    use crate::misc::bit_vector::{pack_bits, unpack_bits};

    let lengths = [3, 9, 5];
    let library = bin::Library {
        gadget_types: lengths
            .iter()
            .enumerate()
            .map(|(index, &length)| bin::GadgetType {
                gtype: u64::try_from(index + 1).unwrap(),
                measurements: vec![bin::gadget_type::Measurement::default(); length],
                readouts: (0..length)
                    .map(|index| bin::gadget_type::Readout {
                        measurement_indices: vec![u64::try_from(index).unwrap()],
                        ..Default::default()
                    })
                    .collect(),
                correction_propagation: Some(zeros(0, 1)),
                readout_propagation: Some(zeros(length, 1)),
                logical_correction: Some(zeros(0, length)),
                physical_correction: Some(zeros(0, length)),
                ..Default::default()
            })
            .collect(),
        program: (1..=3)
            .map(|gid| bin::Instruction {
                create: Some(bin::instruction::Create::Gadget(bin::Gadget {
                    gid,
                    gtype: gid,
                    ..Default::default()
                })),
            })
            .collect(),
        ..Default::default()
    };
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), library.encode_to_vec()).unwrap();
    let controller = Arc::new(StaticController::new(serde_json::json!({"filepath": file.path()})));
    let window = Arc::new(WindowCoordinator::new(
        serde_json::json!({"buffer_radius": 0, "automatic_collection": false, "loss_random_imputation": false}),
        DynDecoder::Mock(Arc::new(MockDecoder::new())),
    ));
    controller
        .start(CoordinatorClient::Local(DynCoordinator::Window(window)))
        .await;
    let outcomes: Vec<bool> = (0..17).map(|index| index % 5 < 2).collect();
    let losses: Vec<bool> = (0..17).map(|index| index % 3 == 1).collect();
    let mut start = 0;
    let mut readouts = Vec::new();
    for length in [2, 5, 4, 6] {
        let end = start + length;
        readouts = controller
            .decode_gadgets(Request::new(coordinator::Outcomes {
                outcomes: Some(BitVector {
                    size: u64::try_from(length).unwrap(),
                    data: pack_bits(&outcomes[start..end]),
                }),
                loss_mask: Some(BitVector {
                    size: u64::try_from(length).unwrap(),
                    data: pack_bits(&losses[start..end]),
                }),
                ..Default::default()
            }))
            .await
            .unwrap()
            .into_inner()
            .readouts;
        start = end;
    }
    assert_eq!(readouts.len(), lengths.len());
    start = 0;
    for (readout, length) in readouts.iter().zip(lengths) {
        let bits = readout.readouts.as_ref().unwrap();
        assert_eq!(unpack_bits(&bits.data, bits.size), outcomes[start..start + length]);
        start += length;
    }
    {
        let state = controller.state.lock().await;
        assert_eq!(state.outcomes.data, pack_bits(&outcomes));
        assert_eq!(state.loss_mask.as_ref().unwrap().data, pack_bits(&losses));
    }
    controller.reset(Request::new(())).await.unwrap();
    let state = controller.state.lock().await;
    assert_eq!(state.outcomes.size, 0);
    assert!(state.outcomes.data.is_empty());
    assert!(state.loss_mask.is_none());
}
