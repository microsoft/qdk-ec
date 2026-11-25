use super::*;
use crate::decoder::DecoderType;
use crate::misc::bit_matrix::{append_bit, zeros};
use crate::misc::bit_vector::{from_sparse_indices, get_bit};

#[allow(
    clippy::too_many_lines,
    reason = "Self-contained three-gadget fixture includes both frame and detector propagation."
)]
fn library() -> bin::Library {
    let mut copy = zeros(1, 2);
    append_bit(&mut copy, 0, 0);
    let common = bin::GadgetType {
        gtype: 1,
        inputs: vec![bin::gadget_type::Port {
            ptype: 1,
            ..Default::default()
        }],
        outputs: vec![bin::gadget_type::Port {
            ptype: 1,
            ..Default::default()
        }],
        measurements: vec![bin::gadget_type::Measurement::default()],
        readouts: vec![bin::gadget_type::Readout {
            measurement_indices: vec![0],
            ..Default::default()
        }],
        correction_propagation: Some(copy.clone()),
        readout_propagation: Some(copy),
        logical_correction: Some(zeros(1, 1)),
        physical_correction: Some(zeros(1, 1)),
        ..Default::default()
    };
    let first = bin::GadgetType {
        gtype: 2,
        inputs: vec![],
        correction_propagation: Some(zeros(1, 1)),
        readout_propagation: Some(zeros(1, 1)),
        ..common.clone()
    };
    let mut library = bin::Library {
        port_types: vec![bin::PortType {
            ptype: 1,
            observables: vec![bin::port_type::Observable::default()],
            ..Default::default()
        }],
        gadget_types: vec![common, first],
        check_model_types: vec![
            bin::CheckModelType {
                ctype: 1,
                remote_gadgets: vec![bin::check_model_type::RemoteGadget {
                    port: Some(bin::check_model_type::remote_gadget::Port::Input(0)),
                    ..Default::default()
                }],
                checks: vec![bin::check_model_type::Check {
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
            },
            bin::CheckModelType {
                ctype: 2,
                checks: vec![bin::check_model_type::Check {
                    measurements: vec![bin::check_model_type::RemoteMeasurement::default()],
                    ..Default::default()
                }],
                ..Default::default()
            },
        ],
        error_model_types: vec![bin::ErrorModelType {
            etype: 1,
            errors: vec![bin::error_model_type::Error {
                checks: vec![bin::error_model_type::RemoteCheck::default()],
                probability: 0.01,
                residual: vec![0],
                readout_flips: vec![0],
                ..Default::default()
            }],
            ..Default::default()
        }],
        ..Default::default()
    };
    for gid in 1..=3 {
        library.program.extend([
            bin::Instruction {
                create: Some(bin::instruction::Create::Gadget(bin::Gadget {
                    gid,
                    gtype: if gid == 1 { 2 } else { 1 },
                    connectors: if gid == 1 {
                        vec![]
                    } else {
                        vec![bin::gadget::Connector { gid: gid - 1, port: 0 }]
                    },
                    ..Default::default()
                })),
            },
            bin::Instruction {
                create: Some(bin::instruction::Create::CheckModel(bin::CheckModel {
                    gid,
                    cid: gid,
                    ctype: if gid == 1 { 2 } else { 1 },
                    ..Default::default()
                })),
            },
            bin::Instruction {
                create: Some(bin::instruction::Create::ErrorModel(bin::ErrorModel {
                    eid: gid,
                    cid: gid,
                    etype: 1,
                    ..Default::default()
                })),
            },
        ]);
    }
    library
}

fn coordinator() -> CompiledWindowCoordinator {
    CompiledWindowCoordinator::new(
        serde_json::json!({"worker_threads": 2, "batch_size": 2}),
        DecoderType::BlackBoxMwpm.create(serde_json::json!({"parallel": 2})),
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn preceding_corrections_adjust_future_windows_without_mutating_original_syndromes() {
    let mut input = library();
    for kind in &mut input.gadget_types {
        kind.readouts[0].measurement_indices.clear();
        kind.readout_propagation = Some(zeros(1, kind.inputs.len() + 1));
    }
    for kind in &mut input.check_model_types {
        kind.remote_gadgets.clear();
        kind.checks[0].measurements = vec![bin::check_model_type::RemoteMeasurement::default()];
    }
    let error_type = &mut input.error_model_types[0];
    error_type.remote_check_models.push(bin::error_model_type::RemoteCheckModel {
        port: Some(bin::error_model_type::remote_check_model::Port::Output(0)),
        ..Default::default()
    });
    error_type.errors[0].checks.push(bin::error_model_type::RemoteCheck {
        remote_check_model: Some(0),
        check_index: 0,
    });
    error_type.errors[0].residual.clear();
    error_type.errors[0].readout_flips.clear();
    let mut terminal = error_type.clone();
    terminal.etype = 2;
    terminal.remote_check_models.clear();
    terminal.errors[0].checks.truncate(1);
    input.error_model_types.push(terminal);
    input.program.last_mut().unwrap().create = Some(bin::instruction::Create::ErrorModel(bin::ErrorModel {
        eid: 3,
        cid: 3,
        etype: 2,
        ..Default::default()
    }));
    let coordinator = coordinator();
    coordinator.load_library(Request::new(input)).await.unwrap();
    for fault in 0..3 {
        let mut syndrome = vec![fault];
        if fault < 2 {
            syndrome.push(fault + 1);
        }
        let result = coordinator
            .ingest(
                coordinator::Outcomes {
                    outcomes: Some(from_sparse_indices(3, &syndrome)),
                    ..Default::default()
                },
                Some(3),
            )
            .await
            .unwrap();
        for (gid, readout) in result.iter().enumerate() {
            assert_eq!(readout.correction_count, u64::from(u64::try_from(gid).unwrap() == fault));
            assert_eq!(
                readout.syndrome_count,
                u64::from(syndrome.contains(&u64::try_from(gid).unwrap()))
            );
        }
        coordinator
            .reset(Request::new(coordinator::ResetRequest::default()))
            .await
            .unwrap();
    }
    coordinator.cancel_pending().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn abandoned_input_cancels_its_epoch_and_reset_isolates_queued_work() {
    let coordinator = Arc::new(coordinator());
    coordinator.load_library(Request::new(library())).await.unwrap();
    let mut releases = Vec::new();
    for _ in 0..2 {
        let (started, ready) = tokio::sync::oneshot::channel();
        let (release, blocked) = std::sync::mpsc::channel();
        coordinator.pool.spawn(move || {
            started.send(()).unwrap();
            blocked.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        });
        ready.await.unwrap();
        releases.push(release);
    }
    let old = coordinator.session.read().await.clone().unwrap();
    let task = tokio::spawn({
        let coordinator = Arc::clone(&coordinator);
        async move {
            coordinator
                .ingest(
                    coordinator::Outcomes {
                        outcomes: Some(from_sparse_indices(3, &[0, 1, 2])),
                        ..Default::default()
                    },
                    Some(3),
                )
                .await
        }
    });
    tokio::task::yield_now().await;
    tokio::task::yield_now().await;
    task.abort();
    let _ = task.await;
    for release in releases {
        release.send(()).unwrap();
    }
    coordinator
        .reset(Request::new(coordinator::ResetRequest::default()))
        .await
        .unwrap();
    assert!(old.epoch.cancelled.load(std::sync::atomic::Ordering::Acquire));
    let result = coordinator
        .ingest(
            coordinator::Outcomes {
                outcomes: Some(from_sparse_indices(3, &[])),
                ..Default::default()
            },
            Some(3),
        )
        .await
        .unwrap();
    assert!(
        result
            .iter()
            .all(|value| value.syndrome_count == 0 && value.correction_count == 0)
    );
    coordinator.cancel_pending().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn immutable_frames_correct_each_single_fault_and_reset_without_recompilation() {
    let coordinator = coordinator();
    coordinator.load_library(Request::new(library())).await.unwrap();
    for fault in [None, Some(0), Some(1), Some(2)] {
        let indices: Vec<_> = fault.map_or_else(Vec::new, |index| (index..3).collect());
        let response = coordinator
            .ingest(
                coordinator::Outcomes {
                    outcomes: Some(from_sparse_indices(3, &indices)),
                    ..Default::default()
                },
                Some(3),
            )
            .await
            .unwrap();
        assert_eq!(response.len(), 3);
        for readout in &response {
            assert!(!get_bit(readout.readouts.as_ref().unwrap(), 0));
        }
        assert_eq!(
            response.iter().map(|value| value.correction_count).sum::<u64>(),
            u64::from(fault.is_some())
        );
        assert!(
            coordinator
                .ingest(
                    coordinator::Outcomes {
                        outcomes: Some(BitVector::default()),
                        ..Default::default()
                    },
                    Some(3),
                )
                .await
                .is_err()
        );
        coordinator
            .reset(Request::new(coordinator::ResetRequest::default()))
            .await
            .unwrap();
    }
    coordinator.cancel_pending().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nonzero_frames_are_readable_before_program_completion_and_reset_clears_them() {
    let coordinator = coordinator();
    assert!(coordinator.readout_bits(1).await.is_err());
    let mut input = library();
    let initial = &mut input.gadget_types[1];
    append_bit(initial.correction_propagation.as_mut().unwrap(), 0, 0);
    append_bit(initial.readout_propagation.as_mut().unwrap(), 0, 0);
    coordinator.load_library(Request::new(input)).await.unwrap();
    assert!(coordinator.readout_bits(0).await.is_err());
    assert!(coordinator.readout_bits(4).await.is_err());
    assert!(coordinator.readout_bits(1).await.unwrap().is_none());
    coordinator
        .ingest(
            coordinator::Outcomes {
                outcomes: Some(from_sparse_indices(1, &[])),
                ..Default::default()
            },
            Some(1),
        )
        .await
        .unwrap();
    let first = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            if let Some(bits) = coordinator.readout_bits(1).await.unwrap() {
                break bits;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(get_bit(&first, 0));
    assert!(coordinator.readout_bits(2).await.unwrap().is_none());
    let result = coordinator
        .ingest(
            coordinator::Outcomes {
                outcomes: Some(from_sparse_indices(2, &[])),
                ..Default::default()
            },
            Some(3),
        )
        .await
        .unwrap();
    for readout in result {
        assert!(get_bit(readout.readouts.as_ref().unwrap(), 0));
        assert_eq!(coordinator.readout_bits(readout.gid).await.unwrap(), readout.readouts);
    }
    coordinator
        .reset(Request::new(coordinator::ResetRequest::default()))
        .await
        .unwrap();
    assert!(coordinator.readout_bits(1).await.unwrap().is_none());
    coordinator.cancel_pending().await;
    assert!(coordinator.readout_bits(1).await.is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn partial_input_does_not_wait_for_future_inputs_and_invalid_features_do_not_advance_it() {
    let coordinator = coordinator();
    coordinator.load_library(Request::new(library())).await.unwrap();
    assert!(
        coordinator
            .ingest(
                coordinator::Outcomes {
                    outcomes: Some(from_sparse_indices(1, &[])),
                    loss_mask: Some(from_sparse_indices(1, &[])),
                    ..Default::default()
                },
                Some(1)
            )
            .await
            .is_err()
    );
    let first = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        coordinator.ingest(
            coordinator::Outcomes {
                outcomes: Some(from_sparse_indices(1, &[])),
                ..Default::default()
            },
            Some(1),
        ),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(first.is_empty());
    coordinator
        .reset(Request::new(coordinator::ResetRequest::default()))
        .await
        .unwrap();
    let result = coordinator
        .ingest(
            coordinator::Outcomes {
                outcomes: Some(from_sparse_indices(3, &[])),
                ..Default::default()
            },
            Some(3),
        )
        .await
        .unwrap();
    assert_eq!(result.iter().map(|value| value.gid).collect::<Vec<_>>(), vec![1, 2, 3]);
    coordinator.cancel_pending().await;
}

#[test]
fn unsupported_models_and_invalid_bindings_fail_during_compilation() {
    let mut loss = library();
    loss.gadget_types[0].loss_model = Some(bin::gadget_type::LossModel::default());
    assert!(plan::Plan::compile(loss, 16).is_err());
    let mut probability = library();
    probability.error_model_types[0].errors[0].probability = 0.6;
    assert!(plan::Plan::compile(probability, 16).is_err());
    let mut bad_reference = library();
    bad_reference.check_model_types[0].checks[0].measurements[1].remote_gadget = Some(3);
    assert!(plan::Plan::compile(bad_reference, 16).is_err());
    let mut cycle = library();
    cycle.check_model_types[0].remote_gadgets[0].previous_remote_gadget = Some(0);
    assert!(plan::Plan::compile(cycle, 16).is_err());
}

#[test]
fn repeated_error_effects_do_not_expand_instance_binding_storage() {
    let mut input = library();
    let error = input.error_model_types[0].errors[0].clone();
    input.error_model_types[0].errors = vec![error; 4096];
    let plan = plan::Plan::compile(input, 16).unwrap();
    for error in plan.errors {
        assert_eq!(error.support.len(), 1);
        assert!(error.support.capacity() <= 4);
    }
}

#[test]
fn inline_frame_maps_match_existing_sparse_matrix_and_modifier_semantics() {
    use binar::{BitVec, Bitwise};
    for rows in [0, 1, 50, 106, 128] {
        for columns in [0, 1, 51, 107, 128] {
            let mut source = zeros(rows, columns);
            for row in 0..rows {
                for column in 0..columns {
                    if (row * 17 + column) % 29 == 0 {
                        append_bit(&mut source, row, column);
                        append_bit(&mut source, row, column);
                    }
                }
            }
            let input = (0..columns).fold(0_u128, |input, bit| input | (u128::from(bit % 3 == 0) << bit));
            let vector: BitVec = (0..columns).map(|bit| input & (1_u128 << bit) != 0).collect();
            let dense = crate::misc::bit_matrix::sparse_to_dense(&source);
            let expected = &dense * &vector.as_view();
            let actual = frame::LinearMap::compile(Some(&source), None, rows, columns)
                .unwrap()
                .apply(input);
            for bit in 0..rows {
                assert_eq!(expected.index(bit), actual & (1_u128 << bit) != 0);
            }
            let modifier = bin::BitMatrixModifier {
                toggle: Some(source.clone()),
                overwrite: None,
            };
            assert_eq!(
                frame::LinearMap::compile(Some(&source), Some(&modifier), rows, columns)
                    .unwrap()
                    .apply(input),
                0
            );
        }
    }
}
