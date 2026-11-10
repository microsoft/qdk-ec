use deq_runtime::bin;
use deq_runtime::coordinator::coordinator_server::Coordinator;
use deq_runtime::coordinator::{MemoryStats, Outcomes, WindowCoordinator};
use deq_runtime::decoder::{DynDecoder, NaiveDecoder};
use deq_runtime::misc::bit_matrix::zeros;
use deq_runtime::util::BitVector;
use std::sync::Arc;
use tonic::Request;

async fn setup() -> WindowCoordinator {
    let coordinator = WindowCoordinator::new(
        serde_json::json!({"buffer_radius": 0, "persistent_decoder": false}),
        DynDecoder::BlackBoxNaive(Arc::new(NaiveDecoder::new(serde_json::json!({})))),
    );
    let port = bin::gadget_type::Port {
        ptype: 1,
        ..Default::default()
    };
    let mut terminal_propagation = zeros(1, 3);
    terminal_propagation.i.push(0);
    terminal_propagation.j.push(0);
    coordinator
        .load_library(Request::new(bin::Library {
            port_types: vec![bin::PortType {
                ptype: 1,
                observables: vec![bin::port_type::Observable::default(); 2],
                ..Default::default()
            }],
            gadget_types: vec![
                bin::GadgetType {
                    gtype: 1,
                    measurements: vec![bin::gadget_type::Measurement::default()],
                    readouts: vec![
                        bin::gadget_type::Readout {
                            measurement_indices: vec![0],
                            ..Default::default()
                        };
                        2
                    ],
                    correction_propagation: Some(zeros(0, 1)),
                    readout_propagation: Some(zeros(2, 1)),
                    logical_correction: Some(zeros(0, 2)),
                    physical_correction: Some(zeros(0, 1)),
                    ..Default::default()
                },
                bin::GadgetType {
                    gtype: 2,
                    outputs: vec![port.clone()],
                    correction_propagation: Some(zeros(2, 1)),
                    readout_propagation: Some(zeros(0, 1)),
                    logical_correction: Some(zeros(2, 0)),
                    physical_correction: Some(zeros(2, 0)),
                    ..Default::default()
                },
                bin::GadgetType {
                    gtype: 3,
                    inputs: vec![port],
                    readouts: vec![bin::gadget_type::Readout::default()],
                    correction_propagation: Some(zeros(0, 3)),
                    readout_propagation: Some(terminal_propagation),
                    logical_correction: Some(zeros(0, 1)),
                    physical_correction: Some(zeros(0, 0)),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }))
        .await
        .unwrap();
    coordinator
}

async fn execute(coordinator: &WindowCoordinator, gadget: bin::Gadget) -> Result<(), tonic::Status> {
    coordinator
        .execute(Request::new(bin::Instruction {
            create: Some(bin::instruction::Create::Gadget(gadget)),
        }))
        .await
        .map(|_| ())
}

async fn source(coordinator: &WindowCoordinator, gid: u64, expected: u64) {
    execute(
        coordinator,
        bin::Gadget {
            gid,
            gtype: 1,
            expected_remote_references: expected,
            ..Default::default()
        },
    )
    .await
    .unwrap();
}

fn consumer(gid: u64, source_gid: u64, indices: &[u64]) -> bin::Gadget {
    let mut matrix = zeros(2, indices.len());
    for column in 0..indices.len() {
        matrix.i.push(0);
        matrix.j.push(u64::try_from(column).unwrap());
    }
    bin::Gadget {
        gid,
        gtype: 2,
        modifier: Some(bin::GadgetModifier {
            remote_conditional_correction: Some(bin::RemoteConditionalCorrection {
                remote_readouts: indices
                    .iter()
                    .map(|&index| bin::remote_conditional_correction::RemoteReadout {
                        gid: source_gid,
                        readout_index: index,
                    })
                    .collect(),
                correction: Some(matrix),
            }),
            ..Default::default()
        }),
        ..Default::default()
    }
}

async fn decode(coordinator: &WindowCoordinator, gid: u64, measured: bool) -> BitVector {
    coordinator
        .decode(Request::new(Outcomes {
            gid,
            outcomes: Some(if measured {
                BitVector {
                    size: 1,
                    data: vec![0x80],
                }
            } else {
                BitVector::default()
            }),
            ..Default::default()
        }))
        .await
        .unwrap()
        .into_inner()
        .readouts
        .unwrap()
}

async fn finish_consumer(coordinator: &WindowCoordinator, gid: u64, terminal: u64, expected: bool) {
    decode(coordinator, gid, false).await;
    execute(
        coordinator,
        bin::Gadget {
            gid: terminal,
            gtype: 3,
            connectors: vec![bin::gadget::Connector { gid, port: 0 }],
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let result = decode(coordinator, terminal, false).await;
    assert_eq!(result.data, vec![if expected { 0x80 } else { 0 }]);
}

async fn stats(coordinator: &WindowCoordinator) -> MemoryStats {
    coordinator.get_memory_stats(Request::new(())).await.unwrap().into_inner()
}

#[tokio::test]
async fn hinted_readouts_outlive_source_state_but_not_their_last_consumer() {
    let coordinator = setup().await;
    source(&coordinator, 1, 2).await;
    decode(&coordinator, 1, true).await;
    let initial = stats(&coordinator).await;
    assert_eq!(
        (
            initial.gadgets,
            initial.frame_gadgets,
            initial.retained_readouts,
            initial.retained_readout_bytes
        ),
        (0, 0, 1, 1)
    );
    for gid in 2..=101 {
        source(&coordinator, gid, 0).await;
        decode(&coordinator, gid, true).await;
    }
    assert_eq!(stats(&coordinator).await.gadgets, 0);
    assert_eq!(stats(&coordinator).await.retained_readouts, 1);
    execute(&coordinator, consumer(102, 1, &[0, 1])).await.unwrap();
    assert_eq!(stats(&coordinator).await.retained_readouts, 1, "one consumer, not two bits");
    execute(&coordinator, consumer(103, 1, &[0])).await.unwrap();
    assert_eq!(stats(&coordinator).await.retained_readouts, 0);
    finish_consumer(&coordinator, 102, 104, false).await;
    finish_consumer(&coordinator, 103, 105, true).await;
    assert_eq!(stats(&coordinator).await.frame_gadgets, 0);
    assert!(
        execute(&coordinator, consumer(106, 1, &[0]))
            .await
            .unwrap_err()
            .message()
            .contains("collected")
    );
}

#[tokio::test]
async fn zero_and_underestimated_hints_accept_pending_consumers_without_losing_values() {
    for hint in [0, 1] {
        let coordinator = setup().await;
        source(&coordinator, 1, hint).await;
        execute(&coordinator, consumer(2, 1, &[0])).await.unwrap();
        execute(&coordinator, consumer(3, 1, &[1])).await.unwrap();
        decode(&coordinator, 1, true).await;
        // Force collection before the readout batch fills to exercise pending consumers.
        coordinator.unload(Request::new(Default::default())).await.unwrap();
        assert_eq!(stats(&coordinator).await.retained_readouts, 0);
        assert!(!coordinator.gadgets.read().await.contains_key(&1));
        finish_consumer(&coordinator, 2, 4, true).await;
        finish_consumer(&coordinator, 3, 5, true).await;
    }
}

#[tokio::test]
async fn an_already_waiting_consumer_keeps_its_dependency_after_the_last_hint_is_spent() {
    let coordinator = Arc::new(setup().await);
    source(&coordinator, 1, 1).await;
    execute(&coordinator, consumer(2, 1, &[0])).await.unwrap();
    let waiting = tokio::spawn({
        let coordinator = Arc::clone(&coordinator);
        async move { decode(&coordinator, 2, false).await }
    });
    let mut published = coordinator.gadgets.read().await[&2].outcomes.subscribe();
    tokio::time::timeout(std::time::Duration::from_secs(5), published.wait_for(Option::is_some))
        .await
        .unwrap()
        .unwrap();
    assert!(!waiting.is_finished());
    decode(&coordinator, 1, true).await;
    tokio::time::timeout(std::time::Duration::from_secs(5), waiting)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stats(&coordinator).await.retained_readouts, 0);
    execute(
        &coordinator,
        bin::Gadget {
            gid: 3,
            gtype: 3,
            connectors: vec![bin::gadget::Connector { gid: 2, port: 0 }],
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(decode(&coordinator, 3, false).await.data, vec![0x80]);
}

#[tokio::test]
async fn rejected_references_do_not_consume_hints_and_reset_clears_unspent_hints() {
    let coordinator = setup().await;
    source(&coordinator, 1, 1).await;
    decode(&coordinator, 1, true).await;
    let invalid = consumer(2, 1, &[0, 99]);
    assert_eq!(
        execute(&coordinator, invalid).await.unwrap_err().code(),
        tonic::Code::InvalidArgument
    );
    assert_eq!(stats(&coordinator).await.retained_readouts, 1);
    execute(&coordinator, consumer(2, 1, &[0])).await.unwrap();
    assert_eq!(stats(&coordinator).await.retained_readouts, 0);
    finish_consumer(&coordinator, 2, 3, true).await;
    source(&coordinator, 4, 99).await;
    decode(&coordinator, 4, true).await;
    assert_eq!(stats(&coordinator).await.retained_readouts, 1);
    coordinator.reset(Request::new(Default::default())).await.unwrap();
    assert_eq!(stats(&coordinator).await.retained_readouts, 0);
    source(&coordinator, 1, 0).await;
    decode(&coordinator, 1, true).await;
    assert!(execute(&coordinator, consumer(2, 1, &[0])).await.is_err());
}
