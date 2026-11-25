use super::*;
use crate::bin;
use crate::coordinator::{CoordinatorClient, DynCoordinator, MockCoordinator};
use crate::util::BitVector;
use prost::Message;

async fn client(measured: bool, completion: Duration) -> StaticDecoderClient {
    let library = bin::Library {
        gadget_types: vec![
            bin::GadgetType {
                gtype: 1,
                measurements: vec![bin::gadget_type::Measurement::default(); usize::from(measured)],
                ..Default::default()
            },
            bin::GadgetType {
                gtype: 2,
                ..Default::default()
            },
        ],
        program: (1..=2)
            .map(|gid| bin::Instruction {
                create: Some(bin::instruction::Create::Gadget(bin::Gadget {
                    gid,
                    gtype: gid,
                    metadata: crate::simulator::program_timing::timing_metadata(
                        0,
                        u64::try_from((completion * u32::try_from(gid).unwrap()).as_nanos()).unwrap(),
                    ),
                    ..Default::default()
                })),
            })
            .collect(),
        ..Default::default()
    };
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), library.encode_to_vec()).unwrap();
    let controller = Arc::new(StaticController::new(serde_json::json!({"filepath": file.path()})));
    controller
        .start(CoordinatorClient::Local(DynCoordinator::Mock(MockCoordinator::new())))
        .await;
    StaticDecoderClient {
        client: None,
        endpoint: tonic::transport::Endpoint::from_static("http://127.0.0.1:1"),
        local: Some(controller),
        schedule: (1..=2)
            .map(|index| DeliveryBatch {
                cumulative_count: usize::from(measured),
                completion: completion * u32::try_from(index).unwrap(),
                completed_gadgets: Some(index),
            })
            .collect(),
        last_latency_secs: 0.0,
    }
}

fn sample(measured: bool) -> ErrorSet {
    ErrorSet {
        measurements: if measured {
            BitVector { size: 1, data: vec![0] }
        } else {
            BitVector::default()
        },
        loss_mask: None,
        errors: vec![],
    }
}

#[test]
fn equal_deadlines_share_delivery_without_losing_empty_gadget_boundaries() {
    let timing = crate::simulator::program_timing::ProgramTiming {
        cumulative_measurements: vec![0, 1, 1, 3],
        completions: vec![
            Duration::ZERO,
            Duration::from_nanos(10),
            Duration::from_nanos(10),
            Duration::from_nanos(20),
        ],
    };
    let batches = DeliveryBatch::from_program(&timing);
    assert_eq!(
        batches
            .iter()
            .map(|batch| (batch.cumulative_count, batch.completion, batch.completed_gadgets))
            .collect::<Vec<_>>(),
        vec![
            (0, Duration::ZERO, Some(1)),
            (1, Duration::from_nanos(10), Some(3)),
            (3, Duration::from_nanos(20), Some(4)),
        ],
    );
}

#[tokio::test]
async fn coalesced_delivery_returns_every_gadget_in_program_order() {
    let mut client = client(true, Duration::ZERO).await;
    client.schedule = DeliveryBatch::from_program(&crate::simulator::program_timing::ProgramTiming {
        cumulative_measurements: vec![1, 1],
        completions: vec![Duration::ZERO; 2],
    });
    client.initialize().await.unwrap();
    for _ in 0..2 {
        let readouts = client.decode(&sample(true)).await.unwrap();
        assert_eq!(readouts.iter().map(|readout| readout.gid).collect::<Vec<_>>(), vec![1, 2]);
        client.reset().await.unwrap();
    }
}

#[tokio::test]
async fn timed_static_waits_for_terminal_gadgets_on_each_shot() {
    for measured in [false, true] {
        let mut client = client(measured, Duration::from_millis(10)).await;
        client.initialize().await.unwrap();
        for _ in 0..2 {
            let origin = Instant::now();
            let readouts = client.decode_at(&sample(measured), origin).await.unwrap();
            assert_eq!(readouts.len(), 2);
            assert!(origin.elapsed() >= Duration::from_millis(20));
            client.reset().await.unwrap();
        }
    }
}

#[tokio::test]
async fn timed_static_uses_shared_origin_without_extra_per_call_delays() {
    let mut client = client(true, Duration::from_secs(1)).await;
    client.initialize().await.unwrap();
    let origin = Instant::now() - Duration::from_secs(3);
    let readouts = tokio::time::timeout(Duration::from_millis(500), client.decode_at(&sample(true), origin))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(readouts.len(), 2);
}

#[tokio::test]
async fn timed_static_rejects_truncated_samples_instead_of_clamping_schedule() {
    let mut client = client(true, Duration::ZERO).await;
    let error = client.decode(&sample(false)).await.unwrap_err();
    assert!(error.to_string().contains("invalid chronological measurement schedule"));
}

#[tokio::test]
async fn native_static_simulator_derives_timing_from_the_binary_program() {
    for measured in [false, true] {
        let client = client(measured, Duration::from_millis(10)).await;
        let controller = client.local.unwrap();
        let circuit = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            circuit.path(),
            format!(
                "#!rhai\n# fn is_logical_error(shot, readouts, measurements) {{ false }}\n{}",
                if measured { "M[tag] 0\n" } else { "" }
            ),
        )
        .unwrap();
        let simulator = StaticSimulator::new(serde_json::json!({
            "filepath": circuit.path(), "shots": 2, "errors": 0, "strict_timing": true, "seed": 42,
        }));
        let (shutdown, receiver) = tokio::sync::oneshot::channel();
        let origin = Instant::now();
        simulator
            .start_with_controller(
                tonic::transport::Endpoint::from_static("http://127.0.0.1:1"),
                shutdown,
                controller,
            )
            .await;
        receiver.await.unwrap();
        assert!(origin.elapsed() >= Duration::from_millis(40));
    }
}
