//! Tests for thread-pooled decoder lifecycle, protocol handling, and execution capacity.

use crate::decoder::SharedDecodingHypergraph;
use crate::decoder::blackbox_decoder::black_box_decoder_server::BlackBoxDecoder;
use crate::decoder::blackbox_decoder::{self, DecodingHypergraph, ParityFactor};
use crate::decoder::thread_pooling::{DecodeError, DecodeRequest, DecoderFactory, DecoderInstance, ThreadPoolingDecoder};
use crate::util::BitVector;
use std::sync::Arc;
use tonic::Request;

fn single_edge_hypergraph() -> DecodingHypergraph {
    DecodingHypergraph {
        vertex_num: 1,
        hyperedges: vec![blackbox_decoder::Hyperedge {
            observable_flips: None,
            vertices: vec![0],
            probability: 0.1,
        }],
    }
}

#[tokio::test]
async fn execution_capacity_defers_construction_and_pooled_instance_checkout() {
    use futures_util::FutureExt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static CONSTRUCTIONS: AtomicUsize = AtomicUsize::new(0);
    struct Instance;
    impl DecoderInstance for Instance {
        fn new(_graph: &DecodingHypergraph, _config: &serde_json::Value) -> Self {
            CONSTRUCTIONS.fetch_add(1, Ordering::Relaxed);
            Self
        }
        fn decode(&mut self, _request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
            Ok(ParityFactor { subgraph: vec![0] })
        }
        fn reset(&mut self) {}
    }

    let decoder = ThreadPoolingDecoder::<Instance>::new(serde_json::json!({"parallel": 1}));
    let hid = decoder
        .load_hypergraph(Request::new(single_edge_hypergraph()))
        .await
        .unwrap()
        .into_inner()
        .hid;
    let occupied = decoder.work_slots.acquire().await.unwrap();
    let mut loading = Box::pin(decoder.load_hypergraph(Request::new(single_edge_hypergraph())));
    let syndrome = BitVector {
        size: 1,
        data: vec![0x80],
    };
    let mut temporary = Box::pin(decoder.decode_shared(
        Arc::new(SharedDecodingHypergraph::new(single_edge_hypergraph())),
        syndrome.clone(),
        None,
        None,
    ));
    let mut loaded = Box::pin(decoder.decode_loaded(Request::new(blackbox_decoder::LoadedDecodingProblem {
        hid,
        syndrome: Some(syndrome.clone()),
        ..Default::default()
    })));
    assert!(loading.as_mut().now_or_never().is_none());
    assert!(temporary.as_mut().now_or_never().is_none());
    assert!(loaded.as_mut().now_or_never().is_none());
    assert_eq!(*decoder.decoding.borrow(), 3);
    assert_eq!(CONSTRUCTIONS.load(Ordering::Relaxed), 1);
    assert_eq!(decoder.loaded.lock().await[&hid].instances.len(), 1);
    drop(loading);
    drop(temporary);
    assert_eq!(*decoder.decoding.borrow(), 1);
    let mut resetting = Box::pin(decoder.reset(Request::new(blackbox_decoder::ResetRequest::default())));
    assert!(resetting.as_mut().now_or_never().is_none());
    drop(occupied);
    assert_eq!(loaded.await.unwrap().into_inner().subgraph, vec![0]);
    resetting.await.unwrap();
    let requests = (0..32).map(|_| {
        decoder.decode_loaded(Request::new(blackbox_decoder::LoadedDecodingProblem {
            hid,
            syndrome: Some(syndrome.clone()),
            ..Default::default()
        }))
    });
    for result in futures_util::future::join_all(requests).await {
        assert_eq!(result.unwrap().into_inner().subgraph, vec![0]);
    }
    assert_eq!(CONSTRUCTIONS.load(Ordering::Relaxed), 1);
    assert_eq!(decoder.loaded.lock().await[&hid].instances.len(), 1);
    assert_eq!(decoder.work_slots.available_permits(), 1);
    assert_eq!(*decoder.decoding.borrow(), 0);
}

#[tokio::test]
async fn unloading_a_graph_preserves_admitted_work_without_restoring_its_pool() {
    use futures_util::FutureExt;

    let decoder = crate::decoder::RelayBPDecoder::<f64>::new(serde_json::json!({"parallel": 1}));
    let mut handles = Vec::new();
    for _ in 0..2 {
        handles.push(
            decoder
                .load_hypergraph(Request::new(single_edge_hypergraph()))
                .await
                .unwrap()
                .into_inner()
                .hid,
        );
    }
    let (started, ready) = tokio::sync::oneshot::channel();
    let (release, waiting) = std::sync::mpsc::channel();
    decoder.thread_pool.spawn(move || {
        started.send(()).unwrap();
        waiting.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
    });
    ready.await.unwrap();
    let request = |hid| {
        Request::new(blackbox_decoder::LoadedDecodingProblem {
            hid,
            syndrome: Some(BitVector {
                size: 1,
                data: vec![0x80],
            }),
            ..Default::default()
        })
    };
    let mut admitted = Box::pin(decoder.decode_loaded(request(handles[0])));
    assert!(admitted.as_mut().now_or_never().is_none());
    decoder.unload_hypergraphs(&handles[..1]).await;
    assert!(!decoder.loaded.lock().await.contains_key(&handles[0]));
    release.send(()).unwrap();
    assert_eq!(admitted.await.unwrap().into_inner().subgraph, vec![0]);
    assert_eq!(decoder.loaded.lock().await.len(), 1);
    assert_eq!(
        decoder.decode_loaded(request(handles[0])).await.unwrap_err().code(),
        tonic::Code::NotFound
    );
    assert_eq!(
        decoder
            .decode_loaded(request(handles[1]))
            .await
            .unwrap()
            .into_inner()
            .subgraph,
        vec![0]
    );
}

#[tokio::test]
async fn same_pool_work_executes_inline_but_foreign_pool_work_does_not() {
    struct Instance;
    impl DecoderInstance for Instance {
        fn new(_graph: &DecodingHypergraph, _config: &serde_json::Value) -> Self {
            Self
        }
        fn decode(&mut self, _request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
            Ok(ParityFactor { subgraph: vec![0] })
        }
        fn reset(&mut self) {}
    }
    let decoder = Arc::new(ThreadPoolingDecoder::<Instance>::new(serde_json::json!({"parallel": 1})));
    let pool = Arc::clone(&decoder.thread_pool);
    let runtime = tokio::runtime::Handle::current();
    let task = crate::misc::pool_executor::spawn(Arc::clone(&pool), runtime.clone(), async move {
        let load = decoder.load_hypergraph(Request::new(single_edge_hypergraph()));
        tokio::pin!(load);
        let std::task::Poll::Ready(response) = futures_util::poll!(&mut load) else {
            panic!("same-pool initialization must not enqueue another job");
        };
        let hid = response.unwrap().into_inner().hid;
        let solve = decoder.decode_loaded(Request::new(blackbox_decoder::LoadedDecodingProblem {
            hid,
            syndrome: Some(BitVector {
                size: 1,
                data: vec![0x80],
            }),
            ..Default::default()
        }));
        tokio::pin!(solve);
        let std::task::Poll::Ready(response) = futures_util::poll!(&mut solve) else {
            panic!("same-pool solve must not enqueue another job");
        };
        assert_eq!(response.unwrap().into_inner().subgraph, vec![0]);
    });
    tokio::time::timeout(std::time::Duration::from_secs(2), task).await.unwrap();

    let decoder = Arc::new(ThreadPoolingDecoder::<Instance>::new(serde_json::json!({"parallel": 1})));
    let (started, ready) = tokio::sync::oneshot::channel();
    let (release, waiting) = std::sync::mpsc::channel();
    decoder.thread_pool.spawn(move || {
        started.send(()).unwrap();
        waiting.recv().unwrap();
    });
    ready.await.unwrap();
    let (sent, received) = tokio::sync::oneshot::channel();
    crate::misc::pool_executor::spawn(pool, runtime, async move {
        decoder.schedule_work(move || {
            sent.send(()).unwrap();
        });
    })
    .await;
    let mut received = Box::pin(received);
    assert!(futures_util::poll!(&mut received).is_pending());
    release.send(()).unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), received)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn live_graph_preparations_have_no_global_eviction_limit() {
    struct Instance;
    impl DecoderInstance for Instance {
        fn new(_graph: &DecodingHypergraph, _config: &serde_json::Value) -> Self {
            Self
        }
        fn decode(&mut self, _request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
            Ok(ParityFactor { subgraph: vec![0] })
        }
        fn reset(&mut self) {}
    }
    let decoder = ThreadPoolingDecoder::<Instance>::new(serde_json::json!({"parallel": 1}));
    let mut graphs = Vec::new();
    let mut preparations = Vec::new();
    for _ in 0..300 {
        let graph = Arc::new(SharedDecodingHypergraph::new(single_edge_hypergraph()));
        decoder
            .decode_shared(
                Arc::clone(&graph),
                BitVector {
                    size: 1,
                    data: vec![0x80],
                },
                None,
                None,
            )
            .await
            .unwrap();
        preparations.push(Arc::downgrade(&graph.prepared::<Instance>(&decoder.original_config).unwrap()));
        graphs.push(graph);
    }
    for (graph, preparation) in graphs.iter().zip(&preparations) {
        assert!(Arc::ptr_eq(
            &preparation.upgrade().expect("live graphs retain their setup"),
            &graph.prepared::<Instance>(&decoder.original_config).unwrap(),
        ));
    }
    graphs.clear();
    assert!(preparations.iter().all(|preparation| preparation.upgrade().is_none()));
}

#[test]
fn shared_graph_rejects_a_different_backend_or_configuration() {
    struct Instance<const KIND: u64>(u64);
    impl<const KIND: u64> DecoderInstance for Instance<KIND> {
        fn new(_graph: &DecodingHypergraph, config: &serde_json::Value) -> Self {
            Self(KIND + config["offset"].as_u64().unwrap())
        }
        fn decode(&mut self, _request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
            unreachable!()
        }
        fn reset(&mut self) {}
    }
    let graph = SharedDecodingHypergraph::new(single_edge_hypergraph());
    let first_config = Arc::new(serde_json::json!({"offset": 1}));
    let second_config = Arc::new(serde_json::json!({"offset": 2}));
    let first = graph.prepared::<Instance<10>>(&first_config).unwrap();
    assert_eq!(first.new_instance(&graph.hypergraph, &first_config).unwrap().0, 11);
    assert!(Arc::ptr_eq(&first, &graph.prepared::<Instance<10>>(&first_config).unwrap()));
    for error in [
        graph.prepared::<Instance<10>>(&second_config).err().unwrap(),
        graph.prepared::<Instance<20>>(&first_config).err().unwrap(),
    ] {
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
        assert!(error.message().contains("bound to another decoder"));
    }
}

#[tokio::test]
async fn cancelled_shared_request_retains_its_graph_until_the_worker_finishes() {
    use futures_util::FutureExt;

    let decoder = crate::decoder::RelayBPDecoder::<f64>::new(serde_json::json!({"parallel": 1}));
    let graph = Arc::new(SharedDecodingHypergraph::new(single_edge_hypergraph()));
    let weak_graph = Arc::downgrade(&graph);
    let (started, ready) = tokio::sync::oneshot::channel();
    let (release, waiting) = std::sync::mpsc::channel();
    decoder.thread_pool.spawn(move || {
        started.send(()).unwrap();
        waiting.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
    });
    ready.await.unwrap();
    let mut decoding = Box::pin(decoder.decode_shared(
        graph,
        BitVector {
            size: 1,
            data: vec![0x80],
        },
        None,
        None,
    ));
    assert!(decoding.as_mut().now_or_never().is_none());
    drop(decoding);
    assert!(weak_graph.upgrade().is_some());
    release.send(()).unwrap();
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        decoder.reset(Request::new(blackbox_decoder::ResetRequest::default())),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(weak_graph.upgrade().is_none());
}

#[tokio::test]
async fn shared_graph_validation_reuses_no_backend_instances_and_invalidates_mutation() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static VALIDATIONS: AtomicUsize = AtomicUsize::new(0);
    static PREPARATIONS: AtomicUsize = AtomicUsize::new(0);
    static INSTANCES: AtomicUsize = AtomicUsize::new(0);

    struct CountingInstance;

    impl DecoderInstance for CountingInstance {
        fn prepare(_graph: &DecodingHypergraph, _config: &serde_json::Value) -> Result<DecoderFactory<Self>, String> {
            PREPARATIONS.fetch_add(1, Ordering::Relaxed);
            Ok(Arc::new(Self::new))
        }

        fn validate_hypergraph(_graph: &DecodingHypergraph, config: &serde_json::Value) -> Result<(), String> {
            VALIDATIONS.fetch_add(1, Ordering::Relaxed);
            if config.get("reject") == Some(&serde_json::Value::Bool(true)) {
                return Err("rejected by changed configuration".to_string());
            }
            Ok(())
        }

        fn new(_graph: &DecodingHypergraph, _config: &serde_json::Value) -> Self {
            INSTANCES.fetch_add(1, Ordering::Relaxed);
            Self
        }

        fn decode(&mut self, _request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
            Ok(ParityFactor { subgraph: vec![0] })
        }

        fn reset(&mut self) {}
    }

    let mut decoder = ThreadPoolingDecoder::<CountingInstance>::new(serde_json::json!({"parallel": 1}));
    let mut graph = Arc::new(SharedDecodingHypergraph::new(single_edge_hypergraph()));
    let syndrome = BitVector {
        size: 1,
        data: vec![0x80],
    };
    decoder
        .decode_shared(Arc::clone(&graph), BitVector { size: 1, data: vec![0] }, None, None)
        .await
        .unwrap();
    assert_eq!(PREPARATIONS.load(Ordering::Relaxed), 0);
    assert_eq!(INSTANCES.load(Ordering::Relaxed), 0);
    for _ in 0..2 {
        assert_eq!(
            decoder
                .decode_shared(Arc::clone(&graph), syndrome.clone(), None, None)
                .await
                .unwrap()
                .subgraph,
            vec![0]
        );
    }
    assert_eq!(VALIDATIONS.load(Ordering::Relaxed), 1);
    assert_eq!(PREPARATIONS.load(Ordering::Relaxed), 1);
    assert_eq!(INSTANCES.load(Ordering::Relaxed), 2);
    assert!(decoder.loaded.lock().await.is_empty());
    let error = decoder
        .decode_shared(Arc::clone(&graph), BitVector::default(), None, None)
        .await
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    Arc::make_mut(&mut decoder.original_config)["reject"] = true.into();
    let error = decoder
        .decode_shared(Arc::clone(&graph), syndrome.clone(), None, None)
        .await
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    assert!(error.message().contains("bound to another decoder configuration"));
    Arc::make_mut(&mut decoder.original_config)["reject"] = false.into();
    graph = Arc::new(SharedDecodingHypergraph::new(graph.graph().clone()));
    decoder
        .decode_shared(Arc::clone(&graph), syndrome.clone(), None, None)
        .await
        .unwrap();
    assert_eq!(PREPARATIONS.load(Ordering::Relaxed), 2);
    let mut modified = graph.graph().clone();
    modified.hyperedges[0].probability = 0.2;
    graph = Arc::new(SharedDecodingHypergraph::new(modified));
    decoder
        .decode_shared(Arc::clone(&graph), syndrome.clone(), None, None)
        .await
        .unwrap();
    assert_eq!(PREPARATIONS.load(Ordering::Relaxed), 3);
    let mut modified = graph.graph().clone();
    modified.hyperedges[0].vertices = vec![1];
    graph = Arc::new(SharedDecodingHypergraph::new(modified));
    let error = decoder.decode_shared(graph, syndrome.clone(), None, None).await.unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    for _ in 0..300 {
        let graph = Arc::new(SharedDecodingHypergraph::new(single_edge_hypergraph()));
        decoder
            .decode_shared(Arc::clone(&graph), syndrome.clone(), None, None)
            .await
            .unwrap();
        let preparation = Arc::downgrade(&graph.prepared::<CountingInstance>(&decoder.original_config).unwrap());
        drop(graph);
        assert!(preparation.upgrade().is_none());
    }
}

#[tokio::test]
async fn shared_preparation_constructs_independent_instances_and_serves_loaded_overflow() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static PREPARATIONS: AtomicUsize = AtomicUsize::new(0);
    static INSTANCES: AtomicUsize = AtomicUsize::new(0);

    struct Instance {
        used: bool,
    }
    impl DecoderInstance for Instance {
        fn prepare(_graph: &DecodingHypergraph, _config: &serde_json::Value) -> Result<DecoderFactory<Self>, String> {
            PREPARATIONS.fetch_add(1, Ordering::Relaxed);
            Ok(Arc::new(Self::new))
        }
        fn new(_graph: &DecodingHypergraph, _config: &serde_json::Value) -> Self {
            INSTANCES.fetch_add(1, Ordering::Relaxed);
            Self { used: false }
        }
        fn decode(&mut self, _request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
            assert!(!self.used);
            self.used = true;
            Ok(ParityFactor { subgraph: vec![0] })
        }
        fn reset(&mut self) {
            self.used = false;
        }
    }

    let decoder = ThreadPoolingDecoder::<Instance>::new(serde_json::json!({"parallel": 2}));
    let graph = Arc::new(SharedDecodingHypergraph::new(single_edge_hypergraph()));
    let syndrome = BitVector {
        size: 1,
        data: vec![0x80],
    };
    let (first, second) = tokio::join!(
        decoder.decode_shared(Arc::clone(&graph), syndrome.clone(), None, None),
        decoder.decode_shared(Arc::clone(&graph), syndrome.clone(), None, None),
    );
    assert_eq!(first.unwrap().subgraph, vec![0]);
    assert_eq!(second.unwrap().subgraph, vec![0]);
    assert_eq!(PREPARATIONS.load(Ordering::Relaxed), 1);
    assert_eq!(INSTANCES.load(Ordering::Relaxed), 2);
    assert!(decoder.loaded.lock().await.is_empty());

    let hid = decoder
        .load_hypergraph(Request::new(single_edge_hypergraph()))
        .await
        .unwrap()
        .into_inner()
        .hid;
    let _reserved = decoder
        .loaded
        .lock()
        .await
        .get_mut(&hid)
        .unwrap()
        .instances
        .pop_back()
        .unwrap();
    for _ in 0..2 {
        assert_eq!(
            decoder
                .decode_loaded(Request::new(blackbox_decoder::LoadedDecodingProblem {
                    hid,
                    syndrome: Some(syndrome.clone()),
                    ..Default::default()
                }))
                .await
                .unwrap()
                .into_inner()
                .subgraph,
            vec![0]
        );
    }
    assert_eq!(PREPARATIONS.load(Ordering::Relaxed), 2);
    assert_eq!(INSTANCES.load(Ordering::Relaxed), 4);
    let loaded = decoder.loaded.lock().await;
    let loaded = &loaded[&hid];
    assert!(matches!(
        loaded
            .prepared
            .new_instance(&Arc::new(loaded.hypergraph.graph().clone()), &decoder.original_config),
        Err(DecodeError::InvalidInput(_))
    ));
    assert!(matches!(
        loaded.prepared.new_instance(
            &loaded.hypergraph.hypergraph,
            &Arc::new(decoder.original_config.as_ref().clone())
        ),
        Err(DecodeError::InvalidInput(_))
    ));
}

#[tokio::test]
async fn failed_preparation_is_reused_without_publishing_a_loaded_graph() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static PREPARATIONS: AtomicUsize = AtomicUsize::new(0);
    struct Instance;
    impl DecoderInstance for Instance {
        fn prepare(_graph: &DecodingHypergraph, _config: &serde_json::Value) -> Result<DecoderFactory<Self>, String> {
            PREPARATIONS.fetch_add(1, Ordering::Relaxed);
            Err("invalid backend preparation".into())
        }
        fn new(_graph: &DecodingHypergraph, _config: &serde_json::Value) -> Self {
            panic!("failed preparation must not construct an instance")
        }
        fn decode(&mut self, _request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
            unreachable!()
        }
        fn reset(&mut self) {}
    }
    let decoder = ThreadPoolingDecoder::<Instance>::new(serde_json::json!({"parallel": 1}));
    let graph = Arc::new(SharedDecodingHypergraph::new(single_edge_hypergraph()));
    let syndrome = BitVector {
        size: 1,
        data: vec![0x80],
    };
    for _ in 0..2 {
        let error = decoder
            .decode_shared(Arc::clone(&graph), syndrome.clone(), None, None)
            .await
            .unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
        assert!(error.message().contains("invalid backend preparation"));
    }
    assert_eq!(PREPARATIONS.load(Ordering::Relaxed), 1);
    let error = decoder
        .load_hypergraph(Request::new(single_edge_hypergraph()))
        .await
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    assert!(decoder.loaded.lock().await.is_empty());
    decoder
        .reset(Request::new(blackbox_decoder::ResetRequest {
            reset_hypergraphs: true,
            ..Default::default()
        }))
        .await
        .unwrap();
    let preparation = Arc::downgrade(&graph.prepared::<Instance>(&decoder.original_config).unwrap());
    drop(graph);
    assert!(preparation.upgrade().is_none());
}

#[tokio::test]
async fn reusable_preparation_preserves_corrections_across_backends() {
    async fn check<T: DecoderInstance + Send + 'static>() {
        let config = serde_json::json!({"parallel": 1});
        let decoder = ThreadPoolingDecoder::<T>::new(config.clone());
        let mut graph = single_edge_hypergraph();
        let mut excluded = graph.hyperedges[0].clone();
        excluded.probability = 0.0;
        graph.hyperedges.insert(0, excluded);
        let graph = Arc::new(SharedDecodingHypergraph::new(graph));
        let syndrome = BitVector {
            size: 1,
            data: vec![0x80],
        };
        let expected = T::new(&graph, &config)
            .decode(DecodeRequest {
                syndrome: &syndrome,
                decoder_seed: None,
                reweights: &[],
                loss: None,
            })
            .unwrap();
        assert_eq!(expected.subgraph, vec![1]);
        for _ in 0..3 {
            assert_eq!(
                decoder
                    .decode_shared(Arc::clone(&graph), syndrome.clone(), None, None)
                    .await
                    .unwrap(),
                expected
            );
        }
        assert!(graph.preparation.get().is_some());
        assert!(decoder.loaded.lock().await.is_empty());
    }
    check::<crate::decoder::relay_bp_decoder::RelayBPDecoderInstance<f64>>().await;
    check::<crate::decoder::relay_bp_decoder::RelayBPDecoderInstance<f32>>().await;
    #[cfg(feature = "mwpm")]
    {
        check::<crate::decoder::mwpm_decoder::MwpmDecoderInstance>().await;
        check::<crate::decoder::uf_decoder::UfDecoderInstance>().await;
    }
    #[cfg(feature = "mwpf")]
    {
        check::<crate::decoder::mwpf_decoder::MwpfDecoderInstance>().await;
        check::<crate::decoder::huf_decoder::HufDecoderInstance>().await;
    }
    #[cfg(feature = "tesseract")]
    check::<crate::decoder::tesseract_decoder::TesseractDecoderInstance>().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn external_decoder_jobs_wake_a_parked_caller_worker() {
    let runtime = tokio::runtime::Handle::current();
    let (finished, result) = tokio::sync::oneshot::channel();
    let caller = std::thread::spawn(move || {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .use_current_thread()
                .build()
                .unwrap(),
        );
        let decoder = Arc::new(crate::decoder::RelayBPDecoder::<f64>::with_thread_pool(
            serde_json::json!({"parallel": 1}),
            Some(Arc::clone(&pool)),
        ));
        let request = runtime.spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            let hid = decoder
                .load_hypergraph(Request::new(single_edge_hypergraph()))
                .await
                .unwrap()
                .into_inner()
                .hid;
            decoder
                .decode_loaded(Request::new(blackbox_decoder::LoadedDecodingProblem {
                    hid,
                    syndrome: Some(BitVector {
                        size: 1,
                        data: vec![0x80],
                    }),
                    ..Default::default()
                }))
                .await
                .unwrap()
                .into_inner()
        });
        let correction = crate::misc::pool_executor::block_on(&pool, &runtime, request)
            .unwrap()
            .unwrap();
        finished.send(correction.subgraph).unwrap();
    });
    assert_eq!(
        tokio::time::timeout(std::time::Duration::from_secs(2), result)
            .await
            .unwrap()
            .unwrap(),
        vec![0],
    );
    caller.join().unwrap();
}

#[cfg(feature = "cli")]
mod cancellation_trace_tests {
    use super::*;
    use fastrace::future::FutureExt as _;
    use std::sync::{OnceLock, mpsc};
    use std::time::Duration;

    fn assert_worker_trace(records: &[fastrace::collector::SpanRecord], trace_id: fastrace::collector::TraceId) {
        let worker = records
            .iter()
            .find(|span| span.name == "decoder_job" && span.trace_id == trace_id)
            .expect("worker trace must survive cancellation of its request");
        let mut previous = 0;
        for event in ["worker_started", "initialized", "solve_finished", "worker_finished"] {
            assert!(worker.events.iter().any(|record| record.name == event), "{event}");
            let key = format!("{event}_offset_ns");
            let offset = worker
                .properties
                .iter()
                .find(|(name, _)| name == &key)
                .unwrap()
                .1
                .parse::<u128>()
                .unwrap();
            assert!(offset >= previous);
            previous = offset;
        }
        assert!(
            worker
                .properties
                .iter()
                .any(|(name, value)| name == "solver_success" && value == "true")
        );
    }

    #[tokio::test]
    async fn cancelled_rpc_keeps_pending_count_until_the_worker_finishes() {
        static STARTED: OnceLock<tokio::sync::Notify> = OnceLock::new();
        static RELEASED: OnceLock<std::sync::Mutex<mpsc::Receiver<()>>> = OnceLock::new();

        struct BlockingInstance;

        impl DecoderInstance for BlockingInstance {
            fn new(_hypergraph: &DecodingHypergraph, _config: &serde_json::Value) -> Self {
                Self
            }

            fn decode(&mut self, _request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
                STARTED.get().unwrap().notify_one();
                RELEASED
                    .get()
                    .unwrap()
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(5))
                    .unwrap();
                Ok(ParityFactor { subgraph: vec![0] })
            }

            fn reset(&mut self) {}
        }

        let (reporter, records) = fastrace::collector::TestReporter::new();
        crate::misc::fastrace::set_reporter(reporter, fastrace::collector::Config::default());
        let (release, receiver) = mpsc::channel();
        STARTED.set(tokio::sync::Notify::new()).unwrap();
        RELEASED.set(std::sync::Mutex::new(receiver)).unwrap();
        let decoder = Arc::new(ThreadPoolingDecoder::<BlockingInstance>::new(
            serde_json::json!({"parallel": 1}),
        ));
        let mut pending = decoder.pending_decodes();
        let hypergraph = single_edge_hypergraph();
        let handle = decoder
            .load_hypergraph(Request::new(hypergraph.clone()))
            .await
            .unwrap()
            .into_inner()
            .hid;
        for loaded in [false, true] {
            tokio::time::timeout(Duration::from_secs(5), pending.wait_for(|count| *count == 0))
                .await
                .unwrap()
                .unwrap();
            let context = fastrace::collector::SpanContext::random();
            let trace = fastrace::Span::root("cancelled_decoder_test", context);
            let request = tokio::spawn({
                let decoder = Arc::clone(&decoder);
                let hypergraph = hypergraph.clone();
                async move {
                    let syndrome = Some(BitVector {
                        size: 1,
                        data: vec![0x80],
                    });
                    if loaded {
                        decoder
                            .decode_loaded(Request::new(blackbox_decoder::LoadedDecodingProblem {
                                hid: handle,
                                syndrome,
                                ..Default::default()
                            }))
                            .await
                    } else {
                        decoder
                            .decode(Request::new(blackbox_decoder::DecodingProblem {
                                hypergraph: Some(hypergraph),
                                syndrome,
                                ..Default::default()
                            }))
                            .await
                    }
                }
                .in_span(trace)
            });
            tokio::time::timeout(Duration::from_secs(5), STARTED.get().unwrap().notified())
                .await
                .unwrap();
            request.abort();
            assert!(request.await.unwrap_err().is_cancelled());
            assert_eq!(*pending.borrow(), 1);
            assert_eq!(decoder.work_slots.available_permits(), 0);
            release.send(()).unwrap();
            tokio::time::timeout(Duration::from_secs(5), pending.wait_for(|count| *count == 0))
                .await
                .unwrap()
                .unwrap();
            assert_eq!(decoder.work_slots.available_permits(), 1);
            if !loaded {
                fastrace::flush();
                assert_worker_trace(&records.lock(), context.trace_id);
            }
        }
    }
}
