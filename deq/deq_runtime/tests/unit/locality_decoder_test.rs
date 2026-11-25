use super::*;

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use crate::decoder::execution_pool::tests::allowed_cpus;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread::ThreadId;

    fn graph() -> DecodingHypergraph {
        DecodingHypergraph {
            vertex_num: 1,
            hyperedges: vec![blackbox_decoder::Hyperedge {
                vertices: vec![0],
                probability: 0.1,
                observable_flips: None,
            }],
        }
    }

    fn configuration() -> serde_json::Value {
        let pools: Vec<_> = allowed_cpus()
            .into_iter()
            .take(2)
            .enumerate()
            .map(|(index, cpu)| serde_json::json!({"name": format!("test-{index}"), "cpus": cpu.to_string()}))
            .collect();
        serde_json::json!({"pools": pools})
    }

    fn request(hid: u64) -> Request<blackbox_decoder::LoadedDecodingProblem> {
        Request::new(blackbox_decoder::LoadedDecodingProblem {
            hid,
            syndrome: Some(BitVector {
                size: 1,
                data: vec![0x80],
            }),
            ..Default::default()
        })
    }

    #[tokio::test]
    async fn cached_zero_requests_share_registry_reads_without_tracking_worker_jobs() {
        struct NeverCalled;
        impl DecoderInstance for NeverCalled {
            fn new(_: &DecodingHypergraph, _: &serde_json::Value) -> Self {
                Self
            }
            fn decode(&mut self, _: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
                panic!("an ordinary zero syndrome must not invoke the backend");
            }
            fn reset(&mut self) {}
        }
        let configuration = configuration();
        let config: ThreadPoolingConfig = serde_json::from_value(configuration.clone()).unwrap();
        let decoder = LocalityDecoder::<NeverCalled> {
            namespace: next_identity(),
            configuration: Arc::new(configuration),
            execution: ExecutionPools::new(&config.pools).unwrap(),
            features: DecoderFeatures::empty(),
            loaded: RwLock::new(HashMap::new()),
            pending: watch::channel(0).0,
            lifecycle: RwLock::new(()),
            instance: PhantomData,
        };
        let handle = decoder.load_hypergraph(Request::new(graph())).await.unwrap().into_inner().hid;
        let registry_reader = decoder.loaded.read().await;
        let pending = decoder.pending.subscribe();
        let mut zero = request(handle).into_inner();
        zero.syndrome.as_mut().unwrap().data[0] = 0;
        let result = tokio::time::timeout(std::time::Duration::from_secs(2), decoder.decode_loaded(Request::new(zero)))
            .await
            .expect("cached decodes must not require exclusive registry access")
            .unwrap();
        assert!(result.into_inner().subgraph.is_empty());
        assert!(!pending.has_changed().unwrap());
        drop(registry_reader);
    }

    #[tokio::test]
    async fn instances_are_prepared_reused_reset_and_destroyed_on_their_owner() {
        static CONSTRUCTED: AtomicUsize = AtomicUsize::new(0);
        static DESTROYED: AtomicUsize = AtomicUsize::new(0);
        static MIGRATED: AtomicUsize = AtomicUsize::new(0);
        struct Owned {
            owner: ThreadId,
            dirty: bool,
        }
        impl DecoderInstance for Owned {
            fn new(_graph: &DecodingHypergraph, _config: &serde_json::Value) -> Self {
                CONSTRUCTED.fetch_add(1, Ordering::Relaxed);
                Self {
                    owner: std::thread::current().id(),
                    dirty: false,
                }
            }
            fn decode(&mut self, _request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
                assert_eq!(self.owner, std::thread::current().id());
                assert!(!self.dirty, "instance state must be reset between requests");
                self.dirty = true;
                std::thread::sleep(std::time::Duration::from_millis(2));
                Ok(ParityFactor { subgraph: vec![0] })
            }
            fn reset(&mut self) {
                assert_eq!(self.owner, std::thread::current().id());
                self.dirty = false;
            }
        }
        impl Drop for Owned {
            fn drop(&mut self) {
                if self.owner != std::thread::current().id() {
                    MIGRATED.fetch_add(1, Ordering::Relaxed);
                }
                DESTROYED.fetch_add(1, Ordering::Relaxed);
            }
        }
        let decoder = DynLocalityDecoder::new::<Owned>(configuration(), None).unwrap();
        let capacity = decoder.execution().capacity();
        let handle = decoder.load_hypergraph(Request::new(graph())).await.unwrap().into_inner().hid;
        for result in futures_util::future::join_all((0..32).map(|_| decoder.decode_loaded(request(handle)))).await {
            assert_eq!(result.unwrap().into_inner().subgraph, vec![0]);
        }
        assert!(CONSTRUCTED.load(Ordering::Relaxed) <= capacity);
        decoder
            .reset(Request::new(blackbox_decoder::ResetRequest::default()))
            .await
            .unwrap();
        assert_eq!(
            decoder.decode_loaded(request(handle)).await.unwrap().into_inner().subgraph,
            vec![0]
        );
        decoder
            .reset(Request::new(blackbox_decoder::ResetRequest {
                reset_hypergraphs: true,
                ..Default::default()
            }))
            .await
            .unwrap();
        assert_eq!(
            decoder.decode_loaded(request(handle)).await.unwrap_err().code(),
            tonic::Code::NotFound
        );
        assert_eq!(CONSTRUCTED.load(Ordering::Relaxed), DESTROYED.load(Ordering::Relaxed));
        assert_eq!(MIGRATED.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn cached_shared_requests_reuse_instances_without_crossing_decoder_namespaces() {
        static CONSTRUCTED: AtomicUsize = AtomicUsize::new(0);
        struct Counting;
        impl DecoderInstance for Counting {
            fn new(_graph: &DecodingHypergraph, _config: &serde_json::Value) -> Self {
                CONSTRUCTED.fetch_add(1, Ordering::Relaxed);
                Self
            }

            fn decode(&mut self, _request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
                Ok(ParityFactor { subgraph: vec![0] })
            }
            fn reset(&mut self) {}
        }
        let first = DynLocalityDecoder::new::<Counting>(configuration(), None).unwrap();
        let second =
            DynLocalityDecoder::new::<Counting>(serde_json::json!({}), Some(Arc::clone(first.execution()))).unwrap();
        let source = Arc::new(SharedDecodingHypergraph::new(graph()));
        for _ in 0..8 {
            assert_eq!(
                first
                    .decode_shared(
                        Arc::clone(&source),
                        BitVector {
                            size: 1,
                            data: vec![0x80]
                        },
                        None,
                        None,
                    )
                    .await
                    .unwrap()
                    .subgraph,
                vec![0]
            );
        }
        assert_eq!(CONSTRUCTED.load(Ordering::Relaxed), 1);
        assert_eq!(
            second
                .decode_shared(
                    source,
                    BitVector {
                        size: 1,
                        data: vec![0x80]
                    },
                    None,
                    None,
                )
                .await
                .unwrap_err()
                .code(),
            tonic::Code::InvalidArgument
        );
        let handle = second.load_hypergraph(Request::new(graph())).await.unwrap().into_inner().hid;
        assert_eq!(
            second.decode_loaded(request(handle)).await.unwrap().into_inner().subgraph,
            vec![0]
        );
        first
            .reset(Request::new(blackbox_decoder::ResetRequest {
                reset_hypergraphs: true,
                ..Default::default()
            }))
            .await
            .unwrap();
        assert_eq!(
            second.decode_loaded(request(handle)).await.unwrap().into_inner().subgraph,
            vec![0]
        );
    }
    #[tokio::test]
    async fn failed_resets_are_reported_and_poisoned_instances_are_discarded() {
        static CONSTRUCTED: AtomicUsize = AtomicUsize::new(0);
        static DESTROYED: AtomicUsize = AtomicUsize::new(0);
        struct ResetPanics;
        impl DecoderInstance for ResetPanics {
            fn new(_: &DecodingHypergraph, _: &serde_json::Value) -> Self {
                CONSTRUCTED.fetch_add(1, Ordering::Relaxed);
                Self
            }
            fn decode(&mut self, _: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
                Ok(ParityFactor { subgraph: vec![0] })
            }
            fn reset(&mut self) {
                panic!("reset failed");
            }
        }
        impl Drop for ResetPanics {
            fn drop(&mut self) {
                DESTROYED.fetch_add(1, Ordering::Relaxed);
            }
        }
        let decoder = DynLocalityDecoder::new::<ResetPanics>(
            serde_json::json!({"pools": [{"cpus": allowed_cpus()[0].to_string()}]}),
            None,
        )
        .unwrap();
        let handle = decoder.load_hypergraph(Request::new(graph())).await.unwrap().into_inner().hid;
        for _ in 0..2 {
            let error = decoder.decode_loaded(request(handle)).await.unwrap_err();
            assert_eq!(error.code(), tonic::Code::Internal);
            assert!(error.message().contains("reset panicked"));
        }
        decoder
            .reset(Request::new(blackbox_decoder::ResetRequest::default()))
            .await
            .unwrap();
        assert_eq!(CONSTRUCTED.load(Ordering::Relaxed), 2);
        assert_eq!(DESTROYED.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn pooled_native_backends_preserve_protocol_results() {
        use crate::decoder::DecoderType;
        let mut kinds = vec![DecoderType::BlackBoxRelayBP, DecoderType::BlackBoxRelayBpF32];
        #[cfg(feature = "mwpm")]
        kinds.extend([DecoderType::BlackBoxMwpm, DecoderType::BlackBoxUf]);
        #[cfg(feature = "mwpf")]
        kinds.extend([DecoderType::BlackBoxMwpf, DecoderType::BlackBoxHuf]);
        #[cfg(feature = "tesseract")]
        kinds.push(DecoderType::BlackBoxTesseract);
        for kind in kinds {
            let decoder = kind.create(configuration());
            assert!(decoder.thread_pool().is_none());
            assert!(decoder.execution_capacity().unwrap() > 0);
            let handle = decoder.load_hypergraph(graph()).await.unwrap().hid;
            for _ in 0..3 {
                assert_eq!(
                    decoder.decode_loaded(request(handle).into_inner()).await.unwrap().subgraph,
                    vec![0]
                );
            }
            decoder.unload_hypergraphs(&[handle]).await.unwrap();
            assert_eq!(
                decoder.decode_loaded(request(handle).into_inner()).await.unwrap_err().code(),
                tonic::Code::NotFound
            );
        }
    }

    #[tokio::test]
    async fn cancellation_keeps_capacity_and_reset_waits_for_real_completion() {
        use futures_util::FutureExt as _;
        use std::sync::{OnceLock, mpsc};
        static STARTED: OnceLock<tokio::sync::Notify> = OnceLock::new();
        static RELEASED: OnceLock<std::sync::Mutex<mpsc::Receiver<()>>> = OnceLock::new();
        struct Blocking;
        impl DecoderInstance for Blocking {
            fn new(_graph: &DecodingHypergraph, _config: &serde_json::Value) -> Self {
                Self
            }
            fn decode(&mut self, _request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
                STARTED.get().unwrap().notify_one();
                RELEASED
                    .get()
                    .unwrap()
                    .lock()
                    .unwrap()
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
                Ok(ParityFactor { subgraph: vec![0] })
            }
            fn reset(&mut self) {}
        }
        let (release, waiting) = mpsc::channel();
        STARTED.set(tokio::sync::Notify::new()).unwrap();
        RELEASED.set(std::sync::Mutex::new(waiting)).unwrap();
        let config = serde_json::json!({"pools": [{"cpus": allowed_cpus()[0].to_string()}]});
        let decoder = DynLocalityDecoder::new::<Blocking>(config, None).unwrap();
        let handle = decoder.load_hypergraph(Request::new(graph())).await.unwrap().into_inner().hid;
        let pending = tokio::spawn({
            let decoder = Arc::clone(&decoder);
            async move { decoder.decode_loaded(request(handle)).await }
        });
        STARTED.get().unwrap().notified().await;
        pending.abort();
        assert!(pending.await.unwrap_err().is_cancelled());
        assert_eq!(decoder.execution().admission.available_permits(), 0);
        let mut reset = Box::pin(decoder.reset(Request::new(blackbox_decoder::ResetRequest {
            reset_hypergraphs: true,
            ..Default::default()
        })));
        assert!(reset.as_mut().now_or_never().is_none());
        release.send(()).unwrap();
        reset.await.unwrap();
        assert_eq!(decoder.execution().admission.available_permits(), 1);
        assert_eq!(
            decoder.decode_loaded(request(handle)).await.unwrap_err().code(),
            tonic::Code::NotFound
        );
    }

    #[tokio::test]
    async fn cold_construction_is_bounded_even_for_many_distinct_graphs() {
        static ACTIVE: AtomicUsize = AtomicUsize::new(0);
        static PEAK: AtomicUsize = AtomicUsize::new(0);
        struct SlowConstruction;
        impl DecoderInstance for SlowConstruction {
            fn new(_graph: &DecodingHypergraph, _config: &serde_json::Value) -> Self {
                let active = ACTIVE.fetch_add(1, Ordering::SeqCst) + 1;
                PEAK.fetch_max(active, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(5));
                ACTIVE.fetch_sub(1, Ordering::SeqCst);
                Self
            }
            fn decode(&mut self, _request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
                Ok(ParityFactor { subgraph: vec![0] })
            }
            fn reset(&mut self) {}
        }
        let cpus = allowed_cpus();
        let first = crate::decoder::execution_pool::numa_node(cpus[0]).unwrap();
        let cpus = cpus
            .into_iter()
            .filter(|&cpu| crate::decoder::execution_pool::numa_node(cpu).unwrap() == first)
            .take(2)
            .map(|cpu| cpu.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let decoder = DynLocalityDecoder::new::<SlowConstruction>(
            serde_json::json!({
                "pools": [{"cpus": cpus, "max_initializations": 1}]
            }),
            None,
        )
        .unwrap();
        let sources: Vec<_> = (0..16).map(|_| Arc::new(SharedDecodingHypergraph::new(graph()))).collect();
        let requests = sources.iter().map(|source| {
            decoder.decode_shared(
                Arc::clone(source),
                BitVector {
                    size: 1,
                    data: vec![0x80],
                },
                None,
                None,
            )
        });
        for result in futures_util::future::join_all(requests).await {
            assert_eq!(result.unwrap().subgraph, vec![0]);
        }
        assert_eq!(PEAK.load(Ordering::SeqCst), 1);
        assert_eq!(ACTIVE.load(Ordering::SeqCst), 0);
    }
}
