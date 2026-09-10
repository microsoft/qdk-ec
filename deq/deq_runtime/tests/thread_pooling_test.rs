//! Integration tests for thread-pooled decoder lifecycle and protocol handling.

use deq_runtime::decoder::DecoderFeatures;
use deq_runtime::decoder::blackbox_decoder::black_box_decoder_server::BlackBoxDecoder;
use deq_runtime::decoder::blackbox_decoder::{self, DecodingHypergraph, ParityFactor};
use deq_runtime::decoder::thread_pooling::{DecodeError, DecodeRequest, DecoderInstance, ThreadPoolingDecoder};
use deq_runtime::util::BitVector;
use std::sync::Arc;
use tonic::Request;

fn single_edge_hypergraph() -> DecodingHypergraph {
    DecodingHypergraph {
        vertex_num: 1,
        hyperedges: vec![blackbox_decoder::Hyperedge {
            vertices: vec![0],
            probability: 0.1,
        }],
    }
}

async fn assert_hid_not_found<T: DecoderInstance + Send + 'static>(decoder: &ThreadPoolingDecoder<T>, hid: u64) {
    let error = BlackBoxDecoder::decode_loaded(
        decoder,
        Request::new(blackbox_decoder::LoadedDecodingProblem {
            hid,
            syndrome: Some(BitVector { size: 1, data: vec![0] }),
            ..Default::default()
        }),
    )
    .await
    .unwrap_err();
    assert_eq!(error.code(), tonic::Code::NotFound);
}

#[tokio::test]
async fn failed_load_does_not_publish_a_hypergraph() {
    struct PanickingDecoderInstance;

    impl DecoderInstance for PanickingDecoderInstance {
        fn new(_hypergraph: &DecodingHypergraph, _config: &serde_json::Value) -> Self {
            panic!("construction failed")
        }

        fn decode(&mut self, _request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
            unreachable!()
        }

        fn reset(&mut self) {}
    }

    let decoder = ThreadPoolingDecoder::<PanickingDecoderInstance>::new(serde_json::json!({}));

    let error = BlackBoxDecoder::load_hypergraph(&decoder, Request::new(single_edge_hypergraph()))
        .await
        .unwrap_err();

    assert_eq!(error.code(), tonic::Code::Internal);
    assert_hid_not_found(&decoder, 1).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_load_does_not_publish_a_hypergraph() {
    use std::sync::{Barrier, OnceLock};

    static STARTED: OnceLock<Arc<Barrier>> = OnceLock::new();
    static RELEASED: OnceLock<Arc<Barrier>> = OnceLock::new();

    struct CancelledLoadDecoderInstance;

    impl DecoderInstance for CancelledLoadDecoderInstance {
        fn new(_hypergraph: &DecodingHypergraph, _config: &serde_json::Value) -> Self {
            STARTED.get().unwrap().wait();
            RELEASED.get().unwrap().wait();
            Self
        }

        fn decode(&mut self, _request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
            Ok(ParityFactor { subgraph: vec![] })
        }

        fn reset(&mut self) {}
    }

    let started = Arc::new(Barrier::new(2));
    let released = Arc::new(Barrier::new(2));
    STARTED.set(started.clone()).unwrap();
    RELEASED.set(released.clone()).unwrap();
    let decoder = Arc::new(ThreadPoolingDecoder::<CancelledLoadDecoderInstance>::new(serde_json::json!(
        {}
    )));
    let loading = tokio::spawn({
        let decoder = decoder.clone();
        async move { BlackBoxDecoder::load_hypergraph(decoder.as_ref(), Request::new(single_edge_hypergraph())).await }
    });
    started.wait();
    loading.abort();
    assert!(loading.await.unwrap_err().is_cancelled());

    released.wait();
    BlackBoxDecoder::reset(decoder.as_ref(), Request::new(blackbox_decoder::ResetRequest::default()))
        .await
        .unwrap();
    assert_hid_not_found(decoder.as_ref(), 1).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reset_waits_for_an_in_flight_load_before_clearing() {
    use std::sync::{Barrier, OnceLock};

    static STARTED: OnceLock<Arc<Barrier>> = OnceLock::new();
    static RELEASED: OnceLock<Arc<Barrier>> = OnceLock::new();

    struct BlockingDecoderInstance;

    impl DecoderInstance for BlockingDecoderInstance {
        fn new(_hypergraph: &DecodingHypergraph, _config: &serde_json::Value) -> Self {
            STARTED.get().unwrap().wait();
            RELEASED.get().unwrap().wait();
            Self
        }

        fn decode(&mut self, _request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
            Ok(ParityFactor { subgraph: vec![] })
        }

        fn reset(&mut self) {}
    }

    let started = Arc::new(Barrier::new(2));
    let released = Arc::new(Barrier::new(2));
    STARTED.set(started.clone()).unwrap();
    RELEASED.set(released.clone()).unwrap();
    let decoder = Arc::new(ThreadPoolingDecoder::<BlockingDecoderInstance>::new(serde_json::json!({})));

    let loading = tokio::spawn({
        let decoder = decoder.clone();
        async move { BlackBoxDecoder::load_hypergraph(decoder.as_ref(), Request::new(single_edge_hypergraph())).await }
    });
    started.wait();
    let resetting = tokio::spawn({
        let decoder = decoder.clone();
        async move {
            BlackBoxDecoder::reset(
                decoder.as_ref(),
                Request::new(blackbox_decoder::ResetRequest {
                    reset_hypergraphs: true,
                    ..Default::default()
                }),
            )
            .await
        }
    });
    tokio::task::yield_now().await;
    assert!(!resetting.is_finished());

    released.wait();
    let hid = loading.await.unwrap().unwrap().into_inner().hid;
    resetting.await.unwrap().unwrap();
    assert_hid_not_found(decoder.as_ref(), hid).await;
}

#[tokio::test]
async fn reset_panic_discards_the_instance_and_releases_the_counter() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CONSTRUCTIONS: AtomicUsize = AtomicUsize::new(0);

    struct ResetPanickingDecoderInstance;

    impl DecoderInstance for ResetPanickingDecoderInstance {
        fn new(_hypergraph: &DecodingHypergraph, _config: &serde_json::Value) -> Self {
            CONSTRUCTIONS.fetch_add(1, Ordering::Relaxed);
            Self
        }

        fn decode(&mut self, _request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
            Ok(ParityFactor { subgraph: vec![0] })
        }

        fn reset(&mut self) {
            panic!("reset failed")
        }
    }

    let decoder = ThreadPoolingDecoder::<ResetPanickingDecoderInstance>::new(serde_json::json!({}));
    let hid = BlackBoxDecoder::load_hypergraph(&decoder, Request::new(single_edge_hypergraph()))
        .await
        .unwrap()
        .into_inner()
        .hid;

    let decode = || {
        BlackBoxDecoder::decode_loaded(
            &decoder,
            Request::new(blackbox_decoder::LoadedDecodingProblem {
                hid,
                syndrome: Some(BitVector {
                    size: 1,
                    data: vec![0b1000_0000],
                }),
                ..Default::default()
            }),
        )
    };
    decode().await.unwrap();
    BlackBoxDecoder::reset(&decoder, Request::new(blackbox_decoder::ResetRequest::default()))
        .await
        .unwrap();
    decode().await.unwrap();
    BlackBoxDecoder::reset(&decoder, Request::new(blackbox_decoder::ResetRequest::default()))
        .await
        .unwrap();

    assert_eq!(CONSTRUCTIONS.load(Ordering::Relaxed), 2);
}

struct InvalidSubgraphDecoderInstance;

impl DecoderInstance for InvalidSubgraphDecoderInstance {
    fn new(_hypergraph: &DecodingHypergraph, _config: &serde_json::Value) -> Self {
        Self
    }

    fn decode(&mut self, _request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
        Ok(ParityFactor { subgraph: vec![1] })
    }

    fn reset(&mut self) {}
}

#[tokio::test]
#[cfg(debug_assertions)]
async fn invalid_backend_edge_is_reported_without_panicking() {
    let decoder = ThreadPoolingDecoder::<InvalidSubgraphDecoderInstance>::new(serde_json::json!({}));
    let hid = BlackBoxDecoder::load_hypergraph(&decoder, Request::new(single_edge_hypergraph()))
        .await
        .unwrap()
        .into_inner()
        .hid;

    let error = BlackBoxDecoder::decode_loaded(
        &decoder,
        Request::new(blackbox_decoder::LoadedDecodingProblem {
            hid,
            syndrome: Some(BitVector {
                size: 1,
                data: vec![0b1000_0000],
            }),
            ..Default::default()
        }),
    )
    .await
    .unwrap_err();

    assert_eq!(error.code(), tonic::Code::Internal);
    assert!(
        error
            .message()
            .contains("decoder returned edge 1, but the hypergraph has 1 edges")
    );
}

struct CombinedDecoderInstance;

impl DecoderInstance for CombinedDecoderInstance {
    fn supported_features(_config: &serde_json::Value) -> DecoderFeatures {
        DecoderFeatures::REWEIGHTS | DecoderFeatures::LOSS
    }

    fn new(_hypergraph: &DecodingHypergraph, _config: &serde_json::Value) -> Self {
        Self
    }

    fn decode(&mut self, request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
        assert_eq!(request.reweights, &[(0, 0.25)]);
        let loss = request.loss.expect("combined request must carry loss");
        assert_eq!(loss.sites[0].source_edges, vec![0]);
        Ok(ParityFactor { subgraph: vec![0] })
    }

    fn reset(&mut self) {}
}

#[tokio::test]
async fn zero_syndrome_still_requires_a_loaded_hypergraph() {
    let decoder = ThreadPoolingDecoder::<CombinedDecoderInstance>::new(serde_json::json!({}));
    assert_hid_not_found(&decoder, 99).await;
}

#[tokio::test]
async fn padding_bits_do_not_make_a_syndrome_nonzero() {
    let decoder = ThreadPoolingDecoder::<CombinedDecoderInstance>::new(serde_json::json!({}));
    let hid = BlackBoxDecoder::load_hypergraph(&decoder, Request::new(single_edge_hypergraph()))
        .await
        .unwrap()
        .into_inner()
        .hid;

    let response = BlackBoxDecoder::decode_loaded(
        &decoder,
        Request::new(blackbox_decoder::LoadedDecodingProblem {
            hid,
            syndrome: Some(BitVector { size: 1, data: vec![1] }),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert!(response.into_inner().subgraph.is_empty());
}

#[tokio::test]
#[cfg(debug_assertions)]
async fn malformed_syndrome_and_reweights_are_rejected() {
    let decoder = ThreadPoolingDecoder::<CombinedDecoderInstance>::new(serde_json::json!({}));
    let hid = BlackBoxDecoder::load_hypergraph(&decoder, Request::new(single_edge_hypergraph()))
        .await
        .unwrap()
        .into_inner()
        .hid;

    for syndrome in [BitVector { size: 1, data: vec![] }, BitVector { size: 2, data: vec![0] }] {
        let error = BlackBoxDecoder::decode_loaded(
            &decoder,
            Request::new(blackbox_decoder::LoadedDecodingProblem {
                hid,
                syndrome: Some(syndrome),
                ..Default::default()
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
    }

    for reweight in [
        blackbox_decoder::EdgeReweight {
            edge: 1,
            probability: 0.2,
        },
        blackbox_decoder::EdgeReweight {
            edge: 0,
            probability: f64::NAN,
        },
    ] {
        let error = BlackBoxDecoder::decode_loaded(
            &decoder,
            Request::new(blackbox_decoder::LoadedDecodingProblem {
                hid,
                syndrome: Some(BitVector {
                    size: 1,
                    data: vec![0b1000_0000],
                }),
                reweights: vec![reweight],
                ..Default::default()
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
    }

    let duplicate_reweight = BlackBoxDecoder::decode_loaded(
        &decoder,
        Request::new(blackbox_decoder::LoadedDecodingProblem {
            hid,
            syndrome: Some(BitVector {
                size: 1,
                data: vec![0b1000_0000],
            }),
            reweights: vec![
                blackbox_decoder::EdgeReweight {
                    edge: 0,
                    probability: 0.2,
                },
                blackbox_decoder::EdgeReweight {
                    edge: 0,
                    probability: 0.3,
                },
            ],
            ..Default::default()
        }),
    )
    .await
    .unwrap_err();
    assert_eq!(duplicate_reweight.code(), tonic::Code::InvalidArgument);

    for site in [
        blackbox_decoder::LossSite {
            source_edges: vec![1],
            probability: 0.2,
            ..Default::default()
        },
        blackbox_decoder::LossSite {
            children: vec![0],
            probability: 0.2,
            ..Default::default()
        },
        blackbox_decoder::LossSite {
            source_edges: vec![0, 0],
            probability: 0.2,
            ..Default::default()
        },
        blackbox_decoder::LossSite {
            heralds: vec![3, 3],
            probability: 0.2,
            ..Default::default()
        },
    ] {
        let error = BlackBoxDecoder::decode_loaded(
            &decoder,
            Request::new(blackbox_decoder::LoadedDecodingProblem {
                hid,
                syndrome: Some(BitVector {
                    size: 1,
                    data: vec![0b1000_0000],
                }),
                loss: Some(blackbox_decoder::LossInfo { sites: vec![site] }),
                ..Default::default()
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
    }
}

#[tokio::test]
#[cfg(debug_assertions)]
async fn invalid_hypergraph_is_rejected_before_construction() {
    let decoder = ThreadPoolingDecoder::<CombinedDecoderInstance>::new(serde_json::json!({}));
    for vertices in [vec![1], vec![0, 0]] {
        let error = BlackBoxDecoder::load_hypergraph(
            &decoder,
            Request::new(DecodingHypergraph {
                vertex_num: 1,
                hyperedges: vec![blackbox_decoder::Hyperedge {
                    vertices,
                    probability: 0.1,
                }],
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn overflow_construction_panic_is_contained_and_releases_the_counter() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Barrier, OnceLock};

    static CONSTRUCTIONS: AtomicUsize = AtomicUsize::new(0);
    static DECODE_STARTED: OnceLock<Arc<Barrier>> = OnceLock::new();
    static DECODE_RELEASED: OnceLock<Arc<Barrier>> = OnceLock::new();

    struct OverflowPanickingDecoderInstance;

    impl DecoderInstance for OverflowPanickingDecoderInstance {
        fn new(_hypergraph: &DecodingHypergraph, _config: &serde_json::Value) -> Self {
            if CONSTRUCTIONS.fetch_add(1, Ordering::Relaxed) > 0 {
                panic!("overflow construction failed")
            }
            Self
        }

        fn decode(&mut self, _request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
            DECODE_STARTED.get().unwrap().wait();
            DECODE_RELEASED.get().unwrap().wait();
            Ok(ParityFactor { subgraph: vec![] })
        }

        fn reset(&mut self) {}
    }

    let started = Arc::new(Barrier::new(2));
    let released = Arc::new(Barrier::new(2));
    DECODE_STARTED.set(started.clone()).unwrap();
    DECODE_RELEASED.set(released.clone()).unwrap();
    let decoder = Arc::new(ThreadPoolingDecoder::<OverflowPanickingDecoderInstance>::new(
        serde_json::json!({ "parallel": 2 }),
    ));
    let hid = BlackBoxDecoder::load_hypergraph(decoder.as_ref(), Request::new(single_edge_hypergraph()))
        .await
        .unwrap()
        .into_inner()
        .hid;
    let request = || {
        Request::new(blackbox_decoder::LoadedDecodingProblem {
            hid,
            syndrome: Some(BitVector {
                size: 1,
                data: vec![0b1000_0000],
            }),
            ..Default::default()
        })
    };

    let first = tokio::spawn({
        let decoder = decoder.clone();
        let request = request();
        async move { BlackBoxDecoder::decode_loaded(decoder.as_ref(), request).await }
    });
    started.wait();
    let second_error = BlackBoxDecoder::decode_loaded(decoder.as_ref(), request()).await.unwrap_err();
    assert_eq!(second_error.code(), tonic::Code::Internal);

    released.wait();
    first.await.unwrap().unwrap();
    BlackBoxDecoder::reset(decoder.as_ref(), Request::new(blackbox_decoder::ResetRequest::default()))
        .await
        .unwrap();
    assert_eq!(CONSTRUCTIONS.load(Ordering::Relaxed), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reset_waits_for_cancelled_one_shot_backend_work() {
    use std::sync::{Barrier, OnceLock};

    static DECODE_STARTED: OnceLock<Arc<Barrier>> = OnceLock::new();
    static DECODE_RELEASED: OnceLock<Arc<Barrier>> = OnceLock::new();

    struct BlockingOneShotDecoderInstance;

    impl DecoderInstance for BlockingOneShotDecoderInstance {
        fn new(_hypergraph: &DecodingHypergraph, _config: &serde_json::Value) -> Self {
            Self
        }

        fn decode(&mut self, _request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
            DECODE_STARTED.get().unwrap().wait();
            DECODE_RELEASED.get().unwrap().wait();
            Ok(ParityFactor { subgraph: vec![] })
        }

        fn reset(&mut self) {}
    }

    let started = Arc::new(Barrier::new(2));
    let released = Arc::new(Barrier::new(2));
    DECODE_STARTED.set(started.clone()).unwrap();
    DECODE_RELEASED.set(released.clone()).unwrap();
    let decoder = Arc::new(ThreadPoolingDecoder::<BlockingOneShotDecoderInstance>::new(
        serde_json::json!({}),
    ));
    let decoding = tokio::spawn({
        let decoder = decoder.clone();
        async move {
            BlackBoxDecoder::decode(
                decoder.as_ref(),
                Request::new(blackbox_decoder::DecodingProblem {
                    hypergraph: Some(single_edge_hypergraph()),
                    syndrome: Some(BitVector {
                        size: 1,
                        data: vec![0b1000_0000],
                    }),
                    loss: None,
                }),
            )
            .await
        }
    });
    started.wait();
    decoding.abort();
    assert!(decoding.await.unwrap_err().is_cancelled());

    let resetting = tokio::spawn({
        let decoder = decoder.clone();
        async move { BlackBoxDecoder::reset(decoder.as_ref(), Request::new(blackbox_decoder::ResetRequest::default())).await }
    });
    tokio::task::yield_now().await;
    assert!(!resetting.is_finished());

    released.wait();
    resetting.await.unwrap().unwrap();
}

#[tokio::test]
async fn loaded_decode_passes_reweights_and_loss_together() {
    let decoder = ThreadPoolingDecoder::<CombinedDecoderInstance>::new(serde_json::json!({}));
    let hid = BlackBoxDecoder::load_hypergraph(&decoder, Request::new(single_edge_hypergraph()))
        .await
        .unwrap()
        .into_inner()
        .hid;

    let response = BlackBoxDecoder::decode_loaded(
        &decoder,
        Request::new(blackbox_decoder::LoadedDecodingProblem {
            hid,
            syndrome: Some(BitVector {
                size: 1,
                data: vec![0b1000_0000],
            }),
            reweights: vec![blackbox_decoder::EdgeReweight {
                edge: 0,
                probability: 0.25,
            }],
            loss: Some(blackbox_decoder::LossInfo {
                sites: vec![blackbox_decoder::LossSite {
                    source_edges: vec![0],
                    probability: 0.2,
                    ..Default::default()
                }],
            }),
        }),
    )
    .await
    .unwrap();

    assert_eq!(response.into_inner().subgraph, vec![0]);
}

#[tokio::test]
async fn unsupported_features_are_rejected_before_backend_dispatch() {
    let decoder = ThreadPoolingDecoder::<InvalidSubgraphDecoderInstance>::new(serde_json::json!({}));
    let hypergraph = single_edge_hypergraph();
    let syndrome = BitVector {
        size: 1,
        data: vec![0b1000_0000],
    };

    let loss_error = BlackBoxDecoder::decode(
        &decoder,
        Request::new(blackbox_decoder::DecodingProblem {
            hypergraph: Some(hypergraph.clone()),
            syndrome: Some(syndrome.clone()),
            loss: Some(blackbox_decoder::LossInfo::default()),
        }),
    )
    .await
    .unwrap_err();
    assert_eq!(loss_error.code(), tonic::Code::FailedPrecondition);

    let hid = BlackBoxDecoder::load_hypergraph(&decoder, Request::new(hypergraph))
        .await
        .unwrap()
        .into_inner()
        .hid;
    let reweight_error = BlackBoxDecoder::decode_loaded(
        &decoder,
        Request::new(blackbox_decoder::LoadedDecodingProblem {
            hid,
            syndrome: Some(syndrome),
            reweights: vec![blackbox_decoder::EdgeReweight {
                edge: 0,
                probability: 0.25,
            }],
            loss: None,
        }),
    )
    .await
    .unwrap_err();
    assert_eq!(reweight_error.code(), tonic::Code::FailedPrecondition);
}

#[tokio::test]
async fn zero_syndrome_with_side_information_reaches_decoder() {
    let decoder = ThreadPoolingDecoder::<CombinedDecoderInstance>::new(serde_json::json!({}));
    let hid = BlackBoxDecoder::load_hypergraph(&decoder, Request::new(single_edge_hypergraph()))
        .await
        .unwrap()
        .into_inner()
        .hid;

    let response = BlackBoxDecoder::decode_loaded(
        &decoder,
        Request::new(blackbox_decoder::LoadedDecodingProblem {
            hid,
            syndrome: Some(BitVector { size: 1, data: vec![0] }),
            reweights: vec![blackbox_decoder::EdgeReweight {
                edge: 0,
                probability: 0.25,
            }],
            loss: Some(blackbox_decoder::LossInfo {
                sites: vec![blackbox_decoder::LossSite {
                    source_edges: vec![0],
                    probability: 0.2,
                    ..Default::default()
                }],
            }),
        }),
    )
    .await
    .unwrap();

    assert_eq!(response.into_inner().subgraph, vec![0]);
}
