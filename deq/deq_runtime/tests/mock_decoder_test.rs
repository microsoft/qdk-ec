//! Tests for MockDecoder

#[cfg(feature = "cli")]
use deq_runtime::decoder::blackbox_decoder::black_box_decoder_server::BlackBoxDecoderServer;
use deq_runtime::decoder::blackbox_decoder::{self, black_box_decoder_server::BlackBoxDecoder};
use deq_runtime::decoder::thread_pooling::DecoderFeatures;
use deq_runtime::decoder::{BlackBoxDecoderClient, MockDecoder};
use deq_runtime::util::BitVector;
use std::sync::Arc;
use tonic::Request;

#[tokio::test]
async fn test_mock_decoder_records_decode_calls() {
    let decoder = MockDecoder::new();

    let hypergraph = blackbox_decoder::DecodingHypergraph {
        vertex_num: 3,
        hyperedges: vec![blackbox_decoder::Hyperedge {
            vertices: vec![0, 1],
            probability: 0.1,
        }],
    };
    let syndrome = BitVector {
        size: 3,
        data: vec![0b101],
    };

    let response = BlackBoxDecoder::decode(
        &decoder,
        Request::new(blackbox_decoder::DecodingProblem {
            hypergraph: Some(hypergraph.clone()),
            syndrome: Some(syndrome.clone()),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert!(response.into_inner().subgraph.is_empty());

    let state = decoder.state.read().await;
    assert_eq!(state.decode_calls.len(), 1);
    assert_eq!(state.decode_calls[0].hypergraph.vertex_num, 3);
    assert_eq!(state.decode_calls[0].syndrome.data, vec![0b101]);
}

#[tokio::test]
async fn test_mock_decoder_load_hypergraph() {
    let decoder = MockDecoder::new();

    let hypergraph = blackbox_decoder::DecodingHypergraph {
        vertex_num: 5,
        hyperedges: vec![],
    };

    let response = BlackBoxDecoder::load_hypergraph(&decoder, Request::new(hypergraph))
        .await
        .unwrap();
    assert_eq!(response.into_inner().hid, 1);

    let hypergraph2 = blackbox_decoder::DecodingHypergraph {
        vertex_num: 10,
        hyperedges: vec![],
    };

    let response2 = BlackBoxDecoder::load_hypergraph(&decoder, Request::new(hypergraph2))
        .await
        .unwrap();
    assert_eq!(response2.into_inner().hid, 2);

    let state = decoder.state.read().await;
    assert_eq!(state.loaded_hypergraphs.len(), 2);
    assert_eq!(state.loaded_hypergraphs[&1].vertex_num, 5);
    assert_eq!(state.loaded_hypergraphs[&2].vertex_num, 10);
}

#[tokio::test]
async fn test_mock_decoder_decode_loaded() {
    let decoder = MockDecoder::new();

    // First load a hypergraph
    let hypergraph = blackbox_decoder::DecodingHypergraph {
        vertex_num: 3,
        hyperedges: vec![],
    };
    let load_response = BlackBoxDecoder::load_hypergraph(&decoder, Request::new(hypergraph))
        .await
        .unwrap();
    let hid = load_response.into_inner().hid;

    // Then decode with it
    let syndrome = BitVector {
        size: 3,
        data: vec![0b011],
    };

    let response = BlackBoxDecoder::decode_loaded(
        &decoder,
        Request::new(blackbox_decoder::LoadedDecodingProblem {
            hid,
            syndrome: Some(syndrome),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert!(response.into_inner().subgraph.is_empty());

    let state = decoder.state.read().await;
    assert_eq!(state.decode_loaded_calls.len(), 1);
    assert_eq!(state.decode_loaded_calls[0].hid, hid);
}

#[tokio::test]
async fn test_decoder_capabilities_are_composable() {
    let decoder = MockDecoder::with_features(DecoderFeatures::REWEIGHTS | DecoderFeatures::LOSS);
    let capabilities = BlackBoxDecoder::get_capabilities(&decoder, Request::new(()))
        .await
        .unwrap()
        .into_inner();

    assert_eq!(
        capabilities.features,
        vec![
            blackbox_decoder::DecoderFeature::Reweights as i32,
            blackbox_decoder::DecoderFeature::Loss as i32,
        ]
    );
}

#[cfg(feature = "cli")]
#[tokio::test]
async fn test_remote_client_queries_and_caches_capabilities() {
    let decoder = Arc::new(MockDecoder::with_features(DecoderFeatures::REWEIGHTS | DecoderFeatures::LOSS));
    let incoming = tonic::transport::server::TcpIncoming::bind("127.0.0.1:0".parse().unwrap()).unwrap();
    let address = incoming.local_addr().unwrap();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let service = BlackBoxDecoderServer::from_arc(decoder.clone());
    let server = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(service)
            .serve_with_incoming_shutdown(incoming, async {
                let _ = shutdown_rx.await;
            })
            .await
            .unwrap();
    });

    let endpoint = tonic::transport::Endpoint::from_shared(format!("http://{address}")).unwrap();
    let mut client = BlackBoxDecoderClient::from_endpoint(endpoint).await;
    assert_eq!(client.features(), DecoderFeatures::REWEIGHTS | DecoderFeatures::LOSS);

    let hid = client
        .load_hypergraph(blackbox_decoder::DecodingHypergraph {
            vertex_num: 1,
            hyperedges: vec![blackbox_decoder::Hyperedge {
                vertices: vec![0],
                probability: 0.1,
            }],
        })
        .await
        .unwrap()
        .hid;
    client
        .decode_loaded(blackbox_decoder::LoadedDecodingProblem {
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
        })
        .await
        .unwrap();
    let state = decoder.state.read().await;
    assert_eq!(state.decode_loaded_calls[0].reweights[0].probability, 0.25);
    assert_eq!(
        state.decode_loaded_calls[0].loss.as_ref().unwrap().sites[0].source_edges,
        vec![0]
    );
    drop(state);

    shutdown_tx.send(()).unwrap();
    server.await.unwrap();
}

#[tokio::test]
async fn test_mock_decoder_accepts_reweights_and_loss_together() {
    let decoder = MockDecoder::new();
    let hid = BlackBoxDecoder::load_hypergraph(
        &decoder,
        Request::new(blackbox_decoder::DecodingHypergraph {
            vertex_num: 1,
            hyperedges: vec![blackbox_decoder::Hyperedge {
                vertices: vec![0],
                probability: 0.1,
            }],
        }),
    )
    .await
    .unwrap()
    .into_inner()
    .hid;
    let loss = blackbox_decoder::LossInfo {
        sites: vec![blackbox_decoder::LossSite {
            source_edges: vec![0],
            probability: 0.2,
            ..Default::default()
        }],
    };

    BlackBoxDecoder::decode_loaded(
        &decoder,
        Request::new(blackbox_decoder::LoadedDecodingProblem {
            hid,
            syndrome: Some(BitVector {
                size: 1,
                data: vec![0b1000_0000],
            }),
            reweights: vec![blackbox_decoder::EdgeReweight {
                edge: 0,
                probability: 0.3,
            }],
            loss: Some(loss.clone()),
        }),
    )
    .await
    .unwrap();

    let state = decoder.state.read().await;
    assert_eq!(state.decode_loaded_calls[0].reweights[0].probability, 0.3);
    assert_eq!(state.decode_loaded_calls[0].loss, Some(loss));
}

#[tokio::test]
async fn test_client_rejects_unsupported_reweights_without_dispatch() {
    let decoder = Arc::new(MockDecoder::with_features(DecoderFeatures::LOSS));
    let mut client = BlackBoxDecoderClient::from_mock(decoder.clone());
    let result = client
        .decode_loaded(blackbox_decoder::LoadedDecodingProblem {
            hid: 1,
            syndrome: Some(BitVector {
                size: 1,
                data: vec![0b1000_0000],
            }),
            reweights: vec![blackbox_decoder::EdgeReweight {
                edge: 0,
                probability: 0.3,
            }],
            loss: None,
        })
        .await;

    assert_eq!(result.unwrap_err().code(), tonic::Code::FailedPrecondition);
    assert!(decoder.state.read().await.decode_loaded_calls.is_empty());
}

#[tokio::test]
async fn test_mock_decoder_custom_response() {
    let decoder = MockDecoder::new();

    let syndrome_data = vec![0b101];
    decoder.set_response(syndrome_data.clone(), vec![0, 2, 5]).await;

    let hypergraph = blackbox_decoder::DecodingHypergraph {
        vertex_num: 3,
        hyperedges: vec![],
    };
    let syndrome = BitVector {
        size: 3,
        data: syndrome_data,
    };

    let response = BlackBoxDecoder::decode(
        &decoder,
        Request::new(blackbox_decoder::DecodingProblem {
            hypergraph: Some(hypergraph),
            syndrome: Some(syndrome),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(response.into_inner().subgraph, vec![0, 2, 5]);
}

#[tokio::test]
async fn test_mock_decoder_reset() {
    let decoder = MockDecoder::new();

    // Load a hypergraph
    let hypergraph = blackbox_decoder::DecodingHypergraph {
        vertex_num: 3,
        hyperedges: vec![],
    };
    BlackBoxDecoder::load_hypergraph(&decoder, Request::new(hypergraph))
        .await
        .unwrap();

    // Reset with hypergraphs
    BlackBoxDecoder::reset(
        &decoder,
        Request::new(blackbox_decoder::ResetRequest {
            reset_hypergraphs: true,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let state = decoder.state.read().await;
    assert_eq!(state.reset_count, 1);
    assert!(state.loaded_hypergraphs.is_empty());
    assert_eq!(state.next_hid, 1);
}

#[tokio::test]
async fn test_mock_decoder_decode_loaded_not_found() {
    let decoder = MockDecoder::new();

    let syndrome = BitVector {
        size: 3,
        data: vec![0b011],
    };

    let result = BlackBoxDecoder::decode_loaded(
        &decoder,
        Request::new(blackbox_decoder::LoadedDecodingProblem {
            hid: 999,
            syndrome: Some(syndrome),
            ..Default::default()
        }),
    )
    .await;

    assert!(result.is_err());
    assert!(result.unwrap_err().message().contains("hid=999"));
}
