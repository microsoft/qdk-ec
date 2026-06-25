//! Tests for MockDecoder

use deq_runtime::decoder::MockDecoder;
use deq_runtime::decoder::blackbox_decoder::{self, black_box_decoder_server::BlackBoxDecoder};
use deq_runtime::util::BitVector;
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
        }),
    )
    .await;

    assert!(result.is_err());
    assert!(result.unwrap_err().message().contains("hid=999"));
}
