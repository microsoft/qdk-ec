//! End-to-end test of the dynamic-library decoder: load the reference plugin
//! `cdylib` through deq's gRPC `BlackBoxDecoder` surface and decode. Proves a
//! binary-only decoder — not Tetracube, just an ABI-conformant `.so` — works
//! through deq's host path: CSR bridging, the shared-instance pool, and the
//! load_hypergraph / decode_loaded flow.
//!
//! Requires the reference plugin to be built:
//!   cargo build -p deq-decoder-reference-plugin

use std::path::PathBuf;
use std::sync::Arc;

use deq_runtime::controller::ParseByName;
use deq_runtime::decoder::DecoderType;
use deq_runtime::decoder::DynLibDecoder;
use deq_runtime::decoder::blackbox_decoder::{self, black_box_decoder_server::BlackBoxDecoder};
use deq_runtime::util::BitVector;
use serde_json::json;
use tonic::Request;

/// Locate the reference plugin's shared library in the build's target dir.
fn plugin_path() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe");
    let profile_dir = exe.ancestors().nth(2).expect("profile dir above deps/").to_path_buf();
    let name = format!(
        "{}deq_decoder_reference_plugin{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );
    profile_dir.join(name)
}

/// Two hyperedges over 3 vertices: edge 0 = {0,1}, edge 1 = {1,2}.
fn sample_hypergraph() -> blackbox_decoder::DecodingHypergraph {
    blackbox_decoder::DecodingHypergraph {
        vertex_num: 3,
        hyperedges: vec![
            blackbox_decoder::Hyperedge {
                vertices: vec![0, 1],
                probability: 0.1,
            },
            blackbox_decoder::Hyperedge {
                vertices: vec![1, 2],
                probability: 0.1,
            },
        ],
    }
}

/// Pack set vertices into deq's dense MSB-first `BitVector`.
fn syndrome(size: u64, set_vertices: &[u64]) -> BitVector {
    let mut data = vec![0u8; usize::try_from(size.div_ceil(8)).unwrap()];
    for &v in set_vertices {
        data[(v / 8) as usize] |= 1 << (7 - (v % 8) as u8);
    }
    BitVector { size, data }
}

#[tokio::test]
async fn load_and_decode_through_grpc_surface() {
    let path = plugin_path();
    assert!(
        path.exists(),
        "reference plugin not found at {} (run `cargo build -p deq-decoder-reference-plugin`)",
        path.display()
    );

    let config = json!({ "parallel": 1, "library": path });
    let decoder = Arc::new(DynLibDecoder::new(config));

    // Load the hypergraph once; decode several syndromes against the handle.
    let hid = BlackBoxDecoder::load_hypergraph(&*decoder, Request::new(sample_hypergraph()))
        .await
        .expect("load_hypergraph")
        .into_inner()
        .hid;

    let decode = |set_vertices: Vec<u64>| {
        let decoder = decoder.clone();
        async move {
            BlackBoxDecoder::decode_loaded(
                &*decoder,
                Request::new(blackbox_decoder::LoadedDecodingProblem {
                    hid,
                    syndrome: Some(syndrome(3, &set_vertices)),
                }),
            )
            .await
            .expect("decode_loaded")
            .into_inner()
            .subgraph
        }
    };

    // The reference decoder returns every hyperedge incident to a set vertex.
    assert_eq!(decode(vec![0]).await, vec![0]); // vertex 0 -> edge {0,1}
    assert_eq!(decode(vec![1]).await, vec![0, 1]); // vertex 1 -> both edges
    assert_eq!(decode(vec![]).await, Vec::<u64>::new()); // no defects -> empty
}

/// The CLI name `black-box-dyn-lib` resolves to the dynlib decoder and builds it.
#[test]
fn cli_name_selects_dynlib() {
    assert_eq!(DecoderType::from_name("black-box-dyn-lib"), Some(DecoderType::BlackBoxDynLib));
    assert!(DecoderType::variant_names().contains(&"black-box-dyn-lib"));

    // `create` returns the dynlib variant for this name.
    let config = json!({ "parallel": 1, "library": plugin_path() });
    let decoder = DecoderType::BlackBoxDynLib.create(config);
    assert!(
        decoder.as_black_box_decoder().is_some(),
        "dynlib variant must expose a blackbox decoder"
    );
}
