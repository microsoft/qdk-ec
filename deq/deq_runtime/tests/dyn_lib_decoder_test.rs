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
use deq_runtime::decoder::blackbox_decoder::{self, black_box_decoder_server::BlackBoxDecoder};
use deq_runtime::decoder::{DecoderType, DynDecoder, DynLibDecoder};
use deq_runtime::util::BitVector;
use serde_json::json;
use tonic::Request;

/// Locate the reference plugin's shared library in the build's target dir.
fn plugin_path() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe");
    let name = format!(
        "{}deq_decoder_reference_plugin{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );
    exe.ancestors()
        .map(|directory| directory.join(&name))
        .find(|candidate| candidate.is_file())
        .unwrap_or_else(|| panic!("could not find reference plugin {name} from {}", exe.display()))
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

    let decode = |set_vertices: Vec<u64>, decoder_seed| {
        let decoder = decoder.clone();
        async move {
            BlackBoxDecoder::decode_loaded(
                &*decoder,
                Request::new(blackbox_decoder::LoadedDecodingProblem {
                    hid,
                    syndrome: Some(syndrome(3, &set_vertices)),
                    decoder_seed,
                    ..Default::default()
                }),
            )
            .await
            .expect("decode_loaded")
            .into_inner()
            .subgraph
        }
    };

    // The reference decoder returns every positive-probability hyperedge incident
    // to a set vertex.
    assert_eq!(decode(vec![0], None).await, vec![0]); // vertex 0 -> edge {0,1}
    assert_eq!(decode(vec![1], None).await, vec![0, 1]); // vertex 1 -> both edges
    assert_eq!(decode(vec![], None).await, Vec::<u64>::new()); // no defects -> empty

    // Odd seeds reverse the reference result; zero follows the even-seed path.
    assert_eq!(decode(vec![1], Some(0)).await, vec![0, 1]);
    assert_eq!(decode(vec![1], Some(1)).await, vec![1, 0]);
}

#[tokio::test]
async fn capabilities_come_from_the_plugin() {
    let decoder = Arc::new(DynLibDecoder::new(json!({ "parallel": 1, "library": plugin_path() })));
    let capabilities = BlackBoxDecoder::get_capabilities(&*decoder, Request::new(()))
        .await
        .expect("get_capabilities")
        .into_inner();
    assert_eq!(
        capabilities.features,
        vec![
            blackbox_decoder::DecoderFeature::Reweights as i32,
            blackbox_decoder::DecoderFeature::Loss as i32,
            blackbox_decoder::DecoderFeature::Seed as i32,
        ]
    );
}

#[tokio::test]
async fn a_capable_plugin_keeps_stable_edge_numbering_for_dormant_edges() {
    // Capability-aware plugins receive dormant edges and return stable indices.
    let hypergraph = blackbox_decoder::DecodingHypergraph {
        vertex_num: 2,
        hyperedges: vec![
            blackbox_decoder::Hyperedge {
                vertices: vec![0],
                probability: 0.1,
            },
            blackbox_decoder::Hyperedge {
                vertices: vec![0],
                probability: 0.0,
            },
            blackbox_decoder::Hyperedge {
                vertices: vec![1],
                probability: 0.1,
            },
        ],
    };
    let decoder = Arc::new(DynLibDecoder::new(json!({ "parallel": 1, "library": plugin_path() })));
    let hid = BlackBoxDecoder::load_hypergraph(&*decoder, Request::new(hypergraph))
        .await
        .expect("load_hypergraph")
        .into_inner()
        .hid;

    let decode = |set_vertices: Vec<u64>, reweights: Vec<blackbox_decoder::EdgeReweight>| {
        let decoder = decoder.clone();
        async move {
            BlackBoxDecoder::decode_loaded(
                &*decoder,
                Request::new(blackbox_decoder::LoadedDecodingProblem {
                    hid,
                    syndrome: Some(syndrome(2, &set_vertices)),
                    reweights,
                    ..Default::default()
                }),
            )
            .await
            .expect("decode_loaded")
            .into_inner()
            .subgraph
        }
    };

    // Had the dormant edge been dropped, edge 2 would come back renumbered as 1.
    assert_eq!(decode(vec![1], vec![]).await, vec![2]);
    // Vertex 0 touches edges 0 and 1, but a dormant edge is not selected.
    assert_eq!(decode(vec![0], vec![]).await, vec![0]);
    // A reweight activates the dormant edge through its stable index.
    let activate = blackbox_decoder::EdgeReweight {
        edge: 1,
        probability: 0.5,
    };
    assert_eq!(decode(vec![0], vec![activate]).await, vec![0, 1]);
}

#[tokio::test]
async fn isolated_zero_vertex_is_supported() {
    let path = plugin_path();
    assert!(
        path.exists(),
        "reference plugin not found at {} (run `cargo build -p deq-decoder-reference-plugin`)",
        path.display()
    );
    let decoder = DynDecoder::BlackBoxDynLib(Arc::new(DynLibDecoder::new(json!({
        "parallel": 1,
        "library": path,
    }))));
    let hypergraph = blackbox_decoder::DecodingHypergraph {
        vertex_num: 2,
        hyperedges: vec![blackbox_decoder::Hyperedge {
            vertices: vec![0],
            probability: 0.1,
        }],
    };
    let syndrome = syndrome(2, &[0]);

    decoder
        .decode(blackbox_decoder::DecodingProblem {
            hypergraph: Some(hypergraph.clone()),
            syndrome: Some(syndrome.clone()),
            loss: None,
            decoder_seed: None,
        })
        .await
        .unwrap();

    let hid = decoder.load_hypergraph(hypergraph).await.unwrap().hid;
    decoder
        .decode_loaded(blackbox_decoder::LoadedDecodingProblem {
            hid,
            syndrome: Some(syndrome),
            ..Default::default()
        })
        .await
        .unwrap();
}

/// The CLI name `black-box-dyn-lib` resolves to the dynlib decoder and builds it.
#[test]
fn cli_name_selects_dynlib() {
    assert_eq!(DecoderType::from_name("black-box-dyn-lib"), Some(DecoderType::BlackBoxDynLib));
    assert!(DecoderType::variant_names().contains(&"black-box-dyn-lib"));

    // `create` returns the dynlib variant for this name.
    let config = json!({ "parallel": 1, "library": plugin_path() });
    let decoder = DecoderType::BlackBoxDynLib.create(config);
    assert!(matches!(decoder, DynDecoder::BlackBoxDynLib(_)));
}
