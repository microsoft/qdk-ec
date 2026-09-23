//! End-to-end test of the C ABI across a real `dlopen` boundary: build this
//! crate's `cdylib`, load it through the `deq_decoder_abi` host loader, and run
//! create/decode/destroy. The in-crate unit tests only call the shims directly;
//! this is the only test that exercises the actual dynamic-library path.

use std::path::PathBuf;

use deq_decoder_abi::host::{DecoderLibrary, HostDecodeRequest, LoadedDecoder};
use deq_decoder_abi::interface::{
    DEQ_DECODER_CAPABILITY_LOSS, DEQ_DECODER_CAPABILITY_REWEIGHTS, DEQ_DECODER_CAPABILITY_SEED,
};
use deq_decoder_abi::plugin::LossSiteView;

/// Locate the reference plugin's shared library next to the test binary
/// (target/<profile>/), using the platform's library prefix/suffix.
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
fn sample_hypergraph() -> (u64, Vec<f64>, Vec<u64>, Vec<u64>) {
    (3, vec![0.1, 0.1], vec![0u64, 2, 4], vec![0u64, 1, 1, 2])
}

/// Pack set vertices into a dense MSB-first bit vector (deq's `BitVector` layout).
fn pack(size: u64, set_vertices: &[u64]) -> Vec<u8> {
    let mut data = vec![0u8; usize::try_from(size.div_ceil(8)).unwrap()];
    for &d in set_vertices {
        data[(d / 8) as usize] |= 1 << (7 - (d % 8) as u8);
    }
    data
}

#[test]
fn load_and_decode_across_dlopen() {
    let path = plugin_path();
    assert!(
        path.exists(),
        "reference plugin cdylib not found at {} (run `cargo build -p deq-decoder-reference-plugin` first)",
        path.display()
    );

    // SAFETY: the path is our own freshly built reference plugin, a trusted artifact.
    let library = unsafe { DecoderLibrary::load(&path) }.expect("load reference plugin");

    let (vertex_num, probs, offsets, vertices) = sample_hypergraph();
    let mut decoder =
        LoadedDecoder::create(library, vertex_num, &probs, &offsets, &vertices, "{}").expect("create decoder");

    let mut out = Vec::new();

    // Vertex 0 set -> only edge 0 ({0,1}) is incident.
    let syndrome = pack(vertex_num, &[0]);
    decoder.decode(vertex_num, &syndrome, &mut out).expect("decode");
    assert_eq!(out, vec![0]);

    // Vertex 1 set -> both edges are incident.
    let syndrome = pack(vertex_num, &[1]);
    decoder.decode(vertex_num, &syndrome, &mut out).expect("decode");
    assert_eq!(out, vec![0, 1]);

    // No vertices set -> empty correction.
    let syndrome = pack(vertex_num, &[]);
    decoder.decode(vertex_num, &syndrome, &mut out).expect("decode");
    assert_eq!(out, [] as [u64; 0]);
    // decoder drops here, calling deq_decoder_destroy across the boundary.
}

#[test]
fn one_handle_per_worker_decodes_in_parallel() {
    let path = plugin_path();
    assert!(path.exists(), "reference plugin cdylib not found at {}", path.display());
    // SAFETY: trusted local artifact.
    let library = unsafe { DecoderLibrary::load(&path) }.expect("load");
    let (vertex_num, probs, offsets, vertices) = sample_hypergraph();

    // The ABI gives each worker its own handle; build one per thread and decode in
    // parallel. Decoders are never shared, matching how deq's host uses them.
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let (probs, offsets, vertices) = (&probs, &offsets, &vertices);
            scope.spawn(move || {
                let mut decoder =
                    LoadedDecoder::create(library, vertex_num, probs, offsets, vertices, "{}").expect("create");
                let syndrome = pack(vertex_num, &[1]);
                let mut out = Vec::new();
                for _ in 0..1000 {
                    decoder.decode(vertex_num, &syndrome, &mut out).expect("decode");
                    assert_eq!(out, vec![0, 1]);
                }
            });
        }
    });
}

fn plain(vertex_num: u64, syndrome: &[u8]) -> HostDecodeRequest<'_> {
    HostDecodeRequest {
        syndrome_size: vertex_num,
        syndrome_data: syndrome,
        ..Default::default()
    }
}

#[test]
fn plugin_advertises_every_capability_it_implements() {
    // SAFETY: the path points to this workspace's reference plugin.
    let library = unsafe { DecoderLibrary::load(plugin_path()) }.expect("load reference plugin");
    assert_eq!(
        library.capabilities(),
        DEQ_DECODER_CAPABILITY_SEED | DEQ_DECODER_CAPABILITY_REWEIGHTS | DEQ_DECODER_CAPABILITY_LOSS
    );
}

#[test]
fn every_optional_field_survives_the_dlopen_boundary() {
    // SAFETY: the path points to this workspace's reference plugin.
    let library = unsafe { DecoderLibrary::load(plugin_path()) }.expect("load reference plugin");
    let (vertex_num, probs, offsets, vertices) = sample_hypergraph();
    let mut decoder =
        LoadedDecoder::create(library, vertex_num, &probs, &offsets, &vertices, "{}").expect("create decoder");
    let syndrome = pack(vertex_num, &[1]);
    let mut out = Vec::new();

    let mut request = plain(vertex_num, &syndrome);
    request.decoder_seed = Some(0);
    decoder.decode_request(&request, &mut out).expect("seed zero");
    assert_eq!(out, vec![0, 1]);

    request.decoder_seed = Some(1);
    decoder.decode_request(&request, &mut out).expect("odd seed");
    assert_eq!(out, vec![1, 0]);

    let mut request = plain(vertex_num, &syndrome);
    request.reweights = &[(0, 0.0)];
    decoder.decode_request(&request, &mut out).expect("reweights");
    assert_eq!(out, vec![1]);

    let sites = [LossSiteView {
        source_edges: &[0],
        continuation_edges: &[],
        probability: 0.2,
        children: &[],
        heralds: &[4, 7],
    }];
    let mut request = plain(vertex_num, &syndrome);
    request.loss = Some(&sites);
    decoder.decode_request(&request, &mut out).expect("loss");
    assert_eq!(out, vec![0, 1, 0]);

    let mut request = plain(vertex_num, &syndrome);
    request.decoder_seed = Some(1);
    request.reweights = &[(0, 0.0)];
    request.loss = Some(&sites);
    decoder.decode_request(&request, &mut out).expect("combined");
    assert_eq!(
        out,
        vec![0, 1],
        "reweight drops edge 0, loss re-adds it, odd seed reverses"
    );
}

#[test]
fn buffer_retry_preserves_the_whole_request() {
    // More than 16 edges forces the host's initial output buffer to be retried.
    let vertex_num = 1u64;
    let edge_count = 20usize;
    let probs = vec![0.1; edge_count];
    let offsets: Vec<u64> = (0..=edge_count as u64).collect();
    let vertices = vec![0u64; edge_count];

    // SAFETY: the path points to this workspace's reference plugin.
    let library = unsafe { DecoderLibrary::load(plugin_path()) }.expect("load reference plugin");
    let mut decoder =
        LoadedDecoder::create(library, vertex_num, &probs, &offsets, &vertices, "{}").expect("create decoder");
    let syndrome = pack(vertex_num, &[0]);
    let mut out = Vec::new();

    let mut request = plain(vertex_num, &syndrome);
    request.decoder_seed = Some(1);
    decoder.decode_request(&request, &mut out).expect("seeded retry");
    let reversed: Vec<u64> = (0..edge_count as u64).rev().collect();
    assert_eq!(out, reversed, "the retry must restart from the same seed");
}

#[test]
fn seeded_results_do_not_depend_on_handle_history() {
    // SAFETY: the path points to this workspace's reference plugin.
    let library = unsafe { DecoderLibrary::load(plugin_path()) }.expect("load reference plugin");
    let (vertex_num, probs, offsets, vertices) = sample_hypergraph();
    let syndrome = pack(vertex_num, &[1]);

    let mut reused =
        LoadedDecoder::create(library, vertex_num, &probs, &offsets, &vertices, "{}").expect("create reused decoder");
    let mut scratch = Vec::new();
    reused
        .decode(vertex_num, &syndrome, &mut scratch)
        .expect("legacy decode");
    let mut reweighted = plain(vertex_num, &syndrome);
    reweighted.reweights = &[(0, 0.0)];
    reused
        .decode_request(&reweighted, &mut scratch)
        .expect("reweighted request");

    let mut fresh =
        LoadedDecoder::create(library, vertex_num, &probs, &offsets, &vertices, "{}").expect("create fresh decoder");

    let mut request = plain(vertex_num, &syndrome);
    request.decoder_seed = Some(1);
    let mut from_reused = Vec::new();
    let mut from_fresh = Vec::new();
    reused
        .decode_request(&request, &mut from_reused)
        .expect("decode with reused handle");
    fresh
        .decode_request(&request, &mut from_fresh)
        .expect("decode with fresh handle");
    assert_eq!(
        from_reused, from_fresh,
        "a prior request must not leak into a later one"
    );
}
