//! End-to-end test of the C ABI across a real `dlopen` boundary: build this
//! crate's `cdylib`, load it through the `deq_decoder_abi` host loader, and run
//! create/decode/destroy. The in-crate unit tests only call the shims directly;
//! this is the only test that exercises the actual dynamic-library path.

use std::path::PathBuf;

use deq_decoder_abi::host::{DecoderLibrary, LoadedDecoder};

/// Locate the reference plugin's shared library next to the test binary
/// (target/<profile>/), using the platform's library prefix/suffix.
fn plugin_path() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe");
    // exe is target/<profile>/deps/<test>-<hash>; the cdylib is two levels up.
    let profile_dir = exe.ancestors().nth(2).expect("profile dir above deps/").to_path_buf();
    let name = format!(
        "{}deq_decoder_reference_plugin{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );
    profile_dir.join(name)
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
    assert!(out.is_empty());
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
