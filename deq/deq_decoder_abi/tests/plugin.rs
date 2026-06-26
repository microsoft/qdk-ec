//! In-process tests for the plugin-side ABI shims. These exercise the generic
//! `*_impl` functions directly (the same code the `declare_decoder!` macro exports),
//! without needing a real shared library.

use core::ffi::c_void;

use deq_decoder_abi::interface::{
    STATUS_BUFFER_TOO_SMALL, STATUS_INVALID_ARG, STATUS_OK, STATUS_PANIC, STATUS_POISONED,
};
use deq_decoder_abi::plugin::{
    DeqDecoder, HypergraphView, OutputBuffer, SyndromeView, create_impl, decode_impl, destroy_impl,
};

/// Returns every hyperedge that contains at least one set vertex.
struct IncidenceDecoder {
    edges: Vec<Vec<u64>>,
}

impl DeqDecoder for IncidenceDecoder {
    fn create(graph: HypergraphView<'_>, _config_json: &[u8]) -> Result<Self, String> {
        let edges = graph.edges().map(|(_, vertices)| vertices.to_vec()).collect();
        Ok(Self { edges })
    }

    fn decode(&mut self, syndrome: SyndromeView<'_>, out: &mut OutputBuffer) -> Result<(), String> {
        let set_vertices: Vec<u64> = syndrome.sparse_indices().collect();
        for (index, vertices) in self.edges.iter().enumerate() {
            if vertices.iter().any(|v| set_vertices.contains(v)) {
                out.push(index as u64);
            }
        }
        Ok(())
    }
}

/// Panics from `decode` to exercise the poison path.
struct PanicDecoder;

impl DeqDecoder for PanicDecoder {
    fn create(_graph: HypergraphView<'_>, _config_json: &[u8]) -> Result<Self, String> {
        Ok(Self)
    }

    fn decode(&mut self, _syndrome: SyndromeView<'_>, _out: &mut OutputBuffer) -> Result<(), String> {
        panic!("boom");
    }
}

/// Two hyperedges: {0,1} and {1,2}, both with probability 0.1, over 3 vertices.
fn sample_csr() -> (u64, Vec<f64>, Vec<u64>, Vec<u64>) {
    let vertex_num = 3;
    let edge_probs = vec![0.1, 0.1];
    let edge_offsets = vec![0u64, 2, 4];
    let edge_vertices = vec![0u64, 1, 1, 2];
    (vertex_num, edge_probs, edge_offsets, edge_vertices)
}

fn create<T: DeqDecoder>(
    vertex_num: u64,
    probs: &[f64],
    offsets: &[u64],
    vertices: &[u64],
    config: &core::ffi::CStr,
) -> Result<*mut c_void, i32> {
    let mut handle: *mut c_void = core::ptr::null_mut();
    let status = unsafe {
        create_impl::<T>(
            vertex_num,
            probs.len() as u64,
            probs.as_ptr(),
            offsets.as_ptr(),
            vertices.as_ptr(),
            vertices.len(),
            config.as_ptr(),
            &raw mut handle,
        )
    };
    if status == STATUS_OK { Ok(handle) } else { Err(status) }
}

/// Pack a sparse list of set vertices into a dense MSB-first bit buffer of
/// `size` bits, mirroring deq's `BitVector` layout.
fn pack(size: u64, set_vertices: &[u64]) -> Vec<u8> {
    let mut data = vec![0u8; usize::try_from(size.div_ceil(8)).unwrap()];
    for &d in set_vertices {
        data[(d / 8) as usize] |= 1 << (7 - (d % 8) as u8);
    }
    data
}

fn decode<T: DeqDecoder>(handle: *mut c_void, size: u64, set_vertices: &[u64], cap: usize) -> (i32, Vec<u64>, usize) {
    let mut out = vec![0u64; cap];
    let mut written = 0usize;
    let data = pack(size, set_vertices);
    let data_ptr = if data.is_empty() {
        core::ptr::null()
    } else {
        data.as_ptr()
    };
    let out_ptr = if cap == 0 {
        core::ptr::null_mut()
    } else {
        out.as_mut_ptr()
    };
    let status = unsafe { decode_impl::<T>(handle, size, data_ptr, data.len(), out_ptr, cap, &raw mut written) };
    out.truncate(written.min(cap));
    (status, out, written)
}

#[test]
fn decode_happy_path() {
    let (vertex_num, probs, offsets, vertices) = sample_csr();
    let handle = create::<IncidenceDecoder>(vertex_num, &probs, &offsets, &vertices, c"{}").unwrap();

    // Vertex 0 set -> only edge 0 ({0,1}) is incident.
    let (status, out, written) = decode::<IncidenceDecoder>(handle, 3, &[0], 8);
    assert_eq!(status, STATUS_OK);
    assert_eq!(written, 1);
    assert_eq!(out, vec![0]);

    // Vertex 1 set -> both edges are incident.
    let (status, out, written) = decode::<IncidenceDecoder>(handle, 3, &[1], 8);
    assert_eq!(status, STATUS_OK);
    assert_eq!(written, 2);
    assert_eq!(out, vec![0, 1]);

    unsafe { destroy_impl::<IncidenceDecoder>(handle) };
}

#[test]
fn decode_buffer_too_small_reports_needed() {
    let (vertex_num, probs, offsets, vertices) = sample_csr();
    let handle = create::<IncidenceDecoder>(vertex_num, &probs, &offsets, &vertices, c"{}").unwrap();

    // Vertex 1 hits both edges, but the buffer holds 0 -> needed = 2.
    let (status, _out, written) = decode::<IncidenceDecoder>(handle, 3, &[1], 0);
    assert_eq!(status, STATUS_BUFFER_TOO_SMALL);
    assert_eq!(written, 2);

    unsafe { destroy_impl::<IncidenceDecoder>(handle) };
}

#[test]
fn create_rejects_bad_offsets() {
    // edge_offsets[0] != 0 is invalid.
    let status = create::<IncidenceDecoder>(3, &[0.1, 0.1], &[1, 2, 4], &[0, 1, 1, 2], c"{}").unwrap_err();
    assert_eq!(status, STATUS_INVALID_ARG);
}

#[test]
fn create_rejects_out_of_range_vertex() {
    // Vertex 9 >= vertex_num 3.
    let status = create::<IncidenceDecoder>(3, &[0.1], &[0, 2], &[0, 9], c"{}").unwrap_err();
    assert_eq!(status, STATUS_INVALID_ARG);
}

#[test]
fn create_rejects_probability_at_one() {
    let status = create::<IncidenceDecoder>(3, &[1.0], &[0, 2], &[0, 1], c"{}").unwrap_err();
    assert_eq!(status, STATUS_INVALID_ARG);
}

#[test]
fn create_rejects_non_positive_probability() {
    for prob in [0.0, -0.1] {
        let status = create::<IncidenceDecoder>(3, &[prob], &[0, 2], &[0, 1], c"{}").unwrap_err();
        assert_eq!(status, STATUS_INVALID_ARG, "probability {prob} should be rejected");
    }
}

#[test]
fn panic_in_decode_poisons_handle() {
    let (vertex_num, probs, offsets, vertices) = sample_csr();
    let handle = create::<PanicDecoder>(vertex_num, &probs, &offsets, &vertices, c"{}").unwrap();

    // Silence the default panic hook so the expected panic does not spam stderr.
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let (status, _out, _written) = decode::<PanicDecoder>(handle, 3, &[0], 8);
    std::panic::set_hook(previous);
    assert_eq!(status, STATUS_PANIC);

    // The handle is now poisoned; subsequent calls fail without invoking the decoder.
    let (status, _out, _written) = decode::<PanicDecoder>(handle, 3, &[0], 8);
    assert_eq!(status, STATUS_POISONED);

    unsafe { destroy_impl::<PanicDecoder>(handle) };
}

#[test]
fn empty_graph_and_empty_syndrome() {
    let handle = create::<IncidenceDecoder>(0, &[], &[0], &[], c"{}").unwrap();
    let (status, out, written) = decode::<IncidenceDecoder>(handle, 0, &[], 8);
    assert_eq!(status, STATUS_OK);
    assert_eq!(written, 0);
    assert!(out.is_empty());
    unsafe { destroy_impl::<IncidenceDecoder>(handle) };
}
