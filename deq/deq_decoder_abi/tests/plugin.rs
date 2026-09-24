//! In-process tests for the plugin-side ABI shims. These exercise the generic
//! `*_impl` functions directly (the same code the `declare_decoder!` macro exports),
//! without needing a real shared library.

use core::ffi::c_void;

use deq_decoder_abi::interface::{
    DEQ_DECODER_CAPABILITY_LOSS, DEQ_DECODER_CAPABILITY_REWEIGHTS, DEQ_DECODER_CAPABILITY_SEED, DeqDecoderCapabilities,
};
use deq_decoder_abi::interface::{
    DeqDecoderDecodeRequest, DeqDecoderEdgeReweight, DeqDecoderLossInfo, DeqDecoderLossSite, STATUS_BUFFER_TOO_SMALL,
    STATUS_INVALID_ARG, STATUS_OK, STATUS_PANIC, STATUS_POISONED,
};
use deq_decoder_abi::plugin::{
    DecodeRequest, DeqDecoder, HypergraphView, OutputBuffer, SyndromeView, create_impl, decode_impl,
    decode_request_impl, destroy_impl,
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
fn create_rejects_negative_probability() {
    for prob in [-0.1, f64::NAN, f64::INFINITY] {
        let status = create::<IncidenceDecoder>(3, &[prob], &[0, 2], &[0, 1], c"{}").unwrap_err();
        assert_eq!(status, STATUS_INVALID_ARG, "probability {prob} should be rejected");
    }
}

#[test]
fn create_accepts_zero_probability_as_a_dormant_edge() {
    create::<IncidenceDecoder>(3, &[0.0], &[0, 2], &[0, 1], c"{}").unwrap();
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
    assert_eq!(out, [] as [u64; 0]);
    unsafe { destroy_impl::<IncidenceDecoder>(handle) };
}

/// Encodes each optional field into the output for request-shim tests.
struct EchoDecoder;

impl DeqDecoder for EchoDecoder {
    const CAPABILITIES: DeqDecoderCapabilities =
        DEQ_DECODER_CAPABILITY_SEED | DEQ_DECODER_CAPABILITY_REWEIGHTS | DEQ_DECODER_CAPABILITY_LOSS;

    fn create(_graph: HypergraphView<'_>, _config_json: &[u8]) -> Result<Self, String> {
        Ok(Self)
    }

    fn decode(&mut self, _syndrome: SyndromeView<'_>, _out: &mut OutputBuffer) -> Result<(), String> {
        Err("legacy decode not used by this test".to_string())
    }

    fn decode_request(&mut self, request: DecodeRequest<'_>, out: &mut OutputBuffer) -> Result<(), String> {
        // Reserve 999 for an absent seed; present seeds start at 1000.
        match request.decoder_seed {
            Some(seed) => out.push(1000 + seed),
            None => out.push(999),
        }
        for reweight in request.reweights {
            out.push(2000 + reweight.edge);
        }
        match request.loss {
            None => out.push(3999),
            Some(loss) => {
                out.push(3000 + loss.sites().len() as u64);
                for site in loss.sites() {
                    for &edge in site.source_edges {
                        out.push(4000 + edge);
                    }
                    for &child in site.children {
                        out.push(5000 + child);
                    }
                    for &herald in site.heralds {
                        out.push(6000 + herald);
                    }
                }
            }
        }
        Ok(())
    }
}

struct DefaultRequestDecoder;

impl DeqDecoder for DefaultRequestDecoder {
    fn create(_graph: HypergraphView<'_>, _config_json: &[u8]) -> Result<Self, String> {
        Ok(Self)
    }

    fn decode(&mut self, syndrome: SyndromeView<'_>, out: &mut OutputBuffer) -> Result<(), String> {
        out.push(syndrome.sparse_indices().count() as u64);
        Ok(())
    }
}

fn decode_request<T: DeqDecoder>(
    handle: *mut c_void,
    request: &DeqDecoderDecodeRequest,
    cap: usize,
) -> (i32, Vec<u64>, usize) {
    let mut out = vec![0u64; cap];
    let mut written = 0usize;
    let out_ptr = if cap == 0 {
        core::ptr::null_mut()
    } else {
        out.as_mut_ptr()
    };
    let status =
        unsafe { decode_request_impl::<T>(handle, core::ptr::from_ref(request), out_ptr, cap, &raw mut written) };
    out.truncate(written.min(cap));
    (status, out, written)
}

/// A loss site with probability 0.2 and no continuation edges. The `'static` slices
/// keep its raw pointers valid for the whole test.
fn loss_site(source_edges: &'static [u64], children: &'static [u64], heralds: &'static [u64]) -> DeqDecoderLossSite {
    DeqDecoderLossSite {
        source_edges: source_edges.as_ptr(),
        source_edge_count: source_edges.len(),
        continuation_edges: core::ptr::null(),
        continuation_edge_count: 0,
        probability: 0.2,
        children: children.as_ptr(),
        child_count: children.len(),
        heralds: heralds.as_ptr(),
        herald_count: heralds.len(),
    }
}

fn plain_request(size: u64, data: &[u8]) -> DeqDecoderDecodeRequest {
    DeqDecoderDecodeRequest {
        syndrome_size: size,
        syndrome_data: data.as_ptr(),
        has_decoder_seed: false,
        decoder_seed: 0,
        reweights: core::ptr::null(),
        reweight_count: 0,
        loss: core::ptr::null(),
    }
}

#[test]
fn absent_seed_zero_seed_and_nonzero_seed_stay_distinct() {
    let (vertex_num, probs, offsets, vertices) = sample_csr();
    let handle = create::<EchoDecoder>(vertex_num, &probs, &offsets, &vertices, c"{}").unwrap();
    let data = pack(vertex_num, &[0]);

    let absent = plain_request(vertex_num, &data);
    let (status, out, _) = decode_request::<EchoDecoder>(handle, &absent, 8);
    assert_eq!(status, STATUS_OK);
    assert_eq!(out, vec![999, 3999], "absent seed must not become seed zero");

    let mut zero = plain_request(vertex_num, &data);
    zero.has_decoder_seed = true;
    zero.decoder_seed = 0;
    let (_, out, _) = decode_request::<EchoDecoder>(handle, &zero, 8);
    assert_eq!(out, vec![1000, 3999], "seed zero is an ordinary deterministic seed");

    let mut seven = plain_request(vertex_num, &data);
    seven.has_decoder_seed = true;
    seven.decoder_seed = 7;
    let (_, out, _) = decode_request::<EchoDecoder>(handle, &seven, 8);
    assert_eq!(out, vec![1007, 3999]);

    unsafe { destroy_impl::<EchoDecoder>(handle) };
}

#[test]
fn present_but_empty_loss_differs_from_absent_loss() {
    let (vertex_num, probs, offsets, vertices) = sample_csr();
    let handle = create::<EchoDecoder>(vertex_num, &probs, &offsets, &vertices, c"{}").unwrap();
    let data = pack(vertex_num, &[0]);

    let empty = DeqDecoderLossInfo {
        sites: core::ptr::null(),
        site_count: 0,
    };
    let mut request = plain_request(vertex_num, &data);
    request.loss = core::ptr::from_ref(&empty);
    let (status, out, _) = decode_request::<EchoDecoder>(handle, &request, 8);
    assert_eq!(status, STATUS_OK);
    assert_eq!(out, vec![999, 3000], "loss supplied with no site is not absent loss");

    unsafe { destroy_impl::<EchoDecoder>(handle) };
}

#[test]
fn reweights_and_loss_arrive_together_intact() {
    let (vertex_num, probs, offsets, vertices) = sample_csr();
    let handle = create::<EchoDecoder>(vertex_num, &probs, &offsets, &vertices, c"{}").unwrap();
    let data = pack(vertex_num, &[0]);

    let reweights = [DeqDecoderEdgeReweight {
        edge: 1,
        probability: 0.25,
    }];
    let sites = [loss_site(&[0], &[0], &[4, 7])];
    let loss = DeqDecoderLossInfo {
        sites: sites.as_ptr(),
        site_count: sites.len(),
    };
    let mut request = plain_request(vertex_num, &data);
    request.has_decoder_seed = true;
    request.decoder_seed = 3;
    request.reweights = reweights.as_ptr();
    request.reweight_count = reweights.len();
    request.loss = core::ptr::from_ref(&loss);

    let (status, out, _) = decode_request::<EchoDecoder>(handle, &request, 16);
    assert_eq!(status, STATUS_OK);
    assert_eq!(out, vec![1003, 2001, 3001, 4000, 5000, 6004, 6007]);

    unsafe { destroy_impl::<EchoDecoder>(handle) };
}

#[test]
fn default_decode_request_rejects_optional_fields_instead_of_ignoring_them() {
    let (vertex_num, probs, offsets, vertices) = sample_csr();
    let handle = create::<DefaultRequestDecoder>(vertex_num, &probs, &offsets, &vertices, c"{}").unwrap();
    let data = pack(vertex_num, &[0, 1]);

    let plain = plain_request(vertex_num, &data);
    let (status, out, _) = decode_request::<DefaultRequestDecoder>(handle, &plain, 4);
    assert_eq!(status, STATUS_OK);
    assert_eq!(out, vec![2], "a plain request still reaches legacy decode");

    let mut seeded = plain_request(vertex_num, &data);
    seeded.has_decoder_seed = true;
    seeded.decoder_seed = 0;
    let (status, _, _) = decode_request::<DefaultRequestDecoder>(handle, &seeded, 4);
    assert_eq!(status, deq_decoder_abi::STATUS_ERROR, "seed zero must be rejected");

    unsafe { destroy_impl::<DefaultRequestDecoder>(handle) };
}

#[test]
fn shim_rejects_out_of_range_and_malformed_references() {
    let (vertex_num, probs, offsets, vertices) = sample_csr();
    let handle = create::<EchoDecoder>(vertex_num, &probs, &offsets, &vertices, c"{}").unwrap();
    let data = pack(vertex_num, &[0]);

    let bad_edge = [DeqDecoderEdgeReweight {
        edge: 99,
        probability: 0.5,
    }];
    let mut request = plain_request(vertex_num, &data);
    request.reweights = bad_edge.as_ptr();
    request.reweight_count = bad_edge.len();
    let (status, _, _) = decode_request::<EchoDecoder>(handle, &request, 8);
    assert_eq!(status, STATUS_INVALID_ARG, "reweight edge out of range");

    let bad_probability = [DeqDecoderEdgeReweight {
        edge: 0,
        probability: f64::NAN,
    }];
    let mut request = plain_request(vertex_num, &data);
    request.reweights = bad_probability.as_ptr();
    request.reweight_count = bad_probability.len();
    let (status, _, _) = decode_request::<EchoDecoder>(handle, &request, 8);
    assert_eq!(status, STATUS_INVALID_ARG, "reweight probability must be finite");

    let mut request = plain_request(vertex_num, &data);
    request.reweight_count = 1;
    let (status, _, _) = decode_request::<EchoDecoder>(handle, &request, 8);
    assert_eq!(status, STATUS_INVALID_ARG, "null pointer with a non-zero count");

    let sites = [loss_site(&[7], &[], &[])];
    let loss = DeqDecoderLossInfo {
        sites: sites.as_ptr(),
        site_count: sites.len(),
    };
    let mut request = plain_request(vertex_num, &data);
    request.loss = core::ptr::from_ref(&loss);
    let (status, _, _) = decode_request::<EchoDecoder>(handle, &request, 8);
    assert_eq!(status, STATUS_INVALID_ARG, "loss site edge out of range");

    let sites = [loss_site(&[], &[3], &[])];
    let loss = DeqDecoderLossInfo {
        sites: sites.as_ptr(),
        site_count: sites.len(),
    };
    let mut request = plain_request(vertex_num, &data);
    request.loss = core::ptr::from_ref(&loss);
    let (status, _, _) = decode_request::<EchoDecoder>(handle, &request, 8);
    assert_eq!(status, STATUS_INVALID_ARG, "child index out of range for the site list");

    let mut request = plain_request(vertex_num, &data);
    request.syndrome_size = vertex_num + 1;
    let (status, _, _) = decode_request::<EchoDecoder>(handle, &request, 8);
    assert_eq!(status, STATUS_INVALID_ARG, "syndrome must match the loaded graph");

    let (status, _, _) = decode_request::<EchoDecoder>(handle, &plain_request(vertex_num, &data), 8);
    assert_eq!(status, STATUS_OK, "the handle survives rejected requests");

    unsafe { destroy_impl::<EchoDecoder>(handle) };
}

#[test]
fn decode_request_repeats_the_seed_on_a_buffer_retry() {
    let (vertex_num, probs, offsets, vertices) = sample_csr();
    let handle = create::<EchoDecoder>(vertex_num, &probs, &offsets, &vertices, c"{}").unwrap();
    let data = pack(vertex_num, &[0]);
    let mut request = plain_request(vertex_num, &data);
    request.has_decoder_seed = true;
    request.decoder_seed = 5;

    let (status, _, needed) = decode_request::<EchoDecoder>(handle, &request, 1);
    assert_eq!(status, STATUS_BUFFER_TOO_SMALL);
    assert_eq!(needed, 2);

    let (status, out, _) = decode_request::<EchoDecoder>(handle, &request, needed);
    assert_eq!(status, STATUS_OK);
    assert_eq!(out, vec![1005, 3999], "the retry decodes the same seeded request");

    unsafe { destroy_impl::<EchoDecoder>(handle) };
}
