//! The frozen ABI contract: version, status codes, exported symbol names, and the
//! `extern "C"` function-pointer types. Both the plugin layer and the host loader build
//! on these definitions, so they are the single source of truth for the boundary.

use core::ffi::{c_char, c_void};

/// ABI revision. The host refuses to load a plugin whose
/// [`deq_decoder_abi_version`](SYM_ABI_VERSION) does not match this value.
///
/// Bump this only for breaking changes to any signature, symbol, or calling
/// convention below.
pub const ABI_VERSION: u32 = 1;

/// The decode call succeeded; `*out_len` holds the number of subgraph indices written.
pub const STATUS_OK: i32 = 0;
/// The output buffer was too small; `*out_len` holds the required length and the
/// buffer contents are unspecified. The caller should retry with a larger buffer.
pub const STATUS_BUFFER_TOO_SMALL: i32 = 1;
/// A generic, recoverable error occurred; see [`last_error`](SYM_LAST_ERROR).
pub const STATUS_ERROR: i32 = -1;
/// An argument violated the ABI contract (e.g. a malformed hypergraph or a null
/// pointer where one is not permitted); see [`last_error`](SYM_LAST_ERROR).
pub const STATUS_INVALID_ARG: i32 = -2;
/// The plugin caught a panic/exception while servicing the call; see
/// [`last_error`](SYM_LAST_ERROR). The handle is now poisoned.
pub const STATUS_PANIC: i32 = -3;
/// The handle was previously poisoned by a panic and can no longer be used.
pub const STATUS_POISONED: i32 = -4;

/// Symbol exported by every plugin: `fn() -> u32` returning [`ABI_VERSION`].
pub const SYM_ABI_VERSION: &[u8] = b"deq_decoder_abi_version\0";
/// Symbol exported by every plugin: the [`CreateFn`] constructor.
pub const SYM_CREATE: &[u8] = b"deq_decoder_create\0";
/// Symbol exported by every plugin: the [`DecodeFn`] hot-path entry point.
pub const SYM_DECODE: &[u8] = b"deq_decoder_decode\0";
/// Symbol exported by every plugin: the [`DestroyFn`] destructor.
pub const SYM_DESTROY: &[u8] = b"deq_decoder_destroy\0";
/// Symbol exported by every plugin: the [`LastErrorFn`] thread-local error reader.
pub const SYM_LAST_ERROR: &[u8] = b"deq_decoder_last_error\0";
/// Optional symbol for request-based decoding. A plugin that
/// exports it must also export [`SYM_CAPABILITIES`]; exporting exactly one of the
/// pair is invalid and the host rejects such a library.
pub const SYM_DECODE_REQUEST: &[u8] = b"deq_decoder_decode_request\0";
/// Optional symbol: the [`CapabilitiesFn`] library-level capability bitmask. Paired
/// with [`SYM_DECODE_REQUEST`].
pub const SYM_CAPABILITIES: &[u8] = b"deq_decoder_capabilities\0";

/// Bits in the library-level capability bitmask returned by [`CapabilitiesFn`].
///
/// deq's internal `DecoderFeatures` uses the same bit values. Compile-time assertions
/// and generated C macros keep both representations aligned.
pub type DeqDecoderCapabilities = u64;

/// The plugin accepts [`DeqDecoderDecodeRequest::decoder_seed`].
pub const DEQ_DECODER_CAPABILITY_SEED: DeqDecoderCapabilities = 1 << 0;
/// The plugin accepts [`DeqDecoderDecodeRequest::reweights`].
pub const DEQ_DECODER_CAPABILITY_REWEIGHTS: DeqDecoderCapabilities = 1 << 1;
/// The plugin accepts [`DeqDecoderDecodeRequest::loss`].
pub const DEQ_DECODER_CAPABILITY_LOSS: DeqDecoderCapabilities = 1 << 2;

/// The capability bits a request needs, given which optional fields it carries.
pub(crate) fn required_capabilities(
    has_decoder_seed: bool,
    has_reweights: bool,
    has_loss: bool,
) -> DeqDecoderCapabilities {
    [
        (has_decoder_seed, DEQ_DECODER_CAPABILITY_SEED),
        (has_reweights, DEQ_DECODER_CAPABILITY_REWEIGHTS),
        (has_loss, DEQ_DECODER_CAPABILITY_LOSS),
    ]
    .into_iter()
    .filter(|&(present, _)| present)
    .fold(0, |bits, (_, bit)| bits | bit)
}

/// The request field names behind `bits`, comma-separated, for error messages.
pub(crate) fn describe_capabilities(bits: DeqDecoderCapabilities) -> String {
    [
        (DEQ_DECODER_CAPABILITY_SEED, "decoder_seed"),
        (DEQ_DECODER_CAPABILITY_REWEIGHTS, "reweights"),
        (DEQ_DECODER_CAPABILITY_LOSS, "loss"),
    ]
    .into_iter()
    .filter(|&(bit, _)| bits & bit != 0)
    .map(|(_, name)| name)
    .collect::<Vec<_>>()
    .join(", ")
}

/// Bits per byte in the packed syndrome representation.
pub const DEQ_DECODER_SYNDROME_BITS_PER_BYTE: u64 = 8;

/// A prior assignment for one request. `probability` replaces the loaded prior of
/// hyperedge `edge`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeqDecoderEdgeReweight {
    /// Hyperedge index into the graph given to [`CreateFn`].
    pub edge: u64,
    /// Replacement probability, finite and in `[0, 1]`.
    pub probability: f64,
}

/// One possible loss site, mirroring deq's internal `LossSite`. Every pointer/count
/// pair is borrowed for the duration of the call; a zero count permits a null pointer.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DeqDecoderLossSite {
    /// Hyperedge indices of the SOURCE generators at this loss location.
    pub source_edges: *const u64,
    /// Number of entries in `source_edges`.
    pub source_edge_count: usize,
    /// Hyperedge indices of the CONTINUATION generators.
    pub continuation_edges: *const u64,
    /// Number of entries in `continuation_edges`.
    pub continuation_edge_count: usize,
    /// Declared probability that loss starts at this site, finite and in `[0, 1]`.
    pub probability: f64,
    /// Forward parent-to-child links: indices into the enclosing `sites` array.
    pub children: *const u64,
    /// Number of entries in `children`.
    pub child_count: usize,
    /// Herald identities. These are opaque identifiers, not indices. The shim does
    /// not range-check them, and the same value may appear at several sites.
    pub heralds: *const u64,
    /// Number of entries in `heralds`.
    pub herald_count: usize,
}

/// Structured loss observation for one shot: a borrowed list of possible sites.
///
/// A non-null [`DeqDecoderDecodeRequest::loss`] with `site_count == 0` is distinct
/// from a null one: it means loss information was supplied and no site was possible.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DeqDecoderLossInfo {
    /// The possible loss sites.
    pub sites: *const DeqDecoderLossSite,
    /// Number of entries in `sites`.
    pub site_count: usize,
}

/// A decode request. Every pointer reachable from this structure is
/// borrowed for the duration of [`DecodeRequestFn`]; the plugin must not retain any
/// of them after the call returns.
///
/// The layout is frozen within one ABI revision: adding, removing, retyping, or
/// reordering a field requires bumping [`ABI_VERSION`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DeqDecoderDecodeRequest {
    /// Logical number of syndrome bits; must equal the graph's `vertex_num`.
    pub syndrome_size: u64,
    /// Dense MSB-first packed syndrome, `syndrome_size.div_ceil(8)` bytes. May be
    /// null only when `syndrome_size == 0`.
    pub syndrome_data: *const u8,
    /// Whether `decoder_seed` carries a value. This field and `decoder_seed` together
    /// represent Rust's `Option<u64>`.
    pub has_decoder_seed: bool,
    /// The seed, meaningful only when `has_decoder_seed` is true. Zero is valid.
    pub decoder_seed: u64,
    /// Borrowed shot-scoped prior assignments; may be null when `reweight_count` is 0.
    pub reweights: *const DeqDecoderEdgeReweight,
    /// Number of entries in `reweights`.
    pub reweight_count: usize,
    /// Borrowed structured loss, or null when none was supplied.
    pub loss: *const DeqDecoderLossInfo,
}

/// `deq_decoder_capabilities() -> u64`.
///
/// Returns the library-level [capability bitmask](DeqDecoderCapabilities). Fixed for
/// the library: it does not vary with the hypergraph or the JSON configuration.
///
/// # Safety
///
/// Takes no arguments and reads no caller memory; it is `unsafe` only because it
/// crosses the ABI boundary.
pub type CapabilitiesFn = unsafe extern "C" fn() -> DeqDecoderCapabilities;

/// `deq_decoder_decode_request(...) -> i32`.
///
/// Decodes one request. Output handling matches [`DecodeFn`]: the selected
/// subgraph is written to `subgraph`/`subgraph_capacity` and `*subgraph_count` is set
/// to the number of `uint64_t` indices written, or to the required count together
/// with [`STATUS_BUFFER_TOO_SMALL`].
///
/// A retry after [`STATUS_BUFFER_TOO_SMALL`] repeats the same request. A seeded plugin
/// must reinitialize its randomness from `decoder_seed` on every call so output
/// capacity cannot change the ordered correction.
///
/// The plugin must return an error for any optional field it does not advertise
/// through [`CapabilitiesFn`].
///
/// For a request with no optional fields, this function must return the same result as
/// [`DecodeFn`]. Older hosts use `decode` for that request; newer hosts use this
/// function.
///
/// # Safety
///
/// `handle` must be a live handle from [`CreateFn`] held exclusively by this caller.
/// `request` must be non-null and point to a valid [`DeqDecoderDecodeRequest`] whose
/// every reachable pointer is either null with a zero count or valid for reads of the
/// stated count, for the duration of the call. `subgraph` must be valid for
/// `subgraph_capacity` writes (or null if and only if the capacity is 0) and must not
/// overlap any request buffer. `subgraph_count` must be non-null.
pub type DecodeRequestFn = unsafe extern "C" fn(
    handle: *mut c_void,
    request: *const DeqDecoderDecodeRequest,
    subgraph: *mut u64,
    subgraph_capacity: usize,
    subgraph_count: *mut usize,
) -> i32;

/// `deq_decoder_abi_version() -> u32`.
///
/// Returns the [`ABI_VERSION`] the plugin was built against. The host calls this first
/// and refuses to use any other symbol on a version mismatch.
pub type AbiVersionFn = unsafe extern "C" fn() -> u32;

/// `deq_decoder_create(...) -> i32`.
///
/// Builds a decoder from a CSR-encoded hypergraph (see the crate-level docs) and a
/// NUL-terminated UTF-8 JSON configuration string. On success writes an opaque,
/// non-null handle to `*out_handle` and returns [`STATUS_OK`]. On failure returns a
/// negative status and records a message retrievable with [`LastErrorFn`].
///
/// # Safety
///
/// All array pointers must be valid for reads of their stated lengths, or null only
/// when the corresponding length is zero. `config_json` must be a valid
/// NUL-terminated C string (or null for an empty config). `out_handle` must be
/// non-null. The returned handle is owned by the caller and must be released with
/// [`DestroyFn`].
pub type CreateFn = unsafe extern "C" fn(
    vertex_num: u64,
    edge_num: u64,
    edge_probs: *const f64,
    edge_offsets: *const u64,
    edge_vertices: *const u64,
    edge_vertices_len: usize,
    config_json: *const c_char,
    out_handle: *mut *mut c_void,
) -> i32;

/// `deq_decoder_decode(...) -> i32`.
///
/// Decodes one syndrome. The syndrome is a dense bit vector mirroring deq's
/// `BitVector`: `syndrome_size` bits packed MSB-first into `syndrome_data`
/// (`syndrome_len == syndrome_size.div_ceil(8)`), where the bit for vertex `i` is set
/// iff `syndrome_data[i / 8] & (1 << (7 - i % 8))` is set. The selected subgraph
/// (hyperedge indices) is written into the caller-owned buffer `out_ptr`/`out_cap`,
/// and `*out_len` is set to the number of indices. If the buffer is too small,
/// returns [`STATUS_BUFFER_TOO_SMALL`] with `*out_len` set to the required length.
///
/// deq gives the plugin **exclusive** access to a handle: `decode` is never called
/// concurrently with another `decode` or with [`DestroyFn`] on the same handle, so a
/// decoder may keep and mutate per-handle state freely without locking. deq achieves
/// parallelism by building one handle per worker (via [`CreateFn`]), not by sharing
/// one handle across threads.
///
/// # Safety
///
/// `handle` must be a live handle from [`CreateFn`] that has not been destroyed, and
/// must not be in use by another `decode`/`destroy` call. `syndrome_data` must be
/// valid for `syndrome_len` reads (or null iff `syndrome_len == 0`). `out_ptr` must be
/// valid for `out_cap` writes (or null iff `out_cap == 0`). `out_len` must be non-null.
pub type DecodeFn = unsafe extern "C" fn(
    handle: *mut c_void,
    syndrome_size: u64,
    syndrome_data: *const u8,
    syndrome_len: usize,
    out_ptr: *mut u64,
    out_cap: usize,
    out_len: *mut usize,
) -> i32;

/// `deq_decoder_destroy(handle)`.
///
/// Releases a handle from [`CreateFn`]. Must be called exactly once per handle and
/// never concurrently with [`DecodeFn`].
///
/// # Safety
///
/// `handle` must be a live handle from [`CreateFn`] that has not already been
/// destroyed. A null handle is ignored.
pub type DestroyFn = unsafe extern "C" fn(handle: *mut c_void);

/// `deq_decoder_last_error() -> *const c_char`.
///
/// Returns the calling thread's most recent error message as a NUL-terminated UTF-8
/// C string, or null if there is none. The pointer borrows thread-local storage owned
/// by the plugin: the caller must not free it, and it is valid only until the next ABI
/// call on the same thread. Read it immediately, on the same thread, after a failing
/// call.
///
/// # Safety
///
/// The returned pointer, if non-null, must be read before the next ABI call on the
/// same thread and must not be freed by the caller.
pub type LastErrorFn = unsafe extern "C" fn() -> *const c_char;
