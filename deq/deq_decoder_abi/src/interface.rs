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
