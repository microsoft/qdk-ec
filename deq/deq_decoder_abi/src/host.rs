//! Host-side loader (deq's side of the boundary). Requires the `host` feature.
//!
//! [`DecoderLibrary::load`] `dlopen`s a plugin, verifies its ABI version, and resolves
//! the exported symbols once. [`LoadedDecoder`] owns one decoder handle and exposes a
//! safe [`decode`](LoadedDecoder::decode) that manages the caller-owned output buffer,
//! including the `BUFFER_TOO_SMALL` retry and a per-handle capacity hint so the
//! expensive search is not re-run on a well-warmed buffer.

use core::ffi::c_void;
use std::path::Path;

use libloading::{Library, Symbol};

use crate::interface::{
    ABI_VERSION, AbiVersionFn, CapabilitiesFn, CreateFn, DEQ_DECODER_CAPABILITY_LOSS, DEQ_DECODER_CAPABILITY_REWEIGHTS,
    DEQ_DECODER_CAPABILITY_SEED, DecodeFn, DecodeRequestFn, DeqDecoderCapabilities, DeqDecoderDecodeRequest,
    DeqDecoderEdgeReweight, DeqDecoderLossInfo, DeqDecoderLossSite, DestroyFn, LastErrorFn, STATUS_BUFFER_TOO_SMALL,
    STATUS_INVALID_ARG, STATUS_OK, SYM_ABI_VERSION, SYM_CAPABILITIES, SYM_CREATE, SYM_DECODE, SYM_DECODE_REQUEST,
    SYM_DESTROY, SYM_LAST_ERROR, describe_capabilities, required_capabilities,
};
use crate::plugin::LossSiteView;

/// An error from loading a plugin or calling one of its ABI functions.
#[derive(Debug)]
pub enum AbiError {
    /// The shared library could not be opened or a symbol was missing.
    Load(String),
    /// The plugin reported a different ABI version than this host supports.
    VersionMismatch { found: u32, expected: u32 },
    /// A plugin function returned a failing status, or the host rejected an argument
    /// with [`STATUS_INVALID_ARG`] before calling one. The string is the plugin's
    /// last-error message or a synthesized description.
    Plugin { status: i32, message: String },
    /// The plugin does not support an optional field in the request.
    UnsupportedRequest(String),
}

impl std::fmt::Display for AbiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AbiError::Load(message) => write!(f, "failed to load decoder plugin: {message}"),
            AbiError::VersionMismatch { found, expected } => write!(
                f,
                "decoder plugin ABI version {found} does not match host version {expected}"
            ),
            AbiError::Plugin { status, message } => {
                write!(f, "decoder plugin call failed (status {status}): {message}")
            }
            AbiError::UnsupportedRequest(message) => {
                write!(f, "decoder plugin does not support this request: {message}")
            }
        }
    }
}

impl std::error::Error for AbiError {}

/// The optional request API.
///
/// A plugin must export both `deq_decoder_decode_request` and
/// `deq_decoder_capabilities`. The loader rejects incomplete pairs.
#[derive(Clone, Copy)]
struct RequestApi {
    decode_request: DecodeRequestFn,
    capabilities: DeqDecoderCapabilities,
}

/// A decode request expressed with borrowed slices.
///
/// [`LoadedDecoder::decode_request`] creates the raw pointer-based
/// [`DeqDecoderDecodeRequest`] for the duration of the plugin call.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostDecodeRequest<'a> {
    /// Logical number of syndrome bits.
    pub syndrome_size: u64,
    /// Dense MSB-first packed syndrome, `syndrome_size.div_ceil(8)` bytes.
    pub syndrome_data: &'a [u8],
    /// The controlled seed, if any. `None` and `Some(0)` are different requests.
    pub decoder_seed: Option<u64>,
    /// Shot-scoped prior assignments as `(edge, probability)` pairs.
    pub reweights: &'a [(u64, f64)],
    /// Structured loss for this shot. `Some(&[])` means loss information was
    /// supplied but no site was possible.
    pub loss: Option<&'a [LossSiteView<'a>]>,
}

/// Every capability bit this ABI revision defines. Private on purpose: a plugin that
/// set `CAPABILITIES` to a public "all" mask would claim, on its next rebuild, every
/// capability a later revision adds.
const KNOWN_CAPABILITIES: DeqDecoderCapabilities =
    DEQ_DECODER_CAPABILITY_SEED | DEQ_DECODER_CAPABILITY_REWEIGHTS | DEQ_DECODER_CAPABILITY_LOSS;

/// The ABI version is checked before this function, so unknown bits indicate a
/// contract mismatch and must not be ignored.
fn check_capability_bits(bits: DeqDecoderCapabilities) -> Result<(), AbiError> {
    let unknown = bits & !KNOWN_CAPABILITIES;
    if unknown != 0 {
        return Err(AbiError::Load(format!(
            "unknown capability bits {unknown:#x}; the plugin uses a different ABI contract"
        )));
    }
    Ok(())
}

/// A loaded decoder plugin: its resolved entry points. The underlying [`Library`] is
/// intentionally never unloaded (it is leaked for the process lifetime) so that code
/// pages and thread-local destructors registered by the plugin remain mapped while any
/// handle (or any thread that ever called into the plugin) is still alive.
pub struct DecoderLibrary {
    create: CreateFn,
    decode: DecodeFn,
    destroy: DestroyFn,
    last_error: LastErrorFn,
    request_api: Option<RequestApi>,
}

// SAFETY: the fields are plain `extern "C"` function pointers, which are themselves
// `Send + Sync`; the ABI contract requires the exported functions to be thread-safe.
unsafe impl Send for DecoderLibrary {}
unsafe impl Sync for DecoderLibrary {}

impl DecoderLibrary {
    /// Load and validate a decoder plugin from a shared-library path.
    ///
    /// The library is leaked (never unloaded). The returned reference therefore lives
    /// for the remainder of the process and can be shared freely.
    ///
    /// # Errors
    ///
    /// Returns [`AbiError::Load`] if the library cannot be opened or a required symbol
    /// is missing, or [`AbiError::VersionMismatch`] if the plugin's ABI version differs
    /// from this host's.
    ///
    /// # Safety
    ///
    /// Loading a shared library runs arbitrary initialization code from `path`. The
    /// caller must ensure `path` refers to a trusted plugin that honors the ABI
    /// contract; it must never come from untrusted or remote input.
    pub unsafe fn load(path: impl AsRef<Path>) -> Result<&'static DecoderLibrary, AbiError> {
        // SAFETY: delegated to the caller's contract on `path`.
        let load_err = |e: libloading::Error| AbiError::Load(e.to_string());
        let library = unsafe { Library::new(path.as_ref()) }.map_err(load_err)?;
        // Leak the library so its code stays mapped for the process lifetime.
        let library: &'static Library = Box::leak(Box::new(library));

        // SAFETY: symbols are resolved from a leaked (permanently mapped) library, so
        // the copied function pointers remain valid for the process lifetime.
        unsafe {
            let abi_version: Symbol<AbiVersionFn> = library.get(SYM_ABI_VERSION).map_err(load_err)?;
            let found = abi_version();
            if found != ABI_VERSION {
                return Err(AbiError::VersionMismatch {
                    found,
                    expected: ABI_VERSION,
                });
            }
            let create: Symbol<CreateFn> = library.get(SYM_CREATE).map_err(load_err)?;
            let decode: Symbol<DecodeFn> = library.get(SYM_DECODE).map_err(load_err)?;
            let destroy: Symbol<DestroyFn> = library.get(SYM_DESTROY).map_err(load_err)?;
            let last_error: Symbol<LastErrorFn> = library.get(SYM_LAST_ERROR).map_err(load_err)?;

            // The request API is optional, but both symbols must be present together.
            let decode_request: Option<Symbol<DecodeRequestFn>> = library.get(SYM_DECODE_REQUEST).ok();
            let capabilities: Option<Symbol<CapabilitiesFn>> = library.get(SYM_CAPABILITIES).ok();
            let request_api = match (decode_request, capabilities) {
                (Some(decode_request), Some(capabilities)) => {
                    let bits = capabilities();
                    check_capability_bits(bits)?;
                    Some(RequestApi {
                        decode_request: *decode_request,
                        capabilities: bits,
                    })
                }
                (None, None) => None,
                _ => {
                    return Err(AbiError::Load(
                        "a plugin must export both deq_decoder_decode_request and deq_decoder_capabilities, or neither"
                            .to_string(),
                    ));
                }
            };
            // The static reference owns function pointers into the leaked library.
            Ok(Box::leak(Box::new(DecoderLibrary {
                create: *create,
                decode: *decode,
                destroy: *destroy,
                last_error: *last_error,
                request_api,
            })))
        }
    }

    /// Read the calling thread's most recent plugin error message, if any.
    fn read_last_error(&self) -> Option<String> {
        // SAFETY: the plugin returns either null or a NUL-terminated C string valid
        // until the next ABI call on this thread; we copy it out immediately.
        let ptr = unsafe { (self.last_error)() };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: non-null per the check above; the plugin guarantees NUL-termination.
        let message = unsafe { core::ffi::CStr::from_ptr(ptr) }.to_string_lossy().into_owned();
        Some(message)
    }

    fn plugin_error(&self, status: i32) -> AbiError {
        let message = self
            .read_last_error()
            .unwrap_or_else(|| "no error message available".to_string());
        AbiError::Plugin { status, message }
    }

    /// The library-level capability bitmask, or zero for a legacy plugin without the
    /// request API.
    #[must_use]
    pub fn capabilities(&self) -> DeqDecoderCapabilities {
        self.request_api.map_or(0, |api| api.capabilities)
    }
}

/// Drive a decode call through the caller-owned output buffer, growing and retrying
/// on [`STATUS_BUFFER_TOO_SMALL`] until the result fits.
///
/// # Safety
///
/// `call` receives `(pointer, capacity, count_out)` for a buffer valid for `capacity`
/// writes, and on [`STATUS_OK`] must have written exactly `*count_out` initialized
/// `u64`s with `*count_out <= capacity`.
unsafe fn decode_with_retry(
    library: &'static DecoderLibrary,
    suggested_cap: &mut usize,
    out: &mut Vec<u64>,
    mut call: impl FnMut(*mut u64, usize, *mut usize) -> i32,
) -> Result<(), AbiError> {
    let mut capacity = (*suggested_cap).max(1);
    loop {
        out.clear();
        out.reserve(capacity);
        let cap = out.capacity();
        let mut written: usize = 0;
        let status = call(out.as_mut_ptr(), cap, &raw mut written);
        match status {
            STATUS_OK => {
                debug_assert!(written <= cap);
                // SAFETY: on STATUS_OK the callee wrote `written <= cap` initialized
                // `u64`s into the buffer, per this function's contract.
                unsafe { out.set_len(written) };
                *suggested_cap = (*suggested_cap).max(written);
                return Ok(());
            }
            STATUS_BUFFER_TOO_SMALL => {
                *suggested_cap = (*suggested_cap).max(written);
                if written <= cap {
                    // Defensive: avoid an infinite loop if the plugin misreports.
                    capacity = cap.saturating_add(1).saturating_mul(2);
                } else {
                    capacity = written;
                }
            }
            _ => return Err(library.plugin_error(status)),
        }
    }
}

/// One live decoder handle built from a hypergraph, bound to its [`DecoderLibrary`].
///
/// A `LoadedDecoder` is owned exclusively by one worker: `decode` takes `&mut self`
/// and the ABI forbids concurrent decode/destroy. deq parallelizes by building one
/// `LoadedDecoder` per worker, not by sharing one. Dropping it destroys the handle.
pub struct LoadedDecoder {
    library: &'static DecoderLibrary,
    handle: *mut c_void,
    /// Capacity hint carried across shots so the `BUFFER_TOO_SMALL` retry settles to
    /// the running high-water mark. Plain `usize` because the handle is never shared.
    suggested_cap: usize,
}

// SAFETY: the handle is a plain pointer the plugin owns; `LoadedDecoder` is moved
// between worker threads but used by one at a time, so it is `Send`. It is
// deliberately not `Sync`: `decode` takes `&mut self` (exclusive per the ABI).
unsafe impl Send for LoadedDecoder {}

impl LoadedDecoder {
    /// Build a decoder handle from a CSR-encoded hypergraph and a JSON config blob.
    ///
    /// See the crate-level docs for the CSR invariants; they are also revalidated by
    /// the plugin. `config_json` is passed through verbatim (typically the bytes of a
    /// serialized JSON object).
    ///
    /// # Errors
    ///
    /// Returns [`AbiError::Plugin`] if the plugin rejects the hypergraph or config, or
    /// returns a null handle on success.
    pub fn create(
        library: &'static DecoderLibrary,
        vertex_num: u64,
        edge_probs: &[f64],
        edge_offsets: &[u64],
        edge_vertices: &[u64],
        config_json: &str,
    ) -> Result<Self, AbiError> {
        let config = std::ffi::CString::new(config_json).map_err(|e| AbiError::Plugin {
            status: STATUS_INVALID_ARG,
            message: format!("config_json contains an interior NUL byte: {e}"),
        })?;
        let mut handle: *mut c_void = core::ptr::null_mut();
        // SAFETY: all slices provide valid pointer/length pairs; `config` is a valid
        // NUL-terminated C string; `out_handle` is a valid local out-param. The plugin
        // revalidates the CSR structure.
        let status = unsafe {
            (library.create)(
                vertex_num,
                edge_probs.len() as u64,
                edge_probs.as_ptr(),
                edge_offsets.as_ptr(),
                edge_vertices.as_ptr(),
                edge_vertices.len(),
                config.as_ptr(),
                &raw mut handle,
            )
        };
        if status != STATUS_OK {
            return Err(library.plugin_error(status));
        }
        if handle.is_null() {
            return Err(AbiError::Plugin {
                status,
                message: "create returned OK but a null handle".to_string(),
            });
        }
        Ok(Self {
            library,
            handle,
            suggested_cap: 16,
        })
    }

    /// Decode one syndrome, writing the selected subgraph (hyperedge indices) into
    /// `out`. `out` is cleared first and resized to the result on success.
    ///
    /// `syndrome_size`/`syndrome_data` mirror deq's `BitVector`: `syndrome_size` bits
    /// packed MSB-first, with `syndrome_data.len() == syndrome_size.div_ceil(8)`.
    ///
    /// Takes `&mut self`: deq gives each handle to one worker exclusively.
    ///
    /// # Errors
    ///
    /// Returns [`AbiError::Plugin`] if the plugin reports a decode failure.
    pub fn decode(&mut self, syndrome_size: u64, syndrome_data: &[u8], out: &mut Vec<u64>) -> Result<(), AbiError> {
        let data_ptr = if syndrome_data.is_empty() {
            core::ptr::null()
        } else {
            syndrome_data.as_ptr()
        };
        let library = self.library;
        let handle = self.handle;
        let call = |ptr: *mut u64, cap: usize, count: *mut usize| {
            // SAFETY: `ptr` is valid for `cap` writes; `data_ptr`/`len` is a valid
            // (possibly empty) slice; `count` is a valid out-param. `handle` is live
            // and held exclusively (`&mut self`).
            unsafe { (library.decode)(handle, syndrome_size, data_ptr, syndrome_data.len(), ptr, cap, count) }
        };
        // SAFETY: the plugin writes `*count` initialized indices into the buffer it is
        // given, per the `DecodeFn` contract.
        unsafe { decode_with_retry(library, &mut self.suggested_cap, out, call) }
    }

    /// Decode a syndrome and any optional fields supported by the plugin.
    ///
    /// A legacy plugin can handle requests with no optional fields through
    /// [`decode`](Self::decode). Unsupported optional fields return an error.
    ///
    /// # Errors
    ///
    /// Returns [`AbiError::UnsupportedRequest`] if the plugin does not accept a field
    /// the request carries, or [`AbiError::Plugin`] if the plugin reports a failure.
    pub fn decode_request(&mut self, request: &HostDecodeRequest<'_>, out: &mut Vec<u64>) -> Result<(), AbiError> {
        // The request ABI carries no byte length, so validate the Rust slice before
        // giving its pointer to the plugin.
        let expected_bytes = request
            .syndrome_size
            .div_ceil(crate::interface::DEQ_DECODER_SYNDROME_BITS_PER_BYTE);
        if u64::try_from(request.syndrome_data.len()) != Ok(expected_bytes) {
            return Err(AbiError::Plugin {
                status: STATUS_INVALID_ARG,
                message: format!(
                    "syndrome_data has {} bytes; syndrome_size {} requires {expected_bytes}",
                    request.syndrome_data.len(),
                    request.syndrome_size
                ),
            });
        }
        let required = required_capabilities(
            request.decoder_seed.is_some(),
            !request.reweights.is_empty(),
            request.loss.is_some(),
        );
        let Some(api) = self.library.request_api else {
            if required == 0 {
                return self.decode(request.syndrome_size, request.syndrome_data, out);
            }
            return Err(AbiError::UnsupportedRequest(format!(
                "fields {} require deq_decoder_decode_request, which the plugin does not export",
                describe_capabilities(required)
            )));
        };
        let missing = required & !api.capabilities;
        if missing != 0 {
            return Err(AbiError::UnsupportedRequest(format!(
                "unsupported fields: {}",
                describe_capabilities(missing)
            )));
        }

        // These buffers must remain alive across every output-buffer retry.
        let reweights: Vec<DeqDecoderEdgeReweight> = request
            .reweights
            .iter()
            .map(|&(edge, probability)| DeqDecoderEdgeReweight { edge, probability })
            .collect();
        let sites: Vec<DeqDecoderLossSite> = request
            .loss
            .unwrap_or(&[])
            .iter()
            .map(|site| DeqDecoderLossSite {
                source_edges: empty_as_null(site.source_edges),
                source_edge_count: site.source_edges.len(),
                continuation_edges: empty_as_null(site.continuation_edges),
                continuation_edge_count: site.continuation_edges.len(),
                probability: site.probability,
                children: empty_as_null(site.children),
                child_count: site.children.len(),
                heralds: empty_as_null(site.heralds),
                herald_count: site.heralds.len(),
            })
            .collect();
        // `Some` with no sites must stay distinguishable from `None`, so the presence
        // of the `LossInfo` follows `request.loss`, not the site count.
        let loss = request.loss.map(|_| DeqDecoderLossInfo {
            sites: empty_as_null(&sites),
            site_count: sites.len(),
        });
        let raw = DeqDecoderDecodeRequest {
            syndrome_size: request.syndrome_size,
            syndrome_data: empty_as_null(request.syndrome_data),
            has_decoder_seed: request.decoder_seed.is_some(),
            decoder_seed: request.decoder_seed.unwrap_or(0),
            reweights: empty_as_null(&reweights),
            reweight_count: reweights.len(),
            loss: loss.as_ref().map_or(core::ptr::null(), core::ptr::from_ref),
        };

        let library = self.library;
        let handle = self.handle;
        let decode_request = api.decode_request;
        let call = |ptr: *mut u64, cap: usize, count: *mut usize| {
            // SAFETY: `raw` and every buffer reachable from it live for this whole
            // loop; `ptr` is valid for `cap` writes and does not overlap them;
            // `count` is a valid out-param; `handle` is live and held exclusively.
            unsafe { decode_request(handle, &raw const raw, ptr, cap, count) }
        };
        // SAFETY: the plugin writes `*count` initialized indices into the buffer it is
        // given, per the `DecodeRequestFn` contract.
        unsafe { decode_with_retry(library, &mut self.suggested_cap, out, call) }
    }
}

/// Return the slice pointer, using null for an empty slice.
///
/// Empty Rust slices may use a non-null dangling pointer, while the ABI permits null
/// whenever the paired count is zero.
fn empty_as_null<T>(slice: &[T]) -> *const T {
    if slice.is_empty() {
        core::ptr::null()
    } else {
        slice.as_ptr()
    }
}

impl Drop for LoadedDecoder {
    fn drop(&mut self) {
        // SAFETY: single-shot destroy of a live handle; the owner guarantees no decode
        // is in flight (the ABI forbids concurrent destroy/decode).
        unsafe { (self.library.destroy)(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::ffi::c_char;

    use crate::interface::{
        DEQ_DECODER_CAPABILITY_LOSS, DEQ_DECODER_CAPABILITY_REWEIGHTS, DEQ_DECODER_CAPABILITY_SEED,
    };

    fn dummy_handle() -> *mut c_void {
        core::ptr::dangling_mut::<u8>().cast::<c_void>()
    }

    unsafe extern "C" fn stub_create(
        _vertex_num: u64,
        _edge_num: u64,
        _edge_probs: *const f64,
        _edge_offsets: *const u64,
        _edge_vertices: *const u64,
        _edge_vertices_len: usize,
        _config_json: *const c_char,
        _out_handle: *mut *mut c_void,
    ) -> i32 {
        STATUS_OK
    }

    unsafe extern "C" fn stub_decode(
        _handle: *mut c_void,
        _syndrome_size: u64,
        _syndrome_data: *const u8,
        _syndrome_len: usize,
        out_ptr: *mut u64,
        out_cap: usize,
        out_len: *mut usize,
    ) -> i32 {
        unsafe { out_len.write(1) };
        if out_cap == 0 {
            return STATUS_BUFFER_TOO_SMALL;
        }
        unsafe { out_ptr.write(11) };
        STATUS_OK
    }

    unsafe extern "C" fn stub_decode_request(
        _handle: *mut c_void,
        _request: *const DeqDecoderDecodeRequest,
        subgraph: *mut u64,
        subgraph_capacity: usize,
        subgraph_count: *mut usize,
    ) -> i32 {
        unsafe { subgraph_count.write(1) };
        if subgraph_capacity == 0 {
            return STATUS_BUFFER_TOO_SMALL;
        }
        unsafe { subgraph.write(22) };
        STATUS_OK
    }

    unsafe extern "C" fn stub_destroy(_handle: *mut c_void) {}

    unsafe extern "C" fn stub_last_error() -> *const c_char {
        core::ptr::null()
    }

    fn library(request_api: Option<RequestApi>) -> &'static DecoderLibrary {
        Box::leak(Box::new(DecoderLibrary {
            create: stub_create,
            decode: stub_decode,
            destroy: stub_destroy,
            last_error: stub_last_error,
            request_api,
        }))
    }

    fn loaded(request_api: Option<RequestApi>) -> LoadedDecoder {
        LoadedDecoder {
            library: library(request_api),
            handle: dummy_handle(),
            suggested_cap: 16,
        }
    }

    fn request_api(capabilities: DeqDecoderCapabilities) -> RequestApi {
        RequestApi {
            decode_request: stub_decode_request,
            capabilities,
        }
    }

    fn plain_request(syndrome: &[u8]) -> HostDecodeRequest<'_> {
        HostDecodeRequest {
            syndrome_size: 8,
            syndrome_data: syndrome,
            ..Default::default()
        }
    }

    #[test]
    fn a_legacy_plugin_serves_a_plain_request_through_decode() {
        let mut decoder = loaded(None);
        let mut out = Vec::new();
        decoder.decode_request(&plain_request(&[0]), &mut out).expect("plain");
        assert_eq!(out, vec![11], "a request with no optional field falls back to decode");
    }

    #[test]
    fn a_legacy_plugin_rejects_an_extended_request_explicitly() {
        let mut decoder = loaded(None);
        let mut out = Vec::new();

        // Seed zero in particular must be rejected: it is a real request, not an
        // absent seed, so it can never fall through to the legacy entry point.
        let mut seeded = plain_request(&[0]);
        seeded.decoder_seed = Some(0);
        let error = decoder.decode_request(&seeded, &mut out).unwrap_err();
        assert!(
            matches!(&error, AbiError::UnsupportedRequest(message) if message.contains("decoder_seed")),
            "unexpected error: {error}"
        );

        let mut reweighted = plain_request(&[0]);
        reweighted.reweights = &[(0, 0.5)];
        let error = decoder.decode_request(&reweighted, &mut out).unwrap_err();
        assert!(
            matches!(&error, AbiError::UnsupportedRequest(message) if message.contains("reweights")),
            "unexpected error: {error}"
        );

        let mut lossy = plain_request(&[0]);
        lossy.loss = Some(&[]);
        let error = decoder.decode_request(&lossy, &mut out).unwrap_err();
        assert!(
            matches!(&error, AbiError::UnsupportedRequest(message) if message.contains("loss")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn a_partially_capable_plugin_rejects_only_the_fields_it_lacks() {
        let mut decoder = loaded(Some(request_api(DEQ_DECODER_CAPABILITY_SEED)));
        let mut out = Vec::new();

        let mut seeded = plain_request(&[0]);
        seeded.decoder_seed = Some(0);
        decoder.decode_request(&seeded, &mut out).expect("seed is advertised");
        assert_eq!(out, vec![22], "an advertised field uses the request entry point");

        let mut lossy = plain_request(&[0]);
        lossy.loss = Some(&[]);
        let error = decoder.decode_request(&lossy, &mut out).unwrap_err();
        assert!(
            matches!(&error, AbiError::UnsupportedRequest(message) if message.contains("loss")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn a_capable_plugin_uses_the_request_entry_point_even_for_a_plain_request() {
        let mut decoder = loaded(Some(request_api(KNOWN_CAPABILITIES)));
        let mut out = Vec::new();
        decoder.decode_request(&plain_request(&[0]), &mut out).expect("plain");
        assert_eq!(out, vec![22]);
    }

    #[test]
    fn unknown_capability_bits_are_rejected_rather_than_masked_off() {
        check_capability_bits(KNOWN_CAPABILITIES).expect("every defined bit is accepted");
        check_capability_bits(0).expect("no capability is accepted");

        let unknown = DEQ_DECODER_CAPABILITY_LOSS | (1 << 40);
        let error = check_capability_bits(unknown).unwrap_err();
        assert!(
            matches!(&error, AbiError::Load(message) if message.contains("0x10000000000")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn capability_bits_match_the_documented_assignments() {
        assert_eq!(DEQ_DECODER_CAPABILITY_SEED, 1);
        assert_eq!(DEQ_DECODER_CAPABILITY_REWEIGHTS, 2);
        assert_eq!(DEQ_DECODER_CAPABILITY_LOSS, 4);
        assert_eq!(KNOWN_CAPABILITIES, 7);
    }

    #[test]
    fn a_syndrome_shorter_than_its_bit_count_is_rejected_before_the_call() {
        // The ABI request carries no syndrome length, so the plugin reads exactly
        // `syndrome_size.div_ceil(8)` bytes. A safe caller must not be able to make
        // it read past the buffer.
        let mut decoder = loaded(Some(request_api(KNOWN_CAPABILITIES)));
        let mut out = Vec::new();

        let mut short = plain_request(&[0]);
        short.syndrome_size = 16;
        let error = decoder.decode_request(&short, &mut out).unwrap_err();
        assert!(
            matches!(&error, AbiError::Plugin { status: STATUS_INVALID_ARG, message } if message.contains("requires 2")),
            "unexpected error: {error}"
        );

        let mut long = plain_request(&[0, 0, 0]);
        long.syndrome_size = 8;
        let error = decoder.decode_request(&long, &mut out).unwrap_err();
        assert!(
            matches!(
                &error,
                AbiError::Plugin {
                    status: STATUS_INVALID_ARG,
                    ..
                }
            ),
            "unexpected error: {error}"
        );

        let mut exact = plain_request(&[0, 0]);
        exact.syndrome_size = 16;
        decoder
            .decode_request(&exact, &mut out)
            .expect("exact length is accepted");

        let empty = HostDecodeRequest::default();
        decoder
            .decode_request(&empty, &mut out)
            .expect("a zero-bit syndrome needs no bytes");
    }
}
