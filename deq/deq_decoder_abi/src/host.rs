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
    ABI_VERSION, AbiVersionFn, CreateFn, DecodeFn, DestroyFn, LastErrorFn, STATUS_BUFFER_TOO_SMALL, STATUS_INVALID_ARG,
    STATUS_OK, SYM_ABI_VERSION, SYM_CREATE, SYM_DECODE, SYM_DESTROY, SYM_LAST_ERROR,
};

/// An error from loading a plugin or calling one of its ABI functions.
#[derive(Debug)]
pub enum AbiError {
    /// The shared library could not be opened or a symbol was missing.
    Load(String),
    /// The plugin reported a different ABI version than this host supports.
    VersionMismatch { found: u32, expected: u32 },
    /// A plugin function returned a failing status; the string is its last-error
    /// message (or a synthesized description when none was available).
    Plugin { status: i32, message: String },
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
        }
    }
}

impl std::error::Error for AbiError {}

/// A loaded decoder plugin: its resolved entry points. The underlying [`Library`] is
/// intentionally never unloaded (it is leaked for the process lifetime) so that code
/// pages and thread-local destructors registered by the plugin remain mapped while any
/// handle (or any thread that ever called into the plugin) is still alive.
pub struct DecoderLibrary {
    create: CreateFn,
    decode: DecodeFn,
    destroy: DestroyFn,
    last_error: LastErrorFn,
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
            // Leaked to hand back `&'static`: these four fn pointers live as long as the
            // (also-leaked) library, so every worker shares one copy with no Arc.
            Ok(Box::leak(Box::new(DecoderLibrary {
                create: *create,
                decode: *decode,
                destroy: *destroy,
                last_error: *last_error,
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
        let mut capacity = self.suggested_cap.max(1);
        loop {
            out.clear();
            out.reserve(capacity);
            let cap = out.capacity();
            let mut written: usize = 0;
            // SAFETY: `out` has capacity `cap` and is logically empty, so its buffer is
            // valid for `cap` writes; `data_ptr`/`syndrome_data.len()` is a valid
            // (possibly empty) slice; `written` is a valid out-param. `self.handle` is
            // live and held exclusively (`&mut self`).
            let status = unsafe {
                (self.library.decode)(
                    self.handle,
                    syndrome_size,
                    data_ptr,
                    syndrome_data.len(),
                    out.as_mut_ptr(),
                    cap,
                    &raw mut written,
                )
            };
            match status {
                STATUS_OK => {
                    debug_assert!(written <= cap);
                    // SAFETY: the plugin wrote `written <= cap` initialized `u64`s.
                    unsafe { out.set_len(written) };
                    self.suggested_cap = self.suggested_cap.max(written);
                    return Ok(());
                }
                STATUS_BUFFER_TOO_SMALL => {
                    // `written` holds the required length; grow and retry.
                    self.suggested_cap = self.suggested_cap.max(written);
                    if written <= cap {
                        // Defensive: avoid an infinite loop if the plugin misreports.
                        capacity = cap.saturating_add(1).saturating_mul(2);
                    } else {
                        capacity = written;
                    }
                }
                _ => return Err(self.library.plugin_error(status)),
            }
        }
    }
}

impl Drop for LoadedDecoder {
    fn drop(&mut self) {
        // SAFETY: single-shot destroy of a live handle; the owner guarantees no decode
        // is in flight (the ABI forbids concurrent destroy/decode).
        unsafe { (self.library.destroy)(self.handle) };
    }
}
