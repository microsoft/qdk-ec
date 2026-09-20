//! Plugin-author interface: implement the safe [`DeqDecoder`] trait and invoke
//! [`declare_decoder!`](crate::declare_decoder) to export the C ABI without writing any `unsafe` code.
//!
//! ```ignore
//! use deq_decoder_abi::plugin::{DeqDecoder, HypergraphView, OutputBuffer, SyndromeView};
//!
//! struct MyDecoder { /* precomputed, immutable state */ }
//!
//! impl DeqDecoder for MyDecoder {
//!     fn create(graph: HypergraphView<'_>, config_json: &[u8]) -> Result<Self, String> {
//!         // build an immutable decoder from the hypergraph + config
//!         Ok(MyDecoder { /* ... */ })
//!     }
//!     fn decode(&mut self, syndrome: SyndromeView<'_>, out: &mut OutputBuffer) -> Result<(), String> {
//!         // run the search; push selected hyperedge indices into `out`.
//!         // `syndrome.sparse_indices()` yields set vertices; `syndrome.data()` is dense.
//!         Ok(())
//!     }
//! }
//!
//! deq_decoder_abi::declare_decoder!(MyDecoder);
//! ```

use core::cell::RefCell;
use core::ffi::{c_char, c_void};

use crate::interface::{ABI_VERSION, STATUS_ERROR, STATUS_INVALID_ARG, STATUS_OK, STATUS_PANIC, STATUS_POISONED};

/// A decoder that can be exported across the C ABI.
///
/// deq gives each decoder instance to a single worker and never calls `decode`
/// concurrently on it, so `decode` takes `&mut self` and the decoder may keep and
/// mutate per-handle state freely without locking. deq parallelizes by building one
/// instance per worker, so all expensive setup (model construction, ordering
/// precomputation) belongs in [`create`](Self::create); a decoder that is costly to
/// build and never mutated may still want to share immutable state behind an `Arc`.
pub trait DeqDecoder: Send + 'static {
    /// Build a decoder from a borrowed hypergraph view and a UTF-8 JSON config blob.
    ///
    /// Return `Err` with a human-readable message on any failure (invalid config,
    /// unsupported hypergraph, etc.); the message is surfaced through the ABI's
    /// thread-local error channel.
    ///
    /// # Errors
    ///
    /// Returns an error message if the configuration or hypergraph is invalid or the
    /// decoder cannot be constructed.
    fn create(graph: HypergraphView<'_>, config_json: &[u8]) -> Result<Self, String>
    where
        Self: Sized;

    /// Decode one syndrome. `syndrome` is a dense bit view (see [`SyndromeView`])
    /// mirroring deq's `BitVector`; push the selected hyperedge indices into `out`.
    /// Use [`SyndromeView::sparse_indices`] for the sparse list of set vertices, or
    /// [`SyndromeView::is_set`] / [`SyndromeView::data`] to read it densely.
    ///
    /// `out` is a fixed-capacity, caller-owned buffer. Pushing beyond its capacity is
    /// safe (the excess is counted but not written) and causes the host to retry with
    /// a larger buffer, so implementations should push all indices unconditionally.
    ///
    /// # Errors
    ///
    /// Returns an error message if the syndrome cannot be decoded.
    fn decode(&mut self, syndrome: SyndromeView<'_>, out: &mut OutputBuffer) -> Result<(), String>;
}

/// A borrowed, dense view of a syndrome, mirroring deq's `BitVector`: `size` bits
/// packed MSB-first into `data` (`data.len() == size.div_ceil(8)`).
#[derive(Clone, Copy)]
pub struct SyndromeView<'a> {
    size: u64,
    data: &'a [u8],
}

impl<'a> SyndromeView<'a> {
    /// The raw packed bytes (MSB-first, one bit per vertex).
    #[must_use]
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Whether the syndrome bit for `vertex` is set. Indices `>= size` read as
    /// `false`.
    #[must_use]
    pub fn is_set(&self, vertex: u64) -> bool {
        if vertex >= self.size {
            return false;
        }
        let byte = (vertex / 8) as usize;
        let mask = 1u8 << (7 - (vertex % 8) as u8);
        self.data[byte] & mask != 0
    }

    /// Iterate over the indices of the set vertices, in ascending order. Mirrors
    /// deq's `to_sparse_indices`.
    pub fn sparse_indices(&self) -> impl Iterator<Item = u64> + '_ {
        (0..self.size).filter(move |&i| self.is_set(i))
    }
}

/// A borrowed, validated view of a CSR-encoded decoding hypergraph passed to
/// [`DeqDecoder::create`].
pub struct HypergraphView<'a> {
    /// Number of vertices (detector axis of the syndrome).
    pub vertex_num: u64,
    edge_probs: &'a [f64],
    edge_offsets: &'a [u64],
    edge_vertices: &'a [u64],
}

impl<'a> HypergraphView<'a> {
    fn edge_num(&self) -> usize {
        self.edge_probs.len()
    }

    fn edge(&self, index: usize) -> (f64, &'a [u64]) {
        let start = usize::try_from(self.edge_offsets[index]).expect("CSR offset exceeds usize");
        let end = usize::try_from(self.edge_offsets[index + 1]).expect("CSR offset exceeds usize");
        (self.edge_probs[index], &self.edge_vertices[start..end])
    }

    /// Iterate over `(probability, vertices)` for every hyperedge in order.
    pub fn edges(&self) -> impl Iterator<Item = (f64, &'a [u64])> + '_ {
        (0..self.edge_num()).map(move |i| self.edge(i))
    }
}

/// A fixed-capacity, caller-owned output buffer that decoders push subgraph indices
/// into. Pushes past the capacity are counted (so the host learns the required size)
/// but not written.
pub struct OutputBuffer {
    ptr: *mut u64,
    cap: usize,
    len: usize,
    needed: usize,
}

impl OutputBuffer {
    /// # Safety
    ///
    /// `ptr` must be valid for `cap` writes of `u64`, or null only if `cap == 0`.
    unsafe fn new(ptr: *mut u64, cap: usize) -> Self {
        Self {
            ptr,
            cap,
            len: 0,
            needed: 0,
        }
    }

    /// Append one hyperedge index. Writes only if capacity remains; always counts
    /// toward the required length.
    pub fn push(&mut self, index: u64) {
        if self.len < self.cap {
            // SAFETY: `len < cap` and `ptr` is valid for `cap` writes (invariant of
            // `new`), so `ptr.add(len)` is in bounds.
            unsafe { self.ptr.add(self.len).write(index) };
            self.len += 1;
        }
        self.needed += 1;
    }
}

thread_local! {
    static LAST_ERROR: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Record `message` as the calling thread's most recent ABI error.
///
/// The buffer is stored NUL-terminated so [`last_error_impl`] can hand back a valid C
/// string. Any interior NUL bytes in `message` are stripped, since they would
/// truncate the C string.
pub fn set_last_error(message: &str) {
    LAST_ERROR.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.clear();
        buf.extend(message.bytes().filter(|&b| b != 0));
        buf.push(0);
    });
}

/// Wrapper that the generated shims box behind the opaque handle. Tracks a poison flag
/// so that a panic during `decode` permanently disables the handle. The handle is
/// owned exclusively by one worker at a time (the ABI forbids concurrent use), so a
/// plain `bool` suffices, with no atomics.
#[doc(hidden)]
pub struct ExportedHandle<T: DeqDecoder> {
    decoder: T,
    poisoned: bool,
}

/// Generic implementation of the `deq_decoder_abi_version` symbol.
#[doc(hidden)]
#[must_use]
pub fn abi_version_impl() -> u32 {
    ABI_VERSION
}

/// Generic implementation of the `deq_decoder_create` symbol.
///
/// # Safety
///
/// Pointers must satisfy the [`crate::interface::CreateFn`] contract.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn create_impl<T: DeqDecoder>(
    vertex_num: u64,
    edge_num: u64,
    edge_probs: *const f64,
    edge_offsets: *const u64,
    edge_vertices: *const u64,
    edge_vertices_len: usize,
    config_json: *const c_char,
    out_handle: *mut *mut c_void,
) -> i32 {
    let result = std::panic::catch_unwind(|| {
        if out_handle.is_null() {
            set_last_error("create: out_handle is null");
            return STATUS_INVALID_ARG;
        }
        let Ok(edge_num) = usize::try_from(edge_num) else {
            set_last_error("create: edge_num exceeds usize");
            return STATUS_INVALID_ARG;
        };
        // SAFETY: the caller guarantees each pointer is valid for the stated length,
        // or null when the length is zero; `empty_slice` special-cases the null case.
        let probs = unsafe { empty_or_slice(edge_probs, edge_num) };
        let offsets = unsafe { empty_or_slice(edge_offsets, edge_num.checked_add(1).unwrap_or(0)) };
        let vertices = unsafe { empty_or_slice(edge_vertices, edge_vertices_len) };
        // SAFETY: the caller guarantees `config_json` is a valid NUL-terminated C
        // string or null; a null pointer is treated as an empty config.
        let config = if config_json.is_null() {
            &[][..]
        } else {
            unsafe { core::ffi::CStr::from_ptr(config_json) }.to_bytes()
        };

        let graph = match validate_hypergraph(vertex_num, edge_num, probs, offsets, vertices) {
            Ok(graph) => graph,
            Err(message) => {
                set_last_error(&message);
                return STATUS_INVALID_ARG;
            }
        };

        match T::create(graph, config) {
            Ok(decoder) => {
                let handle = Box::new(ExportedHandle {
                    decoder,
                    poisoned: false,
                });
                // SAFETY: out_handle checked non-null above.
                unsafe { out_handle.write(Box::into_raw(handle).cast::<c_void>()) };
                STATUS_OK
            }
            Err(message) => {
                set_last_error(&message);
                STATUS_ERROR
            }
        }
    });
    let Ok(status) = result else {
        set_last_error("create: decoder panicked");
        return STATUS_PANIC;
    };
    status
}

/// Generic implementation of the `deq_decoder_decode` symbol.
///
/// # Safety
///
/// Pointers must satisfy the [`crate::interface::DecodeFn`] contract.
#[doc(hidden)]
pub unsafe fn decode_impl<T: DeqDecoder>(
    handle: *mut c_void,
    syndrome_size: u64,
    syndrome_data: *const u8,
    syndrome_len: usize,
    out_ptr: *mut u64,
    out_cap: usize,
    out_len: *mut usize,
) -> i32 {
    if handle.is_null() || out_len.is_null() {
        set_last_error("decode: null handle or out_len");
        return STATUS_INVALID_ARG;
    }
    if syndrome_len as u64 != syndrome_size.div_ceil(8) {
        set_last_error("decode: syndrome_len does not match syndrome_size");
        return STATUS_INVALID_ARG;
    }
    // SAFETY: handle is a live `ExportedHandle<T>` per the contract, given to this
    // worker exclusively (no concurrent decode/destroy), so a unique `&mut` is sound.
    let handle = unsafe { &mut *handle.cast::<ExportedHandle<T>>() };
    if handle.poisoned {
        set_last_error("decode: handle poisoned by a prior panic");
        return STATUS_POISONED;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: caller guarantees validity for the stated lengths (or null at len 0).
        let data = unsafe { empty_or_slice(syndrome_data, syndrome_len) };
        let syndrome = SyndromeView {
            size: syndrome_size,
            data,
        };
        let mut out = unsafe { OutputBuffer::new(out_ptr, out_cap) };
        match handle.decoder.decode(syndrome, &mut out) {
            Ok(()) => {
                let needed = out.needed;
                // SAFETY: out_len checked non-null above.
                unsafe { out_len.write(needed) };
                if needed > out_cap {
                    crate::interface::STATUS_BUFFER_TOO_SMALL
                } else {
                    STATUS_OK
                }
            }
            Err(message) => {
                set_last_error(&message);
                STATUS_ERROR
            }
        }
    }));
    let Ok(status) = result else {
        handle.poisoned = true;
        set_last_error("decode: decoder panicked");
        return STATUS_PANIC;
    };
    status
}

/// Generic implementation of the `deq_decoder_destroy` symbol.
///
/// # Safety
///
/// `handle` must be a live `ExportedHandle<T>` from `create_impl`, not destroyed
/// before, and not used concurrently with any decode.
#[doc(hidden)]
pub unsafe fn destroy_impl<T: DeqDecoder>(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    // SAFETY: reconstitutes the Box created in create_impl; dropping is single-shot
    // per the contract. A panic in the decoder's destructor is contained.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        drop(unsafe { Box::from_raw(handle.cast::<ExportedHandle<T>>()) });
    }));
}

/// Generic implementation of the `deq_decoder_last_error` symbol.
///
/// Returns the calling thread's most recent error as a NUL-terminated C string, or
/// null if there is none. The pointer borrows thread-local storage and is valid until
/// the next ABI call on the same thread.
#[doc(hidden)]
#[must_use]
pub fn last_error_impl() -> *const c_char {
    LAST_ERROR.with(|cell| {
        let buf = cell.borrow();
        // The buffer is either empty (no error) or NUL-terminated (set_last_error).
        if buf.is_empty() {
            core::ptr::null()
        } else {
            buf.as_ptr().cast::<c_char>()
        }
    })
}

/// Build a slice, treating a null pointer with length 0 as empty (avoids the UB of
/// `from_raw_parts(null, 0)`).
///
/// # Safety
///
/// If `len > 0`, `ptr` must be non-null and valid for `len` reads.
unsafe fn empty_or_slice<'a, U>(ptr: *const U, len: usize) -> &'a [U] {
    if len == 0 {
        &[]
    } else {
        // SAFETY: caller guarantees `ptr` valid for `len` reads when `len > 0`.
        unsafe { core::slice::from_raw_parts(ptr, len) }
    }
}

fn validate_hypergraph<'a>(
    vertex_num: u64,
    edge_num: usize,
    edge_probs: &'a [f64],
    edge_offsets: &'a [u64],
    edge_vertices: &'a [u64],
) -> Result<HypergraphView<'a>, String> {
    if edge_probs.len() != edge_num {
        return Err(format!(
            "edge_probs length {} != edge_num {}",
            edge_probs.len(),
            edge_num
        ));
    }
    if edge_offsets.len() != edge_num + 1 {
        return Err(format!(
            "edge_offsets length {} != edge_num + 1 ({})",
            edge_offsets.len(),
            edge_num + 1
        ));
    }
    if edge_offsets[0] != 0 {
        return Err(format!("edge_offsets[0] = {} (must be 0)", edge_offsets[0]));
    }
    if !edge_offsets.is_sorted() {
        return Err("edge_offsets is not monotonically non-decreasing".to_string());
    }
    if edge_offsets[edge_num] != edge_vertices.len() as u64 {
        return Err(format!(
            "edge_offsets[edge_num] = {} != edge_vertices length {}",
            edge_offsets[edge_num],
            edge_vertices.len()
        ));
    }
    for (index, &prob) in edge_probs.iter().enumerate() {
        if !prob.is_finite() || prob <= 0.0 || prob >= 1.0 {
            return Err(format!("hyperedge {index} probability {prob} must be in (0, 1)"));
        }
    }
    for &vertex in edge_vertices {
        if vertex >= vertex_num {
            return Err(format!("vertex {vertex} >= vertex_num {vertex_num}"));
        }
    }
    Ok(HypergraphView {
        vertex_num,
        edge_probs,
        edge_offsets,
        edge_vertices,
    })
}

/// Export the C ABI symbols for a type implementing [`DeqDecoder`].
///
/// Emits `deq_decoder_abi_version`, `deq_decoder_create`, `deq_decoder_decode`,
/// `deq_decoder_destroy`, and `deq_decoder_last_error` as `extern "C"` functions that
/// delegate to the crate's generic, panic-safe implementations. Invoke once per
/// `cdylib` plugin crate.
#[macro_export]
macro_rules! declare_decoder {
    ($decoder:ty) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn deq_decoder_abi_version() -> u32 {
            $crate::plugin::abi_version_impl()
        }

        /// # Safety
        /// See [`deq_decoder_abi::interface::CreateFn`].
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn deq_decoder_create(
            vertex_num: u64,
            edge_num: u64,
            edge_probs: *const f64,
            edge_offsets: *const u64,
            edge_vertices: *const u64,
            edge_vertices_len: usize,
            config_json: *const ::core::ffi::c_char,
            out_handle: *mut *mut ::core::ffi::c_void,
        ) -> i32 {
            unsafe {
                $crate::plugin::create_impl::<$decoder>(
                    vertex_num,
                    edge_num,
                    edge_probs,
                    edge_offsets,
                    edge_vertices,
                    edge_vertices_len,
                    config_json,
                    out_handle,
                )
            }
        }

        /// # Safety
        /// See [`deq_decoder_abi::interface::DecodeFn`].
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn deq_decoder_decode(
            handle: *mut ::core::ffi::c_void,
            syndrome_size: u64,
            syndrome_data: *const u8,
            syndrome_len: usize,
            out_ptr: *mut u64,
            out_cap: usize,
            out_len: *mut usize,
        ) -> i32 {
            unsafe {
                $crate::plugin::decode_impl::<$decoder>(
                    handle,
                    syndrome_size,
                    syndrome_data,
                    syndrome_len,
                    out_ptr,
                    out_cap,
                    out_len,
                )
            }
        }

        /// # Safety
        /// See [`deq_decoder_abi::interface::DestroyFn`].
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn deq_decoder_destroy(handle: *mut ::core::ffi::c_void) {
            unsafe { $crate::plugin::destroy_impl::<$decoder>(handle) }
        }

        /// # Safety
        /// See [`deq_decoder_abi::interface::LastErrorFn`].
        #[unsafe(no_mangle)]
        pub extern "C" fn deq_decoder_last_error() -> *const ::core::ffi::c_char {
            $crate::plugin::last_error_impl()
        }
    };
}
