//! Stable C ABI for deq dynamic decoder plugins.
//!
//! This crate defines the contract that lets `deq` load a quantum-error-correction
//! decoder at runtime from a binary-only shared library (`.so`/`.dylib`/`.dll`),
//! with no recompilation of `deq` and no per-shot serialization. The entire decode
//! search runs in-process at native speed behind a single function-pointer call; the
//! only one-time cost is the dynamic-library load.
//!
//! The crate is split into three parts:
//!
//! * **Interface** ([`interface`]): the ABI version, status codes, exported symbol
//!   names, and `extern "C"` function-pointer types. These are the single source of
//!   truth shared by both sides of the boundary.
//! * **Plugin** ([`plugin`]): a safe Rust trait [`plugin::DeqDecoder`] plus the
//!   [`declare_decoder!`] macro, which a plugin author uses to export the `extern "C"`
//!   symbols without writing any `unsafe` code.
//! * **Host loader** (the `host` module, requires the `host` feature): `deq`'s side, which
//!   `dlopen`s a plugin, validates its ABI version, and exposes safe wrappers.
//!
//! # Boundary discipline
//!
//! Only plain-old-data crosses the boundary: integers, floats, and raw pointers with
//! explicit lengths. No Rust `std` types, no `Vec`, no `String`, no protobuf messages.
//! This keeps the ABI independent of the Rust compiler version, the standard library
//! layout, and the dependency graph on either side.
//!
//! ## Hypergraph encoding (CSR)
//!
//! A decoding hypergraph is passed to [`create`](interface::CreateFn) in compressed
//! sparse row form:
//!
//! * `vertex_num`: number of vertices (detector axis of the syndrome).
//! * `edge_num`: number of hyperedges.
//! * `edge_probs[i]`: independent error probability of hyperedge `i`, in the open
//!   interval `(0, 1)`. Values that are not finite, `<= 0`, or `>= 1` are invalid
//!   and must fail `create`.
//! * `edge_offsets`: length `edge_num + 1`, with `edge_offsets[0] == 0`, monotonically
//!   non-decreasing, and `edge_offsets[edge_num] == edge_vertices_len`.
//! * `edge_vertices`: length `edge_vertices_len`; the vertices of hyperedge `i` are
//!   `edge_vertices[edge_offsets[i] .. edge_offsets[i + 1]]`, each `< vertex_num`.
//!
//! ## Decode I/O
//!
//! [`decode`](interface::DecodeFn) receives the syndrome as a dense bit vector
//! mirroring deq's `BitVector` (`syndrome_size` bits packed MSB-first into
//! `syndrome_data`), and writes the resulting subgraph (the indices of the selected
//! hyperedges) into a caller-owned output buffer. If the buffer is too small, the call
//! returns [`STATUS_BUFFER_TOO_SMALL`] and sets `*out_len` to the required length; the
//! output contents are then unspecified and the caller retries with a larger buffer.
//!
//! ## Threading
//!
//! deq gives each handle to a single worker: [`decode`](interface::DecodeFn) is never
//! called concurrently with another `decode` or with [`destroy`](interface::DestroyFn)
//! on the same handle, so a decoder may keep and mutate per-handle state freely
//! without any locking. deq parallelizes by building one handle per worker (calling
//! `create` more than once), not by sharing a handle across threads. `destroy` is
//! called exactly once per handle.
//!
//! ## Errors
//!
//! Fallible functions return a negative status and record a human-readable message in
//! a thread-local buffer retrievable via [`last_error`](interface::LastErrorFn), which
//! returns a NUL-terminated C string (or null if there is none). The message is valid
//! only until the next ABI call on the same thread, so the caller must read it on the
//! same thread immediately after the failing call.
//!
//! ## Panics / exceptions
//!
//! Rust panics and C++ exceptions must never unwind across the boundary. The plugin
//! layer wraps every entry point in [`std::panic::catch_unwind`] and converts a panic
//! into [`STATUS_PANIC`], marking the handle poisoned so that subsequent calls fail.

#![deny(unsafe_op_in_unsafe_fn)]

pub mod interface;
pub mod plugin;

#[cfg(feature = "host")]
pub mod host;

pub use interface::{
    ABI_VERSION, STATUS_BUFFER_TOO_SMALL, STATUS_ERROR, STATUS_INVALID_ARG, STATUS_OK, STATUS_PANIC, STATUS_POISONED,
};
