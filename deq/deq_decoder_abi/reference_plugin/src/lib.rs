//! Reference deq decoder plugin.
//!
//! A deliberately trivial decoder that exports the full C ABI via
//! [`deq_decoder_abi::declare_decoder!`]. It serves two purposes:
//!
//! * **Header generation:** cbindgen expands this crate to read the real
//!   `#[no_mangle] extern "C"` symbols emitted by the macro, producing
//!   `../include/deq_decoder.h`.
//! * **Integration testing:** the compiled `cdylib` is loaded through the real
//!   `libloading` host path in the ABI crate's integration tests.
//!
//! The decoder itself returns every hyperedge incident to a set vertex: a
//! correct-but-naive "light up everything that touched a defect" rule. It is not
//! a real decoder; it exists only to exercise the boundary deterministically.

use deq_decoder_abi::plugin::{DeqDecoder, HypergraphView, OutputBuffer, SyndromeView};

/// ABI constants surfaced to cbindgen so the generated C header carries them.
///
/// cbindgen cannot evaluate a cross-crate path constant, so these carry literal
/// values. The `const` assertions below fail to compile if any literal ever
/// diverges from its `deq_decoder_abi` source, so they cannot drift silently.
pub mod abi_constants {
    /// ABI revision; see [`deq_decoder_abi::ABI_VERSION`].
    pub const DEQ_DECODER_ABI_VERSION: u32 = 1;
    /// decode succeeded; see [`deq_decoder_abi::STATUS_OK`].
    pub const DEQ_DECODER_STATUS_OK: i32 = 0;
    /// output buffer too small; see [`deq_decoder_abi::STATUS_BUFFER_TOO_SMALL`].
    pub const DEQ_DECODER_STATUS_BUFFER_TOO_SMALL: i32 = 1;
    /// generic recoverable error; see [`deq_decoder_abi::STATUS_ERROR`].
    pub const DEQ_DECODER_STATUS_ERROR: i32 = -1;
    /// invalid argument; see [`deq_decoder_abi::STATUS_INVALID_ARG`].
    pub const DEQ_DECODER_STATUS_INVALID_ARG: i32 = -2;
    /// plugin panicked, handle poisoned; see [`deq_decoder_abi::STATUS_PANIC`].
    pub const DEQ_DECODER_STATUS_PANIC: i32 = -3;
    /// handle already poisoned; see [`deq_decoder_abi::STATUS_POISONED`].
    pub const DEQ_DECODER_STATUS_POISONED: i32 = -4;

    const _: () = {
        assert!(DEQ_DECODER_ABI_VERSION == deq_decoder_abi::ABI_VERSION);
        assert!(DEQ_DECODER_STATUS_OK == deq_decoder_abi::STATUS_OK);
        assert!(DEQ_DECODER_STATUS_BUFFER_TOO_SMALL == deq_decoder_abi::STATUS_BUFFER_TOO_SMALL);
        assert!(DEQ_DECODER_STATUS_ERROR == deq_decoder_abi::STATUS_ERROR);
        assert!(DEQ_DECODER_STATUS_INVALID_ARG == deq_decoder_abi::STATUS_INVALID_ARG);
        assert!(DEQ_DECODER_STATUS_PANIC == deq_decoder_abi::STATUS_PANIC);
        assert!(DEQ_DECODER_STATUS_POISONED == deq_decoder_abi::STATUS_POISONED);
    };
}

/// Stores each hyperedge's vertex list, indexed by hyperedge id.
struct ReferenceDecoder {
    edges: Vec<Vec<u64>>,
}

impl DeqDecoder for ReferenceDecoder {
    fn create(graph: HypergraphView<'_>, _config_json: &[u8]) -> Result<Self, String> {
        let edges = graph.edges().map(|(_, vertices)| vertices.to_vec()).collect();
        Ok(Self { edges })
    }

    fn decode(&mut self, syndrome: SyndromeView<'_>, out: &mut OutputBuffer) -> Result<(), String> {
        for (index, vertices) in self.edges.iter().enumerate() {
            if vertices.iter().any(|&v| syndrome.is_set(v)) {
                out.push(index as u64);
            }
        }
        Ok(())
    }
}

deq_decoder_abi::declare_decoder!(ReferenceDecoder);
