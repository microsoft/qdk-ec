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
//! The decoder itself returns every hyperedge with a positive probability that is
//! incident to a set vertex: a correct-but-naive "light up everything that touched
//! a defect" rule. A zero-probability edge is dormant and stays unselected unless a
//! request reweights it to a positive probability. It is not a real decoder; it
//! exists only to exercise the boundary deterministically.

use deq_decoder_abi::interface::{
    DEQ_DECODER_CAPABILITY_LOSS, DEQ_DECODER_CAPABILITY_REWEIGHTS, DEQ_DECODER_CAPABILITY_SEED, DeqDecoderCapabilities,
    DeqDecoderEdgeReweight,
};
use deq_decoder_abi::plugin::{DecodeRequest, DeqDecoder, HypergraphView, OutputBuffer, SyndromeView};

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
    /// Number of bits per byte in the packed syndrome; see
    /// [`deq_decoder_abi::interface::DEQ_DECODER_SYNDROME_BITS_PER_BYTE`].
    pub const DEQ_DECODER_SYNDROME_BITS_PER_BYTE: u64 = 8;
    /// Indicates support for `decoder_seed`; see
    /// [`deq_decoder_abi::interface::DEQ_DECODER_CAPABILITY_SEED`].
    pub const DEQ_DECODER_CAPABILITY_SEED: u64 = 1 << 0;
    /// Indicates support for `reweights`; see
    /// [`deq_decoder_abi::interface::DEQ_DECODER_CAPABILITY_REWEIGHTS`].
    pub const DEQ_DECODER_CAPABILITY_REWEIGHTS: u64 = 1 << 1;
    /// Indicates support for `loss`; see
    /// [`deq_decoder_abi::interface::DEQ_DECODER_CAPABILITY_LOSS`].
    pub const DEQ_DECODER_CAPABILITY_LOSS: u64 = 1 << 2;

    const _: () = {
        assert!(DEQ_DECODER_ABI_VERSION == deq_decoder_abi::ABI_VERSION);
        assert!(DEQ_DECODER_STATUS_OK == deq_decoder_abi::STATUS_OK);
        assert!(DEQ_DECODER_STATUS_BUFFER_TOO_SMALL == deq_decoder_abi::STATUS_BUFFER_TOO_SMALL);
        assert!(DEQ_DECODER_STATUS_ERROR == deq_decoder_abi::STATUS_ERROR);
        assert!(DEQ_DECODER_STATUS_INVALID_ARG == deq_decoder_abi::STATUS_INVALID_ARG);
        assert!(DEQ_DECODER_STATUS_PANIC == deq_decoder_abi::STATUS_PANIC);
        assert!(DEQ_DECODER_STATUS_POISONED == deq_decoder_abi::STATUS_POISONED);
        assert!(DEQ_DECODER_SYNDROME_BITS_PER_BYTE == deq_decoder_abi::interface::DEQ_DECODER_SYNDROME_BITS_PER_BYTE);
        assert!(DEQ_DECODER_CAPABILITY_SEED == deq_decoder_abi::interface::DEQ_DECODER_CAPABILITY_SEED);
        assert!(DEQ_DECODER_CAPABILITY_REWEIGHTS == deq_decoder_abi::interface::DEQ_DECODER_CAPABILITY_REWEIGHTS);
        assert!(DEQ_DECODER_CAPABILITY_LOSS == deq_decoder_abi::interface::DEQ_DECODER_CAPABILITY_LOSS);
    };
}

/// Stores each hyperedge's loaded prior and vertex list, indexed by hyperedge id.
struct ReferenceDecoder {
    edges: Vec<(f64, Vec<u64>)>,
}

impl ReferenceDecoder {
    /// Every hyperedge with a positive probability that touches a set vertex, in
    /// ascending index order. A reweight assigns its edge's probability for this call
    /// only, replacing the loaded prior.
    ///
    /// Scans `reweights` once per edge, which is fine for a test fixture.
    fn selected_edges(&self, syndrome: SyndromeView<'_>, reweights: &[DeqDecoderEdgeReweight]) -> Vec<u64> {
        (0u64..)
            .zip(&self.edges)
            .filter(|&(index, (prior, vertices))| {
                let probability = reweights
                    .iter()
                    .find(|reweight| reweight.edge == index)
                    .map_or(*prior, |reweight| reweight.probability);
                probability > 0.0 && vertices.iter().any(|&vertex| syndrome.is_set(vertex))
            })
            .map(|(index, _)| index)
            .collect()
    }
}

impl DeqDecoder for ReferenceDecoder {
    const CAPABILITIES: DeqDecoderCapabilities =
        DEQ_DECODER_CAPABILITY_SEED | DEQ_DECODER_CAPABILITY_REWEIGHTS | DEQ_DECODER_CAPABILITY_LOSS;

    fn create(graph: HypergraphView<'_>, _config_json: &[u8]) -> Result<Self, String> {
        let edges = graph
            .edges()
            .map(|(probability, vertices)| (probability, vertices.to_vec()))
            .collect();
        Ok(Self { edges })
    }

    fn decode(&mut self, syndrome: SyndromeView<'_>, out: &mut OutputBuffer) -> Result<(), String> {
        for index in self.selected_edges(syndrome, &[]) {
            out.push(index);
        }
        Ok(())
    }

    /// Gives each optional field a distinct, deterministic effect for ABI tests:
    ///
    /// * reweights assign probabilities: zero switches an edge off, and a positive
    ///   value switches a dormant edge on;
    /// * loss sites append their source edges;
    /// * odd seeds reverse the final order.
    fn decode_request(&mut self, request: DecodeRequest<'_>, out: &mut OutputBuffer) -> Result<(), String> {
        let mut selected = self.selected_edges(request.syndrome, request.reweights);
        if let Some(loss) = request.loss {
            for site in loss.sites() {
                selected.extend_from_slice(site.source_edges);
            }
        }
        if request.decoder_seed.is_some_and(|seed| seed % 2 == 1) {
            selected.reverse();
        }
        for index in selected {
            out.push(index);
        }
        Ok(())
    }
}

deq_decoder_abi::declare_decoder!(ReferenceDecoder);
