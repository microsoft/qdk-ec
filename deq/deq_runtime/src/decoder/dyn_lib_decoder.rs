//! Dynamic-library decoder: load any decoder plugin from a binary-only shared
//! library at runtime via the stable C ABI ([`deq_decoder_abi`]).
//!
//! This is fully decoder-agnostic. The plugin is named only by a filesystem path
//! in the config; the hypergraph is forwarded as CSR and the decoder-specific
//! parameters as an opaque JSON blob that the plugin interprets. Any decoder that
//! exports the ABI — Tetracube, a cudaqx wrapper, a third-party binary — is loaded
//! through this one type.
//!
//! deq gives each worker its own decoder instance (the ABI grants exclusive,
//! non-reentrant access to a handle), so the plugin is hosted by
//! [`ThreadPoolingDecoder`]: one [`LoadedDecoder`] per pooled instance, built via
//! the plugin's `create`. The shared library itself is `dlopen`ed once and cached.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use deq_decoder_abi::host::{DecoderLibrary, HostDecodeRequest, LoadedDecoder};
use deq_decoder_abi::interface::{
    DEQ_DECODER_CAPABILITY_LOSS, DEQ_DECODER_CAPABILITY_REWEIGHTS, DEQ_DECODER_CAPABILITY_SEED,
};
use deq_decoder_abi::plugin::LossSiteView;
use serde::{Deserialize, Serialize};
#[cfg(feature = "cli")]
use structdoc::StructDoc;

use crate::decoder::blackbox_decoder::{DecodingHypergraph, ParityFactor};
use crate::decoder::thread_pooling::{
    DecodeError, DecodeRequest, DecoderInstance, ThreadPoolingConfig, ThreadPoolingDecoder,
};
use crate::decoder::{DecoderFeatures, blackbox_util};

/// ABI capability bits must match the corresponding internal feature bits.
const _: () = {
    assert!(DecoderFeatures::SEED.bits() as u64 == DEQ_DECODER_CAPABILITY_SEED);
    assert!(DecoderFeatures::REWEIGHTS.bits() as u64 == DEQ_DECODER_CAPABILITY_REWEIGHTS);
    assert!(DecoderFeatures::LOSS.bits() as u64 == DEQ_DECODER_CAPABILITY_LOSS);
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct DynLibDecoderConfig {
    /// thread-pool config (parallel = rayon worker count; 0 = `num_cpus`)
    #[serde(flatten)]
    pub thread_pooling_config: ThreadPoolingConfig,

    /// filesystem path to the decoder plugin shared library (.so/.dylib/.dll)
    pub library: PathBuf,

    /// decoder-specific parameters, forwarded verbatim to the plugin as a JSON
    /// object. Its schema is defined by the plugin, not by deq.
    #[cfg_attr(feature = "cli", structdoc(skip))]
    #[serde(default)]
    pub decoder_config: serde_json::Value,
}

/// Process-wide cache of loaded plugin libraries, keyed by path. A library is
/// `dlopen`ed once and never unloaded (see [`DecoderLibrary::load`]); caching the
/// `&'static` avoids re-opening (and re-leaking) it for every hypergraph load.
fn library_cache() -> &'static Mutex<HashMap<PathBuf, &'static DecoderLibrary>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, &'static DecoderLibrary>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn get_or_load_library(path: &Path) -> &'static DecoderLibrary {
    let mut cache = library_cache().lock().unwrap();
    if let Some(library) = cache.get(path) {
        return library;
    }
    // SAFETY: the path comes from trusted local decoder config (never from a
    // remote request); loading runs the plugin's initialization code.
    let library = unsafe { DecoderLibrary::load(path) }
        .unwrap_or_else(|e| panic!("failed to load decoder plugin {}: {e}", path.display()));
    cache.insert(path.to_path_buf(), library);
    library
}

pub struct DynLibInstance {
    loaded: LoadedDecoder,
    /// Plugin-local to stable hyperedge indices, for a plugin that received only the
    /// active edges. `None` when it advertises reweights or loss and so received the
    /// complete graph: its indices are already stable, and `validate_parity_factor`
    /// range-checks them.
    active_edges: Option<Vec<u64>>,
}

/// Convert the plugin's capability bitmask to deq's internal flags.
///
/// The loader has already rejected unknown ABI bits. `from_bits` also catches any
/// future difference between the ABI and internal bit layouts.
fn library_features(library: &'static DecoderLibrary) -> DecoderFeatures {
    let bits = library.capabilities();
    u32::try_from(bits)
        .ok()
        .and_then(DecoderFeatures::from_bits)
        .unwrap_or_else(|| panic!("decoder plugin advertises capability bits {bits:#x} that deq cannot represent"))
}

impl DecoderInstance for DynLibInstance {
    fn supported_features(config: &serde_json::Value) -> DecoderFeatures {
        let config: DynLibDecoderConfig =
            serde_json::from_value(config.clone()).expect("invalid dynamic-library decoder configuration");
        library_features(get_or_load_library(&config.library))
    }

    fn new(hypergraph: &DecodingHypergraph, config: &serde_json::Value) -> Self {
        let config: DynLibDecoderConfig =
            serde_json::from_value(config.clone()).expect("invalid dynamic-library decoder configuration");
        let library = get_or_load_library(&config.library);
        let features = library_features(library);

        // A plugin that accepts reweights or structured loss must see the complete
        // stable graph: a request may activate a dormant zero-prior edge or refer to
        // one, and both address edges by their stable index.
        let complete_graph = features.intersects(DecoderFeatures::REWEIGHTS | DecoderFeatures::LOSS);
        let selected: Vec<u64> = if complete_graph {
            (0..hypergraph.hyperedges.len() as u64).collect()
        } else {
            blackbox_util::active_edge_indices(hypergraph)
        };

        let mut edge_probs = Vec::with_capacity(selected.len());
        let mut edge_offsets = Vec::with_capacity(selected.len() + 1);
        edge_offsets.push(0u64);
        let mut edge_vertices = Vec::new();
        for &edge in &selected {
            let hyperedge = &hypergraph.hyperedges[edge as usize];
            edge_probs.push(hyperedge.probability);
            edge_vertices.extend_from_slice(&hyperedge.vertices);
            edge_offsets.push(edge_vertices.len() as u64);
        }

        let decoder_config = serde_json::to_string(&config.decoder_config).expect("serialize decoder_config");
        let loaded = LoadedDecoder::create(
            library,
            hypergraph.vertex_num,
            &edge_probs,
            &edge_offsets,
            &edge_vertices,
            &decoder_config,
        )
        .unwrap_or_else(|e| panic!("plugin {} failed to build decoder: {e}", config.library.display()));

        let active_edges = (!complete_graph).then_some(selected);
        Self { loaded, active_edges }
    }

    fn decode(&mut self, request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError> {
        // deq's BitVector is already the dense MSB-first packing the ABI expects,
        // so it passes through with no conversion.
        let sites: Vec<LossSiteView<'_>> = request
            .loss
            .map(|loss| {
                loss.sites
                    .iter()
                    .map(|site| LossSiteView {
                        source_edges: &site.source_edges,
                        continuation_edges: &site.continuation_edges,
                        probability: site.probability,
                        children: &site.children,
                        heralds: &site.heralds,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let host_request = HostDecodeRequest {
            syndrome_size: request.syndrome.size,
            syndrome_data: &request.syndrome.data,
            decoder_seed: request.decoder_seed,
            reweights: request.reweights,
            // Present with no sites must stay distinguishable from absent.
            loss: request.loss.map(|_| sites.as_slice()),
        };

        let mut subgraph = Vec::new();
        match self.loaded.decode_request(&host_request, &mut subgraph) {
            Ok(()) => {
                let Some(active_edges) = &self.active_edges else {
                    return Ok(ParityFactor { subgraph });
                };
                let subgraph = subgraph
                    .into_iter()
                    .map(|index| {
                        usize::try_from(index)
                            .ok()
                            .and_then(|index| active_edges.get(index))
                            .copied()
                            .ok_or_else(|| {
                                DecodeError::Backend(format!(
                                    "decoder plugin returned edge {index}; the loaded graph has {} edges",
                                    active_edges.len()
                                ))
                            })
                    })
                    .collect::<Result<_, _>>()?;
                Ok(ParityFactor { subgraph })
            }
            Err(error) => Err(DecodeError::Backend(error.to_string())),
        }
    }

    fn reset(&mut self) {}
}

pub type DynLibDecoder = ThreadPoolingDecoder<DynLibInstance>;
