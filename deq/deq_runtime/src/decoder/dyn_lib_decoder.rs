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

use deq_decoder_abi::host::{DecoderLibrary, LoadedDecoder};
use serde::{Deserialize, Serialize};
#[cfg(feature = "cli")]
use structdoc::StructDoc;

use crate::decoder::blackbox_decoder::{DecodingHypergraph, ParityFactor};
use crate::decoder::thread_pooling::{DecoderInstance, ThreadPoolingConfig, ThreadPoolingDecoder};
use crate::util::BitVector;

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
}

impl DecoderInstance for DynLibInstance {
    fn new(hypergraph: &DecodingHypergraph, config: &serde_json::Value) -> Self {
        let config: DynLibDecoderConfig = serde_json::from_value(config.clone()).expect("invalid DynLibDecoderConfig");
        let library = get_or_load_library(&config.library);

        // Flatten the hypergraph into CSR for the ABI.
        let mut edge_probs = Vec::with_capacity(hypergraph.hyperedges.len());
        let mut edge_offsets = Vec::with_capacity(hypergraph.hyperedges.len() + 1);
        edge_offsets.push(0u64);
        let mut edge_vertices = Vec::new();
        for hyperedge in &hypergraph.hyperedges {
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

        Self { loaded }
    }

    fn decode(&mut self, syndrome: &BitVector) -> ParityFactor {
        // deq's BitVector is already the dense MSB-first packing the ABI expects,
        // so it passes through with no conversion.
        let mut subgraph = Vec::new();
        match self.loaded.decode(syndrome.size, &syndrome.data, &mut subgraph) {
            Ok(()) => ParityFactor { subgraph },
            // DecoderInstance::decode has no error channel; panic so ThreadPoolingDecoder's
            // catch_unwind turns it into a gRPC Status::internal, mirroring PythonDecoder.
            Err(e) => panic!("dylib decode failed: {e}"),
        }
    }

    fn reset(&mut self) {}
}

pub type DynLibDecoder = ThreadPoolingDecoder<DynLibInstance>;
