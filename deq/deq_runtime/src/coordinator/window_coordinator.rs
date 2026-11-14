//! Window coordinator
//!
//! This coordinator implements adaptive window decoding using a **dynamic,
//! decode-driven parallel commit region** algorithm. Each hop-counted gadget
//! (one with physical measurements / syndrome extraction rounds) determines
//! its commit region at decode time based on the current committed/committing
//! landscape, enabling parallel decoding waves.
//!
//! ### Parameters
//!
//! - `buffer_radius`: Minimum hop-distance from any window boundary for a
//!   gadget to be committed. Ensures each committed gadget has sufficient
//!   decoder context on all sides.
//! - `lookahead_radius`: Additional hops to explore beyond the mandatory
//!   zone.  Gives nearby gadgets a chance to be committed in the
//!   same window if they satisfy the `buffer_radius` boundary requirement.
//!   Set to 0 to only explore the mandatory zone.
//! - `window_parallelism`: `sliding` (default) waits for causal history to
//!   commit, retaining spatial parallelism between independent branches.
//!   `fully_parallel` permits temporal parallelism as well, subject to window reservations.
//!   `serial` waits for lower-GID leader candidates before lookahead and decoding.
//! - `forced_gap_strategy`: With forced-gap scoring, `eager`
//!   computes commit-region readout and boundary scores right after the commit,
//!   while `lazy` computes only scores needed by requested logical readouts.
//! - Effective window radius = `buffer_radius + lookahead_radius`.
//!
//! ### Gadget state
//!
//! Commitment and reservation are independent. `committed` means the gadget's
//! correction is fixed; `reserved_by` identifies the active window using its
//! syndrome. Both fields are published together through the existing watch
//! channel, with transitions under the global `gadgets.write()` lock.
//! Reservations block overlapping windows until commit and buffer release.
//!
//! ### Boundary-distance commit rule
//!
//! A hop-counted gadget in the explored window is safe to commit if its
//! minimum hop-distance to any open window boundary ≥ `buffer_radius`.
//! Step 1's blocking BFS guarantees the center gadget always satisfies
//! this condition.
//!
//! Free-hop gadgets (no physical measurements) contribute 0 to hop distance
//! and are absorbed into the commit region when adjacent to it or to
//! already-committed gadgets. With zero buffer radius, every gadget self-commits.
//! Its decoder uses only its own error models, not uncommitted neighbors' errors.
//!
//! ### Five-step window exploration
//!
//! Window construction is decomposed into five clearly separated steps:
//!
//! 1. **`explore_mandatory_zone`** — Mandatory context.  Blocking 0-1 BFS
//!    from the center, up to `buffer_radius` hops.  Waits for unconnected
//!    output ports so the center is guaranteed to have `buffer_radius`
//!    buffer on all sides.
//! 2. **`await_mandatory_zone_syndrome`** — Wait for syndrome.  Blocks
//!    until all check models in the mandatory zone have computed their
//!    syndrome.  While waiting, more gadgets may arrive, improving step 3.
//! 3. **`explore_lookahead_zone`** — Best-effort expansion.  Non-blocking BFS,
//!    `lookahead_radius` additional hops beyond step 1.  Discovers gadgets
//!    that might be committable.
//! 4. **`select_commit_region`** — Maximize commits.  Boundary-distance
//!    BFS from the window edge inward.  Commits ALL gadgets whose
//!    `boundary_dist ≥ buffer_radius`.  No further exploration.
//! 5. **`shrink_window`** — Trim to minimal decoder window.  BFS from
//!    committed gadgets up to `buffer_radius`, intersected with the
//!    explored window.  Produces a smaller window for the decoder.
//!
//! ### Decode flow
//!
//! When a hop-counted gadget's `decode()` is called:
//!   1. Load outcomes and raw readouts.
//!   2. `explore_mandatory_zone()` + `await_mandatory_zone_syndrome()` +
//!      `explore_lookahead_zone()`: discover the window. Before lookahead,
//!      `sliding` waits for causal history; `serial` waits for lower-GID leader candidates.
//!   3. Commit loop: return the frame if already committed; otherwise wait for
//!      conflicting reservations, then `select_commit_region()` + `shrink_window()`.
//!      Include ready remote checks and reserve the entire context atomically.
//!   4. Leader (center GID) runs `decode_and_commit()`.
//!   5. After decode: mark the commit region as committed, clear owned
//!      reservations without changing buffer commitment, then wait for `pauli_frame`.
//!
//! ### Parallel wave decoding
//!
//! The commit loop reserves the geometric window and its ready remote-check
//! dependencies together, including committed check owners. No context is added
//! after reservation. Windows with independent check contexts decode in parallel.
//! Forced-gap queries use a snapshot of that decoding problem and never mutate
//! shared syndromes. With `lookahead_radius > 0`, each window commits multiple
//! gadgets, reducing the number of decode waves.
//!
//! ### Implementation layout
//!
//! Shared state, configuration, and constructors remain here. Private sibling
//! modules own collection and pins (`collection`), dependency-aware discovery
//! (`exploration`), graph construction and commitment (`decoding`), forced-gap
//! history (`scoring`), reusable error-type facts (`error_metadata`), and request
//! publication/reset (`service`).
//! Modules share this coordinator rather than introducing independent state owners.
//!

use crate::bin;
use crate::coordinator;
use crate::coordinator::{DecoderCacheKey, DecoderReweighting, FingerprintSource, LoadedDecoder, LossHandler, LossStrategy};
use crate::decoder::DynDecoder;
use crate::misc::index::CollectedIds;
use crate::misc::pauli_frame_tracker::PauliFrameTracker;
use crate::misc::sync::TaskCounter;
use crate::util::BitVector;
use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
#[cfg(feature = "cli")]
use structdoc::StructDoc;
use tokio::sync::{Mutex, RwLock, Semaphore, watch};
use tokio_util::sync::CancellationToken;

mod collection;
mod decoding;
mod error_metadata;
mod exploration;
mod scoring;
mod service;

use decoding::PreparedWindow;
use error_metadata::CachedErrorRemotes;
pub use exploration::{ExplorePhase, ExploredWindow};
pub use scoring::CommitRegionDecoder;
use scoring::ForcedGapState;

pub mod trace {
    include!("../proto/deq.coordinator.window_coordinator.rs");
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
pub struct WindowCoordinatorConfig {
    /// if sanity check on the parity factor result from the decoder: every decoder
    /// should return a parity factor that exactly produces the observed syndrome
    #[serde(default)]
    pub assert_parity_factor: bool,
    /// Merge hyperedges with the same syndrome (enabled by default). Ignored by
    /// the ``handoff`` loss strategy because structured loss addresses distinct
    /// error mechanisms by edge index and deduplication would destroy that
    /// distinction.
    #[serde(default = "default_true")]
    pub merge_hyperedges: bool,
    /// Reuse loaded backend instances. When disabled, every decode constructs
    /// a fresh instance; bounded immutable graph/template reuse remains enabled.
    #[serde(default = "default_true")]
    pub persistent_decoder: bool,
    /// Transport policy for shot-scoped probability updates: ``auto`` uses
    /// loaded reweights when supported and otherwise materializes a one-shot
    /// graph; ``enabled`` requires decoder support; ``disabled`` always
    /// materializes updates.
    #[serde(default)]
    pub decoder_reweighting: DecoderReweighting,
    /// Compare the selected correction with an opposite-target correction
    /// returned by the same decoder. Scores need not be calibrated.
    /// The caller decides whether to reject results based on these scores.
    ///
    /// Currently disables automatic garbage collection of completed window and
    /// JIT execution state, which is retained until reset for scoring history.
    /// Memory can grow with circuit length, even with `persistent_decoder=false`.
    #[serde(default)]
    pub forced_gap: bool,
    /// Return output-observable scores without adding forced-gap targets.
    /// Empty when any output is internal to a multi-gadget commit region.
    #[serde(default)]
    pub frame_probabilities: bool,
    /// With `forced_gap` enabled, compute commit-region readout and boundary scores
    /// eagerly, or only as requested. Buffer-only outputs are not scoring targets.
    #[serde(default)]
    pub forced_gap_strategy: ForcedGapStrategy,
    /// Window scheduling: ``sliding`` waits for causal history to commit;
    /// ``fully_parallel`` also permits temporal parallelism; ``serial`` waits for lower-GID leader candidates.
    #[serde(default)]
    pub window_parallelism: WindowParallelism,
    /// Minimum hop-distance from any window boundary required for a gadget to
    /// be committed.  Ensures each committed gadget has sufficient decoder
    /// context on all sides.  The effective window radius is
    /// `buffer_radius + lookahead_radius`.
    #[serde(default = "default_buffer_radius")]
    pub buffer_radius: usize,
    /// How many additional hops beyond `buffer_radius` to explore in
    /// best-effort (non-blocking) mode.  The effective exploration radius is
    /// `buffer_radius + lookahead_radius`.  The extra exploration gives nearby
    /// gadgets a chance to be committed if they satisfy the `buffer_radius`
    /// boundary-distance requirement.  Set to 0 to only explore the mandatory
    /// zone.
    ///
    /// Default: same as `buffer_radius`.
    #[serde(default, alias = "center_radius")]
    pub lookahead_radius: Option<usize>,
    /// optional filepath to write a protobuf trace of all execution events;
    /// the trace is written on each reset() call
    #[serde(default)]
    pub trace_filepath: Option<String>,
    /// when ``true`` (the default), each bit of ``Outcomes.outcomes`` whose
    /// position is set in the accompanying ``Outcomes.loss_mask`` is `XOR`ed with
    /// a uniformly random bit before the coordinator computes the syndrome.
    #[serde(default = "default_true")]
    pub loss_random_imputation: bool,
    /// optional seed for the loss-random-imputation RNG.  When ``None``, the
    /// RNG is seeded from the OS entropy pool at coordinator construction.
    #[serde(default)]
    pub loss_random_imputation_seed: Option<u64>,
    /// What to do with atom losses heralded by ``Outcomes.loss_mask``:
    /// ``"reweight"`` (the default) raises Pauli-envelope edges, ``"ignore"``
    /// drops the loss information, and ``"handoff"`` passes structured sites
    /// to a loss-aware decoder. Independent of random imputation. Loss models
    /// come from ``GadgetType.loss_model`` in the loaded library.
    #[serde(default)]
    pub loss_strategy: LossStrategy,
    /// Options for the chosen ``loss_strategy``, parsed and validated at
    /// construction.  ``ignore`` and ``handoff`` take none; ``reweight`` takes
    /// ``{"weight_fraction": f}`` with ``f`` in ``(0, 1]`` (default ``0.5``).
    #[serde(default)]
    #[cfg_attr(feature = "cli", structdoc(leaf = "JSON object"))]
    pub loss_config: serde_json::Value,
}

impl WindowCoordinatorConfig {
    /// Resolved lookahead_radius: defaults to buffer_radius when not specified.
    /// Forced to 0 when buffer_radius is 0 (single-shot isolation mode).
    pub fn lookahead_radius(&self) -> usize {
        if self.buffer_radius == 0 {
            return 0;
        }
        self.lookahead_radius.unwrap_or(self.buffer_radius)
    }

    /// Effective window radius: `buffer_radius + lookahead_radius`.
    /// This is how far the BFS expands from the center gadget.
    pub fn effective_window_radius(&self) -> usize {
        self.buffer_radius + self.lookahead_radius()
    }
}

fn default_true() -> bool {
    true
}

fn default_buffer_radius() -> usize {
    1
}

/// Shared state for window decoding.
///
/// Acquire library and execution-map locks in field order. Release them before
/// waiting for circuit inputs, decoder work, or another task's commitment.
/// The collection-generation and preparation gates serialize whole operations;
/// acquire those gates before taking library or execution-map locks.
pub struct WindowCoordinator {
    /// Fixed for this coordinator's lifetime, including across shot resets.
    pub config: WindowCoordinatorConfig,
    /// library data
    pub port_types: RwLock<HashMap<u64, Arc<bin::PortType>>>,
    pub gadget_types: RwLock<HashMap<u64, Arc<bin::GadgetType>>>,
    pub check_model_types: Arc<RwLock<HashMap<u64, Arc<bin::CheckModelType>>>>,
    pub error_model_types: RwLock<HashMap<u64, Arc<bin::ErrorModelType>>>,
    referenced_error_remotes: RwLock<HashMap<u64, CachedErrorRemotes>>,
    /// execution data
    pub gadgets: Arc<RwLock<HashMap<u64, Gadget>>>,
    pub check_models: Arc<RwLock<HashMap<u64, CheckModel>>>,
    pub error_models: Arc<RwLock<HashMap<u64, ErrorModel>>>,
    /// Gadgets with finalized readouts; live dependencies may still retain their state.
    completed_readouts: Mutex<HashSet<u64>>,
    /// Request generation used to coalesce concurrent garbage-collection passes.
    collection_requests: AtomicU64,
    /// Last successfully collected generation; the mutex serializes collection passes.
    collection_completed: Mutex<u64>,
    /// Active pin counts by gadget ID, protecting state needed by in-flight work.
    pinned_gadgets: Arc<std::sync::Mutex<HashMap<u64, usize>>>,
    /// Retired gadget ID ranges, used to reject ID reuse until reset.
    collected_gids: Mutex<CollectedIds>,
    /// Retired check-model ID ranges, used to reject reuse and stale remote references.
    collected_cids: Mutex<CollectedIds>,
    /// Retired error-model ID ranges, used to reject ID reuse until reset.
    collected_eids: Mutex<CollectedIds>,
    /// Type-unload requests waiting for live models to release their references.
    pending_type_unloads: Mutex<coordinator::UnloadLibrary>,
    /// Error models waiting for a target gadget to get a binding check model.
    /// Key: target gadget GID. Value: list of eids to register in referring_eids
    /// once the check model is created. Cleared on reset.
    pending_referring_by_gid: Mutex<HashMap<u64, Vec<u64>>>,
    /// Incoming absolute-CID referrals waiting for check model creation. Cleared on reset.
    pending_referring_by_cid: Mutex<HashMap<u64, Vec<u64>>>,
    /// Error models blocked on an unconnected output port.
    /// Key: (source_gid, output_port). Value: pending referrals to re-resolve
    /// when the port is connected. Cleared on reset.
    pending_referring_by_port: Mutex<HashMap<(u64, u64), Vec<PendingPortReferral>>>,
    /// next id counters for auto-assignment
    pub next_gid: Mutex<u64>,
    pub next_cid: Mutex<u64>,
    pub next_eid: Mutex<u64>,
    /// Persistent graphs keyed by relative structure, error-type fingerprints,
    /// construction-time modifiers, commit region, and logical target signature.
    pub loaded_decoders: RwLock<HashMap<DecoderCacheKey, LoadedDecoder>>,
    /// Bounded immutable-graph reuse without solver persistence; cleared on shot reset.
    /// Configuration is fixed per coordinator, so only window-specific state belongs in the key.
    prepared_windows: std::sync::Mutex<lru::LruCache<DecoderCacheKey, Arc<PreparedWindow>>>,
    /// Prepared graphs and loaded decoder handles retained across shot resets, using the same key.
    /// Entries are removed when their types retire or the library/decoder is reset.
    persistent_prepared_windows: std::sync::Mutex<HashMap<DecoderCacheKey, Arc<PreparedWindow>>>,
    /// Serializes cache-miss preparation and loading; cache hits bypass this lock.
    prepared_window_preparation: Mutex<()>,
    /// the decoder service
    pub decoder: DynDecoder,
    /// Bounds ready decoding problems, not input admission or pending windows.
    pub decoding_slots: Arc<Semaphore>,
    gap_decoder: Option<DynDecoder>,
    /// Pauli frame tracker
    pub pauli_frame_tracker: Mutex<PauliFrameTracker>,
    /// Forced gap state, if enabled
    forced_gap_state: Option<RwLock<ForcedGapState>>,
    /// Cancelled on reset, shutdown, or decoder failure to abort pending work.
    /// After a decoder failure, reset is required before decoding another shot.
    pub cancellation: RwLock<CancellationToken>,
    /// Tracks active spawned tasks; reset() waits for all to finish before clearing state.
    pub task_counter: Arc<TaskCounter>,
    /// Deterministic loss imputation keyed by seed, gadget, and measurement.
    loss_imputation_seed: Option<u64>,
    /// Validated loss strategy, built from ``config.loss_strategy`` and
    /// ``config.loss_config`` at construction.
    pub loss_handler: LossHandler,
    /// Whether this decoder accepts shot-scoped loaded reweights.
    pub use_loaded_reweights: bool,
    /// accumulated trace for the current shot
    pub trace_shot: Arc<Mutex<trace::Shot>>,
    /// accumulated trace across all shots
    pub trace: Mutex<trace::WindowCoordinatorTrace>,
}

/// Commitment and exclusive window ownership, published together.
///
/// Transitions happen under the global `gadgets.write()` lock.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GadgetState {
    /// Whether the gadget's own correction is fixed.
    pub committed: bool,
    /// Leader of the window reserving this gadget, including buffer-only use.
    pub reserved_by: Option<u64>,
}

impl GadgetState {
    fn release_buffer(&self, leader_gid: u64) -> Option<Self> {
        (self.reserved_by == Some(leader_gid)).then_some(Self {
            committed: self.committed,
            reserved_by: None,
        })
    }
}

pub struct Gadget {
    pub instance: bin::Gadget,
    pub outcomes: watch::Sender<Option<BitVector>>,
    pub probability_modifiers: Vec<(u64, bin::ProbabilityModifier)>,
    pub loss_mask: Option<BitVector>,
    /// Seed from this gadget's loaded `Outcomes`; meaningful only while outcomes are present.
    pub decoder_seed: Option<u64>,
    /// the check model's cid that is binding to this gadget
    pub binding_cid: Option<u64>,
    /// the peer gadgets' gid connected to each output port
    pub outputs: Vec<watch::Sender<Option<bin::gadget::Connector>>>,
    /// the updated pauli frame
    pub pauli_frame: watch::Sender<Option<BitVector>>,
    pub correction_count: u64,
    pub correction_weight: f64,
    /// whether this gadget has no physical measurements (free-hop gate);
    /// free-hop gadgets contribute 0 to hop distance in window exploration.
    /// They may still have check models and error models with physical
    /// errors that need to be corrected.
    pub is_free_hop: bool,
    /// Commitment and reservation state. Other decode tasks watch this for release.
    pub state: watch::Sender<GadgetState>,
}

pub struct CheckModel {
    pub instance: bin::CheckModel,
    /// the list of eid attaching to this check model
    pub attaching_eid_vec: Vec<u64>,
    pub error_model_ready: watch::Sender<Option<()>>,
    /// the modified remote gadgets
    pub modified_remote_gadgets: Arc<Vec<Option<bin::check_model_type::RemoteGadget>>>,
    /// the expanded remote gadgets
    pub expanded_remote_gadgets: Option<Vec<Option<u64>>>,
    /// the syndrome value
    pub syndrome: watch::Sender<Option<BitVector>>,
    pub syndrome_count: u64,
    /// error models (by eid) from OTHER gadgets that reference this check model
    /// via remote check model chains. Used to determine safe terminal condition:
    /// a committed gadget is a safe terminal only if all referring_eids' gadgets
    /// are also committed.
    pub referring_eids: Vec<u64>,
}

#[derive(Clone)]
pub struct ErrorModel {
    pub instance: bin::ErrorModel,
    /// the modified remote check models
    pub modified_remote_check_models: Arc<Vec<Option<bin::error_model_type::RemoteCheckModel>>>,
}

/// Per-coordinator [`FingerprintSource`] adapter for the window
/// `ErrorModel` wrapper.  See [`crate::coordinator::build_modifier_fingerprints`].
impl FingerprintSource for ErrorModel {
    fn instance(&self) -> &bin::ErrorModel {
        &self.instance
    }
    fn modified_remote_check_models(&self) -> &Arc<Vec<Option<bin::error_model_type::RemoteCheckModel>>> {
        &self.modified_remote_check_models
    }
}

/// Deferred referral waiting for an output port to be connected.
/// When the port is connected, the remote check model chain is re-resolved
/// to complete the `referring_eids` registration.
struct PendingPortReferral {
    eid: u64,
    /// Which remote check model index was blocked.
    ri: usize,
    owner_gid: u64,
    modified_remote: Arc<Vec<Option<bin::error_model_type::RemoteCheckModel>>>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(rename_all = "snake_case")]
pub enum WindowParallelism {
    /// Commit causal history first; independent spatial branches may run concurrently.
    #[default]
    Sliding,
    /// Allow spatial and temporal parallelism, subject to window reservations.
    FullyParallel,
    /// Serialize leaders; with zero lookahead gives repeatable windows for the same shot and instruction sequence.
    Serial,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(rename_all = "snake_case")]
pub enum ForcedGapStrategy {
    /// Compute and cache commit-region readout and boundary scores after committing.
    Eager,
    /// Compute and cache only scores required by a requested logical readout.
    #[default]
    Lazy,
}

impl WindowCoordinator {
    pub fn new(config: serde_json::Value, decoder: DynDecoder) -> Self {
        Self::with_gap_decoder(config, decoder, None)
    }

    /// Use `decoder` for hard corrections and `gap_decoder` for forced alternatives.
    /// Passing `None` shares the hard decoder for both operations.
    ///
    /// # Panics
    /// Panics if the configuration is invalid or its loss/reweight settings are unsupported.
    #[must_use]
    pub fn with_gap_decoder(config: serde_json::Value, decoder: DynDecoder, gap_decoder: Option<DynDecoder>) -> Self {
        let config: WindowCoordinatorConfig = serde_json::from_value(config).unwrap();
        assert!(
            !config.frame_probabilities || config.forced_gap,
            "frame_probabilities requires forced_gap"
        );
        let use_loaded_reweights = config
            .decoder_reweighting
            .use_loaded(config.persistent_decoder, decoder.features())
            .unwrap_or_else(|error| panic!("invalid decoder reweighting configuration: {error}"));
        config
            .decoder_reweighting
            .use_loaded(config.persistent_decoder, gap_decoder.as_ref().unwrap_or(&decoder).features())
            .unwrap_or_else(|error| panic!("invalid gap decoder reweighting configuration: {error}"));
        let loss_imputation_seed = if config.loss_random_imputation {
            use rand::Rng;
            Some(config.loss_random_imputation_seed.unwrap_or_else(|| rand::rng().next_u64()))
        } else {
            None
        };
        let loss_handler = LossHandler::new(config.loss_strategy, config.loss_config.clone())
            .unwrap_or_else(|error| panic!("invalid loss configuration: {error}"));
        let forced_gap_state = config.forced_gap.then(|| RwLock::new(ForcedGapState::new()));
        let decoding_slots = Arc::new(Semaphore::new(
            decoder
                .thread_pool()
                .or_else(|| gap_decoder.as_ref().and_then(DynDecoder::thread_pool))
                .map_or(Semaphore::MAX_PERMITS, |pool| {
                    pool.current_num_threads().saturating_mul(2).min(Semaphore::MAX_PERMITS)
                }),
        ));
        assert!(
            !config.forced_gap || !loss_handler.hands_off_to_decoder(),
            "forced_gap does not support loss_strategy \"handoff\"; use \"reweight\" or \"ignore\""
        );
        Self {
            config,
            port_types: Default::default(),
            gadget_types: Default::default(),
            check_model_types: Default::default(),
            error_model_types: Default::default(),
            referenced_error_remotes: RwLock::default(),
            gadgets: Default::default(),
            check_models: Default::default(),
            error_models: Default::default(),
            completed_readouts: Mutex::default(),
            collection_requests: AtomicU64::new(0),
            collection_completed: Mutex::new(0),
            pinned_gadgets: Arc::default(),
            collected_gids: Mutex::default(),
            collected_cids: Mutex::default(),
            collected_eids: Mutex::default(),
            pending_type_unloads: Mutex::default(),
            pending_referring_by_gid: Default::default(),
            pending_referring_by_cid: Mutex::default(),
            pending_referring_by_port: Default::default(),
            next_gid: Mutex::new(1),
            next_cid: Mutex::new(1),
            next_eid: Mutex::new(1),
            loaded_decoders: Default::default(),
            prepared_windows: std::sync::Mutex::new(lru::LruCache::new(std::num::NonZeroUsize::new(128).unwrap())),
            persistent_prepared_windows: std::sync::Mutex::default(),
            prepared_window_preparation: Mutex::new(()),
            decoder,
            decoding_slots,
            gap_decoder,
            pauli_frame_tracker: Default::default(),
            forced_gap_state,
            cancellation: RwLock::default(),
            task_counter: TaskCounter::new(),
            loss_imputation_seed,
            loss_handler,
            use_loaded_reweights,
            trace_shot: Arc::new(Mutex::new(trace::Shot::default())),
            trace: Mutex::new(trace::WindowCoordinatorTrace::default()),
        }
    }

    fn gap_decoder(&self) -> &DynDecoder {
        self.gap_decoder.as_ref().unwrap_or(&self.decoder)
    }

    /// Fire the cancellation token to abort pending decode tasks. Used by
    /// [`crate::server::LocalServer::shutdown`] to propagate runtime shutdown
    /// into the service layer. Unlike the `reset` RPC handler, this does not
    /// wait for tasks to finish, does not refresh the token, and does not
    /// clear coordinator state — it just signals every cancellable point to
    /// bail.
    pub async fn cancel_pending(&self) {
        let token = self.cancellation.read().await;
        token.cancel();
    }

    async fn record_event(&self, event: trace::event::Event) {
        if self.config.trace_filepath.is_some() {
            self.trace_shot.lock().await.events.push(trace::Event {
                timestamp_ns: crate::misc::util::timestamp_ns(),
                event: Some(event),
            });
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/window_coordinator_test.rs"]
mod tests;
