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
//! - `forced_gap_strategy`: With forced-gap scoring, `eager`
//!   computes commit-region readout and boundary scores right after the commit,
//!   while `lazy` computes only scores needed by requested logical readouts.
//! - Effective window radius = `buffer_radius + lookahead_radius`.
//!
//! ### Gadget state machine
//!
//! ```text
//! Uncommitted ──(lock+mark)──> Decoding ──(decode done)──> Committed
//! ```
//!
//! Transitions happen under the global `gadgets.write()` lock.
//! `Decoding` gadgets block overlapping windows from entering their commit
//! phase, ensuring syndrome consistency.
//!
//! ### Boundary-distance commit rule
//!
//! A hop-counted gadget in the explored window is safe to commit if its
//! minimum hop-distance to any open window boundary ≥ `buffer_radius`.
//! Step 1's blocking BFS guarantees the center gadget always satisfies
//! this condition.
//!
//! Free-hop gadgets (no physical measurements) contribute 0 to hop distance
//! and are always absorbed into the commit region when adjacent to it or to
//! already-committed gadgets, to prevent stranding.
//!
//! ### Five-step window exploration
//!
//! Window construction is decomposed into five clearly separated steps:
//!
//! 1. **[`explore_mandatory_zone`]** — Mandatory context.  Blocking 0-1 BFS
//!    from the center, up to `buffer_radius` hops.  Waits for unconnected
//!    output ports so the center is guaranteed to have `buffer_radius`
//!    buffer on all sides.
//! 2. **[`await_mandatory_zone_syndrome`]** — Wait for syndrome.  Blocks
//!    until all check models in the mandatory zone have computed their
//!    syndrome.  While waiting, more gadgets may arrive, improving step 3.
//! 3. **[`explore_lookahead_zone`]** — Best-effort expansion.  Non-blocking BFS,
//!    `lookahead_radius` additional hops beyond step 1.  Discovers gadgets
//!    that might be committable.
//! 4. **[`select_commit_region`]** — Maximize commits.  Boundary-distance
//!    BFS from the window edge inward.  Commits ALL gadgets whose
//!    `boundary_dist ≥ buffer_radius`.  No further exploration.
//! 5. **[`shrink_window`]** — Trim to minimal decoder window.  BFS from
//!    committed gadgets up to `buffer_radius`, intersected with the
//!    explored window.  Produces a smaller window for the decoder.
//!
//! ### Decode flow
//!
//! When a hop-counted gadget's `decode()` is called:
//!   1. Load outcomes and raw readouts.
//!   2. `explore_mandatory_zone()` + `await_mandatory_zone_syndrome()` +
//!      `explore_lookahead_zone()`: discover the window.
//!   3. Commit loop: check own state (Committed/Decoding → wait or retry),
//!      check window for Decoding gadgets (wait if any), then
//!      `select_commit_region()` + `shrink_window()` and mark the decoder
//!      window as `Decoding`.
//!   4. Leader (center GID) runs `decode_and_commit()`.
//!   5. After decode: mark commit_region as `Committed`, release buffer
//!      (mark back to `Uncommitted`), then wait for pauli_frame.
//!
//! ### Parallel wave decoding
//!
//! Because commit regions are computed dynamically, non-overlapping windows
//! can decode in parallel. With `lookahead_radius > 0`, each window commits
//! multiple gadgets, reducing the number of decode waves.
//!

use crate::bin;
use crate::coordinator;
use crate::coordinator::forced_gap_handler::{ForcedGapGraph, ForcedGapProblem};
use crate::coordinator::loss_handler::{RawLossSite, apply_loss_random_imputation, has_loss_model};
use crate::coordinator::reweight_handler::{
    ProjectedErrors, apply_reweights, correction_weights, decode_projected, deduplicate_decoder_input,
    hard_decoding_hypergraph, ignore_edge_isolated_history_vertices, load_projected_decoder, prepare_decoder,
    probability_reweights,
};
use crate::coordinator::{
    DecoderCacheEntry, DecoderCacheKey, DecoderReweighting, FingerprintSource, LossHandler, LossStrategy,
    build_modifier_fingerprints,
};
use crate::decoder::DynDecoder;
use crate::decoder::blackbox_decoder::{self, DecodingHypergraph, EdgeReweight, Hyperedge, ParityFactor};
use crate::decoder::blackbox_util::{assert_parity_factor, is_parity_factor};
use crate::jit::loss_compiler::{GadgetLoss, build_cross_gadget_loss_sites, build_cross_gadget_output_links};
use crate::misc::bit_vector::{self, flip_bit, get_bit, set_bit};
use crate::misc::fastrace::{Event, Span, SpanContext};
use crate::misc::index::{ErrorIndex, WILDCARD};
use crate::misc::pauli_frame_symbolic_propagator::{CorrectionBasis, PauliFrameSymbolicPropagator};
use crate::misc::pauli_frame_tracker::PauliFrameTracker;
use crate::misc::relative_program::{self, RelativeMapping, RelativeProgram};
use crate::misc::sync::{TaskCounter, check_or_receiver, get_or_receiver};
use crate::misc::validation::{
    apply_check_model_reroutes, apply_error_model_reroutes, validate_outcomes, validate_probability_modifier,
};
use crate::util::BitVector;
use binar::{BitVec, BitwiseMut};
use futures_util::future::{join_all, try_join_all};
use hashbrown::{HashMap, HashSet};
use prost::Message;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
#[cfg(feature = "cli")]
use structdoc::StructDoc;
use tokio::sync::{Mutex, RwLock, watch};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status};

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
    /// by default, we load the hypergraph to the decoder service and use it
    /// thereafter; disabling this option will force the coordinator to build
    /// the decoding hypergraph every time and force the decoder service to
    /// build the decoder data structure every time, which could be time consuming
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
    #[serde(default)]
    pub forced_gap: bool,
    /// With `forced_gap` enabled, compute commit-region readout and boundary scores
    /// eagerly, or only as requested. Buffer-only outputs are not scoring targets.
    #[serde(default)]
    pub forced_gap_strategy: ForcedGapStrategy,
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

/// to prevent deadlock, all of the following locks must be acquired in the order of
/// the fields defined below
pub struct WindowCoordinator {
    pub config: WindowCoordinatorConfig,
    /// library data
    pub port_types: RwLock<HashMap<u64, Arc<bin::PortType>>>,
    pub gadget_types: RwLock<HashMap<u64, Arc<bin::GadgetType>>>,
    pub check_model_types: Arc<RwLock<HashMap<u64, Arc<bin::CheckModelType>>>>,
    pub error_model_types: RwLock<HashMap<u64, Arc<bin::ErrorModelType>>>,
    /// execution data
    pub gadgets: Arc<RwLock<HashMap<u64, Gadget>>>,
    pub check_models: Arc<RwLock<HashMap<u64, CheckModel>>>,
    pub error_models: Arc<RwLock<HashMap<u64, ErrorModel>>>,
    /// Error models waiting for a target gadget to get a binding check model.
    /// Key: target gadget GID. Value: list of eids to register in referring_eids
    /// once the check model is created. Cleared on reset.
    pending_referring_by_gid: Mutex<HashMap<u64, Vec<u64>>>,
    /// Error models blocked on an unconnected output port.
    /// Key: (source_gid, output_port). Value: pending referrals to re-resolve
    /// when the port is connected. Cleared on reset.
    pending_referring_by_port: Mutex<HashMap<(u64, u64), Vec<PendingPortReferral>>>,
    /// next id counters for auto-assignment
    pub next_gid: Mutex<u64>,
    pub next_cid: Mutex<u64>,
    pub next_eid: Mutex<u64>,
    /// the loaded decoders, keyed by `(RelativeProgram, mapping.global_eid_of)`.
    ///
    /// See `MonolithicCoordinator::loaded_decoders` for the rationale.  Two
    /// windows with the same `RelativeProgram` structure can still produce
    /// different merged hypergraphs because `decoding_hypergraph()` reads
    /// per-`error_model` modifier state (probability modifier + remote
    /// check-model bias), and which `error_model` each local slot binds to is
    /// determined by `mapping.global_eid_of` — so it must be part of the key.
    pub loaded_decoders: RwLock<HashMap<DecoderCacheKey, WindowDecoderCacheEntry>>,
    /// the decoder service
    pub decoder: DynDecoder,
    gap_decoder: Option<DynDecoder>,
    gap_use_loaded_reweights: bool,
    /// Pauli frame tracker
    pub pauli_frame_tracker: Mutex<PauliFrameTracker>,
    /// Forced gap state, if enabled
    forced_gap_state: Option<RwLock<ForcedGapState>>,
    /// Cancelled on reset()/drop to abort all pending decode/expand tasks.
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

/// State machine for gadget lifecycle in window decoding.
///
/// ```text
/// Uncommitted ──(lock+mark)──> Decoding ──(decode done)──> Committed
/// ```
///
/// Transitions happen under the global `gadgets.write()` lock.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GadgetState {
    /// Not yet part of any commit region.
    Uncommitted,
    /// Part of an active commit region being decoded. `leader_gid` is the
    /// leader (min hop-counted GID) driving the decode.
    Decoding { leader_gid: u64 },
    /// Decode complete, pauli frame set. Acts as a terminal boundary for
    /// future window explorations (BFS stops here).
    Committed,
}

pub struct Gadget {
    pub instance: bin::Gadget,
    pub outcomes: watch::Sender<Option<BitVector>>,
    pub probability_modifiers: Vec<(u64, bin::ProbabilityModifier)>,
    pub loss_mask: Option<BitVector>,
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
    /// Gadget lifecycle state: Uncommitted → Decoding → Committed.
    /// Other decode tasks watch this to detect when blocking gadgets finish.
    pub state: watch::Sender<GadgetState>,
}

pub struct CheckModel {
    pub instance: bin::CheckModel,
    /// the list of eid attaching to this check model
    pub attaching_eid_vec: Vec<u64>,
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

/// Compute the sorted vector of *local* cids that fall in the commit region.
///
/// `committing_cids` is a set of global cids; we filter to those that map to
/// a local cid in this window (i.e., live in `mapping.local_cid_of`), then
/// sort for canonical hashing in [`DecoderCacheKey`].
fn committing_local_cids_sorted(committing_cids: &HashSet<u64>, mapping: &RelativeMapping) -> Vec<u32> {
    let mut out: Vec<u32> = committing_cids
        .iter()
        .filter_map(|gcid| mapping.local_cid_of.get(gcid).map(|&lcid| lcid as u32))
        .collect();
    out.sort_unstable();
    out
}

fn logical_flip_signature(logical_targets: &[CorrectionBasis], mapping: &RelativeMapping) -> Vec<u64> {
    let mut signature = Vec::with_capacity(logical_targets.len() * 3);
    for &target in logical_targets {
        let (kind, gid, index) = match target {
            CorrectionBasis::Readout { gid, index } => (0, gid, index),
            CorrectionBasis::Residual { gid, index } => (1, gid, index),
        };
        signature.extend([
            kind,
            u64::try_from(mapping.local_gid_of[&gid]).unwrap(),
            u64::try_from(index).unwrap(),
        ]);
    }
    signature
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

// ────────────────────────────────────────────────────────────────────────────
// Window exploration data types
// ────────────────────────────────────────────────────────────────────────────

/// Which exploration phase discovered a gadget.
///
/// During window construction, gadgets are tagged with the phase in which
/// they were first added.  This is useful for debugging and visualization:
///
/// - **MandatoryZone** gadgets were found during the mandatory blocking BFS
///   (step 1).  The center gadget is guaranteed to have `buffer_radius`
///   buffer on all sides within this zone.
/// - **LookaheadZone** gadgets were found during the best-effort non-blocking
///   expansion (step 3).  These gadgets may also be committed if they
///   satisfy the `buffer_radius` boundary-distance requirement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExplorePhase {
    MandatoryZone,
    LookaheadZone,
}

/// Result of the five-step window exploration process.
///
/// This struct captures all the information gathered during exploration so
/// that each step's output is inspectable.  It is built incrementally:
///
/// 1. [`explore_mandatory_zone`]: populates `gadgets`, `center_distance`,
///    `phase` for MandatoryZone members, and `frontier` (BFS frontier to
///    continue from).
/// 2. [`await_mandatory_zone_syndrome`]: waits for syndrome; no mutation.
/// 3. [`explore_lookahead_zone`]: extends all fields with LookaheadZone members.
/// 4. [`select_commit_region`]: sets `commit_region` and `committing_cids`.
/// 5. [`shrink_window`]: sets `decoder_window`.
#[derive(Debug)]
pub struct ExploredWindow {
    /// The center gadget that triggered this exploration.
    pub center_gid: u64,
    /// All gadgets discovered during exploration (MandatoryZone ∪ LookaheadZone).
    pub gadgets: HashSet<u64>,
    /// Hop-distance from the center gadget (0-1 BFS distance, free-hops = 0).
    pub center_distance: HashMap<u64, usize>,
    /// Which phase discovered each gadget.
    pub phase: HashMap<u64, ExplorePhase>,
    /// BFS frontier at the end of step 1 — used by step 3 to continue.
    pub frontier: VecDeque<u64>,
    /// (Step 4) Gadgets selected for commit.
    pub commit_region: HashSet<u64>,
    /// (Step 4) Check model CIDs owned by committed gadgets.
    pub committing_cids: HashSet<u64>,
    /// (Step 5) Trimmed decoder window (commit region + minimal buffer).
    pub decoder_window: HashSet<u64>,
}

/// A commit-edge graph retaining every window check row and its edge mapping.
#[derive(Debug)]
pub struct CommitRegionDecoder {
    graph: Arc<ForcedGapGraph>,
    local_edge_of: Vec<Option<u64>>,
}

impl CommitRegionDecoder {
    fn new(
        hypergraph: &DecodingHypergraph,
        logical_flips: &[Vec<u64>],
        retained_edges: &[bool],
        target_count: usize,
        persistent: bool,
    ) -> Self {
        assert_eq!(retained_edges.len(), hypergraph.hyperedges.len());
        assert_eq!(logical_flips.len(), hypergraph.hyperedges.len());
        let mut hyperedges = vec![];
        let mut retained_logical_flips = vec![];
        let local_edge_of = retained_edges
            .iter()
            .enumerate()
            .map(|(edge, &retained)| {
                retained.then(|| {
                    let local_edge = u64::try_from(hyperedges.len()).unwrap();
                    hyperedges.push(hypergraph.hyperedges[edge].clone());
                    retained_logical_flips.push(logical_flips[edge].clone());
                    local_edge
                })
            })
            .collect();
        Self {
            graph: Arc::new(ForcedGapGraph::new(
                Arc::new(DecodingHypergraph {
                    vertex_num: hypergraph.vertex_num,
                    hyperedges,
                }),
                Arc::new(retained_logical_flips),
                target_count,
                persistent,
            )),
            local_edge_of,
        }
    }

    fn project(
        &self,
        hypergraph: &DecodingHypergraph,
        mut syndrome: BitVector,
        baseline: &ParityFactor,
        reweights: Vec<EdgeReweight>,
    ) -> Result<(BitVector, ParityFactor, Vec<EdgeReweight>), Status> {
        if !is_parity_factor(hypergraph, baseline, &syndrome) {
            return Err(Status::internal("forced-gap baseline does not satisfy the syndrome"));
        }
        let subgraph = baseline
            .subgraph
            .iter()
            .filter_map(|&edge| {
                let edge = usize::try_from(edge).unwrap();
                if let Some(local_edge) = self.local_edge_of[edge] {
                    Some(local_edge)
                } else {
                    for &vertex in &hypergraph.hyperedges[edge].vertices {
                        flip_bit(&mut syndrome, vertex);
                    }
                    None
                }
            })
            .collect();
        let reweights = reweights
            .into_iter()
            .filter_map(|reweight| {
                match usize::try_from(reweight.edge)
                    .ok()
                    .and_then(|edge| self.local_edge_of.get(edge))
                {
                    Some(Some(edge)) => Some(Ok(EdgeReweight {
                        edge: *edge,
                        probability: reweight.probability,
                    })),
                    Some(None) => None,
                    None => Some(Err(Status::invalid_argument("reweighted edge is outside the window graph"))),
                }
            })
            .collect::<Result<_, _>>()?;
        Ok((syndrome, ParityFactor { subgraph }, reweights))
    }

    fn problem(
        &self,
        decoder: DynDecoder,
        hypergraph: &DecodingHypergraph,
        syndrome: BitVector,
        baseline: &ParityFactor,
        reweights: Vec<EdgeReweight>,
        use_loaded_reweights: bool,
    ) -> Result<ForcedGapProblem, Status> {
        let (syndrome, baseline, reweights) = self.project(hypergraph, syndrome, baseline, reweights)?;
        Ok(self
            .graph
            .problem(decoder, syndrome, baseline, reweights, use_loaded_reweights))
    }
}

/// A cached full-window hard decoder and its independent commit-only scorer.
pub type WindowDecoderCacheEntry = DecoderCacheEntry<Option<Arc<CommitRegionDecoder>>>;

#[derive(Clone)]
enum ForcedGapScore {
    Ready(Result<f64, Status>),
    Deferred { problem: Arc<ForcedGapProblem>, target: usize },
}

impl ForcedGapScore {
    async fn probability(&self) -> Result<f64, Status> {
        match self {
            Self::Ready(probability) => probability.clone(),
            Self::Deferred { problem, target } => problem.probability(*target).await,
        }
    }
}

struct ForcedGapState {
    symbolic: PauliFrameSymbolicPropagator,
    scores: HashMap<CorrectionBasis, ForcedGapScore>,
}

impl ForcedGapState {
    fn new() -> Self {
        Self {
            symbolic: PauliFrameSymbolicPropagator::new(),
            scores: HashMap::new(),
        }
    }

    fn reset(&mut self) {
        self.symbolic.reset();
        self.scores.clear();
    }

    fn register(&mut self, targets: &[CorrectionBasis], scores: Vec<ForcedGapScore>) {
        assert_eq!(targets.len(), scores.len());
        for (&target, score) in targets.iter().zip(scores) {
            assert!(
                self.scores.insert(target, score).is_none(),
                "forced-gap target registered more than once for {target:?}"
            );
        }
    }

    fn readout_scores(&self, gid: u64) -> Vec<Vec<ForcedGapScore>> {
        self.symbolic
            .readout_components(gid)
            .into_iter()
            .map(|components| {
                components
                    .into_iter()
                    .filter_map(|target| self.scores.get(&target).cloned())
                    .collect()
            })
            .collect()
    }
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
        let use_loaded_reweights = config
            .decoder_reweighting
            .use_loaded(config.persistent_decoder, decoder.features())
            .unwrap_or_else(|error| panic!("invalid decoder reweighting configuration: {error}"));
        let gap_use_loaded_reweights = config
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
            gadgets: Default::default(),
            check_models: Default::default(),
            error_models: Default::default(),
            pending_referring_by_gid: Default::default(),
            pending_referring_by_port: Default::default(),
            next_gid: Mutex::new(1),
            next_cid: Mutex::new(1),
            next_eid: Mutex::new(1),
            loaded_decoders: Default::default(),
            decoder,
            gap_decoder,
            gap_use_loaded_reweights,
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

    /// Wait for a gadget's `pauli_frame` to be set and return it as `Readouts`.
    async fn wait_for_pauli_frame(&self, gid: u64) -> Result<Response<coordinator::Readouts>, Status> {
        let token = self.cancellation.read().await.clone();
        let gadgets = self.gadgets.read().await;
        let gadget = gadgets.get(&gid).ok_or_else(|| Status::not_found(format!("gid={gid}")))?;
        let pauli_frame = get_or_receiver(&gadget.pauli_frame, token.clone());
        drop(gadgets);
        let readouts = match pauli_frame {
            Ok(pf) => Some(pf),
            Err(handle) => handle.await.unwrap_or(None),
        }
        .ok_or_else(|| Status::cancelled("decode cancelled by reset"))?;
        let (syndrome_count, correction_count, correction_weight) = {
            let gadgets = self.gadgets.read().await;
            let gadget = gadgets
                .get(&gid)
                .ok_or_else(|| Status::cancelled("decode cancelled by reset"))?;
            let check_models = self.check_models.read().await;
            let syndrome_count = gadget.binding_cid.map_or(0, |cid| check_models[&cid].syndrome_count);
            (syndrome_count, gadget.correction_count, gadget.correction_weight)
        };
        let probabilities = if self.config.forced_gap {
            self.wait_for_forced_gap_scores(gid, token).await?
        } else {
            vec![]
        };
        Ok((coordinator::Readouts {
            gid,
            readouts: Some(readouts),
            probabilities,
            syndrome_count,
            correction_count,
            correction_weight,
        })
        .into())
    }

    async fn wait_for_forced_gap_scores(&self, gid: u64, token: CancellationToken) -> Result<Vec<f64>, Status> {
        let state = self.forced_gap_state.as_ref().unwrap();
        let readouts = state.read().await.readout_scores(gid);
        let solve = try_join_all(readouts.into_iter().map(|scores| async move {
            let probabilities = try_join_all(scores.iter().map(ForcedGapScore::probability)).await?;
            Ok::<_, Status>(probabilities.into_iter().fold(0.0, f64::max))
        }));
        tokio::select! {
            results = solve => results,
            () = token.cancelled() => {
                Err(Status::cancelled("forced-gap scoring cancelled by reset"))
            }
        }
    }

    async fn register_forced_gap_scores(
        &self,
        logical_targets: &[CorrectionBasis],
        problem: &Result<Arc<ForcedGapProblem>, Status>,
    ) {
        let scores = (0..logical_targets.len())
            .map(|target| match problem {
                Ok(problem) => ForcedGapScore::Deferred {
                    problem: Arc::clone(problem),
                    target,
                },
                Err(error) => ForcedGapScore::Ready(Err(error.clone())),
            })
            .collect();
        self.forced_gap_state
            .as_ref()
            .unwrap()
            .write()
            .await
            .register(logical_targets, scores);
    }

    async fn complete_eager_scores(&self, logical_targets: &[CorrectionBasis], problem: &ForcedGapProblem) {
        let probabilities = join_all((0..logical_targets.len()).map(|target| problem.probability(target))).await;
        let mut state = self.forced_gap_state.as_ref().unwrap().write().await;
        for (&target, probability) in logical_targets.iter().zip(probabilities) {
            *state.scores.get_mut(&target).unwrap() = ForcedGapScore::Ready(probability);
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Five-step window exploration
    //
    // Window exploration is decomposed into five clearly separated steps:
    //
    //   Step 1  explore_mandatory_zone()        — Mandatory context gathering.
    //           Blocking 0-1 BFS from the center, up to `buffer_radius` hops.
    //           Waits for unconnected output ports (future gadgets must arrive
    //           before the center can be safely committed).
    //
    //   Step 2  await_mandatory_zone_syndrome()  — Wait for syndrome.
    //           Blocks until all check models in the mandatory zone have
    //           computed their syndrome.  While waiting, more gadgets may
    //           arrive, improving step 3's reach.
    //
    //   Step 3  explore_lookahead_zone()         — Best-effort expansion.
    //           Continues the BFS from step 1's frontier, expanding up to
    //           `lookahead_radius` additional hops.  Never blocks on unconnected
    //           output ports — includes only what is immediately available.
    //           Also skips hop-counted gadgets whose syndrome has not yet been
    //           computed, so the decoder never waits on lookahead gadgets.
    //           Free-hop gadgets are always included to prevent dangling.
    //           Discovers gadgets that might be committable.
    //
    //   Step 4  select_commit_region() — Maximize commits from explored window.
    //           No further exploration.  Computes boundary distance for every
    //           gadget (inward 0-1 BFS from boundary seeds), then commits ALL
    //           gadgets with boundary_dist ≥ buffer_radius.  The center gadget
    //           is always committed.  Adjacent free-hop gadgets are absorbed.
    //
    //   Step 5  shrink_window()        — Trim to minimal decoder window.
    //           BFS outward from each committed gadget, up to `buffer_radius`
    //           hops, intersected with the explored window.  Produces a smaller
    //           window for the decoder, improving cache reuse and performance.
    //
    // The old `build_window()` combined steps 1 and 3 into a single BFS.
    // The old `compute_commit_region()` performed step 4 but capped commit
    // candidates by center_radius.  Steps 2 and 5 are new.
    // ────────────────────────────────────────────────────────────────────────

    /// **Step 1: Explore mandatory zone** — mandatory blocking BFS.
    ///
    /// Expands a 0-1 BFS from `center_gid` up to `buffer_radius` hops.
    /// Free-hop gadgets contribute 0 to hop distance.  For output ports
    /// within this zone, the BFS **blocks** on unconnected ports (waits for
    /// the downstream gadget to be executed), because the center gadget
    /// cannot be safely committed without `buffer_radius` buffer on all sides.
    ///
    /// Returns an [`ExploredWindow`] with the mandatory-zone gadgets, their
    /// distances, and the BFS frontier ready for step 3.
    /// Returns `None` if cancelled.
    async fn explore_mandatory_zone(&self, center_gid: u64) -> Option<ExploredWindow> {
        let buffer_radius = self.config.buffer_radius;
        let token = self.cancellation.read().await.clone();

        let mut explored = ExploredWindow {
            center_gid,
            gadgets: HashSet::from([center_gid]),
            center_distance: HashMap::from([(center_gid, 0)]),
            phase: HashMap::from([(center_gid, ExplorePhase::MandatoryZone)]),
            frontier: VecDeque::from([center_gid]),
            commit_region: HashSet::new(),
            committing_cids: HashSet::new(),
            decoder_window: HashSet::new(),
        };

        while let Some(fgid) = explored.frontier.pop_front() {
            if token.is_cancelled() {
                return None;
            }
            let my_dist = explored.center_distance[&fgid];
            let mut sync_neighbors: Vec<(u64, bool)> = vec![];
            let mut async_handles: Vec<JoinHandle<Option<bin::gadget::Connector>>> = vec![];

            {
                let gadgets = self.gadgets.read().await;
                let gadget = gadgets.get(&fgid)?;

                // Follow input connectors (always immediately available).
                for connector in &gadget.instance.connectors {
                    if explored.gadgets.contains(&connector.gid) {
                        continue;
                    }
                    let peer = &gadgets[&connector.gid];
                    let peer_dist = if peer.is_free_hop { my_dist } else { my_dist + 1 };
                    if peer_dist <= buffer_radius {
                        explored.gadgets.insert(connector.gid);
                        explored.center_distance.insert(connector.gid, peer_dist);
                        explored.phase.insert(connector.gid, ExplorePhase::MandatoryZone);
                        sync_neighbors.push((connector.gid, peer.is_free_hop));
                    }
                }

                // Follow output ports — blocking wait for unconnected ports,
                // because every direction must have `buffer_radius` buffer.
                // At the boundary (my_dist == buffer_radius), only free-hop
                // neighbors (same distance) could be in range — they don't
                // strengthen the buffer, so we skip blocking for unconnected
                // ports.  This is critical for buffer_radius = 0 (single-shot
                // QEC): each gadget decodes independently with no waits.
                for sender in &gadget.outputs {
                    match get_or_receiver(sender, token.clone()) {
                        Ok(connector) => {
                            if explored.gadgets.contains(&connector.gid) {
                                continue;
                            }
                            let peer = gadgets.get(&connector.gid)?;
                            let peer_dist = if peer.is_free_hop { my_dist } else { my_dist + 1 };
                            if peer_dist <= buffer_radius {
                                explored.gadgets.insert(connector.gid);
                                explored.center_distance.insert(connector.gid, peer_dist);
                                explored.phase.insert(connector.gid, ExplorePhase::MandatoryZone);
                                sync_neighbors.push((connector.gid, peer.is_free_hop));
                            }
                        }
                        Err(handle) => {
                            if my_dist < buffer_radius {
                                async_handles.push(handle);
                            }
                        }
                    }
                }
            }

            // Enqueue sync neighbors: free-hops to front (distance 0), others to back.
            for (gid, is_free_hop) in sync_neighbors {
                let peer_dist = explored.center_distance[&gid];
                if peer_dist < buffer_radius {
                    if is_free_hop {
                        explored.frontier.push_front(gid);
                    } else {
                        explored.frontier.push_back(gid);
                    }
                }
            }

            // Await async output handles.
            for handle in async_handles {
                if let Some(connector) = handle.await.unwrap_or(None) {
                    if explored.gadgets.contains(&connector.gid) {
                        continue;
                    }
                    let gadgets = self.gadgets.read().await;
                    let peer = &gadgets[&connector.gid];
                    let peer_dist = if peer.is_free_hop { my_dist } else { my_dist + 1 };
                    if peer_dist <= buffer_radius {
                        explored.gadgets.insert(connector.gid);
                        explored.center_distance.insert(connector.gid, peer_dist);
                        explored.phase.insert(connector.gid, ExplorePhase::MandatoryZone);
                        if peer_dist < buffer_radius {
                            if peer.is_free_hop {
                                explored.frontier.push_front(connector.gid);
                            } else {
                                explored.frontier.push_back(connector.gid);
                            }
                        }
                    }
                }
            }
        }

        // Rebuild frontier for step 3: all mandatory-zone gadgets at max distance.
        // These are the gadgets whose neighbors were NOT explored because
        // they were at exactly buffer_radius.
        explored.frontier.clear();
        for &gid in &explored.gadgets {
            if explored.center_distance[&gid] == buffer_radius {
                explored.frontier.push_back(gid);
            } else {
                // Free-hop gadgets at distances < buffer_radius might also
                // have unexplored neighbors if all their peers were at
                // buffer_radius, but the BFS already explored them.
                // Only gadgets at exactly the radius boundary remain.
            }
        }

        Some(explored)
    }

    /// **Step 2: Await mandatory-zone syndrome** — wait for syndrome readiness.
    ///
    /// Before exploring the lookahead zone, waits for all syndrome data in
    /// the mandatory zone to become available.  Each check model's syndrome
    /// depends on its owning gadget's outcomes **and** outcomes of remote
    /// gadgets referenced via the check model type.  This method collects
    /// all such gadget GIDs and waits for their outcomes.
    ///
    /// Waiting here serves two purposes:
    /// 1. The decoder cannot run without syndrome — waiting early avoids
    ///    blocking later.
    /// 2. While waiting, more gadgets may be executed by the quantum
    ///    computer, so the subsequent lookahead-zone exploration (step 3)
    ///    discovers more gadgets than it would if run immediately.
    ///
    /// Returns `None` if cancelled.
    async fn await_mandatory_zone_syndrome(&self, explored: &ExploredWindow) -> Option<()> {
        let token = self.cancellation.read().await.clone();

        // Collect CIDs of check models in the mandatory zone.
        let mandatory_cids: Vec<u64> = {
            let gadgets = self.gadgets.read().await;
            explored
                .gadgets
                .iter()
                .filter_map(|&gid| gadgets.get(&gid)?.binding_cid)
                .collect()
        };

        // Wait for each check model's syndrome to be computed.
        // Syndrome computation (spawned during execute) waits internally for
        // the owning gadget's outcomes and all remote gadgets' outcomes, then
        // sets `check_model.syndrome`.  By awaiting syndrome here, we
        // transitively wait for all required physical measurements.
        let mut handles: Vec<JoinHandle<bool>> = vec![];
        {
            let check_models = self.check_models.read().await;
            for cid in &mandatory_cids {
                if let Some(check_model) = check_models.get(cid)
                    && let Err(handle) = check_or_receiver(&check_model.syndrome, token.clone())
                {
                    handles.push(handle);
                }
            }
        }
        futures_util::future::join_all(handles).await;
        if token.is_cancelled() {
            return None;
        }

        Some(())
    }

    /// **Step 3: Explore lookahead zone** — best-effort non-blocking expansion.
    ///
    /// Continues the BFS from step 1's frontier, expanding up to
    /// `lookahead_radius` additional hops.  This step is **entirely
    /// non-blocking** in two ways:
    ///
    /// 1. **Output ports**: only reads what is immediately available from
    ///    the `watch::Sender`.  Unconnected outputs are skipped.
    /// 2. **Syndrome readiness**: only includes hop-counted gadgets whose
    ///    syndrome has already been computed.  Free-hop gadgets are always
    ///    included regardless of syndrome readiness — they typically have
    ///    an empty check model (zero syndrome bits) whose syndrome is
    ///    trivially ready, and including them unconditionally prevents
    ///    dangling free-hops that would block the commit-region absorption
    ///    in step 4.
    ///
    /// The syndrome filter is critical for streaming-mode performance.  In a
    /// streaming scenario the quantum computer executes gadgets ahead of
    /// syndrome availability — the execute call returns before the syndrome
    /// computation finishes.  Including a gadget whose syndrome is not yet
    /// ready would force `decode_and_commit` (step after commit-loop) to
    /// block waiting for it, adding latency that negates the benefit of
    /// non-blocking lookahead.  By excluding such gadgets here, the decoder
    /// window contains only gadgets that are immediately decodable.
    ///
    /// Gadgets discovered here are tagged as `ExplorePhase::LookaheadZone`.
    /// They may still be committed in step 4 if they satisfy the
    /// `buffer_radius` boundary-distance requirement.
    ///
    /// Returns `None` if cancelled.
    async fn explore_lookahead_zone(&self, explored: &mut ExploredWindow) -> Option<()> {
        let lookahead_radius = self.config.lookahead_radius();
        if lookahead_radius == 0 {
            return Some(()); // Nothing to explore beyond the mandatory zone.
        }

        let buffer_radius = self.config.buffer_radius;
        let window_radius = buffer_radius + lookahead_radius;
        let token = self.cancellation.read().await.clone();

        while let Some(fgid) = explored.frontier.pop_front() {
            if token.is_cancelled() {
                return None;
            }
            let my_dist = explored.center_distance[&fgid];
            let mut sync_neighbors: Vec<(u64, bool)> = vec![];

            {
                let gadgets = self.gadgets.read().await;
                let check_models = self.check_models.read().await;
                let gadget = gadgets.get(&fgid)?;

                // Non-blocking syndrome readiness check: a hop-counted gadget
                // is included only if its syndrome has already been computed.
                // Free-hop gadgets are always included — they may have a check
                // model, but it carries zero syndrome bits (trivially ready),
                // and they must not be left dangling outside the window.
                let syndrome_ready = |peer: &Gadget| -> bool {
                    if peer.is_free_hop {
                        return true;
                    }
                    match peer.binding_cid {
                        None => true,
                        Some(cid) => check_models.get(&cid).is_some_and(|cm| cm.syndrome.borrow().is_some()),
                    }
                };

                // Follow input connectors.
                for connector in &gadget.instance.connectors {
                    if explored.gadgets.contains(&connector.gid) {
                        continue;
                    }
                    let peer = &gadgets[&connector.gid];
                    let peer_dist = if peer.is_free_hop { my_dist } else { my_dist + 1 };
                    if peer_dist <= window_radius && syndrome_ready(peer) {
                        explored.gadgets.insert(connector.gid);
                        explored.center_distance.insert(connector.gid, peer_dist);
                        explored.phase.insert(connector.gid, ExplorePhase::LookaheadZone);
                        sync_neighbors.push((connector.gid, peer.is_free_hop));
                    }
                }

                // Follow output ports — non-blocking read only.
                for sender in &gadget.outputs {
                    if let Some(connector) = *sender.borrow() {
                        if explored.gadgets.contains(&connector.gid) {
                            continue;
                        }
                        if let Some(peer) = gadgets.get(&connector.gid) {
                            let peer_dist = if peer.is_free_hop { my_dist } else { my_dist + 1 };
                            if peer_dist <= window_radius && syndrome_ready(peer) {
                                explored.gadgets.insert(connector.gid);
                                explored.center_distance.insert(connector.gid, peer_dist);
                                explored.phase.insert(connector.gid, ExplorePhase::LookaheadZone);
                                sync_neighbors.push((connector.gid, peer.is_free_hop));
                            }
                        }
                    }
                    // Unconnected output → skip (treated as open boundary by step 4).
                }
            }

            // Enqueue sync neighbors.
            for (gid, is_free_hop) in sync_neighbors {
                let peer_dist = explored.center_distance[&gid];
                if peer_dist < window_radius {
                    if is_free_hop {
                        explored.frontier.push_front(gid);
                    } else {
                        explored.frontier.push_back(gid);
                    }
                }
            }
        }

        Some(())
    }

    /// **Step 4: Select commit region** — maximize commits from the explored window.
    ///
    /// This step does **not** explore further.  It works purely on the gadgets
    /// discovered by steps 1 and 3.
    ///
    /// Algorithm:
    /// 1. Identify **boundary seeds**: gadgets with at least one connection
    ///    (input or output) going outside the window.  This includes connections
    ///    to committed gadgets — their corrections are fixed and represent a
    ///    boundary the decoder cannot modify.  Unconnected output ports also
    ///    count as open boundaries.  Only gadgets whose connections ALL stay
    ///    within the window (or have zero ports in a direction) are non-boundary.
    /// 2. Run an inward 0-1 BFS from boundary seeds to compute `boundary_dist`
    ///    for every gadget in the window.  Free-hops contribute 0 to distance.
    ///    Gadgets unreachable from any boundary have `boundary_dist = ∞`
    ///    (e.g., a fully self-contained window with no external connections).
    /// 3. All `Uncommitted` hop-counted gadgets with
    ///    `boundary_dist ≥ buffer_radius` are committed.  The center gadget
    ///    always satisfies this because step 1's blocking BFS guarantees
    ///    `buffer_radius` hops of context in all directions.
    /// 4. Free-hop gadgets adjacent to the commit region (or to already
    ///    committed gadgets) are absorbed iteratively.
    ///
    /// Populates `explored.commit_region` and `explored.committing_cids`.
    fn select_commit_region(&self, explored: &mut ExploredWindow, gadgets: &HashMap<u64, Gadget>) {
        let buffer_radius = self.config.buffer_radius;

        // ── buffer_radius=0: commit only the center gadget ──
        // In single-shot isolation mode, each gadget decodes independently.
        // Skip boundary-distance analysis and free-hop absorption entirely.
        if buffer_radius == 0 {
            explored.commit_region.clear();
            explored.commit_region.insert(explored.center_gid);
            explored.committing_cids.clear();
            if let Some(cid) = gadgets[&explored.center_gid].binding_cid {
                explored.committing_cids.insert(cid);
            }
            return;
        }

        // ── 3.1  Identify boundary seeds ──
        // A gadget is a boundary seed (boundary_dist = 0) if any of its
        // connections go outside the decoder window.  This includes connections
        // to committed gadgets — while their corrections are known, the decoder
        // cannot modify them, so they represent fixed boundary conditions.
        let mut boundary_dist: HashMap<u64, usize> = HashMap::new();
        let mut deque: VecDeque<u64> = VecDeque::new();

        for &gid in &explored.gadgets {
            let gadget = &gadgets[&gid];
            let mut has_open_boundary = false;

            // Check inputs: any connector from outside the window is boundary.
            for connector in &gadget.instance.connectors {
                if !explored.gadgets.contains(&connector.gid) {
                    has_open_boundary = true;
                    break;
                }
            }

            // Check outputs: any output going outside window or unconnected.
            if !has_open_boundary {
                for sender in &gadget.outputs {
                    match sender.borrow().as_ref() {
                        Some(conn) => {
                            if !explored.gadgets.contains(&conn.gid) {
                                has_open_boundary = true;
                                break;
                            }
                        }
                        None => {
                            // Unconnected output port = open boundary.
                            has_open_boundary = true;
                            break;
                        }
                    }
                }
            }

            if has_open_boundary {
                boundary_dist.insert(gid, 0);
                deque.push_back(gid);
            }
        }

        // ── 3.2  Inward 0-1 BFS from boundary seeds ──
        // The step cost is determined by the SOURCE gadget: stepping out of
        // a free-hop gadget is free (it has no measurements, so it doesn't
        // provide decoder context), while stepping out of a hop-counted
        // gadget costs 1 (it provides one layer of syndrome context).
        // This ensures boundary_dist counts how many hop-counted gadgets
        // with actual syndrome data lie between a gadget and the boundary.
        while let Some(fgid) = deque.pop_front() {
            let my_bdist = boundary_dist[&fgid];
            let gadget = &gadgets[&fgid];
            let step = if gadget.is_free_hop { 0 } else { 1 };

            let mut visit = |peer_gid: u64| {
                if !explored.gadgets.contains(&peer_gid) {
                    return;
                }
                let new_dist = my_bdist + step;
                let entry = boundary_dist.entry(peer_gid).or_insert(usize::MAX);
                if new_dist < *entry {
                    *entry = new_dist;
                    let peer = &gadgets[&peer_gid];
                    if peer.is_free_hop {
                        deque.push_front(peer_gid);
                    } else {
                        deque.push_back(peer_gid);
                    }
                }
            };

            for connector in &gadget.instance.connectors {
                visit(connector.gid);
            }
            for sender in &gadget.outputs {
                if let Some(conn) = sender.borrow().as_ref() {
                    visit(conn.gid);
                }
            }
        }

        // Unreached gadgets have no open boundary → infinite distance.
        for &gid in &explored.gadgets {
            boundary_dist.entry(gid).or_insert(usize::MAX);
        }

        // ── 3.3  Build commit region ──
        // All Uncommitted hop-counted gadgets with boundary_dist >= buffer_radius
        // are committed.  The center gadget always satisfies this because
        // step 1's blocking BFS guarantees buffer_radius context in all directions.
        explored.commit_region = HashSet::new();

        for &gid in &explored.gadgets {
            let gadget = &gadgets[&gid];
            if gadget.is_free_hop {
                continue; // handled by absorption below
            }
            if !matches!(*gadget.state.borrow(), GadgetState::Uncommitted) {
                continue; // already committed or being decoded
            }
            let bdist = boundary_dist.get(&gid).copied().unwrap_or(0);
            if bdist >= buffer_radius {
                explored.commit_region.insert(gid);
            }
        }

        // ── 3.4  Absorb free-hop gadgets adjacent to commit region ──
        // Free-hops have no measurements and must not be stranded between
        // committed gadgets and the commit region.
        loop {
            let mut changed = false;
            for &gid in &explored.gadgets {
                if explored.commit_region.contains(&gid) {
                    continue;
                }
                let g = &gadgets[&gid];
                if !g.is_free_hop || !matches!(*g.state.borrow(), GadgetState::Uncommitted) {
                    continue;
                }
                let in_region = |ngid: u64| {
                    explored.commit_region.contains(&ngid)
                        || gadgets
                            .get(&ngid)
                            .is_some_and(|p| matches!(*p.state.borrow(), GadgetState::Committed))
                };
                let adjacent = g.instance.connectors.iter().any(|c| in_region(c.gid))
                    || g.outputs
                        .iter()
                        .any(|s| s.borrow().as_ref().is_some_and(|c| in_region(c.gid)));
                if adjacent {
                    explored.commit_region.insert(gid);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        // ── Collect committing CIDs ──
        debug_assert!(explored.commit_region.iter().all(|g| explored.gadgets.contains(g)));
        debug_assert!(
            explored.commit_region.contains(&explored.center_gid),
            "center gadget {} must be in commit region (boundary_dist={:?}, buffer_radius={})",
            explored.center_gid,
            boundary_dist.get(&explored.center_gid),
            buffer_radius,
        );

        explored.committing_cids.clear();
        for &gid in &explored.commit_region {
            let gadget = &gadgets[&gid];
            if let Some(cid) = gadget.binding_cid {
                explored.committing_cids.insert(cid);
            }
        }
    }

    /// **Step 5: Shrink window** — trim to the minimal decoder window.
    ///
    /// The decoder only needs gadgets within `buffer_radius` hops of any
    /// committed gadget.  This step computes that minimal window:
    ///
    ///   decoder_window = { g ∈ explored.gadgets |
    ///       ∃ c ∈ commit_region : bfs_distance(c, g) ≤ buffer_radius }
    ///
    /// A smaller decoder window improves cache hit rate and decoder performance
    /// by excluding distant lookahead-zone gadgets that do not affect the
    /// committed corrections.
    ///
    /// Populates `explored.decoder_window`.
    fn shrink_window(&self, explored: &mut ExploredWindow, gadgets: &HashMap<u64, Gadget>) {
        let buffer_radius = self.config.buffer_radius;
        explored.decoder_window.clear();

        // BFS outward from every committed gadget, up to buffer_radius hops,
        // restricted to the explored window.
        let mut dist: HashMap<u64, usize> = HashMap::new();
        let mut deque: VecDeque<u64> = VecDeque::new();

        for &cgid in &explored.commit_region {
            dist.insert(cgid, 0);
            deque.push_back(cgid);
            explored.decoder_window.insert(cgid);
        }

        while let Some(fgid) = deque.pop_front() {
            let my_dist = dist[&fgid];
            let gadget = &gadgets[&fgid];

            let mut visit = |peer_gid: u64| {
                if !explored.gadgets.contains(&peer_gid) {
                    return;
                }
                let peer = &gadgets[&peer_gid];
                let step = if peer.is_free_hop { 0 } else { 1 };
                let new_dist = my_dist + step;
                if new_dist > buffer_radius {
                    return;
                }
                let entry = dist.entry(peer_gid).or_insert(usize::MAX);
                if new_dist < *entry {
                    *entry = new_dist;
                    explored.decoder_window.insert(peer_gid);
                    if new_dist < buffer_radius {
                        if peer.is_free_hop {
                            deque.push_front(peer_gid);
                        } else {
                            deque.push_back(peer_gid);
                        }
                    }
                }
            };

            for connector in &gadget.instance.connectors {
                visit(connector.gid);
            }
            for sender in &gadget.outputs {
                if let Some(conn) = sender.borrow().as_ref() {
                    visit(conn.gid);
                }
            }
        }
    }

    /// Runs the decode + commit phase for a single window.
    ///
    /// Builds the decoding problem from the window, calls the decoder, applies
    /// corrections, marks commit_region as Committed, and releases buffer
    /// gadgets (marks remaining Decoding(leader) back to Uncommitted).
    ///
    /// Returns `None` only on cancellation.
    async fn decode_and_commit(
        &self,
        center_gid: u64,
        commit_region: &HashSet<u64>,
        committing_cids: &HashSet<u64>,
        window: &HashSet<u64>,
    ) -> Option<()> {
        let span = Span::root("decode_window", SpanContext::random());
        span.add_property(|| ("center_gid", format!("{center_gid}")));
        span.add_property(|| ("commit_region", format!("{:?}", commit_region)));
        span.add_property(|| ("window", format!("{:?}", window)));

        // first wait for all the outcomes to be loaded to make sure that their check
        // models and error models are completely loaded
        let token = self.cancellation.read().await.clone();
        let mut handles: Vec<JoinHandle<bool>> = vec![];
        {
            let gadgets = self.gadgets.read().await;
            for &window_gid in window {
                let gadget = gadgets.get(&window_gid)?;
                if let Err(handle) = check_or_receiver(&gadget.outcomes, token.clone()) {
                    handles.push(handle);
                }
            }
            // Also wait for outcomes of all commit_region gadgets
            for &cgid in commit_region {
                if !window.contains(&cgid)
                    && let Some(gadget) = gadgets.get(&cgid)
                    && let Err(handle) = check_or_receiver(&gadget.outcomes, token.clone())
                {
                    handles.push(handle);
                }
            }
        }
        futures_util::future::join_all(handles).await;
        if token.is_cancelled() {
            return None;
        }
        span.add_event(Event::new("outcomes_ready"));

        span.add_property(|| ("committing_cids", format!("{:?}", committing_cids)));

        self.record_event(trace::event::Event::Decode(trace::DecodeEvent {
            gid: center_gid,
            is_leader: true,
            leader_gid: center_gid,
            window: {
                let mut v: Vec<_> = window.iter().copied().collect();
                v.sort();
                v
            },
            committing_gids: {
                let mut v: Vec<_> = commit_region.iter().copied().collect();
                v.sort();
                v
            },
            committing_cids: {
                let mut v: Vec<_> = committing_cids.iter().copied().collect();
                v.sort();
                v
            },
        }))
        .await;

        // early return: if no gadget in the commit region has a binding check model
        if committing_cids.is_empty() {
            let gadgets = self.gadgets.read().await;
            let mut tracker = self.pauli_frame_tracker.lock().await;
            for &cgid in commit_region {
                let pauli_frame_gadget = tracker.gadgets.get(&cgid)?;
                let residual = BitVec::zeros(pauli_frame_gadget.num_output_observables());
                let readout_flips = BitVec::zeros(pauli_frame_gadget.num_readouts());
                for (update_gid, pauli_frame) in tracker.load_correction(cgid, residual, readout_flips) {
                    let update_gadget = gadgets.get(&update_gid)?;
                    debug_assert!(update_gadget.pauli_frame.borrow().is_none(), "bug");
                    update_gadget.pauli_frame.send_replace(Some(pauli_frame));
                }
            }
            // transition commit region gadgets: Decoding → Committed
            for &commit_gid in commit_region {
                let gadget = gadgets.get(&commit_gid)?;
                gadget.state.send_replace(GadgetState::Committed);
            }
            // release buffer: mark remaining Decoding(center) gadgets back to Uncommitted
            for &wgid in window {
                if let Some(g) = gadgets.get(&wgid)
                    && matches!(*g.state.borrow(), GadgetState::Decoding { leader_gid } if leader_gid == center_gid)
                {
                    g.state.send_replace(GadgetState::Uncommitted);
                }
            }
            span.add_property(|| ("cid", "None"));
            drop(tracker);
            drop(gadgets);
            return Some(());
        }

        // collect all CIDs in the window (for syndrome waiting)
        let window_cids = {
            let gadgets = self.gadgets.read().await;
            let mut window_cids: HashSet<u64> = HashSet::new();
            for &window_gid in window {
                let gadget = gadgets.get(&window_gid)?;
                if let Some(cid) = gadget.binding_cid {
                    window_cids.insert(cid);
                }
            }
            window_cids
        };
        // wait for these check models to finish expanding their remote gadgets and calculate
        // their check values
        let mut handles: Vec<JoinHandle<bool>> = vec![];
        {
            let check_models = self.check_models.read().await;
            for &window_cid in &window_cids {
                let check_model = check_models.get(&window_cid)?;
                if let Err(handle) = check_or_receiver(&check_model.syndrome, token.clone()) {
                    handles.push(handle);
                }
            }
        }
        futures_util::future::join_all(handles).await;
        if token.is_cancelled() {
            return None;
        }
        span.add_property(|| ("window_cids", format!("{:?}", window_cids)));

        // for error models, we handle them differently from the check models: we don't need to
        // wait for all the remote check models to be expanded before starting to add edges;
        // when a hyperedge connects to some remote check models outside the window, we simply
        // connect the hyperedge to a virtual vertex.

        let mut expanded_gadgets: Vec<relative_program::ExpandedGadget> = vec![];
        {
            let check_model_types = self.check_model_types.read().await;
            let gadgets = self.gadgets.read().await;
            let check_models = self.check_models.read().await;
            let error_models = self.error_models.read().await;
            let decoding_window = if self.config.buffer_radius == 0 {
                window.clone()
            } else {
                Self::retain_committed_check_history(window, &gadgets, &check_models, &error_models)
            };
            let window = &decoding_window;
            let mut gid_vec: Vec<_> = window.iter().copied().collect();
            gid_vec.sort_unstable();

            // Build expanded gadgets for window gadgets.
            // Three configurations:
            //   - Normal (uncommitted with check model): both check_model and error_models
            //   - Check-only (committed with check model): check_model only, error_models = []
            //   - Free-hop (no check model): neither
            for &gid in gid_vec.iter() {
                let gadget = gadgets.get(&gid)?;
                let inputs: Vec<_> = gadget.instance.connectors.iter().cloned().map(Some).collect();
                let outputs: Vec<_> = gadget.outputs.iter().map(|v| *v.borrow()).collect();
                let gtype = gadget.instance.gtype;
                let cid = gadget.binding_cid;
                let is_committed = matches!(*gadget.state.borrow(), GadgetState::Committed);
                let (check_model, error_models) = if let Some(cid) = cid {
                    let check_model = check_models.get(&cid)?;
                    let remote_gadgets = check_model.expanded_remote_gadgets.clone()?;
                    let expanded_check_model = relative_program::ExpandedCheckModel {
                        cid,
                        ctype: check_model.instance.ctype,
                        remote_gadgets,
                        count_checks: check_model_types.get(&check_model.instance.ctype)?.checks.len(),
                    };
                    let expanded_error_models = if is_committed {
                        // Committed gadgets: error effects already applied, exclude
                        // error models from the decoder key and hypergraph.
                        vec![]
                    } else {
                        let mut ems = vec![];
                        for &eid in check_model.attaching_eid_vec.iter() {
                            let error_model = error_models.get(&eid)?;
                            let remote_check_models =
                                Self::expand_remote_check_models_in_window(gid, error_model, &gadgets, window);
                            ems.push(relative_program::ExpandedErrorModel {
                                eid,
                                etype: error_model.instance.etype,
                                remote_check_models,
                            });
                        }
                        ems
                    };
                    (Some(expanded_check_model), expanded_error_models)
                } else {
                    (None, vec![])
                };
                expanded_gadgets.push(relative_program::ExpandedGadget {
                    gid,
                    gtype,
                    inputs,
                    outputs,
                    check_model,
                    error_models,
                });
            }

            // Build expanded gadgets for outside error-contributing gadgets.
            // These are gadgets outside the window whose error models reference
            // check models inside the window (error-only: check_model = None).
            let mut outside_gadget_eids: HashMap<u64, Vec<u64>> = HashMap::new();
            let mut processed_eids: HashSet<u64> = HashSet::new();
            for &gid in gid_vec.iter() {
                let gadget = gadgets.get(&gid)?;
                let Some(cid) = gadget.binding_cid else { continue };
                let check_model = check_models.get(&cid)?;
                for &referring_eid in check_model.referring_eids.iter() {
                    if !processed_eids.insert(referring_eid) {
                        continue;
                    }
                    let error_model = error_models.get(&referring_eid)?;
                    let owner_cid = error_model.instance.cid;
                    let owner_cm = check_models.get(&owner_cid)?;
                    let owner_gid = owner_cm.instance.gid;
                    if window.contains(&owner_gid) {
                        continue; // already in window as normal or check-only
                    }
                    let owner_gadget = gadgets.get(&owner_gid)?;
                    if matches!(*owner_gadget.state.borrow(), GadgetState::Committed) {
                        continue; // committed outside: errors already decoded
                    }
                    outside_gadget_eids.entry(owner_gid).or_default().push(referring_eid);
                }
            }
            let mut outside_gids: Vec<_> = outside_gadget_eids.keys().cloned().collect();
            outside_gids.sort();
            for &gid in outside_gids.iter() {
                let gadget = gadgets.get(&gid)?;
                let inputs: Vec<_> = gadget.instance.connectors.iter().cloned().map(Some).collect();
                // Outside gadgets may have unconnected outputs; read safely.
                let outputs: Vec<_> = gadget.outputs.iter().map(|v| *v.borrow()).collect();
                let eids = &outside_gadget_eids[&gid];
                let mut expanded_error_models = vec![];
                for &eid in eids {
                    let error_model = error_models.get(&eid)?;
                    let remote_check_models = Self::expand_remote_check_models_in_window(gid, error_model, &gadgets, window);
                    expanded_error_models.push(relative_program::ExpandedErrorModel {
                        eid,
                        etype: error_model.instance.etype,
                        remote_check_models,
                    });
                }
                expanded_gadgets.push(relative_program::ExpandedGadget {
                    gid,
                    gtype: gadget.instance.gtype,
                    inputs,
                    outputs,
                    check_model: None, // error-only: no check model
                    error_models: expanded_error_models,
                });
            }
        }
        let (relative_program, mapping) = RelativeProgram::new(&expanded_gadgets);
        let logical_targets = if self.config.forced_gap {
            let gadgets = self.gadgets.read().await;
            let state = self.forced_gap_state.as_ref().unwrap().read().await;
            let mut targets = state.symbolic.readout_targets(
                mapping
                    .global_gid_of
                    .iter()
                    .filter(|gid| commit_region.contains(*gid))
                    .copied(),
            );
            let mut boundaries = vec![];
            for &gid in &mapping.global_gid_of {
                if !commit_region.contains(&gid) {
                    continue;
                }
                for port in 0..gadgets[&gid].outputs.len() {
                    let output = &gadgets[&gid].outputs[port];
                    if output.borrow().as_ref().is_none_or(|peer| !commit_region.contains(&peer.gid)) {
                        boundaries.push((gid, u64::try_from(port).unwrap()));
                    }
                }
            }
            targets.extend(state.symbolic.boundary_targets(&boundaries));
            targets
        } else {
            vec![]
        };
        span.add_event(Event::new("relative_program"));
        span.add_event(Event::new("committing"));

        let (parity_factor, errors, correction_weights, forced_gap_problem) = self
            .decode_parity_factor(committing_cids, &logical_targets, window, &relative_program, &mapping, &span)
            .await;
        let forced_gap_problem = forced_gap_problem.map(|problem| problem.map(Arc::new));
        if let Some(problem) = &forced_gap_problem {
            self.register_forced_gap_scores(&logical_targets, problem).await;
        }
        span.add_event(Event::new("decoded"));

        let selected_errors = Self::global_subgraph_of(&mapping, &errors, &parity_factor.subgraph);
        self.record_event(trace::event::Event::DecodeFinished(trace::DecodeFinishedEvent {
            leader_gid: center_gid,
        }))
        .await;
        span.add_property(|| ("parity_factor", format!("{selected_errors:?}")));

        // update the outcomes of the check models, mark Committed, and release buffer
        self.update_pauli_frame(
            center_gid,
            commit_region,
            committing_cids,
            window,
            &parity_factor,
            &errors,
            &correction_weights,
            &relative_program,
            &mapping,
        )
        .await;
        span.add_event(Event::new("pauli_frame_updated"));

        if self.config.forced_gap_strategy == ForcedGapStrategy::Eager
            && let Some(Ok(problem)) = forced_gap_problem
        {
            self.complete_eager_scores(&logical_targets, &problem).await;
        }
        Some(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn update_pauli_frame(
        &self,
        center_gid: u64,
        commit_region: &HashSet<u64>,
        committing_cids: &HashSet<u64>,
        window: &HashSet<u64>,
        parity_factor: &blackbox_decoder::ParityFactor,
        errors: &ProjectedErrors,
        correction_weights: &[f64],
        relative_program: &RelativeProgram,
        mapping: &RelativeMapping,
    ) {
        let error_model_types = self.error_model_types.read().await;
        let mut gadgets = self.gadgets.write().await;
        let mut check_models = self.check_models.write().await;
        let error_models = self.error_models.read().await;
        let mut tracker = self.pauli_frame_tracker.lock().await;

        // initialize per-gadget residual and readout_flips accumulators
        let mut gadget_residuals: HashMap<u64, BitVec> = HashMap::new();
        let mut gadget_readout_flips: HashMap<u64, BitVec> = HashMap::new();
        for &commit_gid in commit_region {
            let pauli_frame_gadget = tracker.gadgets.get(&commit_gid).unwrap();
            gadget_residuals.insert(commit_gid, BitVec::zeros(pauli_frame_gadget.num_output_observables()));
            gadget_readout_flips.insert(commit_gid, BitVec::zeros(pauli_frame_gadget.num_readouts()));
        }

        // only apply the committed errors
        let mut syndrome_flips: HashMap<u64, HashSet<u64>> = HashMap::new();
        assert_eq!(parity_factor.subgraph.len(), correction_weights.len());
        for (&ei, &weight) in parity_factor.subgraph.iter().zip(correction_weights) {
            let local_error = &errors[ei as usize];
            let local_eid = local_error.eid;
            let eid = mapping.global_eid_of[local_eid];
            let error_index = local_error.error_index;
            let error_model = error_models.get(&eid).unwrap();
            if !committing_cids.contains(&error_model.instance.cid) {
                continue; // skip errors outside the commit region (includes error-only gadgets)
            }
            let error_model_type = error_model_types.get(&error_model.instance.etype).unwrap();
            let error = &error_model_type.errors[error_index];

            // find the gadget that owns this error via its check model
            let error_gadget_gid = check_models.get(&error_model.instance.cid).unwrap().instance.gid;
            let gadget = gadgets.get_mut(&error_gadget_gid).unwrap();
            gadget.correction_count += 1;
            gadget.correction_weight += weight;
            let residual = gadget_residuals.get_mut(&error_gadget_gid).unwrap();
            let readout_flips = gadget_readout_flips.get_mut(&error_gadget_gid).unwrap();

            // update the residual and readout flips for the owning gadget
            for &ri in error.residual.iter() {
                residual.negate_index(ri as usize);
            }
            for &ri in error.readout_flips.iter() {
                readout_flips.negate_index(ri as usize);
            }
            // Update the syndrome of the check models.
            // Remote checks may be projected out (outside window).
            let local_gid = *mapping.local_gid_of.get(&error_gadget_gid).unwrap();
            let local_eid_bias = mapping.local_eid_bias[local_gid];
            let expanded_gadget = &relative_program.local_gadgets[local_gid];
            let expanded_error_model = &expanded_gadget.error_models[local_eid - local_eid_bias];
            assert!(mapping.global_eid_of[expanded_error_model.eid as usize] == eid);
            for check in error.checks.iter() {
                let (cid, check_index) = if let Some(ri) = check.remote_check_model {
                    // Remote check model may be outside the window (projected out
                    // during hypergraph construction). Skip the syndrome flip for it;
                    // a future window decode covering that check model will handle it.
                    let Some(remote_local_cid) = expanded_error_model.remote_check_models[ri as usize] else {
                        continue;
                    };
                    let remote_cid = mapping.global_cid_of[remote_local_cid as usize];
                    let check_index = check.check_index
                        + error_model.modified_remote_check_models[ri as usize]
                            .as_ref()
                            .unwrap()
                            .check_bias;
                    (remote_cid, check_index)
                } else {
                    (error_model.instance.cid, check.check_index)
                };
                if !syndrome_flips.contains_key(&cid) {
                    syndrome_flips.insert(cid, HashSet::new());
                }
                let flips = syndrome_flips.get_mut(&cid).unwrap();
                if flips.contains(&check_index) {
                    flips.remove(&check_index);
                } else {
                    flips.insert(check_index);
                }
            }
        }
        for (cid, flips) in syndrome_flips.iter() {
            let check_model = check_models.get_mut(cid).unwrap();
            let mut syndrome = check_model.syndrome.borrow().clone().unwrap();
            for &check_index in flips.iter() {
                flip_bit(&mut syndrome, check_index);
            }
            check_model.syndrome.send_replace(Some(syndrome));
        }

        // load corrections for all gadgets in the commit region
        for &commit_gid in commit_region {
            let residual = gadget_residuals.remove(&commit_gid).unwrap();
            let readout_flips = gadget_readout_flips.remove(&commit_gid).unwrap();
            let updates = tracker.load_correction(commit_gid, residual, readout_flips);
            for (update_gid, pauli_frame) in updates {
                let update_gadget = gadgets.get(&update_gid).unwrap();
                debug_assert!(update_gadget.pauli_frame.borrow().is_none(), "bug");
                update_gadget.pauli_frame.send_replace(Some(pauli_frame));
            }
        }

        // transition commit region gadgets: Decoding → Committed
        for &commit_gid in commit_region {
            let gadget = gadgets.get_mut(&commit_gid).unwrap();
            gadget.state.send_replace(GadgetState::Committed);
        }

        // release buffer: mark remaining Decoding(center) gadgets back to Uncommitted
        for &wgid in window {
            if let Some(g) = gadgets.get_mut(&wgid)
                && matches!(*g.state.borrow(), GadgetState::Decoding { leader_gid } if leader_gid == center_gid)
            {
                g.state.send_replace(GadgetState::Uncommitted);
            }
        }
    }

    fn global_subgraph_of(mapping: &RelativeMapping, errors: &ProjectedErrors, subgraph: &[u64]) -> Vec<(u64, u64)> {
        subgraph
            .iter()
            .map(|&ei| {
                assert!(
                    (ei as usize) < errors.len(),
                    "decoder returned subgraph edge index {ei} but cached errors has only {} entries — \
                     stale `loaded_decoders` cache (key/value mismatch)",
                    errors.len(),
                );
                let local_error = &errors[ei as usize];
                let local_eid = local_error.eid;
                let eid = mapping.global_eid_of[local_eid];
                let error_index = local_error.error_index;
                (eid, error_index as u64)
            })
            .collect()
    }

    /// Build the possible loss sites for this window from the observed atom
    /// losses, mirroring [`MonolithicCoordinator::build_loss_sites`].
    ///
    /// `config.loss_strategy` is not `ignore` and some gadget currently
    /// loaded recorded observed losses. The generators are keyed by
    /// `(local_eid, error_index)`, matching the `error_reference` that
    /// [`decoding_hypergraph`](Self::decoding_hypergraph) produces, so
    /// The loss handler projects the sites onto this window's hyperedges.
    /// Loss sites are kept per-gadget and ungrouped; cross-gadget continuation
    /// whose herald signature is incomplete in this window is naturally dropped
    /// when its edges fall outside the window.
    async fn build_loss_sites(&self, mapping: &RelativeMapping) -> Vec<RawLossSite> {
        let mut loss_sites = Vec::new();
        if !self.loss_handler.tracks_losses() {
            return loss_sites;
        }
        let port_types = self.port_types.read().await;
        let gadget_types = self.gadget_types.read().await;
        let gadgets = self.gadgets.read().await;
        if !mapping
            .global_gid_of
            .iter()
            .any(|gid| gadgets.get(gid).is_some_and(|gadget| gadget.loss_mask.is_some()))
        {
            return loss_sites;
        }
        let check_models = self.check_models.read().await;

        // Gadgets in this window that carry a loss model, in window order. A
        // gadget with no *observed* loss is still included: a loss can pass
        // through it unheralded and only be resolved by a downstream gadget in the
        // same window (which is why `buffer_radius` must cover the envelope span).
        let mut gid_of_index: Vec<u64> = Vec::new();
        let mut index_of_gid: HashMap<u64, usize> = HashMap::new();
        for &gid in &mapping.global_gid_of {
            let Some(gadget) = gadgets.get(&gid) else { continue };
            if !has_loss_model(&gadget_types, gadget.instance.gtype) {
                continue;
            }
            index_of_gid.insert(gid, gid_of_index.len());
            gid_of_index.push(gid);
        }
        if gid_of_index.is_empty() {
            return loss_sites;
        }

        // The error model whose error list a site's generators index: the first
        // one attached to the gadget, per the `GadgetType.loss_model` contract.
        let local_eid_of_index: Vec<Option<usize>> = gid_of_index
            .iter()
            .map(|&gid| {
                let cid = gadgets.get(&gid)?.binding_cid?;
                let eid = *check_models.get(&cid)?.attaching_eid_vec.first()?;
                mapping.local_eid_of.get(&eid).copied()
            })
            .collect();

        let loss_masks: Vec<BitVector> = gid_of_index
            .iter()
            .map(|&gid| gadgets.get(&gid).and_then(|g| g.loss_mask.clone()).unwrap_or_default())
            .collect();

        // Cross-gadget links: each gadget's output flat slot that a loss can leave
        // on, mapped to the (downstream gadget index, input flat slot) it enters.
        // Links whose downstream gadget is outside this window are dropped, so the
        // chain ends there (a later window covering it reweights it there).
        let gadget_instances: Vec<&bin::Gadget> =
            gid_of_index.iter().map(|&gid| &gadgets.get(&gid).unwrap().instance).collect();
        let output_links_vec = build_cross_gadget_output_links(&gadget_instances, &index_of_gid, &gadget_types, &port_types);

        let gadget_losses: Vec<GadgetLoss> = (0..gid_of_index.len())
            .map(|index| {
                let gtype = gadgets.get(&gid_of_index[index]).unwrap().instance.gtype;
                GadgetLoss {
                    loss_model: gadget_types.get(&gtype).unwrap().loss_model.as_ref().unwrap(),
                    observed: &loss_masks[index],
                    output_links: &output_links_vec[index],
                }
            })
            .collect();

        for site in build_cross_gadget_loss_sites(&gadget_losses) {
            let local_eid = local_eid_of_index[site.gadget_index];
            loss_sites.push(RawLossSite::from_compiled(site, local_eid));
        }
        loss_sites
    }

    async fn decode_parity_factor(
        &self,
        committing_cids: &HashSet<u64>,
        logical_targets: &[CorrectionBasis],
        window_gids: &HashSet<u64>,
        relative_program: &RelativeProgram,
        mapping: &RelativeMapping,
        span: &Span,
    ) -> (
        blackbox_decoder::ParityFactor,
        ProjectedErrors,
        Vec<f64>,
        Option<Result<ForcedGapProblem, Status>>,
    ) {
        let target_count = logical_targets.len();
        // calculate syndrome
        span.add_event(Event::new("calculate_syndrome"));
        let syndrome: BitVector = {
            let mut syndrome = bit_vector::from_sparse_indices(relative_program.count_checks as u64, &[]);
            let check_models = self.check_models.read().await;
            let mut start_index = 0;
            for &cid in &mapping.global_cid_of {
                let check_model = check_models.get(&cid).unwrap();
                let check_syndrome = check_model.syndrome.borrow().clone().unwrap();
                for i in 0..check_syndrome.size {
                    if get_bit(&check_syndrome, i) {
                        set_bit(&mut syndrome, start_index + i, true);
                    }
                }
                start_index += check_syndrome.size;
            }
            assert!(start_index == syndrome.size);
            syndrome
        };
        span.add_event(Event::new("syndrome_calculated"));
        span.add_property(|| ("syndrome", format!("{syndrome:?}")));

        // Assemble the observed atom losses into loss sites within this window.
        // These are shot-dependent only in their *content*: the loaded graph stays
        // fixed and the shot's loss travels with the decode request, so a loss
        // shot is served from the cache like any other.
        let loss_sites = self.build_loss_sites(mapping).await;

        // Deduplication collapses edges that share a syndrome, which renumbers
        // them and, worse for a loss-aware decoder, merges a loss generator with
        // any ordinary Pauli error sharing its syndrome. A coordinator that hands
        // losses to the decoder therefore never deduplicates -- not even on a shot
        // carrying no loss, since the graph it loads then serves later shots that
        // do.
        let deduplicate = self.config.merge_hyperedges && !self.loss_handler.hands_off_to_decoder();

        let logical_flips_are_cacheable = if self.config.forced_gap && target_count != 0 {
            let gadgets = self.gadgets.read().await;
            window_gids.iter().all(|gid| gadgets[gid].instance.modifier.is_none())
        } else {
            true
        };
        let cache_key = if self.config.persistent_decoder && logical_flips_are_cacheable {
            let error_models = self.error_models.read().await;
            let error_model_types = self.error_model_types.read().await;
            // Construction-time modifiers define the base graph and therefore
            // its cache identity. Outcomes.modifiers are shot-scoped assignments
            // projected below; including them here would defeat decoder reuse.
            Some(DecoderCacheKey {
                relative_program: relative_program.clone(),
                error_model_fingerprints: build_modifier_fingerprints(mapping, &error_models, &error_model_types),
                committing_local_cids: committing_local_cids_sorted(committing_cids, mapping),
                logical_flip_signature: logical_flip_signature(logical_targets, mapping),
            })
        } else {
            None
        };

        if let Some(ref cache_key) = cache_key {
            let loaded = self.loaded_decoders.read().await.get(cache_key).cloned();
            if let Some(loaded) = loaded {
                let probability_reweights = self
                    .projected_shot_probability_reweights(mapping, &loaded.decoder.projection)
                    .await;
                let projected =
                    self.loss_handler
                        .project_shot(&loaded.decoder.projection, &probability_reweights, &loss_sites);
                // we can use the loaded decoding hypergraph to call the decoding service
                span.add_event(Event::new("decoding").with_property(|| ("type", "loaded")));
                let decode_syndrome = loaded.decoder.project_syndrome(syndrome.clone());
                let parity_factor = decode_projected(
                    &self.decoder,
                    &loaded.decoder,
                    decode_syndrome.clone(),
                    projected.reweights.clone(),
                    projected.loss,
                    self.use_loaded_reweights,
                )
                .await
                .unwrap();
                if self.config.assert_parity_factor {
                    assert_parity_factor(
                        loaded.decoder.decoding_hypergraph.as_ref().unwrap(),
                        &parity_factor,
                        &decode_syndrome,
                    );
                }
                let weights = loaded.decoder.correction_weights(&parity_factor, &projected.reweights);
                let forced_gap_problem = loaded.scoring.as_ref().map(|scoring| {
                    scoring.problem(
                        self.gap_decoder().clone(),
                        loaded.decoder.decoding_hypergraph.as_ref().unwrap(),
                        decode_syndrome,
                        &parity_factor,
                        projected.reweights,
                        self.gap_use_loaded_reweights,
                    )
                });
                return (parity_factor, projected.errors, weights, forced_gap_problem);
            }
        }

        // when the decoder is not available, construct the decoding hypergraph for the window
        // and instantiate such a decoder
        let committing_eids: HashSet<_> = {
            let error_models = self.error_models.read().await;
            mapping
                .global_eid_of
                .iter()
                .enumerate()
                .filter_map(|(local_eid, eid)| {
                    committing_cids.contains(&error_models[eid].instance.cid).then_some(local_eid)
                })
                .collect()
        };
        let (decoding_hypergraph, errors, logical_flips) = self
            .decoding_hypergraph(committing_cids, logical_targets, window_gids, relative_program, mapping)
            .await;

        let Some(cache_key) = cache_key else {
            let probability_reweights = self.shot_probability_reweights(mapping, &errors).await;
            // Without a persistent decoder every shot ships its whole problem, so
            // probability updates are applied before loss derives its live priors.
            // Loss edges whose checks fall outside the window are absent from the
            // hypergraph and silently skipped -- a future window covering them
            // handles them there.
            let mut decoding_hypergraph = decoding_hypergraph;
            apply_reweights(&mut decoding_hypergraph, probability_reweights.iter().copied());
            let (mut decoding_hypergraph, loss) = self.loss_handler.apply_sites(decoding_hypergraph, &loss_sites, &errors);
            let mut errors = errors;
            let mut logical_flips = logical_flips;
            if deduplicate {
                let prepared = deduplicate_decoder_input(&decoding_hypergraph, &errors, &logical_flips, |error| {
                    usize::from(committing_eids.contains(&error.eid))
                });
                decoding_hypergraph = prepared.hypergraph;
                errors = prepared.representatives;
                logical_flips = prepared.logical_flips.as_ref().clone();
            }
            let mut syndrome = syndrome;
            ignore_edge_isolated_history_vertices(&decoding_hypergraph, &mut syndrome);
            let hard_hypergraph = hard_decoding_hypergraph(decoding_hypergraph.clone(), &logical_flips);
            span.add_event(Event::new("decoding").with_property(|| ("type", "temporary")));
            let parity_factor = self
                .decoder
                .decode(blackbox_decoder::DecodingProblem {
                    hypergraph: Some(hard_hypergraph),
                    syndrome: Some(syndrome.clone()),
                    loss,
                })
                .await
                .unwrap();
            if self.config.assert_parity_factor {
                assert_parity_factor(&decoding_hypergraph, &parity_factor, &syndrome);
            }
            let weights = correction_weights(&decoding_hypergraph, &parity_factor);
            let forced_gap_problem = (target_count != 0).then(|| {
                let retained_edges: Vec<_> = errors.iter().map(|error| committing_eids.contains(&error.eid)).collect();
                CommitRegionDecoder::new(&decoding_hypergraph, &logical_flips, &retained_edges, target_count, false).problem(
                    self.gap_decoder().clone(),
                    &decoding_hypergraph,
                    syndrome,
                    &parity_factor,
                    vec![],
                    false,
                )
            });
            return (parity_factor, errors.into(), weights, forced_gap_problem);
        };

        // Load the stable base graph before any shot's loss is applied. Keep
        // vertex numbering stable and ignore edge-isolated history-boundary bits.
        span.add_event(Event::new("decoding").with_property(|| ("type", "loading")));
        let retain_decoding_hypergraph =
            self.config.forced_gap || !self.use_loaded_reweights || self.config.assert_parity_factor;
        let (projection, prepared) = prepare_decoder(decoding_hypergraph, errors, logical_flips, deduplicate, |error| {
            usize::from(committing_eids.contains(&error.eid))
        });
        let scoring = (target_count != 0).then(|| {
            let retained_edges: Vec<_> = prepared
                .representatives
                .iter()
                .map(|error| committing_eids.contains(&error.eid))
                .collect();
            Arc::new(CommitRegionDecoder::new(
                &prepared.hypergraph,
                &prepared.logical_flips,
                &retained_edges,
                target_count,
                true,
            ))
        });
        let decoder = load_projected_decoder(&self.decoder, projection, prepared, retain_decoding_hypergraph, true)
            .await
            .unwrap();
        let loaded = DecoderCacheEntry { decoder, scoring };
        let probability_reweights = self
            .projected_shot_probability_reweights(mapping, &loaded.decoder.projection)
            .await;
        let decode_syndrome = loaded.decoder.project_syndrome(syndrome);
        let projected = self
            .loss_handler
            .project_shot(&loaded.decoder.projection, &probability_reweights, &loss_sites);
        let mut loaded_decoders = self.loaded_decoders.write().await;
        loaded_decoders.insert(cache_key, loaded.clone());
        drop(loaded_decoders);
        let parity_factor = decode_projected(
            &self.decoder,
            &loaded.decoder,
            decode_syndrome.clone(),
            projected.reweights.clone(),
            projected.loss,
            self.use_loaded_reweights,
        )
        .await
        .unwrap();
        if self.config.assert_parity_factor {
            assert_parity_factor(
                loaded.decoder.decoding_hypergraph.as_ref().unwrap(),
                &parity_factor,
                &decode_syndrome,
            );
        }
        let weights = loaded.decoder.correction_weights(&parity_factor, &projected.reweights);
        let forced_gap_problem = loaded.scoring.as_ref().map(|scoring| {
            scoring.problem(
                self.gap_decoder().clone(),
                loaded.decoder.decoding_hypergraph.as_ref().unwrap(),
                decode_syndrome,
                &parity_factor,
                projected.reweights,
                self.gap_use_loaded_reweights,
            )
        });
        (parity_factor, projected.errors, weights, forced_gap_problem)
    }

    async fn bind_probability_modifiers(
        &self,
        gid: u64,
        modifiers: &[bin::ProbabilityModifier],
    ) -> Result<Vec<(u64, bin::ProbabilityModifier)>, Status> {
        if modifiers.is_empty() {
            return Ok(vec![]);
        }
        let cid = self
            .gadgets
            .read()
            .await
            .get(&gid)
            .and_then(|gadget| gadget.binding_cid)
            .ok_or_else(|| Status::failed_precondition(format!("gid={gid} has no binding check model")))?;
        let eids = self
            .check_models
            .read()
            .await
            .get(&cid)
            .map(|check_model| check_model.attaching_eid_vec.clone())
            .ok_or_else(|| Status::failed_precondition(format!("cid={cid} is not loaded")))?;
        if modifiers.len() > eids.len() {
            return Err(Status::invalid_argument(format!(
                "gid={gid} supplied {} probability modifiers for {} attached error models",
                modifiers.len(),
                eids.len()
            )));
        }
        let error_model_types = self.error_model_types.read().await;
        let error_models = self.error_models.read().await;
        let mut bound = Vec::with_capacity(modifiers.len());
        for (&eid, modifier) in eids.iter().zip(modifiers) {
            let error_model = error_models
                .get(&eid)
                .ok_or_else(|| Status::failed_precondition(format!("eid={eid} is not loaded")))?;
            let error_model_type = error_model_types
                .get(&error_model.instance.etype)
                .ok_or_else(|| Status::failed_precondition(format!("etype={} is not loaded", error_model.instance.etype)))?;
            validate_probability_modifier(modifier, error_model_type.errors.len()).map_err(Status::invalid_argument)?;
            bound.push((eid, modifier.clone()));
        }
        Ok(bound)
    }

    async fn shot_probability_reweights(
        &self,
        mapping: &RelativeMapping,
        error_reference: &[ErrorIndex],
    ) -> Vec<(u64, f64)> {
        let gadgets = self.gadgets.read().await;
        probability_reweights(error_reference, Self::shot_probability_modifiers(mapping, &gadgets))
    }

    async fn projected_shot_probability_reweights(
        &self,
        mapping: &RelativeMapping,
        projection: &crate::coordinator::reweight_handler::DecodeProjection,
    ) -> Vec<(u64, f64)> {
        let gadgets = self.gadgets.read().await;
        projection.probability_reweights(Self::shot_probability_modifiers(mapping, &gadgets))
    }

    fn shot_probability_modifiers<'a>(
        mapping: &RelativeMapping,
        gadgets: &'a HashMap<u64, Gadget>,
    ) -> Vec<(usize, &'a bin::ProbabilityModifier)> {
        let mut modifiers: Vec<_> = mapping
            .global_gid_of
            .iter()
            .filter_map(|gid| gadgets.get(gid))
            .flat_map(|gadget| gadget.probability_modifiers.iter())
            .filter_map(|(eid, modifier)| mapping.local_eid_of.get(eid).map(|&local_eid| (local_eid, modifier)))
            .collect();
        modifiers.sort_unstable_by_key(|&(local_eid, _)| local_eid);
        modifiers
    }

    async fn decoding_hypergraph(
        &self,
        committing_cids: &HashSet<u64>,
        logical_targets: &[CorrectionBasis],
        window_gids: &HashSet<u64>,
        relative_program: &RelativeProgram,
        mapping: &RelativeMapping,
    ) -> (DecodingHypergraph, Arc<Vec<ErrorIndex>>, Vec<Vec<u64>>) {
        #[cfg(feature = "cli")]
        log::info!("constructing decoding hypergraph for cids={committing_cids:?}");
        let error_model_types = self.error_model_types.read().await;
        let error_models = self.error_models.read().await;
        let logical_flip_cache = if self.config.forced_gap && !logical_targets.is_empty() {
            let state = self.forced_gap_state.as_ref().unwrap().read().await;
            Some(
                state
                    .symbolic
                    .logical_flip_cache(window_gids.iter().copied(), logical_targets),
            )
        } else {
            None
        };

        let mut hyperedges: Vec<Hyperedge> = vec![];
        let mut error_reference: Vec<ErrorIndex> = vec![];
        let mut logical_flips = vec![];

        // Iterate over all gadgets' error models. Handles three configurations:
        //   - Normal gadgets (has check_model): local checks + remote checks
        //   - Error-only gadgets (no check_model): only remote checks (local projected out)
        //   - Check-only gadgets (committed, no error_models): skipped (empty iter)
        for (local_gid, expanded_gadget) in relative_program.local_gadgets.iter().enumerate() {
            let has_local_check = expanded_gadget.check_model.is_some();
            let local_start_index = if let Some(ref cm) = expanded_gadget.check_model {
                mapping.start_indices[cm.cid as usize] as u64
            } else {
                0 // unused for error-only gadgets
            };
            let is_in_commit_region = has_local_check
                && expanded_gadget
                    .check_model
                    .as_ref()
                    .is_some_and(|cm| committing_cids.contains(&mapping.global_cid_of[cm.cid as usize]));

            let local_eid_bias = mapping.local_eid_bias[local_gid];
            for (em_index, expanded_em) in expanded_gadget.error_models.iter().enumerate() {
                let local_eid = local_eid_bias + em_index;
                let eid = mapping.global_eid_of[local_eid];
                let error_model = error_models.get(&eid).unwrap();
                let error_model_type = error_model_types.get(&error_model.instance.etype).unwrap();
                let expanded_remotes = &expanded_em.remote_check_models;
                let mut errors = &error_model_type.errors;
                // only when there is modifier to the errors, copy the list of errors and modify
                let modified_errors: Option<Vec<bin::error_model_type::Error>>;
                if let Some(modifier) = &error_model.instance.modifier
                    && let Some(probability_modifier) = &modifier.probability_modifier
                {
                    let mut new_errors = errors.clone();
                    for (error_index, &probability) in probability_modifier.probabilities.iter().enumerate() {
                        new_errors[error_index].probability = probability;
                    }
                    for (&error_index, &probability) in probability_modifier
                        .sparse_indices
                        .iter()
                        .zip(probability_modifier.sparse_probabilities.iter())
                    {
                        new_errors[error_index as usize].probability = probability;
                    }
                    modified_errors = Some(new_errors);
                    errors = modified_errors.as_ref().unwrap();
                }
                for (error_index, error) in errors.iter().enumerate() {
                    // Zero-probability errors are kept, at their prior probability.
                    // They exist for a reason -- an atom loss activates its
                    // Pauli-envelope generators, and a caller may reweight any edge
                    // -- so the hyperedge set stays independent of what happened in
                    // a given shot, keeping edge indices stable across shots.
                    let mut vertices: Vec<u64> = vec![];
                    let mut has_external_check = false;
                    for check in &error.checks {
                        if let Some(ri) = check.remote_check_model {
                            if let Some(remote_local_cid) = expanded_remotes[ri as usize] {
                                let remote_start_index = mapping.start_indices[remote_local_cid as usize] as u64;
                                vertices.push(
                                    remote_start_index
                                        + check.check_index
                                        + error_model.modified_remote_check_models[ri as usize]
                                            .as_ref()
                                            .unwrap()
                                            .check_bias,
                                );
                            } else if is_in_commit_region {
                                // A commit-region error references a check outside the window.
                                // Drop the entire hyperedge: partial projection for committed
                                // errors would corrupt the syndrome seen by future windows.
                                has_external_check = true;
                                break;
                            }
                            // Buffer/outside error models may legitimately reference checks
                            // outside the window — silently omit the vertex (projection).
                        } else if has_local_check {
                            // Local check on a normal window gadget
                            vertices.push(local_start_index + check.check_index);
                        }
                        // Error-only gadgets: local checks belong to outside check model,
                        // project them out.
                    }
                    if has_external_check {
                        continue;
                    }
                    let logical_readout_flips =
                        if is_in_commit_region && let Some(logical_flip_cache) = logical_flip_cache.as_ref() {
                            let gid = mapping.global_gid_of[local_gid];
                            logical_flip_cache.propagated_logical_flips(gid, &error.residual, &error.readout_flips)
                        } else {
                            vec![]
                        };
                    if vertices.is_empty() && logical_readout_flips.is_empty() {
                        continue;
                    }
                    error_reference.push(ErrorIndex {
                        eid: local_eid,
                        error_index,
                    });
                    hyperedges.push(Hyperedge {
                        vertices,
                        probability: error.probability,
                    });
                    logical_flips.push(logical_readout_flips);
                }
            }
        }
        let hypergraph = DecodingHypergraph {
            vertex_num: relative_program.count_checks as u64,
            hyperedges,
        };
        (hypergraph, Arc::new(error_reference), logical_flips)
    }

    /// Keep committed checks until the active error models that can affect them are decided.
    fn retain_committed_check_history(
        window: &HashSet<u64>,
        gadgets: &HashMap<u64, Gadget>,
        check_models: &HashMap<u64, CheckModel>,
        error_models: &HashMap<u64, ErrorModel>,
    ) -> HashSet<u64> {
        let mut retained = window.clone();
        for &owner_gid in window {
            let owner = &gadgets[&owner_gid];
            if matches!(*owner.state.borrow(), GadgetState::Committed) {
                continue;
            }
            let Some(cid) = owner.binding_cid else { continue };
            for eid in &check_models[&cid].attaching_eid_vec {
                let remotes = &error_models[eid].modified_remote_check_models;
                let mut resolved = vec![None; remotes.len()];
                for remote_index in 0..remotes.len() {
                    Self::resolve_remote_check_model_gid(&mut resolved, remote_index, remotes, owner_gid, gadgets);
                    let Some(remote) = &remotes[remote_index] else { continue };
                    let target_gid = if let Some(cid) = remote.absolute_cid {
                        check_models.get(&cid).map(|model| model.instance.gid)
                    } else {
                        resolved[remote_index]
                    };
                    if let Some(target_gid) = target_gid
                        && let Some(target) = gadgets.get(&target_gid)
                        && target.binding_cid.is_some()
                        && matches!(*target.state.borrow(), GadgetState::Committed)
                    {
                        retained.insert(target_gid);
                    }
                }
            }
        }
        retained
    }

    fn expand_remote_check_models_in_window(
        gid: u64,
        error_model: &ErrorModel,
        gadgets: &HashMap<u64, Gadget>,
        window: &HashSet<u64>,
    ) -> Vec<Option<u64>> {
        let remotes = &error_model.modified_remote_check_models;
        let mut resolved = vec![None; remotes.len()];
        for remote_index in 0..remotes.len() {
            Self::resolve_remote_check_model_gid(&mut resolved, remote_index, remotes, gid, gadgets);
        }
        remotes
            .iter()
            .zip(resolved)
            .map(|(remote, resolved_gid)| {
                let remote = remote.as_ref()?;
                remote.absolute_cid.or_else(|| {
                    resolved_gid
                        .filter(|gid| window.contains(gid))
                        .and_then(|gid| gadgets.get(&gid))
                        .and_then(|gadget| gadget.binding_cid)
                })
            })
            .collect()
    }

    /// Resolve a remote check model to its target gadget GID (synchronous).
    /// Does not filter by decoder window or wait for future connections.
    fn resolve_remote_check_model_gid(
        resolved: &mut Vec<Option<u64>>,
        ri: usize,
        remote_check_models: &Vec<Option<bin::error_model_type::RemoteCheckModel>>,
        owner_gid: u64,
        gadgets: &HashMap<u64, Gadget>,
    ) {
        if resolved[ri].is_some() || remote_check_models[ri].is_none() {
            return;
        }
        let rcm = remote_check_models[ri].as_ref().unwrap();
        if rcm.absolute_cid.is_some() {
            // absolute_cid handled separately; not a port-based reference
            resolved[ri] = Some(u64::MAX);
            return;
        }
        let previous = if let Some(prev_ri) = rcm.previous_remote_check_model {
            Self::resolve_remote_check_model_gid(resolved, prev_ri as usize, remote_check_models, owner_gid, gadgets);
            match resolved[prev_ri as usize] {
                Some(gid) if gid != u64::MAX => gid,
                _ => {
                    resolved[ri] = Some(u64::MAX);
                    return;
                }
            }
        } else {
            owner_gid
        };
        let Some(gadget) = gadgets.get(&previous) else {
            resolved[ri] = Some(u64::MAX);
            return;
        };
        match rcm.port.unwrap() {
            bin::error_model_type::remote_check_model::Port::Output(port) => {
                if let Some(conn) = gadget.outputs[port as usize].borrow().as_ref() {
                    resolved[ri] = Some(conn.gid);
                } else {
                    resolved[ri] = Some(u64::MAX); // not yet connected
                }
            }
            bin::error_model_type::remote_check_model::Port::Input(port) => {
                let connector = &gadget.instance.connectors[port as usize];
                resolved[ri] = Some(connector.gid);
            }
        }
    }

    /// Given a remote check model index `ri` that resolved to `u64::MAX`,
    /// walk the resolution chain to find the first unconnected output port
    /// that caused the failure.  Returns `Some((source_gid, port))` if
    /// the blocker is an unconnected output port, or `None` if the failure
    /// was caused by something non-retryable (absolute_cid, missing gadget,
    /// input port, etc.).
    fn find_blocking_port(
        ri: usize,
        resolved_gids: &[Option<u64>],
        remote_check_models: &[Option<bin::error_model_type::RemoteCheckModel>],
        owner_gid: u64,
        gadgets: &HashMap<u64, Gadget>,
    ) -> Option<(u64, u64)> {
        let rcm = remote_check_models[ri].as_ref()?;
        if rcm.absolute_cid.is_some() {
            return None;
        }
        let previous = if let Some(prev_ri) = rcm.previous_remote_check_model {
            if resolved_gids[prev_ri as usize] == Some(u64::MAX) {
                // The previous step is also unresolved — find the root blocker.
                return Self::find_blocking_port(prev_ri as usize, resolved_gids, remote_check_models, owner_gid, gadgets);
            }
            match resolved_gids[prev_ri as usize] {
                Some(gid) if gid != u64::MAX => gid,
                _ => return None,
            }
        } else {
            owner_gid
        };
        let gadget = gadgets.get(&previous)?;
        match rcm.port {
            Some(bin::error_model_type::remote_check_model::Port::Output(port)) => {
                if gadget.outputs[port as usize].borrow().is_none() {
                    Some((previous, port))
                } else {
                    None // port is connected; failure was caused by something else
                }
            }
            _ => None,
        }
    }

    /// expand the remote gadgets referred by the check model; note that this function will
    /// be waiting for the gadget if it has not been connected yet, thus it should be called
    /// in a separate async task without blocking the gRPC request.
    async fn expand_remote_gadgets(
        check_model: &bin::CheckModel,
        modified_remote_gadgets: &Vec<Option<bin::check_model_type::RemoteGadget>>,
        gadgets: &RwLock<HashMap<u64, Gadget>>,
        token: CancellationToken,
    ) -> Vec<Option<u64>> {
        // expand the remote gadgets
        let mut expanded_remote_gid_vec: Vec<Option<u64>> = vec![None; modified_remote_gadgets.len()];
        for ri in 0..modified_remote_gadgets.len() {
            Self::expand_remote_gadget(
                &mut expanded_remote_gid_vec,
                ri,
                modified_remote_gadgets,
                check_model.gid,
                gadgets,
                token.clone(),
            )
            .await;
        }
        expanded_remote_gid_vec
    }

    async fn expand_remote_gadget(
        expanded_remote_gid_vec: &mut Vec<Option<u64>>,
        ri: usize,
        remote_gadgets: &Vec<Option<bin::check_model_type::RemoteGadget>>,
        gid: u64,
        gadgets: &RwLock<HashMap<u64, Gadget>>,
        token: CancellationToken,
    ) {
        if expanded_remote_gid_vec[ri].is_some() || remote_gadgets[ri].is_none() {
            return; // already expanded or nothing to expand
        }
        let remote_gadget = remote_gadgets[ri].as_ref().unwrap();
        // if absolute_gid is provided, use it directly
        if let Some(absolute_gid) = remote_gadget.absolute_gid {
            expanded_remote_gid_vec[ri] = Some(absolute_gid);
            return;
        }
        // expand the dependent remote gadget first
        // (we do not check circular dependency here for simplicity, see ProgSpec)
        let previous = if let Some(previous) = remote_gadget.previous_remote_gadget {
            Box::pin(Self::expand_remote_gadget(
                expanded_remote_gid_vec,
                previous as usize,
                remote_gadgets,
                gid,
                gadgets,
                token.clone(),
            ))
            .await;
            expanded_remote_gid_vec[previous as usize].unwrap()
        } else {
            gid
        };
        let gadgets = gadgets.read().await;
        let gadget = gadgets.get(&previous).unwrap();
        match remote_gadget.port.unwrap() {
            bin::check_model_type::remote_gadget::Port::Output(port) => {
                let next = get_or_receiver(&gadget.outputs[port as usize], token);
                drop(gadgets); // release the read lock
                let next = match next {
                    Ok(next) => Some(next),
                    Err(handle) => handle.await.unwrap_or(None),
                };
                if let Some(next) = next {
                    expanded_remote_gid_vec[ri] = Some(next.gid);
                }
            }
            bin::check_model_type::remote_gadget::Port::Input(port) => {
                let connector = &gadget.instance.connectors[port as usize];
                expanded_remote_gid_vec[ri] = Some(connector.gid);
            }
        }
    }
}

#[tonic::async_trait]
impl coordinator::coordinator_server::Coordinator for WindowCoordinator {
    #[cfg_attr(feature = "cli", fastrace::trace)]
    async fn load_library(&self, request: Request<bin::Library>) -> Result<Response<()>, Status> {
        let _task_guard = self
            .task_counter
            .try_guard()
            .ok_or_else(|| Status::unavailable("coordinator reset in progress"))?;
        let library = request.into_inner();
        self.loss_handler
            .validate_capability(&library.gadget_types, self.decoder.features())?;
        let mut port_types = self.port_types.write().await;
        for port_type in library.port_types.into_iter() {
            if port_types.contains_key(&port_type.ptype) {
                return Err(Status::already_exists(format!("ptype={}", port_type.ptype)));
            }
            port_types.insert(port_type.ptype, Arc::new(port_type));
        }
        drop(port_types);
        let mut gadget_types = self.gadget_types.write().await;
        for gadget_type in library.gadget_types.into_iter() {
            if gadget_types.contains_key(&gadget_type.gtype) {
                return Err(Status::already_exists(format!("gtype={}", gadget_type.gtype)));
            }
            gadget_types.insert(gadget_type.gtype, Arc::new(gadget_type));
        }
        drop(gadget_types);
        let mut check_model_types = self.check_model_types.write().await;
        for check_model_type in library.check_model_types.into_iter() {
            if check_model_types.contains_key(&check_model_type.ctype) {
                return Err(Status::already_exists(format!("ctype={}", check_model_type.ctype)));
            }
            check_model_types.insert(check_model_type.ctype, Arc::new(check_model_type));
        }
        drop(check_model_types);
        let mut error_model_types = self.error_model_types.write().await;
        for error_model_type in library.error_model_types.into_iter() {
            if error_model_types.contains_key(&error_model_type.etype) {
                return Err(Status::already_exists(format!("etype={}", error_model_type.etype)));
            }
            error_model_types.insert(error_model_type.etype, Arc::new(error_model_type));
        }
        drop(error_model_types);
        Ok(().into())
    }

    async fn unload(&self, _unload: Request<coordinator::UnloadLibrary>) -> Result<Response<()>, Status> {
        unimplemented!()
    }

    #[cfg_attr(feature = "cli", fastrace::trace)]
    async fn execute(&self, request: Request<bin::Instruction>) -> Result<Response<coordinator::ExecuteResponse>, Status> {
        let _task_guard = self
            .task_counter
            .try_guard()
            .ok_or_else(|| Status::unavailable("coordinator reset in progress"))?;
        let instruction = request.into_inner();
        let create = instruction
            .create
            .ok_or_else(|| Status::invalid_argument("unknown instruction"))?;
        let id = match create {
            bin::instruction::Create::Gadget(gadget) => {
                let port_types = self.port_types.read().await;
                let gadget_types = self.gadget_types.read().await;
                let mut gadgets = self.gadgets.write().await;
                let gid = if gadget.gid == 0 {
                    // Auto-assign: find next unused gid
                    let mut next_gid = self.next_gid.lock().await;
                    while gadgets.contains_key(&*next_gid) {
                        *next_gid += 1;
                    }
                    let gid = *next_gid;
                    *next_gid += 1;
                    gid
                } else {
                    // User-provided gid
                    gadget.gid
                };
                let gadget_type = gadget_types
                    .get(&gadget.gtype)
                    .ok_or_else(|| Status::not_found(format!("gtype={}", gadget.gtype)))?;
                debug_assert!(gadget.connectors.len() == gadget_type.inputs.len());
                for (port, connector) in gadget.connectors.iter().enumerate() {
                    debug_assert!(gadgets.contains_key(&connector.gid));
                    debug_assert!({
                        let peer_outputs = &gadgets[&connector.gid].outputs;
                        (connector.port as usize) < peer_outputs.len()
                            && peer_outputs[connector.port as usize].borrow().is_none()
                    });
                    gadgets.get_mut(&connector.gid).unwrap().outputs[connector.port as usize]
                        .send_replace(Some(bin::gadget::Connector { gid, port: port as u64 }));
                }
                let is_free_hop = gadget_type.is_free_hop.unwrap_or(gadget_type.measurements.is_empty());
                let mut gadget = gadget;
                gadget.gid = gid;
                gadgets.insert(
                    gid,
                    Gadget {
                        instance: gadget.clone(),
                        outcomes: watch::channel(None).0,
                        probability_modifiers: vec![],
                        binding_cid: None,
                        // important: we should not use vec![;len] syntax because it will create clones
                        outputs: gadget_type.outputs.iter().map(|_| watch::channel(None).0).collect(),
                        pauli_frame: watch::channel(None).0,
                        correction_count: 0,
                        correction_weight: 0.0,
                        is_free_hop,
                        state: watch::channel(GadgetState::Uncommitted).0,
                        loss_mask: None,
                    },
                );
                // Drain pending referrals for newly connected output ports.
                // A port (source_gid, port) was just connected by the connector
                // loop above; re-resolve any deferred error-model references that
                // were blocked on that port.
                {
                    let has_pending = {
                        let pending_by_port = self.pending_referring_by_port.lock().await;
                        gadget
                            .connectors
                            .iter()
                            .any(|c| pending_by_port.contains_key(&(c.gid, c.port)))
                    };
                    if has_pending {
                        let mut check_models = self.check_models.write().await;
                        let mut pending_by_gid = self.pending_referring_by_gid.lock().await;
                        let mut pending_by_port = self.pending_referring_by_port.lock().await;
                        for connector in gadget.connectors.iter() {
                            if let Some(entries) = pending_by_port.remove(&(connector.gid, connector.port)) {
                                for referral in entries {
                                    let mut resolved_gids = vec![None; referral.modified_remote.len()];
                                    for resolved_ri in 0..referral.modified_remote.len() {
                                        Self::resolve_remote_check_model_gid(
                                            &mut resolved_gids,
                                            resolved_ri,
                                            &referral.modified_remote,
                                            referral.owner_gid,
                                            &gadgets,
                                        );
                                    }
                                    let target_gid = resolved_gids[referral.ri].unwrap_or(u64::MAX);
                                    if target_gid == referral.owner_gid {
                                        continue;
                                    }
                                    if target_gid == u64::MAX {
                                        if let Some(blocker) = Self::find_blocking_port(
                                            referral.ri,
                                            &resolved_gids,
                                            &referral.modified_remote,
                                            referral.owner_gid,
                                            &gadgets,
                                        ) {
                                            pending_by_port.entry(blocker).or_default().push(referral);
                                        }
                                    } else if let Some(target_gadget) = gadgets.get(&target_gid) {
                                        if let Some(target_cid) = target_gadget.binding_cid {
                                            if let Some(target_cm) = check_models.get_mut(&target_cid) {
                                                target_cm.referring_eids.push(referral.eid);
                                            }
                                        } else {
                                            pending_by_gid.entry(target_gid).or_default().push(referral.eid);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                let mut tracker = self.pauli_frame_tracker.lock().await;
                tracker.add_gadget(gid, gadget_type, gadget.modifier.as_ref(), &port_types, &gadget.connectors);
                if let Some(state) = &self.forced_gap_state {
                    state.write().await.symbolic.add_gadget(gid, &tracker.gadgets[&gid]);
                }
                self.record_event(trace::event::Event::ExecuteGadget(trace::ExecuteGadgetEvent {
                    gadget: Some(gadget),
                    num_outputs: gadget_type.outputs.len() as u32,
                }))
                .await;
                gid
            }
            bin::instruction::Create::CheckModel(check_model) => {
                let check_model_types = self.check_model_types.read().await;
                let mut gadgets = self.gadgets.write().await;
                let mut check_models = self.check_models.write().await;
                let check_model_type = check_model_types
                    .get(&check_model.ctype)
                    .ok_or_else(|| Status::not_found(format!("ctype={}", check_model.ctype)))?;
                let modified_remote = Arc::new(
                    apply_check_model_reroutes(&check_model_type.remote_gadgets, check_model.modifier.as_ref())
                        .map_err(Status::invalid_argument)?,
                );
                let cid = if check_model.cid == 0 {
                    // Auto-assign: find next unused cid
                    let mut next_cid = self.next_cid.lock().await;
                    while check_models.contains_key(&*next_cid) {
                        *next_cid += 1;
                    }
                    let cid = *next_cid;
                    *next_cid += 1;
                    cid
                } else {
                    // User-provided cid
                    check_model.cid
                };
                let gadget = gadgets.get_mut(&check_model.gid).ok_or_else(|| {
                    Status::invalid_argument(format!("cid={cid} binding to unknown gid={}", check_model.gid))
                })?;
                debug_assert!(check_model_type.gtype == WILDCARD || check_model_type.gtype == gadget.instance.gtype);
                debug_assert!(gadget.binding_cid.is_none());
                gadget.binding_cid.replace(cid);
                // Drain any deferred referring_eids that were waiting for this
                // gadget to get a check model binding.
                let deferred_referring_eids = {
                    let mut pending_by_gid = self.pending_referring_by_gid.lock().await;
                    pending_by_gid.remove(&check_model.gid).unwrap_or_default()
                };
                let mut check_model = check_model;
                check_model.cid = cid;
                check_models.insert(
                    cid,
                    CheckModel {
                        instance: check_model.clone(),
                        attaching_eid_vec: vec![],
                        modified_remote_gadgets: modified_remote.clone(),
                        expanded_remote_gadgets: None,
                        syndrome: watch::channel(None).0,
                        syndrome_count: 0,
                        referring_eids: deferred_referring_eids,
                    },
                );
                self.record_event(trace::event::Event::ExecuteCheckModel(trace::ExecuteCheckModelEvent {
                    check_model: Some(check_model.clone()),
                }))
                .await;
                // expanding the remote gadgets may not be immediately possible if the gadgets
                // are not instantiated yet, so we spawn an async task to do it.
                let gadgets = self.gadgets.clone();
                let check_models = self.check_models.clone();
                let check_model_types = self.check_model_types.clone();
                let num_checks = check_model_type.checks.len();
                let token = self.cancellation.read().await.clone();
                let _guard = self.task_counter.guard();
                let check_model_gid = check_model.gid;
                let trace_shot = self.trace_shot.clone();
                let has_trace = self.config.trace_filepath.is_some();
                tokio::spawn(async move {
                    let _guard = _guard;
                    let expanded_remote_gadgets =
                        Self::expand_remote_gadgets(&check_model, &modified_remote, gadgets.as_ref(), token.clone()).await;
                    // wait for the related gadgets' outcomes to be ready
                    let mut handles: Vec<JoinHandle<bool>> = vec![];
                    {
                        let gadgets = gadgets.read().await;
                        for gid in [check_model.gid]
                            .into_iter()
                            .chain(expanded_remote_gadgets.iter().filter_map(|x| *x))
                        {
                            let gadget = gadgets.get(&gid).unwrap();
                            match check_or_receiver(&gadget.outcomes, token.clone()) {
                                Ok(_) => {}
                                Err(handle) => {
                                    handles.push(handle);
                                }
                            }
                        }
                    }
                    futures_util::future::join_all(handles).await;
                    // If the cancellation token fired while we were awaiting
                    // the outcomes, bail out before reading them: the wait
                    // resolves immediately on cancellation but the outcomes
                    // may still be `None`, which would panic the unwraps below.
                    if token.is_cancelled() {
                        return;
                    }
                    // then calculate the check values based on the expanded remote gadgets
                    let mut syndrome = bit_vector::from_sparse_indices(num_checks as u64, &[]);
                    let check_model_types = check_model_types.read().await;
                    let check_model_type = check_model_types.get(&check_model.ctype).unwrap();
                    let gadgets = gadgets.read().await;
                    let gadget = gadgets.get(&check_model.gid).unwrap();
                    let local_outcomes = gadget.outcomes.borrow().clone().unwrap();
                    let mut syndrome_count = 0;
                    // calculate the syndrome bits
                    for (check_index, check) in check_model_type.checks.iter().enumerate() {
                        let mut is_defect = check.naturally_flipped;
                        for measurement in &check.measurements {
                            if let Some(ri) = measurement.remote_gadget {
                                let remote_gid = expanded_remote_gadgets[ri as usize].unwrap();
                                let remote_gadget = gadgets.get(&remote_gid).unwrap();
                                is_defect ^= get_bit(
                                    remote_gadget.outcomes.borrow().as_ref().unwrap(),
                                    measurement.measurement_index
                                        + modified_remote[ri as usize].as_ref().unwrap().measurement_bias,
                                );
                            } else {
                                is_defect ^= get_bit(&local_outcomes, measurement.measurement_index);
                            }
                        }
                        set_bit(&mut syndrome, check_index as u64, is_defect);
                        syndrome_count += u64::from(is_defect);
                    }
                    drop(gadgets);
                    drop(check_model_types);
                    // save the result into the check model object
                    let mut check_models = check_models.write().await;
                    let check_model = check_models.get_mut(&cid).unwrap();
                    check_model.expanded_remote_gadgets = Some(expanded_remote_gadgets);
                    check_model.syndrome_count = syndrome_count;
                    check_model.syndrome.send_replace(Some(syndrome));
                    drop(check_models);
                    // Record syndrome-ready trace event
                    if has_trace {
                        trace_shot.lock().await.events.push(trace::Event {
                            timestamp_ns: crate::misc::util::timestamp_ns(),
                            event: Some(trace::event::Event::SyndromeReady(trace::SyndromeReadyEvent {
                                cid,
                                gid: check_model_gid,
                            })),
                        });
                    }
                });
                cid
            }
            bin::instruction::Create::ErrorModel(error_model) => {
                let error_model_types = self.error_model_types.read().await;

                // Pre-compute remote check model reroutes
                let error_model_type = error_model_types
                    .get(&error_model.etype)
                    .ok_or_else(|| Status::not_found(format!("etype={}", error_model.etype)))?;
                if let Some(probability_modifier) = error_model
                    .modifier
                    .as_ref()
                    .and_then(|modifier| modifier.probability_modifier.as_ref())
                {
                    validate_probability_modifier(probability_modifier, error_model_type.errors.len())
                        .map_err(Status::invalid_argument)?;
                }
                let modified_remote = Arc::new(
                    apply_error_model_reroutes(&error_model_type.remote_check_models, error_model.modifier.as_ref())
                        .map_err(Status::invalid_argument)?,
                );

                // Acquire locks in ordering: gadgets(read) → check_models(write) →
                // error_models(write).  All three are held throughout to ensure
                // atomicity between resolution and pending registration — prevents
                // TOCTOU races with gadget/check-model creation that could connect
                // the port or set binding_cid between resolution and registration.
                let gadgets = self.gadgets.read().await;
                let mut check_models = self.check_models.write().await;
                let mut error_models = self.error_models.write().await;

                let owner_gid = check_models.get(&error_model.cid).map(|c| c.instance.gid).unwrap_or(0);

                // Resolve remote check models to target GIDs
                let mut resolved_gids: Vec<Option<u64>> = vec![None; modified_remote.len()];
                for ri in 0..modified_remote.len() {
                    Self::resolve_remote_check_model_gid(&mut resolved_gids, ri, &modified_remote, owner_gid, &gadgets);
                }

                // Assign eid
                let eid = if error_model.eid == 0 {
                    let mut next_eid = self.next_eid.lock().await;
                    while error_models.contains_key(&*next_eid) {
                        *next_eid += 1;
                    }
                    let eid = *next_eid;
                    *next_eid += 1;
                    eid
                } else {
                    error_model.eid
                };
                let check_model = check_models.get_mut(&error_model.cid).ok_or_else(|| {
                    Status::invalid_argument(format!("eid={eid} attaching to unknown cid={}", error_model.cid))
                })?;
                debug_assert!(error_model_type.ctype == WILDCARD || error_model_type.ctype == check_model.instance.ctype);
                check_model.attaching_eid_vec.push(eid);

                // Register referring_eids for resolved targets; defer unresolved ones.
                {
                    let mut pending_by_gid = self.pending_referring_by_gid.lock().await;
                    let mut pending_by_port = self.pending_referring_by_port.lock().await;
                    for (ri, target_gid) in resolved_gids.iter().enumerate() {
                        let Some(&target_gid) = target_gid.as_ref() else { continue };
                        if target_gid == owner_gid || modified_remote[ri].is_none() {
                            continue;
                        }
                        if target_gid == u64::MAX {
                            // Output port not connected — defer until the port is connected.
                            if let Some(blocker) =
                                Self::find_blocking_port(ri, &resolved_gids, &modified_remote, owner_gid, &gadgets)
                            {
                                pending_by_port.entry(blocker).or_default().push(PendingPortReferral {
                                    eid,
                                    ri,
                                    owner_gid,
                                    modified_remote: modified_remote.clone(),
                                });
                            }
                        } else if let Some(target_gadget) = gadgets.get(&target_gid) {
                            if let Some(target_cid) = target_gadget.binding_cid {
                                // Fully resolved — register immediately.
                                if let Some(target_cm) = check_models.get_mut(&target_cid) {
                                    target_cm.referring_eids.push(eid);
                                }
                            } else {
                                // Target gadget exists but has no check model yet — defer.
                                pending_by_gid.entry(target_gid).or_default().push(eid);
                            }
                        }
                    }
                }

                let mut error_model = error_model;
                error_model.eid = eid;
                error_models.insert(
                    eid,
                    ErrorModel {
                        instance: error_model.clone(),
                        modified_remote_check_models: modified_remote.clone(),
                    },
                );
                self.record_event(trace::event::Event::ExecuteErrorModel(trace::ExecuteErrorModelEvent {
                    error_model: Some(error_model.clone()),
                }))
                .await;
                eid
            }
        };
        Ok((coordinator::ExecuteResponse { id }).into())
    }

    #[cfg_attr(feature = "cli", fastrace::trace)]
    async fn decode(&self, request: Request<coordinator::Outcomes>) -> Result<Response<coordinator::Readouts>, Status> {
        let outcomes = request.into_inner();
        let _task_guard = self
            .task_counter
            .try_guard()
            .ok_or_else(|| Status::unavailable("coordinator reset in progress"))?;
        let gid = outcomes.gid;
        let probability_modifiers = self.bind_probability_modifiers(gid, &outcomes.modifiers).await?;

        // Load outcomes
        let is_free_hop;
        {
            let gadget_types = self.gadget_types.read().await;
            let mut gadgets = self.gadgets.write().await;
            let gadget = gadgets
                .get_mut(&gid)
                .ok_or_else(|| Status::not_found(format!("gid={}", gid)))?;
            is_free_hop = gadget.is_free_hop;
            let mut outcome_data = outcomes
                .outcomes
                .ok_or_else(|| Status::invalid_argument("missing outcomes"))?;
            let gadget_type = gadget_types
                .get(&gadget.instance.gtype)
                .ok_or_else(|| Status::failed_precondition(format!("gtype={} is not loaded", gadget.instance.gtype)))?;
            validate_outcomes(
                &outcome_data,
                outcomes.loss_mask.as_ref(),
                u64::try_from(gadget_type.measurements.len()).unwrap(),
            )
            .map_err(Status::invalid_argument)?;
            // Apply loss-random-imputation before storing the outcomes so
            // every downstream consumer (syndrome calc, pauli-frame tracker,
            // window decoder) reads a single consistent imputed value per
            // measurement bit.
            if let Some(seed) = self.loss_imputation_seed {
                apply_loss_random_imputation(&mut outcome_data, outcomes.loss_mask.as_ref(), seed, gid);
            }
            // Record the observed atom losses for envelope-matching reweighting:
            // the local measurement indices flagged as losses. Only kept when the
            // strategy is enabled and this gadget type carries a loss model.
            if self.loss_handler.tracks_losses()
                && has_loss_model(&gadget_types, gadget.instance.gtype)
                && let Some(loss_mask) = outcomes.loss_mask.as_ref()
                && (0..loss_mask.size).any(|index| get_bit(loss_mask, index))
            {
                gadget.loss_mask = Some(loss_mask.clone());
            }
            gadget.outcomes.send_replace(Some(outcome_data));
            gadget.probability_modifiers = probability_modifiers;
            let mut readouts = Vec::with_capacity(gadget_type.readouts.len());
            let data: BitVector = gadget.outcomes.borrow().as_ref().unwrap().clone();
            for readout in gadget_type.readouts.iter() {
                let mut value = false;
                for &mi in readout.measurement_indices.iter() {
                    value ^= get_bit(&data, mi);
                }
                readouts.push(value);
            }
            self.pauli_frame_tracker.lock().await.load_raw(gid, &readouts, &data);
        }

        if is_free_hop && self.config.buffer_radius > 0 {
            // free-hop gadget: wait for pauli_frame set by commit region leader.
            // When buffer_radius=0 (isolation mode), free-hops fall through to
            // the standard 5-step path and self-commit as their own center.
            self.record_event(trace::event::Event::Decode(trace::DecodeEvent {
                gid,
                is_leader: false,
                ..Default::default()
            }))
            .await;
            return self.wait_for_pauli_frame(gid).await;
        }

        // Hop-counted gadget: five-step window exploration then commit loop.

        // Step 1: Explore mandatory zone (blocking BFS up to buffer_radius).
        let mut explored = self
            .explore_mandatory_zone(gid)
            .await
            .ok_or_else(|| Status::cancelled("decode cancelled by reset"))?;

        // Step 2: Wait for mandatory-zone syndrome to be ready.
        // While waiting, more gadgets may arrive, improving step 3's reach.
        self.await_mandatory_zone_syndrome(&explored)
            .await
            .ok_or_else(|| Status::cancelled("decode cancelled by reset"))?;

        // Step 3: Explore lookahead zone (non-blocking BFS, lookahead_radius more hops).
        self.explore_lookahead_zone(&mut explored)
            .await
            .ok_or_else(|| Status::cancelled("decode cancelled by reset"))?;

        // Commit loop: check window for Decoding gadgets, run steps 3+4,
        // mark entire window as Decoding, then proceed.
        loop {
            let token = self.cancellation.read().await.clone();
            if token.is_cancelled() {
                return Err(Status::cancelled("decode cancelled by reset"));
            }

            let blocking_gids: Vec<u64>;
            {
                let mut gadgets = self.gadgets.write().await;

                // Check if this gadget has been claimed by another task
                let my_state = gadgets
                    .get(&gid)
                    .map(|g| g.state.borrow().clone())
                    .unwrap_or(GadgetState::Uncommitted);
                match my_state {
                    GadgetState::Decoding { leader_gid: other } if other != gid => {
                        // Claimed by another leader (commit region or buffer).
                        // Wait for state to resolve, then re-check.
                        let gadget = gadgets.get(&gid).ok_or_else(|| Status::not_found(format!("gid={}", gid)))?;
                        let mut rx = gadget.state.subscribe();
                        let token_c = token.clone();
                        drop(gadgets);
                        let resolved = tokio::select! {
                            result = rx.wait_for(|s| !matches!(s, GadgetState::Decoding { .. })) => {
                                result.ok().map(|r| r.clone())
                            }
                            _ = token_c.cancelled() => None
                        };
                        match resolved {
                            Some(GadgetState::Committed) => {
                                // Committed by the other leader. Emit non-leader event, wait for pauli_frame.
                                self.record_event(trace::event::Event::Decode(trace::DecodeEvent {
                                    gid,
                                    is_leader: false,
                                    leader_gid: other,
                                    ..Default::default()
                                }))
                                .await;
                                return self.wait_for_pauli_frame(gid).await;
                            }
                            Some(GadgetState::Uncommitted) => {
                                // Was in the other leader's buffer; released. Retry commit loop.
                                continue;
                            }
                            _ => {
                                // Cancelled or unexpected. Retry (cancellation checked at top of loop).
                                continue;
                            }
                        }
                    }
                    GadgetState::Committed => {
                        // Already committed by another leader. Emit non-leader event, wait for pauli_frame.
                        drop(gadgets);
                        self.record_event(trace::event::Event::Decode(trace::DecodeEvent {
                            gid,
                            is_leader: false,
                            ..Default::default()
                        }))
                        .await;
                        return self.wait_for_pauli_frame(gid).await;
                    }
                    _ => {}
                }

                // Check if any gadget in the window is Decoding
                let mut blocked: Vec<u64> = Vec::new();
                for &wgid in &explored.gadgets {
                    if let Some(g) = gadgets.get(&wgid)
                        && matches!(*g.state.borrow(), GadgetState::Decoding { .. })
                    {
                        blocked.push(wgid);
                    }
                }

                if blocked.is_empty() {
                    // Step 4: Select commit region.
                    self.select_commit_region(&mut explored, &gadgets);

                    // Step 5: Shrink window to minimal decoder window.
                    self.shrink_window(&mut explored, &gadgets);

                    // Emit WindowExploreEvent trace.
                    let mandatory_zone_gids: Vec<u64> = explored
                        .gadgets
                        .iter()
                        .filter(|g| explored.phase.get(*g) == Some(&ExplorePhase::MandatoryZone))
                        .copied()
                        .collect();
                    let lookahead_zone_gids: Vec<u64> = explored
                        .gadgets
                        .iter()
                        .filter(|g| explored.phase.get(*g) == Some(&ExplorePhase::LookaheadZone))
                        .copied()
                        .collect();
                    self.record_event(trace::event::Event::WindowExplore(trace::WindowExploreEvent {
                        center_gid: gid,
                        mandatory_zone_gids,
                        lookahead_zone_gids,
                        commit_region_gids: explored.commit_region.iter().copied().collect(),
                        decoder_window_gids: explored.decoder_window.iter().copied().collect(),
                    }))
                    .await;

                    // Mark commit region as Decoding(gid)
                    for &cgid in &explored.commit_region {
                        if let Some(g) = gadgets.get_mut(&cgid) {
                            g.state.send_replace(GadgetState::Decoding { leader_gid: gid });
                        }
                    }
                    // Mark buffer (decoder_window gadgets not in commit region)
                    for &wgid in &explored.decoder_window {
                        if explored.commit_region.contains(&wgid) {
                            continue;
                        }
                        if let Some(g) = gadgets.get_mut(&wgid)
                            && matches!(*g.state.borrow(), GadgetState::Uncommitted)
                        {
                            g.state.send_replace(GadgetState::Decoding { leader_gid: gid });
                        }
                    }
                    break;
                }

                blocking_gids = blocked;
            }

            // Wait for blocking gadgets to finish (become non-Decoding)
            let gadgets = self.gadgets.read().await;
            let mut watchers: Vec<JoinHandle<()>> = Vec::new();
            for &bgid in &blocking_gids {
                if let Some(g) = gadgets.get(&bgid) {
                    let mut rx = g.state.subscribe();
                    let token = token.clone();
                    watchers.push(tokio::spawn(async move {
                        tokio::select! {
                            result = rx.wait_for(|s| !matches!(s, GadgetState::Decoding { .. })) => {
                                result.map(|_| ()).unwrap_or(())
                            }
                            _ = token.cancelled() => {}
                        }
                    }));
                }
            }
            drop(gadgets);
            futures_util::future::join_all(watchers).await;
        }

        // Decode and commit (builds relative program, reads fresh syndromes, calls decoder,
        // applies corrections, marks Committed, releases buffer).
        self.decode_and_commit(
            gid,
            &explored.commit_region,
            &explored.committing_cids,
            &explored.decoder_window,
        )
        .await
        .ok_or_else(|| Status::cancelled("decode cancelled by reset"))?;

        // Wait for pauli_frame (may depend on predecessors committed by other tasks,
        // which is now possible since buffer has been released in update_pauli_frame).
        self.wait_for_pauli_frame(gid).await
    }

    #[cfg_attr(feature = "cli", fastrace::trace)]
    async fn reset(&self, request: Request<coordinator::ResetRequest>) -> Result<Response<()>, Status> {
        let flags = request.into_inner();
        let _pause = self
            .task_counter
            .try_pause()
            .ok_or_else(|| Status::unavailable("coordinator reset already in progress"))?;
        // Cancel all pending async tasks, wait for them to finish, then
        // install a fresh token so post-reset operations proceed normally.
        {
            let token = self.cancellation.read().await;
            token.cancel();
        }
        self.task_counter.wait_for_zero().await;
        {
            let mut token = self.cancellation.write().await;
            *token = CancellationToken::new();
        }
        if flags.reset_library {
            self.port_types.write().await.clear();
            self.gadget_types.write().await.clear();
            self.check_model_types.write().await.clear();
            self.error_model_types.write().await.clear();
        }
        self.gadgets.write().await.clear();
        self.check_models.write().await.clear();
        self.error_models.write().await.clear();
        self.pending_referring_by_gid.lock().await.clear();
        self.pending_referring_by_port.lock().await.clear();
        *self.next_gid.lock().await = 1;
        *self.next_cid.lock().await = 1;
        *self.next_eid.lock().await = 1;
        self.pauli_frame_tracker.lock().await.reset();
        if let Some(state) = &self.forced_gap_state {
            state.write().await.reset();
        }
        if flags.reset_library || flags.reset_decoder_service {
            self.loaded_decoders.write().await.clear();
        }
        // since decoders reset asynchronously, wait for all the decoders to finish
        self.decoder
            .reset(blackbox_decoder::ResetRequest {
                reset_hypergraphs: flags.reset_decoder_service,
                ..Default::default()
            })
            .await
            .map_err(|e| Status::internal(format!("reset decoder service error: {}", e)))?;
        if let Some(decoder) = &self.gap_decoder {
            decoder
                .reset(blackbox_decoder::ResetRequest {
                    reset_hypergraphs: flags.reset_decoder_service,
                    ..Default::default()
                })
                .await
                .map_err(|error| Status::internal(format!("reset gap decoder service error: {error}")))?;
        }
        // flush current shot into the trace and write to file if configured
        {
            let shot = std::mem::take(&mut *self.trace_shot.lock().await);
            let mut trace = self.trace.lock().await;
            trace.shots.push(shot);
            if let Some(filepath) = &self.config.trace_filepath {
                let buf = trace.encode_to_vec();
                std::fs::write(filepath, buf).map_err(|e| Status::internal(format!("failed to write trace: {e}")))?;
            }
        }
        Ok(().into())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/window_coordinator_test.rs"]
mod tests;
