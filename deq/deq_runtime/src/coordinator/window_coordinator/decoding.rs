//! Graph preparation, decoder reuse, and atomic correction commitment.

use super::collection::StatePins;
use super::scoring::{CausalGapProblem, original_scoring_baseline};
use super::{ErrorModel, ForcedGapStrategy, Gadget, GadgetState, WindowCoordinator, trace};
use crate::coordinator::decoder_cache_key::{ErrorModelFingerprint, etype_digest};
use crate::coordinator::loss_handler::{RawLossSite, has_loss_model};
use crate::coordinator::reweight_handler::{
    ProjectedErrors, apply_reweights, correction_weights, decode_projected, deduplicate_decoder_input,
    edge_isolated_vertices, hard_decoding_hypergraph, ignore_edge_isolated_history_vertices, load_projected_decoder,
    prepare_decoder, probability_reweights,
};
use crate::coordinator::{DecoderCacheKey, build_modifier_fingerprints};
use crate::decoder::SharedDecodingHypergraph;
use crate::decoder::blackbox_decoder::{self, DecodingHypergraph, Hyperedge, ParityFactor};
use crate::decoder::blackbox_util::assert_parity_factor;
use crate::jit::loss_compiler::{GadgetLoss, build_cross_gadget_loss_sites, build_cross_gadget_output_links};
use crate::misc::bit_vector::{self, flip_bit, get_bit, set_bit};
use crate::misc::fastrace::{Event, Span, SpanContext};
use crate::misc::index::ErrorIndex;
use crate::misc::pauli_frame_symbolic_propagator::CorrectionBasis;
use crate::misc::relative_program::{self, RelativeMapping, RelativeProgram};
use crate::misc::sync::{check_or_future, get_or_future};
use crate::util::BitVector;
use crate::{bin, coordinator};
use binar::{BitVec, BitwiseMut};
#[cfg(feature = "cli")]
use fastrace::future::FutureExt;
use hashbrown::{HashMap, HashSet};
use std::sync::Arc;
use tonic::{Response, Status};

struct WindowSpan {
    span: Span,
    #[cfg(feature = "cli")]
    started: std::time::Instant,
    #[cfg(feature = "cli")]
    timings: std::sync::Mutex<Vec<(&'static str, u128)>>,
}

impl WindowSpan {
    fn new(center_gid: u64, commit_region: &HashSet<u64>, window: &HashSet<u64>) -> Self {
        let span = Span::root("decode_window", SpanContext::random());
        #[cfg(feature = "cli")]
        let started = std::time::Instant::now();
        #[cfg(feature = "cli")]
        let span = span.with_properties(|| {
            [
                ("center_gid", center_gid.to_string()),
                ("commit_region", format!("{commit_region:?}")),
                ("window", format!("{window:?}")),
            ]
        });
        #[cfg(not(feature = "cli"))]
        let _ = (center_gid, commit_region, window);
        Self {
            span,
            #[cfg(feature = "cli")]
            started,
            #[cfg(feature = "cli")]
            timings: std::sync::Mutex::new(Vec::with_capacity(6)),
        }
    }

    fn record_timing(&self, name: &'static str) {
        #[cfg(feature = "cli")]
        {
            let offset = self.started.elapsed().as_nanos();
            self.timings.lock().unwrap().push((name, offset));
        }
        #[cfg(not(feature = "cli"))]
        let _ = name;
    }

    fn timed_event(&self, name: &'static str) {
        self.record_timing(name);
        self.span.add_event(Event::new(name));
    }
}

#[cfg(feature = "cli")]
impl Drop for WindowSpan {
    fn drop(&mut self) {
        let timings = std::mem::take(self.timings.get_mut().unwrap());
        let span = std::mem::replace(&mut self.span, Span::noop());
        // Submit boundaries in the root record, not separate cross-thread messages.
        drop(span.with_properties(|| {
            timings
                .into_iter()
                .map(|(name, offset)| (format!("{name}_offset_ns"), offset.to_string()))
        }));
    }
}

impl std::ops::Deref for WindowSpan {
    type Target = Span;

    fn deref(&self) -> &Span {
        &self.span
    }
}

pub(super) struct PreparedWindow {
    pub(super) hypergraph: Arc<SharedDecodingHypergraph>,
    pub(super) errors: Arc<Vec<ErrorIndex>>,
    pub(super) ignored_vertices: Vec<u64>,
    pub(super) loaded_hid: tokio::sync::OnceCell<u64>,
}

/// Compute the sorted vector of local CIDs in the commit region.
///
/// `committing_cids` is a set of global cids; we filter to those that map to
/// a local cid in this window (i.e., live in `mapping.local_cid_of`), then
/// sort for canonical hashing in [`DecoderCacheKey`].
pub(super) fn committing_local_cids_sorted(committing_cids: &HashSet<u64>, mapping: &RelativeMapping) -> Vec<u32> {
    let mut out: Vec<u32> = committing_cids
        .iter()
        .filter_map(|gcid| mapping.local_cid_of.get(gcid).map(|&lcid| lcid as u32))
        .collect();
    out.sort_unstable();
    out
}

pub(super) fn logical_flip_signature(logical_targets: &[CorrectionBasis], mapping: &RelativeMapping) -> Vec<u64> {
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

impl WindowCoordinator {
    pub(super) async fn finish_decode(&self, gid: u64, pins: StatePins) -> Result<Response<coordinator::Readouts>, Status> {
        let result = self.wait_for_pauli_frame(gid).await;
        drop(pins);
        if result.is_ok() {
            self.collect_after_readout().await?;
        }
        result
    }

    /// Wait for a gadget's `pauli_frame` to be set and return it as `Readouts`.
    pub(super) async fn wait_for_pauli_frame(&self, gid: u64) -> Result<Response<coordinator::Readouts>, Status> {
        let token = self.cancellation.read().await.clone();
        let gadgets = self.gadgets.read().await;
        let gadget = gadgets.get(&gid).ok_or_else(|| Status::not_found(format!("gid={gid}")))?;
        let pauli_frame = get_or_future(&gadget.pauli_frame, token.clone());
        drop(gadgets);
        let readouts = match pauli_frame {
            Ok(pf) => Some(pf),
            Err(waiting) => waiting.await,
        }
        .ok_or_else(|| Status::cancelled("decode cancelled"))?;
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
            self.wait_for_forced_gap_scores(gid, token.clone()).await?
        } else {
            vec![]
        };
        let frame_probabilities = if self.config.frame_probabilities {
            let scores = self.forced_gap_state.as_ref().unwrap().read().await.output_scores(gid);
            Self::resolve_forced_gap_scores(scores, token).await?
        } else {
            vec![]
        };
        self.pauli_frame_tracker.lock().await.finalize(gid);
        self.completed_readouts.lock().await.insert(gid);
        Ok((coordinator::Readouts {
            gid,
            readouts: Some(readouts),
            probabilities,
            syndrome_count,
            correction_count,
            correction_weight,
            frame_probabilities,
        })
        .into())
    }

    /// Runs the decode + commit phase for a single window.
    ///
    /// Builds the decoding problem from the window, calls the decoder, applies
    /// corrections, marks the commit region as committed, and clears owned
    /// reservations without changing buffer gadgets' commitment.
    ///
    /// Returns an error on cancellation, inconsistent internal state, or decoder failure.
    pub(super) async fn decode_and_commit(
        &self,
        center_gid: u64,
        commit_region: &HashSet<u64>,
        committing_cids: &HashSet<u64>,
        window: &HashSet<u64>,
    ) -> Result<(), Status> {
        let cancelled = || Status::cancelled("decode cancelled");
        let missing = |entity: &'static str, id: u64| {
            move || Status::internal(format!("window centered at gid={center_gid}: missing {entity} for id={id}"))
        };
        let span = WindowSpan::new(center_gid, commit_region, window);

        // first wait for all the outcomes to be loaded to make sure that their check
        // models and error models are completely loaded
        let token = self.cancellation.read().await.clone();
        let mut pins = StatePins::new(&self.pinned_gadgets);
        let mut handles = vec![];
        {
            let gadgets = self.gadgets.read().await;
            pins.retain(window.iter().chain(commit_region).copied());
            for &window_gid in window {
                let gadget = gadgets.get(&window_gid).ok_or_else(missing("gadget", window_gid))?;
                if let Err(handle) = check_or_future(&gadget.outcomes, token.clone()) {
                    handles.push(handle);
                }
            }
            // Also wait for outcomes of all commit_region gadgets
            for &cgid in commit_region {
                if !window.contains(&cgid)
                    && let Some(gadget) = gadgets.get(&cgid)
                    && let Err(handle) = check_or_future(&gadget.outcomes, token.clone())
                {
                    handles.push(handle);
                }
            }
        }
        futures_util::future::join_all(handles).await;
        if token.is_cancelled() {
            return Err(cancelled());
        }
        span.timed_event("outcomes_ready");

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
            let gadgets = self.gadgets.write().await;
            let mut tracker = self.pauli_frame_tracker.lock().await;
            for &cgid in commit_region {
                let pauli_frame_gadget = tracker.gadgets.get(&cgid).ok_or_else(missing("pauli-frame gadget", cgid))?;
                let residual = BitVec::zeros(pauli_frame_gadget.num_output_observables());
                let readout_flips = BitVec::zeros(pauli_frame_gadget.num_readouts());
                for (update_gid, pauli_frame) in tracker.load_correction(cgid, residual, readout_flips) {
                    let update_gadget = gadgets.get(&update_gid).ok_or_else(missing("gadget", update_gid))?;
                    debug_assert!(update_gadget.pauli_frame.borrow().is_none(), "bug");
                    update_gadget.pauli_frame.send_replace(Some(pauli_frame));
                }
            }
            // Fix commit-region corrections and release their reservations.
            for &commit_gid in commit_region {
                let gadget = gadgets.get(&commit_gid).ok_or_else(missing("gadget", commit_gid))?;
                gadget.state.send_replace(GadgetState {
                    committed: true,
                    reserved_by: None,
                });
            }
            // Release buffer reservations without undoing committed history.
            for &wgid in window {
                let Some(g) = gadgets.get(&wgid) else { continue };
                let released = g.state.borrow().release_buffer(center_gid);
                if let Some(state) = released {
                    g.state.send_replace(state);
                }
            }
            span.add_property(|| ("cid", "None"));
            drop(tracker);
            drop(gadgets);
            return Ok(());
        }

        // collect all CIDs in the window (for syndrome waiting)
        let window_cids = {
            let gadgets = self.gadgets.read().await;
            let mut window_cids: HashSet<u64> = HashSet::new();
            for &window_gid in window {
                let gadget = gadgets.get(&window_gid).ok_or_else(missing("gadget", window_gid))?;
                if let Some(cid) = gadget.binding_cid {
                    window_cids.insert(cid);
                }
            }
            window_cids
        };
        // wait for these check models to finish expanding their remote gadgets and calculate
        // their check values
        let mut handles = vec![];
        {
            let check_models = self.check_models.read().await;
            for &window_cid in &window_cids {
                let check_model = check_models.get(&window_cid).ok_or_else(missing("check model", window_cid))?;
                if let Err(handle) = check_or_future(&check_model.syndrome, token.clone()) {
                    handles.push(handle);
                }
            }
        }
        futures_util::future::join_all(handles).await;
        if token.is_cancelled() {
            return Err(cancelled());
        }
        span.add_property(|| ("window_cids", format!("{:?}", window_cids)));

        // for error models, we handle them differently from the check models: we don't need to
        // wait for all the remote check models to be expanded before starting to add edges;
        // when a hyperedge connects to some remote check models outside the window, we simply
        // connect the hyperedge to a virtual vertex.

        let terminal_gids = self.await_error_models(window).await?;
        span.timed_event("inputs_ready");
        // Bound graph construction only after every circuit dependency is ready.
        // Ready work includes graph preparation, but never input-dependent waits.
        let _decoding_slot = tokio::select! {
            permit = self.decoding_slots.acquire() => permit.map_err(|_| Status::unavailable("window decoding closed"))?,
            () = token.cancelled() => return Err(cancelled()),
        };
        span.timed_event("graph_slot_acquired");
        if let Some(pool) = self.decoder.thread_pool() {
            crate::misc::pool_executor::flush_local(pool);
        }
        let mut selected_error_models = HashMap::new();
        let mut expanded_gadgets: Vec<relative_program::ExpandedGadget> = vec![];
        {
            let check_model_types = self.check_model_types.read().await;
            let error_model_types = self.error_model_types.read().await;
            let gadgets = self.gadgets.read().await;
            let check_models = self.check_models.read().await;
            let error_models = self.error_models.read().await;
            let mut gid_vec: Vec<_> = window.iter().copied().collect();
            gid_vec.sort_unstable();

            // Build expanded gadgets for window gadgets.
            // Three configurations:
            //   - Normal (uncommitted with check model): both check_model and error_models
            //   - Check-only (committed with check model): check_model only, error_models = []
            //   - Free-hop (no check model): neither
            for &gid in gid_vec.iter() {
                let gadget = gadgets.get(&gid).ok_or_else(missing("gadget", gid))?;
                let inputs: Vec<_> = gadget.instance.connectors.iter().cloned().map(Some).collect();
                let outputs: Vec<_> = gadget.outputs.iter().map(|v| *v.borrow()).collect();
                let gtype = gadget.instance.gtype;
                let cid = gadget.binding_cid;
                let is_committed = gadget.state.borrow().committed;
                let (check_model, error_models) = if let Some(cid) = cid {
                    let check_model = check_models.get(&cid).ok_or_else(missing("check model", cid))?;
                    let remote_gadgets = check_model
                        .expanded_remote_gadgets
                        .clone()
                        .ok_or_else(missing("expanded remote gadgets for check model", cid))?;
                    let expanded_check_model = relative_program::ExpandedCheckModel {
                        cid,
                        ctype: check_model.instance.ctype,
                        remote_gadgets,
                        count_checks: check_model_types
                            .get(&check_model.instance.ctype)
                            .ok_or_else(missing("check model type", check_model.instance.ctype))?
                            .checks
                            .len(),
                    };
                    let expanded_error_models = if is_committed {
                        // Committed gadgets: error effects already applied, exclude
                        // error models from the decoder key and hypergraph.
                        vec![]
                    } else {
                        let mut ems = vec![];
                        if check_model.attaching_eid_vec.is_empty()
                            && terminal_gids.contains(&gid)
                            && let Some(terminal) = &check_model.instance.terminal_error_model
                        {
                            let terminal_type = &error_model_types[&terminal.etype];
                            let model = ErrorModel {
                                instance: terminal.clone(),
                                modified_remote_check_models: Arc::new(
                                    terminal_type.remote_check_models.iter().cloned().map(Some).collect(),
                                ),
                            };
                            let remote_check_models =
                                Self::expand_remote_check_models_in_window(gid, &model, &gadgets, window);
                            selected_error_models.insert(terminal.eid, model);
                            ems.push(relative_program::ExpandedErrorModel {
                                eid: terminal.eid,
                                etype: terminal.etype,
                                remote_check_models,
                            });
                        }
                        for &eid in check_model.attaching_eid_vec.iter() {
                            let error_model = error_models.get(&eid).ok_or_else(missing("error model", eid))?;
                            selected_error_models.insert(eid, error_model.clone());
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
            // Isolation mode must not explain local syndrome with corrections it cannot commit.
            if self.config.buffer_radius > 0 {
                for &gid in gid_vec.iter() {
                    let gadget = gadgets.get(&gid).ok_or_else(missing("gadget", gid))?;
                    let Some(cid) = gadget.binding_cid else { continue };
                    let check_model = check_models.get(&cid).ok_or_else(missing("check model", cid))?;
                    for &referring_eid in check_model.referring_eids.iter() {
                        if !processed_eids.insert(referring_eid) {
                            continue;
                        }
                        let error_model = error_models
                            .get(&referring_eid)
                            .ok_or_else(missing("error model", referring_eid))?;
                        let owner_cid = error_model.instance.cid;
                        let owner_cm = check_models.get(&owner_cid).ok_or_else(missing("check model", owner_cid))?;
                        let owner_gid = owner_cm.instance.gid;
                        if window.contains(&owner_gid) {
                            continue; // already in window as normal or check-only
                        }
                        let owner_gadget = gadgets.get(&owner_gid).ok_or_else(missing("gadget", owner_gid))?;
                        if owner_gadget.state.borrow().committed {
                            continue; // committed outside: errors already decoded
                        }
                        outside_gadget_eids.entry(owner_gid).or_default().push(referring_eid);
                    }
                }
            }
            let mut outside_gids: Vec<_> = outside_gadget_eids.keys().cloned().collect();
            outside_gids.sort();
            for &gid in outside_gids.iter() {
                let gadget = gadgets.get(&gid).ok_or_else(missing("gadget", gid))?;
                pins.retain([gid]);
                let inputs: Vec<_> = gadget.instance.connectors.iter().cloned().map(Some).collect();
                // Outside gadgets may have unconnected outputs; read safely.
                let outputs: Vec<_> = gadget.outputs.iter().map(|v| *v.borrow()).collect();
                let eids = &outside_gadget_eids[&gid];
                let mut expanded_error_models = vec![];
                for &eid in eids {
                    let error_model = error_models.get(&eid).ok_or_else(missing("error model", eid))?;
                    selected_error_models.insert(eid, error_model.clone());
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
                for (port, output) in gadgets[&gid].outputs.iter().enumerate() {
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
            .decode_parity_factor(
                committing_cids,
                &logical_targets,
                window,
                &relative_program,
                &mapping,
                &span,
                &selected_error_models,
            )
            .await?;
        let forced_gap_problem = forced_gap_problem.map(|problem| problem.map(Arc::new));
        if let Some(problem) = &forced_gap_problem {
            self.register_forced_gap_scores(&logical_targets, problem).await;
        }
        span.timed_event("decoded");

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
            &selected_error_models,
        )
        .await;
        span.timed_event("pauli_frame_updated");

        if self.config.forced_gap_strategy == ForcedGapStrategy::Eager
            && let Some(Ok(problem)) = forced_gap_problem
        {
            self.complete_eager_scores(&logical_targets, &problem).await;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn update_pauli_frame(
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
        error_models: &HashMap<u64, ErrorModel>,
    ) {
        let error_model_types = self.error_model_types.read().await;
        let mut gadgets = self.gadgets.write().await;
        let mut check_models = self.check_models.write().await;
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
            let local_gid = *mapping.local_gid_of.get(&error_gadget_gid).unwrap();
            let local_eid_bias = mapping.local_eid_bias[local_gid];
            let expanded_gadget = &relative_program.local_gadgets[local_gid];
            let expanded_error_model = &expanded_gadget.error_models[local_eid - local_eid_bias];
            assert!(mapping.global_eid_of[expanded_error_model.eid as usize] == eid);
            for check in error.checks.iter() {
                let (cid, check_index) = if let Some(ri) = check.remote_check_model {
                    let remote_local_cid = expanded_error_model.remote_check_models[ri as usize]
                        .expect("committed errors must have all checks inside the window");
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

        // Fix commit-region corrections and release their reservations.
        for &commit_gid in commit_region {
            let gadget = gadgets.get_mut(&commit_gid).unwrap();
            gadget.state.send_replace(GadgetState {
                committed: true,
                reserved_by: None,
            });
        }

        // Release buffer reservations without undoing committed history.
        for &wgid in window {
            let Some(g) = gadgets.get_mut(&wgid) else { continue };
            let released = g.state.borrow().release_buffer(center_gid);
            if let Some(state) = released {
                g.state.send_replace(state);
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
    /// losses, matching the monolithic coordinator's loss-site projection.
    ///
    /// `config.loss_strategy` is not `ignore` and some gadget currently
    /// loaded recorded observed losses. The generators are keyed by
    /// `(local_eid, error_index)`, matching the `error_reference` that
    /// [`decoding_hypergraph`](Self::decoding_hypergraph) produces, so
    /// The loss handler projects the sites onto this window's hyperedges.
    /// Loss sites are kept per-gadget and ungrouped; cross-gadget continuation
    /// whose herald signature is incomplete in this window is naturally dropped
    /// when its edges fall outside the window.
    async fn build_loss_sites(&self, relative_program: &RelativeProgram, mapping: &RelativeMapping) -> Vec<RawLossSite> {
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
                let local_gid = *mapping.local_gid_of.get(&gid)?;
                relative_program.local_gadgets[local_gid]
                    .error_models
                    .first()
                    .map(|model| model.eid as usize)
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

    pub(super) async fn prepared_window_key(
        &self,
        committing_cids: &HashSet<u64>,
        relative_program: &RelativeProgram,
        mapping: &RelativeMapping,
        error_models: &HashMap<u64, ErrorModel>,
    ) -> DecoderCacheKey {
        let types = self.error_model_types.read().await;
        let metadata = self.referenced_error_remotes.read().await;
        let fingerprints = mapping
            .global_eid_of
            .iter()
            .map(|eid| {
                let model = &error_models[eid];
                let model_type = &types[&model.instance.etype];
                let digest = metadata
                    .get(&model.instance.etype)
                    .and_then(|cached| cached.digest_for_type(model_type))
                    .unwrap_or_else(|| etype_digest(model_type));
                let normalized = model
                    .modified_remote_check_models
                    .iter()
                    .map(|remote| {
                        remote.as_ref().map(|remote| bin::error_model_type::RemoteCheckModel {
                            check_bias: remote.check_bias,
                            ..Default::default()
                        })
                    })
                    .collect();
                ErrorModelFingerprint::new(&model.instance, Arc::new(normalized), digest)
            })
            .collect();
        DecoderCacheKey {
            relative_program: relative_program.clone(),
            error_model_fingerprints: fingerprints,
            committing_local_cids: committing_local_cids_sorted(committing_cids, mapping),
            logical_flip_signature: vec![],
        }
    }

    pub(super) async fn prepare_window(
        &self,
        committing_cids: &HashSet<u64>,
        window_gids: &HashSet<u64>,
        relative_program: &RelativeProgram,
        mapping: &RelativeMapping,
        error_models: &HashMap<u64, ErrorModel>,
        deduplicate: bool,
    ) -> Arc<PreparedWindow> {
        let (mut hypergraph, mut errors, mut logical_flips) = self
            .decoding_hypergraph(committing_cids, &[], window_gids, relative_program, mapping, error_models)
            .await;
        let owned_error_models: HashSet<_> = mapping
            .global_eid_of
            .iter()
            .enumerate()
            .filter_map(|(local_eid, eid)| committing_cids.contains(&error_models[eid].instance.cid).then_some(local_eid))
            .collect();
        if deduplicate {
            let prepared = deduplicate_decoder_input(&hypergraph, &errors, &logical_flips, |error| {
                usize::from(owned_error_models.contains(&error.eid))
            });
            hypergraph = prepared.hypergraph;
            errors = prepared.representatives;
            prepared.logical_flips.as_ref().clone_into(&mut logical_flips);
        }
        let hypergraph = hard_decoding_hypergraph(hypergraph, &logical_flips);
        let ignored_vertices = edge_isolated_vertices(&hypergraph);
        let hypergraph = Arc::new(SharedDecodingHypergraph::new(hypergraph));
        Arc::new(PreparedWindow {
            hypergraph,
            errors,
            ignored_vertices,
            loaded_hid: tokio::sync::OnceCell::new(),
        })
    }

    fn cached_prepared_window(&self, key: &DecoderCacheKey) -> Option<Arc<PreparedWindow>> {
        if self.config.persistent_decoder {
            self.persistent_prepared_windows.lock().unwrap().get(key).cloned()
        } else {
            self.prepared_windows.lock().unwrap().get(key).cloned()
        }
    }

    async fn decode_temporary(
        &self,
        hypergraph: Arc<SharedDecodingHypergraph>,
        syndrome: BitVector,
        loss: Option<blackbox_decoder::LossInfo>,
        decoder_seed: Option<u64>,
        span: &WindowSpan,
        prepared_window: Option<&PreparedWindow>,
    ) -> Result<ParityFactor, Status> {
        span.record_timing("decoding");
        span.add_event(Event::new("decoding").with_property(|| {
            (
                "type",
                if self.config.persistent_decoder {
                    "persistent_prepared"
                } else {
                    "temporary"
                },
            )
        }));
        let decoding = async {
            if self.config.persistent_decoder
                && let Some(prepared) = prepared_window
            {
                let hid = prepared
                    .loaded_hid
                    .get_or_try_init(|| async {
                        self.decoder
                            .load_hypergraph(hypergraph.graph().clone())
                            .await
                            .map(|response| response.hid)
                    })
                    .await?;
                return self
                    .decoder
                    .decode_loaded(blackbox_decoder::LoadedDecodingProblem {
                        hid: *hid,
                        syndrome: Some(syndrome),
                        loss,
                        decoder_seed,
                        reweights: vec![],
                    })
                    .await;
            }
            self.decoder.decode_shared(hypergraph, syndrome, loss, decoder_seed).await
        };
        #[cfg(feature = "cli")]
        let decoding = decoding.in_span(
            Span::enter_with_parent("decoder_request", span)
                .with_property(|| ("pooled", self.decoder.thread_pool().is_some().to_string())),
        );
        decoding.await
    }

    #[allow(clippy::too_many_arguments)]
    async fn decode_parity_factor(
        &self,
        committing_cids: &HashSet<u64>,
        logical_targets: &[CorrectionBasis],
        window_gids: &HashSet<u64>,
        relative_program: &RelativeProgram,
        mapping: &RelativeMapping,
        span: &WindowSpan,
        error_models: &HashMap<u64, ErrorModel>,
    ) -> Result<
        (
            blackbox_decoder::ParityFactor,
            ProjectedErrors,
            Vec<f64>,
            Option<Result<CausalGapProblem, Status>>,
        ),
        Status,
    > {
        let decoder_seed = {
            let gadgets = self.gadgets.read().await;
            crate::coordinator::common_decoder_seed(
                window_gids
                    .iter()
                    .map(|gid| &gadgets[gid])
                    .filter(|gadget| gadget.outcomes.borrow().is_some())
                    .map(|gadget| gadget.decoder_seed),
            )?
        };
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
        let loss_sites = self.build_loss_sites(relative_program, mapping).await;

        // Deduplication collapses edges that share a syndrome, which renumbers
        // them and, worse for a loss-aware decoder, merges a loss generator with
        // any ordinary Pauli error sharing its syndrome. A coordinator that hands
        // losses to the decoder therefore never deduplicates -- not even on a shot
        // carrying no loss, since the graph it loads then serves later shots that
        // do.
        let deduplicate = self.config.merge_hyperedges && !self.loss_handler.hands_off_to_decoder();

        let can_prepare = !self.config.forced_gap
            && !self.config.frame_probabilities
            && target_count == 0
            && !self.loss_handler.hands_off_to_decoder()
            && loss_sites.is_empty()
            && error_models.values().all(|model| {
                model
                    .instance
                    .modifier
                    .as_ref()
                    .is_none_or(|modifier| modifier.probability_modifier.is_none())
            })
            && {
                let gadgets = self.gadgets.read().await;
                mapping.global_gid_of.iter().all(|gid| {
                    gadgets
                        .get(gid)
                        .is_some_and(|gadget| gadget.outcomes.borrow().is_some() && gadget.probability_modifiers.is_empty())
                })
            };
        if can_prepare {
            let key = self
                .prepared_window_key(committing_cids, relative_program, mapping, error_models)
                .await;
            // Cache hits must not enter an async handoff queue behind graph preparation.
            let cached = self.cached_prepared_window(&key);
            let (prepared, cache_hit) = if let Some(prepared) = cached {
                (prepared, true)
            } else {
                let _preparation = self.prepared_window_preparation.lock().await;
                let cached = self.cached_prepared_window(&key);
                if let Some(prepared) = cached {
                    (prepared, true)
                } else {
                    let prepared = self
                        .prepare_window(
                            committing_cids,
                            window_gids,
                            relative_program,
                            mapping,
                            error_models,
                            deduplicate,
                        )
                        .await;
                    if self.config.persistent_decoder {
                        self.persistent_prepared_windows
                            .lock()
                            .unwrap()
                            .insert(key, Arc::clone(&prepared));
                    } else {
                        self.prepared_windows.lock().unwrap().put(key, Arc::clone(&prepared));
                    }
                    (prepared, false)
                }
            };
            span.add_property(|| ("prepared_window", if cache_hit { "hit" } else { "miss" }));
            span.add_property(|| ("graph_edges", prepared.hypergraph.hyperedges.len().to_string()));
            let mut syndrome = syndrome;
            for &vertex in &prepared.ignored_vertices {
                set_bit(&mut syndrome, vertex, false);
            }
            let parity_factor = self
                .decode_temporary(
                    Arc::clone(&prepared.hypergraph),
                    syndrome.clone(),
                    None,
                    decoder_seed,
                    span,
                    Some(&prepared),
                )
                .await?;
            if self.config.assert_parity_factor {
                assert_parity_factor(&prepared.hypergraph, &parity_factor, &syndrome);
            }
            let weights = correction_weights(&prepared.hypergraph, &parity_factor);
            return Ok((
                parity_factor,
                ProjectedErrors::shared(Arc::clone(&prepared.errors)),
                weights,
                None,
            ));
        }

        let logical_flips_are_cacheable = if self.config.forced_gap && target_count != 0 {
            let gadgets = self.gadgets.read().await;
            window_gids.iter().all(|gid| gadgets[gid].instance.modifier.is_none())
        } else {
            true
        };
        let cache_key = if self.config.persistent_decoder && logical_flips_are_cacheable {
            let error_model_types = self.error_model_types.read().await;
            // Construction-time modifiers define the base graph and therefore
            // its cache identity. Outcomes.modifiers are shot-scoped assignments
            // projected below; including them here would defeat decoder reuse.
            Some(DecoderCacheKey {
                relative_program: relative_program.clone(),
                error_model_fingerprints: build_modifier_fingerprints(mapping, error_models, &error_model_types),
                committing_local_cids: committing_local_cids_sorted(committing_cids, mapping),
                logical_flip_signature: logical_flip_signature(logical_targets, mapping),
            })
        } else {
            None
        };

        let mut preparation = None;
        if let Some(ref cache_key) = cache_key {
            let mut loaded = self.loaded_decoders.read().await.get(cache_key).cloned();
            if loaded.is_none() {
                preparation = Some(self.prepared_window_preparation.lock().await);
                loaded = self.loaded_decoders.read().await.get(cache_key).cloned();
            }
            if let Some(loaded) = loaded {
                drop(preparation);
                let probability_reweights = self.projected_shot_probability_reweights(mapping, &loaded.projection).await;
                let projected = self
                    .loss_handler
                    .project_shot(&loaded.projection, &probability_reweights, &loss_sites);
                // we can use the loaded decoding hypergraph to call the decoding service
                span.record_timing("decoding");
                span.add_event(Event::new("decoding").with_property(|| ("type", "loaded")));
                let decode_syndrome = loaded.project_syndrome(syndrome.clone());
                let parity_factor = decode_projected(
                    &self.decoder,
                    &loaded,
                    decode_syndrome.clone(),
                    decoder_seed,
                    projected.reweights.clone(),
                    projected.loss,
                    self.use_loaded_reweights,
                )
                .await?;
                if self.config.assert_parity_factor {
                    assert_parity_factor(loaded.decoding_hypergraph.as_ref().unwrap(), &parity_factor, &decode_syndrome);
                }
                let weights = loaded.correction_weights(&parity_factor, &projected.reweights);
                let forced_gap_problem = if self.config.forced_gap {
                    let mut graph = loaded.projection.base_hypergraph.clone();
                    apply_reweights(&mut graph, probability_reweights.iter().copied());
                    let (graph, _) = self
                        .loss_handler
                        .apply_sites(graph, &loss_sites, &loaded.projection.base_errors);
                    let baseline =
                        original_scoring_baseline(&loaded.projection.base_errors, &projected.errors, &parity_factor);
                    let scoring_errors = ProjectedErrors::shared(Arc::clone(&loaded.projection.base_errors));
                    Some(
                        self.causal_gap_problem(
                            graph,
                            decode_syndrome,
                            decoder_seed,
                            &baseline,
                            &scoring_errors,
                            committing_cids,
                            logical_targets,
                            window_gids,
                            relative_program,
                            mapping,
                            error_models,
                        )
                        .await,
                    )
                } else {
                    None
                };
                return Ok((parity_factor, projected.errors, weights, forced_gap_problem));
            }
        }

        // when the decoder is not available, construct the decoding hypergraph for the window
        // and instantiate such a decoder
        let committing_eids: HashSet<_> = {
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
            .decoding_hypergraph(
                committing_cids,
                logical_targets,
                window_gids,
                relative_program,
                mapping,
                error_models,
            )
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
            let scoring_input = self
                .config
                .forced_gap
                .then(|| (decoding_hypergraph.clone(), Arc::clone(&errors)));
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
            let parity_factor = self
                .decode_temporary(
                    Arc::new(SharedDecodingHypergraph::new(hard_hypergraph)),
                    syndrome.clone(),
                    loss,
                    decoder_seed,
                    span,
                    None,
                )
                .await?;
            if self.config.assert_parity_factor {
                assert_parity_factor(&decoding_hypergraph, &parity_factor, &syndrome);
            }
            let weights = correction_weights(&decoding_hypergraph, &parity_factor);
            let errors: ProjectedErrors = errors.into();
            let forced_gap_problem = if let Some((graph, original_errors)) = scoring_input {
                let baseline = original_scoring_baseline(&original_errors, &errors, &parity_factor);
                let scoring_errors = ProjectedErrors::shared(original_errors);
                Some(
                    self.causal_gap_problem(
                        graph,
                        syndrome,
                        decoder_seed,
                        &baseline,
                        &scoring_errors,
                        committing_cids,
                        logical_targets,
                        window_gids,
                        relative_program,
                        mapping,
                        error_models,
                    )
                    .await,
                )
            } else {
                None
            };
            return Ok((parity_factor, errors, weights, forced_gap_problem));
        };

        // Load the stable base graph before any shot's loss is applied. Keep
        // vertex numbering stable and ignore edge-isolated history-boundary bits.
        span.record_timing("decoding");
        span.add_event(Event::new("decoding").with_property(|| ("type", "loading")));
        let retain_decoding_hypergraph =
            self.config.forced_gap || !self.use_loaded_reweights || self.config.assert_parity_factor;
        let (projection, prepared) = prepare_decoder(decoding_hypergraph, errors, logical_flips, deduplicate, |error| {
            usize::from(committing_eids.contains(&error.eid))
        });
        let loaded = load_projected_decoder(&self.decoder, projection, prepared, retain_decoding_hypergraph, true).await?;
        let probability_reweights = self.projected_shot_probability_reweights(mapping, &loaded.projection).await;
        let decode_syndrome = loaded.project_syndrome(syndrome);
        let projected = self
            .loss_handler
            .project_shot(&loaded.projection, &probability_reweights, &loss_sites);
        let mut loaded_decoders = self.loaded_decoders.write().await;
        loaded_decoders.insert(cache_key, loaded.clone());
        drop(loaded_decoders);
        drop(preparation);
        let parity_factor = decode_projected(
            &self.decoder,
            &loaded,
            decode_syndrome.clone(),
            decoder_seed,
            projected.reweights.clone(),
            projected.loss,
            self.use_loaded_reweights,
        )
        .await?;
        if self.config.assert_parity_factor {
            assert_parity_factor(loaded.decoding_hypergraph.as_ref().unwrap(), &parity_factor, &decode_syndrome);
        }
        let weights = loaded.correction_weights(&parity_factor, &projected.reweights);
        let forced_gap_problem = if self.config.forced_gap {
            let mut graph = loaded.projection.base_hypergraph.clone();
            apply_reweights(&mut graph, probability_reweights.iter().copied());
            let (graph, _) = self
                .loss_handler
                .apply_sites(graph, &loss_sites, &loaded.projection.base_errors);
            let baseline = original_scoring_baseline(&loaded.projection.base_errors, &projected.errors, &parity_factor);
            let scoring_errors = ProjectedErrors::shared(Arc::clone(&loaded.projection.base_errors));
            Some(
                self.causal_gap_problem(
                    graph,
                    decode_syndrome,
                    decoder_seed,
                    &baseline,
                    &scoring_errors,
                    committing_cids,
                    logical_targets,
                    window_gids,
                    relative_program,
                    mapping,
                    error_models,
                )
                .await,
            )
        } else {
            None
        };
        Ok((parity_factor, projected.errors, weights, forced_gap_problem))
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
        error_models: &HashMap<u64, ErrorModel>,
    ) -> (DecodingHypergraph, Arc<Vec<ErrorIndex>>, Vec<Vec<u64>>) {
        #[cfg(feature = "cli")]
        log::info!("constructing decoding hypergraph for cids={committing_cids:?}");
        let error_model_types = self.error_model_types.read().await;
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
                    if is_in_commit_region
                        && error.checks.iter().any(|check| {
                            check
                                .remote_check_model
                                .is_some_and(|remote| expanded_remotes[remote as usize].is_none())
                        })
                    {
                        continue;
                    }
                    // Zero-probability errors are kept, at their prior probability.
                    // They exist for a reason -- an atom loss activates its
                    // Pauli-envelope generators, and a caller may reweight any edge
                    // -- so the hyperedge set stays independent of what happened in
                    // a given shot, keeping edge indices stable across shots.
                    let mut vertices: Vec<u64> = vec![];
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
                            }
                        } else if has_local_check {
                            // Local check on a normal window gadget
                            vertices.push(local_start_index + check.check_index);
                        }
                        // Error-only gadgets: local checks belong to outside check model,
                        // project them out.
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
                        observable_flips: None,
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
}
