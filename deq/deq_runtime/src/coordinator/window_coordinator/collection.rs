//! Dependency-aware retention, state pins, and deferred type retirement.

use super::{CheckModel, ErrorModel, Gadget, WindowCoordinator};
use crate::coordinator::DecoderCacheKey;
use hashbrown::{HashMap, HashSet};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tonic::Status;

/// Minimum readout-request batch, avoiding frequent scans of small live graphs.
const MIN_COLLECTION_PROGRESS_INTERVAL: usize = 32;
/// Scale larger batches to roughly one-eighth of the live gadget count to amortize full scans.
/// These thresholds trade scan frequency for reclamation delay, not correctness.
const COLLECTION_PROGRESS_DIVISOR: usize = 8;

/// Keeps selected gadget state alive across awaits, releasing pins when the task finishes or is cancelled.
pub(super) struct StatePins {
    /// Shared per-gadget counts of guards that still need the state.
    registry: Arc<std::sync::Mutex<HashMap<u64, usize>>>,
    /// IDs owned by this guard, counted once each in the registry.
    ids: HashSet<u64>,
}

impl StatePins {
    /// Create an empty guard sharing the coordinator's pin registry.
    pub(super) fn new(registry: &Arc<std::sync::Mutex<HashMap<u64, usize>>>) -> Self {
        Self {
            registry: Arc::clone(registry),
            ids: HashSet::new(),
        }
    }

    /// Add pins without releasing earlier ones; repeated IDs do not increase this guard's counts.
    /// Hold the gadget-map lock while discovering and pinning state to prevent a collection race.
    pub(super) fn retain(&mut self, ids: impl IntoIterator<Item = u64>) {
        let mut registry = self.registry.lock().unwrap();
        for id in ids {
            if self.ids.insert(id) {
                *registry.entry(id).or_default() += 1;
            }
        }
    }
}

impl Drop for StatePins {
    /// Release this guard's counts synchronously, including when its owning future is dropped.
    fn drop(&mut self) {
        let mut registry = self.registry.lock().unwrap();
        for id in &self.ids {
            let count = registry.get_mut(id).expect("live state pin");
            *count -= 1;
            if *count == 0 {
                registry.remove(id);
            }
        }
    }
}

impl WindowCoordinator {
    /// Whether history can be discarded without breaking forced-gap scoring or tracing.
    pub(crate) fn automatic_collection(&self) -> bool {
        !self.config.forced_gap && self.config.trace_filepath.is_none()
    }

    /// Retain unfinished, reserved, or extendable gadgets and their geometric decoding context.
    /// Traversal includes one hop beyond the configured window radius; free-hop gadgets add no cost.
    pub(super) fn live_window_gadgets(
        &self,
        gadgets: &HashMap<u64, Gadget>,
        completed: &HashSet<u64>,
        checks: &HashMap<u64, CheckModel>,
    ) -> HashSet<u64> {
        let radius = self.config.effective_window_radius().saturating_add(1);
        let mut distances = HashMap::new();
        let mut pending = VecDeque::new();
        for (&gid, gadget) in gadgets {
            if !completed.contains(&gid)
                || !gadget.state.borrow().committed
                || gadget.state.borrow().reserved_by.is_some()
                || gadget.outputs.iter().any(|output| output.borrow().is_none())
                || gadget
                    .binding_cid
                    .and_then(|cid| checks.get(&cid))
                    .is_some_and(|check| check.error_model_ready.borrow().is_none())
            {
                distances.insert(gid, 0);
                pending.push_back(gid);
            }
        }
        while let Some(gid) = pending.pop_front() {
            let distance = distances[&gid];
            if distance >= radius {
                continue;
            }
            let gadget = &gadgets[&gid];
            let neighbors = gadget.instance.connectors.iter().map(|peer| peer.gid).chain(
                gadget
                    .outputs
                    .iter()
                    .filter_map(|peer| peer.borrow().as_ref().map(|peer| peer.gid)),
            );
            for neighbor in neighbors {
                let Some(peer) = gadgets.get(&neighbor) else { continue };
                let next_distance = distance + usize::from(!peer.is_free_hop);
                if next_distance <= radius && distances.get(&neighbor).is_none_or(|previous| next_distance < *previous) {
                    distances.insert(neighbor, next_distance);
                    if peer.is_free_hop {
                        pending.push_front(neighbor);
                    } else {
                        pending.push_back(neighbor);
                    }
                }
            }
        }
        distances.into_keys().collect()
    }

    /// Retain pending syndrome inputs and remote-check owners needed by uncommitted errors.
    /// Unexpanded relative measurement paths conservatively retain the whole current graph.
    ///
    /// Returns an error if an error model has no owning check model.
    pub(super) fn retain_pending_dependencies(
        retained: &mut HashSet<u64>,
        gadgets: &HashMap<u64, Gadget>,
        checks: &HashMap<u64, CheckModel>,
        errors: &HashMap<u64, ErrorModel>,
    ) -> Result<(), Status> {
        for check in checks.values() {
            if check.syndrome.borrow().is_none() {
                if let Some(remotes) = &check.expanded_remote_gadgets {
                    for &gid in remotes.iter().flatten() {
                        retained.insert(gid);
                    }
                } else {
                    for remote in check.modified_remote_gadgets.iter().flatten() {
                        if let Some(gid) = remote.absolute_gid {
                            retained.insert(gid);
                        } else {
                            retained.extend(gadgets.keys());
                        }
                    }
                }
            }
        }
        for model in errors.values() {
            let Some(check) = checks.get(&model.instance.cid) else {
                return Err(Status::internal("error model has no owning check model"));
            };
            if gadgets[&check.instance.gid].state.borrow().committed {
                continue;
            }
            let mut resolved = vec![None; model.modified_remote_check_models.len()];
            for index in 0..resolved.len() {
                Self::resolve_remote_check_model_gid(
                    &mut resolved,
                    index,
                    &model.modified_remote_check_models,
                    check.instance.gid,
                    gadgets,
                );
            }
            for gid in resolved.into_iter().flatten().filter(|gid| *gid != u64::MAX) {
                retained.insert(gid);
            }
            for remote in model.modified_remote_check_models.iter().flatten() {
                if let Some(check) = remote.absolute_cid.and_then(|cid| checks.get(&cid)) {
                    retained.insert(check.instance.gid);
                }
            }
        }
        Ok(())
    }

    /// Preserve paths between retained nodes, not all ancestors: terminal-model selection
    /// can walk outside a window and reenter it, so removing these paths would change its result.
    pub(super) fn retain_connecting_paths(retained: &mut HashSet<u64>, gadgets: &HashMap<u64, Gadget>) {
        let mut ancestors = retained.clone();
        let mut pending: Vec<_> = retained.iter().copied().collect();
        while let Some(gid) = pending.pop() {
            if let Some(gadget) = gadgets.get(&gid) {
                for input in &gadget.instance.connectors {
                    if gadgets.contains_key(&input.gid) && ancestors.insert(input.gid) {
                        pending.push(input.gid);
                    }
                }
            }
        }
        let mut pending: Vec<_> = retained.iter().copied().collect();
        while let Some(gid) = pending.pop() {
            if let Some(gadget) = gadgets.get(&gid) {
                for output in &gadget.outputs {
                    if let Some(peer) = output.borrow().as_ref()
                        && ancestors.contains(&peer.gid)
                        && retained.insert(peer.gid)
                    {
                        pending.push(peer.gid);
                    }
                }
            }
        }
    }

    /// Request a full collection pass unless history must be retained.
    /// Concurrent requests share a pass when its starting generation already covers them.
    pub(super) async fn collect_garbage(&self) -> Result<(), Status> {
        if !self.automatic_collection() {
            return Ok(());
        }
        self.collection_requests.fetch_add(1, Ordering::AcqRel);
        let mut completed = self.collection_completed.lock().await;
        // Requests arriving during the pass may release more state and need a later pass.
        let requested = self.collection_requests.load(Ordering::Acquire);
        if *completed != requested {
            self.collect_garbage_pass().await?;
            *completed = requested;
        }
        Ok(())
    }

    /// Amortize collection over readout completions instead of scanning after every readout.
    /// Once every live gadget has completed its readout, collect without waiting for the budget.
    pub(super) async fn collect_after_readout(&self) -> Result<(), Status> {
        if !self.automatic_collection() {
            return Ok(());
        }
        self.collection_requests.fetch_add(1, Ordering::AcqRel);
        let mut completed = self.collection_completed.lock().await;
        let requested = self.collection_requests.load(Ordering::Acquire);
        if *completed == requested {
            return Ok(());
        }
        let (live, finished) = {
            let gadgets = self.gadgets.read().await;
            (gadgets.len(), self.completed_readouts.lock().await.len())
        };
        let progress_budget =
            u64::try_from((live / COLLECTION_PROGRESS_DIVISOR).max(MIN_COLLECTION_PROGRESS_INTERVAL)).unwrap();
        // Finished programs and explicit unloads still drain collection immediately.
        if finished < live && requested.wrapping_sub(*completed) < progress_budget {
            return Ok(());
        }
        self.collect_garbage_pass().await?;
        *completed = requested;
        Ok(())
    }

    /// Retire execution state first, then declarations no longer referenced by surviving models.
    /// Graph locks are released before type retirement acquires library locks.
    async fn collect_garbage_pass(&self) -> Result<(), Status> {
        self.collect_gadget_state().await?;
        self.collect_unused_types().await;
        Ok(())
    }

    /// Remove gadgets and models outside the retention set, including stale referral entries.
    /// Preserve retired ID ranges and any hinted logical readouts needed by future consumers.
    pub(super) async fn collect_gadget_state(&self) -> Result<(), Status> {
        let mut gadgets = self.gadgets.write().await;
        let mut checks = self.check_models.write().await;
        let mut errors = self.error_models.write().await;
        let mut completed = self.completed_readouts.lock().await;
        if completed.is_empty() {
            // Every gadget is rooted by its unfinished readout.
            return Ok(());
        }
        let mut retained = self.live_window_gadgets(&gadgets, &completed, &checks);
        retained.extend(self.pinned_gadgets.lock().unwrap().keys());
        Self::retain_pending_dependencies(&mut retained, &gadgets, &checks, &errors)?;
        Self::retain_connecting_paths(&mut retained, &gadgets);
        let mut collected_gadgets = self.collected_gids.lock().await;
        let mut collected_checks = self.collected_cids.lock().await;
        let mut collected_errors = self.collected_eids.lock().await;
        let mut tracker = self.pauli_frame_tracker.lock().await;
        checks.retain(|cid, check| {
            if retained.contains(&check.instance.gid) {
                return true;
            }
            collected_checks.insert(*cid..=*cid);
            for eid in &check.attaching_eid_vec {
                collected_errors.insert(*eid..=*eid);
            }
            false
        });
        gadgets.retain(|gid, _| {
            if retained.contains(gid) {
                return true;
            }
            collected_gadgets.insert(*gid..=*gid);
            tracker.collect_gadget(*gid);
            false
        });
        errors.retain(|_, model| checks.contains_key(&model.instance.cid));
        for check in checks.values_mut() {
            check.referring_eids.retain(|eid| errors.contains_key(eid));
        }
        self.pending_referring_by_gid.lock().await.retain(|_, eids| {
            eids.retain(|eid| errors.contains_key(eid));
            !eids.is_empty()
        });
        self.pending_referring_by_cid.lock().await.retain(|_, eids| {
            eids.retain(|eid| errors.contains_key(eid));
            !eids.is_empty()
        });
        self.pending_referring_by_port.lock().await.retain(|_, referrals| {
            referrals.retain(|referral| errors.contains_key(&referral.eid));
            !referrals.is_empty()
        });
        completed.retain(|gid| retained.contains(gid));
        Ok(())
    }

    /// Fulfil pending type unloads only when no live model or terminal fallback references them.
    /// Unload backend handles and evict dependent graph caches before removing the declarations.
    pub(super) async fn collect_unused_types(&self) {
        {
            let unload = self.pending_type_unloads.lock().await;
            if unload.ctype.is_empty() && unload.etype.is_empty() {
                return;
            }
        }
        let mut check_types = self.check_model_types.write().await;
        let mut error_types = self.error_model_types.write().await;
        let mut remote_indices = self.referenced_error_remotes.write().await;
        let checks = self.check_models.read().await;
        let errors = self.error_models.read().await;
        let mut unload = self.pending_type_unloads.lock().await;
        let live_checks: HashSet<_> = checks.values().map(|check| check.instance.ctype).collect();
        let mut live_errors: HashSet<_> = errors.values().map(|model| model.instance.etype).collect();
        live_errors.extend(
            checks
                .values()
                .filter_map(|check| check.instance.terminal_error_model.as_ref().map(|model| model.etype)),
        );
        let retired_checks: HashSet<_> = unload.ctype.iter().copied().filter(|id| !live_checks.contains(id)).collect();
        let retired_errors: HashSet<_> = unload.etype.iter().copied().filter(|id| !live_errors.contains(id)).collect();
        if retired_checks.is_empty() && retired_errors.is_empty() {
            return;
        }
        let uses_retired_type = |key: &DecoderCacheKey| {
            key.relative_program.local_gadgets.iter().any(|gadget| {
                gadget
                    .check_model
                    .as_ref()
                    .is_some_and(|check| retired_checks.contains(&check.ctype))
                    || gadget.error_models.iter().any(|error| retired_errors.contains(&error.etype))
            })
        };
        let mut loaded = self.loaded_decoders.write().await;
        let mut retired_handles: Vec<_> = loaded
            .iter()
            .filter(|(key, _)| uses_retired_type(key))
            .map(|(_, decoder)| decoder.hid)
            .collect();
        {
            let prepared = self.persistent_prepared_windows.lock().unwrap();
            retired_handles.extend(
                prepared
                    .iter()
                    .filter(|(key, _)| uses_retired_type(key))
                    .filter_map(|(_, window)| window.loaded_hid.get().copied()),
            );
        }
        // Live models pin types until their windows finish. Unload before removing
        // cache entries so cancellation cannot orphan backend handles.
        if !retired_handles.is_empty() {
            self.decoder.unload_hypergraphs(&retired_handles).await;
        }
        loaded.retain(|key, _| !uses_retired_type(key));
        self.persistent_prepared_windows
            .lock()
            .unwrap()
            .retain(|key, _| !uses_retired_type(key));
        {
            let mut prepared = self.prepared_windows.lock().unwrap();
            let retired_keys: Vec<_> = prepared
                .iter()
                .filter(|(key, _)| uses_retired_type(key))
                .map(|(key, _)| key.clone())
                .collect();
            for key in retired_keys {
                prepared.pop(&key);
            }
        }
        unload.ctype.retain(|id| {
            if live_checks.contains(id) {
                true
            } else {
                check_types.remove(id);
                false
            }
        });
        unload.etype.retain(|id| {
            if live_errors.contains(id) {
                true
            } else {
                error_types.remove(id);
                remote_indices.remove(id);
                false
            }
        });
    }
}
