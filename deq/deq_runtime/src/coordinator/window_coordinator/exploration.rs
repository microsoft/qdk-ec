//! History readiness, window discovery, commit-region selection, and remote context.

use super::collection::StatePins;
use super::error_metadata::referenced_error_remotes;
use super::{ErrorModel, Gadget, WindowCoordinator, WindowParallelism};
use crate::bin;
use crate::misc::sync::{check_or_future, get_or_future};
use futures_util::future::join_all;
use hashbrown::{HashMap, HashSet};
use std::collections::VecDeque;
use tokio_util::sync::CancellationToken;
use tonic::Status;

/// Cooperative scheduling heuristic; correctness does not depend on the exact positive value.
const COOPERATIVE_MAX_HISTORY_BATCH_SIZE: usize = 64;

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
/// 1. `explore_mandatory_zone`: populates `gadgets`, `center_distance`,
///    `phase` for MandatoryZone members, and `frontier` (BFS frontier to
///    continue from).
/// 2. `await_mandatory_zone_syndrome`: waits for syndrome; no mutation.
/// 3. `explore_lookahead_zone`: extends all fields with LookaheadZone members.
/// 4. `select_commit_region`: sets `commit_region` and `committing_cids`.
/// 5. `shrink_window`: sets `decoder_window`.
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

impl ExploredWindow {
    fn enqueue_interior(&mut self, gid: u64, radius: usize, is_free_hop: bool) {
        if self.center_distance[&gid] < radius {
            if is_free_hop {
                self.frontier.push_front(gid);
            } else {
                self.frontier.push_back(gid);
            }
        }
    }
}

impl WindowCoordinator {
    pub(super) async fn wait_for_history_commitment(&self, gid: u64) -> Result<(), Status> {
        if self.config.window_parallelism == WindowParallelism::FullyParallel {
            return Ok(());
        }
        let token = self.cancellation.read().await.clone();
        if self.config.window_parallelism == WindowParallelism::Sliding {
            return self.wait_for_sliding_history(gid, token).await;
        }
        let watchers: Vec<_> = {
            let gadgets = self.gadgets.read().await;
            gadgets
                .iter()
                .filter(|(other_gid, gadget)| {
                    **other_gid < gid
                        && (self.config.buffer_radius == 0 || !gadget.is_free_hop)
                        && !gadget.state.borrow().committed
                })
                .map(|(_, gadget)| gadget.state.subscribe())
                .collect()
        };
        for mut watcher in watchers {
            tokio::select! {
                result = watcher.wait_for(|state| state.committed) => {
                    result.map_err(|_| Status::cancelled("history predecessor removed"))?;
                }
                () = token.cancelled() => return Err(Status::cancelled("history commitment wait cancelled")),
            }
        }
        Ok(())
    }

    async fn wait_for_sliding_history(&self, gid: u64, token: CancellationToken) -> Result<(), Status> {
        let mut pending = vec![gid];
        let mut visited = HashSet::new();
        let mut pins = StatePins::new(&self.pinned_gadgets);
        while !pending.is_empty() {
            if token.is_cancelled() {
                return Err(Status::cancelled("history commitment wait cancelled"));
            }
            let mut watcher = None;
            {
                let gadgets = self.gadgets.read().await;
                // Bound lock-held work even for long chains of committed or free-hop gadgets.
                for _ in 0..COOPERATIVE_MAX_HISTORY_BATCH_SIZE {
                    let Some(current) = pending.pop() else { break };
                    if visited.contains(&current) {
                        continue;
                    }
                    let gadget = gadgets
                        .get(&current)
                        .ok_or_else(|| Status::not_found(format!("gid={current}")))?;
                    let committed = gadget.state.borrow().committed;
                    if current != gid && !gadget.is_free_hop && !committed {
                        pending.push(current);
                        watcher = Some(gadget.state.subscribe());
                        break;
                    }
                    visited.insert(current);
                    // A final frame requires corrections and final frames from every input,
                    // including remote readouts. Commitment alone does not close history:
                    // another leader may have absorbed this gadget before its ancestors.
                    if current != gid && committed && gadget.pauli_frame.borrow().is_some() {
                        continue;
                    }
                    pending.extend(gadget.instance.connectors.iter().map(|input| input.gid));
                    if let Some(remote) = gadget
                        .instance
                        .modifier
                        .as_ref()
                        .and_then(|modifier| modifier.remote_conditional_correction.as_ref())
                    {
                        pending.extend(
                            remote
                                .remote_readouts
                                .iter()
                                .map(|readout| readout.gid)
                                .filter(|gid| gadgets.contains_key(gid)),
                        );
                    }
                }
                pins.retain(pending.iter().copied());
            }
            if let Some(mut watcher) = watcher {
                tokio::select! {
                    result = watcher.wait_for(|state| state.committed) => {
                        result.map_err(|_| Status::cancelled("history predecessor removed"))?;
                    }
                    () = token.cancelled() => return Err(Status::cancelled("history commitment wait cancelled")),
                }
            } else if !pending.is_empty() {
                tokio::task::yield_now().await;
            }
        }
        Ok(())
    }

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
    pub(super) async fn explore_mandatory_zone(&self, center_gid: u64, pins: &mut StatePins) -> Option<ExploredWindow> {
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

        if buffer_radius == 0 {
            explored.frontier.clear();
            return Some(explored);
        }

        while let Some(fgid) = explored.frontier.pop_front() {
            if token.is_cancelled() {
                return None;
            }
            let my_dist = explored.center_distance[&fgid];
            let mut sync_neighbors: Vec<(u64, bool)> = vec![];
            let mut async_handles = vec![];

            {
                let gadgets = self.gadgets.read().await;
                let gadget = gadgets.get(&fgid)?;
                pins.retain([fgid]);

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
                    match get_or_future(sender, token.clone()) {
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
                pins.retain(explored.gadgets.iter().copied());
            }

            // Enqueue sync neighbors: free-hops to front (distance 0), others to back.
            for (gid, is_free_hop) in sync_neighbors {
                explored.enqueue_interior(gid, buffer_radius, is_free_hop);
            }

            // Await async output handles.
            for handle in async_handles {
                if let Some(connector) = handle.await {
                    if explored.gadgets.contains(&connector.gid) {
                        continue;
                    }
                    let gadgets = self.gadgets.read().await;
                    let peer = &gadgets[&connector.gid];
                    pins.retain([connector.gid]);
                    let peer_dist = if peer.is_free_hop { my_dist } else { my_dist + 1 };
                    if peer_dist <= buffer_radius {
                        explored.gadgets.insert(connector.gid);
                        explored.center_distance.insert(connector.gid, peer_dist);
                        explored.phase.insert(connector.gid, ExplorePhase::MandatoryZone);
                        explored.enqueue_interior(connector.gid, buffer_radius, peer.is_free_hop);
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
    pub(super) async fn await_mandatory_zone_syndrome(&self, explored: &ExploredWindow) -> Option<()> {
        let token = self.cancellation.read().await.clone();
        let mut outcomes = Vec::new();
        {
            let gadgets = self.gadgets.read().await;
            for gid in &explored.gadgets {
                let gadget = gadgets.get(gid)?;
                if let Err(handle) = check_or_future(&gadget.outcomes, token.clone()) {
                    outcomes.push(handle);
                }
            }
        }
        join_all(outcomes).await;
        if token.is_cancelled() {
            return None;
        }

        // Collect CIDs of check models in the mandatory zone.
        // Outcomes are submitted after check-model binding. Reading CIDs before
        // that point could commit a newly connected gadget without its checks.
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
        let mut handles = vec![];
        {
            let check_models = self.check_models.read().await;
            for cid in &mandatory_cids {
                if let Some(check_model) = check_models.get(cid)
                    && let Err(handle) = check_or_future(&check_model.syndrome, token.clone())
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
    /// 2. **Syndrome readiness**: includes only gadgets with published outcomes
    ///    and, when a check model is bound, an available syndrome. This also
    ///    applies to free hops: outcome publication makes their binding stable.
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
    pub(super) async fn explore_lookahead_zone(&self, explored: &mut ExploredWindow, pins: &mut StatePins) -> Option<()> {
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

                // A connected peer may still be binding its check model.
                // Only published outcomes make its binding stable.
                let syndrome_ready = |peer: &Gadget| -> bool {
                    if peer.outcomes.borrow().is_none() {
                        return false;
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
                pins.retain(explored.gadgets.iter().copied());
            }

            // Enqueue sync neighbors.
            for (gid, is_free_hop) in sync_neighbors {
                explored.enqueue_interior(gid, window_radius, is_free_hop);
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
    ///    (input or output) going outside the explored window, or an unconnected
    ///    output port. Committed gadgets inside the window still provide context;
    ///    commitment alone does not make them boundary seeds.
    /// 2. Run an inward 0-1 BFS from boundary seeds to compute `boundary_dist`
    ///    for every gadget in the window.  Free-hops contribute 0 to distance.
    ///    Gadgets unreachable from any boundary have `boundary_dist = ∞`
    ///    (e.g., a fully self-contained window with no external connections).
    /// 3. All uncommitted, unreserved hop-counted gadgets with
    ///    `boundary_dist ≥ buffer_radius` are committed.  The center gadget
    ///    always satisfies this because step 1's blocking BFS guarantees
    ///    `buffer_radius` hops of context in all directions.
    /// 4. Free-hop gadgets adjacent to the commit region (or to already
    ///    committed gadgets) are absorbed iteratively.
    ///
    /// Populates `explored.commit_region` and `explored.committing_cids`.
    pub(super) fn select_commit_region(&self, explored: &mut ExploredWindow, gadgets: &HashMap<u64, Gadget>) {
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
        // connections go outside the explored window or remain unconnected.
        // Committed gadgets inside the window still contribute syndrome context.
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
            let state = gadget.state.borrow();
            if state.committed || state.reserved_by.is_some() {
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
                let state = g.state.borrow();
                if !g.is_free_hop || state.committed || state.reserved_by.is_some() {
                    continue;
                }
                drop(state);
                let in_region = |ngid: u64| {
                    explored.commit_region.contains(&ngid) || gadgets.get(&ngid).is_some_and(|p| p.state.borrow().committed)
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
    pub(super) fn shrink_window(&self, explored: &mut ExploredWindow, gadgets: &HashMap<u64, Gadget>) {
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

    pub(super) async fn available_remote_check_owners(&self, owners: &HashSet<u64>) -> HashMap<u64, HashSet<u64>> {
        let error_model_types = self.error_model_types.read().await;
        let remote_indices = self.referenced_error_remotes.read().await;
        let gadgets = self.gadgets.read().await;
        let check_models = self.check_models.read().await;
        let error_models = self.error_models.read().await;
        let mut dependencies = HashMap::new();
        for &owner_gid in owners {
            let owner = &gadgets[&owner_gid];
            if owner.state.borrow().committed {
                continue;
            }
            let Some(cid) = owner.binding_cid else { continue };
            let mut targets = HashSet::new();
            for eid in &check_models[&cid].attaching_eid_vec {
                let error_model = &error_models[eid];
                let remotes = &error_model.modified_remote_check_models;
                let model_type = &error_model_types[&error_model.instance.etype];
                let used_remotes = remote_indices
                    .get(&error_model.instance.etype)
                    .and_then(|cached| cached.indices_for_type(model_type))
                    .unwrap_or_else(|| referenced_error_remotes(model_type));
                let mut resolved = vec![None; remotes.len()];
                for &remote_index in used_remotes.iter() {
                    let remote_index = usize::try_from(remote_index).unwrap();
                    Self::resolve_remote_check_model_gid(&mut resolved, remote_index, remotes, owner_gid, &gadgets);
                    let Some(remote) = &remotes[remote_index] else { continue };
                    let target_cid = remote.absolute_cid.or_else(|| {
                        resolved[remote_index]
                            .and_then(|gid| gadgets.get(&gid))
                            .and_then(|gadget| gadget.binding_cid)
                    });
                    if let Some(check_model) = target_cid.and_then(|cid| check_models.get(&cid))
                        && check_model.syndrome.borrow().is_some()
                    {
                        targets.insert(check_model.instance.gid);
                    }
                }
            }
            dependencies.insert(owner_gid, targets);
        }
        dependencies
    }

    pub(super) async fn await_error_models(&self, window: &HashSet<u64>) -> Result<HashSet<u64>, Status> {
        let token = self.cancellation.read().await.clone();
        let mut readiness = vec![];
        let mut terminal_gids = HashSet::new();
        {
            let gadgets = self.gadgets.read().await;
            let checks = self.check_models.read().await;
            let mut terminal_boundaries = None;
            for gid in window {
                let Some(gadget) = gadgets.get(gid) else { continue };
                let Some(check) = gadget.binding_cid.and_then(|cid| checks.get(&cid)) else {
                    continue;
                };
                let expected = check.instance.error_model_count.unwrap_or(1);
                if gadget.state.borrow().committed || check.attaching_eid_vec.len() as u64 >= expected {
                    continue;
                }
                if expected == 1
                    && check.instance.terminal_error_model.is_some()
                    && terminal_boundaries
                        .get_or_insert_with(|| Self::terminal_boundary_gids(&gadgets, window))
                        .contains(gid)
                {
                    terminal_gids.insert(*gid);
                } else if let Err(handle) = check_or_future(&check.error_model_ready, token.clone()) {
                    readiness.push(handle);
                }
            }
        }
        join_all(readiness).await;
        if token.is_cancelled() {
            return Err(Status::cancelled("error model readiness wait cancelled"));
        }
        Ok(terminal_gids)
    }

    pub(super) fn terminal_boundary_gids(gadgets: &HashMap<u64, Gadget>, window: &HashSet<u64>) -> HashSet<u64> {
        let mut terminal_gids = window.clone();
        let mut pending: Vec<_> = window
            .iter()
            .filter_map(|gid| gadgets.get(gid))
            .flat_map(|gadget| gadget.instance.connectors.iter().map(|connector| connector.gid))
            .collect();
        let mut visited = HashSet::new();
        while let Some(current) = pending.pop() {
            if window.contains(&current) {
                terminal_gids.remove(&current);
                continue;
            }
            if visited.insert(current)
                && let Some(gadget) = gadgets.get(&current)
            {
                pending.extend(gadget.instance.connectors.iter().map(|connector| connector.gid));
            }
        }
        terminal_gids
    }

    pub(super) fn expand_remote_check_models_in_window(
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
    pub(super) fn resolve_remote_check_model_gid(
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
    pub(super) fn find_blocking_port(
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
}
