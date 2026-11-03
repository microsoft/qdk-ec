//! Request validation, publication, decode orchestration, and reset lifecycle.

use super::collection::StatePins;
use super::error_metadata::{CachedErrorRemotes, terminal_check_shape, terminal_shape_fits};
use super::{CheckModel, ErrorModel, ExplorePhase, Gadget, GadgetState, PendingPortReferral, WindowCoordinator, trace};
use crate::coordinator::loss_handler::{apply_loss_random_imputation, has_loss_model};
use crate::decoder::blackbox_decoder;
use crate::misc::bit_vector;
use crate::misc::bit_vector::{get_bit, set_bit};
use crate::misc::index::WILDCARD;
use crate::misc::sync::{check_or_future, get_or_future};
use crate::misc::validation::{
    apply_check_model_reroutes, apply_error_model_reroutes, validate_outcomes, validate_probability_modifier,
};
use crate::util::BitVector;
use crate::{bin, coordinator};
use hashbrown::{HashMap, HashSet};
use prost::Message;
use std::sync::Arc;
use tokio::sync::{RwLock, watch};
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status};

#[tonic::async_trait]
impl coordinator::coordinator_server::Coordinator for WindowCoordinator {
    async fn get_memory_stats(&self, _request: Request<()>) -> Result<Response<coordinator::MemoryStats>, Status> {
        let check_types = self.check_model_types.read().await;
        let error_types = self.error_model_types.read().await;
        let gadgets = self.gadgets.read().await;
        let checks = self.check_models.read().await;
        let errors = self.error_models.read().await;
        let collected_id_ranges = self.collected_gids.lock().await.len()
            + self.collected_cids.lock().await.len()
            + self.collected_eids.lock().await.len();
        let tracker = self.pauli_frame_tracker.lock().await;
        Ok(Response::new(coordinator::MemoryStats {
            automatic_collection: self.automatic_collection(),
            persistent_decoder: self.config.persistent_decoder,
            gadgets: gadgets.len() as u64,
            check_models: checks.len() as u64,
            error_models: errors.len() as u64,
            check_model_types: check_types.len() as u64,
            error_model_types: error_types.len() as u64,
            frame_gadgets: tracker.gadgets.len() as u64,
            retained_readouts: tracker.retained_readout_count() as u64,
            retained_readout_bytes: tracker.retained_readout_bytes() as u64,
            collected_id_ranges: collected_id_ranges as u64,
            ..Default::default()
        }))
    }

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
        let mut remote_indices = self.referenced_error_remotes.write().await;
        for error_model_type in library.error_model_types.into_iter() {
            if error_model_types.contains_key(&error_model_type.etype) {
                return Err(Status::already_exists(format!("etype={}", error_model_type.etype)));
            }
            let error_model_type = Arc::new(error_model_type);
            remote_indices.insert(error_model_type.etype, CachedErrorRemotes::new(&error_model_type));
            error_model_types.insert(error_model_type.etype, error_model_type);
        }
        drop(error_model_types);
        Ok(().into())
    }

    async fn unload(&self, unload: Request<coordinator::UnloadLibrary>) -> Result<Response<()>, Status> {
        let _guard = self
            .task_counter
            .try_guard()
            .ok_or_else(|| Status::unavailable("coordinator reset in progress"))?;
        let unload = unload.into_inner();
        if !unload.gtype.is_empty() || !self.automatic_collection() {
            return Err(Status::failed_precondition(
                "deferred unloading requires automatic collection and only check/error types",
            ));
        }
        {
            let mut pending = self.pending_type_unloads.lock().await;
            pending.ctype.extend(unload.ctype);
            pending.etype.extend(unload.etype);
            pending.ctype.sort_unstable();
            pending.ctype.dedup();
            pending.etype.sort_unstable();
            pending.etype.dedup();
        }
        self.collect_garbage().await?;
        Ok(Response::new(()))
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
        let mut collect_after_publication = false;
        let id = match create {
            bin::instruction::Create::Gadget(gadget) => {
                let port_types = self.port_types.read().await;
                let gadget_types = self.gadget_types.read().await;
                let mut gadgets = self.gadgets.write().await;
                if gadget.gid != 0
                    && (gadgets.contains_key(&gadget.gid) || self.collected_gids.lock().await.contains(&gadget.gid))
                {
                    return Err(Status::invalid_argument("gadget id already exists or has been collected"));
                }
                let gadget_type = gadget_types
                    .get(&gadget.gtype)
                    .ok_or_else(|| Status::not_found(format!("gtype={}", gadget.gtype)))?;
                if gadget.connectors.len() != gadget_type.inputs.len() {
                    return Err(Status::invalid_argument("connector count does not match gadget inputs"));
                }
                for (index, connector) in gadget.connectors.iter().enumerate() {
                    if gadget.connectors[..index].contains(connector) {
                        return Err(Status::invalid_argument("input port is connected more than once"));
                    }
                    let peer = gadgets
                        .get(&connector.gid)
                        .ok_or_else(|| Status::invalid_argument("connector refers to an unknown or retired gadget"))?;
                    let port = usize::try_from(connector.port)
                        .map_err(|_| Status::invalid_argument("connector output port exceeds usize"))?;
                    let output = peer
                        .outputs
                        .get(port)
                        .ok_or_else(|| Status::invalid_argument("connector output port is out of range"))?;
                    if output.borrow().is_some() {
                        return Err(Status::failed_precondition("connector output port is already connected"));
                    }
                }
                self.pauli_frame_tracker
                    .lock()
                    .await
                    .validate_remote_readouts(gadget.modifier.as_ref())
                    .map_err(Status::invalid_argument)?;
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
                    let mut next_gid = self.next_gid.lock().await;
                    *next_gid = (*next_gid).max(gadget.gid.saturating_add(1));
                    gadget.gid
                };
                for (port, connector) in gadget.connectors.iter().enumerate() {
                    let output = usize::try_from(connector.port).expect("connector was validated");
                    gadgets.get_mut(&connector.gid).unwrap().outputs[output]
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
                        state: watch::channel(GadgetState::default()).0,
                        loss_mask: None,
                        decoder_seed: None,
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
                                            if let Some(target_cm) = check_models.get_mut(&target_cid)
                                                && !target_cm.referring_eids.contains(&referral.eid)
                                            {
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
                tracker
                    .add_committing_gadget(&gadget, gadget_type, &port_types)
                    .map_err(Status::invalid_argument)?;
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
                let check_model_type = check_model_types
                    .get(&check_model.ctype)
                    .ok_or_else(|| Status::not_found(format!("ctype={}", check_model.ctype)))?;
                let error_model_types = self.error_model_types.read().await;
                if let Some(terminal) = &check_model.terminal_error_model {
                    if check_model.error_model_count.unwrap_or(1) != 1 {
                        return Err(Status::invalid_argument(
                            "terminal fallback requires exactly one full error model",
                        ));
                    }
                    let model_type = error_model_types
                        .get(&terminal.etype)
                        .ok_or_else(|| Status::not_found(format!("terminal etype={}", terminal.etype)))?;
                    let valid_checks = self
                        .referenced_error_remotes
                        .read()
                        .await
                        .get(&terminal.etype)
                        .and_then(|cached| cached.terminal_for_type(model_type, check_model_type.checks.len()))
                        .unwrap_or_else(|| {
                            terminal_shape_fits(terminal_check_shape(model_type), check_model_type.checks.len())
                        });
                    if terminal.eid == 0
                        || (model_type.ctype != WILDCARD && model_type.ctype != check_model.ctype)
                        || !valid_checks
                    {
                        return Err(Status::invalid_argument(
                            "terminal model requires a nonzero eid, compatible ctype, and valid local or reserved future checks",
                        ));
                    }
                    if let Some(modifier) = &terminal.modifier {
                        if !modifier.reroute_remote_check_models.is_empty() {
                            return Err(Status::invalid_argument("terminal model cannot reroute checks"));
                        }
                        if let Some(probabilities) = &modifier.probability_modifier {
                            validate_probability_modifier(probabilities, model_type.errors.len())
                                .map_err(Status::invalid_argument)?;
                        }
                    }
                }
                drop(error_model_types);
                let mut gadgets = self.gadgets.write().await;
                let mut check_models = self.check_models.write().await;
                let modified_remote = Arc::new(
                    apply_check_model_reroutes(&check_model_type.remote_gadgets, check_model.modifier.as_ref())
                        .map_err(Status::invalid_argument)?,
                );
                for remote in modified_remote.iter().flatten() {
                    if let Some(gid) = remote.absolute_gid
                        && !gadgets.contains_key(&gid)
                    {
                        return Err(Status::invalid_argument(format!(
                            "physical measurement source gid={gid} is unknown or collected"
                        )));
                    }
                }
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
                    if self.collected_cids.lock().await.contains(&check_model.cid)
                        || check_models.contains_key(&check_model.cid)
                    {
                        return Err(Status::already_exists("check id already exists or has been collected"));
                    }
                    let mut next_cid = self.next_cid.lock().await;
                    *next_cid = (*next_cid).max(check_model.cid.saturating_add(1));
                    check_model.cid
                };
                if cid == crate::misc::index::FUTURE_CHECK_CID {
                    return Err(Status::invalid_argument("check model id is reserved for future checks"));
                }
                if let Some(terminal) = &check_model.terminal_error_model
                    && terminal.cid != 0
                    && terminal.cid != cid
                {
                    return Err(Status::invalid_argument(
                        "terminal model must belong to its containing check model",
                    ));
                }
                let gadget = gadgets.get_mut(&check_model.gid).ok_or_else(|| {
                    Status::invalid_argument(format!("cid={cid} binding to unknown gid={}", check_model.gid))
                })?;
                if gadget.outcomes.borrow().is_some() {
                    return Err(Status::failed_precondition(
                        "check models must be bound before outcomes are submitted",
                    ));
                }
                debug_assert!(check_model_type.gtype == WILDCARD || check_model_type.gtype == gadget.instance.gtype);
                debug_assert!(gadget.binding_cid.is_none());
                gadget.binding_cid.replace(cid);
                let mut deferred_referring_eids = {
                    let mut pending_by_gid = self.pending_referring_by_gid.lock().await;
                    pending_by_gid.remove(&check_model.gid).unwrap_or_default()
                };
                {
                    let mut pending_absolute_referrals = self.pending_referring_by_cid.lock().await;
                    deferred_referring_eids.extend(pending_absolute_referrals.remove(&cid).unwrap_or_default());
                }
                let mut seen_eids = HashSet::new();
                deferred_referring_eids.retain(|eid| seen_eids.insert(*eid));
                let mut check_model = check_model;
                check_model.cid = cid;
                if let Some(terminal) = check_model.terminal_error_model.as_mut() {
                    terminal.cid = cid;
                }
                check_models.insert(
                    cid,
                    CheckModel {
                        instance: check_model.clone(),
                        attaching_eid_vec: vec![],
                        error_model_ready: watch::channel((check_model.error_model_count == Some(0)).then_some(())).0,
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
                let syndrome_published = check_models[&cid].syndrome.clone();
                let gadgets = self.gadgets.clone();
                let check_models = self.check_models.clone();
                let check_model_types = self.check_model_types.clone();
                let token = self.cancellation.read().await.clone();
                let _guard = self.task_counter.guard();
                let trace_shot = self.trace_shot.clone();
                let has_trace = self.config.trace_filepath.is_some();
                tokio::spawn(async move {
                    let _guard = _guard;
                    let expanded_remote_gadgets =
                        match Self::expand_remote_gadgets(&check_model, &modified_remote, gadgets.as_ref(), token.clone())
                            .await
                        {
                            Ok(expanded) => expanded,
                            Err(error) => {
                                if error.code() != tonic::Code::Cancelled {
                                    eprintln!("check-model cid={cid} expansion failed: {error}");
                                    token.cancel();
                                }
                                return;
                            }
                        };
                    if token.is_cancelled() {
                        return;
                    }
                    check_models.write().await.get_mut(&cid).unwrap().expanded_remote_gadgets =
                        Some(expanded_remote_gadgets.clone());
                    // wait for the related gadgets' outcomes to be ready
                    let mut handles = vec![];
                    {
                        let gadgets = gadgets.read().await;
                        if syndrome_published.borrow().is_some() {
                            return;
                        }
                        for gid in [check_model.gid]
                            .into_iter()
                            .chain(expanded_remote_gadgets.iter().filter_map(|x| *x))
                        {
                            let gadget = gadgets.get(&gid).unwrap();
                            match check_or_future(&gadget.outcomes, token.clone()) {
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
                    let published = match Self::try_publish_syndrome(
                        cid,
                        &gadgets,
                        &check_models,
                        &check_model_types,
                        Some(&syndrome_published),
                    )
                    .await
                    {
                        Ok(published) => published,
                        Err(error) => {
                            eprintln!("check-model cid={cid} syndrome publication failed: {error}");
                            token.cancel();
                            return;
                        }
                    };
                    if has_trace && let Some(gid) = published {
                        trace_shot.lock().await.events.push(trace::Event {
                            timestamp_ns: crate::misc::util::timestamp_ns(),
                            event: Some(trace::event::Event::SyndromeReady(trace::SyndromeReadyEvent { cid, gid })),
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
                for remote in modified_remote.iter().flatten() {
                    if let Some(cid) = remote.absolute_cid
                        && self.collected_cids.lock().await.contains(&cid)
                    {
                        return Err(Status::invalid_argument(format!(
                            "remote check model cid={cid} has been collected"
                        )));
                    }
                }

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
                    let mut next_eid = self.next_eid.lock().await;
                    *next_eid = (*next_eid).max(error_model.eid.saturating_add(1));
                    error_model.eid
                };
                if error_models.contains_key(&eid) || self.collected_eids.lock().await.contains(&eid) {
                    return Err(Status::already_exists(format!("eid={eid}")));
                }
                let check_model = check_models.get_mut(&error_model.cid).ok_or_else(|| {
                    Status::invalid_argument(format!("eid={eid} attaching to unknown cid={}", error_model.cid))
                })?;
                if check_model.attaching_eid_vec.len() as u64 >= check_model.instance.error_model_count.unwrap_or(1) {
                    return Err(Status::failed_precondition("all declared error models are already attached"));
                }
                debug_assert!(error_model_type.ctype == WILDCARD || error_model_type.ctype == check_model.instance.ctype);
                check_model.attaching_eid_vec.push(eid);
                if check_model.attaching_eid_vec.len() as u64 == check_model.instance.error_model_count.unwrap_or(1) {
                    check_model.error_model_ready.send_replace(Some(()));
                }
                // Register referring_eids for resolved targets; defer unresolved ones.
                {
                    let mut pending_by_gid = self.pending_referring_by_gid.lock().await;
                    let mut pending_absolute_referrals = self.pending_referring_by_cid.lock().await;
                    let mut pending_by_port = self.pending_referring_by_port.lock().await;
                    for (ri, target_gid) in resolved_gids.iter().enumerate() {
                        let Some(remote) = modified_remote[ri].as_ref() else {
                            continue;
                        };
                        if let Some(target_cid) = remote.absolute_cid {
                            if target_cid == error_model.cid || target_cid == crate::misc::index::FUTURE_CHECK_CID {
                                continue;
                            }
                            let referrals = if let Some(target_cm) = check_models.get_mut(&target_cid) {
                                &mut target_cm.referring_eids
                            } else {
                                pending_absolute_referrals.entry(target_cid).or_default()
                            };
                            if !referrals.contains(&eid) {
                                referrals.push(eid);
                            }
                            continue;
                        }
                        let Some(&target_gid) = target_gid.as_ref() else { continue };
                        if target_gid == owner_gid {
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
                                if let Some(target_cm) = check_models.get_mut(&target_cid)
                                    && !target_cm.referring_eids.contains(&eid)
                                {
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
                collect_after_publication = self.completed_readouts.lock().await.contains(&owner_gid);
                eid
            }
        };
        if collect_after_publication {
            self.collect_garbage().await?;
        }
        Ok((coordinator::ExecuteResponse { id }).into())
    }

    #[cfg_attr(feature = "cli", fastrace::trace)]
    async fn decode(&self, request: Request<coordinator::Outcomes>) -> Result<Response<coordinator::Readouts>, Status> {
        let outcomes = request.into_inner();
        let _task_guard = self
            .task_counter
            .try_guard()
            .ok_or_else(|| Status::unavailable("coordinator reset in progress"))?;
        let mut pins = StatePins::new(&self.pinned_gadgets);
        let gid = outcomes.gid;
        let probability_modifiers = self.bind_probability_modifiers(gid, &outcomes.modifiers).await?;

        // Load outcomes
        let is_free_hop;
        let binding_cid;
        {
            let gadget_types = self.gadget_types.read().await;
            let mut gadgets = self.gadgets.write().await;
            pins.retain([gid]);
            let gadget = gadgets
                .get_mut(&gid)
                .ok_or_else(|| Status::not_found(format!("gid={}", gid)))?;
            if gadget.outcomes.borrow().is_some() {
                return Err(Status::already_exists(format!("gid={gid} outcomes already loaded")));
            }
            is_free_hop = gadget.is_free_hop;
            binding_cid = gadget.binding_cid;
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
            gadget.decoder_seed = outcomes.decoder_seed;
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
        if crate::misc::pool_executor::inline_cpu_allowed()
            && let Some(cid) = binding_cid
            && let Some(gid) =
                Self::try_publish_syndrome(cid, &self.gadgets, &self.check_models, &self.check_model_types, None).await?
        {
            self.record_event(trace::event::Event::SyndromeReady(trace::SyndromeReadyEvent { cid, gid }))
                .await;
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
            return self.finish_decode(gid, pins).await;
        }

        // Hop-counted gadget: five-step window exploration then commit loop.

        // Step 1: Explore mandatory zone (blocking BFS up to buffer_radius).
        let mut explored = self
            .explore_mandatory_zone(gid, &mut pins)
            .await
            .ok_or_else(|| Status::cancelled("decode cancelled"))?;

        // Step 2: Wait for mandatory-zone syndrome to be ready.
        // While waiting, more gadgets may arrive, improving step 3's reach.
        self.await_mandatory_zone_syndrome(&explored)
            .await
            .ok_or_else(|| Status::cancelled("decode cancelled"))?;

        self.await_error_models(&explored.gadgets).await?;
        self.wait_for_history_commitment(gid).await?;

        // Step 3: Explore lookahead zone (non-blocking BFS, lookahead_radius more hops).
        self.explore_lookahead_zone(&mut explored, &mut pins)
            .await
            .ok_or_else(|| Status::cancelled("decode cancelled"))?;

        // Commit loop: check reservations, select the commit region and context,
        // then reserve the entire decoder window before proceeding.
        let reservation = loop {
            let token = self.cancellation.read().await.clone();
            if token.is_cancelled() {
                return Err(Status::cancelled("decode cancelled"));
            }

            self.await_error_models(&explored.gadgets).await?;
            let remote_check_owners = if self.config.buffer_radius == 0 {
                HashMap::new()
            } else {
                self.available_remote_check_owners(&explored.gadgets).await
            };
            let blocking_gids: Vec<u64>;
            {
                let mut gadgets = self.gadgets.write().await;

                // Check if this gadget has been claimed by another task
                let my_state = gadgets.get(&gid).map(|g| g.state.borrow().clone()).unwrap_or_default();
                if my_state.committed {
                    // Already committed by another leader. Emit non-leader event, wait for pauli_frame.
                    drop(gadgets);
                    self.record_event(trace::event::Event::Decode(trace::DecodeEvent {
                        gid,
                        is_leader: false,
                        ..Default::default()
                    }))
                    .await;
                    return self.finish_decode(gid, pins).await;
                }
                if let Some(other) = my_state.reserved_by
                    && other != gid
                {
                    // Claimed by another leader (commit region or buffer).
                    // Wait for state to resolve, then re-check.
                    let gadget = gadgets.get(&gid).ok_or_else(|| Status::not_found(format!("gid={gid}")))?;
                    let mut rx = gadget.state.subscribe();
                    let token_c = token.clone();
                    drop(gadgets);
                    let resolved = tokio::select! {
                        result = rx.wait_for(|state| state.reserved_by.is_none()) => {
                            result.ok().map(|state| state.clone())
                        }
                        () = token_c.cancelled() => None
                    };
                    if resolved.is_some_and(|state| state.committed) {
                        self.record_event(trace::event::Event::Decode(trace::DecodeEvent {
                            gid,
                            is_leader: false,
                            leader_gid: other,
                            ..Default::default()
                        }))
                        .await;
                        return self.finish_decode(gid, pins).await;
                    }
                    continue;
                }

                // Check if any gadget in the window is reserved.
                let mut blocked: Vec<u64> = Vec::new();
                for &wgid in &explored.gadgets {
                    if let Some(g) = gadgets.get(&wgid)
                        && g.state.borrow().reserved_by.is_some()
                    {
                        blocked.push(wgid);
                    }
                }

                if blocked.is_empty() {
                    // Step 4: Select commit region.
                    self.select_commit_region(&mut explored, &gadgets);

                    // Step 5: Shrink window to minimal decoder window.
                    self.shrink_window(&mut explored, &gadgets);

                    let remote_context: HashSet<_> = explored
                        .decoder_window
                        .iter()
                        .filter(|gid| !gadgets[*gid].state.borrow().committed)
                        .filter_map(|gid| remote_check_owners.get(gid))
                        .flatten()
                        .copied()
                        .collect();
                    explored.decoder_window.extend(remote_context);
                    pins.retain(explored.decoder_window.iter().copied());
                    blocked.extend(
                        explored
                            .decoder_window
                            .iter()
                            .copied()
                            .filter(|gid| gadgets[gid].state.borrow().reserved_by.is_some()),
                    );
                }

                if blocked.is_empty() {
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

                    // Reserve the entire window without changing commitment.
                    for &wgid in &explored.decoder_window {
                        if let Some(g) = gadgets.get_mut(&wgid) {
                            g.state.send_modify(|state| state.reserved_by = Some(gid));
                        }
                    }
                    // Dropping a leader must not strand reservations or permit partial retries.
                    break token.drop_guard();
                }

                blocking_gids = blocked;
            }

            // Wait for blocking gadgets to release their reservations.
            let gadgets = self.gadgets.read().await;
            let mut watchers = Vec::new();
            for &bgid in &blocking_gids {
                if let Some(g) = gadgets.get(&bgid) {
                    let mut rx = g.state.subscribe();
                    let token = token.clone();
                    watchers.push(async move {
                        tokio::select! {
                            result = rx.wait_for(|state| state.reserved_by.is_none()) => {
                                result.map(|_| ()).unwrap_or(())
                            }
                            _ = token.cancelled() => {}
                        }
                    });
                }
            }
            drop(gadgets);
            futures_util::future::join_all(watchers).await;
        };

        // Decode and commit (builds relative program, reads fresh syndromes, calls decoder,
        // applies corrections, marks Committed, releases buffer).
        if let Err(error) = self
            .decode_and_commit(
                gid,
                &explored.commit_region,
                &explored.committing_cids,
                &explored.decoder_window,
            )
            .await
        {
            eprintln!("Window decode failed for gid={gid}: {error}");
            self.cancel_pending().await;
            return Err(error);
        }
        reservation.disarm();

        // Wait for pauli_frame (may depend on predecessors committed by other tasks,
        // which is now possible since buffer has been released in update_pauli_frame).
        self.finish_decode(gid, pins).await
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
        self.prepared_windows.lock().unwrap().clear();
        {
            let mut token = self.cancellation.write().await;
            *token = CancellationToken::new();
        }
        if flags.reset_library {
            self.port_types.write().await.clear();
            self.gadget_types.write().await.clear();
            self.check_model_types.write().await.clear();
            self.error_model_types.write().await.clear();
            self.referenced_error_remotes.write().await.clear();
        }
        self.gadgets.write().await.clear();
        self.check_models.write().await.clear();
        self.error_models.write().await.clear();
        self.completed_readouts.lock().await.clear();
        self.collected_gids.lock().await.clear();
        self.collected_cids.lock().await.clear();
        self.collected_eids.lock().await.clear();
        self.collect_unused_types().await;
        self.pending_referring_by_gid.lock().await.clear();
        self.pending_referring_by_cid.lock().await.clear();
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
            self.persistent_prepared_windows.lock().unwrap().clear();
        }
        // since decoders reset asynchronously, wait for all the decoders to finish
        self.decoder
            .reset(blackbox_decoder::ResetRequest {
                reset_hypergraphs: flags.reset_decoder_service || flags.reset_library,
                ..Default::default()
            })
            .await
            .map_err(|e| Status::internal(format!("reset decoder service error: {}", e)))?;
        if let Some(decoder) = &self.gap_decoder {
            decoder
                .reset(blackbox_decoder::ResetRequest {
                    reset_hypergraphs: flags.reset_decoder_service || flags.reset_library,
                    ..Default::default()
                })
                .await
                .map_err(|error| Status::internal(format!("reset gap decoder service error: {error}")))?;
        }
        // flush current shot into the trace and write to file if configured
        if let Some(filepath) = &self.config.trace_filepath {
            let shot = std::mem::take(&mut *self.trace_shot.lock().await);
            let mut trace = self.trace.lock().await;
            trace.shots.push(shot);
            let buf = trace.encode_to_vec();
            std::fs::write(filepath, buf).map_err(|e| Status::internal(format!("failed to write trace: {e}")))?;
        }
        Ok(().into())
    }
}

impl WindowCoordinator {
    /// Decode using a caller-pool slot when available, otherwise use the service path.
    ///
    /// The `Arc` lets queued work retain the coordinator. The RPC service trait
    /// borrows `self` and runs on Tokio, without a caller-pool slot; both paths
    /// execute the same service implementation.
    ///
    /// # Errors
    /// Returns the service's decoding error or a status if reset prevents admission.
    pub async fn decode(
        self: &Arc<Self>,
        request: Request<coordinator::Outcomes>,
    ) -> Result<Response<coordinator::Readouts>, Status> {
        if let Some(pool) = self.decoder.thread_pool()
            && crate::misc::pool_executor::can_submit_inline(pool)
            && let Ok(runtime) = tokio::runtime::Handle::try_current()
        {
            return self.queue_decode(request, runtime)?.await;
        }
        coordinator::coordinator_server::Coordinator::decode(self.as_ref(), request).await
    }

    pub(super) fn queue_decode(
        self: &Arc<Self>,
        request: Request<coordinator::Outcomes>,
        runtime: tokio::runtime::Handle,
    ) -> Result<async_task::Task<Result<Response<coordinator::Readouts>, Status>>, Status> {
        use futures_util::FutureExt as _;

        let pool = self
            .decoder
            .thread_pool()
            .cloned()
            .ok_or_else(|| Status::failed_precondition("no decoder pool"))?;
        // Admission precedes queuing so reset also waits for unpolled work.
        let admission = self
            .task_counter
            .try_guard()
            .ok_or_else(|| Status::unavailable("coordinator reset in progress"))?;
        let window = Arc::clone(self);
        Ok(crate::misc::pool_executor::spawn(pool, runtime, async move {
            let _admission = admission;
            if let Ok(result) =
                std::panic::AssertUnwindSafe(coordinator::coordinator_server::Coordinator::decode(window.as_ref(), request))
                    .catch_unwind()
                    .await
            {
                result
            } else {
                eprintln!("window decode task panicked");
                window.cancel_pending().await;
                Err(Status::internal("window decode task panicked"))
            }
        }))
    }

    pub(super) async fn try_publish_syndrome(
        cid: u64,
        gadgets: &RwLock<HashMap<u64, Gadget>>,
        check_models: &RwLock<HashMap<u64, CheckModel>>,
        check_types: &RwLock<HashMap<u64, Arc<bin::CheckModelType>>>,
        completed: Option<&watch::Sender<Option<BitVector>>>,
    ) -> Result<Option<u64>, Status> {
        let (gid, ctype, expanded, modified, published) = {
            let checks = check_models.read().await;
            if completed.is_some_and(|signal| signal.borrow().is_some()) {
                return Ok(None);
            }
            let check = checks
                .get(&cid)
                .ok_or_else(|| Status::internal(format!("missing syndrome check cid={cid}")))?;
            if check.syndrome.borrow().is_some() {
                return Ok(None);
            }
            let Some(expanded) = &check.expanded_remote_gadgets else {
                return Ok(None);
            };
            (
                check.instance.gid,
                check.instance.ctype,
                expanded.clone(),
                Arc::clone(&check.modified_remote_gadgets),
                check.syndrome.clone(),
            )
        };
        let types = check_types.read().await;
        if published.borrow().is_some() {
            return Ok(None);
        }
        let kind = types
            .get(&ctype)
            .cloned()
            .ok_or_else(|| Status::internal(format!("missing syndrome type ctype={ctype}")))?;
        drop(types);
        let (local, remotes) = {
            let gadgets = gadgets.read().await;
            if published.borrow().is_some() {
                return Ok(None);
            }
            let gadget = gadgets
                .get(&gid)
                .ok_or_else(|| Status::internal(format!("missing syndrome owner gid={gid}")))?;
            let Some(local) = gadget.outcomes.borrow().clone() else {
                return Ok(None);
            };
            let mut remotes = Vec::with_capacity(expanded.len());
            for remote in &expanded {
                if let Some(remote) = remote {
                    let gadget = gadgets
                        .get(remote)
                        .ok_or_else(|| Status::internal(format!("missing syndrome input gid={remote}")))?;
                    let Some(outcomes) = gadget.outcomes.borrow().clone() else {
                        return Ok(None);
                    };
                    remotes.push(Some(outcomes));
                } else {
                    remotes.push(None);
                }
            }
            (local, remotes)
        };
        let mut syndrome = bit_vector::from_sparse_indices(u64::try_from(kind.checks.len()).unwrap(), &[]);
        let mut count = 0;
        for (index, check) in kind.checks.iter().enumerate() {
            let mut defect = check.naturally_flipped;
            for measurement in &check.measurements {
                if let Some(remote) = measurement.remote_gadget {
                    let remote = usize::try_from(remote).map_err(|_| Status::internal("syndrome remote index overflow"))?;
                    let outcomes = remotes
                        .get(remote)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| Status::internal("unresolved syndrome remote input"))?;
                    let mapping = modified
                        .get(remote)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| Status::internal("missing syndrome remote mapping"))?;
                    defect ^= get_bit(outcomes, measurement.measurement_index + mapping.measurement_bias);
                } else {
                    defect ^= get_bit(&local, measurement.measurement_index);
                }
            }
            set_bit(&mut syndrome, u64::try_from(index).unwrap(), defect);
            count += u64::from(defect);
        }
        let mut checks = check_models.write().await;
        // Another publisher may already have committed or corrected this syndrome.
        if published.borrow().is_some() {
            return Ok(None);
        }
        let check = checks
            .get_mut(&cid)
            .ok_or_else(|| Status::internal(format!("missing syndrome check cid={cid}")))?;
        check.syndrome_count = count;
        check.syndrome.send_replace(Some(syndrome));
        Ok(Some(gid))
    }

    async fn bind_probability_modifiers(
        &self,
        gid: u64,
        modifiers: &[bin::ProbabilityModifier],
    ) -> Result<Vec<(u64, bin::ProbabilityModifier)>, Status> {
        if modifiers.is_empty() {
            return Ok(vec![]);
        }
        self.await_error_models(&HashSet::from([gid])).await?;
        let cid = self
            .gadgets
            .read()
            .await
            .get(&gid)
            .and_then(|gadget| gadget.binding_cid)
            .ok_or_else(|| Status::failed_precondition(format!("gid={gid} has no binding check model")))?;
        let (mut eids, terminal) = self
            .check_models
            .read()
            .await
            .get(&cid)
            .map(|check_model| {
                (
                    check_model.attaching_eid_vec.clone(),
                    check_model.instance.terminal_error_model.clone(),
                )
            })
            .ok_or_else(|| Status::failed_precondition(format!("cid={cid} is not loaded")))?;
        if eids.is_empty()
            && let Some(terminal) = &terminal
        {
            eids.push(terminal.eid);
        }
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
                .map(|model| &model.instance)
                .or_else(|| terminal.as_ref().filter(|model| model.eid == eid))
                .ok_or_else(|| Status::failed_precondition(format!("eid={eid} is not loaded")))?;
            let error_model_type = error_model_types
                .get(&error_model.etype)
                .ok_or_else(|| Status::failed_precondition(format!("etype={} is not loaded", error_model.etype)))?;
            validate_probability_modifier(modifier, error_model_type.errors.len()).map_err(Status::invalid_argument)?;
            bound.push((eid, modifier.clone()));
        }
        Ok(bound)
    }

    /// expand the remote gadgets referred by the check model; note that this function will
    /// be waiting for the gadget if it has not been connected yet, thus it should be called
    /// in a separate async task without blocking the gRPC request.
    pub(super) async fn expand_remote_gadgets(
        check_model: &bin::CheckModel,
        modified_remote_gadgets: &Vec<Option<bin::check_model_type::RemoteGadget>>,
        gadgets: &RwLock<HashMap<u64, Gadget>>,
        token: CancellationToken,
    ) -> Result<Vec<Option<u64>>, Status> {
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
            .await?;
        }
        Ok(expanded_remote_gid_vec)
    }

    async fn expand_remote_gadget(
        expanded_remote_gid_vec: &mut Vec<Option<u64>>,
        ri: usize,
        remote_gadgets: &Vec<Option<bin::check_model_type::RemoteGadget>>,
        gid: u64,
        gadgets: &RwLock<HashMap<u64, Gadget>>,
        token: CancellationToken,
    ) -> Result<(), Status> {
        if token.is_cancelled() {
            return Err(Status::cancelled("remote gadget expansion cancelled"));
        }
        if expanded_remote_gid_vec[ri].is_some() || remote_gadgets[ri].is_none() {
            return Ok(());
        }
        let remote_gadget = remote_gadgets[ri].as_ref().unwrap();
        // if absolute_gid is provided, use it directly
        if let Some(absolute_gid) = remote_gadget.absolute_gid {
            expanded_remote_gid_vec[ri] = Some(absolute_gid);
            return Ok(());
        }
        // expand the dependent remote gadget first
        // (we do not check circular dependency here for simplicity, see ProgSpec)
        let previous = if let Some(previous) = remote_gadget.previous_remote_gadget {
            let previous =
                usize::try_from(previous).map_err(|_| Status::invalid_argument("remote predecessor index exceeds usize"))?;
            Box::pin(Self::expand_remote_gadget(
                expanded_remote_gid_vec,
                previous,
                remote_gadgets,
                gid,
                gadgets,
                token.clone(),
            ))
            .await?;
            expanded_remote_gid_vec[previous].ok_or_else(|| Status::internal("remote predecessor did not resolve"))?
        } else {
            gid
        };
        let gadgets = gadgets.read().await;
        let gadget = gadgets
            .get(&previous)
            .ok_or_else(|| Status::internal(format!("remote path lost gid={previous}")))?;
        match remote_gadget.port.unwrap() {
            bin::check_model_type::remote_gadget::Port::Output(port) => {
                let port =
                    usize::try_from(port).map_err(|_| Status::invalid_argument("remote output port exceeds usize"))?;
                let next = get_or_future(
                    gadget
                        .outputs
                        .get(port)
                        .ok_or_else(|| Status::invalid_argument("remote output port is out of range"))?,
                    token,
                );
                drop(gadgets); // release the read lock
                let next = match next {
                    Ok(next) => Some(next),
                    Err(waiting) => waiting.await,
                };
                let next = next.ok_or_else(|| Status::cancelled("remote output wait cancelled"))?;
                expanded_remote_gid_vec[ri] = Some(next.gid);
            }
            bin::check_model_type::remote_gadget::Port::Input(port) => {
                let connector = &gadget.instance.connectors[port as usize];
                expanded_remote_gid_vec[ri] = Some(connector.gid);
            }
        }
        Ok(())
    }
}
