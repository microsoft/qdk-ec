//! Mock coordinator for testing
//!
//! Records all `load_library` and `execute` calls for verification.
//! Provides `get_effective_types()` to expand modifiers and compute effective type definitions.

use crate::bin::{self, instruction};
use crate::coordinator::{self, coordinator_server};
use hashbrown::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

/// A mock coordinator that records all operations for testing.
pub struct MockCoordinator {
    pub state: RwLock<MockCoordinatorState>,
}

#[derive(Default)]
pub struct MockCoordinatorState {
    /// All libraries loaded via load_library calls
    pub libraries: Vec<bin::Library>,
    /// All instructions executed via execute calls
    pub instructions: Vec<bin::Instruction>,
    /// Next gid to assign
    pub next_gid: u64,
    /// Next cid to assign
    pub next_cid: u64,
    /// Next eid to assign
    pub next_eid: u64,
    /// Gadget types by gtype
    pub gadget_types: HashMap<u64, bin::GadgetType>,
    /// Check model types by ctype
    pub check_model_types: HashMap<u64, bin::CheckModelType>,
    /// Error model types by etype
    pub error_model_types: HashMap<u64, bin::ErrorModelType>,
    /// Gadget instances by gid
    pub gadgets: HashMap<u64, bin::Gadget>,
    /// Check model instances by cid
    pub check_models: HashMap<u64, bin::CheckModel>,
    /// Error model instances by eid
    pub error_models: HashMap<u64, bin::ErrorModel>,
    /// Output connections: maps (gid, output_port) -> connected_gid
    /// Built from gadget connectors for reverse lookup
    pub outputs: HashMap<(u64, u64), u64>,
}

impl MockCoordinatorState {
    /// Update the outputs map when a gadget is added
    fn register_gadget_outputs(&mut self, gadget: &bin::Gadget) {
        for connector in gadget.connectors.iter() {
            // connector.gid's output port connector.port is connected to this gadget's input
            self.outputs.insert((connector.gid, connector.port), gadget.gid);
        }
    }

    /// Expand a single remote gadget reference to an absolute gid.
    /// Uses the same logic as `MonolithicCoordinator::expand_remote_gadget`.
    pub fn expand_remote_gadget(
        &self,
        expanded: &mut Vec<Option<u64>>,
        ri: usize,
        remote_gadgets: &[Option<bin::check_model_type::RemoteGadget>],
        gid: u64,
    ) {
        assert!(ri < remote_gadgets.len());
        if expanded[ri].is_some() || remote_gadgets[ri].is_none() {
            return; // already expanded or nothing to expand
        }
        let remote_gadget = remote_gadgets[ri].as_ref().unwrap();
        // if absolute_gid is provided, use it directly
        if let Some(absolute_gid) = remote_gadget.absolute_gid {
            expanded[ri] = Some(absolute_gid);
            return;
        }

        let previous = if let Some(previous) = remote_gadget.previous_remote_gadget {
            self.expand_remote_gadget(expanded, previous as usize, remote_gadgets, gid);
            expanded[previous as usize].unwrap()
        } else {
            gid
        };

        match remote_gadget.port.as_ref().expect("remote gadget port not set") {
            bin::check_model_type::remote_gadget::Port::Output(port) => {
                // Same as monolithic: look up output connection
                expanded[ri] = self.outputs.get(&(previous, *port)).copied();
            }
            bin::check_model_type::remote_gadget::Port::Input(port) => {
                let gadget = self.gadgets.get(&previous).expect("gadget not found during expansion");
                let connector = &gadget.connectors[*port as usize];
                expanded[ri] = Some(connector.gid);
            }
        }
    }

    /// Expand all remote gadgets for a check model.
    pub fn expand_remote_gadgets(
        &self,
        check_model: &bin::CheckModel,
        remote_gadgets: &[Option<bin::check_model_type::RemoteGadget>],
    ) -> Vec<Option<u64>> {
        let mut expanded = vec![None; remote_gadgets.len()];
        for ri in 0..remote_gadgets.len() {
            self.expand_remote_gadget(&mut expanded, ri, remote_gadgets, check_model.gid);
        }
        expanded
    }

    /// Expand a single remote check model reference to an absolute cid.
    /// Uses the same logic as `MonolithicCoordinator::expand_remote_check_model`.
    pub fn expand_remote_check_model(
        &self,
        expanded_gids: &mut Vec<Option<u64>>,
        ri: usize,
        remote_check_models: &[Option<bin::error_model_type::RemoteCheckModel>],
        gid: u64,
    ) {
        assert!(ri < remote_check_models.len());
        if expanded_gids[ri].is_some() || remote_check_models[ri].is_none() {
            return;
        }
        let remote_check_model = remote_check_models[ri].as_ref().unwrap();

        if remote_check_model.absolute_cid.is_some() {
            expanded_gids[ri] = Some(u64::MAX - 1); // sentinel for absolute_cid
            return;
        }

        let previous = if let Some(previous) = remote_check_model.previous_remote_check_model {
            self.expand_remote_check_model(expanded_gids, previous as usize, remote_check_models, gid);
            expanded_gids[previous as usize].unwrap()
        } else {
            gid
        };

        match remote_check_model.port.as_ref().expect("remote check model port not set") {
            bin::error_model_type::remote_check_model::Port::Output(port) => {
                // Same as monolithic: look up output connection
                expanded_gids[ri] = self.outputs.get(&(previous, *port)).copied();
            }
            bin::error_model_type::remote_check_model::Port::Input(port) => {
                let gadget = self.gadgets.get(&previous).expect("gadget not found during expansion");
                let connector = &gadget.connectors[*port as usize];
                expanded_gids[ri] = Some(connector.gid);
            }
        }
    }

    /// Expand all remote check models for an error model, returning absolute cids.
    pub fn expand_remote_check_models(
        &self,
        error_model: &bin::ErrorModel,
        remote_check_models: &[Option<bin::error_model_type::RemoteCheckModel>],
    ) -> Vec<Option<u64>> {
        let check_model = self.check_models.get(&error_model.cid).expect("check model not found");
        let gid = check_model.gid;

        let mut expanded_gids = vec![None; remote_check_models.len()];
        for ri in 0..remote_check_models.len() {
            self.expand_remote_check_model(&mut expanded_gids, ri, remote_check_models, gid);
        }

        expanded_gids
            .into_iter()
            .enumerate()
            .map(|(ri, opt_gid)| {
                opt_gid.and_then(|gid| {
                    if gid == u64::MAX - 1 {
                        remote_check_models[ri].as_ref().unwrap().absolute_cid
                    } else {
                        self.check_models.values().find(|cm| cm.gid == gid).map(|cm| cm.cid)
                    }
                })
            })
            .collect()
    }
}

/// Effective check model type with modifiers applied.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectiveCheckModelType {
    pub ctype: u64,
    pub gtype: u64,
    pub checks: Vec<bin::check_model_type::Check>,
    /// Resolved absolute gids for remote gadgets. Indices not referenced by any check are 0.
    pub remote_gadgets: Vec<u64>,
}

/// Effective error model type with modifiers applied.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectiveErrorModelType {
    pub etype: u64,
    pub ctype: u64,
    pub errors: Vec<bin::error_model_type::Error>,
    /// Resolved absolute cids for remote check models. Indices not referenced by any error are 0.
    pub remote_check_models: Vec<u64>,
}

/// Collection of effective types computed from instances with modifiers applied.
#[derive(Clone, Debug, Default)]
pub struct EffectiveTypes {
    pub check_model_types: HashMap<u64, EffectiveCheckModelType>,
    pub error_model_types: HashMap<u64, EffectiveErrorModelType>,
}

impl MockCoordinator {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            state: RwLock::new(MockCoordinatorState {
                next_gid: 1,
                next_cid: 1,
                next_eid: 1,
                ..Default::default()
            }),
        })
    }

    /// Compute effective types by applying modifiers from instances to their types.
    ///
    /// For each check model instance, applies the modifier's reroute_remote_gadgets
    /// to produce an effective check model type.
    ///
    /// For each error model instance, applies the modifier's reroute_remote_check_models
    /// and probability_modifier to produce an effective error model type.
    pub async fn get_effective_types(&self) -> EffectiveTypes {
        let state = self.state.read().await;
        let mut effective = EffectiveTypes::default();

        // Process check models
        for (&cid, check_model) in &state.check_models {
            let base_type = state
                .check_model_types
                .get(&check_model.ctype)
                .expect("check model type not found");

            // Build modified remote gadgets list (with reroutes applied)
            let mut modified_remote_gadgets: Vec<Option<bin::check_model_type::RemoteGadget>> =
                base_type.remote_gadgets.iter().cloned().map(Some).collect();

            if let Some(modifier) = &check_model.modifier {
                for reroute in &modifier.reroute_remote_gadgets {
                    let idx = reroute.remote_gadget_index as usize;
                    while idx >= modified_remote_gadgets.len() {
                        modified_remote_gadgets.push(None);
                    }
                    modified_remote_gadgets[idx] = reroute.value.clone();
                }
            }

            // Expand remote gadgets to absolute gids
            let expanded = state.expand_remote_gadgets(check_model, &modified_remote_gadgets);

            // Collect indices referenced by checks and find max
            let mut referenced_indices = std::collections::HashSet::new();
            for check in &base_type.checks {
                for measurement in &check.measurements {
                    if let Some(idx) = measurement.remote_gadget {
                        referenced_indices.insert(idx as usize);
                    }
                }
            }
            let max_referenced = referenced_indices.iter().max().map(|&m| m + 1).unwrap_or(0);

            // Build final Vec<u64>, truncated to max referenced index
            let remote_gadgets: Vec<u64> = expanded
                .iter()
                .take(max_referenced)
                .enumerate()
                .map(|(idx, opt_gid)| {
                    if referenced_indices.contains(&idx) {
                        opt_gid.expect("expanded remote gadget missing")
                    } else {
                        0
                    }
                })
                .collect();

            effective.check_model_types.insert(
                cid,
                EffectiveCheckModelType {
                    ctype: check_model.ctype,
                    gtype: base_type.gtype,
                    checks: base_type.checks.clone(),
                    remote_gadgets,
                },
            );
        }

        // Process error models
        for (&eid, error_model) in &state.error_models {
            let base_type = state
                .error_model_types
                .get(&error_model.etype)
                .expect("error model type not found");

            // Build modified remote check models list (with reroutes applied)
            let mut modified_remote_check_models: Vec<Option<bin::error_model_type::RemoteCheckModel>> =
                base_type.remote_check_models.iter().cloned().map(Some).collect();

            let mut errors = base_type.errors.clone();

            if let Some(modifier) = &error_model.modifier {
                for reroute in &modifier.reroute_remote_check_models {
                    let idx = reroute.remote_check_model_index as usize;
                    while idx >= modified_remote_check_models.len() {
                        modified_remote_check_models.push(None);
                    }
                    modified_remote_check_models[idx] = reroute.value.clone();
                }

                // Apply probability modifier
                if let Some(prob_modifier) = &modifier.probability_modifier {
                    for (idx, &prob) in prob_modifier.probabilities.iter().enumerate() {
                        assert!(idx < errors.len());
                        errors[idx].probability = prob;
                    }
                    for (&idx, &prob) in prob_modifier
                        .sparse_indices
                        .iter()
                        .zip(prob_modifier.sparse_probabilities.iter())
                    {
                        assert!((idx as usize) < errors.len());
                        errors[idx as usize].probability = prob;
                    }
                }
            }

            // Expand remote check models to absolute cids
            let expanded = state.expand_remote_check_models(error_model, &modified_remote_check_models);

            // Collect indices referenced by errors and find max
            let mut referenced_indices = std::collections::HashSet::new();
            for error in &errors {
                for check in &error.checks {
                    if let Some(idx) = check.remote_check_model {
                        referenced_indices.insert(idx as usize);
                    }
                }
            }
            let max_referenced = referenced_indices.iter().max().map(|&m| m + 1).unwrap_or(0);

            // Build final Vec<u64>, truncated to max referenced index
            let remote_check_models: Vec<u64> = expanded
                .iter()
                .take(max_referenced)
                .enumerate()
                .map(|(idx, opt_cid)| {
                    if referenced_indices.contains(&idx) {
                        opt_cid.expect("expanded remote check model missing")
                    } else {
                        0
                    }
                })
                .collect();

            effective.error_model_types.insert(
                eid,
                EffectiveErrorModelType {
                    etype: error_model.etype,
                    ctype: base_type.ctype,
                    errors,
                    remote_check_models,
                },
            );
        }

        effective
    }

    /// Clear all recorded state.
    pub async fn clear(&self) {
        let mut state = self.state.write().await;
        state.libraries.clear();
        state.instructions.clear();
        state.gadget_types.clear();
        state.check_model_types.clear();
        state.error_model_types.clear();
        state.gadgets.clear();
        state.check_models.clear();
        state.error_models.clear();
        state.outputs.clear();
        state.next_gid = 1;
        state.next_cid = 1;
        state.next_eid = 1;
    }
}

impl Default for MockCoordinator {
    fn default() -> Self {
        Self {
            state: RwLock::new(MockCoordinatorState {
                next_gid: 1,
                next_cid: 1,
                next_eid: 1,
                ..Default::default()
            }),
        }
    }
}

#[tonic::async_trait]
impl coordinator_server::Coordinator for MockCoordinator {
    async fn load_library(&self, request: Request<bin::Library>) -> Result<Response<()>, Status> {
        let library = request.into_inner();
        let mut state = self.state.write().await;

        // Store types
        for gadget_type in &library.gadget_types {
            state.gadget_types.insert(gadget_type.gtype, gadget_type.clone());
        }
        for check_model_type in &library.check_model_types {
            state
                .check_model_types
                .insert(check_model_type.ctype, check_model_type.clone());
        }
        for error_model_type in &library.error_model_types {
            state
                .error_model_types
                .insert(error_model_type.etype, error_model_type.clone());
        }

        // Record the library
        state.libraries.push(library);

        Ok(Response::new(()))
    }

    async fn unload(&self, _request: Request<coordinator::UnloadLibrary>) -> Result<Response<()>, Status> {
        Ok(Response::new(()))
    }

    async fn execute(&self, request: Request<bin::Instruction>) -> Result<Response<coordinator::ExecuteResponse>, Status> {
        let instruction = request.into_inner();
        let mut state = self.state.write().await;

        let id = match &instruction.create {
            Some(instruction::Create::Gadget(gadget)) => {
                // Validate that the gadget type is registered
                assert!(
                    state.gadget_types.contains_key(&gadget.gtype),
                    "gadget references unknown gtype {}: was the library loaded?",
                    gadget.gtype
                );
                let gid = if gadget.gid == 0 {
                    // Auto-assign: find next unused gid
                    while state.gadgets.contains_key(&state.next_gid) {
                        state.next_gid += 1;
                    }
                    let gid = state.next_gid;
                    state.next_gid += 1;
                    gid
                } else {
                    // User-provided gid
                    gadget.gid
                };
                let mut gadget = gadget.clone();
                gadget.gid = gid;
                state.register_gadget_outputs(&gadget);
                state.gadgets.insert(gid, gadget);
                gid
            }
            Some(instruction::Create::CheckModel(check_model)) => {
                // Validate that the check model type is registered
                assert!(
                    state.check_model_types.contains_key(&check_model.ctype),
                    "check model references unknown ctype {}: was the library loaded?",
                    check_model.ctype
                );
                let cid = if check_model.cid == 0 {
                    // Auto-assign: find next unused cid
                    while state.check_models.contains_key(&state.next_cid) {
                        state.next_cid += 1;
                    }
                    let cid = state.next_cid;
                    state.next_cid += 1;
                    cid
                } else {
                    // User-provided cid
                    check_model.cid
                };
                let mut check_model = check_model.clone();
                check_model.cid = cid;
                state.check_models.insert(cid, check_model);
                cid
            }
            Some(instruction::Create::ErrorModel(error_model)) => {
                // Validate that the error model type is registered
                assert!(
                    state.error_model_types.contains_key(&error_model.etype),
                    "error model references unknown etype {}: was the library loaded?",
                    error_model.etype
                );
                let eid = if error_model.eid == 0 {
                    // Auto-assign: find next unused eid
                    while state.error_models.contains_key(&state.next_eid) {
                        state.next_eid += 1;
                    }
                    let eid = state.next_eid;
                    state.next_eid += 1;
                    eid
                } else {
                    // User-provided eid
                    error_model.eid
                };
                let mut error_model = error_model.clone();
                error_model.eid = eid;
                state.error_models.insert(eid, error_model);
                eid
            }
            None => return Err(Status::invalid_argument("missing instruction")),
        };

        // Record the instruction
        state.instructions.push(instruction);

        Ok(Response::new(coordinator::ExecuteResponse { id }))
    }

    async fn decode(&self, request: Request<coordinator::Outcomes>) -> Result<Response<coordinator::Readouts>, Status> {
        let outcomes = request.into_inner();
        // Return empty readouts - this is a mock
        Ok(Response::new(coordinator::Readouts {
            gid: outcomes.gid,
            readouts: Some(crate::util::BitVector::default()),
            probabilities: vec![],
        }))
    }

    async fn reset(&self, request: Request<coordinator::ResetRequest>) -> Result<Response<()>, Status> {
        let flags = request.into_inner();
        let mut state = self.state.write().await;
        if flags.reset_library {
            state.gadget_types.clear();
            state.check_model_types.clear();
            state.error_model_types.clear();
            state.libraries.clear();
        }
        state.instructions.clear();
        state.gadgets.clear();
        state.check_models.clear();
        state.error_models.clear();
        state.outputs.clear();
        state.next_gid = 1;
        state.next_cid = 1;
        state.next_eid = 1;
        Ok(Response::new(()))
    }
}
